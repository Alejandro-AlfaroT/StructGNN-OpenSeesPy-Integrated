"""Train the baseline causal hybrid GNN-LSTM surrogate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from rc_hybrid_surrogate.data import (
    HybridDataset,
    NormalizationStats,
    collate_hybrid,
    discover_samples,
    grouped_split,
    move_batch,
    write_split_manifest,
)
from rc_hybrid_surrogate.losses import masked_smooth_l1
from rc_hybrid_surrogate.metrics import ResponseMetrics
from rc_hybrid_surrogate.model import HybridGNNLSTM


MODEL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_ROOT.parent
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "RC Structure" / "outputs" / "parameterized_2500"
DEFAULT_CONFIG = MODEL_ROOT / "configs" / "baseline.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        help="Root containing hybrid_sample.npz files; repeat for multiple devices.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(MODEL_ROOT / "outputs" / "baseline"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict) -> HybridGNNLSTM:
    return HybridGNNLSTM(
        graph_hidden_dim=int(config["graph_hidden_dim"]),
        graph_layers=int(config["graph_layers"]),
        condition_dim=int(config["condition_dim"]),
        lstm_hidden_dim=int(config["lstm_hidden_dim"]),
        lstm_layers=int(config["lstm_layers"]),
        dropout=float(config["dropout"]),
    )


def make_loader(records, stats, config, shuffle: bool) -> DataLoader:
    dataset = HybridDataset(
        records,
        normalization=stats,
        sequence_stride=int(config["sequence_stride"]),
        maximum_steps=config.get("maximum_steps"),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["data_loader_workers"]),
        collate_fn=collate_hybrid,
        pin_memory=torch.cuda.is_available(),
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    delay = 0.05
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.5)


def run_epoch(
    model,
    loader,
    device,
    target_mean,
    target_std,
    huber_beta,
    optimizer=None,
    scaler=None,
    gradient_clip_norm=1.0,
    mixed_precision=False,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    metrics = ResponseMetrics()
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                prediction = model(batch)
                loss = masked_smooth_l1(
                    prediction, batch["target"], batch["mask"], beta=huber_beta
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().item())
        batches += 1
        metrics.update(
            prediction.detach(),
            batch["target"],
            batch["mask"],
            target_mean,
            target_std,
        )
    result = {"loss": total_loss / max(batches, 1)}
    result.update(metrics.compute())
    return result


def smoke_test(records, config, device):
    selected = records[: min(2, len(records))]
    if not selected:
        raise RuntimeError("No valid hybrid samples were discovered.")
    stats = NormalizationStats.calculate(selected)
    smoke_config = dict(config)
    smoke_config.update(
        {
            "batch_size": min(2, len(selected)),
            "sequence_stride": 1,
            "maximum_steps": 128,
            "graph_hidden_dim": 32,
            "graph_layers": 2,
            "condition_dim": 32,
            "lstm_hidden_dim": 48,
            "lstm_layers": 1,
        }
    )
    loader = make_loader(selected, stats, smoke_config, shuffle=False)
    batch = move_batch(next(iter(loader)), device)
    model = make_model(smoke_config).to(device)
    prediction = model(batch)
    loss = masked_smooth_l1(prediction, batch["target"], batch["mask"])
    loss.backward()
    print(
        json.dumps(
            {
                "status": "smoke_test_passed",
                "device": str(device),
                "samples": batch["sample_ids"],
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.detach().item()),
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    roots = args.dataset_root or [str(DEFAULT_DATASET_ROOT)]
    records = discover_samples(roots)
    print(f"Discovered {len(records)} successful hybrid samples.")
    device = torch.device(args.device)
    if args.smoke_test:
        smoke_test(records, config, device)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = grouped_split(
        records,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    write_split_manifest(output_dir / "splits.json", splits)
    print(
        "Split sizes: "
        + ", ".join(f"{name}={len(values)}" for name, values in splits.items())
    )
    normalization_path = output_dir / "normalization.json"
    if normalization_path.exists():
        stats = NormalizationStats.load(normalization_path)
    else:
        print("Calculating normalization from the training split only...")
        stats = NormalizationStats.calculate(splits["train"])
        stats.save(normalization_path)

    train_loader = make_loader(splits["train"], stats, config, shuffle=True)
    validation_loader = make_loader(splits["validation"], stats, config, shuffle=False)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    mixed_precision = bool(config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    target_mean, target_std = stats.target_tensors(device)
    history = []
    best_validation = float("inf")
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            target_mean,
            target_std,
            float(config["huber_beta"]),
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(config["gradient_clip_norm"]),
            mixed_precision=mixed_precision,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            target_mean,
            target_std,
            float(config["huber_beta"]),
            mixed_precision=mixed_precision,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = {
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "normalization_path": str(normalization_path),
        }
        _atomic_torch_save(checkpoint, output_dir / "latest.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            _atomic_torch_save(checkpoint, output_dir / "best.pt")
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_rmse={validation_metrics['rmse_in']:.5f} in "
            f"val_peak_mae={validation_metrics['peak_mae_in']:.5f} in"
        )


if __name__ == "__main__":
    main()
