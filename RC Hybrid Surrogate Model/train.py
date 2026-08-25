"""Train the baseline causal hybrid GNN-LSTM surrogate."""

from __future__ import annotations

import argparse
from datetime import datetime
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
    stratified_grouped_split,
    write_split_manifest,
)
from rc_hybrid_surrogate.losses import masked_smooth_l1
from rc_hybrid_surrogate.metrics import ResponseMetrics
from rc_hybrid_surrogate.model import HybridGNNLSTM
from rc_hybrid_surrogate.plotting import LossPlotter
from rc_hybrid_surrogate.features import EngineeredFeatureCache, load_group_intensity_scores


MODEL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_ROOT.parent
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "RC Structure" / "outputs" / "parameterized_2500"
DEFAULT_CONFIG = MODEL_ROOT / "configs" / "full_2500_physics10_stratified_lstm256_dropout012.json"
DEFAULT_OUTPUT_DIR = MODEL_ROOT / "outputs" / "full_2500_physics10_stratified_lstm256_dropout012_v1"
DEFAULT_ENGINEERED_FEATURE_CACHE = (
    MODEL_ROOT / "derived_features" / "engineered_features.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        help="Root containing hybrid_sample.npz files; repeat for multiple devices.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--engineered-feature-cache",
        default=str(DEFAULT_ENGINEERED_FEATURE_CACHE),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--live-plot",
        dest="live_plot",
        action="store_true",
        help="Make loss_curves.html auto-refresh while training.",
    )
    plot_group.add_argument(
        "--no-live-plot",
        dest="live_plot",
        action="store_false",
        help="Write loss_curves.html without automatic browser refresh.",
    )
    parser.set_defaults(live_plot=True)
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
        engineered_dim=int(config.get("engineered_feature_dim", 0)),
    )


def make_loader(
    records, stats, config, shuffle: bool, engineered_features=None
) -> DataLoader:
    dataset = HybridDataset(
        records,
        normalization=stats,
        engineered_features=engineered_features,
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


def smoke_test(records, config, device, engineered_features=None):
    selected = records[: min(2, len(records))]
    if not selected:
        raise RuntimeError("No valid hybrid samples were discovered.")
    stats = NormalizationStats.calculate(selected, engineered_features)
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
    loader = make_loader(
        selected,
        stats,
        smoke_config,
        shuffle=False,
        engineered_features=engineered_features,
    )
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


def _print_split_intensity_summary(splits, group_scores, engineered_feature_cache_path) -> None:
    """Best-effort diagnostic: is one split systematically more severe than train?

    With only a few dozen distinct ground-motion pairs, a plain random split
    can hand validation a subset that is much stronger or weaker than train
    purely by chance, which shows up as a validation-loss plateau that no
    amount of regularization fixes. This never blocks training.
    """
    if group_scores is None:
        try:
            group_scores = load_group_intensity_scores(engineered_feature_cache_path)
        except (OSError, ValueError) as error:
            print(f"Skipping split intensity summary: {error}")
            return

    groups_by_split = {
        name: {record.record_group for record in records}
        for name, records in splits.items()
    }
    if not groups_by_split.get("train"):
        return
    train_mean = sum(group_scores[group] for group in groups_by_split["train"]) / len(
        groups_by_split["train"]
    )
    print(
        "Ground-motion intensity score by split "
        "(0=mildest, 1=most severe of the discovered pairs):"
    )
    for name in ("train", "validation", "test"):
        groups = groups_by_split.get(name)
        if not groups:
            continue
        mean_score = sum(group_scores[group] for group in groups) / len(groups)
        flag = ""
        if name != "train" and abs(mean_score - train_mean) > 0.15:
            flag = (
                "  <- differs from train by more than 0.15; this split may be "
                "systematically easier or harder than train"
            )
        print(f"  {name}: groups={len(groups)} mean_score={mean_score:.3f}{flag}")


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    roots = args.dataset_root or [str(DEFAULT_DATASET_ROOT)]
    records = discover_samples(roots)
    print(f"Discovered {len(records)} successful hybrid samples.")
    engineered_features = None
    engineered_dimension = int(config.get("engineered_feature_dim", 0))
    if engineered_dimension > 0:
        engineered_features = EngineeredFeatureCache.load(args.engineered_feature_cache)
        selected_groups = config.get("engineered_feature_groups")
        if selected_groups:
            engineered_features = engineered_features.select_groups(selected_groups)
        if engineered_features.dimension != engineered_dimension:
            raise ValueError(
                "Configured engineered_feature_dim does not match the feature cache: "
                f"{engineered_dimension} != {engineered_features.dimension}."
            )
        missing = [
            record.sample_id
            for record in records
            if record.sample_id not in engineered_features.values
        ]
        if missing:
            raise ValueError(
                f"Engineered feature cache is missing {len(missing)} samples. "
                "Run build_engineered_features.py before training."
            )
        print(
            f"Loaded {engineered_features.dimension} engineered features "
            f"for {len(records)} samples"
            + (
                f" from groups: {', '.join(selected_groups)}."
                if selected_groups
                else "."
            )
        )
    device = torch.device(args.device)
    if args.smoke_test:
        smoke_test(records, config, device, engineered_features)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_scores = None
    if config.get("stratify_split", False):
        group_scores = load_group_intensity_scores(args.engineered_feature_cache)
        splits = stratified_grouped_split(
            records,
            group_scores,
            train_fraction=float(config["train_fraction"]),
            validation_fraction=float(config["validation_fraction"]),
            seed=int(config["seed"]),
            test_group_count=config.get("test_group_count"),
        )
    else:
        splits = grouped_split(
            records,
            train_fraction=float(config["train_fraction"]),
            validation_fraction=float(config["validation_fraction"]),
            seed=int(config["seed"]),
            test_group_count=config.get("test_group_count"),
        )
    write_split_manifest(output_dir / "splits.json", splits)
    print(
        "Split sizes: "
        + ", ".join(f"{name}={len(values)}" for name, values in splits.items())
    )
    _print_split_intensity_summary(splits, group_scores, args.engineered_feature_cache)
    normalization_path = output_dir / "normalization.json"
    if normalization_path.exists():
        stats = NormalizationStats.load(normalization_path)
    else:
        print("Calculating normalization from the training split only...")
        stats = NormalizationStats.calculate(splits["train"], engineered_features)
        stats.save(normalization_path)

    train_loader = make_loader(
        splits["train"],
        stats,
        config,
        shuffle=True,
        engineered_features=engineered_features,
    )
    validation_loader = make_loader(
        splits["validation"],
        stats,
        config,
        shuffle=False,
        engineered_features=engineered_features,
    )
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = None
    if config.get("lr_scheduler_patience") is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("lr_scheduler_factor", 0.5)),
            patience=int(config["lr_scheduler_patience"]),
            min_lr=float(config.get("minimum_learning_rate", 1.0e-6)),
        )
    mixed_precision = bool(config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    target_mean, target_std = stats.target_tensors(device)
    history = []
    best_validation = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
    loss_plotter = LossPlotter(output_dir / "loss_curves.html", live=args.live_plot)

    run_start_time = time.time()
    print(
        "Training run starting: "
        f"start_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"config={Path(args.config).name} "
        f"output_dir={output_dir} "
        f"device={device} "
        f"batch_size={config['batch_size']} "
        f"epochs={config['epochs']} (budget) "
        f"early_stopping_patience={early_stopping_patience}"
    )

    for epoch in range(1, int(config["epochs"]) + 1):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
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
            "learning_rate": learning_rate_used,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        loss_plotter.update(history)
        improved = validation_metrics["loss"] < best_validation
        if improved:
            best_validation = validation_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step(validation_metrics["loss"])
        checkpoint = {
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "normalization_path": str(normalization_path),
            "best_validation_loss": best_validation,
        }
        _atomic_torch_save(checkpoint, output_dir / "latest.pt")
        if improved:
            _atomic_torch_save(checkpoint, output_dir / "best.pt")
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_rmse={validation_metrics['rmse_in']:.5f} in "
            f"val_peak_mae={validation_metrics['peak_mae_in']:.5f} in "
            f"lr={next_learning_rate:.2e} "
            f"patience={epochs_without_improvement}/{early_stopping_patience}"
        )
        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "Early stopping: validation loss did not improve for "
                f"{early_stopping_patience} consecutive epochs. "
                f"Best validation loss={best_validation:.6f}."
            )
            break

    elapsed_seconds = time.time() - run_start_time
    print(
        "Training run finished: "
        f"end_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"elapsed={elapsed_seconds / 60.0:.1f} min "
        f"epochs_run={epoch} "
        f"best_validation_loss={best_validation:.6f}"
    )


if __name__ == "__main__":
    main()
