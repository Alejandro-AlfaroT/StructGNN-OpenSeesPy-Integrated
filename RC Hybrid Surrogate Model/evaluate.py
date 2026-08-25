"""Evaluate a trained hybrid surrogate on its frozen held-out test split."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rc_hybrid_surrogate.data import (
    HybridDataset,
    NormalizationStats,
    collate_hybrid,
    discover_samples,
    move_batch,
)
from rc_hybrid_surrogate.features import (
    EngineeredFeatureCache,
    load_group_intensity_scores,
)
from rc_hybrid_surrogate.model import HybridGNNLSTM


MODEL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_ROOT.parent
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "RC Structure" / "outputs" / "parameterized_2500"
DEFAULT_RUN_DIR = (
    MODEL_ROOT
    / "outputs"
    / "full_2500_physics10_stratified_lstm256_dropout012_v1"
)
DEFAULT_ENGINEERED_FEATURE_CACHE = (
    MODEL_ROOT / "derived_features" / "engineered_features.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to evaluate; defaults to <run-dir>/best.pt.",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        help="Current root containing hybrid_sample.npz files; repeat if needed.",
    )
    parser.add_argument(
        "--engineered-feature-cache",
        default=str(DEFAULT_ENGINEERED_FEATURE_CACHE),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination; defaults to <run-dir>/test_evaluation.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


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


def load_frozen_test_records(split_path: Path, dataset_roots) -> tuple[list, dict]:
    """Resolve saved test IDs against the current dataset without regenerating a split."""
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_rows = manifest.get("splits") or {}
    required = {"train", "validation", "test"}
    if not required.issubset(split_rows):
        raise ValueError(f"Split manifest is missing: {sorted(required - set(split_rows))}")

    groups = {
        name: {str(row["record_group"]) for row in split_rows[name]}
        for name in required
    }
    if (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    ):
        raise ValueError("Saved split manifest contains record-group leakage.")

    test_ids = [str(row["sample_id"]) for row in split_rows["test"]]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Saved test split contains duplicate sample IDs.")
    discovered = {record.sample_id: record for record in discover_samples(dataset_roots)}
    missing = [sample_id for sample_id in test_ids if sample_id not in discovered]
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} saved test samples under the supplied "
            f"dataset root(s), including {missing[:3]}."
        )
    records = [discovered[sample_id] for sample_id in test_ids]
    return records, manifest


def _safe_relative_error(error: float, reference: float) -> float:
    return 100.0 * error / reference if abs(reference) > 1.0e-8 else math.nan


def calculate_case_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    normalized_prediction: np.ndarray,
    normalized_target: np.ndarray,
    huber_beta: float,
) -> dict[str, float]:
    """Calculate physical-unit and normalized-objective metrics for one case."""
    error = prediction - target
    peak_prediction = np.max(np.abs(prediction), axis=0)
    peak_target = np.max(np.abs(target), axis=0)
    peak_error = np.abs(peak_prediction - peak_target)
    resultant_prediction = np.linalg.norm(prediction, axis=1)
    resultant_target = np.linalg.norm(target, axis=1)
    peak_resultant_prediction = float(np.max(resultant_prediction))
    peak_resultant_target = float(np.max(resultant_target))
    peak_resultant_error = abs(peak_resultant_prediction - peak_resultant_target)

    difference = torch.from_numpy(normalized_prediction - normalized_target)
    absolute = difference.abs()
    beta = float(huber_beta)
    huber = torch.where(
        absolute < beta,
        0.5 * difference.square() / beta,
        absolute - 0.5 * beta,
    )
    return {
        "loss": float(huber.mean().item()),
        "rmse_in": float(np.sqrt(np.mean(np.square(error)))),
        "mae_in": float(np.mean(np.abs(error))),
        "rmse_x_in": float(np.sqrt(np.mean(np.square(error[:, 0])))),
        "rmse_y_in": float(np.sqrt(np.mean(np.square(error[:, 1])))),
        "mae_x_in": float(np.mean(np.abs(error[:, 0]))),
        "mae_y_in": float(np.mean(np.abs(error[:, 1]))),
        "true_peak_x_in": float(peak_target[0]),
        "predicted_peak_x_in": float(peak_prediction[0]),
        "peak_error_x_in": float(peak_error[0]),
        "peak_error_x_percent": _safe_relative_error(peak_error[0], peak_target[0]),
        "true_peak_y_in": float(peak_target[1]),
        "predicted_peak_y_in": float(peak_prediction[1]),
        "peak_error_y_in": float(peak_error[1]),
        "peak_error_y_percent": _safe_relative_error(peak_error[1], peak_target[1]),
        "true_peak_resultant_in": peak_resultant_target,
        "predicted_peak_resultant_in": peak_resultant_prediction,
        "peak_error_resultant_in": peak_resultant_error,
        "peak_error_resultant_percent": _safe_relative_error(
            peak_resultant_error, peak_resultant_target
        ),
    }


def _metadata_for_record(record, engineered_features, intensity_scores) -> dict:
    metadata_path = record.path.with_name("hybrid_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(record.path, allow_pickle=False) as sample:
        global_values = np.asarray(sample["global_features"], dtype=np.float64)
        record_values = np.asarray(sample["record_features"], dtype=np.float64)
    global_names = list(metadata.get("global_feature_keys") or [])
    record_names = list(metadata.get("record_feature_keys") or [])
    globals_by_name = dict(zip(global_names, global_values))
    x_record = dict(zip(record_names, record_values[0]))
    y_record = dict(zip(record_names, record_values[1]))
    engineered = {}
    if engineered_features is not None:
        engineered = dict(
            zip(
                engineered_features.feature_names,
                engineered_features.vector(record.sample_id),
            )
        )
    return {
        "sample_id": record.sample_id,
        "case_id": record.case_id,
        "run_name": record.run_name,
        "record_group": record.record_group,
        "record_id_x": record.record_id_x,
        "record_id_y": record.record_id_y,
        "num_stories": int(round(globals_by_name.get("num_floor", math.nan))),
        "num_bay_x": int(round(globals_by_name.get("num_bay_x", math.nan))),
        "num_bay_y": int(round(globals_by_name.get("num_bay_y", math.nan))),
        "story_height_ft": float(globals_by_name.get("story_h_in", math.nan)) / 12.0,
        "period_x_sec": float(engineered.get("period_x_mode_1_sec", math.nan)),
        "period_y_sec": float(engineered.get("period_y_mode_1_sec", math.nan)),
        "pga_x_g": float(x_record.get("pga_g", math.nan)),
        "pga_y_g": float(y_record.get("pga_g", math.nan)),
        "duration_sec": float(max(x_record.get("duration_sec", 0.0), y_record.get("duration_sec", 0.0))),
        "intensity_score": float(intensity_scores.get(record.record_group, math.nan)),
        "source_path": str(record.path),
    }


def _aggregate_rows(rows: list[dict]) -> dict[str, float | int]:
    return {
        "case_count": len(rows),
        "mean_rmse_in": float(np.mean([row["rmse_in"] for row in rows])),
        "median_rmse_in": float(np.median([row["rmse_in"] for row in rows])),
        "p90_rmse_in": float(np.percentile([row["rmse_in"] for row in rows], 90)),
        "mean_mae_in": float(np.mean([row["mae_in"] for row in rows])),
        "mean_peak_error_in": float(
            np.mean(
                [
                    0.5 * (row["peak_error_x_in"] + row["peak_error_y_in"])
                    for row in rows
                ]
            )
        ),
        "mean_resultant_peak_error_in": float(
            np.mean([row["peak_error_resultant_in"] for row in rows])
        ),
    }


def _group_rows(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return [
        {"group_by": key, "group": value, **_aggregate_rows(group_rows)}
        for value, group_rows in sorted(groups.items())
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17202a}.axis{stroke:#566573;stroke-width:1}'
        '.grid{stroke:#d5d8dc;stroke-width:1}.truth{stroke:#1f618d;fill:none;stroke-width:1.5}'
        '.prediction{stroke:#c0392b;fill:none;stroke-width:1.25}.point{fill:#2874a6;fill-opacity:.7}</style>'
        f"{body}</svg>"
    )


def _parity_plot(rows: list[dict], output_path: Path) -> None:
    width, height = 960, 440
    pieces = ['<text x="480" y="28" text-anchor="middle" font-size="20">Held-out test peak-displacement parity</text>']
    for panel, direction in enumerate(("x", "y")):
        actual = np.asarray([row[f"true_peak_{direction}_in"] for row in rows])
        predicted = np.asarray([row[f"predicted_peak_{direction}_in"] for row in rows])
        maximum = max(float(actual.max()), float(predicted.max()), 1.0e-6) * 1.05
        left, top, size = 75 + panel * 470, 65, 320
        bottom = top + size
        pieces.extend(
            [
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + size}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                f'<line x1="{left}" y1="{bottom}" x2="{left + size}" y2="{top}" stroke="#707b7c" stroke-dasharray="5 4"/>',
                f'<text x="{left + size / 2}" y="{top - 14}" text-anchor="middle" font-size="17">Roof {direction.upper()}</text>',
                f'<text x="{left + size / 2}" y="{bottom + 38}" text-anchor="middle" font-size="13">OpenSees peak (in)</text>',
                f'<text x="{left - 48}" y="{top + size / 2}" text-anchor="middle" font-size="13" transform="rotate(-90 {left - 48} {top + size / 2})">Predicted peak (in)</text>',
                f'<text x="{left}" y="{bottom + 18}" text-anchor="middle" font-size="11">0</text>',
                f'<text x="{left + size}" y="{bottom + 18}" text-anchor="middle" font-size="11">{maximum:.2f}</text>',
            ]
        )
        for x_value, y_value in zip(actual, predicted):
            x = left + size * float(x_value) / maximum
            y = bottom - size * float(y_value) / maximum
            pieces.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="3.5"/>')
    output_path.write_text(_svg_document(width, height, "".join(pieces)), encoding="utf-8")


def _story_error_plot(rows: list[dict], output_path: Path) -> None:
    width, height = 820, 460
    stories = sorted({int(row["num_stories"]) for row in rows})
    maximum = max(float(row["rmse_in"]) for row in rows) * 1.08
    left, top, plot_width, plot_height = 80, 55, 700, 330
    bottom = top + plot_height
    pieces = [
        '<text x="410" y="28" text-anchor="middle" font-size="20">Test error by building height</text>',
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<text x="{left + plot_width / 2}" y="{bottom + 52}" text-anchor="middle" font-size="14">Number of stories</text>',
        f'<text x="25" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 25 {top + plot_height / 2})">Per-case RMSE (in)</text>',
    ]
    for index, story in enumerate(stories):
        x = left + (index + 0.5) * plot_width / len(stories)
        values = [float(row["rmse_in"]) for row in rows if int(row["num_stories"]) == story]
        pieces.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" font-size="12">{story}</text>')
        for item_index, value in enumerate(values):
            jitter = ((item_index % 7) - 3) * 3.0
            y = bottom - plot_height * value / maximum
            pieces.append(f'<circle class="point" cx="{x + jitter:.2f}" cy="{y:.2f}" r="3.2"/>')
        mean_y = bottom - plot_height * float(np.mean(values)) / maximum
        pieces.append(f'<line x1="{x - 24:.2f}" y1="{mean_y:.2f}" x2="{x + 24:.2f}" y2="{mean_y:.2f}" stroke="#c0392b" stroke-width="3"/>')
    pieces.append(f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="11">{maximum:.2f}</text>')
    output_path.write_text(_svg_document(width, height, "".join(pieces)), encoding="utf-8")


def _history_plot(result: dict, label: str, output_path: Path, stride: int) -> None:
    record = result["record"]
    metadata_path = record.path.with_name("hybrid_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(record.path, allow_pickle=False) as sample:
        record_features = np.asarray(sample["record_features"], dtype=np.float64)
    feature_names = list(metadata.get("record_feature_keys") or [])
    dt_index = feature_names.index("dt_sec") if "dt_sec" in feature_names else None
    dt = float(record_features[0, dt_index]) if dt_index is not None else 1.0
    time = np.arange(result["target"].shape[0]) * dt * stride

    width, height = 1050, 620
    left, plot_width, plot_height = 80, 930, 220
    sample_indices = np.linspace(0, len(time) - 1, min(len(time), 1200), dtype=int)
    pieces = [
        f'<text x="{width / 2}" y="27" text-anchor="middle" font-size="19">{html.escape(label.title())} test case: {html.escape(record.sample_id)} — RMSE {result["metrics"]["rmse_in"]:.3f} in</text>',
        '<line class="truth" x1="790" y1="48" x2="825" y2="48"/><text x="832" y="52" font-size="12">OpenSees</text>',
        '<line class="prediction" x1="905" y1="48" x2="940" y2="48"/><text x="947" y="52" font-size="12">Surrogate</text>',
    ]
    for direction, top in enumerate((70, 330)):
        bottom = top + plot_height
        values = np.concatenate((result["target"][:, direction], result["prediction"][:, direction]))
        limit = max(float(np.max(np.abs(values))) * 1.08, 1.0e-6)
        pieces.extend(
            [
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                f'<line class="grid" x1="{left}" y1="{top + plot_height / 2}" x2="{left + plot_width}" y2="{top + plot_height / 2}"/>',
                f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-size="13" transform="rotate(-90 24 {top + plot_height / 2})">Roof {("X", "Y")[direction]} (in)</text>',
                f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="10">{limit:.2f}</text>',
                f'<text x="{left - 8}" y="{bottom}" text-anchor="end" font-size="10">{-limit:.2f}</text>',
            ]
        )
        for values_array, css_class in ((result["target"][:, direction], "truth"), (result["prediction"][:, direction], "prediction")):
            coordinates = []
            for index in sample_indices:
                x = left + plot_width * index / max(len(time) - 1, 1)
                y = top + plot_height / 2 - (plot_height / 2) * float(values_array[index]) / limit
                coordinates.append(f"{x:.2f},{y:.2f}")
            pieces.append(f'<polyline class="{css_class}" points="{" ".join(coordinates)}"/>')
    pieces.append(f'<text x="{left + plot_width / 2}" y="600" text-anchor="middle" font-size="14">Time (s), 0–{time[-1]:.1f}</text>')
    output_path.write_text(_svg_document(width, height, "".join(pieces)), encoding="utf-8")


def _write_html_report(output_path: Path, summary: dict, rows: list[dict], examples: list[tuple[str, str]]) -> None:
    metric_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>"
        for label, value in (
            ("Cases", summary["test_cases"]),
            ("Record groups", summary["test_record_groups"]),
            ("Checkpoint epoch", summary["checkpoint_epoch"]),
            ("Huber loss", f"{summary['metrics']['loss']:.6f}"),
            ("RMSE", f"{summary['metrics']['rmse_in']:.4f} in"),
            ("MAE", f"{summary['metrics']['mae_in']:.4f} in"),
            ("Peak MAE", f"{summary['metrics']['peak_mae_in']:.4f} in"),
            ("Resultant peak MAE", f"{summary['metrics']['resultant_peak_mae_in']:.4f} in"),
        )
    )
    worst = sorted(rows, key=lambda row: row["rmse_in"], reverse=True)[:10]
    worst_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td><td>{row['num_stories']}</td>"
        f"<td>{row['rmse_in']:.4f}</td><td>{row['mae_in']:.4f}</td>"
        f"<td>{row['peak_error_resultant_in']:.4f}</td></tr>"
        for row in worst
    )
    example_html = "".join(
        f"<h3>{html.escape(label.title())} case</h3><img src=\"{html.escape(filename)}\">"
        for label, filename in examples
    )
    output_path.write_text(
        f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Held-out test evaluation</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;margin:16px 0}}th,td{{border:1px solid #ccd1d1;padding:7px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #ddd;margin-bottom:18px}}.note{{background:#f4f6f7;padding:12px 16px;border-left:4px solid #2874a6}}</style></head><body>
<h1>Held-out test evaluation</h1><p class=\"note\">This report evaluates the frozen test split. Do not tune model choices against these cases.</p>
<table>{metric_rows}</table>
<h2>Peak predictions</h2><img src=\"peak_parity.svg\"><h2>Error by stories</h2><img src=\"error_by_stories.svg\">
<h2>Representative histories</h2>{example_html}
<h2>Ten highest-RMSE cases</h2><table><tr><th>Sample</th><th>Stories</th><th>RMSE (in)</th><th>MAE (in)</th><th>Resultant peak error (in)</th></tr>{worst_rows}</table>
<p>See <code>per_case_metrics.csv</code>, <code>grouped_metrics.csv</code>, and <code>summary.json</code> for machine-readable results.</p></body></html>""",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "best.pt"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "test_evaluation"
    roots = args.dataset_root or [str(DEFAULT_DATASET_ROOT)]
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    test_records, manifest = load_frozen_test_records(run_dir / "splits.json", roots)
    print(f"Resolved frozen test split: {len(test_records)} cases.")

    engineered_features = None
    engineered_dimension = int(config.get("engineered_feature_dim", 0))
    if engineered_dimension:
        engineered_features = EngineeredFeatureCache.load(args.engineered_feature_cache)
        groups = config.get("engineered_feature_groups")
        if groups:
            engineered_features = engineered_features.select_groups(groups)
        if engineered_features.dimension != engineered_dimension:
            raise ValueError(
                "Checkpoint engineered-feature dimension does not match the cache: "
                f"{engineered_dimension} != {engineered_features.dimension}."
            )

    stats = NormalizationStats.load(run_dir / "normalization.json")
    dataset = HybridDataset(
        test_records,
        normalization=stats,
        engineered_features=engineered_features,
        sequence_stride=int(config["sequence_stride"]),
        maximum_steps=config.get("maximum_steps"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("data_loader_workers", 0)),
        collate_fn=collate_hybrid,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_mean, target_std = stats.target_tensors(device)
    intensity_scores = load_group_intensity_scores(args.engineered_feature_cache)

    case_rows = []
    inference_results = []
    all_predictions = []
    all_targets = []
    all_normalized_predictions = []
    all_normalized_targets = []
    mixed_precision = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    record_by_id = {record.sample_id: record for record in test_records}
    with torch.inference_mode():
        for batch in loader:
            moved = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision):
                normalized_prediction = model(moved)
            for index, sample_id in enumerate(batch["sample_ids"]):
                length = int(batch["lengths"][index].item())
                normalized_pred = normalized_prediction[index, :length].float()
                normalized_true = moved["target"][index, :length].float()
                prediction = normalized_pred * target_std + target_mean
                target = normalized_true * target_std + target_mean
                prediction_np = prediction.cpu().numpy().astype(np.float64)
                target_np = target.cpu().numpy().astype(np.float64)
                normalized_pred_np = normalized_pred.cpu().numpy().astype(np.float64)
                normalized_true_np = normalized_true.cpu().numpy().astype(np.float64)
                record = record_by_id[sample_id]
                metrics = calculate_case_metrics(
                    prediction_np,
                    target_np,
                    normalized_pred_np,
                    normalized_true_np,
                    float(config["huber_beta"]),
                )
                row = {
                    **_metadata_for_record(record, engineered_features, intensity_scores),
                    "time_steps": length,
                    **metrics,
                }
                case_rows.append(row)
                inference_results.append(
                    {"record": record, "prediction": prediction_np, "target": target_np, "metrics": metrics}
                )
                all_predictions.append(prediction_np)
                all_targets.append(target_np)
                all_normalized_predictions.append(normalized_pred_np)
                all_normalized_targets.append(normalized_true_np)

    prediction = np.concatenate(all_predictions)
    target = np.concatenate(all_targets)
    normalized_prediction = np.concatenate(all_normalized_predictions)
    normalized_target = np.concatenate(all_normalized_targets)
    error = prediction - target
    normalized_error = normalized_prediction - normalized_target
    absolute = np.abs(normalized_error)
    beta = float(config["huber_beta"])
    normalized_huber = np.where(absolute < beta, 0.5 * normalized_error**2 / beta, absolute - 0.5 * beta)
    peak_errors = [0.5 * (row["peak_error_x_in"] + row["peak_error_y_in"]) for row in case_rows]
    summary = {
        "evaluation_time": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "run_directory": str(run_dir),
        "device": str(device),
        "test_cases": len(case_rows),
        "test_record_groups": len({row["record_group"] for row in case_rows}),
        "test_sample_ids": [row["sample_id"] for row in case_rows],
        "split_grouping": manifest.get("grouping"),
        "metrics": {
            "loss": float(np.mean(normalized_huber)),
            "rmse_in": float(np.sqrt(np.mean(np.square(error)))),
            "mae_in": float(np.mean(np.abs(error))),
            "rmse_x_in": float(np.sqrt(np.mean(np.square(error[:, 0])))),
            "rmse_y_in": float(np.sqrt(np.mean(np.square(error[:, 1])))),
            "mae_x_in": float(np.mean(np.abs(error[:, 0]))),
            "mae_y_in": float(np.mean(np.abs(error[:, 1]))),
            "peak_mae_in": float(np.mean(peak_errors)),
            "resultant_peak_mae_in": float(np.mean([row["peak_error_resultant_in"] for row in case_rows])),
            "median_case_rmse_in": float(np.median([row["rmse_in"] for row in case_rows])),
            "p90_case_rmse_in": float(np.percentile([row["rmse_in"] for row in case_rows], 90)),
            "worst_case_rmse_in": float(max(row["rmse_in"] for row in case_rows)),
        },
    }

    case_rows.sort(key=lambda row: row["sample_id"])
    grouped_rows = _group_rows(case_rows, "num_stories") + _group_rows(case_rows, "record_group")
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    _write_csv(output_dir / "grouped_metrics.csv", grouped_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _parity_plot(case_rows, output_dir / "peak_parity.svg")
    _story_error_plot(case_rows, output_dir / "error_by_stories.svg")
    ranked = sorted(inference_results, key=lambda result: result["metrics"]["rmse_in"])
    selected = [("best", ranked[0]), ("median", ranked[len(ranked) // 2]), ("worst", ranked[-1])]
    examples = []
    for label, result in selected:
        filename = f"history_{label}.svg"
        _history_plot(result, label, output_dir / filename, int(config["sequence_stride"]))
        examples.append((label, filename))
    _write_html_report(output_dir / "report.html", summary, case_rows, examples)

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
