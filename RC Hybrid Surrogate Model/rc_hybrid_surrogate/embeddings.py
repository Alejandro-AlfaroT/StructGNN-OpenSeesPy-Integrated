"""Raw parameter-space exports for TensorBoard's Embedding Projector."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import SampleRecord, grouped_split


TARGET_PEAK_FALLBACK = [
    "max_abs_roof_disp_x_in",
    "max_abs_roof_disp_y_in",
    "max_story_drift_resultant_ratio",
    "max_story_drift_resultant_story",
    "analysis_failed",
    "completed_fraction",
    "num_recovery_steps",
    "recovery_rate",
]


def _clean_tsv(value) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _value_map(names, values) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, values)}


def load_raw_parameter_space(records: Sequence[SampleRecord]):
    """Return structural/case matrices, feature names, and rich case metadata."""
    structural_rows = []
    case_rows = []
    metadata_rows = []
    structural_names = None
    case_names = None

    splits = grouped_split(records)
    split_by_id = {
        record.sample_id: split
        for split, split_records in splits.items()
        for record in split_records
    }

    for record in records:
        metadata_path = record.path.with_name("hybrid_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(record.path, allow_pickle=False) as sample:
            structural = np.asarray(sample["global_features"], dtype=np.float64)
            record_features = np.asarray(sample["record_features"], dtype=np.float64)
            targets = np.asarray(sample["target_peak"], dtype=np.float64)

        global_names = list(metadata.get("global_feature_keys") or [])
        record_names = list(metadata.get("record_feature_keys") or [])
        target_names = list(metadata.get("target_peak_columns") or TARGET_PEAK_FALLBACK)
        if len(global_names) != structural.size:
            raise ValueError(f"Global feature-name mismatch in {record.path}")
        if record_features.shape != (2, len(record_names)):
            raise ValueError(f"Record feature-name mismatch in {record.path}")
        current_case_names = global_names + [
            f"record_x_{name}" for name in record_names
        ] + [f"record_y_{name}" for name in record_names]
        if structural_names is None:
            structural_names = global_names
            case_names = current_case_names
        elif structural_names != global_names or case_names != current_case_names:
            raise ValueError(f"Feature schema differs in {record.path}")

        structural_rows.append(structural)
        case_rows.append(np.concatenate((structural, record_features.reshape(-1))))
        global_values = _value_map(global_names, structural)
        record_x = _value_map(record_names, record_features[0])
        record_y = _value_map(record_names, record_features[1])
        target_values = _value_map(target_names, targets)
        metadata_rows.append(
            {
                "sample_id": record.sample_id,
                "case_id": record.case_id,
                "run_name": record.run_name,
                "split": split_by_id[record.sample_id],
                "record_id_x": record.record_id_x,
                "record_id_y": record.record_id_y,
                "num_floor": int(round(global_values.get("num_floor", 0))),
                "num_bay_x": int(round(global_values.get("num_bay_x", 0))),
                "num_bay_y": int(round(global_values.get("num_bay_y", 0))),
                "bay_x_in": global_values.get("bay_x_in", np.nan),
                "bay_y_in": global_values.get("bay_y_in", np.nan),
                "story_h_in": global_values.get("story_h_in", np.nan),
                "pga_x_g": record_x.get("pga_g", np.nan),
                "pga_y_g": record_y.get("pga_g", np.nan),
                "duration_x_sec": record_x.get("duration_sec", np.nan),
                "duration_y_sec": record_y.get("duration_sec", np.nan),
                "scale_factor_x": record_x.get("scale_factor", np.nan),
                "scale_factor_y": record_y.get("scale_factor", np.nan),
                "peak_roof_x_in": target_values.get("max_abs_roof_disp_x_in", np.nan),
                "peak_roof_y_in": target_values.get("max_abs_roof_disp_y_in", np.nan),
                "peak_drift_ratio": target_values.get(
                    "max_story_drift_resultant_ratio", np.nan
                ),
                "peak_drift_percent": 100.0
                * target_values.get("max_story_drift_resultant_ratio", np.nan),
                "recovery_steps": int(
                    round(target_values.get("num_recovery_steps", 0.0))
                ),
                "recovery_rate": target_values.get("recovery_rate", np.nan),
                "source_path": str(record.path),
            }
        )

    return (
        np.vstack(structural_rows),
        np.vstack(case_rows),
        structural_names,
        case_names,
        metadata_rows,
    )


def standardize(matrix: np.ndarray):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1.0e-12] = 1.0
    return (matrix - mean) / std, mean, std


def principal_components(matrix: np.ndarray, components: int = 3):
    centered = matrix - matrix.mean(axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(centered)).to(torch.float64)
    _, singular_values, right = torch.linalg.svd(tensor, full_matrices=False)
    coordinates = tensor @ right[:components].T
    variance = torch.square(singular_values)
    denominator = variance.sum()
    explained = (
        variance[:components] / denominator
        if float(denominator) > 0.0
        else torch.zeros(components, dtype=torch.float64)
    )
    return coordinates.numpy(), explained.numpy(), right[:components].numpy()


def _write_vectors(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter="\t", fmt="%.8g")


def _write_metadata(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _clean_tsv(value) for key, value in row.items()})


def export_parameter_space(records: Sequence[SampleRecord], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    structural, combined, structural_names, case_names, metadata = load_raw_parameter_space(
        records
    )
    summary = {"sample_count": len(records), "spaces": {}}
    for name, matrix, feature_names in (
        ("structure", structural, structural_names),
        ("structure_record", combined, case_names),
    ):
        destination = output_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        standardized, mean, std = standardize(matrix)
        coordinates, explained, loadings = principal_components(standardized)
        _write_vectors(destination / "vectors.tsv", standardized)
        _write_vectors(destination / "raw_vectors.tsv", matrix)
        _write_metadata(destination / "metadata.tsv", metadata)
        (destination / "feature_names.txt").write_text(
            "\n".join(feature_names) + "\n", encoding="utf-8"
        )
        (destination / "standardization.json").write_text(
            json.dumps(
                {"feature_names": feature_names, "mean": mean.tolist(), "std": std.tolist()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        pca_rows = []
        for row, coordinate in zip(metadata, coordinates):
            pca_rows.append(
                {
                    **{key: value for key, value in row.items() if key != "source_path"},
                    "pc1": coordinate[0],
                    "pc2": coordinate[1],
                    "pc3": coordinate[2],
                }
            )
        with (destination / "pca_coordinates.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(pca_rows[0]))
            writer.writeheader()
            writer.writerows(pca_rows)
        with (destination / "pca_loadings.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=["feature", "pc1", "pc2", "pc3"])
            writer.writeheader()
            for feature, loading in zip(feature_names, loadings.T):
                writer.writerow(
                    {
                        "feature": feature,
                        "pc1": loading[0],
                        "pc2": loading[1],
                        "pc3": loading[2],
                    }
                )
        summary["spaces"][name] = {
            "dimensions": int(matrix.shape[1]),
            "explained_variance_ratio": explained.tolist(),
            "directory": str(destination),
        }
    (output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
