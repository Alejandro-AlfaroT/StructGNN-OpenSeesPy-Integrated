"""Reproducible PCA analysis of an exported RC parameter-space embedding.

Companion to ``RC Hybrid Surrogate Model/export_raw_embeddings.py``: reads the
``raw_vectors.tsv`` (+ sibling ``feature_names.txt`` / ``metadata.tsv``) that
script writes and reports what the TensorFlow Embedding Projector's PCA view
never exposes on its own -- explained variance, per-component loadings named
back to real feature columns, a target-leakage smoke test, and an optional
static scatter figure -- all reproducible from files already in the repo
instead of a browser session's memory.

Pass ``--features`` pointing at the RAW (unstandardized) export
(``raw_vectors.tsv``), not the pre-standardized ``vectors.tsv`` that gets
uploaded to the Projector: this script does its own standardization by
default so ``--no-standardize`` can meaningfully compare against it. Feeding
it an already-standardized file makes ``--no-standardize`` a no-op.

Example usage:

    python Analyze_Parameter_Space.py \\
        --features ".../structure_record/raw_vectors.tsv" --top-loadings 8

    python Analyze_Parameter_Space.py \\
        --features ".../structure_record/raw_vectors.tsv" \\
        --no-standardize --top-loadings 8

    python Analyze_Parameter_Space.py \\
        --features ".../structure_record/raw_vectors.tsv" \\
        --color-by num_floor --color-by peak_drift_percent \\
        --figure pca_projection.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t")


def _read_feature_names(path: Path, expected: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature-name file not found: {path}. Pass --feature-names explicitly "
            "if it isn't named feature_names.txt next to --features."
        )
    names = path.read_text(encoding="utf-8").splitlines()
    if len(names) != expected:
        raise ValueError(
            f"{path} has {len(names)} names but the feature matrix has "
            f"{expected} columns."
        )
    return names


def _read_metadata(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _coerce_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1.0e-12] = 1.0
    return (matrix - mean) / std


def pca(matrix: np.ndarray, components: int):
    """Mean-centered PCA via SVD. Returns (coordinates, explained_ratio, loadings)."""
    centered = matrix - matrix.mean(axis=0)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    total = variance.sum()
    explained = variance / total if total > 0.0 else np.zeros_like(variance)
    coordinates = centered @ right[:components].T
    return coordinates, explained, right[:components]


def components_for_variance(
    explained: np.ndarray, thresholds=(0.90, 0.95, 0.99)
) -> dict[str, int]:
    cumulative = np.cumsum(explained)
    return {
        f"{threshold:.0%}": int(np.searchsorted(cumulative, threshold) + 1)
        for threshold in thresholds
    }


def leakage_check(
    matrix: np.ndarray,
    feature_names: list[str],
    metadata: list[dict],
    target_column: str,
    top_n: int = 5,
):
    if not metadata or target_column not in metadata[0]:
        return None
    target = np.array([_coerce_float(row[target_column]) for row in metadata])
    if np.all(np.isnan(target)) or np.nanstd(target) == 0.0:
        return []
    correlations = np.array(
        [np.corrcoef(matrix[:, i], target)[0, 1] for i in range(matrix.shape[1])]
    )
    order = np.argsort(-np.abs(correlations))[:top_n]
    return [(feature_names[i], float(correlations[i])) for i in order]


def print_loadings(feature_names: list[str], loadings: np.ndarray, top_n: int) -> None:
    for index, component in enumerate(loadings):
        ranked = np.argsort(-np.abs(component))[:top_n]
        print(f"PC{index + 1} top {top_n} loadings:")
        for i in ranked:
            print(f"    {feature_names[i]:35s} {component[i]:+.3f}")


def make_figure(
    coordinates: np.ndarray,
    metadata: list[dict],
    color_by: list[str],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metadata:
        raise ValueError("--color-by requires a metadata.tsv with matching rows.")
    available = sorted(metadata[0].keys())
    missing = [column for column in color_by if column not in available]
    if missing:
        raise ValueError(
            f"--color-by column(s) not found in metadata: {missing}. "
            f"Available columns: {available}"
        )

    figure, axes = plt.subplots(
        1, len(color_by), figsize=(6 * len(color_by), 5), squeeze=False
    )
    for axis, column in zip(axes[0], color_by):
        values = np.array([_coerce_float(row[column]) for row in metadata])
        scatter = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=values, s=6, cmap="viridis"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_title(f"colored by {column}")
        figure.colorbar(scatter, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Raw (unstandardized), tab-separated feature matrix with no header "
        "(e.g. raw_vectors.tsv from export_raw_embeddings.py).",
    )
    parser.add_argument(
        "--feature-names",
        default=None,
        help="One name per line, same order as the matrix columns. Defaults to "
        "feature_names.txt next to --features.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Tab-separated metadata, one row per sample in the same order as "
        "--features. Defaults to metadata.tsv next to --features; omitted "
        "entirely if that file doesn't exist.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Run PCA on raw values instead of z-scored features, to compare "
        "against the standardized run.",
    )
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--top-loadings", type=int, default=8)
    parser.add_argument(
        "--leak-check",
        default="peak_drift_percent",
        help="Metadata column to correlate every feature against as a target-"
        "leakage smoke test. Empty string disables.",
    )
    parser.add_argument(
        "--color-by",
        action="append",
        default=None,
        help="Metadata column to color a PC1-vs-PC2 scatter panel by; repeat "
        "for multiple panels. Requires --figure.",
    )
    parser.add_argument(
        "--figure",
        default=None,
        help="Output image path for the --color-by scatter figure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features_path = Path(args.features)
    matrix = _read_matrix(features_path)

    names_path = (
        Path(args.feature_names)
        if args.feature_names
        else features_path.with_name("feature_names.txt")
    )
    feature_names = _read_feature_names(names_path, matrix.shape[1])

    metadata_path = (
        Path(args.metadata) if args.metadata else features_path.with_name("metadata.tsv")
    )
    metadata = _read_metadata(metadata_path)

    working = matrix if args.no_standardize else standardize(matrix)
    print(
        f"{matrix.shape[0]} samples x {matrix.shape[1]} features "
        f"({'raw scale' if args.no_standardize else 'standardized'})"
    )

    components = max(args.components, 2)
    coordinates, explained, loadings = pca(working, components)
    print("Explained variance ratio:", np.round(explained[: args.components], 4).tolist())
    print(
        "Cumulative:",
        np.round(np.cumsum(explained)[: args.components], 4).tolist(),
    )
    print("Components needed for variance thresholds:", components_for_variance(explained))
    print()
    print_loadings(feature_names, loadings[: args.components], args.top_loadings)

    if args.leak_check:
        leak = leakage_check(working, feature_names, metadata, args.leak_check)
        if leak is None:
            print(f"\nSkipping leakage check: '{args.leak_check}' not found in metadata.")
        elif not leak:
            print(f"\nSkipping leakage check: '{args.leak_check}' has no variance.")
        else:
            print(f"\nLeakage check -- top |correlation| with '{args.leak_check}':")
            for name, r in leak:
                print(f"  {name:35s} r={r:+.3f}")

    if args.figure:
        if not args.color_by:
            raise SystemExit("--figure requires at least one --color-by column.")
        make_figure(coordinates, metadata, args.color_by, Path(args.figure))
    elif args.color_by:
        raise SystemExit("--color-by requires --figure to write the scatter to.")


if __name__ == "__main__":
    main()
