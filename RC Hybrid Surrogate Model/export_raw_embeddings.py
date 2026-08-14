"""Export raw RC parameter spaces for TensorBoard's Embedding Projector."""

import argparse
import json
from pathlib import Path

from rc_hybrid_surrogate.data import discover_samples
from rc_hybrid_surrogate.embeddings import export_parameter_space


MODEL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        help="Root containing hybrid samples; repeat for non-overlapping devices.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODEL_ROOT / "outputs" / "raw_parameter_embeddings"),
    )
    args = parser.parse_args()
    roots = args.dataset_root or [
        str(REPOSITORY_ROOT / "RC Structure" / "outputs" / "parameterized_2500")
    ]
    records = discover_samples(roots)
    if not records:
        raise RuntimeError("No successful hybrid samples were discovered.")
    summary = export_parameter_space(records, args.output_dir)
    print(json.dumps(summary, indent=2))
    print("Load vectors.tsv and metadata.tsv from either space into projector.tensorflow.org.")


if __name__ == "__main__":
    main()
