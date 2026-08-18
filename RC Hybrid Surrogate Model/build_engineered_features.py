"""Build derived motion/modal features without modifying OpenSees outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rc_hybrid_surrogate.data import discover_samples
from rc_hybrid_surrogate.features import build_engineered_feature_cache


MODEL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_ROOT.parent
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "RC Structure" / "outputs" / "parameterized_2500"
DEFAULT_OUTPUT = MODEL_ROOT / "derived_features" / "engineered_features.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        help="Root containing hybrid_sample.npz files; repeat for multiple roots.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    roots = args.dataset_root or [str(DEFAULT_DATASET_ROOT)]
    records = discover_samples(roots)
    if not records:
        raise RuntimeError("No valid hybrid samples were discovered.")
    summary = build_engineered_feature_cache(records, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
