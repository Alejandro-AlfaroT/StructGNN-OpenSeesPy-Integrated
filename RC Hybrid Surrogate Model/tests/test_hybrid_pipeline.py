import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from rc_hybrid_surrogate.data import (
    HybridDataset,
    NormalizationStats,
    collate_hybrid,
    discover_samples,
    grouped_split,
)
from rc_hybrid_surrogate.losses import masked_smooth_l1
from rc_hybrid_surrogate.model import HybridGNNLSTM
from rc_hybrid_surrogate.embeddings import export_parameter_space


def _write_sample(root: Path, case_number: int, record_number: int, steps: int):
    case_id = f"case_{case_number:04d}"
    run_name = f"peer_{record_number}"
    output = root / "cases" / case_id / "dataset" / run_name
    output.mkdir(parents=True)
    rng = np.random.default_rng(case_number)
    np.savez_compressed(
        output / "hybrid_sample.npz",
        x=rng.normal(size=(3, 8)).astype(np.float32),
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        edge_attr=rng.normal(size=(4, 14)).astype(np.float32),
        global_features=rng.normal(size=43).astype(np.float32),
        record_features=rng.normal(size=(2, 6)).astype(np.float32),
        ground_motion=rng.normal(size=(steps, 2)).astype(np.float32),
        response_time_history=rng.normal(size=(steps, 3)).astype(np.float32),
        target_peak=np.asarray(
            [1.0, 2.0, 0.01, 2.0, 0.0, 1.0, 3.0, 0.02], dtype=np.float32
        ),
    )
    metadata = {
        "run_name": run_name,
        "analysis_failed": False,
        "completed_steps": steps,
        "npts_requested": steps,
        "record_id_x": f"record_{record_number}_x",
        "record_id_y": f"record_{record_number}_y",
        "global_feature_keys": [f"global_{index}" for index in range(43)],
        "record_feature_keys": [f"record_{index}" for index in range(6)],
        "target_peak_columns": [
            "max_abs_roof_disp_x_in",
            "max_abs_roof_disp_y_in",
            "max_story_drift_resultant_ratio",
            "max_story_drift_resultant_story",
            "analysis_failed",
            "completed_fraction",
            "num_recovery_steps",
            "recovery_rate",
        ],
    }
    (output / "hybrid_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


class HybridPipelineTests(unittest.TestCase):
    def test_parameter_embedding_export_has_projector_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(1, 7):
                _write_sample(root, index, index, steps=8)
            records = discover_samples([root])
            destination = root / "embeddings"

            summary = export_parameter_space(records, destination)

            self.assertEqual(summary["sample_count"], 6)
            self.assertEqual(summary["spaces"]["structure"]["dimensions"], 43)
            self.assertEqual(
                summary["spaces"]["structure_record"]["dimensions"], 55
            )
            vector_lines = (destination / "structure_record" / "vectors.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            metadata_lines = (
                destination / "structure_record" / "metadata.tsv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(vector_lines), 6)
            self.assertEqual(len(metadata_lines), 7)

    def test_record_grouped_splits_do_not_leak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(1, 7):
                _write_sample(root, index, index, steps=8 + index)
            records = discover_samples([root])
            splits = grouped_split(records, seed=7)

            groups = {
                name: {record.record_group for record in values}
                for name, values in splits.items()
            }
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])
            self.assertEqual(sum(map(len, splits.values())), 6)

    def test_variable_graph_sequence_batch_forward_and_backward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_sample(root, 1, 1, steps=11)
            _write_sample(root, 2, 2, steps=7)
            records = discover_samples([root])
            stats = NormalizationStats.calculate(records)
            dataset = HybridDataset(records, stats)
            loader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_hybrid,
                shuffle=False,
            )
            batch = next(iter(loader))
            model = HybridGNNLSTM(
                graph_hidden_dim=16,
                graph_layers=2,
                condition_dim=12,
                lstm_hidden_dim=20,
                lstm_layers=1,
                dropout=0.0,
            )
            prediction = model(batch)
            loss = masked_smooth_l1(prediction, batch["target"], batch["mask"])
            loss.backward()

            self.assertEqual(tuple(prediction.shape), (2, 11, 2))
            self.assertEqual(int(batch["mask"].sum()), 18)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
            )


if __name__ == "__main__":
    unittest.main()
