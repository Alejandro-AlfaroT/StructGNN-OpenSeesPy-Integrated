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
    stratified_grouped_split,
)
from rc_hybrid_surrogate.losses import masked_smooth_l1
from rc_hybrid_surrogate.model import HybridGNNLSTM
from rc_hybrid_surrogate.embeddings import export_parameter_space
from rc_hybrid_surrogate.plotting import LossPlotter
from rc_hybrid_surrogate.features import (
    ENGINEERED_FEATURE_NAMES,
    MOTION_FEATURE_NAMES,
    EngineeredFeatureCache,
    GRAVITY_IN_PER_SEC2,
    load_group_intensity_scores,
    pseudo_spectral_acceleration,
)
from evaluate import calculate_case_metrics


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
    def test_case_evaluation_metrics_use_physical_peaks(self):
        target = np.asarray([[0.0, 0.0], [2.0, -4.0]], dtype=np.float64)
        prediction = np.asarray([[0.0, 0.0], [1.0, -2.0]], dtype=np.float64)

        metrics = calculate_case_metrics(
            prediction,
            target,
            prediction,
            target,
            huber_beta=0.5,
        )

        self.assertAlmostEqual(metrics["rmse_in"], np.sqrt(1.25))
        self.assertAlmostEqual(metrics["mae_in"], 0.75)
        self.assertAlmostEqual(metrics["true_peak_x_in"], 2.0)
        self.assertAlmostEqual(metrics["predicted_peak_y_in"], 2.0)
        self.assertAlmostEqual(metrics["peak_error_x_in"], 1.0)
        self.assertAlmostEqual(metrics["peak_error_y_in"], 2.0)

    def test_engineered_feature_group_selection(self):
        full = EngineeredFeatureCache(
            feature_names=list(ENGINEERED_FEATURE_NAMES),
            values={
                "case_0001/peer_1": np.arange(
                    len(ENGINEERED_FEATURE_NAMES), dtype=np.float32
                )
            },
        )

        selected = full.select_groups(["modal", "interaction"])

        self.assertEqual(selected.dimension, 10)
        self.assertEqual(tuple(selected.vector("case_0001/peer_1").shape), (10,))
        self.assertNotIn("x_pgv_in_per_sec", selected.feature_names)
        self.assertIn("period_x_mode_1_sec", selected.feature_names)
        self.assertIn("psa_x_at_period_x_g", selected.feature_names)

    def test_response_spectrum_peaks_near_sine_period(self):
        dt = 0.01
        time = np.arange(0.0, 20.0, dt)
        acceleration = (
            0.1 * GRAVITY_IN_PER_SEC2 * np.sin(2.0 * np.pi * time)
        )
        periods = np.asarray([0.2, 1.0, 2.0], dtype=np.float64)

        spectrum = pseudo_spectral_acceleration(acceleration, dt, periods)

        self.assertEqual(tuple(spectrum.shape), (3,))
        self.assertTrue(np.isfinite(spectrum).all())
        self.assertEqual(int(np.argmax(spectrum)), 1)

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

    def test_fixed_test_group_count_moves_remaining_groups_to_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(1, 7):
                _write_sample(root, index, index, steps=8)
            records = discover_samples([root])
            splits = grouped_split(
                records,
                train_fraction=0.5,
                validation_fraction=0.15,
                test_group_count=1,
                seed=7,
            )

            groups = {
                name: {record.record_group for record in values}
                for name, values in splits.items()
            }
            self.assertEqual(len(groups["train"]), 3)
            self.assertEqual(len(groups["validation"]), 2)
            self.assertEqual(len(groups["test"]), 1)
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])

    def test_group_intensity_scores_rank_groups_by_severity(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "engineered_features.json"
            feature_count = len(MOTION_FEATURE_NAMES)
            payload = {
                "motion_groups": {
                    "low": {"motion_features": [1.0] * feature_count},
                    "mid": {"motion_features": [5.0] * feature_count},
                    "high": {"motion_features": [10.0] * feature_count},
                }
            }
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

            scores = load_group_intensity_scores(cache_path)

            self.assertEqual(set(scores), {"low", "mid", "high"})
            self.assertAlmostEqual(scores["low"], 0.0)
            self.assertAlmostEqual(scores["mid"], 0.5)
            self.assertAlmostEqual(scores["high"], 1.0)

    def test_stratified_split_balances_group_scores_across_splits(self):
        # Regression guard: a plain random shuffle of a small group pool can
        # hand one split a systematically more (or less) severe subset by
        # chance (see RC Hybrid Surrogate Model/README.md discussion of the
        # 54-group interim dataset). Stratifying by score should keep every
        # split's mean close to the overall mean instead.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(1, 21):
                _write_sample(root, index, index, steps=8)
            records = discover_samples([root])
            group_scores = {
                record.record_group: (index - 1) / 19.0
                for index, record in enumerate(records, start=1)
            }

            splits = stratified_grouped_split(
                records,
                group_scores,
                train_fraction=0.7,
                validation_fraction=0.15,
                seed=3,
            )

            groups = {
                name: {record.record_group for record in values}
                for name, values in splits.items()
            }
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])
            self.assertEqual(len(groups["train"]), 14)
            self.assertEqual(len(groups["validation"]), 3)
            self.assertEqual(len(groups["test"]), 3)

            for name in ("train", "validation", "test"):
                mean_score = sum(group_scores[g] for g in groups[name]) / len(groups[name])
                self.assertLess(abs(mean_score - 0.5), 0.15)

    def test_loss_plotter_writes_live_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "loss_curves.html"
            plotter = LossPlotter(output, live=True)
            plotter.update(
                [
                    {
                        "epoch": 1,
                        "train": {"loss": 0.5},
                        "validation": {"loss": 0.7},
                    },
                    {
                        "epoch": 2,
                        "train": {"loss": 0.4},
                        "validation": {"loss": 0.6},
                    },
                ]
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            contents = output.read_text(encoding="utf-8")
            self.assertIn('http-equiv="refresh"', contents)
            self.assertIn("Training loss: 0.400000", contents)
            self.assertIn("Validation loss: 0.600000", contents)

    def test_variable_graph_sequence_batch_forward_and_backward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_sample(root, 1, 1, steps=11)
            _write_sample(root, 2, 2, steps=7)
            records = discover_samples([root])
            engineered = EngineeredFeatureCache(
                feature_names=["feature_0", "feature_1", "feature_2", "feature_3"],
                values={
                    record.sample_id: np.asarray(
                        [index, index + 1, index + 2, index + 3], dtype=np.float32
                    )
                    for index, record in enumerate(records)
                },
            )
            stats = NormalizationStats.calculate(records, engineered)
            dataset = HybridDataset(records, stats, engineered_features=engineered)
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
                engineered_dim=4,
            )
            prediction = model(batch)
            loss = masked_smooth_l1(prediction, batch["target"], batch["mask"])
            loss.backward()

            self.assertEqual(tuple(prediction.shape), (2, 11, 2))
            self.assertEqual(tuple(batch["engineered_features"].shape), (2, 4))
            self.assertEqual(int(batch["mask"].sum()), 18)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
            )


if __name__ == "__main__":
    unittest.main()
