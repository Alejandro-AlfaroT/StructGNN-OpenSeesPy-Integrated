"""Contract tests for the hybrid_gnn_lstm_v3 sample reader.

The fixtures are synthesized, but they are no longer guesses: the layout below
was checked against a real v3 export (case_0001/peer_25, 245 nodes / 532
elements / 1064 hinges / 14000 steps). The first version of these tests wrote
per-entity histories as 3-D [steps, entities, components], which the exporter
never produces -- it writes them 2-D and entity-major -- and that mismatch hid
a validator check that could never fire. The shapes here mirror the exporter.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from rc_hybrid_surrogate.schema_v3 import (  # noqa: E402
    SCHEMA_VERSION,
    SchemaError,
    load_sample,
)


def write_v3_sample(
    directory: Path,
    steps: int = 32,
    stride: int = 8,
    num_nodes: int = 6,
    num_elements: int = 5,
    num_hinges: int = 10,
    num_joints: int = 4,
    schema_version: str = SCHEMA_VERSION,
    x_only: bool = False,
    include_present_mask: bool = True,
    **overrides,
):
    """Write a minimal but structurally valid v3 sample."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    force_steps = np.arange(0, steps, stride, dtype=np.int64)
    num_stories = 3

    arrays = {
        "x": rng.normal(size=(num_nodes, 8)).astype(np.float32),
        "edge_index": np.asarray(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64
        ),
        "edge_attr": rng.normal(size=(4, 14)).astype(np.float32),
        "element_attr": rng.normal(size=(num_elements, 6)).astype(np.float32),
        "global_features": rng.normal(size=43).astype(np.float32),
        "record_features": rng.normal(size=(2, 6)).astype(np.float32),
        "ground_motion": rng.normal(size=(steps, 2)).astype(np.float32),
        # full rate
        "floor_disp": rng.normal(size=(steps, num_stories, 2)).astype(np.float32),
        "floor_vel": rng.normal(size=(steps, num_stories, 2)).astype(np.float32),
        "floor_accel": rng.normal(size=(steps, num_stories, 2)).astype(np.float32),
        "story_drift_history": rng.normal(size=(steps, num_stories * 2)).astype(np.float32),
        "base_shear": rng.normal(size=(steps, 2)).astype(np.float32),
        # Strided, and FLATTENED entity-major: [steps, entities * components].
        "element_force_history": rng.normal(
            size=(force_steps.size, num_elements * 12)
        ).astype(np.float32),
        "element_force_tag_order": np.arange(1, num_elements + 1, dtype=np.int64),
        "element_force_steps": force_steps,
        "hinge_rotation": rng.normal(
            size=(force_steps.size, num_hinges * 2)
        ).astype(np.float32),
        "hinge_tag_order": np.arange(1, num_hinges + 1, dtype=np.int64),
        "hinge_rotation_steps": force_steps,
        # Joints are a strict subset of nodes in real exports.
        "joint_force_history": rng.normal(
            size=(force_steps.size, num_joints * 6)
        ).astype(np.float32),
        "joint_force_node_order": np.arange(1, num_joints + 1, dtype=np.int64),
    }
    if include_present_mask:
        arrays["ground_motion_present"] = np.asarray(
            [1.0, 0.0 if x_only else 1.0], dtype=np.float32
        )
    arrays.update(overrides)

    np.savez_compressed(directory / "hybrid_sample.npz", **arrays)
    (directory / "hybrid_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "run_name": directory.name,
                "x_only": x_only,
                "record_id_x": "RSN1_X",
                "record_id_y": None if x_only else "RSN1_Y",
                "element_force_history_columns": [
                    "axial_i", "shear_y_i", "shear_z_i", "torsion_i",
                    "moment_y_i", "moment_z_i", "axial_j", "shear_y_j",
                    "shear_z_j", "torsion_j", "moment_y_j", "moment_z_j",
                ],
                "joint_force_history_columns": ["fx", "fy", "fz", "mx", "my", "mz"],
            }
        ),
        encoding="utf-8",
    )
    return directory / "hybrid_sample.npz"


class SchemaV3Tests(unittest.TestCase):
    def test_loads_a_valid_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(Path(temporary) / "peer_1")
            sample = load_sample(path)

            self.assertEqual(sample.num_nodes, 6)
            self.assertEqual(sample.num_elements, 5)
            self.assertEqual(sample.num_steps, 32)
            self.assertEqual(sample.metadata["schema_version"], SCHEMA_VERSION)

    def test_v2_sample_is_rejected_by_version_not_by_missing_array(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1", schema_version="hybrid_gnn_lstm_v2"
            )
            with self.assertRaises(SchemaError) as caught:
                load_sample(path)
            message = str(caught.exception)
            self.assertIn("hybrid_gnn_lstm_v2", message)
            self.assertIn(SCHEMA_VERSION, message)

    def test_strided_steps_align_to_the_full_rate_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(Path(temporary) / "peer_1", steps=32, stride=8)
            sample = load_sample(path)

            steps = sample.strided_steps("element_force_history")
            np.testing.assert_array_equal(steps, np.asarray([0, 8, 16, 24]))
            self.assertEqual(
                sample.require("element_force_history").shape[0], steps.shape[0]
            )
            # Every strided index must be gatherable from a full-rate tensor.
            self.assertLess(int(steps.max()), sample.num_steps)

    def test_joint_forces_share_the_element_stride_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(Path(temporary) / "peer_1")
            sample = load_sample(path)

            np.testing.assert_array_equal(
                sample.strided_steps("joint_force_history"),
                sample.strided_steps("element_force_history"),
            )

    def test_strided_array_without_its_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "peer_1"
            write_v3_sample(directory)
            # Drop the index array the forces depend on.
            with np.load(directory / "hybrid_sample.npz", allow_pickle=False) as handle:
                arrays = {k: handle[k] for k in handle.files if k != "element_force_steps"}
            np.savez_compressed(directory / "hybrid_sample.npz", **arrays)

            with self.assertRaises(SchemaError) as caught:
                load_sample(directory / "hybrid_sample.npz")
            self.assertIn("element_force_steps", str(caught.exception))

    def test_full_rate_length_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1",
                steps=32,
                base_shear=np.zeros((31, 2), dtype=np.float32),
            )
            with self.assertRaises(SchemaError) as caught:
                load_sample(path)
            self.assertIn("base_shear", str(caught.exception))

    def test_element_force_element_count_mismatch_is_rejected(self):
        # The order array disagrees with element_attr: 99 elements covered,
        # 5 described. This is the check that silently never fired while the
        # fixtures wrote 3-D arrays.
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1",
                element_force_history=np.zeros((4, 99 * 12), dtype=np.float32),
                element_force_tag_order=np.arange(99, dtype=np.int64),
            )
            with self.assertRaises(SchemaError) as caught:
                load_sample(path)
            message = str(caught.exception)
            self.assertIn("element_force_history", message)
            self.assertIn("99", message)

    def test_entity_history_reshapes_entity_major(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1", num_elements=5, steps=32, stride=8
            )
            sample = load_sample(path)

            self.assertEqual(sample.num_components("element_force_history"), 12)
            reshaped = sample.as_entity_history("element_force_history")
            self.assertEqual(reshaped.shape, (4, 5, 12))
            # Entity-major: element j owns the j-th block of 12 columns.
            flat = sample.require("element_force_history")
            np.testing.assert_array_equal(reshaped[:, 2, :], flat[:, 24:36])

    def test_joint_order_is_a_subset_of_nodes(self):
        # A per-node decoder must target the joint order, not every node.
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1", num_nodes=6, num_joints=4
            )
            sample = load_sample(path)

            joints = sample.entity_order("joint_force_history")
            self.assertEqual(joints.shape[0], 4)
            self.assertLess(joints.shape[0], sample.num_nodes)
            self.assertEqual(sample.num_components("joint_force_history"), 6)

    def test_ragged_entity_width_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1",
                # 5 elements cannot divide 61 columns.
                element_force_history=np.zeros((4, 61), dtype=np.float32),
            )
            with self.assertRaises(SchemaError) as caught:
                load_sample(path)
            self.assertIn("whole multiple", str(caught.exception))

    def test_component_count_must_match_the_declared_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1",
                # 6 components per element, but metadata declares 12.
                element_force_history=np.zeros((4, 5 * 6), dtype=np.float32),
            )
            with self.assertRaises(SchemaError) as caught:
                load_sample(path)
            self.assertIn("element_force_history_columns", str(caught.exception))

    def test_diverged_run_is_rejected_by_the_physical_ceiling(self):
        # A diverged solve writes structurally valid arrays holding nonsense.
        # Real case: pilot30_s3 case_0003 collapsed at ~5.6% drift, then blew
        # up to a drift ratio of 135 with hinge rotations of 1.9e12 rad.
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1",
                story_drift_peaks=np.full((3, 3), 135.09, dtype=np.float32),
            )
            with self.assertRaises(SchemaError) as caught:
                load_sample(path)
            message = str(caught.exception)
            self.assertIn("physical ceiling", message)
            self.assertIn("diverged", message)

    def test_a_genuine_collapse_below_the_ceiling_still_loads(self):
        # 8% drift is a real collapse, not a divergence, and must survive.
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1",
                story_drift_peaks=np.full((3, 3), 0.08, dtype=np.float32),
            )
            sample = load_sample(path)
            self.assertAlmostEqual(
                float(sample.require("story_drift_peaks").max()), 0.08, places=5
            )

    def test_non_finite_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "peer_1"
            write_v3_sample(directory)
            with np.load(directory / "hybrid_sample.npz", allow_pickle=False) as handle:
                arrays = {k: handle[k] for k in handle.files}
            arrays["base_shear"] = arrays["base_shear"].copy()
            arrays["base_shear"][0, 0] = np.inf
            np.savez_compressed(directory / "hybrid_sample.npz", **arrays)

            with self.assertRaises(SchemaError) as caught:
                load_sample(directory / "hybrid_sample.npz")
            self.assertIn("non-finite", str(caught.exception))

    def test_analysis_failed_is_surfaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(Path(temporary) / "peer_1")
            self.assertFalse(load_sample(path).analysis_failed)

    def test_absent_y_component_is_distinguishable_from_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            both = load_sample(write_v3_sample(Path(temporary) / "a"))
            x_only = load_sample(
                write_v3_sample(Path(temporary) / "b", x_only=True)
            )

            np.testing.assert_array_equal(both.ground_motion_present(), [1.0, 1.0])
            np.testing.assert_array_equal(x_only.ground_motion_present(), [1.0, 0.0])

    def test_present_mask_falls_back_to_metadata_for_older_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_v3_sample(
                Path(temporary) / "peer_1", x_only=True, include_present_mask=False
            )
            sample = load_sample(path)
            np.testing.assert_array_equal(sample.ground_motion_present(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
