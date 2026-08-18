import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


RC_STRUCTURE_DIR = Path(__file__).resolve().parents[1]
if str(RC_STRUCTURE_DIR) not in sys.path:
    sys.path.insert(0, str(RC_STRUCTURE_DIR))

import Structure_Parameters as sp
from Analysis.NTHA import _analysis_step_schedule
from Data_Generation import Generate_Hybrid_Dataset
from Data_Generation import Generate_Parameterized_Dataset
from Data_Generation import Hybrid_Exporter
from Data_Generation.Graph_Exporter import collect_global_parameters
from Ground_Motion_Main import (
    analysis_run_name,
    validate_ntha_output_compatibility,
)
from Loads import Ground_Motion
from Loads.Ground_Motion import GroundMotionRecord
from Model import IMK_Hinges


class GenerationRegressionTests(unittest.TestCase):
    def test_parameterized_scheduler_uses_lightweight_record_catalog(self):
        expected = [
            item[1]
            for item in Generate_Hybrid_Dataset._all_pair_keys(
                "peer_mle_all", None, 15000
            )
        ]

        actual = Generate_Parameterized_Dataset.eligible_record_ids(
            "peer_mle_all", 15000
        )

        self.assertEqual(actual, expected)

    def test_parameterized_progress_uses_cached_completion_state(self):
        cases = [
            {"case_id": "case_0001", "result_id": 1},
            {"case_id": "case_0002", "result_id": 2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.object(
                Generate_Parameterized_Dataset,
                "complete",
                side_effect=AssertionError("progress must not rescan case artifacts"),
            ):
                state = Generate_Parameterized_Dataset.progress(
                    root=root,
                    cases=cases,
                    results={},
                    active={"case_0002": {"pid": 123}},
                    status="running",
                    started="test-start",
                    completed_ids={"case_0001"},
                    write_manifest=True,
                    invocation_case_ids=["case_0001", "case_0002"],
                    case_start=1,
                    case_end=2,
                )

            self.assertEqual(state["completed_count"], 1)
            self.assertEqual(state["remaining_count"], 1)
            self.assertEqual(state["active_cases"], {"case_0002": {"pid": 123}})
            self.assertEqual(state["invocation_completed_count"], 1)
            self.assertEqual(state["invocation_remaining_case_ids"], ["case_0002"])

    def test_device_case_ranges_are_non_overlapping(self):
        cases = [{"case_id": f"case_{index:04d}"} for index in range(1, 11)]
        completed = {"case_0001", "case_0006"}

        scope_a, selected_a = Generate_Parameterized_Dataset.select_pending_cases(
            cases, completed, case_start=1, case_end=5
        )
        scope_b, selected_b = Generate_Parameterized_Dataset.select_pending_cases(
            cases, completed, case_start=6, case_end=10
        )

        ids_a = {case["case_id"] for case in selected_a}
        ids_b = {case["case_id"] for case in selected_b}
        self.assertEqual(len(scope_a), 5)
        self.assertEqual(len(scope_b), 5)
        self.assertFalse(ids_a & ids_b)
        self.assertEqual(ids_a, {"case_0002", "case_0003", "case_0004", "case_0005"})
        self.assertEqual(ids_b, {"case_0007", "case_0008", "case_0009", "case_0010"})

        with self.assertRaises(ValueError):
            Generate_Parameterized_Dataset.select_pending_cases(
                cases, completed, case_start=0, case_end=5
            )

    def test_stop_request_is_atomic_and_detectable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "STOP_GENERATION.json"

            payload = Generate_Hybrid_Dataset._request_stop(stop_path)

            self.assertTrue(Generate_Hybrid_Dataset._stop_requested(stop_path))
            self.assertEqual(
                json.loads(stop_path.read_text(encoding="utf-8")),
                payload,
            )
            self.assertFalse((stop_path.parent / f".{stop_path.name}.tmp").exists())

    def test_atomic_checkpoint_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "generation_state.json"
            real_replace = Generate_Hybrid_Dataset.os.replace
            attempts = []

            def transient_lock(source, destination):
                attempts.append((source, destination))
                if len(attempts) < 3:
                    raise PermissionError(5, "Access is denied")
                return real_replace(source, destination)

            with (
                mock.patch.object(
                    Generate_Hybrid_Dataset.os,
                    "replace",
                    side_effect=transient_lock,
                ),
                mock.patch.object(Generate_Hybrid_Dataset.time, "sleep") as sleep,
            ):
                Generate_Hybrid_Dataset._write_json_atomic(
                    output_path,
                    {"status": "completed"},
                )

            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {"status": "completed"},
            )

    def test_running_child_is_terminated_when_stop_is_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stop_path = temp_path / "STOP_GENERATION.json"
            log_path = temp_path / "child.log"
            Generate_Hybrid_Dataset._request_stop(stop_path)

            result = Generate_Hybrid_Dataset._run_command(
                [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                cwd=temp_path,
                log_path=log_path,
                stop_path=stop_path,
                poll_interval_sec=0.01,
            )

            self.assertTrue(result["interrupted"])
            self.assertEqual(result["stop_reason"], "stop_file")
            self.assertNotEqual(result["returncode"], 0)
            self.assertIn("will be retried on resume", log_path.read_text(encoding="utf-8"))

    def test_resume_reuses_successful_ntha_when_sample_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            args = SimpleNamespace(
                ntha_root=str(temp_path / "ntha"),
                dataset_dir=str(temp_path / "dataset"),
                max_npts=15000,
                result_id=[1],
                set_name="peer_mle_all",
                split=None,
                limit=1,
                start_index=0,
                _stop_path=str(temp_path / "dataset" / "STOP_GENERATION.json"),
                x_only=False,
                scale_factor=None,
                damping_ratio=0.05,
                rayleigh_mode_i=0,
                rayleigh_mode_j=2,
                dt_factor=1.0,
                skip_existing=True,
                python_exe=sys.executable,
                catalog_summary=False,
            )

            with (
                mock.patch.object(
                    Generate_Hybrid_Dataset,
                    "_all_pair_keys",
                    return_value=[("peer_result_id:1", 1, 100)],
                ),
                mock.patch.object(
                    Generate_Hybrid_Dataset,
                    "_status_success",
                    return_value=True,
                ),
                mock.patch.object(Generate_Hybrid_Dataset, "_run_command") as run_command,
            ):
                summaries = Generate_Hybrid_Dataset.run_ntha_batch(args)

            run_command.assert_not_called()
            self.assertTrue(summaries[0]["skipped"])
            self.assertIn("sample will be compiled", summaries[0]["reason"])

    def test_time_series_uses_scaled_values_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "record.txt"
            source.write_text("1.0\n2.0\n", encoding="utf-8")
            before = source.read_bytes()
            record = GroundMotionRecord(
                record_id="TEST",
                dt_sec=0.01,
                acceleration=np.asarray([1.0, 2.0]),
                units="g",
                scale_factor=2.0,
                source_path=str(source),
            )

            with mock.patch.object(Ground_Motion.ops, "timeSeries") as time_series:
                returned = Ground_Motion.define_path_time_series(10, record)

            self.assertIsNone(returned)
            self.assertEqual(before, source.read_bytes())
            args = time_series.call_args.args
            self.assertIn("-values", args)
            values_index = args.index("-values") + 1
            self.assertAlmostEqual(args[values_index], 2.0 * 386.4)
            self.assertAlmostEqual(args[values_index + 1], 4.0 * 386.4)
            self.assertNotIn("-filePath", args)

    def test_analysis_substeps_preserve_full_excitation_duration(self):
        x = GroundMotionRecord("X", 0.01, np.zeros(100))
        y = GroundMotionRecord("Y", 0.01, np.zeros(120))

        dt, steps = _analysis_step_schedule(x, dt_factor=0.5)
        self.assertAlmostEqual(dt, 0.005)
        self.assertEqual(steps, 200)

        dt, steps = _analysis_step_schedule(x, y, dt_factor=1.0)
        self.assertAlmostEqual(dt, 0.01)
        self.assertEqual(steps, 120)

        with self.assertRaises(ValueError):
            _analysis_step_schedule(x, dt_factor=0.0)

    def test_exporter_recovers_stale_path_and_applies_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ground_motion_dir = Path(temp_dir) / "Ground_Motions"
            processed_dir = ground_motion_dir / "processed"
            metadata_dir = ground_motion_dir / "metadata"
            processed_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)
            processed_file = processed_dir / "TEST_in_per_sec2.txt"
            np.savetxt(processed_file, np.asarray([1.0, 2.0, 3.0]))
            manifest = metadata_dir / "record_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "record_id",
                        "processed_file",
                        "processed_units",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_id": "TEST",
                        "processed_file": "processed/TEST_in_per_sec2.txt",
                        "processed_units": "in/sec^2",
                    }
                )

            summary = {
                "record_id": "TEST",
                "dt_sec": 1.0,
                "analysis_dt_sec": 0.5,
                "units": "in/sec^2",
                "scale_factor": 2.0,
                "source_path": r"C:\old\desktop\processed\TEST_in_per_sec2.txt",
            }
            with (
                mock.patch.object(
                    Hybrid_Exporter,
                    "GROUND_MOTION_DIR",
                    ground_motion_dir,
                ),
                mock.patch.object(
                    Hybrid_Exporter,
                    "GROUND_MOTION_MANIFEST",
                    manifest,
                ),
            ):
                acceleration = Hybrid_Exporter._load_acceleration_from_summary(
                    summary,
                    target_steps=5,
                )

            np.testing.assert_allclose(
                acceleration,
                np.asarray([2.0, 3.0, 4.0, 5.0, 6.0]),
            )

    def test_manifest_paths_are_relative_and_resolvable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "dataset"
            target = Path(temp_dir) / "ntha" / "peer_1"
            base.mkdir()
            target.mkdir(parents=True)

            portable = Hybrid_Exporter._portable_path(target, base)

            self.assertFalse(Path(portable).is_absolute())
            self.assertEqual((base / portable).resolve(), target.resolve())

    def test_output_identity_rejects_mixed_model_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "peer_1"
            output_dir.mkdir()
            current = collect_global_parameters()
            current["num_floor"] = int(current["num_floor"]) + 1
            (output_dir / "global_parameters.json").write_text(
                json.dumps(current),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "different model configurations"):
                validate_ntha_output_compatibility(output_dir)

    def test_analysis_variants_get_distinct_output_names(self):
        self.assertEqual(analysis_run_name("peer_1"), "peer_1")
        self.assertEqual(
            analysis_run_name("peer_1", scale_factor=2.0),
            "peer_1__sf_2",
        )
        self.assertEqual(
            analysis_run_name("peer_1", x_only=True, dt_factor=0.5),
            "peer_1__x_only__dtf_0p5",
        )

    def test_imk_material_definition_uses_documented_argument_order(self):
        # Regression guard: this call has previously shipped with a wrong
        # OpenSees IMKBilin argument order (a nonexistent "A" deterioration
        # mode standing in for the FmaxFy/FresFy strength ratios). Pin the
        # exact positional order the current IMKBilin signature requires.
        with mock.patch.object(IMK_Hinges.ops, "uniaxialMaterial") as uniaxial_material:
            IMK_Hinges._define_imk_peak_material(
                mat_tag=101,
                elastic_stiffness=5000.0,
                yield_moment=250.0,
            )

        uniaxial_material.assert_called_once()
        args = uniaxial_material.call_args.args
        self.assertEqual(len(args), 23)
        self.assertEqual(
            args,
            (
                sp.IMK_MATERIAL_TYPE,
                101,
                5000.0,
                sp.IMK_THETA_P_POS,
                sp.IMK_THETA_PC_POS,
                sp.IMK_THETA_U_POS,
                250.0,
                getattr(sp, "IMK_FMAXFY_POS", 1.10),
                getattr(sp, "IMK_FRESFY_POS", sp.IMK_RES_POS),
                sp.IMK_THETA_P_NEG,
                sp.IMK_THETA_PC_NEG,
                sp.IMK_THETA_U_NEG,
                250.0,
                getattr(sp, "IMK_FMAXFY_NEG", 1.10),
                getattr(sp, "IMK_FRESFY_NEG", sp.IMK_RES_NEG),
                sp.IMK_LAMBDA_S,
                sp.IMK_LAMBDA_C,
                sp.IMK_LAMBDA_K,
                sp.IMK_C_S,
                sp.IMK_C_C,
                sp.IMK_C_K,
                sp.IMK_D_POS,
                sp.IMK_D_NEG,
            ),
        )

    def test_hinge_stiffness_mode_selects_correct_basis(self):
        # Regression guard: hinge calibration previously used a
        # "yield_rotation" basis that was later changed to
        # "member_stiffness_factor". Pin the behavior of every supported
        # mode so a future edit can't silently change which basis is used.
        length = 120.0

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "yield_rotation"):
            components = IMK_Hinges.imk_hinge_stiffness_components(
                "column", "rot_z", length
            )
        self.assertEqual(components["mode"], "yield_rotation")
        self.assertAlmostEqual(
            components["selected_stiffness"], components["yield_based_stiffness"]
        )

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "member_stiffness_factor"):
            components = IMK_Hinges.imk_hinge_stiffness_components(
                "column", "rot_z", length
            )
        self.assertEqual(components["mode"], "member_stiffness_factor")
        self.assertAlmostEqual(
            components["selected_stiffness"], components["member_based_stiffness"]
        )

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "max"):
            components = IMK_Hinges.imk_hinge_stiffness_components(
                "column", "rot_z", length
            )
        self.assertAlmostEqual(
            components["selected_stiffness"],
            max(
                components["yield_based_stiffness"],
                components["member_based_stiffness"],
            ),
        )

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "bogus"):
            with self.assertRaises(ValueError):
                IMK_Hinges.imk_hinge_stiffness_components("column", "rot_z", length)

    def test_member_orientation_ties_expected_dofs(self):
        self.assertEqual(IMK_Hinges._orientation("column")[1], (1, 2, 3, 6))
        self.assertEqual(IMK_Hinges._orientation("beam_x")[1], (1, 2, 3, 4))
        self.assertEqual(IMK_Hinges._orientation("beam_y")[1], (1, 2, 3, 5))

        with self.assertRaises(ValueError):
            IMK_Hinges._orientation("brace")


if __name__ == "__main__":
    unittest.main()
