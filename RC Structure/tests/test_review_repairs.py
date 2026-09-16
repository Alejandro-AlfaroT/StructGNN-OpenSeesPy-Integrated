"""Regression counterexamples from the September verification/export review.

Tests inspect exports in memory. They do not run SAP or a design sweep.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
import Structure_Parameters as sp
from Design import Compare_SAP2000 as compare
from Design import Export_SAP2000 as export
from Design import Verify_Designs as verify
from Design import Verification_Integrity as integrity
from Design.Config import DesignConfig
from Design.SMRF_Demands import live_load_patterns, strength_load_combinations


def export_record():
    patterns = live_load_patterns(2, 2)
    unit = {"beams": [{"axis": "y", "line_index": 0, "span_index": 0,
                       "node_loads": [[0.5, 3.0]], "node_couples": [[0.5, 10.0, 20.0]]}], "columns": []}
    return {
        "geometry": {"num_bay_x": 2, "num_bay_y": 2, "num_floor": 1,
                     "bay_x_in": 120., "bay_y_in": 180., "story_h_in": 120.},
        "sections": {"b_col_in": 20., "h_col_in": 20., "b_beam_in": 10., "h_beam_in": 18.,
                     "fc_col_ksi": 8., "fc_beam_ksi": 4.},
        "reinforcement": {"col_top_bars": 4, "col_side_bars": 2, "col_stirrup_legs": 4,
                          "col_bar_size": 8, "col_stirrup_bar_size": 4, "col_stirrup_spacing_in": 4.,
                          "beam_top_bars": 4, "beam_bot_bars": 3, "beam_bar_size": 7},
        "slab": {"thickness_in": 5., "concrete_unit_weight_kcf": .15, "concrete_fc_ksi": 4.},
        "floor_loads": {"slab_thickness_in": 5., "floor_superimposed_dead_load_ksf": .05,
                        "floor_live_load_ksf": .06, "seismic_live_load_fraction": 0.},
        "demand": {"model_period_sec": .5},
        "drift_screen": {"assumptions": {"cd": 5.5}},
        "demand_basis": {"live_load_patterns": patterns,
                         "drift": {"column_stiffness_modifier": .7, "beam_stiffness_modifier": .35}},
        "design_actions": {"combinations": strength_load_combinations(1., live_patterns=patterns)},
        "floor_transfer": {"mesh_per_bay": 16, "unit_cases": {
            key: copy.deepcopy(unit) for key in ["dead", "live"] + ["live_pattern_" + p["id"] for p in patterns]}},
    }


class SAPExportRepairs(unittest.TestCase):
    def setUp(self):
        self.snapshot = dict(vars(sp))
        self.record = export_record()
        self.elf_patch = patch.object(export, "elf_story_forces", return_value={
            "story_forces_kip": [10.], "base_shear_kip": 10., "design_period_sec": .5,
            "cs": .1, "seismic_weight_kip": 100.})
        self.elf_patch.start()

    def tearDown(self):
        self.elf_patch.stop()
        for key in set(vars(sp)) - set(self.snapshot):
            delattr(sp, key)
        vars(sp).update(self.snapshot)

    def test_y_beam_local_couples_rotate_to_global(self):
        model, _ = export.build(self.record, "frame")
        rows = dict(model.tables)["FRAME LOADS - POINT"]
        rows = [r for r in rows if r["LoadPat"] == "DEAD_FLOOR" and r["Type"] == "Moment"]
        self.assertEqual({r["Dir"]: r["Force"] for r in rows}, {"X": -20., "Y": 10.})

    def test_every_saved_combination_and_pattern_is_preserved(self):
        model, target = export.build(self.record, "frame")
        tables = dict(model.tables)
        for combo in self.record["design_actions"]["combinations"]:
            rows = [r for r in tables["COMBINATION DEFINITIONS"] if r["ComboName"] == combo["id"]]
            live = "LIVE" if combo["live_pattern"] == "all" else "LIVE_" + combo["live_pattern"]
            expected = {"DEAD_FLOOR": combo["dead"], "SELF_WT": combo["dead"],
                        live: combo["live"], "EQX": combo["ex"], "EQY": combo["ey"]}
            self.assertEqual({r["CaseName"]: r["ScaleFactor"] for r in rows},
                             {k: v for k, v in expected.items() if v})
        self.assertEqual(len(target["required_strength_combinations"]), 24)
        live_cases = [r for r in tables["LOAD CASE DEFINITIONS"] if r["Case"].startswith("LIVE")]
        self.assertEqual(len(live_cases), 7)
        self.assertTrue(all(r["DesignType"] == "Live" for r in live_cases))

    def test_mesh_is_subdivisions_per_edge_and_area_patterns_match_panels(self):
        model, _ = export.build(self.record, "slab")
        tables = dict(model.tables)
        self.assertTrue(all(r["N1"] == r["N2"] == 16 for r in tables["AREA AUTO MESH ASSIGNMENTS"]))
        rows = tables["AREA LOADS - UNIFORM"]
        for pattern in self.record["demand_basis"]["live_load_patterns"]:
            self.assertEqual({r["Area"] for r in rows if r["LoadPat"] == "LIVE_" + pattern["id"]},
                             {j * 2 + i + 1 for i, j in pattern["panels"]})

    def test_weight_modifiers_remove_overlaps_without_changing_stiffness(self):
        model, target = export.build(self.record, "frame")
        weights = target["member_weight_modifiers"]
        self.assertAlmostEqual(weights["column"], 115 / 120)
        self.assertAlmostEqual(weights["x"], 13 / 18 * 100 / 120)
        self.assertAlmostEqual(weights["y"], 13 / 18 * 160 / 180)
        gamma = .15 / 1728
        corrected = (9 * 20 * 20 * 120 * weights["column"] +
                     6 * 10 * 18 * 120 * weights["x"] + 6 * 10 * 18 * 180 * weights["y"]) * gamma
        exact = (9 * 20 * 20 * 115 + 6 * 10 * 13 * 100 + 6 * 10 * 13 * 160) * gamma
        self.assertAlmostEqual(corrected, exact)
        self.assertTrue(all(r["I22Mod"] in (.7, .35) for r in dict(model.tables)["FRAME PROPERTY MODIFIERS"]))

    def test_missing_transfer_or_combinations_fail_closed(self):
        self.record["floor_transfer"]["unit_cases"].pop("live")
        with self.assertRaisesRegex(ValueError, "Missing floor-transfer"):
            export.build(self.record, "frame")
        self.record["design_actions"]["combinations"] = []
        with self.assertRaisesRegex(ValueError, "combinations are required"):
            export.build(self.record, "slab")


class SAPComparisonRepairs(unittest.TestCase):
    def test_rad_units_row_is_not_a_joint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "joints.csv"
            path.write_text("Joint,OutputCase,U1,R1\nText,Text,in,rad\n1,DRIFT_X,0.1,0.2\n")
            self.assertEqual(len(compare.read_csv(path)), 1)

    def test_missing_tables_and_steel_area_are_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            (Path(tmp) / "beam_design.csv").write_text("Frame,AsTop\n1,0.8\n")
            targets = {"case": "test", "variant": "frame", "elf": {"story_forces_kip": [1]},
                       "drift_screen": {}, "dcr": {"beam": .8}}
            report = compare.compare(targets, tmp)
            self.assertEqual(report["status"], "incomplete")
            self.assertTrue(any(r["label"] == "beam DCR" and r["status"] == "unavailable" for r in report["checks"]))

    def test_nonfinite_ratios_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            (Path(tmp) / "beam_design.csv").write_text("Frame,Ratio\n1,0.8\n2,NaN\n")
            targets = {"case": "test", "variant": "frame", "elf": {"story_forces_kip": [1]},
                       "drift_screen": {}, "dcr": {"beam": .8}}
            report = compare.compare(targets, tmp)
            self.assertTrue(any(r["label"] == "beam DCR coverage" for r in report["checks"]))


class VerificationRepairs(unittest.TestCase):
    def setUp(self):
        self.case = verify.plan_cases(1)[0]
        self.expected = integrity.expected_identity(self.case, DesignConfig.from_structure_parameters())

    def artifact(self):
        c = self.case
        return {"request_identity": self.expected, "schema_version": self.expected["schema"],
                "seismic": {"site_label": c["seismic_site"]},
                "geometry": {"num_bay_x": c["num_bay_x"], "num_bay_y": c["num_bay_y"], "num_floor": c["num_floor"],
                             "bay_x_in": c["bay_x_width_ft"] * 12., "bay_y_in": c["bay_y_width_ft"] * 12.,
                             "story_h_in": c["story_height_ft"] * 12.}, "sections": {}, "dcr": {}}

    def saved(self):
        return {"case": self.case, "status": "designed", "accepted": True, "probe_assertions": False, "probe_date": None}

    def test_identity_calculation_leaves_geometry_and_site_unchanged(self):
        before = dict(vars(sp))
        integrity.expected_identity(verify.plan_cases(2)[1], DesignConfig.from_structure_parameters())
        for key in ("NUM_FLOOR", "BAY_X", "BAY_Y", "SEISMIC_SITE_LABEL", "ASCE_SDS", "FLOOR_TRANSFER"):
            self.assertEqual(getattr(sp, key), before[key])

    def test_missing_design_and_digest_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = integrity.checked_result(self.case, tmp, self.saved(), self.expected, False, None)
            self.assertEqual(result["status"], "error")
            (Path(tmp) / "design.json").write_text(json.dumps(self.artifact()))
            result = integrity.checked_result(self.case, tmp, {**self.saved(), "design_sha256": "wrong"},
                                              self.expected, False, None)
            self.assertEqual(result["status"], "error")

    def test_changed_sources_geometry_and_probe_cannot_reuse_acceptance(self):
        record = self.artifact()
        record["geometry"]["num_floor"] += 1
        with self.assertRaisesRegex(ValueError, "geometry"):
            integrity.validate_record(record, self.expected)
        record = self.artifact()
        record["request_identity"] = {**self.expected, "source_sha256": {"different": "code"}}
        with self.assertRaisesRegex(ValueError, "identity"):
            integrity.validate_record(record, self.expected)
        with tempfile.TemporaryDirectory() as tmp:
            result = integrity.checked_result(self.case, tmp, self.saved(), self.expected, True, "2026-09-16")
            self.assertEqual(result["status"], "error")

    def test_qualification_is_recomputed_instead_of_trusting_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "design.json").write_text(json.dumps(self.artifact()))
            q = {"accepted": False, "counts": {"fail": 1}, "checks": [{"id": "broken", "status": "fail"}]}
            with patch("Design.SMRF_Qualification.qualify_design", return_value=q) as qualify:
                result = integrity.checked_result(self.case, tmp, self.saved(), self.expected, False, None)
            self.assertEqual(result["status"], "designed")
            self.assertFalse(result["accepted"])
            self.assertEqual(result["fail_ids"], ["broken"])
            qualify.assert_called_once()

    def test_interrupted_lock_and_temp_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".design.json.lock"
            temp = Path(tmp) / ".design.json.example.tmp"
            lock.write_text("owned")
            temp.write_text("partial")
            with self.assertRaises(RuntimeError):
                verify.clear_interrupted_design(tmp)
            self.assertEqual(lock.read_text(), "owned")
            self.assertEqual(temp.read_text(), "partial")

    def test_launcher_does_not_restart_a_designed_case_with_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "result.json").write_text(json.dumps(self.saved()))
            with patch.object(verify.subprocess, "run") as launch:
                result, cached = verify._launch(sys.executable, self.case, tmp, False)
            self.assertTrue(cached)
            self.assertEqual(result["status"], "error")
            launch.assert_not_called()
            self.assertTrue(json.loads((Path(tmp) / "result.json").read_text())["accepted"])

    def test_worker_crash_cannot_reuse_an_earlier_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "result.json").write_text(json.dumps({**self.saved(), "status": "error",
                                                              "attempt_id": "earlier", "error": "earlier failure"}))
            with patch.object(verify.subprocess, "run", return_value=subprocess.CompletedProcess([], 42, "", "crash")):
                result, cached = verify._launch(sys.executable, self.case, tmp, False)
            self.assertFalse(cached)
            self.assertIn("42", result["error"])

    def test_summary_main_does_not_count_missing_artifact_as_accepted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            case_dir = Path(tmp) / self.case["case_id"]
            case_dir.mkdir()
            (case_dir / "result.json").write_text(json.dumps(self.saved()))
            verify.main(["--count", "1", "--summarize-only", "--output-root", tmp])
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(summary["cases"][0]["status"], "error")
            self.assertIn("0 accepted", (Path(tmp) / "summary.md").read_text())

    def test_new_probe_manifest_requires_explicit_common_date(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = ["--count", "1", "--summarize-only", "--probe-assertions", "--output-root", tmp]
            with self.assertRaisesRegex(SystemExit, "require --probe-date"):
                verify.main(args)
            verify.main(args + ["--probe-date", "2026-09-16"])
            self.assertEqual(json.loads((Path(tmp) / "plan.json").read_text())["probe_date"], "2026-09-16")
            with self.assertRaisesRegex(SystemExit, "date differs"):
                verify.main(args + ["--probe-date", "2026-09-17"])

    def test_lease_excludes_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".lease"
            script = ("import sys; sys.path.insert(0, sys.argv[1]); "
                      "from Design.Verification_Integrity import exclusive_lease; "
                      "lease=exclusive_lease(sys.argv[2]); lease.__enter__(); lease.__exit__(None,None,None)")
            with integrity.exclusive_lease(path):
                child = subprocess.run([sys.executable, "-B", "-c", script, str(RC), str(path)], capture_output=True, text=True)
                self.assertNotEqual(child.returncode, 0)
                self.assertIn("Another process owns", child.stderr)
            with integrity.exclusive_lease(path):
                pass

    def test_invalid_plans_duplicate_results_and_missing_date_are_rejected(self):
        for kwargs in ({"num_cases": 0}, {"num_cases": 1, "geometry_offset": -1}, {"num_cases": 1, "seismic_sites": ()}):
            with self.assertRaises(ValueError):
                verify.plan_cases(**kwargs)
        with self.assertRaises(ValueError):
            verify.probe_config()
        # The PROBE declaration must itself be a valid, serializable declaration:
        # the repair that made the date mandatory passed the datetime.date class
        # as the declaration date, so every PROBE design would have been refused
        # ("declaration_date is blank") and the identity could not be serialized.
        probe = verify.probe_config("2026-09-16")
        self.assertEqual(probe.demands.declaration_date, "2026-09-16")
        self.assertEqual(probe.demands.problems(), [])
        self.assertTrue(probe.demands.declared())
        self.assertTrue(probe.slab_actions.all_asserted())
        self.assertEqual(probe.verification.assertion_date, "2026-09-16")
        identity = integrity.expected_identity(self.case, probe)
        self.assertEqual(identity["policy"]["demands"]["declaration_date"], "2026-09-16")
        json.dumps(identity, allow_nan=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                verify.summarize([self.saved(), self.saved()], Path(tmp), False, [self.case])
        self.assertIsNone(verify.methodology_sha256({}))


if __name__ == "__main__":
    unittest.main()
