"""Fixture demonstration of the direction-explicit pushover diagnostic (benchmark preparation, 2026-09-20).

The sign conventions the recorder relies on are checked on a cantilever
column, then a small designed frame is pushed a few steps in +Y, -Y and +X
and every equilibrium and sign check must pass on real solver output.
"""
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops                                              # noqa: E402
import Structure_Parameters as sp                                              # noqa: E402
from Analysis import Pushover_Diagnostic as diag                                # noqa: E402
from Design import Design_Driver as driver                                      # noqa: E402


class SignConventionTests(unittest.TestCase):
    def test_column_story_shear_and_moment_rules_on_a_cantilever(self):
        for direction, load in (("x", (10.0, 0.0)), ("y", (0.0, 10.0))):
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0.0, 0.0, 0.0)
            ops.node(2, 0.0, 0.0, 120.0)
            ops.fix(1, 1, 1, 1, 1, 1, 1)
            ops.geomTransf("PDelta", 1, 1, 0, 0)
            ops.element("elasticBeamColumn", 1, 1, 2, 400.0, 5000.0, 2000.0, 20000.0, 13333.0, 13333.0, 1)
            ops.timeSeries("Linear", 1)
            ops.pattern("Plain", 1, 1)
            ops.load(2, load[0], load[1], 0.0, 0.0, 0.0, 0.0)
            ops.constraints("Plain")
            ops.numberer("Plain")
            ops.system("BandGeneral")
            ops.test("NormDispIncr", 1e-8, 20)
            ops.algorithm("Linear")
            ops.integrator("LoadControl", 1.0)
            ops.analysis("Static")
            self.assertEqual(ops.analyze(1), 0)
            f = list(ops.eleResponse(1, "localForce"))
            # The column carries +10 kip of story shear in the loaded direction.
            self.assertAlmostEqual(diag.COLUMN_STORY_SHEAR[direction](f), 10.0, places=6, msg=direction)
            # Internal moment: zero at the free end, 1200 kip-in at the base, linear, and M(L) equals the j-end value.
            moment = diag.COLUMN_MOMENT_ALONG[direction]
            self.assertAlmostEqual(abs(moment(f, 0.0)), 1200.0, places=4, msg=direction)
            self.assertAlmostEqual(moment(f, 120.0), 0.0, places=6, msg=direction)
            self.assertAlmostEqual(moment(f, 120.0), f[diag.COLUMN_END_MOMENT_INDEX[direction][1]], places=6)
            self.assertAlmostEqual(abs(moment(f, 60.0)), 600.0, places=4)
            ops.wipe()


class FixtureDemonstrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"COVER", "NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            cls.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.NUM_BAY_X, sp.NUM_BAY_Y = 2, 1
        sp.NUM_FLOOR = 2
        sp.NUM_MODES = 3 * sp.NUM_FLOOR
        try:
            record = driver.design_structure(max_section_iter=1, max_steel_iter=1, verbose=False)
        finally:
            ops.wipe()
        cls.record = json.loads(json.dumps(record, allow_nan=False))
        driver.apply_design(cls.record)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        ops.wipe()
        cls.tmp.cleanup()
        cls.stack.close()

    def run_case(self, direction, sign, steps=6, **changes):
        settings = diag.DiagnosticSettings(direction=direction, sign=sign, du_in=0.02, max_steps=steps,
                                           target_roof_drift_ratio=0.05, load_pattern="elf", **changes)
        out = Path(self.tmp.name) / f"{direction}{'p' if sign > 0 else 'm'}"
        summary = diag.run_diagnostic(settings, out)
        lines = [json.loads(line) for line in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        return summary, lines

    def test_settings_are_validated_and_recorded(self):
        with self.assertRaises(ValueError):
            diag.DiagnosticSettings(direction="z").validate()
        with self.assertRaises(ValueError):
            diag.DiagnosticSettings(sign=0.5).validate()
        with self.assertRaises(ValueError):
            diag.DiagnosticSettings(load_pattern="explicit", explicit_weights=[1.0]).validate()
        summary, _ = self.run_case("y", 1.0, steps=2)
        recorded = summary["settings"]
        for key in ("direction", "sign", "load_pattern", "gravity_dead_factor", "gravity_live_factor", "du_in", "max_steps",
                    "target_roof_drift_ratio", "tolerance", "max_iterations", "recovery", "stop_on_strength_loss_fraction"):
            self.assertIn(key, recorded)
        self.assertEqual(recorded["case"], "y+")
        self.assertEqual(summary["pattern"]["basis"]["kind"], "ASCE 7-22 12.8.3 vertical distribution")
        self.assertAlmostEqual(sum(summary["pattern"]["weights_kip"]), 1.0)
        self.assertEqual(summary["gravity"]["settings"]["live_factor"], 0.25)
        self.assertIn("not_represented", summary["model_audit"])
        self.assertTrue(any("probable" in item for item in summary["model_audit"]["not_represented"]))
        self.assertEqual(summary["model_audit"]["cage"]["col_bar_size"], self.record["reinforcement"]["col_bar_size"])

    def test_y_push_passes_every_equilibrium_and_sign_check(self):
        summary, lines = self.run_case("y", 1.0)
        checks = summary["checks"]
        self.assertTrue(checks["all_pass"], checks)
        self.assertTrue(checks["sign_check"]["same_sign"])
        self.assertTrue(checks["sign_check"]["reaction_opposes_load"])
        self.assertTrue(checks["control_direction_check"]["moves_with_sign"])
        self.assertLess(abs(checks["control_direction_check"]["other_direction_in"]), 1e-6)   # no X motion under a Y push
        self.assertLess(checks["base_shear_max_residual_kip"], 1e-6)
        self.assertLess(checks["story_shear_max_residual_kip"], 1e-6)
        self.assertLess(checks["joint_balance_max_relative_residual"], 1e-6)
        self.assertLess(checks["column_linearity_max_residual_kip_in"], 1e-6)
        self.assertIsNotNone(summary["gravity"]["expected_vertical_load_kip"])
        self.assertLess(abs(summary["gravity"]["vertical_equilibrium_residual_kip"]), 1e-6 * summary["gravity"]["expected_vertical_load_kip"])
        self.assertEqual(summary["completed_steps"], 6)
        self.assertEqual(lines[0]["kind"], "gravity_state")
        step = lines[-1]
        # Story shears: the column-to-line sums equal the applied shear above each story, line by line summed.
        applied = step["applied_loads_kip"]
        for k in (1, 2):
            self.assertAlmostEqual(step["line_story_shears"][str(k)]["total_kip"], sum(applied[k - 1:]), places=6)
        self.assertEqual(sorted(step["line_story_shears"]["1"]["by_line_kip"]), ["0", "1", "2"])   # three y-frames (i = 0, 1, 2)
        # Every column's moment diagram is linear and its two end shears balance.
        for row in step["columns"]:
            self.assertLess(abs(row["linearity_residual_kip_in"]), 1e-6)
            self.assertLess(abs(row["shear_end_residual_kip"]), 1e-6)
            self.assertIn("increment", row)
            self.assertEqual(row["face_offsets_in"][0], 0.0 if row["story"] == 1 else sp.H_BEAM / 2.0)
        # Joint balances: the increment sums to zero and the sharing ratio is reported or undefined, never clipped.
        for joint in step["joint_balances"]:
            self.assertLess(abs(joint["relative_residual"]), 1e-6)
            share = joint["share_to_column_below"]
            self.assertTrue(share is None or 0.0 <= share <= 1.0)
        roof = [j for j in step["joint_balances"] if j["floor"] == 2]
        self.assertTrue(all(j["column_above_kip_in"] == 0.0 and j["share_to_column_below"] == 1.0 for j in roof))
        # Hinges carry the installed strengths per sign, and the active spring is the loaded bending.
        hinge = next(h for h in step["hinges"] if h["member_type"] == "beam_y")
        self.assertEqual(hinge["active_spring_index"], 0)
        self.assertIn("moment_over_fy", hinge["state"]["y"])
        column = next(h for h in step["hinges"] if h["member_type"] == "column")
        self.assertEqual(column["active_spring_index"], 0)                    # Y push bends about global X
        # Edge displacements are signed and grow with the push.
        edges = step["edge_displacements"]["2"]
        self.assertGreater(edges["edge_a_in"], 0.0)
        self.assertGreater(edges["edge_b_in"], 0.0)

    def test_negative_y_and_positive_x_pushes_are_supported(self):
        summary_m, lines_m = self.run_case("y", -1.0, steps=4)
        self.assertTrue(summary_m["checks"]["all_pass"], summary_m["checks"])
        self.assertLess(lines_m[-1]["control_displacement_in"], 0.0)
        self.assertLess(lines_m[-1]["base_shear_kip"], 0.0)
        self.assertGreater(lines_m[-1]["load_factor"], 0.0)                      # lambda stays positive: sign lives in the pattern
        summary_x, lines_x = self.run_case("x", 1.0, steps=4)
        self.assertTrue(summary_x["checks"]["all_pass"], summary_x["checks"])
        self.assertLess(abs(summary_x["checks"]["control_direction_check"]["other_direction_in"]), 1e-6)
        self.assertEqual(sorted(lines_x[-1]["line_story_shears"]["1"]["by_line_kip"]), ["0", "1"])   # two x-frames (j = 0, 1)
        column = next(h for h in lines_x[-1]["hinges"] if h["member_type"] == "column")
        self.assertEqual(column["active_spring_index"], 1)                     # X push bends about global Y
        # The two Y pushes mirror each other on this symmetric frame.
        summary_p, lines_p = self.run_case("y", 1.0, steps=4)
        self.assertAlmostEqual(lines_p[-1]["base_shear_kip"], -lines_m[-1]["base_shear_kip"], delta=1e-3 * abs(lines_p[-1]["base_shear_kip"]))

    def test_gravity_state_is_kept_apart_from_the_lateral_increment(self):
        summary, lines = self.run_case("y", 1.0, steps=3)
        gravity = lines[0]
        self.assertEqual(gravity["load_factor"], 0.0)
        self.assertAlmostEqual(gravity["reaction_totals"][2], summary["gravity"]["expected_vertical_load_kip"], delta=1e-6 * summary["gravity"]["expected_vertical_load_kip"])
        step = lines[-1]
        g_by_tag = {row["tag"]: row for row in gravity["columns"]}
        for row in step["columns"]:
            base = g_by_tag[row["tag"]]
            self.assertAlmostEqual(row["increment"]["story_shear_kip"], row["story_shear_kip"] - base["story_shear_kip"], places=9)
            self.assertAlmostEqual(row["increment"]["moment_center_j_kip_in"], row["moment_center_j_kip_in"] - base["moment_center_j_kip_in"], places=9)


if __name__ == "__main__":
    unittest.main()
