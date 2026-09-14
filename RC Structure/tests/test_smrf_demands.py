import json
import math
import sys
import unittest
from pathlib import Path


RC_ROOT = Path(__file__).resolve().parents[1]
if str(RC_ROOT) not in sys.path:
    sys.path.insert(0, str(RC_ROOT))

from Design.SMRF_Demands import (
    demand_scope_checks,
    evaluate_drift_and_stability,
    floor_load_factor,
    story_node_deltas,
    strength_load_combinations,
)


def sample_story(**updates):
    result = {
        "id": 1, "height_in": 144,
        "node_deltas_in": {"edge_a": {"x": 0.1, "y": 0.15},
                           "edge_b": {"x": -0.2, "y": -0.12}},
        "expected_node_ids": ["edge_a", "edge_b"],
        "story_shear_kip": {"x": 100, "y": 80},
        "gravity_above_kip": 1000,
    }
    result.update(updates)
    return result


class StrengthCombinationTests(unittest.TestCase):
    def test_gravity_and_all_orthogonal_signs_are_present(self):
        combinations = strength_load_combinations(1.0)
        self.assertEqual(len(combinations), 18)
        self.assertEqual(len({c["id"] for c in combinations}), 18)
        self.assertEqual(combinations[0]["dead"], 1.4)
        self.assertEqual(combinations[1]["live"], 1.6)
        for family in ("seismic_high_gravity", "seismic_low_gravity"):
            pairs = {(round(c["ex"], 3), round(c["ey"], 3)) for c in combinations if c["family"] == family}
            expected = {(sx * a, sy * b) for a, b in ((1.3, 0.39), (0.39, 1.3))
                        for sx in (-1, 1) for sy in (-1, 1)}
            self.assertEqual(pairs, expected)

    def test_vertical_seismic_and_full_live_factors(self):
        combinations = strength_load_combinations(1.5, redundancy_factor=1.0)
        high = [c for c in combinations if c["family"] == "seismic_high_gravity"]
        low = [c for c in combinations if c["family"] == "seismic_low_gravity"]
        self.assertTrue(all(c["live"] == 1.0 and c["dead"] == 1.5 for c in high))
        self.assertTrue(all(c["live"] == 0 and abs(c["dead"] - 0.6) < 1e-12 for c in low))

    def test_invalid_combination_parameters_rejected(self):
        for kwargs in ({"sds": -1}, {"sds": math.nan}, {"sds": True},
                       {"sds": 1, "redundancy_factor": 0.9},
                       {"sds": 1, "orthogonal_fraction": 0.2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                strength_load_combinations(**kwargs)

    def test_weighted_floor_load_preserves_separate_dead_and_live(self):
        actual = floor_load_factor(1.4, 1.0, 0.15, 0.05)
        self.assertAlmostEqual(actual, 1.3)
        self.assertAlmostEqual(floor_load_factor(0.7, 0, 0.15, 0.05), 0.525)
        with self.assertRaises(ValueError):
            floor_load_factor(1, 1, 0, 0)


class DriftStabilityTests(unittest.TestCase):
    def test_all_nodes_enveloped_and_cd_applied_once(self):
        result = evaluate_drift_and_stability([sample_story()])
        self.assertTrue(result["accepted"])
        x, y = result["stories"]
        self.assertEqual(x["governing_node"], "edge_b")
        self.assertAlmostEqual(x["design_drift_in"], 0.2 * 5.5)
        self.assertAlmostEqual(x["drift_ratio"], 1.1 / 144)
        self.assertAlmostEqual(y["design_drift_in"], 0.15 * 5.5)
        self.assertAlmostEqual(x["allowable_drift_ratio"], 0.02 / 1.3)

    def test_sdc_c_does_not_divide_drift_limit_by_rho(self):
        result = evaluate_drift_and_stability([sample_story()], seismic_design_category="C")
        self.assertEqual(result["stories"][0]["allowable_drift_ratio"], 0.02)

    def test_theta_uses_elastic_not_amplified_drift(self):
        result = evaluate_drift_and_stability([sample_story()])
        x = result["stories"][0]
        self.assertAlmostEqual(x["theta"], 1000 * 0.2 / (100 * 144))
        self.assertAlmostEqual(x["theta_limit"], 0.1)

    def test_drift_exceedance_is_a_design_failure(self):
        story = sample_story(node_deltas_in={"edge_a": {"x": 0.1, "y": 0.15},
                                           "edge_b": {"x": 0.8, "y": -0.12}})
        result = evaluate_drift_and_stability([story])
        self.assertFalse(result["accepted"])
        self.assertIn("demands.story_drift", result["failed"])

    def test_numerical_failure_does_not_assert_instability(self):
        result = evaluate_drift_and_stability([sample_story()], analysis_succeeded=False)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["stories"], [])
        self.assertEqual(result["checks"][0]["status"], "not_evaluated")

    def test_missing_node_coverage_cannot_pass(self):
        story = sample_story()
        story.pop("expected_node_ids")
        result = evaluate_drift_and_stability([story])
        self.assertFalse(result["accepted"])
        self.assertIn("demands.node_coverage", result["not_evaluated"])
        story["expected_node_ids"] = ["edge_a", "edge_b", "missing"]
        result = evaluate_drift_and_stability([story])
        self.assertIn("demands.node_coverage", result["failed"])

    def test_zero_shear_missing_gravity_and_nan_are_unknown(self):
        for change in ({"story_shear_kip": {"x": 0, "y": 80}},
                       {"gravity_above_kip": None},
                       {"node_deltas_in": {"edge_a": {"x": math.nan, "y": 0.1},
                                          "edge_b": {"x": 0.2, "y": 0.1}}}):
            with self.subTest(change=change):
                result = evaluate_drift_and_stability([sample_story(**change)])
                self.assertFalse(result["accepted"])
                self.assertIn("demands.stability", result["not_evaluated"])
                json.dumps(result, allow_nan=False)

    def test_theta_above_point_one_requires_amplification(self):
        # beta=0.5 allows theta up to 0.1818, but first-order 0.12 still
        # needs amplification of displacement AND all member demands.
        story = sample_story(gravity_above_kip=8640, height_in=144,
                             story_shear_kip={"x": 100, "y": 100},
                             node_deltas_in={"a": {"x": 0.2, "y": 0.0}},
                             expected_node_ids=["a"], directions=["x"])
        story["gravity_above_kip"] = 8640  # theta = 0.12
        result = evaluate_drift_and_stability([story], beta=0.5)
        self.assertAlmostEqual(result["stories"][0]["theta"], 0.12)
        self.assertAlmostEqual(result["stories"][0]["required_second_order_amplification"], 1 / 0.88)
        self.assertIn("demands.second_order_effects", result["not_evaluated"])
        self.assertFalse(result["accepted"])
        second_order = evaluate_drift_and_stability([story], beta=0.5, second_order_included=True)
        self.assertAlmostEqual(second_order["stories"][0]["theta"], 0.12 / 1.12)
        self.assertTrue(second_order["accepted"])

    def test_beta_floor_is_enforced_and_empty_results_do_not_pass(self):
        with self.assertRaises(ValueError):
            evaluate_drift_and_stability([sample_story()], beta=0.4)
        self.assertFalse(evaluate_drift_and_stability([])["accepted"])

    def test_aligned_floor_subtraction_uses_relative_drift(self):
        upper = {"a": {"x": 2, "y": -1}, "b": {"x": 3, "y": 0.1}}
        lower = {"a": {"x": 1.5, "y": -0.4}, "b": {"x": 2.8, "y": 0}}
        deltas = story_node_deltas(upper, lower)
        self.assertAlmostEqual(deltas["a"]["x"], 0.5)
        self.assertAlmostEqual(deltas["a"]["y"], -0.6)
        with self.assertRaises(ValueError):
            story_node_deltas(upper, {"a": lower["a"]})

    def test_scope_prerequisites_are_not_silently_passed(self):
        checks = demand_scope_checks()
        self.assertTrue(all(c["status"] == "not_evaluated" for c in checks))
        self.assertIn("demands.elf_eligibility", {c["id"] for c in checks})
        self.assertIn("demands.accidental_torsion", {c["id"] for c in checks})
        self.assertIn("demands.load_scope", {c["id"] for c in checks})


if __name__ == "__main__":
    unittest.main()
