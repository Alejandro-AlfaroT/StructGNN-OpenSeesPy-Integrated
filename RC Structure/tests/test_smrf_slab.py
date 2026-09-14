import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Slab import choose_slab, evaluate_slab, _beam_inertia, _required_thickness


def geometry(**changes):
    data = dict(num_bay_x=3, num_bay_y=3, num_floor=4,
                bay_x_in=240.0, bay_y_in=240.0, story_h_in=144.0)
    data.update(changes)
    return data


def sections(**changes):
    data = dict(b_beam_in=18.0, h_beam_in=24.0, fc_beam_ksi=4.0)
    data.update(changes)
    return data


class SlabTests(unittest.TestCase):
    def test_equation_boundaries_and_discontinuous_edge_factor(self):
        self.assertIsNone(_required_thickness(240, 1, .2, False))
        result = _required_thickness(240, 1, 2.0, False)
        self.assertAlmostEqual(result["required_thickness_in"], 240 * 1.1 / 45)
        self.assertEqual(result["absolute_minimum_in"], 5)
        self.assertEqual(_required_thickness(240, 1, 2.00001, False)["absolute_minimum_in"], 3.5)
        weak = _required_thickness(240, 1, 2.0, True)
        self.assertAlmostEqual(weak["equation_thickness_in"], result["equation_thickness_in"])
        self.assertAlmostEqual(weak["required_thickness_in"], result["required_thickness_in"] * 1.1)
        # The 10% applies to the expression, not the absolute table floor.
        self.assertEqual(_required_thickness(100, 1, 1.0, True)["required_thickness_in"], 5.0)

    def test_one_thickness_and_complete_panel_inventory(self):
        result = choose_slab(geometry(), sections(), {})
        self.assertEqual(result["panel_count_per_floor"], 9)
        self.assertEqual(result["panel_count_building"], 36)
        self.assertEqual({p["thickness_in"] for p in result["panels"]}, {result["thickness_in"]})
        self.assertTrue(all(p["status"] == "pass" for p in result["panels"]))
        self.assertEqual(sum(p["discontinuous_edge_count"] == 0 for p in result["panels"]), 1)
        self.assertEqual(sum(p["discontinuous_edge_count"] == 1 for p in result["panels"]), 4)
        self.assertEqual(sum(p["discontinuous_edge_count"] == 2 for p in result["panels"]), 4)
        self.assertEqual(result["thickness_in"], 5.5)
        self.assertAlmostEqual(result["self_weight_ksf"], .15 * 5.5 / 12)
        self.assertAlmostEqual(result["total_dead_load_ksf"], .05 + .15 * 5.5 / 12)

    def test_single_panel_has_four_discontinuous_edges(self):
        result = choose_slab(geometry(num_bay_x=1, num_bay_y=1), sections(), {})
        self.assertEqual(result["panels"][0]["discontinuous_edge_count"], 4)
        self.assertEqual({e["flange_count"] for e in result["panels"][0]["edges"]}, {1})

    def test_weak_perimeter_beams_trigger_panel_edge_increase(self):
        result = choose_slab(geometry(), sections(b_beam_in=12, h_beam_in=14), {})
        corner = result["panels"][0]
        interior = next(p for p in result["panels"] if p["panel_type"] == "interior")
        self.assertEqual(corner["weak_discontinuous_edges"], ["x_min", "y_min"])
        self.assertEqual(corner["discontinuous_edge_factor"], 1.1)
        self.assertEqual(interior["discontinuous_edge_factor"], 1.0)
        self.assertGreater(corner["required_thickness_in"], interior["required_thickness_in"])
        self.assertEqual(result["governing_panel_id"], corner["panel_id"])

    def test_gross_t_and_l_inertia_and_conservative_edge_strip(self):
        # Independent three-rectangle composite section: 18x24 web plus
        # two 18x6 projecting slab strips whose centroid is y=3 from top.
        h, bw, hb = 6., 18., 24.
        aw, af = bw * hb, 2 * 18 * h
        yc = (aw * 12 + af * 3) / (aw + af)
        expected_i = bw * hb**3 / 12 + aw * (12-yc)**2 + 2 * 18 * h**3 / 12 + af * (3-yc)**2
        interior, projection, fw = _beam_inertia(bw, hb, h, 2, 240)
        exterior, _, _ = _beam_inertia(bw, hb, h, 1, 240)
        self.assertAlmostEqual(interior, expected_i)
        self.assertEqual(projection, 18.)
        self.assertEqual(fw, 54.)
        self.assertLess(exterior, interior)
        result = choose_slab(geometry(), sections(), {})
        edges = result["panels"][0]["edges"]
        self.assertEqual({e["slab_strip_width_in"] for e in edges}, {240.})

    def test_span_growth_never_uses_independent_panel_thicknesses(self):
        small = choose_slab(geometry(), sections(), {})
        large = choose_slab(geometry(bay_x_in=300, bay_y_in=300), sections(), {})
        self.assertGreater(large["thickness_in"], small["thickness_in"])
        self.assertEqual(len({p["thickness_in"] for p in large["panels"]}), 1)
        self.assertEqual(large["required_thickness_in"], max(p["required_thickness_in"] for p in large["panels"]))

    def test_alpha_recomputed_at_each_trial(self):
        result = choose_slab(geometry(bay_x_in=300, bay_y_in=300), sections(), {})
        history = result["trial_history"]
        self.assertGreater(len(history), 1)
        self.assertGreater(history[0]["alpha_fm_max"], history[-1]["alpha_fm_max"])
        self.assertEqual(history[-1]["status"], "pass")
        self.assertTrue(all(t["status"] != "pass" for t in history[:-1]))

    def test_aspect_ratio_is_clear_span_ratio_and_boundary_two_allowed(self):
        boundary = choose_slab(geometry(bay_x_in=222, bay_y_in=120), sections(), {})
        self.assertEqual(boundary["panels"][0]["beta"], 2.)
        with self.assertRaisesRegex(ValueError, "one-way"):
            choose_slab(geometry(bay_x_in=222.001, bay_y_in=120), sections(), {})

    def test_exhaustion_weak_beams_and_story_conflicts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "No slab thickness passes"):
            choose_slab(geometry(bay_x_in=360, bay_y_in=360), sections(), {"maximum_thickness_in": 5})
        with self.assertRaisesRegex(ValueError, "No slab thickness passes"):
            choose_slab(geometry(), sections(h_beam_in=6), {})
        with self.assertRaisesRegex(ValueError, "story height"):
            choose_slab(geometry(story_h_in=24), sections(), {})

    def test_scope_and_nonfinite_inputs_rejected(self):
        for patch in ({"fy_ksi": 80}, {"prestressed": True}, {"concrete_type": "lightweight"},
                      {"slab_system": "flat_plate"}, {"thickness_increment_in": 0},
                      {"maximum_thickness_in": math.inf}, {"minimum_thickness_in": -1},
                      {"live_load_mass_fraction": 1.1}, {"superimposed_dead_load_ksf": -1}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                choose_slab(geometry(), sections(), patch)
        for patch in ({"num_floor": True}, {"num_bay_x": 1.5}, {"bay_x_in": math.nan}, {"bay_y_in": 18}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                choose_slab(geometry(**patch), sections(), {})

    def test_record_is_deterministic_json_and_does_not_mutate_inputs(self):
        g, s, p = geometry(), sections(), {"superimposed_dead_load_ksf": .04}
        before = copy.deepcopy((g, s, p))
        first = choose_slab(g, s, p)
        self.assertEqual(first, choose_slab(g, s, p))
        self.assertEqual(first, json.loads(json.dumps(first, allow_nan=False)))
        self.assertEqual(before, (g, s, p))
        checks = evaluate_slab(first)
        self.assertEqual(checks[0]["status"], "pass")
        self.assertEqual(checks[1]["status"], "pass")
        # The screen answers only thickness; the strip design carries the rest.
        self.assertEqual({c["id"] for c in checks}, {"slab_thickness_evidence", "slab_thickness_screen"})
        self.assertTrue(all(c["status"] == "pass" for c in checks))

    def test_tampered_saved_evidence_and_missing_record_cannot_pass(self):
        result = choose_slab(geometry(), sections(), {})
        for path in ("thickness_in", "total_dead_load_ksf", "thickness_screen_passed", "method_version"):
            changed = copy.deepcopy(result)
            changed[path] = None
            self.assertEqual(evaluate_slab(changed)[0]["status"], "fail")
        changed = copy.deepcopy(result)
        changed["panels"][0]["alpha_fm"] *= 2
        self.assertEqual(evaluate_slab(changed)[0]["status"], "fail")
        self.assertEqual(evaluate_slab(None)[0]["status"], "not_evaluated")
        self.assertEqual(evaluate_slab({"inputs": {}})[0]["status"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
