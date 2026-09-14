"""Numerical diagnostics, not engineering qualification of the floor model.

The independent benchmark uses the Navier sine series for a uniformly loaded
simply supported thin rectangular plate: D*biharmonic(w)=q. See the university
derivation at https://ocw.tudelft.nl/courses/advanced-structural-analysis/subjects/plate-theory-ii/.
Finite-thickness MITC4 results are compared at h/L=.01, not exactly equated to
the Kirchhoff thin-plate limit. Moments are sampled at Gauss points, not center.
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design.SMRF_Floor_Analysis import analyze_floor


def slab(**changes):
    result = {"thickness_in": 2.0, "concrete_fc_ksi": 4.0,
              "concrete_unit_weight_kcf": .15, "superimposed_dead_load_ksf": .05}
    result.update(changes)
    return result


def geometry(**changes):
    result = {"num_bay_x": 1, "num_bay_y": 1, "bay_x_in": 200., "bay_y_in": 200.}
    result.update(changes)
    return result


def loadcase(**changes):
    result = {"id": "uniform_test", "dead_factor": 0., "live_factor": 1.,
              "live_load_ksf": .00144, "live_pattern": "all"}
    result.update(changes)
    return result


def navier_center(a, b, h, fc, nu, q):
    """Independent thin-plate center displacement and x/y moments per width.

    q_mn=16q/(pi^2*m*n) for odd m,n; w_mn=q_mn/(D*pi^4*k^4).
    Mx=-D(w,xx+nu*w,yy), My=-D(w,yy+nu*w,xx).
    """
    ec = 57.0 * math.sqrt(fc * 1000.0)
    stiffness = ec * h**3 / (12.0 * (1.0 - nu**2))
    w = mx = my = 0.0
    for m in range(1, 100, 2):
        for n in range(1, 100, 2):
            ax, by = (m / a)**2, (n / b)**2
            sine = math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
            coefficient = 16 * q * sine / (math.pi**6 * m * n * (ax + by)**2)
            w += coefficient / stiffness
            mx += coefficient * math.pi**2 * (ax + nu * by)
            my += coefficient * math.pi**2 * (by + nu * ax)
    return w, mx, my


class FloorAnalysisTests(unittest.TestCase):
    def setUp(self):
        ops.wipe()
        self.addCleanup(ops.wipe)

    def test_single_panel_mesh_refinement_to_independent_navier_solution(self):
        expected_w, expected_mx, expected_my = navier_center(200, 200, 2, 4, .2, .00144 / 144)
        errors = []
        moment_errors = []
        for mesh in (4, 8, 16):
            r = analyze_floor(slab(), geometry(), {}, loadcase(), mesh)
            self.assertEqual(r["status"], "diagnostic_complete")
            self.assertTrue(r["equilibrium"]["numerical_balance_passed"])
            self.assertLess(r["equilibrium"]["vertical_relative_error"], 1e-8)
            errors.append(abs(r["maximum_downward_displacement_in"] / expected_w - 1))
            face = r["panels"][0]["direct_bending_face_envelopes_kip_in_per_in"]
            self.assertEqual(face["x"]["top"], 0.)
            self.assertEqual(face["y"]["top"], 0.)
            self.assertAlmostEqual(face["x"]["bottom"], face["y"]["bottom"], places=10)
            moment_errors.append(abs(face["x"]["bottom"] / expected_mx - 1))
            self.assertLess(abs(face["y"]["bottom"] / expected_my - 1), .08)
            self.assertFalse(r["verified"])
            self.assertFalse(r["demand_evidence"]["verified"])
        self.assertTrue(errors[2] < errors[1] < errors[0])
        self.assertTrue(moment_errors[2] < moment_errors[1] < moment_errors[0])
        self.assertLess(errors[2], .002)
        self.assertLess(moment_errors[2], .005)

    def test_rectangular_plate_preserves_local_x_y_moment_ordering(self):
        a, b = 200., 300.
        expected_w, expected_mx, expected_my = navier_center(a, b, 2, 4, .2, .00144 / 144)
        r = analyze_floor(slab(), geometry(bay_x_in=a, bay_y_in=b), {}, loadcase(), 12)
        face = r["panels"][0]["direct_bending_face_envelopes_kip_in_per_in"]
        self.assertGreater(face["x"]["bottom"], face["y"]["bottom"])
        self.assertLess(abs(face["x"]["bottom"] / expected_mx - 1), .02)
        self.assertLess(abs(face["y"]["bottom"] / expected_my - 1), .02)
        self.assertLess(abs(r["maximum_downward_displacement_in"] / expected_w - 1), .01)

    def test_multibay_continuity_has_hogging_and_unique_reaction_accounting(self):
        r = analyze_floor(slab(thickness_in=6), geometry(num_bay_x=2, num_bay_y=2), {},
                          loadcase(dead_factor=1.2, live_factor=1.6, live_load_ksf=.05), 4)
        self.assertEqual(r["mesh"]["node_count"], 81)
        self.assertEqual(len(r["panels"]), 4)
        self.assertEqual({p["panel_id"] for p in r["panels"]},
                         {"panel_x1_y1", "panel_x2_y1", "panel_x1_y2", "panel_x2_y2"})
        self.assertTrue(all(len(p["gauss_point_resultants"]) == 64 for p in r["panels"]))
        self.assertTrue(any(p["direct_bending_face_envelopes_kip_in_per_in"]["x"]["top"] > 0
                            for p in r["panels"]))
        self.assertTrue(any(p["direct_bending_face_envelopes_kip_in_per_in"]["y"]["top"] > 0
                            for p in r["panels"]))
        expected = (1.2 * (.15 * 6 / 12 + .05) + 1.6 * .05) * 400 * 400 / 144
        self.assertAlmostEqual(r["equilibrium"]["applied_downward_kip"], expected)
        lines = sum(x["downward_load_to_line_kip"] for x in r["beam_line_reactions"])
        intersections = sum(x["upward_reaction_kip"] for x in r["intersection_reactions"])
        self.assertAlmostEqual(lines + intersections, r["equilibrium"]["support_upward_kip"])
        nodes = [n for line in r["beam_line_reactions"] for n in line["node_tags_excluding_intersections"]]
        nodes += [r["node"] for r in r["intersection_reactions"]]
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(set(nodes), {r["node"] for r in r["support_node_reactions"]})

    def test_explicit_live_pattern_changes_pressure_not_dead_load(self):
        g = geometry(num_bay_x=2)
        r = analyze_floor(slab(), g, {}, loadcase(dead_factor=1.2, live_factor=1.6,
                                                 live_load_ksf=.05, live_pattern=[[1, 0]]), 4)
        panels = sorted(r["panels"], key=lambda p: p["i"])
        dead = 1.2 * (.15 * 2 / 12 + .05)
        self.assertAlmostEqual(panels[0]["pressure_ksf"], dead)
        self.assertAlmostEqual(panels[1]["pressure_ksf"], dead + 1.6 * .05)
        self.assertFalse(r["demand_evidence"]["all_live_patterns_enveloped"])
        expected = (2 * dead + 1.6 * .05) * 200 * 200 / 144
        self.assertAlmostEqual(r["equilibrium"]["applied_downward_kip"], expected)
        self.assertTrue(r["equilibrium"]["numerical_balance_passed"])

    def test_live_none_means_dead_only_and_input_is_not_mutated(self):
        inputs = [slab(), geometry(), {}, loadcase(dead_factor=1., live_pattern="none")]
        before = copy.deepcopy(inputs)
        r = analyze_floor(*inputs)
        self.assertEqual(before, inputs)
        self.assertAlmostEqual(r["equilibrium"]["applied_downward_kip"], .075 * 200 * 200 / 144)
        self.assertEqual(ops.getNodeTags(), [])
        self.assertEqual(ops.getEleTags(), [])
        json.dumps(r, allow_nan=False)

    def test_existing_frame_domain_is_preserved_and_refused(self):
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(2345, 1, 2, 3)
        with self.assertRaisesRegex(RuntimeError, "empty OpenSees domain"):
            analyze_floor(slab(), geometry(), {}, loadcase())
        self.assertEqual(ops.getNodeTags(), [2345])
        self.assertEqual(ops.nodeCoord(2345), [1., 2., 3.])

    def test_missing_pattern_and_invalid_pattern_indices_are_rejected(self):
        case = loadcase()
        del case["live_pattern"]
        with self.assertRaisesRegex(ValueError, "live_pattern"):
            analyze_floor(slab(), geometry(), {}, case)
        for pattern in ("checkerboard", [[0, 0], [0, 0]], [[1, 0]], [[0.0, 0]], [[True, 0]], {}):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                analyze_floor(slab(), geometry(), {}, loadcase(live_pattern=pattern))

    def test_numerical_input_failures_are_rejected_before_creating_domain(self):
        for mesh in (True, 1, 25, 2.5):
            with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                analyze_floor(slab(), geometry(), {}, loadcase(), mesh)
        for key, value in (("dead_factor", -1), ("live_factor", math.inf), ("live_load_ksf", math.nan)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_floor(slab(), geometry(), {}, loadcase(**{key: value}))
        with self.assertRaises(ValueError):
            analyze_floor(slab(), geometry(), {"slab_poisson_ratio": .5}, loadcase())
        with self.assertRaises(ValueError):
            analyze_floor(slab(), geometry(num_bay_x=100), {}, loadcase(), 24)
        self.assertEqual(ops.getNodeTags(), [])

    def test_solver_failure_stays_unverified_and_cleans_scratch_domain(self):
        with mock.patch("Design.SMRF_Floor_Analysis.ops.analyze", return_value=-3):
            r = analyze_floor(slab(), geometry(), {}, loadcase())
        self.assertEqual(r["status"], "analysis_failed")
        self.assertFalse(r["verified"])
        self.assertNotIn("panels", r)
        self.assertEqual(ops.getNodeTags(), [])

    def test_twisting_is_saved_but_never_silently_ignored_in_design(self):
        r = analyze_floor(slab(), geometry(), {}, loadcase())
        self.assertFalse(r["demand_evidence"]["twisting_design_processed"])
        self.assertFalse(r["demand_evidence"]["reinforcement_design_ready"])
        self.assertIsNone(r["panels"][0]["design_moments_including_twisting"])
        twisting = r["panels"][0]["resultant_extrema"]["mxy_raw"]
        self.assertLess(twisting["minimum"]["value"], 0)
        self.assertGreater(twisting["maximum"]["value"], 0)


if __name__ == "__main__":
    unittest.main()
