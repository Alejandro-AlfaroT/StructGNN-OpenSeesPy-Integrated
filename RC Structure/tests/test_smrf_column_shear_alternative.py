"""Column design shear (ACI 318-19 18.7.6.1.1): the selectable column-own probable-strength method.

Prepared for the engineering-method review of 2026-09-20. The joint-limited
method stays the production default and its numbers are unchanged; the
alternative is named, its inputs are exposed, and its arithmetic is checked
against the review packet's controls for the dv150_v10 case 0074 section.
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))                        # sibling fixture module

from Design import SMRF_Capacity_Design as capacity                              # noqa: E402
from Design.SMRF_Capacity_Design import (COLUMN_SHEAR_METHOD_COLUMN_OWN, COLUMN_SHEAR_METHOD_JOINT_LIMITED,  # noqa: E402
                                         CLEAR_HEIGHT_PHYSICAL, LOCAL_FORCE_CONVENTION, build_capacity_design,
                                         column_action_envelopes, design_column_shear, probable_beam_strengths,
                                         probable_moment_over_axial_range, probable_section_curve,
                                         section_moment_at_axial)
from test_smrf_capacity_design import state                                     # noqa: E402


def _layers(h, cover, ab, top, side):
    """Bending through h: top/bottom bars in the extreme layers, one bar per side face at each interior level."""
    return [(top * ab, cover)] + [(2 * ab, cover + (h - 2 * cover) * k / (side + 1)) for k in range(1, side + 1)] + [(top * ab, h - cover)]


def _layers_about_z(b, cover, ab, top, side):
    """Bending through b: corners plus side bars in the outer layers (Design_Driver._col_steel_layers_about_z)."""
    outer = (2 + side) * ab
    layers = [(outer, cover)]
    for count in (max(2, top), max(2, top)):
        if count > 2:
            layers += [(ab, cover + (b - 2 * cover) * k / (count - 1)) for k in range(1, count - 1)]
    layers.append((outer, b - cover))
    return sorted(layers, key=lambda layer: layer[1])


# dv150_v10 case 0074: 20x20 in, f'c 8 ksi, twenty #6 (6 top, 6 bottom, 4 per side face), centroid offset 2.375 in.
CASE_0074_SECTION = {"b_in": 20.0, "h_in": 20.0, "fc_ksi": 8.0, "fy_ksi": 75.0, "es_ksi": 29000.0,
                     "layers": _layers(20.0, 2.375, 0.44, 6, 4)}
CASE_0074_STORY_1_RANGE = (-8.593751870951518, 233.62558834756982)      # all 28 combinations, both ends, 21 columns
# The review packet's controls (column_shear_alternative.json, worked_example.md).
PACKET_MPR_ROOT = 6734.716688377332
PACKET_MPR_CURVE_160 = 6734.701908888259
PACKET_VE_STORY_1 = 132.05326839955552
PACKET_VS_LIMIT = 252.22846786197627
PACKET_PHI_V_SCREEN = 189.1713508964822


def hoop_demanding_state():
    """A frame where the column-own Ve changes the hoop demand and the section verdict.

    28x28 in f'c 6 ksi columns with #9 bars (4 top, 3 per side face) over
    14x20 in beams with 3#7 top / 2#7 bottom: strong columns, weak beams.
    Under the joint-limited method the beams cap Ve near 204 kip; the
    column's own probable strength gives about 351 kip, past what #4 hoops
    at 3 in provide and past the 8 sqrt(f'c) b d section limit.
    """
    ab, db = 1.0, 1.128
    cover = 1.5 + 0.5 + db / 2.0
    return state(geometry={"story_h_in": 120.0, "num_floor": 4},
                 sections={"b_col_in": 28.0, "h_col_in": 28.0, "fc_col_ksi": 6.0,
                           "b_beam_in": 14.0, "h_beam_in": 20.0, "fc_beam_ksi": 5.0},
                 beam={"bar_size": 7, "top_bars": 3, "bot_bars": 2, "centroid_offset_in": 1.5 + 0.5 + 0.875 / 2.0},
                 column={"bar_size": 9, "top_bars": 4, "bot_bars": 4, "side_bars": 3, "centroid_offset_in": cover,
                         "layers": _layers(28.0, cover, ab, 4, 3), "layers_about_z": _layers_about_z(28.0, cover, ab, 4, 3)},
                 column_axial_envelope={s: (20.0 * (5 - s), 150.0 * (5 - s)) for s in range(1, 5)},
                 column_shear_demand={s: 15.0 for s in range(1, 5)})


def strengths_for(st):
    return {f"{a}_{p}": probable_beam_strengths(st, p, a) for a in ("x", "y") for p in ("edge", "interior")}


class ExactSectionTests(unittest.TestCase):
    def test_exact_solution_reproduces_the_packet_root_value_at_the_range_maximum(self):
        mpr = section_moment_at_axial(b=20.0, h=20.0, fc=8.0, fy=75.0, es=29000.0, layers=CASE_0074_SECTION["layers"],
                                      axial=CASE_0074_STORY_1_RANGE[1])
        self.assertAlmostEqual(mpr, PACKET_MPR_ROOT, delta=1e-5 * PACKET_MPR_ROOT)
        envelope = probable_moment_over_axial_range(CASE_0074_SECTION, *CASE_0074_STORY_1_RANGE)
        self.assertEqual(envelope["location"], "range_max")
        self.assertAlmostEqual(envelope["at_axial_kip"], CASE_0074_STORY_1_RANGE[1])
        self.assertAlmostEqual(envelope["mpr_kip_in"], PACKET_MPR_ROOT, delta=1e-6 * PACKET_MPR_ROOT)
        self.assertAlmostEqual(envelope["curve_peak_kip_in"], PACKET_MPR_CURVE_160, delta=1e-9 * PACKET_MPR_CURVE_160)
        self.assertLess(envelope["relative_difference_curve_vs_exact"], 1e-5)
        self.assertLess(envelope["relative_difference_refined_vs_exact"], envelope["relative_difference_curve_vs_exact"])
        # The story-envelope form of the packet: twice the maximum over the merged range at 102 in.
        self.assertAlmostEqual(2.0 * envelope["mpr_kip_in"] / 102.0, PACKET_VE_STORY_1, delta=1e-6)

    def test_exact_solution_lies_on_the_sweep_and_refuses_loads_outside_the_domain(self):
        section = CASE_0074_SECTION
        curve = probable_section_curve(20.0, 20.0, 8.0, 75.0, 29000.0, section["layers"], n_pts=1600)
        for p, m in curve[::97]:
            if 0.0 < p < 0.79 * curve[-1][0]:
                self.assertAlmostEqual(section_moment_at_axial(20.0, 20.0, 8.0, 75.0, 29000.0, section["layers"], p),
                                       m, delta=2e-3 * max(1.0, m))
        with self.assertRaisesRegex(ValueError, "outside"):
            section_moment_at_axial(20.0, 20.0, 8.0, 75.0, 29000.0, section["layers"], 1.0e5)
        with self.assertRaisesRegex(ValueError, "outside"):
            section_moment_at_axial(20.0, 20.0, 8.0, 75.0, 29000.0, section["layers"], -1.0e4)

    def test_interior_maximum_is_found_on_the_vertices_not_on_nine_samples(self):
        """An axial range straddling the balanced point: the peak lies between the nine legacy samples."""
        section = CASE_0074_SECTION
        curve = probable_section_curve(20.0, 20.0, 8.0, 75.0, 29000.0, section["layers"])
        p_bal, _ = max(curve, key=lambda point: point[1])
        p_min, p_max = p_bal - 137.0, p_bal + 123.0
        nine = max(capacity._moment_at(curve, p_min + (p_max - p_min) * k / 8.0) for k in range(9))
        envelope = probable_moment_over_axial_range(section, p_min, p_max)
        self.assertEqual(envelope["location"], "interior")
        self.assertGreater(envelope["mpr_kip_in"], nine * (1.0 + 1e-6))
        self.assertGreaterEqual(envelope["curve_peak_kip_in"], nine)
        self.assertTrue(p_min < envelope["at_axial_kip"] < p_max)
        # The range ends are candidates too: a range ending at the peak reports it as the end.
        end = probable_moment_over_axial_range(section, p_min, envelope["at_axial_kip"])
        self.assertIn(end["location"], ("range_max", "interior"))
        self.assertAlmostEqual(end["mpr_kip_in"], envelope["mpr_kip_in"], delta=1e-6 * envelope["mpr_kip_in"])


class ActionEnvelopeTests(unittest.TestCase):
    def test_local_force_convention_against_a_cantilever_column(self):
        """A fixed-base column with the frame's transformation under a global X, then Y, tip force."""
        import openseespy.opensees as ops
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
            forces = list(ops.eleResponse(1, "localForce"))
            shear_index = LOCAL_FORCE_CONVENTION["shear_index"][direction]
            moment_index = LOCAL_FORCE_CONVENTION["moment_index"][direction]
            other = "y" if direction == "x" else "x"
            self.assertAlmostEqual(abs(forces[shear_index["i"]]), 10.0, places=6, msg=direction)
            self.assertAlmostEqual(abs(forces[shear_index["j"]]), 10.0, places=6, msg=direction)
            self.assertAlmostEqual(abs(forces[moment_index["i"]]), 1200.0, places=4, msg=direction)
            self.assertAlmostEqual(forces[LOCAL_FORCE_CONVENTION["shear_index"][other]["i"]], 0.0, places=8)
            self.assertAlmostEqual(forces[LOCAL_FORCE_CONVENTION["moment_index"][other]["i"]], 0.0, places=6)
            ops.wipe()

    def test_envelopes_keep_per_end_ranges_per_direction_shears_and_sources(self):
        def member(n, vy, vz, pi, pj):
            return {"member_type": "column", "local_force_kip_kipin": [n, vy, vz, 0.0, 0.0, 0.0, -n, -vy, -vz, 0.0, 0.0, 0.0],
                    "axial_i_kip": pi, "axial_j_kip": pj}
        combinations = [
            {"id": "gravity", "analysis_succeeded": True,
             "members": {"1": member(100.0, 1.0, 2.0, 100.0, 95.0), "2": member(50.0, 0.5, 0.5, 50.0, 45.0),
                         "3": member(40.0, 0.2, 0.3, 40.0, 38.0), "4": member(20.0, 0.1, 0.1, 20.0, 18.0),
                         "9": {"member_type": "beam_x", "local_force_kip_kipin": [0.0] * 12, "axial_i_kip": 0.0, "axial_j_kip": 0.0}}},
            {"id": "seismic_X", "analysis_succeeded": True,
             "members": {"1": member(-5.0, 3.0, 12.0, -5.0, -8.0), "2": member(60.0, 2.0, 11.0, 60.0, 55.0),
                         "3": member(30.0, 1.0, 4.0, 30.0, 28.0), "4": member(10.0, 0.5, 3.0, 10.0, 8.0)}},
            {"id": "seismic_Y", "analysis_succeeded": True,
             "members": {"1": member(90.0, 9.0, 1.0, 90.0, 85.0), "2": member(55.0, 8.0, 1.0, 55.0, 50.0),
                         "3": member(35.0, 6.0, 0.5, 35.0, 33.0), "4": member(15.0, 5.0, 0.5, 15.0, 13.0)}},
            {"id": "broken", "analysis_succeeded": False, "members": {}},
        ]
        result = column_action_envelopes(combinations, columns_per_story=2)
        self.assertEqual(result["combinations_used"], 3)
        self.assertEqual(result["combinations_skipped"], ["broken"])
        self.assertEqual(result["axial"][1], (-8.0, 100.0))
        self.assertEqual(result["shear"][1], 12.0)
        story_1 = result["detail"][1]
        self.assertEqual(story_1["axial_by_end"]["i"]["min_kip"], -5.0)
        self.assertEqual(story_1["axial_by_end"]["j"]["min_kip"], -8.0)
        self.assertEqual(story_1["axial_by_end"]["i"]["max_source"], {"combination": "gravity", "column_tag": 1})
        self.assertEqual(story_1["axial_by_end"]["j"]["min_source"], {"combination": "seismic_X", "column_tag": 1})
        self.assertEqual(story_1["shear_by_direction_kip"], {"x": 12.0, "y": 9.0})
        self.assertEqual(story_1["shear_sources"]["x"], {"combination": "seismic_X", "column_tag": 1, "end": "i", "local_index": 2})
        self.assertEqual(story_1["shear_sources"]["y"]["combination"], "seismic_Y")
        self.assertEqual(story_1["observations"], 6)
        self.assertEqual(result["detail"][2]["shear_by_direction_kip"], {"x": 4.0, "y": 6.0})
        with self.assertRaisesRegex(ValueError, "no successful"):
            column_action_envelopes([{"id": "x", "analysis_succeeded": False, "members": {}}], 2)


class MethodSelectionTests(unittest.TestCase):
    def test_default_is_the_joint_limited_method_with_its_numbers_unchanged(self):
        st = state()
        strengths = strengths_for(st)
        result = design_column_shear(st, strengths)
        self.assertEqual(result["column_shear_method"], COLUMN_SHEAR_METHOD_JOINT_LIMITED)
        entry = result["stories"]["x"][1]
        ln = 120.0 - 16.0
        p_min, p_max = st["column_axial_envelope"][1]
        diagram = capacity.column_probable_pm(st)
        legacy_mpr = max(capacity._moment_at(diagram, p_min + (p_max - p_min) * k / 8.0) for k in range(9))
        self.assertAlmostEqual(entry["mpr_column_kip_in"], legacy_mpr)
        self.assertAlmostEqual(entry["ve_own_kip"], 2.0 * legacy_mpr / ln)
        self.assertAlmostEqual(entry["ve_kip"], max(min(entry["ve_own_kip"], entry["ve_joint_limited_kip"]), entry["vu_analysis_kip"]))
        self.assertEqual(entry["vu_analysis_kip"], 12.0)
        self.assertEqual(entry["clear_height_in"], ln)
        self.assertEqual(entry["method"], COLUMN_SHEAR_METHOD_JOINT_LIMITED)
        # The alternative's numbers are recorded beside it without being selected.
        self.assertIn("ve_own_envelope_kip", entry)
        self.assertIn("mpr_by_end_and_sense", entry["column_own"])
        self.assertEqual(entry["column_own"]["axial_range_basis"],
                         "per-end ranges not recorded: the story range is used for both ends")
        full = build_capacity_design(st)
        self.assertEqual(full["column_shear_method"], COLUMN_SHEAR_METHOD_JOINT_LIMITED)
        self.assertEqual(full["method_version"], capacity.METHOD_VERSION)
        section_check = next(c for c in full["checks"] if c["id"] == "column.capacity_shear_section")
        self.assertEqual(section_check["details"]["column_shear_method"], COLUMN_SHEAR_METHOD_JOINT_LIMITED)

    def test_column_own_method_removes_the_sharing_reduction_and_keeps_the_analysis_floor(self):
        st = state()
        strengths = strengths_for(st)
        result = design_column_shear(st, strengths, method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        self.assertEqual(result["column_shear_method"], COLUMN_SHEAR_METHOD_COLUMN_OWN)
        for axis in ("x", "y"):
            for story, entry in result["stories"][axis].items():
                own = entry["column_own"]
                self.assertAlmostEqual(entry["ve_own_envelope_kip"], own["sway_sums_kip_in"][own["governing_sway"]] / entry["clear_height_in"])
                self.assertAlmostEqual(entry["ve_kip"], max(entry["ve_own_envelope_kip"], entry["vu_analysis_kip"]))
                self.assertGreaterEqual(entry["ve_kip"], entry["ve_joint_limited_kip"] - 1e-9)   # no reduction below the beams' delivery
                self.assertGreaterEqual(entry["ve_own_story_envelope_kip"], entry["ve_own_envelope_kip"] - 1e-9)
                self.assertAlmostEqual(own["form_difference_kip"], entry["ve_own_story_envelope_kip"] - entry["ve_own_envelope_kip"])
                self.assertEqual(entry["phi"], 0.75)
                self.assertIn("exact", own["strength_basis"])
                self.assertEqual(own["section"]["bar_count"], 4 + 4 + 2 * 4)
                for end in ("i", "j"):
                    for sense in ("positive", "negative"):
                        self.assertIn("at_axial_kip", own["mpr_by_end_and_sense"][end][sense])
        # A large analysis shear becomes the floor.
        heavy = state(column_shear_demand={s: 500.0 for s in range(1, 9)})
        floor = design_column_shear(heavy, strengths_for(heavy), method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        self.assertEqual(floor["stories"]["x"][1]["ve_kip"], 500.0)
        # The state's own selection is honoured, and unknown names are refused.
        selected = build_capacity_design({**st, "column_shear_method": COLUMN_SHEAR_METHOD_COLUMN_OWN})
        self.assertEqual(selected["column_shear_method"], COLUMN_SHEAR_METHOD_COLUMN_OWN)
        with self.assertRaisesRegex(ValueError, "unknown column shear method"):
            design_column_shear({**st, "column_shear_method": "something_else"}, strengths)

    def test_symmetric_sections_reduce_to_twice_the_maximum_and_asymmetric_ones_do_not(self):
        # A mirror-symmetric cage (the shared fixture's rounded interior layers are not quite symmetric).
        st = state(column={"layers": _layers(26.0, 2.375, 0.44, 4, 4)})
        strengths = strengths_for(st)
        own = design_column_shear(st, strengths, method=COLUMN_SHEAR_METHOD_COLUMN_OWN)["stories"]["x"][1]["column_own"]
        for end in ("i", "j"):
            self.assertAlmostEqual(own["mpr_by_end_and_sense"][end]["positive"]["mpr_kip_in"],
                                   own["mpr_by_end_and_sense"][end]["negative"]["mpr_kip_in"], delta=1e-6)
        self.assertAlmostEqual(own["sway_sums_kip_in"]["a"], own["sway_sums_kip_in"]["b"], delta=1e-6)
        # Same story range at both ends (no per-end record): the general form equals the story envelope.
        self.assertAlmostEqual(own["form_difference_kip"], 0.0, delta=1e-9)
        # 6 top / 2 bottom bars: the two senses differ and the governing sway pairs the strong sense at one
        # end with the weak sense at the other, so Ve is neither 2 max / ln nor 2 min / ln.
        cover = 2.375
        asym = state(column={"top_bars": 6, "bot_bars": 2, "side_bars": 4,
                             "layers": [(6 * 0.44, cover)] + [(2 * 0.44, cover + (26.0 - 2 * cover) * k / 5) for k in range(1, 5)]
                                       + [(2 * 0.44, 26.0 - cover)]})
        entry = design_column_shear(asym, strengths_for(asym), method=COLUMN_SHEAR_METHOD_COLUMN_OWN)["stories"]["x"][1]
        own = entry["column_own"]
        pos = own["mpr_by_end_and_sense"]["i"]["positive"]["mpr_kip_in"]
        neg = own["mpr_by_end_and_sense"]["i"]["negative"]["mpr_kip_in"]
        self.assertNotAlmostEqual(pos, neg, delta=1.0)
        strong, weak = max(pos, neg), min(pos, neg)
        self.assertAlmostEqual(own["sway_sums_kip_in"][own["governing_sway"]], strong + weak, delta=1e-6)
        self.assertLess(entry["ve_own_envelope_kip"], 2.0 * strong / entry["clear_height_in"] - 1e-6)
        self.assertGreater(entry["ve_own_envelope_kip"], 2.0 * weak / entry["clear_height_in"] + 1e-6)
        self.assertAlmostEqual(entry["ve_own_story_envelope_kip"], 2.0 * strong / entry["clear_height_in"], delta=1e-6)

    def test_rectangular_column_directions_are_not_interchanged(self):
        st = state(sections={"b_col_in": 20.0, "h_col_in": 26.0},
                   column={"layers_about_z": [(2.64, 2.375), (1.76, 10.0), (2.64, 17.625)]})
        strengths = strengths_for(st)
        with patch.object(capacity, "probable_moment_over_axial_range", wraps=capacity.probable_moment_over_axial_range) as env:
            result = design_column_shear(st, strengths, method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        sections = {(call.args[0]["b_in"], call.args[0]["h_in"]) for call in env.call_args_list}
        self.assertEqual(sections, {(20.0, 26.0), (26.0, 20.0)})
        x, y = result["stories"]["x"][1]["column_own"]["section"], result["stories"]["y"][1]["column_own"]["section"]
        self.assertEqual((x["b_in"], x["h_in"], x["bending_depth_in"], x["shear_width_in"]), (20.0, 26.0, 26.0, 20.0))
        self.assertEqual((y["b_in"], y["h_in"], y["bending_depth_in"], y["shear_width_in"]), (26.0, 20.0, 20.0, 26.0))
        self.assertEqual([tuple(t) for t in y["layers_area_in2_depth_in"]], [tuple(t) for t in st["column"]["layers_about_z"]])
        self.assertNotAlmostEqual(result["stories"]["x"][1]["ve_own_envelope_kip"], result["stories"]["y"][1]["ve_own_envelope_kip"])
        self.assertEqual(result["vs_limit_by_direction_kip"]["x"], 8.0 * math.sqrt(4000.0) * 20.0 * (26.0 - 2.375) / 1000.0)
        self.assertEqual(result["vs_limit_by_direction_kip"]["y"], 8.0 * math.sqrt(4000.0) * 26.0 * (20.0 - 2.375) / 1000.0)

    def test_clear_height_conventions_are_explicit_and_reported_as_a_method_difference(self):
        st = state()
        strengths = strengths_for(st)
        uniform = design_column_shear(st, strengths, method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        physical = design_column_shear({**st, "column_clear_height_convention": CLEAR_HEIGHT_PHYSICAL}, strengths,
                                       method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        base_u, base_p = uniform["stories"]["x"][1], physical["stories"]["x"][1]
        self.assertEqual(base_u["face_to_face_clear_height_in"], 104.0)
        self.assertEqual(base_u["physical_clear_height_in"], 112.0)                     # base to the soffit of a 16-in beam
        self.assertEqual((base_u["clear_height_in"], base_p["clear_height_in"]), (104.0, 112.0))
        self.assertAlmostEqual(base_u["ve_own_envelope_at_physical_height_kip"], base_p["ve_own_envelope_kip"])
        self.assertAlmostEqual(base_u["ve_own_envelope_kip"] * 104.0, base_p["ve_own_envelope_kip"] * 112.0, delta=1e-6)
        self.assertGreater(base_u["ve_own_envelope_kip"], base_p["ve_own_envelope_kip"])   # the uniform convention is conservative
        self.assertEqual(uniform["stories"]["x"][2]["physical_clear_height_in"], 104.0)
        self.assertEqual(uniform["clear_height_convention"], "uniform_face_to_face")
        self.assertEqual(physical["clear_height_convention"], CLEAR_HEIGHT_PHYSICAL)
        # The joint-limited method never moves off the face-to-face height.
        legacy = design_column_shear({**st, "column_clear_height_convention": CLEAR_HEIGHT_PHYSICAL}, strengths)
        self.assertEqual(legacy["stories"]["x"][1]["clear_height_in"], 104.0)
        self.assertEqual(legacy["clear_height_convention"], "uniform_face_to_face")
        with self.assertRaisesRegex(ValueError, "clear height convention"):
            design_column_shear({**st, "column_clear_height_convention": "guess"}, strengths)

    def test_per_direction_analysis_shear_is_used_when_recorded(self):
        st = state()
        detail = {s: {"axial_by_end": {"i": {"min_kip": 10.0 * (9 - s), "max_kip": 40.0 * (9 - s), "min_source": None, "max_source": None},
                                       "j": {"min_kip": 10.0 * (9 - s) - 2.0, "max_kip": 40.0 * (9 - s) - 2.0, "min_source": None, "max_source": None}},
                      "shear_by_direction_kip": {"x": 9.0, "y": 12.0}, "shear_sources": {"x": None, "y": None},
                      "shear_any_direction_kip": 12.0, "observations": 1}
                  for s in range(1, 9)}
        st["column_action_envelopes"] = detail
        strengths = strengths_for(st)
        own = design_column_shear(st, strengths, method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        self.assertEqual(own["stories"]["x"][1]["vu_analysis_kip"], 9.0)
        self.assertEqual(own["stories"]["y"][1]["vu_analysis_kip"], 12.0)
        self.assertEqual(own["stories"]["x"][1]["vu_direction_kip"], 9.0)
        self.assertEqual(own["stories"]["x"][1]["vu_any_direction_kip"], 12.0)
        self.assertIn("x-frame shear axis", own["stories"]["x"][1]["vu_basis"])
        # Per-end ranges: the top end's lower load gives a lower Mpr there, so the general form sits below the story envelope.
        entry = own["stories"]["x"][1]
        self.assertEqual(entry["column_own"]["axial_by_end_kip"]["j"]["max_kip"], 40.0 * 8 - 2.0)
        self.assertGreater(entry["column_own"]["form_difference_kip"], 0.0)
        # The joint-limited method keeps the legacy any-direction scalar.
        legacy = design_column_shear(st, strengths)
        self.assertEqual(legacy["stories"]["x"][1]["vu_analysis_kip"], 12.0)


class HoopDemandTests(unittest.TestCase):
    def test_the_alternative_changes_the_hoop_demand_and_the_section_verdict_on_a_strong_column_frame(self):
        st = hoop_demanding_state()
        strengths = strengths_for(st)
        limited = design_column_shear(st, strengths)
        own = design_column_shear(st, strengths, method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        gov_l, gov_o = limited["governing"], own["governing"]
        self.assertLess(gov_l["ve_kip"], gov_o["ve_kip"] * 0.7)                        # the beams capped Ve well below the column's own
        self.assertAlmostEqual(gov_l["ve_kip"], gov_l["ve_joint_limited_kip"])
        self.assertAlmostEqual(gov_o["ve_kip"], gov_o["ve_own_envelope_kip"])
        self.assertTrue(limited["section_adequate"])
        self.assertFalse(own["section_adequate"])                                        # Vs required > 8 sqrt(f'c) b d
        self.assertGreater(gov_o["vs_required_kip"], own["vs_limit_by_direction_kip"][gov_o["axis"]])
        self.assertIsNotNone(limited["hoops"])
        self.assertIsNotNone(own["hoops"])
        self.assertNotEqual((limited["hoops"]["bar_size"], limited["hoops"]["spacing_in"]),
                            (own["hoops"]["bar_size"], own["hoops"]["spacing_in"]))
        self.assertLess(own["hoops"]["spacing_from_shear_in"], limited["hoops"]["spacing_from_shear_in"])
        # The provided-steel screen with each method's own hoops.
        self.assertTrue(limited["screen"]["available"] and own["screen"]["available"])
        self.assertLess(limited["screen"]["governing_ratio"], own["screen"]["governing_ratio"])
        # Through the capacity design: the section check fails under the alternative only.
        limited_full = build_capacity_design(st)
        own_full = build_capacity_design({**st, "column_shear_method": COLUMN_SHEAR_METHOD_COLUMN_OWN})
        status = lambda full: next(c["status"] for c in full["checks"] if c["id"] == "column.capacity_shear_section")   # noqa: E731
        self.assertEqual((status(limited_full), status(own_full)), ("pass", "fail"))

    def test_case_0074_controls_reproduce_through_the_designer(self):
        """The packet's screen for the saved 0074 cage: Ve 132.0533 kip, phi V 189.1714 kip, ratio 0.698."""
        cover = 2.375
        st = state(geometry={"num_bay_x": 2, "num_bay_y": 6, "num_floor": 6, "bay_x_in": 120.0, "bay_y_in": 168.0, "story_h_in": 120.0},
                   sections={"b_col_in": 20.0, "h_col_in": 20.0, "fc_col_ksi": 8.0, "b_beam_in": 10.0, "h_beam_in": 18.0, "fc_beam_ksi": 4.0},
                   beam={"bar_size": 6, "top_bars": 3, "bot_bars": 2, "centroid_offset_in": cover},
                   column={"bar_size": 6, "top_bars": 6, "bot_bars": 6, "side_bars": 4, "centroid_offset_in": cover,
                           "layers": _layers(20.0, cover, 0.44, 6, 4), "layers_about_z": _layers_about_z(20.0, cover, 0.44, 6, 4)},
                   column_axial_envelope={1: CASE_0074_STORY_1_RANGE, **{s: (-5.0, 200.0 - 30.0 * s) for s in range(2, 7)}},
                   column_shear_demand={1: 10.827065444656537, **{s: 10.0 for s in range(2, 7)}})
        own = design_column_shear(st, strengths_for(st), method=COLUMN_SHEAR_METHOD_COLUMN_OWN)
        entry = own["stories"]["x"][1]
        self.assertAlmostEqual(entry["ve_own_story_envelope_kip"], PACKET_VE_STORY_1, delta=1e-6)
        self.assertAlmostEqual(entry["ve_kip"], PACKET_VE_STORY_1, delta=1e-6)      # same range at both ends: general == envelope
        self.assertEqual(entry["clear_height_in"], 102.0)
        self.assertTrue(entry["vc_zero"])                                            # P_min -8.6 kip < Ag f'c / 20 = 160 kip
        self.assertAlmostEqual(own["vs_limit_by_direction_kip"]["x"], PACKET_VS_LIMIT)
        # The saved cage (#4, six legs each way, 4 in) screened as the packet did: Vc = 0, Vs capped at the section limit.
        hoops = {"bar_size": 4, "spacing_in": 4.0, "by_direction": {axis: {"av_in2": 6 * 0.20, "d_in": 17.625, "legs": 6} for axis in ("x", "y")}}
        screen = capacity.column_shear_screen(own["stories"], hoops, own["vs_limit_by_direction_kip"], 60.0)
        row = screen["rows"]["x"][1]
        self.assertAlmostEqual(row["phi_vn_capped_kip"], PACKET_PHI_V_SCREEN, delta=1e-6)
        self.assertAlmostEqual(row["vs_provided_kip"], 317.25, delta=1e-9)
        self.assertAlmostEqual(row["ratio"], PACKET_VE_STORY_1 / PACKET_PHI_V_SCREEN, delta=1e-6)
        self.assertTrue(row["pass"])


class EvidenceIdentityTests(unittest.TestCase):
    def test_recomputation_reads_old_evidence_as_the_joint_limited_method_and_flags_a_changed_method(self):
        from Design.SMRF_Design_Evidence import capacity_design_recomputation
        st = state()
        saved = json.loads(json.dumps(build_capacity_design(st), allow_nan=False))
        record = {"capacity_design": saved}
        with patch("Design.SMRF_Design_Evidence.capacity_state_from_record", return_value=copy.deepcopy(st)):
            self.assertTrue(capacity_design_recomputation(record)["consistent"])
            # Evidence written before the method was named: read as the joint-limited method, no difference.
            legacy = copy.deepcopy(record)
            del legacy["capacity_design"]["column_shear_method"]
            self.assertTrue(capacity_design_recomputation(legacy)["consistent"])
            # Evidence produced with the alternative against a state that selects the default: a named difference.
            other = copy.deepcopy(record)
            other["capacity_design"]["column_shear_method"] = COLUMN_SHEAR_METHOD_COLUMN_OWN
            result = capacity_design_recomputation(other)
            self.assertFalse(result["consistent"])
            self.assertTrue(any("column_shear_method" in d for d in result["differences"]))


# ---------------------------------------------------------------------------------------------------
# CS-N1 (engineering review of 2026-09-20): the section model's axial force jumps down where the
# Whitney block reaches a bar layer (discrete displaced-concrete subtraction), so a load inside a
# jump has two equilibrium roots and the old single-root bisection could pick the smaller moment.
# The reviewer's probes (section_root_counterexamples.json, values copied here) and the repaired
# branch-aware solver.
# ---------------------------------------------------------------------------------------------------

REVIEW_0074 = {"b_in": 20.0, "h_in": 20.0, "fc_ksi": 8.0, "fy_ksi": 75.0, "es_ksi": 29000.0,
               "layers": [(2.64, 2.375), (0.88, 5.425), (0.88, 8.475), (0.88, 11.525), (0.88, 14.575), (2.64, 17.625)]}
REVIEW_HOOP_FIXTURE = {"b_in": 28.0, "h_in": 28.0, "fc_ksi": 6.0, "fy_ksi": 75.0, "es_ksi": 29000.0,
                       "layers": [(4.0, 2.564), (2.0, 8.282), (2.0, 14.0), (2.0, 19.718), (4.0, 25.436)]}
# (section, block-entry depth, target axial, reviewer roots [(c, M)], old single-root moment)
REVIEW_PROBES = [
    (REVIEW_0074, 2.375, -38.699452631578936, [(3.5982605686674143, 5038.136091961434), (3.7102012336811976, 5027.511788012749)], 5027.511788007589),
    (REVIEW_0074, 5.425, 622.4942488479262, [(8.324832185780743, 8420.380285384956), (8.367515829948491, 8410.632635025366)], 8410.632635023678),
    (REVIEW_0074, 8.475, 1304.086159292035, [(13.060897464955083, 8788.969329208729)], 8788.969329208756),
    (REVIEW_0074, 11.525, 1867.3814056399133, [(17.703826890700576, 8175.177357184233), (17.75772826237926, 8171.133729408968)], 8171.133729409281),
    (REVIEW_0074, 14.575, 2360.1172075471695, [(22.393903871912038, 6520.046053565814), (22.452260447057405, 6519.825832085847)], 6519.825832088182),
    (REVIEW_HOOP_FIXTURE, 2.564, -307.06080000000003, [(3.3634417045126836, 8965.520658150279), (3.474644310036896, 8965.517296544516)], 8965.520658138892),
    (REVIEW_HOOP_FIXTURE, 8.282, 984.5707108427914, [(11.071948352593083, 18902.78270672484)], 18902.78270671835),
    (REVIEW_HOOP_FIXTURE, 14.0, 2267.8002857142856, [(18.63345476979832, 19197.047797435527), (18.699914313535096, 19175.570018214177)], 19175.570018211893),
    (REVIEW_HOOP_FIXTURE, 19.718, 3325.173943969977, [(26.25159041858023, 15132.77034076982), (26.329763739289874, 15130.485115812467)], 15132.770340778432),
]


def _args(section):
    return (section["b_in"], section["h_in"], section["fc_ksi"], section["fy_ksi"], section["es_ksi"], section["layers"])


class BranchAwareSolverTests(unittest.TestCase):
    def test_branches_split_at_every_block_entry_depth_and_are_monotone(self):
        branches = capacity.section_branches(*_args(REVIEW_0074))
        b1 = 0.65                                                          # f'c 8 ksi
        self.assertEqual(len(branches), 7)                                 # six distinct depths -> seven branches
        self.assertEqual([b["block_entry_depth_in"] for b in branches][1:], [2.375, 5.425, 8.475, 11.525, 14.575, 17.625])
        for k, branch in enumerate(branches[1:], start=1):
            self.assertAlmostEqual(branch["c_lo"], branch["block_entry_depth_in"] / b1)
            self.assertEqual(sum(branch["displaced"]), k)
            # The jump at the branch start is the displaced concrete of the entering layer, 0.85 f'c A_k.
            area = sum(a for a, d in REVIEW_0074["layers"] if d == branch["block_entry_depth_in"])
            before = capacity._section_forces_masked(*_args(REVIEW_0074), branch["c_lo"], branches[k - 1]["displaced"])[0]
            after = capacity._section_forces_masked(*_args(REVIEW_0074), branch["c_lo"], branch["displaced"])[0]
            self.assertAlmostEqual(before - after, 0.85 * 8.0 * area, places=9)
            # Monotone non-decreasing axial force within the branch.
            previous = None
            for j in range(41):
                c = branch["c_lo"] + (branch["c_hi"] - branch["c_lo"]) * j / 40.0
                value = capacity._section_forces_masked(*_args(REVIEW_0074), c, branch["displaced"])[0]
                if previous is not None:
                    self.assertGreaterEqual(value, previous - 1e-9)
                previous = value

    def test_reviewer_counterexamples_return_every_root_and_select_the_largest_moment(self):
        for section, depth, target, expected_roots, old_moment in REVIEW_PROBES:
            roots = capacity.section_equilibrium_roots(*_args(section), target)
            found = {round(r["c_in"], 6): r for r in roots}
            for c_expected, m_expected in expected_roots:
                self.assertIn(round(c_expected, 6), found, (depth, c_expected))
                self.assertAlmostEqual(found[round(c_expected, 6)]["moment_kip_in"], m_expected, delta=1e-6 * m_expected)
            for r in roots:
                self.assertLess(abs(r["residual_kip"]), 1e-9, (depth, r))
                # Every root is a root of the raw model too, not only of its masked branch.
                raw_p, raw_m = capacity._section_forces(*_args(section), r["c_in"])
                self.assertLess(abs(raw_p - target), 1e-9, (depth, r["c_in"]))
                self.assertAlmostEqual(raw_m, r["moment_kip_in"], delta=1e-9 * max(1.0, raw_m))
            largest = max(r["moment_kip_in"] for r in roots)
            self.assertGreaterEqual(largest, max(m for _, m in expected_roots) - 1e-6)
            self.assertGreaterEqual(largest, old_moment - 1e-6)
            selected = capacity.section_moment_at_axial(*_args(section), target)
            self.assertAlmostEqual(selected, largest, delta=1e-9 * largest)
            detail = capacity.section_moment_at_axial(*_args(section), target, detail=True)
            self.assertEqual(detail["root_count"], len(roots))
            self.assertLess(detail["max_residual_kip"], 1e-9)
            # The envelope at zero width and over a range straddling the jump never falls below the largest root.
            zero_width = probable_moment_over_axial_range(section, target, target, dense_check_points=4000)
            self.assertAlmostEqual(zero_width["mpr_kip_in"], largest, delta=1e-9 * largest)
            self.assertTrue(zero_width["numerics"]["dense_sweep"]["envelope_covers_sweep"])
            self.assertLess(zero_width["numerics"]["round_trip_moment_residual_kip_in"], 1e-6)
            around = probable_moment_over_axial_range(section, target - 5.0, target + 5.0, dense_check_points=4000)
            self.assertGreaterEqual(around["mpr_kip_in"], largest - 1e-9 * largest)
            self.assertTrue(around["numerics"]["dense_sweep"]["envelope_covers_sweep"])

    def test_two_reviewer_single_root_probes_carry_a_second_larger_root_in_the_preceding_branch(self):
        """Reported back to the review: under the declared model these loads also sit inside a jump."""
        for section, depth, target, expected_roots, _old in REVIEW_PROBES:
            if len(expected_roots) != 1:
                continue
            roots = sorted(capacity.section_equilibrium_roots(*_args(section), target), key=lambda r: r["c_in"])
            self.assertEqual(len(roots), 2, depth)
            self.assertAlmostEqual(roots[1]["c_in"], expected_roots[0][0], places=6)
            self.assertGreater(roots[0]["moment_kip_in"], roots[1]["moment_kip_in"])
            self.assertEqual(roots[0]["branch"] + 1, roots[1]["branch"])
            raw_p, _ = capacity._section_forces(*_args(section), roots[0]["c_in"])
            self.assertLess(abs(raw_p - target), 1e-9)

    def test_envelope_evidence_on_the_controls_and_across_several_jumps(self):
        control = probable_moment_over_axial_range(CASE_0074_SECTION, *CASE_0074_STORY_1_RANGE, dense_check_points=20000)
        self.assertAlmostEqual(control["mpr_kip_in"], PACKET_MPR_ROOT, delta=1e-6 * PACKET_MPR_ROOT)   # unchanged by the repair
        self.assertEqual(control["location"], "range_max")
        self.assertEqual(len(control["numerics"]["roots_at_peak_axial"]), 1)                     # a single root in the saved range
        self.assertTrue(control["numerics"]["dense_sweep"]["envelope_covers_sweep"])
        self.assertLess(control["numerics"]["max_root_residual_kip"], 1e-9)
        self.assertLess(control["numerics"]["round_trip_moment_residual_kip_in"], 1e-6)
        # A range spanning four block-entry jumps: the envelope covers the dense sweep.
        wide = probable_moment_over_axial_range(REVIEW_0074, -100.0, 2400.0, dense_check_points=20000)
        self.assertTrue(wide["numerics"]["dense_sweep"]["envelope_covers_sweep"])
        self.assertGreaterEqual(len(wide["numerics"]["branches_checked"]), 5)
        self.assertGreaterEqual(wide["mpr_kip_in"], max(m for probe in REVIEW_PROBES[:5] for _, m in probe[3]) - 1e-6)
        self.assertLess(wide["numerics"]["dense_sweep"]["envelope_minus_dense_kip_in"], 5.0)     # the sweep sits just below the peak
        # Golden-section brackets closed to the neutral-axis tolerance.
        for width in wide["numerics"]["golden_section_bracket_widths_in"]:
            self.assertLessEqual(width, 1e-14 * 20.0 * 1.01)

    def test_out_of_domain_loads_and_ranges_are_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            capacity.section_equilibrium_roots(*_args(REVIEW_0074), 1.0e5)
        with self.assertRaisesRegex(ValueError, "domain"):
            probable_moment_over_axial_range(REVIEW_0074, -100.0, 1.0e5)


if __name__ == "__main__":
    unittest.main()
