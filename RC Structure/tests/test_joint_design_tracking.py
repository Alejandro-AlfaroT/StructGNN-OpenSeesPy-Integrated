"""Physical connectivity/area/evidence checks; no nonlinear analysis."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.Joint_Design_Tracking import build_joint_design_tracking


def record():
    return {"geometry": {"num_bay_x": 2, "num_bay_y": 2, "num_floor": 2,
                         "bay_x_in": 240., "bay_y_in": 300., "story_h_in": 168.},
            "sections": {"b_col_in": 40., "h_col_in": 20., "b_beam_in": 10., "h_beam_in": 24.},
            "capacity_design": {"joints": {"joints": {"joint_shear/floor/corner/x": {
                "aj_in2": 600., "vj_kip": 100., "phi_vn_kip": 200., "passes": True,
                "beams_in_direction": 1, "beams_perpendicular": 1, "evidence_complete": True,
                "slab_steel_in_tension_included": True}}}},
            "qualification": {"accepted": False, "checks": []}, "slab_reinforcement": {"layout": None}}


class JointTrackingTests(unittest.TestCase):
    def test_rectangular_axes_keep_shear_area_distinct_from_panel_face(self):
        r = build_joint_design_tracking(record())
        self.assertEqual((r["physical_joint_count"], r["directional_row_count"]), (18, 36))
        first = {row["axis"]: row for row in r["rows"] if row["node_tag"] == 10}
        self.assertEqual(first["x"]["geometry"]["effective_shear_area_aj_in2"], 600.)
        self.assertEqual(first["y"]["geometry"]["effective_shear_area_aj_in2"], 800.)
        self.assertEqual(first["x"]["geometry"]["panel_face_area_in2"], 480.)
        self.assertEqual(first["y"]["geometry"]["panel_face_area_in2"], 960.)
        self.assertEqual(first["x"]["saved_shear_dcr"], .5)
        self.assertEqual(first["x"]["tracking_status"], "unresolved")
        self.assertFalse(first["x"]["slab_layout_established"])
        self.assertTrue(first["x"]["saved_slab_inclusion_claim"])

    def test_roof_and_edges_follow_actual_members(self):
        r = build_joint_design_tracking(record())
        rows = {x["id"]: x for x in r["rows"]}
        # Nodes at floor 1: 10=(0,0), 11=(1,0), 13=(0,1), 14=(1,1).
        self.assertEqual(rows["joint_11/x"]["kind"], "edge_x")
        self.assertEqual(rows["joint_13/x"]["kind"], "edge_y")
        self.assertEqual(rows["joint_14/x"]["kind"], "interior")
        self.assertEqual(rows["joint_11/x"]["xyz_in"], [240., 0., 168.])
        self.assertEqual(len(rows["joint_14/x"]["beams_in_axis"]), 2)
        for row in r["rows"]:
            self.assertEqual(len(row["columns"]), 1 if row["level"] == "roof" else 2)
        self.assertFalse(r["base_connections_included"])

    def test_missing_evidence_stays_missing_and_saved_mismatch_is_visible(self):
        source = record()
        source["capacity_design"]["joints"]["joints"]["joint_shear/floor/corner/x"]["aj_in2"] = 999.
        rows = build_joint_design_tracking(source)["rows"]
        first = rows[0]
        self.assertEqual(first["tracking_status"], "inconsistent")
        missing = rows[1]
        self.assertIsNone(missing["saved_category_calculation"])
        self.assertIsNone(missing["saved_shear_dcr"])
        self.assertEqual(missing["saved_shear_status"], "not_evaluated")
        self.assertEqual(missing["tracking_status"], "unresolved")

    def test_report_does_not_mutate_or_alias_design(self):
        source = record()
        before = copy.deepcopy(source)
        report = build_joint_design_tracking(source)
        report["rows"][0]["saved_category_calculation"]["aj_in2"] = -1.
        self.assertEqual(source, before)


if __name__ == "__main__":
    unittest.main()
