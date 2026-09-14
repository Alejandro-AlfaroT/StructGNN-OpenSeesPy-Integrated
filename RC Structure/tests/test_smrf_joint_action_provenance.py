import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Demands import strength_load_combinations
from Design.SMRF_Design_Evidence import analysis_input_signature, joint_evidence


def record():
    # Only the wrapper's signed record inputs are needed here. The physical
    # inventory/section adapter has its own independent detailed tests.
    value = {"geometry": {}, "sections": {}, "reinforcement": {}, "materials": {},
             "floor_loads": {}, "seismic": {"sds": .75}, "demand": {}}
    value["design_actions"] = {
        "analysis_input_sha256": analysis_input_signature(value),
        "combinations": [dict(case, analysis_succeeded=True,
                              axial_reference="joint_faces", members={})
                         for case in strength_load_combinations(.75)],
    }
    return value


class JointActionProvenanceTests(unittest.TestCase):
    def test_canonical_factors_survive_json_roundtrip_and_reach_adapter(self):
        value = json.loads(json.dumps(record()))
        with mock.patch("Design.SMRF_Design_Evidence.build_joint_evidence", return_value={"checked": True}) as build:
            self.assertEqual(joint_evidence(value), {"checked": True})
        self.assertEqual(build.call_args.kwargs["expected_combination_ids"],
                         [case["id"] for case in strength_load_combinations(.75)])
        self.assertEqual(build.call_args.args[1], value["design_actions"]["combinations"])

    def test_same_case_id_with_changed_coefficient_is_rejected(self):
        for key in ("dead", "live", "ex", "ey"):
            value = record()
            value["design_actions"]["combinations"][2][key] += .05
            # This is the gap the descriptor validation closes: the frame
            # inputs are unchanged, so their signature legitimately matches.
            self.assertEqual(analysis_input_signature(value), value["design_actions"]["analysis_input_sha256"])
            with self.subTest(key=key), mock.patch("Design.SMRF_Design_Evidence.build_joint_evidence") as build:
                with self.assertRaisesRegex(ValueError, key + " coefficient"):
                    joint_evidence(value)
                build.assert_not_called()

    def test_missing_nonnumeric_nonfinite_and_boolean_factors_are_rejected(self):
        for key in ("dead", "live", "ex", "ey"):
            for bad in (None, "1.4", True, float("nan"), float("inf"), -float("inf")):
                value = record()
                value["design_actions"]["combinations"][0][key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, key + " coefficient"):
                    joint_evidence(value)
            value = record()
            value["design_actions"]["combinations"][0].pop(key)
            with self.subTest(key=key, absent=True), self.assertRaisesRegex(ValueError, key + " coefficient"):
                joint_evidence(value)

    def test_missing_wrong_family_and_unknown_id_are_rejected(self):
        for field, bad in (("family", None), ("family", "wrong"), ("id", None), ("id", "unknown")):
            value = record()
            value["design_actions"]["combinations"][0][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                joint_evidence(value)

    def test_altered_sds_with_resigned_frame_cannot_reuse_old_factors(self):
        value = record()
        value["seismic"]["sds"] += .1
        value["design_actions"]["analysis_input_sha256"] = analysis_input_signature(value)
        with self.assertRaisesRegex(ValueError, "dead coefficient"):
            joint_evidence(value)

    def test_stale_frame_signature_is_still_rejected_first(self):
        value = record()
        value["sections"]["b_col_in"] = 30
        with self.assertRaisesRegex(ValueError, "do not match"):
            joint_evidence(value)

    def test_changed_transfer_or_demand_basis_invalidates_solved_actions(self):
        for key, changed in (("floor_transfer", {"unit_cases": {"dead": "changed"}}),
                             ("gravity_load_model", "slab_transfer"),
                             ("demand_basis", {"patterns_in_strength_envelope": True})):
            value = record()
            value[key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "do not match"):
                joint_evidence(value)

    def test_changed_live_pattern_with_unchanged_factors_is_rejected(self):
        value = record()
        value["design_actions"]["combinations"][0]["live_pattern"] = "wrong"
        with self.assertRaisesRegex(ValueError, "live_pattern"):
            joint_evidence(value)

    def test_missing_combination_inventory_still_reaches_adapter_for_incomplete_check(self):
        value = record()
        value["design_actions"]["combinations"].pop()
        with mock.patch("Design.SMRF_Design_Evidence.build_joint_evidence", return_value={}) as build:
            joint_evidence(value)
        self.assertEqual(len(build.call_args.args[1]), 17)
        self.assertEqual(len(build.call_args.kwargs["expected_combination_ids"]), 18)


if __name__ == "__main__":
    unittest.main()
