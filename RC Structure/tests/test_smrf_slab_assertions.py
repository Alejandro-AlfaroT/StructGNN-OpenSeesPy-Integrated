"""Assertion provenance is bookkeeping, never engineering certification."""
import copy
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.Config import SlabActionAssertions
from Design.SMRF_Slab_Actions import build_slab_action_evidence, evaluate_slab_actions
from Design.SMRF_Common import assertion_provenance_valid


def signed():
    return SlabActionAssertions(
        analysis_applicability_verified=True, all_floors_enveloped=True,
        load_pattern_envelope_verified=True, spatial_envelope_per_unit_width=True,
        twisting_moment_resolution_verified=True, zero_membrane_force_verified=True,
        verified=True, asserted_by="test fixture only", assertion_date="2026-09-13",
        assertion_basis="Unit test, not engineering verification")


class SlabAssertionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slab = {"thickness_in": 5., "concrete_fc_ksi": 4., "concrete_unit_weight_kcf": .15,
                    "superimposed_dead_load_ksf": .05}
        cls.geometry = {"num_bay_x": 1, "num_bay_y": 1, "bay_x_in": 200., "bay_y_in": 200.}
        cls.sections = {"b_beam_in": 12., "h_beam_in": 18., "fc_beam_ksi": 4., "b_col_in": 18., "h_col_in": 18.}
        cls.inputs = {"thickness_in": 5., "fc_ksi": 4., "fy_ksi": 60., "max_aggregate_size_in": .75,
                      "exposure": "sheltered_interior", "steel_specification": "ASTM A706",
                      "concrete_type": "normalweight", "num_floor": 1, "panel_ids": ["panel_x1_y1"]}

    def evidence(self, policy):
        return build_slab_action_evidence(self.slab, self.geometry, self.sections, .05, self.inputs,
                                          mesh_per_bay=4, assertions=asdict(policy))

    def test_named_dated_literal_assertions_are_required(self):
        self.assertTrue(signed().all_asserted())
        for key, value in (("verified", "false"), ("verified", 1), ("asserted_by", " "),
                           ("assertion_basis", ""), ("assertion_date", "bad-date"),
                           ("assertion_date", "2026-02-30")):
            policy = signed()
            setattr(policy, key, value)
            with self.subTest(key=key, value=value):
                self.assertFalse(policy.all_asserted())
                evidence = self.evidence(policy)
                self.assertFalse(evidence["verified"])
                self.assertEqual(evaluate_slab_actions(evidence)[0]["status"], "not_evaluated")

    def test_signed_evidence_retains_provenance_but_does_not_clear_independent_review(self):
        evidence = self.evidence(signed())
        self.assertTrue(evidence["assertion_provenance_valid"])
        self.assertTrue(evidence["engineering_assertions"]["verified"])
        # This single-panel fixture has a governing face shear in bottom
        # tension. A signature cannot turn an unresolved top-row shear basis
        # into valid design evidence.
        self.assertFalse(evidence["shear_recovery"]["top_rows_recovered_at_face"])
        self.assertFalse(evidence["verified"])
        checks = {c["id"]: c for c in evaluate_slab_actions(evidence)}
        self.assertEqual(checks["floor.qualified_slab_actions"]["status"], "not_evaluated")
        self.assertEqual(checks["floor.independent_hand_verification"]["status"], "not_evaluated")

    def test_cached_true_flags_cannot_replace_missing_assertions(self):
        evidence = self.evidence(signed())
        for value in ({}, {"asserted_by": " ", "assertion_date": "2026-09-13", "assertion_basis": "test"}):
            changed = copy.deepcopy(evidence)
            changed["engineering_assertions"] = value
            self.assertEqual(evaluate_slab_actions(changed)[0]["status"], "not_evaluated")
        self.assertFalse(assertion_provenance_valid(None))


if __name__ == "__main__":
    unittest.main()
