"""Demand basis: SDC, ELF eligibility, live-load arrangements, torsion, declarations."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Structure_Parameters as sp  # noqa: E402
from Design.SMRF_Demands import (elf_eligibility, evaluate_demand_basis, live_load_patterns,  # noqa: E402
                                 seismic_design_category, strength_load_combinations)
from Design.SMRF_Qualification import _apply_verification_assertions  # noqa: E402
from Design.SMRF_Common import not_evaluated  # noqa: E402
from Loads.Seismic_ELF import accidental_torsion_moment  # noqa: E402


class SeismicDesignCategoryTests(unittest.TestCase):
    def test_tables_11_6_1_and_11_6_2(self):
        self.assertEqual(seismic_design_category(0.10, 0.05, 0.05), "A")
        self.assertEqual(seismic_design_category(0.20, 0.05, 0.05), "B")
        self.assertEqual(seismic_design_category(0.20, 0.15, 0.10), "C")    # SD1 governs
        self.assertEqual(seismic_design_category(0.45, 0.18, 0.18), "C")
        self.assertEqual(seismic_design_category(0.50, 0.25, 0.25), "D")    # threshold is inclusive
        self.assertEqual(seismic_design_category(1.00, 0.60, 0.60), "D")
        self.assertEqual(seismic_design_category(1.25, 0.75, 0.75), "E")    # S1 >= 0.75, RC II
        self.assertEqual(seismic_design_category(1.25, 0.75, 0.75, "IV"), "F")
        self.assertEqual(seismic_design_category(0.20, 0.05, 0.05, "IV"), "C")
        with self.assertRaises(ValueError):
            seismic_design_category(1.0, 0.6, 0.6, "V")

    def test_elf_eligibility_table_12_6_1(self):
        self.assertTrue(elf_eligibility("D", 126.0, True, 1.0, 0.6)[0])
        self.assertFalse(elf_eligibility("D", 126.0, False, 1.0, 0.6)[0])       # any irregularity
        self.assertTrue(elf_eligibility("D", 200.0, True, 1.5, 0.6)[0])         # T < 3.5 Ts
        self.assertFalse(elf_eligibility("E", 200.0, True, 3.0, 0.6)[0])
        self.assertTrue(elf_eligibility("C", 300.0, False, 5.0, 0.6)[0])


class LiveLoadPatternTests(unittest.TestCase):
    def test_arrangements_cover_alternate_and_adjacent_spans(self):
        patterns = {p["id"]: p for p in live_load_patterns(3, 2)}
        self.assertEqual(set(patterns), {"alternate_x_even", "alternate_x_odd", "alternate_y_even", "alternate_y_odd",
                                         "adjacent_x_support_1", "adjacent_x_support_2", "adjacent_y_support_1"})
        self.assertEqual(sorted(patterns["alternate_x_even"]["panels"]), [[0, 0], [0, 1], [2, 0], [2, 1]])
        self.assertEqual(sorted(patterns["adjacent_x_support_2"]["panels"]), [[1, 0], [1, 1], [2, 0], [2, 1]])
        # A single bay has no odd spans and no interior supports.
        self.assertEqual([p["id"] for p in live_load_patterns(1, 1)], ["alternate_x_even", "alternate_y_even"])

    def test_pattern_combinations_extend_the_canonical_rule(self):
        base = strength_load_combinations(1.0)
        self.assertEqual(len(base), 18)
        self.assertTrue(all(c["live_pattern"] == "all" for c in base))
        patterned = strength_load_combinations(1.0, live_patterns=live_load_patterns(3, 3))
        extra = [c for c in patterned if c["family"] == "gravity_pattern"]
        self.assertEqual(len(extra), 8)
        self.assertTrue(all(c["dead"] == 1.2 and c["live"] == 1.6 and c["ex"] == c["ey"] == 0.0 for c in extra))
        self.assertEqual(len({c["id"] for c in patterned}), 26)


class TorsionTests(unittest.TestCase):
    def test_accidental_torsion_moment_uses_the_perpendicular_plan_dimension(self):
        with mock.patch.object(sp, "NUM_BAY_X", 3), mock.patch.object(sp, "NUM_BAY_Y", 2), \
             mock.patch.object(sp, "BAY_X", 240.0), mock.patch.object(sp, "BAY_Y", 200.0):
            self.assertAlmostEqual(accidental_torsion_moment("x", 10.0), 0.05 * 400.0 * 10.0)
            self.assertAlmostEqual(accidental_torsion_moment("y", 10.0), 0.05 * 720.0 * 10.0)
            self.assertAlmostEqual(accidental_torsion_moment("x", 10.0, 0.05, 2.0), 2.0 * 0.05 * 400.0 * 10.0)


def _record(declared=True, torsion_ratio=1.067, patterns=True):
    policy = {"risk_category": "II", "site_class": "C", "occupancy": "office", "partition_allowance_ksf": 0.010,
              "storage_live_fraction_in_weight": 0.0, "roof_live_load_ksf": 0.020, "snow_load_ksf": 0.0,
              "wind_governs": False, "rain_ponding_excluded": True, "accidental_torsion_ratio": 0.05,
              "live_load_patterning": True, "declared_by": "test" if declared else "",
              "declaration_date": "2026-09-13", "declaration_basis": "fixture" if declared else ""}
    return {
        "seismic": {"sds": 1.0, "sd1": 0.6, "s1": 0.6, "site_label": "sdc_d_high"},
        "geometry": {"num_bay_x": 3, "num_bay_y": 3, "num_floor": 8, "story_h_in": 120.0},
        "floor_loads": {"floor_live_load_ksf": 0.05, "floor_superimposed_dead_load_ksf": 0.05,
                        "seismic_live_load_fraction": 0.0, "total_floor_seismic_weight_kip": 200.0},
        "demand_basis": {"policy": policy,
                         "torsion": {"ratio": 0.05, "amplification": 1.0, "max_drift_ratio": torsion_ratio,
                                     "torsional_irregularity": "none" if torsion_ratio <= 1.2 else "1a"},
                         "regularity": {"regular": True}, "design_period_sec": 1.0, "ts_sec": 0.6,
                         "period_basis": {"capped_at_cu_ta": False},
                         "live_load_patterns": live_load_patterns(3, 3) if patterns else [],
                         "patterns_in_strength_envelope": patterns,
                         "drift": {"cd": 5.5, "rho_for_drift_load": 1.0, "beam_stiffness_modifier": 0.35,
                                   "column_stiffness_modifier": 0.70, "second_order_included": True}},
    }


class DemandBasisEvaluationTests(unittest.TestCase):
    def test_declared_record_evaluates_every_item(self):
        statuses = {c["id"]: c for c in evaluate_demand_basis(_record())}
        for key in ("site_hazard", "elf_eligibility", "accidental_torsion", "load_scope",
                    "live_load_patterning", "effective_seismic_weight", "drift_analysis_basis"):
            self.assertEqual(statuses[f"demands.{key}"]["status"], "pass", key)
        self.assertEqual(statuses["demands.site_hazard"]["details"]["derived_sdc"], "D")

    def test_undeclared_policy_keeps_declarations_open_but_computed_items_evaluated(self):
        statuses = {c["id"]: c["status"] for c in evaluate_demand_basis(_record(declared=False))}
        for key in ("site_hazard", "elf_eligibility", "load_scope", "effective_seismic_weight"):
            self.assertEqual(statuses[f"demands.{key}"], "not_evaluated", key)
        for key in ("accidental_torsion", "live_load_patterning", "drift_analysis_basis"):
            self.assertEqual(statuses[f"demands.{key}"], "pass", key)

    def test_mislabelled_site_and_torsional_irregularity_fail_or_block_elf(self):
        record = _record()
        record["seismic"] = {"sds": 0.50, "sd1": 0.25, "s1": 0.25, "site_label": "sdc_c"}
        statuses = {c["id"]: c for c in evaluate_demand_basis(record)}
        self.assertEqual(statuses["demands.site_hazard"]["status"], "fail")
        self.assertEqual(statuses["demands.site_hazard"]["details"]["derived_sdc"], "D")
        irregular = evaluate_demand_basis(_record(torsion_ratio=1.3))
        statuses = {c["id"]: c["status"] for c in irregular}
        self.assertEqual(statuses["demands.elf_eligibility"], "fail")      # Type 1a bars ELF in SDC D
        self.assertEqual(statuses["demands.accidental_torsion"], "pass")   # 1.3 <= 1.4, Ax applies

    def test_empty_record_is_all_not_evaluated(self):
        self.assertTrue(all(c["status"] == "not_evaluated" for c in evaluate_demand_basis({})))


class VerificationAssertionTests(unittest.TestCase):
    def test_assertions_need_an_author_and_a_basis(self):
        checks = [not_evaluated("floor.independent_hand_verification", "review", "open"),
                  not_evaluated("slab_fire_resistance", "4.11", "open")]
        unsigned = {"demand_basis": {"verification": {"floor_hand_check_verified": True, "asserted_by": "", "assertion_basis": ""}}}
        self.assertTrue(all(c["status"] == "not_evaluated" for c in _apply_verification_assertions(unsigned, checks)))
        signed = {"demand_basis": {"verification": {"floor_hand_check_verified": True, "fire_resistance_scope_accepted": False,
                                                    "asserted_by": "engineer", "assertion_date": "2026-09-13",
                                                    "assertion_basis": "hand check of case 7"}}}
        result = {c["id"]: c for c in _apply_verification_assertions(signed, checks)}
        self.assertEqual(result["floor.independent_hand_verification"]["status"], "pass")
        self.assertEqual(result["floor.independent_hand_verification"]["details"]["asserted_by"], "engineer")
        self.assertEqual(result["slab_fire_resistance"]["status"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
