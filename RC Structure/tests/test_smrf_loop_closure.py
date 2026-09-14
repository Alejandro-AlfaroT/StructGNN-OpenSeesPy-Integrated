"""The section search closes on what qualification evaluates.

Covers the planner that moves the ladder rungs, the depth step for beams,
checks re-run on captured actions, and the strong-column steel selection.
"""
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Design import Design_Driver as driver
from Design.ACI_Checks import run_checks_phase1
from Design.Config import DesignConfig
from Design.Section_Design import beam_ladder, column_ladder
from RC_Design_Check import get_element_tags


def flags(**overrides):
    base = {"drift_ok": True, "scwb_ok": True, "joint_scwb_failed": False, "capacity_accepted": True,
            "beam_section_adequate": True, "beam_hoops_selected": True, "column_section_adequate": True,
            "column_hoops_selected": True, "joints_all_pass": True, "anchorage_all_pass": True}
    base.update(overrides)
    return base


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.columns = column_ladder()
        self.beams = beam_ladder(span_in=240., story_height_in=144.)

    def test_beam_ladder_is_ordered_by_capacity_and_offers_a_wider_variant(self):
        from Design.Section_Design import flexural_capacity_proxy, beam_width_for_depth, BEAM_WIDTH_STEP_IN
        proxies = [flexural_capacity_proxy(r) for r in self.beams]
        self.assertEqual(proxies, sorted(proxies))
        depths = {r[1] for r in self.beams}
        for h in depths:
            widths = sorted({r[0] for r in self.beams if r[1] == h})
            self.assertEqual(widths, [beam_width_for_depth(h), beam_width_for_depth(h) + BEAM_WIDTH_STEP_IN])

    def test_next_deeper_beam_skips_concrete_strength_and_width_rungs(self):
        index = next(i for i, r in enumerate(self.beams) if r[:3] == (10., 18., 5.))
        deeper = driver._next_deeper_beam_index(self.beams, index)
        self.assertGreater(self.beams[deeper][1], 18.)
        self.assertTrue(all(r[1] <= 18. for r in self.beams[index:deeper]))
        self.assertIsNone(driver._next_deeper_beam_index(self.beams, len(self.beams) - 1))

    def test_next_wider_beam_keeps_the_depth(self):
        index = next(i for i, r in enumerate(self.beams) if r[:3] == (10., 18., 5.))
        wider = driver._next_wider_beam_index(self.beams, index)
        self.assertEqual(self.beams[wider][1], 18.)
        self.assertEqual(self.beams[wider][0], 14.)
        widest = max(i for i, r in enumerate(self.beams) if r[1] == 18.)
        self.assertIsNone(driver._next_wider_beam_index(self.beams, widest))

    def test_drift_failure_takes_the_next_beam_depth_and_holds_the_column(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (30., 30., 4.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 18., 6.))
        worst = {"beam": 0.87, "column": 0.75}       # in band; column well under its ceiling
        next_column, next_beam, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0, flags=flags(drift_ok=False))
        self.assertIn("drift", reasons)
        self.assertGreater(self.beams[next_beam][1], 18.)
        self.assertGreaterEqual(next_column, ci)      # never steps down while a requirement is unmet

    def test_compatible_beam_index_prefers_the_next_feasible_rung_above(self):
        sp_saved = {k: getattr(sp, k) for k in ("BAY_X", "BAY_Y", "STORY_H", "B_COL", "H_COL", "SLAB_THICKNESS_IN",
                                                 "BEAM_BAR_SIZE", "BEAM_STIRRUP_BAR_SIZE")}
        try:
            sp.BAY_X = sp.BAY_Y = 120.; sp.STORY_H = 168.; sp.B_COL = sp.H_COL = 36.; sp.SLAB_THICKNESS_IN = None
            beams = beam_ladder(span_in=120., story_height_in=168.)
            target = next(i for i, r in enumerate(beams) if r[:2] == (12., 24.))   # ln = 84 < 4d: forbidden
            chosen = driver._compatible_beam_index(beams, target)
            self.assertGreaterEqual(chosen, target)
            self.assertLess(beams[chosen][1], 24.)                # a wider, shallower rung above it
            self.assertGreaterEqual(beams[chosen][0], 12.)
        finally:
            for k, v in sp_saved.items():
                setattr(sp, k, v)

    def test_beam_capacity_shear_takes_the_wider_variant_first(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (30., 30., 6.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 20., 5.))
        worst = {"beam": 0.86, "column": 0.75}
        _c, next_beam, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0,
            flags=flags(beam_section_adequate=False))
        self.assertIn("beam_capacity_shear", reasons)
        # width first: the shear section answers to bw d and short spans cap d
        self.assertEqual(self.beams[next_beam][:2], (14., 20.))

    def test_joint_scwb_failure_takes_the_next_column_size(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (26., 26., 8.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 16., 8.))
        worst = {"beam": 0.83, "column": 0.87}
        next_column, next_beam, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=ci,
            flags=flags(scwb_ok=False, joint_scwb_failed=True))
        self.assertIn("joint_scwb", reasons)
        self.assertEqual(self.columns[next_column][0], 28.)
        self.assertGreaterEqual(next_beam, bi)        # strength may still move the beam; never down

    def test_step_down_allowed_only_when_every_requirement_is_met(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (36., 36., 4.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 18., 6.))
        worst = {"beam": 0.85, "column": 0.6}
        met, _b, reasons = driver._plan_next_rungs(self.columns, self.beams, ci, bi, worst, 0.75, 1.0,
                                                   scwb_index=0, flags=flags())
        self.assertEqual(reasons, [])
        self.assertLess(met, ci)                      # strength alone may shrink an oversize column
        held, _b, _r = driver._plan_next_rungs(self.columns, self.beams, ci, bi, worst, 0.75, 1.0,
                                               scwb_index=0, flags=flags(drift_ok=False))
        self.assertEqual(held, ci)                    # top of the ladder: drift grows the beam instead

    def test_joint_shear_jumps_to_the_rung_whose_joint_area_covers_the_shortfall(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (24., 24., 4.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 20., 4.))
        worst = {"beam": 0.89, "column": 0.5}
        next_column, _b, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0,
            flags=flags(joints_all_pass=False, joint_shear_ratio=1.6))
        self.assertIn("joint_shear_or_anchorage", reasons)
        b, h, fc = self.columns[next_column]
        self.assertGreaterEqual(b * h * fc ** .5, 24. * 24. * 2. * 1.6 - 1e-9)
        # and not further than the first rung that covers it
        b0, h0, fc0 = self.columns[next_column - 1]
        self.assertLess(b0 * h0 * fc0 ** .5, 24. * 24. * 2. * 1.6)

    def test_top_of_beam_ladder_falls_back_to_the_column(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (30., 30., 4.))
        bi = len(self.beams) - 1
        worst = {"beam": 0.5, "column": 0.5}
        next_column, next_beam, _r = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0, flags=flags(drift_ok=False))
        self.assertEqual(next_beam, bi)
        self.assertGreater(next_column, ci)


class CapturedActionCheckTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 2, 1, 2
        sp.NUM_MODES = sp.NUM_FLOOR + 2
        sp.SLAB_THICKNESS_IN = None
        sp.FLOOR_TRANSFER = None
        sp.SLAB_REINFORCEMENT = None
        sp.SLAB_ACTIONS = None

    def tearDown(self):
        driver.ops.wipe()
        self.stack.close()

    def test_checks_on_captured_actions_match_the_live_domain(self):
        cfg = DesignConfig.from_structure_parameters()
        driver._sync_cfg_to_sp(cfg)
        driver._analyze_combination({"dead": 1.2, "live": 1.6, "ex": 1.0, "ey": 0.3, "live_pattern": "all"}, 0.5)
        cols, bx, by = get_element_tags()
        live = run_checks_phase1(cols, bx + by, cfg)
        captured = driver._capture_element_actions(1.2)
        driver.ops.wipe()                              # the live domain is gone
        offline = run_checks_phase1(cols, bx + by, cfg, member_actions=captured)
        self.assertEqual(set(live), set(offline))
        for tag, result in live.items():
            for name, state in result.limit_states.items():
                self.assertAlmostEqual(state.dcr, offline[tag].limit_states[name].dcr, places=12, msg=(tag, name))
            self.assertEqual(result.section_info.get("Pu_kip"), offline[tag].section_info.get("Pu_kip"))

    def test_missing_captured_member_is_an_error_not_a_zero(self):
        cfg = DesignConfig.from_structure_parameters()
        cols, bx, by = get_element_tags()
        with self.assertRaises(KeyError):
            run_checks_phase1(cols, bx + by, cfg, member_actions={})


class StrongColumnSteelTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR"}:
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.B_COL = sp.H_COL = 24.
        sp.FC_COL_KSI = 5.
        sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS = 8, 3, 3, 2
        sp.COL_BAR_AREA = sp.rebar_area(8)
        sp.SLAB_THICKNESS_IN = 6.
        sp.FLOOR_TRANSFER = None
        sp.SLAB_REINFORCEMENT = None

    def tearDown(self):
        self.stack.close()

    def _joint_state(self, factor):
        """A failed check whose required sum is ``factor`` times the current column Mn."""
        from Design.SMRF_Joint_Adapter import record_section_capacity
        state = driver._state_record_core()
        p = 40.
        mn = min(record_section_capacity(state, "column", "x", face, p)["mn_kip_in"] for face in ("positive", "negative"))
        demand = factor * mn
        sway = {"nominal_strengths": True,
                "column_capacities": [{"tag": 1, "end": "j", "factored_axial_kip": p, "axial_min_kip": p * .5,
                                       "axial_max_kip": p * 1.5, "mn_kip_in": mn, "axial_envelope_checked": True}],
                "beam_capacities": []}
        check = {"id": "scwb", "status": "fail", "demand": demand, "capacity": mn,
                 "details": {"ratio_provided": mn / demand * 1.2, "sum_mnc_kip_in": mn}}
        return {"_failing": [(check, sway, "x")], "_state": state, "all_pass": False}, mn

    def test_least_cage_meeting_the_joint_rule_is_selected(self):
        cfg = DesignConfig()
        joint, current = self._joint_state(factor=1.35)
        update, exhausted = driver._scwb_column_steel(cfg, joint)
        self.assertFalse(exhausted)
        self.assertIsNotNone(update)
        from Design.SMRF_Joint_Adapter import record_section_capacity
        state = driver._state_record_core()
        state["reinforcement"].update(col_bar_size=update["bar_size"], col_top_bars=update["n_top"],
                                      col_bot_bars=update["n_bot"], col_side_bars=update["n_side"],
                                      col_bar_area_in2=sp.rebar_area(update["bar_size"]),
                                      col_bar_diameter_in=sp.rebar_diameter(update["bar_size"]),
                                      col_longitudinal_centroid_offset_in=sp.longitudinal_cover_in("column", update["bar_size"]))
        provided = min(record_section_capacity(state, "column", "x", face, p)["mn_kip_in"]
                       for p in (20., 40., 60.) for face in ("positive", "negative"))
        self.assertGreaterEqual(provided, current * 1.35)
        ast = (2 * update["n_top"] + 2 * update["n_side"]) * sp.rebar_area(update["bar_size"])
        self.assertLessEqual(ast / (sp.B_COL * sp.H_COL), cfg.rebar.rho_col_practical_max + 1e-12)
        # Least steel: no lighter admissible cage satisfies the rule.
        from Redesign import _col_candidates
        ag = sp.B_COL * sp.H_COL
        for candidate in _col_candidates(cfg.rebar.rho_col_min * ag, ast - 1e-9, cfg):
            trial = driver._state_record_core()
            trial["reinforcement"].update(col_bar_size=candidate[0], col_top_bars=candidate[1], col_bot_bars=candidate[2],
                                          col_side_bars=candidate[3], col_bar_area_in2=sp.rebar_area(candidate[0]),
                                          col_bar_diameter_in=sp.rebar_diameter(candidate[0]),
                                          col_longitudinal_centroid_offset_in=sp.longitudinal_cover_in("column", candidate[0]))
            lighter = min(record_section_capacity(trial, "column", "x", face, p)["mn_kip_in"]
                          for p in (20., 40., 60.) for face in ("positive", "negative"))
            self.assertLess(lighter, current * 1.35, candidate)

    def test_unreachable_requirement_is_reported_exhausted_not_faked(self):
        cfg = DesignConfig()
        joint, _current = self._joint_state(factor=50.)
        update, exhausted = driver._scwb_column_steel(cfg, joint)
        self.assertTrue(exhausted)
        self.assertIsNotNone(update)                  # the closest cage is installed so the shortfall is measured

    def test_no_failure_means_no_change(self):
        self.assertEqual(driver._scwb_column_steel(DesignConfig(), {"_failing": [], "_state": {}}), (None, False))


if __name__ == "__main__":
    unittest.main()
