"""
Design/Report.py
Terminal reporting for Phase 1 (elastic) design check results.

Accepts the typed MemberCheckResult objects from Design/ACI_Checks.py.
Produces the same table layout as RC_Design_Check.print_summary() but with
per-limit-state columns, a min-controlled flag, and a penalty-reason column.

Entry point
-----------
    from Design.Report import print_phase1_summary

    col_fail, beam_fail = print_phase1_summary(results, cfg, verbose=False)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import Structure_Parameters as sp
from Design.CheckResult import MemberCheckResult
from Design.Config import DesignConfig


def print_phase1_summary(
    results: Dict[int, MemberCheckResult],
    cfg: DesignConfig,
    verbose: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Print the Phase 1 design check summary table to stdout.

    Parameters
    ----------
    results  : dict[int, MemberCheckResult] from run_checks_phase1()
    cfg      : DesignConfig (used for section header info)
    verbose  : if True, print every element; if False, print only failures
               NOTE: currently all rows are printed regardless of this flag
               to mirror the behaviour requested for print_summary().

    Returns
    -------
    col_fail, beam_fail : lists of failing element tags
    """
    col_fail: List[int]  = []
    beam_fail: List[int] = []

    col_dcr_PM_max = col_dcr_V_max = 0.0
    beam_dcr_F_max = beam_dcr_V_max = 0.0
    n_min_ctrl = 0

    print("\n" + "=" * 76)
    print("  ACI 318-19 Phase 1 Design Check")
    print(
        f"  Column : {sp.B_COL}\" x {sp.H_COL}\"  "
        f"fc={sp.FC_COL_KSI} ksi  "
        f"({sp.COL_TOP_BARS}T+{sp.COL_BOT_BARS}B+{sp.COL_SIDE_BARS}S) "
        f"#{sp.COL_BAR_SIZE}"
    )
    print(
        f"  Beam   : {sp.B_BEAM}\" x {sp.H_BEAM}\"  "
        f"fc={sp.FC_BEAM_KSI} ksi  "
        f"{sp.BEAM_TOP_BARS}T+{sp.BEAM_BOT_BARS}B #{sp.BEAM_BAR_SIZE}"
    )
    print(
        f"  DCR target band: [{cfg.dcr.dcr_band_lo:.2f}, {cfg.dcr.dcr_band_hi:.2f}]  "
        f"target: {cfg.dcr.dcr_target:.2f}  "
        f"objective: {cfg.dcr.objective}"
    )
    print("=" * 76)

    # ── Columns ──────────────────────────────────────────────────────────────
    print("\n  COLUMNS")
    print(
        f"  {'Tag':>5}  {'Pu(k)':>7}  {'Mu(k·in)':>9}  "
        f"{'φMn(k·in)':>11}  {'DCR P-M':>7}  {'Vu(k)':>6}  "
        f"{'DCR V':>6}  {'Min?':>4}  Status"
    )
    print("  " + "-" * 79)

    for tag, r in results.items():
        if r.member_type != "column":
            continue

        ls_pm    = r.limit_states.get("PM")
        ls_shear = r.limit_states.get("shear")

        if ls_pm is None or ls_shear is None:
            continue

        col_dcr_PM_max = max(col_dcr_PM_max, ls_pm.dcr)
        col_dcr_V_max  = max(col_dcr_V_max,  ls_shear.dcr)
        if not r.ok:
            col_fail.append(tag)
        if r.any_min_controlled:
            n_min_ctrl += 1

        flag = "Y" if r.any_min_controlled else " "
        print(
            f"  {tag:>5}  {ls_pm.demand:>7.1f}  {ls_pm.demand:>9.1f}  "
            f"{ls_pm.capacity:>11.1f}  {ls_pm.dcr:>7.3f}  "
            f"{ls_shear.demand:>6.2f}  {ls_shear.dcr:>6.3f}  "
            f"{flag:>4}  {'OK' if r.ok else 'FAIL'}"
        )
        if r.penalty_reasons:
            for reason in r.penalty_reasons:
                print(f"         [penalty] {reason}")

    print(
        f"\n  Column P-M  max DCR: {col_dcr_PM_max:.3f}  "
        f"({'OK' if col_dcr_PM_max <= 1.0 else 'FAIL'})"
    )
    print(
        f"  Column shear max DCR: {col_dcr_V_max:.3f}  "
        f"({'OK' if col_dcr_V_max <= 1.0 else 'FAIL'})"
    )

    # ── Beams ─────────────────────────────────────────────────────────────────
    print("\n  BEAMS")
    print(
        f"  {'Tag':>5}  {'Mu(k·in)':>9}  {'φMn(k·in)':>10}  "
        f"{'DCR F':>6}  {'Vu(k)':>6}  {'DCR V':>6}  {'Min?':>4}  Status"
    )
    print("  " + "-" * 68)

    for tag, r in results.items():
        if r.member_type != "beam":
            continue

        ls_pos   = r.limit_states.get("flexure_pos")
        ls_neg   = r.limit_states.get("flexure_neg")
        ls_shear = r.limit_states.get("shear")

        if ls_pos is None or ls_neg is None or ls_shear is None:
            continue

        # Governing flexural direction
        if ls_pos.dcr >= ls_neg.dcr:
            ls_flex = ls_pos
        else:
            ls_flex = ls_neg

        beam_dcr_F_max = max(beam_dcr_F_max, ls_pos.dcr, ls_neg.dcr)
        beam_dcr_V_max = max(beam_dcr_V_max, ls_shear.dcr)
        if not r.ok:
            beam_fail.append(tag)
        if r.any_min_controlled:
            n_min_ctrl += 1

        flag = "Y" if r.any_min_controlled else " "
        print(
            f"  {tag:>5}  {ls_flex.demand:>9.1f}  {ls_flex.capacity:>10.1f}  "
            f"{ls_flex.dcr:>6.3f}  {ls_shear.demand:>6.2f}  "
            f"{ls_shear.dcr:>6.3f}  {flag:>4}  {'OK' if r.ok else 'FAIL'}"
        )
        if r.penalty_reasons:
            for reason in r.penalty_reasons:
                print(f"         [penalty] {reason}")

    print(
        f"\n  Beam flexure max DCR: {beam_dcr_F_max:.3f}  "
        f"({'OK' if beam_dcr_F_max <= 1.0 else 'FAIL'})"
    )
    print(
        f"  Beam shear   max DCR: {beam_dcr_V_max:.3f}  "
        f"({'OK' if beam_dcr_V_max <= 1.0 else 'FAIL'})"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_fail = len(col_fail) + len(beam_fail)
    print("\n" + "=" * 76)
    if n_fail == 0:
        print("  OVERALL: ALL ELEMENTS PASS")
    else:
        print(f"  OVERALL: {n_fail} ELEMENT(S) FAIL")
        if col_fail:
            print(f"  Failed columns : {col_fail}")
        if beam_fail:
            print(f"  Failed beams   : {beam_fail}")

    if n_min_ctrl:
        print(
            f"  Min-controlled : {n_min_ctrl} element(s) sized by code minimum  "
            "(flagged, not rejected)"
        )

    print("=" * 76 + "\n")

    return col_fail, beam_fail
