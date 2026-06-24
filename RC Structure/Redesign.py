"""
Redesign.py
Iterative ACI 318-19 steel redesign for the RC frame.

Strategy
--------
Inner loop (cheap):  build → gravity only → design check → resize steel → repeat
Outer  (once):       modal + pushover / NTHA after steel has converged

The redesign adjusts for both under-design (DCR > DCR_MAX) and over-design
(DCR < DCR_MIN) by finding the smallest discrete (bar_size, bar_count)
combination whose capacity falls in the target band [DCR_MIN, DCR_MAX].

All target DCR values are read from Structure_Parameters (DESIGN_DCR_MIN/MAX).
"""

import contextlib
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Gravity import run_gravity_analysis
from Design.ACI_Checks import run_checks_phase1, to_legacy_dict
from Design.Config import DesignConfig
from Loads.Gravity_Loads import apply_gravity_loads
from Model.Build_Model import build_model
from RC_Design_Check import get_element_tags, run_checks


@contextlib.contextmanager
def _silence_c_output():
    """Suppress C-level stdout (fd 1) and stderr (fd 2) for the duration.

    OpenSeesPy prints material-creation banners (e.g. the IMK attribution
    line) directly from C code to stderr, bypassing sys.stdout/sys.stderr.
    Redirecting both file descriptors to devnull is the only reliable way
    to silence them on Windows.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (legacy fallbacks — prefer DesignConfig)
# ─────────────────────────────────────────────────────────────────────────────

PHI_FLEX = 0.90
ECU      = 0.003


# ─────────────────────────────────────────────────────────────────────────────
# Beam: required steel from moment demand (ACI §22.3)
# ─────────────────────────────────────────────────────────────────────────────

def _beam_As_min(fc_ksi, fy_ksi, b, d):
    """ACI 318-19 §9.6.1.2 minimum beam steel."""
    fc_psi = fc_ksi * 1000.0
    fy_psi = fy_ksi * 1000.0
    return max(3.0 * math.sqrt(fc_psi) / fy_psi, 200.0 / fy_psi) * b * d


def _beam_As_for_Mu(Mu, fc, fy, b, d, phi=PHI_FLEX):
    """
    Solve φ·As·fy·(d − As·fy / (1.7·fc·b)) = Mu for As (quadratic formula).
    Returns As_required, or math.inf if Mu exceeds the section's capacity.
    """
    if Mu <= 0.0:
        return 0.0
    A = phi * fy**2 / (1.7 * fc * b)
    B = phi * fy * d
    disc = B**2 - 4.0 * A * Mu
    if disc < 0.0:
        return math.inf
    return (B - math.sqrt(disc)) / (2.0 * A)


# ─────────────────────────────────────────────────────────────────────────────
# Column: P-M interaction for explicit Ast  (mirrors RC_Design_Check but
# accepts Ast as a parameter instead of reading from sp)
# ─────────────────────────────────────────────────────────────────────────────

def _beta1(fc_ksi):
    return max(0.65, min(0.85, 0.85 - 0.05 * (fc_ksi - 4.0)))


def _layers_for_Ast(Ast):
    """
    Distribute Ast proportionally to the current top/side/bot layout in sp.
    Returns [(area, d_from_comp_face), ...].
    """
    cover  = sp.COVER
    h      = sp.H_COL
    n_top  = sp.COL_TOP_BARS
    n_bot  = sp.COL_BOT_BARS
    n_side = sp.COL_SIDE_BARS * 2   # total side bars (2 faces)
    n_tot  = n_top + n_bot + n_side

    layers = [
        (Ast * n_top / n_tot, cover),
        (Ast * n_bot / n_tot, h - cover),
    ]
    if n_side > 0:
        layers.insert(1, (Ast * n_side / n_tot, h / 2.0))
    return layers


def _phi_Mn_at_Pu(Pu, Ast, n_pts=60):
    """
    Return φMn capacity at the given Pu for a column with total steel Ast.
    Uses the same sweep as build_column_PM_diagram in RC_Design_Check.
    Returns None if Pu is outside the diagram.
    """
    fc     = sp.FC_COL_KSI
    fy     = sp.FY_KSI
    Es     = sp.ES_KSI
    b      = sp.B_COL
    h      = sp.H_COL
    layers = _layers_for_Ast(Ast)

    b1  = _beta1(fc)
    Ag  = b * h
    hc  = h / 2.0
    dt  = max(d for _, d in layers)
    Ast_tot = sum(A for A, _ in layers)

    P0      = 0.85 * fc * (Ag - Ast_tot) + fy * Ast_tot
    Pn_max  = 0.80 * P0
    diagram = [(0.65 * Pn_max, 0.0)]

    for i in range(n_pts):
        c  = 0.001 + (4.0 * h - 0.001) * i / (n_pts - 1)
        a  = min(b1 * c, h)
        Cc = 0.85 * fc * b * a
        Pn = Cc
        Mn = Cc * (hc - a / 2.0)
        for As_i, d_i in layers:
            eps_i  = ECU * (c - d_i) / c
            fs_i   = max(-fy, min(fy, Es * eps_i))
            fs_net = fs_i - (0.85 * fc if d_i <= a else 0.0)
            F_i    = As_i * fs_net
            Pn    += F_i
            Mn    += F_i * (hc - d_i)
        eps_t = ECU * (dt - c) / c
        if eps_t <= 0.002:
            phi = 0.65
        elif eps_t >= 0.005:
            phi = 0.90
        else:
            phi = 0.65 + (eps_t - 0.002) / 0.003 * 0.25
        diagram.append((phi * Pn, phi * abs(Mn)))

    # Interpolate envelope at Pu
    pts = sorted(diagram, key=lambda p: -p[0])
    cap = None
    for i in range(len(pts) - 1):
        P1, M1 = pts[i]
        P2, M2 = pts[i + 1]
        if P2 <= Pu <= P1:
            dP = P1 - P2
            t  = (Pu - P2) / dP if dP > 1e-9 else 0.0
            M  = M2 + t * (M1 - M2)
            cap = M if cap is None else max(cap, M)
    return cap


def _bisect_Ast(Pu, Mu_target, tol=1e-2, n_iter=40,
                rho_min=0.01, rho_max=0.08):
    """
    Binary search for Ast such that φMn_cap(Pu, Ast) ≈ Mu_target.

    Searches between rho_min and rho_max (ACI 318-19 §10.6.1 bounds by default).
    Returns Ast (in²), or None if the target cannot be met within the bounds.
    """
    Ag      = sp.B_COL * sp.H_COL
    Ast_lo  = rho_min * Ag
    Ast_hi  = rho_max * Ag

    # Verify bounds span the target
    cap_lo = _phi_Mn_at_Pu(Pu, Ast_lo)
    cap_hi = _phi_Mn_at_Pu(Pu, Ast_hi)

    if cap_lo is None or cap_hi is None:
        return None

    # If even max steel can't meet Mu_target, return None (geometry must change)
    if cap_hi < Mu_target:
        return None

    # If even min steel exceeds Mu_target, Ast_lo is already sufficient
    if cap_lo >= Mu_target:
        return Ast_lo

    for _ in range(n_iter):
        Ast_mid = (Ast_lo + Ast_hi) / 2.0
        cap = _phi_Mn_at_Pu(Pu, Ast_mid)
        if cap is None or cap < Mu_target:
            Ast_lo = Ast_mid
        else:
            Ast_hi = Ast_mid
        if (Ast_hi - Ast_lo) / Ast_hi < tol:
            break

    return (Ast_lo + Ast_hi) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation and selection
# ─────────────────────────────────────────────────────────────────────────────

def _col_candidates(Ast_lo, Ast_hi, cfg=None):
    """
    All (bar_size, n_top, n_bot, n_side_per_face, Ast) column combos where:
      - Ast_lo ≤ Ast ≤ Ast_hi
      - ACI ρ limits satisfied
      - n_top == n_bot (symmetric)
      - n_side_per_face in col_n_side_options from cfg
    """
    cfg        = cfg or DesignConfig()
    Ag         = sp.B_COL * sp.H_COL
    candidates = []
    for bar_size in cfg.rebar.bar_sizes_col:
        Ab = sp.rebar_area(bar_size)
        for n_top in cfg.rebar.col_n_top_iter():
            n_bot = n_top
            for n_side_pf in cfg.rebar.col_n_side_options:
                n_total = n_top + n_bot + n_side_pf * 2
                Ast     = n_total * Ab
                rho     = Ast / Ag
                if not (cfg.rebar.rho_col_min <= rho <= cfg.rebar.rho_col_max):
                    continue
                if Ast_lo <= Ast <= Ast_hi:
                    candidates.append((bar_size, n_top, n_bot, n_side_pf, Ast))
    return candidates


def _beam_candidates(As_top_lo, As_top_hi, As_bot_lo, As_bot_hi, cfg=None):
    """
    All (bar_size, n_top, n_bot) beam combos where both top and bottom
    steel fall within their respective [lo, hi] bands.
    ACI §9.6.1.2 minimum is enforced inside lo for each direction.
    """
    cfg        = cfg or DesignConfig()
    candidates = []
    for bar_size in cfg.rebar.bar_sizes_beam:
        Ab = sp.rebar_area(bar_size)
        for n_top in cfg.rebar.beam_n_iter():
            As_top = n_top * Ab
            if not (As_top_lo <= As_top <= As_top_hi):
                continue
            for n_bot in cfg.rebar.beam_n_iter():
                As_bot = n_bot * Ab
                if not (As_bot_lo <= As_bot <= As_bot_hi):
                    continue
                candidates.append((bar_size, n_top, n_bot))
    return candidates


def _pick_col(candidates, Pu: float = 0.0, Mu: float = 0.0, cfg=None):
    """
    Select the best column candidate according to the objective function.

    "min_deviation"  → candidate whose expected DCR is closest to dcr_target.
    "min_volume"     → smallest Ast (original behaviour).

    Expected DCR is estimated by computing φMn at Pu for the candidate Ast
    and comparing to Mu.
    """
    if not candidates:
        return None
    cfg = cfg or DesignConfig()

    if cfg.dcr.objective == "min_volume" or Mu < 1e-4:
        return min(candidates, key=lambda c: c[4])

    target = cfg.dcr.dcr_target

    def _dcr_estimate(c):
        bar_size, n_top, n_bot, n_side_pf, Ast = c
        cap = _phi_Mn_at_Pu(Pu, Ast)
        if cap is None or cap < 1e-6:
            return 999.0
        return Mu / cap

    return min(candidates, key=lambda c: abs(_dcr_estimate(c) - target))


def _pick_beam(candidates, Mu_pos: float = 0.0, Mu_neg: float = 0.0, cfg=None):
    """
    Select the best beam candidate according to the objective function.

    "min_deviation"  → candidate whose governing DCR is closest to dcr_target.
    "min_volume"     → smallest total As (original behaviour).
    """
    if not candidates:
        return None
    cfg    = cfg or DesignConfig()
    fc     = sp.FC_BEAM_KSI
    fy     = sp.FY_KSI
    b      = sp.B_BEAM
    d      = sp.H_BEAM - sp.COVER

    if cfg.dcr.objective == "min_volume":
        return min(candidates, key=lambda c: (c[1] + c[2]) * sp.rebar_area(c[0]))

    target = cfg.dcr.dcr_target

    def _dcr_estimate(c):
        bar_size, n_top, n_bot = c
        Ab       = sp.rebar_area(bar_size)
        As_top   = n_top * Ab
        As_bot   = n_bot * Ab
        a_top    = As_top * fy / (0.85 * fc * b)
        a_bot    = As_bot * fy / (0.85 * fc * b)
        phi_Mn_neg = PHI_FLEX * As_top * fy * (d - a_top / 2.0) if As_top > 0 else 1e-9
        phi_Mn_pos = PHI_FLEX * As_bot * fy * (d - a_bot / 2.0) if As_bot > 0 else 1e-9
        dcr_neg    = Mu_neg / phi_Mn_neg if phi_Mn_neg > 1e-6 else 0.0
        dcr_pos    = Mu_pos / phi_Mn_pos if phi_Mn_pos > 1e-6 else 0.0
        return max(dcr_neg, dcr_pos)

    return min(candidates, key=lambda c: abs(_dcr_estimate(c) - target))


# ─────────────────────────────────────────────────────────────────────────────
# Redesign from check results
# ─────────────────────────────────────────────────────────────────────────────

def redesign_steel(design_results: dict, cfg: Optional[DesignConfig] = None):
    """
    Compute updated bar specs from a set of design check results.

    Accepts either the new MemberCheckResult dict (from run_checks_phase1)
    or the legacy flat dict (from run_checks) — both are supported.

    Logic
    -----
    For each member type, identify the critical demand, then:
      1. Ast_lo : minimum steel so φCap ≥ Demand            (hard, DCR ≤ 1.0)
      2. Ast_hi : maximum steel so φCap ≤ Demand / band_lo  (soft upper)
      3. Search the discrete (bar_size, counts) space and select the candidate
         that minimises the objective (min_deviation | min_volume).
    Infeasible designs get a penalty reason logged rather than silent rejection.

    Returns
    -------
    col_update  : dict {bar_size, n_top, n_bot, n_side} or None
    beam_update : dict {bar_size, n_top, n_bot}         or None
    converged   : bool  — True when all flexural DCRs are within the soft band
    penalty_log : list[str] — reasons for any penalties applied this iteration
    """
    cfg         = cfg or DesignConfig()
    DCR_MIN     = cfg.dcr.dcr_band_lo
    DCR_MAX     = cfg.dcr.dcr_hard_max
    penalty_log: List[str] = []

    # ── Normalise result format ──────────────────────────────────────────────
    # Support both MemberCheckResult (new) and flat dict (legacy)
    def _get(r, *keys):
        for k in keys:
            if hasattr(r, k):
                return getattr(r, k)
            if isinstance(r, dict) and k in r:
                return r[k]
        return None

    def _is_col(r):
        t = _get(r, "member_type", "type")
        return t == "column"

    def _is_beam(r):
        t = _get(r, "member_type", "type")
        return t == "beam"

    cols  = {t: r for t, r in design_results.items() if _is_col(r)}
    beams = {t: r for t, r in design_results.items() if _is_beam(r)}

    # ── Convergence check ────────────────────────────────────────────────────
    def _col_flex_dcr(r):
        # MemberCheckResult
        if hasattr(r, "limit_states"):
            ls = r.limit_states.get("PM")
            return ls.dcr if ls else 0.0
        return r.get("dcr_PM", 0.0)

    def _beam_flex_dcr(r):
        if hasattr(r, "limit_states"):
            pos = r.limit_states.get("flexure_pos")
            neg = r.limit_states.get("flexure_neg")
            return max(
                pos.dcr if pos else 0.0,
                neg.dcr if neg else 0.0,
            )
        return max(r.get("dcr_pos", 0.0), r.get("dcr_neg", 0.0))

    def _shear_dcr(r):
        if hasattr(r, "limit_states"):
            ls = r.limit_states.get("shear")
            return ls.dcr if ls else 0.0
        return r.get("dcr_V", 0.0)

    col_dcrs   = [_col_flex_dcr(r)  for r in cols.values()]
    beam_dcrs  = [_beam_flex_dcr(r) for r in beams.values()]
    shear_dcrs = [_shear_dcr(r)     for r in design_results.values()]

    flexural_dcrs = [d for d in col_dcrs + beam_dcrs if d > 0]
    flexural_converged = (
        bool(flexural_dcrs)
        and all(DCR_MIN <= d <= cfg.dcr.dcr_band_hi for d in flexural_dcrs)
    )
    shear_converged = all(d <= DCR_MAX for d in shear_dcrs)
    converged = flexural_converged and shear_converged

    if flexural_converged and not shear_converged:
        penalty_log.append(
            "Flexural DCRs in band but shear DCR exceeds 1.0; "
            "shear governed by stirrup spacing — adjust manually."
        )
        return None, None, False, penalty_log

    # ── Column redesign ──────────────────────────────────────────────────────
    col_update = None
    if cols:
        Ag          = sp.B_COL * sp.H_COL
        Ast_aci_min = cfg.rebar.rho_col_min * Ag
        Ast_aci_max = cfg.rebar.rho_col_max * Ag

        # Critical column: highest P-M DCR
        def _col_sort_key(r):
            return _col_flex_dcr(r)

        crit = max(cols.values(), key=_col_sort_key)

        # Extract Pu and Mu from either result format
        if hasattr(crit, "limit_states"):
            ls_pm = crit.limit_states.get("PM")
            Pu    = crit.section_info.get("Pu_kip", ls_pm.demand if ls_pm else 0.0)
            Mu    = crit.section_info.get("Mu_kip_in", ls_pm.demand if ls_pm else 0.0)
        else:
            Pu = crit.get("Pu", 0.0)
            Mu = crit.get("Mu", 0.0)

        _rho_min = cfg.rebar.rho_col_min
        _rho_max = cfg.rebar.rho_col_max
        if Mu > 0:
            Ast_lo = _bisect_Ast(Pu, Mu, rho_min=_rho_min, rho_max=_rho_max)           # DCR = 1.0 lower bound
            Ast_hi = _bisect_Ast(Pu, Mu / DCR_MIN, rho_min=_rho_min, rho_max=_rho_max) # soft upper
        else:
            Ast_lo = Ast_aci_min
            Ast_hi = _bisect_Ast(Pu, Pu / DCR_MIN, rho_min=_rho_min, rho_max=_rho_max) if Pu > 0 else Ast_aci_min

        Ast_lo = Ast_lo if Ast_lo is not None else Ast_aci_min
        Ast_hi = Ast_hi if Ast_hi is not None else Ast_aci_max

        if abs(Ast_hi - Ast_lo) < 0.05:
            Ast_hi = min(Ast_lo * 1.5, Ast_aci_max)

        candidates = _col_candidates(Ast_lo, Ast_hi, cfg)

        if not candidates:
            penalty_log.append(
                f"Column: no candidate in Ast band [{Ast_lo:.2f}, {Ast_hi:.2f}] in²; "
                "falling back to ACI minimum arrangement."
            )
            candidates = _col_candidates(Ast_aci_min, Ast_aci_min * 1.5, cfg)

        best = _pick_col(candidates, Pu=Pu, Mu=Mu, cfg=cfg)
        if best:
            bar_size, n_top, n_bot, n_side_pf, _ = best
            col_update = {
                "bar_size": bar_size,
                "n_top":    n_top,
                "n_bot":    n_bot,
                "n_side":   n_side_pf,
            }
        else:
            penalty_log.append(
                "Column: no feasible bar combination found — "
                "section geometry may need to change."
            )

    # ── Beam redesign ────────────────────────────────────────────────────────
    beam_update = None
    if beams:
        fc = sp.FC_BEAM_KSI
        fy = sp.FY_KSI
        b  = sp.B_BEAM
        d  = sp.H_BEAM - sp.COVER

        def _mu_neg(r):
            if hasattr(r, "limit_states"):
                ls = r.limit_states.get("flexure_neg")
                return ls.demand if ls else 0.0
            return r.get("Mu_neg", 0.0)

        def _mu_pos(r):
            if hasattr(r, "limit_states"):
                ls = r.limit_states.get("flexure_pos")
                return ls.demand if ls else 0.0
            return r.get("Mu_pos", 0.0)

        Mu_neg_crit = max(_mu_neg(r) for r in beams.values())
        Mu_pos_crit = max(_mu_pos(r) for r in beams.values())

        As_code_min = _beam_As_min(fc, fy, b, d)

        As_top_lo = max(_beam_As_for_Mu(Mu_neg_crit, fc, fy, b, d), As_code_min)
        As_top_hi = max(_beam_As_for_Mu(Mu_neg_crit / DCR_MIN, fc, fy, b, d), As_code_min)
        As_bot_lo = max(_beam_As_for_Mu(Mu_pos_crit, fc, fy, b, d), As_code_min)
        As_bot_hi = max(_beam_As_for_Mu(Mu_pos_crit / DCR_MIN, fc, fy, b, d), As_code_min)

        if abs(As_top_hi - As_top_lo) < 0.05:
            As_top_hi = As_top_lo * 1.5
        if abs(As_bot_hi - As_bot_lo) < 0.05:
            As_bot_hi = As_bot_lo * 1.5

        candidates = _beam_candidates(As_top_lo, As_top_hi, As_bot_lo, As_bot_hi, cfg)

        if not candidates:
            step = sp.rebar_area(cfg.rebar.bar_sizes_beam[0])
            penalty_log.append(
                f"Beam: no candidate in As bands "
                f"top=[{As_top_lo:.2f},{As_top_hi:.2f}] "
                f"bot=[{As_bot_lo:.2f},{As_bot_hi:.2f}] in²; "
                "widening to code-min fallback."
            )
            candidates = _beam_candidates(
                As_code_min, As_code_min + step * 3,
                As_code_min, As_code_min + step * 3,
                cfg,
            )

        best = _pick_beam(candidates, Mu_pos=Mu_pos_crit, Mu_neg=Mu_neg_crit, cfg=cfg)
        if best:
            bar_size, n_top, n_bot = best
            beam_update = {"bar_size": bar_size, "n_top": n_top, "n_bot": n_bot}
        else:
            penalty_log.append(
                "Beam: no feasible bar combination found — "
                "section geometry may need to change."
            )

    return col_update, beam_update, converged, penalty_log


def apply_updates(col_update, beam_update):
    """Mutate Structure_Parameters in-place with the new bar specs."""
    if col_update:
        sp.COL_BAR_SIZE  = col_update['bar_size']
        sp.COL_TOP_BARS  = col_update['n_top']
        sp.COL_BOT_BARS  = col_update['n_bot']
        sp.COL_SIDE_BARS = col_update['n_side']   # per face
        sp.COL_BAR_AREA  = sp.rebar_area(col_update['bar_size'])

    if beam_update:
        sp.BEAM_BAR_SIZE = beam_update['bar_size']
        sp.BEAM_TOP_BARS = beam_update['n_top']
        sp.BEAM_BOT_BARS = beam_update['n_bot']
        sp.BEAM_BAR_AREA = sp.rebar_area(beam_update['bar_size'])


def run_iterative_redesign(
    cfg: Optional[DesignConfig] = None,
    max_iter: int = 10,
    verbose: bool = True,
):
    """
    Iterative gravity-driven steel redesign loop.

    Parameters
    ----------
    cfg      : DesignConfig controlling bar pools, ACI limits, DCR targets,
               and objective function.  Defaults to DesignConfig() (reads
               DESIGN_DCR_MIN/MAX from Structure_Parameters).
    max_iter : maximum redesign iterations
    verbose  : print per-iteration progress

    Returns
    -------
    converged      : bool
    n_iter         : int
    gravity_results: dict  — gravity analysis results from final iteration
    design_results : dict  — legacy flat-dict schema (backward-compatible)
    """
    cfg = cfg or DesignConfig()

    if verbose:
        print("\n" + "-" * 60)
        print(
            "  Iterative Steel Redesign  "
            f"(target DCR: {cfg.dcr.dcr_band_lo:.2f} - {cfg.dcr.dcr_band_hi:.2f}  "
            f"obj: {cfg.dcr.objective})"
        )
        print("-" * 60)

    gravity_results = None
    phase1_results  = {}   # dict[int, MemberCheckResult]
    legacy_results  = {}   # dict compatible with print_summary / summarize_design_results

    for iteration in range(1, max_iter + 1):
        if verbose:
            print(
                f"\n  Iteration {iteration}/{max_iter}"
                f"  |  Column: #{sp.COL_BAR_SIZE} "
                f"{sp.COL_TOP_BARS}T+{sp.COL_BOT_BARS}B+{sp.COL_SIDE_BARS * 2}S"
                f"  |  Beam: #{sp.BEAM_BAR_SIZE} "
                f"{sp.BEAM_TOP_BARS}T+{sp.BEAM_BOT_BARS}B"
            )

        with _silence_c_output():
            build_model()
        apply_gravity_loads()

        try:
            gravity_results = run_gravity_analysis()
        except RuntimeError as exc:
            print(f"  [Redesign] Gravity failed: {exc}")
            return False, iteration, None, {}

        col_tags, beam_x_tags, beam_y_tags = get_element_tags()

        # Use new Phase 1 checks internally
        phase1_results = run_checks_phase1(col_tags, beam_x_tags + beam_y_tags, cfg)
        legacy_results = to_legacy_dict(phase1_results)

        # Collect per-type DCR ranges for progress output
        col_pm_dcrs    = [r.dcr("PM")    for r in phase1_results.values() if r.member_type == "column"]
        col_v_dcrs     = [r.dcr("shear") for r in phase1_results.values() if r.member_type == "column"]
        beam_flex_dcrs = [
            max(r.dcr("flexure_pos"), r.dcr("flexure_neg"))
            for r in phase1_results.values() if r.member_type == "beam"
        ]
        beam_v_dcrs    = [r.dcr("shear") for r in phase1_results.values() if r.member_type == "beam"]

        if verbose and col_pm_dcrs:
            print(f"    Column P-M DCR  min={min(col_pm_dcrs):.3f}  max={max(col_pm_dcrs):.3f}")
            print(f"    Column V   DCR  min={min(col_v_dcrs):.3f}  max={max(col_v_dcrs):.3f}")
            print(f"    Beam flex  DCR  min={min(beam_flex_dcrs):.3f}  max={max(beam_flex_dcrs):.3f}")
            print(f"    Beam V     DCR  min={min(beam_v_dcrs):.3f}  max={max(beam_v_dcrs):.3f}")

        col_upd, beam_upd, converged, penalty_log = redesign_steel(phase1_results, cfg)

        if penalty_log and verbose:
            for reason in penalty_log:
                print(f"    [penalty] {reason}")

        if converged:
            if verbose:
                print(f"\n  Converged in {iteration} iteration(s).")
            return True, iteration, gravity_results, legacy_results

        if verbose:
            if col_upd:
                print(
                    f"    -> Column update: #{col_upd['bar_size']}  "
                    f"{col_upd['n_top']}T+{col_upd['n_bot']}B"
                    f"+{col_upd['n_side'] * 2}S"
                )
            else:
                print("    -> Column: no feasible update found")
            if beam_upd:
                print(
                    f"    -> Beam   update: #{beam_upd['bar_size']}  "
                    f"{beam_upd['n_top']}T+{beam_upd['n_bot']}B"
                )
            else:
                print("    -> Beam  : no feasible update found")

        if col_upd is None and beam_upd is None:
            print(
                "  [Redesign] No feasible longitudinal-bar update found. "
                "Check penalty log above for details."
            )
            return False, iteration, gravity_results, legacy_results

        previous = (
            sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS,
            sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS,
        )
        apply_updates(col_upd, beam_upd)
        current = (
            sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS,
            sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS,
        )
        if current == previous:
            design_passes = all(r.ok for r in phase1_results.values())
            if design_passes:
                if verbose:
                    print(
                        "  [Redesign] Bar layout stable; all checks pass — "
                        "stopping at closest discrete/code-min layout."
                    )
                return True, iteration, gravity_results, legacy_results
            if verbose:
                print("  [Redesign] Bar layout stable; some checks still fail — stopping.")
            return False, iteration, gravity_results, legacy_results

    if verbose:
        print(f"\n  Did not converge in {max_iter} iterations.")
    return False, max_iter, gravity_results, legacy_results
