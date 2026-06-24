"""
Design/ACI_Checks.py
Phase 1 elastic ACI 318-19 design checks returning typed result objects.

This module contains the pure-math check logic.  Force extraction (which
requires a live OpenSees model) stays in RC_Design_Check.py and is imported
here rather than duplicated.

Entry point
-----------
    from Design.ACI_Checks import run_checks_phase1, to_legacy_dict
    from Design.Config import DesignConfig

    cfg     = DesignConfig()
    results = run_checks_phase1(col_tags, beam_tags, cfg)
    # results : dict[int, MemberCheckResult]

    legacy  = to_legacy_dict(results)
    # legacy  : dict compatible with RC_Design_Check.print_summary()

Limit states tracked per element type
--------------------------------------
  Column  :  P-M interaction (§22.4), shear (§22.5)
             slenderness (§6.2.5) — scaffolded, TODO
             development length (§25.4) — scaffolded, TODO
  Beam    :  flexure positive (§22.3), flexure negative (§22.3)
             shear (§22.5)
             development length (§25.4) — scaffolded, TODO
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import Structure_Parameters as sp
from Design.CheckResult import LimitStateResult, MemberCheckResult
from Design.Config import DesignConfig, RebarConfig, MaterialConfig

# Force extraction (OpenSees-coupled layer) stays in RC_Design_Check
from RC_Design_Check import get_element_tags, _extract_forces


# ---------------------------------------------------------------------------
# ACI 318-19 section constants
# ---------------------------------------------------------------------------

PHI_FLEX: float = 0.90     # ACI Table 21.2.1 — tension-controlled
PHI_SHEAR: float = 0.75    # ACI Table 21.2.1 — shear


# ---------------------------------------------------------------------------
# ACI helper functions
# ---------------------------------------------------------------------------

def _beta1(fc_ksi: float) -> float:
    """ACI 318-19 §22.2.2.4.3  (0.65 ≤ β₁ ≤ 0.85)."""
    return max(0.65, min(0.85, 0.85 - 0.05 * (fc_ksi - 4.0)))


def _phi_PM(eps_t: float) -> float:
    """ACI 318-19 Table 21.2.2 — tied-column φ factor (0.65 → 0.90)."""
    if eps_t <= 0.002:
        return 0.65
    if eps_t >= 0.005:
        return 0.90
    return 0.65 + (eps_t - 0.002) / 0.003 * 0.25


# ---------------------------------------------------------------------------
# Effective depths
# ---------------------------------------------------------------------------

def _col_d(cfg: DesignConfig) -> float:
    return cfg.sections.h_col_in - cfg.rebar.cover_in


def _beam_d(cfg: DesignConfig) -> float:
    return cfg.sections.h_beam_in - cfg.rebar.cover_in


# ---------------------------------------------------------------------------
# ACI §9.6.1.2 — beam minimum longitudinal steel
# ---------------------------------------------------------------------------

def _beam_As_min(fc_ksi: float, fy_ksi: float, b: float, d: float) -> float:
    """ACI 318-19 §9.6.1.2 minimum beam steel area (in²)."""
    fc_psi = fc_ksi * 1000.0
    fy_psi = fy_ksi * 1000.0
    return max(3.0 * math.sqrt(fc_psi) / fy_psi, 200.0 / fy_psi) * b * d


# ---------------------------------------------------------------------------
# Column P-M interaction diagram
# ---------------------------------------------------------------------------

def _col_steel_layers(
    cfg: DesignConfig,
) -> List[Tuple[float, float]]:
    """
    Build the column steel layer list from Structure_Parameters.
    Returns [(area_in2, dist_from_compression_face_in), ...].
    """
    cover = cfg.rebar.cover_in
    h     = cfg.sections.h_col_in
    Ab    = sp.COL_BAR_AREA
    layers = [
        (sp.COL_TOP_BARS * Ab, cover),
        (sp.COL_BOT_BARS * Ab, h - cover),
    ]
    if sp.COL_SIDE_BARS > 0:
        layers.insert(1, (sp.COL_SIDE_BARS * Ab, h / 2.0))
    return layers


def build_pm_diagram(cfg: DesignConfig, n_pts: int = 120) -> List[Tuple[float, float]]:
    """
    φPn–φMn interaction diagram for the current column section.

    Sweeps neutral-axis depth c from near-zero (tension-controlled) to 4h
    (near pure compression).  All geometry and materials come from cfg and sp.

    Returns
    -------
    list of (phi_Pn_kips, phi_Mn_kip_in)
        Compression positive for φPn.  φMn ≥ 0.
    """
    fc     = cfg.materials.fc_col_ksi if isinstance(cfg.materials.fc_col_ksi, float) else sp.FC_COL_KSI
    fy     = cfg.materials.fy_ksi     if isinstance(cfg.materials.fy_ksi,     float) else sp.FY_KSI
    Es     = cfg.materials.es_ksi
    b      = cfg.sections.b_col_in
    h      = cfg.sections.h_col_in
    layers = _col_steel_layers(cfg)

    ecu = 0.003
    b1  = _beta1(fc)
    Ag  = b * h
    Ast = sum(A for A, _ in layers)
    hc  = h / 2.0
    dt  = max(d for _, d in layers)

    P0     = 0.85 * fc * (Ag - Ast) + fy * Ast   # ACI §22.4.2.2
    Pn_max = 0.80 * P0                            # ACI §22.4.2.1
    diagram: List[Tuple[float, float]] = [(0.65 * Pn_max, 0.0)]

    for i in range(n_pts):
        c  = 0.001 + (4.0 * h - 0.001) * i / (n_pts - 1)
        a  = min(b1 * c, h)
        Cc = 0.85 * fc * b * a
        Pn = Cc
        Mn = Cc * (hc - a / 2.0)

        for As_i, d_i in layers:
            eps_i  = ecu * (c - d_i) / c
            fs_i   = max(-fy, min(fy, Es * eps_i))
            fs_net = fs_i - (0.85 * fc if d_i <= a else 0.0)
            F_i    = As_i * fs_net
            Pn    += F_i
            Mn    += F_i * (hc - d_i)

        eps_t = ecu * (dt - c) / c
        phi   = _phi_PM(eps_t)
        diagram.append((phi * Pn, phi * abs(Mn)))

    return diagram


def _interpolate_pm_capacity(Pu: float, diagram: List[Tuple[float, float]]) -> Optional[float]:
    """Return the interpolated φMn capacity at the given Pu.  None if out of range."""
    pts = sorted(diagram, key=lambda p: -p[0])
    cap: Optional[float] = None
    for i in range(len(pts) - 1):
        P1, M1 = pts[i]
        P2, M2 = pts[i + 1]
        if P2 <= Pu <= P1:
            dP = P1 - P2
            t  = (Pu - P2) / dP if dP > 1e-9 else 0.0
            M  = M2 + t * (M1 - M2)
            cap = M if cap is None else max(cap, M)
    return cap


# ---------------------------------------------------------------------------
# Per-limit-state check functions  →  LimitStateResult
# ---------------------------------------------------------------------------

def check_column_pm(
    Pu: float,
    Muz: float,
    Muy: float,
    diagram: List[Tuple[float, float]],
    cfg: DesignConfig,
) -> LimitStateResult:
    """
    ACI 318-19 §22.4 column P-M interaction.

    Biaxial bending via resultant moment: Mu = √(Muz² + Muy²).
    DCR = distance from origin to demand point
          / distance from origin to capacity surface along the same load angle.

    min_controlled: True when Ast ≤ ρ_min × Ag × 1.01.
    """
    Mu     = math.sqrt(Muz**2 + Muy**2)
    Ag     = cfg.sections.b_col_in * cfg.sections.h_col_in
    Ast    = (sp.COL_TOP_BARS + sp.COL_BOT_BARS + sp.COL_SIDE_BARS * 2) * sp.COL_BAR_AREA
    min_controlled = Ast <= cfg.rebar.rho_col_min * Ag * 1.01

    # Pure-compression case (zero eccentricity)
    if Mu < 1e-4:
        phi_Pmax = diagram[0][0]
        cap      = phi_Pmax if phi_Pmax > 1e-6 else 1e-9
        dcr      = Pu / cap
        return LimitStateResult(
            name="PM", demand=Pu, capacity=cap,
            dcr=dcr, ok=dcr <= 1.0, min_controlled=min_controlled,
        )

    phi_Mn_cap = _interpolate_pm_capacity(Pu, diagram)

    if phi_Mn_cap is None or phi_Mn_cap < 1e-6:
        return LimitStateResult(
            name="PM", demand=Mu, capacity=0.0,
            dcr=999.0, ok=False, min_controlled=min_controlled,
        )

    dcr = Mu / phi_Mn_cap
    return LimitStateResult(
        name="PM", demand=Mu, capacity=phi_Mn_cap,
        dcr=dcr, ok=dcr <= 1.0, min_controlled=min_controlled,
    )


def check_beam_flexure_pos(
    Mu_pos: float,
    cfg: DesignConfig,
) -> LimitStateResult:
    """
    ACI 318-19 §22.3 — positive (sagging) beam flexure.
    Capacity governed by BEAM_BOT_BARS.

    min_controlled: True when As_bot ≤ As_min × 1.01.
    """
    fc  = cfg.materials.fc_beam_ksi if isinstance(cfg.materials.fc_beam_ksi, float) else sp.FC_BEAM_KSI
    fy  = cfg.materials.fy_ksi      if isinstance(cfg.materials.fy_ksi,      float) else sp.FY_KSI
    b   = cfg.sections.b_beam_in
    d   = _beam_d(cfg)

    As_bot = sp.BEAM_BOT_BARS * sp.BEAM_BAR_AREA
    a      = As_bot * fy / (0.85 * fc * b)
    phi_Mn = PHI_FLEX * As_bot * fy * (d - a / 2.0)

    As_min        = _beam_As_min(fc, fy, b, d)
    min_controlled = As_bot <= As_min * 1.01

    demand  = Mu_pos
    cap     = max(phi_Mn, 1e-9)
    dcr     = demand / cap if demand > 0.0 else 0.0

    return LimitStateResult(
        name="flexure_pos", demand=demand, capacity=cap,
        dcr=dcr, ok=dcr <= 1.0, min_controlled=min_controlled,
    )


def check_beam_flexure_neg(
    Mu_neg: float,
    cfg: DesignConfig,
) -> LimitStateResult:
    """
    ACI 318-19 §22.3 — negative (hogging) beam flexure.
    Capacity governed by BEAM_TOP_BARS.

    min_controlled: True when As_top ≤ As_min × 1.01.
    """
    fc  = cfg.materials.fc_beam_ksi if isinstance(cfg.materials.fc_beam_ksi, float) else sp.FC_BEAM_KSI
    fy  = cfg.materials.fy_ksi      if isinstance(cfg.materials.fy_ksi,      float) else sp.FY_KSI
    b   = cfg.sections.b_beam_in
    d   = _beam_d(cfg)

    As_top = sp.BEAM_TOP_BARS * sp.BEAM_BAR_AREA
    a      = As_top * fy / (0.85 * fc * b)
    phi_Mn = PHI_FLEX * As_top * fy * (d - a / 2.0)

    As_min        = _beam_As_min(fc, fy, b, d)
    min_controlled = As_top <= As_min * 1.01

    demand  = Mu_neg
    cap     = max(phi_Mn, 1e-9)
    dcr     = demand / cap if demand > 0.0 else 0.0

    return LimitStateResult(
        name="flexure_neg", demand=demand, capacity=cap,
        dcr=dcr, ok=dcr <= 1.0, min_controlled=min_controlled,
    )


def check_shear(
    Vu: float,
    Nu: float,
    bw: float,
    d: float,
    fc_ksi: float,
    Av: float,
    s: float,
    fy_ksi: float,
    lambda_: float = 1.0,
) -> LimitStateResult:
    """
    ACI 318-19 §22.5 — simplified shear strength.

    Vc = 2λ√f'c · bw · d  +  Nu / (6·Ag)   (compression boost for columns)
    Vs = Av · fy · d / s
    φVn = 0.75(Vc + Vs)

    Nu   : axial compression, kip (positive = compression); 0 for beams.
    Av   : total shear-steel area across all stirrup legs (in²).
    s    : stirrup spacing (in).
    """
    Ag     = bw * (d / 0.9)
    Vc     = (2.0 * lambda_ * math.sqrt(fc_ksi * 1000.0) * bw * d) / 1000.0
    Vc    += max(0.0, Nu) / (6.0 * Ag)
    Vs     = Av * fy_ksi * d / s
    phi_Vn = PHI_SHEAR * (Vc + Vs)

    cap = max(phi_Vn, 1e-9)
    dcr = Vu / cap

    return LimitStateResult(
        name="shear", demand=Vu, capacity=cap,
        dcr=dcr, ok=dcr <= 1.0,
    )


# ---------------------------------------------------------------------------
# Scaffold checks — flagged, not fully implemented
# ---------------------------------------------------------------------------

def check_slenderness(
    member_type: str,
    L_in: float,
    b_in: float,
    h_in: float,
) -> LimitStateResult:
    """
    ACI 318-19 §6.2.5 — column slenderness.

    TODO: implement the moment-magnification procedure for slender columns
    (non-sway: §6.6.4, sway: §6.6.5).  Currently returns a passing scaffold
    result so the limit state slot exists in the output schema.

    The flag min_controlled is reused here as "scaffold_not_implemented".
    """
    # ACI §6.2.5 — for non-sway frames, slenderness negligible if klu/r ≤ 22
    # r ≈ 0.3h for rectangular sections (ACI §10.10.1.2)
    r = 0.3 * min(b_in, h_in)
    klu_over_r = 1.0 * L_in / r   # k=1.0 conservative for non-sway

    return LimitStateResult(
        name="slenderness",
        demand=klu_over_r,
        capacity=22.0,            # ACI §6.2.5 non-sway limit
        dcr=klu_over_r / 22.0,
        ok=True,
        min_controlled=True,      # sentinel: scaffold not fully implemented
    )


def check_development_length(
    bar_size: int,
    fc_ksi: float,
    fy_ksi: float,
    cover_in: float,
    available_length_in: float,
) -> LimitStateResult:
    """
    ACI 318-19 §25.4 — development length feasibility.

    TODO: implement full development length / lap splice check including
    bar spacing, confinement reinforcement, epoxy coating, and
    lightweight concrete factors.

    Currently provides a simplified ld estimate for straight bars in
    normal-weight concrete with adequate cover (simplified §25.4.2.3).
    """
    # Simplified ld: ld = (3/40) × (fy / (λ√f'c)) × (Ab / (cb + Ktr))
    # Using conservative cb = cover, Ktr = 0, λ = 1.0
    fy_psi  = fy_ksi * 1000.0
    fc_psi  = fc_ksi * 1000.0
    Ab      = sp.rebar_area(bar_size)
    db      = sp.rebar_diameter(bar_size)
    cb_ktr  = max(cover_in + db / 2.0, 2.5 * db)  # cap per §25.4.2.3
    ld      = (3.0 / 40.0) * (fy_psi / (1.0 * math.sqrt(fc_psi))) * (db / (cb_ktr / db))
    ld      = max(ld, 12.0)   # ACI §25.4.2.1 minimum

    feasible = available_length_in >= ld
    dcr      = ld / max(available_length_in, 1e-6)

    return LimitStateResult(
        name="dev_length",
        demand=ld,
        capacity=available_length_in,
        dcr=dcr,
        ok=feasible,
        min_controlled=False,
    )


# ---------------------------------------------------------------------------
# Strong-column-weak-beam — scaffold (requires joint connectivity)
# ---------------------------------------------------------------------------

def check_scwb_joint(
    sum_phi_Mn_col: float,
    sum_phi_Mn_beam: float,
    ratio_min: float = 1.2,
) -> LimitStateResult:
    """
    ACI 318-19 §18.7.3 — strong-column-weak-beam capacity ratio.

    TODO: compute sum_phi_Mn_col and sum_phi_Mn_beam at each joint from
    the element properties.  This requires iterating over beam-column
    joints in the structural grid and computing column capacities at the
    axial load level present during the lateral load case.

    Parameters
    ----------
    sum_phi_Mn_col   : ΣφMnc at the joint (kip·in)
    sum_phi_Mn_beam  : ΣφMnb at the joint (kip·in)
    ratio_min        : required ratio (ACI §18.7.3: ≥ 1.2)
    """
    ratio = sum_phi_Mn_col / max(sum_phi_Mn_beam, 1e-9)
    return LimitStateResult(
        name="scwb",
        demand=sum_phi_Mn_beam,
        capacity=sum_phi_Mn_col / ratio_min,
        dcr=ratio_min / max(ratio, 1e-9),   # DCR<1 means column is stronger
        ok=ratio >= ratio_min,
        min_controlled=True,   # sentinel: scaffold
    )


# ---------------------------------------------------------------------------
# Main Phase 1 entry point
# ---------------------------------------------------------------------------

def run_checks_phase1(
    col_tags: List[int],
    beam_tags: List[int],
    cfg: DesignConfig,
) -> Dict[int, MemberCheckResult]:
    """
    Run all Phase 1 ACI 318-19 checks for the current OpenSees model state.

    Parameters
    ----------
    col_tags   : column element tags (from get_element_tags())
    beam_tags  : beam element tags (X + Y combined)
    cfg        : DesignConfig controlling materials, rebar, and section dims

    Returns
    -------
    dict[int, MemberCheckResult]
        One entry per element tag.  Each result carries per-limit-state DCRs,
        the governing limit state, ok flag, and min_controlled flags.
    """
    fy   = cfg.materials.fy_ksi if isinstance(cfg.materials.fy_ksi, float) else sp.FY_KSI
    Av   = cfg.rebar.stirrup_area_in2

    pm_diagram = build_pm_diagram(cfg)
    results: Dict[int, MemberCheckResult] = {}

    # ── Columns ─────────────────────────────────────────────────────────────
    for tag in col_tags:
        P, Vy, Vz, My, Mz, _, _, _, _ = _extract_forces(tag)
        Vu = math.sqrt(Vy**2 + Vz**2)

        ls_pm    = check_column_pm(P, Mz, My, pm_diagram, cfg)
        ls_shear = check_shear(
            Vu, P,
            bw=cfg.sections.b_col_in, d=_col_d(cfg),
            fc_ksi=(cfg.materials.fc_col_ksi
                    if isinstance(cfg.materials.fc_col_ksi, float)
                    else sp.FC_COL_KSI),
            Av=Av, s=cfg.rebar.stirrup_spacing_col_in, fy_ksi=fy,
        )
        ls_slend = check_slenderness(
            "column",
            L_in=sp.STORY_H,
            b_in=cfg.sections.b_col_in,
            h_in=cfg.sections.h_col_in,
        )

        Ast = (sp.COL_TOP_BARS + sp.COL_BOT_BARS + sp.COL_SIDE_BARS * 2) * sp.COL_BAR_AREA
        Mu = math.sqrt(Mz**2 + My**2)
        result = MemberCheckResult(
            ele_tag=tag,
            member_type="column",
            limit_states={
                "PM":          ls_pm,
                "shear":       ls_shear,
                "slenderness": ls_slend,
            },
            section_info={
                "bar_size":    sp.COL_BAR_SIZE,
                "n_top":       sp.COL_TOP_BARS,
                "n_bot":       sp.COL_BOT_BARS,
                "n_side":      sp.COL_SIDE_BARS,
                "Ast_in2":     Ast,
                "rho":         Ast / (cfg.sections.b_col_in * cfg.sections.h_col_in),
                "b_in":        cfg.sections.b_col_in,
                "h_in":        cfg.sections.h_col_in,
                "Pu_kip":      P,
                "Mu_kip_in":   Mu,
                "Muy_kip_in":  My,
                "Muz_kip_in":  Mz,
                "Vu_kip":      Vu,
            },
        )
        result._recompute()
        results[tag] = result

    # ── Beams ────────────────────────────────────────────────────────────────
    for tag in beam_tags:
        P, Vy, Vz, My, Mz, My_i, My_j, Mz_i, Mz_j = _extract_forces(tag)
        end_moments = (My_i, My_j, Mz_i, Mz_j)
        Mu_pos = max(0.0, max(end_moments))
        Mu_neg = max(0.0, max(-m for m in end_moments))
        Vu     = math.sqrt(Vy**2 + Vz**2)

        ls_flex_pos = check_beam_flexure_pos(Mu_pos, cfg)
        ls_flex_neg = check_beam_flexure_neg(Mu_neg, cfg)
        ls_shear    = check_shear(
            Vu, 0.0,
            bw=cfg.sections.b_beam_in, d=_beam_d(cfg),
            fc_ksi=(cfg.materials.fc_beam_ksi
                    if isinstance(cfg.materials.fc_beam_ksi, float)
                    else sp.FC_BEAM_KSI),
            Av=Av, s=cfg.rebar.stirrup_spacing_beam_in, fy_ksi=fy,
        )

        result = MemberCheckResult(
            ele_tag=tag,
            member_type="beam",
            limit_states={
                "flexure_pos": ls_flex_pos,
                "flexure_neg": ls_flex_neg,
                "shear":       ls_shear,
            },
            section_info={
                "bar_size":    sp.BEAM_BAR_SIZE,
                "n_top":       sp.BEAM_TOP_BARS,
                "n_bot":       sp.BEAM_BOT_BARS,
                "As_top_in2":  sp.BEAM_TOP_BARS * sp.BEAM_BAR_AREA,
                "As_bot_in2":  sp.BEAM_BOT_BARS * sp.BEAM_BAR_AREA,
                "b_in":        cfg.sections.b_beam_in,
                "h_in":        cfg.sections.h_beam_in,
            },
        )
        result._recompute()
        results[tag] = result

    return results


# ---------------------------------------------------------------------------
# Backward-compatibility converter
# ---------------------------------------------------------------------------

def to_legacy_dict(phase1_results: Dict[int, MemberCheckResult]) -> dict:
    """
    Convert Phase 1 results to the flat-dict schema expected by
    RC_Design_Check.print_summary() and Main.summarize_design_results().

    This allows Phase 1 internals to use the new typed objects while the
    rest of the pipeline continues working without changes.
    """
    legacy = {}
    for tag, r in phase1_results.items():
        if r.member_type == "column":
            ls_pm    = r.limit_states.get("PM",    LimitStateResult("PM",    0, 1, 0, True))
            ls_shear = r.limit_states.get("shear", LimitStateResult("shear", 0, 1, 0, True))
            Pu = r.section_info.get("Pu_kip", ls_pm.demand)
            Mu = r.section_info.get("Mu_kip_in", ls_pm.demand)
            legacy[tag] = {
                "type":        "column",
                "Pu":          Pu,
                "Mu":          Mu,
                "phi_Mn_cap":  ls_pm.capacity,
                "Vu":          ls_shear.demand,
                "phi_Vn":      ls_shear.capacity,
                "dcr_PM":      ls_pm.dcr,
                "dcr_V":       ls_shear.dcr,
                "ok_PM":       ls_pm.ok,
                "ok_V":        ls_shear.ok,
                "ok":          r.ok,
                # Extra Phase 1 fields not in legacy schema
                "min_controlled":        r.any_min_controlled,
                "governing_limit_state": r.governing_limit_state,
                "penalty_reasons":       r.penalty_reasons,
            }
        else:
            ls_pos   = r.limit_states.get("flexure_pos", LimitStateResult("flexure_pos", 0, 1, 0, True))
            ls_neg   = r.limit_states.get("flexure_neg", LimitStateResult("flexure_neg", 0, 1, 0, True))
            ls_shear = r.limit_states.get("shear",       LimitStateResult("shear",       0, 1, 0, True))
            legacy[tag] = {
                "type":        "beam",
                "Mu_pos":      ls_pos.demand,
                "Mu_neg":      ls_neg.demand,
                "phi_Mn_pos":  ls_pos.capacity,
                "phi_Mn_neg":  ls_neg.capacity,
                "Mu_y":        ls_pos.demand,   # approximate
                "Mu_z":        ls_neg.demand,
                "Vu":          ls_shear.demand,
                "phi_Vn":      ls_shear.capacity,
                "dcr_pos":     ls_pos.dcr,
                "dcr_neg":     ls_neg.dcr,
                "dcr_V":       ls_shear.dcr,
                "ok_F":        ls_pos.ok and ls_neg.ok,
                "ok_V":        ls_shear.ok,
                "ok":          r.ok,
                # Extra Phase 1 fields
                "min_controlled":        r.any_min_controlled,
                "governing_limit_state": r.governing_limit_state,
                "penalty_reasons":       r.penalty_reasons,
            }
    return legacy
