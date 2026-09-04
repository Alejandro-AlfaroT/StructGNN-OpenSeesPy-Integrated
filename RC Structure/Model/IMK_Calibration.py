"""
Model/IMK_Calibration.py
========================

Per-member IMK backbone calibration.

Before this module the hinge backbone was a set of global constants: every
element in every structure used theta_p = 0.020, theta_pc = 0.060 and a yield
moment computed with no axial load. That is defensible for beams, where axial
force is negligible, but it is wrong for columns, whose moment capacity and
plastic rotation capacity both depend strongly on how hard they are being
squeezed. With one constant backbone, a lightly loaded roof column and a
heavily loaded ground-floor column degrade identically, and no soft-story
mechanism can emerge from the model no matter what the ground motion does.

Two calibrations are provided:

  * yield moment from the nominal P-M interaction surface at the member's
    axial load, rather than the zero-axial flexural capacity
  * plastic and post-capping rotation capacity from Haselton et al. (2008),
    PEER 2007/03, which are the calibrations underlying the ASCE 41 and
    FEMA P-58 concrete hinge parameters

Beams are the zero-axial case of the same equations, so both member types go
through one path.

Unit system: kip, inch, ksi. The Haselton expressions are dimensional and
were fit in MPa, so concrete strength and steel yield are converted locally.
"""

from __future__ import annotations

import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from RC_Design_Check import _beta1, _col_steel_layers


KSI_TO_MPA = 6.894757

# Haselton et al. (2008) fit bounds. Outside these the expressions are
# extrapolation, so inputs are clamped and the clamp is reported.
NU_MIN, NU_MAX = 0.0, 0.70
RHO_SH_MIN, RHO_SH_MAX = 0.002, 0.020

# a_sl = 1 where bar slip from the joint is possible, which is the normal
# condition for a cast-in-place frame.
BOND_SLIP_INDICATOR = 1.0

THETA_P_FLOOR = 0.005
THETA_PC_FLOOR = 0.010
THETA_PC_CAP = 0.10


def _clamp(value, low, high):
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Nominal P-M interaction
# ---------------------------------------------------------------------------

def column_pm_nominal(n_pts=120):
    """Nominal (unfactored) column P-M interaction surface.

    RC_Design_Check.build_column_PM_diagram returns phi-factored capacities
    because it exists to check designs. A hinge represents actual strength,
    not design strength, so the strength-reduction factor must not appear
    here. The sweep mirrors that function so the two stay comparable.

    Returns a list of (Pn_kip, Mn_kip_in), compression positive.
    """
    fc = sp.FC_COL_KSI
    fy = sp.FY_KSI
    es = sp.ES_KSI
    b = sp.B_COL
    h = sp.H_COL
    layers = _col_steel_layers()

    ecu = 0.003
    b1 = _beta1(fc)
    ag = b * h
    ast = sum(area for area, _ in layers)
    hc = h / 2.0

    p0 = 0.85 * fc * (ag - ast) + fy * ast
    diagram = [(0.80 * p0, 0.0)]

    for index in range(n_pts):
        c = 0.001 + (4.0 * h - 0.001) * index / (n_pts - 1)
        a = min(b1 * c, h)
        compression = 0.85 * fc * b * a
        axial = compression
        moment = compression * (hc - a / 2.0)

        for area, depth in layers:
            strain = ecu * (c - depth) / c
            stress = _clamp(es * strain, -fy, fy)
            net = stress - (0.85 * fc if depth <= a else 0.0)
            axial += net * area
            moment += net * area * (hc - depth)

        diagram.append((axial, abs(moment)))

    return sorted(diagram, key=lambda point: point[0])


def column_moment_at_axial(axial_kip, diagram=None):
    """Nominal moment capacity at a given axial load, by interpolation.

    Axial loads outside the surface are clamped to its ends: below the
    tension end the section is governed by flexure, and above the balance
    point the surface already falls to zero moment at pure compression.
    """
    diagram = diagram or column_pm_nominal()
    axials = [point[0] for point in diagram]
    moments = [point[1] for point in diagram]
    if axial_kip <= axials[0]:
        return moments[0]
    if axial_kip >= axials[-1]:
        return moments[-1]
    for index in range(1, len(diagram)):
        if axial_kip <= axials[index]:
            span = axials[index] - axials[index - 1]
            if span <= 0.0:
                return moments[index]
            weight = (axial_kip - axials[index - 1]) / span
            return moments[index - 1] + weight * (moments[index] - moments[index - 1])
    return moments[-1]


# ---------------------------------------------------------------------------
# Axial demand estimate
# ---------------------------------------------------------------------------

def column_gravity_axial(story_index, grid_i, grid_j):
    """Gravity axial load in a column, kips, compression positive.

    Hinge properties are fixed when the model is built, before any analysis
    has run, so the axial load is estimated from tributary gravity rather
    than read from a solved state. Every floor at or above this column's top
    contributes its tributary load, plus the self weight of the columns above.

    story_index is one-based: story 1 is the column between the base and the
    first elevated floor.
    """
    floors_above = max(0, sp.NUM_FLOOR - story_index + 1)
    floor_load = sp.node_gravity_load_kip(grid_i, grid_j) * floors_above
    self_weight = sp.col_self_weight_kip_per_in() * sp.STORY_H * floors_above
    return floor_load + self_weight


def column_grid_position(node_tag):
    """Recover (story_index, grid_i, grid_j) from a column's lower node."""
    x, y, z = ops.nodeCoord(node_tag)
    grid_i = int(round(x / sp.BAY_X)) if sp.BAY_X > 0 else 0
    grid_j = int(round(y / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
    floor_below = int(round(z / sp.STORY_H)) if sp.STORY_H > 0 else 0
    return floor_below + 1, grid_i, grid_j


def axial_load_ratio(axial_kip, member_type):
    """Normalized axial load nu = P / (Ag f'c)."""
    if member_type == "column":
        gross_area = sp.B_COL * sp.H_COL
        strength = sp.FC_COL_KSI
    else:
        gross_area = sp.B_BEAM * sp.H_BEAM
        strength = sp.FC_BEAM_KSI
    if gross_area <= 0.0 or strength <= 0.0:
        return 0.0
    return max(0.0, axial_kip / (gross_area * strength))


# ---------------------------------------------------------------------------
# Haselton et al. (2008) rotation capacities
# ---------------------------------------------------------------------------

def transverse_steel_ratio(member_type):
    """rho_sh = Ash / (b * s) for the member's confinement."""
    if member_type == "column":
        legs, bar_size, spacing, width = (
            sp.COL_STIRRUP_LEGS, sp.COL_STIRRUP_BAR_SIZE,
            sp.COL_STIRRUP_SPACING, sp.B_COL,
        )
    else:
        legs, bar_size, spacing, width = (
            sp.BEAM_STIRRUP_LEGS, sp.BEAM_STIRRUP_BAR_SIZE,
            sp.BEAM_STIRRUP_SPACING, sp.B_BEAM,
        )
    if spacing <= 0.0 or width <= 0.0:
        return RHO_SH_MIN
    return legs * sp.rebar_area(bar_size) / (width * spacing)


def longitudinal_steel_ratio(member_type):
    """rho = As / (b * d) using the tension-side longitudinal steel."""
    if member_type == "column":
        area = max(sp.COL_TOP_BARS, sp.COL_BOT_BARS) * sp.COL_BAR_AREA
        width, depth = sp.B_COL, sp.H_COL - sp.COVER
    else:
        area = max(sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS) * sp.BEAM_BAR_AREA
        width, depth = sp.B_BEAM, sp.H_BEAM - sp.COVER
    if width <= 0.0 or depth <= 0.0:
        return 0.0
    return area / (width * depth)


def _stability_index(member_type):
    """s_n = (s / d_b) * sqrt(fy_MPa / 100), the rebar buckling index."""
    if member_type == "column":
        spacing = sp.COL_STIRRUP_SPACING
        bar_diameter = sp.rebar_diameter(sp.COL_BAR_SIZE)
    else:
        spacing = sp.BEAM_STIRRUP_SPACING
        bar_diameter = sp.rebar_diameter(sp.BEAM_BAR_SIZE)
    if bar_diameter <= 0.0:
        return 1.0
    return (spacing / bar_diameter) * math.sqrt(sp.FY_KSI * KSI_TO_MPA / 100.0)


def haselton_theta_p(member_type, nu):
    """Plastic rotation capacity to the capping point.

    Haselton et al. (2008), PEER 2007/03, equation 4.3 (calibrated form):

        theta_p = 0.12 (1 + 0.55 a_sl) (0.16)^nu (0.02 + 40 rho_sh)^0.43
                  (0.54)^(0.01 f'c_MPa) (0.66)^(0.1 s_n) (2.27)^(10 rho)

    Beams are the nu = 0 case, where the (0.16)^nu term is unity.
    """
    nu = _clamp(nu, NU_MIN, NU_MAX)
    rho_sh = _clamp(transverse_steel_ratio(member_type), RHO_SH_MIN, RHO_SH_MAX)
    rho = longitudinal_steel_ratio(member_type)
    fc_mpa = (sp.FC_COL_KSI if member_type == "column" else sp.FC_BEAM_KSI) * KSI_TO_MPA
    s_n = _stability_index(member_type)

    theta_p = (
        0.12
        * (1.0 + 0.55 * BOND_SLIP_INDICATOR)
        * (0.16 ** nu)
        * ((0.02 + 40.0 * rho_sh) ** 0.43)
        * (0.54 ** (0.01 * fc_mpa))
        * (0.66 ** (0.1 * s_n))
        * (2.27 ** (10.0 * rho))
    )
    return max(THETA_P_FLOOR, theta_p)


def haselton_theta_pc(member_type, nu):
    """Post-capping rotation capacity.

    Haselton et al. (2008), equation 4.4:

        theta_pc = 0.76 (0.031)^nu (0.02 + 40 rho_sh)^1.02,  capped at 0.10

    The cap is part of the published expression: the fit is unreliable for
    very well confined members and would otherwise predict implausibly
    ductile post-peak behaviour.
    """
    nu = _clamp(nu, NU_MIN, NU_MAX)
    rho_sh = _clamp(transverse_steel_ratio(member_type), RHO_SH_MIN, RHO_SH_MAX)
    theta_pc = 0.76 * (0.031 ** nu) * ((0.02 + 40.0 * rho_sh) ** 1.02)
    return max(THETA_PC_FLOOR, min(THETA_PC_CAP, theta_pc))


def backbone_for_member(member_type, axial_kip=0.0, pm_diagram=None):
    """Full per-member IMK backbone.

    Returns yield moment, plastic and post-capping rotations, ultimate
    rotation, and the axial ratio that produced them. The axial ratio is
    carried through so it can be exported as a node/element feature: it is
    the single number that explains why two otherwise identical columns
    behave differently.
    """
    use_calibration = getattr(sp, "IMK_USE_CALIBRATED_BACKBONE", True)
    nu = axial_load_ratio(axial_kip, member_type)

    if not use_calibration:
        return {
            "axial_kip": axial_kip,
            "axial_ratio": nu,
            "theta_p": sp.IMK_THETA_P_POS,
            "theta_pc": sp.IMK_THETA_PC_POS,
            "theta_u": sp.IMK_THETA_U_POS,
            "source": "fixed",
        }

    theta_p = haselton_theta_p(member_type, nu)
    theta_pc = haselton_theta_pc(member_type, nu)
    return {
        "axial_kip": axial_kip,
        "axial_ratio": nu,
        "theta_p": theta_p,
        "theta_pc": theta_pc,
        "theta_u": theta_p + theta_pc + sp.IMK_BEAM_THETA_Y,
        "source": "haselton_2008",
    }
