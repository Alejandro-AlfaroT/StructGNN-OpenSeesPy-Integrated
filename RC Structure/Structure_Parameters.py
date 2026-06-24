# ==================================================
# Structure_Parameters.py
# ==================================================
# Global parameters for RC 3D frame model
#
# Unit system:
#   Force  = kip
#   Length = inch
#   Time   = second
#   Stress = ksi
# ==================================================

import math

# Geometry
NUM_BAY_X = 3
NUM_BAY_Y = 3
NUM_FLOOR = 8

BAY_X = 120.0
BAY_Y = 120.0
STORY_H = 120.0

# Material strengths
FC_COL_KSI = 5.0
FC_BEAM_KSI = 4.0

FY_KSI = 60.0
ES_KSI = 29000.0
STEEL_B = 0.01

# Section dimensions
B_COL = 18.0
H_COL = 18.0

B_BEAM = 12.0
H_BEAM = 18.0

COVER = 1.5

# Rebar database
REBAR = {
    3: {"area": 0.11, "diameter": 0.375},
    4: {"area": 0.20, "diameter": 0.500},
    5: {"area": 0.31, "diameter": 0.625},
    6: {"area": 0.44, "diameter": 0.750},
    7: {"area": 0.60, "diameter": 0.875},
    8: {"area": 0.79, "diameter": 1.000},
    9: {"area": 1.00, "diameter": 1.128},
    10: {"area": 1.27, "diameter": 1.270},
    11: {"area": 1.56, "diameter": 1.410},
    14: {"area": 2.25, "diameter": 1.693},
    18: {"area": 4.00, "diameter": 2.257},
}


def rebar_area(bar_size):
    return REBAR[bar_size]["area"]


def rebar_diameter(bar_size):
    return REBAR[bar_size]["diameter"]


# Reinforcement layouts
COL_BAR_SIZE = 8
COL_TOP_BARS = 4
COL_BOT_BARS = 4
COL_SIDE_BARS = 2
COL_BAR_AREA = rebar_area(COL_BAR_SIZE)

BEAM_BAR_SIZE = 6
BEAM_TOP_BARS = 2
BEAM_BOT_BARS = 2
BEAM_SIDE_BARS = 0
BEAM_BAR_AREA = rebar_area(BEAM_BAR_SIZE)

# Design DCR target band (iterative steel redesign)
DESIGN_DCR_MIN = 0.60   # lower bound — avoid over-design
DESIGN_DCR_MAX = 0.95   # upper bound — demand must be met

# Loading / mass
G = 386.4

# Floor gravity load — specified as a uniform area load (ksf).
# Dead load includes slab self-weight + superimposed dead (finishes, MEP, etc.)
# Live load is the design occupancy load.
# Both are applied in full for gravity analysis (unfactored service loads).
# The combined value is used for seismic mass (ASCE 7 §12.7.2 effective weight).
FLOOR_DEAD_LOAD_KSF = 0.15   # kip/ft²
FLOOR_LIVE_LOAD_KSF = 0.05   # kip/ft²

FX_FLOOR = 10.0

# Gravity load application.
# Options:
#   "nodal"        = tributary-area nodal loads at floor grid nodes
#   "beam_uniform" = tributary uniform loads on beams
#
# The nodal option is the baseline for the IMK hinge model because applying
# uniform element loads directly to the elastic spine can create a large
# gravity-only sidesway before pushover starts.
GRAVITY_LOAD_MODEL = "nodal"

# Material tags
COVER_COL_TAG = 1
CORE_COL_TAG = 2
COVER_BEAM_TAG = 3
CORE_BEAM_TAG = 4
STEEL_TAG = 5

# Section tags
COL_SEC_TAG = 101
BEAM_SEC_TAG = 102

# Beam integration tags
COL_INTEG_TAG = 1
BEAM_INTEG_TAG = 2
NUM_INT_PTS = 5

# Geometric transformation tags
COL_TRANSF_TAG = 1
BEAM_X_TRANSF_TAG = 2
BEAM_Y_TRANSF_TAG = 3

# Element formulation
# Options:
#   "fiber" = distributed forceBeamColumn fiber sections
#   "imk"   = elastic interior members with IMKPeakOriented rotational hinges
ELEMENT_FORMULATION = "imk"
IMK_APPLY_TO_COLUMNS = False
IMK_APPLY_TO_BEAMS = True

# IMK lumped hinge tags
IMK_HINGE_NODE_TAG_BASE = 2000000
IMK_MATERIAL_TAG_BASE = 3000000
IMK_HINGE_ELEMENT_TAG_BASE = 4000000

# IMK hinge calibration defaults.
# These are intentionally centralized because IMK behavior should eventually be
# calibrated from design/code equations or experiments rather than hidden in the
# element builder.
IMK_MATERIAL_TYPE = "IMKBilin"
IMK_HINGE_STIFFNESS_FACTOR = 20.0
IMK_BEAM_THETA_Y = 0.005
IMK_COLUMN_THETA_Y = 0.004
IMK_POST_YIELD_RATIO_POS = 0.02
IMK_POST_YIELD_RATIO_NEG = 0.02
IMK_LAMBDA_S = 10.0
IMK_LAMBDA_C = 10.0
IMK_LAMBDA_A = 10.0
IMK_LAMBDA_K = 10.0
IMK_C_S = 1.0
IMK_C_C = 1.0
IMK_C_A = 1.0
IMK_C_K = 1.0
IMK_THETA_P_POS = 0.020
IMK_THETA_P_NEG = 0.020
IMK_THETA_PC_POS = 0.060
IMK_THETA_PC_NEG = 0.060
IMK_RES_POS = 0.20
IMK_RES_NEG = 0.20
IMK_THETA_U_POS = 0.120
IMK_THETA_U_NEG = 0.120
IMK_D_POS = 1.0
IMK_D_NEG = 1.0

# Fiber section mesh settings
CORE_PATCH_NY = 12
CORE_PATCH_NZ = 12

COVER_PATCH_N_LONG = 12
COVER_PATCH_N_SHORT = 2

# Analysis settings
GRAVITY_TOL = 1.0e-5
GRAVITY_MAX_ITER = 100
GRAVITY_STEPS = 20

PUSHOVER_TOL = 1.0e-5
PUSHOVER_MAX_ITER = 50
PUSHOVER_FALLBACK_TOL = 5.0e-5
PUSHOVER_FALLBACK_MAX_ITER = 100

PUSHOVER_STEPS = 1200
PUSHOVER_DU = 0.05
PUSHOVER_DEBUG_EVERY = 10
PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO = 0.05
PUSHOVER_REDESIGN_MAX_ATTEMPTS = 3
PUSHOVER_STOP_ON_LOAD_REVERSAL = True
PUSHOVER_LOAD_REVERSAL_LIMIT = 0.0
PUSHOVER_LOAD_REVERSAL_MIN_PEAK_FACTOR = 0.02

NUM_MODES = NUM_FLOOR + 2
EIGENVALUE_TOL = 1.0e-8

PENALTY_ALPHA_SP = 1.0e9
PENALTY_ALPHA_MP = 1.0e9


def concrete_ec_ksi(fc_ksi):
    fc_psi = fc_ksi * 1000.0
    return 57000.0 * math.sqrt(fc_psi) / 1000.0


def concrete_shear_modulus_ksi(ec_ksi):
    return 0.4 * ec_ksi


def rect_area(b, h):
    return b * h


def rect_iy(b, h):
    return h * b**3 / 12.0


def rect_iz(b, h):
    return b * h**3 / 12.0


def approx_rect_j(b, h):
    return rect_iy(b, h) + rect_iz(b, h)


def rc_nominal_moment_ksi(fc_ksi, width_in, depth_in, steel_area_in2):
    d = depth_in - COVER
    a = steel_area_in2 * FY_KSI / (0.85 * fc_ksi * width_in)
    a = min(a, 0.85 * depth_in)
    return steel_area_in2 * FY_KSI * (d - 0.5 * a)


def beam_nominal_moment_y():
    steel_area = max(BEAM_TOP_BARS, BEAM_BOT_BARS) * BEAM_BAR_AREA
    return rc_nominal_moment_ksi(FC_BEAM_KSI, B_BEAM, H_BEAM, steel_area)


def beam_nominal_moment_z():
    steel_area = max(BEAM_TOP_BARS, BEAM_BOT_BARS) * BEAM_BAR_AREA
    return rc_nominal_moment_ksi(FC_BEAM_KSI, H_BEAM, B_BEAM, steel_area)


def column_nominal_moment_y():
    steel_area = max(COL_TOP_BARS, COL_BOT_BARS) * COL_BAR_AREA
    return rc_nominal_moment_ksi(FC_COL_KSI, B_COL, H_COL, steel_area)


def column_nominal_moment_z():
    steel_area = max(COL_TOP_BARS, COL_BOT_BARS) * COL_BAR_AREA
    return rc_nominal_moment_ksi(FC_COL_KSI, H_COL, B_COL, steel_area)


def floor_load_ksi():
    """Total gravity floor load converted to kip/in²."""
    return (FLOOR_DEAD_LOAD_KSF + FLOOR_LIVE_LOAD_KSF) / 144.0


def node_tributary_area_in2(i, j):
    """
    Tributary floor area (in²) for a structural node at grid position (i, j).
    Corner nodes receive a quarter bay, edge nodes a half bay, interior nodes
    a full bay in each direction.
    """
    tx = (BAY_X / 2.0) if (i == 0 or i == NUM_BAY_X) else BAY_X
    ty = (BAY_Y / 2.0) if (j == 0 or j == NUM_BAY_Y) else BAY_Y
    return tx * ty


def node_gravity_load_kip(i, j):
    """Tributary gravity load (kips, downward) at grid node (i, j)."""
    return floor_load_ksi() * node_tributary_area_in2(i, j)


def node_seismic_mass(i, j):
    """Translational seismic mass (kip·s²/in) at grid node (i, j)."""
    return node_gravity_load_kip(i, j) / G


def total_floor_gravity_load():
    """Total gravity load per floor (kips)."""
    return floor_load_ksi() * (BAY_X * NUM_BAY_X) * (BAY_Y * NUM_BAY_Y)


def beam_tributary_width_in(beam_axis, perp_index):
    """
    Tributary width (in) for a beam.

    beam_axis   : 'x' or 'y'
    perp_index  : position index perpendicular to the beam span.
                  For X-beams this is the j-index (Y-direction).
                  For Y-beams this is the i-index (X-direction).
    """
    if beam_axis == "x":
        n_max  = NUM_BAY_Y
        bay_perp = BAY_Y
    else:
        n_max  = NUM_BAY_X
        bay_perp = BAY_X
    return (bay_perp / 2.0) if (perp_index == 0 or perp_index == n_max) else bay_perp


def beam_gravity_wz_kip_per_in(beam_axis, perp_index):
    """Uniform downward load (kip/in, negative) on a beam at perp_index."""
    trib = beam_tributary_width_in(beam_axis, perp_index)
    return -floor_load_ksi() * trib


def pushover_target_disp():
    return PUSHOVER_STEPS * PUSHOVER_DU
