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

# Transverse reinforcement used in shear design checks.
COL_STIRRUP_BAR_SIZE = 4
COL_STIRRUP_LEGS = 2
COL_STIRRUP_SPACING = 6.0

BEAM_STIRRUP_BAR_SIZE = 4
BEAM_STIRRUP_LEGS = 2
BEAM_STIRRUP_SPACING = 6.0

STIRRUP_MIN_SPACING = 3.0
STIRRUP_SPACING_STEP = 1.0

# Design DCR target band (iterative steel redesign)
DESIGN_DCR_MIN = 0.60   # lower bound — avoid over-design
DESIGN_DCR_MAX = 0.95   # upper bound — demand must be met

# Loading / mass
G = 386.4

# Concrete unit weight (normal-weight concrete)
CONCRETE_UNIT_WEIGHT_KCF = 0.150              # kip/ft³
CONCRETE_UNIT_WEIGHT_KCI = CONCRETE_UNIT_WEIGHT_KCF / 1728.0  # kip/in³

# Floor gravity load — specified as a uniform area load (ksf).
# Dead load should cover superimposed dead only (finishes, MEP, partitions, slab).
# Structural self-weight of beams and columns is applied separately as element
# loads in Gravity_Loads.py — do not double-count it here.
# Live load is the design occupancy load (50 psf = office, ASCE 7 Table 4.3-1).
# Both are applied in full for gravity analysis (unfactored service loads).
# The combined value is used for seismic mass (ASCE 7 §12.7.2 effective weight).
FLOOR_DEAD_LOAD_KSF = 0.10   # kip/ft²  (superimposed dead: slab + finishes + MEP)
FLOOR_LIVE_LOAD_KSF = 0.05   # kip/ft²  (office occupancy)

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

# IMK rotational spring stiffness calibration.
#   "yield_rotation"          = Ke = My / theta_y_input
#   "member_stiffness_factor" = Ke = stiffness_factor * 6EI/L
#   "max"                     = legacy rule: max(My/theta_y_input, stiffness_factor*6EI/L)
#
# Use "member_stiffness_factor" for concentrated-plasticity springs so the
# elastic beam-column element carries member flexibility and the zeroLength
# hinge remains stiff until its IMKBilin backbone reaches yield strength.
IMK_HINGE_STIFFNESS_MODE = "member_stiffness_factor"
IMK_HINGE_STIFFNESS_FACTOR = 20.0
IMK_BEAM_THETA_Y = 0.005
IMK_COLUMN_THETA_Y = 0.004
IMK_LAMBDA_S = 10.0
IMK_LAMBDA_C = 10.0
IMK_LAMBDA_K = 10.0
IMK_C_S = 1.0
IMK_C_C = 1.0
IMK_C_K = 1.0
IMK_THETA_P_POS = 0.020
IMK_THETA_P_NEG = 0.020
IMK_THETA_PC_POS = 0.060
IMK_THETA_PC_NEG = 0.060
IMK_RES_POS = 0.20
IMK_RES_NEG = 0.20

# Current OpenSees IMKBilin uses maximum/residual strength ratios, not
# post-yield hardening ratios, in the material command.
IMK_FMAXFY_POS = 1.10
IMK_FMAXFY_NEG = 1.10
IMK_FRESFY_POS = IMK_RES_POS
IMK_FRESFY_NEG = IMK_RES_NEG
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
PUSHOVER_TRACK_DESIGN_ENVELOPE = True
PUSHOVER_DESIGN_ENVELOPE_EVERY = 5
PUSHOVER_DESIGN_ENVELOPE_EVENT_MIN_STEP_GAP = 5
PUSHOVER_DESIGN_ENVELOPE_PEAK_SHEAR_REL_STEP = 0.05
PUSHOVER_DESIGN_ENVELOPE_DRIFT_JUMP_REL_STEP = 0.10
PUSHOVER_DCR_CHECK_ROOF_DRIFT_RATIO = 0.02
PUSHOVER_USE_TARGET_DCR_FOR_REDESIGN = True
PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO = 0.05
PUSHOVER_REDESIGN_MAX_ATTEMPTS = 3
PUSHOVER_ENVELOPE_REDESIGN_MAX_ATTEMPTS = 6
PUSHOVER_ENVELOPE_STOP_ON_STALLED_DCR = True
PUSHOVER_ENVELOPE_STALL_MIN_RERUNS = 2
PUSHOVER_ENVELOPE_STALL_DCR_TOL = 0.01
PUSHOVER_STOP_ON_LOAD_REVERSAL = True
PUSHOVER_LOAD_REVERSAL_LIMIT = 0.0
PUSHOVER_LOAD_REVERSAL_MIN_PEAK_FACTOR = 0.02

NUM_MODES = NUM_FLOOR + 2
EIGENVALUE_TOL = 1.0e-8

# ASCE 7 approximate fundamental period sanity check.
# US customary form: Ta = Ct * h_n^x, with h_n in feet.
# Default constants are for concrete moment-resisting frames.
ASCE_PERIOD_SYSTEM = "concrete_moment_frame"
ASCE_PERIOD_CT = 0.016
ASCE_PERIOD_EXPONENT = 0.90
ASCE_PERIOD_WARN_RATIO_LOW = 0.50
ASCE_PERIOD_WARN_RATIO_HIGH = 2.00


def building_height_ft():
    """Total structural height in feet."""
    return NUM_FLOOR * STORY_H / 12.0


def asce_approx_fundamental_period_sec():
    """ASCE approximate fundamental period Ta = Ct * h_n^x."""
    return ASCE_PERIOD_CT * building_height_ft() ** ASCE_PERIOD_EXPONENT


def asce_fundamental_period_check(model_period_sec=None, modal_source=None):
    """
    Return a compact period sanity check against ASCE approximate Ta.

    This is intended as an early flag, not a design acceptance criterion.
    """
    ta = asce_approx_fundamental_period_sec()
    check = {
        "equation": "Ta = Ct * h_n^x",
        "system": ASCE_PERIOD_SYSTEM,
        "ct": ASCE_PERIOD_CT,
        "exponent": ASCE_PERIOD_EXPONENT,
        "height_ft": building_height_ft(),
        "asce_ta_sec": ta,
        "modal_source": modal_source,
        "model_period_sec": model_period_sec,
        "period_ratio_to_asce_ta": None,
        "warn_ratio_low": ASCE_PERIOD_WARN_RATIO_LOW,
        "warn_ratio_high": ASCE_PERIOD_WARN_RATIO_HIGH,
        "passes_reasonableness_check": None,
        "warning": None,
    }

    if model_period_sec is None or ta <= 0.0:
        return check

    ratio = model_period_sec / ta
    check["period_ratio_to_asce_ta"] = ratio
    check["passes_reasonableness_check"] = (
        ASCE_PERIOD_WARN_RATIO_LOW <= ratio <= ASCE_PERIOD_WARN_RATIO_HIGH
    )

    if ratio > ASCE_PERIOD_WARN_RATIO_HIGH:
        check["warning"] = (
            "Model period is much longer than ASCE approximate Ta; "
            "check mass, stiffness, hinges, gravity state, and constraints."
        )
    elif ratio < ASCE_PERIOD_WARN_RATIO_LOW:
        check["warning"] = (
            "Model period is much shorter than ASCE approximate Ta; "
            "check mass, stiffness, rigid constraints, and units."
        )

    return check


# Primary post-gravity analysis mode.
# Options:
#   "ntha"     = nonlinear time-history analysis from Ground_Motions manifest
#   "pushover" = static nonlinear pushover with redesign/envelope checks
ANALYSIS_METHOD = "ntha"

# NTHA / ground-motion settings used by Main.py when ANALYSIS_METHOD = "ntha".
NTHA_SET_NAME = "peer_mle_all"
NTHA_SPLIT = None
NTHA_RESULT_ID = 1
NTHA_RECORD_ID_X = None
NTHA_RECORD_ID_Y = None
NTHA_X_ONLY = False
NTHA_SCALE_FACTOR = None
NTHA_DAMPING_RATIO = 0.05
NTHA_RAYLEIGH_MODE_I = 0
NTHA_RAYLEIGH_MODE_J = 2
NTHA_DT_FACTOR = 1.0
NTHA_PROGRESS_EVERY = 100
NTHA_PRINT_CATALOG_SUMMARY = True

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


def col_self_weight_kip_per_in():
    """Column self-weight per unit length (kip/in), downward."""
    return CONCRETE_UNIT_WEIGHT_KCI * B_COL * H_COL


def beam_self_weight_kip_per_in():
    """Beam self-weight per unit length (kip/in), downward."""
    return CONCRETE_UNIT_WEIGHT_KCI * B_BEAM * H_BEAM


def total_structural_self_weight_per_floor():
    """
    Total self-weight of all column and beam elements in one floor (kips).

    Columns  : NUM_FLOOR stories × (NUM_BAY_X+1) × (NUM_BAY_Y+1) columns,
               each STORY_H tall.
    Beams    : X-beams + Y-beams, each spanning one bay (BAY_X or BAY_Y).
    """
    n_col    = (NUM_BAY_X + 1) * (NUM_BAY_Y + 1)
    n_beam_x = NUM_BAY_X * (NUM_BAY_Y + 1)
    n_beam_y = (NUM_BAY_X + 1) * NUM_BAY_Y
    w_col_floor  = col_self_weight_kip_per_in()  * STORY_H * n_col
    w_beam_floor = beam_self_weight_kip_per_in() * BAY_X   * (n_beam_x + n_beam_y)
    return w_col_floor + w_beam_floor
