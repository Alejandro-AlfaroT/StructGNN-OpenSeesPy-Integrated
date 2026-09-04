"""
Gravity_Loads.py
Apply floor gravity loads as tributary-area-based uniform beam loads.

Each beam receives a downward uniform load (kip/in) equal to the floor area
load intensity (kip/in²) multiplied by its tributary width:
  - Edge beams   → half the adjacent bay width
  - Interior beams → full bay width (half from each side)

"""

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.nodes import node_tag


def _column_count():
    return sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)


def _apply_nodal_gravity_loads():
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                ops.load(
                    node_tag(k, i, j),
                    0.0,
                    0.0,
                    -sp.node_gravity_load_kip(i, j),
                    0.0,
                    0.0,
                    0.0,
                )


def _apply_beam_uniform_gravity_loads():
    """
    Apply tributary-area-scaled uniform loads to all floor beams.

    The floor load is split equally between X-beams and Y-beams so the total
    applied load matches the nodal approach (180 kips/floor for the default
    parameters).  Each beam receives half the full-tributary intensity:

        wz = 0.5 × floor_load_ksi × tributary_width_perp

    Without the 0.5 factor, both beam directions would each apply the full
    floor load independently, doubling the total to 360 kips/floor.

    X-beams span in the X-direction; tributary width is in the Y-direction
    and depends on the beam's j-position (edge vs. interior).

    Y-beams span in the Y-direction; tributary width is in the X-direction
    and depends on the beam's i-position (edge vs. interior).
    """
    ele_tag = _column_count() + 1

    # ── X beams (half-tributary in the Y-direction) ───────────────────────────
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            wz = sp.beam_gravity_wz_kip_per_in("x", j) / 2.0
            for i in range(sp.NUM_BAY_X):
                ops.eleLoad("-ele", ele_tag, "-type", "-beamUniform", 0.0, wz, 0.0)
                ele_tag += 1

    # ── Y beams (half-tributary in the X-direction) ───────────────────────────
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                wz = sp.beam_gravity_wz_kip_per_in("y", i) / 2.0
                ops.eleLoad("-ele", ele_tag, "-type", "-beamUniform", 0.0, wz, 0.0)
                ele_tag += 1


def _apply_element_self_weight():
    """
    Apply distributed self-weight to all physical column and beam elements.

    Column self-weight acts along the column axis (global -Z = local -x for a
    vertical element), applied as the axial Wx component of beamUniform.

    Beam self-weight acts downward (global -Z = local -z for a horizontal
    element), applied as the Wz component — the same convention used by
    _apply_beam_uniform_gravity_loads().

    IMK hinge zero-length elements (tags >= IMK_HINGE_ELEMENT_TAG_BASE) are
    skipped; they have zero length so their self-weight is zero and they do
    not support distributed element loads.
    """
    n_col    = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    n_beam_x = sp.NUM_FLOOR * sp.NUM_BAY_X       * (sp.NUM_BAY_Y + 1)
    n_beam_y = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * sp.NUM_BAY_Y

    w_col  = sp.col_self_weight_kip_per_in()   # kip/in, magnitude
    w_beam = sp.beam_self_weight_kip_per_in()  # kip/in, magnitude

    # Columns — self-weight along local -x (axial, downward for vertical column)
    for tag in range(1, n_col + 1):
        ops.eleLoad("-ele", tag, "-type", "-beamUniform", 0.0, 0.0, -w_col)

    # Beams — self-weight along local -z (transverse, downward for horizontal beam)
    for tag in range(n_col + 1, n_col + n_beam_x + n_beam_y + 1):
        ops.eleLoad("-ele", tag, "-type", "-beamUniform", 0.0, -w_beam, 0.0)


def apply_gravity_loads():
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    if sp.GRAVITY_LOAD_MODEL == "nodal":
        _apply_nodal_gravity_loads()
    elif sp.GRAVITY_LOAD_MODEL == "beam_uniform":
        _apply_beam_uniform_gravity_loads()
    else:
        raise ValueError(f"Unknown GRAVITY_LOAD_MODEL: {sp.GRAVITY_LOAD_MODEL}")

    _apply_element_self_weight()
