"""
Gravity_Loads.py
Apply floor gravity loads as tributary-area-based uniform beam loads.

Each beam receives a downward uniform load (kip/in) equal to the floor area
load intensity (kip/in²) multiplied by its tributary width:
  - Edge beams   → half the adjacent bay width
  - Interior beams → full bay width (half from each side)

This correctly accounts for the tributary area of each beam position, unlike
the previous approach of applying equal point loads to every node.
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

    X-beams span in the X-direction; tributary width is in the Y-direction
    and depends on the beam's j-position.

    Y-beams span in the Y-direction; tributary width is in the X-direction
    and depends on the beam's i-position.
    """
    ele_tag = _column_count() + 1

    # ── X beams ──────────────────────────────────────────────────────────────
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            wz = sp.beam_gravity_wz_kip_per_in("x", j)
            for i in range(sp.NUM_BAY_X):
                ops.eleLoad("-ele", ele_tag, "-type", "-beamUniform", 0.0, wz, 0.0)
                ele_tag += 1

    # ── Y beams ──────────────────────────────────────────────────────────────
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                wz = sp.beam_gravity_wz_kip_per_in("y", i)
                ops.eleLoad("-ele", ele_tag, "-type", "-beamUniform", 0.0, wz, 0.0)
                ele_tag += 1


def apply_gravity_loads():
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    if sp.GRAVITY_LOAD_MODEL == "nodal":
        _apply_nodal_gravity_loads()
        return

    if sp.GRAVITY_LOAD_MODEL == "beam_uniform":
        _apply_beam_uniform_gravity_loads()
        return

    raise ValueError(f"Unknown GRAVITY_LOAD_MODEL: {sp.GRAVITY_LOAD_MODEL}")
