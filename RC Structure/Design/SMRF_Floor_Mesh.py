"""Bounded rectangular floor grids with explicit coordinate provenance.

Explicit offsets repeat within each bay. They are a numerical mesh request,
not a claim of convergence or engineering applicability. Uniform defaults
retain their existing limit; the larger explicit-grid budget is opt-in.
"""
from __future__ import annotations

import hashlib
import json
import math

MAX_EXPLICIT_SHELLS = 45000


def _offsets(values, length, name):
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise ValueError(f"{name} must contain at least three bay coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain finite numeric coordinates")
    values = [float(v) for v in values]
    if values[0] != 0.0 or values[-1] != length:
        raise ValueError(f"{name} must start at zero and end at its bay length")
    if any(b - a <= 1e-8 for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing, with cells wider than 1e-8 in")
    return values


def floor_mesh(nx, ny, lx, ly, mesh_per_bay, *, mesh_spec=None, uniform_shell_limit=8192):
    if mesh_spec is None:
        count = mesh_per_bay
        if (isinstance(count, bool) or not isinstance(count, (int, float))
                or not math.isfinite(count) or int(count) != count or not 2 <= count <= 24):
            raise ValueError("Uniform floor mesh requires 2-24 integer subdivisions per bay")
        count = int(count)
        ox = [i * lx / count for i in range(count + 1)]
        oy = [i * ly / count for i in range(count + 1)]
        budget, kind, solver = uniform_shell_limit, "uniform", "BandGeneral"
    else:
        if not isinstance(mesh_spec, dict) or set(mesh_spec) != {"x_offsets_in", "y_offsets_in", "max_shells"}:
            raise ValueError("Explicit mesh requires x_offsets_in, y_offsets_in and max_shells only")
        budget = mesh_spec["max_shells"]
        if isinstance(budget, bool) or not isinstance(budget, int) or not 4 <= budget <= MAX_EXPLICIT_SHELLS:
            raise ValueError(f"Explicit mesh max_shells must be an integer from 4 to {MAX_EXPLICIT_SHELLS}")
        ox = _offsets(mesh_spec["x_offsets_in"], lx, "x_offsets_in")
        oy = _offsets(mesh_spec["y_offsets_in"], ly, "y_offsets_in")
        kind, solver = "explicit_rectangular", "UmfPack"
    mx, my = len(ox) - 1, len(oy) - 1
    count = nx * ny * mx * my
    if count > budget:
        raise ValueError(f"Requested floor mesh has {count} shells, exceeding explicit budget {budget}")
    xs = [i * lx + offset for i in range(nx) for offset in ox[:-1]] + [nx * lx]
    ys = [j * ly + offset for j in range(ny) for offset in oy[:-1]] + [ny * ly]
    dx = [b - a for a, b in zip(xs, xs[1:])]
    dy = [b - a for a, b in zip(ys, ys[1:])]
    coordinates = {"x_coordinates_in": xs, "y_coordinates_in": ys}
    digest = hashlib.sha256(json.dumps(coordinates, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return {"kind": kind, **coordinates, "x_offsets_in": ox, "y_offsets_in": oy,
            "coordinate_sha256": digest, "shell_count": count, "node_count": len(xs) * len(ys),
            "subdivisions_per_bay": mx if mx == my else None,
            "subdivisions_x_per_bay": mx, "subdivisions_y_per_bay": my,
            "dx_in": lx / mx if mesh_spec is None else None,
            "dy_in": ly / my if mesh_spec is None else None,
            "minimum_cell_width_in": min(*dx, *dy),
            "maximum_cell_aspect_ratio": max(max(dx) / min(dy), max(dy) / min(dx)),
            "shell_budget": budget, "solver": solver}


def nested_refinement(coarse, fine):
    """Require retained nodes and a genuinely finer grid in at least one axis."""
    increased = False
    for key in ("x_coordinates_in", "y_coordinates_in"):
        a, b = coarse[key], fine[key]
        if not set(a).issubset(b):
            raise ValueError("Refinement meshes must retain every previous coordinate")
        increased |= len(b) > len(a)
    if not increased:
        raise ValueError("Refinement must add coordinates; repeated grids are not refinement")
