"""
Design/Section_Design.py
========================

Discrete section-dimension and concrete-strength ladders for the iterative
design loop, plus the heuristics that pick a rung from a demand-capacity
ratio.

Redesign.py resizes longitudinal bars and stirrup spacing. That is only the
last ~20% of the design space: once a section is at its maximum practical
steel ratio, the only way to reduce DCR further is a bigger section or
stronger concrete. Without this module the loop saturates and every geometry
converges on whatever section Structure_Parameters happened to start with.

Ladder ordering
---------------
Rungs are ordered by increasing capacity. Concrete strength is stepped before
dimensions at a given size because it is the cheaper way to buy capacity, and
because deeper members attract more seismic demand through added mass and
stiffness. Only when the strength options at a size are exhausted does the
section grow.

Unit system: kip, inch, ksi.
"""

from __future__ import annotations

import math


# ACI 318-19 18.7.2.1: special moment frame columns, least dimension >= 12 in
# and least/perpendicular dimension ratio >= 0.4. Square columns satisfy the
# ratio automatically and keep biaxial capacity symmetric.
COLUMN_SIZES_IN = (14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0)

# ACI 318-19 18.6.2.1: special moment frame beams, width >= max(0.3h, 10 in).
BEAM_DEPTHS_IN = (16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0)

CONCRETE_STRENGTHS_KSI = (4.0, 5.0, 6.0, 8.0)

COLUMN_MIN_DIMENSION_IN = 12.0
BEAM_MIN_WIDTH_IN = 10.0
BEAM_MIN_WIDTH_RATIO = 0.30

# Proportioning limits that keep generated frames buildable.
#
# The deep limit comes from ACI 318-19 18.6.2.1(a), clear span >= 4d. With
# cover that is roughly h <= span/5. The shallow limit is a serviceability
# floor rather than a code requirement.
BEAM_MIN_SPAN_DEPTH_RATIO = 5.0    # h <= span / 5   (ACI 18.6.2.1 ln >= 4d)
BEAM_MAX_SPAN_DEPTH_RATIO = 20.0   # h >= span / 20  (deflection control)
BEAM_MAX_DEPTH_FRACTION_OF_STORY = 0.40


def beam_width_for_depth(depth_in):
    """Beam width paired with a depth: about h/2, snapped to 2 in, ACI-legal."""
    raw = max(depth_in / 2.0, BEAM_MIN_WIDTH_RATIO * depth_in, BEAM_MIN_WIDTH_IN)
    snapped = 2.0 * math.ceil(raw / 2.0)
    return max(snapped, BEAM_MIN_WIDTH_IN)


def column_ladder():
    """Ordered (b, h, fc) rungs for columns, increasing capacity."""
    return [
        (size, size, fc)
        for size in COLUMN_SIZES_IN
        for fc in CONCRETE_STRENGTHS_KSI
    ]


def beam_ladder(span_in=None, story_height_in=None):
    """Ordered (b, h, fc) rungs for beams, filtered by span proportioning."""
    rungs = []
    for depth in BEAM_DEPTHS_IN:
        if span_in is not None:
            if depth > span_in / BEAM_MIN_SPAN_DEPTH_RATIO:
                continue
            if depth < span_in / BEAM_MAX_SPAN_DEPTH_RATIO:
                continue
        if story_height_in is not None:
            if depth > BEAM_MAX_DEPTH_FRACTION_OF_STORY * story_height_in:
                continue
        for fc in CONCRETE_STRENGTHS_KSI:
            rungs.append((beam_width_for_depth(depth), depth, fc))

    if not rungs:
        # Proportioning excluded every listed depth. Rather than silently
        # returning the unfiltered ladder -- which would let the design loop
        # pick a section the geometry cannot accept -- fall back to the single
        # closest legal depth and let the caller see that it was clamped.
        target = (span_in / 10.0) if span_in else BEAM_DEPTHS_IN[0]
        depth = min(BEAM_DEPTHS_IN, key=lambda value: abs(value - target))
        if story_height_in is not None:
            limit = BEAM_MAX_DEPTH_FRACTION_OF_STORY * story_height_in
            depth = min(depth, max(BEAM_DEPTHS_IN[0], limit))
        rungs = [(beam_width_for_depth(depth), depth, fc) for fc in CONCRETE_STRENGTHS_KSI]
    return rungs


def nearest_rung_index(ladder, b_in, h_in, fc_ksi):
    """Index of the rung closest to a given section, for resuming a search."""
    def distance(rung):
        rb, rh, rfc = rung
        return (rb - b_in) ** 2 + (rh - h_in) ** 2 + 4.0 * (rfc - fc_ksi) ** 2

    return min(range(len(ladder)), key=lambda i: distance(ladder[i]))


def flexural_capacity_proxy(rung):
    """Relative flexural capacity of a rung: b*h^2*sqrt(fc).

    Used only to order and to size a jump. The real capacity comes from the
    ACI checks in RC_Design_Check.py; this just avoids walking the ladder one
    rung at a time when the demand is far from the target.
    """
    b, h, fc = rung
    return b * h * h * math.sqrt(fc)


def suggest_rung_index(ladder, current_index, governing_dcr, target_dcr):
    """Jump to the rung whose capacity proxy should land DCR near the target.

    Returns the current index when the demand-capacity ratio is already at the
    target, so a converged design does not oscillate.
    """
    if not ladder:
        raise ValueError("Section ladder is empty.")
    current_index = max(0, min(current_index, len(ladder) - 1))
    if governing_dcr <= 0.0 or target_dcr <= 0.0:
        return current_index

    required = flexural_capacity_proxy(ladder[current_index]) * (governing_dcr / target_dcr)
    feasible = [i for i, rung in enumerate(ladder) if flexural_capacity_proxy(rung) >= required]
    if not feasible:
        return len(ladder) - 1
    return min(feasible)


def rung_summary(rung, member_type):
    b, h, fc = rung
    return {
        "member_type": member_type,
        "b_in": b,
        "h_in": h,
        "fc_ksi": fc,
        "capacity_proxy": flexural_capacity_proxy(rung),
    }


def validate_rung(rung, member_type, span_in=None, story_height_in=None):
    """Raise if a rung violates the ACI dimensional limits it claims to honor."""
    b, h, fc = rung
    if member_type == "column":
        if min(b, h) < COLUMN_MIN_DIMENSION_IN:
            raise ValueError(f"Column least dimension {min(b, h)} in < 12 in (ACI 18.7.2.1).")
        if min(b, h) / max(b, h) < 0.4:
            raise ValueError("Column dimension ratio < 0.4 (ACI 18.7.2.1).")
    else:
        if b < max(BEAM_MIN_WIDTH_RATIO * h, BEAM_MIN_WIDTH_IN):
            raise ValueError(f"Beam width {b} in violates ACI 18.6.2.1 for depth {h} in.")
        if story_height_in is not None and h > BEAM_MAX_DEPTH_FRACTION_OF_STORY * story_height_in:
            raise ValueError(f"Beam depth {h} in exceeds {BEAM_MAX_DEPTH_FRACTION_OF_STORY} of story height.")
    if fc <= 0.0:
        raise ValueError("Concrete strength must be positive.")
    return True
