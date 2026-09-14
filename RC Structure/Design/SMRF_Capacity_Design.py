"""Capacity design at every joint: probable strengths, shear, joint shear, anchorage.

Pure computation on an explicit ``state`` dictionary (no OpenSees, no
Structure_Parameters); the driver assembles the state and applies the
selected transverse steel. Everything here is ACI 318-19 Chapter 18:

* Beam probable moment Mpr (18.6.5.1): the developed beam-plus-slab section
  of SMRF_Beam_Slab_Strength with steel at 1.25 fy and phi = 1.
* Beam design shear Ve: Mpr equilibrium over the clear span plus the
  factored gravity face reactions from the slab transfer (SMRF_Joints
  .beam_capacity_shear_envelope), for the high- and low-gravity seismic
  combinations. Vc = 0 in the hinge zone when the mechanism shear is at least
  half of Ve and the axial load is below Ag fc/20 (18.6.5.2); Vs limited to
  8 sqrt(fc) bw d (22.5.1.2). Hoop spacing/legs/bar are chosen from Ve, the
  18.6.4.4 spacing bounds and a 1-in grid from 3 in.
* Column design shear Ve (18.7.6.1.1): 2 Mpr,col / ln from the column's own
  probable P-M strength over its factored axial range, not exceeding the
  shear the beams' probable strengths can deliver through the joints
  (floor joints split the beam moments equally above/below; a roof joint
  sends all of it into the single column). Vc per 18.7.6.2.1 / 22.5.5.1,
  hoops from Vs, the 18.7.5.4 confinement area and the 18.7.5.3 spacing
  limits (so from hx with every face bar tied).
* Joint shear (18.8.4): Vj = T + C - Vcol with T = 1.25 fy As including the
  slab bars in the effective flange, Vcol from the same mechanism, Aj per
  18.8.4.3 (SMRF_Joints.rectangular_joint_area), gamma per Table 18.8.4.3
  from the faces confined by beams at least 3/4 of the column width and
  from column continuity (terminating at the roof), phi = 0.85.
* Terminating beam bars (18.8.5.1): standard 90-degree hooks, ldh =
  fy db / (65 lambda sqrt(fc)) >= max(8 db, 6 in) inside the column depth
  less cover and hoop; through bars keep the 18.8.2.3 depth check.

Steel and concrete units: kip, inch, ksi (sqrt(fc) terms use psi).
"""
from __future__ import annotations

import math

from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths
from Design.SMRF_Common import make_check, not_evaluated
from Design.SMRF_Joints import (MPR_BASIS, beam_capacity_shear_envelope, rectangular_joint_area)

METHOD_VERSION = "aci318_19_capacity_design_v1"
_BAR = {3: (0.375, 0.11), 4: (0.5, 0.20), 5: (0.625, 0.31), 6: (0.75, 0.44), 7: (0.875, 0.60),
        8: (1.0, 0.79), 9: (1.128, 1.00), 10: (1.27, 1.27), 11: (1.41, 1.56)}
PHI_SHEAR = 0.75
PHI_JOINT = 0.85
SPACING_GRID_IN = 1.0
SPACING_MIN_IN = 3.0
STIRRUP_LADDER = ((4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4), (5, 6))   # (bar, legs)
GAMMA = {"continuous": {4: 20.0, 3: 15.0, 2: 15.0, 0: 12.0},
         "terminating": {4: 15.0, 3: 12.0, 2: 12.0, 0: 8.0}}


def _grid_down(value):
    steps = math.floor((value - SPACING_MIN_IN) / SPACING_GRID_IN + 1e-9)
    return SPACING_MIN_IN + max(0, steps) * SPACING_GRID_IN if value >= SPACING_MIN_IN else None


def probable_beam_strengths(state, position, axis):
    """Composite beam Mn and Mpr (1.25 fy, phi = 1) for one family."""
    beam, slab = state["beam"], state["slab"]
    geometry = {"bay_x_in": state["geometry"]["bay_x_in"], "bay_y_in": state["geometry"]["bay_y_in"],
                "h_col_in": state["sections"]["h_col_in"], "b_col_in": state["sections"]["b_col_in"]}
    section = {"b_in": state["sections"]["b_beam_in"], "h_in": state["sections"]["h_beam_in"],
               "fc_ksi": state["sections"]["fc_beam_ksi"], "fy_ksi": state["materials"]["fy_ksi"],
               "bar_size": beam["bar_size"], "top_bars": beam["top_bars"], "bot_bars": beam["bot_bars"],
               "centroid_offset_in": beam["centroid_offset_in"]}
    layout = slab.get("layout")
    slab_arg = {"thickness_in": slab["thickness_in"] if layout is not None else 0.0}
    nominal = composite_beam_strengths(section, slab_arg, layout, geometry, axis, position)
    probable = composite_beam_strengths({**section, "fy_ksi": 1.25 * section["fy_ksi"]},
                                        slab_arg, layout, geometry, axis, position)
    ab = _BAR[beam["bar_size"]][1]
    tension_hogging = beam["top_bars"] * ab + probable["slab_steel_in_flange_in2"]
    tension_sagging = beam["bot_bars"] * ab
    return {"axis": axis, "position": position,
            "mn_positive_kip_in": nominal["mn_positive_kip_in"], "mn_negative_kip_in": nominal["mn_negative_kip_in"],
            "mpr_positive_kip_in": probable["mn_positive_kip_in"], "mpr_negative_kip_in": probable["mn_negative_kip_in"],
            "effective_flange_width_in": probable["effective_flange_width_in"],
            "tension_steel_hogging_in2": tension_hogging, "tension_steel_sagging_in2": tension_sagging,
            "clear_span_in": nominal["clear_span_in"], "mpr_basis": MPR_BASIS}


def _span_face_reactions(beam, span_in, column_depth_in, factor):
    """Upward joint-face reactions of one span's transfer loads, zero end moments.

    The free body is the beam between the joint faces (18.6.5.1): a load at
    centerline fraction x sits at a' = x L - c/2 from the left face; loads
    inside a column footprint go straight to that column and are excluded.
    A point force P gives P (1 - a'/ln), P a'/ln; a local-y couple M (the
    slab's vertical-plane bending couple at that node, positive clockwise
    with local z up) gives -M/ln at the left face and +M/ln at the right.
    """
    clear = span_in - column_depth_in
    left = right = 0.0
    for x, p in beam["node_loads"]:
        a = x * span_in - 0.5 * column_depth_in
        if 0.0 < a < clear:
            left += factor * p * (1.0 - a / clear)
            right += factor * p * a / clear
    for x, _torsion, couple in beam.get("node_couples", []):
        a = x * span_in - 0.5 * column_depth_in
        if 0.0 < a < clear:
            left -= factor * couple / clear
            right += factor * couple / clear
    return left, right


def _drop_weight(beam, axis, geometry, sections):
    """Physical beam weight per inch between the faces; recovered from the smeared value if absent."""
    if beam.get("drop_weight_kip_per_in") is not None:
        return beam["drop_weight_kip_per_in"]
    span = geometry["bay_x_in"] if axis == "x" else geometry["bay_y_in"]
    depth = sections["h_col_in"] if axis == "x" else sections["b_col_in"]
    return beam["self_weight_kip_per_in"][axis] / (1.0 - depth / span)


def _face_reactions(transfer, axis, position, geometry, factor_dead, factor_live, beam_weight_per_in,
                    column_depth_in=None):
    """Envelope of the upward joint-face reactions of a family under factored gravity alone.

    Every span of the family and every unit case that can load it are
    evaluated end by end: dead once, live as the envelope over the all-panel
    case and each saved ACI 6.4.2 pattern (a pattern can load one span more
    than the full floor does). The left value is the largest left-face
    reaction over spans, the right value the largest right-face reaction,
    so neither end nor sway direction is judged on another end's span. With
    a zero live factor the live envelope is zero. Beam self-weight acts on
    the clear span.
    """
    nx, ny = geometry["num_bay_x"], geometry["num_bay_y"]
    if axis == "x":
        lines = [j for j in range(ny + 1) if (j in (0, ny)) == (position == "edge")]
    else:
        lines = [i for i in range(nx + 1) if (i in (0, nx)) == (position == "edge")]
    span = geometry["bay_x_in"] if axis == "x" else geometry["bay_y_in"]
    depth = column_depth_in if column_depth_in is not None else 0.0
    clear = span - depth
    # beam_weight_per_in is the PHYSICAL drop weight per inch between the
    # faces (state["beam"]["drop_weight_kip_per_in"]), not the smeared
    # centerline line weight, which already carries the clear fraction.
    self_weight = factor_dead * beam_weight_per_in * clear / 2.0
    cases = (transfer or {}).get("unit_cases", {}) if transfer else {}
    if not cases:
        return [self_weight, self_weight], "member self-weight only; no slab transfer supplied"
    dead, live = {}, {}
    for name, case in cases.items():
        factor = factor_dead if name == "dead" else factor_live
        if factor == 0.0 or not case:
            continue
        target = dead if name == "dead" else live
        for beam in case["beams"]:
            if beam["axis"] != axis or beam["line_index"] not in lines:
                continue
            key = (beam["line_index"], beam["span_index"])
            left, right = _span_face_reactions(beam, span, depth, factor)
            if name == "dead":
                target[key] = [left, right]
            else:
                # Live envelope per end over the all-panel case and every pattern.
                current = target.setdefault(key, [-math.inf, -math.inf])
                current[0] = max(current[0], left)
                current[1] = max(current[1], right)
    spans = set(dead) | set(live)
    if not spans:
        return [self_weight, self_weight], "member self-weight only; family has no transfer spans"
    left_by_span = {key: dead.get(key, [0.0, 0.0])[0] + (live[key][0] if key in live else 0.0) for key in spans}
    right_by_span = {key: dead.get(key, [0.0, 0.0])[1] + (live[key][1] if key in live else 0.0) for key in spans}
    left_key = max(left_by_span, key=left_by_span.get)
    right_key = max(right_by_span, key=right_by_span.get)
    return ([left_by_span[left_key] + self_weight, right_by_span[right_key] + self_weight],
            f"joint-face free body over the clear span; left governed by span {left_key}, right by span {right_key}; "
            f"live enveloped over {len([n for n in cases if n != 'dead'])} unit case(s); plus beam self-weight")


def design_beam_shear(state, strengths, transfer):
    """Ve at the faces for both seismic gravity families and the hinge-zone hoops."""
    sections, mats, beam, geometry = state["sections"], state["materials"], state["beam"], state["geometry"]
    fc, fy = sections["fc_beam_ksi"], mats["fy_ksi"]
    bw, h = sections["b_beam_in"], sections["h_beam_in"]
    d = h - beam["centroid_offset_in"]
    sds = state["sds"]
    families = {}
    worst_ve, worst_mechanism = 0.0, 0.0
    for key, s in strengths.items():
        entries = []
        for family_name, dead, live in (("seismic_high_gravity", 1.2 + 0.2 * sds, 1.0),
                                        ("seismic_low_gravity", 0.9 - 0.2 * sds, 0.0)):
            reactions, basis = _face_reactions(transfer, s["axis"], s["position"], geometry, dead, live,
                                               _drop_weight(beam, s["axis"], geometry, sections),
                                               column_depth_in=(sections["h_col_in"] if s["axis"] == "x"
                                                                else sections["b_col_in"]))
            data = {"id": f"{key}/{family_name}", "clear_span_in": s["clear_span_in"],
                    "mpr_left_positive_kip_in": s["mpr_positive_kip_in"], "mpr_left_negative_kip_in": s["mpr_negative_kip_in"],
                    "mpr_right_positive_kip_in": s["mpr_positive_kip_in"], "mpr_right_negative_kip_in": s["mpr_negative_kip_in"],
                    "gravity_reactions_kip": reactions, "mpr_basis": MPR_BASIS,
                    "gravity_basis": "factored_zero_end_moment_reactions",
                    "gravity_reaction_basis": basis, "gravity_family": family_name,
                    "dead_factor": dead, "live_factor": live}
            envelope = beam_capacity_shear_envelope(data)
            data["envelope"] = envelope
            entries.append(data)
            worst_ve = max(worst_ve, envelope["left_required_kip"], envelope["right_required_kip"])
            worst_mechanism = max(worst_mechanism, envelope["mechanism_shear_positive_kip"],
                                  envelope["mechanism_shear_negative_kip"])
        families[key] = entries
    # Hinge-zone concrete contribution (18.6.5.2): beams carry no axial load.
    vc_zero = worst_mechanism >= 0.5 * worst_ve
    vc = 0.0 if vc_zero else 2.0 * math.sqrt(fc * 1000.0) * bw * d / 1000.0
    vs_required = max(0.0, worst_ve / PHI_SHEAR - vc)
    vs_limit = 8.0 * math.sqrt(fc * 1000.0) * bw * d / 1000.0
    db = _BAR[beam["bar_size"]][0]
    bounds = {"d_over_4": d / 4.0, "six_db": 6.0 * db, "absolute": 6.0}
    selected = None
    for bar, legs in STIRRUP_LADDER:
        av = legs * _BAR[bar][1]
        s_shear = av * fy * d / vs_required if vs_required > 0 else float("inf")
        s_cap = min(s_shear, *bounds.values())
        spacing = _grid_down(s_cap)
        if spacing is not None:
            selected = {"bar_size": bar, "legs": legs, "spacing_in": spacing, "av_in2": av,
                        "spacing_from_shear_in": s_shear, "phi_vs_kip": PHI_SHEAR * av * fy * d / spacing,
                        "phi_vn_kip": PHI_SHEAR * (vc + av * fy * d / spacing)}
            break
    result = {"families": families, "ve_kip": worst_ve, "mechanism_shear_kip": worst_mechanism,
              "vc_zero_hinge_zone": vc_zero, "vc_kip": vc, "vs_required_kip": vs_required,
              "vs_limit_kip": vs_limit, "section_adequate": vs_required <= vs_limit,
              "spacing_bounds_in": bounds, "hoops": selected,
              "basis": ("Vc = 0 in the hinge zone when the Mpr mechanism shear is at least half of Ve (18.6.5.2); "
                        "hoops at <= min(d/4, 6db, 6 in) and Av fyt d / Vs; uniform along the member")}
    for entries in families.values():
        for data in entries:
            data["phi_vn_left_kip"] = selected["phi_vn_kip"] if selected else 0.0
            data["phi_vn_right_kip"] = selected["phi_vn_kip"] if selected else 0.0
            data["shear_capacity_requirements_checked"] = bool(selected) and result["section_adequate"]
            data["hoops"] = selected
    return result


def column_probable_pm(state, fy_factor=1.25, n_pts=160):
    """Probable P-M surface of the column section (steel at fy_factor * fy)."""
    sections, col, mats = state["sections"], state["column"], state["materials"]
    b, h, fc = sections["b_col_in"], sections["h_col_in"], sections["fc_col_ksi"]
    fy, es = fy_factor * mats["fy_ksi"], mats["es_ksi"]
    layers = col["layers"]
    b1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 4.0)))
    ag, ast, hc = b * h, sum(a for a, _ in layers), h / 2.0
    p0 = 0.85 * fc * (ag - ast) + fy * ast
    diagram = [(0.80 * p0, 0.0)]
    for index in range(n_pts):
        c = 0.001 + (4.0 * h - 0.001) * index / (n_pts - 1)
        a = min(b1 * c, h)
        compression = 0.85 * fc * b * a
        axial, moment = compression, compression * (hc - a / 2.0)
        for area, depth in layers:
            strain = 0.003 * (c - depth) / c
            stress = max(-fy, min(fy, es * strain))
            net = stress - (0.85 * fc if depth <= a else 0.0)
            axial += net * area
            moment += net * area * (hc - depth)
        diagram.append((axial, abs(moment)))
    return sorted(diagram)


def _moment_at(diagram, axial):
    axials = [p for p, _ in diagram]
    moments = [m for _, m in diagram]
    if axial <= axials[0]:
        return moments[0]
    if axial >= axials[-1]:
        return moments[-1]
    for k in range(1, len(diagram)):
        if axial <= axials[k]:
            w = (axial - axials[k - 1]) / max(1e-12, axials[k] - axials[k - 1])
            return moments[k - 1] + w * (moments[k] - moments[k - 1])
    return moments[-1]


def design_column_shear(state, strengths):
    """Column Ve per story, its hoops from shear, confinement and spacing limits."""
    sections, col, mats, geometry = state["sections"], state["column"], state["materials"], state["geometry"]
    b, h, fc = sections["b_col_in"], sections["h_col_in"], sections["fc_col_ksi"]
    fy = mats["fy_ksi"]
    ln = geometry["story_h_in"] - sections["h_beam_in"]
    d = h - col["centroid_offset_in"]
    ag = b * h
    diagram = column_probable_pm(state)
    # Beam probable moments a joint can deliver to its columns, per direction.
    def beams_at(kind, axis):
        interior = strengths[f"{axis}_interior"]
        edge = strengths[f"{axis}_edge"]
        if kind == "interior":
            return interior["mpr_negative_kip_in"] + interior["mpr_positive_kip_in"]
        if kind == "edge":            # one interior beam perpendicular, two edge beams parallel
            return max(edge["mpr_negative_kip_in"] + edge["mpr_positive_kip_in"],
                       max(interior["mpr_negative_kip_in"], interior["mpr_positive_kip_in"]))
        return max(edge["mpr_negative_kip_in"], edge["mpr_positive_kip_in"])
    joint_delivery = {kind: max(beams_at(kind, "x"), beams_at(kind, "y")) for kind in ("interior", "edge", "corner")}
    stories = {}
    worst = None
    for story in range(1, geometry["num_floor"] + 1):
        p_min, p_max = state["column_axial_envelope"].get(story, (0.0, 0.0))
        samples = [p_min + (p_max - p_min) * k / 8.0 for k in range(9)]
        mpr_col = max(_moment_at(diagram, p) for p in samples)
        ve_own = 2.0 * mpr_col / ln
        top_is_roof = story == geometry["num_floor"]
        # Interior column: floor joint below gives half the beam sum, joint above
        # gives half (or all of it at the roof).
        m_bottom = joint_delivery["interior"] / 2.0 if story > 1 else mpr_col
        m_top = joint_delivery["interior"] if top_is_roof else joint_delivery["interior"] / 2.0
        ve_joint_limited = (m_top + m_bottom) / ln
        vu = state["column_shear_demand"].get(story, 0.0)
        ve = max(min(ve_own, ve_joint_limited), vu)
        mechanism = min(ve_own, ve_joint_limited)
        vc_zero = mechanism >= 0.5 * ve and p_min < ag * fc / 20.0
        nu_psi = max(0.0, p_min) * 1000.0 / ag
        vc = 0.0 if vc_zero else (2.0 * math.sqrt(fc * 1000.0) + min(nu_psi / 6.0, 0.05 * fc * 1000.0)) * b * d / 1000.0
        vs_required = max(0.0, ve / PHI_SHEAR - vc)
        stories[story] = {"axial_min_kip": p_min, "axial_max_kip": p_max, "mpr_column_kip_in": mpr_col,
                          "ve_own_kip": ve_own, "ve_joint_limited_kip": ve_joint_limited, "vu_analysis_kip": vu,
                          "ve_kip": ve, "vc_zero": vc_zero, "vc_kip": vc, "vs_required_kip": vs_required,
                          "roof_story": top_is_roof}
        if worst is None or vs_required > worst["vs_required_kip"]:
            worst = stories[story]
    vs_limit = 8.0 * math.sqrt(fc * 1000.0) * b * d / 1000.0
    # Confinement (18.7.5.4) with every face bar tied by a leg or crosstie.
    cc = col["clear_cover_in"]
    ach = (b - 2.0 * cc) * (h - 2.0 * cc)
    bc = max(b, h) - 2.0 * cc
    p_max_all = max(v["axial_max_kip"] for v in stories.values()) if stories else 0.0
    ratio = max(0.3 * (ag / ach - 1.0) * fc / fy, 0.09 * fc / fy)
    high_axial = p_max_all > 0.3 * ag * fc
    if high_axial:
        kf = max(1.0, fc / 25.0 + 0.6)
        perimeter_bars = col["top_bars"] + col["bot_bars"] + 2 * col["side_bars"]
        kn = perimeter_bars / max(1, perimeter_bars - 2)
        ratio = max(ratio, 0.2 * kf * kn * p_max_all / (fy * ach))
    # Every face bar is tied, so hx is the larger face's bar spacing.
    face_bars = max(col["top_bars"], col["bot_bars"], col["side_bars"] + 2)
    offset = col["centroid_offset_in"]
    hx = max((b - 2.0 * offset) / max(1, max(col["top_bars"], col["bot_bars"]) - 1),
             (h - 2.0 * offset) / max(1, col["side_bars"] + 1))
    so = max(4.0, min(6.0, 4.0 + (14.0 - hx) / 3.0))
    if high_axial and hx > 8.0:
        hx_ok = False
    else:
        hx_ok = True
    db = _BAR[col["bar_size"]][0]
    # The hx-dependent so (4-6 in) is recorded, but the methodology keeps a
    # conservative 4-in cap until the supported-bar layout is verified.
    bounds = {"quarter_min_dimension": min(b, h) / 4.0, "six_db": 6.0 * db, "so": so,
              "conservative_so_cap": 4.0}
    selected = None
    for bar, legs in STIRRUP_LADDER:
        if legs < face_bars:
            continue                      # every face bar needs a leg or crosstie
        av = legs * _BAR[bar][1]
        s_shear = av * fy * d / worst["vs_required_kip"] if worst and worst["vs_required_kip"] > 0 else float("inf")
        s_conf = av / (ratio * bc)
        spacing = _grid_down(min(s_shear, s_conf, *bounds.values()))
        if spacing is not None:
            selected = {"bar_size": bar, "legs": legs, "spacing_in": spacing, "av_in2": av,
                        "spacing_from_shear_in": s_shear, "spacing_from_confinement_in": s_conf,
                        "ash_provided_per_in": av / spacing, "ash_required_per_in": ratio * bc,
                        "phi_vn_kip": PHI_SHEAR * ((worst["vc_kip"] if worst else 0.0) + av * fy * d / spacing)}
            break
    return {"stories": stories, "governing": worst, "vs_limit_kip": vs_limit,
            "section_adequate": (worst["vs_required_kip"] if worst else 0.0) <= vs_limit,
            "confinement": {"ach_in2": ach, "bc_in": bc, "ash_ratio_required": ratio, "high_axial": high_axial,
                            "hx_in": hx, "so_in": so, "hx_within_8in_when_required": hx_ok},
            "spacing_bounds_in": bounds, "hoops": selected, "joint_delivery_kip_in": joint_delivery,
            "basis": ("Ve = min(2 Mpr,col/ln over the factored axial range, joint-limited beam Mpr delivery) >= Vu; "
                      "Vc = 0 when mechanism shear >= Ve/2 and Pu < Ag fc/20 (18.7.6.2.1); hoops from Vs, "
                      "18.7.5.4 confinement and 18.7.5.3 spacing with every face bar tied")}


def joint_kinds(num_bay_x, num_bay_y):
    """The distinct joint connectivities of a uniform rectangular frame.

    Returns {kind: {axis: (family in that direction, beam count in that
    direction, perpendicular beam count)}}. A column on the y = 0 / y = Ly
    perimeter line (``edge_x``) carries two edge x-beams and one interior
    y-beam that terminates in it; a column on the x = 0 / x = Lx line
    (``edge_y``) the mirror. Both directions of both kinds are enumerated:
    the terminating interior-family beam and the pair of edge-family beams
    are different joints. Kinds that the plan does not contain are omitted.
    """
    kinds = {"corner": {"x": ("x_edge", 1, 1), "y": ("y_edge", 1, 1)}}
    if num_bay_x >= 2:
        kinds["edge_x"] = {"x": ("x_edge", 2, 1), "y": ("y_interior", 1, 2)}
    if num_bay_y >= 2:
        kinds["edge_y"] = {"x": ("x_interior", 1, 2), "y": ("y_edge", 2, 1)}
    if num_bay_x >= 2 and num_bay_y >= 2:
        kinds["interior"] = {"x": ("x_interior", 2, 2), "y": ("y_interior", 2, 2)}
    return kinds


def design_joint_shear(state, strengths, column):
    """Vj against phi Vn for every joint kind, direction and column continuity."""
    sections, mats, geometry = state["sections"], state["materials"], state["geometry"]
    fy = mats["fy_ksi"]
    fc_joint = sections["fc_col_ksi"]
    bw = sections["b_beam_in"]
    story_h = geometry["story_h_in"]
    results, evidence = {}, []
    kinds = joint_kinds(geometry["num_bay_x"], geometry["num_bay_y"])
    for level in ("floor", "roof"):
        continuity = "terminating" if level == "roof" else "continuous"
        for kind, per_axis in kinds.items():
            for axis in ("x", "y"):
                depth = sections["h_col_in"] if axis == "x" else sections["b_col_in"]
                width = sections["b_col_in"] if axis == "x" else sections["h_col_in"]
                family, count, perpendicular_count = per_axis[axis]
                fam = strengths[family]
                t_hog = 1.25 * fy * fam["tension_steel_hogging_in2"]
                t_sag = 1.25 * fy * fam["tension_steel_sagging_in2"]
                if count == 2:
                    forces = [t_hog, t_sag]
                    sum_mpr = fam["mpr_negative_kip_in"] + fam["mpr_positive_kip_in"]
                else:
                    forces = [max(t_hog, t_sag)]
                    sum_mpr = max(fam["mpr_negative_kip_in"], fam["mpr_positive_kip_in"])
                vcol = sum_mpr / story_h * (2.0 if level == "roof" else 1.0)
                vj = abs(sum(forces) - vcol)
                # Confinement: a beam confines the face it frames into when it is
                # at least 3/4 of that face's width. "Two opposite faces" means
                # both faces in one direction, not two adjacent ones.
                parallel = count if bw >= 0.75 * width else 0
                perpendicular = perpendicular_count if bw >= 0.75 * depth else 0
                confined = parallel + perpendicular
                category = (4 if confined == 4 else 3 if confined == 3
                            else 2 if (parallel == 2 or perpendicular == 2) else 0)
                gamma = GAMMA[continuity][category]
                aj = rectangular_joint_area(column_depth_in=depth, column_width_in=width,
                                            beam_width_in=bw, beam_center_offset_in=0.0)
                vn = gamma * math.sqrt(fc_joint * 1000.0) * aj / 1000.0
                entry = {"id": f"joint_shear/{level}/{kind}/{axis}", "level": level, "kind": kind, "axis": axis,
                         "beam_family": family, "beams_in_direction": count, "beams_perpendicular": perpendicular_count,
                         "beam_face_forces_kip": forces, "column_shear_kip": vcol,
                         "probable_face_forces_complete": True, "column_shear_consistent_with_mpr": True,
                         "nominal_vn_kip": vn, "capacity_basis": "ACI318-19_Table18.8.4.3",
                         "capacity_topology_and_confinement_checked": False,   # set once joint hoops are designed
                         "gamma": gamma, "confined_faces": confined, "column_continuity": continuity,
                         "aj_in2": aj, "vj_kip": vj, "phi_vn_kip": PHI_JOINT * vn,
                         "passes": vj <= PHI_JOINT * vn,
                         "slab_steel_in_tension_included": True}
                results[entry["id"]] = entry
                evidence.append(entry)
    # Joint transverse reinforcement (18.8.3.1): the column end-zone hoops
    # continue through the joint at their spacing and area (18.7.5.2-.4);
    # the 18.8.3.2 relaxation applies only when all four faces are confined.
    hoops = column.get("hoops") if column else None
    all_confined = all(v["confined_faces"] == 4 for v in results.values())
    joint_transverse = {
        "hoops": hoops, "basis": "18.8.3.1: column hoops per 18.7.5.2--18.7.5.4 continue through the joint depth",
        "relaxation_18_8_3_2_applicable": all_confined,
        "designed": hoops is not None,
        "column_shear_basis": ("Vcol = sum of beam Mpr at the joint divided by the story height, columns above and "
                               "below sharing equally (inflection at mid-height); at a terminating roof column the "
                               "single column takes the whole sum, so Vcol = 2 sum Mpr / H"),
    }
    for entry in evidence:
        entry["capacity_topology_and_confinement_checked"] = joint_transverse["designed"]
        entry["joint_transverse_reinforcement"] = joint_transverse["basis"]
        entry["column_shear_basis"] = joint_transverse["column_shear_basis"]
    return {"joints": results, "evidence": evidence, "all_pass": all(v["passes"] for v in results.values()),
            "joint_transverse": joint_transverse,
            "basis": ("Vj = 1.25 fy (As,top + slab bars in flange) + 1.25 fy As,bot - Vcol; Vcol = sum Mpr / H "
                      "(2 sum Mpr / H at a terminating roof column); gamma from Table 18.8.4.3 with faces confined "
                      "by beams >= 3/4 the column width; Aj per 18.8.4.3; phi = 0.85")}


def design_anchorage(state):
    """Terminating beam bars at exterior joints (18.8.5.1) and through bars (18.8.2.3).

    Through bars exist only in a direction with an interior joint (two or
    more bays); a one-bay direction terminates every bar with a hook.
    """
    sections, beam, mats, col = state["sections"], state["beam"], state["materials"], state["column"]
    geometry = state.get("geometry") or {}
    fy, fc = mats["fy_ksi"], sections["fc_col_ksi"]
    db = _BAR[beam["bar_size"]][0]
    ldh = max(fy * 1000.0 * db / (65.0 * math.sqrt(fc * 1000.0)), 8.0 * db, 6.0)
    hoop_db = _BAR[col["stirrup_bar_size"]][0]
    results = {}
    for axis in ("x", "y"):
        depth = sections["h_col_in"] if axis == "x" else sections["b_col_in"]
        bays = geometry.get("num_bay_x" if axis == "x" else "num_bay_y")
        through = True if bays is None else bays >= 2
        available = depth - col["clear_cover_in"] - hoop_db
        results[axis] = {"ldh_required_in": ldh, "embedment_available_in": available,
                         "hook": "standard 90-degree hook into the confined core, 18.8.5.1",
                         "passes": ldh <= available,
                         "through_bars_present": through,
                         "through_bar_depth_required_in": 20.0 * db, "through_bar_depth_available_in": depth,
                         "through_bar_passes": (20.0 * db <= depth) if through else True}
    return {"directions": results, "all_pass": all(v["passes"] and v["through_bar_passes"] for v in results.values()),
            "basis": ("ldh = fy db / (65 lambda sqrt(fc)) >= max(8db, 6 in); embedment = column depth - cover - hoop; "
                      "through bars need a joint depth of 20 db (normalweight, 18.8.2.3) where an interior joint exists")}


def straight_development_length(bar, fc_ksi, fy_ksi, top_bar=False):
    """ACI 318-19 25.4.2.4 straight development, uncoated Grade 60, normalweight.

    Clear spacing >= db with cover >= db and hoops throughout (or spacing >=
    2db and cover >= db) selects the /25 (#6 and smaller) or /20 (#7 and
    larger) form; psi_t = 1.3 for bars with more than 12 in of fresh
    concrete below them.
    """
    db = _BAR[bar][0]
    divisor = 25.0 if bar <= 6 else 20.0
    psi_t = 1.3 if top_bar else 1.0
    return max(12.0, fy_ksi * 1000.0 * psi_t / (divisor * math.sqrt(fc_ksi * 1000.0)) * db)


def design_splices(state):
    """Where and how longitudinal bars can be spliced (18.6.3.3, 18.7.4.3, 25.5, 18.2.7).

    Beams: Class B lap (1.3 ld) only outside the 2h hinge zones and enclosed
    by hoops at <= min(d/4, 4 in); when the middle of the clear span cannot
    hold 1.3 ld, Type 2 mechanical splices (18.2.7.1, permitted anywhere)
    are the design. Columns: Class B lap within the center half of the
    clear height, enclosed by the end-zone hoops; otherwise mechanical.
    """
    sections, beam, col, mats, geometry = state["sections"], state["beam"], state["column"], state["materials"], state["geometry"]
    fy = mats["fy_ksi"]
    result = {}
    # Beams: worst (shortest) clear span in either direction.
    clear = min(geometry["bay_x_in"] - sections["h_col_in"], geometry["bay_y_in"] - sections["b_col_in"])
    h = sections["h_beam_in"]
    lap_top = 1.3 * straight_development_length(beam["bar_size"], sections["fc_beam_ksi"], fy, top_bar=h - beam["centroid_offset_in"] > 12.0)
    lap_bot = 1.3 * straight_development_length(beam["bar_size"], sections["fc_beam_ksi"], fy, top_bar=False)
    available = clear - 2.0 * (2.0 * h)
    beam_lap_fits = max(lap_top, lap_bot) <= available
    result["beam"] = {"class_b_lap_top_in": lap_top, "class_b_lap_bottom_in": lap_bot,
                      "available_between_hinge_zones_in": available, "lap_splice_feasible": beam_lap_fits,
                      "splice_type": "class_B_lap_outside_hinge_zones" if beam_lap_fits else "type_2_mechanical_18.2.7",
                      "hoops_over_lap_max_spacing_in": min((h - beam["centroid_offset_in"]) / 4.0, 4.0),
                      "basis": "18.6.3.3 (no laps within 2h of the face or where yielding is expected); 25.5.2.1 Class B"}
    ln = geometry["story_h_in"] - h
    lap_col = 1.3 * straight_development_length(col["bar_size"], sections["fc_col_ksi"], fy, top_bar=False)
    center_half = ln / 2.0
    col_lap_fits = lap_col <= center_half
    result["column"] = {"class_b_lap_in": lap_col, "center_half_clear_height_in": center_half,
                        "lap_splice_feasible": col_lap_fits,
                        "splice_type": "class_B_lap_center_half" if col_lap_fits else "type_2_mechanical_18.2.7",
                        "basis": "18.7.4.3 (laps only in the center half, tension laps, hoops per 18.7.5.2/.3); 25.5.2.1 Class B"}
    result["all_designed"] = True
    return result


def build_capacity_design(state):
    """All capacity-design evidence for the current state, plus its checks."""
    strengths = {f"{axis}_{position}": probable_beam_strengths(state, position, axis)
                 for axis in ("x", "y") for position in ("edge", "interior")}
    beams = design_beam_shear(state, strengths, state.get("transfer"))
    columns = design_column_shear(state, strengths)
    joints = design_joint_shear(state, strengths, columns)
    anchorage = design_anchorage(state)
    splices = design_splices(state)
    checks = []
    checks.append(make_check("detailing.splices_designed", "ACI 318-19 18.6.3.3 / 18.7.4.3 / 18.2.7 / 25.5",
                             1, 1, "==", details={"beam": splices["beam"], "column": splices["column"]}))
    checks.append(make_check("beam.capacity_shear_section", "ACI 318-19 22.5.1.2 with 18.6.5.1 Ve",
                             beams["vs_required_kip"], beams["vs_limit_kip"], "<=", "kip"))
    checks.append(make_check("beam.hoops_selected", "ACI 318-19 18.6.4.4 / 18.6.5", int(beams["hoops"] is not None), 1, "=="))
    checks.append(make_check("column.capacity_shear_section", "ACI 318-19 22.5.1.2 with 18.7.6.1.1 Ve",
                             columns["governing"]["vs_required_kip"] if columns["governing"] else 0.0,
                             columns["vs_limit_kip"], "<=", "kip"))
    checks.append(make_check("column.hoops_selected", "ACI 318-19 18.7.5.3 / 18.7.5.4 / 18.7.6",
                             int(columns["hoops"] is not None), 1, "=="))
    checks.append(make_check("column.supported_bar_spacing", "ACI 318-19 18.7.5.2",
                             int(columns["confinement"]["hx_within_8in_when_required"]), 1, "=="))
    for entry in joints["evidence"]:
        checks.append(make_check("joint.shear_screen", "ACI 318-19 18.8.4", entry["vj_kip"], entry["phi_vn_kip"],
                                 "<=", "kip", entry["id"]))
    for axis, item in anchorage["directions"].items():
        checks.append(make_check("joint.terminating_bar_hook", "ACI 318-19 18.8.5.1", item["ldh_required_in"],
                                 item["embedment_available_in"], "<=", "in", f"exterior/{axis}"))
        if item["through_bars_present"]:
            checks.append(make_check("joint.through_bar_depth", "ACI 318-19 18.8.2.3",
                                     item["through_bar_depth_required_in"], item["through_bar_depth_available_in"],
                                     "<=", "in", f"interior/{axis}"))
    beam_evidence = [data for entries in beams["families"].values() for data in entries]
    return {"method_version": METHOD_VERSION, "beam_strengths": strengths, "beams": beams,
            "columns": columns, "joints": joints, "anchorage": anchorage, "splices": splices,
            "transverse": {"beam": beams["hoops"], "column": columns["hoops"]},
            "joint_evidence": {"beam_capacity_shear": beam_evidence, "joint_shear": joints["evidence"]},
            "checks": checks,
            "accepted": all(c["status"] == "pass" for c in checks)}
