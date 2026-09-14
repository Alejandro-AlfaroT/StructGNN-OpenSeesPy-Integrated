import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.IMK_Calibration import (
    backbone_for_member,
    column_axial_domain,
    column_gravity_axial,
    column_grid_position,
    column_moment_at_axial,
    column_pm_nominal,
)


# Per-hinge backbones recorded as the model is built, keyed by physical
# element tag. Consumers that need to know what a specific hinge was actually
# given -- mechanism checks, hysteresis recording, dataset export -- read this
# instead of recomputing from globals, which cannot capture the axial
# dependence of a column.
_HINGE_REGISTRY = {}

# The nominal P-M sweep is identical for every column in a build, so it is
# computed once per model rather than once per member.
_PM_DIAGRAM_CACHE = {}


def reset_hinge_registry():
    """Clear per-build hinge state. Called when a new model is built."""
    _HINGE_REGISTRY.clear()
    _PM_DIAGRAM_CACHE.clear()


def hinge_registry():
    """All recorded hinge backbones, keyed by physical element tag."""
    return dict(_HINGE_REGISTRY)


def hinge_backbone(ele_tag):
    """Backbone recorded for one element, or None if it has no IMK hinges."""
    return _HINGE_REGISTRY.get(int(ele_tag))


def _cached_pm_diagram():
    key = (sp.B_COL, sp.H_COL, sp.FC_COL_KSI, sp.COL_BAR_SIZE,
           sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS)
    if key not in _PM_DIAGRAM_CACHE:
        _PM_DIAGRAM_CACHE.clear()
        _PM_DIAGRAM_CACHE[key] = column_pm_nominal()
    return _PM_DIAGRAM_CACHE[key]


def hinge_node_tag(ele_tag, end_id):
    return sp.IMK_HINGE_NODE_TAG_BASE + 10 * ele_tag + end_id


def hinge_element_tag(ele_tag, end_id):
    return sp.IMK_HINGE_ELEMENT_TAG_BASE + 10 * ele_tag + end_id


def imk_material_tag(ele_tag, end_id, local_rot_dir):
    return sp.IMK_MATERIAL_TAG_BASE + 100 * ele_tag + 10 * end_id + local_rot_dir


def _member_length(n_i, n_j):
    xi, yi, zi = ops.nodeCoord(n_i)
    xj, yj, zj = ops.nodeCoord(n_j)
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    length = math.sqrt(dx * dx + dy * dy + dz * dz)

    if length <= 0.0:
        raise ValueError(f"Zero-length physical member between nodes {n_i} and {n_j}.")

    return length


def beam_yield_moments(member_type, n_i, n_j):
    """(hogging, sagging) yield moments for a beam member, kip-in.

    Slab-aware designs use the developed beam-plus-slab strengths from
    Design.SMRF_Beam_Slab_Strength -- the same numbers the SCWB screen and
    the joint qualification compare columns against -- for the beam's
    family (x/y, perimeter/interior line). Without a slab the two signs
    come from the actual top and bottom bars. The legacy bundled-load model
    keeps its symmetric max(top, bottom) strength.
    """
    if sp.SLAB_THICKNESS_IN is None:
        symmetric = sp.beam_nominal_moment_y()
        return symmetric, symmetric, {"basis": "legacy symmetric max(top, bottom) bars", "family": None}
    from Design.SMRF_Beam_Slab_Strength import (composite_beam_strengths, beam_family,
                                                perimeter_slab_bar_anchorage, exterior_ends)
    xi, yi, _ = ops.nodeCoord(n_i)
    if member_type == "beam_x":
        line = int(round(yi / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
        span = int(round(xi / sp.BAY_X)) if sp.BAY_X > 0 else 0
    else:
        line = int(round(xi / sp.BAY_X)) if sp.BAY_X > 0 else 0
        span = int(round(yi / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
    axis, position = beam_family(member_type, line, sp.NUM_BAY_X, sp.NUM_BAY_Y)
    exterior = exterior_ends(member_type, span, sp.NUM_BAY_X, sp.NUM_BAY_Y)
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout")
    beam = {"b_in": sp.B_BEAM, "h_in": sp.H_BEAM, "fc_ksi": sp.FC_BEAM_KSI, "fy_ksi": sp.FY_KSI,
            "bar_size": sp.BEAM_BAR_SIZE, "top_bars": sp.BEAM_TOP_BARS, "bot_bars": sp.BEAM_BOT_BARS,
            "centroid_offset_in": sp.longitudinal_cover_in("beam")}
    slab = {"thickness_in": sp.SLAB_THICKNESS_IN if layout is not None else 0.0}
    geometry = {"bay_x_in": sp.BAY_X, "bay_y_in": sp.BAY_Y, "h_col_in": sp.H_COL, "b_col_in": sp.B_COL}
    family = composite_beam_strengths(beam, slab, layout, geometry, axis, position)
    anchorage = perimeter_slab_bar_anchorage(layout, axis, sp.B_BEAM, sp.BEAM_CLEAR_COVER_IN,
                                             sp.rebar_diameter(sp.BEAM_STIRRUP_BAR_SIZE), sp.FC_BEAM_KSI, sp.FY_KSI)
    hogging, sagging = family["mn_negative_kip_in"], family["mn_positive_kip_in"]
    undeveloped = layout is not None and anchorage is not None and not anchorage["developed"]
    # An exterior end whose slab bars are not developed at the perimeter
    # yields without them in either sign: rectangular hogging, and sagging
    # with the flange concrete but no bottom mat (SMRF_Beam_Slab_Strength).
    hogging_ends = {end: (family["mn_undeveloped_negative_kip_in"] if (undeveloped and exterior[end]) else hogging)
                    for end in ("i", "j")}
    sagging_ends = {end: (family["mn_undeveloped_positive_kip_in"] if (undeveloped and exterior[end]) else sagging)
                    for end in ("i", "j")}
    return hogging, sagging, {
        "basis": "beam plus developed slab mats in the ACI 6.3.2 flange" if layout is not None
                 else "rectangular beam, actual top and bottom bars; slab reinforcement not established",
        "family": f"{axis}_{position}",
        "effective_flange_width_in": family["effective_flange_width_in"],
        "slab_steel_in_flange_in2": family["slab_steel_in_flange_in2"],
        "hogging_i_kip_in": hogging_ends["i"], "hogging_j_kip_in": hogging_ends["j"],
        "sagging_i_kip_in": sagging_ends["i"], "sagging_j_kip_in": sagging_ends["j"],
        "exterior_ends": exterior, "exterior_anchorage": anchorage}


def _member_properties(member_type, axial_kip=0.0):
    """Elastic properties for an IMK member, on effective (cracked) stiffness.

    The stiffness modifier is applied once, here, so it reaches both the
    elastic element between the springs and the spring calibration itself
    (Ke = n * 6EI/L), keeping the two consistent.
    """
    modifier = sp.section_stiffness_modifier(member_type)

    if member_type == "column":
        # Zero-axial flexural capacity understates a column badly: it ignores
        # the compression and side steel, and ignores axial load entirely.
        # Both moments come off the nominal P-M surface instead.
        if getattr(sp, "IMK_USE_CALIBRATED_BACKBONE", True):
            diagram = _cached_pm_diagram()
            low, high = column_axial_domain(diagram)
            if not low <= axial_kip <= high:
                # Outside the nominal surface the section has no flexural
                # strength: it cannot carry the axial load. A hinge without
                # strength is not a model of anything, so refuse rather than
                # build a column that fails under gravity.
                raise ValueError(f"Column gravity axial estimate {axial_kip:.1f} kip is outside the nominal "
                                 f"P-M surface [{low:.1f}, {high:.1f}] kip (0.80 P0 cap, ACI 318-19 22.4.2.1); "
                                 "the section cannot carry it and the hinge cannot be calibrated.")
            capacity = column_moment_at_axial(axial_kip, diagram)
            my = mz = capacity
        else:
            my, mz = sp.column_nominal_moment_y(), sp.column_nominal_moment_z()
        return {
            "area": sp.rect_area(sp.B_COL, sp.H_COL),
            "e": sp.concrete_ec_ksi(sp.FC_COL_KSI),
            "g": sp.concrete_shear_modulus_ksi(sp.concrete_ec_ksi(sp.FC_COL_KSI)),
            "j": modifier * sp.approx_rect_j(sp.B_COL, sp.H_COL),
            "iy": modifier * sp.rect_iy(sp.B_COL, sp.H_COL),
            "iz": modifier * sp.rect_iz(sp.B_COL, sp.H_COL),
            "my": my,
            "mz": mz,
            "theta_y": sp.IMK_COLUMN_THETA_Y,
            "stiffness_modifier": modifier,
        }

    return {
        "area": sp.rect_area(sp.B_BEAM, sp.H_BEAM),
        "e": sp.concrete_ec_ksi(sp.FC_BEAM_KSI),
        "g": sp.concrete_shear_modulus_ksi(sp.concrete_ec_ksi(sp.FC_BEAM_KSI)),
        "j": modifier * sp.approx_rect_j(sp.B_BEAM, sp.H_BEAM),
        "iy": modifier * sp.rect_iy(sp.B_BEAM, sp.H_BEAM),
        "iz": modifier * sp.rect_iz(sp.B_BEAM, sp.H_BEAM),
        "my": sp.beam_nominal_moment_y(),
        "mz": sp.beam_nominal_moment_z(),
        "theta_y": sp.IMK_BEAM_THETA_Y,
        "stiffness_modifier": modifier,
    }


def imk_member_properties(member_type):
    return _member_properties(member_type).copy()


def _elastic_rotational_stiffness(e_mod, inertia, length):
    return 6.0 * e_mod * inertia / length


def _hinge_yield_data(props, rot_dir):
    if rot_dir in (5, "rot_y"):
        return props["my"], props["iy"]

    if rot_dir in (6, "rot_z"):
        return props["mz"], props["iz"]

    raise ValueError(f"Unknown IMK hinge rotation direction: {rot_dir}")


def imk_hinge_stiffness_components(member_type, rot_dir, length):
    props = _member_properties(member_type)
    yield_moment, inertia = _hinge_yield_data(props, rot_dir)

    if props["theta_y"] <= 0.0:
        raise ValueError(f"IMK theta_y must be positive for {member_type} {rot_dir}.")

    yield_based_stiffness = yield_moment / props["theta_y"]
    member_based_stiffness = (
        sp.IMK_HINGE_STIFFNESS_FACTOR
        * _elastic_rotational_stiffness(props["e"], inertia, length)
    )

    mode = getattr(sp, "IMK_HINGE_STIFFNESS_MODE", "yield_rotation")

    if mode == "yield_rotation":
        selected_stiffness = yield_based_stiffness
    elif mode == "member_stiffness_factor":
        selected_stiffness = member_based_stiffness
    elif mode == "max":
        selected_stiffness = max(yield_based_stiffness, member_based_stiffness)
    else:
        raise ValueError(
            "Unknown IMK_HINGE_STIFFNESS_MODE "
            f"{mode!r}; use 'yield_rotation', 'member_stiffness_factor', or 'max'."
        )

    return {
        "mode": mode,
        "yield_moment": yield_moment,
        "target_theta_y": props["theta_y"],
        "yield_based_stiffness": yield_based_stiffness,
        "member_based_stiffness": member_based_stiffness,
        "selected_stiffness": selected_stiffness,
        "actual_theta_y": yield_moment / selected_stiffness,
    }


def imk_hinge_stiffness(member_type, rot_dir, length):
    return imk_hinge_stiffness_components(
        member_type,
        rot_dir,
        length,
    )["selected_stiffness"]


def imk_hinge_thresholds(member_type, rot_dir, length):
    components = imk_hinge_stiffness_components(member_type, rot_dir, length)
    theta_y = components["actual_theta_y"]
    theta_p = max(sp.IMK_THETA_P_POS, sp.IMK_THETA_P_NEG)

    return {
        "theta_y": theta_y,
        "target_theta_y": components["target_theta_y"],
        "theta_cap": theta_y + theta_p,
        "theta_u": max(sp.IMK_THETA_U_POS, sp.IMK_THETA_U_NEG),
        "stiffness_mode": components["mode"],
        "selected_stiffness": components["selected_stiffness"],
    }


def imk_elastic_inertia_factor():
    """(n+1)/n stiffness correction for the elastic segment between springs.

    With a rotational spring of stiffness Ke = n * (6EI/L) at each end, the
    elastic element between them must use I * (n+1)/n so the composite
    spring-element-spring assembly reproduces the real member stiffness
    (Ibarra & Krawinkler 2005; Zareian & Medina 2010). Without it the member
    is softer than intended by exactly that factor.

    Only the member-stiffness calibration carries this correction. The
    yield-rotation mode sets Ke from My/theta_y, which is not referenced to
    6EI/L, so no correction applies.
    """
    mode = getattr(sp, "IMK_HINGE_STIFFNESS_MODE", "yield_rotation")
    if mode != "member_stiffness_factor":
        return 1.0

    n = float(sp.IMK_HINGE_STIFFNESS_FACTOR)
    if n <= 0.0:
        raise ValueError("IMK_HINGE_STIFFNESS_FACTOR must be positive.")
    return (n + 1.0) / n


def _orientation(member_type):
    if member_type == "column":
        return (0.0, 0.0, 1.0, 1.0, 0.0, 0.0), (1, 2, 3, 6)

    if member_type == "beam_x":
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1, 2, 3, 4)

    if member_type == "beam_y":
        return (0.0, 1.0, 0.0, -1.0, 0.0, 0.0), (1, 2, 3, 5)

    raise ValueError(f"Unknown member_type: {member_type}")


def _define_imk_peak_material(mat_tag, elastic_stiffness, yield_moment, backbone=None,
                              yield_moment_negative=None):
    """
    Define OpenSees IMKBilin using the current OpenSees argument order:

        Ke, dp_pos, dpc_pos, du_pos, Fy_pos, FmaxFy_pos, FresFy_pos,
        dp_neg, dpc_neg, du_neg, Fy_neg, FmaxFy_neg, FresFy_neg,
        Lamda_S, Lamda_C, Lamda_K, c_S, c_C, c_K, D_pos, D_neg

    All positive/negative-direction backbone parameters are passed as positive
    values, matching the OpenSees IMKBilin documentation. ``yield_moment`` is
    the positive-deformation strength; ``yield_moment_negative`` (default:
    the same) the negative-deformation strength.
    """
    if yield_moment_negative is None:
        yield_moment_negative = yield_moment

    # Rotation capacities come from the per-member backbone when one is
    # supplied. Falling back to the globals keeps the uncalibrated path usable.
    theta_p = backbone["theta_p"] if backbone else sp.IMK_THETA_P_POS
    theta_pc = backbone["theta_pc"] if backbone else sp.IMK_THETA_PC_POS
    theta_u = backbone["theta_u"] if backbone else sp.IMK_THETA_U_POS

    fmaxfy_pos = getattr(sp, "IMK_FMAXFY_POS", 1.10)
    fmaxfy_neg = getattr(sp, "IMK_FMAXFY_NEG", 1.10)
    fresfy_pos = getattr(sp, "IMK_FRESFY_POS", sp.IMK_RES_POS)
    fresfy_neg = getattr(sp, "IMK_FRESFY_NEG", sp.IMK_RES_NEG)

    ops.uniaxialMaterial(
        sp.IMK_MATERIAL_TYPE,
        mat_tag,
        elastic_stiffness,
        theta_p,
        theta_pc,
        theta_u,
        yield_moment,
        fmaxfy_pos,
        fresfy_pos,
        theta_p,
        theta_pc,
        theta_u,
        yield_moment_negative,
        fmaxfy_neg,
        fresfy_neg,
        sp.IMK_LAMBDA_S,
        sp.IMK_LAMBDA_C,
        sp.IMK_LAMBDA_K,
        sp.IMK_C_S,
        sp.IMK_C_C,
        sp.IMK_C_K,
        sp.IMK_D_POS,
        sp.IMK_D_NEG,
    )


def _create_hinge_node(source_node, hinge_node):
    ops.node(hinge_node, *ops.nodeCoord(source_node))


def _create_end_hinge(
    ele_tag, end_id, retained_node, hinge_node, member_type, props, length, backbone=None
):
    orient, tied_dofs = _orientation(member_type)
    for dof in tied_dofs:
        ops.equalDOF(retained_node, hinge_node, dof)

    mat_y = imk_material_tag(ele_tag, end_id, 5)
    mat_z = imk_material_tag(ele_tag, end_id, 6)
    ke_y = imk_hinge_stiffness(member_type, "rot_y", length)
    ke_z = imk_hinge_stiffness(member_type, "rot_z", length)

    # Beam hinges are asymmetric. Measured on the zeroLength springs (see
    # tests/test_beam_hinge_asymmetry.py): hogging is POSITIVE spring
    # deformation at end i and NEGATIVE at end j, for beam_x and beam_y alike.
    # Both strengths can differ between the ends: at an exterior end whose
    # slab bars are not developed at the perimeter the hinge yields without
    # them in either sign (beam_yield_moments).
    hogging = props.get("my_hogging_i" if end_id == 1 else "my_hogging_j", props.get("my_hogging", props["my"]))
    sagging = props.get("my_sagging_i" if end_id == 1 else "my_sagging_j", props.get("my_sagging", props["my"]))
    if end_id == 1:
        positive, negative = hogging, sagging
    else:
        positive, negative = sagging, hogging
    _define_imk_peak_material(mat_y, ke_y, positive, backbone, negative)
    _define_imk_peak_material(mat_z, ke_z, props["mz"], backbone)

    ops.element(
        "zeroLength",
        hinge_element_tag(ele_tag, end_id),
        retained_node,
        hinge_node,
        "-mat",
        mat_y,
        mat_z,
        "-dir",
        5,
        6,
        "-orient",
        *orient,
    )


def create_imk_member(ele_tag, n_i, n_j, member_type, transf_tag):
    # A column's backbone depends on how hard it is being squeezed, so the
    # gravity axial load is estimated from tributary area before its hinge
    # properties are fixed. Beams carry no meaningful axial force.
    axial_kip = 0.0
    if member_type == "column":
        story_index, grid_i, grid_j = column_grid_position(n_i)
        axial_kip = column_gravity_axial(story_index, grid_i, grid_j)

    props = _member_properties(member_type, axial_kip=axial_kip)
    strength_basis = None
    if member_type in ("beam_x", "beam_y"):
        hogging, sagging, strength_basis = beam_yield_moments(member_type, n_i, n_j)
        props.update(my_hogging=hogging, my_sagging=sagging, my=max(hogging, sagging),
                     my_hogging_i=strength_basis.get("hogging_i_kip_in", hogging),
                     my_hogging_j=strength_basis.get("hogging_j_kip_in", hogging),
                     my_sagging_i=strength_basis.get("sagging_i_kip_in", sagging),
                     my_sagging_j=strength_basis.get("sagging_j_kip_in", sagging))
    backbone = backbone_for_member(member_type, axial_kip=axial_kip)
    length = _member_length(n_i, n_j)
    i_hinge_node = hinge_node_tag(ele_tag, 1)
    j_hinge_node = hinge_node_tag(ele_tag, 2)

    _create_hinge_node(n_i, i_hinge_node)
    _create_hinge_node(n_j, j_hinge_node)
    _create_end_hinge(ele_tag, 1, n_i, i_hinge_node, member_type, props, length, backbone)
    _create_end_hinge(ele_tag, 2, n_j, j_hinge_node, member_type, props, length, backbone)

    _HINGE_REGISTRY[int(ele_tag)] = {
        "ele_tag": int(ele_tag),
        "member_type": member_type,
        # The element spans the two hinge nodes, so ops.eleNodes reports those
        # rather than the structural joints. Anything assembling forces at
        # joints needs the physical end nodes recorded here.
        "node_i": int(n_i),
        "node_j": int(n_j),
        "length_in": length,
        "yield_moment_y_kip_in": props["my"],
        "yield_moment_y_hogging_kip_in": props.get("my_hogging", props["my"]),
        "yield_moment_y_hogging_i_kip_in": props.get("my_hogging_i", props.get("my_hogging", props["my"])),
        "yield_moment_y_hogging_j_kip_in": props.get("my_hogging_j", props.get("my_hogging", props["my"])),
        "yield_moment_y_sagging_kip_in": props.get("my_sagging", props["my"]),
        "yield_moment_y_sagging_i_kip_in": props.get("my_sagging_i", props.get("my_sagging", props["my"])),
        "yield_moment_y_sagging_j_kip_in": props.get("my_sagging_j", props.get("my_sagging", props["my"])),
        "exterior_slab_anchorage": (strength_basis or {}).get("exterior_anchorage"),
        "yield_moment_z_kip_in": props["mz"],
        "strength_basis": (strength_basis or {}).get("basis"),
        "beam_family": (strength_basis or {}).get("family"),
        "theta_y_target": props["theta_y"],
        # The spring's OWN elastic limit. theta_y_target is a member-level
        # nominal (a fixed 0.004 for columns, 0.005 for beams), but the spring
        # is deliberately IMK_HINGE_STIFFNESS_FACTOR times the member
        # stiffness, so it yields at My/Ke -- 26x to 65x smaller across the
        # pilot. Recorded hinge rotation is spring rotation, so plastic
        # rotation must be measured from this, not from the member nominal.
        # With asymmetric beam strengths the spring yields first in the
        # weaker (sagging) direction; that is the rotation the yielded flag
        # must be measured against.
        "theta_y_spring_y": (
            min(props.get("my_hogging", props["my"]), props.get("my_sagging", props["my"]))
            / imk_hinge_stiffness(member_type, "rot_y", length)
            if imk_hinge_stiffness(member_type, "rot_y", length) > 0 else 0.0
        ),
        # Per end, where the ends differ (exterior ends without developed
        # slab bars): the spring at that end yields in its weaker direction.
        "theta_y_spring_y_i": (
            min(props.get("my_hogging_i", props.get("my_hogging", props["my"])),
                props.get("my_sagging_i", props.get("my_sagging", props["my"])))
            / imk_hinge_stiffness(member_type, "rot_y", length)
            if imk_hinge_stiffness(member_type, "rot_y", length) > 0 else 0.0
        ),
        "theta_y_spring_y_j": (
            min(props.get("my_hogging_j", props.get("my_hogging", props["my"])),
                props.get("my_sagging_j", props.get("my_sagging", props["my"])))
            / imk_hinge_stiffness(member_type, "rot_y", length)
            if imk_hinge_stiffness(member_type, "rot_y", length) > 0 else 0.0
        ),
        "theta_y_spring_y_hogging": (
            props.get("my_hogging", props["my"]) / imk_hinge_stiffness(member_type, "rot_y", length)
            if imk_hinge_stiffness(member_type, "rot_y", length) > 0 else 0.0
        ),
        "theta_y_spring_z": (
            props["mz"] / imk_hinge_stiffness(member_type, "rot_z", length)
            if imk_hinge_stiffness(member_type, "rot_z", length) > 0 else 0.0
        ),
        "stiffness_modifier": props["stiffness_modifier"],
        **backbone,
    }

    inertia_factor = imk_elastic_inertia_factor()
    ops.element(
        "elasticBeamColumn",
        ele_tag,
        i_hinge_node,
        j_hinge_node,
        props["area"],
        props["e"],
        props["g"],
        props["j"],
        inertia_factor * props["iy"],
        inertia_factor * props["iz"],
        transf_tag,
    )
