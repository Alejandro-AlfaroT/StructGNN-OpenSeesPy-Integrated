import math

import openseespy.opensees as ops

import Structure_Parameters as sp


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


def _member_properties(member_type):
    if member_type == "column":
        return {
            "area": sp.rect_area(sp.B_COL, sp.H_COL),
            "e": sp.concrete_ec_ksi(sp.FC_COL_KSI),
            "g": sp.concrete_shear_modulus_ksi(sp.concrete_ec_ksi(sp.FC_COL_KSI)),
            "j": sp.approx_rect_j(sp.B_COL, sp.H_COL),
            "iy": sp.rect_iy(sp.B_COL, sp.H_COL),
            "iz": sp.rect_iz(sp.B_COL, sp.H_COL),
            "my": sp.column_nominal_moment_y(),
            "mz": sp.column_nominal_moment_z(),
            "theta_y": sp.IMK_COLUMN_THETA_Y,
        }

    return {
        "area": sp.rect_area(sp.B_BEAM, sp.H_BEAM),
        "e": sp.concrete_ec_ksi(sp.FC_BEAM_KSI),
        "g": sp.concrete_shear_modulus_ksi(sp.concrete_ec_ksi(sp.FC_BEAM_KSI)),
        "j": sp.approx_rect_j(sp.B_BEAM, sp.H_BEAM),
        "iy": sp.rect_iy(sp.B_BEAM, sp.H_BEAM),
        "iz": sp.rect_iz(sp.B_BEAM, sp.H_BEAM),
        "my": sp.beam_nominal_moment_y(),
        "mz": sp.beam_nominal_moment_z(),
        "theta_y": sp.IMK_BEAM_THETA_Y,
    }


def imk_member_properties(member_type):
    return _member_properties(member_type).copy()


def _elastic_rotational_stiffness(e_mod, inertia, length):
    return 6.0 * e_mod * inertia / length


def imk_hinge_stiffness(member_type, rot_dir, length):
    props = _member_properties(member_type)

    if rot_dir in (5, "rot_y"):
        yield_moment = props["my"]
        inertia = props["iy"]
    elif rot_dir in (6, "rot_z"):
        yield_moment = props["mz"]
        inertia = props["iz"]
    else:
        raise ValueError(f"Unknown IMK hinge rotation direction: {rot_dir}")

    yield_based_stiffness = yield_moment / props["theta_y"]
    member_based_stiffness = (
        sp.IMK_HINGE_STIFFNESS_FACTOR
        * _elastic_rotational_stiffness(props["e"], inertia, length)
    )

    return max(yield_based_stiffness, member_based_stiffness)


def imk_hinge_thresholds(member_type, rot_dir, length):
    props = _member_properties(member_type)

    if rot_dir in (5, "rot_y"):
        yield_moment = props["my"]
    elif rot_dir in (6, "rot_z"):
        yield_moment = props["mz"]
    else:
        raise ValueError(f"Unknown IMK hinge rotation direction: {rot_dir}")

    theta_y = yield_moment / imk_hinge_stiffness(member_type, rot_dir, length)
    theta_p = max(sp.IMK_THETA_P_POS, sp.IMK_THETA_P_NEG)

    return {
        "theta_y": theta_y,
        "theta_cap": theta_y + theta_p,
        "theta_u": max(sp.IMK_THETA_U_POS, sp.IMK_THETA_U_NEG),
    }


def _orientation(member_type):
    if member_type == "column":
        return (0.0, 0.0, 1.0, 1.0, 0.0, 0.0), (1, 2, 3, 6)

    if member_type == "beam_x":
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1, 2, 3, 4)

    if member_type == "beam_y":
        return (0.0, 1.0, 0.0, -1.0, 0.0, 0.0), (1, 2, 3, 5)

    raise ValueError(f"Unknown member_type: {member_type}")


def _define_imk_peak_material(mat_tag, elastic_stiffness, yield_moment):
    ops.uniaxialMaterial(
        sp.IMK_MATERIAL_TYPE,
        mat_tag,
        elastic_stiffness,
        sp.IMK_POST_YIELD_RATIO_POS,
        sp.IMK_POST_YIELD_RATIO_NEG,
        yield_moment,
        yield_moment,
        sp.IMK_LAMBDA_S,
        sp.IMK_LAMBDA_C,
        sp.IMK_LAMBDA_A,
        sp.IMK_LAMBDA_K,
        sp.IMK_C_S,
        sp.IMK_C_C,
        sp.IMK_C_A,
        sp.IMK_C_K,
        sp.IMK_THETA_P_POS,
        sp.IMK_THETA_P_NEG,
        sp.IMK_THETA_PC_POS,
        sp.IMK_THETA_PC_NEG,
        sp.IMK_RES_POS,
        sp.IMK_RES_NEG,
        sp.IMK_THETA_U_POS,
        sp.IMK_THETA_U_NEG,
        sp.IMK_D_POS,
        sp.IMK_D_NEG,
    )


def _create_hinge_node(source_node, hinge_node):
    ops.node(hinge_node, *ops.nodeCoord(source_node))


def _create_end_hinge(ele_tag, end_id, retained_node, hinge_node, member_type, props, length):
    orient, tied_dofs = _orientation(member_type)
    for dof in tied_dofs:
        ops.equalDOF(retained_node, hinge_node, dof)

    mat_y = imk_material_tag(ele_tag, end_id, 5)
    mat_z = imk_material_tag(ele_tag, end_id, 6)
    ke_y = imk_hinge_stiffness(member_type, "rot_y", length)
    ke_z = imk_hinge_stiffness(member_type, "rot_z", length)

    _define_imk_peak_material(mat_y, ke_y, props["my"])
    _define_imk_peak_material(mat_z, ke_z, props["mz"])

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
    props = _member_properties(member_type)
    length = _member_length(n_i, n_j)
    i_hinge_node = hinge_node_tag(ele_tag, 1)
    j_hinge_node = hinge_node_tag(ele_tag, 2)

    _create_hinge_node(n_i, i_hinge_node)
    _create_hinge_node(n_j, j_hinge_node)
    _create_end_hinge(ele_tag, 1, n_i, i_hinge_node, member_type, props, length)
    _create_end_hinge(ele_tag, 2, n_j, j_hinge_node, member_type, props, length)

    ops.element(
        "elasticBeamColumn",
        ele_tag,
        i_hinge_node,
        j_hinge_node,
        props["area"],
        props["e"],
        props["g"],
        props["j"],
        props["iy"],
        props["iz"],
        transf_tag,
    )
