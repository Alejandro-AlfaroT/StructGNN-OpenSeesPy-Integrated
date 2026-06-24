import Structure_Parameters as sp
import openseespy.opensees as ops
from Model.IMK_Hinges import create_imk_member
from Model.nodes import node_tag


def define_geometric_transformations():
    ops.geomTransf("PDelta", sp.COL_TRANSF_TAG, 1, 0, 0)
    ops.geomTransf("Linear", sp.BEAM_X_TRANSF_TAG, 0, 0, 1)
    ops.geomTransf("Linear", sp.BEAM_Y_TRANSF_TAG, 0, 0, 1)


def _create_frame_member(ele_tag, n_i, n_j, member_type, transf_tag, integ_tag):
    use_imk = (
        sp.ELEMENT_FORMULATION == "imk"
        and (
            (member_type == "column" and sp.IMK_APPLY_TO_COLUMNS)
            or (member_type in {"beam_x", "beam_y"} and sp.IMK_APPLY_TO_BEAMS)
        )
    )

    if use_imk:
        create_imk_member(ele_tag, n_i, n_j, member_type, transf_tag)
        return

    if sp.ELEMENT_FORMULATION in {"fiber", "imk"}:
        ops.element(
            "forceBeamColumn",
            ele_tag,
            n_i,
            n_j,
            transf_tag,
            integ_tag,
        )
        return

    raise ValueError(f"Unknown ELEMENT_FORMULATION: {sp.ELEMENT_FORMULATION}")


def create_column_elements(start_ele_tag=1):
    ele_tag = start_ele_tag

    for k in range(sp.NUM_FLOOR):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k + 1, i, j)

                _create_frame_member(
                    ele_tag,
                    n_i,
                    n_j,
                    "column",
                    sp.COL_TRANSF_TAG,
                    sp.COL_INTEG_TAG,
                )
                ele_tag += 1

    return ele_tag


def create_beam_x_elements(start_ele_tag):
    ele_tag = start_ele_tag

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k, i + 1, j)

                _create_frame_member(
                    ele_tag,
                    n_i,
                    n_j,
                    "beam_x",
                    sp.BEAM_X_TRANSF_TAG,
                    sp.BEAM_INTEG_TAG,
                )
                ele_tag += 1

    return ele_tag


def create_beam_y_elements(start_ele_tag):
    ele_tag = start_ele_tag

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k, i, j + 1)

                _create_frame_member(
                    ele_tag,
                    n_i,
                    n_j,
                    "beam_y",
                    sp.BEAM_Y_TRANSF_TAG,
                    sp.BEAM_INTEG_TAG,
                )
                ele_tag += 1

    return ele_tag


def create_elements():
    define_geometric_transformations()

    next_ele_tag = create_column_elements(start_ele_tag=1)
    next_ele_tag = create_beam_x_elements(next_ele_tag)
    create_beam_y_elements(next_ele_tag)
