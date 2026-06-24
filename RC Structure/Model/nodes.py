#==============================
#            NODES
#==============================


import Structure_Parameters as sp
import openseespy.opensees as ops


def node_tag(k, i, j):
    return (
        k * ((sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1))
        + j * (sp.NUM_BAY_X + 1)
        + i
        + 1
    )


def diaphragm_master_tag(k):
    return 100000 + k


def create_nodes():
    for k in range(sp.NUM_FLOOR + 1):
        z = k * sp.STORY_H
        for j in range(sp.NUM_BAY_Y + 1):
            y = j * sp.BAY_Y
            for i in range(sp.NUM_BAY_X + 1):
                x = i * sp.BAY_X
                ops.node(node_tag(k, i, j), x, y, z)

    x_center = sp.NUM_BAY_X * sp.BAY_X / 2.0
    y_center = sp.NUM_BAY_Y * sp.BAY_Y / 2.0
    for k in range(1, sp.NUM_FLOOR + 1):
        ops.node(diaphragm_master_tag(k), x_center, y_center, k * sp.STORY_H)


def fix_base_nodes():
    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            ops.fix(node_tag(0, i, j), 1, 1, 1, 1, 1, 1)


def roof_master_node():
    return diaphragm_master_tag(sp.NUM_FLOOR)
