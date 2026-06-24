#============================
#        Diaphragms
#============================

import Structure_Parameters as sp
import openseespy.opensees as ops
from Model.nodes import diaphragm_master_tag, node_tag


def floor_master_node(k):
    return diaphragm_master_tag(k)


def floor_nodes(k):
    nodes = []

    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            nodes.append(node_tag(k, i, j))

    return nodes


def create_rigid_diaphragms():
    for k in range(1, sp.NUM_FLOOR + 1):
        master = floor_master_node(k)
        slaves = [n for n in floor_nodes(k) if n != master]

        ops.fix(master, 0, 0, 1, 1, 1, 0)
        ops.rigidDiaphragm(3, master, *slaves)
