#==============================
#           Mass
#==============================


import Structure_Parameters as sp
import openseespy.opensees as ops
from Model.nodes import node_tag


def assign_nodal_masses():
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                n = node_tag(k, i, j)
                m = sp.node_seismic_mass(i, j)
                ops.mass(n, m, m, 1.0e-8, 0.0, 0.0, 0.0)
