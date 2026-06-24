import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.diaphragms import floor_master_node


def apply_lateral_loads():
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)

    for k in range(1, sp.NUM_FLOOR + 1):
        master = floor_master_node(k)
        ops.load(master, sp.FX_FLOOR, 0.0, 0.0, 0.0, 0.0, 0.0)
