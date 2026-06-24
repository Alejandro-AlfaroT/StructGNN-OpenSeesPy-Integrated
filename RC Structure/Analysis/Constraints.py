import openseespy.opensees as ops

import Structure_Parameters as sp


def apply_analysis_constraints():
    if sp.ELEMENT_FORMULATION == "imk":
        ops.constraints("Penalty", sp.PENALTY_ALPHA_SP, sp.PENALTY_ALPHA_MP)
        return

    ops.constraints("Transformation")
