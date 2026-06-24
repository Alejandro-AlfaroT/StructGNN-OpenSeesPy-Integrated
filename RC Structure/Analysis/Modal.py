import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Constraints import apply_analysis_constraints
from Model.nodes import roof_master_node


def run_modal_analysis():
    ops.wipeAnalysis()
    apply_analysis_constraints()
    ops.numberer("RCM")
    ops.system("BandGeneral")

    eigenvalues = ops.eigen(sp.NUM_MODES)
    roof_node = roof_master_node()

    modes = []
    for mode, eigenvalue in enumerate(eigenvalues, start=1):
        if eigenvalue <= sp.EIGENVALUE_TOL:
            modes.append(
                {
                    "mode": mode,
                    "lambda": eigenvalue,
                    "valid": False,
                    "period": None,
                    "omega": None,
                    "frequency": None,
                    "roof_eigenvector": None,
                }
            )
            continue

        omega = math.sqrt(eigenvalue)
        period = 2.0 * math.pi / omega
        frequency = omega / (2.0 * math.pi)

        modes.append(
            {
                "mode": mode,
                "lambda": eigenvalue,
                "valid": True,
                "period": period,
                "omega": omega,
                "frequency": frequency,
                "roof_eigenvector": (
                    ops.nodeEigenvector(roof_node, mode, 1),
                    ops.nodeEigenvector(roof_node, mode, 2),
                    ops.nodeEigenvector(roof_node, mode, 3),
                ),
            }
        )

    return modes
