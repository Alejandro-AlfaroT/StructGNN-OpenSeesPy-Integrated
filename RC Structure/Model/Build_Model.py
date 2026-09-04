#==========================
#      Model Builder
#==========================

import openseespy.opensees as ops
from Model.Sections import define_sections
from Model.diaphragms import create_rigid_diaphragms
from Model.IMK_Hinges import reset_hinge_registry
from Model.elements import create_elements
from Model.mass import assign_nodal_masses
from Model.nodes import create_nodes, fix_base_nodes


def build_model():
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # Hinge backbones are per-build state: sections and axial loads change
    # between builds during the design search, so stale entries would
    # misreport what the current model actually contains.
    reset_hinge_registry()

    create_nodes()
    fix_base_nodes()
    define_sections()
    create_elements()
    create_rigid_diaphragms()
    assign_nodal_masses()