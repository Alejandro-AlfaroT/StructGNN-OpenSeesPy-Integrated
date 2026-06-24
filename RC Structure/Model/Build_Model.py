#==========================
#      Model Builder
#==========================

import openseespy.opensees as ops
from Model.Sections import define_sections
from Model.diaphragms import create_rigid_diaphragms
from Model.elements import create_elements
from Model.mass import assign_nodal_masses
from Model.nodes import create_nodes, fix_base_nodes


def build_model():
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    create_nodes()
    fix_base_nodes()
    define_sections()
    create_elements()
    create_rigid_diaphragms()
    assign_nodal_masses()