"""Diagnostic 3D scissors panel with two explicitly calibrated IMKPinching laws.

Global Z is vertical. This small-rotation, orthogonal joint-shear idealization
is deliberately separate from the production frame builder. No member slip,
joint strength, stiffness, mass, or pinching parameters are inferred here.
See JOINT_PANEL.md for kinematics, work conjugacy, and applicability limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import openseespy.opensees as ops

from Model.IMK_Materials import CyclicParameters, RotationalBackbone, define_rotational_imk


def _positive(name, value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class JointGeometry:
    center: tuple[float, float, float]
    dx: float
    dy: float
    hz: float

    def validate(self):
        if len(self.center) != 3 or not all(math.isfinite(v) for v in self.center):
            raise ValueError("Joint center must contain three finite coordinates in inches")
        for key in ("dx", "dy", "hz"):
            _positive(key, getattr(self, key))

    @property
    def volume(self):
        return self.dx * self.dy * self.hz

    def offsets(self):
        return {"x_minus": (-self.dx / 2, 0., 0.), "x_plus": (self.dx / 2, 0., 0.),
                "y_minus": (0., -self.dy / 2, 0.), "y_plus": (0., self.dy / 2, 0.),
                "z_minus": (0., 0., -self.hz / 2), "z_plus": (0., 0., self.hz / 2)}


@dataclass(frozen=True)
class JointTags:
    column_core: int
    beam_core: int
    x_minus: int
    x_plus: int
    y_minus: int
    y_plus: int
    z_minus: int
    z_plus: int
    element: int
    material_rx: int
    material_ry: int

    def nodes(self):
        return {key: value for key, value in asdict(self).items()
                if key not in ("element", "material_rx", "material_ry")}

    def validate(self):
        for key, value in asdict(self).items():
            _positive(key, value)
            if not isinstance(value, int):
                raise ValueError(f"{key} must be an integer tag")
        if len(set(self.nodes().values())) != 8:
            raise ValueError("All eight panel nodes must have distinct tags")
        if self.material_rx == self.material_ry:
            raise ValueError("The two joint planes require distinct material tags")


@dataclass(frozen=True)
class JointShearCalibration:
    ke: float
    positive: RotationalBackbone
    negative: RotationalBackbone
    cyclic: CyclicParameters
    calibration_id: str
    status: str
    source_refs: tuple[str, ...]
    deformation_scope: str = "joint_shear_only"

    def provenance(self, plane):
        return {"calibration_id": self.calibration_id, "status": self.status,
                "deformation_scope": self.deformation_scope, "source_refs": list(self.source_refs),
                "plane": plane, "coordinate": "theta_beam_core - theta_column_core",
                "topology": "orthogonal_scissors_3d_v1"}

    def validate(self):
        _positive("ke", self.ke)
        for branch in (self.positive, self.negative):
            branch.arguments()
            if branch.du <= branch.fy / self.ke:
                raise ValueError("Ultimate joint rotation must exceed its elastic yield rotation")
        self.cyclic.arguments("IMKPinching")
        for name in ("calibration_id", "status"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"An explicit {name} is required")
        if not self.source_refs or any(not isinstance(v, str) or not v.strip() for v in self.source_refs):
            raise ValueError("Identify the parameter source, including synthetic fixture sources")
        if self.deformation_scope != "joint_shear_only":
            raise ValueError("This panel supports joint shear only; reconcile member slip before adding slip")
        json.dumps(self.provenance("validation"), allow_nan=False)


def build_joint_panel(geometry: JointGeometry, tags: JointTags, *, rx: JointShearCalibration,
                      ry: JointShearCalibration):
    """Add eight massless nodes and one two-material spring to a 3D/6DOF domain.

    The caller owns tag allocation, supports, loads, and constraint handling.
    All tags must be unused; material-tag uniqueness against the existing
    domain remains the caller's responsibility (OpenSees checks installation).
    No fixities, diaphragm, analysis settings, or physical member are created.
    """
    geometry.validate()
    tags.validate()
    rx.validate()
    ry.validate()
    if ops.getNDM() != [3] or ops.getNDF() != [6]:
        raise ValueError("Joint panel requires an active 3D model with six DOFs per node")
    if set(tags.nodes().values()).intersection(ops.getNodeTags()):
        raise ValueError("Panel node tags collide with existing domain nodes")
    if tags.element in ops.getEleTags():
        raise ValueError("Panel element tag already exists")
    installed = {}
    for name, calibration, mat_tag, plane in (("rx", rx, tags.material_rx, "yz"),
                                              ("ry", ry, tags.material_ry, "xz")):
        installed[name] = define_rotational_imk(
            "IMKPinching", mat_tag, calibration.ke, calibration.positive, calibration.negative,
            calibration.cyclic, provenance=calibration.provenance(plane))
    ops.node(tags.column_core, *geometry.center)
    ops.node(tags.beam_core, *geometry.center)
    # The cores share translations and Rz, but Rx/Ry remain independent.
    ops.equalDOF(tags.column_core, tags.beam_core, 1, 2, 3, 6)
    ops.element("zeroLength", tags.element, tags.column_core, tags.beam_core,
                "-mat", tags.material_rx, tags.material_ry, "-dir", 4, 5,
                "-orient", 1., 0., 0., 0., 1., 0., "-doRayleigh", 0)
    arms = []
    for face, offset in geometry.offsets().items():
        node = getattr(tags, face)
        ops.node(node, *(c + d for c, d in zip(geometry.center, offset)))
        core = tags.column_core if face.startswith("z_") else tags.beam_core
        ops.rigidLink("beam", core, node)
        arms.append({"core": core, "face": node, "offset_in": list(offset)})
    return {"schema_version": "orthogonal_scissors_3d_v1", "status": "diagnostic_subassembly",
            "geometry_in": asdict(geometry), "volume_in3": geometry.volume,
            "tags": asdict(tags), "installed_materials": installed, "rigid_arms": arms,
            "units": {"length": "in", "rotation": "rad", "moment": "kip-in", "stress": "ksi"},
            "plane_mapping": {"yz": {"direction": 4, "gamma_sign": 1},
                              "xz": {"direction": 5, "gamma_sign": -1}},
            "added_mass": 0., "bond_slip_included": False, "rayleigh_in_spring": False,
            "limitations": ["small_rotation_rigid_arms", "global_axis_aligned_common_panel_height",
                            "independent_yz_xz_shear_laws", "rigid_xy_distortion",
                            "four_beam_faces_share_one_core_rotation_and_torsion_kinematics",
                            "no_axial_or_biaxial_strength_interaction", "no_frame_integration"]}


def joint_response(panel):
    """Read conjugate spring response and independently reconstruct face shear.

    The equivalent stress conversion applies to a uniform rectangular panel
    of the stated dimensions. It does not determine effective joint width or
    turn a specimen's applied lateral force into panel shear demand.
    """
    tags = panel["tags"]
    geo = panel["geometry_in"]
    q = list(ops.eleResponse(tags["element"], "deformation"))
    moment = list(ops.eleResponse(tags["element"], "basicForce"))
    if len(q) != 2 or len(moment) != 2:
        raise RuntimeError("Expected two rotational responses from the joint spring")
    def disp(face, dof):
        return ops.nodeDisp(tags[face], dof)
    gamma_yz = ((disp("z_plus", 2) - disp("z_minus", 2)) / geo["hz"]
                + (disp("y_plus", 3) - disp("y_minus", 3)) / geo["dy"])
    gamma_xz = ((disp("z_plus", 1) - disp("z_minus", 1)) / geo["hz"]
                + (disp("x_plus", 3) - disp("x_minus", 3)) / geo["dx"])
    return {"q_rx": q[0], "q_ry": q[1], "moment_rx_kip_in": moment[0], "moment_ry_kip_in": moment[1],
            "gamma_yz_from_faces": gamma_yz, "gamma_xz_from_faces": gamma_xz,
            "tau_yz_ksi": moment[0] / panel["volume_in3"],
            "tau_xz_ksi": -moment[1] / panel["volume_in3"]}


def clear_span_between_panels(i: JointGeometry, j: JointGeometry, axis: str):
    """Return the positive face-to-face span for collinear, axis-aligned panels.

    Geometry helper only: it does not modify existing member stiffness,
    calibration, gravity loads, member identity, or mass assignment.
    """
    i.validate()
    j.validate()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y, or z")
    k = "xyz".index(axis)
    if any(abs(j.center[d] - i.center[d]) > 1e-9 for d in range(3) if d != k):
        raise ValueError("Panel centers must be collinear along the selected global axis")
    dimension = ("dx", "dy", "hz")[k]
    span = abs(j.center[k] - i.center[k]) - (getattr(i, dimension) + getattr(j, dimension)) / 2
    _positive("clear span", span)
    return span
