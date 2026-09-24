"""Mechanical checks of the diagnostic 3D panel; synthetic inputs, not RC fits."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Model.Joint_Panel import (JointGeometry, JointTags, JointShearCalibration,
                               build_joint_panel, joint_response, clear_span_between_panels)
from Model.IMK_Materials import CyclicParameters, RotationalBackbone, define_rotational_imk

GEOMETRY = JointGeometry((3., -5., 100.), 24., 30., 20.)
TAGS = JointTags(1, 2, 3, 4, 5, 6, 7, 8, 21, 11, 12)
RESPONSE_COLUMNS = ("q_rx", "q_ry", "moment_rx_kip_in", "moment_ry_kip_in",
                    "gamma_yz_from_faces", "gamma_xz_from_faces", "tau_yz_ksi", "tau_xz_ksi")


def fixture_calibration(scale=1.):
    return JointShearCalibration(
        ke=1e5 * scale,
        positive=RotationalBackbone(.01, .05, .09, 100. * scale, 1.1, .2),
        negative=RotationalBackbone(.01, .05, .09, 70. * scale, 1.1, .2),
        cyclic=CyclicParameters(10., 10., 10., 1., 1., 1., 1., 1.,
                                lamda_a=10., c_a=1., kappa_f=.5, kappa_d=.5),
        calibration_id=f"synthetic_joint_mechanics_v1_scale_{scale:g}",
        status="verification_only_not_experimental",
        source_refs=("tests/test_joint_panel.py synthetic fixture; not a specimen fit",))


def make_panel():
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    return build_joint_panel(GEOMETRY, TAGS, rx=fixture_calibration(), ry=fixture_calibration(1.7))


def analysis(handler="Lagrange", penalty=1e12):
    # Lagrange enforces chained rigidLink/equalDOF constraints without penalty
    # compliance. Its matrix is indefinite: use a general (not SPD) solver.
    if handler == "Penalty":
        ops.constraints("Penalty", penalty, penalty)
    else:
        ops.constraints("Lagrange", 1e5, 1e5)
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-10, 60)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.)
    ops.analysis("Static")


def arm_residual(panel):
    error = 0.
    for arm in panel["rigid_arms"]:
        core = np.array(ops.nodeDisp(arm["core"]))
        face = np.array(ops.nodeDisp(arm["face"]))
        expected = np.r_[core[:3] + np.cross(core[3:], arm["offset_in"]), core[3:]]
        error = max(error, float(np.max(np.abs(face - expected))))
    return error


def run_prescribed_joint(history, *, handler="Lagrange", penalty=1e12):
    """Prescribe the two core rotations; record every committed equilibrium state."""
    path = np.asarray(history, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not np.all(path[0] == 0):
        raise ValueError("history must start at [0, 0] and have two rotation columns")
    panel = make_panel()
    try:
        ops.fix(TAGS.column_core, 1, 1, 1, 1, 1, 1)
        for index, dof in enumerate((4, 5)):
            ops.timeSeries("Path", index + 1, "-dt", 1., "-values", *path[:, index].tolist())
            ops.pattern("Plain", index + 1, index + 1)
            ops.sp(TAGS.beam_core, dof, 1.)
        analysis(handler, penalty)
        rows = [np.zeros(len(RESPONSE_COLUMNS))]
        max_arm_error = 0.
        for _ in path[1:]:
            if ops.analyze(1) != 0:
                raise AssertionError(f"Panel failed at pseudo-time {ops.getTime()}")
            response = joint_response(panel)
            rows.append([response[key] for key in RESPONSE_COLUMNS])
            max_arm_error = max(max_arm_error, arm_residual(panel))
            q_nodes = np.array(ops.nodeDisp(TAGS.beam_core))[3:5] - np.array(ops.nodeDisp(TAGS.column_core))[3:5]
            np.testing.assert_allclose(rows[-1][:2], q_nodes, atol=1e-11, rtol=1e-10)
        return np.asarray(rows), panel, max_arm_error
    finally:
        ops.wipe()


def run_transverse_torsion_probe(stiffness_ratio):
    """Quantify the shared beam-core assumption, not physical RC accuracy.

    Two elastic X beams restrain Rx through torsion while a moment is applied
    to the beam core in the Y-Z panel plane. Ratio = sum(GJ/L)/panel Ke.
    """
    panel = make_panel()
    try:
        ops.fix(TAGS.column_core, 1, 1, 1, 1, 1, 1)
        if stiffness_ratio > 0:
            length, shear_modulus = 216., 1500.
            torsion_j = stiffness_ratio * 1e5 * length / (2 * shear_modulus)
            ops.geomTransf("Linear", 31, 0., 0., 1.)
            for face, far_tag, ele_tag, sign in ((TAGS.x_minus, 9, 22, -1.), (TAGS.x_plus, 10, 23, 1.)):
                xyz = np.array(ops.nodeCoord(face)) + [sign * length, 0., 0.]
                ops.node(far_tag, *xyz)
                ops.fix(far_tag, 1, 1, 1, 1, 1, 1)
                ops.element("elasticBeamColumn", ele_tag, face, far_tag, 400., 3600., shear_modulus,
                            torsion_j, 30000., 40000., 31)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(TAGS.beam_core, 0., 0., 0., 10., 0., 0.)
        analysis()
        if ops.analyze(1) != 0:
            raise AssertionError("Transverse torsion probe failed")
        response = joint_response(panel)
        beam_torque = sum(ops.eleForce(tag)[3] for tag in (22, 23)) if stiffness_ratio > 0 else 0.
        return {"torsion_to_panel_stiffness_ratio": stiffness_ratio, "applied_moment_kip_in": 10.,
                "panel_rotation_rad": response["q_rx"], "panel_moment_kip_in": response["moment_rx_kip_in"],
                "transverse_beam_torque_kip_in": beam_torque,
                "x_beam_face_rx_rad": ops.nodeDisp(TAGS.x_plus, 4),
                "analytical_panel_rotation_rad": 10. / (1e5 * (1 + stiffness_ratio))}
    finally:
        ops.wipe()


class JointPanelTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_geometry_axis_assignment_and_zero_added_mass(self):
        panel = make_panel()
        self.assertEqual(ops.nodeCoord(TAGS.column_core), ops.nodeCoord(TAGS.beam_core))
        self.assertNotEqual(TAGS.column_core, TAGS.beam_core)
        self.assertEqual(ops.eleNodes(TAGS.element), [TAGS.column_core, TAGS.beam_core])
        self.assertEqual(len(ops.getNodeTags()), 8)
        for face, offset in GEOMETRY.offsets().items():
            np.testing.assert_allclose(ops.nodeCoord(getattr(TAGS, face)), np.array(GEOMETRY.center) + offset)
        for tag in ops.getNodeTags():
            self.assertEqual(ops.nodeMass(tag), [0.] * 6)
        self.assertFalse(panel["bond_slip_included"])
        self.assertEqual(panel["plane_mapping"]["yz"]["direction"], 4)
        self.assertEqual(panel["plane_mapping"]["xz"]["direction"], 5)

    def test_rejects_bad_geometry_tags_and_incomplete_calibration_before_mutation(self):
        cases = [(replace(GEOMETRY, hz=0.), TAGS, fixture_calibration()),
                 (GEOMETRY, replace(TAGS, beam_core=1), fixture_calibration()),
                 (GEOMETRY, TAGS, replace(fixture_calibration(), deformation_scope="shear_plus_slip")),
                 (GEOMETRY, TAGS, replace(fixture_calibration(), source_refs=())),
                 (GEOMETRY, TAGS, replace(fixture_calibration(), cyclic=replace(fixture_calibration().cyclic, kappa_f=None)))]
        for geo, tags, calibration in cases:
            with self.subTest(geometry=geo, tags=tags, calibration=calibration):
                ops.wipe()
                ops.model("basic", "-ndm", 3, "-ndf", 6)
                with self.assertRaises(ValueError):
                    build_joint_panel(geo, tags, rx=fixture_calibration(), ry=calibration)
                self.assertEqual(ops.getNodeTags(), [])
                self.assertEqual(ops.getEleTags(), [])
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        with self.assertRaisesRegex(ValueError, "3D"):
            build_joint_panel(GEOMETRY, TAGS, rx=fixture_calibration(), ry=fixture_calibration())
        self.assertEqual(ops.getNodeTags(), [])
        make_panel()
        with self.assertRaisesRegex(ValueError, "collide"):
            build_joint_panel(GEOMETRY, TAGS, rx=fixture_calibration(), ry=fixture_calibration())

    def test_face_load_couples_match_rotations_equilibrium_and_virtual_work(self):
        panel = make_panel()
        ops.fix(TAGS.column_core, 1, 1, 1, 1, 1, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        loads = {TAGS.y_plus: .1, TAGS.y_minus: -.1, TAGS.x_plus: -.2, TAGS.x_minus: .2}
        for tag, fz in loads.items():
            ops.load(tag, 0., 0., fz, 0., 0., 0.)
        analysis()
        self.assertEqual(ops.analyze(1), 0)
        result = joint_response(panel)
        moment = np.array([.1 * GEOMETRY.dy, .2 * GEOMETRY.dx])
        q = moment / np.array([1e5, 1.7e5])
        np.testing.assert_allclose([result["moment_rx_kip_in"], result["moment_ry_kip_in"]], moment, rtol=1e-10)
        np.testing.assert_allclose([result["q_rx"], result["q_ry"]], q, rtol=1e-10)
        np.testing.assert_allclose([result["gamma_yz_from_faces"], result["gamma_xz_from_faces"]], [q[0], -q[1]], atol=1e-12)
        face_work = sum(fz * ops.nodeDisp(tag, 3) for tag, fz in loads.items())
        self.assertAlmostEqual(face_work, float(moment @ q), places=12)
        stress_work = GEOMETRY.volume * (result["tau_yz_ksi"] * result["gamma_yz_from_faces"]
                                         + result["tau_xz_ksi"] * result["gamma_xz_from_faces"])
        self.assertAlmostEqual(stress_work, face_work, places=12)
        self.assertLess(arm_residual(panel), 1e-12)

    def test_rigid_body_motion_has_no_panel_shear(self):
        panel = make_panel()
        rigid_motion = np.array([.01, -.02, .03, .0002, -.0003, .0001])
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for dof, value in enumerate(rigid_motion, 1):
            ops.sp(TAGS.column_core, dof, float(value))
        analysis()
        self.assertEqual(ops.analyze(1), 0)
        result = joint_response(panel)
        np.testing.assert_allclose(list(result.values()), 0., atol=1e-10)
        np.testing.assert_allclose(ops.nodeDisp(TAGS.beam_core), rigid_motion, atol=1e-12)
        self.assertLess(arm_residual(panel), 1e-12)

    def test_axial_translation_and_torsion_transfer_without_spurious_joint_shear(self):
        panel = make_panel()
        ops.fix(TAGS.column_core, 1, 1, 1, 1, 1, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        # Axial face loads have no moment about the core. Rz is tied explicitly.
        ops.load(TAGS.x_plus, 7., 0., 0., 0., 0., 0.)
        ops.load(TAGS.y_plus, 0., -9., 0., 0., 0., 0.)
        ops.load(TAGS.z_plus, 0., 0., 11., 0., 0., 0.)
        ops.load(TAGS.beam_core, 0., 0., 0., 0., 0., 13.)
        analysis()
        self.assertEqual(ops.analyze(1), 0)
        np.testing.assert_allclose(list(joint_response(panel).values()), 0., atol=1e-10)
        self.assertLess(arm_residual(panel), 1e-12)
        for tag in ops.getNodeTags():
            np.testing.assert_allclose(ops.nodeDisp(tag), 0., atol=1e-12)

    def test_cyclic_biaxial_response_matches_two_independent_native_materials(self):
        from test_imk_materials import run_fixture, rotation_history
        path_x = rotation_history([.005, -.005, .02, -.02, .03, -.03, .03, -.03, 0.])
        # Different simultaneous paths exercise both directions and coupling errors.
        path_y = -.7 * path_x
        rows, _, residual = run_prescribed_joint(np.column_stack([path_x, path_y]))
        self.assertTrue(np.isfinite(rows).all())
        self.assertLess(residual, 1e-11)
        for index, (path, scale) in enumerate(((path_x, 1.), (path_y, 1.7))):
            spring, _ = run_fixture("IMKPinching", path, moment_scale=scale)
            np.testing.assert_allclose(rows[:, index], spring[:, 1], atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(rows[:, index + 2], spring[:, 0], atol=1e-7, rtol=1e-8)
        np.testing.assert_allclose(rows[:, 4], rows[:, 0], atol=1e-12)
        np.testing.assert_allclose(rows[:, 5], -rows[:, 1], atol=1e-12)
        spring_work = np.sum(np.diff(rows[:, :2], axis=0) * (rows[1:, 2:4] + rows[:-1, 2:4]) / 2)
        panel_work = GEOMETRY.volume * np.sum(np.diff(rows[:, 4:6], axis=0) * (rows[1:, 6:8] + rows[:-1, 6:8]) / 2)
        self.assertGreater(spring_work, 0.)
        self.assertAlmostEqual(spring_work, panel_work, places=9)

    def test_penalty_solution_converges_toward_exact_constraint_fixture(self):
        from test_imk_materials import rotation_history
        x = rotation_history([.008, -.008, .012, -.012, 0.], step=.0004)
        path = np.column_stack([x, -.6 * x])
        exact, _, _ = run_prescribed_joint(path)
        coarse, _, coarse_error = run_prescribed_joint(path, handler="Penalty", penalty=1e10)
        fine, _, fine_error = run_prescribed_joint(path, handler="Penalty", penalty=1e12)
        # Unloaded rigid faces already satisfy their constraints to roundoff;
        # the loaded support/core penalty compliance must converge instead.
        self.assertLess(max(fine_error, coarse_error), 1e-10)
        coarse_force_error = np.max(abs(coarse[:, 2:4] - exact[:, 2:4]))
        fine_force_error = np.max(abs(fine[:, 2:4] - exact[:, 2:4]))
        self.assertLess(fine_force_error, coarse_force_error)
        np.testing.assert_allclose(fine[:, :2], exact[:, :2], atol=1e-8, rtol=0)
        np.testing.assert_allclose(fine[:, 2:4], exact[:, 2:4], atol=.01, rtol=0)

    def test_peak_oriented_member_hinge_and_clear_beam_match_tip_compliance(self):
        panel = make_panel()
        ops.fix(TAGS.column_core, 1, 1, 1, 1, 1, 1)
        length, force, elastic_modulus, iy, hinge_ke = 216., .01, 3600., 30000., 1e5
        face = np.array(ops.nodeCoord(TAGS.x_plus))
        ops.node(9, *face)
        ops.node(10, *(face + [length, 0., 0.]))
        calibration = fixture_calibration()
        define_rotational_imk(
            "IMKPeakOriented", 13, hinge_ke, calibration.positive, calibration.negative,
            replace(calibration.cyclic, kappa_f=None, kappa_d=None),
            provenance={"calibration_id": "synthetic_member_joint_compliance_fixture",
                        "status": "verification_only_not_experimental", "deformation_scope": "member_flexure_only_fixture"})
        ops.equalDOF(TAGS.x_plus, 9, 1, 2, 3, 4, 6)
        ops.element("zeroLength", 22, TAGS.x_plus, 9, "-mat", 13, "-dir", 5)
        ops.geomTransf("Linear", 31, 0., 0., 1.)
        ops.element("elasticBeamColumn", 23, 9, 10, 400., elastic_modulus, 1500., 20000., iy, 40000., 31)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(10, 0., 0., -force, 0., 0., 0.)
        analysis()
        self.assertEqual(ops.analyze(1), 0)
        # Virtual work: joint arm a contributes to its lever; the face hinge
        # carries P*L; only the clear beam length contributes elastic curvature.
        a = GEOMETRY.dx / 2
        expected_tip = -force * ((length + a) ** 2 / 1.7e5 + length ** 2 / hinge_ke
                                 + length ** 3 / (3 * elastic_modulus * iy))
        self.assertAlmostEqual(ops.nodeDisp(10, 3), expected_tip, places=10)
        self.assertAlmostEqual(joint_response(panel)["moment_ry_kip_in"], force * (length + a), places=9)
        self.assertAlmostEqual(ops.eleResponse(22, "basicForce")[0], force * length, places=9)
        self.assertAlmostEqual(joint_response(panel)["moment_rx_kip_in"], 0., places=10)

    def test_floor_diaphragm_preserves_both_vertical_joint_shear_rotations(self):
        panel = make_panel()
        # Represent a floor master attached to the column core's UX/UY/RZ.
        # Leave core Rx/Ry supported, while the beam core can shear the panel.
        master = 90
        center = np.array(GEOMETRY.center)
        ops.node(master, *(center + [-100., -100., 0.]))
        ops.fix(master, 1, 1, 1, 1, 1, 1)
        ops.fix(TAGS.column_core, 0, 0, 1, 1, 1, 0)
        ops.rigidDiaphragm(3, master, TAGS.column_core)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(TAGS.beam_core, 0., 0., 0., 10., 17., 0.)
        analysis()
        self.assertEqual(ops.analyze(1), 0)
        response = joint_response(panel)
        np.testing.assert_allclose([response["q_rx"], response["q_ry"]], [.0001, .0001], rtol=1e-10)
        np.testing.assert_allclose([response["gamma_yz_from_faces"], response["gamma_xz_from_faces"]], [.0001, -.0001], atol=1e-12)

    def test_clear_span_uses_panel_faces_in_all_three_directions(self):
        for axis, distance, other_size, expected in (("x", 240., 28., 214.),
                                                     ("y", 300., 26., 272.),
                                                     ("z", 144., 24., 122.)):
            index = "xyz".index(axis)
            center = list(GEOMETRY.center)
            center[index] += distance
            other = replace(GEOMETRY, center=tuple(center), **{("dx", "dy", "hz")[index]: other_size})
            self.assertEqual(clear_span_between_panels(GEOMETRY, other, axis), expected)
            self.assertEqual(clear_span_between_panels(other, GEOMETRY, axis), expected)
        with self.assertRaises(ValueError):
            clear_span_between_panels(GEOMETRY, GEOMETRY, "x")
        with self.assertRaises(ValueError):
            clear_span_between_panels(GEOMETRY, replace(GEOMETRY, center=(500., 10., 100.)), "x")

    def test_perpendicular_beam_torsion_competes_with_panel_shear(self):
        # This deliberately documents the prototype's coupling. Passing this
        # equation is not approval of this topology for a three-dimensional RC joint.
        for ratio in (0., .1, 1., 10.):
            with self.subTest(ratio=ratio):
                result = run_transverse_torsion_probe(ratio)
                self.assertAlmostEqual(result["panel_rotation_rad"], result["analytical_panel_rotation_rad"], places=12)
                self.assertAlmostEqual(result["x_beam_face_rx_rad"], result["panel_rotation_rad"], places=12)
                self.assertAlmostEqual(result["panel_moment_kip_in"], 10. / (1 + ratio), places=8)
                self.assertAlmostEqual(result["panel_moment_kip_in"] + result["transverse_beam_torque_kip_in"], 10., places=8)


if __name__ == "__main__":
    unittest.main()
