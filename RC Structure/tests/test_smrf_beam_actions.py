"""Static closed-form and OpenSees checks for span-interior beam demands."""
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design.SMRF_Beam_Actions import recover_beam_bending, current_beam_bending, applied_element_loads
from RC_Design_Check import beam_flexure_demands


def forces(length, mi, vi, w=0., points=()):
    f = [0.] * 12
    f[4], f[2] = mi, vi
    f[8] = -(vi + w*length + sum(p for a, p in points))
    f[10] = -(mi + vi*length + w*length**2/2 + sum(p*length*(1-a) for a, p in points))
    return f


class AnalyticalBeamTests(unittest.TestCase):
    def test_single_member_fixed_fixed_uniform_interior_sagging_is_not_lost(self):
        length, w = 240., 1.
        f = forces(length, -w*length**2/12, w*length/2, -w)
        self.assertEqual(beam_flexure_demands(f[4], f[10])[0], 0.)
        result = recover_beam_bending(length, f, -w)
        self.assertAlmostEqual(result["full_span"]["mu_positive_kip_in"], w*length**2/24)
        self.assertAlmostEqual(result["full_span"]["mu_negative_kip_in"], w*length**2/12)
        self.assertEqual(result["full_span"]["positive_x_in"], length/2)

    def test_simply_supported_uniform(self):
        result = recover_beam_bending(120., forces(120., 0., 60., -1.), -1.)
        self.assertAlmostEqual(result["full_span"]["mu_positive_kip_in"], 120.**2/8)
        self.assertEqual(result["full_span"]["mu_negative_kip_in"], 0.)

    def test_off_center_point_load_peak_is_at_load_not_midspan(self):
        length, p, a = 120., 10., .3
        loads = [(a, -p)]
        result = recover_beam_bending(length, forces(length, 0., p*(1-a), points=loads), point_z_loads=loads)
        self.assertAlmostEqual(result["full_span"]["positive_x_in"], a*length)
        self.assertAlmostEqual(result["full_span"]["mu_positive_kip_in"], p*length*a*(1-a))

    def test_combined_load_root_between_points_and_faces(self):
        loads = [(.25, -2.), (.25, -3.)]
        f = forces(100., -500., 70., -1., loads)
        result = recover_beam_bending(100., f, -1., loads, (10., 15.))
        self.assertEqual(result["full_span"]["positive_x_in"], 65.)
        self.assertEqual(result["point_z_loads"], [[.25, -5.]])
        self.assertEqual(result["clear_span"]["interval_in"], [10., 85.])
        self.assertLess(result["clear_span"]["mu_negative_kip_in"], result["full_span"]["mu_negative_kip_in"])

    def test_no_element_loads_linear_diagram(self):
        result = recover_beam_bending(100., forces(100., -100., 3.))
        self.assertEqual(result["full_span"]["mu_positive_kip_in"], 200.)
        self.assertEqual(result["full_span"]["mu_negative_kip_in"], 100.)

    def test_uplift_changes_tension_face(self):
        result = recover_beam_bending(100., forces(100., 0., -50., 1.), 1.)
        self.assertEqual(result["full_span"]["mu_positive_kip_in"], 0.)
        self.assertEqual(result["full_span"]["mu_negative_kip_in"], 1250.)

    def test_mismatched_end_forces_loads_and_bad_input_are_rejected(self):
        f = forces(100., 0., 50., -1.)
        with self.assertRaisesRegex(ValueError, "equilibrium"):
            recover_beam_bending(100., f, -.5)
        for kwargs in ({"length_in": -1.}, {"uniform_z_kip_per_in": float("nan")},
                       {"point_z_loads": [(0., 1.)]}, {"point_z_loads": [(True, 1.)]},
                       {"face_offsets_in": (60., 40.)}):
            args = {"length_in": 100., "local_force": f, "uniform_z_kip_per_in": -1., **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                recover_beam_bending(**args)


class OpenSeesBeamTests(unittest.TestCase):
    def setUp(self):
        ops.wipe()
        self.addCleanup(ops.wipe)

    def build(self, axis="x", pinned=False):
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0., 0., 0.)
        ops.node(2, 240. if axis == "x" else 0., 240. if axis == "y" else 0., 0.)
        if pinned:
            restraints = [1, 1, 1, int(axis == "x"), int(axis == "y"), 1]
            ops.fix(1, *restraints)
            ops.fix(2, *restraints)
        else:
            ops.fix(1, 1, 1, 1, 1, 1, 1)
            ops.fix(2, int(axis != "x"), int(axis != "y"), 1, 1, 1, 1)
        ops.geomTransf("Linear", 1, 0, 0, 1)
        ops.element("elasticBeamColumn", 1, 1, 2, 400., 3600., 1500., 30000., 13333., 5333., 1)

    def solve(self, step=1.):
        ops.system("BandGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", step)
        ops.algorithm("Linear")
        ops.analysis("Static")
        self.assertEqual(ops.analyze(1), 0)

    def test_native_load_query_and_signs_match_fixed_fixed_in_both_axes(self):
        for axis in ("x", "y"):
            with self.subTest(axis=axis):
                ops.wipe()
                self.build(axis)
                ops.timeSeries("Linear", 1)
                ops.pattern("Plain", 1, 1)
                ops.eleLoad("-ele", 1, "-type", "-beamUniform", 0., -1., 0.)
                self.solve()
                result = current_beam_bending([1])[1]
                self.assertAlmostEqual(result["full_span"]["mu_positive_kip_in"], 2400.)
                self.assertAlmostEqual(result["full_span"]["mu_negative_kip_in"], 4800.)
                self.assertEqual(len(ops.getEleTags()), 1)  # read-only extraction

    def test_point_load_pinned_span_and_current_factor(self):
        self.build(pinned=True)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.eleLoad("-ele", 1, "-type", "-beamPoint", 0., -10., .3)
        self.solve(.5)
        result = current_beam_bending([1])[1]
        self.assertEqual(result["point_z_loads"], [[.3, -5.]])
        self.assertAlmostEqual(result["full_span"]["mu_positive_kip_in"], 5.*240.*.3*.7)

    def test_multiple_patterns_and_frozen_gravity_do_not_use_reset_time(self):
        self.build()
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.eleLoad("-ele", 1, "-type", "-beamUniform", 0., -1., 0.)
        ops.timeSeries("Constant", 2, "-factor", 2.)
        ops.pattern("Plain", 2, 2)
        ops.eleLoad("-ele", 1, "-type", "-beamPoint", 0., -10., .5)
        self.solve(.5)
        before = current_beam_bending([1])[1]
        ops.loadConst("-time", 0.)
        after = current_beam_bending([1])[1]
        self.assertEqual(before, after)
        self.assertEqual(after["uniform_z_kip_per_in"], -.5)
        self.assertEqual(after["point_z_loads"], [[.5, -20.]])
        self.assertAlmostEqual(after["full_span"]["mu_positive_kip_in"], .5*240.**2/24 + 20.*240./8)

    def test_wiping_domain_does_not_reuse_previous_loads(self):
        self.build()
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.eleLoad("-ele", 1, "-type", "-beamUniform", 0., -1., 0.)
        self.solve()
        self.assertTrue(applied_element_loads())
        ops.wipe()
        self.assertEqual(applied_element_loads(), {})

    def test_unsupported_load_class_is_not_silently_skipped(self):
        with mock.patch.object(ops, "getPatterns", return_value=[1]), \
             mock.patch.object(ops, "getEleLoadTags", return_value=[1]), \
             mock.patch.object(ops, "getEleLoadClassTags", return_value=[999]), \
             mock.patch.object(ops, "getEleLoadData", return_value=[1.]), \
             mock.patch.object(ops, "getLoadFactor", return_value=1.):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                applied_element_loads()


if __name__ == "__main__":
    unittest.main()
