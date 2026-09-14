"""Column P-M under biaxial bending: each axis against its own surface, load-contour interaction."""
import contextlib
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Design.ACI_Checks import build_pm_diagram, build_pm_diagrams, check_column_pm, _interpolate_pm_capacity
from Design.Config import DesignConfig


class BiaxialColumnTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name in ("B_COL", "H_COL", "FC_COL_KSI", "COL_BAR_SIZE", "COL_BAR_AREA", "COL_TOP_BARS", "COL_BOT_BARS",
                     "COL_SIDE_BARS", "SLAB_THICKNESS_IN", "COL_STIRRUP_BAR_SIZE", "COL_CLEAR_COVER_IN"):
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.SLAB_THICKNESS_IN = 6.0
        sp.B_COL = sp.H_COL = 24.0
        sp.FC_COL_KSI = 5.0
        sp.COL_BAR_SIZE, sp.COL_BAR_AREA = 8, sp.rebar_area(8)
        sp.COL_STIRRUP_BAR_SIZE = 4

    def tearDown(self):
        self.stack.close()

    def _cfg(self, top, side):
        sp.COL_TOP_BARS = sp.COL_BOT_BARS = top
        sp.COL_SIDE_BARS = side
        cfg = DesignConfig.from_structure_parameters()
        cfg.sections.b_col_in, cfg.sections.h_col_in = sp.B_COL, sp.H_COL
        cfg.materials.fc_col_ksi = sp.FC_COL_KSI
        return cfg

    def test_symmetric_cage_has_identical_surfaces(self):
        cfg = self._cfg(top=4, side=2)                       # 4 bars per face all round
        y, z = build_pm_diagram(cfg, axis="y"), build_pm_diagram(cfg, axis="z")
        for (py, my), (pz, mz) in zip(y, z):
            self.assertAlmostEqual(py, pz, places=6)
            self.assertAlmostEqual(my, mz, places=6)

    def test_unequal_faces_give_different_surfaces_and_the_weak_axis_is_read_against_its_own(self):
        cfg = self._cfg(top=5, side=1)                       # heavy top/bottom faces, light sides
        diagrams = build_pm_diagrams(cfg)
        P = 200.0
        phi_y = _interpolate_pm_capacity(P, diagrams["y"])
        phi_z = _interpolate_pm_capacity(P, diagrams["z"])
        self.assertGreater(phi_y, 1.03 * phi_z)                # bending about z has less lever-arm steel
        # A pure moment about z at 95% of its own strength: under the old
        # resultant-against-y reading this passed comfortably; it is 0.95 now.
        M = 0.95 * phi_z
        result = check_column_pm(P, M, 0.0, diagrams, cfg)
        self.assertAlmostEqual(result.dcr, 0.95, places=6)
        self.assertLess(M / phi_y, 0.92)                       # what the old check would have reported
        # The same moment about y is well within the stronger surface.
        self.assertAlmostEqual(check_column_pm(P, 0.0, M, diagrams, cfg).dcr, M / phi_y, places=6)

    def test_load_contour_combination(self):
        cfg = self._cfg(top=4, side=2)
        diagrams = build_pm_diagrams(cfg)
        P = 150.0
        phi = _interpolate_pm_capacity(P, diagrams["y"])
        # Equal components at 0.6 of the uniaxial strength each.
        result = check_column_pm(P, 0.6 * phi, 0.6 * phi, diagrams, cfg)
        alpha = cfg.dcr.biaxial_contour_exponent
        self.assertAlmostEqual(result.dcr, (2 * 0.6 ** alpha) ** (1 / alpha), places=9)
        # Linear contour is the conservative bound; the resultant would have said 0.85.
        cfg.dcr.biaxial_contour_exponent = 1.0
        self.assertAlmostEqual(check_column_pm(P, 0.6 * phi, 0.6 * phi, diagrams, cfg).dcr, 1.2, places=9)
        self.assertLess(0.6 * math.sqrt(2), 1.2)
        # Reported demand is the equivalent uniaxial moment about y.
        cfg.dcr.biaxial_contour_exponent = 1.5
        result = check_column_pm(P, 0.6 * phi, 0.6 * phi, diagrams, cfg)
        self.assertAlmostEqual(result.demand, result.dcr * phi, places=9)
        self.assertAlmostEqual(result.capacity, phi, places=9)

    def test_pure_compression_uses_the_smaller_axial_cap(self):
        cfg = self._cfg(top=5, side=1)
        diagrams = build_pm_diagrams(cfg)
        cap = min(max(p for p, _m in diagrams["y"]), max(p for p, _m in diagrams["z"]))
        result = check_column_pm(0.5 * cap, 0.0, 0.0, diagrams, cfg)
        self.assertAlmostEqual(result.dcr, 0.5, places=9)

    def test_axial_cap_holds_whatever_the_moment(self):
        """The cross-check's counterexample: above phi Pn,max, 1 kip-in of moment must not make the column pass."""
        cfg = self._cfg(top=4, side=2)
        diagrams = build_pm_diagrams(cfg)
        cap = max(p for p, _m in diagrams["y"])
        self.assertEqual(sum(1 for p, _m in diagrams["y"] if p == cap), 1)     # the sweep is cut at phi Pn,max: one envelope point there
        self.assertTrue(all(p <= cap + 1e-9 for p, _m in diagrams["y"]))
        over = 1.05 * cap
        self.assertAlmostEqual(check_column_pm(over, 0.0, 0.0, diagrams, cfg).dcr, 1.05, places=9)
        for moment in (1.0, 500.0):
            result = check_column_pm(over, 0.0, moment, diagrams, cfg)
            self.assertAlmostEqual(result.dcr, 1.05, places=9)
            self.assertFalse(result.ok)
        # Just under the cap the axial ratio still governs a small moment.
        result = check_column_pm(0.97 * cap, 0.0, 10.0, diagrams, cfg)
        self.assertAlmostEqual(result.dcr, 0.97, places=9)
        # The steel picker's own sweep is cut the same way.
        from Redesign import _phi_Mn_at_Pu
        sp.COL_TOP_BARS = sp.COL_BOT_BARS = 4; sp.COL_SIDE_BARS = 2
        self.assertIsNone(_phi_Mn_at_Pu(over, 12 * sp.rebar_area(8)))
        # And the nominal surface used for SCWB and the hinges stops at 0.80 P0.
        from Model.IMK_Calibration import column_pm_nominal
        nominal = column_pm_nominal()
        ag = sp.B_COL * sp.H_COL; ast = 12 * sp.rebar_area(8)
        self.assertAlmostEqual(max(p for p, _m in nominal), 0.80 * (0.85 * sp.FC_COL_KSI * (ag - ast) + sp.FY_KSI * ast), places=6)

    def test_nominal_lookup_follows_the_upper_envelope_at_the_cap(self):
        """Fourth cross-check: two moments at the cap made the lookup read 0 just below it and the cap moment above it."""
        from Model.IMK_Calibration import column_pm_nominal_for, column_moment_at_axial, column_axial_domain, _col_steel_layers
        sp.COL_TOP_BARS = sp.COL_BOT_BARS = 4; sp.COL_SIDE_BARS = 2
        diagram = column_pm_nominal_for(sp.B_COL, sp.H_COL, sp.FC_COL_KSI, _col_steel_layers())
        low, cap = column_axial_domain(diagram)
        self.assertEqual(sum(1 for p, _m in diagram if p == cap), 1)          # one point at the cap: the envelope
        at_cap = column_moment_at_axial(cap, diagram)
        below = column_moment_at_axial(cap - 1.0, diagram)
        previous = max(m for p, m in diagram if p < cap)
        self.assertGreater(at_cap, 0.0)
        self.assertGreaterEqual(below, at_cap)                                # the envelope rises away from the cap
        self.assertLessEqual(below, previous)
        self.assertGreater(below, 0.5 * at_cap)                               # never toward zero
        # Above the cap the section carries no moment at all; the design check fails it on the axial ratio.
        self.assertEqual(column_moment_at_axial(cap + 1.0, diagram), 0.0)
        self.assertEqual(column_moment_at_axial(1.2 * cap, diagram), 0.0)
        self.assertEqual(column_moment_at_axial(low - 1.0, diagram), 0.0)
        # Interior lookups interpolate the envelope.
        (p1, m1), (p2, m2) = sorted(diagram)[10:12]
        self.assertAlmostEqual(column_moment_at_axial(0.5 * (p1 + p2), diagram), 0.5 * (m1 + m2), places=9)
        # The design-side lookup reads the same envelope just below its cap.
        cfg = self._cfg(top=4, side=2)
        design = build_pm_diagram(cfg)
        design_cap = max(p for p, _m in design)
        self.assertGreater(_interpolate_pm_capacity(design_cap - 1e-6, design), 0.99 * _interpolate_pm_capacity(design_cap, design))
        self.assertIsNone(_interpolate_pm_capacity(design_cap + 1.0, design))
        # A hinge cannot be calibrated for a column whose gravity load is outside the surface.
        from Model import IMK_Hinges
        IMK_Hinges._PM_DIAGRAM_CACHE.clear()
        with self.assertRaisesRegex(ValueError, "outside the nominal P-M surface"):
            IMK_Hinges._member_properties("column", axial_kip=1.1 * cap)
        self.assertAlmostEqual(IMK_Hinges._member_properties("column", axial_kip=0.5 * cap)["my"],
                               column_moment_at_axial(0.5 * cap, diagram), places=6)

    def test_single_diagram_is_accepted_for_both_axes(self):
        cfg = self._cfg(top=4, side=2)
        y = build_pm_diagram(cfg, axis="y")
        P = 100.0
        phi = _interpolate_pm_capacity(P, y)
        self.assertAlmostEqual(check_column_pm(P, 0.0, 0.5 * phi, y, cfg).dcr, 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
