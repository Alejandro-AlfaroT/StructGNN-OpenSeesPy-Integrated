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
        cap = min(diagrams["y"][0][0], diagrams["z"][0][0])
        result = check_column_pm(0.5 * cap, 0.0, 0.0, diagrams, cfg)
        self.assertAlmostEqual(result.dcr, 0.5, places=9)

    def test_axial_cap_holds_whatever_the_moment(self):
        """The cross-check's counterexample: above phi Pn,max, 1 kip-in of moment must not make the column pass."""
        cfg = self._cfg(top=4, side=2)
        diagrams = build_pm_diagrams(cfg)
        cap = max(p for p, _m in diagrams["y"])
        self.assertAlmostEqual(cap, diagrams["y"][0][0], places=9)             # the sweep is cut at phi Pn,max
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

    def test_single_diagram_is_accepted_for_both_axes(self):
        cfg = self._cfg(top=4, side=2)
        y = build_pm_diagram(cfg, axis="y")
        P = 100.0
        phi = _interpolate_pm_capacity(P, y)
        self.assertAlmostEqual(check_column_pm(P, 0.0, 0.5 * phi, y, cfg).dcr, 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
