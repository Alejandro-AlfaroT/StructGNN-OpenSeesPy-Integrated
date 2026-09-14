"""Beam end-moment sign mapping: numerical localForce sign -> sagging/hogging.

The design checks read OpenSees localForce for each beam and must decide
which end moments load the top bars and which load the bottom bars. The
convention is pinned here against OpenSees itself, on the same transform the
frame builders use (vecxz = (0, 0, 1)), so a change in either the builder or
the mapping fails loudly instead of silently symmetrising beam steel.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openseespy.opensees as ops  # noqa: E402

from RC_Design_Check import beam_flexure_demands  # noqa: E402


def _fixed_fixed_span_local_forces(span=240.0, w=1.0):
    """Two-element fixed-fixed span under uniform downward load.

    Physical state: hogging w*L^2/12 at both supports, sagging w*L^2/24 at
    midspan. Returns the localForce vectors of the left and right elements.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, span / 2, 0.0, 0.0)
    ops.node(3, span, 0.0, 0.0)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.fix(3, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 0, 0, 1)
    for tag, ni, nj in ((1, 1, 2), (2, 2, 3)):
        ops.element("elasticBeamColumn", tag, ni, nj,
                    400.0, 3600.0, 1500.0, 30000.0, 13333.0, 5333.0, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.eleLoad("-ele", 1, 2, "-type", "-beamUniform", 0.0, -w)
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    left = list(ops.eleResponse(1, "localForce"))
    right = list(ops.eleResponse(2, "localForce"))
    ops.wipe()
    return left, right


class BeamFlexureDemandMappingTests(unittest.TestCase):
    def test_pure_sign_cases(self):
        # end i: negative My is hogging; end j: positive My is hogging.
        self.assertEqual(beam_flexure_demands(-100.0, 0.0), (0.0, 100.0))
        self.assertEqual(beam_flexure_demands(+100.0, 0.0), (100.0, 0.0))
        self.assertEqual(beam_flexure_demands(0.0, +100.0), (0.0, 100.0))
        self.assertEqual(beam_flexure_demands(0.0, -100.0), (100.0, 0.0))

    def test_gravity_hogging_pair_is_not_a_sagging_demand(self):
        # Single-element beam with hogging at both ends: My_i = -M, My_j = +M.
        pos, neg = beam_flexure_demands(-300.0, 300.0)
        self.assertEqual(neg, 300.0)
        self.assertEqual(pos, 0.0)

    def test_sway_double_curvature_keeps_both_signs(self):
        # Sway: sagging at i (+), hogging at j (+), different magnitudes.
        pos, neg = beam_flexure_demands(514.0, 590.0)
        self.assertEqual((pos, neg), (514.0, 590.0))

    def test_convention_matches_opensees_fixed_fixed_span(self):
        span, w = 240.0, 1.0
        hog, sag = w * span ** 2 / 12.0, w * span ** 2 / 24.0
        left, right = _fixed_fixed_span_local_forces(span, w)
        # Left element: i at the support (hogging), j at midspan (sagging).
        pos, neg = beam_flexure_demands(left[4], left[10])
        self.assertAlmostEqual(neg, hog, places=6)
        self.assertAlmostEqual(pos, sag, places=6)
        # Right element: i at midspan (sagging), j at the support (hogging).
        pos, neg = beam_flexure_demands(right[4], right[10])
        self.assertAlmostEqual(pos, sag, places=6)
        self.assertAlmostEqual(neg, hog, places=6)
        # Vertical bending is the local-y moment on this transform; Mz stays 0.
        self.assertAlmostEqual(left[5], 0.0, places=9)
        self.assertAlmostEqual(left[11], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()


class BarPickerTieBreakTests(unittest.TestCase):
    """Ties in the governing-DCR objective must go to the lighter cage."""

    def setUp(self):
        import Structure_Parameters as sp
        from Design.Config import DesignConfig
        self.sp = sp
        self.cfg = DesignConfig()
        self.saved = (sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI, sp.FY_KSI)
        sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI, sp.FY_KSI = 10.0, 18.0, 4.0, 60.0

    def tearDown(self):
        (self.sp.B_BEAM, self.sp.H_BEAM,
         self.sp.FC_BEAM_KSI, self.sp.FY_KSI) = self.saved

    def test_symmetric_demand_noise_does_not_add_a_top_bar(self):
        import Redesign as rd
        candidates = [(5, 3, 3), (5, 3, 4), (5, 4, 3), (5, 4, 4)]
        # Equal demands up to float noise: with 3 bars governing in both
        # directions the three lighter candidates tie at the same DCR.
        base = 676.0
        for mu_pos, mu_neg in ((base, base), (base, base + 2e-5), (base + 2e-5, base)):
            pick = rd._pick_beam(candidates, Mu_pos=mu_pos, Mu_neg=mu_neg, cfg=self.cfg)
            self.assertEqual(pick, (5, 3, 3), (mu_pos, mu_neg))

    def test_real_preference_still_wins(self):
        import Redesign as rd
        candidates = [(5, 3, 3), (5, 4, 3)]
        # Hogging clearly larger: 3T sits near DCR 1.0 and 4T near 0.75-0.80
        # under either cover convention, so 4T is nearer the 0.85 target.
        pick = rd._pick_beam(candidates, Mu_pos=600.0, Mu_neg=780.0, cfg=self.cfg)
        self.assertEqual(pick, (5, 4, 3))
