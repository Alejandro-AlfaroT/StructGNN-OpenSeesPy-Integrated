"""Symmetric beam design policy; no structural analyses are launched."""
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
import Redesign as rd
from Design.Config import DesignConfig


class SymmetricBeamTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in {
            'SLAB_THICKNESS_IN': 6.0, 'B_BEAM': 10.0, 'H_BEAM': 18.0,
            'B_COL': 20.0, 'H_COL': 20.0, 'NUM_BAY_X': 2, 'NUM_BAY_Y': 6,
            'FC_BEAM_KSI': 4.0, 'FY_KSI': 60.0, 'BEAM_CLEAR_COVER_IN': 1.5,
            'BEAM_STIRRUP_BAR_SIZE': 4, 'AGGREGATE_MAX_SIZE_IN': 0.75,
            'BEAM_BAR_SIZE': 6, 'BEAM_TOP_BARS': 3, 'BEAM_BOT_BARS': 2,
        }.items():
            self.stack.enter_context(mock.patch.object(sp, key, value))
        self.cfg = DesignConfig()
        self.cfg.rebar.bar_sizes_beam = [6]
        self.cfg.rebar.beam_n_range = (2, 3)

    def test_larger_demand_governs_both_faces(self):
        self.assertEqual(rd._beam_candidates(1.2, 1.4, .8, 1.0, self.cfg), [(6, 3, 3)])

    def test_legacy_independent_search_is_explicit(self):
        self.cfg.rebar.beam_symmetric = False
        self.assertEqual(rd._beam_candidates(1.2, 1.4, .8, 1.0, self.cfg), [(6, 3, 2)])

    def test_spacing_still_excludes_impossible_cages(self):
        sp.B_BEAM = 7.0
        self.assertEqual(rd._beam_candidates(1.2, 1.4, .8, 1.0, self.cfg), [])

    def test_in_band_asymmetric_cage_is_not_converged(self):
        result = {1: {'type': 'beam', 'Mu_neg': 900., 'Mu_pos': 500.,
                      'dcr_neg': .85, 'dcr_pos': .70, 'dcr_V': .2}}
        _, update, converged, _ = rd.redesign_steel(result, self.cfg)
        self.assertFalse(converged)
        self.assertIsNotNone(update)
        self.assertEqual(update['n_top'], update['n_bot'])

    def test_fallback_preserves_demand_minimum(self):
        # An impossible demand must not fall back to a small code-min cage.
        result = {1: {'type': 'beam', 'Mu_neg': 10000., 'Mu_pos': 500.,
                      'dcr_neg': 9., 'dcr_pos': .70, 'dcr_V': 1.2}}
        _, update, converged, _ = rd.redesign_steel(result, self.cfg)
        self.assertFalse(converged)
        self.assertIsNone(update)


if __name__ == '__main__':
    unittest.main()
