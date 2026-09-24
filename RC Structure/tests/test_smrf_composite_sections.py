"""Native equilibrium and independent beam-theory checks for section recovery."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design.SMRF_Composite_Sections import recover_floor_cut, transport_wrench
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity
from tests.test_smrf_coupled_analysis import SLAB, GEOMETRY, SECTIONS, DEAD, ZERO
from tools.review_composite_sections import analytical_cantilever


class CompositeSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geometry = dict(GEOMETRY,num_bay_x=2,num_bay_y=2)
        case = dict(DEAD,id='corner',dead_factor=1.2,live_factor=1.6,live_pattern=[[0,0]])
        cls.result = analyze_coupled_gravity(SLAB,geometry,SECTIONS,[ZERO,case],mesh_per_bay=4)

    def test_asymmetric_all_floor_cuts_balance_every_force_and_moment(self):
        original = copy.deepcopy(self.result)
        for floor in (1,2):
            for axis, positions in (('x',(120.,360.)),('y',(100.,300.))):
                for position in positions:
                    with self.subTest(floor=floor,axis=axis,position=position):
                        cut = recover_floor_cut(self.result,floor,axis,position)
                        self.assertTrue(cut['numerical_balance_passed'])
                        self.assertFalse(cut['engineering_verified'])
                        for side in ('left','right'):
                            residual = cut['sides'][side]['free_body_residual']
                            self.assertLess(max(abs(v) for v in residual[:3]),1e-8)
                            self.assertLess(max(abs(v) for v in residual[3:]),1e-6)
                        json.dumps(cut,allow_nan=False)
        self.assertEqual(original,self.result)

    def test_reference_shift_obeys_independent_wrench_translation(self):
        a = recover_floor_cut(self.result,2,'x',120.)
        delta = [0.,37.,-19.]
        reference = [x+y for x,y in zip(a['reference_in'],delta)]
        b = recover_floor_cut(self.result,2,'x',120.,reference_in=reference)
        self.assertTrue(b['numerical_balance_passed'])
        for side in ('left','right'):
            f = a['sides'][side]['total_force_moment']
            expected = f[:3]+[f[3]-delta[1]*f[2]+delta[2]*f[1],
                              f[4]-delta[2]*f[0],f[5]+delta[1]*f[0]]
            for actual,want in zip(b['sides'][side]['total_force_moment'],expected):
                self.assertAlmostEqual(actual,want,places=7)

    def test_missing_eccentricity_breaks_free_body_equilibrium(self):
        broken = copy.deepcopy(self.result)
        for b in broken['web_segment_actions']:
            if b['floor'] == 2:
                b['node_positions_in'] = [[p[0],p[1],2*GEOMETRY['story_h_in']] for p in b['node_positions_in']]
                b['body_load_position_in'][2] = 2*GEOMETRY['story_h_in']
        cut = recover_floor_cut(broken,2,'x',120.)
        self.assertFalse(cut['numerical_balance_passed'])
        self.assertGreater(cut['sides']['left']['moment_relative_error'],1e-4)

    def test_missing_pressure_at_cut_breaks_action_continuity(self):
        broken = copy.deepcopy(self.result)
        for s in broken['shell_resultants']:
            if s['floor'] == 2:
                s['applied_nodal_force_kip_kip_in'] = [0.]*24
        cut = recover_floor_cut(broken,2,'x',120.)
        self.assertFalse(cut['numerical_balance_passed'])
        self.assertGreater(cut['force_continuity_relative_error'],1e-4)

    def test_missing_and_duplicate_inventories_are_rejected(self):
        for key in ('shell_resultants','web_segment_actions','floor_boundary_actions'):
            for duplicate in (False,True):
                with self.subTest(key=key,duplicate=duplicate):
                    broken = copy.deepcopy(self.result)
                    index = next(i for i,r in enumerate(broken[key]) if r['floor']==2)
                    entry = broken[key].pop(index)
                    if duplicate:
                        broken[key].extend([entry,entry])
                    with self.assertRaisesRegex(ValueError,'inventory'):
                        recover_floor_cut(broken,2,'x',120.)

    def test_invalid_cuts_and_nonfinite_data_are_rejected(self):
        for position in (0.,240.,480.,121.,float('nan')):
            with self.subTest(position=position), self.assertRaises(ValueError):
                recover_floor_cut(self.result,2,'x',position)
        for floor in (0,3,True,1.5):
            with self.assertRaises(ValueError):
                recover_floor_cut(self.result,floor,'x',120.)
        broken = copy.deepcopy(self.result)
        broken['shell_resultants'][0]['global_nodal_force_kip_kip_in'][0] = float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            recover_floor_cut(broken,1,'x',120.)
        with self.assertRaises(ValueError):
            recover_floor_cut(self.result,2,'x',120.,numerical_tolerance=float('nan'))

    def test_old_or_failed_results_cannot_be_used(self):
        for field,value in (('status','analysis_failed'),('section_action_schema',None)):
            with self.assertRaisesRegex(ValueError,'completed coupled solve'):
                recover_floor_cut(dict(self.result,**{field:value}),2,'x',120.)

    def test_analytical_constant_curvature_matches_displacement_and_components(self):
        for mesh in (4,8,16):
            with self.subTest(mesh=mesh):
                r = analytical_cantilever(mesh)
                self.assertTrue(r['numerical_passed'])
                left = r['cut']['sides']['left']
                self.assertAlmostEqual(left['total_force_moment'][4],1000.,places=5)
                self.assertLess(max(abs(v) for v in left['total_force_moment'][:3]),1e-7)
                self.assertGreater(left['components']['web_eccentricity'][4]/1000.,.70)
                self.assertEqual(ops.getNodeTags(),[])

    def test_benchmark_preserves_existing_domain(self):
        ops.model('basic','-ndm',3,'-ndf',6)
        ops.node(987,0.,0.,0.)
        try:
            with self.assertRaisesRegex(RuntimeError,'preserved'):
                analytical_cantilever()
            self.assertEqual(ops.getNodeTags(),[987])
        finally:
            ops.wipe()

    def test_transport_has_physical_sign_for_axial_force_below_midplane(self):
        self.assertEqual(transport_wrench([10.,0.,0.,0.,0.,0.],[0.,0.,-14.],[0.,0.,0.]),
                         [10.,0.,0.,0.,-140.,0.])


if __name__ == '__main__':
    unittest.main()
