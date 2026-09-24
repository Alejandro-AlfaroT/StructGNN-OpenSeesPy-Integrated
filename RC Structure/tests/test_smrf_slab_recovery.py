"""Independent manufactured fields and native checks for physical-face recovery."""
import copy
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Slab_Recovery import recover_panel_faces
from Design.SMRF_Slab_Actions import build_slab_action_evidence, compare_slab_action_refinement, evaluate_slab_actions

def panel(xs, ys, field, offset=(0.,0.)):
    points=[]
    for j,(y0,y1) in enumerate(zip(ys,ys[1:])):
        for i,(x0,x1) in enumerate(zip(xs,xs[1:])):
            element=1+i+(len(xs)-1)*j
            for gp,(sx,sy) in enumerate(((-1,-1),(1,-1),(1,1),(-1,1)),1):
                x=(x0+x1)/2+sx*(x1-x0)/(2*math.sqrt(3))+offset[0]
                y=(y0+y1)/2+sy*(y1-y0)/(2*math.sqrt(3))+offset[1]
                points.append(dict(element=element,gauss_point=gp,x_in=x,y_in=y,**field(x,y,i,j)))
    return {'panel_id':'panel_x1_y1','i':0,'j':0,'gauss_point_resultants':points}

def field(x,y,i,j):
    return dict(mx=-3+x*.01+y*.02+x*y*.001,my=-2,mxy_raw=.5,
                qx_raw=1+x*.1+y*.2+x*y*.003,qy_raw=2-x*.2+y*.1)

class SlabRecoveryTests(unittest.TestCase):
    geometry=dict(num_bay_x=1,num_bay_y=1,bay_x_in=40.,bay_y_in=40.)

    def fixture_evidence(self, tensor):
        result = dict(status='transfer_complete', max_abs_membrane_kip_per_in=0.,
                      equilibrium=dict(numerical_balance_passed=True),
                      panels=[panel([0,10,20,30,40], [0,10,20,30,40], tensor)])
        slab = dict(thickness_in=5., concrete_fc_ksi=4., concrete_unit_weight_kcf=.15,
                    superimposed_dead_load_ksf=.05)
        inputs = dict(thickness_in=5.,fc_ksi=4.,fy_ksi=60.,max_aggregate_size_in=.75,
                      exposure='sheltered_interior',steel_specification='ASTM A706',
                      concrete_type='normalweight',num_floor=1,panel_ids=['panel_x1_y1'])
        flags = {key:True for key in ('verified','analysis_applicability_verified','all_floors_enveloped',
            'load_pattern_envelope_verified','spatial_envelope_per_unit_width',
            'twisting_moment_resolution_verified','zero_membrane_force_verified')}
        flags.update(asserted_by='test fixture only',assertion_date='2026-09-24',
                     assertion_basis='Manufactured field; not engineering certification')
        before = copy.deepcopy(result)
        with mock.patch('Design.SMRF_Slab_Actions.analyze_floor',return_value=result):
            evidence = build_slab_action_evidence(slab,self.geometry,dict(b_beam_in=14.),.05,
                                                  inputs,mesh_per_bay=4,assertions=flags)
        self.assertEqual(result,before)
        return evidence

    def test_supported_faces_can_satisfy_assertions_without_clearing_independent_review(self):
        evidence = self.fixture_evidence(lambda x,y,i,j:dict(mx=-1.,my=-2.,mxy_raw=0.,qx_raw=1.,qy_raw=2.))
        self.assertTrue(evidence['verified'])
        checks = {c['id']:c for c in evaluate_slab_actions(evidence)}
        self.assertEqual(checks['floor.qualified_slab_actions']['status'],'pass')
        self.assertEqual(checks['floor.independent_hand_verification']['status'],'not_evaluated')

    def test_tied_shear_cannot_discard_an_unresolved_tension_face_in_either_order(self):
        for sign in (-1.,1.):
            with self.subTest(sign=sign):
                evidence = self.fixture_evidence(lambda x,y,i,j:dict(mx=sign*(x-20),my=-2.,
                    mxy_raw=0.,qx_raw=1.,qy_raw=0.))
                self.assertFalse(evidence['verified'])
                row = next(r for r in evidence['strips'] if r['axis']=='x' and r['face']=='top')
                self.assertFalse(row['tension_face_at_shear_verified'])
                self.assertAlmostEqual(row['vu_kip_per_ft'],12.)

    def test_exact_bilinear_field_at_physical_faces_on_nonuniform_grid(self):
        p=panel([0,3,8,14,25,40],[0,5,15,26,40],field)
        before=copy.deepcopy(p)
        recovered=recover_panel_faces(p,self.geometry,14.)
        self.assertEqual(p,before)
        self.assertEqual(len(recovered['coverage']),4)
        for s in recovered['samples']:
            exact=field(s['x_in'],s['y_in'],0,0)
            for key,value in exact.items():self.assertAlmostEqual(s['raw_resultants'][key],value,places=11)
        # x=7 lies in [3,8], not the first [0,3] row beside the centerline.
        xs=[s for s in recovered['samples'] if s['axis']=='x' and s['support']=='lower']
        self.assertTrue(all(abs(s['cell_bounds_in']['x_lo']-3)<1e-8 for s in xs))

    def test_clear_span_ownership_and_no_cross_element_averaging(self):
        xs=[0,7,20,33,40];ys=[0,7,20,33,40]
        def discontinuous(x,y,i,j):return dict(mx=-100-i,my=-100-j,mxy_raw=0.,qx_raw=float(i),qy_raw=float(j))
        r=recover_panel_faces(panel(xs,ys,discontinuous),self.geometry,14.)
        for s in r['samples']:
            expected=1. if s['support']=='lower' else 2.
            self.assertAlmostEqual(s['raw_resultants']['qx_raw' if s['axis']=='x' else 'qy_raw'],expected)
        # A transverse cell boundary has separate limits; it is not smoothed.
        limits=[s['raw_resultants']['my'] for s in r['samples'] if s['axis']=='x' and s['support']=='lower' and abs(s['y_in']-20)<1e-8]
        self.assertEqual(sorted(round(v,8) for v in limits),[-102.,-101.])

    def test_missing_duplicate_malformed_and_nonfinite_data_fail_closed(self):
        original=panel([0,10,20,30,40],[0,10,20,30,40],field)
        variants=[]
        p=copy.deepcopy(original);p['gauss_point_resultants'].pop(0);variants.append(p)
        p=copy.deepcopy(original);p['gauss_point_resultants'].append(p['gauss_point_resultants'][0]);variants.append(p)
        p=copy.deepcopy(original);p['gauss_point_resultants'][2]['x_in']+=1;variants.append(p)
        p=copy.deepcopy(original);p['gauss_point_resultants'][0]['mx']=math.nan;variants.append(p)
        p=copy.deepcopy(original);p['gauss_point_resultants'][0]['gauss_point']=True;variants.append(p)
        p=copy.deepcopy(original);p['gauss_point_resultants']=[x for x in p['gauss_point_resultants'] if x['element']!=5];variants.append(p)
        for p in variants:
            with self.subTest(points=len(p['gauss_point_resultants'])),self.assertRaises(ValueError):
                recover_panel_faces(p,self.geometry,14.)

    def test_face_tension_comes_from_recovered_tensor_not_nearest_gauss_point(self):
        # The lower x face is x=7: mx=+0.4 (bottom tension), whereas the
        # closest GP in [0,10] is x=7.887: mx<0 (top tension). Its qx=10
        # governs over the qx=1 at the opposite face. The old nearest-GP
        # classification would incorrectly confirm top reinforcement here.
        def tensor(x,y,i,j):return dict(mx=7.4-x,my=1.,mxy_raw=0.,qx_raw=10. if i==0 else 1.,qy_raw=0.)
        p=panel([0,10,20,30,40],[0,10,20,30,40],tensor)
        result={'status':'transfer_complete','max_abs_membrane_kip_per_in':0.,
                'equilibrium':{'numerical_balance_passed':True},'panels':[p]}
        slab=dict(thickness_in=5.,concrete_fc_ksi=4.,concrete_unit_weight_kcf=.15,superimposed_dead_load_ksf=.05)
        sections=dict(b_beam_in=14.,h_beam_in=18.,fc_beam_ksi=4.)
        inputs=dict(thickness_in=5.,fc_ksi=4.,fy_ksi=60.,max_aggregate_size_in=.75,exposure='sheltered_interior',
                    steel_specification='ASTM A706',concrete_type='normalweight',num_floor=1,panel_ids=['panel_x1_y1'])
        with mock.patch('Design.SMRF_Slab_Actions.analyze_floor',side_effect=lambda *a,**k:copy.deepcopy(result)):
            flags={k:True for k in ('verified','analysis_applicability_verified','all_floors_enveloped',
                'load_pattern_envelope_verified','spatial_envelope_per_unit_width','twisting_moment_resolution_verified','zero_membrane_force_verified')}
            flags.update(asserted_by='unit test only',assertion_date='2026-09-24',assertion_basis='manufactured fixture, no engineering approval')
            evidence=build_slab_action_evidence(slab,self.geometry,sections,.05,inputs,mesh_per_bay=4,assertions=flags)
        row=next(r for r in evidence['strips'] if r['axis']=='x' and r['face']=='top')
        self.assertAlmostEqual(row['vu_kip_per_ft'],120.)
        self.assertFalse(row['tension_face_at_shear_verified'])
        self.assertFalse(evidence['shear_recovery']['top_rows_recovered_at_face'])
        self.assertFalse(evidence['verified'])
        self.assertEqual(row['shear_location']['top_demand_at_face_kip_in_per_in'],0.)

    def test_refinement_preserves_unstable_rows_and_rejects_different_problems(self):
        a=dict(physical_model_sha256='a'*64,analysis_model_sha256='b'*64,slab_input_sha256='c'*64,
               cases=[{'id':'case','live_pattern':'all'}],strips=[dict(panel_id='panel_x1_y1',axis='x',face='top',mu_kip_in_per_ft=10.,vu_kip_per_ft=0.)])
        b=copy.deepcopy(a);b['analysis_model_sha256']='d'*64
        b['strips'][0].update(mu_kip_in_per_ft=10.1,vu_kip_per_ft=.1)
        r=compare_slab_action_refinement(a,b,moment_tolerance=.02,shear_tolerance=.05)
        self.assertFalse(r['all_within_tolerance'])
        self.assertFalse(r['engineering_verified'])
        self.assertTrue(r['comparisons'][0]['within_tolerance'])
        self.assertIsNone(r['comparisons'][1]['relative_change'])
        for key,value in (('physical_model_sha256','z'*64),('cases',[]),('strips',[])):
            changed=copy.deepcopy(b);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                compare_slab_action_refinement(a,changed,moment_tolerance=.02,shear_tolerance=.05)

if __name__=='__main__':unittest.main()
