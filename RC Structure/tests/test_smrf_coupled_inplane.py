"""Declared in-plane restraint and constraint-handler options of the coupled diagnostic.

These pin software behaviour (reproduction, kinematics, equilibrium closure,
rejection of unsupported combinations). They do not decide which restraint
is the right floor idealization.
"""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design.SMRF_Composite_Sections import recover_floor_cut
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity, METHOD_VERSION, SECTION_ACTION_SCHEMA
from tests.test_smrf_coupled_analysis import SLAB, GEOMETRY, SECTIONS, DEAD, ZERO

GEOMETRY_2X2 = dict(GEOMETRY, num_bay_x=2, num_bay_y=2)
CORNER = dict(DEAD, id='corner', dead_factor=1.2, live_factor=1.6, live_pattern=[[0, 0]])


def solve(**options):
    return analyze_coupled_gravity(SLAB, GEOMETRY_2X2, SECTIONS, [ZERO, CORNER], mesh_per_bay=4, **options)


def column_forces(result):
    return {c['tag']: c['local_force_kip_kip_in'] for c in result['column_actions']}


def joint_rigid_residual(result, floor):
    """Largest in-plane departure of the column joints from the master's rigid field."""
    entry = result['floors'][floor - 1]
    master = entry['diaphragm_master_displacement_rotation']
    cx, cy = GEOMETRY_2X2['num_bay_x']*GEOMETRY_2X2['bay_x_in']/2, GEOMETRY_2X2['num_bay_y']*GEOMETRY_2X2['bay_y_in']/2
    worst = 0.
    for joint in entry['column_joint_displacements']:
        x, y = joint['grid_i']*GEOMETRY_2X2['bay_x_in'], joint['grid_j']*GEOMETRY_2X2['bay_y_in']
        u = joint['displacement_rotation']
        expected = [master[0]-master[5]*(y-cy), master[1]+master[5]*(x-cx), master[5]]
        worst = max(worst, abs(u[0]-expected[0]), abs(u[1]-expected[1]), abs(u[5]-expected[2]))
    return worst


class InplaneRestraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ops.wipe()
        cls.baseline = solve()
        cls.lagrange = solve(constraint_handler='Lagrange')
        cls.joints = solve(inplane_restraint='rigid_joints', constraint_handler='Lagrange')
        cls.floor = solve(inplane_restraint='rigid_floor', constraint_handler='Lagrange')

    def setUp(self):
        ops.wipe()
        self.addCleanup(ops.wipe)

    def test_default_record_declares_the_finite_membrane_transformation_assembly(self):
        r = self.baseline
        self.assertEqual(r['status'], 'diagnostic_complete')
        self.assertEqual(r['method_version'], METHOD_VERSION)
        self.assertEqual(r['section_action_schema'], SECTION_ACTION_SCHEMA)
        for record in (r['inputs'], r['assembly']):
            self.assertEqual((record['inplane_restraint'], record['constraint_handler'], record['penalty_alpha']),
                             ('finite_membrane', 'Transformation', None))
        self.assertEqual(r['assembly']['constrained_node_count_per_floor'], 0)
        self.assertEqual(r['assembly']['diaphragm_masters'], [])
        self.assertEqual(r['diaphragm_constraint_actions'], [])
        self.assertEqual(r['diaphragm_max_residual'], 0.)
        self.assertTrue(r['constraint_action_check']['numerical_balance_passed'])
        self.assertIsNone(r['floors'][0]['diaphragm_master_displacement_rotation'])
        json.dumps(r, allow_nan=False)

    def test_lagrange_reproduces_the_transformation_baseline(self):
        self.assertEqual(self.lagrange['status'], 'diagnostic_complete')
        a, b = column_forces(self.baseline), column_forces(self.lagrange)
        self.assertEqual(set(a), set(b))
        self.assertLess(max(abs(x-y) for t in a for x, y in zip(a[t], b[t])), 1e-7)
        webs = max(abs(x-y) for p, q in zip(self.baseline['web_segment_actions'], self.lagrange['web_segment_actions'])
                   for x, y in zip(p['local_force_kip_kip_in'], q['local_force_kip_kip_in']))
        self.assertLess(webs, 1e-7)
        for p, q in zip(self.baseline['floors'], self.lagrange['floors']):
            for m, n in zip(p['column_joint_displacements'], q['column_joint_displacements']):
                for x, y in zip(m['displacement_rotation'], n['displacement_rotation']):
                    self.assertAlmostEqual(x, y, places=10)
        self.assertEqual(self.lagrange['inputs']['constraint_handler'], 'Lagrange')

    def test_rigid_joints_tie_the_joints_and_rigid_floor_ties_every_slab_node(self):
        nx, ny, mesh = GEOMETRY_2X2['num_bay_x'], GEOMETRY_2X2['num_bay_y'], 4
        for result, tied in ((self.joints, (nx+1)*(ny+1)), (self.floor, (nx*mesh+1)*(ny*mesh+1))):
            with self.subTest(restraint=result['inputs']['inplane_restraint']):
                self.assertEqual(result['status'], 'diagnostic_complete')
                self.assertEqual(result['assembly']['constrained_node_count_per_floor'], tied)
                self.assertEqual([m['constrained_nodes'] for m in result['assembly']['diaphragm_masters']], [tied, tied])
                self.assertEqual(len(result['diaphragm_constraint_actions']), 2*tied)
                self.assertLess(result['diaphragm_max_residual'], 1e-10)
                for floor in (1, 2):
                    self.assertLess(joint_rigid_residual(result, floor), 1e-10)
                self.assertTrue(result['equilibrium']['numerical_balance_passed'])
                self.assertTrue(result['shell_to_frame_equilibrium']['numerical_balance_passed'])
        # Interior nodes carry constraint forces only under the rigid floor.
        interior = [a for a in self.floor['diaphragm_constraint_actions'] if a['grid_i'] is None]
        self.assertEqual(len(interior), 2*((nx*mesh+1)*(ny*mesh+1)-(nx+1)*(ny+1)))
        self.assertTrue(all(a['grid_i'] is not None for a in self.joints['diaphragm_constraint_actions']))
        self.assertGreater(max(math.hypot(a['force_moment'][0], a['force_moment'][1]) for a in interior), 1e-6)

    def test_restraint_changes_the_column_actions_and_the_two_restraints_differ(self):
        f, j, r = column_forces(self.baseline), column_forces(self.joints), column_forces(self.floor)
        def largest(a, b):
            return max(abs(x-y) for t in a for x, y in zip(a[t], b[t]))
        self.assertGreater(largest(f, j), 1e-3)
        self.assertGreater(largest(f, r), 1e-3)
        self.assertGreater(largest(j, r), 1e-6)

    def test_constraint_actions_close_node_equilibrium_and_balance_about_the_master(self):
        for result in (self.joints, self.floor):
            with self.subTest(restraint=result['inputs']['inplane_restraint']):
                check = result['constraint_action_check']
                self.assertTrue(check['numerical_balance_passed'])
                self.assertLess(check['free_node_max_residual_kip'], 1e-9)
                for entry in check['per_floor_diaphragm_balance']:
                    self.assertLess(entry['force_relative_error'], 1e-10)
                    self.assertLess(entry['moment_relative_error'], 1e-10)
                    self.assertLess(entry['max_out_of_plane_component'], 1e-9)
                    self.assertLess(max(abs(v) for v in entry['sum_about_master']), 1e-7)

    def test_rigid_joint_cuts_close_only_with_the_constraint_actions(self):
        for floor in (1, 2):
            for axis, positions in (('x', (120., 360.)), ('y', (100., 300.))):
                for position in positions:
                    with self.subTest(floor=floor, axis=axis, position=position):
                        cut = recover_floor_cut(self.joints, floor, axis, position)
                        self.assertTrue(cut['numerical_balance_passed'])
        zeroed = copy.deepcopy(self.joints)
        for action in zeroed['diaphragm_constraint_actions']:
            action['force_moment'] = [0.]*6
        cut = recover_floor_cut(zeroed, 2, 'x', 120.)
        self.assertFalse(cut['numerical_balance_passed'])
        missing = copy.deepcopy(self.joints)
        missing['diaphragm_constraint_actions'].pop(0)
        with self.assertRaisesRegex(ValueError, 'constraint-action inventory'):
            recover_floor_cut(missing, 1, 'x', 120.)
        stale = copy.deepcopy(self.joints)
        stale['assembly']['inplane_restraint'] = 'finite_membrane'
        with self.assertRaisesRegex(ValueError, 'constraint-action inventory'):
            recover_floor_cut(stale, 1, 'x', 120.)

    def test_rigid_floor_cuts_through_constrained_nodes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'external point action'):
            recover_floor_cut(self.floor, 2, 'x', 120.)

    def test_unsupported_and_invalid_options_are_rejected_before_any_domain_is_built(self):
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        ops.node(987, 0., 0., 0.)
        bad = [dict(inplane_restraint='rigid_joints'), dict(inplane_restraint='rigid_floor'),
               dict(inplane_restraint='rigid'), dict(constraint_handler='Plain'),
               dict(constraint_handler='Penalty', penalty_alpha=0.), dict(constraint_handler='Penalty', penalty_alpha=True),
               dict(constraint_handler='Penalty', penalty_alpha=float('nan'))]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                solve(**options)
            self.assertEqual(ops.getNodeTags(), [987])
        with self.assertRaisesRegex(ValueError, 'Transformation'):
            solve(inplane_restraint='rigid_joints', constraint_handler='Transformation')

    def test_penalty_handler_records_its_alpha_and_reports_its_slip(self):
        result = solve(inplane_restraint='rigid_joints', constraint_handler='Penalty', penalty_alpha=1e9)
        self.assertEqual(result['inputs']['penalty_alpha'], 1e9)
        self.assertEqual(result['assembly']['penalty_alpha'], 1e9)
        self.assertTrue(math.isfinite(result['diaphragm_max_residual']))
        self.assertTrue(math.isfinite(result['rigid_offset_max_residual']))
        self.assertIn(result['status'], ('diagnostic_complete', 'equilibrium_failed'))


if __name__ == '__main__':
    unittest.main()
