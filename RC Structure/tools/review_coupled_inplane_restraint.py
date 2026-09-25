"""Matched in-plane restraint comparison inside the coupled shell/web/column assembly.

The bare-frame variants of 2026-09-24 showed that releasing the frame's
rigid diaphragm moves column-end moments by about a third of the governing
demand. This review asks the converse question in the coupled model itself:
with the same webs, offsets, columns, materials and loads, what does a
declared rigid in-plane restraint do to the column actions, to the
whole-floor composite section actions and to the floor deformation?

Every run is a fresh coupled solve. The finite-membrane Transformation run
must reproduce the saved coupled reference column forces exactly; the
finite-membrane Lagrange run measures the handler alone; the restrained runs
use the same handler. Signed column-end moments are compared at both ends
of every column against the anchor and against the saved bare-frame
solution, each normalized by the saved governing design moment at that end
(not by the changed moment and not a capacity check). Whole-floor cuts are
recovered where the free bodies are defined (a cut cannot pass through a
constrained node, so the rigid-floor cuts are recorded as undefined).

These are attribution experiments only; nothing here corrects a demand or
selects a floor idealization.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys
import time

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
import openseespy.opensees as ops
from Design.SMRF_Composite_Sections import recover_floor_cut
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity

FIXED_DESIGN_SHA256 = 'f49291440328b9097efd44345246d11a420365a0bbd1dea8c810276890c43871'
ANCHOR = 'finite_membrane_transformation'
VARIANTS = {
    ANCHOR: dict(inplane_restraint='finite_membrane', constraint_handler='Transformation'),
    'finite_membrane_lagrange': dict(inplane_restraint='finite_membrane', constraint_handler='Lagrange'),
    'rigid_joints_lagrange': dict(inplane_restraint='rigid_joints', constraint_handler='Lagrange'),
    'rigid_floor_lagrange': dict(inplane_restraint='rigid_floor', constraint_handler='Lagrange'),
}
END_INDEXES = {'i': (4, 5), 'j': (10, 11)}   # local My, Mz at each end
AXIAL_INDEXES = {'i': 0, 'j': 6}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    raw = (json.dumps(data, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def rigid_fit(joints, cx, cy):
    """Least-squares rigid in-plane motion (tx, ty, rz) of a floor's column joints.

    Residual is the largest joint departure from that fitted field, in inches.
    Exact diaphragm ties give machine zero; a finite membrane does not.
    """
    rows, rhs = [], []
    for x, y, ux, uy in joints:
        rows.append([1., 0., -(y-cy)]); rhs.append(ux)
        rows.append([0., 1., (x-cx)]); rhs.append(uy)
    ata = [[math.fsum(r[a]*r[b] for r in rows) for b in range(3)] for a in range(3)]
    atb = [math.fsum(r[a]*v for r, v in zip(rows, rhs)) for a in range(3)]
    # 3x3 Gaussian elimination with partial pivoting.
    m = [ata[a]+[atb[a]] for a in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(3):
            if r != c:
                f = m[r][c]/m[c][c]
                m[r] = [a-f*b for a, b in zip(m[r], m[c])]
    fit = [m[a][3]/m[a][a] for a in range(3)]
    residual = max(abs(r[0]*fit[0]+r[1]*fit[1]+r[2]*fit[2]-v) for r, v in zip(rows, rhs))
    return fit, residual


def governing_moments(record):
    governing = {}
    for combo in record['design_actions']['combinations']:
        for tag, member in combo['members'].items():
            if member['member_type'] == 'column':
                for end, idx in END_INDEXES.items():
                    governing[tag, end] = max(governing.get((tag, end), 0.),
                                              math.hypot(*(member['local_force_kip_kipin'][i] for i in idx)))
    return governing


def column_rows(result, anchor, frame, governing, g):
    per_story = (g['num_bay_x']+1)*(g['num_bay_y']+1)
    anchor_by_tag = {c['tag']: c for c in anchor['column_actions']}
    rows = []
    for col in result['column_actions']:
        base = anchor_by_tag[col['tag']]
        if (base['story'], base['grid_i'], base['grid_j']) != (col['story'], col['grid_i'], col['grid_j']):
            raise ValueError('Column inventory differs between variants')
        frame_tag = str((col['story']-1)*per_story + col['grid_j']*(g['num_bay_x']+1) + col['grid_i'] + 1)
        for end, idx in END_INDEXES.items():
            v = [col['local_force_kip_kip_in'][i] for i in idx]
            a = [base['local_force_kip_kip_in'][i] for i in idx]
            f = [frame['columns'][frame_tag][i] for i in idx]
            denominator = governing[frame_tag, end]
            if denominator <= 0:
                raise ValueError('Governing moment required at every column end')
            rows.append(dict(tag=col['tag'], frame_tag=int(frame_tag), story=col['story'], grid_i=col['grid_i'],
                             grid_j=col['grid_j'], end=end, moments=v, anchor_moments=a, frame_moments=f,
                             axial_kip=col['local_force_kip_kip_in'][AXIAL_INDEXES[end]],
                             anchor_axial_kip=base['local_force_kip_kip_in'][AXIAL_INDEXES[end]],
                             frame_axial_kip=frame['columns'][frame_tag][AXIAL_INDEXES[end]],
                             governing_saved_demand=denominator,
                             change_from_anchor_over_governing=math.dist(v, a)/denominator,
                             gap_to_frame_over_governing=math.dist(v, f)/denominator,
                             anchor_gap_to_frame_over_governing=math.dist(a, f)/denominator))
    return rows


def floor_fits(result, g):
    cx, cy = g['num_bay_x']*g['bay_x_in']/2., g['num_bay_y']*g['bay_y_in']/2.
    fits = []
    for entry in result['floors']:
        joints = [(j['grid_i']*g['bay_x_in'], j['grid_j']*g['bay_y_in'],
                   j['displacement_rotation'][0], j['displacement_rotation'][1])
                  for j in entry['column_joint_displacements']]
        fit, residual = rigid_fit(joints, cx, cy)
        fits.append(dict(floor=entry['floor'], fitted_tx_ty_rz=fit, maximum_nonrigid_displacement_in=residual,
                         maximum_downward_displacement_in=entry['maximum_downward_displacement_in']))
    return fits


def all_cuts(result, g):
    return [recover_floor_cut(result, floor, axis, (bay+.5)*g[f'bay_{axis}_in'])
            for floor in range(1, g['num_floor']+1) for axis in ('x', 'y') for bay in range(g[f'num_bay_{axis}'])]


def cut_rows(cuts, anchor_cuts):
    """Left-side vertical bending and in-plane axial resultants against the anchor's."""
    bending = {'x': 4, 'y': 3}   # My across an x cut, Mx across a y cut
    axial = {'x': 0, 'y': 1}
    peak = {axis: max(abs(c['sides']['left']['total_force_moment'][bending[axis]])
                      for c in anchor_cuts if c['axis'] == axis) for axis in ('x', 'y')}
    rows = []
    for cut, base in zip(cuts, anchor_cuts):
        if (cut['floor'], cut['axis'], cut['position_in']) != (base['floor'], base['axis'], base['position_in']):
            raise ValueError('Cut inventories differ')
        axis, left, left0 = cut['axis'], cut['sides']['left'], base['sides']['left']
        m, m0 = left['total_force_moment'][bending[axis]], left0['total_force_moment'][bending[axis]]
        n, n0 = left['total_force_moment'][axial[axis]], left0['total_force_moment'][axial[axis]]
        rows.append(dict(floor=cut['floor'], axis=axis, position_in=cut['position_in'],
                         vertical_bending_kip_in=m, anchor_vertical_bending_kip_in=m0,
                         bending_change_over_anchor_axis_peak=abs(m-m0)/peak[axis],
                         inplane_axial_kip=n, anchor_inplane_axial_kip=n0,
                         components_kip_in={k: left['components'][k][bending[axis]]
                                            for k in ('shell_native', 'web_intrinsic', 'web_eccentricity', 'external')},
                         anchor_components_kip_in={k: left0['components'][k][bending[axis]]
                                                   for k in ('shell_native', 'web_intrinsic', 'web_eccentricity', 'external')},
                         numerical_balance_passed=cut['numerical_balance_passed']))
    return rows, peak


def run_review(design, references, output, meshes):
    raw = gzip.decompress(design.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FIXED_DESIGN_SHA256:
        raise ValueError('This review protocol is bounded to the fixed wider candidate')
    output.mkdir(parents=True, exist_ok=False)
    record = json.loads(raw)
    g = record['geometry']
    governing = governing_moments(record)
    sections = dict(record['sections'])
    sections.setdefault('beam_stiffness_modifier', 1.)
    sections.setdefault('column_stiffness_modifier', 1.)
    summary = dict(source_design_sha256=digest, source_file_sha256=sha(design), production_enabled=False,
                   engineering_verified=False, variants=VARIANTS, anchor=ANCHOR, meshes=meshes,
                   source_hashes={str(p.relative_to(RC)): sha(p) for p in
                                  [RC/'Design/SMRF_Coupled_Analysis.py', RC/'Design/SMRF_Composite_Sections.py', Path(__file__)]},
                   attempts=[], cases=[],
                   interpretation='Matched-assembly restraint sensitivity only; no corrected demands, no P-M checks, '
                                  'no selected floor idealization.')
    for case in (record['coupled_comparison']['case'], record['coupled_comparison']['asymmetric']['case']):
        frame_path = references/f"{case['id']}_frame.json"
        frame = json.loads(frame_path.read_text())
        summary['source_hashes'][str(frame_path)] = sha(frame_path)
        for mesh in meshes:
            name = f"{case['id']}_m{mesh}"
            entry = dict(name=name, mesh=mesh, case=case, variants={})
            summary['cases'].append(entry)
            results, cuts = {}, {}
            for key, options in VARIANTS.items():
                attempt = dict(name=name, variant=key, status='started')
                summary['attempts'].append(attempt)
                save(output/'summary.json', summary)
                start = time.perf_counter()
                try:
                    result = analyze_coupled_gravity(record['slab'], g, sections, [case]*g['num_floor'],
                                                     mesh_per_bay=mesh, **options)
                    save(output/f'{name}_{key}_raw.json.gz', result)
                    if result['status'] != 'diagnostic_complete':
                        raise RuntimeError(result['status'])
                    results[key] = result
                    item = dict(status='completed', rigid_offset_max_residual=result['rigid_offset_max_residual'],
                                diaphragm_max_residual=result['diaphragm_max_residual'],
                                constraint_action_check=result['constraint_action_check']['numerical_balance_passed'],
                                floor_fits=floor_fits(result, g))
                    if key == ANCHOR:
                        ref_path = references/f"{case['id']}_coupled_m{mesh}.json.gz"
                        original = json.loads(gzip.decompress(ref_path.read_bytes()))
                        old = {c['tag']: c['local_force_kip_kip_in'] for c in original['column_actions']}
                        difference = max(abs(f-v) for c in result['column_actions']
                                         for f, v in zip(c['local_force_kip_kip_in'], old[c['tag']]))
                        if difference > 1e-7:
                            raise RuntimeError(f'Anchor did not reproduce the saved coupled reference: {difference}')
                        item.update(reference=str(ref_path), reference_sha256=sha(ref_path),
                                    reference_max_absolute_difference=difference)
                    rows = column_rows(result, results[ANCHOR], frame, governing, g)
                    save(output/f'{name}_{key}_column_ends.json', rows)
                    item.update(column_end_count=len(rows),
                                maximum_change_from_anchor=max(rows, key=lambda r: r['change_from_anchor_over_governing']),
                                maximum_gap_to_frame=max(rows, key=lambda r: r['gap_to_frame_over_governing']),
                                anchor_maximum_gap_to_frame=max(rows, key=lambda r: r['anchor_gap_to_frame_over_governing']),
                                mean_gap_to_frame=math.fsum(r['gap_to_frame_over_governing'] for r in rows)/len(rows),
                                mean_change_from_anchor=math.fsum(r['change_from_anchor_over_governing'] for r in rows)/len(rows))
                    try:
                        cuts[key] = all_cuts(result, g)
                        save(output/f'{name}_{key}_cuts.json.gz', cuts[key])
                        if not all(c['numerical_balance_passed'] for c in cuts[key]):
                            raise RuntimeError('Whole-floor section equilibrium failed')
                        crows, peak = cut_rows(cuts[key], cuts[ANCHOR])
                        save(output/f'{name}_{key}_cut_rows.json', crows)
                        item.update(cut_count=len(crows), anchor_axis_peak_kip_in=peak,
                                    maximum_bending_change=max(crows, key=lambda r: r['bending_change_over_anchor_axis_peak']),
                                    maximum_force_relative_error=max(c['sides'][s]['force_relative_error'] for c in cuts[key] for s in ('left', 'right')),
                                    maximum_moment_relative_error=max(c['sides'][s]['moment_relative_error'] for c in cuts[key] for s in ('left', 'right')))
                    except ValueError as exc:
                        # A cut through constrained nodes has no defined free bodies.
                        item.update(cuts='undefined', cut_rejection=f'{type(exc).__name__}: {exc}')
                    entry['variants'][key] = item
                    attempt['status'] = 'completed'
                except Exception as exc:
                    attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                finally:
                    attempt['elapsed_seconds'] = time.perf_counter()-start
                    save(output/'summary.json', summary)
                    print(json.dumps({k: attempt[k] for k in ('name', 'variant', 'status', 'elapsed_seconds')}
                                     | ({'error': attempt['error']} if 'error' in attempt else {})), flush=True)
    if sha(design) != summary['source_file_sha256']:
        raise RuntimeError('Source design changed during review')
    summary['numerical_passed'] = all(a['status'] == 'completed' for a in summary['attempts'])
    save(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=RC/'outputs/validation_structure_20260924/design.json.gz')
    parser.add_argument('--references', type=Path, default=RC/'outputs/slab_face_recovery_20260924/compatibility')
    parser.add_argument('--meshes', default='8,10', help='comma-separated shells per bay; a saved reference must exist for each')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meshes = [int(m) for m in args.meshes.split(',')]
    raise SystemExit(0 if run_review(args.design, args.references, args.output, meshes)['numerical_passed'] else 1)
