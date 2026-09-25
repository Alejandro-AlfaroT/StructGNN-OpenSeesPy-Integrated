"""Metrics and a Markdown table from a review_coupled_inplane_restraint output folder.

Reads the saved per-variant column-end and cut rows, writes ``metrics.json``
(paired effects, gap distributions, worst ends, mesh sensitivity of every
effect between the solved meshes, floor rigid-motion fits, cut resultants)
and prints the table used in the review. Nothing is re-solved.
"""
from pathlib import Path
import argparse
import json
import math


def percentiles(values):
    v = sorted(values)
    n = len(v)
    return {'p50': v[n//2], 'p90': v[int(.9*n)], 'max': v[-1], 'mean': math.fsum(v)/n,
            'count_over_5pct': sum(x > .05 for x in v), 'count': n}


def load_rows(root, name, key, kind):
    path = root/f'{name}_{key}_{kind}.json'
    return json.load(open(path, encoding='utf-8')) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='folder written by review_coupled_inplane_restraint.py')
    args = parser.parse_args()
    root = args.output
    summary = json.load(open(root/'summary.json', encoding='utf-8'))
    anchor = summary['anchor']
    metrics = {'source_design_sha256': summary['source_design_sha256'], 'anchor': anchor,
               'numerical_passed': summary.get('numerical_passed'), 'engineering_verified': False,
               'normalization': 'Column-end moment changes and gaps are the two-component (My, Mz) vector '
                                'distance divided by the saved governing design moment at that same end. '
                                'Cut changes are divided by the anchor axis peak of the vertical-bending resultant.',
               'cases': {}, 'mesh_sensitivity': {}}
    lines = ['| case | mesh | variant | max change vs anchor | mean change | max gap to frame | mean gap | ends >5% gap | max cut bending change |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    by_case_mesh = {}
    for case in summary['cases']:
        name, mesh = case['name'], case['mesh']
        entry = metrics['cases'].setdefault(name, {'mesh': mesh, 'case': case['case'], 'variants': {}})
        for key, v in case['variants'].items():
            if v.get('status') != 'completed':
                entry['variants'][key] = {'status': v.get('status')}
                continue
            rows = load_rows(root, name, key, 'column_ends')
            changes = [r['change_from_anchor_over_governing'] for r in rows]
            gaps = [r['gap_to_frame_over_governing'] for r in rows]
            worst_change = max(rows, key=lambda r: r['change_from_anchor_over_governing'])
            worst_gap = max(rows, key=lambda r: r['gap_to_frame_over_governing'])
            item = {'status': 'completed', 'assembly': summary['variants'][key],
                    'change_from_anchor': percentiles(changes), 'gap_to_frame': percentiles(gaps),
                    'worst_change_end': worst_change, 'worst_gap_end': worst_gap,
                    'residuals': {'rigid_offset': v['rigid_offset_max_residual'],
                                  'diaphragm': v['diaphragm_max_residual'],
                                  'constraint_actions_closed': v['constraint_action_check']},
                    'floor_fits': v['floor_fits']}
            if 'reference_max_absolute_difference' in v:
                item['reference_reproduction_max_abs_difference'] = v['reference_max_absolute_difference']
            cut_text = 'undefined'
            if 'cut_count' in v:
                cuts = load_rows(root, name, key, 'cut_rows')
                item['cuts'] = {'count': v['cut_count'], 'anchor_axis_peak_kip_in': v['anchor_axis_peak_kip_in'],
                                'max_bending_change_over_axis_peak': v['maximum_bending_change'],
                                'max_free_body_relative_error': max(v['maximum_force_relative_error'], v['maximum_moment_relative_error']),
                                'rows': cuts}
                cut_text = f"{100*v['maximum_bending_change']['bending_change_over_anchor_axis_peak']:.4f}%"
            else:
                item['cuts'] = {'status': 'undefined', 'reason': v.get('cut_rejection')}
            entry['variants'][key] = item
            by_case_mesh[case['case']['id'], mesh, key] = (rows, item)
            lines.append(f"| {case['case']['id']} | {mesh} | {key} | {100*item['change_from_anchor']['max']:.4f}% | "
                         f"{100*item['change_from_anchor']['mean']:.4f}% | {100*item['gap_to_frame']['max']:.4f}% | "
                         f"{100*item['gap_to_frame']['mean']:.4f}% | {item['gap_to_frame']['count_over_5pct']}/{item['gap_to_frame']['count']} | {cut_text} |")
    meshes = sorted({m for _, m, _ in by_case_mesh})
    if len(meshes) >= 2:
        coarse, fine = meshes[-2], meshes[-1]
        for (cid, mesh, key), (rows, item) in by_case_mesh.items():
            if mesh != fine or (cid, coarse, key) not in by_case_mesh:
                continue
            crows = {(r['story'], r['grid_i'], r['grid_j'], r['end']): r for r in by_case_mesh[cid, coarse, key][0]}
            frows = {(r['story'], r['grid_i'], r['grid_j'], r['end']): r for r in rows}
            if set(crows) != set(frows):
                raise ValueError('Column-end inventories differ between meshes')
            sens = {'meshes': [coarse, fine],
                    'max_abs_difference_of_change_over_governing': max(
                        abs(frows[k]['change_from_anchor_over_governing']-crows[k]['change_from_anchor_over_governing']) for k in frows),
                    'max_abs_difference_of_gap_over_governing': max(
                        abs(frows[k]['gap_to_frame_over_governing']-crows[k]['gap_to_frame_over_governing']) for k in frows),
                    'max_abs_moment_vector_difference_over_governing': max(
                        math.dist(frows[k]['moments'], crows[k]['moments'])/frows[k]['governing_saved_demand'] for k in frows)}
            citem = by_case_mesh[cid, coarse, key][1]
            if 'rows' in item.get('cuts', {}) and 'rows' in citem.get('cuts', {}):
                fc = {(r['floor'], r['axis'], r['position_in']): r for r in item['cuts']['rows']}
                cc = {(r['floor'], r['axis'], r['position_in']): r for r in citem['cuts']['rows']}
                peak = item['cuts']['anchor_axis_peak_kip_in']
                sens['max_abs_difference_of_cut_bending_over_axis_peak'] = max(
                    abs(fc[k]['vertical_bending_kip_in']-cc[k]['vertical_bending_kip_in'])/peak[k[1]] for k in fc)
                sens['max_abs_difference_of_cut_bending_change_over_axis_peak'] = max(
                    abs(fc[k]['bending_change_over_anchor_axis_peak']-cc[k]['bending_change_over_anchor_axis_peak']) for k in fc)
            metrics['mesh_sensitivity'][f'{cid}/{key}'] = sens
    (root/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    print('\nmesh sensitivity (fine minus coarse):')
    for k, s in metrics['mesh_sensitivity'].items():
        print(f"  {k}: change {100*s['max_abs_difference_of_change_over_governing']:.4f}%  gap {100*s['max_abs_difference_of_gap_over_governing']:.4f}%"
              f"  moments {100*s['max_abs_moment_vector_difference_over_governing']:.4f}%"
              + (f"  cut bending {100*s['max_abs_difference_of_cut_bending_over_axis_peak']:.4f}%  cut change {100*s['max_abs_difference_of_cut_bending_change_over_axis_peak']:.4f}%"
                 if 'max_abs_difference_of_cut_bending_over_axis_peak' in s else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
