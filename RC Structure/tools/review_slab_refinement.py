"""Reproduce the fixed desktop candidate through the integrated refinement path.

All loads are solved separately. Original designs and benchmark arrays are
read-only; raw solutions, failed attempts and source hashes go to a new folder.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
import time

import numpy as np

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence
from Design.SMRF_Slab_Actions import evaluate_slab_actions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=RC / 'outputs/validation_structure_20260924/design.json.gz')
    parser.add_argument('--reference-dir', type=Path, default=RC / 'outputs/slab_beam_benchmark_20260924')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = gzip.decompress(args.design.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'f49291440328b9097efd44345246d11a420365a0bbd1dea8c810276890c43871':
        raise ValueError('This reproduction protocol is bounded to the unchanged wider candidate')
    record = json.loads(raw)
    geometry, slab = record['geometry'], record['slab']
    sections = {k: record['sections'][k] for k in ('b_beam_in', 'h_beam_in', 'fc_beam_ksi', 'b_col_in', 'h_col_in')}
    names = ['actual_graded_24', 'actual_graded_48', 'actual_graded_60']
    specs, source_hashes = [], {'design_decompressed': digest, str(args.design): sha(args.design)}
    for name in names:
        path = args.reference_dir / (name + '.npz')
        with np.load(path) as saved:
            specs.append(dict(x_offsets_in=saved['xs'][saved['xs'] <= geometry['bay_x_in']].tolist(),
                              y_offsets_in=saved['ys'][saved['ys'] <= geometry['bay_y_in']].tolist(),
                              max_shells=45000))
        source_hashes[str(path)] = sha(path)
    policy = dict(meshes=specs, moment_tolerance=.05, shear_tolerance=.05,
                  tolerance_basis='Inherited 5% all-strip investigation screen from the desktop benchmark; '
                                  'bounded support-region refinement, not an ACI criterion or global error bound')
    write(args.output / 'policy.json', policy)
    source_hashes.update({str(p.relative_to(RC)): sha(p) for p in (RC / 'Design').glob('*.py')})
    write(args.output / 'provenance.json', dict(source_design_sha256=digest, sources=source_hashes,
                                               raw_cases_independently_solved=True, production_enabled=False))
    attempts = []

    def observe(level, case, result):
        index = len(attempts)
        target = args.output / f'level{level}_case{index}.json.gz'
        with gzip.open(target, 'wt', encoding='utf-8') as f:
            json.dump(result, f, allow_nan=False)
        item = dict(level=level, case=case, status=result['status'], raw_path=target.name, sha256=sha(target))
        if result['status'] == 'transfer_complete':
            pressure = case['dead_factor'] * .125 + case['live_factor'] * .05
            factor = pressure / .23
            with np.load(args.reference_dir / (names[level] + '.npz')) as saved:
                tensor = np.empty_like(saved['resultants'])
                ex = saved['resultants'].shape[1]
                for panel in result['panels']:
                    for point in panel['gauss_point_resultants']:
                        element = point['element'] - 1
                        tensor[element // ex, element % ex, point['gauss_point'] - 1] = [
                            point[k] for k in ('mx', 'my', 'mxy_raw', 'qx_raw', 'qy_raw')]
                reference = saved['resultants'] * factor
                delta = np.max(abs(tensor - reference), axis=(0, 1, 2))
                scales = np.max(abs(reference), axis=(0, 1, 2))
                w = np.array([p['uz_in'] for p in result['vertical_displacements']])
                reference_w = saved['displacement'][:, 2] * factor
                item['independent_assembly_comparison'] = dict(
                    fields=['mx', 'my', 'mxy_raw', 'qx_raw', 'qy_raw'],
                    max_absolute_difference=delta.tolist(),
                    relative_to_field_peak=(delta / scales).tolist(),
                    displacement_peak_normalized_difference=float(np.max(abs(w - reference_w)) / np.max(abs(reference_w))),
                    tolerance=1e-7,
                    all_within_tolerance=bool(np.max(delta / scales) <= 1e-7 and
                                             np.max(abs(w - reference_w)) / np.max(abs(reference_w)) <= 1e-7))
            item['mesh'] = {k: result['mesh'][k] for k in ('shell_count', 'coordinate_sha256', 'solver')}
            item['equilibrium'] = result['equilibrium']
            item['transfer_equilibrium'] = result['transfer_equilibrium']
        attempts.append(item)
        write(args.output / 'attempts.json', attempts)
        print(json.dumps({k: item[k] for k in ('level', 'case', 'status')}), flush=True)

    start = time.perf_counter()
    evidence = build_refined_slab_action_evidence(slab, geometry, sections, .05,
                                                 record['slab_reinforcement_inputs'], policy,
                                                 assertions=record['slab_actions']['engineering_assertions'],
                                                 case_observer=observe)
    write(args.output / 'slab_actions.json', evidence)
    summary = dict(elapsed_seconds=time.perf_counter() - start, source_design_sha256=digest,
                   refinement_status=evidence['refinement']['status'], verified=evidence['verified'],
                   production_enabled=False, checks=evaluate_slab_actions(evidence),
                   comparisons=[], raw_case_count=len(attempts),
                   independent_assembly_passed=bool(attempts) and all(
                       a.get('independent_assembly_comparison', {}).get('all_within_tolerance') is True for a in attempts))
    for comparison in evidence['refinement']['comparisons']:
        rows = comparison['comparisons']
        summary['comparisons'].append(dict(all_within_tolerance=comparison['all_within_tolerance'],
                                          failed_count=sum(not r['within_tolerance'] for r in rows),
                                          comparison_count=len(rows),
                                          max_moment_relative_change=max(r['relative_change'] or 0. for r in rows if r['metric']=='mu_kip_in_per_ft'),
                                          max_shear_relative_change=max(r['relative_change'] or 0. for r in rows if r['metric']=='vu_kip_per_ft')))
    if hashlib.sha256(gzip.decompress(args.design.read_bytes())).hexdigest() != digest:
        raise RuntimeError('Source design changed during review')
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['refinement_status'] == 'passed' and summary['independent_assembly_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
