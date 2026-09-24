"""Reproducible composite-section benchmark and fixed-candidate gravity review.

This exports diagnostic evidence only; it does not change a design or qualify
member reinforcement. Run in a dedicated process with an empty OpenSees domain.
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
from Design.SMRF_Composite_Sections import recover_planar_cut, recover_floor_cut
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity


def save(path, data):
    raw = (json.dumps(data, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analytical_cantilever(mesh=8):
    """Exact constant-curvature elastic composite solution, with nu=0.

    Apply the analytical end stress resultants to the flange and web separately:
    N_a = E A_a (z_a-z_c) kappa, M_a = E I_a kappa. Their net force is zero
    and their total moment is 1000 kip-in. This avoids end shear lag and tests
    stiffness, eccentric coupling and recovered actions against beam theory.
    It is deliberately not a test of arbitrary end loading, cracking or torsion.
    """
    if isinstance(mesh, bool) or mesh not in (4, 8, 16):
        raise ValueError('Benchmark mesh must be 4, 8 or 16')
    if ops.getNodeTags() or ops.getEleTags():
        raise RuntimeError('Benchmark requires an empty domain; existing model was preserved')
    e, nu, length, width, hs, bw, hw, zw, moment, ny = 4000., 0., 240., 60., 6., 14., 22., -14., 1000., 4
    af, aw = width*hs, bw*hw
    zc = aw*zw/(af+aw)
    ifl, iw = width*hs**3/12., bw*hw**3/12.
    inertia = ifl+iw+af*zc**2+aw*(zw-zc)**2
    curvature = moment/(e*inertia)
    nf, nw = e*af*(-zc)*curvature, e*aw*(zw-zc)*curvature
    positions, nodes, web_nodes, shells, webs, external = {}, {}, {}, [], [], []
    try:
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        for i in range(mesh+1):
            for j in range(ny+1):
                n = len(positions)+1
                positions[n] = [length*i/mesh, width*(j/ny-.5), 0.]
                nodes[i,j] = n
                ops.node(n, *positions[n])
                if i == 0:
                    ops.fix(n, 1,1,1,1,1,1)
        for i in range(mesh+1):
            n = len(positions)+1
            positions[n] = [length*i/mesh, 0., zw]
            ops.node(n, *positions[n])
            web_nodes[i] = n
            ops.rigidLink('beam', nodes[i,ny//2], n)
        ops.section('ElasticMembranePlateSection', 1, e, nu, hs, 0.)
        ops.geomTransf('Linear', 1, 0,0,1)
        tag = 0
        for i in range(mesh):
            for j in range(ny):
                tag += 1
                ns = [nodes[i,j],nodes[i+1,j],nodes[i+1,j+1],nodes[i,j+1]]
                ops.element('ShellMITC4', tag, *ns, 1)
                shells.append(dict(tag=tag, node_positions_in=[positions[n] for n in ns],
                                   applied_nodal_force_kip_kip_in=[0.]*24))
            tag += 1
            ns = [web_nodes[i],web_nodes[i+1]]
            iz = hw*bw**3/12.
            ops.element('elasticBeamColumn', tag, *ns, aw,e,e/2,iw+iz,iw,iz,1)
            webs.append(dict(tag=tag, node_positions_in=[positions[n] for n in ns],
                             body_load_force_kip=[0.]*3, body_load_position_in=[length*(i+.5)/mesh,0.,zw]))
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        for j in range(ny+1):
            fraction = (.5 if j in (0,ny) else 1.)/ny
            f = [nf*fraction,0.,0.,0.,e*ifl*curvature*fraction,0.]
            ops.load(nodes[mesh,j], *f)
            external.append(dict(position_in=positions[nodes[mesh,j]], force_moment=f, source='tip_flange'))
        f = [nw,0.,0.,0.,e*iw*curvature,0.]
        ops.load(web_nodes[mesh], *f)
        external.append(dict(position_in=positions[web_nodes[mesh]], force_moment=f, source='tip_web'))
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('UmfPack')
        ops.algorithm('Linear')
        ops.integrator('LoadControl', 1.)
        ops.analysis('Static')
        code = ops.analyze(1)
        if code:
            raise RuntimeError(f'Analytical benchmark solve failed: {code}')
        ops.reactions()
        # Native nodeReaction does not assemble MP forces onto the retained node.
        # Include the clamped web slave and transport its reaction at its real z.
        for n in [nodes[0,j] for j in range(ny+1)]+[web_nodes[0]]:
            external.append(dict(position_in=positions[n], force_moment=list(ops.nodeReaction(n)), source='root'))
        for s in shells:
            s['global_nodal_force_kip_kip_in'] = list(ops.eleForce(s['tag']))
        for w in webs:
            w['global_force_kip_kip_in'] = list(ops.eleForce(w['tag']))
        cut = recover_planar_cut(shells, webs, external, axis='x', position_in=length/2,
                                 reference_in=[length/2,0.,0.])
        expected = [-zc*curvature*length,0.,-curvature*length**2/2,0.,curvature*length,0.]
        observed = list(ops.nodeDisp(nodes[mesh,ny//2]))
        displacement_error = max(abs(observed[i]/expected[i]-1.) for i in (0,2,4))
        expected_components = dict(shell_native=e*ifl*curvature, web_intrinsic=e*iw*curvature,
                                   web_eccentricity=zw*nw)
        component_error = max(abs(cut['sides']['left']['components'][name][4]-v)/moment
                              for name,v in expected_components.items())
        return dict(mesh=mesh, status='completed', engineering_verified=False,
                    inputs=dict(e_ksi=e, nu=nu, length_in=length, flange_width_in=width, slab_thickness_in=hs,
                                web_width_in=bw, web_depth_in=hw, web_offset_in=zw, applied_moment_kip_in=moment),
                    analytical=dict(neutral_axis_z_in=zc, inertia_in4=inertia, curvature_per_in=curvature,
                                    flange_axial_kip=nf, web_axial_kip=nw, tip_displacement_rotation=expected,
                                    left_cut_my_components_kip_in=expected_components),
                    observed_tip_displacement_rotation=observed,
                    maximum_nonzero_displacement_relative_error=displacement_error,
                    maximum_component_error_over_applied_moment=component_error,
                    numerical_passed=(displacement_error<1e-8 and component_error<1e-8 and cut['numerical_balance_passed']),
                    cut=cut, shells=shells, webs=webs, external_actions=external)
    finally:
        ops.wipe()


def run_review(design, references, output):
    raw = gzip.decompress(design.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'f49291440328b9097efd44345246d11a420365a0bbd1dea8c810276890c43871':
        raise ValueError('This review protocol is bounded to the fixed wider candidate')
    output.mkdir(parents=True, exist_ok=False)
    record = json.loads(raw)
    g = record['geometry']
    summary = dict(source_design_sha256=digest, source_file_sha256=sha(design), production_enabled=False,
                   engineering_verified=False, source_hashes={str(p.relative_to(RC)):sha(p) for p in
                   [RC/'Design/SMRF_Coupled_Analysis.py', RC/'Design/SMRF_Composite_Sections.py', Path(__file__)]},
                   attempts=[], benchmarks=[], cases=[])
    for mesh in (4,8,16):
        benchmark = analytical_cantilever(mesh)
        save(output/f'cantilever_m{mesh}.json.gz', benchmark)
        summary['benchmarks'].append({k:benchmark[k] for k in ('mesh','numerical_passed',
            'maximum_nonzero_displacement_relative_error','maximum_component_error_over_applied_moment')})
    for case in (record['coupled_comparison']['case'], record['coupled_comparison']['asymmetric']['case']):
        for mesh in (8,10):
            name = f"{case['id']}_m{mesh}"
            attempt = dict(name=name, status='started')
            summary['attempts'].append(attempt)
            save(output/'summary.json', summary)
            start = time.perf_counter()
            try:
                # Saved reference spells out the same gross-stiffness defaults.
                sections = dict(record['sections'])
                sections.setdefault('beam_stiffness_modifier', 1.)
                sections.setdefault('column_stiffness_modifier', 1.)
                result = analyze_coupled_gravity(record['slab'],g,sections,[case]*g['num_floor'],mesh_per_bay=mesh)
                save(output/(name+'_raw.json.gz'), result)
                if result['status'] != 'diagnostic_complete':
                    raise RuntimeError(result['status'])
                ref_path = references/f"{case['id']}_coupled_m{mesh}.json.gz"
                original = json.loads(gzip.decompress(ref_path.read_bytes()))
                if result['inputs'] != original['inputs']:
                    raise ValueError('Saved coupled reference input mismatch')
                old = {c['tag']:c['local_force_kip_kip_in'] for c in original['column_actions']}
                if set(old) != {c['tag'] for c in result['column_actions']}:
                    raise ValueError('Column inventory mismatch')
                difference = max(abs(f-v) for c in result['column_actions']
                                 for f,v in zip(c['local_force_kip_kip_in'],old[c['tag']]))
                if difference > 1e-7:
                    raise RuntimeError('Export-only change altered saved column forces')
                cuts = [recover_floor_cut(result,floor,axis,(bay+.5)*g[f'bay_{axis}_in'])
                        for floor in range(1,g['num_floor']+1) for axis in ('x','y') for bay in range(g[f'num_bay_{axis}'])]
                save(output/(name+'_cuts.json.gz'), cuts)
                if not all(c['numerical_balance_passed'] for c in cuts):
                    raise RuntimeError('Whole-floor section equilibrium failed')
                summary['cases'].append(dict(name=name, mesh=mesh, case=case, cut_count=len(cuts),
                    column_count=len(old), reference_sha256=sha(ref_path), reference=str(ref_path),
                    maximum_column_force_change=difference,
                    maximum_force_relative_error=max(c['sides'][s]['force_relative_error'] for c in cuts for s in ('left','right')),
                    maximum_moment_relative_error=max(c['sides'][s]['moment_relative_error'] for c in cuts for s in ('left','right')),
                    maximum_continuity_relative_error=max(max(c['force_continuity_relative_error'],c['moment_continuity_relative_error']) for c in cuts)))
                attempt['status'] = 'completed'
            except Exception as exc:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                attempt['elapsed_seconds'] = time.perf_counter()-start
                save(output/'summary.json', summary)
                print(json.dumps(attempt), flush=True)
    if sha(design) != summary['source_file_sha256']:
        raise RuntimeError('Source design changed during review')
    summary['numerical_passed'] = all(b['numerical_passed'] for b in summary['benchmarks']) and all(
        a['status']=='completed' for a in summary['attempts'])
    save(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=RC/'outputs/validation_structure_20260924/design.json.gz')
    parser.add_argument('--references', type=Path, default=RC/'outputs/slab_face_recovery_20260924/compatibility')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run_review(args.design,args.references,args.output)['numerical_passed'] else 1)
