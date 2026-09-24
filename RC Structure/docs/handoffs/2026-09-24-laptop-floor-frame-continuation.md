# Floor/frame compatibility and composite section recovery — September 24

Added equilibrium-complete section recovery to the coupled gravity diagnostic.
Controlled comparisons identify in-plane floor restraint as the strongest of
the tested sensitivities. Compatibility remains open; releasing the diaphragm
is not an established correction.

## Measured effects

Twelve bare-frame analyses cover six variants and two gravity patterns. Every
comparison includes both ends of all 126 columns, with the same saved loads.
The two baseline variants reproduce the desktop frame forces exactly.

The table uses paired variants differing in the named assumption only. Values
are the largest change in the two-component column-end moment vector divided by
the **saved governing moment demand at that same end**. They are not percentages
of the changed case moment, capacity utilization, or a uniform correction factor.

| Isolated assumption | Full live | Alternate-X live |
|---|---:|---:|
| Column PDelta → Linear | 0.0000% | 0.0179% |
| Penalty → Transformation | 0.1477% | 0.1298% |
| Beam flange inertia convention | 0.8552% | 0.7406% |
| Release in-plane diaphragm ties | 33.3749% | 28.0879% |

The frame and floor routines use different effective-flange conventions. Swapping
only the frame inertia has a much smaller effect here than releasing its in-plane
ties. Retaining the alternate flange convention still gives diaphragm sensitivities
of 33.0973% and 27.8581%, respectively.

Releasing ties leaves worst gaps to the finite-membrane coupled model of 14.8253%
and 12.4591% of governing demand. The original gaps were 18.5499% and 16.7536%.
Thus a lower maximum gap does not make the released frame an accepted model.
Differences also vary by location and sign.

A least-squares rigid-motion fit to each floor's column-joint translations gives
maximum nonrigid residuals of 0.002068 in and 0.001741 in in the coupled model,
compared with 0.006328 in and 0.005624 in in the released frame. Exact diaphragm
ties give effectively zero. This confirms three different in-plane deformation
treatments; it does not by itself decide which approximation is appropriate.

## Implemented recovery

`SMRF_Coupled_Analysis.py` now exports native shell nodal forces, the corresponding
applied nodal loads, global beam/column end actions, their actual coordinates,
beam body-load resultants, and column actions on each floor. The stiffness,
connectivity and loading are unchanged. Four fresh coupled solves reproduce
every saved column end-force component **exactly**, at meshes 8 and 10 per bay,
for both gravity patterns.

`SMRF_Composite_Sections.py` recovers sections spanning the complete floor width:

- Transport every action to an explicit reference with **M_ref = M_node + r × F**.
- Include slab actions, web bending, and the moment from web axial force acting
  below the slab midplane. Preserve the in-plane lever arms as separate terms.
- Subtract each cut-side shell's applied loads at cut nodes from its native
  resisting actions. Those nodal loads remain in that side's external-load ledger.
- Independently check opposite-face continuity and force/moment equilibrium of
  both free bodies, including beam self-weight and adjacent column actions.
- Reject incomplete inventories, nonfinite records, cuts through element
  interiors or column lines, and records lacking the native-action schema.

The force transport follows the work-conjugate form of the documented small-rotation
[OpenSees rigidLink constraint](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/mp_constraint/rigidLink.html).
Future constraints must also respect the documented restriction on chained retained
nodes in the [Transformation handler](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/constraint/TransformationMethod.html).

## Independent checks

An elastic shell-flange/web cantilever with zero Poisson ratio receives analytical
constant-curvature end tractions. Hand calculation uses the composite neutral
axis and parallel-axis inertia. On three meshes, the largest relative error in
nonzero tip displacement/rotation is 2.31e-12; the largest component error divided
by applied moment is 1.21e-12.

For its 1000 kip-in applied moment about the slab midplane, the left cut recovers
23.46 kip-in from slab bending, 269.84 from web bending, and **706.70 from web axial
force × offset**. The 70.67% fraction belongs to this benchmark, not every building
beam. It shows why web bending alone is an incomplete composite moment.

All **192 whole-floor cuts** from the four building solves pass the numerical
equilibrium checks. The largest normalized free-body residual is 1.84e-14
(normalization uses the summed magnitudes of the external actions). Mesh 8 → 10
changes in vertical-bending cut resultants, normalized by each axis's fine-mesh
peak across floors, are 0.2204%/0.8304% for X/Y under full live and 0.2180%/0.8144%
under alternate live. These are whole-floor checks, not local stress convergence.

**152 targeted tests pass.** They include the previous slab/refinement
checks plus asymmetric native cuts, unloaded lower-floor transfer, reference
translation, missing offset/pressure detection, invalid inventories, analytical
cantilever recovery and existing-domain preservation.

## Next modeling step

Use a common coupled gravity assembly to isolate the diaphragm assumption while
preserving web offsets, floor support flexibility and material properties. Any
rigid-floor comparison needs a constraint formulation without unsupported MP
chains. Compare signed column forces and complete composite actions, rather than
only base reactions or web moments.

Then define a physically justified allocation of slab membrane and bending forces
to reinforcement and beam sections, with checks against double counting shared
slab steel. The new whole-floor cuts cannot directly size individual beams or
justify equal top/bottom reinforcement. Slab steel selection, member/joint capacity
checks and experimental IMK calibration remain prerequisites for production data.

## Files and reproducibility

Work is in branch `continuation/slab-refinement-20260924`, based on desktop commit
`3fec8de60d61a7bf21efe94a241e0d7a96c67f95`. The original laptop checkout, source design and SSD handoff are preserved.
The source design's decompressed SHA256 remains
`f49291440328b9097efd44345246d11a420365a0bbd1dea8c810276890c43871`.

From the restored checkout's `RC Structure` directory, using the OpPy environment:

```powershell
python -B tools/review_floor_frame_assumptions.py --output <new-assumptions-folder>
python -B tools/review_composite_sections.py --output <new-composite-folder>
```

Both scripts refuse an existing output folder. Detailed raw evidence is in
`../floor_frame_assumptions_20260924/` and
`../composite_section_recovery_20260924/verified_run/`. The parent composite folder
retains the first four runs rejected by the input-identity guard: their explicit
stiffness defaults were missing from the recorded input dictionary. The verified
rerun spells out those defaults and checks exact input and force reproduction.

`metrics.json` retains every paired comparison, rigid-motion fit and mesh check.
`tests.log` and `test_results.json` record validation. `continuation.patch` and
`continuation_code.zip` contain cumulative source changes since the desktop base,
including the previous slab refinement work. They exclude transferred unrelated
IMK/joint edits. `code_manifest.json` records file hashes. Nothing was committed
or pushed; production and engineering-qualification flags remain false.
