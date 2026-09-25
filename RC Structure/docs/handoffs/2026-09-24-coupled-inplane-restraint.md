# In-plane restraint matched inside the coupled assembly — September 24

Continuation of the floor/frame compatibility investigation (see
`2026-09-24-laptop-floor-frame-continuation.md`). The bare-frame experiments
had shown that releasing the frame's rigid diaphragm moves column-end
gravity moments by about a third of the governing demand. This step asks
the converse inside the coupled shell/web/column model itself: with the
same webs, offsets, columns, materials, mesh and loads, what does a declared
rigid in-plane restraint do to the column actions, to the whole-floor
composite section actions and to the floor deformation?

Result in one line: the production frame's diaphragm — rigid in-plane ties
at the column joints — is the largest single identified cause of the
frame/coupled gravity gap; placing the same tie in the coupled assembly
removes about 60% of it. The rigid-membrane limit is a different assumption
and moves away from the frame. Neither result selects a floor idealization;
what they do is decompose the gap into a diaphragm term and a residual
section/joint-representation term, and the recommendation below is built
on that decomposition.

## What changed in the code

`Design/SMRF_Coupled_Analysis.analyze_coupled_gravity` gains two declared,
recorded assembly options (method
`smrf_monolithic_eccentric_shell_web_column_gravity_v3_inplane_restraint_options`):

- `inplane_restraint`: `finite_membrane` (default, unchanged model),
  `rigid_joints` (one `rigidDiaphragm` per floor over the column-joint slab
  nodes, ux/uy/rz tied to an unloaded master at the plan centroid — the
  production frame's diaphragm placed in this assembly) or `rigid_floor`
  (every slab node of the floor tied — the rigid-membrane limit).
- `constraint_handler`: `Transformation` (default), `Lagrange` or `Penalty`
  (with `penalty_alpha`). A diaphragm constrains slab nodes that retain web
  nodes, the constraint chain the Transformation handler does not support
  (its documentation), so the restrained variants require Lagrange or
  Penalty and the module refuses the combination outright. The handler is
  declared so that a finite-membrane Lagrange run can be compared with the
  Transformation baseline before any restrained run is read.

Both options are written into `inputs` (so the request identity carries
them) and into a new `assembly` record with the master nodes and the tied
node count. The forces the diaphragm exerts on the slab nodes are recovered
from node equilibrium (element resisting forces of the node and of its
rigid-linked web node, transported, minus the applied load; the link's own
force pair cancels in that sum) and exported as
`diaphragm_constraint_actions` (schema
`native_global_actions_applied_loads_and_constraint_actions_v2`). Three
checks enter the status: the diaphragm kinematic residual, the residual of
the same node balance at untied nodes (must vanish), and the sum of the
constraint actions about each master (must vanish in force and moment,
with zero out-of-plane components). Under a rigid floor the interior slab
nodes are added to the shell-to-frame interface so its global balance still
closes. Default-path column forces reproduce the saved references exactly.

`Design/SMRF_Composite_Sections.recover_floor_cut` accepts the v2 schema:
the constraint actions of the floor enter the free bodies as external
actions (count checked against the assembly record), and a cut through a
constrained node is rejected because a point action on the cut belongs to
neither side — so rigid-floor cuts are recorded as undefined, by design.
`tools/review_composite_sections.py` compares inputs to the pre-option
references only for the default assembly.

New: `tools/review_coupled_inplane_restraint.py` (the experiment),
`tools/summarize_coupled_inplane_restraint.py` (metrics table, mesh
sensitivity) and `tests/test_smrf_coupled_inplane.py` (8 tests: default
record declaration, Lagrange reproduces Transformation to 1e-7 kip and
1e-10 in, joint/floor tie counts and kinematics, constraint actions close
node equilibrium and balance about the master, rigid-joint cuts close only
with the constraint actions and fail with them zeroed, rigid-floor cuts
rejected, unsupported combinations refused before any domain is built,
Penalty slip reported). Fixture leak fixed in
`tests/test_hinge_hysteresis_fixture.py` (a failing `setUpClass` left its
2x1x2 geometry patched for every later module of the same discover run).

## Protocol

Fixed wider candidate (2x6 bays, 6 stories, 240-in bays, 168-in stories,
14x28 beams, 36x36 columns, 6-in slab; design SHA256
`f49291440328b9097efd44345246d11a420365a0bbd1dea8c810276890c43871`), the two
saved gravity cases (1.2D+1.6L, live on all panels and on the alternate-X
strip), meshes 8 and 10 per bay, four variants each, all fresh solves:

| variant | restraint | handler |
|---|---|---|
| `finite_membrane_transformation` | finite membrane | Transformation (anchor; must reproduce the saved coupled reference) |
| `finite_membrane_lagrange` | finite membrane | Lagrange (handler alone) |
| `rigid_joints_lagrange` | column-joint slab nodes tied | Lagrange |
| `rigid_floor_lagrange` | every slab node tied | Lagrange |

Every column end (126 columns, both ends, local My and Mz) is compared with
the anchor and with the saved bare-frame solution, each as a vector
distance divided by the saved governing design moment at that same end —
not by the changed moment, and not a capacity check. All 48 whole-floor
cuts per solve are recovered where defined; the joint translations of each
floor are fitted to a rigid in-plane motion. Raw solves, cuts, rows and
`metrics.json` are in `outputs/coupled_inplane_restraint_20260924/`
(untracked; copied to the research workspace).

## Numerical verification

- The anchor reproduces the saved coupled reference column forces with
  zero difference at both meshes and both cases.
- The handler alone changes nothing: finite-membrane Lagrange equals the
  Transformation anchor to 0.0000% of governing at every column end and
  every cut; rigid-link residual 1.7e-18 in. Lagrange with UmfPack takes
  the link/diaphragm chain exactly: diaphragm residual at most 1.7e-18 in,
  fitted rigid-motion residual of the tied joints at most 2e-18 in (the
  finite membrane leaves 1.42e-3 to 2.07e-3 in, floor 5 worst, as before).
- Constraint actions close node equilibrium (untied-node residual below
  1e-8 of the applied load) and sum to zero force and moment about every
  master; out-of-plane components vanish. Base and interface balances pass
  in every variant. All 96 finite-membrane and rigid-joint cuts balance with
  free-body residuals at most 2.0e-14.

## Measured effects (percent of the governing design moment at the same end)

| case | mesh | variant | max change vs anchor | mean change | max gap to frame | mean gap | ends >5% gap | max cut bending change |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| all panels | 10 | finite membrane (anchor) | — | — | 18.5499% | 3.6780% | 52/252 | — |
| all panels | 10 | rigid joints | 14.9059% | 3.2951% | 11.4585% | 1.4424% | 6/252 | 12.3705% |
| all panels | 10 | rigid floor | 32.6132% | 7.9317% | 19.3906% | 4.5323% | 82/252 | undefined |
| alternate X | 10 | finite membrane (anchor) | — | — | 16.7536% | 3.1701% | 27/252 | — |
| alternate X | 10 | rigid joints | 14.4401% | 2.9239% | 9.6849% | 1.2935% | 4/252 | 11.8694% |
| alternate X | 10 | rigid floor | 30.8094% | 6.8280% | 18.0574% | 3.8959% | 70/252 | undefined |
| all panels | 8 | rigid joints | 14.3387% | 3.0632% | 10.4176% | 1.3226% | 4/252 | 11.5417% |
| all panels | 8 | rigid floor | 30.2984% | 7.2108% | 17.5621% | 4.0816% | 74/252 | undefined |
| alternate X | 8 | rigid joints | 13.9836% | 2.7080% | 8.7791% | 1.1801% | 2/252 | 11.0062% |
| alternate X | 8 | rigid floor | 28.6750% | 6.1957% | 16.3578% | 3.4988% | 55/252 | undefined |

Mesh sensitivity of the effects themselves (mesh 8 to 10, largest change at
any column end): rigid joints 0.84% of governing, rigid floor 2.52%; the
anchor's own moments move 0.74% between the meshes and the cut bending
resultants 0.83% of the axis peak. The effects are 15 to 30 times their
mesh sensitivity.

What the numbers say:

1. **The joint diaphragm is the frame's assumption, and it is most of the
   gap.** Tying the joints in the coupled assembly moves the worst gap to
   the frame from 18.55% to 11.46% (all panels) and 16.75% to 9.68%
   (alternate X), the mean from 3.68% to 1.44% and 3.17% to 1.29%, and the
   number of ends further than 5% from the frame from 52 to 6 and 27 to 4.
   Which ends move: the roof-story exterior and corner columns, by up to
   14.9% of their governing demand.
2. **The rigid-membrane limit is not the frame.** Tying every slab node
   changes the column ends twice as much (32.6% max, 7.9% mean) and moves
   *away* from the frame (19.39% max gap, 82 ends over 5%, story-5 corner
   columns at 642/655 kip-in against the frame's 542/544 and the
   finite-membrane 526/532). It also reduces the roof floor's largest
   downward displacement by 4.9% (0.16526 to 0.15718 in): a rigid membrane
   is an infinitely stiff flange. It must not be used as a proxy for the
   frame or for the slab.
3. **The mechanism is composite, not just kinematic.** At the interior x
   cuts of the lower floors the joint ties change the whole-floor in-plane
   axial resultant from +23.4 to -117.5 kip (floor 1) and +5.0 to -157.5
   kip (floor 3): the ties carry about 140 kip of in-plane force per cut
   that the finite-membrane assembly carries as web axial force at its
   eccentricity. The web-eccentricity component of the cut moment drops
   from -4051.6 to -3519.9 kip-in at floor 1 while the slab-bending and
   web-bending components stay within 1% (-1682 to -1681, -1621 to -1613),
   and the whole-floor vertical bending resultant falls 7.4% at floor 1
   (-7354.5 to -6813.9), 8.9% at floor 3 and 1.5% at the roof. Ties at the
   joints therefore change the composite floor section's own force
   distribution, not only the columns.
4. **The residual after matching the diaphragm is a
   representation term.** With the joints tied, the worst remaining end is
   the roof-story interior-edge column at grid (1,5)/(1,1), lower end: local
   Mz 238.2 kip-in in the coupled assembly against 319.0 in the frame (188.2
   finite membrane), 11.46% of the 705 kip-in governing demand; the frame is
   the conservative side there. The story-1 column at the same grid, upper
   end, reverses sign: -11.2 against +26.3 (frame) and +25.8 (finite),
   6.36% of 590 kip-in. Story 5 at the same grid: 179.1 against 250.8 and
   212.7, 5.38% of 1332. These sit on the interior-edge line of the 2-bay
   direction, where the frame's centerline T-section beam and the coupled
   eccentric web plus slab plate disagree most on the rotational restraint
   they give the column.

## Recommendation for gravity/design consistency

The evidence supports the following, in order; none of it is implemented
beyond item 1's building blocks, and none of it is an engineering
qualification.

1. **Decompose the compatibility evidence in the design loop.** The
   production compatibility check (`SMRF_Coupled_Comparison`) runs the
   coupled reference once, finite membrane, and compares it with the frame.
   Run it twice — finite membrane (the physical reference) and
   `rigid_joints` under Lagrange (the frame's own assumption) — and record
   three numbers per column end: frame against rigid-joints (the
   section/joint representation term), rigid-joints against finite
   membrane (the diaphragm term) and frame against finite membrane (the
   total). The judgement item `floor.compatibility_idealization_reviewed`
   then argues from decomposed terms instead of one number. Cost on the
   fixed candidate: one extra 16-second solve at mesh 10; no change to the
   production stiffness, loads or transfer; the options and the constraint
   export already exist and are tested.

2. **Make the frame's gravity restraint match the physics, not the coupled
   model match the frame.** Under gravity there is no in-plane load and the
   slab is a finite elastic membrane; the rigid diaphragm is a lateral-load
   idealization (ASCE 7-22 12.3.1.2 permits it for concrete slabs of low
   span-to-depth ratio without horizontal irregularity), and ACI 318-19
   6.6.3.1.1 asks the frame analysis to represent the stiffness actually
   present. The consistent direction is therefore to release the joint
   diaphragm for the gravity state — the D/L strength combinations and the
   gravity part that precedes the seismic combinations — and keep it for
   the lateral-load runs, superposing in the linear design frame. On this
   candidate the bare-frame experiment already measured what that does: the
   worst gap to the finite-membrane reference falls from 18.5% to 14.8%
   (all panels) and 16.8% to 12.5% (alternate X), with column-end changes of
   up to a third of the governing demand at some ends. That is a partial
   closure, not a fix, and it re-opens every column P-M check; it is a
   methodology change that needs the engineer's decision, not a code edit
   made here. Keeping the diaphragm everywhere is not defensible as
   "conservative": the frame is conservative at the roof interior-edge
   columns and unconservative elsewhere (the story-1 sign reversal, the
   corner ends the earlier review listed).

3. **Close the representation term with a matched beam-representation
   comparison in the same coupled assembly**, the same way this step
   closed the diaphragm term: (a) web offset set to zero (slab and web
   bend about their own centroids at a shared curvature, non-composite)
   and (b) the frame-equivalent — web inertia replaced by the 6.3.2
   T-section inertia less the slab plate's own share, at zero offset — each
   under rigid joints and compared with the rigid-joints solve above and
   with the frame. Whichever reproduces the frame identifies what the
   frame's centerline T-section is missing; the residual after that is the
   joint-region idealization. The Transformation/Lagrange equivalence
   established here means the comparisons can stay exact.

4. **Recover member design forces from the whole-floor cuts by a disjoint
   interval partition.** The recovery is linear in the element inventory,
   so a cut can be partitioned by transverse interval: each beam's
   interval is its ACI 318-19 6.3.2 effective flange plus its web (the
   composite beam section), the remaining intervals are slab strips, the
   intervals are disjoint and cover the width, and the member sections sum
   exactly to the whole-floor cut — demand is never counted twice. The
   flange interval is the same one whose slab mats `SMRF_Beam_Slab_Strength`
   credits to the beam, so the demand partition and the strength partition
   coincide; slab-strip reinforcement is then designed for its own
   intervals' membrane-plus-bending resultants (the extension the
   methodology already names as missing). The partition is an allocation
   rule, not a property of the stress field — the plate moment does not
   stop at the flange line — so its strip results must be checked against
   the Wood-Armer strip envelopes now in use before either replaces the
   other. Not implemented; `recover_planar_cut` already accepts an
   arbitrary shell inventory, so the partition is a bounded addition.

## Residual discrepancies stated

- After matching the diaphragm, 6 of 252 column ends (all panels) and 4 of
  252 (alternate X) remain more than 5% of governing from the frame, the
  worst 11.46% and 9.68%; their locations are listed above.
- The rigid-floor cuts are undefined by construction (constraint forces at
  the cut nodes), so the composite-action comparison covers the finite and
  rigid-joint assemblies only.
- The rigid-joints change is 0.84% of governing mesh-sensitive between
  meshes 8 and 10; the rigid-floor change 2.52%. Neither mesh is a
  converged local stress field; both are whole-member and whole-floor
  resultants.
- The Penalty handler is available and tested for reporting its slip, not
  used for any number above.

## Failed or superseded trials

None failed. The Transformation handler with a diaphragm was not attempted
as a run: the module refuses it before building a domain, and the test
pins that refusal. Codex's earlier bare-frame variants
(`outputs/floor_frame_assumptions_20260924/`) remain the measurement of the
released frame and are not repeated here.

## What remains before dataset generation

Unchanged from the previous handoff, plus: the decomposed compatibility
evidence (item 1) and the engineer's decision on the gravity restraint
(item 2) precede any change to the demand basis; the representation term
(item 3) and the member-force partition (item 4) precede slab steel
selection and the beam/slab and joint capacity checks; experimental IMK
calibration is untouched. Note also that the continuation patch's slab
refinement gate now stops every PROBE-assertion design that has no
`FloorAnalysisConfig.slab_refinement` plan (`Verify_Designs.probe_config`,
`Evidence_Summary`, the hinge-hysteresis fixture): the refinement plan
needs a per-geometry mesh generator or a declared default before the
150-case verification run can be repeated on this branch.

## Reproduction

From this checkout's `RC Structure` directory, OpPy environment
(`C:/Users/andro/anaconda3/envs/OpPy/python.exe`, OpenSees 3.8.0):

```powershell
python -X utf8 -B tools/review_coupled_inplane_restraint.py --output <new-folder>
python -X utf8 -B tools/summarize_coupled_inplane_restraint.py <that-folder>
python -X utf8 -B -m unittest tests.test_smrf_coupled_inplane tests.test_smrf_coupled_analysis tests.test_smrf_composite_sections
```

The review refuses an existing output folder and the fixed design's SHA256
is checked before and after. The full run took 4.2 minutes; the full test
suite (545 tests) passes except the hinge-hysteresis fixture noted above.
Production and engineering-qualification flags remain false.
