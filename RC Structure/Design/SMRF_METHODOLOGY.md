# SMRF methodology overhaul — implementation status

Status: **partial implementation; not approved for dataset generation**.
Last reviewed: 2026-09-13.

### Monolithic coupled-gravity diagnostic

`SMRF_Coupled_Analysis.analyze_coupled_gravity` now solves all floors, shell
slabs, eccentric downstand beam webs and full-height columns in one elastic
gravity model. Each web excludes the slab thickness and lies Hbeam/2 below
the slab midsurface; a six-DOF rigid link enforces compatibility, without
chaining constrained/retained nodes. Columns join the floor intersections to
a fixed base: elevated joints can settle and rotate. No arbitrary column-load
redistribution or independently prescribed slab support reactions are used.

The slab-to-frame interface is extracted as six-component nodal actions at
the **solved coupled displacement**. Both its force/moment balance and the
whole building's base balance are checked. Slab/SDL/live pressures, clear-span
beam-drop self-weight and slab-excluded column self-weight have separate
ledgers. That interface cannot be reused as fixed loads on a different bare
frame without retaining the stiffness coupling.

This remains a separate diagnostic, not the production design/NTHA model.
It uses elastic membrane stiffness rather than a rigid diaphragm; gross
slabs and explicit optional web/column bending modifiers; linear column
transforms; and the inherited Iy+Iz torsion proxy (not verified Saint-Venant
torsion). Centerline joint regions are not solid models. Web moments alone
are not composite beam strengths. Slab membrane forces are now nonzero, so
the existing zero-membrane reinforcement routine must **not** consume these
results without a membrane-plus-bending design/recovery method.

`Review_Coupled_Gravity.py` makes a reproducible review from a saved design,
requires a new output directory, records the source SHA256, and runs no NTHA.
The 8-story, 3-by-3 review from `smrf_evidence_review_20260913/design.json`
completed uniform 1.2D+1.6L and roof-corner live cases at 4 and 8 subdivisions
per bay. Results are in `outputs/smrf_coupled_gravity_review_20260913`.
Force/moment errors were below 4e-13. Roof maximum downward displacement
changed from 0.05985448 to 0.06149691 in under full gravity, and 0.00334049
to 0.00348410 in under roof-corner live load. Those roughly 3–4% changes are
not a convergence certificate: peak local membrane forces increased from
0.52165 to 1.08374 kip/in and 0.03499 to 0.08078 kip/in, respectively.
Investigate spatial recovery/idealized connection concentrations before
using those peaks for reinforcement. Historical source designs were not edited.

The 308-test suite includes independent symmetric-column PL/EA comparisons,
roof load transmitted through an unloaded floor, actual load/weight accounting,
asymmetric force/moment balance at meshes 2/4/6, rigid-offset compatibility,
nonzero membrane action, support-stiffness sensitivity, protected domains,
failed-solve cleanup, source preservation and no-clobber review outputs.

Next: validate composite section and slab membrane/bending recovery, resolve
torsion/mesh/local connection modeling, then integrate the verified demand
path into the design loop with its seismic/rigid-diaphragm assumptions.
The coupled model is now also run automatically against every selected
frame as the load-path check described under "Transfer resolution" below.
`GENERATION_RELEASE_READY` remains False.

Implementation references: [OpenSees rigid links](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/mp_constraint/rigidLink.html)
and [Transformation constraints](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/constraint/TransformationMethod.html).

### Span recovery and interface audit follow-up

The end-only beam-demand gap described below is now addressed for the current
static project beam model. `SMRF_Beam_Actions` reads the actual OpenSees
element loads by pattern and their current factors, including frozen gravity.
It reconstructs the piecewise-quadratic vertical bending diagram, checks
end-force/load equilibrium, and evaluates its extrema at ends, point forces
and zero-shear locations. Both legacy and Phase 1 beam strength checks now
use its full-centerline envelope. The design record stores each beam/case's
loads and recovery; qualification recomputes that bounded software check.
Old end-only records remain unqualified rather than gaining a fabricated
interior envelope. Optional face intervals are supported by the pure routine,
but production strength checks still use the entire centerline span; this
does not establish joint-face capacity design or beam biaxial/axial adequacy.

The floor diagnostic now also saves `raw_solved_interface`: unmodified
vertical nodal forces **and both global interface couples**, before the
footprint redistribution. Its corner-loaded 3-by-2 example balances force
and moments to approximately 1e-13 when the couples are included. The
unmodified vertical-load first moments alone are 2001.857426 and
1682.869912 kip-in; global couples My=-1.857426 and Mx=16.203245 kip-in
restore the applied 2000 and 1666.666667 kip-in resultants. The subsequent
redistribution adds a separate change in the vertical-load first moments.
This separates two sources of the transfer error from a solver failure.

That raw interface is **diagnostic only**, not a newly approved transfer:
`verified=False`, `applied_to_frame=False`. It is not installed into
`Gravity_Loads`, does not change the beam/column compatibility model, and
does not clear the existing asymmetric-transfer failure. Next work remains
the compatible slab/beam/frame interface, support-face shear recovery and
final capacity/constructability verification. Generation remains disabled.

Regression coverage includes fixed-fixed and simply supported analytical
beams, an off-centre point force, combined forces, uplift, both framing axes,
multiple time-series factors and `loadConst` time reset, rejected stale or
unsupported loads, saved-envelope recomputation, and raw interface balance
for three asymmetric patterns at meshes 4/8/12 per bay. Full suite: 299 tests.

### Follow-up review: exported loads and verification safeguards

The latest code review found a concrete counterexample to the transfer's
previous force-only success flag. A 3-by-2 floor with 240-by-200-in bays,
5-in slab, 12-by-18-in beams and 18-in-square columns was loaded at 0.05 ksf
on panel (0,0) only, at eight subdivisions per bay. The shell reactions
balance. The exported vertical beam/column loads also total 16.666667 kip,
but their X first moment is 1959.880010 rather than 2000 kip-in, and their
Y first moment is 1649.816506 rather than 1666.666667 kip-in. The shear-only
extraction plus column-footprint redistribution is therefore not an exact
mechanical transfer under this pattern. Prior mesh-stability and symmetric
force-balance tests did not detect it.

The v2 transfer now checks actual exported force **and both first moments**,
and rejects that case. Cached transfer validation recomputes these quantities
from the load inventory, checks every beam/column exactly once, and verifies
all patterned cases against their declared panel pressures. Old v1 transfer
records are not silently upgraded. The member-action fingerprint now includes
the transfer, gravity model and demand policy; canonical load-case checking
also verifies the live pattern. Qualification repeats the transfer audit.

**Consequence at the time:** multibay design-only searches stopped on
asymmetric patterns that the old force-only test allowed. Patterns were not
disabled, the tolerance was not loosened and no loads were rescaled.

### Transfer resolution (2026-09-13, later)

The two sources of the first-moment loss were removed rather than masked.
The footprint redistribution is gone: the column node keeps whatever the
point-support floor model delivers to it, recorded as
`column_direct_fraction` (a mesh artifact that falls with refinement: 16% at
8/bay, 6% at 12, 3% at 16 on 20x17-ft bays; the finite-footprint share is
recorded beside it for information). And the solved interface couples are
exported with the forces: at every beam-line node the torsion and bending
couples the slab applies (`node_couples`), and at every column node the
couples from the beam ends. With them the exported inventory balances force
and both global first moments to solver precision for every pattern at every
mesh (transfer schema v3, `smrf_floor_transfer_v3_force_and_couple_export`);
the reviewer's corner-loaded 3-by-2 counterexample now balances to ~1e-13.
`validate_floor_transfer` recomputes force and the couple-inclusive moment
balance from the saved inventory, requires a couple row at every load
position, and rejects an altered, relocated or missing couple.

How `Gravity_Loads` applies the interior couples to the single-element
beams matters mechanically, and the first version got it wrong: it split
each couple to the two end joints in proportion to position, which keeps
the global balance exact but gives no beam response at all on a fixed-ended
element (a 120 kip-in couple at midspan of a fixed 240-in beam produces a
0.75-kip reaction and 30 kip-in end moments; the split produces nothing --
the cross-check's counterexample, reproduced in
`tests/test_smrf_transfer_mechanics.py`). Now the vertical-plane bending
couple is carried as a force pair inside the element (+M/2a at x - a,
-M/2a at x + a, a = 1 in, `_bending_couple_pair`), whose fixed-end actions
and internal moment jump the element carries exactly through the same
`-beamPoint` machinery as the forces; the torsion couple keeps the linear
split, which is its consistent nodal load (linear twist). The column-node
couples act at the column joints, which are real nodes. On the real design
frame this reproduces a reference frame whose beams are subdivided at every
transfer node and loaded there with the forces and both couples as nodal
loads: reactions to 1e-4, joint displacements and rotations to 1e-3, beam
end forces to 2e-3 and the bending diagram just outside each 2-in pair
window to 3e-3 of the end-moment scale, for full live, an asymmetric
pattern and dead only. The pair is a tested approximation to the point
couple, not the point couple: inside the 2-in window the element carries a
shear of M/2a that is an artifact of the representation. Nothing reads it
as a demand -- the strength checks use end forces, the capacity design uses
the joint-face free body, the span envelope uses moments -- and halving or
doubling the arm changes reactions by less than 1e-5 and joint rotations and
beam end moments by less than 1e-4 of their scale (tested). The same bending
couples enter the capacity-design face reactions as -M/ln, +M/ln on the
clear span.

The compatibility question is an evaluated comparison plus a judgement,
not a non-overridable stop: `Design/SMRF_Coupled_Comparison` runs the
monolithic shell/web/column model of the selected frame (1.2D + 1.6L with
member weight, both models at gross stiffness) and compares it with the bare
frame loaded by the transfer, for all panels and for the first saved ACI
6.4.2 pattern. The evaluated quantity is the vertical load path:
`floor.coupled_frame_compatibility` / `floor.coupled_total_load` and their
`_pattern` counterparts pass when every column's base vertical reaction is
within 5% of the coupled model's and the totals agree to 1e-6 -- each case
enforced on its own -- all recomputed at qualification from the saved
per-column rows after those rows are checked for a complete, unduplicated
column inventory with finite values (`floor.coupled_summary_consistent`
fails if the saved summary no longer matches its rows), and only from a
comparison carrying this frame's input signature. Recorded beside them, for the judgement and
not as acceptance: the base-moment ratio frame/coupled per column group,
and the coupled reference's own mesh sensitivity (verticals and base
moments at the next coarser mesh). After the couple correction above, the
4x4x7 centre case reads corner/edge base-moment ratios 0.82/0.84 (they were
0.69/0.69 with the split), the 2x2x4 cases 0.96-1.02, with the coupled
moments themselves moving 1-8% between meshes 6 and 8. Not compared, and
stated in the record: beam end moments (the coupled model shares them
between web and slab), floor displacements, the cracked-stiffness state,
joint flexibility and membrane action. Those, and whether the recorded
moment ratios are acceptable for the frame's purpose, are the judgement
item `floor.compatibility_idealization_reviewed`, cleared by
`IndependentVerification.floor_frame_compatibility_reviewed` after review;
the vertical criterion is the only one the code asserts is justified. So
that the judgement is made against the demand it affects, the record also
carries `moment_significance`: per column kind, the coupled model's gravity
base moment, the bare frame's, and the largest factored base moment over
every strength combination the frame was designed for. On the 4x4x7 centre
case the coupled gravity base moment is 3-4% of the governing design
moment and the 17% gap between the models is 0.5-0.6% of it; on the 2x2x4
cases the gap is 0.1-0.2% of the governing moment. The compatibility
question is not closed by these numbers; they are what closing it will be
argued from.

The two demand gaps named above are closed in the same way: beam interior
gravity extrema are recovered from the actual element loads
(`SMRF_Beam_Actions`, `beam.interior_flexure_envelope`), and slab support-face
shear is recovered at the beam face (see "Slab strip actions",
`floor.support_face_shear_recovery`).

Slab and independent-verification assertions now require literal Boolean
flags, a nonblank named author/basis, and a valid YYYY-MM-DD date. These are
provenance checks, not authentication or engineering verification. No review
assertions or generation-release setting were enabled by this follow-up.

See [SMRF_REVIEW_20260913.md](SMRF_REVIEW_20260913.md) for the bounded review,
regression coverage and next implementation order. The descriptions below
record the work already implemented; they do not override these open findings.

The agreed structural system has moment frames on every grid line in both
directions. These frames carry gravity and seismic loads. There is no separate
gravity-only framing system. Fixed supports and rigid floor diaphragms remain
analysis assumptions; foundation, diaphragm and collector design is outside
this initial scope. This work does not establish complete building-code compliance.

## Approved material and detailing assumptions

New research candidates use sheltered interior exposure, normalweight concrete,
ASTM A706 Grade 60 reinforcement, and 3/4-inch maximum aggregate. Beam and column
clear cover is **1.5 inches to the outside of the hoops**, not to longitudinal
bar centers. The longitudinal offset is calculated independently for each member
and candidate bar size: clear cover + hoop diameter + half the longitudinal bar
diameter. Candidate bar-fit checks also include the aggregate-dependent spacing.

The supported sheltered slab bar family (#4, #5, #6) uses a separate minimum
clear cover of **0.75 inch**, under ACI 318-19 Table 20.5.1.3.1. This is not a
waiver of fire-resistance, durability, or construction-tolerance requirements.
It must not be applied to exterior/wet/exposed slabs without revisiting scope.
The slab strength routine rejects unsupported exposure, steel grade or concrete
type rather than silently adopting these assumptions for unrelated input data.

## Implemented and connected

- A separate elastic cracked-stiffness design model with the same physical
  connectivity as the nonlinear model. No yielding springs are used for the
  preliminary strength/drift demand calculations. Response-history member
  formulations are unchanged; the new slab-aware load/mass state is separate
  from the explicitly retained legacy load mode.
- An explicit D/L/seismic strength-combination subset: gravity, positive and
  negative X/Y, 100/30 orthogonal effects, vertical seismic dead-load effects,
  and conservative rho=1.3. This is not the complete applicable load inventory.
- All member/combination results reach the reinforcement search. Final demand
  results cannot refer to the state before the last reinforcement update.
  The elastic design frame (gross properties with constant stiffness
  modifiers) does not depend on the bars, so each combination is solved once
  per section and the reinforcement iterations re-run only the ACI checks on
  the captured actions (`run_checks_phase1(..., member_actions=...)`); a
  test pins the captured-action checks to the live-domain checks.
- Separate rho=1 QEx/QEy drift runs, floor-node drift envelopes, Cd/Ie
  amplification, and a story P-Delta screen. The current conservative screen
  uses Risk II solely-moment-frame D/E/F drift limits.
- The section search closes on what qualification evaluates (see "Section
  search closure" below): drift steps the beam to the next depth, the beam
  capacity-shear section steps it to the wider variant of its depth, a
  joint that fails 18.7.3.2 after the steel pass steps the column size,
  joint shear jumps the column to the rung whose joint area covers the
  shortfall, and no member steps down while any requirement is unmet.
- Beam clear-span checking in both framing directions using column dimensions.
  Empty section ladders now reject the candidate rather than invent a fallback.
- The beam-width code limit is correctly distinguished from a stricter
  practical sizing preference. Column longitudinal steel is limited to 1–6%
  (18.7.4.1); the strong-column escalation selects at most
  `rho_col_practical_max` (4%, a stated proportioning preference) before the
  section grows, and offers only cages whose face bars the hoop ladder can
  support (corners and alternate bars, 25.7.2.3: at most 2L - 2 bars per
  face with L legs).
- Single-layer bar-fit, aggregate-dependent clear-spacing and beam
  reinforcement-balance candidate filters. The hoop/crosstie arrangement is
  generated from the bar positions (`SMRF_Cage_Layout`, see "Capacity
  design"); hook geometry and placement drawings are not.
- One automatically selected slab thickness for the entire building, including
  the roof. Every bay panel is checked against the declared beam-supported
  two-way thickness rules; the critical panel governs. Selection is repeated
  before analysis whenever beam dimensions change, and saved/restored with
  the winning frame candidate. See the slab section below for the limits.
- Separate computed slab weight and superimposed dead load. Beam-drop and
  column weight accounting excludes slab-height overlap; beam weight uses
  clear spans. Gravity, nodal mass, ELF weight, stability weight, tributary IMK
  axial estimates and exported metadata share this load inventory. The existing
  Y-beam length error for unequal bays is also corrected.
- Pure detailing checks for geometry, steel ratios, bar spacing, continuity,
  end-zone lengths and Grade-60 hoop spacing, when their inputs are supplied.
- Conservative scalar hoop spacing and end-zone selection before every steel-
  iteration solve, including after longitudinal bar sizes change. Beam limits
  use effective depth and bar diameter; columns use a conservative 4-inch cap
  in place of an unverified `hx`-dependent allowance. The chosen spacing is
  applied uniformly along members. Zone geometry is recorded; the
  hoop/crosstie arrangement is generated (below), but this is not a zoned
  confinement-material model.
- Member-specific cover and actual perimeter-bar positions in the new-mode
  section-strength calculations, candidate selection, fiber sections and IMK
  geometric inputs. Intermediate column side bars are no longer all treated
  as a single centroid layer in the new mode. Legacy mode retains its previous
  cover convention and side-steel idealization. This migration is not a
  completed confinement design or an independently validated nonlinear model.
- A physical elevated-joint inventory matching the elastic member connectivity,
  including one actual roof column. Saved signed local element actions retain
  simultaneous combinations, compression/tension signs and end references.
  The adapter checks the expected combination inventory before forming axial
  envelopes and nominal uniaxial column capacities for both planes/signs.
  Missing joint-face references, load cases, section equilibrium or developed
  slab strength do not become passing SCWB evidence. The legacy top-corner
  estimate remains a sizing proxy, not the governing joint qualification.
- Pure joint helpers for nominal SCWB sums, probable-moment beam shear
  equilibrium, through-bar joint depth and explicit joint-shear evidence.
  Capacity-design shear, slab-enhanced beam strengths and terminating-bar
  anchorage are not yet connected as completed design checks.
- An independent shell-floor diagnostic with continuous shared panel nodes,
  explicit gravity/load-pattern cases, equilibrium reporting, displacement
  and signed Gauss-point resultants. Its supporting beam lines are vertically
  rigid with free rotations; it is not a validated elastic floor/frame system.
  Diagnostic reactions and raw shell moments are **not** fed into frame design
  or relabeled as verified slab reinforcement demands.
- A flexible-beam slab-to-frame gravity transfer (`SMRF_Floor_Transfer`,
  `SMRF_Floor_Analysis` with `support_model="flexible_beams"`): the same shell
  mesh carrying elastic ACI 318-19 8.4.1.8 gross T/L-beams on every centerline,
  held only at column intersections. Unit dead and unit live (all panels)
  pressures are solved once per candidate slab/section; the discrete node
  loads each beam receives and the column footprint share are saved in
  `design.json['floor_transfer']` and applied to every floor of the elastic
  design model and the IMK response model in place of tributary nodal loads
  (`gravity_load_model: slab_transfer`). Floor pressure therefore reaches the
  frame once, through this path; beam drop and column self-weight remain
  separate element loads. Beams now carry gravity end moments in both models.
- A separate reinforcement/strength routine for independently verified slab
  action envelopes: one common bar size, four uniform x/y top/bottom spacings,
  actual crossing-layer depths, minimum steel, spacing/cover/clearance,
  strain-compatible flexure, tension control and ACI 318-19 size- and
  reinforcement-dependent one-way shear. No reinforcement is selected for
  current candidates because their qualified out-of-plane slab demand evidence
  is still missing. See the slab-strength and diagnostic sections below.
- Per-check `pass`, `fail`, and `not_evaluated` evidence. Empty or duplicated
  checklists cannot pass. Preferred DCR utilization is separate from code checks.
- Design artifacts save longitudinal steel, transverse bar sizes/legs/spacings,
  explicit member clear covers and resulting centroid offsets, signed action
  inventories, joint evidence, floor diagnostics, scope and a portable
  input/source SHA256 identity.
  Changed methodology/inputs reject an existing cache instead of overwriting it.
  A case-local reservation prevents two compliant workers designing one case.
- Solved member actions are bound to the selected geometry, sections, bars,
  materials and loads by a separate fingerprint. Saved combination factors
  and families are checked against the declared rule. Saved bar areas,
  diameters and cover offsets must match their bar numbers; slab strength
  evidence must match the current frame's slab. Detailing cannot replace the
  authoritative frame dimensions or reinforcement values.

## Important open work — do not label these checks complete

1. **Qualified slab action generation.** The flexible-beam transfer now
   feeds slab gravity to the frame (see "Slab-to-frame gravity transfer");
   still needed are governing live-load pattern/combination envelopes,
   spatial design demands, twisting-to-reinforcement conversion, support-face
   shear, and independent verification of the floor idealization. The
   rigid-beam-line floor model remains a diagnostic only.
2. **Slab reinforcement and beam participation.** Implemented, not verified
   (see "Slab strip actions and reinforcement" and "Beam-plus-slab
   strengths"): strip actions are computed from the flexible-beam floor
   model; whether they are applicable, enveloped and verified is asserted
   by the engineer in `Design.Config.SlabActionAssertions` after review,
   never by the code. Without those assertions no reinforcement is selected,
   the slab contribution to beam strength is unknown (not zero), and the
   joint SCWB checks stay `not_evaluated`. With them, a four-layer mat is
   selected with development, continuity, crack-control, deflection,
   integrity and corner checks, the two-way shear path is settled only if
   also assessed, and developed beam-plus-slab strengths feed SCWB, the joint
   evidence and the beam hinges. Open: the review itself (item 7), fire
   resistance, and the applicability of 8.6.1.2 to a beam-supported slab.
3. **Capacity design at every joint.** Implemented, not verified (see
   "Capacity design"): the SCWB screen sizes columns for the governing roof
   joint; probable beam/column shear, joint shear with the 318-19 confinement
   categories and joint transverse steel, hooked terminating-bar anchorage
   and splice feasibility are computed and fed to the joint checks. Open:
   independent verification of joint-face action transport, beam axial
   effects, the sway combinations and the column-shear distribution
   assumption (item 7).
4. **Transverse reinforcement design.** Implemented, not verified: hoop bar,
   legs and spacing from probable shear, 18.7.5.4 confinement area and the
   18.6.4.4 beam bounds, with the leg count bounded and realized by the
   generated hoop/crosstie arrangement (`SMRF_Cage_Layout`: corners and
   alternate bars supported, 6-in clear rule, hx from the arrangement,
   crossties only where a bar can be engaged); column spacing keeps the
   conservative 4-in cap in place of the hx-dependent `so` (`so` is
   recorded). The same hoops set rho_sh in the IMK backbones. Open:
   135-degree hook geometry and crosstie end alternation (stated, not
   drawn), middle-zone versus end-zone spacing (one spacing is used along
   the member), and verification (item 7).
5. **Anchorage and constructability.** Hooked anchorage at exterior joints,
   through-bar depth and splice type/location are computed; congestion,
   mechanical-splice staggering and bar-placement drawings remain open
   (`detailing.congestion_and_placement`).
6. **Demand verification.** Implemented against a declared policy (see
   "Demand basis"): SDC is derived from SDS/SD1/S1 (11.6), ELF eligibility
   from Table 12.6-1 with regularity by construction and the torsional
   check, accidental torsion is applied in every seismic combination with
   Ax from the drift runs, live-load arrangements per ACI 6.4.2 enter the
   1.2D+1.6L family through the floor transfer, and the seismic-weight and
   drift bases are evaluated against the declared occupancy/site
   assumptions. Site class, risk category, occupancy, partition allowance,
   roof/snow/wind/rain scope are declarations in
   `Design.Config.DemandPolicy` and stay `not_evaluated` until declared with
   an author and basis. Column P-M under biaxial bending (2026-09-14): each
   axis is checked against its own phi P-M surface -- bending about y with
   the top and bottom bars in the extreme layers, bending about z with the
   side faces extreme (`ACI_Checks.build_pm_diagrams`) -- and the two are
   combined with the Bresler load contour, (Muy/phiMny)^a + (Muz/phiMnz)^a
   <= 1, a = `DCRTargets.biaxial_contour_exponent` (1.5; 1.0 is the linear
   contour, conservative for every section). The earlier resultant moment
   against the y surface alone read a demand about z against the strength
   about y. The steel search sizes the cage against the equivalent uniaxial
   moment (DCR x phiMny) so it targets the biaxial ratio; the true
   components and both capacities are recorded per member. Column DCRs rose
   5-10% on the sweep cases; sections did not change (columns are
   capacity-protected and sit at DCR 0.4-0.6). The axial strength cap
   (22.4.2.1, phi Pn,max = 0.65 x 0.80 P0 for tied columns) is enforced
   whatever the moment: every P-M sweep (design phi surfaces, the steel
   picker's, the nominal surface the hinges and SCWB read) is cut at the
   cap, and the check reports max(contour ratio, Pu / phi Pn,max). The
   third cross-check found the cap enforced only at zero moment, so a
   column at 1.05 phi Pn,max passed with 1 kip-in of bending; that case is
   now a test. The fourth found the cut surface carrying two points at the
   cap (the sweep's seeded pure-compression point and the crossing), so the
   nominal lookup the hinges and the preliminary SCWB sizing use read
   almost zero just below the cap and the cap moment above it. The cut
   surface is now the upper envelope M(P) with one point at the cap;
   `column_moment_at_axial` interpolates on it and returns 0 outside the
   domain, and a hinge refuses a column whose gravity estimate is outside
   the surface rather than calibrate to nothing. Still open: independent verification of the contour
   exponent, action transformations and force signs (item 7,
   `strength_model_verified`). The
   `sdc_c` preset was re-valued to SDS 0.40 / SD1 0.19 / S1 0.19 on
   2026-09-13: its former 0.50 / 0.25 pair sat on the Table 11.6-1/11.6-2
   thresholds and derived as SDC D. The legacy mode retains its old D+100%L
   seismic weight.
7. **Independent validation.** Hand-check representative frames/joints and
   confirm design-to-nonlinear-model consistency, bar coordinates, confinement
   zones and export/cache metadata before a new calibration pilot.
   Unit tests confirm software behavior, not an engineering certification.
   The items this settles are asserted in
   `Design.Config.IndependentVerification` (author, date, basis required);
   the qualification records the assertion and never makes it.

## Section search closure

A design-only sweep of the plan space on 2026-09-13
(`outputs/smrf_sweep_20260913/sweep_summary.md`; 2-6 bays, 4-9 stories,
10-15 ft bays, 10-14 ft stories, all five hazard presets) found that 5 of 12
corner cases stopped on frames qualification then failed, all for one
reason: the section loop closed on the DCR band, the SCWB sizing screen and
a column-only drift step, but not on the per-joint results qualification
evaluates. Specifically: the screen's top-corner service axial read 1.2027
where the joint envelope read 1.1946 (100 joints failed); at the top of the
column ladder the joint rule needed steel the strength pick never selects;
drift grew only the column while the strength rule shrank it again; and the
beam capacity-shear section limit on 90-in clear spans advanced one f'c rung
per iteration until the cap.

The loop now closes on every evaluated requirement (`_plan_next_rungs`,
pure and tested):

- strength sizes both members toward their targets as before;
- drift (12.8.6, Table 12.12-1 with rho) takes the next beam *depth*; the
  beam capacity-shear section (22.5.1.2 with 18.6.5.1 Ve) takes the *wider*
  variant of the current depth first, because on short clear spans
  18.6.2.1(a) (ln >= 4d) caps the depth and bw d is the lever -- the beam
  ladder now offers each depth at its paired width and one 4-in wider
  variant, ordered by the capacity proxy b h^2 sqrt(f'c) so a jump lands on
  the lightest rung that should carry the demand; both fall back to the
  column only at the top of the beam ladder. When the clear span forbids a
  target rung the search lands on the next feasible rung above it, never a
  lighter one;
- beam bars are offered only where 18.8.2.3 lets them pass through the
  joints (20 db within the column dimension parallel to the bars, in each
  direction with an interior joint), and the through-bar depth is a
  capacity-design check the loop sees, so a column that no admissible bar
  fits grows;
- a joint that still fails 18.7.3.2 after the steel pass (cage exhausted at
  the practical ratio) takes the next column size;
- joint shear (18.8.4.1) jumps the column to the first rung whose
  b h sqrt(f'c) covers the worst Vj / phi Vn, the next evaluation deciding
  (gamma can fall when a wider column loses a confined face); column shear,
  hoop feasibility and anchorage take the next column size;
- while any requirement is unmet, neither member steps down, so one failure
  cannot be traded for another and the search cannot cycle;
- `max_section_iter` is 10 (was 6) and the search still stops at the first
  rung pair that meets every evaluated requirement.

The record states what the beam rung answers to: `dcr.governed_by` is
`demand` (in the band), `minimum_section`, `drift`, `capacity_design`,
`scwb` or `search_limit`, from the step reasons recorded per iteration
(`history[*].step_reasons`, `next_rungs`). A drift- or shear-governed frame
legitimately carries a beam DCR below the band; the band is a preference,
not a code requirement, and qualification does not read it.

One more closure came out of the rerun: the joint rule is priced on bar
positions, which sit inside the hoops the capacity design selects for the
cage (cover + hoop diameter + db/2). The steel pass now installs those
hoops before the in-loop joint check, so it prices the section
qualification will see (a #4 -> #5 hoop is ~0.4% of column Mn: 1.204 in
the loop had become 1.1995 in qualification on the 4x4x7 case).

On the final code all twelve corner cases reach `accepted` under probe
assertions (`outputs/smrf_sweep_20260913_final/sweep_summary_final.md`):
for example 2x2x4 at 15-ft bays and 14-ft stories, sdc_e_near (30x30 /
14x26, drift 0.79 of allowable, joint SCWB 1.212); the 2x2x9 tower (36x36
/ 18x28); 2x6x9 at 10-ft bays (28x28 fc8 / 16x22 -- the wide variant
carried the capacity shear the 12-in web could not); 6x6x9 at 10-ft bays
and 10-ft stories, sdc_e (24x24 fc8 / 10x18 with #8 beam bars per
18.8.2.3); 6x6x9 at 15-ft bays, sdc_e_near (36x36 fc8 with #11 at 1.7% /
16x30). Total design time for the twelve fell from 288 to 122 min. The
full evidence trail in `design.json` is kept unchanged by decision
(2026-09-13): verification comes first, and a 150-geometry design
verification run precedes the dataset run.

## Reaching an accepted design and lifting the generation gate

Three declaration/assertion blocks in `Design/Config.py` carry everything a
person must decide; each is part of the design request identity:

1. `SlabActionAssertions` -- after reviewing `design.json['slab_actions']`
   for a representative case (strip moments/shears, pattern rule, membrane
   check, equilibrium) assert the analysis flags and, if agreed, the
   two-way shear path.
2. `DemandPolicy` -- declare site class, risk category, occupancy, partition
   allowance, roof/snow/wind/rain scope with `declared_by` and
   `declaration_basis`.
3. `IndependentVerification` -- after the item-7 hand checks, assert the
   floor hand check, strength-model verification, detailing/model
   consistency, the 8.6.1.2 assessment, fire scope, congestion/placement
   acceptance and the floor/frame compatibility idealization (informed by
   the saved coupled comparison), with `asserted_by` and `assertion_basis`.
   The hoop/crosstie arrangement is generated and checked by code
   (`detailing.cage_layout` is evaluated, not asserted); what remains for
   the placement judgement is hook geometry, crosstie alternation and
   congestion (`detailing.congestion_and_placement`).

What an assertion cannot do: at qualification the capacity design and the
beam-plus-slab strengths are rebuilt from the record's own sections, bars,
slab layout, transfer and saved actions (`recomputed_evidence`) and
compared with the saved copies -- every capacity check, the hoops, joint
shear, anchorage and acceptance (`qualification.capacity_evidence_recomputed`),
every slab-strength entry (`qualification.slab_strength_evidence_recomputed`)
-- and `qualification.hoops_match_design` requires the transverse steel the
model and IMK calibration read to be the hoops the recomputed design
selects. All three are evaluated and outside the assertion map. Downstream
qualification then consumes only the recomputed objects: the joint evidence
is rebuilt from the recomputed capacity groups and slab strengths, and no
nested value of a saved copy is read once it has been compared (a saved
`joint_evidence` is never consumed). When a recomputation differs, the
saved evidence is withheld and its dependent items stay unevaluated. A
tampered or stale artifact therefore fails or stays open whatever is
asserted, and a tamper in a saved nested duplicate simply has no effect
(`tests/test_smrf_integration.py`, tampered-evidence test).

With all three made, `qualify_design` reports `accepted: true` for a
compliant candidate (no failures, nothing unevaluated); without them the
same candidate is sized on proxies and reports exactly which items are
open. `GENERATION_RELEASE_READY` in `SMRF_Qualification.py` is lifted by
hand once the three blocks are filled from real review, and generation is
then automatic for every case: each case designs, qualifies and runs
without further intervention, and a case whose qualification fails (for
example a mislabelled site) is refused rather than analysed.

## Uniform slab thickness and load policy

Edit `SlabConfig` in `Design/Config.py`, or pass a configured `DesignConfig`.
The candidate design driver selects the thickness; do not manually set a slab
thickness in `Structure_Parameters.py` to bypass the selector.

| Policy | Current default | Meaning |
|---|---:|---|
| `minimum_thickness_in` | 5.0 in | Practical lower bound on trial thicknesses |
| `maximum_thickness_in` | 14.0 in | Search ceiling, not a fallback thickness |
| `thickness_increment_in` | 0.5 in | Spacing of trials starting at the lower bound |
| `superimposed_dead_load_ksf` | 0.05 ksf / 50 psf | Provisional finishes/MEP/partition allowance, **excluding concrete** |
| `live_load_mass_fraction` | 0.0 | Provisional ordinary-office seismic mass assumption; gravity still includes full L |

These are visible research defaults, not a verified occupancy/load schedule.
The old 0.15 ksf bundled dead load is used **only** when no slab has been
selected (legacy mode); it is not added to the new computed slab weight.

The supported slab is monolithic, nonprestressed, normalweight concrete with
Grade 60 reinforcement and concrete strength matching its beams. All panels
have beams on four sides. It has no openings, cantilevers or drops, and one
geometry/thickness/load family at all levels. For every thickness trial:

1. Calculate beam-face clear spans and their aspect ratio.
2. Recalculate gross T/L beam stiffness for slab sizing using ACI 318-19
   8.4.1.8, with one flange on perimeter beams and two on interior beams.
   A full transverse-bay slab strip is retained at perimeter beams as a
   conservative stiffness-ratio assumption; adjacent flange projections cannot
   overlap. This is a thickness-screen stiffness, **not** an update to the
   frame's cracked beam stiffness or its nominal/probable moment capacity.
3. Evaluate Table 8.3.1.2 on every panel, including the 8.3.1.2.1 discontinuous
   edge adjustment where applicable, using the trial thickness consistently.
4. Select the first trial passing every panel and apply it to every floor.

Clear-span aspect ratio above 2, weak-beam `alpha_fm <= 0.2` cases requiring
the separate 8.3.1.1 procedure, and unsupported systems do not receive a
fabricated passing design. Thickness-ladder exhaustion triggers a bounded retry
with deeper compatible beam sections; invalid policy/unsupported aspect ratio
is not treated as sizing exhaustion. If no combination works, the search raises
a diagnostic rather than selecting the maximum anyway. Production slab
strength/shear/reinforcement and the applicability of explicit long-term
deflection checks remain `not_evaluated`: implementing a capacity routine does
not supply its missing design demands or complete the separate checks.
Do not use a direct-design slab moment procedure without checking its scope
(including continuity/span-count conditions).

Load accounting is an explicitly idealized centerline model: slab weight over
the centerline floor footprint, beam drops over column-face clear spans, and
columns over story height minus slab thickness. Equivalent member line weights
are applied over the centerline elements. Member mass is lumped as one column
story at its upper floor plus half each incident beam. Perimeter/joint surfaces
are not a physical concrete quantity takeoff. The legacy nodal/half-tributary beam
floor-load options remain as the request fallback (`GRAVITY_LOAD_MODEL`) and
are **not** a solved two-way slab load path; a slab-aware design replaces them
with the solved transfer described below.

`design.json` saves the slab input policy, thickness trials, panel evidence,
governing panel, load inventory and frame iteration history. The artifact schema
is `rc_smrf_candidate_v6_slab_transfer`; policy/source changes reject old caches.
Cached slab evidence is recalculated and checked against the frame before use.
Geometry overrides invalidate a prior selection. New-mode `global_parameters`
and `hybrid_metadata` include `floor_loads` and `reinforcement_geometry`
provenance (cover, bars, hoop/core geometry, aggregate and material basis).
Historical missing provenance remains unknown. The 43-column hybrid
feature order is preserved; slab fields are not silently appended to an old
trained model. Existing datasets and projector files are not rewritten or mixed
with the new methodology.

## Slab reinforcement routine and demand contract

`SMRF_Slab_Reinforcement.design_slab_reinforcement` is an isolated, deterministic
capacity-and-layout routine. It accepts the chosen slab thickness/materials,
complete panel inventory and floor count, plus separately supplied verified
out-of-plane demands. It does not estimate demands using a generic span
coefficient or use the rigid-diaphragm frame analysis as a plate solution.

Every panel must supply both x/y directions and top/bottom tension faces,
including explicit zeros. Factored moment and support-face shear magnitudes
are per foot of slab width. Evidence must identify its analysis source/model
hash, applicable analysis method, the full-floor and live-load-pattern envelope,
spatial maxima rather than averaged strip moments, treatment of twisting,
and the shear tension face. A slab-input hash prevents reusing these actions
after thickness, material, panel inventory or floor count changes. Supported
actions are uniform area gravity loads without concentrated loading or membrane
axial force; the calling analysis must establish those scope conditions.

The routine searches #4/#5/#6 bars and allowed spacings. It uses a common bar
size but permits separate x/y top/bottom spacings, uniform across the building.
Two orthogonal bars at each face have different effective depths; the opposite
face mats must fit within the selected thickness. It checks ACI minimum tension
steel, critical-section maximum spacing, cover and clear spacing, rectangular
strain-compatible flexure and the required tension-controlled strain. One-way
shear uses ACI 318-19 Table 22.5.5.1(c), with longitudinal reinforcement ratio
and size effect, without an obsolete constant `2 sqrt(fc)` lower bound.

`screen_passed` refers only to those strip section checks. `accepted` remains
false even if they all pass: punching/local minimum steel, anchorage/detailing,
serviceability and fire scope have not been established. Saved capacities and
bar selections are recomputed when audited. Empty, stale, malformed or merely
unverified action evidence does not generate a fallback bar layout. Verification
flags are upstream engineering assertions, not evidence that this routine has
independently checked the source analysis.

## Independent floor diagnostic and numerical benchmarks

`SMRF_Floor_Analysis` builds an elastic `ShellMITC4` floor with shared nodes
across every bay. Every beam centerline is a vertical support; rotations remain
free. The model uses gross isotropic slab stiffness, no cracking or creep, and
no elastic supporting beams. This is **not** the production NTHA floor model.
It refuses an existing nonempty OpenSees frame domain and uses only a separate
scratch model after the candidate frame actions have been retained.

When enabled in `FloorAnalysisConfig`, candidate evidence includes 1.4D,
1.2D+1.6L with full/even/odd live patterns, service D+L, and a refined full-live
mesh. These patterns do not prove a complete governing live-load envelope.
Reports retain nodal support reactions, separate beam-line/intersection
reaction accounting, vertical displacement, and signed Gauss-point moments,
twisting and transverse shears. Resultants use **kip-in/in** and **kip/in**;
these differ by a factor of 12 from the reinforcement routine's per-foot units.
Raw Gauss-point bending envelopes are not support-face/extrapolated design
actions and have no accepted twisting-to-reinforcement transformation.

The rigid-line floor diagnostic always retains `verified: false`. Numerical
convergence and equilibrium do not approve its support idealization or
qualify production slab reinforcement. `transferred_to_frame_design` is true
only when the separate flexible-beam transfer below is present.

## Slab-to-frame gravity transfer

`Design/SMRF_Floor_Transfer.build_floor_transfer` runs the flexible-beam
floor model twice per candidate (unit dead pressure = slab self-weight + SDL;
unit live pressure on every panel). Beam segments on the slab mid-plane share
the shell nodes and carry the slab screen's gross T/L-section inertia
(`SMRF_Slab._beam_inertia`; one flange on perimeter lines, two on interior
lines) less the flange's own mid-plane term, which the shells provide. Only
column intersections are held vertically; column shortening and joint
rotation belong to the frame model, so beam moments from the floor model are
informational only.

What each beam receives is read from its solved segments: at an interior
beam-line node the slab delivers the shear jump (a vertical force) and the
torsion and bending couple jumps; the column node takes the reaction less
the beam end shears, plus the beam-end couples. Nothing is redistributed. A
point support in a plate mesh draws load that a real column bearing over its
footprint does not (16% of the floor at 8 subdivisions per bay, 3% at 16 on
20x17-ft bays), so the transfer uses the finest even mesh within 8192 shells
(`transfer_mesh_per_bay`, 14-16 per bay for the dataset floors) and records
the direct-to-column fraction and the finite-footprint share for
information. Beam totals converge from below with refinement and reproduce
the classical two-way tributary for interior beams (within 3% at 16/bay).
The exported inventory balances force and both global first moments, couples
included, to solver precision for every pattern.

`Loads/Gravity_Loads` applies the record on every floor as `-beamPoint`
element loads at the saved span fractions (dead and live factors kept
separate, live patterns selectable), the exported couples as nodal moments
at the beam-end and column joints, and the column direct loads; the
tributary seismic mass is unchanged. `apply_design` validates the cached
transfer against the frame geometry, sections, slab thickness, pressure
totals and the couple-inclusive balance before restoring it. Tests pin:
exact force and moment balance for asymmetric patterns at meshes 4/8/12,
the stiff-bending/free-torsion limit against the rigid-line diagnostic,
mirror symmetry, the falling point-support share, the same floor total as
the nodal path under equal factors, tampered/relocated/missing loads and
couples, and balanced base reactions on the elastic frame.

Still open here: T-beam centroid offset/membrane action in the bare frame
(quantified per design by the coupled comparison, judged in
`floor_frame_compatibility_reviewed`) and independent hand verification of
a representative floor.

## Slab strip actions and reinforcement

Shear path (basis corrected twice, 2026-09-14). In ACI 318-19 Chapter 8,
8.4.3 is the slab's one-way shear (8.4.3.1: Vu at the face of the support)
and 8.4.4 is two-way shear (8.4.4.1.1: evaluated in the vicinity of
columns, loads and reaction areas at the 22.6.4 critical sections); the
earlier text had 8.4.4.1 as one-way, which it is not. What the checks
record: the strip routine checks the slab for one-way shear at the beam
faces (8.4.3.1); the slab's reactions are line reactions on beams and every
column stands at the intersection of an x-beam line and a y-beam line
(`columns_at_beam_intersections`), so the slab's load reaches the columns
through the beams; how much of the panel shear the beams take is the
beam-supported-slab rule of ACI 318-14 8.10.8 (not in the 2019 body text;
R8.2.1 keeps the method available): Table 8.10.8.1 gives 100% of the
45-degree-tributary shear to a beam with alpha_f1 l2/l1 >= 1.0 (8.10.8.2
interpolates below that), 8.10.8.3 adds the loads applied to the beam
directly including its stem, and 8.10.8.4 requires resistance to the total
shear on the panel. The criterion recorded is the Table 8.10.8.1 one --
alpha_f1 l2/l1 with l1 the beam's span and l2 the transverse width, on
every edge of every panel (`alpha_f_l2_l1_min`; alpha_f alone is not it, a
stiff beam on a long narrow panel can fall below 1.0 while alpha_f does
not) -- and the beams are designed for the plate's actual reactions (their
sum is checked against the floor load by the transfer ledger) plus the drop
weight, which is 8.10.8.3/8.10.8.4 by analysis rather than by tributary.
Using a direct-design-method clause as the load-path criterion for a slab
analysed as a plate is an interpretation, and it is the engineer's:
`two_way_shear_path_assessed`. Beam intersections alone establish nothing;
they are one of the three conditions (intersections, alpha_f1 l2/l1 >= 1.0,
the assessment), and the check is `not_evaluated` without all three.

`Design/SMRF_Slab_Actions.build_slab_action_evidence` produces the demand
evidence the strip routine requires, from the same flexible-beam floor model
as the transfer: 1.4D and 1.2D+1.6L (ACI 5.3.1), with live-load patterns per
6.4.3.3 -- all panels when L <= 0.75D (the project's office floors), otherwise
3/4 factored live on checkerboards and on the panels adjacent to each interior
support line, enveloped with the all-panel case. At every Gauss point outside
the beam widths the sagging-positive mx, my and raw mxy are resolved with the
Wood-Armer rules into x/y top and bottom design moments; the panel envelope is
the maximum over those points, cases and (identical) floors. Support-face
shear is recovered at the beam face: in the element row adjacent to each
support line the two Gauss points sharing a transverse coordinate are
interpolated/extrapolated linearly to the face position (centerline plus or
minus half the beam width) -- MITC4 transverse shear is constant across that
pair, so this is the adjacent element's shear -- for both supports and every
row, and the top-face row carries the maximum with its tension side
confirmed from the twisting-resolved top demand there; the bottom-face row
carries the maximum shear in the pure sagging zone, where the bottom mat is
the longitudinal steel the one-way shear strength relies on
(`floor.support_face_shear_recovery`). Membrane resultants are shown to vanish.

The verification flags the strip routine reads are **engineering
assertions**, made in `Design.Config.SlabActionAssertions` (with
`asserted_by`, `assertion_date` and `assertion_basis`) after reviewing this
evidence; the module records each item's numerical basis separately and
combines the two, so a flag is True only when asserted and its numerical
precondition holds. Equilibrium, benchmarks and unit tests establish software
behavior, not verification. Assertions are part of the design request
identity, so a design made under them records who asserted what. The
computed evidence is always saved as `design.json['slab_actions']` for
review; only asserted evidence becomes routine input. The qualification
report keeps `floor.independent_hand_verification` open regardless.

`SMRF_Slab_Reinforcement.design_slab_reinforcement` then selects one bar size
and four spacings (method v2). With floor context it also settles: straight
development per 25.4.2.4 against half the shortest clear span (continuous
uniform mats), continuity/extensions by construction, crack-control spacing
(24.3.2 at fs = 2/3 fy), deflection control by minimum thickness (8.3.2.1),
two continuous bottom bars through the column core (8.7.4.2, a placement
requirement) and corner reinforcement (8.7.3.1). The shear path (the beams
take the whole panel shear, ACI 318-14 Table 8.10.8.1 at alpha_f1 l2/l1 >=
1.0 on every edge; the slab is checked for one-way shear at the beam faces,
318-19 8.4.3.1; no slab-column critical section, 8.4.4.1.1) is a code
interpretation and is settled only when `two_way_shear_path_assessed` is
asserted; otherwise it stays `not_evaluated` with the reasoning recorded.
Fire resistance and the 8.6.1.2 applicability question stay `not_evaluated`.
The selected record is `design.json['slab_reinforcement']`, recomputed by
the qualification audit and restored by `apply_design`. Within the approved
#4/#5/#6 family, the 2h spacing limit governs a 5-in slab, so the mats are
#4 @ 10 in each face.

## Capacity design

`Design/SMRF_Capacity_Design.build_capacity_design` runs after every steel
pass on an explicit state (sections, bars, slab layout, floor transfer,
column axial and shear envelopes from the final factored cases) and returns
the evidence the joint checks consume, the hoops it selected, and its own
checks; the driver installs the hoops before the state is captured, so
`rho_sh` in the IMK backbones is the designed value.

- SCWB screen: members are uniform over height, so the governing joint is an
  interior roof joint -- one column at the low roof axial load against a
  hogging beam and a sagging beam: Mnc >= 1.2 (Mnb- + Mnb+) of the strongest
  family. Floor joints need only 0.6 (Mnb- + Mnb+) and never govern. There is
  no roof exemption in the code text and the joint qualification applies
  none. Candidate rungs are priced with at least 1% steel (18.7.4.1). The
  screen sizes the section; acceptance is the per-joint check below.
- SCWB at every joint, inside the search: each steel iteration runs the
  joint adapter and `scwb_check` on the installed cage with the solved
  actions (column Mn enveloped over every factored combination and both
  compression faces; beam Mn with the developed slab where established),
  and when a joint fails, replaces the column cage with the least admissible
  cage that satisfies every failed joint, priced with the adapter's own
  section capacity at the ends of each column's axial envelope
  (`_scwb_column_steel`). The next steel iteration confirms it on all joints.
  A cage above `rho_col_practical_max` is never installed; the closest cage
  is, the shortfall is recorded (`scwb_joint.steel_exhausted`) and the
  section search steps the column size. Checks the adapter cannot evaluate
  (no established slab strength) stay unevaluated here as in qualification.
- Beam hoops likewise: `beam_cage` supports the top and bottom bars per
  18.6.4.4 / 25.7.2.3 (corners and alternate bars, 6-in clear rule,
  supported spacing at most 14 in) and bounds the legs between that minimum
  and the bars a crosstie can engage on both faces; a 7-bar layer in a
  16-in beam needs a 4-leg set, a 3-bar layer in a 10-in beam a closed hoop.
- Beam Mpr: the composite section with steel at 1.25 fy, phi = 1. Ve = Mpr
  equilibrium over the clear span plus the factored gravity face reactions
  from the slab transfer, for (1.2 + 0.2 SDS) D + 1.0 L and (0.9 - 0.2 SDS) D.
  The face reactions are a joint-face free body over the clear span: every
  span of the family, every unit case that can load it (dead; live as the
  envelope over the all-panel case and each saved pattern), point forces and
  bending couples at their positions from the face, loads inside a column
  footprint to the column, the beam's physical drop weight
  (`beam_drop_weight_kip_per_in`, b (h - t)) on the clear span -- the frame
  element carries the same weight smeared over its centerline length, and
  the two ledgers agree (smeared L/2 = physical ln/2, tested) -- and each end
  is governed by its own worst span (the cross-check's counterexample: a
  span with the largest total is not the span with the largest end
  reaction).
  Vc = 0 in the hinge zone when the mechanism shear is at least half of Ve
  (18.6.5.2); Vs <= 8 sqrt(fc) bw d or the beam rung grows. Hoops from
  Av fyt d / Vs and min(d/4, 6 db, 6 in) on a 1-in grid from 3 in, uniform
  along the member.
- Column Ve (18.7.6.1.1), per frame direction: min(2 Mpr,col / ln over the
  story's factored axial range, the beam probable moments the joints in that
  direction can deliver -- half to each column at a floor joint, all of it
  at the roof) and never less than the analysis shear. Mpr,col is the
  probable strength about that direction's axis (the x frame bends the
  column through h with the top/bottom bars in the extreme layers; the y
  frame through b with the corner and side-face bars extreme), the depth d
  and the width in Vc follow the direction, and Vs <= 8 sqrt(fc) b d is
  checked in each. Vc = 0 when the mechanism shear is at least half of Ve
  and Pu < Ag fc / 20, else 22.5.5.1 with axial load. Hoops from Vs, the
  18.7.5.4 confinement area (Ag/Ach and 0.09 fc/fyt; the Pu > 0.3 Ag fc form
  when it applies) and 18.7.5.3 spacing (b/4, 6 db, and the conservative
  4-in cap retained in place of so = 4 + (14 - hx)/3, which is recorded).
  One hoop bar and one spacing serve both directions; the leg count is
  chosen per direction, as 18.7.5.4 defines Ash per direction: the legs
  that cross the b faces (the hoop's two legs parallel to h plus the
  crossties on the b-face bars) carry Av for shear along h and Ash
  perpendicular to bc = b - 2 cover, and the legs that cross the h faces
  the reverse. For a hoop bar every constructible leg pair is priced and
  the pick is the largest spacing, then the fewest legs; the record stores
  `col_stirrup_legs_by_direction` beside `col_stirrup_legs`, which is the
  lighter direction and is what the hinge calibration (rho_sh) and the
  legacy shear checks read. The 150-case verification run found 12 of 150
  designs stuck without a column hoop under the earlier single count both
  ways: 3-top/4-side and 4-top/3-side bar layouts need 3 legs one way and 4
  or 5 the other, which no common count satisfies, and the section loop
  escalated to 36x36 with `column_capacity_shear` as the reason every
  iteration.
  The leg count is one the cage can hold: `SMRF_Cage_Layout.column_cage`
  places the bars on each face from cover, hoop and bar diameters, finds
  the bars 25.7.2.3 (through 18.7.5.2(d)) requires to be supported --
  corners and alternate bars, no unsupported bar more than 6 in clear from
  a supported one, hx of supported bars at most 14 in (18.7.5.2(e)), every
  bar and 8 in under high axial load (18.7.5.2(f)) -- and bounds the legs
  across each face pair between that minimum and the number of bars a
  crosstie can engage (18.7.5.2(b)), independently per direction. The
  selected arrangement (which bars carry crossties, both directions) is
  saved with the design, hx is its supported-bar spacing, and Av and Ash
  use the legs it realizes across each direction's faces. The cross-check
  found the earlier every-face-bar-tied rule
  crediting six legs to a column with three top bars; a 3-top-bar column
  now gets a 3-leg set (hoop plus one crosstie) at the spacing that shear
  and confinement then need, or the section grows. `detailing.cage_layout`
  and `column.cage_layout` / `beam.cage_layout` evaluate the arrangement.
- Joint shear (18.8.4): Vj = 1.25 fy (beam top bars + slab bars in the
  effective flange) + 1.25 fy (opposite beam bottom bars) - Vcol, with
  Vcol = sum Mpr / H (2 sum Mpr / H at a terminating roof column), for every
  joint connectivity the plan contains -- corner, a column on the x
  perimeter line (two edge x-beams, one terminating interior y-beam), a
  column on the y perimeter line (the mirror) and interior -- in both
  directions and at floor and roof (`joint_kinds`). gamma per
  Table 18.8.4.3: a face is confined when its beam is at least 3/4 of the
  column width; "two opposite faces" means both faces of one direction. Aj
  per 18.8.4.3 through `SMRF_Joints.rectangular_joint_area`; phi = 0.85. A
  failing joint grows the column rung. Joint transverse reinforcement
  (18.8.3.1) is the column end-zone hoops continued through the joint; the
  18.8.3.2 relaxation is recorded as applicable only when all four faces are
  confined. Vcol assumes the beam probable moments split equally between the
  columns above and below with inflection at mid-height (all to the single
  column at the roof); that distribution is an assumption to verify.
- Terminating bars (18.8.5.1): ldh = fy db / (65 sqrt(fc)) >= max(8 db, 6 in)
  against column depth - cover - hoop; through bars keep 18.8.2.3.
- Splices: beams may lap (Class B, 1.3 ld) only between the 2h hinge zones
  under hoops at <= min(d/4, 4 in); when that length does not fit (it does
  not on 10-ft bays with 26-in columns) the design is Type 2 mechanical
  splices (18.2.7). Columns lap in the center half of the clear height.

On the 3x3x8 review geometry, with the slab action assertions made, the
outcome is 26x26 columns (rho 1.04%, #5 six-leg hoops at 4 in), 10x16 fc-8
beams with 2#8 (#4 hoops at 3 in), joint shear passing everywhere with the
roof interior joint closest (gamma = 8), and a qualification report with no
failures and open items limited to the `demands.*` scope statements (item 6),
the independent-verification items (item 7), congestion/placement, fire and
8.6.1.2 applicability. Without the assertions the same run sizes the frame
on rectangular beam strengths as a proxy and reports the slab contribution,
SCWB and slab actions as `not_evaluated`; it is not an accepted design.

## Demand basis

A declaration is validated, not trusted (2026-09-14): `DemandPolicy` must
carry a non-blank author, an ISO calendar date and a non-blank basis, a site
class and risk category from ASCE 7-22's lists, a non-blank occupancy,
finite nonnegative loads, an accidental-torsion ratio of at least 5% and
Boolean flags (`SMRF_Demands.demand_policy_problems`). Site class and risk
category must be exactly the ASCE 7-22 spelling: the fourth cross-check
found the validator folding " D " to D while the 11.4.8 site-specific flag
read the value verbatim, so the padded spelling dodged the flag. The
validator no longer normalises anything and the flag reads the declared
value as declared. The design refuses a partly filled or invalid
declaration outright, and at qualification every `demands.*` item stays
unevaluated with the rejection listed -- a whitespace basis, a mistyped or
padded site class is never approved evidence.

`Design/SMRF_Demands.evaluate_demand_basis` reads the design record: the
declared `DemandPolicy`, the load inventory, the saved torsion assessment
and the drift/ELF assumptions. Items:

- `demands.site_hazard`: SDC derived from SDS/SD1/S1 and risk category
  (Tables 11.6-1/11.6-2, S1 >= 0.75 rule) must match the preset label; a
  declared Site Class D/E/F with S1 >= 0.2 flags the 11.4.8 site-specific
  requirement.
- `demands.elf_eligibility`: Table 12.6-1 with height, the torsional
  irregularity from the torsion runs, and regularity by construction
  (rectangular grid, frames on every line, uniform stories/sections).
- `demands.accidental_torsion`: 5% eccentricity applied at every level in
  every seismic combination (signed with the force; the doubly symmetric
  building's member envelope is the same for either sign), Ax from
  delta_max/delta_avg at the two extreme frames with torsion applied
  (Table 12.3-1: 1a > 1.2, 1b > 1.4; Ax = (dmax/1.2 davg)^2 <= 3).
- `demands.live_load_patterning`: ACI 6.4.2 alternate-span and adjacent-
  span arrangements as slab panel patterns, solved in the floor model as
  unit live cases of the transfer and combined as 1.2D + 1.6L; the
  canonical combination rule and the saved action inventory carry them.
- `demands.load_scope`, `demands.effective_seismic_weight`: evaluated
  against the declared roof live/snow/wind/rain scope and the 12.7.2
  inventory (slab + SDL with a >= 10 psf partition allowance for offices,
  no storage fraction, member weight).
- `demands.drift_analysis_basis`: 0.35/0.70 Ig cracked stiffness, rho = 1
  drift forces at the Cu Ta-capped period (conservative), Cd = 5.5, P-Delta,
  full D+L gravity state, derived SDC for the limits.

## Beam-plus-slab strengths

Where the slab ends: at a beam's exterior end the slab bars in its flange run
to the building edge, and the only concrete beyond the critical section is
the perimeter beam. The mats are credited there only if a standard hook
into that beam develops them -- ACI 318-19 25.4.3.1 (the db^1.5 form, psi_r
from the mat spacing, psi_c from f'c, psi_o = 1 with the beam continuous
along the perimeter; at least 8 db and 6 in) against the beam width less the
far-side cover and hoop (`perimeter_slab_bar_anchorage`). Where the hook
fits, the entries and hinges keep the composite hogging strength and record
the hook; where it does not, neither mat is counted at that end in either
sign: the hogging entries carry `slab_basis = terminated_undeveloped` with
zero slab contribution, the sagging entries carry
`flange_concrete_undeveloped_bars` -- the flange concrete stays in
compression and needs no development, but the bottom mat is dropped, since
undeveloped bars are not tension steel either (the fourth cross-check
found it still counted: 1242.61 vs 1110.56 kip-in on the test fixture) --
and the joint rule accepts both bases (a nonzero term only with a strength
basis). The hinge at that end yields at those strengths in both directions
(`yield_moment_y_hogging_i/j`, `yield_moment_y_sagging_i/j`; the spring's
own yield rotation is per end, `theta_y_spring_y_i/j`, and the plastic
rotation post-processing uses the end's value). The slab design itself
checks the same hook (`slab_perimeter_bar_anchorage`, both axes) and the
bar ladder checks each candidate spacing with its own hook (psi_r = 1.6
below 6 db): a spacing whose hook does not fit is not offered, a bar with
no fitting spacing is not offered -- pricing every bar at the tightest
spacing offered, as the ladder first did, rejected #5 mats that fit at
their selected 12 in. Interior ends are
developed by continuity (25.4.2.4 within half the adjacent clear span,
`slab_bar_development`). The probable strengths the capacity design uses
for beam shear and joint shear keep the composite value everywhere, which
is the conservative side for a demand. On the review case #4 mats need 6.0
in and the 10-in perimeter beam offers 8.0; #6 mats (9.7 in) would not fit
and the ladder would not select them.

`Design/SMRF_Beam_Slab_Strength` computes, per beam family (x/y, perimeter
or interior line), the rectangular and the composite nominal moments by
strain compatibility with every bar layer discrete: beam top and bottom bars
plus the slab mats within the Table 6.3.2.1 effective flange (interior:
min(8h, sw/2, ln/8) each side; perimeter: min(6h, sw/2, ln/12) one side).
Negative bending puts the web in compression and both slab mats in tension;
positive bending puts the flange in compression. The results go three
places, deliberately the same numbers: `design.json['beam_slab_strengths']`
entries (`developed_effective_width`, increment = composite - rectangular)
for the joint adapter's SCWB checks; the driver's SCWB screen
(`_scwb_required_column_moment`, 1.2 x the largest composite Mn); and the
IMK beam hinges, which are now asymmetric (`Model/IMK_Hinges.beam_yield_moments`).
The spring sign was measured: hogging is positive deformation at end i and
negative at end j (`tests/test_beam_hinge_asymmetry.py`), so each end's
IMKBilin receives (My-, My+) or (My+, My-) accordingly. On the 3x3x8 review
geometry the interior-beam negative strength roughly doubles (1049 -> 1999
kip-in with 3#6 top bars) and the column screen moves from 18x18 to 22x22;
the exact joint check then fails only at roof joints, where a single column
faces two slab-enhanced beams -- open item 3.

The numerical tests currently establish:

- A uniformly loaded, simply supported 200-by-200-inch thin plate (h/L=0.01)
  approaches an independently evaluated Navier sine-series solution as the
  per-bay mesh increases from 4 to 8 to 16. At 16 subdivisions, tested relative
  tolerances are below 0.2% for center deflection and 0.5% for the sampled
  bending-moment maximum. These are test bounds, not accuracy guarantees for
  an arbitrary multibay floor.
- A 200-by-300-inch rectangular plate preserves the x/y moment ordering and
  agrees with the thin-plate reference within tested 2% moment and 1%
  displacement bounds at 12 subdivisions.
- A 2-by-2-bay mesh develops continuity-related hogging moments and balances
  vertical force and first moments; beam-line plus separately retained
  intersection reactions account for every supported node exactly once.
- Explicit pattern changes affect live pressure without deleting dead load;
  failures stay unverified and clear only the diagnostic's scratch domain.
- Slab-section hand calculations check strain compatibility, tension control,
  concrete stress-block limits, reinforcement/size effects in shear, and
  rejection of impossible layouts, incomplete inventories and stale evidence.
- Joint-section tests compare known neutral-axis equilibrium, unequal positive/
  negative beam reinforcement and both rectangular column axes; joint inventory
  and local axial-sign tests use actual elastic-builder conventions.

These tests validate bounded algorithms and implementation conventions, not
full SMRF engineering qualification. Representative coupled floors, frame
demands, anchorage/cage details and nonlinear behavior still require independent
engineering checks before a new pilot can establish the methodology.

## Generation safeguard and diagnostic use

`Ground_Motion_Main.py` stops before design/NTHA when the new methodology is
not release-ready. This intentionally prevents launching many incomplete
designs while the overhaul is in progress. It does not terminate any existing
external processes. No historical output or parameter plan is migrated.

`--design-only` performs the candidate search and saves its qualification report
in `design.json`, including the enabled independent floor diagnostics, then
returns without NTHA. It is diagnostic, not acceptance.
Use a **new output root** and an explicit `--design-file` when exercising it;
otherwise the historic parent-directory design-path convention still applies.
An old design artifact is rejected, never silently upgraded.

`--skip-design` remains an explicit legacy-analysis option. It must not be used
to bypass qualification for a dataset claimed to consist of designed SMRFs.
Neither code tests nor a manually edited `accepted: true` clears open checks.

`Design/Evidence_Summary.py` writes the evidence summary the assertion
review is made from: it designs the representative case (3x3x8, 10-ft bays,
sdc_d_high by default) under PROBE slab-action and demand assertions with
`IndependentVerification` empty, or takes a saved `design.json`, and writes
`evidence_summary.md` beside it. Every quantity in the document is computed
from the artifact or measured on the model rebuilt from it (IMK period,
gravity sway, hinge yield moments per family, load ledger, hogging against
each beam's fixed-end reference) with the hand reference beside it where a
hand formula exists; the 2026-09-14 review of the document found two
sentences that had been typed in from an earlier design and removed the
possibility.

`Design/Verify_Designs.py` is the design-only verification run over the
generation plan's own cases: it reproduces `build_plan`'s geometry and
hazard sampling (same seed, shuffle and round-robin; the first N cases are
the first N the dataset run will design), designs each in a fresh
interpreter with the committed `Design/Config.py` through
`load_or_create_design`, keeps every `design.json`, resumes, and writes
`summary.md` / `summary.csv` / `summary.json` with the failed and open item
histograms. `--probe-assertions` fills the three blocks with PROBE values
(labelled in every request identity) to exercise the pipeline before the
real assertions exist; its summary says so and certifies nothing. Three
plan cases ran at 8 min per case with 100-180 MB each, so 150 cases are
about 5-6 h on four workers and 20 GB. For the multi-machine run see
`Design/DESIGN_VERIFICATION_RUN.md`: `--plan-only` prints the plan SHA to
compare across machines, `--case-start/--case-end` take a disjoint slice,
`--request-stop` stops gracefully, the PROBE stamp date is fixed in
`plan.json` so the whole run has one request identity, and
`--summarize-only` rebuilds the summary after the slices are copied together.

In the generation scheduler a case whose saved design was refused by
qualification is terminal (`design_refused` in `case_results.json` and
`generation_state.json`, with the failed and unevaluated item ids): the
child would refuse it again on every resume at the cost of a full design.
`--retry-refused` re-designs them after the plan or the design inputs change.

After the remaining design work is verified, freeze one new plan and methodology
version, distribute disjoint case ranges, and compare hashes on all computers.
Keep old and new datasets separate until their compatibility is assessed.

## Reference basis

- [ACI 318-19 / reapproved 2022, publisher](https://www.concrete.org/store/productdetail.aspx?ItemID=318U19)
- [ACI 318-19 original text, slab provisions 8.3.1.2 and 8.4.1.8](https://www.ocf.berkeley.edu/~chiep/wp-content/uploads/2024/01/CE-123-ACI-318-19.pdf)
- [ASCE/SEI 7-22, publisher](https://www.asce.org/publications-and-news/codes-and-standards/asce-sei-7-22)
- [NIST SMRF design guide, second edition](https://nvlpubs.nist.gov/nistpubs/gcr/2016/NIST.GCR.16-917-40.pdf)
- [OpenSees ShellMITC4 formulation](https://opensees.berkeley.edu/OpenSees/manuals/usermanual/640.htm)
- [OpenSees ElasticMembranePlateSection](https://opensees.berkeley.edu/OpenSees/manuals/usermanual/231.htm)
- [TU Delft plate theory and Navier solution](https://ocw.tudelft.nl/courses/advanced-structural-analysis/subjects/plate-theory-ii/)

The NIST guide uses ACI 318-14 and ASCE 7-16; it is a methodology reference,
not the source for every numeric 2019/2022 provision. Implemented checks carry
their intended clauses and require an independent engineering review.
