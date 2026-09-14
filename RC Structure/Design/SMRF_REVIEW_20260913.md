# SMRF follow-up review — 2026-09-13

## Cross-check follow-up (Astra, 2026-09-13) -- resolution

1. Confirmed: the pattern comparison enforced the per-column criterion only.
   `floor.coupled_total_load_pattern` now enforces the pattern total on its
   own, and both cases' rows must carry a complete, unduplicated, finite
   column inventory before anything is evaluated. The 1.04-scaling and
   duplicate/missing-identity reproductions are tests.
2. Confirmed: qualification compared summaries and then read nested saved
   duplicates. Now the capacity design and the slab strengths are recomputed
   from the record, compared (`capacity_evidence_recomputed`,
   `slab_strength_evidence_recomputed`, `hoops_match_design`), and only the
   recomputed objects are consumed downstream -- the joint evidence is
   rebuilt from them and a saved `joint_evidence` is never read. The
   `beams.hoops.bar_size = 11` reproduction now has no effect on any
   consumed value (test, on a JSON round-tripped record so nested copies do
   not alias).
3. Confirmed: the smeared centerline line weight was cut at the faces and
   multiplied by the clear span (15% of the beam-weight term). The free body
   now uses the physical drop weight (`beam_drop_weight_kip_per_in`) on the
   clear span; a ledger test pins smeared L/2 = physical ln/2.

Engineering notes: the compatibility question stays open, and the record
now carries `moment_significance` so it is judged against the demand it
affects (4x4x7: coupled gravity base moments are 3-4% of the governing
design moment; the 17% model gap is 0.5-0.6% of it). `cage_layout_verified`
is not set by any probe; the sweep probe leaves `detailing.cage_layout` open.
The force pair's window shear is an artifact, not a demand; arm sensitivity
is tested (0.5 / 1 / 2 in). `GENERATION_RELEASE_READY` is still False.

## Cross-check (Astra, 2026-09-13) -- findings and resolution

Six findings; all six verified and acted on, no thresholds changed:

1. High, confirmed: interior bending couples were split to the end joints
   (no beam response on a fixed-ended element). Now carried as in-element
   force pairs; torsion keeps its consistent linear split. Verified on the
   real design frame against a subdivided explicit-node reference
   (reactions, joint displacements/rotations, beam end forces, bending
   diagram; full live, asymmetric pattern, dead only) and on the
   counterexample itself: `tests/test_smrf_transfer_mechanics.py`.
2. High, confirmed as a scope limit: the coupled comparison evaluates the
   vertical load path only. It now also runs the first ACI 6.4.2 pattern
   (evaluated, same criterion) and the coupled reference's own mesh
   sensitivity, records the base-moment ratios and states what is not
   compared; acceptance of the moment agreement stays the engineer's
   judgement item. After fix 1 the 4x4x7 corner/edge base-moment ratios
   read 0.82/0.84 (were 0.69/0.69).
3. High, confirmed: `_face_reactions` chose the span with the largest sum.
   Now a joint-face free body over the clear span, every span and unit case
   (patterns enveloped), couples included, each end governed by its own
   worst span (`tests/test_smrf_capacity_design.py`, the reviewer's numbers).
4. High, confirmed: stale summaries and altered hoops survived. The coupled
   summary is recomputed from its rows and bound to the frame's input
   signature; the capacity design is recomputed from the record and compared
   check by check (`qualification.capacity_evidence_recomputed`); the model's
   hoops must equal the designed hoops (`qualification.hoops_match_design`).
   Neither is assertable; a failure withholds the saved evidence. The
   reviewer's two mutations now fail (tampered-evidence test).
5. Medium, confirmed: quantity checks stated more than the detailing
   establishes. Confinement and hx checks now say they assume every face bar
   tied; the arrangement is the new open item `detailing.cage_layout`
   (`IndependentVerification.cage_layout_verified`).
6. Medium, confirmed: only one edge orientation was enumerated. `joint_kinds`
   now covers corner, x-edge, y-edge and interior in both directions (the
   terminating interior-family beam is its own case).

`GENERATION_RELEASE_READY` is still False.

## Parameter-space sweep and search closure (2026-09-13, latest)

Twelve design-only corner cases of the generation plan space were run under
probe assertions (`outputs/smrf_sweep_20260913/sweep_summary.md`). None
crashed; the transfer validated and the coupled comparison held within 5%
in every case. Five did not reach `accepted`, all because the section search
closed on proxies (DCR band, SCWB sizing screen, column-only drift step) and
not on the per-joint results qualification evaluates. The search now closes
on every evaluated requirement -- see "Section search closure" in
`SMRF_METHODOLOGY.md` -- and the elastic combinations are solved once per
section instead of once per steel iteration (the frame stiffness does not
depend on the bars; a test pins the captured-action checks to the live
domain). On the final code all twelve cases reach `accepted` under probe
assertions (`outputs/smrf_sweep_20260913_final/sweep_summary_final.md`).
`GENERATION_RELEASE_READY` is still False; the assertion blocks are still
empty in the committed configuration.

## Resolution of the transfer findings (2026-09-13, later)

The three follow-ups agreed with the user were implemented; see
"Transfer resolution" in `SMRF_METHODOLOGY.md`:

1. The footprint redistribution was removed and the solved interface couples
   are exported with the forces (transfer schema v3). The exported inventory
   now balances force and both first moments to ~1e-13 for the corner-loaded
   3-by-2 counterexample and for every pattern at meshes 4/8/12; validation
   recomputes the couple-inclusive balance and rejects tampered, relocated or
   missing couples. Multibay designs with ACI 6.4.2 patterns run again.
2. `floor.coupled_frame_compatibility` is an evaluated comparison against the
   coupled monolithic model (`SMRF_Coupled_Comparison`, run for every
   selected frame): column base verticals within 5%, totals to 1e-6, moment
   ratios recorded. The residual idealization is the judgement item
   `floor.compatibility_idealization_reviewed`
   (`IndependentVerification.floor_frame_compatibility_reviewed`).
3. Slab support-face shear is recovered at the beam face from the adjacent
   Gauss pair with the tension face confirmed (`floor.support_face_shear_recovery`).

`GENERATION_RELEASE_READY` is still False; the assertion blocks are still
empty in the committed configuration.

## Subsequent implementation update

The beam span-interior finding below has now been implemented and tested in
`SMRF_Beam_Actions.py`, connected to both strength-check paths and persisted
per member/load case. Recovery uses actual pattern-factored OpenSees loads,
not the domain clock or inferred gravity factors. It includes full uniform
and point forces and checks exact extrema, with endpoint equilibrium checks.
Qualification recomputes saved envelopes; missing old evidence is unknown,
not automatically passed. Full suite now contains 299 passing tests.

The floor diagnostic additionally retains the raw interface couples and
forces. With those couples, the asymmetric FE interface balances; removing
the couples and redistributing column forces changes its moments. This is
a diagnostic result, not a repair of full slab/frame compatibility. The
raw interface is not applied to the production frame and the failed
shear-only transfer remains blocked. See the latest methodology section.

The remainder of this document is the earlier review snapshot; its beam
interior-demand item has been superseded by this bounded implementation.

Scope: review the user's new floor-transfer, slab-action, capacity-design and
loading changes, then continue the outstanding verification work. This is a
bounded software/mechanics review, not a clause-by-clause engineering approval.
No dataset, plan or previous design output was changed; no generation launched.

## Preserved work

The new code connects flexible-beam floor load distributions to frame gravity,
adds panel-pattern cases, slab Wood-Armer actions and reinforcement selection,
composite beam/slab strengths, probable-strength capacity design, accidental
torsion and demand-scope declarations. The revised beam end-moment signs and
bar-picker tie-break have regression coverage. Baseline: 275 tests passed.

## Findings addressed in this follow-up

1. **High: transfer success checked vertical force only.** A corner-loaded
   3-by-2 floor conserves total force but not its first moments after export.
   The 8-per-bay regression produces 16.666667 kip, with X first moment
   1959.880010 vs 2000 kip-in (2.006% difference) and Y first moment
   1649.816506 vs 1666.666667 kip-in (1.011% difference).
   `SMRF_Floor_Analysis` now records and checks the exported first moments,
   and `build_floor_transfer` refuses the inconsistent load case. The gate is
   fixed; the mechanical origin is **not** repaired by this safeguard.
2. **High: cached summaries could conceal altered applied loads.** Validation
   used saved beam/column totals and pass flags. Actual point loads could be
   changed, removed, duplicated or relocated without a corresponding failure.
   `validate_floor_transfer` now checks finite signed loads, exact member
   inventories, positions, descriptors, pressure totals and first moments for
   every unit case, including patterns. Signed uplift remains permitted.
   The transfer schema/method changed; old records are not auto-migrated.
3. **High: saved action provenance omitted the new load path.** The frame
   fingerprint now includes gravity model, floor transfer and demand basis.
   Live-pattern descriptors are checked against the canonical combinations.
   Changed load distributions can no longer reuse the old action fingerprint.
4. **Medium: assertion truthiness bypassed explicit review bookkeeping.**
   Values such as the string `"false"` or integer `1`, or flags without a
   named/datable basis, could be treated as slab verification. Literal True
   plus nonblank author/basis and an ISO calendar date are now required.
   Cached generated evidence is checked too. This does not verify an author's
   identity or establish that an engineering review actually took place.

## Engineering/implementation findings still open

- **Floor/frame compatibility:** the current common-floor model has vertical
  column pins, gross concentric effective T/L bending stiffness and a
  shear-only transfer with footprint redistribution. It is not a coupled
  multistory frame with eccentric slab/beam compatibility and transferred
  interface moments. Force balance or an asserted hand-check does not close
  this gap. New non-overridable `floor.coupled_frame_compatibility` check.
- **Beam interior moments:** production beams are single physical elements.
  `run_checks_phase1` and `RC_Design_Check` use only local-y end moments.
  Correct signs do not capture span-interior sagging from point/uniform loads.
  For example, a fixed-fixed UDL beam has negative ends of wL²/12 and positive
  midspan wL²/24; the corrected end-only mapper returns zero positive demand.
  Recover exact extrema between load discontinuities and verify both axes,
  support faces and units. New `beam.interior_flexure_envelope` open check.
- **Slab shear sections:** the action builder samples Gauss points outside
  beam widths, but labels the result `support_face_maximum`. It does not
  extrapolate to or integrate at the actual face. Its tension-face filter
  also follows direct mx/my, while flexural design uses twisting-resolved
  actions. Verify the recovery and face assignment before treating these as
  support-face shears. New `floor.support_face_shear_recovery` open check.
- **Capacity audit:** the qualification adapter consumes saved capacity checks
  and derived joint evidence. Independent recomputation of that complete
  state, simultaneous column P-M/shear demands, anchorage/congestion and the
  slab-inclusive capacity mechanism still warrant focused review. No claim
  that all new capacity formulas or their applicability were approved here.

## Next implementation order

1. Recover loaded-beam interior/face moments with independent analytical tests.
2. Replace/validate the floor-to-frame interface mechanics: do not mask the
   patterned-load failure by disabling patterns or inserting arbitrary forces.
   Retain numerical equilibrium tests for asymmetric patterns at several meshes.
3. Recover slab support-face shear and verify twisting/tension-face treatment.
4. Recompute capacity-design evidence against the final selected frame;
   complete representative hand comparisons and constructability checks.
5. Only then make a fresh design-only candidate and a separately authorized
   small nonlinear pilot. Keep `GENERATION_RELEASE_READY = False` meanwhile.

## Verification

The full unittest suite passes after these changes (286 tests). Added tests
cover the real asymmetric equilibrium failure, a builder refusal, altered
loads with unchanged summary flags, relocation without weight change,
duplicate/missing members, malformed values, patterned-case validation,
zero live load, stale transfer methods, changed action provenance, and
unsigned/non-Boolean/cached verification assertions. The existing in-memory
one-bay design-only smoke checks still run and remain unqualified.

The automatic numerical threshold is 1e-8; moment errors are scaled by the
larger of the applied first moment and total force times one bay length.
The percentages above instead use each applied first moment as denominator.

## Primary implementation references

- [OpenSees elastic beam-column element](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/elements/elasticBeamColumn.html)
  defines the 3D section/stiffness and force-response interface.
- [OpenSees rigid-link formulation](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/mp_constraint/rigidLink.html)
  documents the small-rotation eccentric compatibility relationship relevant
  to a future coupled implementation; no new rigid links were installed here.

No new code-compliance assertions follow from these references or the tests.
