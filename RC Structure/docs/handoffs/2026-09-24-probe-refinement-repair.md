# PROBE integration repair for the slab refinement gate — September 24

Answers `PROBE_INTEGRATION_ADDENDUM.md`. The laptop continuation's
refinement gate was correct and its callers were not: both PROBE factories
asserted the slab-action flags with `FloorAnalysisConfig.slab_refinement`
left `None`, so every PROBE design (the 150-case verification path, the
evidence-review design and the hinge-hysteresis fixture) reached
"Slab reinforcement not selected". This repair gives the gate working
callers, a defined plan-selection path and regression coverage. It does
not validate the mesh recipe for every geometry, and the 5% tolerance
remains an investigation screen.

## What changed

1. **A named recipe form of the plan** (`Design/SMRF_Floor_Mesh.py`:
   `GRADED_FACE_V1`, `graded_face_levels`, `resolve_recipe_plan`). A
   policy is either explicit (`meshes`, tolerances, basis — unchanged) or
   `{recipe, levels, max_shells, moment_tolerance, shear_tolerance,
   tolerance_basis}`; a dictionary carrying both is rejected. The recipe
   is resolved for the current bay dimensions and beam face at every call,
   so the policy in the request identity carries no coordinates and
   `Verify_Designs.methodology_sha256` agrees across geometries (tested),
   while the resolved coordinates, their hashes and the recipe inputs
   travel with the evidence.
   - `graded_face_v1` is the fixed-candidate benchmark pattern generalized:
     two nodes between the column line and the beam face at 2/7 and 4/7 of
     the face offset, the face itself, then nine nodes over the clear
     half-span at the benchmark's fractions, mirrored; level 0 keeps every
     other interior node (12 cells per bay), level 1 the full pattern (24),
     level 2 every midpoint (48), level 3 the midpoints of the near-face
     band (60). For a 240-in bay and 14-in beam the levels reproduce the
     verified `actual_graded_24/48/60` coordinates exactly (tested to
     1e-9); for any other bay or beam the face and the grading follow the
     actual dimensions, each direction from its own bay length.
   - Levels are nested and increasing, so the affordable ones under
     `max_shells` are a prefix. Fewer than two is an explicit
     `unresolved_budget` result recorded in the evidence with the dropped
     levels and their shell counts: nothing is solved, nothing is coarsened,
     nothing passes, and `evaluate_slab_actions` stays `not_evaluated`.
2. **`SMRF_Slab_Refinement`** (`bounded_explicit_slab_refinement_v2_recipes`)
   resolves recipe plans from the live sections, records `resolution`,
   `recipe_inputs` and `resolved_meshes` in the report, and
   `refinement_verified` re-resolves the recipe from the recorded inputs
   and requires the same meshes — a record whose recorded beam width, bay
   or coordinates no longer resolve to its meshes fails verification
   (tested). Explicit plans behave exactly as before; the 13 earlier
   refinement tests pass unchanged.
3. **`Design_Driver._update_slab_reinforcement`** refuses an asserted
   design with no plan before any solve, naming
   `FloorAnalysisConfig.slab_refinement`, passes the current sections to
   the resolver (a beam rung change re-resolves the plan), and names the
   refinement status (`unresolved_budget`, `comparison_failed`,
   `analysis_failed`) in the layout-not-selected error.
4. **One shared PROBE plan**, `Design.Config.PROBE_SLAB_REFINEMENT`
   (`graded_face_v1`, 4 levels, 45,000 shells, 5%/5%, PROBE-labelled basis),
   carried by both `Verify_Designs.probe_config` and
   `Evidence_Summary.probe_config`; their assertion scopes are unchanged
   (Evidence_Summary's verification block stays empty; tested).
5. **Plan identity recorded**: `DESIGN_VERIFICATION_RUN.md` now carries the
   plan SHA of the 12-16 ft story range
   (`CD813D4D66A4958B51069F67F9010EE243E6544D02E7ACB51E916EEA7653B94E`) with
   the note that the dv150 v7-v10 roots belong to the previous plan.

## Coverage

`tests/test_smrf_probe_refinement.py` (13 tests): benchmark reproduction
and the nested coarse level; face placement and scaling for three
bay/beam pairs; a rectangular floor resolved per direction; the budget
prefix for 2x6, 3x3 (evidence-review geometry), 3x5, 4x6 and the plan's
6x6 maximum inventory (12/24 only at 45,000 shells) and an unresolved
budget; a beam-size change rebuilding the coordinates and their hash;
policy validation before resolution (conflicting, unknown, out-of-range,
missing keys; explicit path unchanged); recipe evidence recording its
resolution and verifying; tampered recorded inputs, meshes or geometry
failing verification; an unresolved budget solving nothing and unable to
qualify; both PROBE factories sharing the plan; geometry-independent
methodology identity; and the driver's early missing-plan diagnostic.
`tests/test_hinge_hysteresis_fixture.py` (the real fixture, a 2x1x2
design under `Verify_Designs.probe_config`) passes again: 9 tests, 11 s,
with the fixture-cleanup fix preserved.

Budget arithmetic for the generation ranges (14-in beam, 45,000 shells):
2x6 and 3x3 floors take all four levels (12/24/48/60), 3x5 three, 4x6 and
6x6 two (12/24). On the largest floors the screen is therefore the 12-to-24
comparison, coarser than the benchmark's 48-to-60, and a rejection there
is a legitimate result of the declared budget, not a malfunction; raising
`MAX_EXPLICIT_SHELLS` or the recipe is a decision, not a repair.

## Representative native runs (public `Verify_Designs --probe-assertions` path)

Plan SHA `CD813D4D...` (12-16 ft stories), PROBE date 2026-09-24, one fresh
interpreter per case, this checkout at `bec742c0` plus this patch.

- **case_0001, 6x6x4, 10x15-ft bays, sdc_c** (the plan's largest floor):
  refused in 90 s with `Slab reinforcement not selected (slab refinement
  comparison_failed: final pair of levels [12, 24] cells per bay: 140 of
  288 strip comparisons outside tolerance, largest relative change
  mu_kip_in_per_ft 0.1020, vu_kip_per_ft 0.1438)`. Only the 12- and
  24-cell levels fit 45,000 shells on 36 bays (48 cells needs 82,944, 60
  needs 129,600), and that pair does not agree to 5%. This is the declared
  budget producing a legitimate convergence rejection, reported with its
  numbers; nothing was coarsened or passed.
- **case_0062, 3x2x4, 14x15-ft bays, sdc_e** (the plan's smallest):
  designed in about 5 minutes (5 section iterations, 33 MB `design.json`),
  28x28 columns, 14x26 beams, slab mats #4. All four levels fit
  (12/24/48/60 cells per bay, resolved for 168x180-in bays and the 14-in
  beam); the 12-to-24 pair failed (14 of 48 strips, up to 8.7% moment and
  14.2% shear), the 24-to-48 pair failed (12 of 48, 3.1% and 7.0%) and the
  final 48-to-60 pair passed (1.5% and 0.6%), the same pattern the fixed
  candidate showed. The slab actions are `verified` under the PROBE
  assertions, reinforcement is selected, no qualification check fails and
  one stays open, `demands.torsional_irregularity` (the provisional
  line-strength model: `Verify_Designs.probe_config` does not assert
  `story_strength_model_verified`, so a PROBE run reports that item open by
  construction; unrelated to this repair). The summary step of the same
  launch then reported the case as an untrusted cached design because this
  patch's error-message edit landed in `Design_Driver.py` while the case
  was designing — the integrity check working as intended, and the reason a
  batch must run from a committed tree.

Over the whole 150-case plan at the 45,000-shell budget and a 14-in beam,
71 cases resolve all four levels (final pair 48/60, the benchmark's), 34
three (24/48 — the pair the fixed candidate failed at 7.3% shear) and 45
only two (12/24). So under the current budget roughly half of the plan
would be screened at a pair coarser than the one that was verified, and
the 6x6 result above suggests many of those will be refused. That is the
truth of the current recipe and budget, not a defect in the gate; lifting
`MAX_EXPLICIT_SHELLS` (82,944 shells covers 48/bay on every floor; 129,600
covers 60/bay) with the solve time and memory that implies, or changing
the recipe, is an engineering and runtime decision to make before the
batch is described as ready to repeat.
