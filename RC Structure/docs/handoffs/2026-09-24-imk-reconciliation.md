# Desktop IMK edits reconciled onto the integration branch — September 24

The desktop `main` checkout (`C:\Users\andro\StructGNN`, at 91afdd40) carried
uncommitted IMK edits made before 3fec8de6 rewrote `Model/IMK_Hinges` around
the `Model/IMK_Materials` adapter. They are reconciled here by hand, on
`continuation/slab-refinement-20260924`, selectively.

## Taken as written

- `Structure_Parameters.py`: `IMK_MATERIAL_TYPE = "IMKPeakOriented"` (member
  flexure; IMKPinching stays the joint material through its own adapter and
  topology), the new `IMK_DETERIORATION_MODE = "haselton_2008"`, the
  formulation comment, and `NTHA_SCALE_FACTOR = 2.25` (the requested total
  amplitude multiplier; a run setting, flagged below).
- `Model/IMK_Calibration.py`: `deterioration_for_member` — PEER 2007/03
  Eq. 3.20, lambda = 170.7 (0.27)^nu (0.10)^(s/d), translated to the OpenSees
  convention as Lamda = lambda x theta_y,member for the strength modes S and
  C, with A and K suppressed by a large finite capacity (1e12); the Eq.
  3.10/3.16 references; the fixed backbone's negative-side rotations; and the
  column ultimate rotation now using the column yield rotation (it used the
  beam's).
- `Model/IMK_Hinges.py`: `props`/`family`/`axial_kip` keywords on the
  stiffness functions and `_create_end_hinge` passing the member's own
  properties, so an edge beam's spring uses its line's T-section I and a
  column's its axial state — the inconsistency the hysteresis diagnostic had
  recorded as "not corrected"; one `equalDOF` call per hinge; `ke_y/ke_z`
  recorded in the registry.
- `Design/Config.py`, `Redesign.py`, `tests/test_beam_symmetric_design.py`:
  `RebarConfig.beam_symmetric = True` (equal beam top/bottom bar counts by
  default; `False` reproduces the independent-layer search) with its
  candidate, convergence and fallback logic. **Flag for review:** this
  changes the SMRF steel search (`Design_Driver` uses `Redesign.redesign_steel`)
  and therefore every design identity, and the laptop floor/frame review
  states that the composite-section work "has not established that
  reinforcement should be made symmetric". It is the desktop's stated
  policy, carried as authored, not endorsed by the floor/frame evidence.
- Test updates: the IMKBilin signature pin patches the material type
  explicitly; the slab-anchorage test passes the beam family.

## Adapted, not copied

The desktop rewrote `_define_imk_peak_material` for an older `IMK_Hinges`;
on this branch the material goes through `IMK_Materials.define_rotational_imk`,
whose rule is that Lamda arrives as a command-level value ("never multiplied
by yield rotation here; translate upstream"). The desktop's translation is
exactly that upstream step, so: the backbone carries
`lambda_opensees_by_mode_rad` and `_define_imk_peak_material` builds
`CyclicParameters` from it when present (else the `IMK_LAMBDA_*` constants
pass unchanged), recording `deterioration_source`, a calibration id
`haselton_2008_eq3_20_nominal_member_theta_y_v1` and the translation inputs
in the material provenance. The status stays
`provisional_not_experimentally_calibrated`: a column regression extended to
beams on nominal yield rotations is a research calibration. The branch's
explicit experimental energy-profile path (`IMK_ENERGY_MAPPING_MODE`) is
untouched and, when a reviewed profile exists, still takes precedence. The
desktop's stricter check that the ultimate rotation clears the capping
rotation (yield + plastic) is in the adapter.

Consumers that recomputed the spring stiffness without the member's
properties now read the installed value: `Pushover_Diagnostic.hinge_inventory`
(which the hysteresis diagnostic and the cyclic fixture use) and
`Mechanism_Checks` (spring yield rotation and Ke from the registry).
`Record_Hinge_Hysteresis.backbone_reference` keeps its documented
zero-axial fallback.

Two desktop tests that were untracked there and first surfaced in the
desktop's stash (`desktop-imk-edits-before-9a587d6c`) are carried too:
`tests/test_haselton_deterioration_modes.py` (translated Lamda in the
command slots of both materials, the direct mode passing the constants,
strength loss with a constant unloading slope on a cycled material) and
`tests/test_imk_rc_corrections.py` (the PeakOriented signature with a
negative-side backbone, the theta_y factor and its independence from the
spring stiffness factor, the installed edge-beam spring against its own
line's I under Transformation constraints, cyclic strength loss with the
legacy material still available). They exercise the desktop's uniform
`lambda_opensees_rad` backbone form and a deterioration-only backbone, so
the adapter accepts both: a uniform value applies to every mode, and a
backbone without rotations falls back to the global constants.

## Not taken

`.idea/StructGNN.iml`; the untracked `OpPy_environment.yml` at the repository
root (an environment export, not code).

## Tests

`tests/test_imk_deterioration_mapping.py` (new): the Eq. 3.20 arithmetic and
theta_y factor, beam extrapolation and axial-ratio clamp flags, the command
slots the translated values land in for IMKPeakOriented and IMKBilin, the
constants passing unchanged without a translation, and the capping-rotation
guard. `tests/test_hinge_hysteresis_fixture.py` pins `IMKBilin` and
`IMK_DETERIORATION_MODE = "direct"`: it measures installed strengths with
the diagnostic's exact Bilin tangent accounting, which the diagnostic does
not provide for PeakOriented (virgin-envelope exceedance only, no
accumulated plastic rotation), and the Haselton capacities already take a
few tenths of a percent off a spring that first yields after a reversal
(0.4% on this fixture). Full suite: see the commit.

## Open after this reconciliation

- A PeakOriented cyclic measurement fixture (envelope-based yield detection,
  reversal yielding) does not exist; the production member material is
  covered by the synthetic energy-mapping and asymmetry tests only.
- `Model/Joint_Panel.py` (IMKPinching) is still not part of the production
  frame; wiring it in requires partitioning bond slip out of the member
  springs (Haselton a_sl = 1 already counts it).
- `IMK_Hinges.imk_hinge_thresholds` still reads the global `IMK_THETA_*`
  constants; `Mechanism_Checks` and the registry consumers override them
  with installed values, `Record_Hinge_Hysteresis.backbone_reference` does not.
- `NTHA_SCALE_FACTOR = 2.25` is now the production run setting; no NTHA was
  run here to confirm it is what the next batch wants.
- The 2026-09-23 bundle's uncommitted documents (`Model/IMK_MATERIALS.md`,
  `Model/JOINT_PANEL.md`, `JOINT_PANEL_REVIEW_20260923.md`,
  `HANDOFF_IMK_2026-09-23.md`, `joint_histories.npz`) are still absent;
  `tests/run_imk_verification.py` fails on the missing `JOINT_PANEL.md`.
- Every saved design predates these `Model/` changes and is invalidated by
  the request identity, by design.
