# Research handoff: IMK member transition and 3D joint review

Updated 2026-09-23. Branch: `dv150-v11-rule-and-search-corrections`.
Implementation started from `b90f57c5d0a0c2b08eed6f12ada2850ff3c7b1d9`.

## Start here

The PeakOriented member diagnostic is complete. A finite 3D Pinching joint
prototype is implemented and mechanically tested, but **is not installed in
the building model**. The new review found a material 3D compatibility
assumption and an applicability question for code-conforming SMRFs. Read
[the joint review](<RC Structure/Model/JOINT_PANEL_REVIEW_20260923.md>) before
fitting parameters or integrating joints. Do not restart the completed
member-only comparison or launch a population rerun to answer those questions.

**90 bounded tests passed** on OpenSees 3.8.0. Tests and synthetic plots can be
reproduced without the SSD. The user is taking the SSD and will handle this
commit/push manually. At handoff preparation, no commit or push had been made
by Codex; original HEAD and Git index were unchanged. The original `.git`
directory has sandbox deny ACLs despite approved path access. No ACLs were
altered. Run the Git commands below from the user's normal terminal.

## User decisions and research context

- Research: randomly generated, code-compliant 3D RC SMRFs; calibrate seismic
  loading for an inelastic response domain, then generate data and train the
  GNN+LSTM surrogate. Study adaptive sampling before adopting it. A GNN+FNO
  comparison follows the first completed model. This handoff concerns the
  structural model, not surrogate training.
- The user and Codex are doing this scripting. Earlier Claude assignments do
  not authorize sending messages or assigning new work to Claude now.
- Intended constitutive assignment: IMKPeakOriented for beam/column flexural
  end springs; IMKPinching is being investigated for joint behavior separately.
- Keep the modern OpenSees equation `E_ref = Lamda * Fy`. For rotational
  materials Fy is moment. Do not multiply command-level Lamda by yield rotation.
- `Lamda_A = 10`, `c_A = 1` were explicitly approved for diagnostics only.
  No experimental joint calibration, kappa values, or production acceptance
  has been approved. Synthetic fixture kappaF=kappaD=0.5 is not a joint default.
- Learn from primary sources when uncertain. Visual agreement of a hysteresis
  loop is not evidence of experimental accuracy or correct deformation accounting.
- The user requested automatic pushing of future project changes; the current
  push is being handled by the user after a Git metadata permission problem.
  Preserve unrelated edits; do not force-push or include raw datasets.

## What is implemented

| Area | Current state |
|---|---|
| Modern material adapter | `Model/IMK_Materials.py`: explicit Bilin/PeakOriented/Pinching signatures, validation, units, exact installed commands, calibration provenance, parameter hashes. |
| Member integration | `Model/IMK_Hinges.py`: Bilin or PeakOriented flexural springs, preserved beam-end strength asymmetry; installed material metadata in registry. Joint Pinching cannot silently replace a member material. |
| Selection | `Structure_Parameters.IMK_MATERIAL_TYPE` still defaults to **IMKBilin**. `Design/Pilot_Ground_Motion_Diagnostic.py --member-material IMKPeakOriented` selects the provisional diagnostic profile. The global production model has not been switched. |
| Hysteresis analysis | Material-aware v3 diagnostic; modern reloading is not misclassified using the Bilin plastic-increment proxy. Accumulated plastic rotation is unavailable for Peak/Pinching; virgin-envelope exceedance can miss degraded reversal yielding. Work/storage energy fields remain estimates. |
| Output compatibility | Cyclic parameter identity and energy convention are included in global output identity; mismatched settings cannot silently reuse output. |
| Joint prototype | `Model/Joint_Panel.py`: two distinct coincident cores, six finite faces, small-rotation rigid arms, Pinching about global X/Y, explicit shear-only inputs, zero added mass. No frame-builder integration. |
| Verification | Material, member, joint, export/regression, and recorder tests; portable `tests/run_imk_verification.py`. |

See [material notes](<RC Structure/Model/IMK_MATERIALS.md>) and
[joint equations and assumptions](<RC Structure/Model/JOINT_PANEL.md>).

`Design/SMRF_Qualification.py` still has `GENERATION_RELEASE_READY = False`.
The production path still requires accepted designs and release readiness.
Neither the material tests nor the fixed-design diagnostic lifts these gates.
Earlier M1/design rulings are not re-adjudicated by this handoff.

## Results already in hand

The paired member-only pilot used fixed `case_0074`, RSN864 Landers/Joshua Tree
JOS000/JOS090, PEER result ID 11 in `peer_strong_63`, scale 1.0. Both laws
completed 2,200 steps over 44 seconds. All 1,272 spring stiffnesses, strengths
and rotation limits matched, as did initial periods and gravity. No joint
springs were present.

| Peak story drift | Bilin | PeakOriented |
|---|---:|---:|
| X | 0.6753% | 0.7983% |
| Y | 0.9702% | 0.8630% |
| Simultaneous resultant | 1.0123% | 1.0227% |

Peak run: 940.93 s solve, 124 successful KrylovNewton recoveries, zero failed
recovery, no subdivision or truncation. The comparison independently checked
saved response arrays and drift. Historical processed ground-motion byte
hashes were unavailable; IDs, time steps, lengths, scale and PGA matched.
Coverage is one design and one pair; this is not experimental validation.

Portable evidence: [comparison](<RC Structure/docs/handoffs/2026-09-23-imk-joints/evidence/paired_member_pilot/comparison.md>),
[comparison JSON](<RC Structure/docs/handoffs/2026-09-23-imk-joints/evidence/paired_member_pilot/comparison.json>),
and [paired loops](<RC Structure/docs/handoffs/2026-09-23-imk-joints/evidence/paired_member_pilot/paired_hinge_loops.png>).

The synthetic joint demonstration completed 1,880 simultaneous, nonproportional
biaxial loading steps without failure. Face shear and spring rotations agree
to roundoff. A PeakOriented face hinge plus clear cantilever matches analytical
tip compliance. A new transverse-beam test quantifies the prototype relation
`q = M/(Kj + sum(GJ/L))`; it does not validate that kinematic assumption against
a real joint. See the review's numerical table and disposition.

Current evidence: [test log](<RC Structure/docs/handoffs/2026-09-23-imk-joints/evidence/current_verification/tests.log>),
[verification JSON](<RC Structure/docs/handoffs/2026-09-23-imk-joints/evidence/current_verification/verification.json>),
and [joint plot](<RC Structure/docs/handoffs/2026-09-23-imk-joints/evidence/current_verification/joint_pinching_3d.png>).
The JSON includes exact fixture commands, source hashes, response columns,
constraint sensitivity, work checks, coverage and exclusions. The NPZ in the
same directory contains the synthetic histories.

## Next work, in order

1. Read `Model/JOINT_PANEL_REVIEW_20260923.md`. Resolve the intended role of
   degrading joints in the verified SMRF domain before choosing an empirical
   degradation law. The primary-source distinction is documented there.
2. Establish suitable 3D joint kinematics. The shared beam core couples
   perpendicular-member torsion into panel rotation. Compare physical face
   boundary conditions and work-conjugate coordinates, not arbitrary internal
   core restraints. The installed `Joint3D` constructor failed the isolated
   capability probe; the exact command and warning are in the evidence folder.
3. Extract applicable specimen data from the screened primary sources.
   Kurose et al. report 88-2 is a candidate with slabs and bidirectional loading,
   but it is scanned and its specimen-level applicability has not been checked.
   Do not borrow degrading parameters from deficient exterior joints and label
   them SMRF calibration. Source URLs and SHA-256 hashes are in
   `evidence/research_source_manifest.json`.
4. Resolve deformation partition. Existing member calibration still has
   `BOND_SLIP_INDICATOR = 1`. The joint prototype is shear-only. Adding a combined
   shear/slip fit can double-count slip. Setting that indicator to zero alone
   does not establish validated flexure-only member behavior.
5. Integrate a selected, justified joint model at member faces. Reconcile clear
   spans, stiffness/calibration lengths, physical member IDs, gravity transfer,
   self-weight, masses, diaphragm constraints and damping. Avoid tying released
   hinge rotations through a diaphragm. Check both transverse beam sets.
6. Run a single matched fixed-design diagnostic after those choices are concrete.
   Reconnect the SSD for case data. Use a fresh output directory, preserve raw
   records, and keep diagnostics outside dataset qualification. Population
   generation and surrogate training are later work.

## Reproduce the current checks without the SSD

From `RC Structure`, in the existing OpenSees environment:

```powershell
python -B tests/run_imk_verification.py --output outputs/imk_verification_resume
```

This runs 90 tests and writes JSON, NPZ, a PNG, and a test log. The two forced
NTHA failures printed by the recorder fixture are intentional recovery tests;
the final unittest result must still have zero failures/errors. No generation,
training, SSD read, or ground-motion pilot is started by this command.

For just the 11 joint mechanics tests:

```powershell
python -B -m unittest discover -s tests -p test_joint_panel.py -v
```

Original interpreter: `C:\Users\andro\anaconda3\envs\OpPy\python.exe`,
OpenSees 3.8.0. Check the environment on the receiving machine rather than
assuming that absolute path exists. Raw byte hashes can differ across checkout
line endings; the verification reports record the exact bytes used in each run.
Design request identity uses BOM-stripped, newline-normalized UTF-8 text hashes;
do not compare those directly with raw file hashes.

## Data that is not carried by this Git handoff

- Fixed record: `D:\StructGNN_outputs\dv150_v10\case_0074\design.json`.
  SHA-256: `87a336d0d0d5ee2add6131d181d3775530f343cd2ed05ddd3bc98e9c21415410`.
- Bilin raw pilot: `D:\StructGNN_outputs\diag_gm_pilot_20260921\run2\case_0074\peer_11_scale_1`.
- Peak raw pilot: `C:\Users\andro\Documents\Codex\2026-09-17\heyyyyy\outputs\imk_model_change_20260922\peak_pilot\case_0074\peer_11_scale_1`.
- Research PDFs/text: `C:\Users\andro\Documents\Codex\2026-09-17\heyyyyy\outputs\joint_review_20260923\sources`.
- Ground-motion raw/processed files are Git-ignored; verify their availability
  before any new pilot on a different machine.

Raw pilot arrays, the design population, ground-motion files and downloaded
papers are not committed. The smaller derived comparison and fixture evidence
are included. No SSD access is needed to read this handoff.

## Commit, push, and resume

From the original repository root in the user's PowerShell terminal:

```powershell
git add -- @(Get-Content 'RC Structure/docs/handoffs/2026-09-23-imk-joints/files_to_stage.txt')
git diff --cached --stat
git commit -m "Add modern IMK diagnostics and reviewed 3D joint prototype"
git push origin dv150-v11-rule-and-search-corrections
```

The staging manifest names only this work and its evidence. It excludes the
pre-existing untracked `grep.exe.stackdump`, caches, raw runs and downloaded
papers. Review any independently staged changes before committing.

On the receiving checkout, switch to `dv150-v11-rule-and-search-corrections`,
pull with `git pull --ff-only`, and read this file. A useful continuation prompt:

> Read HANDOFF_IMK_2026-09-23.md and the joint review it links. Continue with the
> joint applicability/3D compatibility decision before experimental parameter
> fitting or frame integration. Preserve the completed PeakOriented comparison
> and modern energy convention. The user wants primary-source reasoning when
> uncertain. Do not launch generation or a ground-motion run as setup.

This root handoff and `RC Structure/docs/handoffs/2026-09-23-imk-joints/` are
removable after their contents are no longer needed. Removing them does not
remove the implementation, permanent model notes, tests, or verification runner.
