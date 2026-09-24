# Slab refinement integration and gravity sensitivity — 2026-09-24

Implemented and checked in `C:\Users\andro\Documents\Codex\2026-09-10\structural-research-analysis\work\desktop_continuation_20260924`, branch
`continuation/slab-refinement-20260924`, based on desktop commit
`3fec8de60d61a7bf21efe94a241e0d7a96c67f95`.
The original laptop checkout and SSD handoff were preserved. All 187 transferred
files (64,276,319 bytes) passed SHA256 verification before restoration.

## Completed work

- Integrated explicit rectangular meshes into the existing floor solver, including
  cell areas, Gauss coordinates, supports, beam connectivity and physical load fractions.
  Full coordinates, solver and mesh hash accompany each result. Explicit requests
  have a declared shell budget capped at 45,000; uniform defaults retain 8,192.
- Added a bounded refinement policy to `FloorAnalysisConfig.slab_refinement` and
  wired it into the slab reinforcement step of `Design_Driver`. The policy supplies
  nested meshes, separate moment/shear tolerances and a stated tolerance basis.
  Each geometry needs its own explicit mesh plan; these coordinates are not a
  validated automatic mesh generator for all dataset topologies.
- Retained every completed level, unsuccessful comparison and attempted-case failure.
  A failed fine solve cannot qualify earlier coarse results. Final comparisons cover
  every panel, axis and face. The last pair must satisfy the declared tolerances.
- Required numerical refinement before generated demands can be marked verified.
  Qualification and reinforcement selection recompute the comparison and check its
  identity against the returned demands. Cached flags cannot substitute for evidence.
  Physical signatures now include complete load cases, including load magnitudes.
- Updated the integration fixture to preserve rejected inputs and exercise the actual
  refinement path. Kept the initial failed test report rather than erasing it.

## Actual fixed-candidate reproduction

The saved design's decompressed SHA256 remains
`f49291440328b9097efd44345246d11a420365a0bbd1dea8c810276890c43871`.
Six native analyses separately solve 1.4D and 1.2D+1.6L on each of the transferred
graded 24, 48 and 60 meshes. Unlike the earlier postprocessing, both load cases
were actually solved at every level. Runtime: 297.5 seconds.

| Mesh comparison | Maximum moment change | Maximum shear change | Failed comparisons |
|---|---:|---:|---:|
| Graded 24 → 48 | 2.818% | 7.269% | 24 / 96 |
| Graded 48 → 60 | 1.415% | 0.550% | 0 / 96 |

The final all-strip screen passes its inherited 5% diagnostic criterion. This
remains bounded support refinement, not an ACI acceptance limit or proof of global
convergence. Every native solution reproduces the saved independent assembly;
the largest raw-resultant difference normalized by its field peak is
3.67e-11. Force and first-moment balance pass for both the
floor supports and exported beam/column transfer. Raw results, attempts and hashes
are in `../slab_refinement_integration_20260924/`.

## Gravity compatibility: one suspected cause ruled out

Reproduced both saved bare-frame gravity cases exactly, then replaced only the
column PDelta transformation with Linear in an isolated diagnostic process.
The saved transfer, rigid diaphragms, gross sections and all applied loads stayed
fixed. Compared both ends of all 126 columns against the unchanged coupled
10/bay solutions, retaining all 252 signed comparisons per case.

| Case | Largest original gap / governing demand | Largest matched-Linear gap / governing demand | Largest transformation effect / governing demand |
|---|---:|---:|---:|
| Full live | 18.550% | 18.550% | 2.37e-13% |
| Alternate-X live | 16.754% | 16.754% | 0.0179% |

Thus the PDelta/Linear mismatch does not explain the main discrepancy in these
two gravity cases. This does not establish its importance under seismic loading.
The largest full-live discrepancy remains sixth-story column 122, lower end:
319.014 versus 188.171 kip-in in local Mz. The frame is conservative at that end;
the earlier nonconservative locations remain unresolved. These are differences
normalized by saved demands, not revised P-M capacity checks. Full results are
in `../gravity_transform_sensitivity_20260924/`.

## Validation and remaining work

**142 targeted tests pass**, covering slab recovery, reinforcement,
assertions, anchorage, loads, floor transfer, coupled diagnostics, the new mesh
workflow and design-loop integration. Native comparisons cover rectangular
nonuniform grids, asymmetric patterns, actual load positions, preserved existing
domains, and uniform-grid equivalence. Failure checks cover missing/duplicate/
over-budget meshes, failed fine solves, unstable strips and stale cached flags.

The final consumer guard was added after the six native solves. It does not
change the solver or refinement algorithms; the final tests and saved-evidence
recheck cover it. `final_validation.json` records both that change and final code
hashes; the original numerical-run provenance remains untouched.

The next physical task is a controlled all-story floor/frame stiffness and
kinematic comparison, followed by complete composite section force recovery.
The current independent-floor transfer and coupled shell/web model use different
support and membrane treatments. Do not close compatibility from vertical base
reactions or apply a blanket moment factor. A coupled design route also needs
membrane-plus-bending slab reinforcement, not the current zero-membrane routine.
Then select verified slab steel, rerun beam/slab capacity checks and resume joint
design/SCWB. Experimental IMK energy calibration remains outstanding.

No slab layout was selected for the fixed candidate, no full redesign or seismic
analysis was run, and no production/qualification assertion was granted.

## Reproduction

From the restored checkout's `RC Structure` directory:

```powershell
python -B tools/review_slab_refinement.py --output <new-output-folder>
python -B tools/review_gravity_transform_sensitivity.py --output <another-new-output-folder>
```

The scripts default to the restored four handoff output folders. Both refuse an
existing output folder and preserve the source design identity. Use the OpPy
environment with OpenSeesPy and NumPy. See `continuation.patch` and
`continuation_code.zip` for portable source changes; no commit or push was made.
