# Repeatable analysis workflow

## 1. Inventory and freeze the analysis scope

Start with a device profile and a new output ID: UTC timestamp + device + short unique ID. Preserve the profile, code revision/dirty state, interpreter and library versions, source file hashes, and scan start/end times. A metadata preflight is a non-atomic scan. For a reproducible full analysis, restrict to stable completed files, record the exact source roster, and verify fingerprints before and after reading. Mark changed files for a later pass.

Use relative source paths beneath a named dataset root. Normalize Windows backslashes when reading manifests on another OS. Resolve paths against the proper dataset root and reject escape paths. Compare parameter-plan hashes across copies of the same experiment; different experiments should not be forced to have identical plans. Treat the generator repository's revision at analysis time separately from an unknown generation-time revision.

Build a case/run index containing dataset ID, plan hash, source device, case ID, run name, record IDs/components, scale factors, geometry/design identity, schema version, source files and hashes, requested/completed steps, status and timestamps. Include planned-but-absent cases as a separate coverage table. Multiple runs may belong to one case. Root-level manifests can be stale after merging copies.

## 2. QA/QC and triage

Reconcile the plan, root manifests, case-level manifests, per-run status, summaries, and compiled samples. Report each denominator. Check missing/empty/unreadable files, duplicate scientific identities, conflicting copies, NaN/Inf, missing required fields by schema, array shapes and metadata columns, valid node/edge indices, and masked or unrecorded values.

Check strictly increasing time, expected versus recorded duration, time-step/stride consistency, matching channels, recorded-step indexes for hinge/force histories, and premature termination. A recorded first time may be dt rather than zero. Do not demand equal row counts for histories recorded at different strides.

Store independent columns rather than one destructive pass/fail filter:

| Column | Example values |
|---|---|
| `solver_state` | complete, incomplete, failed, unknown |
| `export_state` | present, missing, corrupt, schema_unknown |
| `collapse_label` | true, false, unknown; with exporter source |
| `response_window` | full, truncated, censored, unknown |
| `qa_disposition` | usable, review, unusable_for_this_analysis |
| `reason_codes` | nonfinite, status_conflict, divergent_tail, unit_unknown, duplicate_conflict |

Create a review queue with case/run/source path, metric, observed value, rule and threshold origin, evidence, impact, and proposed next step. Exclude rows only from a named derived analysis with a recorded reason; retain them in the master index.

## 3. Validate structural metrics

Check source units and formulas before setting tolerances. Recompute a small, stratified selection first: typical, extreme, failed, collapse-labelled, and multiple schema versions. Expand the scope when mismatches warrant it.

Validate roof X/Y peaks and simultaneous resultant, interstory drift by story/direction, base shear, modal periods, residual response window, hinge damage/yield/capping measures, and DCR fields when available. Compare force components only in the same local/global frame. Verify relative versus absolute acceleration and gravity offsets. Check force/reaction balance only with matching load, inertia, damping, sign, and time conventions; do not require base shear to equal a static load during transient response.

Separate the existing summary drift envelope from simultaneous resultant drift (see DATA_CONTRACT.md). Separate elastic-reference periods from post-gravity tangent periods. Project collapse/inelastic gates are research-label rules, not automatically engineering acceptance limits. Record any threshold changes as new analysis rules with sensitivity results.

## 4. EDA and outliers

Summarize coverage, missingness, solver/label status, and response distributions by schema, record pair, scale/intensity, geometry, stories, seismic site, and device where relevant. Plot distributions/ECDFs, paired histories, response versus intensity, and geometry/record coverage. Use log axes only with explicit handling of zero/nonpositive data.

Use robust within-group statistics (IQR/MAD or model residuals) to nominate outliers. Report small groups, zero spread, and thresholds; do not choose an automatic global z-score cutoff as a scientific rejection rule. Assess whether extremes are plausible structural behavior, sampling effects, numerical artifacts, or mislabeled units.

## 5. Calibration and adaptive-sampling comparison

Confirm set membership from plans and provenance: pilot, calibration fitting observations, fixed baseline, calibrated generation, and each adaptive round. The laptop did not expose a confirmed adaptive dataset. Do not relabel the ground-motion expansion as adaptive sampling without evidence.

Compare input support (geometry, record/event, intensity, spectral/period features) before outcome distributions. Match geometry and record identities where applicable. Compare achieved versus target drift bands, scale-factor clipping, solver completion, physical-window length, collapse labels, coverage of rare responses, and fit residuals on held-out groups. Preserve censoring; a truncated peak is not a fully observed peak. Include changed code/schema/selection policies as potential confounders. Use paired differences or cluster/group uncertainty when observations share geometry, record, or event.

For adaptive rounds, retain round ID, parent dataset/model, acquisition score/reason if available, seed, candidate pool, rejected candidates, and fixed evaluation set. Never imply uniform population performance from a deliberately enriched adaptive set without justified weighting. Show raw and comparable-support summaries separately.

## 6. Deliver clean outputs

Use `outputs/<analysis_id>/` with the following products as applicable:

```text
report.md                    findings, coverage, limitations, next actions
provenance.json              inputs, hashes, definitions, config, versions
data_dictionary.md           units, formulas, nullable fields, source mappings
tables/run_index.csv         full indexed roster with QA and label columns
tables/review_queue.csv      exact cases/runs and evidence to investigate
tables/analysis_ready.csv    declared eligibility filter, no raw-data edits
tables/set_comparison.csv    denominators and comparison definitions
figures/                    labelled PNG/SVG/PDF figures
```

Use Parquet in addition to CSV when useful and available. Preserve full numeric precision in machine-readable tables and round only presentation tables. Figure labels must state units, cohort, metric definition, and censoring/filter rules. Keep scripts reusable in `scripts/`; use `work/` for temporary calculations. Transfer completed output folders to the laptop for review; keep the authoritative raw data on the desktop and carry stable dataset copies on the SSD as described in SSD_WORKFLOW.md.

## Requests to reuse after setup

- “Audit the configured desktop datasets. Reconcile manifests against runs, check schemas/status, and produce a reason-coded review queue before filtering anything.”
- “Investigate the most extreme drift, roof displacement, and hinge-damage runs. Compare summaries with synchronized histories and separate plausible collapse from divergence.”
- “Compare the pilot/calibration sets with the confirmed adaptive rounds. Show input coverage, target-versus-achieved drift, censoring, and changes in response distributions.”
- “Prepare clean analysis tables and figures from the approved eligibility rules, preserving provenance and every excluded run's reason.”
