# Observed research data contract

Inspected on the laptop on 2026-09-10. Revalidate on the desktop before analysis. Source repository at inspection: `C:/Users/andro/StructGNN`, HEAD `0442f20765356aaab42a0f844538f72511ccb835`. The saved Codex project is the nested `RC Structure` folder, but Git resolves to the parent repository.

## Input hierarchy

```text
RC Structure/
  Structure_Parameters.py
  Ground_Motion_Main.py
  Data_Generation/
    Generate_Parameterized_Dataset.py
    Hybrid_Exporter.py
    Calibrate_Intensity.py
    Analyze_Parameter_Space.py
    Check_Dataset_DCR.py
    MULTI_DEVICE_SETUP.md
  Ground_Motions/metadata/record_manifest.csv
  outputs/<dataset>/
    parameter_plan.json + parameter_plan.csv
    parameterized_manifest.json + parameterized_manifest.csv
    generation_state.json + case_results.json
    cases/<case_id>/
      design.json                         newer cases; not universal
      controller.log
      ntha/<run_name>/
        status.json + summary.json
        global_parameters.json
        record_summary_x.json + record_summary_y.json
        time_history.csv + story_drift_peaks.csv
        modal_results*.json + modal_diagnostics.json
        nodes.csv + edges.csv + elements.csv
        response_arrays.npz               newer examples
        hinge_backbone.csv                newer examples
      dataset/<run_name>/
        hybrid_metadata.json + hybrid_sample.npz
```

Use JSON plans for structured `runs`; a CSV cell in the pilot manifest contained a Python-like representation of a list. Never evaluate a CSV cell as code. Older root manifests have one sample path per row; newer ones summarize several planned/completed runs per case. Traverse the declared case runs and case manifests.

## Schema and metrics

Two sampled older datasets used `hybrid_gnn_lstm_v2`; sampled pilots used `hybrid_gnn_lstm_v3`. This is sampled evidence, not a complete schema census.

Common NPZ keys include `x`, `edge_index`, `edge_attr`, `time_seconds`, `ground_motion`, `response_time_history`, `story_drift_peaks`, `global_features`, `record_features`, and `target_peak`. V3 adds such fields as `ground_motion_present`, `damage_metrics`, floor/hinge histories, force histories and recorded-step arrays. Read the metadata's column names. The sampled v3 force/hinge histories had 750 rows versus 6000 main time steps, with separate step indexes. The older sampled NTHA folders lacked `response_arrays.npz`; their absence is not automatically corruption for v2.

| Metric | Source / interpretation |
|---|---|
| Roof displacement | `summary.max_abs_roof_disp_x_in/y_in`; `time_history.csv` X/Y and resultant, in inches |
| Summary resultant drift | `summary.max_story_drift_resultant.peak_drift_resultant_ratio`; Ground_Motion_Main.py combines separate directional drift peaks per story using sqrt(dx²+dy²). These peaks need not occur at the same time. |
| Simultaneous resultant drift | Newer exporter `damage.peak_interstory_drift_ratio`; derived from synchronized X/Y story histories, with the exporter's physical-window handling. It is a different metric from the summary envelope above. |
| Residual drift | Exporter averages the maximum story resultant across the final 2% of retained steps; this is its specific tail statistic, not universally a post-shaking residual. A truncated run needs a censoring flag. |
| Roof drift | Exporter uses peak resultant roof displacement divided by number of floors × story height, on its retained window. Verify variable story heights before reusing that simplification. |
| Base shear | `time_history.csv` has signed X/Y values in kip; exporter `peak_base_shear_kip` takes the largest absolute array component, not a resultant peak. |
| Periods | Keep `elastic_reference_period_mode_1_sec` separate from `post_gravity_tangent_period_mode_1_sec` and higher modes. |
| DCR | Inspect matching design-check stage, force convention, demand/capacity definitions, and available columns before aggregating. Do not mix gravity, pushover, and NTHA DCRs. |

Units observed include inches/feet, kip, ksi, seconds, dimensionless drift ratios and separately named percent fields. Acceleration units and response reference frames must be checked in the exporter and record metadata. Use `np.load(..., allow_pickle=False)` for NPZ. Legacy `Data/Static_Linear_Analysis` contains a different graph.pt workflow; keep it out of RC transient comparisons unless explicitly requested.

## Solver, damage and calibration semantics

The current generator's successful status check requires `npts_requested > 0`, matching completed steps, and `failed == false`. Its complete-run check also requires a compiled sample and accepts an exporter collapse label. Thus “completed” at case level does not prove a full-duration solve. A process return code alone is insufficient.

The currently inspected exporter has research gates: interstory drift 0.010, roof drift 0.00775, hinge damage 0.25, yielded fraction 0.10; its inelastic label requires all four. Its collapse label uses drift at least 0.10 OR failed status with drift at least 0.010. Physical drift and damage-ratio ceilings are 0.20 and 20.0. Preserve these as versioned implementation facts, not universal acceptance limits or independently verified collapse truth. The code's physical-window selection and hinge filtering require scrutiny when a run diverges; do not assume every retained point is physically valid.

The calibration artifact `outputs/intensity_calibration_pilot30.json` reports schema `rc_intensity_calibration_v1`, 88 fitting observations, and fitted R² about 0.928. That is a stored in-sample fit statistic, not independent validation. Current calibration collection uses the summary drift envelope, skips some failed/implausible runs, and fits a log-linear model. Inspect the exact code and source roster before claiming a formal censoring model. Calibrated-generation targets and synchronized drift QA must not be mixed without acknowledging their different definitions.

The surrogate sibling `RC Hybrid Surrogate Model` documents grouping by ordered X/Y record pair and training-only normalization. Check actual saved splits and normalization for a given run. Reversed pairs and shared events merit an additional leakage audit. The inspected laptop sibling did not expose a local training `outputs` folder at top level; discover desktop checkpoints explicitly.

## Multi-device context

The repository's MULTI_DEVICE_SETUP.md describes original partitions 1–1250, 1251–1899, and 1900–2500, and separate expansion IDs 2501–3800. For expansion scheduling, positions 1–1300 map to IDs 2501–3800; do not confuse scheduler positions with case IDs. These are historical generation instructions, not current assignments inferred for this laptop.

That guide mentions `RC Structure/environment-generation.yml`, which was not found during this inspection. Use a verified interpreter or a reviewed environment specification instead of assuming that command works. The existing root `OpPy_environment.yml` is platform-specific and includes more than ordinary analysis needs.
