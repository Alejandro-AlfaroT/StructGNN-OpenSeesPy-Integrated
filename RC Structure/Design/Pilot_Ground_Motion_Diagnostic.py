"""Fixed-design ground-motion diagnostic with full-rate hinge loops (one case, one record pair, one worker).

    python Design/Pilot_Ground_Motion_Diagnostic.py --root D:/StructGNN_outputs/dv150_v10 --case case_0074 \
        --result-id 11 --set-name peer_strong_63 --scale 1.0 --output-root <isolated folder>
        [--damping-ratio 0.05 --rayleigh-mode-i 0 --rayleigh-mode-j 2 --dt-factor 1.0 --x-only --plot-limit 24]

What it does, in order: read the saved record and hash it; apply the
record's OWN geometry (record["geometry"]) and seismic site
(record["seismic"], checked value by value after applying the named
entry) -- no generation plan is consulted; install the saved design
explicitly (Design_Driver.apply_design -- no cache loading, no
source-identity check, no redesign); build the current IMK model with the
production NTHA conventions (Ground_Motion_Main.build_gravity_modal_state:
elastic reference modal analysis, D + L gravity at factor 1.0 through the
saved slab transfer, then the post-gravity modal diagnostic); measure the
gravity state; verify the installed sections, reinforcement, slab and
hinge registry against the record (per member, end and sign for the beam
strengths); attach a full-rate material recorder to every hinge spring in
the domain; load the record pair with its component ids; run
Analysis.NTHA.run_ntha with Rayleigh damping from the reference modes;
save the production outputs; read the recorders within the retained window;
evaluate every spring against its own installed backbone (path-aware and
virgin-backbone yield calls); justify a history stride against the full
rate; plot the yielded loops; hash the record again; write manifest.json.
A failed or empty solve still writes the manifest with the observed
failure and no fabricated response. Diagnostic evidence about the declared
model, kept apart from design qualification and training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
sys.path.insert(0, str(RC / "Data_Generation"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(RC.parent), timeout=20).stdout.strip() or None
    except Exception:                                   # noqa: BLE001 -- identity is reported, never required
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-id", type=int, required=True, help="PEER result_id pair in the record set")
    parser.add_argument("--set-name", default="peer_strong_63")
    parser.add_argument("--scale", type=float, default=1.0, help="scale factor applied to both components (the catalog's processed files are in in/s^2; the loader converts units)")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--damping-ratio", type=float, default=0.05)
    parser.add_argument("--rayleigh-mode-i", type=int, default=0)
    parser.add_argument("--rayleigh-mode-j", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument("--x-only", action="store_true")
    parser.add_argument("--plot-limit", type=int, default=24)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    import openseespy.opensees as ops
    import Structure_Parameters as sp
    import Geometry_Overrides as go
    from Design import Design_Driver as driver
    from Analysis.NTHA import run_ntha
    from Analysis import Hinge_Hysteresis_Diagnostic as hd
    from Loads.Ground_Motion import load_ground_motion_pair_by_result_id, summarize_record, significant_duration_5_95, find_manifest_row
    import Ground_Motion_Main as gm

    started = time.time()
    record_path = Path(args.root) / args.case / "design.json"
    sha_before = sha256_file(record_path)
    record = json.loads(record_path.read_bytes())
    out = Path(args.output_root) / args.case / f"peer_{args.result_id}_scale_{args.scale:g}{'_x_only' if args.x_only else ''}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output folder {out} is not empty; the pilot writes into a fresh folder only")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"diagnostic_version": hd.DIAGNOSTIC_VERSION, "label": args.label, "case_id": args.case,
                "record_path": str(record_path), "record_sha256_before": sha_before, "status": "started"}

    def write_manifest():
        manifest["record_sha256_after"] = sha256_file(record_path)
        manifest["record_unchanged"] = manifest["record_sha256_before"] == manifest["record_sha256_after"]
        manifest["elapsed_sec"] = time.time() - started
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    try:
        # --- geometry and site from the record itself ---------------------------------------------------
        geometry, seismic = record["geometry"], record["seismic"]
        overrides = {"NUM_BAY_X": int(geometry["num_bay_x"]), "NUM_BAY_Y": int(geometry["num_bay_y"]), "NUM_FLOOR": int(geometry["num_floor"]),
                     "STORY_H": float(geometry["story_h_in"]), "BAY_X": float(geometry["bay_x_in"]), "BAY_Y": float(geometry["bay_y_in"])}
        go.apply_geometry_overrides(overrides, variant_name=args.case, emit=False)
        sp.apply_seismic_site(seismic["site_label"])
        site_check = {k: {"record": seismic[k], "installed": getattr(sp, attr), "match": math.isclose(float(seismic[k]), float(getattr(sp, attr)), rel_tol=1e-12)}
                      for k, attr in (("sds", "ASCE_SDS"), ("sd1", "ASCE_SD1"), ("s1", "ASCE_S1"), ("r", "ASCE_R"))}
        if not all(v["match"] for v in site_check.values()):
            raise RuntimeError(f"the named site entry does not reproduce the record's seismic values: {site_check}")
        driver.apply_design(record)
        current_identity = driver.design_request_identity()
        record_identity = record.get("request_identity") or {}
        record_sources = record_identity.get("source_sha256") or {}
        current_sources = current_identity["source_sha256"]
        changed_sources = sorted(k for k in set(record_sources) | set(current_sources) if record_sources.get(k) != current_sources.get(k))
        manifest.update({
            "geometry_applied_from": "record['geometry']", "geometry": overrides, "site_applied_from": "record['seismic']['site_label'], values checked",
            "site_check": site_check,
            "identity": {"record_schema_version": record.get("schema_version"), "record_request_identity_sha256": record_identity.get("sha256"),
                         "record_qualification_accepted": (record.get("qualification") or {}).get("accepted"),
                         "record_design_source_files": len(record_sources), "current_source_files": len(current_sources),
                         "source_files_changed_since_design": changed_sources, "source_files_changed_count": len(changed_sources),
                         "current_source_sha256": current_sources, "git_head": git_head(),
                         "basis": "a fixed-design diagnostic under the CURRENT model and sources; the record was designed under its own "
                                  "request identity, whose source hashes are compared file by file above"}})

        # --- model, gravity, verification, recorders --------------------------------------------------------
        gravity, reference_modal, post_modal, reference_check, tangent_check = gm.build_gravity_modal_state()
        verification = hd.verify_installed_design(record)
        audit, inventory, hinges = hd.run_audit()
        gravity_measured = hd.measured_gravity_state(inventory)
        coverage = hd.attach_hinge_recorders(out / "recorders", hinges)
        valid = [m for m in reference_modal if m.get("valid")]
        print(f"[{args.case}] installed design verified: {verification['consistent']} ({len(verification['differences'])} differences); "
              f"hinges {len(hinges)} (domain {coverage['domain_hinge_count']}); T1 {valid[0]['period']:.4f} s, T2 {valid[1]['period']:.4f} s; "
              f"gravity reaction {gravity_measured['vertical_reaction_kip']:.2f} kip vs ledger {gravity_measured['expected_vertical_load_kip']:.2f} kip", flush=True)
        for d in verification["differences"]:
            print("   DIFFERENCE:", d, flush=True)
        manifest.update({"installed_design_verification": verification, "model_audit": audit,
                         "model": {"periods_elastic_reference_sec": [m["period"] for m in valid[:6]],
                                   "mode_1_roof_direction": {"x": valid[0]["roof_eigenvector"][0], "y": valid[0]["roof_eigenvector"][1]},
                                   "periods_post_gravity_tangent_sec": [m["period"] for m in post_modal if m.get("valid")][:3],
                                   "asce_reference_period_check": reference_check, "asce_post_gravity_period_check": tangent_check,
                                   "gravity": {"floor_factor": 1.0, "dead_factor": 1.0, "live_factor": 1.0, "self_weight_factor": 1.0,
                                               "load_model": sp.effective_gravity_load_model(), **gravity_measured,
                                               "basis": "Ground_Motion_Main.build_gravity_modal_state: apply_gravity_loads() at floor factor 1.0 (D + L), "
                                                        "member self-weight, then run_gravity_analysis; the production NTHA gravity state"},
                                   "damping": {"ratio": args.damping_ratio, "rayleigh_modes_0_based": [args.rayleigh_mode_i, args.rayleigh_mode_j],
                                               "basis": "mass + initial-stiffness proportional Rayleigh from the elastic reference modal periods (production NTHA convention)"},
                                   "integrator": "Newmark 0.5 / 0.25; Newton, then KrylovNewton, then 10x subdivided KrylovNewton recovery (Analysis.NTHA._try_step)",
                                   "dt_factor": args.dt_factor},
                         "hinges": {"coverage": coverage}})

        # --- records ------------------------------------------------------------------------------------------
        key, record_x, record_y = load_ground_motion_pair_by_result_id(args.result_id, set_name=args.set_name, scale_factor=args.scale)
        if args.x_only:
            record_y = None
        components = {}
        for direction, r in (("x", record_x), ("y", record_y)):
            if r is None:
                continue
            row = find_manifest_row(r.record_id)
            components[direction] = {**summarize_record(r), "component": row.get("component"), "component_angle_deg": row.get("component_angle_deg"),
                                     "event": row.get("event_name"), "year": row.get("event_year"), "station": row.get("station_name"),
                                     "rsn": row.get("station_id"), "magnitude": row.get("magnitude"), "rrup_km": row.get("rrup_km"),
                                     "vs30_m_per_s": row.get("vs30_m_per_s"), "mechanism": row.get("mechanism"),
                                     "raw_file": row.get("raw_file"), "processed_file": row.get("processed_file"),
                                     "processed_units": row.get("processed_units"), "raw_units": row.get("raw_units"),
                                     "significant_duration_5_95_sec": significant_duration_5_95(r),
                                     "excitation": f"UniformExcitation DOF {1 if direction == 'x' else 2}, Path series at the record dt, factor 1 on the "
                                                   f"scaled record converted to in/s^2"}
        manifest["records"] = {"pair_key": key, "set_name": args.set_name, "scale_factor": args.scale, "components": components,
                               "scale_basis": "initial scale set by the caller; Data_Generation.Calibrate_Intensity's drift model at the reference "
                                              "period is an estimate, not a promise of a drift band or of yielding"}
        print(f"[{args.case}] records {key}: X {record_x.record_id}" + (f", Y {record_y.record_id}" if record_y else " (x only)") +
              f"; scale {args.scale:g}; dt {record_x.dt_sec} s; npts {record_x.npts}; PGA {record_x.pga_g:.3f} g", flush=True)

        # --- solve --------------------------------------------------------------------------------------------
        t0 = time.time()
        results = None
        try:
            results = run_ntha(record_x, record_y=record_y, damping_ratio=args.damping_ratio, modal_results=reference_modal,
                               rayleigh_mode_i=args.rayleigh_mode_i, rayleigh_mode_j=args.rayleigh_mode_j, dt_factor=args.dt_factor,
                               log_path=out / "opensees_ntha.log")
        finally:
            solve_seconds = time.time() - t0
            hd.close_recorders()
        status = results["status"]
        time_history = results["time_history"]
        analysis_dt = results["record_summary_x"]["analysis_dt_sec"]
        scheduled_steps = results["record_summary_x"]["analysis_steps"]
        scheduled_duration = analysis_dt * scheduled_steps
        record_duration = record_x.duration_sec if record_y is None else max(record_x.duration_sec, record_y.duration_sec)
        last_time = float(time_history[-1]) if time_history else None
        manifest["model"]["damping"].update({"a0": results["rayleigh_a0"], "a1": results["rayleigh_a1"]})
        manifest["model"].update({"analysis_dt_sec": analysis_dt, "analysis_steps_scheduled": scheduled_steps})
        manifest["solver"] = {"completed_steps": status["completed_steps"], "requested_steps": status["npts_requested"], "failed": status["failed"],
                              "failed_step": status["failed_step"], "failed_time_sec": status["failed_time_sec"],
                              "convergence_summary": status["convergence_summary"], "solve_seconds": solve_seconds,
                              "retained_window_sec": [float(time_history[0]) if time_history else None, last_time],
                              "scheduled_duration_sec": scheduled_duration, "record_duration_sec_dt_times_npts_minus_1": record_duration,
                              "truncated": bool(status["failed"]) or status["completed_steps"] < scheduled_steps,
                              "truncation_basis": "completed steps below the scheduled dt x npts steps, or a failed step"}
        print(f"[{args.case}] NTHA: {status['completed_steps']}/{status['npts_requested']} steps, failed={status['failed']}, {solve_seconds:.0f} s", flush=True)
        file_counts = gm.save_ntha_outputs(out / "ntha", results, gravity, reference_modal) if time_history else {"note": "no completed step: production outputs not written"}
        ops.wipe()
        manifest["production_outputs"] = {"dir": str(out / "ntha"), **file_counts}

        # --- response ------------------------------------------------------------------------------------------
        if time_history:
            n = sp.NUM_FLOOR
            peak_x = max(results["peak_story_drift_x"]); peak_y = max(results["peak_story_drift_y"])
            drift_hist = results["story_drift_history"]
            resultant = max((math.hypot(row[k], row[n + k]) for row in drift_hist for k in range(n)), default=0.0)
            manifest["response"] = {
                "peak_story_drift_x": peak_x, "peak_story_drift_x_story": 1 + results["peak_story_drift_x"].index(peak_x),
                "peak_story_drift_y": peak_y, "peak_story_drift_y_story": 1 + results["peak_story_drift_y"].index(peak_y),
                "peak_story_drift_resultant_simultaneous": resultant,
                "peak_story_drift_definition": "max over completed steps and stories of |u_k - u_{k-1}| / h at the floor masters, per direction; the "
                                               "resultant is the simultaneous hypot of the two directions at the same step and story",
                "max_abs_roof_disp_in": {"x": max(abs(v) for v in results["roof_disp_x"]), "y": max(abs(v) for v in results["roof_disp_y"])},
                "peak_base_shear_kip": {"x": max(abs(v) for v in results["base_shear_x"]), "y": max(abs(v) for v in results["base_shear_y"])}}
        else:
            manifest["response"] = {"missing": True, "reason": "no completed step"}

        # --- hinges --------------------------------------------------------------------------------------------
        recorded = hd.read_hinge_recorders(coverage, time_limit=last_time)
        rows_eval = hd.evaluate_hinge_histories(recorded, hinges)
        summary = hd.yield_summary(rows_eval)
        strides = hd.stride_justification(recorded, hinges, rows_eval) if summary["springs_with_response"] else {"missing": True}
        plots = hd.plot_hinge_loops(recorded, hinges, rows_eval, out / "loops", limit=args.plot_limit) if summary["springs_with_response"] else []
        histories = hd.write_histories_npz(recorded, out / "hinge_histories_full_rate.npz") if summary["springs_with_response"] else None
        subdivided = status["convergence_summary"]["recovery_strategy_counts"]
        subdivide_steps = sum(v for k, v in subdivided.items() if "subdivide" in k)
        alignment = {axis: {"rows_in_window": recorded[axis]["rows"], "rows_dropped_beyond_window": recorded[axis]["rows_dropped_beyond_window"],
                            "time_strictly_increasing": recorded[axis]["time_strictly_increasing"],
                            "first_time": float(recorded[axis]["time"][0]) if recorded[axis]["rows"] else None,
                            "last_time": float(recorded[axis]["time"][-1]) if recorded[axis]["rows"] else None,
                            "ntha_snapshots": len(time_history), "last_snapshot_time": last_time,
                            "expected_rows_from_recovery": len(time_history) + 9 * subdivide_steps,
                            "rows_match_snapshots_plus_substeps": recorded[axis]["rows"] == len(time_history) + 9 * subdivide_steps}
                     for axis in ("y", "z")}
        (out / "hinge_evaluation.json").write_text(json.dumps(rows_eval, indent=1), encoding="utf-8")
        manifest["hinges"].update({"recorder_alignment": alignment, "summary": summary, "stride_justification": strides,
                                   "yielded": [r for r in rows_eval if r.get("yielded")],
                                   "criteria_disagreements": [r for r in rows_eval if r.get("criteria_disagree")][:50],
                                   "histories_npz": histories, "evaluation_json": str(out / "hinge_evaluation.json"), "plots": plots})
        manifest["status"] = "completed" if not status["failed"] else "solver_failed"
        if time_history:
            print(f"[{args.case}] peak interstory drift X {100 * peak_x:.3f}% (story {manifest['response']['peak_story_drift_x_story']}), "
                  f"Y {100 * peak_y:.3f}% (story {manifest['response']['peak_story_drift_y_story']}); yielded springs {summary['yielded_springs']} / "
                  f"{summary['springs']} on {summary['yielded_members']} members; max plastic rotation {summary['max_plastic_rotation_peak']:.4f} rad", flush=True)
    except Exception:                                   # noqa: BLE001 -- the observed failure is the result
        manifest["status"] = "error"
        manifest["error"] = traceback.format_exc()
        print(manifest["error"], flush=True)
        try:
            hd.close_recorders()
        except Exception:                               # noqa: BLE001
            pass
    write_manifest()
    print(f"[{args.case}] {manifest['status']}; record unchanged {manifest['record_unchanged']}; {manifest['elapsed_sec']:.0f} s; manifest {out / 'manifest.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
