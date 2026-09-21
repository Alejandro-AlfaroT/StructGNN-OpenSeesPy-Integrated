"""Run the direction-explicit pushover diagnostic on a saved fixed design (one worker, isolated output).

    python Design/Benchmark_Diagnostic.py --root D:/StructGNN_outputs/dv150_v10 --case case_0074 \
        --runs y+ --output-root <folder> [--du 0.05 --max-steps 1200 --target-drift 0.04 \
        --gravity-dead 1.0 --gravity-live 0.25 --pattern elf --count 150]

Applies the case's geometry and site, installs the saved design
(Design_Driver.apply_design: sections, cage, hoops, slab, transfer, slab
reinforcement) and runs Analysis.Pushover_Diagnostic.run_diagnostic for each
requested run in sequence, writing <output-root>/<case>/<run>/summary.json
and steps.jsonl plus a manifest with the record's SHA-256 and the source
identity. The saved record is read only; nothing is redesigned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
sys.path.insert(0, str(RC / "Data_Generation"))


def parse_run(text):
    text = text.strip().lower()
    if len(text) != 2 or text[0] not in "xy" or text[1] not in "+-":
        raise argparse.ArgumentTypeError("runs are x+, x-, y+ or y-")
    return text[0], 1.0 if text[1] == "+" else -1.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="verification root holding <case>/design.json")
    parser.add_argument("--case", required=True)
    parser.add_argument("--runs", nargs="+", type=parse_run, default=[("y", 1.0)])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--count", type=int, default=150, help="plan size the case ids belong to")
    parser.add_argument("--du", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--target-drift", type=float, default=0.04)
    parser.add_argument("--gravity-dead", type=float, default=1.0)
    parser.add_argument("--gravity-live", type=float, default=0.25)
    parser.add_argument("--pattern", default="elf", choices=("elf", "uniform", "triangular"))
    parser.add_argument("--stop-fraction", type=float, default=0.2)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    import Structure_Parameters as sp
    import Geometry_Overrides as go
    from Design import Design_Driver as driver
    from Design.Verify_Designs import plan_cases
    from Analysis.Pushover_Diagnostic import DiagnosticSettings, run_diagnostic

    cases = {c["case_id"]: c for c in plan_cases(args.count)}
    case = cases[args.case]
    record_path = Path(args.root) / args.case / "design.json"
    raw = record_path.read_bytes()
    record = json.loads(raw)
    overrides = {"NUM_BAY_X": case["num_bay_x"], "NUM_BAY_Y": case["num_bay_y"], "NUM_FLOOR": case["num_floor"],
                 "STORY_H": 12.0 * case["story_height_ft"], "BAY_X": 12.0 * case["bay_x_width_ft"],
                 "BAY_Y": 12.0 * case["bay_y_width_ft"]}
    go.apply_geometry_overrides(overrides, variant_name=case["geometry_name"], emit=False)
    sp.apply_seismic_site(case["seismic_site"])
    driver.apply_design(record)
    period = driver._model_period()
    out_root = Path(args.output_root) / args.case
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"case": case, "record": str(record_path), "record_sha256": hashlib.sha256(raw).hexdigest(), "record_bytes": len(raw),
                "schema_version": record.get("schema_version"), "request_identity_sha256": (record.get("request_identity") or {}).get("sha256"),
                "model_period_sec": period, "source_identity": driver.design_request_identity()["source_sha256"],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
    for direction, sign in args.runs:
        name = f"{direction}{'+' if sign > 0 else '-'}"
        settings = DiagnosticSettings(direction=direction, sign=sign, load_pattern=args.pattern, model_period_sec=period,
                                      gravity_dead_factor=args.gravity_dead, gravity_live_factor=args.gravity_live,
                                      du_in=args.du, max_steps=args.max_steps, target_roof_drift_ratio=args.target_drift,
                                      stop_on_strength_loss_fraction=args.stop_fraction, record_every=args.record_every,
                                      label=args.label or f"{args.case} {name}")
        print(f"[{args.case}] run {name}: du {args.du} in, {args.max_steps} steps max, target drift {args.target_drift}, "
              f"gravity D x {args.gravity_dead} + L x {args.gravity_live}, pattern {args.pattern}", flush=True)
        summary = run_diagnostic(settings, out_root / name, verbose=args.verbose)
        manifest["runs"][name] = {"stop_reason": summary["stop_reason"], "completed_steps": summary["completed_steps"],
                                  "peak_base_shear_kip": summary["peak_base_shear_kip"], "checks": summary["checks"],
                                  "event_count": summary["event_count"], "elapsed_sec": summary["elapsed_sec"]}
        print(f"[{args.case}] {name}: {summary['stop_reason']} after {summary['completed_steps']} steps, peak base shear "
              f"{summary['peak_base_shear_kip']:.2f} kip, checks all_pass={summary['checks']['all_pass']}", flush=True)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
