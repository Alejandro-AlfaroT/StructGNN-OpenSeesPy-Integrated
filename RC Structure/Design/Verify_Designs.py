"""Design-only verification run over the generation plan's own geometries.

Reproduces the geometry and hazard sampling of
Data_Generation/Generate_Parameterized_Dataset.build_plan (same seed, same
shuffle, same round-robin over sites), designs each case with the committed
Design/Config.py in a fresh interpreter, and tabulates what qualification
found. No ground motions are selected and no time-history analysis runs.

    python Design/Verify_Designs.py --count 150 --workers 4 --output-root outputs/design_verification_20260914

Resumable: a case whose result.json says "designed" is skipped. Each case
keeps its full design.json. ``--probe-assertions`` fills the three assertion
blocks with PROBE values (labelled in every artifact's request identity) so
the pipeline can be exercised before the real assertions exist; it is not a
certification and the summary says so.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))
sys.path.insert(0, str(RC_DIR / "Data_Generation"))

from Generate_Parameterized_Dataset import RANGES, SEED, SEISMIC_SITES  # noqa: E402

PROBE = "PROBE -- design verification run, not a certification"


def plan_cases(num_cases, seed=SEED, geometry_offset=0, seismic_sites=SEISMIC_SITES):
    """The first ``num_cases`` cases of the generation plan, geometry and site only."""
    geometries = list(product(*RANGES.values()))
    if geometry_offset + num_cases > len(geometries):
        raise ValueError(f"{geometry_offset + num_cases} exceeds the {len(geometries)} plan geometries.")
    random.Random(seed).shuffle(geometries)
    sites = list(seismic_sites)
    random.Random(seed + 2).shuffle(sites)
    cases = []
    for local_index, values in enumerate(geometries[geometry_offset:geometry_offset + num_cases], 1):
        bx, by, floors, story_ft, width_x_ft, width_y_ft = values
        case_index = geometry_offset + local_index
        cases.append({"case_id": f"case_{case_index:04d}", "num_bay_x": bx, "num_bay_y": by, "num_floor": floors,
                      "story_height_ft": story_ft, "bay_x_width_ft": width_x_ft, "bay_y_width_ft": width_y_ft,
                      "seismic_site": sites[(case_index - 1) % len(sites)],
                      "geometry_name": (f"case_{case_index:04d}_bx{bx}_by{by}_s{floors}_"
                                        f"sh{story_ft}ft_bwx{width_x_ft}ft_bwy{width_y_ft}ft_{sites[(case_index - 1) % len(sites)]}")})
    return cases


def probe_config():
    from Design.Config import DesignConfig, SlabActionAssertions, DemandPolicy, IndependentVerification
    slab_flags = ("analysis_applicability_verified", "all_floors_enveloped", "load_pattern_envelope_verified",
                  "spatial_envelope_per_unit_width", "twisting_moment_resolution_verified",
                  "zero_membrane_force_verified", "verified", "two_way_shear_path_assessed")
    verify_flags = ("floor_hand_check_verified", "strength_model_verified", "detailing_model_consistency_verified",
                    "slab_column_local_steel_assessed", "fire_resistance_scope_accepted",
                    "congestion_and_placement_accepted", "floor_frame_compatibility_reviewed")
    stamp = dict(asserted_by=PROBE, assertion_date=time.strftime("%Y-%m-%d"), assertion_basis=PROBE)
    return DesignConfig(
        slab_actions=SlabActionAssertions(**{k: True for k in slab_flags}, **stamp),
        demands=DemandPolicy(declared_by=PROBE, declaration_date=time.strftime("%Y-%m-%d"), declaration_basis=PROBE),
        verification=IndependentVerification(**{k: True for k in verify_flags}, **stamp))


def run_worker(case, out_dir, probe):
    """Design one case in this interpreter; write result.json; never raise."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"case": case, "status": "started", "probe_assertions": probe, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = io.StringIO()
    t0 = time.perf_counter()
    try:
        import Structure_Parameters as sp
        import Geometry_Overrides as go
        from Design.Config import DesignConfig
        from Design.Design_Driver import load_or_create_design
        overrides = {"NUM_BAY_X": case["num_bay_x"], "NUM_BAY_Y": case["num_bay_y"], "NUM_FLOOR": case["num_floor"],
                     "STORY_H": 12.0 * case["story_height_ft"], "BAY_X": 12.0 * case["bay_x_width_ft"],
                     "BAY_Y": 12.0 * case["bay_y_width_ft"]}
        with contextlib.redirect_stdout(log):
            go.apply_geometry_overrides(overrides, variant_name=case["geometry_name"], emit=True)
            sp.apply_seismic_site(case["seismic_site"])
            cfg = probe_config() if probe else DesignConfig.from_structure_parameters()
            record, created = load_or_create_design(out_dir / "design.json", cfg=cfg, verbose=True)
        q = record["qualification"]
        by_status = {}
        for c in q["checks"]:
            by_status.setdefault(c["status"], set()).add(c["id"].split(":")[0])
        s, reb = record["sections"], record["reinforcement"]
        result.update({
            "status": "designed", "created": created, "elapsed_s": time.perf_counter() - t0,
            "design_json_bytes": (out_dir / "design.json").stat().st_size,
            "accepted": q["accepted"], "counts": q["counts"],
            "fail_ids": sorted(by_status.get("fail", [])), "not_evaluated_ids": sorted(by_status.get("not_evaluated", [])),
            "sections": s, "iterations": record.get("iterations"), "governed_by": record["dcr"]["governed_by"],
            "dcr": {"beam": record["dcr"]["beam"], "column": record["dcr"]["column"]},
            "column_bars": [reb["col_bar_size"], reb["col_top_bars"], reb["col_side_bars"]],
            "beam_bars": [reb["beam_bar_size"], reb["beam_top_bars"], reb["beam_bot_bars"]],
            "column_hoops": [reb["col_stirrup_bar_size"], reb["col_stirrup_legs"], reb["col_stirrup_spacing_in"]],
            "beam_hoops": [reb["beam_stirrup_bar_size"], reb["beam_stirrup_legs"], reb["beam_stirrup_spacing_in"]],
            "coupled_max_vertical_difference": (record.get("coupled_comparison") or {}).get("max_column_vertical_relative_difference"),
            "schema_version": record.get("schema_version"),
        })
    except Exception as exc:                              # noqa: BLE001
        result.update({"status": "error", "elapsed_s": time.perf_counter() - t0,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (out_dir / "log.txt").write_text(log.getvalue(), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def _launch(python_exe, case, out_dir, probe):
    out_dir = Path(out_dir)
    existing = out_dir / "result.json"
    if existing.exists():
        try:
            saved = json.loads(existing.read_text(encoding="utf-8"))
            if saved.get("status") == "designed":
                return saved, True
        except (OSError, ValueError):
            pass
    command = [python_exe, "-B", str(Path(__file__).resolve()), "--worker", json.dumps(case), str(out_dir)]
    if probe:
        command.append("--probe-assertions")
    t0 = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=str(RC_DIR), timeout=6 * 3600)
    (out_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if existing.exists():
        try:
            return json.loads(existing.read_text(encoding="utf-8")), False
        except (OSError, ValueError):
            pass
    return {"case": case, "status": "error", "elapsed_s": time.perf_counter() - t0,
            "error": f"worker exited {proc.returncode} without a result", "stderr_tail": (proc.stderr or "")[-2000:]}, False


def summarize(results, root, probe):
    rows = sorted(results, key=lambda r: r["case"]["case_id"])
    designed = [r for r in rows if r.get("status") == "designed"]
    accepted = [r for r in designed if r.get("accepted")]
    errors = [r for r in rows if r.get("status") == "error"]
    open_items = Counter(i for r in designed for i in r.get("not_evaluated_ids", []))
    fail_items = Counter(i for r in designed for i in r.get("fail_ids", []))
    total_time = sum(r.get("elapsed_s", 0.0) for r in designed)
    total_bytes = sum(r.get("design_json_bytes", 0) for r in designed)
    lines = [f"# Design verification run -- {time.strftime('%Y-%m-%d %H:%M')}\n",
             f"{len(rows)} plan cases; {len(designed)} designed, {len(accepted)} accepted, "
             f"{len(designed) - len(accepted)} not accepted, {len(errors)} errors.",
             f"Assertions: {'PROBE values in all three blocks -- exercises the pipeline, certifies nothing' if probe else 'the committed Design/Config.py'}.",
             f"Design time {total_time / 60:.0f} min serial-equivalent ({total_time / max(1, len(designed)) / 60:.1f} min/case); "
             f"design.json total {total_bytes / 1e9:.2f} GB.\n"]
    if fail_items:
        lines.append("## Failed checks (cases with the item)\n")
        lines += [f"- `{k}`: {v}" for k, v in fail_items.most_common()]
        lines.append("")
    if open_items:
        lines.append("## Open (not evaluated) items (cases with the item)\n")
        lines += [f"- `{k}`: {v}" for k, v in open_items.most_common()]
        lines.append("")
    if errors:
        lines.append("## Errors\n")
        lines += [f"- {r['case']['case_id']}: {r.get('error')}" for r in errors]
        lines.append("")
    lines.append("## Cases\n")
    lines.append("| case | plan | site | result | fail | open | min | MB | sections | col bars | beam bars | col hoops | beam hoops | governed | coupled max |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["case"]
        plan = f"{c['num_bay_x']}x{c['num_bay_y']}x{c['num_floor']} {c['bay_x_width_ft']}x{c['bay_y_width_ft']} ft / {c['story_height_ft']} ft"
        if r.get("status") != "designed":
            lines.append(f"| {c['case_id']} | {plan} | {c['seismic_site']} | ERROR | | | | | {str(r.get('error', ''))[:80]} | | | | | | |")
            continue
        s = r["sections"]
        lines.append(f"| {c['case_id']} | {plan} | {c['seismic_site']} | {'accepted' if r['accepted'] else 'open'} | "
                     f"{r['counts']['fail']} | {r['counts']['not_evaluated']} | {r['elapsed_s'] / 60:.1f} | {r['design_json_bytes'] / 1e6:.0f} | "
                     f"{s['b_col_in']:g}x{s['h_col_in']:g} fc{s['fc_col_ksi']:g} / {s['b_beam_in']:g}x{s['h_beam_in']:g} fc{s['fc_beam_ksi']:g} | "
                     f"#{r['column_bars'][0]} {r['column_bars'][1]}T/{r['column_bars'][2]}S | #{r['beam_bars'][0]} {r['beam_bars'][1]}T/{r['beam_bars'][2]}B | "
                     f"#{r['column_hoops'][0]}-{r['column_hoops'][1]}L@{r['column_hoops'][2]:g} | #{r['beam_hoops'][0]}-{r['beam_hoops'][1]}L@{r['beam_hoops'][2]:g} | "
                     f"{r['governed_by']} | {100 * (r.get('coupled_max_vertical_difference') or 0):.1f}% |")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "num_bay_x", "num_bay_y", "num_floor", "story_height_ft", "bay_x_width_ft", "bay_y_width_ft",
                         "seismic_site", "status", "accepted", "fail", "not_evaluated", "elapsed_s", "design_json_bytes",
                         "b_col_in", "h_col_in", "fc_col_ksi", "b_beam_in", "h_beam_in", "fc_beam_ksi", "governed_by",
                         "fail_ids", "not_evaluated_ids"])
        for r in rows:
            c, s = r["case"], r.get("sections") or {}
            writer.writerow([c["case_id"], c["num_bay_x"], c["num_bay_y"], c["num_floor"], c["story_height_ft"],
                             c["bay_x_width_ft"], c["bay_y_width_ft"], c["seismic_site"], r.get("status"), r.get("accepted"),
                             (r.get("counts") or {}).get("fail"), (r.get("counts") or {}).get("not_evaluated"),
                             round(r.get("elapsed_s", 0.0), 1), r.get("design_json_bytes"),
                             s.get("b_col_in"), s.get("h_col_in"), s.get("fc_col_ksi"), s.get("b_beam_in"), s.get("h_beam_in"),
                             s.get("fc_beam_ksi"), r.get("governed_by"),
                             ";".join(r.get("fail_ids", [])), ";".join(r.get("not_evaluated_ids", []))])
    (root / "summary.json").write_text(json.dumps({"probe_assertions": probe, "cases": rows}, indent=1, default=str),
                                       encoding="utf-8")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", nargs=2, metavar=("CASE_JSON", "OUT_DIR"), help=argparse.SUPPRESS)
    parser.add_argument("--count", type=int, default=150, help="Number of plan cases, from the first (default 150).")
    parser.add_argument("--geometry-offset", type=int, default=0, help="Skip this many plan geometries first.")
    parser.add_argument("--seed", type=int, default=SEED, help="Plan seed (default: the generation plan's).")
    parser.add_argument("--sites", nargs="*", default=None, help="Hazard labels to round-robin over (default: the plan's).")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Only these plan case ids.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", default=None, help="Default: outputs/design_verification_<date>.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--probe-assertions", action="store_true",
                        help="Fill the three assertion blocks with PROBE values (labelled); exercises the pipeline only.")
    args = parser.parse_args(argv)
    if args.worker:
        case = json.loads(args.worker[0])
        result = run_worker(case, args.worker[1], args.probe_assertions)
        print(json.dumps({k: result.get(k) for k in ("status", "elapsed_s", "accepted", "counts", "error")}, default=str))
        return 0
    root = Path(args.output_root or (RC_DIR / "outputs" / f"design_verification_{time.strftime('%Y%m%d')}"))
    root.mkdir(parents=True, exist_ok=True)
    cases = plan_cases(args.count, seed=args.seed, geometry_offset=args.geometry_offset,
                       seismic_sites=tuple(args.sites) if args.sites else SEISMIC_SITES)
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c["case_id"] in wanted]
    (root / "plan.json").write_text(json.dumps({"seed": args.seed, "geometry_offset": args.geometry_offset,
                                                 "count": args.count, "cases": cases}, indent=1), encoding="utf-8")
    print(f"{len(cases)} cases, {args.workers} workers, root {root}, "
          f"{'PROBE assertions' if args.probe_assertions else 'committed Design/Config.py'}", flush=True)
    results = []

    def run(case):
        t0 = time.perf_counter()
        result, cached = _launch(args.python_exe, case, root / case["case_id"], args.probe_assertions)
        tag = "cached" if cached else f"{time.perf_counter() - t0:6.0f}s"
        counts = result.get("counts") or {}
        print(f"[{time.strftime('%H:%M:%S')}] {case['case_id']} {tag:>8}  {result.get('status')}  "
              f"accepted={result.get('accepted')}  fail={counts.get('fail')} open={counts.get('not_evaluated')}  "
              f"{result.get('error', '') or ''}", flush=True)
        return result

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, cases))
    lines = summarize(results, root, args.probe_assertions)
    print("\n".join(lines[:12]))
    print(f"summary: {root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
