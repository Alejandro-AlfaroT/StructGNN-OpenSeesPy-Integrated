"""Deterministic, parallel, stop/resume-safe RC dataset scheduler."""

import argparse
import csv
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


RC_DIR = Path(
    os.environ.get(
        "RC_STRUCTURE_DIR",
        r"C:\Users\andro\Documents\GitHub\StructGNN-OpenSeesPy-Integrated\RC Structure",
    )
).resolve()
sys.path.insert(0, str(RC_DIR))


SEED = 20260731
RANGES = {
    "num_bay_x": tuple(range(2, 7)),
    "num_bay_y": tuple(range(2, 7)),
    "num_floor": tuple(range(2, 9)),
    "story_height_ft": tuple(range(10, 15)),
    "bay_width_ft": tuple(range(10, 16)),
}
STOP_NAME = "STOP_GENERATION.json"
RECORD_MANIFEST = RC_DIR / "Ground_Motions" / "metadata" / "record_manifest.csv"
RECORD_SETS = RC_DIR / "Ground_Motions" / "metadata" / "record_sets.csv"
PEER_RESULT_RE = re.compile(r"PEER result_id=([0-9]+)")
ATOMIC_REPLACE_ATTEMPTS = 10
ATOMIC_REPLACE_INITIAL_DELAY_SEC = 0.05


def now():
    return datetime.now(timezone.utc).isoformat()


def replace_with_retry(source, destination):
    """Replace a file atomically, tolerating brief Windows sharing locks."""
    delay = ATOMIC_REPLACE_INITIAL_DELAY_SEC
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.5)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    replace_with_retry(temp, path)


def write_csv(path, rows):
    path = Path(path)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(temp, path)


def eligible_record_ids(set_name, max_npts):
    """Read eligible PEER pairs without importing OpenSees into the scheduler."""
    with RECORD_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        manifest = {row["record_id"]: row for row in csv.DictReader(file)}
    with RECORD_SETS.open(newline="", encoding="utf-8-sig") as file:
        selected_ids = {
            row["record_id"]
            for row in csv.DictReader(file)
            if row.get("set_name") == set_name
        }
    pairs = {}
    for record_id in selected_ids:
        row = manifest.get(record_id)
        if not row or (row.get("usable") or "true").strip().lower() not in {
            "1", "true", "yes", "y",
        }:
            continue
        match = PEER_RESULT_RE.search(row.get("notes") or "")
        if match:
            pairs.setdefault(int(match.group(1)), []).append(row)
    eligible = []
    for result_id, rows in pairs.items():
        if len(rows) < 2:
            continue
        pair_npts = max(int(float(row.get("npts") or 0)) for row in rows)
        if max_npts is None or max_npts <= 0 or pair_npts <= max_npts:
            eligible.append(result_id)
    return sorted(eligible)


def build_plan(num_cases, record_ids, seed=SEED):
    geometries = list(product(*RANGES.values()))
    if not 1 <= num_cases <= len(geometries):
        raise ValueError(f"num_cases must be within 1..{len(geometries)}")
    if not record_ids:
        raise ValueError("No eligible seismic record pairs were found.")
    random.Random(seed).shuffle(geometries)
    records = sorted(set(map(int, record_ids)))
    random.Random(seed + 1).shuffle(records)
    cases = []
    for index, values in enumerate(geometries[:num_cases], 1):
        bx, by, floors, story_ft, width_ft = values
        case_id = f"case_{index:04d}"
        cases.append(
            {
                "case_index": index,
                "case_id": case_id,
                "geometry_name": (
                    f"{case_id}_bx{bx}_by{by}_s{floors}_"
                    f"sh{story_ft}ft_bw{width_ft}ft"
                ),
                "result_id": records[(index - 1) % len(records)],
                "num_bay_x": bx,
                "num_bay_y": by,
                "num_floor": floors,
                "story_height_ft": story_ft,
                "story_height_in": story_ft * 12,
                "bay_width_ft": width_ft,
                "bay_x_in": width_ft * 12,
                "bay_y_in": width_ft * 12,
            }
        )
    return cases


def load_plan(root, num_cases, records, seed, max_npts):
    path = root / "parameter_plan.json"
    expected = {
        "version": 1,
        "seed": seed,
        "num_cases": num_cases,
        "max_npts": max_npts,
        "ranges": {key: list(value) for key, value in RANGES.items()},
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing plan conflicts with this command: {mismatches}. "
                "Choose another --output-root for a new plan."
            )
        return payload["cases"]
    cases = build_plan(num_cases, records, seed)
    payload = dict(expected)
    payload.update(
        {
            "created_at": now(),
            "bay_width_policy": "same width in X and Y",
            "record_policy": "seeded shuffle then balanced round-robin",
            "eligible_result_ids": records,
            "cases": cases,
        }
    )
    write_json(path, payload)
    write_csv(root / "parameter_plan.csv", cases)
    return cases


def paths_for(root, case):
    base = root / "cases" / case["case_id"]
    run = f"peer_{case['result_id']}"
    return {
        "base": base,
        "ntha": base / "ntha",
        "dataset": base / "dataset",
        "status": base / "ntha" / run / "status.json",
        "sample": base / "dataset" / run / "hybrid_sample.npz",
        "child_stop": base / "dataset" / STOP_NAME,
        "log": base / "controller.log",
    }


def successful_status(path):
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        requested = int(value.get("npts_requested") or 0)
        completed = int(value.get("completed_steps") or 0)
        return requested > 0 and requested == completed and not value.get("failed")
    except (OSError, ValueError, TypeError):
        return False


def complete(root, case):
    paths = paths_for(root, case)
    return successful_status(paths["status"]) and paths["sample"].exists()


def scan_completed(root, cases):
    """Perform the expensive filesystem scan once when a scheduler starts."""
    return {case["case_id"] for case in cases if complete(root, case)}


def select_pending_cases(cases, completed_ids, case_start=1, case_end=None, run_limit=None):
    """Select unfinished cases from an inclusive, one-based device range."""
    case_end = len(cases) if case_end is None else case_end
    if case_start < 1 or case_end < case_start or case_end > len(cases):
        raise ValueError(
            f"case range must satisfy 1 <= start <= end <= {len(cases)}; "
            f"received {case_start}..{case_end}."
        )
    scope = cases[case_start - 1:case_end]
    selected = [case for case in scope if case["case_id"] not in completed_ids]
    if run_limit is not None:
        selected = selected[:run_limit]
    return scope, selected


def command_for(args, case, paths):
    return [
        args.python_exe,
        "-B",
        str(RC_DIR / "Data_Generation" / "Generate_Hybrid_Dataset.py"),
        "--run-ntha",
        "--result-id", str(case["result_id"]),
        "--set-name", args.set_name,
        "--max-npts", str(args.max_npts),
        "--python-exe", args.python_exe,
        "--ntha-root", str(paths["ntha"]),
        "--dataset-dir", str(paths["dataset"]),
        "--geometry-name", case["geometry_name"],
        "--num-bay-x", str(case["num_bay_x"]),
        "--num-bay-y", str(case["num_bay_y"]),
        "--num-floor", str(case["num_floor"]),
        "--bay-x", str(case["bay_x_in"]),
        "--bay-y", str(case["bay_y_in"]),
        "--story-h", str(case["story_height_in"]),
    ]


def request_child_stop(path):
    if not path.exists():
        write_json(path, {"requested_at": now(), "reason": "parent_stop"})


def run_case(args, case, stop_event, active, lock):
    root = Path(args.output_root)
    paths = paths_for(root, case)
    paths["ntha"].mkdir(parents=True, exist_ok=True)
    paths["dataset"].mkdir(parents=True, exist_ok=True)
    if complete(root, case):
        return {"case_id": case["case_id"], "status": "completed", "skipped": True}
    if stop_event.is_set():
        return {"case_id": case["case_id"], "status": "pending", "skipped": True}
    command = command_for(args, case, paths)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    started = time.perf_counter()
    with paths["log"].open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=RC_DIR, env=env, stdout=log,
            stderr=subprocess.STDOUT, text=True, creationflags=flags,
        )
        with lock:
            active[case["case_id"]] = {"pid": process.pid}
        stop_sent = False
        while process.poll() is None:
            if stop_event.is_set() and not stop_sent:
                request_child_stop(paths["child_stop"])
                stop_sent = True
            time.sleep(0.5)
        returncode = process.wait()
        with lock:
            active.pop(case["case_id"], None)
    status = "completed" if complete(root, case) else (
        "pending" if stop_event.is_set() else "failed"
    )
    return {
        "case_id": case["case_id"], "status": status,
        "skipped": False, "returncode": returncode,
        "elapsed_sec": time.perf_counter() - started, "updated_at": now(),
    }


def load_results(root):
    path = root / "case_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("cases", {})


def progress(
    root, cases, results, active, status, started, completed_ids,
    write_manifest=False, invocation_case_ids=None, case_start=1, case_end=None,
):
    completed = [case["case_id"] for case in cases if case["case_id"] in completed_ids]
    failed = [
        case["case_id"]
        for case in cases
        if case["case_id"] not in completed_ids
        and results.get(case["case_id"], {}).get("status") == "failed"
    ]
    remaining = [case["case_id"] for case in cases if case["case_id"] not in completed_ids]
    state = {
        "status": status, "started_at": started, "updated_at": now(),
        "planned_count": len(cases), "completed_count": len(completed),
        "failed_count": len(failed), "remaining_count": len(remaining),
        "active_cases": active, "completed_case_ids": completed,
        "failed_case_ids": failed, "remaining_case_ids": remaining,
    }
    if invocation_case_ids is not None:
        invocation_case_ids = list(invocation_case_ids)
        invocation_completed = [
            case_id for case_id in invocation_case_ids if case_id in completed_ids
        ]
        invocation_failed = [
            case_id
            for case_id in invocation_case_ids
            if case_id not in completed_ids
            and results.get(case_id, {}).get("status") == "failed"
        ]
        invocation_remaining = [
            case_id for case_id in invocation_case_ids if case_id not in completed_ids
        ]
        state.update(
            {
                "case_range_start": case_start,
                "case_range_end": len(cases) if case_end is None else case_end,
                "invocation_selected_count": len(invocation_case_ids),
                "invocation_completed_count": len(invocation_completed),
                "invocation_failed_count": len(invocation_failed),
                "invocation_remaining_count": len(invocation_remaining),
                "invocation_case_ids": invocation_case_ids,
                "invocation_completed_case_ids": invocation_completed,
                "invocation_failed_case_ids": invocation_failed,
                "invocation_remaining_case_ids": invocation_remaining,
            }
        )
    write_json(root / "generation_state.json", state)
    if write_manifest:
        rows = []
        for case in cases:
            paths = paths_for(root, case)
            result = results.get(case["case_id"], {})
            row = dict(case)
            row.update(
                {
                    "status": (
                        "completed"
                        if case["case_id"] in completed_ids
                        else result.get("status", "pending")
                    ),
                    "sample_npz": os.path.relpath(paths["sample"], root),
                    "ntha_status": os.path.relpath(paths["status"], root),
                    "controller_log": os.path.relpath(paths["log"], root),
                    "returncode": result.get("returncode"),
                    "elapsed_sec": result.get("elapsed_sec"),
                }
            )
            rows.append(row)
        write_json(root / "case_results.json", {"updated_at": now(), "cases": results})
        write_json(root / "parameterized_manifest.json", rows)
        write_csv(root / "parameterized_manifest.csv", rows)
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cases", type=int, default=2500)
    parser.add_argument("--run-limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--case-start",
        type=int,
        default=1,
        help="First one-based case number assigned to this device (inclusive).",
    )
    parser.add_argument(
        "--case-end",
        type=int,
        help="Last one-based case number assigned to this device (inclusive).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--set-name", default="peer_mle_all")
    parser.add_argument("--max-npts", type=int, default=15000)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--refresh-status",
        action="store_true",
        help="Rescan existing case artifacts and rebuild manifests without running analyses.",
    )
    parser.add_argument("--request-stop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or (args.run_limit is not None and args.run_limit < 1):
        raise ValueError("workers and run-limit must be positive.")
    root = Path(args.output_root).resolve()
    args.output_root = str(root)
    root.mkdir(parents=True, exist_ok=True)
    stop_path = root / STOP_NAME
    if args.request_stop:
        write_json(stop_path, {"requested_at": now(), "reason": "user_requested_stop"})
        print(f"Stop requested: {stop_path}")
        return
    records = eligible_record_ids(args.set_name, args.max_npts)
    cases = load_plan(root, args.num_cases, records, args.seed, args.max_npts)
    print(f"Plan: {root / 'parameter_plan.json'} ({len(cases)} cases, {len(records)} records)")
    if args.plan_only:
        return
    results = load_results(root)
    completed_ids = scan_completed(root, cases)
    if args.refresh_status:
        previous_state_path = root / "generation_state.json"
        previous_state = (
            json.loads(previous_state_path.read_text(encoding="utf-8"))
            if previous_state_path.exists()
            else {}
        )
        state = progress(
            root,
            cases,
            results,
            {},
            "refreshed",
            previous_state.get("started_at") or now(),
            completed_ids,
            write_manifest=True,
        )
        print(
            f"Refreshed manifests: {state['completed_count']}/{state['planned_count']} "
            f"complete; {state['failed_count']} failed; "
            f"{state['remaining_count']} remaining."
        )
        return
    scope, selected = select_pending_cases(
        cases,
        completed_ids,
        case_start=args.case_start,
        case_end=args.case_end,
        run_limit=args.run_limit,
    )
    invocation_case_ids = [case["case_id"] for case in selected]
    resolved_case_end = args.case_end if args.case_end is not None else len(cases)
    started = now()
    stop_event, active, lock = threading.Event(), {}, threading.Lock()
    progress(
        root, cases, results, {}, "running", started, completed_ids,
        write_manifest=True,
        invocation_case_ids=invocation_case_ids,
        case_start=args.case_start,
        case_end=resolved_case_end,
    )
    print(
        f"Launching {len(selected)} pending cases from assigned range "
        f"{args.case_start}..{resolved_case_end} ({len(scope)} planned) "
        f"with {args.workers} workers."
    )
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {executor.submit(run_case, args, case, stop_event, active, lock): case for case in selected}
    pending = set(futures)
    last_save = 0.0
    try:
        while pending:
            if stop_path.exists() and not stop_event.is_set():
                stop_event.set()
                print("Stop detected; active cases are checkpointing.")
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                case = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"case_id": case["case_id"], "status": "failed", "error": repr(exc)}
                results[case["case_id"]] = result
                if result.get("status") == "completed":
                    completed_ids.add(case["case_id"])
                print(f"{case['case_id']}: {result['status']}")
            if done or time.monotonic() - last_save > 5:
                with lock:
                    snapshot = dict(active)
                progress(
                    root, cases, results, snapshot,
                    "stopping" if stop_event.is_set() else "running",
                    started, completed_ids, write_manifest=bool(done),
                    invocation_case_ids=invocation_case_ids,
                    case_start=args.case_start,
                    case_end=resolved_case_end,
                )
                last_save = time.monotonic()
    except KeyboardInterrupt:
        stop_event.set()
        print("Ctrl+C received; active cases are checkpointing.")
        for case in selected:
            request_child_stop(paths_for(root, case)["child_stop"])
        for future, case in futures.items():
            try:
                result = future.result()
                results[case["case_id"]] = result
                if result.get("status") == "completed":
                    completed_ids.add(case["case_id"])
            except Exception as exc:
                results[case["case_id"]] = {"case_id": case["case_id"], "status": "failed", "error": repr(exc)}
    finally:
        executor.shutdown(wait=True)
    with lock:
        snapshot = dict(active)
    final = "stopped" if stop_event.is_set() else (
        "completed" if len(completed_ids) == len(cases) else "pilot_completed"
    )
    state = progress(
        root, cases, results, snapshot, final, started, completed_ids,
        write_manifest=True,
        invocation_case_ids=invocation_case_ids,
        case_start=args.case_start,
        case_end=resolved_case_end,
    )
    if stop_path.exists():
        stop_path.unlink()
    print(f"Completed {state['completed_count']}/{state['planned_count']}; remaining {state['remaining_count']}.")
    print(
        "Current invocation completed "
        f"{state['invocation_completed_count']}/{state['invocation_selected_count']}; "
        f"remaining {state['invocation_remaining_count']}."
    )


if __name__ == "__main__":
    main()
