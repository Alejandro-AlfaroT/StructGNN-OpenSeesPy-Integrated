"""Deterministic, parallel, stop/resume-safe RC dataset scheduler."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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

from Run_Naming import analysis_run_name  # noqa: E402


SEED = 20260731

# Two- and three-story frames are excluded. They need roughly four times the
# spectral demand of an eight-story frame to reach the same drift, which made
# a single intensity rule unreliable across the set, and they are the shortest
# periods where the equal-displacement assumption behind that rule is weakest.
# The upper bound moves to nine stories to keep the geometry pool large enough
# and because taller frames are the more inelastic ones.
RANGES = {
    "num_bay_x": tuple(range(2, 7)),
    "num_bay_y": tuple(range(2, 7)),
    "num_floor": tuple(range(4, 10)),
    "story_height_ft": tuple(range(10, 15)),
    "bay_width_ft": tuple(range(10, 16)),
}

# Design hazard, assigned per case. Geometry alone barely moves design demand,
# so without this axis the design loop returns near-identical members for
# different buildings. Labels must match Structure_Parameters.SEISMIC_SITE_OPTIONS.
SEISMIC_SITES = ("sdc_c", "sdc_d_low", "sdc_d_high", "sdc_e", "sdc_e_near")

# Ground-motion scale factors applied to each record pair. Running one
# structure at several intensities is what separates structural response from
# record identity; with one intensity per structure the two are confounded and
# the surrogate cannot learn an intensity-to-response mapping.
#
# These are placeholders. The calibration step replaces them with per-case
# factors targeting a drift distribution; until then a uniform ladder keeps
# the schema and the run naming exercised.
DEFAULT_INTENSITY_LEVELS = (1.0, 2.0, 3.0)
DEFAULT_RECORDS_PER_CASE = 2

# Bumped when plan semantics change. Version 3 adds the hazard axis, the
# intensity ladder, and multiple records per case, so a version 1 or 2 plan
# describes a different experiment and must not be silently reused.
PLAN_VERSION = 3
STOP_NAME = "STOP_GENERATION.json"
RECORD_MANIFEST = RC_DIR / "Ground_Motions" / "metadata" / "record_manifest.csv"
RECORD_SETS = RC_DIR / "Ground_Motions" / "metadata" / "record_sets.csv"
PEER_RESULT_RE = re.compile(r"PEER result_id=([0-9]+)")
ATOMIC_REPLACE_ATTEMPTS = 10
ATOMIC_REPLACE_INITIAL_DELAY_SEC = 0.05
NPTS_POLICY_OVERRIDE = "allow_npts_policy_excluded=true"
NPTS_POLICY_EXCLUSION = "omitted_from_generation=max_npts_gt_15000"


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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def eligible_record_ids(set_name, max_npts):
    """Read eligible PEER pairs without importing OpenSees into the scheduler."""
    with RECORD_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        manifest = {row["record_id"]: row for row in csv.DictReader(file)}
    with RECORD_SETS.open(newline="", encoding="utf-8-sig") as file:
        selected = {
            row["record_id"]: row
            for row in csv.DictReader(file)
            if row.get("set_name") == set_name
        }
    pairs = {}
    for record_id, set_row in selected.items():
        row = manifest.get(record_id)
        if not row:
            continue
        usable = (row.get("usable") or "true").strip().lower() in {
            "1", "true", "yes", "y",
        }
        policy_override = (
            NPTS_POLICY_OVERRIDE in (set_row.get("notes") or "").lower()
            and NPTS_POLICY_EXCLUSION in (row.get("notes") or "").lower()
        )
        if not usable and not policy_override:
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


def build_runs(case_records, intensity_levels, scale_factors=None):
    """Expand a case's record and intensity assignments into concrete runs.

    Run names are produced by the same helper the analysis uses, so the
    scheduler can tell a finished run from a pending one without importing
    OpenSees or guessing at the naming convention.
    """
    runs = []
    per_record = len(intensity_levels)
    for record_index, result_id in enumerate(case_records):
        for slot in range(per_record):
            scale = (
                scale_factors[record_index * per_record + slot]
                if scale_factors is not None
                else intensity_levels[slot]
            )
            runs.append(
                {
                    "run_index": len(runs) + 1,
                    "result_id": int(result_id),
                    "scale_factor": float(scale),
                    "run_name": analysis_run_name(
                        f"peer_{int(result_id)}", scale_factor=float(scale)
                    ),
                }
            )
    return runs


def eligible_record_pairs(set_name, max_npts):
    """Map each eligible PEER result_id to its (X, Y) record ids.

    eligible_record_ids answers which pairs may be used; the calibration also
    needs to know which physical records those are, so it can read their
    spectra. Component order follows the manifest's horizontal_component note
    so X and Y stay consistent with what the analysis actually applies.
    """
    with RECORD_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        manifest = list(csv.DictReader(file))
    allowed = set(eligible_record_ids(set_name, max_npts))

    grouped = {}
    for row in manifest:
        match = PEER_RESULT_RE.search(row.get("notes") or "")
        if not match:
            continue
        result_id = int(match.group(1))
        if result_id not in allowed:
            continue
        component = 2 if "horizontal_component=2" in (row.get("notes") or "") else 1
        grouped.setdefault(result_id, {})[component] = row["record_id"]

    pairs = {}
    for result_id, components in grouped.items():
        first = components.get(1)
        second = components.get(2, first)
        if first:
            pairs[result_id] = (first, second)
    return pairs


def _calibrated_scale_factors(calibration, case_geometry, record_pairs, case_records, targets):
    """Scale factors predicted to land each run on its target drift."""
    from Calibrate_Intensity import (
        estimated_period,
        geometric_mean_sa,
        scale_factor_for,
    )

    coefficients = calibration["coefficients"]
    spectra = calibration["record_spectra_g"]
    period = estimated_period(
        case_geometry["num_floor"],
        case_geometry["story_height_in"],
        calibration["period_ratio"],
    )

    factors = []
    index = 0
    for result_id in case_records:
        pair = record_pairs.get(int(result_id))
        for _ in range(len(targets) // max(1, len(case_records))):
            target = targets[index % len(targets)]
            index += 1
            if not pair:
                factors.append(1.0)
                continue
            unscaled = geometric_mean_sa(spectra, pair[0], pair[1], period)
            factors.append(
                round(scale_factor_for(target, period, unscaled, coefficients), 3)
            )
    return factors, period


def build_plan(
    num_cases,
    record_ids,
    seed=SEED,
    geometry_offset=0,
    case_id_offset=0,
    seismic_sites=SEISMIC_SITES,
    intensity_levels=DEFAULT_INTENSITY_LEVELS,
    records_per_case=DEFAULT_RECORDS_PER_CASE,
    calibration=None,
    record_pairs=None,
):
    geometries = list(product(*RANGES.values()))
    if geometry_offset < 0 or case_id_offset < 0:
        raise ValueError("geometry-offset and case-id-offset must be nonnegative.")
    if not 1 <= num_cases or geometry_offset + num_cases > len(geometries):
        raise ValueError(
            "Requested geometry slice must fit within the "
            f"{len(geometries)} available combinations; received offset="
            f"{geometry_offset}, num_cases={num_cases}."
        )
    if not record_ids:
        raise ValueError("No eligible seismic record pairs were found.")
    if not seismic_sites:
        raise ValueError("At least one seismic site is required.")
    if not intensity_levels:
        raise ValueError("At least one intensity level is required.")
    records = sorted(set(map(int, record_ids)))
    if records_per_case < 1:
        raise ValueError(f"records-per-case must be positive; received {records_per_case}.")
    if records_per_case > len(records):
        # A catalog smaller than the request is a degenerate catalog, not a
        # user error. Clamp rather than halt, but say so: silently assigning
        # fewer records than asked for would be invisible in the plan.
        print(
            f"WARNING: records-per-case {records_per_case} exceeds the "
            f"{len(records)} eligible record pair(s); clamping to {len(records)}."
        )
        records_per_case = len(records)
    random.Random(seed).shuffle(geometries)
    random.Random(seed + 1).shuffle(records)
    sites = list(seismic_sites)
    random.Random(seed + 2).shuffle(sites)

    # With a calibration, intensity is chosen per run to hit a target drift
    # rather than taken from a fixed ladder. Targets are drawn once for the
    # whole plan so the realized drift distribution matches the intent across
    # the dataset, not merely within each case.
    runs_per_case = records_per_case * len(intensity_levels)
    plan_targets = None
    if calibration is not None:
        from Calibrate_Intensity import target_drift_sequence

        plan_targets = target_drift_sequence(num_cases * runs_per_case, seed=seed)
        record_pairs = record_pairs or {}

    cases = []
    selected_geometries = geometries[geometry_offset:geometry_offset + num_cases]
    for local_index, values in enumerate(selected_geometries, 1):
        bx, by, floors, story_ft, width_ft = values
        case_index = case_id_offset + local_index
        case_id = f"case_{case_index:04d}"

        # Round-robin over both axes so hazard and records stay balanced
        # across any contiguous slice of the plan, including the per-device
        # ranges the scheduler hands out.
        site = sites[(case_index - 1) % len(sites)]
        start = ((case_index - 1) * records_per_case) % len(records)
        case_records = [
            records[(start + offset) % len(records)]
            for offset in range(records_per_case)
        ]

        scale_factors = None
        estimated_period_sec = None
        if plan_targets is not None:
            start = (local_index - 1) * runs_per_case
            scale_factors, estimated_period_sec = _calibrated_scale_factors(
                calibration,
                {"num_floor": floors, "story_height_in": story_ft * 12},
                record_pairs,
                case_records,
                plan_targets[start:start + runs_per_case],
            )

        cases.append(
            {
                "case_index": case_index,
                "case_id": case_id,
                "geometry_name": (
                    f"{case_id}_bx{bx}_by{by}_s{floors}_"
                    f"sh{story_ft}ft_bw{width_ft}ft_{site}"
                ),
                "seismic_site": site,
                "result_ids": case_records,
                "num_bay_x": bx,
                "num_bay_y": by,
                "num_floor": floors,
                "story_height_ft": story_ft,
                "story_height_in": story_ft * 12,
                "bay_width_ft": width_ft,
                "bay_x_in": width_ft * 12,
                "bay_y_in": width_ft * 12,
                "runs": build_runs(case_records, intensity_levels, scale_factors),
                "estimated_period_sec": estimated_period_sec,
            }
        )
    return cases


def plan_csv_rows(cases):
    """Flatten cases for the CSV mirror of the plan.

    The runs list cannot be a CSV cell, so it is summarized here. The JSON
    plan remains the authoritative record of every individual run.
    """
    rows = []
    for case in cases:
        row = {key: value for key, value in case.items() if key not in {"runs", "result_ids"}}
        runs = case_runs(case)
        row.update(
            {
                "result_ids": " ".join(str(value) for value in case.get("result_ids", [])),
                "intensity_levels": " ".join(
                    f"{value:g}" for value in sorted({run["scale_factor"] for run in runs})
                ),
                "run_names": " ".join(run["run_name"] for run in runs),
                "num_runs": len(runs),
            }
        )
        rows.append(row)
    return rows


def load_plan(
    root,
    num_cases,
    records,
    seed,
    max_npts,
    set_name="peer_mle_all",
    geometry_offset=0,
    case_id_offset=0,
    seismic_sites=SEISMIC_SITES,
    intensity_levels=DEFAULT_INTENSITY_LEVELS,
    records_per_case=DEFAULT_RECORDS_PER_CASE,
    calibration=None,
    record_pairs=None,
):
    path = root / "parameter_plan.json"
    # Every field is compared on every version. Earlier revisions left
    # set_name out of the version 1 comparison, so changing the record set
    # silently reused a plan built for a different catalog.
    expected = {
        "version": PLAN_VERSION,
        "seed": seed,
        "num_cases": num_cases,
        "max_npts": max_npts,
        "ranges": {key: list(value) for key, value in RANGES.items()},
        "set_name": set_name,
        "geometry_offset": geometry_offset,
        "case_id_offset": case_id_offset,
        "seismic_sites": list(seismic_sites),
        "intensity_levels": [float(value) for value in intensity_levels],
        "records_per_case": records_per_case,
        "eligible_result_ids": list(records),
        # A plan built against a calibration describes a different experiment
        # from one built on the uniform ladder, so the fingerprint is part of
        # the conflict check.
        "intensity_source": "calibration" if calibration else "uniform_ladder",
        "calibration_fingerprint": (
            calibration["coefficients"] if calibration else None
        ),
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
    cases = build_plan(
        num_cases,
        records,
        seed,
        geometry_offset=geometry_offset,
        case_id_offset=case_id_offset,
        seismic_sites=seismic_sites,
        intensity_levels=intensity_levels,
        records_per_case=records_per_case,
        calibration=calibration,
        record_pairs=record_pairs,
    )
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
    write_csv(root / "parameter_plan.csv", plan_csv_rows(cases))
    return cases


def paths_for(root, case):
    """Case-level paths. The design artifact is shared by every run below it."""
    base = root / "cases" / case["case_id"]
    return {
        "base": base,
        "ntha": base / "ntha",
        "dataset": base / "dataset",
        "design": base / "design.json",
        "child_stop": base / "dataset" / STOP_NAME,
        "log": base / "controller.log",
    }


def case_runs(case):
    """Runs belonging to a case: one per (record pair, intensity) combination."""
    return case.get("runs") or []


def run_paths_for(root, case, run):
    base = root / "cases" / case["case_id"]
    name = run["run_name"]
    return {
        "status": base / "ntha" / name / "status.json",
        "sample": base / "dataset" / name / "hybrid_sample.npz",
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


def complete_run(root, case, run):
    paths = run_paths_for(root, case, run)
    return successful_status(paths["status"]) and paths["sample"].exists()


def completed_run_count(root, case):
    return sum(1 for run in case_runs(case) if complete_run(root, case, run))


def complete(root, case):
    """A case is finished only when every one of its runs is finished."""
    runs = case_runs(case)
    return bool(runs) and all(complete_run(root, case, run) for run in runs)


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
    command = [
        args.python_exe,
        "-B",
        str(RC_DIR / "Data_Generation" / "Generate_Hybrid_Dataset.py"),
        "--run-ntha",
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
        "--seismic-site", str(case["seismic_site"]),
    ]
    # The child expands records against intensities, matching build_runs.
    for result_id in case["result_ids"]:
        command.extend(["--result-id", str(result_id)])
    for scale in sorted({run["scale_factor"] for run in case_runs(case)}):
        command.extend(["--scale-factor", str(scale)])
    return command


def request_child_stop(path):
    if not path.exists():
        write_json(path, {"requested_at": now(), "reason": "parent_stop"})


def run_case(args, case, stop_event, active, lock):
    root = Path(args.output_root)
    paths = paths_for(root, case)
    paths["ntha"].mkdir(parents=True, exist_ok=True)
    paths["dataset"].mkdir(parents=True, exist_ok=True)
    if complete(root, case):
        return {
            "case_id": case["case_id"], "status": "completed", "skipped": True,
            "completed_runs": len(case_runs(case)),
            "planned_runs": len(case_runs(case)),
        }
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
    finished_runs = completed_run_count(root, case)
    status = "completed" if finished_runs == len(case_runs(case)) else (
        "pending" if stop_event.is_set() else "failed"
    )
    return {
        "case_id": case["case_id"], "status": status,
        "skipped": False, "returncode": returncode,
        "completed_runs": finished_runs,
        "planned_runs": len(case_runs(case)),
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
                    "planned_runs": len(case_runs(case)),
                    "completed_runs": result.get(
                        "completed_runs",
                        len(case_runs(case)) if case["case_id"] in completed_ids else None,
                    ),
                    "case_dir": os.path.relpath(paths["base"], root),
                    "design_json": os.path.relpath(paths["design"], root),
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
    parser.add_argument(
        "--seismic-site",
        action="append",
        dest="seismic_sites",
        help=(
            "Design hazard label to include in the plan; repeat to set the "
            "pool. Defaults to all of: " + ", ".join(SEISMIC_SITES) + "."
        ),
    )
    parser.add_argument(
        "--intensity-level",
        action="append",
        type=float,
        dest="intensity_levels",
        help=(
            "Ground-motion scale factor applied to every record pair; repeat "
            "for a ladder. Defaults to "
            + " ".join(f"{value:g}" for value in DEFAULT_INTENSITY_LEVELS)
            + ". Total runs per case = records-per-case x intensity levels."
        ),
    )
    parser.add_argument(
        "--intensity-calibration",
        default=None,
        help=(
            "Calibration artifact from Calibrate_Intensity.py. When given, "
            "each run scale factor is chosen to hit a target drift instead "
            "of using the uniform ladder."
        ),
    )
    parser.add_argument(
        "--records-per-case",
        type=int,
        default=DEFAULT_RECORDS_PER_CASE,
        help="Distinct record pairs assigned to each structure.",
    )
    parser.add_argument("--max-npts", type=int, default=15000)
    parser.add_argument(
        "--geometry-offset",
        type=int,
        default=0,
        help="Skip this many entries in the deterministic shuffled geometry pool.",
    )
    parser.add_argument(
        "--case-id-offset",
        type=int,
        default=0,
        help="Add this offset to generated case IDs; 2500 starts at case_2501.",
    )
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
    seismic_sites = tuple(args.seismic_sites or SEISMIC_SITES)
    intensity_levels = tuple(args.intensity_levels or DEFAULT_INTENSITY_LEVELS)
    for site in seismic_sites:
        if site not in SEISMIC_SITES:
            raise ValueError(
                f"Unknown seismic site {site!r}; expected one of {list(SEISMIC_SITES)}."
            )
    calibration = None
    record_pairs = None
    if args.intensity_calibration:
        from Calibrate_Intensity import load_calibration

        calibration = load_calibration(args.intensity_calibration)
        record_pairs = eligible_record_pairs(args.set_name, args.max_npts)
        fit = calibration.get("fit", {})
        print(
            f"Intensity calibration: {args.intensity_calibration} "
            f"(fitted={fit.get('fitted')}, "
            f"pilot runs={calibration.get('pilot_observation_count')})"
        )
        if not fit.get("fitted"):
            print(f"  WARNING: {fit.get('reason')}")

    cases = load_plan(
        root,
        args.num_cases,
        records,
        args.seed,
        args.max_npts,
        set_name=args.set_name,
        geometry_offset=args.geometry_offset,
        case_id_offset=args.case_id_offset,
        seismic_sites=seismic_sites,
        intensity_levels=intensity_levels,
        records_per_case=args.records_per_case,
        calibration=calibration,
        record_pairs=record_pairs,
    )
    total_runs = sum(len(case_runs(case)) for case in cases)
    print(
        f"Plan: {len(cases)} cases x "
        f"{len(intensity_levels)} intensity level(s) = {total_runs} analyses; "
        f"hazards {list(seismic_sites)}"
    )
    plan_csv = root / "parameter_plan.csv"
    print(f"Plan: {root / 'parameter_plan.json'} ({len(cases)} cases, {len(records)} records)")
    print(f"Plan SHA256 (parameter_plan.csv): {sha256_file(plan_csv)}")
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
