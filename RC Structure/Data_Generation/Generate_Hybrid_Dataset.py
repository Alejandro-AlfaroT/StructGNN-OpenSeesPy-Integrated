"""
Data_Generation/Generate_Hybrid_Dataset.py
==========================================

Automation entrypoint for RC Structure hybrid GNN + LSTM data generation.

Two modes are supported:
  1. Compile existing NTHA result folders into hybrid_sample.npz files.
  2. Optionally run Ground_Motion_Main.py first, then compile the new outputs.

Examples
--------
Compile existing successful NTHA outputs:
    python -B Data_Generation/Generate_Hybrid_Dataset.py --compile-existing

Run PEER result IDs 1 and 2, then compile:
    python -B Data_Generation/Generate_Hybrid_Dataset.py --run-ntha --result-id 1 --result-id 2

Run the first 5 pairs from the manifest, then compile:
    python -B Data_Generation/Generate_Hybrid_Dataset.py --run-ntha --limit 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Geometry_Overrides import (
    add_geometry_arguments,
    geometry_cli_args_for_command,
    geometry_name_from_args,
    geometry_overrides_from_args,
)
from Hybrid_Exporter import compile_ntha_root


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _status_success(path):
    status_path = Path(path) / "status.json"
    if not status_path.exists():
        return False
    with status_path.open(encoding="utf-8") as file:
        status = json.load(file)
    requested = int(status.get("npts_requested") or 0)
    completed = int(status.get("completed_steps") or 0)
    return not bool(status.get("failed")) and requested > 0 and completed == requested


def _pair_max_npts(row_x, row_y):
    return max(
        int(float(row_x.get("npts") or 0)),
        int(float(row_y.get("npts") or 0)),
    )


def _all_pair_keys(set_name, split, max_npts):
    # Import lazily so compile-only mode has no OpenSees/manifest dependency.
    sys.path.insert(0, str(_repo_root()))
    from Loads.Ground_Motion import ground_motion_pair_rows

    keys = []
    for key, row_x, row_y in ground_motion_pair_rows(set_name=set_name, split=split):
        match = re.match(r"peer_result_id:([0-9]+)$", key)
        if not match:
            continue

        result_id = int(match.group(1))
        npts = _pair_max_npts(row_x, row_y)
        if max_npts is not None and npts > max_npts:
            continue

        keys.append((key, result_id, npts))

    return sorted(keys, key=lambda item: item[1])


def _load_pair_keys(set_name, split, limit, start_index, max_npts):
    keys = _all_pair_keys(set_name, split, max_npts)
    keys = keys[start_index:]
    if limit is not None:
        keys = keys[:limit]
    return keys


def _run_command(command, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(str(item) for item in command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "returncode": completed.returncode,
        "elapsed_sec": time.perf_counter() - start,
        "log_path": str(log_path),
    }


def run_ntha_batch(args):
    rc_dir = _repo_root()
    ntha_root = Path(args.ntha_root)
    logs_dir = Path(args.dataset_dir) / "run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    max_npts = args.max_npts if args.max_npts and args.max_npts > 0 else None

    if args.result_id:
        allowed = {
            result_id: (key, result_id, npts)
            for key, result_id, npts in _all_pair_keys(args.set_name, args.split, max_npts)
        }
        selected = []
        filtered_ids = []
        for rid in args.result_id:
            if rid in allowed:
                selected.append(allowed[rid])
            else:
                filtered_ids.append(rid)
        if filtered_ids:
            print(
                "Skipping PEER result_id(s) above max_npts="
                f"{max_npts}: {filtered_ids}"
            )
    else:
        selected = _load_pair_keys(
            args.set_name,
            args.split,
            args.limit,
            args.start_index,
            max_npts,
        )

    summaries = []
    for key, result_id, npts in selected:
        run_name = _safe_name(key.replace("peer_result_id:", "peer_"))
        out_dir = ntha_root / run_name
        sample_npz = Path(args.dataset_dir) / run_name / "hybrid_sample.npz"

        if args.skip_existing and _status_success(out_dir) and sample_npz.exists():
            summaries.append(
                {
                    "key": key,
                    "result_id": result_id,
                    "run_name": run_name,
                    "npts": npts,
                    "skipped": True,
                    "reason": "successful NTHA and compiled sample already exist",
                }
            )
            print(f"Skipping {run_name}: existing successful NTHA + hybrid sample.")
            continue

        command = [
            args.python_exe or sys.executable,
            "-B",
            str(rc_dir / "Ground_Motion_Main.py"),
            "--result-id",
            str(result_id),
            "--set-name",
            args.set_name,
            "--output-dir",
            str(ntha_root),
            "--damping-ratio",
            str(args.damping_ratio),
            "--rayleigh-mode-i",
            str(args.rayleigh_mode_i),
            "--rayleigh-mode-j",
            str(args.rayleigh_mode_j),
            "--dt-factor",
            str(args.dt_factor),
        ]
        if args.split:
            command.extend(["--split", args.split])
        if args.x_only:
            command.append("--x-only")
        if args.scale_factor is not None:
            command.extend(["--scale-factor", str(args.scale_factor)])
        if args.catalog_summary:
            command.append("--catalog-summary")
        command.extend(geometry_cli_args_for_command(args))

        print(f"Running NTHA {run_name} ({npts} points) -> {out_dir}")
        run_result = _run_command(command, cwd=rc_dir, log_path=logs_dir / f"{run_name}.log")
        run_result.update(
            {
                "key": key,
                "result_id": result_id,
                "run_name": run_name,
                "npts": npts,
                "output_dir": str(out_dir),
                "skipped": False,
            }
        )
        summaries.append(run_result)
        print(
            f"Finished {run_name}: returncode={run_result['returncode']}, "
            f"elapsed={run_result['elapsed_sec'] / 60.0:.1f} min"
        )

    return summaries


def parse_args():
    rc_dir = _repo_root()
    default_ntha_root = rc_dir / "outputs" / "ntha"
    default_dataset_dir = rc_dir / "outputs" / "hybrid_dataset"

    parser = argparse.ArgumentParser(
        description="Run and/or compile RC Structure NTHA records for hybrid GNN/LSTM training."
    )
    parser.add_argument("--run-ntha", action="store_true", help="Run NTHA before compiling samples.")
    parser.add_argument("--compile-existing", action="store_true", help="Compile existing NTHA folders.")
    parser.add_argument("--result-id", type=int, action="append", help="Specific PEER result_id to run. May be repeated.")
    parser.add_argument("--set-name", default="peer_mle_all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-npts",
        type=int,
        default=15000,
        help=(
            "Omit ground-motion pairs whose longer horizontal component has "
            "more than this many acceleration points. Use 0 to disable."
        ),
    )
    parser.add_argument("--x-only", action="store_true")
    parser.add_argument("--scale-factor", type=float, default=None)
    parser.add_argument("--damping-ratio", type=float, default=0.05)
    parser.add_argument("--rayleigh-mode-i", type=int, default=0)
    parser.add_argument("--rayleigh-mode-j", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--ntha-root", default=str(default_ntha_root))
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--catalog-summary", action="store_true")
    add_geometry_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    geometry_name = geometry_name_from_args(args)
    default_ntha_root = _repo_root() / "outputs" / "ntha"
    default_dataset_dir = _repo_root() / "outputs" / "hybrid_dataset"
    if geometry_name:
        if Path(args.ntha_root) == default_ntha_root:
            args.ntha_root = str(default_ntha_root / geometry_name)
        if Path(args.dataset_dir) == default_dataset_dir:
            args.dataset_dir = str(default_dataset_dir / geometry_name)
    ntha_root = Path(args.ntha_root)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    automation_summary = {
        "run_ntha": args.run_ntha,
        "compile_existing": args.compile_existing or not args.run_ntha,
        "ntha_root": str(ntha_root),
        "dataset_dir": str(dataset_dir),
        "max_npts": args.max_npts if args.max_npts and args.max_npts > 0 else None,
        "geometry_variant": geometry_name or "baseline",
        "geometry_overrides": geometry_overrides_from_args(args),
        "ntha_runs": [],
        "compiled_samples": [],
    }

    if args.run_ntha:
        automation_summary["ntha_runs"] = run_ntha_batch(args)

    if args.compile_existing or not args.run_ntha or args.run_ntha:
        rows = compile_ntha_root(
            ntha_root,
            dataset_dir=dataset_dir,
            require_success=not args.allow_failed,
            overwrite=args.overwrite,
        )
        automation_summary["compiled_samples"] = rows
        print(f"Hybrid manifest: {dataset_dir / 'hybrid_manifest.csv'}")

    with (dataset_dir / "automation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(automation_summary, file, indent=2)
    print(f"Automation summary: {dataset_dir / 'automation_summary.json'}")


if __name__ == "__main__":
    main()
