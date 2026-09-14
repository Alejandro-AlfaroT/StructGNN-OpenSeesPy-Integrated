"""Read-only input inventory; writes dated reports only inside this analysis workspace."""
import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

WORKSPACE = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 32 * 1024 * 1024


def inside(path, root):
    return path == root or root in path.parents


def child(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty relative path is required")
    portable = value.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", portable) or portable.startswith("/"):
        raise ValueError(f"Expected relative path beneath {root}: {value}")
    result = (root / portable).resolve()
    if not inside(result, root):
        raise ValueError(f"Path leaves its configured root: {value}")
    return result


def validate(profile, workspace=WORKSPACE):
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported profile schema_version")
    for key in ("project_root", "python_executable"):
        if not profile.get(key):
            raise ValueError(f"Set {key} in the selected device profile; no paths were guessed")
        if not Path(profile[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute local path")
    project = Path(profile["project_root"]).resolve()
    if not (project / "Data_Generation").is_dir() or not (project / "Structure_Parameters.py").is_file():
        raise ValueError("project_root must be the local RC Structure folder")
    configured_python = Path(profile["python_executable"]).resolve()
    if configured_python != Path(sys.executable).resolve():
        raise ValueError(f"Run this script with the configured interpreter: {configured_python}")
    base_value = profile.get("dataset_base", "outputs")
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError("Set dataset_base to the mounted SSD datasets folder, or outputs for local source data")
    base = Path(base_value).resolve() if Path(base_value).is_absolute() else child(project, base_value)
    if not base.is_dir():
        raise ValueError(f"dataset_base is not a directory: {base}")
    specs = profile.get("datasets", [])
    ids = [s["id"] for s in specs]
    if not specs or len(ids) != len(set(ids)):
        raise ValueError("Provide at least one dataset with unique IDs")
    datasets = [(s, child(base, s["path"])) for s in specs]
    output = child(workspace.resolve(), profile.get("output_root", "outputs"))
    sources = [project, base] + [p for _, p in datasets]
    if profile.get("surrogate_root"):
        sources.append(Path(profile["surrogate_root"]).resolve())
    if any(inside(output, p) for p in sources):
        raise ValueError("Output must be outside all source project/data folders")
    count = profile.get("metadata_sample_cases_per_dataset", 3)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
        raise ValueError("metadata_sample_cases_per_dataset must be 1 through 20")
    return project, base, datasets, output


def read_json(path, evidence):
    before = path.stat()
    if before.st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds preflight size limit: {path}")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while reading: {path}")
    evidence.append({"path": str(path), "bytes": len(raw), "mtime_ns": after.st_mtime_ns,
                     "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw.decode("utf-8-sig"))


def read_if_present(path, evidence, errors):
    if not path.is_file():
        return None
    try:
        return read_json(path, evidence)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return None


def inventory(spec, root, sample_count, evidence):
    result = {"id": spec["id"], "role": spec.get("role"), "path": str(root),
              "exists": root.is_dir(), "errors": [], "warnings": []}
    if not result["exists"]:
        result["warnings"].append("Dataset not present on this device")
        return result
    errors = result["errors"]
    plan = read_if_present(root / "parameter_plan.json", evidence, errors)
    state = read_if_present(root / "generation_state.json", evidence, errors)
    manifest = read_if_present(root / "parameterized_manifest.json", evidence, errors)
    if isinstance(plan, dict):
        result["plan"] = {k: v for k, v in plan.items() if not isinstance(v, (dict, list))}
        result["plan_case_count"] = len(plan["cases"]) if isinstance(plan.get("cases"), list) else None
    if isinstance(state, dict):
        result["generation_state_claims"] = {k: state.get(k) for k in (
            "status", "updated_at", "planned_count", "completed_count", "failed_count", "remaining_count")}
    if isinstance(manifest, list) and all(isinstance(r, dict) for r in manifest):
        result["manifest_row_count"] = len(manifest)
        result["manifest_status_claims"] = dict(collections.Counter(str(r.get("status", "unknown")) for r in manifest))
    cases = sorted(p for p in (root / "cases").glob("case_*") if p.is_dir() and inside(p.resolve(), root))
    result["observed_case_directory_count"] = len(cases)
    claimed = result.get("manifest_status_claims", {}).get("completed")
    if claimed is not None and claimed != len(cases):
        result["warnings"].append("Case-directory count differs from manifest completed count; reconcile runs before interpreting coverage")
    result["sample_selection"] = f"First {sample_count} case directories in lexical order; not a representative QA sample"
    result["sample_metadata"] = []
    for case in cases[:sample_count]:
        for path in sorted(case.glob("dataset/*/hybrid_metadata.json"))[:5]:
            if not inside(path.resolve(), root):
                errors.append(f"Skipped metadata path outside root: {path}")
                continue
            data = read_if_present(path, evidence, errors)
            if isinstance(data, dict):
                item = {"case_id": case.name, "path": str(path.relative_to(root)),
                        "sample_npz_exists": path.with_name("hybrid_sample.npz").is_file()}
                for key in ("schema_version", "run_name", "num_time_steps", "num_stories", "analysis_failed",
                            "completed_steps", "npts_requested", "collapse", "is_inelastic", "record_id_x", "record_id_y"):
                    item[key] = data.get(key)
                result["sample_metadata"].append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    profile_path = Path(args.profile).resolve()
    evidence = []
    profile = read_json(profile_path, evidence)
    project, base, datasets, output = validate(profile)
    started = dt.datetime.now(dt.timezone.utc)
    report = {"schema_version": 1, "scan_kind": "non_atomic_metadata_inventory",
              "started_at": started.isoformat(), "profile": profile,
              "observed_hostname": platform.node(), "platform": platform.platform(),
              "python_executable": sys.executable, "python_version": platform.python_version(),
              "packages": {}, "sources_read": evidence, "datasets": [], "calibration_artifacts": []}
    for name in ("numpy", "scipy", "matplotlib", "pandas", "seaborn", "pyarrow", "openseespy", "torch", "torch-geometric"):
        try:
            report["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        revision = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "--no-optional-locks", "-C", str(project), "status", "--short"], capture_output=True, text=True, timeout=10)
        report["git"] = {"head": revision.stdout.strip() if revision.returncode == 0 else None,
                         "working_changes": status.stdout.splitlines() if status.returncode == 0 else None,
                         "read_error": (revision.stderr + status.stderr).strip() or None}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["git"] = {"read_error": str(exc)}
    for spec, root in datasets:
        report["datasets"].append(inventory(spec, root, profile["metadata_sample_cases_per_dataset"], evidence))
    for value in profile.get("calibration_artifacts", []):
        path = child(base, value)
        errors = []
        data = read_if_present(path, evidence, errors)
        report["calibration_artifacts"].append({"path": str(path), "exists": path.is_file(), "errors": errors,
            "schema_version": data.get("schema_version") if isinstance(data, dict) else None,
            "fit_claims": data.get("fit") if isinstance(data, dict) else None})
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    device = re.sub(r"[^A-Za-z0-9_-]+", "_", profile.get("device_id", platform.node()))[:60]
    analysis_id = started.strftime("%Y%m%dT%H%M%SZ") + "_" + device + "_" + uuid.uuid4().hex[:8]
    destination = output / analysis_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Local analysis readiness", "", f"Device: {device}; role: {profile.get('role')}", "",
             "Metadata inventory only. Counts are not verified usable-run totals. Sources were not frozen atomically.", "",
             "| Dataset | Present | Case folders | Manifest status claims |", "|---|---|---:|---|"]
    for row in report["datasets"]:
        lines.append(f"| {row['id']} | {row['exists']} | {row.get('observed_case_directory_count', '—')} | {row.get('manifest_status_claims', {})} |")
    lines += ["", "## Items to resolve", ""]
    messages = [f"{r['id']}: {msg}" for r in report["datasets"] for msg in r["errors"] + r["warnings"]]
    if report["git"].get("read_error"):
        messages.append("Git provenance was partly unavailable; see preflight.json. No Git settings were changed.")
    missing = [n for n, v in report["packages"].items() if v is None]
    if missing:
        messages.append("Packages not present (not all required for analysis): " + ", ".join(missing))
    lines += ["- " + m for m in messages] or ["No metadata-level readiness issue found in this bounded scan."]
    lines += ["", "Next: reconcile individual runs on the desktop before QA filtering or set comparisons."]
    (destination / "readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(destination))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Preflight could not proceed: {exc}", file=sys.stderr)
        sys.exit(2)
