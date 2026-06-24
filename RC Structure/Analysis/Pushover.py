import json
from pathlib import Path

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Constraints import apply_analysis_constraints
from Analysis.Diagnostics import (
    collect_failed_element_diagnostics,
    parse_failed_element_tags,
)
from Analysis.Mechanism_Checks import MechanismTracker
from Model.nodes import node_tag, roof_master_node


def _base_shear_x():
    ops.reactions()
    shear = 0.0

    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            base_node = node_tag(0, i, j)
            shear += ops.nodeReaction(base_node, 1)

    return -shear


def _restore_default_algorithm():
    ops.test("NormDispIncr", sp.PUSHOVER_TOL, sp.PUSHOVER_MAX_ITER)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", roof_master_node(), 1, sp.PUSHOVER_DU)


def _run_recovery_step(roof_node, algorithm, tolerance, max_iter, du_factor=1.0):
    ops.test("NormDispIncr", tolerance, max_iter)
    ops.algorithm(*algorithm)
    ops.integrator("DisplacementControl", roof_node, 1, sp.PUSHOVER_DU * du_factor)
    try:
        ok = ops.analyze(1)
    finally:
        _restore_default_algorithm()
    return ok


def _try_recovery_step(roof_node):
    strategies = [
        ("NewtonRelaxed", ("Newton",), sp.PUSHOVER_FALLBACK_TOL, sp.PUSHOVER_FALLBACK_MAX_ITER, 1.0),
    ]

    for name, algorithm, tolerance, max_iter, du_factor in strategies:
        ok = _run_recovery_step(roof_node, algorithm, tolerance, max_iter, du_factor)
        if ok == 0:
            return ok, name

    return ok, strategies[-1][0]


def _roof_drift_ratio(roof_disp):
    return roof_disp / (sp.NUM_FLOOR * sp.STORY_H)


def _start_opensees_log(log_path):
    if log_path is None:
        return None

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        try:
            log_path.unlink()
        except PermissionError:
            stem = log_path.stem
            suffix = log_path.suffix
            for idx in range(1, 1000):
                candidate = log_path.with_name(f"{stem}_{idx}{suffix}")
                if not candidate.exists():
                    log_path = candidate
                    break
            else:
                raise

    ops.logFile(str(log_path))
    return log_path


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(data, file, indent=2)

    return path


def _write_failure_snapshot(log_path, status):
    if log_path is None:
        return None

    snapshot_path = Path(log_path).with_name("pushover_failure_snapshot.json")
    return _write_json(snapshot_path, status)


def _write_failed_element_diagnostics(log_path, diagnostics):
    if log_path is None:
        return None

    diagnostics_path = Path(log_path).with_name("failed_element_diagnostics.json")
    return _write_json(diagnostics_path, diagnostics)


def _write_progress_snapshot(log_path, status):
    if log_path is None:
        return None

    progress_path = Path(log_path).with_name("pushover_progress.json")
    return _write_json(progress_path, status)


def _write_status_snapshot(log_path, status):
    if log_path is None:
        return None

    status_path = Path(log_path).with_name("pushover_status.json")
    return _write_json(status_path, status)


def run_pushover(debug=True, log_path=None):
    roof_node = roof_master_node()
    log_path = _start_opensees_log(log_path)

    ops.wipeAnalysis()
    ops.system("BandGeneral")
    apply_analysis_constraints()
    ops.numberer("RCM")
    _restore_default_algorithm()
    ops.integrator("DisplacementControl", roof_node, 1, sp.PUSHOVER_DU)
    ops.analysis("Static")

    roof_disp = []
    roof_drift = []
    base_shear = []
    load_factors = []
    convergence = []
    mechanism_tracker = MechanismTracker()
    status = {
        "completed_steps": 0,
        "requested_steps": sp.PUSHOVER_STEPS,
        "target_disp": sp.pushover_target_disp(),
        "failed": False,
        "failed_step": None,
        "failed_roof_disp": None,
        "failed_base_shear": None,
        "failed_element_tags": [],
        "failure_reason": None,
        "opensees_log": str(log_path) if log_path else None,
        "analysis_state": "running",
    }
    _write_progress_snapshot(log_path, status)
    _write_status_snapshot(log_path, status)
    peak_load_factor = 0.0

    for step in range(sp.PUSHOVER_STEPS):
        algorithm_used = "Newton"
        ok = ops.analyze(1)

        if ok != 0:
            ok, algorithm_used = _try_recovery_step(roof_node)

        ux = ops.nodeDisp(roof_node, 1)
        drift = _roof_drift_ratio(ux)
        shear = _base_shear_x()

        if ok != 0:
            failed_element_tags = parse_failed_element_tags(log_path) if log_path else []
            status["failed"] = True
            status["failed_step"] = step
            status["failed_roof_disp"] = ux
            status["failed_base_shear"] = shear
            status["failed_element_tags"] = failed_element_tags
            status["failure_reason"] = "analysis_nonconvergence"
            status["analysis_state"] = "failed"
            status["failure_snapshot"] = str(_write_failure_snapshot(log_path, status))
            _write_status_snapshot(log_path, status)
            print(
                f"Pushover failed at step {step}: "
                f"Uroof={ux:.4f} in, drift={100.0 * drift:.2f}%, "
                f"Vbase={shear:.3f} kip"
            )
            break

        roof_disp.append(ux)
        roof_drift.append(drift)
        base_shear.append(shear)
        load_factors.append(ops.getLoadFactor(2))
        load_factor = load_factors[-1]
        peak_load_factor = max(peak_load_factor, load_factor)
        mechanism_tracker.record_step(step, ux, shear)
        convergence.append(
            {
                "step": step,
                "ok": ok,
                "algorithm": algorithm_used,
                "roof_disp": ux,
                "roof_drift": drift,
                "base_shear": shear,
                "load_factor": load_factor,
            }
        )
        status["completed_steps"] = step + 1
        status["current_roof_disp"] = ux
        status["current_base_shear"] = shear
        status["current_load_factor"] = load_factor

        if (
            sp.PUSHOVER_STOP_ON_LOAD_REVERSAL
            and peak_load_factor >= sp.PUSHOVER_LOAD_REVERSAL_MIN_PEAK_FACTOR
            and load_factor <= sp.PUSHOVER_LOAD_REVERSAL_LIMIT
        ):
            status["failed"] = True
            status["failed_step"] = step
            status["failed_roof_disp"] = ux
            status["failed_base_shear"] = shear
            status["failed_element_tags"] = []
            status["failure_reason"] = "lateral_load_factor_reversal"
            status["analysis_state"] = "failed"
            status["failure_snapshot"] = str(_write_failure_snapshot(log_path, status))
            _write_progress_snapshot(log_path, status)
            _write_status_snapshot(log_path, status)
            print(
                f"Pushover stopped at step {step}: "
                f"load factor reversed to {load_factor:.4f}, "
                f"Uroof={ux:.4f} in, drift={100.0 * drift:.2f}%, "
                f"Vbase={shear:.3f} kip"
            )
            break

        if debug and step % sp.PUSHOVER_DEBUG_EVERY == 0:
            _write_progress_snapshot(log_path, status)
            _write_status_snapshot(log_path, status)
            print(
                f"step={step:04d}, Uroof={ux:.3f} in, "
                f"drift={100.0 * drift:.2f}%, Vbase={shear:.2f} kip, "
                f"lambda={load_factor:.4f}, alg={algorithm_used}"
            )

    if not status["failed"]:
        status["analysis_state"] = "completed"
    _write_progress_snapshot(log_path, status)
    _write_status_snapshot(log_path, status)

    failed_element_diagnostics = collect_failed_element_diagnostics(
        status["failed_element_tags"]
    )

    if status["failed"]:
        diagnostics_path = _write_failed_element_diagnostics(
            log_path,
            failed_element_diagnostics,
        )
        if diagnostics_path is not None:
            status["failed_element_diagnostics"] = str(diagnostics_path)
        status["failure_snapshot"] = str(_write_failure_snapshot(log_path, status))

    return {
        "roof_node": roof_node,
        "roof_disp": roof_disp,
        "roof_drift": roof_drift,
        "base_shear": base_shear,
        "load_factors": load_factors,
        "convergence": convergence,
        "mechanism_diagnostics": mechanism_tracker.results(),
        "failed_element_diagnostics": failed_element_diagnostics,
        "status": status,
        "final_disp": (
            ops.nodeDisp(roof_node, 1),
            ops.nodeDisp(roof_node, 2),
            ops.nodeDisp(roof_node, 3),
        ),
    }
