"""Explicit bounded slab-demand refinement, with retained unsuccessful attempts.

Agreement is a numerical screen. It does not resolve the support model,
floor/frame compatibility, membrane reinforcement, or engineering assertions.
"""
from __future__ import annotations

import copy
import math
import time

from Design.SMRF_Floor_Mesh import floor_mesh, nested_refinement


def _plan(geometry, policy):
    required = {"meshes", "moment_tolerance", "shear_tolerance", "tolerance_basis"}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError(f"Slab refinement requires exactly {sorted(required)}")
    for key in ("moment_tolerance", "shear_tolerance"):
        v = policy[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 1:
            raise ValueError("Refinement tolerances must be explicit finite numbers between zero and one")
    if not isinstance(policy["tolerance_basis"], str) or not policy["tolerance_basis"].strip():
        raise ValueError("Refinement requires an explicit tolerance_basis")
    specs = policy["meshes"]
    if not isinstance(specs, (list, tuple)) or not 2 <= len(specs) <= 8:
        raise ValueError("Bounded refinement requires between two and eight explicit meshes")
    from Design.SMRF_Floor_Analysis import _integer, _number
    nx, ny = (_integer(geometry[k], k) for k in ("num_bay_x", "num_bay_y"))
    lx, ly = (_number(geometry[k], k) for k in ("bay_x_in", "bay_y_in"))
    grids = [floor_mesh(nx, ny, lx, ly, 4, mesh_spec=spec) for spec in specs]
    for a, b in zip(grids, grids[1:]):
        nested_refinement(a, b)
    return grids


def refinement_verified(evidence):
    """Recompute the final screen and bind it to the actual returned demands."""
    from Design.SMRF_Slab_Actions import compare_slab_action_refinement
    try:
        report = evidence["refinement"]
        if report["status"] != "passed" or report["all_within_tolerance"] is not True:
            return False
        grids = _plan(report["geometry"], report["policy"])
        levels = report["levels"]
        if len(levels) != len(grids) or any(level["status"] != "completed" for level in levels):
            return False
        for level, grid in zip(levels, grids):
            actions = level["actions"]
            meshes = actions["solved_meshes"]
            if [m["case_id"] for m in meshes] != [c["id"] for c in actions["cases"]]:
                return False
            if not meshes or any(m["mesh"] != grid for m in meshes):
                return False
            if not actions["equilibrium"] or not all(r["numerical_balance_passed"] for r in actions["equilibrium"]):
                return False
        coarse, fine = levels[-2]["actions"], levels[-1]["actions"]
        for key in ("analysis_model_sha256", "physical_model_sha256", "slab_input_sha256", "cases",
                    "strips", "solved_meshes", "equilibrium", "shear_recovery", "max_abs_membrane_kip_per_in"):
            if evidence[key] != fine[key]:
                return False
        comparison = compare_slab_action_refinement(
            coarse, fine, moment_tolerance=report["policy"]["moment_tolerance"],
            shear_tolerance=report["policy"]["shear_tolerance"])
        return comparison == report["comparisons"][-1] and comparison["all_within_tolerance"] is True
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def build_refined_slab_action_evidence(slab_record, geometry, sections, live_load_ksf, slab_inputs,
                                       policy, *, assertions=None, case_observer=None):
    """Run all requested levels, retaining failed comparisons and solve errors.

    ``case_observer(level_index, loadcase, result)`` can persist full raw
    solutions immediately. The returned record always retains attempted
    case summaries, successful demand records and failure reasons. Failed
    fine solves cannot qualify a previous coarse result. Invalid plans are
    rejected before creating any OpenSees domain.
    """
    from Design.SMRF_Slab_Actions import build_slab_action_evidence, compare_slab_action_refinement, _FLAGS
    grids = _plan(geometry, policy)
    report = {"required": True, "method": "bounded_explicit_slab_refinement_v1",
              "status": "running", "all_within_tolerance": False, "engineering_verified": False,
              "geometry": copy.deepcopy(geometry), "policy": copy.deepcopy(policy),
              "tolerance_basis": policy["tolerance_basis"], "levels": [], "comparisons": []}
    latest = {flag: False for flag in _FLAGS}
    latest.update(strips=[], cases=[], equilibrium=[], numerical_preconditions={},
                  numerical_basis={}, engineering_assertions=copy.deepcopy(assertions or {}))
    for index, (spec, grid) in enumerate(zip(policy["meshes"], grids)):
        level = {"index": index, "requested_mesh": grid, "status": "started", "attempted_cases": []}
        report["levels"].append(level)
        start = time.perf_counter()

        def observe(case, result):
            level["attempted_cases"].append({"loadcase": copy.deepcopy(case), **{
                key: copy.deepcopy(result[key]) for key in
                ("status", "error", "analysis_return_code", "mesh", "equilibrium", "transfer_equilibrium")
                if key in result}})
            if case_observer is not None:
                case_observer(index, case, result)

        try:
            actions = build_slab_action_evidence(
                slab_record, geometry, sections, live_load_ksf, slab_inputs,
                assertions=assertions, mesh_spec=spec, case_observer=observe)
            if not actions["solved_meshes"] or any(m["mesh"] != grid for m in actions["solved_meshes"]):
                raise ValueError("Solved mesh provenance differs from the requested refinement mesh")
            level.update(status="completed", actions=actions)
            latest = actions
            if index:
                report["comparisons"].append(compare_slab_action_refinement(
                    report["levels"][index - 1]["actions"], actions,
                    moment_tolerance=policy["moment_tolerance"], shear_tolerance=policy["shear_tolerance"]))
        except Exception as exc:
            level.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["status"] = "analysis_failed"
            break
        finally:
            level["elapsed_seconds"] = time.perf_counter() - start
    if report["status"] != "analysis_failed":
        report["all_within_tolerance"] = report["comparisons"][-1]["all_within_tolerance"]
        report["status"] = "passed" if report["all_within_tolerance"] else "comparison_failed"
    evidence = copy.deepcopy(latest)
    evidence["refinement"] = report
    passed = refinement_verified(evidence)
    evidence["numerical_preconditions"]["mesh_refinement_verified"] = passed
    evidence["numerical_preconditions"]["verified"] = (
        passed and evidence["numerical_preconditions"].get("physical_recovery_valid") is True)
    evidence["numerical_basis"]["mesh_refinement_verified"] = (
        f"Final pair of {len(grids)} declared meshes: {report['status']}; {policy['tolerance_basis']}. "
        "Local strip-demand agreement only; not support-model or engineering verification.")
    evidence["verified"] = (
        evidence.get("assertion_provenance_valid") is True
        and all(evidence["engineering_assertions"].get(flag) is True for flag in _FLAGS)
        and evidence["numerical_preconditions"]["verified"])
    return evidence
