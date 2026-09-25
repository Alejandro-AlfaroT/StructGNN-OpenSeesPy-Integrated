"""Explicit bounded slab-demand refinement, with retained unsuccessful attempts.

Agreement is a numerical screen. It does not resolve the support model,
floor/frame compatibility, membrane reinforcement, or engineering assertions.

A plan is either explicit (``meshes`` with offsets per bay) or a named
recipe (``recipe`` + ``levels`` + ``max_shells``) resolved for the current
bay dimensions and beam face at every call, so the policy in the request
identity is geometry-independent while the resolved coordinates and their
hashes travel with the evidence. A recipe whose affordable levels are fewer
than two yields an explicit ``unresolved_budget`` result: no solve, no
coarsening, no pass.
"""
from __future__ import annotations

import copy
import math
import time

from Design.SMRF_Floor_Mesh import floor_mesh, nested_refinement, resolve_recipe_plan

EXPLICIT_KEYS = {"meshes", "moment_tolerance", "shear_tolerance", "tolerance_basis"}
RECIPE_KEYS = {"recipe", "levels", "max_shells", "moment_tolerance", "shear_tolerance", "tolerance_basis"}


def _validate_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("Slab refinement policy must be a dictionary")
    keys = set(policy)
    if keys == EXPLICIT_KEYS:
        kind = "explicit"
    elif keys == RECIPE_KEYS:
        kind = "recipe"
    elif "meshes" in keys and "recipe" in keys:
        raise ValueError("Slab refinement takes an explicit mesh plan or a named recipe, not both")
    else:
        raise ValueError(f"Slab refinement requires exactly {sorted(EXPLICIT_KEYS)} or {sorted(RECIPE_KEYS)}")
    for key in ("moment_tolerance", "shear_tolerance"):
        v = policy[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 1:
            raise ValueError("Refinement tolerances must be explicit finite numbers between zero and one")
    if not isinstance(policy["tolerance_basis"], str) or not policy["tolerance_basis"].strip():
        raise ValueError("Refinement requires an explicit tolerance_basis")
    return kind


def _plan(geometry, policy, *, sections=None, recipe_inputs=None):
    """Grids of the plan, plus the recipe resolution (None for an explicit plan).

    A recipe resolves from the live ``sections`` (beam width) or, when
    checking a saved record, from the ``recipe_inputs`` it recorded.
    """
    kind = _validate_policy(policy)
    from Design.SMRF_Floor_Analysis import _integer, _number
    nx, ny = (_integer(geometry[k], k) for k in ("num_bay_x", "num_bay_y"))
    lx, ly = (_number(geometry[k], k) for k in ("bay_x_in", "bay_y_in"))
    resolution = None
    if kind == "explicit":
        specs = policy["meshes"]
        if not isinstance(specs, (list, tuple)) or not 2 <= len(specs) <= 8:
            raise ValueError("Bounded refinement requires between two and eight explicit meshes")
    else:
        if recipe_inputs is not None:
            source = {"b_beam_in": recipe_inputs["beam_width_in"]}
            if (recipe_inputs.get("num_bay_x"), recipe_inputs.get("num_bay_y"), recipe_inputs.get("bay_x_in"),
                    recipe_inputs.get("bay_y_in")) != (nx, ny, lx, ly):
                raise ValueError("Recorded recipe inputs disagree with the geometry")
        elif sections is not None:
            source = sections
        else:
            raise ValueError("A recipe plan needs the beam section to resolve")
        resolution = resolve_recipe_plan({"num_bay_x": nx, "num_bay_y": ny, "bay_x_in": lx, "bay_y_in": ly},
                                         source, policy)
        specs = resolution["meshes"]
    grids = [floor_mesh(nx, ny, lx, ly, 4, mesh_spec=spec) for spec in specs]
    for a, b in zip(grids, grids[1:]):
        nested_refinement(a, b)
    return grids, resolution


def refinement_verified(evidence):
    """Recompute the final screen and bind it to the actual returned demands."""
    from Design.SMRF_Slab_Actions import compare_slab_action_refinement
    try:
        report = evidence["refinement"]
        if report["status"] != "passed" or report["all_within_tolerance"] is not True:
            return False
        grids, resolution = _plan(report["geometry"], report["policy"], recipe_inputs=report.get("recipe_inputs"))
        if (resolution is None) != (report.get("resolution") is None):
            return False
        if resolution is not None and (resolution["status"] != "resolved"
                                       or resolution["meshes"] != report["resolved_meshes"]):
            return False
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
    rejected before creating any OpenSees domain; a recipe that does not
    resolve to two affordable levels returns unverified evidence without
    solving anything.
    """
    from Design.SMRF_Slab_Actions import build_slab_action_evidence, compare_slab_action_refinement, _FLAGS
    grids, resolution = _plan(geometry, policy, sections=sections)
    report = {"required": True, "method": "bounded_explicit_slab_refinement_v2_recipes",
              "status": "running", "all_within_tolerance": False, "engineering_verified": False,
              "geometry": copy.deepcopy(geometry), "policy": copy.deepcopy(policy),
              "tolerance_basis": policy["tolerance_basis"], "levels": [], "comparisons": [],
              "resolution": copy.deepcopy(resolution),
              "recipe_inputs": None if resolution is None else copy.deepcopy(resolution["inputs"]),
              "resolved_meshes": None if resolution is None else copy.deepcopy(resolution["meshes"])}
    latest = {flag: False for flag in _FLAGS}
    latest.update(strips=[], cases=[], equilibrium=[], numerical_preconditions={},
                  numerical_basis={}, engineering_assertions=copy.deepcopy(assertions or {}))
    specs = policy["meshes"] if resolution is None else resolution["meshes"]
    if resolution is not None and resolution["status"] != "resolved":
        report["status"] = resolution["status"]
        report["status_detail"] = resolution["detail"]
        specs, grids = [], []
    for index, (spec, grid) in enumerate(zip(specs, grids)):
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
    if report["status"] == "running":
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
