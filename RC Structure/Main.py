import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure the terminal can render Unicode (φ, etc.) on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Gravity import run_gravity_analysis
from Analysis.Modal import run_modal_analysis
from Analysis.Pushover import run_pushover
from Analysis.Diagnostics import print_failed_element_diagnostics
from Data_Generation.Graph_Exporter import save_analysis_sample
from Loads.Gravity_Loads import apply_gravity_loads
from Loads.Lateral_Loads import apply_lateral_loads
from Model.Build_Model import build_model
from Plotting.Plot_Results import (
    plot_pushover_curve,
    plot_roof_drift_curve,
    save_pushover_curve,
)
from Design.Config import DesignConfig
from RC_Design_Check import get_element_tags, print_summary, run_checks
from Redesign import apply_updates, run_iterative_redesign


FAILURE_ARTIFACTS = (
    "failed_element_diagnostics.json",
    "pushover_failure_snapshot.json",
)


def should_show_plots():
    value = os.environ.get("RC_SHOW_PLOTS", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def clear_failure_artifacts(outputs_dir):
    for filename in FAILURE_ARTIFACTS:
        path = outputs_dir / filename
        if path.exists():
            path.unlink()


def print_gravity_results(results):
    print("Static gravity analysis succeeded")
    print("\nGravity-only roof control node displacement:")
    print(f"Ux = {results['ux']:.6e} in")
    print(f"Uy = {results['uy']:.6e} in")
    print(f"Uz = {results['uz']:.6e} in (diaphragm master UZ is constrained)")
    print("\nGravity-only physical roof floor vertical displacement:")
    print(f"Uz min = {results['roof_floor_uz_min']:.6e} in")
    print(f"Uz max = {results['roof_floor_uz_max']:.6e} in")
    print(f"Uz avg = {results['roof_floor_uz_avg']:.6e} in")


def print_modal_results(modes):
    print("\nModal Properties:")
    for item in modes:
        if not item["valid"]:
            print(
                f"Mode {item['mode']}: "
                f"skipped invalid eigenvalue lambda = {item['lambda']:.12e}"
            )
            continue

        print(
            f"Mode {item['mode']}: "
            f"T = {item['period']:.6f} sec, "
            f"omega = {item['omega']:.6f} rad/sec, "
            f"f = {item['frequency']:.6f} Hz"
        )


def print_pushover_results(results, gravity_results):
    status = results["status"]
    print(f"\nPushover completed {status['completed_steps']} steps")

    if status["failed"]:
        print(f"Pushover failed at step {status['failed_step']}")
        if status.get("failed_element_tags"):
            print(f"Failed element tags: {status['failed_element_tags']}")
        print_failed_element_diagnostics(results.get("failed_element_diagnostics", []))

    ux, uy, uz = results["final_disp"]
    print("\nGravity + lateral roof control node displacement:")
    print(f"Ux = {ux:.6e} in")
    print(f"Uy = {uy:.6e} in")
    print(f"Uz = {uz:.6e} in (diaphragm master UZ is constrained)")

    print("\nIncrement due to lateral load:")
    print(f"dUx = {ux - gravity_results['ux']:.6e} in")
    print(f"dUy = {uy - gravity_results['uy']:.6e} in")
    print(f"dUz = {uz - gravity_results['uz']:.6e} in")

    mechanism = results.get("mechanism_diagnostics", {}).get("summary", {})
    if mechanism:
        print("\nMechanism diagnostics:")
        for target, snapshot in mechanism.get("story_drift_targets", {}).items():
            print(
                f"Roof drift {target}: max story drift = "
                f"{100.0 * snapshot['max_abs_story_drift_ratio']:.3f}% "
                f"between floors {snapshot['max_floor_pair']} "
                f"(story {snapshot['max_story']})"
            )
            ranking = snapshot.get("story_drift_ranking", [])
            if ranking:
                comparison = ", ".join(
                    f"{row['floor_pair']}: "
                    f"{100.0 * row['abs_story_drift_ratio']:.3f}%"
                    for row in ranking
                )
                print(f"  Interstory drift ranking: {comparison}")

        for target in mechanism.get("unreached_story_drift_targets", []):
            print(f"Roof drift {target}: not reached in this analysis")

        max_interstory = mechanism.get("max_interstory_drift_overall")
        if max_interstory:
            print(
                "Peak interstory drift over pushover: "
                f"{100.0 * max_interstory['abs_story_drift_ratio']:.3f}% "
                f"between floors {max_interstory['floor_pair']} "
                f"at roof drift {100.0 * max_interstory['roof_drift_ratio']:.3f}%"
            )

        first_cap = mechanism.get("first_hinge_cap")
        if first_cap:
            first_cap_group = mechanism.get("first_hinge_cap_group", [])
            print(
                "First hinge cap: "
                f"element {first_cap['physical_ele_tag']} "
                f"end {first_cap['end']} {first_cap['rot_dir']} "
                f"at roof drift {100.0 * first_cap['roof_drift_ratio']:.3f}% "
                f"({len(first_cap_group)} hinges at this step)"
            )
        else:
            print("First hinge cap: none reached")

        first_column = mechanism.get("first_column_nominal_yield")
        if first_column:
            print(
                "First column nominal yield: "
                f"element {first_column['ele_tag']} "
                f"end {first_column['end']} "
                f"at roof drift {100.0 * first_column['roof_drift_ratio']:.3f}%"
            )
        else:
            print("First column nominal yield: none detected")

        first_ultimate = mechanism.get("first_hinge_ultimate_exceedance")
        if first_ultimate:
            print(
                "First theta_u exceedance: "
                f"element {first_ultimate['physical_ele_tag']} "
                f"end {first_ultimate['end']} {first_ultimate['rot_dir']} "
                f"at roof drift {100.0 * first_ultimate['roof_drift_ratio']:.3f}%"
            )
        else:
            print("Theta_u exceedance: none detected")

        print(
            "Beam-sway before column mechanism indicator: "
            f"{mechanism.get('beam_sway_before_column_mechanism_indicator')}"
        )


def design_results_to_jsonable(design_results):
    return {str(k): v for k, v in design_results.items()}


def write_design_results(path, design_results):
    with path.open("w") as file:
        json.dump(design_results_to_jsonable(design_results), file, indent=2)


def summarize_design_results(
    design_results,
    phase,
    redesign_ok=None,
    n_iter=None,
    notes=None,
):
    values = list(design_results.values())
    columns = [row for row in values if row["type"] == "column"]
    beams = [row for row in values if row["type"] == "beam"]
    failures = [int(tag) for tag, row in design_results.items() if not row["ok"]]

    def max_value(rows, key, default=0.0):
        return max((row[key] for row in rows), default=default)

    beam_flex_max = max(
        (max(row["dcr_pos"], row["dcr_neg"]) for row in beams),
        default=0.0,
    )

    status = {
        "phase": phase,
        "all_design_checks_pass": len(failures) == 0,
        "num_failed_elements": len(failures),
        "failed_element_tags": failures,
        "max_column_pm_dcr": max_value(columns, "dcr_PM"),
        "max_column_shear_dcr": max_value(columns, "dcr_V"),
        "max_beam_flexure_dcr": beam_flex_max,
        "max_beam_shear_dcr": max_value(beams, "dcr_V"),
        "final_column_reinforcement": {
            "bar_size": sp.COL_BAR_SIZE,
            "top_bars": sp.COL_TOP_BARS,
            "bottom_bars": sp.COL_BOT_BARS,
            "side_bars_per_face": sp.COL_SIDE_BARS,
            "bar_area_in2": sp.COL_BAR_AREA,
        },
        "final_beam_reinforcement": {
            "bar_size": sp.BEAM_BAR_SIZE,
            "top_bars": sp.BEAM_TOP_BARS,
            "bottom_bars": sp.BEAM_BOT_BARS,
            "bar_area_in2": sp.BEAM_BAR_AREA,
        },
    }

    if redesign_ok is not None:
        status["redesign_converged"] = redesign_ok
    if n_iter is not None:
        status["redesign_iterations"] = n_iter
    if notes is not None:
        status["notes"] = notes

    return status


def combine_design_statuses(
    gravity_status,
    pushover_final_status,
    pushover_redesign_history=None,
):
    status = {
        "all_design_checks_pass": (
            gravity_status["all_design_checks_pass"]
            and pushover_final_status["all_design_checks_pass"]
        ),
        "gravity": gravity_status,
        "pushover_final": pushover_final_status,
        "final_column_reinforcement": pushover_final_status["final_column_reinforcement"],
        "final_beam_reinforcement": pushover_final_status["final_beam_reinforcement"],
        "notes": [
            "Gravity checks use the post-redesign gravity-analysis state.",
            "Pushover-final checks use element forces at the final converged pushover state.",
            "Pushover-final checks are not peak-over-history envelopes; add per-step force tracking before treating them as lateral design envelopes.",
            "Shear DCRs are included in pass/fail status, but the current redesign loop only changes longitudinal bars.",
        ],
    }

    if pushover_redesign_history is not None:
        status["pushover_redesign_history"] = pushover_redesign_history
        final_attempt = pushover_redesign_history[-1] if pushover_redesign_history else {}
        final_failed_drift = final_attempt.get("failed_roof_drift_ratio")
        min_required_drift = final_attempt.get(
            "minimum_acceptable_roof_drift_ratio",
            sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO,
        )
        pushover_drift_pass = (
            not final_attempt.get("failed", False)
            or (
                final_failed_drift is not None
                and final_failed_drift >= min_required_drift
            )
        )
        status["pushover_minimum_drift_pass"] = pushover_drift_pass
        status["minimum_acceptable_roof_drift_ratio"] = min_required_drift
        status["overall_acceptance_pass"] = (
            status["all_design_checks_pass"]
            and pushover_drift_pass
        )
        if not pushover_drift_pass:
            status["notes"].append(
                "Pushover failed before the configured minimum roof drift; the design checks pass, but the nonlinear drift acceptance target does not."
            )

    return status


def run_final_pushover_design_check(outputs_dir, pushover_failed=False):
    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    design_results = run_checks(col_tags, beam_x_tags + beam_y_tags)

    print("\nFinal-state design check at end of pushover:")
    print_summary(design_results, verbose=False)

    write_design_results(outputs_dir / "design_check_pushover_final.json", design_results)

    notes = [
        "Demand snapshot taken from the final converged pushover state.",
        "This is a final-state check only, not a peak-over-history envelope.",
    ]
    if pushover_failed:
        notes.append(
            "Pushover did not converge to the requested target; this check is a failed-state snapshot and should not be treated as a valid final lateral design envelope."
        )

    status = summarize_design_results(
        design_results,
        phase="pushover_final",
        notes=notes,
    )

    return status, design_results


def reinforcement_snapshot():
    return {
        "column": {
            "bar_size": sp.COL_BAR_SIZE,
            "top_bars": sp.COL_TOP_BARS,
            "bottom_bars": sp.COL_BOT_BARS,
            "side_bars_per_face": sp.COL_SIDE_BARS,
            "bar_area_in2": sp.COL_BAR_AREA,
        },
        "beam": {
            "bar_size": sp.BEAM_BAR_SIZE,
            "top_bars": sp.BEAM_TOP_BARS,
            "bottom_bars": sp.BEAM_BOT_BARS,
            "bar_area_in2": sp.BEAM_BAR_AREA,
        },
    }


def _next_value(current, values):
    for value in sorted(set(values)):
        if value > current:
            return value
    return None


def _next_count(current, bounds):
    _, upper = bounds
    if current < upper:
        return current + 1
    return None


def _failed_member_classes(failed_element_tags):
    if not failed_element_tags:
        return {"column", "beam"}

    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    columns = set(col_tags)
    beams = set(beam_x_tags + beam_y_tags)
    classes = set()

    for tag in failed_element_tags:
        if tag in columns:
            classes.add("column")
        elif tag in beams:
            classes.add("beam")

    return classes or {"column", "beam"}


def _column_strengthening_update(cfg):
    next_bar = _next_value(sp.COL_BAR_SIZE, cfg.rebar.bar_sizes_col)
    if next_bar is not None:
        return {
            "bar_size": next_bar,
            "n_top": sp.COL_TOP_BARS,
            "n_bot": sp.COL_BOT_BARS,
            "n_side": sp.COL_SIDE_BARS,
        }

    next_top = _next_count(sp.COL_TOP_BARS, cfg.rebar.col_n_top_range)
    if next_top is not None:
        return {
            "bar_size": sp.COL_BAR_SIZE,
            "n_top": next_top,
            "n_bot": next_top,
            "n_side": sp.COL_SIDE_BARS,
        }

    next_side = _next_value(sp.COL_SIDE_BARS, cfg.rebar.col_n_side_options)
    if next_side is not None:
        return {
            "bar_size": sp.COL_BAR_SIZE,
            "n_top": sp.COL_TOP_BARS,
            "n_bot": sp.COL_BOT_BARS,
            "n_side": next_side,
        }

    return None


def _beam_strengthening_update(cfg):
    next_bar = _next_value(sp.BEAM_BAR_SIZE, cfg.rebar.bar_sizes_beam)
    if next_bar is not None:
        return {
            "bar_size": next_bar,
            "n_top": sp.BEAM_TOP_BARS,
            "n_bot": sp.BEAM_BOT_BARS,
        }

    next_top = _next_count(sp.BEAM_TOP_BARS, cfg.rebar.beam_n_range)
    next_bot = _next_count(sp.BEAM_BOT_BARS, cfg.rebar.beam_n_range)
    if next_top is not None or next_bot is not None:
        return {
            "bar_size": sp.BEAM_BAR_SIZE,
            "n_top": next_top if next_top is not None else sp.BEAM_TOP_BARS,
            "n_bot": next_bot if next_bot is not None else sp.BEAM_BOT_BARS,
        }

    return None


def apply_pushover_failure_redesign(pushover_results, cfg):
    failed_tags = pushover_results["status"].get("failed_element_tags", [])
    classes = _failed_member_classes(failed_tags)
    before = reinforcement_snapshot()
    col_update = _column_strengthening_update(cfg) if "column" in classes else None
    beam_update = _beam_strengthening_update(cfg) if "beam" in classes else None

    if col_update is None and beam_update is None:
        return {
            "changed": False,
            "reason": "No stronger longitudinal reinforcement option is available in the configured search space.",
            "failed_member_classes": sorted(classes),
            "before": before,
            "after": before,
        }

    apply_updates(col_update, beam_update)
    after = reinforcement_snapshot()

    return {
        "changed": before != after,
        "reason": "Pushover failed before the minimum acceptable roof drift; strengthened affected member class(es).",
        "failed_member_classes": sorted(classes),
        "column_update": col_update,
        "beam_update": beam_update,
        "before": before,
        "after": after,
    }


def _pushover_failure_roof_drift(pushover_results):
    status = pushover_results["status"]
    if status.get("failed_roof_disp") is not None:
        return status["failed_roof_disp"] / (sp.NUM_FLOOR * sp.STORY_H)
    if pushover_results["roof_drift"]:
        return pushover_results["roof_drift"][-1]
    return 0.0


def needs_pushover_failure_redesign(pushover_results):
    status = pushover_results["status"]
    if not status.get("failed"):
        return False
    return _pushover_failure_roof_drift(pushover_results) < sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO


def pushover_log_path(outputs_dir, attempt_index):
    return outputs_dir / f"opensees_pushover_attempt_{attempt_index}.log"


def run_modal_and_pushover(outputs_dir, attempt_index):
    build_model()
    print("\nModel rebuilt with final steel for modal/pushover.")

    apply_gravity_loads()
    gravity_results = run_gravity_analysis()

    modal_results = run_modal_analysis()
    print_modal_results(modal_results)

    apply_lateral_loads()
    pushover_results = run_pushover(log_path=pushover_log_path(outputs_dir, attempt_index))
    print_pushover_results(pushover_results, gravity_results)

    return gravity_results, modal_results, pushover_results


def main():
    t_start = time.perf_counter()
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    project_dir = Path(__file__).resolve().parent
    outputs_dir = project_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    clear_failure_artifacts(outputs_dir)

    # ── Iterative steel redesign (gravity-driven inner loop) ─────────────────
    cfg = DesignConfig.from_structure_parameters()
    redesign_ok, n_iter, gravity_results, design_results = run_iterative_redesign(
        cfg=cfg, max_iter=10, verbose=True
    )
    if not redesign_ok:
        print(f"\n[WARNING] Steel redesign did not converge in {n_iter} iterations. "
              "Proceeding with last iteration's design.")
    print_gravity_results(gravity_results)

    # Design check after convergence
    col_fail, beam_fail = print_summary(design_results, verbose=False)
    gravity_design_status = summarize_design_results(
        design_results,
        phase="gravity",
        redesign_ok=redesign_ok,
        n_iter=n_iter,
        notes=[
            "Demand snapshot taken from the gravity-analysis state after iterative redesign.",
            "Shear DCRs are included in pass/fail status, but the current redesign loop only changes longitudinal bars.",
        ],
    )
    write_design_results(outputs_dir / "design_check_gravity.json", design_results)
    write_design_results(outputs_dir / "design_check.json", design_results)
    with (outputs_dir / "design_status_gravity.json").open("w") as file:
        json.dump(gravity_design_status, file, indent=2)

    if not gravity_design_status["all_design_checks_pass"]:
        print(
            "[WARNING] Gravity design checks still fail: "
            f"{gravity_design_status['num_failed_elements']} element(s), "
            f"max column shear DCR={gravity_design_status['max_column_shear_dcr']:.3f}, "
            f"max column P-M DCR={gravity_design_status['max_column_pm_dcr']:.3f}, "
            f"max beam flexure DCR={gravity_design_status['max_beam_flexure_dcr']:.3f}."
        )

    # ── One-time modal + pushover after steel has converged ──────────────────
    # rebuild model with final steel configuration (run_iterative_redesign leaves
    # the OpenSees domain in a post-gravity state; rebuild cleanly for pushover)
    build_model()
    print("\nModel rebuilt with final steel for modal/pushover.")

    apply_gravity_loads()
    gravity_results = run_gravity_analysis()

    modal_results = run_modal_analysis()
    print_modal_results(modal_results)

    apply_lateral_loads()
    pushover_results = run_pushover(log_path=pushover_log_path(outputs_dir, 0))
    print_pushover_results(pushover_results, gravity_results)
    pushover_redesign_history = [
        {
            "attempt_index": 0,
            "reinforcement": reinforcement_snapshot(),
            "completed_steps": pushover_results["status"]["completed_steps"],
            "failed": pushover_results["status"]["failed"],
            "failed_step": pushover_results["status"]["failed_step"],
            "failed_roof_drift_ratio": (
                _pushover_failure_roof_drift(pushover_results)
                if pushover_results["status"]["failed"]
                else None
            ),
            "minimum_acceptable_roof_drift_ratio": sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO,
            "failed_element_tags": pushover_results["status"].get("failed_element_tags", []),
        }
    ]
    redesign_attempts = 0

    while needs_pushover_failure_redesign(pushover_results):
        if redesign_attempts >= sp.PUSHOVER_REDESIGN_MAX_ATTEMPTS:
            print(
                "[WARNING] Pushover still failed before "
                f"{100.0 * sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO:.1f}% roof drift "
                f"after {sp.PUSHOVER_REDESIGN_MAX_ATTEMPTS} redesign attempt(s)."
            )
            break

        redesign_update = apply_pushover_failure_redesign(pushover_results, cfg)
        pushover_redesign_history[-1]["redesign_update"] = redesign_update

        if not redesign_update["changed"]:
            print(
                "[WARNING] Pushover failed before "
                f"{100.0 * sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO:.1f}% roof drift, "
                "but no stronger reinforcement option was available."
            )
            break

        redesign_attempts += 1
        print(
            "\nRetrying modal/pushover after pushover-failure redesign "
            f"(attempt {redesign_attempts + 1})."
        )
        gravity_results, modal_results, pushover_results = run_modal_and_pushover(
            outputs_dir,
            redesign_attempts,
        )
        pushover_redesign_history.append(
            {
                "attempt_index": redesign_attempts,
                "reinforcement": reinforcement_snapshot(),
                "completed_steps": pushover_results["status"]["completed_steps"],
                "failed": pushover_results["status"]["failed"],
                "failed_step": pushover_results["status"]["failed_step"],
                "failed_roof_drift_ratio": (
                    _pushover_failure_roof_drift(pushover_results)
                    if pushover_results["status"]["failed"]
                    else None
                ),
                "minimum_acceptable_roof_drift_ratio": sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO,
                "failed_element_tags": pushover_results["status"].get("failed_element_tags", []),
            }
        )
    (
        pushover_final_design_status,
        pushover_final_design_results,
    ) = run_final_pushover_design_check(
        outputs_dir,
        pushover_failed=pushover_results["status"]["failed"],
    )
    design_status = combine_design_statuses(
        gravity_design_status,
        pushover_final_design_status,
        pushover_redesign_history=pushover_redesign_history,
    )

    with (outputs_dir / "design_status.json").open("w") as file:
        json.dump(design_status, file, indent=2)

    if not pushover_results["status"]["failed"]:
        clear_failure_artifacts(outputs_dir)

    with (outputs_dir / "pushover_status.json").open("w") as file:
        json.dump(pushover_results["status"], file, indent=2)

    with (outputs_dir / "mechanism_diagnostics.json").open("w") as file:
        json.dump(pushover_results["mechanism_diagnostics"], file, indent=2)

    csv_path = save_pushover_curve(
        pushover_results,
        outputs_dir / "pushover_curve.csv",
    )
    fig, _ = plot_pushover_curve(pushover_results, show=False)
    fig.savefig(outputs_dir / "pushover_curve.png", dpi=200)
    drift_fig, _ = plot_roof_drift_curve(pushover_results, show=False)
    drift_fig.savefig(outputs_dir / "pushover_drift_curve.png", dpi=200)

    sample_summary = save_analysis_sample(
        outputs_dir / "samples" / "baseline_frame",
        gravity_results,
        modal_results,
        pushover_results,
        design_status=design_status,
        design_check_results={
            "gravity": design_results,
            "pushover_final": pushover_final_design_results,
        },
    )

    print(f"\nSaved pushover curve CSV: {csv_path}")
    print(f"Saved pushover curve plot: {outputs_dir / 'pushover_curve.png'}")
    print(f"Saved pushover drift plot: {outputs_dir / 'pushover_drift_curve.png'}")
    print(
        "Saved graph sample: "
        f"{outputs_dir / 'samples' / 'baseline_frame'} "
        f"({sample_summary['num_nodes']} nodes, "
        f"{sample_summary['num_elements']} elements, "
        f"{sample_summary['num_element_end_force_rows']} element-end force rows)"
    )

    if should_show_plots():
        plot_pushover_curve(pushover_results, show=True)
        plot_roof_drift_curve(pushover_results, show=True)

    ops.wipe()

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
