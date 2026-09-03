"""Re-run one generated case with recorders on its IMK hinges, and plot the
moment-rotation hysteresis.

Generated NTHA output only stores force/rotation ENVELOPES, never the full
moment-rotation history, so proving the IMK hinge model behaves correctly
(elastic slope matches Ke, yield at My, capping at FmaxFy*My, post-capping
degradation) requires re-running the case with element recorders attached.

This rebuilds the model exactly the way Ground_Motion_Main.py does
(build_gravity_modal_state() then run_ntha()) so the model is identical to
the one that produced the dataset, attaches recorders to the requested
hinges, runs the analysis, and plots the result.

Important: hinges only exist where Structure_Parameters enables them.
With IMK_APPLY_TO_COLUMNS = False, column elements have no hinge at all and
their recorders will produce a file containing only a time column -- point
this at beam elements in that configuration.

Usage
-----
    python Record_Hinge_Hysteresis.py \\
        --case-dir "<root>/cases/case_3331" --set-name peer_expansion_30k \\
        --element 266 --element 262

Geometry and result_id are read from the case's own automation_summary.json,
so the rerun matches the original generation run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


RC_DIR = Path(
    os.environ.get("RC_STRUCTURE_DIR", Path(__file__).resolve().parents[1])
).resolve()
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import numpy as np  # noqa: E402
import openseespy.opensees as ops  # noqa: E402

import Structure_Parameters as sp  # noqa: E402
from Analysis.NTHA import run_ntha  # noqa: E402
from Geometry_Overrides import apply_geometry_overrides  # noqa: E402
from Ground_Motion_Main import build_gravity_modal_state  # noqa: E402
from Loads.Ground_Motion import load_ground_motion_pair_by_result_id  # noqa: E402
from Model.IMK_Hinges import (  # noqa: E402
    hinge_element_tag,
    imk_hinge_stiffness_components,
    imk_hinge_thresholds,
    _member_properties,
)

MATERIAL_DIRECTIONS = ((1, "roty"), (2, "rotz"))


def read_case_settings(case_dir: Path):
    """Recover geometry overrides and result_id from a generated case."""
    summary_path = case_dir / "dataset" / "automation_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No automation_summary.json under {case_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overrides = summary.get("geometry_overrides") or {}
    runs = summary.get("ntha_runs") or []
    if not overrides or not runs:
        raise SystemExit(f"{summary_path} is missing geometry_overrides or ntha_runs.")
    return overrides, int(runs[0]["result_id"]), runs[0]["run_name"]


def attach_recorders(output_dir: Path, element_tags):
    for ele_tag in element_tags:
        for end_id in (1, 2):
            hinge_ele = hinge_element_tag(ele_tag, end_id)
            for material_index, label in MATERIAL_DIRECTIONS:
                path = output_dir / f"hinge_{ele_tag}_end{end_id}_{label}.out"
                ops.recorder(
                    "Element", "-file", str(path), "-time",
                    "-ele", hinge_ele, "material", material_index, "stressStrain",
                )
            print(f"  recorders: element {ele_tag} end {end_id} -> hinge {hinge_ele}")


def backbone_reference(member_type: str, length: float):
    """Backbone values the recorded loop should agree with."""
    properties = _member_properties(member_type)
    components = imk_hinge_stiffness_components(member_type, "rot_y", length)
    thresholds = imk_hinge_thresholds(member_type, "rot_y", length)
    return {
        "My": properties["my"],
        "Ke": components["selected_stiffness"],
        "theta_y": components["actual_theta_y"],
        "theta_cap": thresholds["theta_cap"],
        "theta_u": thresholds["theta_u"],
        "FmaxFy": getattr(sp, "IMK_FMAXFY_POS", 1.10),
        "FresFy": getattr(sp, "IMK_FRESFY_POS", sp.IMK_RES_POS),
    }


def plot_recordings(output_dir: Path, reference: dict, figure_path: Path):
    """Plot every recorder that captured a real (non-degenerate) response."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = []
    for path in sorted(output_dir.glob("hinge_*.out")):
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 3:
            print(f"  {path.name}: no material response recorded (hinge absent) -- skipped")
            continue
        moment, rotation = data[:, 1], data[:, 2]
        if np.abs(rotation).max() < 1e-12:
            print(f"  {path.name}: rotation is identically zero (constrained DOF) -- skipped")
            continue
        active.append((path.stem, moment, rotation))

    if not active:
        print("No hinge produced a usable moment-rotation history; nothing to plot.")
        return None

    active.sort(key=lambda item: -np.abs(item[2]).max())
    active = active[:4]

    columns = min(2, len(active))
    rows = (len(active) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(6.5 * columns, 5.0 * rows), squeeze=False
    )

    for axis, (name, moment, rotation) in zip(axes.flat, active):
        axis.plot(rotation * 100.0, moment, lw=0.6, color="tab:blue", alpha=0.85)
        span = np.array([-reference["theta_y"], reference["theta_y"]]) * 1.4
        axis.plot(span * 100.0, span * reference["Ke"], "k--", lw=1.2, label="elastic Ke")
        for sign in (1.0, -1.0):
            axis.axhline(sign * reference["My"], color="tab:red", ls=":", lw=1)
            axis.axhline(
                sign * reference["FmaxFy"] * reference["My"],
                color="tab:orange", ls=":", lw=1,
            )
            axis.axvline(
                sign * reference["theta_cap"] * 100.0, color="tab:green", ls="-.", lw=1
            )
        axis.set_xlabel("Hinge rotation (%)")
        axis.set_ylabel("Hinge moment (kip-in)")
        axis.set_title(
            f"{name}\npeak rotation = {np.abs(rotation).max() * 100:.3f}% "
            f"({np.abs(rotation).max() / reference['theta_y']:.0f} x theta_y)"
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="lower right")

    for axis in axes.flat[len(active):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    print(f"\nWrote {figure_path}")
    return figure_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, help="Generated case directory.")
    parser.add_argument(
        "--set-name",
        required=True,
        help="Record set the case was generated with (e.g. peer_mle_all).",
    )
    parser.add_argument(
        "--element",
        action="append",
        type=int,
        required=True,
        help="Spine element tag whose hinges to record; repeat for several.",
    )
    parser.add_argument(
        "--member-type",
        default="beam",
        help="Member type for the backbone reference values ('beam' or 'column').",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--damping-ratio", type=float, default=0.05)
    parser.add_argument("--rayleigh-mode-i", type=int, default=0)
    parser.add_argument("--rayleigh-mode-j", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    overrides, result_id, run_name = read_case_settings(case_dir)

    output_dir = Path(
        args.output_dir or RC_DIR / "outputs" / f"hinge_hysteresis_{case_dir.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    apply_geometry_overrides(overrides, variant_name=f"{case_dir.name}_hysteresis")

    _, reference_modal_results, *_ = build_gravity_modal_state()
    print(f"Gravity+modal done at {time.perf_counter() - start:.0f}s", flush=True)

    attach_recorders(output_dir, args.element)

    _, record_x, record_y = load_ground_motion_pair_by_result_id(
        result_id, set_name=args.set_name
    )
    print(f"Loaded records: {record_x.record_id} / {record_y.record_id}", flush=True)

    results = run_ntha(
        record_x,
        record_y=record_y,
        damping_ratio=args.damping_ratio,
        modal_results=reference_modal_results,
        rayleigh_mode_i=args.rayleigh_mode_i,
        rayleigh_mode_j=args.rayleigh_mode_j,
        dt_factor=args.dt_factor,
        log_path=output_dir / "opensees_ntha.log",
    )
    ops.wipe()

    status = results["status"]
    print(
        f"NTHA done: {status['completed_steps']}/{status['npts_requested']} steps, "
        f"failed={status['failed']}, elapsed={time.perf_counter() - start:.0f}s",
        flush=True,
    )

    span = sp.BAY_Y if args.member_type != "column" else sp.STORY_H
    reference = backbone_reference(args.member_type, span)
    print(
        "\nBackbone reference: "
        f"My={reference['My']:.1f} kip-in  Ke={reference['Ke']:.3e} kip-in/rad  "
        f"theta_y={reference['theta_y'] * 100:.4f}%  "
        f"theta_cap={reference['theta_cap'] * 100:.3f}%  "
        f"FmaxFy*My={reference['FmaxFy'] * reference['My']:.1f} kip-in"
    )
    plot_recordings(output_dir, reference, output_dir / "hysteresis_plot.png")


if __name__ == "__main__":
    main()
