import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import openseespy.opensees as ops

import Structure_Parameters as sp
from Geometry_Overrides import (
    add_geometry_arguments,
    apply_geometry_from_args,
)
from Analysis.Gravity import run_gravity_analysis
from Analysis.Modal import run_modal_analysis
from Analysis.NTHA import envelope_to_end_force_rows, run_ntha
from Data_Generation.Graph_Exporter import (
    collect_element_rows,
    collect_global_parameters,
    collect_graph_edge_rows,
    collect_node_rows,
)
from Loads.Gravity_Loads import apply_gravity_loads
from Loads.Ground_Motion import (
    ground_motion_catalog_summary,
    load_ground_motion_pair_by_result_id,
    load_ground_motion_pairs,
    load_ground_motion_record,
    summarize_record,
)
from Model.Build_Model import build_model


OUTPUT_IDENTITY_KEYS = (
    "num_bay_x",
    "num_bay_y",
    "num_floor",
    "bay_x_in",
    "bay_y_in",
    "story_h_in",
    "fc_col_ksi",
    "fc_beam_ksi",
    "fy_ksi",
    "b_col_in",
    "h_col_in",
    "b_beam_in",
    "h_beam_in",
    "col_bar_size",
    "col_top_bars",
    "col_bot_bars",
    "col_side_bars",
    "beam_bar_size",
    "beam_top_bars",
    "beam_bot_bars",
    "beam_side_bars",
    "element_formulation",
    "imk_apply_to_columns",
    "imk_apply_to_beams",
    "imk_material_type",
    "imk_hinge_stiffness_mode",
    "imk_hinge_stiffness_factor",
)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _variant_value(value):
    return (
        f"{float(value):g}"
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def analysis_run_name(
    base_name,
    *,
    x_only=False,
    scale_factor=None,
    damping_ratio=0.05,
    rayleigh_mode_i=0,
    rayleigh_mode_j=2,
    dt_factor=1.0,
):
    name = _safe_name(base_name)
    suffixes = []
    if x_only and not name.endswith("__x_only"):
        suffixes.append("x_only")
    if scale_factor is not None and abs(float(scale_factor) - 1.0) > 1.0e-12:
        suffixes.append(f"sf_{_variant_value(scale_factor)}")
    if abs(float(damping_ratio) - 0.05) > 1.0e-12:
        suffixes.append(f"zeta_{_variant_value(damping_ratio)}")
    if int(rayleigh_mode_i) != 0 or int(rayleigh_mode_j) != 2:
        suffixes.append(f"rayleigh_{int(rayleigh_mode_i)}_{int(rayleigh_mode_j)}")
    if abs(float(dt_factor) - 1.0) > 1.0e-12:
        suffixes.append(f"dtf_{_variant_value(dt_factor)}")
    if suffixes:
        name += "__" + "__".join(suffixes)
    return name


def validate_ntha_output_compatibility(output_dir, overwrite_existing=False):
    output_dir = Path(output_dir)
    if overwrite_existing or not output_dir.exists():
        return

    generated_files = (
        output_dir / "status.json",
        output_dir / "summary.json",
        output_dir / "global_parameters.json",
    )
    if not any(path.exists() for path in generated_files):
        return

    parameters_path = output_dir / "global_parameters.json"
    if not parameters_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing NTHA output without configuration "
            f"metadata: {output_dir}. Use a different output directory or "
            "--overwrite-existing."
        )

    with parameters_path.open(encoding="utf-8") as file:
        previous = json.load(file)
    current = collect_global_parameters()
    differences = {
        key: {"existing": previous.get(key), "requested": current.get(key)}
        for key in OUTPUT_IDENTITY_KEYS
        if previous.get(key) != current.get(key)
    }
    if differences:
        preview = ", ".join(
            f"{key}={values['existing']!r}->{values['requested']!r}"
            for key, values in list(differences.items())[:8]
        )
        raise RuntimeError(
            "Refusing to mix different model configurations in "
            f"{output_dir}. Differences: {preview}. "
            "Use a different output directory or --overwrite-existing."
        )


def _story_drift_rows(results):
    rows = []
    for story, (dx, dy) in enumerate(
        zip(results["peak_story_drift_x"], results["peak_story_drift_y"]),
        start=1,
    ):
        rows.append(
            {
                "story": story,
                "floor_pair": f"{story - 1}-{story}",
                "peak_drift_x_ratio": dx,
                "peak_drift_y_ratio": dy,
                "peak_drift_x_percent": 100.0 * dx,
                "peak_drift_y_percent": 100.0 * dy,
                "peak_drift_resultant_ratio": (dx * dx + dy * dy) ** 0.5,
                "peak_drift_resultant_percent": 100.0 * (dx * dx + dy * dy) ** 0.5,
            }
        )
    return rows


def _time_history_rows(results):
    rows = []
    for i, t in enumerate(results["time_history"]):
        ux = results["roof_disp_x"][i]
        uy = results["roof_disp_y"][i]
        rows.append(
            {
                "step": i,
                "time_sec": t,
                "roof_disp_x_in": ux,
                "roof_disp_y_in": uy,
                "roof_disp_resultant_in": (ux * ux + uy * uy) ** 0.5,
            }
        )
    return rows


def _node_envelope_rows(results):
    dof_names = ("ux", "uy", "uz", "rx", "ry", "rz")
    rows = []
    for node_tag, envelope in sorted(
        results["node_envelope"].items(),
        key=lambda item: int(item[0]),
    ):
        row = {"node_tag": int(node_tag)}
        for i, name in enumerate(dof_names):
            row[f"{name}_max"] = envelope["max"][i]
            row[f"{name}_min"] = envelope["min"][i]
            row[f"{name}_abs_max"] = max(
                abs(envelope["max"][i]),
                abs(envelope["min"][i]),
            )
        rows.append(row)
    return rows


def _result_summary(results, output_dir, elapsed_sec):
    drift_rows = _story_drift_rows(results)
    max_drift_x = max(drift_rows, key=lambda row: row["peak_drift_x_ratio"])
    max_drift_y = max(drift_rows, key=lambda row: row["peak_drift_y_ratio"])
    max_drift_resultant = max(
        drift_rows,
        key=lambda row: row["peak_drift_resultant_ratio"],
    )

    roof_x = results["roof_disp_x"]
    roof_y = results["roof_disp_y"]
    status = results["status"]

    return {
        "status": status,
        "record_summary_x": results["record_summary_x"],
        "record_summary_y": results["record_summary_y"],
        "damping_ratio": results["damping_ratio"],
        "rayleigh_a0": results["rayleigh_a0"],
        "rayleigh_a1": results["rayleigh_a1"],
        "elapsed_sec": elapsed_sec,
        "max_abs_roof_disp_x_in": max((abs(v) for v in roof_x), default=0.0),
        "max_abs_roof_disp_y_in": max((abs(v) for v in roof_y), default=0.0),
        "max_story_drift_x": max_drift_x,
        "max_story_drift_y": max_drift_y,
        "max_story_drift_resultant": max_drift_resultant,
        "output_dir": str(output_dir),
    }


def save_ntha_outputs(output_dir, results, gravity_results, modal_results):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_rows = collect_node_rows()
    element_rows = collect_element_rows()
    edge_rows = collect_graph_edge_rows(element_rows)
    end_force_rows = envelope_to_end_force_rows(
        results["element_envelope"],
        element_rows,
    )
    story_rows = _story_drift_rows(results)
    time_rows = _time_history_rows(results)
    node_env_rows = _node_envelope_rows(results)

    _write_json(output_dir / "status.json", results["status"])
    _write_json(output_dir / "record_summary_x.json", results["record_summary_x"])
    if results["record_summary_y"] is not None:
        _write_json(output_dir / "record_summary_y.json", results["record_summary_y"])
    _write_json(output_dir / "gravity_results.json", gravity_results)
    _write_json(output_dir / "modal_results.json", modal_results)
    _write_json(output_dir / "global_parameters.json", collect_global_parameters())
    _write_json(output_dir / "hinge_envelope.json", results["hinge_envelope"])

    _write_csv(output_dir / "nodes.csv", list(node_rows[0].keys()), node_rows)
    _write_csv(output_dir / "elements.csv", list(element_rows[0].keys()), element_rows)
    _write_csv(output_dir / "edges.csv", list(edge_rows[0].keys()), edge_rows)
    _write_csv(output_dir / "time_history.csv", list(time_rows[0].keys()), time_rows)
    _write_csv(output_dir / "story_drift_peaks.csv", list(story_rows[0].keys()), story_rows)
    _write_csv(output_dir / "node_envelope.csv", list(node_env_rows[0].keys()), node_env_rows)
    if end_force_rows:
        _write_csv(
            output_dir / "element_end_force_envelope.csv",
            list(end_force_rows[0].keys()),
            end_force_rows,
        )

    return {
        "num_time_steps": len(time_rows),
        "num_nodes": len(node_rows),
        "num_elements": len(element_rows),
        "num_edges": len(edge_rows),
        "num_element_end_force_envelope_rows": len(end_force_rows),
    }



def first_valid_modal_period(modes):
    for item in modes:
        if item.get("valid") and item.get("period") is not None:
            return item["period"]
    return None


def asce_period_check_from_modal_results(modes, modal_source):
    return sp.asce_fundamental_period_check(
        model_period_sec=first_valid_modal_period(modes),
        modal_source=modal_source,
    )

def build_gravity_modal_state():
    ops.wipe()
    build_model()
    reference_modal_results = run_modal_analysis()
    reference_period_check = asce_period_check_from_modal_results(
        reference_modal_results,
        "elastic_reference_before_gravity",
    )
    apply_gravity_loads()
    gravity_results = run_gravity_analysis()
    post_gravity_modal_results = run_modal_analysis()
    tangent_period_check = asce_period_check_from_modal_results(
        post_gravity_modal_results,
        "post_gravity_tangent_diagnostic",
    )
    return gravity_results, reference_modal_results, post_gravity_modal_results, reference_period_check, tangent_period_check


def run_one(record_x, record_y, args, output_dir):
    gravity_results, reference_modal_results, modal_results, reference_period_check, tangent_period_check = build_gravity_modal_state()

    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start

    file_counts = save_ntha_outputs(
        output_dir,
        results,
        gravity_results,
        reference_modal_results,
    )
    _write_json(output_dir / "modal_results_elastic_reference.json", reference_modal_results)
    _write_json(output_dir / "modal_results_post_gravity_tangent.json", modal_results)
    reference_valid_modes = [mode for mode in reference_modal_results if mode.get("valid")]
    tangent_valid_modes = [mode for mode in modal_results if mode.get("valid")]
    modal_diagnostics = {
        "rayleigh_modal_source": "elastic_reference_before_gravity",
        "elastic_reference_period_mode_1_sec": reference_valid_modes[0]["period"] if len(reference_valid_modes) > 0 else None,
        "elastic_reference_period_mode_2_sec": reference_valid_modes[1]["period"] if len(reference_valid_modes) > 1 else None,
        "post_gravity_tangent_period_mode_1_sec": tangent_valid_modes[0]["period"] if len(tangent_valid_modes) > 0 else None,
        "post_gravity_tangent_period_mode_2_sec": tangent_valid_modes[1]["period"] if len(tangent_valid_modes) > 1 else None,
        "gravity_roof_ux_in": gravity_results.get("ux"),
        "gravity_roof_uy_in": gravity_results.get("uy"),
        "asce_reference_period_check": reference_period_check,
        "asce_post_gravity_period_check": tangent_period_check,
    }
    _write_json(output_dir / "modal_diagnostics.json", modal_diagnostics)

    summary = _result_summary(results, output_dir, elapsed)
    summary.update(file_counts)
    summary.update(modal_diagnostics)
    _write_json(output_dir / "summary.json", summary)

    status = results["status"]
    drift = summary["max_story_drift_resultant"]
    print(
        f"NTHA {record_x.record_id}"
        f"{' + ' + record_y.record_id if record_y else ''}: "
        f"{status['completed_steps']}/{status['npts_requested']} steps, "
        f"failed={status['failed']}, "
        f"max resultant story drift={100.0 * drift['peak_drift_resultant_ratio']:.3f}% "
        f"at story {drift['story']}"
    )

    return summary


def select_records(args):
    if args.record_id_x:
        record_x = load_ground_motion_record(
            args.record_id_x,
            scale_factor=args.scale_factor,
        )
        record_y = None
        if args.record_id_y and not args.x_only:
            record_y = load_ground_motion_record(
                args.record_id_y,
                scale_factor=args.scale_factor,
            )
        return [(f"manual:{record_x.record_id}", record_x, record_y)]

    if args.result_id is not None:
        key, record_x, record_y = load_ground_motion_pair_by_result_id(
            args.result_id,
            set_name=args.set_name,
            split=args.split,
            scale_factor=args.scale_factor,
        )
        if args.x_only:
            record_y = None
        return [(key, record_x, record_y)]

    pairs = load_ground_motion_pairs(
        set_name=args.set_name,
        split=args.split,
        limit=args.limit,
        start_index=args.start_index,
        scale_factor=args.scale_factor,
    )
    if args.x_only:
        pairs = [(key, record_x, None) for key, record_x, _record_y in pairs]
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RC Structure nonlinear time-history analyses from the ground-motion manifest."
    )
    parser.add_argument("--record-id-x", help="Specific manifest record_id for X-direction excitation.")
    parser.add_argument("--record-id-y", help="Specific manifest record_id for Y-direction excitation.")
    parser.add_argument("--result-id", type=int, help="PEER search result_id pair to run.")
    parser.add_argument("--set-name", default="peer_mle_all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--x-only", action="store_true", help="Run only the X component.")
    parser.add_argument("--scale-factor", type=float, default=None)
    parser.add_argument("--damping-ratio", type=float, default=0.05)
    parser.add_argument("--rayleigh-mode-i", type=int, default=0)
    parser.add_argument("--rayleigh-mode-j", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "ntha"),
    )
    parser.add_argument("--catalog-summary", action="store_true")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow replacing an existing run even when its model configuration differs.",
    )
    add_geometry_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    geometry_name = apply_geometry_from_args(args)

    if args.catalog_summary:
        print(json.dumps(ground_motion_catalog_summary(), indent=2))
        if not args.record_id_x and args.result_id is None:
            return

    selected = select_records(args)
    if not selected:
        raise RuntimeError("No ground-motion records selected.")

    base_output_dir = Path(args.output_dir)
    default_output_dir = Path(__file__).resolve().parent / "outputs" / "ntha"
    if geometry_name and base_output_dir == default_output_dir:
        base_output_dir = base_output_dir / geometry_name
    run_summaries = []

    for key, record_x, record_y in selected:
        name = _safe_name(key.replace("peer_result_id:", "peer_"))
        if args.record_id_x:
            name = _safe_name(
                record_x.record_id + (f"__{record_y.record_id}" if record_y else "__x_only")
            )
        name = analysis_run_name(
            name,
            x_only=record_y is None,
            scale_factor=record_x.scale_factor,
            damping_ratio=args.damping_ratio,
            rayleigh_mode_i=args.rayleigh_mode_i,
            rayleigh_mode_j=args.rayleigh_mode_j,
            dt_factor=args.dt_factor,
        )

        output_dir = base_output_dir / name
        validate_ntha_output_compatibility(
            output_dir,
            overwrite_existing=args.overwrite_existing,
        )
        summary = run_one(record_x, record_y, args, output_dir)
        summary["pair_key"] = key
        run_summaries.append(summary)

    _write_json(base_output_dir / "ntha_batch_summary.json", run_summaries)
    print(f"Saved NTHA batch summary: {base_output_dir / 'ntha_batch_summary.json'}")


if __name__ == "__main__":
    main()
