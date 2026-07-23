"""
Data_Generation/Hybrid_Exporter.py
==================================

Compile completed RC Structure NTHA output folders into compact tensor samples
for hybrid GNN + LSTM training.

Expected input folder
---------------------
An NTHA result directory produced by Ground_Motion_Main.py or Main.py, e.g.
outputs/ntha/peer_1, containing:
  - nodes.csv
  - edges.csv
  - elements.csv
  - time_history.csv
  - story_drift_peaks.csv
  - node_envelope.csv
  - element_end_force_envelope.csv
  - global_parameters.json
  - record_summary_x.json
  - record_summary_y.json, optional for X-only runs
  - status.json
  - summary.json

Output
------
hybrid_sample.npz plus hybrid_metadata.json. The NPZ is intentionally neutral:
it can be loaded into PyTorch Geometric, vanilla PyTorch, or any custom hybrid
dataset class without depending on pandas.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ELEMENT_TYPE_ID = {
    "column": 0,
    "beam_x": 1,
    "beam_y": 2,
}

NODE_FEATURE_COLUMNS = [
    "x_in",
    "y_in",
    "z_in",
    "floor",
    "grid_i",
    "grid_j",
    "is_base",
    "is_floor_master",
]

EDGE_ATTR_COLUMNS = [
    "element_type_id",
    "length_in",
    "dir_x",
    "dir_y",
    "dir_z",
    "b_in",
    "h_in",
    "area_in2",
    "fc_ksi",
    "fy_ksi",
    "bar_area_in2",
    "top_bars",
    "bot_bars",
    "side_bars",
]

ELEMENT_FEATURE_COLUMNS = [
    "element_type_id",
    "length_in",
    "dir_x",
    "dir_y",
    "dir_z",
    "b_in",
    "h_in",
    "area_in2",
    "fc_ksi",
    "fy_ksi",
    "bar_area_in2",
    "top_bars",
    "bot_bars",
    "side_bars",
]

TIME_TARGET_COLUMNS = [
    "roof_disp_x_in",
    "roof_disp_y_in",
    "roof_disp_resultant_in",
]

STORY_DRIFT_COLUMNS = [
    "peak_drift_x_ratio",
    "peak_drift_y_ratio",
    "peak_drift_resultant_ratio",
]

NODE_ENVELOPE_COLUMNS = [
    "ux_max",
    "ux_min",
    "ux_abs_max",
    "uy_max",
    "uy_min",
    "uy_abs_max",
    "uz_max",
    "uz_min",
    "uz_abs_max",
    "rx_max",
    "rx_min",
    "rx_abs_max",
    "ry_max",
    "ry_min",
    "ry_abs_max",
    "rz_max",
    "rz_min",
    "rz_abs_max",
]

ELEMENT_FORCE_COLUMNS = [
    "axial_kip_max",
    "axial_kip_min",
    "shear_y_kip_max",
    "shear_y_kip_min",
    "shear_z_kip_max",
    "shear_z_kip_min",
    "torsion_kip_in_max",
    "torsion_kip_in_min",
    "moment_y_kip_in_max",
    "moment_y_kip_in_min",
    "moment_z_kip_in_max",
    "moment_z_kip_in_min",
]

GLOBAL_FEATURE_KEYS = [
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
    "col_bar_area_in2",
    "col_stirrup_bar_size",
    "col_stirrup_legs",
    "col_stirrup_spacing_in",
    "beam_bar_size",
    "beam_top_bars",
    "beam_bot_bars",
    "beam_side_bars",
    "beam_bar_area_in2",
    "beam_stirrup_bar_size",
    "beam_stirrup_legs",
    "beam_stirrup_spacing_in",
    "floor_dead_load_ksf",
    "floor_live_load_ksf",
    "total_floor_gravity_load_kip",
    "element_formulation",
    "imk_apply_to_columns",
    "imk_apply_to_beams",
    "imk_hinge_stiffness_factor",
    "imk_beam_theta_y",
    "imk_column_theta_y",
    "imk_theta_p_pos",
    "imk_theta_pc_pos",
    "imk_theta_u_pos",
    "imk_fmaxfy_pos",
    "imk_fresfy_pos",
]

RECORD_FEATURE_KEYS = [
    "dt_sec",
    "npts",
    "duration_sec",
    "scale_factor",
    "pga_g",
    "pga_in_per_sec2",
]

TARGET_PEAK_COLUMNS = [
    "max_abs_roof_disp_x_in",
    "max_abs_roof_disp_y_in",
    "max_story_drift_resultant_ratio",
    "max_story_drift_resultant_story",
    "analysis_failed",
    "completed_fraction",
    "num_recovery_steps",
    "recovery_rate",
]


def _read_csv(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


def _to_float(value, default=math.nan):
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return default

    lowered = text.lower()
    if lowered in {"true", "yes", "y"}:
        return 1.0
    if lowered in {"false", "no", "n"}:
        return 0.0

    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value, default=-1):
    number = _to_float(value, default=math.nan)
    if math.isnan(number):
        return default
    return int(number)


def _rows_to_array(rows, columns, dtype=np.float32):
    return np.asarray(
        [[_to_float(row.get(column), default=0.0) for column in columns] for row in rows],
        dtype=dtype,
    )


def _global_to_float(key, value):
    if key == "element_formulation":
        text = str(value or "").strip().lower()
        return {"fiber": 0.0, "imk": 1.0}.get(text, 0.0)
    return _to_float(value, default=0.0)


def _json_to_feature_array(data, keys):
    data = data or {}
    return np.asarray([_global_to_float(key, data.get(key)) for key in keys], dtype=np.float32)


def _load_acceleration_from_summary(record_summary, target_steps):
    if not record_summary:
        return np.zeros(target_steps, dtype=np.float32)

    source_path = record_summary.get("source_path")
    if not source_path:
        return np.zeros(target_steps, dtype=np.float32)

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground motion source file not found: {path}")

    accel = np.loadtxt(path, dtype=np.float64)
    accel = np.atleast_1d(accel).astype(np.float32)

    if accel.size >= target_steps:
        return accel[:target_steps]

    padded = np.zeros(target_steps, dtype=np.float32)
    padded[: accel.size] = accel
    return padded


def _status_features(status):
    status = status or {}
    requested = _to_float(status.get("npts_requested"), default=0.0)
    completed = _to_float(status.get("completed_steps"), default=0.0)
    convergence = status.get("convergence_summary") or {}
    return {
        "analysis_failed": 1.0 if status.get("failed") else 0.0,
        "completed_fraction": completed / requested if requested > 0 else 0.0,
        "num_recovery_steps": _to_float(convergence.get("num_recovery_steps"), default=0.0),
        "recovery_rate": _to_float(convergence.get("recovery_rate"), default=0.0),
    }


def _target_peak_array(summary, status):
    drift = (summary or {}).get("max_story_drift_resultant") or {}
    status_values = _status_features(status)
    values = {
        "max_abs_roof_disp_x_in": _to_float((summary or {}).get("max_abs_roof_disp_x_in"), default=0.0),
        "max_abs_roof_disp_y_in": _to_float((summary or {}).get("max_abs_roof_disp_y_in"), default=0.0),
        "max_story_drift_resultant_ratio": _to_float(drift.get("peak_drift_resultant_ratio"), default=0.0),
        "max_story_drift_resultant_story": _to_float(drift.get("story"), default=0.0),
        **status_values,
    }
    return np.asarray([values[column] for column in TARGET_PEAK_COLUMNS], dtype=np.float32)


def _completed_ok(status):
    if not status:
        return False
    requested = _to_int(status.get("npts_requested"), default=0)
    completed = _to_int(status.get("completed_steps"), default=0)
    return not bool(status.get("failed")) and requested > 0 and completed == requested


def required_files_present(ntha_dir):
    ntha_dir = Path(ntha_dir)
    required = [
        "nodes.csv",
        "edges.csv",
        "elements.csv",
        "time_history.csv",
        "story_drift_peaks.csv",
        "node_envelope.csv",
        "element_end_force_envelope.csv",
        "global_parameters.json",
        "record_summary_x.json",
        "status.json",
        "summary.json",
    ]
    missing = [name for name in required if not (ntha_dir / name).exists()]
    return missing


def compile_hybrid_sample(
    ntha_dir,
    output_dir=None,
    require_success=True,
    overwrite=False,
):
    """
    Compile one NTHA result directory into hybrid_sample.npz.

    Parameters
    ----------
    ntha_dir : str or Path
        Completed NTHA output directory.
    output_dir : str or Path, optional
        Destination directory. Defaults to ntha_dir.
    require_success : bool
        If True, skip failed or partial analyses.
    overwrite : bool
        If False and hybrid_sample.npz already exists, keep it.
    """
    ntha_dir = Path(ntha_dir)
    output_dir = Path(output_dir) if output_dir is not None else ntha_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "hybrid_sample.npz"
    metadata_path = output_dir / "hybrid_metadata.json"
    if sample_path.exists() and not overwrite:
        metadata = _read_json(metadata_path, default={}) or {}
        metadata.setdefault("run_name", ntha_dir.name)
        metadata.setdefault("sample_npz", str(sample_path))
        if not metadata_path.exists() or metadata.get("run_name") != ntha_dir.name:
            _write_json(metadata_path, metadata)
        return metadata

    missing = required_files_present(ntha_dir)
    if missing:
        raise FileNotFoundError(f"{ntha_dir} is missing required NTHA files: {missing}")

    nodes = _read_csv(ntha_dir / "nodes.csv")
    edges = _read_csv(ntha_dir / "edges.csv")
    elements = _read_csv(ntha_dir / "elements.csv")
    time_rows = _read_csv(ntha_dir / "time_history.csv")
    story_rows = _read_csv(ntha_dir / "story_drift_peaks.csv")
    node_env_rows = _read_csv(ntha_dir / "node_envelope.csv")
    force_env_rows = _read_csv(ntha_dir / "element_end_force_envelope.csv")

    status = _read_json(ntha_dir / "status.json", default={}) or {}
    summary = _read_json(ntha_dir / "summary.json", default={}) or {}
    global_parameters = _read_json(ntha_dir / "global_parameters.json", default={}) or {}
    record_x = _read_json(ntha_dir / "record_summary_x.json", default={}) or {}
    record_y = _read_json(ntha_dir / "record_summary_y.json", default=None)

    if require_success and not _completed_ok(status):
        raise RuntimeError(f"NTHA run is not a complete successful sample: {ntha_dir}")

    n_steps = len(time_rows)
    if n_steps <= 0:
        raise RuntimeError(f"No time history rows found in {ntha_dir}")

    node_tags = np.asarray([_to_int(row.get("node_tag")) for row in nodes], dtype=np.int64)
    element_tags = np.asarray([_to_int(row.get("ele_tag")) for row in elements], dtype=np.int64)
    edge_element_tags = np.asarray([_to_int(row.get("ele_tag")) for row in edges], dtype=np.int64)

    x = _rows_to_array(nodes, NODE_FEATURE_COLUMNS)
    edge_index = np.asarray(
        [
            [_to_int(row.get("source")) - 1 for row in edges],
            [_to_int(row.get("target")) - 1 for row in edges],
        ],
        dtype=np.int64,
    )
    edge_attr = _rows_to_array(edges, EDGE_ATTR_COLUMNS)
    element_attr = _rows_to_array(elements, ELEMENT_FEATURE_COLUMNS)

    time_seconds = np.asarray([_to_float(row.get("time_sec"), default=0.0) for row in time_rows], dtype=np.float32)
    gm_x = _load_acceleration_from_summary(record_x, n_steps)
    gm_y = _load_acceleration_from_summary(record_y, n_steps)
    ground_motion = np.stack([gm_x, gm_y], axis=1).astype(np.float32)
    response_time_history = _rows_to_array(time_rows, TIME_TARGET_COLUMNS)

    story_drift_peaks = _rows_to_array(story_rows, STORY_DRIFT_COLUMNS)
    story_ids = np.asarray([_to_int(row.get("story")) for row in story_rows], dtype=np.int64)
    node_env_by_tag = {
        _to_int(row.get("node_tag")): row
        for row in node_env_rows
    }
    node_envelope = np.zeros((len(node_tags), len(NODE_ENVELOPE_COLUMNS)), dtype=np.float32)
    node_envelope_mask = np.zeros(len(node_tags), dtype=np.float32)
    for index, tag in enumerate(node_tags):
        row = node_env_by_tag.get(int(tag))
        if row is None:
            continue
        node_envelope[index, :] = _rows_to_array([row], NODE_ENVELOPE_COLUMNS)[0]
        node_envelope_mask[index] = 1.0
    node_envelope_tags = node_tags.copy()

    element_type_lookup = {row["ele_tag"]: ELEMENT_TYPE_ID.get(row.get("element_type"), -1) for row in elements}
    element_force_index = np.asarray(
        [
            [
                _to_int(row.get("ele_tag")),
                _to_int(row.get("node_tag")),
                0 if row.get("end") == "i" else 1,
                element_type_lookup.get(row.get("ele_tag"), ELEMENT_TYPE_ID.get(row.get("element_type"), -1)),
            ]
            for row in force_env_rows
        ],
        dtype=np.int64,
    )
    element_force_envelope = _rows_to_array(force_env_rows, ELEMENT_FORCE_COLUMNS)

    global_features = _json_to_feature_array(global_parameters, GLOBAL_FEATURE_KEYS)
    record_features = np.stack(
        [
            _json_to_feature_array(record_x, RECORD_FEATURE_KEYS),
            _json_to_feature_array(record_y, RECORD_FEATURE_KEYS),
        ],
        axis=0,
    )
    target_peak = _target_peak_array(summary, status)

    np.savez_compressed(
        sample_path,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_tags=node_tags,
        element_tags=element_tags,
        edge_element_tags=edge_element_tags,
        element_attr=element_attr,
        time_seconds=time_seconds,
        ground_motion=ground_motion,
        response_time_history=response_time_history,
        story_ids=story_ids,
        story_drift_peaks=story_drift_peaks,
        node_envelope_tags=node_envelope_tags,
        node_envelope=node_envelope,
        node_envelope_mask=node_envelope_mask,
        element_force_index=element_force_index,
        element_force_envelope=element_force_envelope,
        global_features=global_features,
        record_features=record_features,
        target_peak=target_peak,
    )

    metadata = {
        "schema_version": "hybrid_gnn_lstm_v1",
        "run_name": ntha_dir.name,
        "source_ntha_dir": str(ntha_dir),
        "sample_npz": str(sample_path),
        "num_nodes": int(x.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "num_elements": int(element_attr.shape[0]),
        "num_time_steps": int(n_steps),
        "num_stories": int(len(story_rows)),
        "analysis_failed": bool(status.get("failed")),
        "completed_steps": _to_int(status.get("completed_steps"), default=0),
        "npts_requested": _to_int(status.get("npts_requested"), default=0),
        "record_id_x": record_x.get("record_id"),
        "record_id_y": (record_y or {}).get("record_id"),
        "node_feature_columns": NODE_FEATURE_COLUMNS,
        "edge_attr_columns": EDGE_ATTR_COLUMNS,
        "element_feature_columns": ELEMENT_FEATURE_COLUMNS,
        "ground_motion_columns": ["accel_x_in_per_sec2", "accel_y_in_per_sec2"],
        "response_time_history_columns": TIME_TARGET_COLUMNS,
        "story_drift_peak_columns": STORY_DRIFT_COLUMNS,
        "node_envelope_columns": NODE_ENVELOPE_COLUMNS,
        "node_envelope_mask_description": "1.0 where node envelope was recorded; base nodes are 0.0 when absent from NTHA node_envelope.csv.",
        "element_force_index_columns": ["ele_tag", "node_tag", "end_id", "element_type_id"],
        "element_force_envelope_columns": ELEMENT_FORCE_COLUMNS,
        "global_feature_keys": GLOBAL_FEATURE_KEYS,
        "record_feature_keys": RECORD_FEATURE_KEYS,
        "target_peak_columns": TARGET_PEAK_COLUMNS,
        "summary": {
            "max_abs_roof_disp_x_in": summary.get("max_abs_roof_disp_x_in"),
            "max_abs_roof_disp_y_in": summary.get("max_abs_roof_disp_y_in"),
            "max_story_drift_resultant": summary.get("max_story_drift_resultant"),
            "elapsed_sec": summary.get("elapsed_sec"),
        },
    }
    _write_json(metadata_path, metadata)
    return metadata


def discover_ntha_runs(ntha_root):
    ntha_root = Path(ntha_root)
    if not ntha_root.exists():
        return []
    return sorted(
        path for path in ntha_root.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    )


def compile_ntha_root(
    ntha_root,
    dataset_dir=None,
    require_success=True,
    overwrite=False,
    limit=None,
):
    ntha_root = Path(ntha_root)
    if dataset_dir is None:
        dataset_dir = ntha_root.parent / "hybrid_dataset"
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    runs = discover_ntha_runs(ntha_root)
    if limit is not None:
        runs = runs[:limit]

    for run_dir in runs:
        out_dir = dataset_dir / run_dir.name
        already_compiled = (out_dir / "hybrid_sample.npz").exists()
        row = {
            "run_name": run_dir.name,
            "source_ntha_dir": str(run_dir),
            "sample_dir": str(out_dir),
            "sample_npz": str(out_dir / "hybrid_sample.npz"),
            "compiled": False,
            "skipped": False,
            "error": "",
        }
        try:
            metadata = compile_hybrid_sample(
                run_dir,
                output_dir=out_dir,
                require_success=require_success,
                overwrite=overwrite,
            )
            row.update(
                {
                    "compiled": True,
                    "skipped": already_compiled and not overwrite,
                    "num_nodes": metadata.get("num_nodes"),
                    "num_edges": metadata.get("num_edges"),
                    "num_elements": metadata.get("num_elements"),
                    "num_time_steps": metadata.get("num_time_steps"),
                    "record_id_x": metadata.get("record_id_x"),
                    "record_id_y": metadata.get("record_id_y"),
                    "analysis_failed": metadata.get("analysis_failed"),
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if require_success:
                rows.append(row)
                continue
        rows.append(row)

    fieldnames = [
        "run_name",
        "source_ntha_dir",
        "sample_dir",
        "sample_npz",
        "compiled",
        "skipped",
        "num_nodes",
        "num_edges",
        "num_elements",
        "num_time_steps",
        "record_id_x",
        "record_id_y",
        "analysis_failed",
        "error",
    ]
    _write_csv(dataset_dir / "hybrid_manifest.csv", fieldnames, rows)
    _write_json(dataset_dir / "hybrid_manifest.json", rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile RC Structure NTHA outputs into hybrid GNN/LSTM tensor samples."
    )
    parser.add_argument("--ntha-dir", help="Compile one NTHA run directory.")
    parser.add_argument(
        "--ntha-root",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "ntha"),
        help="Root directory containing NTHA run folders.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Destination dataset directory. Defaults to outputs/hybrid_dataset.",
    )
    parser.add_argument("--allow-failed", action="store_true", help="Compile failed/partial runs too.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing hybrid_sample.npz files.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    require_success = not args.allow_failed

    if args.ntha_dir:
        metadata = compile_hybrid_sample(
            args.ntha_dir,
            output_dir=args.dataset_dir,
            require_success=require_success,
            overwrite=args.overwrite,
        )
        print(json.dumps(metadata, indent=2))
        return

    rows = compile_ntha_root(
        args.ntha_root,
        dataset_dir=args.dataset_dir,
        require_success=require_success,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    compiled = sum(1 for row in rows if row.get("compiled"))
    failed = [row for row in rows if row.get("error")]
    print(f"Compiled {compiled}/{len(rows)} NTHA runs.")
    if failed:
        print(f"Skipped/failed {len(failed)} run(s). See hybrid_manifest.csv.")


if __name__ == "__main__":
    main()
