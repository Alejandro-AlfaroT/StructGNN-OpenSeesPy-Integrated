import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

import Structure_Parameters as sp


PEER_NPTS_RE = re.compile(r"NPTS\s*=\s*(\d+)", re.IGNORECASE)
PEER_DT_RE = re.compile(r"DT\s*=\s*([0-9.+\-Ee]+)", re.IGNORECASE)

GROUND_MOTION_DIR = Path(__file__).resolve().parents[1] / "Ground_Motions"
MANIFEST_PATH = GROUND_MOTION_DIR / "metadata" / "record_manifest.csv"
RECORD_SETS_PATH = GROUND_MOTION_DIR / "metadata" / "record_sets.csv"
NPTS_POLICY_OVERRIDE = "allow_npts_policy_excluded=true"
NPTS_POLICY_EXCLUSION = "omitted_from_generation=max_npts_gt_15000"


@dataclass
class GroundMotionRecord:
    record_id: str
    dt_sec: float
    acceleration: np.ndarray
    units: str = "g"
    scale_factor: float = 1.0
    source_path: str | None = None

    @property
    def npts(self):
        return int(self.acceleration.size)

    @property
    def duration_sec(self):
        return self.dt_sec * max(self.npts - 1, 0)

    @property
    def scaled_acceleration(self):
        return self.scale_factor * self.acceleration

    @property
    def pga_g(self):
        return float(np.max(np.abs(_to_g(self.scaled_acceleration, self.units))))

    @property
    def pga_in_per_sec2(self):
        accel = to_in_per_sec2(self.scaled_acceleration, self.units)
        return float(np.max(np.abs(accel)))


def _to_g(acceleration, units):
    units = _normalize_units(units)
    acceleration = np.asarray(acceleration, dtype=float)

    if units == "g":
        return acceleration
    if units == "in/sec^2":
        return acceleration / sp.G
    if units == "cm/sec^2":
        return acceleration / 981.0
    if units == "m/sec^2":
        return acceleration / 9.81

    raise ValueError(f"Unsupported acceleration units: {units}")


def _normalize_units(units):
    value = units.strip().lower().replace(" ", "")

    aliases = {
        "g": "g",
        "grav": "g",
        "gravity": "g",
        "in/s2": "in/sec^2",
        "in/sec2": "in/sec^2",
        "in/sec^2": "in/sec^2",
        "cm/s2": "cm/sec^2",
        "cm/sec2": "cm/sec^2",
        "cm/sec^2": "cm/sec^2",
        "m/s2": "m/sec^2",
        "m/sec2": "m/sec^2",
        "m/sec^2": "m/sec^2",
    }

    if value not in aliases:
        raise ValueError(f"Unsupported acceleration units: {units}")

    return aliases[value]


def to_in_per_sec2(acceleration, units):
    units = _normalize_units(units)
    acceleration = np.asarray(acceleration, dtype=float)

    if units == "g":
        return acceleration * sp.G
    if units == "in/sec^2":
        return acceleration
    if units == "cm/sec^2":
        return acceleration / 2.54
    if units == "m/sec^2":
        return acceleration * 39.37007874015748

    raise ValueError(f"Unsupported acceleration units: {units}")


def read_peer_at2(path, record_id=None, scale_factor=1.0):
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    npts = None
    dt_sec = None
    data_start = None

    for index, line in enumerate(lines):
        npts_match = PEER_NPTS_RE.search(line)
        dt_match = PEER_DT_RE.search(line)

        if npts_match and dt_match:
            npts = int(npts_match.group(1))
            dt_sec = float(dt_match.group(1))
            data_start = index + 1
            break

    if npts is None or dt_sec is None or data_start is None:
        raise ValueError(f"Could not find PEER NPTS/DT header in {path}")

    values = []
    for line in lines[data_start:]:
        for token in line.replace(",", " ").split():
            values.append(float(token))

    if len(values) < npts:
        raise ValueError(f"Expected {npts} acceleration values in {path}, got {len(values)}")

    acceleration = np.asarray(values[:npts], dtype=float)

    return GroundMotionRecord(
        record_id=record_id or path.stem,
        dt_sec=dt_sec,
        acceleration=acceleration,
        units="g",
        scale_factor=scale_factor,
        source_path=str(path),
    )


def read_plain_acceleration(path, dt_sec, units="g", record_id=None, scale_factor=1.0):
    path = Path(path)
    values = []

    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.replace(",", " ").split():
            values.append(float(token))

    if not values:
        raise ValueError(f"No acceleration values found in {path}")

    return GroundMotionRecord(
        record_id=record_id or path.stem,
        dt_sec=dt_sec,
        acceleration=np.asarray(values, dtype=float),
        units=units,
        scale_factor=scale_factor,
        source_path=str(path),
    )


def write_acceleration_file(record, output_path, units="in/sec^2"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _normalize_units(units) == "in/sec^2":
        acceleration = to_in_per_sec2(record.scaled_acceleration, record.units)
    elif _normalize_units(units) == "g":
        acceleration = _to_g(record.scaled_acceleration, record.units)
    else:
        raise ValueError("write_acceleration_file currently supports 'g' and 'in/sec^2'.")

    np.savetxt(output_path, acceleration, fmt="%.10e")
    return output_path


def define_path_time_series(series_tag, record, accel_file_path=None, factor=1.0):
    if accel_file_path is None:
        acceleration = to_in_per_sec2(
            record.scaled_acceleration,
            record.units,
        )
        ops.timeSeries(
            "Path",
            series_tag,
            "-dt",
            record.dt_sec,
            "-values",
            *acceleration.tolist(),
            "-factor",
            factor,
        )
        return None

    ops.timeSeries(
        "Path",
        series_tag,
        "-dt",
        record.dt_sec,
        "-filePath",
        str(Path(accel_file_path)),
        "-factor",
        factor,
    )

    return accel_file_path


def apply_uniform_excitation(pattern_tag, series_tag, direction):
    ops.pattern("UniformExcitation", pattern_tag, direction, "-accel", series_tag)


def load_record_manifest(path=MANIFEST_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def load_record_sets(path=RECORD_SETS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def find_manifest_row(record_id, manifest_path=MANIFEST_PATH):
    for row in load_record_manifest(manifest_path):
        if row.get("record_id") == record_id:
            return row

    raise KeyError(f"Ground motion record_id not found in manifest: {record_id}")


def _manifest_path(row, key):
    value = (row.get(key) or "").strip()
    if not value:
        return None

    path = Path(value)
    if path.is_absolute():
        return path

    return GROUND_MOTION_DIR / path


def _manifest_float(row, key, default=0.0):
    value = (row.get(key) or "").strip()
    if not value:
        return default

    return float(value)


def _manifest_bool(row, key, default=True):
    value = (row.get(key) or "").strip().lower()
    if not value:
        return default

    return value in {"1", "true", "yes", "y"}


def _peer_result_id(row):
    notes = row.get("notes") or ""
    match = re.search(r"PEER result_id=([0-9]+)", notes)
    if match:
        return int(match.group(1))

    return None


def _record_pair_key(row):
    peer_id = _peer_result_id(row)
    if peer_id is not None:
        return f"peer_result_id:{peer_id}"

    return "|".join(
        str(row.get(key) or "").strip()
        for key in ("database", "event_name", "station_id", "station_name")
    )


def _record_pair_sort_key(key):
    """Sort PEER result IDs numerically instead of lexicographically."""
    match = re.fullmatch(r"peer_result_id:([0-9]+)", str(key))
    if match:
        return 0, int(match.group(1))
    return 1, str(key)


def load_ground_motion_record(
    record_id,
    manifest_path=MANIFEST_PATH,
    prefer_processed=True,
    scale_factor=None,
    require_usable=True,
):
    """
    Load a GroundMotionRecord directly from record_manifest.csv.

    Processed catalog files are already converted to the model unit system
    (in/sec^2), so prefer them for OpenSees input. If a processed file is not
    available, this falls back to the raw PEER AT2 acceleration file.
    """

    row = find_manifest_row(record_id, manifest_path)

    if require_usable and not _manifest_bool(row, "usable", default=True):
        raise ValueError(f"Ground motion record is marked unusable: {record_id}")

    if scale_factor is None:
        scale_factor = _manifest_float(row, "scale_factor", default=1.0)

    processed_path = _manifest_path(row, "processed_file")
    raw_path = _manifest_path(row, "raw_file")

    if prefer_processed and processed_path is not None and processed_path.exists():
        return read_plain_acceleration(
            processed_path,
            dt_sec=_manifest_float(row, "dt_sec"),
            units=row.get("processed_units") or "in/sec^2",
            record_id=record_id,
            scale_factor=scale_factor,
        )

    if raw_path is not None and raw_path.exists():
        return read_peer_at2(
            raw_path,
            record_id=record_id,
            scale_factor=scale_factor,
        )

    missing = []
    if processed_path is not None:
        missing.append(str(processed_path))
    if raw_path is not None:
        missing.append(str(raw_path))

    raise FileNotFoundError(
        f"No ground motion file found for {record_id}. Checked: {missing}"
    )


def load_ground_motion_set(
    set_name="peer_mle_all",
    split=None,
    limit=None,
    manifest_path=MANIFEST_PATH,
    record_sets_path=RECORD_SETS_PATH,
    prefer_processed=True,
    scale_factor=None,
):
    """
    Load a named record set as GroundMotionRecord objects.

    record_sets.csv controls grouping/splits, while record_manifest.csv controls
    file paths and physical metadata.
    """

    set_rows = [
        row for row in load_record_sets(record_sets_path)
        if row.get("set_name") == set_name
        and (split is None or row.get("split") == split)
    ]

    if limit is not None:
        set_rows = set_rows[:limit]

    records = []
    for row in set_rows:
        row_scale = row.get("scale_factor")
        records.append(
            load_ground_motion_record(
                row["record_id"],
                manifest_path=manifest_path,
                prefer_processed=prefer_processed,
                scale_factor=(
                    scale_factor
                    if scale_factor is not None
                    else float(row_scale) if row_scale not in (None, "") else None
                ),
            )
        )

    return records


def ground_motion_pair_rows(
    set_name="peer_mle_all",
    split=None,
    manifest_path=MANIFEST_PATH,
    record_sets_path=RECORD_SETS_PATH,
):
    """
    Return paired horizontal manifest rows.

    PEER result_id is used when present. Otherwise rows are grouped by database,
    event, station_id, and station_name. Pairs are sorted by component angle
    where available so the output order is stable.
    """

    manifest_rows = {
        row["record_id"]: row
        for row in load_record_manifest(manifest_path)
    }

    selected_rows = []
    for set_row in load_record_sets(record_sets_path):
        if (
            set_row.get("set_name") != set_name
            or (split is not None and set_row.get("split") != split)
            or set_row.get("record_id") not in manifest_rows
        ):
            continue
        row = dict(manifest_rows[set_row["record_id"]])
        policy_override = (
            NPTS_POLICY_OVERRIDE in (set_row.get("notes") or "").lower()
            and NPTS_POLICY_EXCLUSION in (row.get("notes") or "").lower()
        )
        row["_allow_npts_policy_excluded"] = policy_override
        selected_rows.append(row)

    groups = {}
    for row in selected_rows:
        if (
            not _manifest_bool(row, "usable", default=True)
            and not row.get("_allow_npts_policy_excluded")
        ):
            continue
        groups.setdefault(_record_pair_key(row), []).append(row)

    pairs = []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue

        rows = sorted(
            rows,
            key=lambda item: (
                _manifest_float(item, "component_angle_deg", default=999.0),
                item.get("record_id") or "",
            ),
        )
        pairs.append((key, rows[0], rows[1]))

    return sorted(pairs, key=lambda item: _record_pair_sort_key(item[0]))


def load_ground_motion_pairs(
    set_name="peer_mle_all",
    split=None,
    limit=None,
    start_index=0,
    manifest_path=MANIFEST_PATH,
    record_sets_path=RECORD_SETS_PATH,
    prefer_processed=True,
    scale_factor=None,
):
    pairs = ground_motion_pair_rows(
        set_name=set_name,
        split=split,
        manifest_path=manifest_path,
        record_sets_path=record_sets_path,
    )

    pairs = pairs[start_index:]
    if limit is not None:
        pairs = pairs[:limit]

    loaded = []
    for key, row_x, row_y in pairs:
        loaded.append(
            (
                key,
                load_ground_motion_record(
                    row_x["record_id"],
                    manifest_path=manifest_path,
                    prefer_processed=prefer_processed,
                    scale_factor=scale_factor,
                    require_usable=not row_x.get("_allow_npts_policy_excluded", False),
                ),
                load_ground_motion_record(
                    row_y["record_id"],
                    manifest_path=manifest_path,
                    prefer_processed=prefer_processed,
                    scale_factor=scale_factor,
                    require_usable=not row_y.get("_allow_npts_policy_excluded", False),
                ),
            )
        )

    return loaded


def load_ground_motion_pair_by_result_id(
    result_id,
    set_name="peer_mle_all",
    split=None,
    manifest_path=MANIFEST_PATH,
    record_sets_path=RECORD_SETS_PATH,
    prefer_processed=True,
    scale_factor=None,
):
    target_key = f"peer_result_id:{int(result_id)}"

    for key, record_x, record_y in load_ground_motion_pairs(
        set_name=set_name,
        split=split,
        manifest_path=manifest_path,
        record_sets_path=record_sets_path,
        prefer_processed=prefer_processed,
        scale_factor=scale_factor,
    ):
        if key == target_key:
            return key, record_x, record_y

    raise KeyError(f"No ground motion pair found for PEER result_id={result_id}")


def ground_motion_catalog_summary(
    manifest_path=MANIFEST_PATH,
    record_sets_path=RECORD_SETS_PATH,
):
    rows = load_record_manifest(manifest_path)
    usable = [row for row in rows if _manifest_bool(row, "usable", default=True)]
    pairs = ground_motion_pair_rows(
        manifest_path=manifest_path,
        record_sets_path=record_sets_path,
    )

    def numeric_values(key):
        values = []
        for row in usable:
            value = (row.get(key) or "").strip()
            if value:
                values.append(float(value))
        return values

    pga = numeric_values("pga_g")
    magnitude = numeric_values("magnitude")
    rrup = numeric_values("rrup_km")
    vs30 = numeric_values("vs30_m_per_s")

    return {
        "num_manifest_rows": len(rows),
        "num_usable_records": len(usable),
        "num_record_pairs": len(pairs),
        "num_unique_events": len({row.get("event_name") for row in usable}),
        "pga_g_min": min(pga) if pga else None,
        "pga_g_max": max(pga) if pga else None,
        "magnitude_min": min(magnitude) if magnitude else None,
        "magnitude_max": max(magnitude) if magnitude else None,
        "rrup_km_min": min(rrup) if rrup else None,
        "rrup_km_max": max(rrup) if rrup else None,
        "vs30_m_per_s_min": min(vs30) if vs30 else None,
        "vs30_m_per_s_max": max(vs30) if vs30 else None,
    }


def summarize_record(record):
    source_path = record.source_path
    if source_path:
        try:
            source_path = Path(source_path).resolve().relative_to(
                GROUND_MOTION_DIR.resolve()
            ).as_posix()
        except ValueError:
            source_path = str(source_path)

    return {
        "record_id": record.record_id,
        "dt_sec": record.dt_sec,
        "npts": record.npts,
        "duration_sec": record.duration_sec,
        "units": record.units,
        "scale_factor": record.scale_factor,
        "pga_g": record.pga_g,
        "pga_in_per_sec2": record.pga_in_per_sec2,
        "source_path": source_path,
    }


def significant_duration_5_95(record):
    accel_in_sec2 = to_in_per_sec2(record.scaled_acceleration, record.units)
    arias_increment = accel_in_sec2 * accel_in_sec2 * record.dt_sec
    cumulative = np.cumsum(arias_increment)

    if cumulative.size == 0 or cumulative[-1] <= 0.0:
        return math.nan

    normalized = cumulative / cumulative[-1]
    t5 = np.interp(0.05, normalized, np.arange(record.npts) * record.dt_sec)
    t95 = np.interp(0.95, normalized, np.arange(record.npts) * record.dt_sec)
    return float(t95 - t5)
