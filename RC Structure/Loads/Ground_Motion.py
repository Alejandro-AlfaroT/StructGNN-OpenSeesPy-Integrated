asimport csv
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
        accel_file_path = (
            GROUND_MOTION_DIR
            / "processed"
            / f"{record.record_id}_in_per_sec2.txt"
        )
        write_acceleration_file(record, accel_file_path, units="in/sec^2")

    ops.timeSeries(
        "Path",
        series_tag,
        "-dt",
        record.dt_sec,
        "-filePath",
        str(accel_file_path),
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


def summarize_record(record):
    return {
        "record_id": record.record_id,
        "dt_sec": record.dt_sec,
        "npts": record.npts,
        "duration_sec": record.duration_sec,
        "units": record.units,
        "scale_factor": record.scale_factor,
        "pga_g": record.pga_g,
        "pga_in_per_sec2": record.pga_in_per_sec2,
        "source_path": record.source_path,
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
