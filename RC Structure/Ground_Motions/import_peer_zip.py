"""
Import a PEER NGA-West2 time-series zip into the RC Structure ground-motion
catalog.

The importer expects a PEER search-result zip containing:
    - _SearchResults.csv
    - horizontal acceleration files (*.AT2)
    - optional velocity/displacement files (*.VT2, *.DT2)

It writes:
    Ground_Motions/raw/*.AT2
    Ground_Motions/processed/*_in_per_sec2.txt
    Ground_Motions/metadata/record_manifest.csv
    Ground_Motions/metadata/record_sets.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import shutil
import sys
import zipfile
from pathlib import Path


GROUND_MOTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = GROUND_MOTION_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Loads.Ground_Motion import (  # noqa: E402
    read_peer_at2,
    significant_duration_5_95,
    write_acceleration_file,
)


MANIFEST_COLUMNS = [
    "record_id",
    "source",
    "database",
    "event_name",
    "event_year",
    "station_name",
    "station_id",
    "component",
    "component_angle_deg",
    "tectonic_region",
    "magnitude",
    "mechanism",
    "rrup_km",
    "rjb_km",
    "rx_km",
    "vs30_m_per_s",
    "site_class",
    "dt_sec",
    "npts",
    "duration_sec",
    "raw_units",
    "processed_units",
    "pga_g",
    "pgv_cm_s",
    "arias_intensity",
    "significant_duration_5_95_sec",
    "scale_factor",
    "raw_file",
    "processed_file",
    "scaled_file",
    "download_url",
    "citation",
    "license_notes",
    "usable",
    "notes",
]


SEARCH_COLUMNS = {
    "result_id": 0,
    "record_sequence_number": 2,
    "scale_factor": 4,
    "duration_5_95": 7,
    "arias_intensity": 8,
    "event_name": 9,
    "event_year": 10,
    "station_name": 11,
    "magnitude": 12,
    "mechanism": 13,
    "rjb": 14,
    "rrup": 15,
    "vs30": 16,
    "horizontal_1": 19,
    "horizontal_2": 20,
    "vertical": 21,
}


def _clean(value):
    return str(value).strip().strip('"')


def _float_or_blank(value):
    value = _clean(value)
    if not value or value == "-":
        return ""
    try:
        return float(value)
    except ValueError:
        return ""


def _site_class(vs30):
    try:
        vs30 = float(vs30)
    except (TypeError, ValueError):
        return ""

    if vs30 >= 1500.0:
        return "A"
    if vs30 >= 760.0:
        return "B"
    if vs30 >= 360.0:
        return "C"
    if vs30 >= 180.0:
        return "D"
    return "E"


def _component_label(filename):
    stem = Path(filename).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[-1]
    return stem


def _component_angle(filename):
    label = _component_label(filename)
    digits = ""
    for char in reversed(label):
        if not char.isdigit():
            break
        digits = char + digits
    if len(digits) == 3:
        return int(digits)
    return ""


def _read_peer_search_rows(zip_file):
    try:
        raw = zip_file.read("_SearchResults.csv")
    except KeyError as exc:
        raise FileNotFoundError("PEER zip is missing _SearchResults.csv") from exc

    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = None

    for index, line in enumerate(lines):
        if line.startswith("Result ID,"):
            header_index = index
            break

    if header_index is None:
        raise ValueError("Could not locate PEER selected-record table header.")

    reader = csv.reader(lines[header_index:])
    next(reader)
    rows = [row for row in reader if row and row[0].strip().isdigit()]

    if not rows:
        raise ValueError("No selected-record rows found in PEER search report.")

    return rows, text


def _read_peer_series_from_zip(zip_file, filename):
    values = []
    raw = zip_file.read(filename)
    text = raw.decode("utf-8", errors="replace")
    data_started = False

    for line in text.splitlines():
        if "NPTS" in line.upper() and "DT" in line.upper():
            data_started = True
            continue
        if not data_started:
            continue
        for token in line.replace(",", " ").split():
            values.append(float(token))

    return values


def _pgv_cm_s(zip_file, velocity_filename):
    if not velocity_filename:
        return ""

    candidate = str(Path(velocity_filename).with_suffix(".VT2"))
    try:
        values = _read_peer_series_from_zip(zip_file, candidate)
    except KeyError:
        return ""

    if not values:
        return ""
    return max(abs(value) for value in values)


def _relative(path):
    return path.relative_to(GROUND_MOTION_DIR).as_posix()


def _manifest_has_records(path):
    if not path.exists():
        return False
    with path.open(newline="") as file:
        rows = list(csv.reader(file))
    return len(rows) > 1


def import_peer_zip(zip_path, overwrite=False):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    raw_dir = GROUND_MOTION_DIR / "raw"
    processed_dir = GROUND_MOTION_DIR / "processed"
    metadata_dir = GROUND_MOTION_DIR / "metadata"
    manifest_path = metadata_dir / "record_manifest.csv"
    sets_path = metadata_dir / "record_sets.csv"

    if _manifest_has_records(manifest_path) and not overwrite:
        raise RuntimeError(
            f"{manifest_path} already contains records. Re-run with --overwrite."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    set_rows = []

    with zipfile.ZipFile(zip_path) as zip_file:
        selected_rows, search_report = _read_peer_search_rows(zip_file)
        available = set(zip_file.namelist())

        search_report_path = metadata_dir / "_SearchResults.csv"
        search_report_path.write_text(search_report, encoding="utf-8")

        for selected in selected_rows:
            result_id = _clean(selected[SEARCH_COLUMNS["result_id"]])
            rsn = _clean(selected[SEARCH_COLUMNS["record_sequence_number"]])
            scale_factor = _float_or_blank(selected[SEARCH_COLUMNS["scale_factor"]])
            event_name = _clean(selected[SEARCH_COLUMNS["event_name"]])
            event_year = _clean(selected[SEARCH_COLUMNS["event_year"]])
            station_name = _clean(selected[SEARCH_COLUMNS["station_name"]])
            magnitude = _float_or_blank(selected[SEARCH_COLUMNS["magnitude"]])
            mechanism = _clean(selected[SEARCH_COLUMNS["mechanism"]])
            rjb = _float_or_blank(selected[SEARCH_COLUMNS["rjb"]])
            rrup = _float_or_blank(selected[SEARCH_COLUMNS["rrup"]])
            vs30 = _float_or_blank(selected[SEARCH_COLUMNS["vs30"]])
            arias_intensity = _float_or_blank(
                selected[SEARCH_COLUMNS["arias_intensity"]]
            )
            peer_duration = _float_or_blank(selected[SEARCH_COLUMNS["duration_5_95"]])
            vertical_file = _clean(selected[SEARCH_COLUMNS["vertical"]])

            for component_index, column in enumerate(
                (SEARCH_COLUMNS["horizontal_1"], SEARCH_COLUMNS["horizontal_2"]),
                start=1,
            ):
                filename = _clean(selected[column])
                if filename not in available:
                    raise FileNotFoundError(f"{filename} is listed but not in zip.")

                raw_path = raw_dir / filename
                processed_path = processed_dir / f"{Path(filename).stem}_in_per_sec2.txt"

                with zip_file.open(filename) as src, raw_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                record = read_peer_at2(
                    raw_path,
                    record_id=Path(filename).stem,
                    scale_factor=scale_factor if scale_factor != "" else 1.0,
                )
                write_acceleration_file(record, processed_path, units="in/sec^2")

                duration_5_95 = peer_duration
                if duration_5_95 == "" or not math.isfinite(float(duration_5_95)):
                    duration_5_95 = significant_duration_5_95(record)

                row = {
                    "record_id": record.record_id,
                    "source": "PEER",
                    "database": "NGA-West2",
                    "event_name": event_name,
                    "event_year": event_year,
                    "station_name": station_name,
                    "station_id": f"RSN{rsn}",
                    "component": _component_label(filename),
                    "component_angle_deg": _component_angle(filename),
                    "tectonic_region": "",
                    "magnitude": magnitude,
                    "mechanism": mechanism,
                    "rrup_km": rrup,
                    "rjb_km": rjb,
                    "rx_km": "",
                    "vs30_m_per_s": vs30,
                    "site_class": _site_class(vs30),
                    "dt_sec": record.dt_sec,
                    "npts": record.npts,
                    "duration_sec": record.duration_sec,
                    "raw_units": "g",
                    "processed_units": "in/sec^2",
                    "pga_g": record.pga_g,
                    "pgv_cm_s": _pgv_cm_s(zip_file, filename),
                    "arias_intensity": arias_intensity,
                    "significant_duration_5_95_sec": duration_5_95,
                    "scale_factor": record.scale_factor,
                    "raw_file": _relative(raw_path),
                    "processed_file": _relative(processed_path),
                    "scaled_file": "",
                    "download_url": "",
                    "citation": "PEER NGA-West2 ground motion database",
                    "license_notes": "User-provided PEER download; follow PEER terms of use.",
                    "usable": "True",
                    "notes": (
                        f"Imported from {zip_path.name}; PEER result_id={result_id}; "
                        f"horizontal_component={component_index}; vertical_file={vertical_file}"
                    ),
                }
                manifest_rows.append(row)
                set_rows.append(
                    {
                        "set_name": "peer_mle_all",
                        "record_id": record.record_id,
                        "split": "all",
                        "weight": 1.0,
                        "scale_factor": record.scale_factor,
                        "notes": f"PEER result_id={result_id}; RSN={rsn}",
                    }
                )

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with sets_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["set_name", "record_id", "split", "weight", "scale_factor", "notes"],
        )
        writer.writeheader()
        writer.writerows(set_rows)

    return {
        "zip_path": str(zip_path),
        "num_horizontal_components": len(manifest_rows),
        "num_record_pairs": len(manifest_rows) // 2,
        "manifest_path": str(manifest_path),
        "record_sets_path": str(sets_path),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zip_path",
        nargs="?",
        default=r"C:\Users\andro\Downloads\MLEarthquakeRecords.zip",
        help="Path to the PEER ground-motion zip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest and record-set CSV files.",
    )
    args = parser.parse_args()

    summary = import_peer_zip(args.zip_path, overwrite=args.overwrite)

    print("Imported PEER ground motions")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
