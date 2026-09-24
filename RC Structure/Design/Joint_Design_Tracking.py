"""Read-only, per-physical-joint tracking of a saved regular SMRF design.

Reuses the existing joint-area geometry rule; it does not redesign or promote
category-level capacity-design results to independently solved joint demands.
Base anchorage, nonlinear joint calibration, and detailing drawings are outside
this inventory. Missing evidence remains explicit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from Design.SMRF_Joint_Adapter import physical_joint_inventory
from Design.SMRF_Joints import rectangular_joint_area

SCHEMA = "physical_joint_design_tracking_v1"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def build_joint_design_tracking(record):
    geometry, sections = record["geometry"], record["sections"]
    inventory = physical_joint_inventory(geometry)
    bay_x, bay_y, story_h = (_positive(geometry[k], k) for k in ("bay_x_in", "bay_y_in", "story_h_in"))
    bc, hc, bw, hb = (_positive(sections[k], k) for k in ("b_col_in", "h_col_in", "b_beam_in", "h_beam_in"))
    capacity = record.get("capacity_design") or {}
    joint_design = capacity.get("joints") or {}
    categories = joint_design.get("joints") or {}
    transverse = joint_design.get("joint_transverse") or {}
    qualification = record.get("qualification") or {}
    checks = qualification.get("checks") or []
    slab = record.get("slab_reinforcement") or {}
    slab_layout = bool(slab.get("layout"))
    slab_accepted = slab.get("accepted") is True
    rows = []
    for joint in inventory["joints"]:
        i, j = joint["grid_i"], joint["grid_j"]
        x_edge, y_edge = i in (0, geometry["num_bay_x"]), j in (0, geometry["num_bay_y"])
        kind = "corner" if x_edge and y_edge else "edge_y" if x_edge else "edge_x" if y_edge else "interior"
        level = "roof" if joint["is_roof"] else "floor"
        for axis, depth, width in (("x", hc, bc), ("y", bc, hc)):
            category_id = f"joint_shear/{level}/{kind}/{axis}"
            saved = categories.get(category_id)
            area = rectangular_joint_area(column_depth_in=depth, column_width_in=width,
                                          beam_width_in=bw, beam_center_offset_in=0.)
            issues = []
            area_matches = None
            topology_matches = None
            dcr = None
            if saved is None:
                issues.append("No saved category-level joint shear calculation")
            else:
                saved_area = saved.get("aj_in2")
                area_matches = (isinstance(saved_area, (float, int)) and not isinstance(saved_area, bool)
                                and math.isfinite(saved_area) and math.isclose(area, saved_area, rel_tol=1e-10))
                if not area_matches:
                    issues.append("Saved shear area does not match physical geometry")
                topology_matches = (saved.get("beams_in_direction") == len(joint["beams_" + axis])
                    and saved.get("beams_perpendicular") == len(joint["beams_y" if axis == "x" else "beams_x"]))
                if not topology_matches:
                    issues.append("Saved category beam counts do not match physical connectivity")
                demand, cap = saved.get("vj_kip"), saved.get("phi_vn_kip")
                if (isinstance(demand, (float, int)) and not isinstance(demand, bool) and math.isfinite(demand)
                        and demand >= 0 and isinstance(cap, (float, int)) and not isinstance(cap, bool)
                        and math.isfinite(cap) and cap > 0):
                    dcr = demand / cap
                else:
                    issues.append("Finite nonnegative demand and positive capacity are missing")
                if saved.get("evidence_complete") is not True:
                    issues.append("Saved joint shear evidence is incomplete")
            if not slab_layout or not slab_accepted:
                issues.append("Slab reinforcement/contribution is not qualified; saved demand remains provisional")
            scwb = [c for c in checks if c.get("id") == "scwb"
                    and c.get("location", "").startswith(f"{joint['id']}/{axis}/")]
            if len(scwb) != 2 or any(c.get("status") != "pass" for c in scwb):
                issues.append("Both signed SCWB checks are not established")
            if qualification.get("accepted") is not True:
                issues.append("The source design is not qualified")
            saved_status = ("not_evaluated" if saved is None or saved.get("passes") not in (True, False)
                            else "pass" if saved["passes"] is True else "fail")
            tracking_status = ("inconsistent" if area_matches is False or topology_matches is False
                               else "fail" if saved_status == "fail" or (dcr is not None and dcr > 1.)
                               else "unresolved" if issues or saved_status != "pass" else "evidence_recorded")
            anchorage = ((capacity.get("anchorage") or {}).get("directions") or {}).get(axis)
            rows.append({"id": f"{joint['id']}/{axis}", "joint_id": joint["id"], "node_tag": joint["node_tag"],
                "floor": joint["floor"], "grid_i": i, "grid_j": j,
                "xyz_in": [i * bay_x, j * bay_y, joint["floor"] * story_h],
                "axis": axis, "level": level, "kind": kind,
                "columns": joint["columns"], "beams_in_axis": joint["beams_" + axis],
                "beams_transverse": joint["beams_y" if axis == "x" else "beams_x"],
                "geometry": {"column_depth_in": depth, "column_width_in": width,
                    "beam_width_in": bw, "beam_depth_in": hb, "beam_center_offset_in": 0.,
                    "offset_basis": "centered all-grid-line regular frame builder",
                    "effective_joint_width_bj_in": area / depth, "effective_shear_area_aj_in2": area,
                    "panel_face_area_in2": depth * hb, "panel_volume_in3": bc * hc * hb,
                    "area_basis": "SMRF_Joints.rectangular_joint_area; Aj = bj * column depth"},
                "saved_category_id": category_id, "saved_category_calculation": saved,
                "demand_resolution": "shared level/plan-category calculation; not node-specific NTHA demand",
                "saved_area_matches_geometry": area_matches, "saved_topology_matches_geometry": topology_matches,
                "saved_shear_status": saved_status, "saved_shear_dcr": dcr,
                "slab_layout_established": slab_layout, "slab_reinforcement_accepted": slab_accepted,
                "saved_slab_inclusion_claim": None if saved is None else saved.get("slab_steel_in_tension_included"),
                "scwb_checks": scwb, "anchorage_category_calculation": anchorage,
                "anchorage_relevant_branch": "through_bars" if len(joint["beams_" + axis]) == 2 else "terminating_hook",
                "hoop_design": transverse.get("hoops"),
                "hoop_relaxation_applied": transverse.get("relaxation_18_8_3_2_applied"),
                "fabrication_paths_verified": None,
                "nonlinear_joint_calibration_status": "not_established_by_design_record",
                "tracking_status": tracking_status, "open_reasons": issues,
                "source_design_accepted": qualification.get("accepted") is True})
    # Snapshot all nested evidence; callers cannot mutate the source through this report.
    return json.loads(json.dumps({"schema_version": SCHEMA,
        "scope": "regular shared-section frame; elevated physical joints, two shear directions",
        "base_connections_included": False, "geometry": geometry, "sections": sections,
        "reinforcement": record.get("reinforcement"),
        "physical_joint_count": inventory["elevated_joint_count"], "directional_row_count": len(rows),
        "tracking_status_counts": dict(Counter(r["tracking_status"] for r in rows)),
        "source_qualification_counts": qualification.get("counts"),
        "source_design_accepted": qualification.get("accepted") is True, "rows": rows}, allow_nan=False))


def markdown_report(report):
    lines = ["# Joint design tracking", "", f"Physical joints: {report['physical_joint_count']}; directional rows: {report['directional_row_count']}.",
        "", "Elevated joints only. Demands below are saved category calculations, repeated at the matching physical joints.",
        "This inventory does not establish nonlinear calibration, fabrication details, or production acceptance.", "",
        "| Saved category | Rows | Aj (in²) | Vj (kip) | phi Vn (kip) | DCR | Saved shear | Tracking |",
        "|---|---:|---:|---:|---:|---:|---|---|"]
    groups = {}
    for row in report["rows"]:
        groups.setdefault(row["saved_category_id"], []).append(row)
    def number(value):
        return "—" if value is None else f"{value:.3f}"
    for key, rows in sorted(groups.items()):
        row = rows[0]
        saved = row["saved_category_calculation"] or {}
        statuses = ", ".join(sorted({r["tracking_status"] for r in rows}))
        lines.append(f"| {key} | {len(rows)} | {number(row['geometry']['effective_shear_area_aj_in2'])} | "
                     f"{number(saved.get('vj_kip'))} | {number(saved.get('phi_vn_kip'))} | "
                     f"{number(row['saved_shear_dcr'])} | {row['saved_shear_status']} | {statuses} |")
    reasons = sorted({reason for row in report["rows"] for reason in row["open_reasons"]})
    lines.extend(["", "Open evidence:", ""] + [f"- {reason}" for reason in reasons])
    lines.extend(["", "Aj is the effective joint shear cross section. Panel face area (column depth × beam depth) is a different geometric quantity.",
                  "Exact bar, hook and crosstie paths are not established by this report.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.design.resolve().parent == args.output_root.resolve():
        raise ValueError("Use a separate output directory to preserve the input design package")
    raw = args.design.read_bytes()
    report = build_joint_design_tracking(json.loads(raw))
    report["source_design_path"] = str(args.design.resolve())
    report["source_design_sha256"] = hashlib.sha256(raw).hexdigest()
    report["tracker_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "joint_design_tracking.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "joint_design_tracking.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("physical_joint_count", "directional_row_count", "tracking_status_counts", "source_design_sha256")}))


if __name__ == "__main__":
    main()
