import json
import re
from pathlib import Path

import openseespy.opensees as ops

import Structure_Parameters as sp
from Data_Generation.Graph_Exporter import collect_element_rows


FAILED_ELEMENT_RE = re.compile(r"element:\s*(\d+)")


def parse_failed_element_tags(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    text = log_path.read_text(errors="ignore")
    tags = []
    seen = set()

    for match in FAILED_ELEMENT_RE.finditer(text):
        tag = int(match.group(1))
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)

    return tags


def _safe_ele_response(ele_tag, *args):
    try:
        response = ops.eleResponse(ele_tag, *args)
    except Exception as exc:
        return {"error": str(exc)}

    if response is None:
        return None

    return list(response)


def _safe_ele_force(ele_tag):
    try:
        return list(ops.eleForce(ele_tag))
    except Exception as exc:
        return {"error": str(exc)}


def _safe_node_coord(node_tag):
    try:
        return list(ops.nodeCoord(node_tag))
    except Exception as exc:
        return {"error": str(exc)}


def _safe_ele_nodes(ele_tag):
    try:
        return list(ops.eleNodes(ele_tag))
    except Exception:
        return []


def _fallback_element_row(ele_tag):
    nodes = _safe_ele_nodes(ele_tag)
    node_i = nodes[0] if len(nodes) >= 1 else None
    node_j = nodes[1] if len(nodes) >= 2 else None

    if ele_tag >= sp.IMK_HINGE_ELEMENT_TAG_BASE:
        element_type = "imk_hinge"
    else:
        element_type = "unknown"

    return {
        "ele_tag": ele_tag,
        "node_i": node_i,
        "node_j": node_j,
        "element_type": element_type,
        "length_in": 0.0 if element_type == "imk_hinge" else None,
        "section_tag": None,
        "transf_tag": None,
        "integration_tag": None,
    }


def collect_failed_element_diagnostics(element_tags):
    element_rows = {row["ele_tag"]: row for row in collect_element_rows()}
    diagnostics = []

    for ele_tag in element_tags:
        element = element_rows.get(ele_tag, _fallback_element_row(ele_tag))
        node_i = element.get("node_i")
        node_j = element.get("node_j")

        diagnostics.append(
            {
                "ele_tag": ele_tag,
                "element_type": element.get("element_type"),
                "node_i": node_i,
                "node_j": node_j,
                "node_i_coord": _safe_node_coord(node_i) if node_i else None,
                "node_j_coord": _safe_node_coord(node_j) if node_j else None,
                "length_in": element.get("length_in"),
                "section_tag": element.get("section_tag"),
                "transf_tag": element.get("transf_tag"),
                "integration_tag": element.get("integration_tag"),
                "local_force": _safe_ele_response(ele_tag, "localForce"),
                "global_force": _safe_ele_force(ele_tag),
                "basic_force": _safe_ele_response(ele_tag, "basicForce"),
                "basic_deformation": _safe_ele_response(ele_tag, "basicDeformation"),
            }
        )

    return diagnostics


def print_failed_element_diagnostics(diagnostics):
    if not diagnostics:
        print("No failed element tags were detected in the OpenSees log.")
        return

    print("\nFailed element diagnostics:")
    for item in diagnostics:
        basic_deformation = item.get("basic_deformation")
        basic_force = item.get("basic_force")
        local_force = item.get("local_force")

        print(
            f"Element {item['ele_tag']}: "
            f"type={item['element_type']}, "
            f"nodes=({item['node_i']}, {item['node_j']}), "
            f"length={item['length_in']} in, "
            f"section={item['section_tag']}, "
            f"transf={item['transf_tag']}, "
            f"integration={item['integration_tag']}"
        )
        print(f"  node_i_coord={item.get('node_i_coord')}")
        print(f"  node_j_coord={item.get('node_j_coord')}")
        print(f"  basic_deformation={basic_deformation}")
        print(f"  basic_force={basic_force}")
        print(f"  local_force={local_force}")


def save_failed_element_diagnostics(path, diagnostics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(diagnostics, file, indent=2)

    return path
