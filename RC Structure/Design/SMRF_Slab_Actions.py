"""Slab strip action evidence from the flexible-beam floor model.

Produces the ``demand_evidence`` that ``SMRF_Slab_Reinforcement`` requires:
one row per (panel, bar axis, face) with the governing factored moment and
support-face shear per foot, resolved for twisting, enveloped over every
Gauss point in the panel that lies outside the beam widths, every factored
combination and every required live-load pattern, on one common floor that
represents all floors.

Load cases (ACI 318-19 5.3.1): 1.4D and 1.2D + 1.6L. Live-load patterns
follow 6.4.3.3: when L <= 0.75 D, full factored live load on all panels
governs (6.4.3.3(a)); otherwise 3/4 of the factored live load on alternate
panels (checkerboards, for positive moments) and on the two panels adjacent
to each interior support line (for negative moments), never less than the
all-panel case (6.4.3.3(b) and (c)).

Twisting is resolved with the Wood-Armer rules for orthogonal x/y
reinforcement, using the floor model's sagging-positive mx, my and raw mxy.
Shear rows: the top-face (support) row carries the transverse shear
recovered AT the beam face -- in the element row adjacent to each support
line the two Gauss points that share a transverse coordinate are
interpolated/extrapolated linearly to the face position (centerline +/- half
the beam width), row by row and for both supports of the strip, and the
maximum is kept; the tension face there is verified from the twisting-
resolved top demand at the nearest point. The bottom-face row carries the
maximum shear in the pure sagging zone (Wood-Armer bottom demand positive,
top demand zero), which is where the bottom mat is the longitudinal tension
steel the one-way shear strength relies on.

The verification flags the strip routine reads are engineering assertions
(``Design.Config.SlabActionAssertions``), made after reviewing this
analysis; this module records the numerical basis of each item
(``numerical_basis``) and combines it with the assertion. A flag is True
only when it is asserted AND its numerical precondition holds. Nothing here
certifies its own output: without assertions the evidence is complete but
unverified, and the strip routine selects no reinforcement.
"""
from __future__ import annotations

import hashlib
import json
import math

from Design.SMRF_Floor_Analysis import analyze_floor, transfer_mesh_per_bay
from Design.SMRF_Slab_Reinforcement import slab_input_signature
from Design.SMRF_Common import make_check, not_evaluated, assertion_provenance_valid

METHOD_VERSION = "smrf_slab_actions_wood_armer_face_shear_v2"
PATTERN_LIVE_FRACTION = 0.75
PATTERN_RULE_ALL = "ACI 318-19 6.4.3.3(a): L <= 0.75 D, full factored live load on all panels governs"
PATTERN_RULE_PARTIAL = ("ACI 318-19 6.4.3.3(b)/(c): 3/4 factored live load on alternate panels and on the "
                        "panels adjacent to each interior support line, enveloped with the all-panel case")
_FLAGS = ("verified", "analysis_applicability_verified", "all_floors_enveloped",
          "load_pattern_envelope_verified", "spatial_envelope_per_unit_width",
          "twisting_moment_resolution_verified", "zero_membrane_force_verified")


def wood_armer(mx, my, mxy):
    """Design moments per unit width for orthogonal x/y bars.

    Inputs are sagging-positive bending resultants and the twisting
    resultant, any consistent unit. Returns nonnegative demands
    ``(bottom_x, bottom_y, top_x, top_y)`` in the same unit.
    """
    t = abs(mxy)
    # Bottom (sagging) steel.
    bx, by = mx + t, my + t
    if bx < 0 and by < 0:
        bx = by = 0.0
    elif bx < 0:
        bx, by = 0.0, my + (t * t / abs(mx) if mx else 0.0)
    elif by < 0:
        bx, by = mx + (t * t / abs(my) if my else 0.0), 0.0
    # Top (hogging) steel: the design moments are negative; report magnitudes.
    tx, ty = mx - t, my - t
    if tx > 0 and ty > 0:
        tx = ty = 0.0
    elif tx > 0:
        tx, ty = 0.0, my - (t * t / abs(mx) if mx else 0.0)
    elif ty > 0:
        tx, ty = mx - (t * t / abs(my) if my else 0.0), 0.0
    return max(0.0, bx), max(0.0, by), max(0.0, -tx), max(0.0, -ty)


def slab_load_cases(dead_ksf, live_ksf, nx, ny):
    """Factored slab cases and the pattern rule that produced them."""
    cases = [{"id": "1.4D", "dead_factor": 1.4, "live_factor": 0.0, "live_pattern": "none"},
             {"id": "1.2D+1.6L_all", "dead_factor": 1.2, "live_factor": 1.6, "live_pattern": "all"}]
    if live_ksf <= PATTERN_LIVE_FRACTION * dead_ksf:
        return cases, PATTERN_RULE_ALL
    partial = 1.6 * PATTERN_LIVE_FRACTION
    even = [[i, j] for j in range(ny) for i in range(nx) if (i + j) % 2 == 0]
    odd = [[i, j] for j in range(ny) for i in range(nx) if (i + j) % 2 == 1]
    cases.append({"id": "1.2D+1.2L_even", "dead_factor": 1.2, "live_factor": partial, "live_pattern": even})
    cases.append({"id": "1.2D+1.2L_odd", "dead_factor": 1.2, "live_factor": partial, "live_pattern": odd})
    for line in range(1, ny):
        panels = [[i, j] for j in (line - 1, line) for i in range(nx)]
        cases.append({"id": f"1.2D+1.2L_adjacent_x_line_{line}", "dead_factor": 1.2,
                      "live_factor": partial, "live_pattern": panels})
    for line in range(1, nx):
        panels = [[i, j] for i in (line - 1, line) for j in range(ny)]
        cases.append({"id": f"1.2D+1.2L_adjacent_y_line_{line}", "dead_factor": 1.2,
                      "live_factor": partial, "live_pattern": panels})
    return cases, PATTERN_RULE_PARTIAL


def _model_signature(slab_record, geometry, sections, mesh, case_ids):
    payload = {"method": METHOD_VERSION, "slab": slab_record, "geometry": geometry,
               "sections": sections, "mesh_per_bay": mesh, "support_model": "flexible_beams",
               "cases": case_ids}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_slab_action_evidence(slab_record, geometry, sections, live_load_ksf, slab_inputs,
                               mesh_per_bay=None, uniform_all_floors=True, assertions=None):
    """Slab strip demand evidence for ``design_slab_reinforcement``.

    ``sections`` needs the beam section and column footprint (as for the
    floor transfer) so beam widths can be excluded from the strip envelope.
    ``slab_inputs`` is the strip routine's input dictionary; its signature
    binds this evidence to that slab. ``uniform_all_floors`` must be True for
    the single-floor envelope to represent every floor. ``assertions`` is a
    mapping of the engineering assertions (see Design.Config
    .SlabActionAssertions); absent assertions leave every flag False.
    """
    if not uniform_all_floors:
        raise ValueError("Slab action evidence for non-uniform floors is not implemented.")
    nx, ny = geometry["num_bay_x"], geometry["num_bay_y"]
    lx, ly = geometry["bay_x_in"], geometry["bay_y_in"]
    half_beam = sections["b_beam_in"] / 2.0
    dead_ksf = (slab_record["concrete_unit_weight_kcf"] * slab_record["thickness_in"] / 12.0
                + slab_record["superimposed_dead_load_ksf"])
    mesh = mesh_per_bay or transfer_mesh_per_bay(nx, ny)
    cases, pattern_rule = slab_load_cases(dead_ksf, live_load_ksf, nx, ny)
    for case in cases:
        case["live_load_ksf"] = live_load_ksf
    envelope = {}   # (panel_id, axis, face) -> {"mu", "vu", "location"}
    max_membrane = 0.0
    equilibrium = []
    for case in cases:
        result = analyze_floor(slab_record, geometry, sections, case, mesh_per_bay=mesh,
                               support_model="flexible_beams")
        if result["status"] != "transfer_complete":
            raise RuntimeError(f"Slab action case {case['id']} failed: status {result['status']}.")
        max_membrane = max(max_membrane, result["max_abs_membrane_kip_per_in"])
        equilibrium.append({"case_id": case["id"], **result["equilibrium"]})
        dx, dy = lx / mesh, ly / mesh
        for panel in result["panels"]:
            pi, pj = panel["i"], panel["j"]
            points = panel["gauss_point_resultants"]
            # Moments: Wood-Armer envelope over the points outside the beam widths.
            for point in points:
                x, y = point["x_in"], point["y_in"]
                to_x_line = min(abs(y - j * ly) for j in range(ny + 1))
                to_y_line = min(abs(x - i * lx) for i in range(nx + 1))
                if to_x_line < half_beam or to_y_line < half_beam:
                    continue        # over a beam; not a slab strip section
                bx, by, tx, ty = wood_armer(point["mx"], point["my"], point["mxy_raw"])
                point["_wa"] = (bx, by, tx, ty)
                for (axis, face), moment in ((("x", "bottom"), bx), (("y", "bottom"), by),
                                             (("x", "top"), tx), (("y", "top"), ty)):
                    key = (panel["panel_id"], axis, face)
                    entry = envelope.setdefault(key, {"mu": 0.0, "vu": 0.0, "mu_location": None, "vu_location": None,
                                                      "shear_basis": None, "tension_face_verified": True})
                    mu = 12.0 * moment
                    if mu > entry["mu"]:
                        entry.update(mu=mu, mu_location={"case_id": case["id"], "x_in": x, "y_in": y,
                                                         "element": point["element"], "gauss_point": point["gauss_point"]})
            # Shear at the support faces, from the element row adjacent to each support line.
            by_element = {}
            for point in points:
                by_element.setdefault(point["element"], {})[point["gauss_point"]] = point
            for axis in ("x", "y"):
                key = (panel["panel_id"], axis, "top")
                entry = envelope.setdefault(key, {"mu": 0.0, "vu": 0.0, "mu_location": None, "vu_location": None,
                                                  "shear_basis": None, "tension_face_verified": True})
                for element, gps in by_element.items():
                    if len(gps) != 4:
                        continue
                    cell_i = int(math.floor(gps[1]["x_in"] / dx)); cell_j = int(math.floor(gps[1]["y_in"] / dy))
                    if axis == "x":
                        # supports are the y-lines at x = pi*lx and (pi+1)*lx; strips span x
                        pairs = [((gps[1], gps[2]), "x_in", "qx_raw", 2), ((gps[4], gps[3]), "x_in", "qx_raw", 3)]
                        left_cell, right_cell = cell_i == pi * mesh, cell_i == (pi + 1) * mesh - 1
                        faces = ([pi * lx + half_beam] if left_cell else []) + ([(pi + 1) * lx - half_beam] if right_cell else [])
                        tension_index = 2
                    else:
                        pairs = [((gps[1], gps[4]), "y_in", "qy_raw", 1), ((gps[2], gps[3]), "y_in", "qy_raw", 1)]
                        left_cell, right_cell = cell_j == pj * mesh, cell_j == (pj + 1) * mesh - 1
                        faces = ([pj * ly + half_beam] if left_cell else []) + ([(pj + 1) * ly - half_beam] if right_cell else [])
                        tension_index = 3
                    for face_position in faces:
                        for (a, b), coordinate, shear_key, _ in pairs:
                            x1, x2 = a[coordinate], b[coordinate]
                            q1, q2 = a[shear_key], b[shear_key]
                            q_face = q1 + (q2 - q1) * (face_position - x1) / (x2 - x1)
                            nearest = a if abs(x1 - face_position) <= abs(x2 - face_position) else b
                            top_demand = nearest.get("_wa", (0, 0, 0, 0))[tension_index] if "_wa" in nearest else wood_armer(
                                nearest["mx"], nearest["my"], nearest["mxy_raw"])[tension_index]
                            vu = 12.0 * abs(q_face)
                            if vu > entry["vu"]:
                                entry.update(vu=vu, shear_basis="linear recovery to the beam face from the two adjacent Gauss points",
                                             tension_face_verified=top_demand > 0.0,
                                             vu_location={"case_id": case["id"], "element": element, "face_position_in": face_position,
                                                          "gauss_positions_in": [x1, x2], "gauss_shears_kip_per_in": [q1, q2],
                                                          "top_demand_at_nearest_kip_in_per_in": top_demand})
                # Bottom row: maximum shear in the pure sagging zone (bottom mat is the tension steel there).
                key = (panel["panel_id"], axis, "bottom")
                entry = envelope.setdefault(key, {"mu": 0.0, "vu": 0.0, "mu_location": None, "vu_location": None,
                                                  "shear_basis": None, "tension_face_verified": True})
                for point in points:
                    if "_wa" not in point:
                        continue
                    bx, by, tx, ty = point["_wa"]
                    sagging = (bx > 0 and tx == 0.0) if axis == "x" else (by > 0 and ty == 0.0)
                    if not sagging:
                        continue
                    vu = 12.0 * abs(point["qx_raw"] if axis == "x" else point["qy_raw"])
                    if vu > entry["vu"]:
                        entry.update(vu=vu, shear_basis="maximum Gauss-point shear in the pure sagging zone",
                                     vu_location={"case_id": case["id"], "x_in": point["x_in"], "y_in": point["y_in"],
                                                  "element": point["element"], "gauss_point": point["gauss_point"]})
    if not envelope:
        raise RuntimeError("No slab Gauss points lie outside the beam widths; refine the mesh.")
    strips = []
    face_recovery_complete = True
    for (panel_id, axis, face), entry in sorted(envelope.items()):
        if face == "top" and entry.get("shear_basis") is None:
            face_recovery_complete = False
        if not entry.get("tension_face_verified", True):
            face_recovery_complete = False
        strips.append({"panel_id": panel_id, "axis": axis, "face": face,
                       "mu_kip_in_per_ft": entry["mu"], "vu_kip_per_ft": entry["vu"],
                       "demand_location": (f"{panel_id}: Wood-Armer {face} {axis} envelope over Gauss points outside beam "
                                           f"widths; moment at {entry['mu_location']}; shear: {entry.get('shear_basis')} at {entry['vu_location']}"),
                       "moment_location": entry["mu_location"], "shear_location": entry["vu_location"],
                       "shear_basis": entry.get("shear_basis"),
                       "tension_face_at_shear_verified": bool(entry.get("tension_face_verified", True))})
    membrane_free = max_membrane <= 1e-9
    balanced = all(item["numerical_balance_passed"] for item in equilibrium)
    asserted = dict(assertions or {})
    numerical = {
        "analysis_applicability_verified": ("Linear elastic ShellMITC4 plate on elastic ACI 8.4.1.8 T-beams, uniform area "
                                            "gravity only, no openings/drops/concentrated loads (SMRF_Floor_Analysis)."),
        "all_floors_enveloped": ("Uniform slab thickness, superimposed dead load and live load on every floor including the "
                                 "roof, so one common floor represents all floors."),
        "load_pattern_envelope_verified": pattern_rule,
        "spatial_envelope_per_unit_width": ("Maximum over every Gauss point of every element in the panel outside the beam "
                                            "widths; no strip averaging."),
        "twisting_moment_resolution_verified": "Wood-Armer resolution of mx, my, mxy into orthogonal x/y top and bottom demands.",
        "zero_membrane_force_verified": f"In-plane DOFs fixed; max |p11,p22,p12| = {max_membrane:.3e} kip/in.",
        "verified": (f"Numerical only: every case in equilibrium to 1e-8 ({balanced}), membrane-free response "
                     f"({membrane_free}). Engineering verification is asserted in Design.Config.SlabActionAssertions "
                     "after an independent review (methodology item 7); it is not set by this module."),
    }
    preconditions = {"zero_membrane_force_verified": membrane_free, "verified": membrane_free and balanced}
    provenance_ok = assertion_provenance_valid(asserted)
    evidence = {flag: provenance_ok and asserted.get(flag) is True and preconditions.get(flag, True)
                for flag in _FLAGS}
    evidence.update({
        "method": "plate_finite_element",
        "source": f"Design/SMRF_Slab_Actions.py {METHOD_VERSION}; Design/SMRF_Floor_Analysis.py flexible_beams",
        "load_combination_basis": "ACI 318-19 5.3.1 (1.4D; 1.2D+1.6L) with " + pattern_rule,
        "analysis_model_sha256": _model_signature(slab_record, geometry, sections, mesh, [c["id"] for c in cases]),
        "slab_input_sha256": slab_input_signature(slab_inputs),
        "shear_envelope_basis": "support_face_maximum",
        "load_scope": "uniform_area_gravity_no_concentrated_loads",
        "strips": strips,
        "numerical_basis": numerical,
        "engineering_assertions": {key: asserted.get(key) for key in (*_FLAGS, "asserted_by", "assertion_date", "assertion_basis")},
        "numerical_preconditions": preconditions,
        "assertion_provenance_valid": provenance_ok,
        "cases": [{k: v for k, v in case.items()} for case in cases],
        "pattern_rule": pattern_rule,
        "mesh_per_bay": mesh,
        "beam_half_width_excluded_in": half_beam,
        "shear_recovery": {"method": "linear_recovery_to_beam_face_from_adjacent_gauss_pair",
                           "top_rows_recovered_at_face": face_recovery_complete,
                           "bottom_rows": "maximum Gauss-point shear in the pure sagging zone"},
        "equilibrium": equilibrium,
        "max_abs_membrane_kip_per_in": max_membrane,
        "independent_hand_check": False,
        "units": {"moment": "kip-in/ft", "shear": "kip/ft"},
    })
    return evidence


def evaluate_slab_actions(evidence):
    """Qualification checks for saved slab action evidence."""
    if not isinstance(evidence, dict):
        return [not_evaluated("floor.qualified_slab_actions", "ACI 318-19 Chapters 6 and 8",
                              "No slab action evidence was generated.")]
    checks = []
    assertions = evidence.get("engineering_assertions") or {}
    flags_ok = (assertion_provenance_valid(assertions)
                and all(assertions.get(flag) is True and evidence.get(flag) is True for flag in _FLAGS))
    details = {"numerical_basis": evidence.get("numerical_basis"), "pattern_rule": evidence.get("pattern_rule"),
               "cases": [c["id"] for c in evidence.get("cases", [])],
               "engineering_assertions": evidence.get("engineering_assertions")}
    if flags_ok:
        checks.append(make_check("floor.qualified_slab_actions", "ACI 318-19 5.3.1, 6.4.3.3, 8.4; Wood-Armer",
                                 1, 1, "==", details=details))
    else:
        missing = [flag for flag in _FLAGS if evidence.get(flag) is not True]
        if not assertion_provenance_valid(assertions):
            missing.append("named_dated_assertion_basis")
        missing.extend("assertion." + flag for flag in _FLAGS if assertions.get(flag) is not True)
        checks.append({**not_evaluated("floor.qualified_slab_actions", "ACI 318-19 5.3.1, 6.4.3.3, 8.4; Wood-Armer",
                                       f"Slab action evidence is computed but not asserted as verified: {missing}. "
                                       "Assert in Design.Config.SlabActionAssertions after independent review."),
                       "details": {**details, "reason": f"unasserted or unmet: {missing}"}})
    balanced = all(item.get("numerical_balance_passed") for item in evidence.get("equilibrium", [])) \
        and bool(evidence.get("equilibrium"))
    checks.append(make_check("floor.slab_action_equilibrium", "Reaction/first-moment balance of every slab case",
                             int(balanced), 1, "=="))
    checks.append(not_evaluated("floor.independent_hand_verification", "Engineering review",
                                "Hand-check a representative floor's strip moments, shears and beam transfer against the model."))
    recovery = evidence.get("shear_recovery") or {}
    if recovery.get("top_rows_recovered_at_face") is True:
        checks.append(make_check("floor.support_face_shear_recovery", "Slab design section / shear recovery",
                                 1, 1, "==", details={"method": recovery.get("method"),
                                                      "bottom_rows": recovery.get("bottom_rows"),
                                                      "basis": "top rows: shear interpolated/extrapolated linearly to the beam face "
                                                               "from the adjacent Gauss pair, both supports, every element row "
                                                               "(MITC4 transverse shear is constant across that pair, so this is "
                                                               "the adjacent element's shear); tension face confirmed from the "
                                                               "twisting-resolved top demand there"}))
    else:
        checks.append(not_evaluated("floor.support_face_shear_recovery", "Slab design section / shear recovery",
                                    "Face shear was not recovered for every support row, or a face's tension side "
                                    "was not the top; review the shear_location details."))
    return checks
