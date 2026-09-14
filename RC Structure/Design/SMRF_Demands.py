"""Pure, explicitly scoped SMRF load-combination and drift evidence helpers.

Units are kip and inch; no OpenSees imports or model state are used here. These
helpers do not establish ELF eligibility or complete building-code compliance.
ASCE/SEI 7-22 Chapters 2 and 12 are the declared basis. Primary references:
https://www.asce.org/publications-and-news/codes-and-standards/asce-sei-7-22
https://nvlpubs.nist.gov/nistpubs/gcr/2016/NIST.GCR.16-917-40.pdf

The NIST guide explains the approach but predates ASCE 7-22. In particular,
7-22 permits a 0.10 lower bound on theta_max and limits beta >= 1.25/Omega0;
these newer details must not be taken from an older worked example.
"""
from __future__ import annotations

import math
from datetime import date
from collections.abc import Mapping

from Design.SMRF_Common import make_check, not_evaluated, summarize_checks


def _number(value, name, minimum=None, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return number


def live_load_patterns(num_bay_x, num_bay_y):
    """ACI 318-19 6.4.2 arrangements for the frame beams, as floor panel sets.

    Beams are loaded by the slab, so a span pattern is a panel pattern:
    alternate spans (panel columns/rows of one parity, for maximum positive
    moment) and two adjacent spans about every interior support line (for
    maximum negative moment there). Patterns are zero-based [i, j] panels.
    """
    patterns = []
    for axis, count, other in (("x", num_bay_x, num_bay_y), ("y", num_bay_y, num_bay_x)):
        for parity, name in ((0, "even"), (1, "odd")):
            panels = [[i, j] for i in range(num_bay_x) for j in range(num_bay_y)
                      if ((i if axis == "x" else j) % 2) == parity]
            if panels:
                patterns.append({"id": f"alternate_{axis}_{name}", "rule": "ACI 318-19 6.4.2(c)", "panels": panels})
        for support in range(1, count):
            panels = [[i, j] for i in range(num_bay_x) for j in range(num_bay_y)
                      if (i if axis == "x" else j) in (support - 1, support)]
            patterns.append({"id": f"adjacent_{axis}_support_{support}", "rule": "ACI 318-19 6.4.2(b)",
                             "panels": panels})
    return patterns


def strength_load_combinations(sds, redundancy_factor=1.3,
                               orthogonal_fraction=0.3, live_patterns=()):
    """Return the signed D/L/QEx/QEy strength-load descriptors.

    ``live_patterns`` (from ``live_load_patterns``) adds one 1.2D + 1.6L
    gravity combination per arrangement, carrying ``live_pattern`` = its id;
    the unpatterned combinations carry ``live_pattern`` = "all".

    ``dead`` applies to ALL dead load, including member self-weight. ``live``
    applies only to L. ``ex``/``ey`` multiply signed strength-level *unfactored*
    ELF actions QEx/QEy: they already include rho and must not be multiplied by
    it again. Both 100X/30Y and 30X/100Y are included with independent signs.

    This is a D/L/seismic subset, not all ASCE Chapter 2 combinations. Roof
    live, snow, rain, wind and other loads require a separate applicability
    determination. L is not reduced to 0.5. Unpatterned full L is not always
    conservative, so live-load patterning remains a separate prerequisite.
    Rho=1.3 is a conservative assumption, not an assessment of redundancy.
    """
    sds = _number(sds, "sds", minimum=0)
    rho = _number(redundancy_factor, "redundancy_factor", minimum=1)
    fraction = _number(orthogonal_fraction, "orthogonal_fraction", minimum=0.3)
    if fraction > 1:
        raise ValueError("orthogonal_fraction must not exceed 1.")
    combinations = [
        {"id": "gravity_1.4D", "family": "gravity", "dead": 1.4,
         "live": 0.0, "ex": 0.0, "ey": 0.0, "live_pattern": "all"},
        {"id": "gravity_1.2D_1.6L", "family": "gravity", "dead": 1.2,
         "live": 1.6, "ex": 0.0, "ey": 0.0, "live_pattern": "all"},
    ]
    for pattern in live_patterns:
        combinations.append({"id": f"gravity_1.2D_1.6L_{pattern['id']}", "family": "gravity_pattern",
                             "dead": 1.2, "live": 1.6, "ex": 0.0, "ey": 0.0,
                             "live_pattern": pattern["id"]})
    for family, dead, live in (
        ("seismic_high_gravity", 1.2 + 0.2 * sds, 1.0),
        ("seismic_low_gravity", 0.9 - 0.2 * sds, 0.0),
    ):
        for primary, fx, fy in (("X", 1.0, fraction), ("Y", fraction, 1.0)):
            for sx in (-1, 1):
                for sy in (-1, 1):
                    combinations.append({
                        "id": f"{family}_{primary}_x{sx:+d}_y{sy:+d}",
                        "family": family, "dead": dead, "live": live,
                        "ex": rho * sx * fx, "ey": rho * sy * fy, "live_pattern": "all",
                    })
    return combinations


def floor_load_factor(dead_factor, live_factor, dead_load, live_load):
    """Weighted multiplier for a uniform combined-D+L floor-load interface.

    This is only valid when D and L share the same floor distribution. Apply
    ``dead_factor`` separately to member self-weight. It does not implement
    live-load patterning or represent different roof/floor load distributions.
    """
    d = _number(dead_load, "dead_load", minimum=0)
    l = _number(live_load, "live_load", minimum=0)
    df = _number(dead_factor, "dead_factor")
    lf = _number(live_factor, "live_factor")
    if d + l <= 0:
        raise ValueError("The combined floor dead and live load must be positive.")
    return (df * d + lf * l) / (d + l)


def story_node_deltas(upper_displacements, lower_displacements):
    """Subtract aligned floor-node XY displacements using stable grid IDs.

    Values are mappings ``{'x': ux, 'y': uy}``; base displacements must be
    explicitly provided (zero for the declared fixed base). A missing node
    raises rather than silently removing an edge from the drift calculation.
    """
    if not upper_displacements or set(upper_displacements) != set(lower_displacements):
        raise ValueError("Upper and lower floors require identical nonempty grid-node IDs.")
    return {
        node: {direction: _number(upper[direction], "upper displacement") -
               _number(lower_displacements[node][direction], "lower displacement")
               for direction in ("x", "y")}
        for node, upper in upper_displacements.items()
    }


def seismic_design_category(sds, sd1, s1, risk_category="II"):
    """ASCE 7-22 11.6: SDC from Tables 11.6-1/11.6-2, with the S1 >= 0.75 rule."""
    risk = str(risk_category).upper()
    if risk not in ("I", "II", "III", "IV"):
        raise ValueError("risk_category must be I, II, III or IV.")
    if s1 >= 0.75:
        return "F" if risk == "IV" else "E"
    def from_sds(value):
        if value < 0.167:
            return "A"
        if value < 0.33:
            return "B"
        if value < 0.50:
            return "C"
        return "D"
    def from_sd1(value):
        if value < 0.067:
            return "A"
        if value < 0.133:
            return "B"
        if value < 0.20:
            return "C"
        return "D"
    short, one = from_sds(sds), from_sd1(sd1)
    if risk == "IV":
        short = {"A": "A", "B": "C", "C": "D", "D": "D"}[short]
        one = {"A": "A", "B": "C", "C": "D", "D": "D"}[one]
    return max(short, one)


def elf_eligibility(sdc, height_ft, regular, period_sec, ts_sec):
    """ASCE 7-22 Table 12.6-1 for Risk Category I/II buildings."""
    if sdc in ("A", "B", "C"):
        return True, "Table 12.6-1: SDC B/C, all structures permitted (SDC A per 11.7)"
    if regular and height_ft <= 160.0:
        return True, "Table 12.6-1: no structural irregularities and height <= 160 ft"
    if regular and period_sec < 3.5 * ts_sec:
        return True, "Table 12.6-1: no irregularities, T < 3.5 Ts"
    return False, "Table 12.6-1: ELF not permitted for this height/period/irregularity combination"


SITE_CLASSES = ("A", "B", "BC", "C", "CD", "D", "DE", "E", "F")
RISK_CATEGORIES = ("I", "II", "III", "IV")


def demand_policy_problems(policy):
    """Why a DemandPolicy is not an acceptable declaration; empty when it is.

    A declaration is a named, dated basis with values in their domains. A
    blank or whitespace author, date or basis, a date that is not an ISO
    calendar date, a site class or risk category outside ASCE 7-22's, a
    blank occupancy, a nonfinite or negative load value, an accidental
    torsion ratio below the 5% of 12.8.4.2, or a non-Boolean flag is a
    problem, listed explicitly so a typo never reads as approved evidence.
    """
    problems = []
    if not isinstance(policy, dict):
        return ["DemandPolicy is missing"]
    for key in ("declared_by", "declaration_date", "declaration_basis"):
        value = policy.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key} is blank")
    stamp = policy.get("declaration_date")
    if isinstance(stamp, str) and stamp.strip():
        try:
            if date.fromisoformat(stamp).isoformat() != stamp:
                problems.append("declaration_date is not an ISO calendar date")
        except ValueError:
            problems.append("declaration_date is not an ISO calendar date")
    site_class = policy.get("site_class")
    if not isinstance(site_class, str) or site_class.strip().upper() not in SITE_CLASSES:
        problems.append(f"site_class {site_class!r} is not one of {SITE_CLASSES}")
    risk = policy.get("risk_category")
    if not isinstance(risk, str) or risk.strip().upper() not in RISK_CATEGORIES:
        problems.append(f"risk_category {risk!r} is not one of {RISK_CATEGORIES}")
    occupancy = policy.get("occupancy")
    if not isinstance(occupancy, str) or not occupancy.strip():
        problems.append("occupancy is blank")
    for key in ("partition_allowance_ksf", "storage_live_fraction_in_weight", "roof_live_load_ksf", "snow_load_ksf"):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            problems.append(f"{key} {value!r} is not a finite nonnegative number")
    fraction = policy.get("storage_live_fraction_in_weight")
    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and math.isfinite(fraction) and fraction > 1:
        problems.append("storage_live_fraction_in_weight exceeds 1")
    ratio = policy.get("accidental_torsion_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0.05:
        problems.append(f"accidental_torsion_ratio {ratio!r} is below the 5% of ASCE 7-22 12.8.4.2")
    for key in ("wind_governs", "rain_ponding_excluded", "live_load_patterning"):
        if not isinstance(policy.get(key), bool):
            problems.append(f"{key} is not Boolean")
    return problems


def evaluate_demand_basis(record):
    """Evaluate the demand-scope items from the saved policy, loads and results.

    Everything here reads what the design record carries: the declared
    ``DemandPolicy`` (request identity), the load inventory, the torsion
    assessment and the drift/ELF assumptions. Items stay not_evaluated
    while the policy carries no declaration basis.
    """
    policy = ((record.get("request_identity") or {}).get("policy") or {}).get("demands") \
        or (record.get("demand_basis") or {}).get("policy") or {}
    basis = record.get("demand_basis") or {}
    seismic = record.get("seismic") or {}
    loads = record.get("floor_loads") or {}
    geometry = record.get("geometry") or {}
    checks = []
    problems = demand_policy_problems(policy)
    declared = not problems
    def declaration_missing(key, clause, what):
        return not_evaluated(f"demands.{key}", clause,
                             f"{what}: declare DemandPolicy (declared_by, declaration_date, declaration_basis) in "
                             f"Design/Config.py. Rejected: {'; '.join(problems) if problems else 'not declared'}.")
    # Site hazard and SDC.
    try:
        sdc = seismic_design_category(seismic["sds"], seismic["sd1"], seismic["s1"], policy.get("risk_category", "II"))
    except (KeyError, ValueError, TypeError):
        sdc = None
    label = seismic.get("site_label") or ""
    label_sdc = label.split("_")[1].upper() if label.startswith("sdc_") and len(label.split("_")) > 1 else None
    if sdc is None or not declared:
        checks.append(declaration_missing("site_hazard", "ASCE 7-22 Chapters 11 and 21",
                                          "Site class and risk category are declarations"))
    else:
        site_specific = policy.get("site_class", "").upper() in ("D", "E", "F") and seismic["s1"] >= 0.2
        checks.append(make_check("demands.site_hazard", "ASCE 7-22 11.4.8, 11.6, Tables 11.6-1/11.6-2",
                                 int(label_sdc == sdc and not site_specific), 1, "==",
                                 details={"derived_sdc": sdc, "site_label": label, "label_sdc": label_sdc,
                                          "site_class": policy.get("site_class"), "risk_category": policy.get("risk_category"),
                                          "site_specific_ground_motion_required": site_specific,
                                          "basis": "SDS/SD1/S1 are declared design values; 11.4.8 site-specific analysis "
                                                   "is required for Site Class D/E/F with S1 >= 0.2 unless its exceptions apply"}))
    # ELF eligibility.
    regularity = basis.get("regularity") or {}
    torsion = basis.get("torsion") or {}
    height_ft = geometry.get("num_floor", 0) * geometry.get("story_h_in", 0.0) / 12.0
    if sdc is None or not regularity or not declared:
        checks.append(declaration_missing("elf_eligibility", "ASCE 7-22 Table 12.6-1; 12.3", "ELF eligibility"))
    else:
        regular = regularity.get("regular") is True and torsion.get("torsional_irregularity") == "none"
        permitted, reason = elf_eligibility(sdc, height_ft, regular, basis.get("design_period_sec", 0.0),
                                            basis.get("ts_sec", 0.0))
        checks.append(make_check("demands.elf_eligibility", "ASCE 7-22 Table 12.6-1; 12.3.2; 12.2.5.5",
                                 int(permitted), 1, "==",
                                 details={"height_ft": height_ft, "regular": regular, "reason": reason,
                                          "regularity": regularity, "torsional_irregularity": torsion.get("torsional_irregularity"),
                                          "period_cap": basis.get("period_basis")}))
    # Accidental torsion.
    if not torsion:
        checks.append(not_evaluated("demands.accidental_torsion", "ASCE 7-22 12.8.4.2-12.8.4.3",
                                    "No torsion assessment was saved with this design."))
    else:
        checks.append(make_check("demands.accidental_torsion", "ASCE 7-22 12.8.4.2-12.8.4.3; Table 12.3-1",
                                 torsion.get("max_drift_ratio", 99.0), 1.4, "<=", "delta_max/delta_avg",
                                 details={**torsion, "basis": "5% eccentricity at every level in every seismic "
                                          "combination, amplified by Ax; ratio from the drift runs with torsion applied"}))
    # Load scope.
    if not declared:
        checks.append(declaration_missing("load_scope", "ASCE 7-22 Chapter 2", "Applicable loads are a declaration"))
    else:
        roof_ok = loads.get("floor_live_load_ksf", 0.0) >= policy.get("roof_live_load_ksf", 0.0) + policy.get("snow_load_ksf", 0.0)
        checks.append(make_check("demands.load_scope", "ASCE 7-22 Chapter 2; 4.8; 7; 8; 26",
                                 int(roof_ok and not policy.get("wind_governs", False) and policy.get("rain_ponding_excluded", False)), 1, "==",
                                 details={"roof_live_load_ksf": policy.get("roof_live_load_ksf"),
                                          "floor_live_applied_to_roof_ksf": loads.get("floor_live_load_ksf"),
                                          "snow_load_ksf": policy.get("snow_load_ksf"), "wind_governs": policy.get("wind_governs"),
                                          "rain_ponding_excluded": policy.get("rain_ponding_excluded"),
                                          "declared_by": policy.get("declared_by"), "declaration_basis": policy.get("declaration_basis")}))
    # Live-load patterning.
    patterns = basis.get("live_load_patterns")
    if patterns is None:
        checks.append(not_evaluated("demands.live_load_patterning", "ACI 318-19 6.4.2",
                                    "No live-load arrangements were analyzed."))
    else:
        checks.append(make_check("demands.live_load_patterning", "ACI 318-19 6.4.2(a)-(c)",
                                 int(len(patterns) > 0 and basis.get("patterns_in_strength_envelope") is True), 1, "==",
                                 details={"arrangements": [p["id"] for p in patterns],
                                          "basis": "alternate spans and two adjacent spans about every interior support, "
                                                   "as slab panel patterns through the floor transfer, in the 1.2D+1.6L family"}))
    # Effective seismic weight.
    if not declared:
        checks.append(declaration_missing("effective_seismic_weight", "ASCE 7-22 12.7.2", "The weight inventory needs the occupancy declaration"))
    else:
        partitions = policy.get("partition_allowance_ksf", 0.0)
        sdl = loads.get("floor_superimposed_dead_load_ksf") or 0.0
        storage = policy.get("storage_live_fraction_in_weight", 0.0)
        live_fraction = loads.get("seismic_live_load_fraction", 0.0)
        office = policy.get("occupancy", "") == "office"
        ok = (partitions >= 0.010 and partitions <= sdl and (storage == 0.0 if office else storage >= 0.25)
              and live_fraction >= storage and policy.get("snow_load_ksf", 0.0) == 0.0)
        checks.append(make_check("demands.effective_seismic_weight", "ASCE 7-22 12.7.2", int(ok), 1, "==",
                                 details={"inventory": "slab self-weight + SDL (finishes/MEP/partitions) + beam drops + columns",
                                          "partition_allowance_ksf": partitions, "superimposed_dead_load_ksf": sdl,
                                          "storage_live_fraction": storage, "seismic_live_load_fraction": live_fraction,
                                          "occupancy": policy.get("occupancy"),
                                          "total_floor_seismic_weight_kip": loads.get("total_floor_seismic_weight_kip")}))
    # Drift analysis basis.
    drift = basis.get("drift") or {}
    if not drift:
        checks.append(not_evaluated("demands.drift_analysis_basis", "ASCE 7-22 12.8.6; ACI 318-19 6.6.3",
                                    "No drift basis was saved with this design."))
    else:
        ok = (abs(drift.get("beam_stiffness_modifier", 0) - 0.35) < 1e-9 and abs(drift.get("column_stiffness_modifier", 0) - 0.70) < 1e-9
              and drift.get("cd") == 5.5 and drift.get("rho_for_drift_load") == 1.0 and drift.get("second_order_included") is True)
        checks.append(make_check("demands.drift_analysis_basis", "ASCE 7-22 12.8.6, 12.8.7; ACI 318-19 Table 6.6.3.1.1(a)",
                                 int(ok), 1, "==", details={**drift, "basis": "elastic cracked stiffness 0.35 Ig beams / 0.70 Ig columns, "
                                                            "rho = 1 drift forces at the Cu Ta-capped period (conservative for drift), Cd = 5.5, "
                                                            "P-Delta stability coefficient, full D+L gravity state"}))
    return checks


def demand_scope_checks():
    """Kept for callers that have no design record: every item not_evaluated."""
    return evaluate_demand_basis({})


def evaluate_drift_and_stability(stories, cd=5.5, importance_factor=1.0,
                                 drift_limit_ratio=0.02, redundancy_factor=1.3,
                                 seismic_design_category="D", beta=1.0,
                                 overstrength_factor=3.0,
                                 second_order_included=False,
                                 analysis_succeeded=True):
    """Evaluate node-envelope drift and story stability from elastic results.

    Each story has ``id``, ``height_in``, ``node_deltas_in`` mapping grid IDs to
    ``{'x': dx, 'y': dy}``, ``expected_node_ids`` listing *all* structural floor
    grid IDs, ``story_shear_kip`` mapping directions to shears, and
    ``gravity_above_kip`` (P at and above that story, not just that floor).
    ``directions`` may be supplied to evaluate a unidirectional analysis; it
    defaults to ('x', 'y'). Input deltas come from QEx/QEy drift analyses with
    rho=1, not factored strength combinations or nonlinear time histories.

    Cd/Ie amplifies elastic drift. Risk-II 0.02h is the base limit (no low-rise
    allowance); solely moment-frame SDC D/E/F limits are divided by rho.
    theta=P*delta_elastic/(V*h); 7-22 permits theta_max at least 0.10. The
    optional theta/(1+theta) correction applies only to analysis that already
    includes P-Delta. A first-order theta>0.10 requires a subsequent design
    amplification/reanalysis, recorded as not_evaluated here, not silently
    applied to an unrelated member-demand envelope.
    """
    cd = _number(cd, "cd", positive=True)
    ie = _number(importance_factor, "importance_factor", positive=True)
    limit = _number(drift_limit_ratio, "drift_limit_ratio", positive=True)
    if limit > 0.02:
        raise ValueError("This Risk-II research scope does not use drift allowances above 0.02.")
    rho = _number(redundancy_factor, "redundancy_factor", minimum=1)
    beta = _number(beta, "beta", positive=True)
    omega = _number(overstrength_factor, "overstrength_factor", positive=True)
    if beta < 1.25 / omega:
        raise ValueError("ASCE 7-22 beta cannot be less than 1.25/Omega0.")
    sdc = str(seismic_design_category).upper()
    if sdc not in ("A", "B", "C", "D", "E", "F"):
        raise ValueError("seismic_design_category must be A through F.")
    limit /= rho if sdc in ("D", "E", "F") else 1.0
    theta_max = max(0.10, min(0.5 / (beta * cd), 0.25))
    checks, rows = [], []
    stories = list(stories)
    if not analysis_succeeded:
        checks.append(not_evaluated(
            "demands.drift_analysis", "ASCE 7-22 12.8.6-12.8.7",
            "Analysis did not complete; drift and stability are unknown. A solver failure alone does not prove structural instability."))
        return {"checks": checks, "stories": rows, **summarize_checks(checks)}
    if not stories:
        checks.append(not_evaluated("demands.story_results", "ASCE 7-22 12.8.6",
                                    "No story results were supplied."))
    for index, story in enumerate(stories, 1):
        location = f"story:{story.get('id', index)}"
        directions = tuple(story.get("directions", ("x", "y")))
        if not directions or any(d not in ("x", "y") for d in directions) or len(set(directions)) != len(directions):
            raise ValueError("Each story directions must be a nonempty unique subset of x/y.")
        deltas = story.get("node_deltas_in", {})
        expected = story.get("expected_node_ids")
        if not isinstance(deltas, Mapping):
            deltas = {}
        if expected is None:
            checks.append(not_evaluated("demands.node_coverage", "ASCE 7-22 12.8.6.5",
                                        "Expected grid-node IDs are missing; edge coverage cannot be confirmed.", location))
        else:
            expected = list(expected)
            matched = bool(expected) and len(expected) == len(set(expected)) and set(deltas) == set(expected)
            checks.append(make_check("demands.node_coverage", "ASCE 7-22 12.8.6.5",
                                     int(matched), 1, comparison="==", location=location,
                                     details={"expected_nodes": len(expected), "supplied_nodes": len(deltas)}))
        for direction in directions:
            loc = f"{location}/{direction}"
            try:
                height = _number(story.get("height_in"), "height_in", positive=True)
                if not deltas:
                    raise ValueError("No structural grid-node drifts were supplied.")
                values = {str(node): abs(_number(value[direction], "node delta"))
                          for node, value in deltas.items()}
                governing_node = max(values, key=values.get)
                elastic_delta = values[governing_node]
            except (ValueError, KeyError, TypeError) as exc:
                for key in ("story_drift", "stability"):
                    checks.append(not_evaluated(f"demands.{key}", "ASCE 7-22 12.8.6-12.8.7", str(exc), loc))
                continue
            amplified = elastic_delta * cd / ie
            ratio = amplified / height
            row = {"location": loc, "governing_node": governing_node,
                   "elastic_drift_in": elastic_delta, "design_drift_in": amplified,
                   "drift_ratio": ratio, "allowable_drift_ratio": limit,
                   "theta": None, "theta_limit": theta_max,
                   "required_second_order_amplification": None}
            checks.append(make_check("demands.story_drift", "ASCE 7-22 12.8.6; 12.12.1.1; Table 12.12-1",
                                     ratio, limit, location=loc, units="in/in",
                                     details={"governing_node": governing_node, "cd": cd, "importance_factor": ie}))
            try:
                p = _number(story.get("gravity_above_kip"), "gravity_above_kip", minimum=0)
                shear = abs(_number(story.get("story_shear_kip", {}).get(direction), "story_shear_kip"))
                if shear <= 0:
                    raise ValueError("Nonzero corresponding story shear is required to evaluate stability.")
                theta_raw = p * elastic_delta / (shear * height)
                theta = theta_raw / (1 + theta_raw) if second_order_included else theta_raw
                row.update(theta=theta, theta_raw=theta_raw)
                checks.append(make_check("demands.stability", "ASCE 7-22 12.8.7 Eqs. 12.8-18, 12.8-19",
                                         theta, theta_max, units="dimensionless", location=loc,
                                         details={"second_order_included": second_order_included, "beta": beta}))
                needs_amplification = theta > 0.10 and not second_order_included
                row["required_second_order_amplification"] = 1 / (1 - theta) if needs_amplification and theta < 1 else 1.0 if not needs_amplification else None
                if needs_amplification:
                    checks.append(not_evaluated("demands.second_order_effects", "ASCE 7-22 12.8.7",
                                                "First-order theta exceeds 0.10; displacement AND member-demand amplification/reanalysis has not been performed.", loc))
            except (ValueError, TypeError, AttributeError) as exc:
                checks.append(not_evaluated("demands.stability", "ASCE 7-22 12.8.7", str(exc), loc))
            rows.append(row)
    return {"checks": checks, "stories": rows,
            "assumptions": {"risk_category": "II", "system": "solely_moment_frames",
                            "sdc": sdc, "rho": rho, "cd": cd, "importance_factor": ie,
                            "drift_load_rho": 1.0, "theta_max_lower_bound": 0.10,
                            "not_complete_code_acceptance": True},
            **summarize_checks(checks)}
