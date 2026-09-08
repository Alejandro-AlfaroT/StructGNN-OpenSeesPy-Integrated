"""
Design/Design_Driver.py
=======================

Runs a complete ACI 318-19 design for the current geometry and writes the
result as a reusable artifact.

Why this exists
---------------
Redesign.run_iterative_redesign resizes longitudinal bars against a
gravity-only demand, and it was only ever wired into Main.py. The dataset
generator calls Ground_Motion_Main, which had no design step at all, so every
generated structure kept the Structure_Parameters defaults regardless of its
height or span. This driver closes both gaps:

  * demand comes from the ASCE 7-22 seismic combination (1.2D + 0.5L + E),
    not gravity alone, so member sizes respond to building height
  * the search covers section dimensions and concrete strength as well as
    reinforcement, via the ladders in Design/Section_Design.py

Design depends only on geometry and loads, never on the ground motion, so the
result is cached per structure and reused by every record and intensity run of
that case. Building it once instead of once per analysis is the difference
between designing a thousand structures and designing several thousand.

Unit system: kip, inch, ksi.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Gravity import run_gravity_analysis
from Analysis.Modal import run_modal_analysis
from Design.ACI_Checks import run_checks_phase1
from Model.IMK_Calibration import (
    column_gravity_axial,
    column_moment_at_axial,
    column_pm_nominal_for,
)
from RC_Design_Check import _col_steel_layers
from Design.Config import DesignConfig
from Design.Section_Design import (
    beam_ladder,
    column_ladder,
    nearest_rung_index,
    suggest_rung_index,
    validate_rung,
)
from Loads.Gravity_Loads import apply_gravity_loads
from Loads.Seismic_ELF import apply_elf_loads
from Model.Build_Model import build_model
from RC_Design_Check import get_element_tags
from Redesign import apply_updates, redesign_steel


DESIGN_ARTIFACT_NAME = "design.json"
DESIGN_SCHEMA_VERSION = "rc_design_v1"

_STATE_KEYS = (
    "B_COL", "H_COL", "FC_COL_KSI", "B_BEAM", "H_BEAM", "FC_BEAM_KSI",
    "COL_BAR_SIZE", "COL_TOP_BARS", "COL_BOT_BARS", "COL_SIDE_BARS", "COL_BAR_AREA",
    "BEAM_BAR_SIZE", "BEAM_TOP_BARS", "BEAM_BOT_BARS", "BEAM_SIDE_BARS", "BEAM_BAR_AREA",
    "COL_STIRRUP_SPACING", "BEAM_STIRRUP_SPACING",
)


@contextlib.contextmanager
def _quiet():
    """Suppress OpenSees C-level banners emitted during repeated model builds."""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for handle in saved:
            os.close(handle)
        os.close(devnull)


def _capture_state():
    return {key: getattr(sp, key) for key in _STATE_KEYS}


def _restore_state(state):
    for key, value in state.items():
        setattr(sp, key, value)


def _apply_rung(rung, member_type):
    """Write one ladder rung into Structure_Parameters."""
    b, h, fc = rung
    if member_type == "column":
        sp.B_COL, sp.H_COL, sp.FC_COL_KSI = b, h, fc
    else:
        sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI = b, h, fc


def _sync_cfg_to_sp(cfg):
    """Mirror the current Structure_Parameters section state into cfg.

    The ACI checks read sections and concrete strengths from cfg --
    check_column_pm takes Ag from cfg.sections, and build_pm_diagram builds
    the entire interaction surface from it -- while the ladder writes them to
    sp. cfg was constructed once at entry (DesignConfig.from_structure_
    parameters) and never refreshed, so after any escalation the checks
    measured the new, larger, heavier model against the capacity of the
    section the design STARTED with: demand growing, capacity frozen.

    That is why column DCR rose with section size instead of falling
    (case_0013: 1.104 at 26x26 up to 2.005 at 36x36) and why any case needing
    a column escalation could not converge. Cases that accept on the first
    iteration were unaffected, because cfg still matched sp -- which is why
    this stayed hidden.
    """
    if cfg is None:
        return
    cfg.sections.b_col_in = sp.B_COL
    cfg.sections.h_col_in = sp.H_COL
    cfg.sections.b_beam_in = sp.B_BEAM
    cfg.sections.h_beam_in = sp.H_BEAM
    cfg.materials.fc_col_ksi = sp.FC_COL_KSI
    cfg.materials.fc_beam_ksi = sp.FC_BEAM_KSI


def _model_period():
    """First valid elastic period of the current model, for the ELF period cap."""
    with _quiet():
        ops.wipe()
        build_model()
        modes = run_modal_analysis()
    for mode in modes:
        if mode.get("valid") and mode.get("period"):
            return float(mode["period"])
    return None


def _analyze_combination(direction, model_period_sec):
    """Build and analyze 1.2D + 0.5L + E in one direction."""
    with _quiet():
        ops.wipe()
        build_model()
    apply_gravity_loads(
        floor_factor=sp.seismic_combination_floor_factor(),
        self_weight_factor=sp.SEISMIC_COMBINATION_DEAD_FACTOR,
    )
    elf = apply_elf_loads(direction, model_period_sec=model_period_sec)
    with _quiet():
        run_gravity_analysis()
    return elf


def _governing_dcrs(cfg):
    """Worst PM/flexure/shear DCR for columns and for beams in the current state."""
    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    results = run_checks_phase1(col_tags, beam_x_tags + beam_y_tags, cfg)

    column_dcr = 0.0
    beam_dcr = 0.0
    for result in results.values():
        if result.member_type == "column":
            column_dcr = max(column_dcr, result.dcr("PM"), result.dcr("shear"))
        else:
            beam_dcr = max(
                beam_dcr,
                result.dcr("flexure_pos"),
                result.dcr("flexure_neg"),
                result.dcr("shear"),
            )
    return column_dcr, beam_dcr, results


def _steel_pass(cfg, model_period_sec, max_steel_iter):
    """Resize reinforcement at the current sections until the spec stops changing.

    Both orthogonal ELF cases are analyzed each pass and the worse governs, so
    an unequal bay count in X and Y cannot leave one direction under-designed.
    """
    worst = {"column": 0.0, "beam": 0.0}
    elf_used = None

    for _ in range(max_steel_iter):
        worst = {"column": 0.0, "beam": 0.0}
        governing_results = None
        governing_score = -1.0

        for direction in ("x", "y"):
            elf_used = _analyze_combination(direction, model_period_sec)
            column_dcr, beam_dcr, results = _governing_dcrs(cfg)
            worst["column"] = max(worst["column"], column_dcr)
            worst["beam"] = max(worst["beam"], beam_dcr)
            if column_dcr + beam_dcr > governing_score:
                governing_score = column_dcr + beam_dcr
                governing_results = results

        column_update, beam_update, converged, _penalties = redesign_steel(
            governing_results, cfg
        )
        if converged or (not column_update and not beam_update):
            break
        apply_updates(column_update, beam_update, cfg)

    return worst, elf_used


def _scwb_required_column_moment():
    """Column nominal moment needed to satisfy ACI 318-19 18.7.3.

    The code requires sum(Mnc) >= 1.2 * sum(Mnb) at each joint. For an
    interior joint in one direction the sums are two columns (above and
    below) against two beams (left and right), so the per-member form is
    Mnc >= 1.2 * Mnb. Both orthogonal directions use the same sections here,
    so the governing beam capacity is the larger of the two axes.
    """
    beam_moment = max(sp.beam_nominal_moment_y(), sp.beam_nominal_moment_z())
    return sp.SCWB_RATIO_MIN * beam_moment


def _scwb_governing_axial():
    """Factored axial force that minimises column flexural strength.

    ACI 318-19 18.7.3.2 wants Mnc "calculated for the factored axial force
    ... resulting in the lowest flexural strength". Below the balance point
    Mn falls with axial load, so the governing joint is the lightest-loaded
    column: the top story at a corner, where tributary area is smallest.
    """
    return column_gravity_axial(max(1, sp.NUM_FLOOR), 0, 0)


def _column_nominal_moment(section=None):
    """Column Mn from the nominal P-M surface at the governing axial load.

    sp.column_nominal_moment_y() is a singly-reinforced BEAM formula: it
    counts only max(top, bottom) bars -- ignoring the side steel and the
    opposite face -- and assumes zero axial load. For a column that
    understates Mn by roughly 2x at zero axial and 4x under service
    compression, which drove this search to demand columns several times
    larger than ACI actually requires, and made SCWB unsatisfiable for
    frames that in fact comply. Model/IMK_Hinges.py already takes hinge
    capacity off the P-M surface for exactly this reason; using it here
    makes the design side agree with the model that gets analysed.

    section is (b, h, fc); defaults to the section currently installed.
    """
    if section is None:
        section = (sp.B_COL, sp.H_COL, sp.FC_COL_KSI)
    b, h, fc = section
    diagram = column_pm_nominal_for(b, h, fc, _col_steel_layers(h=h))
    return column_moment_at_axial(_scwb_governing_axial(), diagram)


def _next_larger_column_index(ladder, index):
    """First rung with a strictly larger cross-section than ladder[index].

    The ladder interleaves concrete strengths within each size (18x18 at 4, 5,
    6, 8 ksi, then 20x20 at 4, ...), so stepping one rung usually only raises
    f'c. That is the wrong lever for a gravity stability failure: flexural
    stiffness goes as E*I, and E rises with sqrt(f'c), so 5 -> 8 ksi buys about
    26% while 18 -> 20 in buys (20/18)^4 = 1.52x. Skip to the next real size.

    Returns None when the ladder has no larger section.
    """
    b, h, _fc = ladder[index]
    for candidate in range(index + 1, len(ladder)):
        cb, ch, _ = ladder[candidate]
        if cb > b or ch > h:
            return candidate
    return None


def _smallest_scwb_column_index(ladder):
    """First column rung satisfying strong-column/weak-beam.

    Returns (index, satisfied). When no rung can satisfy the rule the largest
    is returned with satisfied=False, so an exhausted ladder is distinguishable
    from a real match -- previously both came back as a bare index and running
    out of column looked identical to succeeding on the last one.
    """
    required = _scwb_required_column_moment()
    for index, section in enumerate(ladder):
        if _column_nominal_moment(section) >= required:
            return index, True
    return len(ladder) - 1, False


def design_structure(cfg=None, max_section_iter=6, max_steel_iter=6, verbose=True):
    """Design the current geometry to the configured DCR band.

    Returns a JSON-serializable record of the final sections, reinforcement,
    governing DCRs, and the ELF demand they were designed against.
    """
    cfg = cfg or DesignConfig.from_structure_parameters()
    target = cfg.dcr.dcr_target
    band_lo, band_hi = cfg.dcr.dcr_band_lo, cfg.dcr.dcr_band_hi

    span = max(sp.BAY_X, sp.BAY_Y)
    columns = column_ladder()
    beams = beam_ladder(span_in=span, story_height_in=sp.STORY_H)

    column_index = nearest_rung_index(columns, sp.B_COL, sp.H_COL, sp.FC_COL_KSI)
    beam_index = nearest_rung_index(beams, sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI)

    history = []
    visited = set()
    best = None
    gravity_failures = []

    # Gravity escalations get their own budget. They are not design
    # iterations -- "this section cannot stand up" is a search step, not an
    # evaluation -- and letting them consume max_section_iter left case_0013
    # with 4 escalations, 2 real iterations, and an overstressed column at
    # DCR 1.10. The escalation budget is bounded by the ladder itself.
    iteration = 0
    gravity_escalations = 0
    while iteration < max_section_iter:
        _apply_rung(columns[column_index], "column")
        _apply_rung(beams[beam_index], "beam")
        _sync_cfg_to_sp(cfg)
        validate_rung(columns[column_index], "column")
        validate_rung(beams[beam_index], "beam", span_in=span, story_height_in=sp.STORY_H)

        period = _model_period()
        try:
            worst, elf = _steel_pass(cfg, period, max_steel_iter)
        except RuntimeError as error:
            if "gravity analysis failed" not in str(error).lower():
                raise
            # This section cannot stand up under its own gravity load. Columns
            # use a PDelta transform, so that is a stability failure, not a
            # numerical one: case_0013 (9 stories, 117 ft, 18x18 columns) went
            # unstable at 85% of applied gravity. Escalating the column is the
            # correct response. Aborting the case -- the old behaviour -- lost
            # it entirely, and because only tall slender frames fail this way,
            # that silently biased the dataset against exactly the buildings
            # most worth having in it.
            gravity_failures.append(
                {
                    # Escalation order, not design iteration: `iteration` is
                    # deliberately not advanced here, so recording it would
                    # label every escalation "1".
                    "escalation": gravity_escalations + 1,
                    "design_iterations_used": iteration,
                    "column_section": list(columns[column_index]),
                    "beam_section": list(beams[beam_index]),
                    "error": str(error),
                }
            )
            larger = _next_larger_column_index(columns, column_index)
            if larger is None:
                raise RuntimeError(
                    f"{error} No column in the ladder can carry gravity for "
                    f"this geometry ({sp.NUM_FLOOR} stories, "
                    f"{sp.NUM_FLOOR * sp.STORY_H:.0f} in tall); the ladder "
                    f"tops out at {tuple(columns[-1])}."
                ) from error
            column_index = larger
            gravity_escalations += 1
            if gravity_escalations > len(columns):
                raise RuntimeError(
                    "Gravity escalation did not terminate; the column ladder "
                    "is inconsistent."
                ) from error
            continue

        iteration += 1
        scwb_ok = _column_nominal_moment() >= _scwb_required_column_moment()

        entry = {
            "iteration": iteration,
            "column_section": list(columns[column_index]),
            "beam_section": list(beams[beam_index]),
            "column_dcr": worst["column"],
            "beam_dcr": worst["beam"],
            "model_period_sec": period,
            "base_shear_kip": (elf or {}).get("base_shear_kip"),
            "column_bars": [sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS],
            "beam_bars": [sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS],
            "scwb_satisfied": scwb_ok,
            # Must come off the same P-M capacity as scwb_ok above. Leaving
            # this on sp.column_nominal_moment_y() recorded the old beam-formula
            # ratio beside the new pass/fail flag, so a history row could read
            # "1.171, satisfied" against a reported 2.971 for the same section.
            "scwb_ratio": (
                _column_nominal_moment() / sp.beam_nominal_moment_y()
                if sp.beam_nominal_moment_y() > 0 else None
            ),
            "beam_at_ladder_floor": beam_index == 0,
            "column_at_ladder_floor": column_index == 0,
        }
        history.append(entry)
        if verbose:
            print(
                "  [design] iter {}: col {:.0f}x{:.0f} fc{:.0f} DCR={:.3f} | "
                "beam {:.0f}x{:.0f} fc{:.0f} DCR={:.3f} | T1={:.3f}s".format(
                    iteration,
                    columns[column_index][0], columns[column_index][1], columns[column_index][2],
                    worst["column"],
                    beams[beam_index][0], beams[beam_index][1], beams[beam_index][2],
                    worst["beam"],
                    period or 0.0,
                )
            )

        # Beams are the intended yielding elements, so they carry the DCR
        # band. Columns are capacity-protected by ACI 318-19 18.7.3: forcing
        # them into the same band would require them to be weaker than the
        # beams they are required to out-strength, which is precisely the
        # soft-story behaviour the model exists to study. They get a strength
        # ceiling and the SCWB rule instead.
        deviation = abs(worst["beam"] - target)
        if max(worst["beam"], worst["column"]) > cfg.dcr.dcr_hard_max:
            deviation += 10.0
        if not scwb_ok:
            deviation += 5.0
        if best is None or deviation < best["deviation"]:
            best = {"deviation": deviation, "entry": entry, "state": _capture_state()}

        # A design is only finished when the beam is in band, the column is
        # under its strength ceiling, AND strong column / weak beam holds.
        # Leaving SCWB out let an early in-band beam terminate the search with
        # columns weaker than the beams framing into them.
        accepted = (
            band_lo <= worst["beam"] <= band_hi
            and worst["column"] <= cfg.dcr.dcr_hard_max
            and scwb_ok
        )
        if accepted:
            break

        key = (column_index, beam_index)
        if key in visited:
            break
        visited.add(key)

        next_beam = suggest_rung_index(beams, beam_index, max(worst["beam"], 1e-6), target)

        # Size the column to the larger of what strength and SCWB demand.
        strength_index = suggest_rung_index(
            columns, column_index, max(worst["column"], 1e-6), cfg.dcr.dcr_hard_max
        )
        scwb_index, _scwb_reachable = _smallest_scwb_column_index(columns)
        next_column = max(strength_index, scwb_index)

        if next_column == column_index and next_beam == beam_index:
            break
        column_index, beam_index = next_column, next_beam

    if best is None:
        raise RuntimeError(
            f"No usable design found in {max_section_iter} section iterations; "
            f"{len(gravity_failures)} of them could not carry gravity. "
            "Raise max_section_iter or extend the column ladder."
        )

    _restore_state(best["state"])
    _sync_cfg_to_sp(cfg)
    final = best["entry"]
    governing = max(final["column_dcr"], final["beam_dcr"])

    # Evaluated against the restored (final) section, so the reported figures
    # describe the design that is actually written out.
    column_mn = _column_nominal_moment()
    beam_mn = sp.beam_nominal_moment_y()
    scwb_satisfied = column_mn >= _scwb_required_column_moment()

    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "geometry": {
            "num_bay_x": sp.NUM_BAY_X,
            "num_bay_y": sp.NUM_BAY_Y,
            "num_floor": sp.NUM_FLOOR,
            "bay_x_in": sp.BAY_X,
            "bay_y_in": sp.BAY_Y,
            "story_h_in": sp.STORY_H,
        },
        "sections": {
            "b_col_in": sp.B_COL,
            "h_col_in": sp.H_COL,
            "fc_col_ksi": sp.FC_COL_KSI,
            "b_beam_in": sp.B_BEAM,
            "h_beam_in": sp.H_BEAM,
            "fc_beam_ksi": sp.FC_BEAM_KSI,
        },
        "reinforcement": {
            "col_bar_size": sp.COL_BAR_SIZE,
            "col_top_bars": sp.COL_TOP_BARS,
            "col_bot_bars": sp.COL_BOT_BARS,
            "col_side_bars": sp.COL_SIDE_BARS,
            "beam_bar_size": sp.BEAM_BAR_SIZE,
            "beam_top_bars": sp.BEAM_TOP_BARS,
            "beam_bot_bars": sp.BEAM_BOT_BARS,
        },
        "dcr": {
            "column": final["column_dcr"],
            "beam": final["beam_dcr"],
            "governing": governing,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "beam_in_band": band_lo <= final["beam_dcr"] <= band_hi,
            "column_within_ceiling": final["column_dcr"] <= cfg.dcr.dcr_hard_max,
            # SCWB belongs here. The search loop already refused to stop
            # without it, but this published flag left it out, so a design
            # that fell back after exhausting the ladder with columns weaker
            # than their beams was still reported as accepted -- and anything
            # downstream filtering on this flag believed it.
            "accepted": (
                band_lo <= final["beam_dcr"] <= band_hi
                and final["column_dcr"] <= cfg.dcr.dcr_hard_max
                and scwb_satisfied
            ),
            "exceeds_capacity": governing > cfg.dcr.dcr_hard_max,
            "target_basis": "beam flexure carries the DCR band; columns are capacity-protected",
            "governed_by": (
                "demand"
                if band_lo <= final["beam_dcr"] <= band_hi
                else "minimum_section"
                if final.get("beam_at_ladder_floor") and final["beam_dcr"] < band_lo
                else "search_limit"
            ),
        },
        "scwb": {
            "ratio_min": sp.SCWB_RATIO_MIN,
            "column_nominal_moment_kip_in": column_mn,
            "beam_nominal_moment_kip_in": beam_mn,
            "column_axial_kip": _scwb_governing_axial(),
            "column_moment_basis": "nominal P-M surface at the governing axial load",
            "ratio_provided": (column_mn / beam_mn if beam_mn > 0 else None),
            "satisfied": scwb_satisfied,
        },
        "seismic": {
            "sds": sp.ASCE_SDS,
            "sd1": sp.ASCE_SD1,
            "s1": sp.ASCE_S1,
            "r": sp.ASCE_R,
            "site_label": getattr(sp, "SEISMIC_SITE_LABEL", None),
        },
        "demand": {
            "basis": "ASCE 7-22 12.8 ELF, combination 1.2D + 0.5L + E",
            "model_period_sec": final["model_period_sec"],
            "base_shear_kip": final["base_shear_kip"],
        },
        "iterations": len(history),
        "gravity_failures": gravity_failures,
        "history": history,
    }


def apply_design(record):
    """Apply a stored design artifact to Structure_Parameters."""
    sections = record["sections"]
    rebar = record["reinforcement"]
    sp.B_COL = sections["b_col_in"]
    sp.H_COL = sections["h_col_in"]
    sp.FC_COL_KSI = sections["fc_col_ksi"]
    sp.B_BEAM = sections["b_beam_in"]
    sp.H_BEAM = sections["h_beam_in"]
    sp.FC_BEAM_KSI = sections["fc_beam_ksi"]
    sp.COL_BAR_SIZE = rebar["col_bar_size"]
    sp.COL_TOP_BARS = rebar["col_top_bars"]
    sp.COL_BOT_BARS = rebar["col_bot_bars"]
    sp.COL_SIDE_BARS = rebar["col_side_bars"]
    sp.COL_BAR_AREA = sp.rebar_area(sp.COL_BAR_SIZE)
    sp.BEAM_BAR_SIZE = rebar["beam_bar_size"]
    sp.BEAM_TOP_BARS = rebar["beam_top_bars"]
    sp.BEAM_BOT_BARS = rebar["beam_bot_bars"]
    sp.BEAM_BAR_AREA = sp.rebar_area(sp.BEAM_BAR_SIZE)
    return record


def load_or_create_design(design_path, cfg=None, verbose=True):
    """Read a cached design artifact, or run the design and write it.

    Returns (record, created). The artifact is written atomically so a
    concurrent generation worker never reads a half-written design.
    """
    design_path = Path(design_path)
    if design_path.exists():
        record = json.loads(design_path.read_text(encoding="utf-8"))
        apply_design(record)
        return record, False

    record = design_structure(cfg=cfg, verbose=verbose)
    design_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = design_path.with_name(f".{design_path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, design_path)
    return record, True
