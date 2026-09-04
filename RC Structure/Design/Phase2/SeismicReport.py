"""
Design/Phase2/SeismicReport.py


TODO
----
  - Implement print_phase2_summary() using RotationCheckResult and
    SCWBCheckResult from AcceptanceCriteria.py.
  - Add a Phase 2 section to the JSON sample output in Graph_Exporter.py.
"""
from __future__ import annotations

from typing import Dict, List


def print_phase2_summary(
    rotation_results: List,
    force_check_results: Dict,
    scwb_results: List,
    performance_level: str = "LS",
) -> None:
    """
    Print Phase 2 acceptance check summary.

    Parameters
    ----------
    rotation_results     : list of RotationCheckResult from RotationCheck
    force_check_results  : dict[ele_tag, bool] from ForceCheck
    scwb_results         : list of SCWBCheckResult from JointSCWBCheck
    performance_level    : "IO" | "LS" | "CP"

    TODO: implement table formatting and summary statistics.
    """
    print("\n" + "=" * 70)
    print(f"  Phase 2 Seismic Acceptance Check  (Performance Level: {performance_level})")
    print("=" * 70)

    # ── Hinge rotation results ─────────────────────────────────────────────
    print("\n  HINGE ROTATION ACCEPTANCE  (deformation-controlled)")
    if not rotation_results:
        print("  [TODO] No hinge rotation results available.")
        print("         Implement RotationCheck.evaluate() in AcceptanceCriteria.py.")
    else:
        scaffold_count = sum(1 for r in rotation_results if getattr(r, "scaffold", False))
        total = len(rotation_results)
        fail  = sum(1 for r in rotation_results if not r.ok)
        print(f"  Total hinge events : {total}")
        print(f"  Scaffold results   : {scaffold_count}  (not yet computed)")
        print(f"  Failures           : {fail}")

    # ── Force-controlled results ───────────────────────────────────────────
    print("\n  FORCE-CONTROLLED CHECKS  (DCR ≤ 1.0, Phase 1 lateral case)")
    if not force_check_results:
        print("  [TODO] Run Phase 1 checks under lateral load combination.")
    else:
        n_fail = sum(1 for ok in force_check_results.values() if not ok)
        print(f"  Elements checked : {len(force_check_results)}")
        print(f"  Failures         : {n_fail}")

    # ── Strong-column-weak-beam ────────────────────────────────────────────
    print("\n  STRONG-COLUMN-WEAK-BEAM  (ACI §18.7.3 / ASCE 41 §10.4.2.2)")
    if not scwb_results:
        print("  [TODO] Implement JointSCWBCheck.evaluate() in AcceptanceCriteria.py.")
    else:
        n_fail = sum(1 for r in scwb_results if not r.ok)
        print(f"  Joints checked : {len(scwb_results)}")
        print(f"  Failures       : {n_fail}")

    print("\n" + "=" * 70 + "\n")


def phase2_results_to_dict(
    rotation_results: List,
    force_check_results: Dict,
    scwb_results: List,
    performance_level: str = "LS",
) -> Dict:
    """
    Convert Phase 2 results to a JSON-serialisable dict for Graph_Exporter.

    TODO: extend Graph_Exporter.save_analysis_sample() to call this and
    include Phase 2 data in the graph sample.
    """
    return {
        "performance_level": performance_level,
        "rotation_checks": [
            {
                "ele_tag":       getattr(r, "ele_tag", 0),
                "end":           getattr(r, "end", ""),
                "rot_dir":       getattr(r, "rot_dir", ""),
                "demand_rad":    getattr(r, "demand_rad", 0.0),
                "capacity_rad":  getattr(r, "capacity_rad", 0.0),
                "dcr_rotation":  getattr(r, "dcr_rotation", 0.0),
                "ok":            getattr(r, "ok", True),
                "scaffold":      getattr(r, "scaffold", True),
            }
            for r in rotation_results
        ],
        "force_checks": {
            str(tag): ok for tag, ok in force_check_results.items()
        },
        "scwb_checks": [
            {
                "floor":                   getattr(r, "floor", 0),
                "grid_i":                  getattr(r, "grid_i", 0),
                "grid_j":                  getattr(r, "grid_j", 0),
                "sum_phi_Mn_col_kip_in":   getattr(r, "sum_phi_Mn_col_kip_in", 0.0),
                "sum_phi_Mn_beam_kip_in":  getattr(r, "sum_phi_Mn_beam_kip_in", 0.0),
                "ratio":                   getattr(r, "ratio", 0.0),
                "required_ratio":          getattr(r, "required_ratio", 1.2),
                "ok":                      getattr(r, "ok", True),
                "scaffold":                getattr(r, "scaffold", True),
            }
            for r in scwb_results
        ],
    }
