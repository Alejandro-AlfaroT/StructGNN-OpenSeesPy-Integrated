"""
Design/Phase2/AcceptanceCriteria.py
Phase 2 performance-based acceptance criteria scaffold.

Separates deformation-controlled checks (rotation demand vs. acceptance
criteria) from force-controlled checks (DCR ≤ 1.0 from Phase 1).

Structure
---------
  PerformanceLevel  — IO / LS / CP enum
  RotationCriteria  — a, b, c parameters per ASCE 41-17
  RotationCheck     — compares rotation demand to acceptance criteria
  ForceCheck        — wraps Phase 1 DCR ≤ 1.0 check for lateral case
  JointSCWBCheck    — strong-column-weak-beam scaffold

TODO
----
  - Populate ASCE 41-17 Table 10-7 (beam) and Table 10-8 (column) entries.
  - Implement RotationCheck.evaluate() using actual hinge rotation output
    from the pushover analysis (currently a placeholder).
  - Implement JointSCWBCheck.evaluate() with joint connectivity lookup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Performance level
# ---------------------------------------------------------------------------

class PerformanceLevel(str, Enum):
    """
    ASCE 41-17 §2.3.1 structural performance levels.

    IO  — Immediate Occupancy
    LS  — Life Safety
    CP  — Collapse Prevention
    """
    IO = "IO"
    LS = "LS"
    CP = "CP"

    @classmethod
    def from_str(cls, s: str) -> "PerformanceLevel":
        return cls(s.upper())


# ---------------------------------------------------------------------------
# ASCE 41-17 rotation acceptance criteria parameters
# ---------------------------------------------------------------------------

@dataclass
class RotationCriteria:
    """
    ASCE 41-17 deformation acceptance criteria for one member end.

    Corresponds to the a, b, c parameters in ASCE 41-17 Tables 10-7/10-8
    and the acceptance criteria θ values in those same tables.

    Attributes
    ----------
    a           : plastic rotation capacity (chord rotation, rad)
    b           : total rotation at loss of lateral strength (rad)
    c           : residual strength ratio
    theta_IO    : acceptance criteria — Immediate Occupancy (rad)
    theta_LS    : acceptance criteria — Life Safety (rad)
    theta_CP    : acceptance criteria — Collapse Prevention (rad)
    member_type : "column" | "beam"
    condition   : conforming ("C") or non-conforming ("NC") transverse reinf.

    TODO: populate from ASCE 41-17 Tables 10-7 and 10-8 via interpolation
    """
    a: float = 0.02          # plastic rotation capacity
    b: float = 0.05          # total rotation at strength loss
    c: float = 0.20          # residual strength ratio

    theta_IO: float = 0.005  # IO acceptance limit
    theta_LS: float = 0.015  # LS acceptance limit
    theta_CP: float = 0.020  # CP acceptance limit

    member_type: str = "beam"
    condition: str = "C"     # "C" = conforming, "NC" = non-conforming

    def acceptance_limit(self, level: PerformanceLevel) -> float:
        """Return θ acceptance limit (rad) for the given performance level."""
        return {
            PerformanceLevel.IO: self.theta_IO,
            PerformanceLevel.LS: self.theta_LS,
            PerformanceLevel.CP: self.theta_CP,
        }[level]


# ---------------------------------------------------------------------------
# Rotation check result
# ---------------------------------------------------------------------------

@dataclass
class RotationCheckResult:
    """
    Result of a deformation-controlled (rotation) acceptance check.

    Attributes
    ----------
    ele_tag         : element tag
    end             : "i" or "j"
    rot_dir         : "pos" or "neg" (moment direction)
    demand_rad      : rotation demand from pushover (rad)
    capacity_rad    : acceptance limit for the target performance level (rad)
    dcr_rotation    : demand / capacity
    ok              : dcr_rotation ≤ 1.0
    performance_level : "IO" | "LS" | "CP"
    scaffold        : True when the result is a placeholder (not yet computed)
    """
    ele_tag: int = 0
    end: str = "i"
    rot_dir: str = "pos"
    demand_rad: float = 0.0
    capacity_rad: float = 0.0
    dcr_rotation: float = 0.0
    ok: bool = True
    performance_level: str = "LS"
    scaffold: bool = True


# ---------------------------------------------------------------------------
# Rotation check
# ---------------------------------------------------------------------------

class RotationCheck:
    """
    Compare hinge rotation demands to ASCE 41-17 acceptance criteria.

    TODO
    ----
    1. Read rotation demands from the pushover hinge event log
       (MechanismTracker in Analysis/Mechanism_Checks.py tracks θ_peak).
    2. Call criteria.acceptance_limit(performance_level) for each hinge.
    3. Return a RotationCheckResult per hinge end.

    Currently returns placeholder scaffold results.
    """

    def __init__(
        self,
        performance_level: PerformanceLevel = PerformanceLevel.LS,
        hinge_model=None,        # HingeModel instance (from HingeModel.py)
    ) -> None:
        self.performance_level = performance_level
        self.hinge_model       = hinge_model

    def evaluate(
        self,
        hinge_events: List[Dict],
        element_props: Dict[int, Dict],
    ) -> List[RotationCheckResult]:
        """
        Evaluate rotation acceptance for all hinge events.

        Parameters
        ----------
        hinge_events    : list of hinge event dicts from MechanismTracker
        element_props   : dict[ele_tag, section_props] for capacity lookup

        Returns
        -------
        list of RotationCheckResult
            One entry per hinge end per direction that reached θ > 0.
        """
        # TODO: implement full ASCE 41 evaluation
        results: List[RotationCheckResult] = []
        for event in hinge_events:
            results.append(RotationCheckResult(
                ele_tag=event.get("physical_ele_tag", 0),
                end=event.get("end", "i"),
                rot_dir=event.get("rot_dir", "pos"),
                demand_rad=event.get("theta_peak", 0.0),
                capacity_rad=0.0,     # TODO: lookup from RotationCriteria
                dcr_rotation=0.0,     # TODO: compute
                ok=True,              # TODO: compute
                performance_level=self.performance_level.value,
                scaffold=True,
            ))
        return results


# ---------------------------------------------------------------------------
# Force check (Phase 1 DCR for lateral case)
# ---------------------------------------------------------------------------

class ForceCheck:
    """
    Force-controlled acceptance check: DCR ≤ 1.0 for each limit state.

    This wraps the Phase 1 check results for use in the Phase 2 evaluation
    pipeline.  Force-controlled members (e.g. shear-critical columns) are
    checked here; deformation-controlled members go through RotationCheck.

    Per ASCE 41-17 §7.5.2, the force-controlled demand uses a lower-bound
    estimate of capacity.  The current implementation uses the same φ-reduced
    capacity as Phase 1 (conservative for force-controlled actions).
    """

    def evaluate(self, phase1_results: Dict) -> Dict[int, bool]:
        """
        Return dict[ele_tag, ok] from Phase 1 check results.

        Parameters
        ----------
        phase1_results : dict[int, MemberCheckResult] or legacy flat dict

        Returns
        -------
        dict[int, bool]  — True when all force-controlled limit states pass.
        """
        out = {}
        for tag, r in phase1_results.items():
            # Accept both MemberCheckResult and legacy flat dict
            if hasattr(r, "ok"):
                out[tag] = r.ok
            else:
                out[tag] = r.get("ok", True)
        return out


# ---------------------------------------------------------------------------
# Strong-column-weak-beam check (scaffold)
# ---------------------------------------------------------------------------

@dataclass
class SCWBCheckResult:
    """
    ACI 318-19 §18.7.3 / ASCE 41-17 joint capacity ratio result.

    TODO: compute sum_phi_Mn_col and sum_phi_Mn_beam from element properties
    and current axial load during the lateral analysis.
    """
    floor: int = 0
    grid_i: int = 0
    grid_j: int = 0
    sum_phi_Mn_col_kip_in: float = 0.0
    sum_phi_Mn_beam_kip_in: float = 0.0
    ratio: float = 0.0
    required_ratio: float = 1.2
    ok: bool = True
    scaffold: bool = True


class JointSCWBCheck:
    """
    Strong-column-weak-beam capacity ratio check at every beam-column joint.

    TODO
    ----
    1. Iterate over all interior joints in the structural grid.
    2. For each joint, collect φMnc for columns above and below at the
       axial load level from the gravity analysis.
    3. Collect φMnb for beams framing into the joint in each direction.
    4. Compute ratio = ΣφMnc / ΣφMnb and compare to required_ratio.
    """

    def __init__(self, required_ratio: float = 1.2) -> None:
        self.required_ratio = required_ratio

    def evaluate(self, phase1_results: Dict) -> List[SCWBCheckResult]:
        """
        Return scaffold SCWB results.

        TODO: implement joint-level capacity ratio computation.
        """
        # Placeholder — returns an empty list until implemented
        return []
