"""
Design/Phase2/PushoverConfig.py
Configuration dataclass for nonlinear static (pushover) analysis.

Used by Analysis/Pushover.py and the Phase 2 evaluation pipeline.
Currently Analysis/Pushover.py reads settings directly from
Structure_Parameters; migrate it to accept a PushoverConfig once
Phase 2 is fully activated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import Structure_Parameters as sp


# ---------------------------------------------------------------------------
# Load pattern types
# ---------------------------------------------------------------------------

LOAD_PATTERN_UNIFORM      = "uniform"
LOAD_PATTERN_TRIANGULAR   = "triangular"
LOAD_PATTERN_FIRST_MODE   = "first_mode"

# ---------------------------------------------------------------------------
# Target displacement method types
# ---------------------------------------------------------------------------

TARGET_DISP_COEFFICIENT   = "coefficient_method"    # ASCE 41-17 §7.4.3
TARGET_DISP_CAPACITY_SPECTRUM = "capacity_spectrum"  # ATC-40 / FEMA 440 — TODO


@dataclass
class PushoverConfig:
    """
    Configuration for nonlinear static pushover analysis.

    Attributes
    ----------
    load_pattern
        Lateral force distribution applied during pushover.
        "uniform"     — equal force to every floor (ASCE 41-17 §7.4.3.3.2)
        "triangular"  — linearly increasing with height
        "first_mode"  — scaled by first-mode shape (requires prior modal run)

    control_node_floor
        Floor index (1-based) of the roof control node.  Defaults to the
        top floor (NUM_FLOOR).

    target_disp_method
        Method for computing the target displacement.
        "coefficient_method"  — ASCE 41-17 §7.4.3  (implemented)
        "capacity_spectrum"   — ATC-40 / FEMA 440   (TODO)

    target_roof_drift_ratio
        Roof drift ratio (Δ/H) used as the pushover target when
        target_disp_method is not the coefficient method.
        Default 0.04 (4 %).

    performance_level
        Target performance level for Phase 2 acceptance checks.
        "IO" | "LS" | "CP" (from ASCE 41-17).

    du_in
        Displacement increment per step (in).  Reads from sp by default.

    max_steps
        Maximum number of pushover steps.  Reads from sp by default.

    tol / max_iter / fallback_tol / fallback_max_iter
        Newton solver settings.
    """
    load_pattern: str = LOAD_PATTERN_UNIFORM
    control_node_floor: int = field(default_factory=lambda: sp.NUM_FLOOR)

    target_disp_method: str = TARGET_DISP_COEFFICIENT
    target_roof_drift_ratio: float = 0.04   # used when not coefficient method

    performance_level: str = "LS"

    du_in: float = field(default_factory=lambda: sp.PUSHOVER_DU)
    max_steps: int = field(default_factory=lambda: sp.PUSHOVER_STEPS)

    tol: float = field(default_factory=lambda: sp.PUSHOVER_TOL)
    max_iter: int = field(default_factory=lambda: sp.PUSHOVER_MAX_ITER)
    fallback_tol: float = field(default_factory=lambda: sp.PUSHOVER_FALLBACK_TOL)
    fallback_max_iter: int = field(
        default_factory=lambda: sp.PUSHOVER_FALLBACK_MAX_ITER
    )

    def target_disp_in(self) -> float:
        """Target displacement (in) from step count × increment."""
        return self.max_steps * self.du_in

    def is_coefficient_method(self) -> bool:
        return self.target_disp_method == TARGET_DISP_COEFFICIENT

    # TODO: implement coefficient_method() → target_disp_in using
    #       C0, C1, C2, Sa, T from ASCE 41-17 §7.4.3.3.
    def coefficient_method_target(
        self,
        Sa: float,
        T1: float,
        C0: float = 1.3,
        C1: float = 1.0,
        C2: float = 1.0,
    ) -> float:
        """
        ASCE 41-17 §7.4.3.3 target displacement estimate.

        δt = C0 · C1 · C2 · Sa · (T1²/4π²) · g

        Parameters
        ----------
        Sa  : spectral acceleration at T1 (g)
        T1  : fundamental period (sec)
        C0  : modification factor (shape factor, ASCE 41 Table 7-5)
        C1  : inelastic displacement modifier
        C2  : degradation modifier

        TODO: compute C1 and C2 from structure properties per ASCE 41 §7.4.3.
        """
        g_in_per_s2 = 386.4   # in/s²
        return C0 * C1 * C2 * Sa * (T1**2 / (4.0 * 3.14159265**2)) * g_in_per_s2
