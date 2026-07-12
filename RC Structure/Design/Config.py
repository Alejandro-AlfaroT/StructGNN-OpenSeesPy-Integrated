"""
Design/Config.py
Configuration dataclasses for the RC frame design pipeline.

All design-decision parameters live here.  Structure_Parameters.py retains
geometry scalars and OpenSees solver settings that feed the model builder
directly; this module is the design-decision layer on top of it.

Usage
-----
    from Design.Config import DesignConfig

    cfg = DesignConfig()                          # all defaults from sp
    cfg = DesignConfig(dcr=DCRTargets(dcr_band_lo=0.70, dcr_target=0.80))
    cfg = DesignConfig.from_structure_parameters()   # explicit mirror of sp
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import Structure_Parameters as sp


# ---------------------------------------------------------------------------
# Scalar-or-range type
# ---------------------------------------------------------------------------
# A parameter that can be fixed (scalar) or drawn uniformly from [lo, hi]
# during a batch generation run.
ScalarOrRange = Union[float, Tuple[float, float]]


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class RebarConfig:
    """Longitudinal and transverse reinforcement constraints.

    bar_sizes_col / bar_sizes_beam
        ASTM standard bar numbers available for selection (e.g. 8 → #8).

    col_n_top_range / beam_n_range
        Inclusive (min, max) bar count search bounds.
        Column: n_top == n_bot enforced (symmetric section).
        Beam: n_top and n_bot solved independently.

    col_n_side_options
        Candidate side-bar counts per face (total side = 2 × value).

    rho_col_min / rho_col_max
        ACI 318-19 §10.6.1.1 longitudinal steel ratio limits.

    cover_in
        Clear cover to longitudinal bar centroid (in).

    stirrup_*
        Transverse reinforcement properties used for shear checks.
        These were previously hardcoded in RC_Design_Check.py.

    section_round_increment_in
        Snap section dimensions to this increment (0 = no snapping).
    """
    bar_sizes_col: List[int] = field(
        default_factory=lambda: [5, 6, 7, 8, 9, 10, 11]
    )
    bar_sizes_beam: List[int] = field(
        default_factory=lambda: [4, 5, 6, 7, 8, 9, 10]
    )

    col_n_top_range: Tuple[int, int] = (2, 6)     # inclusive; n_bot = n_top
    col_n_side_options: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4]    # bars per face
    )
    beam_n_range: Tuple[int, int] = (2, 7)         # inclusive, per layer

    rho_col_min: float = 0.01
    rho_col_max: float = 0.08

    cover_in: float = field(default_factory=lambda: sp.COVER)

    # Transverse reinforcement
    stirrup_bar_size: int = field(default_factory=lambda: sp.COL_STIRRUP_BAR_SIZE)
    stirrup_legs: int = field(default_factory=lambda: sp.COL_STIRRUP_LEGS)
    col_stirrup_bar_size: int = field(default_factory=lambda: sp.COL_STIRRUP_BAR_SIZE)
    col_stirrup_legs: int = field(default_factory=lambda: sp.COL_STIRRUP_LEGS)
    beam_stirrup_bar_size: int = field(default_factory=lambda: sp.BEAM_STIRRUP_BAR_SIZE)
    beam_stirrup_legs: int = field(default_factory=lambda: sp.BEAM_STIRRUP_LEGS)
    stirrup_spacing_col_in: float = field(default_factory=lambda: sp.COL_STIRRUP_SPACING)
    stirrup_spacing_beam_in: float = field(default_factory=lambda: sp.BEAM_STIRRUP_SPACING)
    stirrup_spacing_min_in: float = field(default_factory=lambda: sp.STIRRUP_MIN_SPACING)
    stirrup_spacing_step_in: float = field(default_factory=lambda: sp.STIRRUP_SPACING_STEP)

    section_round_increment_in: float = 2.0

    # ---------------------------------------------------------------------------
    def col_n_top_iter(self):
        lo, hi = self.col_n_top_range
        return range(lo, hi + 1)

    def beam_n_iter(self):
        lo, hi = self.beam_n_range
        return range(lo, hi + 1)

    @property
    def stirrup_area_in2(self) -> float:
        """Total stirrup shear area (both legs) in²."""
        return self.stirrup_legs * sp.rebar_area(self.stirrup_bar_size)

    @property
    def col_stirrup_area_in2(self) -> float:
        """Total column stirrup/tie shear area in in^2."""
        return self.col_stirrup_legs * sp.rebar_area(self.col_stirrup_bar_size)

    @property
    def beam_stirrup_area_in2(self) -> float:
        """Total beam stirrup shear area in in^2."""
        return self.beam_stirrup_legs * sp.rebar_area(self.beam_stirrup_bar_size)


@dataclass
class MaterialConfig:
    """Concrete and steel material properties.

    Each value may be a fixed float or a (lo, hi) tuple for randomised
    sampling.  Call .resolve() to get a MaterialConfig with all fields
    as scalars before using them in calculations.
    """
    fc_col_ksi: ScalarOrRange = field(default_factory=lambda: sp.FC_COL_KSI)
    fc_beam_ksi: ScalarOrRange = field(default_factory=lambda: sp.FC_BEAM_KSI)
    fy_ksi: ScalarOrRange = field(default_factory=lambda: sp.FY_KSI)
    es_ksi: float = field(default_factory=lambda: sp.ES_KSI)

    def resolve(self, rng=None) -> "MaterialConfig":
        """Return a MaterialConfig with all ScalarOrRange fields as floats.

        Parameters
        ----------
        rng : numpy.random.Generator or None
            If provided, uses rng.uniform for reproducible draws.
            If None, falls back to the standard library random module.
        """
        def _draw(v):
            if isinstance(v, tuple):
                lo, hi = v
                return rng.uniform(lo, hi) if rng is not None else random.uniform(lo, hi)
            return float(v)

        return MaterialConfig(
            fc_col_ksi=_draw(self.fc_col_ksi),
            fc_beam_ksi=_draw(self.fc_beam_ksi),
            fy_ksi=_draw(self.fy_ksi),
            es_ksi=self.es_ksi,
        )

    def is_resolved(self) -> bool:
        return all(
            isinstance(v, (int, float))
            for v in (self.fc_col_ksi, self.fc_beam_ksi, self.fy_ksi)
        )


@dataclass
class SectionConfig:
    """Member cross-section dimensions."""
    b_col_in: float = field(default_factory=lambda: sp.B_COL)
    h_col_in: float = field(default_factory=lambda: sp.H_COL)
    b_beam_in: float = field(default_factory=lambda: sp.B_BEAM)
    h_beam_in: float = field(default_factory=lambda: sp.H_BEAM)


@dataclass
class LoadConfig:
    """Gravity and lateral load combination settings.

    mode
        "gravity"          → gravity-only analysis (D + L).
        "gravity_lateral"  → gravity + seismic (ASCE 7 load combination).

    factor_dead / factor_live
        ASCE 7-22 §2.3.1 load combination factors for strength design.
        Default: 1.2D + 1.6L (combination 2).

    strong_column_weak_beam
        When True and mode == "gravity_lateral", enforce ACI 318-19 §18.7.3:
        ΣφMnc / ΣφMnb ≥ scwb_ratio_min at every beam-column joint.
        TODO: joint-level capacity ratio check is scaffolded in
        Design/Phase2/AcceptanceCriteria.py.
    """
    mode: str = "gravity"

    factor_dead: float = 1.2
    factor_live: float = 1.6

    strong_column_weak_beam: bool = False
    scwb_ratio_min: float = 1.2

    def is_lateral(self) -> bool:
        return self.mode == "gravity_lateral"


@dataclass
class DCRTargets:
    """Demand/capacity ratio acceptance criteria and optimisation target.

    Hard constraint
        DCR > dcr_hard_max → member fails; design is rejected.
        Default = 1.0 (ACI strength design).

    Soft band  [dcr_band_lo, dcr_band_hi]
        Optimisation feasibility region.  Designs within the band are
        preferred; those outside incur a penalty in the objective function
        but are not hard-rejected unless DCR > dcr_hard_max.

    dcr_target
        Centroid used for the "min_deviation" objective:
            objective = Σ (DCR_i − dcr_target)²
        A value inside the soft band (e.g. 0.85) centres the distribution.

    objective
        "min_deviation"  minimize Σ (DCR_i − dcr_target)²  ← default
        "min_volume"     minimize total longitudinal steel volume
    """
    dcr_hard_max: float = 1.0

    dcr_band_lo: float = field(default_factory=lambda: sp.DESIGN_DCR_MIN)
    dcr_band_hi: float = field(default_factory=lambda: sp.DESIGN_DCR_MAX)
    dcr_target: float = 0.85

    objective: str = "min_deviation"

    def in_band(self, dcr: float) -> bool:
        return self.dcr_band_lo <= dcr <= self.dcr_band_hi

    def hard_fail(self, dcr: float) -> bool:
        return dcr > self.dcr_hard_max


@dataclass
class IterationConfig:
    """Controls for the iterative redesign loop.

    convergence_tol
        The loop considers itself converged when the bar specification does
        not change between successive iterations (discrete bar search).
        This tolerance is also used as a guard for near-zero objective
        improvement in future continuous optimisation extensions.

    seed
        Random seed for reproducible batch generation runs (passed to
        MaterialConfig.resolve()).  None → non-reproducible.

    penalty_weight
        Multiplier applied to the objective for infeasible designs (DCR > 1.0
        or ACI limit violations) so that the optimizer still has a gradient to
        follow rather than encountering a silent rejection.
    """
    max_iter: int = 10
    convergence_tol: float = 1e-3
    seed: Optional[int] = None
    penalty_weight: float = 1e3


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class DesignConfig:
    """Root design configuration object.

    Aggregates all sub-configs.  Construct with defaults (reads from
    Structure_Parameters) or override any sub-config explicitly.

    Examples
    --------
    cfg = DesignConfig()
    cfg = DesignConfig(dcr=DCRTargets(dcr_band_lo=0.70, dcr_target=0.80))
    cfg = DesignConfig(loads=LoadConfig(mode="gravity_lateral",
                                        strong_column_weak_beam=True))
    """
    rebar: RebarConfig = field(default_factory=RebarConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    sections: SectionConfig = field(default_factory=SectionConfig)
    loads: LoadConfig = field(default_factory=LoadConfig)
    dcr: DCRTargets = field(default_factory=DCRTargets)
    iteration: IterationConfig = field(default_factory=IterationConfig)

    @classmethod
    def from_structure_parameters(cls) -> "DesignConfig":
        """Construct a DesignConfig that mirrors the current Structure_Parameters."""
        return cls(
            rebar=RebarConfig(
                cover_in=sp.COVER,
                col_stirrup_bar_size=sp.COL_STIRRUP_BAR_SIZE,
                col_stirrup_legs=sp.COL_STIRRUP_LEGS,
                beam_stirrup_bar_size=sp.BEAM_STIRRUP_BAR_SIZE,
                beam_stirrup_legs=sp.BEAM_STIRRUP_LEGS,
                stirrup_spacing_col_in=sp.COL_STIRRUP_SPACING,
                stirrup_spacing_beam_in=sp.BEAM_STIRRUP_SPACING,
                stirrup_spacing_min_in=sp.STIRRUP_MIN_SPACING,
                stirrup_spacing_step_in=sp.STIRRUP_SPACING_STEP,
            ),
            materials=MaterialConfig(
                fc_col_ksi=sp.FC_COL_KSI,
                fc_beam_ksi=sp.FC_BEAM_KSI,
                fy_ksi=sp.FY_KSI,
                es_ksi=sp.ES_KSI,
            ),
            sections=SectionConfig(
                b_col_in=sp.B_COL,
                h_col_in=sp.H_COL,
                b_beam_in=sp.B_BEAM,
                h_beam_in=sp.H_BEAM,
            ),
            dcr=DCRTargets(
                dcr_band_lo=sp.DESIGN_DCR_MIN,
                dcr_band_hi=sp.DESIGN_DCR_MAX,
            ),
        )
