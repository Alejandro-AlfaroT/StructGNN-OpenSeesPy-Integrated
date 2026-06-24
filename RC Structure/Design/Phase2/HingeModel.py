"""
Design/Phase2/HingeModel.py
Abstract interface and concrete scaffold subclasses for nonlinear hinge models.

Architecture
------------
HingeModel (ABC)
  ├── ASCE41DefaultHinge    — Table-based backbone (ASCE 41-17 Tables 10-7, 10-8)
  ├── ASCE41MFactorHinge    — m-factor approach (ASCE 41-17 §7.5)
  └── UserDefinedHinge      — user-supplied moment-rotation dict, usable now

The model builder (Model/IMK_Hinges.py) should call
    backbone = hinge_model.get_backbone(member_type, section_props)
and use the returned BackboneParams to set up the IMK material, making
the hinge source swappable without touching the rest of the pipeline.

TODO: implement ASCE41DefaultHinge and ASCE41MFactorHinge backbone logic.
      Current IMK_Hinges.py reads calibration from Structure_Parameters
      directly; migrate it to call get_backbone() once the tables are ready.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Backbone parameter container
# ---------------------------------------------------------------------------

@dataclass
class BackboneParams:
    """
    Moment-rotation backbone parameters compatible with the OpenSees
    IMKBilin (IMK Peak-Oriented) uniaxial material.

    Field names match the OpenSees IMKBilin argument order so callers can
    unpack directly:
        ops.uniaxialMaterial("IMKBilin", tag, K0,
            as_Plus, as_Neg,
            My_Plus, My_Neg,
            Lamda_S, Lamda_C, Lamda_A, Lamda_K,
            c_S, c_C, c_A, c_K,
            theta_p_Plus, theta_p_Neg,
            theta_pc_Plus, theta_pc_Neg,
            Res_Pos, Res_Neg,
            theta_u_Plus, theta_u_Neg,
            D_Plus, D_Neg)

    Attributes
    ----------
    K0               : initial rotational stiffness (kip·in/rad)
    as_pos / as_neg  : post-yield hardening ratio (dimensionless)
    My_pos / My_neg  : yield moment (kip·in)
    theta_p_pos/neg  : plastic rotation capacity (rad)
    theta_pc_pos/neg : post-cap rotation capacity (rad)
    theta_u_pos/neg  : ultimate rotation (rad)
    Res_pos / Res_neg: residual strength ratio
    lambda_*         : cyclic deterioration parameters (S, C, A, K)
    c_*              : exponent parameters (S, C, A, K)
    D_pos / D_neg    : rate-of-deterioration multipliers
    """
    K0: float = 0.0

    as_pos: float = 0.02
    as_neg: float = 0.02

    My_pos: float = 0.0
    My_neg: float = 0.0

    lambda_S: float = 10.0
    lambda_C: float = 10.0
    lambda_A: float = 10.0
    lambda_K: float = 10.0

    c_S: float = 1.0
    c_C: float = 1.0
    c_A: float = 1.0
    c_K: float = 1.0

    theta_p_pos: float  = 0.020
    theta_p_neg: float  = 0.020
    theta_pc_pos: float = 0.060
    theta_pc_neg: float = 0.060

    Res_pos: float = 0.20
    Res_neg: float = 0.20

    theta_u_pos: float = 0.120
    theta_u_neg: float = 0.120

    D_pos: float = 1.0
    D_neg: float = 1.0

    # Extra metadata (not passed to OpenSees but useful for reporting)
    source: str = ""          # e.g. "ASCE41_Table10-7", "UserDefined"
    performance_level: str = ""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class HingeModel(ABC):
    """
    Swappable interface for nonlinear hinge backbone parameters.

    Subclasses implement get_backbone() to return BackboneParams for a given
    member type and section properties.  The rest of the pipeline (IMK_Hinges,
    Pushover, etc.) calls this method rather than computing backbone params
    inline, so the hinge source can be changed in one place.
    """

    @abstractmethod
    def get_backbone(
        self,
        member_type: str,
        section_props: Dict,
        axial_load: float = 0.0,
    ) -> BackboneParams:
        """
        Return the moment-rotation backbone for one member end.

        Parameters
        ----------
        member_type   : "column" or "beam"
        section_props : dict with keys b_in, h_in, fc_ksi, fy_ksi,
                        Ast_in2 (columns) or As_top_in2/As_bot_in2 (beams),
                        L_in (clear length), cover_in
        axial_load    : axial compression (kip, positive) for columns

        Returns
        -------
        BackboneParams
        """

    def description(self) -> str:
        """Human-readable description of this hinge source."""
        return type(self).__name__


# ---------------------------------------------------------------------------
# Subclass 1: ASCE 41-17 default tables  (scaffold — TODO)
# ---------------------------------------------------------------------------

class ASCE41DefaultHinge(HingeModel):
    """
    ASCE 41-17 Table 10-7 / 10-8 default backbone parameters for RC beams
    and columns.

    TODO
    ----
    1. Implement interpolation into ASCE 41-17 Table 10-7 (beams) and
       Table 10-8 (columns) based on:
         - Columns: P/(Ag·f'c), ρ_bal, transverse reinforcement condition
         - Beams:   ρ, ρ', V/(bw·d·√f'c), transverse reinforcement
    2. Map ASCE 41 a, b, c parameters to BackboneParams fields.
    3. Add conforming vs. non-conforming transverse reinforcement flag.
    """

    def __init__(self, conforming_transverse: bool = True) -> None:
        self.conforming_transverse = conforming_transverse

    def get_backbone(
        self,
        member_type: str,
        section_props: Dict,
        axial_load: float = 0.0,
    ) -> BackboneParams:
        # TODO: replace with actual ASCE 41-17 table interpolation
        params = BackboneParams(source="ASCE41_Table_TODO")
        params.K0 = self._stiffness(section_props)
        params.My_pos = section_props.get("phi_Mn_pos", 1.0)
        params.My_neg = section_props.get("phi_Mn_neg", 1.0)
        return params

    @staticmethod
    def _stiffness(section_props: Dict) -> float:
        """Elastic rotational stiffness EI/L (kip·in/rad)."""
        E  = section_props.get("E_ksi",  3605.0)   # Ec for fc=4 ksi
        b  = section_props.get("b_in",   12.0)
        h  = section_props.get("h_in",   18.0)
        L  = section_props.get("L_in",   120.0)
        I  = b * h**3 / 12.0
        return E * I / L if L > 0 else 0.0

    def description(self) -> str:
        cond = "conforming" if self.conforming_transverse else "non-conforming"
        return f"ASCE41DefaultHinge ({cond} transverse reinforcement) — scaffold"


# ---------------------------------------------------------------------------
# Subclass 2: ASCE 41-17 m-factor approach  (scaffold — TODO)
# ---------------------------------------------------------------------------

class ASCE41MFactorHinge(HingeModel):
    """
    ASCE 41-17 §7.5 m-factor (force-based deformation) acceptance criteria.

    Used for linear procedures (LSP / LDP).  Produces equivalent backbone
    parameters scaled by m-factors for IO, LS, CP.

    TODO
    ----
    1. Implement m-factor lookup from ASCE 41-17 Tables 10-3 and 10-4.
    2. Map m · Qy to rotation acceptance criteria in BackboneParams.
    3. Add performance level selector.
    """

    def __init__(self, performance_level: str = "LS") -> None:
        self.performance_level = performance_level

    def get_backbone(
        self,
        member_type: str,
        section_props: Dict,
        axial_load: float = 0.0,
    ) -> BackboneParams:
        # TODO: m-factor lookup and mapping
        params = BackboneParams(
            source="ASCE41_MFactor_TODO",
            performance_level=self.performance_level,
        )
        params.K0 = ASCE41DefaultHinge._stiffness(section_props)
        return params

    def description(self) -> str:
        return f"ASCE41MFactorHinge (level={self.performance_level}) — scaffold"


# ---------------------------------------------------------------------------
# Subclass 3: User-defined backbone  (usable now)
# ---------------------------------------------------------------------------

class UserDefinedHinge(HingeModel):
    """
    User-supplied moment-rotation backbone parameters.

    Pass a dict of BackboneParams field values.  Any field not supplied
    falls back to the BackboneParams default.  Useful for sensitivity studies
    and for overriding specific parameters without reimplementing the full
    ASCE 41 procedure.

    Example
    -------
    hinge = UserDefinedHinge(
        beam_params={"theta_p_pos": 0.03, "theta_u_pos": 0.15},
        col_params= {"theta_p_pos": 0.02, "theta_u_pos": 0.10},
    )
    """

    def __init__(
        self,
        beam_params: Optional[Dict] = None,
        col_params: Optional[Dict] = None,
    ) -> None:
        self._beam = beam_params or {}
        self._col  = col_params  or {}

    def get_backbone(
        self,
        member_type: str,
        section_props: Dict,
        axial_load: float = 0.0,
    ) -> BackboneParams:
        overrides = self._col if member_type == "column" else self._beam
        params    = BackboneParams(source="UserDefined")
        params.K0 = ASCE41DefaultHinge._stiffness(section_props)
        for k, v in overrides.items():
            if hasattr(params, k):
                setattr(params, k, v)
        return params

    def description(self) -> str:
        return "UserDefinedHinge"
