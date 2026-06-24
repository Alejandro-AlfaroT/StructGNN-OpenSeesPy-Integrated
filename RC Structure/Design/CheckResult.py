"""
Design/CheckResult.py
Typed result objects for Phase 1 (elastic) ACI 318-19 design checks.

These replace the flat dicts previously returned by RC_Design_Check.run_checks().

Structure
---------
  LimitStateResult   — one ACI limit state (P-M, flexure, shear, …)
  MemberCheckResult  — all limit states for a single frame element

The old flat-dict format is still accepted by RC_Design_Check.print_summary()
for backward compatibility.  Use to_legacy_dict() in Design/ACI_Checks.py to
convert when calling code that expects the old schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-limit-state result
# ---------------------------------------------------------------------------

@dataclass
class LimitStateResult:
    """Result for a single ACI 318-19 limit state check.

    Attributes
    ----------
    name
        Identifier: "PM" | "flexure_pos" | "flexure_neg" | "shear"
        | "slenderness" | "dev_length".

    demand
        Governing demand quantity in consistent units
        (kip·in for moment, kip for force).

    capacity
        φ-reduced capacity in the same units.

    dcr
        demand / capacity ratio.

    ok
        True when dcr ≤ 1.0 (hard ACI acceptance).

    min_controlled
        True when the capacity is set by the ACI code-minimum
        reinforcement (§9.6.1.2 for beams, §10.6.1.1 for columns),
        not by the actual demand.  The member is NOT rejected, but
        the flag is carried through to the GNN graph so the model
        can learn that this member was not demand-sized.
    """
    name: str
    demand: float
    capacity: float
    dcr: float
    ok: bool
    min_controlled: bool = False

    def deviation(self, target: float = 0.85) -> float:
        """Signed deviation of DCR from the optimisation target."""
        return self.dcr - target

    def squared_deviation(self, target: float = 0.85) -> float:
        return self.deviation(target) ** 2


# ---------------------------------------------------------------------------
# Per-element aggregated result
# ---------------------------------------------------------------------------

@dataclass
class MemberCheckResult:
    """Aggregated Phase 1 result for one frame element.

    Attributes
    ----------
    ele_tag
        OpenSees element tag.

    member_type
        "column" or "beam".

    limit_states
        Mapping from limit-state name to LimitStateResult.
        Populated by run_checks_phase1().

    governing_limit_state
        Name of the limit state with the highest DCR.

    ok
        True when all limit states pass (DCR ≤ 1.0).

    any_min_controlled
        True when at least one limit state's capacity is code-minimum governed.

    section_info
        Snapshot of the section properties used for this check:
        bar_size, bar counts, Ast (columns) or As_top/bot (beams), etc.

    penalty_reasons
        Non-empty list of strings when the design is infeasible and was
        penalised rather than silently rejected.  Carry through to the
        output report so the user can see why a design didn't meet criteria.
    """
    ele_tag: int
    member_type: str
    limit_states: Dict[str, LimitStateResult] = field(default_factory=dict)
    governing_limit_state: str = ""
    ok: bool = True
    any_min_controlled: bool = False
    section_info: Dict = field(default_factory=dict)
    penalty_reasons: List[str] = field(default_factory=list)

    # ── Convenience accessors ────────────────────────────────────────────────

    def dcr(self, limit_state: str) -> float:
        """DCR for a named limit state, or 0.0 if not present."""
        ls = self.limit_states.get(limit_state)
        return ls.dcr if ls is not None else 0.0

    def max_dcr(self) -> float:
        """Highest DCR across all limit states."""
        if not self.limit_states:
            return 0.0
        return max(ls.dcr for ls in self.limit_states.values())

    def governing_dcr(self) -> float:
        """DCR of the governing (highest-DCR) limit state."""
        return self.dcr(self.governing_limit_state)

    def objective_contribution(self, target: float = 0.85) -> float:
        """Sum of squared deviations across flexural limit states.

        Shear and scaffold limit states (slenderness, dev_length) are
        excluded from the deviation objective — they have no DCR target,
        only a hard pass/fail.
        """
        flex_names = {"PM", "flexure_pos", "flexure_neg"}
        return sum(
            ls.squared_deviation(target)
            for name, ls in self.limit_states.items()
            if name in flex_names
        )

    # ── Internal maintenance ─────────────────────────────────────────────────

    def _recompute(self) -> None:
        """Recompute derived fields from limit_states.

        Call after all limit states have been added.
        """
        if not self.limit_states:
            return

        scaffold_limit_states = {"slenderness", "dev_length"}
        hard_limit_states = {
            name: ls
            for name, ls in self.limit_states.items()
            if name not in scaffold_limit_states
        }
        active_limit_states = hard_limit_states or self.limit_states

        gov = max(active_limit_states.values(), key=lambda ls: ls.dcr)
        self.governing_limit_state = gov.name
        self.ok = all(ls.ok for ls in active_limit_states.values())
        self.any_min_controlled = any(
            ls.min_controlled for ls in active_limit_states.values()
        )
