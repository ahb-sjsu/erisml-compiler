"""Compiler tier definitions.

The compiler supports three tiers, each appropriate for a different deployment
context. Tier 1 is the silicon-target spine: Tiers 2 and 3 are extraction
frontends that feed structured input into the same evaluator.

    Tier 1  GEOMETRIC   FSM + Mahalanobis evaluator; pre-parsed structured input.
                        Bounded memory, deterministic dispatch, fixed-point feasible.
                        Castable to FPGA/ASIC. Targets: real-time safety interlocks,
                        autonomous-vehicle ethical gates, model-level kill switches.

    Tier 2  RULES       Natural-language text -> pattern/keyword extraction ->
                        same Tier 1 evaluator. Deterministic, offline, no network.
                        Targets: edge devices, on-device alignment, air-gapped contexts.

    Tier 3  LLM         Natural-language text -> LLM-driven extraction ->
                        same Tier 1 evaluator. Highest expressiveness, batch only.
                        Targets: research annotation, RLEF dataset generation,
                        corpus analysis, deep extraction.

Tiers 2 and 3 emit the same canonical IR that Tier 1 ingests. The evaluator
core is shared across all three tiers.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path


class CompilerTier(str, Enum):
    """Three tiers, listed in order of decreasing real-time feasibility and
    increasing expressive capability."""

    GEOMETRIC = "geometric"
    RULES = "rules"
    LLM = "llm"

    @classmethod
    def auto_detect(cls, input_path: Path) -> "CompilerTier":
        """Pick a sensible default tier based on the input file extension.

        - .json -> structured-input -> GEOMETRIC (Tier 1)
        - .txt / .md -> natural-language text -> RULES (Tier 2)
        - everything else: raise.

        LLM (Tier 3) is never auto-selected; it must be explicitly requested.
        """
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            return cls.GEOMETRIC
        if suffix in {".txt", ".md"}:
            return cls.RULES
        raise ValueError(
            f"Cannot auto-detect tier from extension {suffix!r}. "
            f"Specify --tier explicitly."
        )

    @property
    def silicon_castable(self) -> bool:
        """Whether this tier's evaluator core can be cast into silicon
        (fixed-point arithmetic, bounded memory, deterministic dispatch).

        Note: all three tiers share the same Tier-1 evaluator core; what
        differs is the input pathway. The boolean here describes whether the
        *full pipeline* fits silicon constraints, not whether the evaluator
        does.
        """
        return self == CompilerTier.GEOMETRIC

    @property
    def requires_network(self) -> bool:
        """Whether the tier requires network access (LLM API)."""
        return self == CompilerTier.LLM

    @property
    def description(self) -> str:
        descriptions = {
            CompilerTier.GEOMETRIC: (
                "Tier 1: FSM + Mahalanobis evaluator on pre-parsed structured "
                "input. Real-time, silicon-castable."
            ),
            CompilerTier.RULES: (
                "Tier 2: Rule-based natural-language extraction feeding the "
                "Tier-1 evaluator. Deterministic, offline."
            ),
            CompilerTier.LLM: (
                "Tier 3: LLM-driven natural-language extraction feeding the "
                "Tier-1 evaluator. Highest expressiveness; batch only."
            ),
        }
        return descriptions[self]
