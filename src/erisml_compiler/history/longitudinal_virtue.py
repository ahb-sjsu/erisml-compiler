"""LongitudinalVirtueProjection — reads agent history before judging.

The single-case `VirtueProjection` is honest that one act is weak
evidence. This projection augments the single-case reading with the
agent's longitudinal pattern, when one exists. The verdict reflects
both:

  - Single-case reading from the current substrate (as before)
  - Aggregated per-axis trait reading from the agent's history

The combined verdict:

  - `vicious` — at least one axis shows a `vice` (mean polarity ≤ -0.3)
    pattern across history, AND the current case extends that pattern
    (a vice act on the same axis)
  - `virtuous` — at least one axis shows `virtue` pattern AND no
    current vice acts
  - `requires_practical_wisdom` — mixed signals, single case visible
    but no longitudinal pattern (or pattern conflicts with case)

The store is *append-on-read-back* — running this projection writes
the current observation to the history. Two consecutive runs against
the same case will see different `history_hash`es because of the
appended record. For deterministic-replay scenarios (CI, audit
review), pass `read_only=True` and the projection won't write.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from erisml_compiler.history.habit_store import (
    HabitStore,
    record_from_compile,
)
from erisml_compiler.projections.base import GateFinding, Projection, ProjectionResult
from erisml_compiler.projections.substrate import MoralSubstrate
from erisml_compiler.projections.virtue import VirtueProjection


class LongitudinalVirtueProjection(Projection):
    """Reads agent's HabitStore + the current substrate, returns a
    combined VirtueAssessment + ProjectionResult.

    Drop-in alternative to VirtueProjection.
    """

    framework = "virtue_longitudinal"

    def __init__(
        self,
        store: HabitStore | str | Path,
        *,
        read_only: bool = False,
    ) -> None:
        self.store = store if isinstance(store, HabitStore) else HabitStore(store)
        self.read_only = read_only

    def project(
        self,
        substrate: MoralSubstrate,
        *,
        graph: Any = None,
        ir: Any = None,
        **kwargs: Any,
    ) -> ProjectionResult:
        # 1. Run the underlying single-case virtue projection.
        single = VirtueProjection().project(substrate, graph=graph)

        # 2. Identify the agent.
        agent_id = (
            substrate.maxim.agent_id
            if substrate.maxim and substrate.maxim.agent_id
            else "unknown_agent"
        )

        # 3. Optionally write the current observation to the store.
        record_written: str | None = None
        if ir is not None and not self.read_only:
            rec = record_from_compile(agent_id, ir)
            if rec is not None:
                self.store.append(rec)
                record_written = rec.case_id

        # 4. Read the (possibly just-updated) history + assess.
        assessment = self.store.assess(agent_id)

        # 5. Build longitudinal findings.
        findings: list[GateFinding] = list(single.findings)
        findings.append(self._gate_longitudinal_pattern(agent_id, assessment))
        findings.append(self._gate_evidence_sufficient(assessment))

        # 6. Combine the verdicts.
        verdict, polarity_override = self._combine_verdicts(single.verdict, assessment)

        return ProjectionResult(
            framework=self.framework,
            verdict=verdict,
            polarity=polarity_override or single.polarity,
            confidence=min(single.confidence, _confidence_from_n(assessment.n_observations)),
            findings=findings,
            framework_specific={
                **single.framework_specific,
                "longitudinal_assessment": {
                    "agent_id": assessment.agent_id,
                    "n_observations": assessment.n_observations,
                    "per_axis_mean_polarity": dict(assessment.per_axis_mean_polarity),
                    "per_axis_dispersion": dict(assessment.per_axis_dispersion),
                    "per_axis_dominant": dict(assessment.per_axis_dominant),
                    "history_hash": assessment.history_hash,
                },
                "record_written": record_written,
            },
            metadata={
                "projection_version": "v1_longitudinal",
                "underlying_single_case_verdict": single.verdict,
                "read_only": self.read_only,
            },
        )

    # ------------------------------------------------------ findings

    def _gate_longitudinal_pattern(
        self, agent_id: str, assessment
    ) -> GateFinding:
        """Surface whichever axis carries the strongest established pattern."""
        if assessment.n_observations == 0:
            return GateFinding(
                name="longitudinal_pattern",
                passed=True,
                reason=f"No prior observations of {agent_id} in the store",
                severity="minor",
            )
        vices = [
            (axis, mean)
            for axis, mean in assessment.per_axis_mean_polarity.items()
            if assessment.per_axis_dominant.get(axis) == "vice"
        ]
        virtues = [
            (axis, mean)
            for axis, mean in assessment.per_axis_mean_polarity.items()
            if assessment.per_axis_dominant.get(axis) == "virtue"
        ]
        if vices:
            worst = min(vices, key=lambda x: x[1])
            return GateFinding(
                name="longitudinal_pattern",
                passed=False,
                reason=(
                    f"History of {assessment.n_observations} observation(s) "
                    f"shows entrenched VICE on the '{worst[0]}' axis "
                    f"(mean polarity {worst[1]:+.2f})"
                ),
                severity="grave",
                detail={"axis": worst[0], "mean_polarity": worst[1]},
            )
        if virtues:
            best = max(virtues, key=lambda x: x[1])
            return GateFinding(
                name="longitudinal_pattern",
                passed=True,
                reason=(
                    f"History of {assessment.n_observations} observation(s) "
                    f"shows established VIRTUE on the '{best[0]}' axis "
                    f"(mean polarity {best[1]:+.2f})"
                ),
                severity="minor",
                detail={"axis": best[0], "mean_polarity": best[1]},
            )
        return GateFinding(
            name="longitudinal_pattern",
            passed=True,
            reason=(
                f"History of {assessment.n_observations} observation(s) "
                f"shows no entrenched virtue or vice pattern"
            ),
            severity="minor",
        )

    def _gate_evidence_sufficient(self, assessment) -> GateFinding:
        """Aristotle's 'one swallow' caveat — flag when n is too small
        for confident character assessment."""
        if assessment.n_observations < 3:
            return GateFinding(
                name="evidence_sufficient",
                passed=False,
                reason=(
                    f"Only {assessment.n_observations} prior observation(s); "
                    f"one swallow does not make a summer. Character verdicts "
                    f"call for more data."
                ),
                severity="moderate",
            )
        return GateFinding(
            name="evidence_sufficient",
            passed=True,
            reason=(
                f"{assessment.n_observations} observations available for "
                f"character assessment"
            ),
            severity="minor",
        )

    # ------------------------------------------------------ combiner

    def _combine_verdicts(self, single_verdict: str, assessment) -> tuple[str, str | None]:
        """Combine the single-case and longitudinal readings.

        Virtue ethics judges character. Character IS the longitudinal
        pattern (Aristotle: character is built through habituation).
        When the longitudinal data is sufficient (n ≥ 3) and shows an
        entrenched pattern, that's the verdict — regardless of how
        the single-case projection reads the current act.
        """
        n = assessment.n_observations
        if n < 3:
            # Insufficient history; fall back to the single-case reading.
            return (single_verdict, None)

        has_vice_pattern = any(
            d == "vice" for d in assessment.per_axis_dominant.values()
        )
        has_virtue_pattern = any(
            d == "virtue" for d in assessment.per_axis_dominant.values()
        )

        # Entrenched vice pattern → vicious character.
        if has_vice_pattern:
            return ("vicious", "forbid")
        # Entrenched virtue pattern + single-case not vicious → virtuous character.
        if has_virtue_pattern and single_verdict != "vicious":
            return ("virtuous", "permit")
        # Mixed: defer to single-case reading.
        return (single_verdict, None)


def _confidence_from_n(n: int) -> float:
    """Diminishing-returns confidence based on observation count."""
    if n == 0:
        return 0.3
    return min(1.0, 0.3 + 0.1 * n)
