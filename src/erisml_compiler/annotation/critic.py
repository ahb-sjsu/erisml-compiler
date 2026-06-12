"""Critic pass: cross-extractor consensus check.

Given two extractors (primary and critic), run both on the same document,
diff the results, and flag disagreements. The output of the primary
extractor is preserved as the main IR, with `requires_review=True` set on
fields the two extractors disagreed about.

Typical pairings:
    primary=llm,  critic=rule       # LLM extraction sanity-checked by rules
    primary=llm,  critic=llm(diff model)  # cross-model consensus
    primary=rule, critic=mock       # ensure mock fixture matches rule output
"""
from __future__ import annotations

from dataclasses import dataclass, field

from erisml_compiler.annotation.base import Extractor, ExtractorResult


@dataclass
class CriticReport:
    """Per-document agreement diagnostics."""

    primary_extractor: str
    critic_extractor: str
    stakeholder_id_overlap: float = 0.0
    commitment_id_overlap: float = 0.0
    fact_kind_overlap: float = 0.0
    canonical_form_agreement: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def overall_agreement(self) -> float:
        scores = [
            self.stakeholder_id_overlap,
            self.commitment_id_overlap,
            self.fact_kind_overlap,
            1.0 if self.canonical_form_agreement else 0.0,
        ]
        return sum(scores) / len(scores)


def _set_overlap(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def critic_pass(
    primary: ExtractorResult,
    critic: ExtractorResult,
    primary_name: str,
    critic_name: str,
    review_threshold: float = 0.5,
) -> tuple[ExtractorResult, CriticReport]:
    """Run a critic pass over the primary extractor's output.

    Strategy:
        1. Compute per-class agreement scores (stakeholder ids, commitment
           ids, fact kinds, canonical form).
        2. For each stakeholder in `primary` that has no counterpart in
           `critic` (by id), set `requires_review=True` on the entry.
        3. Build a CriticReport with the agreement scores plus narrative
           notes about specific disagreements.
        4. If overall_agreement is below `review_threshold`, flag a
           document-level review note.

    Returns the (possibly-modified) primary result and the CriticReport.
    """
    # ----- per-class agreement scores -----
    p_stake_ids = {s.id for s in primary.stakeholders}
    c_stake_ids = {s.id for s in critic.stakeholders}
    stake_overlap = _set_overlap(p_stake_ids, c_stake_ids)

    p_commit_ids = {c.id for c in primary.commitments}
    c_commit_ids = {c.id for c in critic.commitments}
    commit_overlap = _set_overlap(p_commit_ids, c_commit_ids)

    p_fact_kinds = {f.kind for f in primary.ethical_facts}
    c_fact_kinds = {f.kind for f in critic.ethical_facts}
    fact_overlap = _set_overlap(p_fact_kinds, c_fact_kinds)

    canonical_agreement = (
        primary.canonical_form is not None
        and primary.canonical_form == critic.canonical_form
    )

    notes: list[str] = []
    if not canonical_agreement and primary.canonical_form != critic.canonical_form:
        notes.append(
            f"Canonical-form mismatch: primary={primary.canonical_form!r}, "
            f"critic={critic.canonical_form!r}"
        )

    # ----- mark divergent stakeholders for review -----
    for s in primary.stakeholders:
        if s.id not in c_stake_ids:
            s.requires_review = True
            notes.append(f"Stakeholder {s.id!r} not corroborated by critic.")

    # ----- missing kinds in primary -----
    missing_in_primary = c_fact_kinds - p_fact_kinds
    if missing_in_primary:
        notes.append(
            f"Critic detected fact kinds primary missed: {sorted(missing_in_primary)}"
        )
    missing_in_critic = p_fact_kinds - c_fact_kinds
    if missing_in_critic:
        notes.append(
            f"Primary detected fact kinds critic missed: {sorted(missing_in_critic)}"
        )

    report = CriticReport(
        primary_extractor=primary_name,
        critic_extractor=critic_name,
        stakeholder_id_overlap=stake_overlap,
        commitment_id_overlap=commit_overlap,
        fact_kind_overlap=fact_overlap,
        canonical_form_agreement=canonical_agreement,
        notes=notes,
    )

    if report.overall_agreement < review_threshold:
        notes.insert(
            0,
            f"Document-level review recommended: overall agreement "
            f"{report.overall_agreement:.2f} below threshold {review_threshold:.2f}.",
        )

    # Store the report in extractor_metadata for downstream consumers.
    primary.extractor_metadata = dict(primary.extractor_metadata)
    primary.extractor_metadata["critic_report"] = {
        "primary": primary_name,
        "critic": critic_name,
        "stakeholder_overlap": stake_overlap,
        "commitment_overlap": commit_overlap,
        "fact_kind_overlap": fact_overlap,
        "canonical_form_agreement": canonical_agreement,
        "overall_agreement": report.overall_agreement,
        "notes": notes,
    }

    return primary, report


class CriticExtractor(Extractor):
    """A composing extractor: runs `primary` and `critic` extractors and
    fuses their outputs via the critic pass."""

    name = "critic"

    def __init__(
        self,
        primary: Extractor,
        critic: Extractor,
        review_threshold: float = 0.5,
    ):
        self.primary = primary
        self.critic = critic
        self.review_threshold = review_threshold
        self.name = f"critic({primary.name},{critic.name})"

    def extract(self, document, segments) -> ExtractorResult:
        primary_result = self.primary.extract(document, segments)
        try:
            critic_result = self.critic.extract(document, segments)
        except Exception as exc:
            # Critic failure leaves primary intact but logs the issue.
            primary_result.extractor_metadata = dict(primary_result.extractor_metadata)
            primary_result.extractor_metadata["critic_error"] = str(exc)
            return primary_result

        fused, _report = critic_pass(
            primary_result,
            critic_result,
            self.primary.name,
            self.critic.name,
            review_threshold=self.review_threshold,
        )
        return fused
