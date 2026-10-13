"""MoralStreamer: yield events as the pipeline progresses.

Each StreamEvent is a structured record describing what just changed in
moral state. The captioner renders the stream as live text; the HTML
report writer consumes the same stream to build the timeline section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from erisml_compiler.ir.schemas import CompilerIR


@dataclass
class StreamEvent:
    kind: str  # "pass_start" | "pass_end" | "event_processed" | "fact_detected" | "timeline_entry" | "verdict"
    time_index: int = 0
    label: str = ""
    detail: dict = field(default_factory=dict)


class MoralStreamer:
    """Replay a fully-compiled IR as a stream of StreamEvents.

    Phase 1 emits events post-hoc (after the pipeline completes); Phase 2+
    will emit them incrementally during pipeline execution.
    """

    def __init__(self, ir: CompilerIR):
        self.ir = ir

    def __iter__(self) -> Iterator[StreamEvent]:
        return self.stream()

    def stream(self) -> Iterator[StreamEvent]:
        # 1) Document loaded.
        yield StreamEvent(
            kind="pass_start",
            label=f"Document loaded: {self.ir.document.doc_id}",
            detail={
                "title": self.ir.document.title,
                "language": self.ir.document.language,
                "sha256": self.ir.document.sha256,
            },
        )

        # 2) Each event in temporal order.
        events_sorted = sorted(self.ir.events, key=lambda e: e.time_index)
        for evt in events_sorted:
            yield StreamEvent(
                kind="event_processed",
                time_index=evt.time_index,
                label=evt.type,
                detail={
                    "actor": evt.actor,
                    "target": evt.target,
                    "content": evt.content,
                    "id": evt.id,
                },
            )

        # 3) Ethical facts detected.
        for fact in self.ir.ethical_facts:
            yield StreamEvent(
                kind="fact_detected",
                label=f"{fact.kind}: {fact.description}",
                detail={
                    "id": fact.id,
                    "severity": fact.severity,
                    "subjects": fact.subjects,
                    "confidence": fact.confidence,
                },
            )

        # 4) Timeline (DEME's moral vector after each event, erisml-lib's nine dimensions).
        for entry in self.ir.timeline:
            yield StreamEvent(
                kind="timeline_entry",
                time_index=entry.time_index,
                label=entry.event_label,
                detail=entry.vector.to_dict(),
            )

        # 6) Verdict.
        if self.ir.deme_verdict:
            verdict = self.ir.deme_verdict
            yield StreamEvent(
                kind="verdict",
                label=f"DEME verdict: {verdict.verdict}",
                detail={
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "escalation_required": verdict.escalation_required,
                    "moral_residue": verdict.moral_residue,
                },
            )
