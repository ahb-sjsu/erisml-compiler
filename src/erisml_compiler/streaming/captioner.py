"""Terminal captioner: render a stream of StreamEvents as live text."""

from __future__ import annotations

import sys
from typing import IO, Iterable

from erisml_compiler.streaming.streamer import StreamEvent

GLYPHS = {
    "pass_start": "[#]",
    "pass_end": "[/]",
    "event_processed": "[t]",
    "fact_detected": " > ",
    "em_output": " * ",
    "timeline_entry": " ~ ",
    "verdict": "[*]",
}


class TerminalCaptioner:
    """Renders StreamEvents to a text stream (default: stdout)."""

    def __init__(self, stream: IO[str] | None = None):
        self.stream = stream or sys.stdout

    def render(self, events: Iterable[StreamEvent]) -> None:
        for event in events:
            self._render_one(event)

    def _render_one(self, event: StreamEvent) -> None:
        glyph = GLYPHS.get(event.kind, "   ")
        prefix = f"{glyph}"
        if event.kind == "event_processed":
            prefix = f"[t={event.time_index}] "
            line = f"{prefix}{event.label}"
            d = event.detail
            if d.get("actor"):
                line += f"  actor={d['actor']}"
            if d.get("target"):
                line += f"  target={d['target']}"
            if d.get("content"):
                line += f"  content={d['content']!r}"
        elif event.kind == "fact_detected":
            line = f" {glyph} fact {event.detail.get('id', '')}: {event.label}"
        elif event.kind == "em_output":
            line = (
                f" {glyph} EM[{event.detail.get('module','?')}] "
                f"value={event.detail.get('value', 0):+.3f} "
                f"conf={event.detail.get('confidence', 0):.2f}"
            )
        elif event.kind == "timeline_entry":
            d = event.detail
            line = (
                f" {glyph} t={event.time_index} {event.label}  "
                f"harm={d.get('physical_harm', 0):+.2f} "
                f"ext={d.get('third_party_externality', 0):+.2f} "
                f"fid={d.get('vow_fidelity', 0):+.2f} "
                f"care={d.get('care_protection', 0):+.2f} "
                f"legit={d.get('legitimacy_trust', 0):+.2f}"
            )
        elif event.kind == "verdict":
            d = event.detail
            line = (
                f"\n{glyph} DEME VERDICT: {event.label}\n"
                f"    confidence: {d.get('confidence', 0):.2f}\n"
                f"    rationale:  {d.get('rationale','')}\n"
                f"    escalation: {d.get('escalation_required', False)}\n"
            )
            if d.get("moral_residue"):
                line += f"    residue:    {d['moral_residue']}\n"
        else:
            line = f"{prefix} {event.label}"
        print(line, file=self.stream)
