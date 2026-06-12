"""Paragraph-based segmenter.

MVP: split on blank-line boundaries. Each non-empty paragraph becomes a
segment. Type-tagging is heuristic for now: segments containing strong
decision verbs ('shall', 'must', 'choose', 'must decide') are classified as
`decision_point`; quoted segments become `dialogue`; the rest are
`background`. Phase 2 will replace heuristic tagging with a learned
classifier.
"""
from __future__ import annotations

import re

from erisml_compiler.ir.schemas import Segment, SegmentType

DECISION_HINTS = re.compile(
    r"\b(shall|must|choose|decide|forced to|whether to|will not|cannot)\b",
    flags=re.IGNORECASE,
)
NORM_HINTS = re.compile(
    r"\b(prohibit|require|permit|forbidden|allowed|obligat|duty|owes?)\b",
    flags=re.IGNORECASE,
)
DIALOGUE_HINTS = re.compile(r'^[\'"]|[\'"]\s*$')


def _classify(text: str) -> SegmentType:
    s = text.strip()
    if DIALOGUE_HINTS.search(s):
        return "dialogue"
    if NORM_HINTS.search(s):
        return "norm_statement"
    if DECISION_HINTS.search(s):
        return "decision_point"
    return "background"


def segment_paragraphs(raw_text: str) -> list[Segment]:
    """Split on blank-line boundaries and emit Segment objects with spans."""
    segments: list[Segment] = []
    cursor = 0
    paragraphs = re.split(r"\n\s*\n", raw_text)
    for idx, para in enumerate(paragraphs, start=1):
        # Find this paragraph's character span in the original text.
        start = raw_text.find(para, cursor)
        if start < 0:
            start = cursor
        end = start + len(para)
        cursor = end
        text = para.strip()
        if not text:
            continue
        seg_id = f"seg_{idx:03d}"
        segments.append(
            Segment(
                segment_id=seg_id,
                type=_classify(text),
                text=text,
                span=(start, end),
            )
        )
    return segments
