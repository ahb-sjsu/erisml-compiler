"""Load a Tier-1 structured-input file (JSON) into a CompilerIR skeleton.

Tier 1 inputs are pre-parsed event streams. The expected JSON schema:

    {
        "doc_id": "...",
        "title": "...",
        "agents": [{"id": "speaker", "type": "individual", "roles": [...]}, ...],
        "events": [
            {"time_index": 0, "type": "vow_made", "actor": "speaker", "content": "...", ...},
            ...
        ],
        "commitments": [...],   # optional; otherwise inferred from events
        "ethical_facts": [...]  # optional; otherwise inferred from events
    }

This loader produces a partially-populated IR. The pipeline's tensorisation
and EM-DAG passes fill in the rest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from erisml_compiler.ir.schemas import (
    Commitment,
    CompilerIR,
    Document,
    EthicalFact,
    Event,
    Stakeholder,
)


def load_structured_input(path: str | Path) -> CompilerIR:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = json.loads(raw)
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    document = Document(
        doc_id=data.get("doc_id", p.stem),
        title=data.get("title", p.stem),
        source=str(p),
        domain=data.get("domain"),
        language=data.get("language", "en"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        sha256=sha,
        raw_text=json.dumps(data, indent=2),
    )
    stakeholders = [Stakeholder(**a) for a in data.get("agents", [])]
    events = [Event(**e) for e in data.get("events", [])]
    commitments = [Commitment(**c) for c in data.get("commitments", [])]
    ethical_facts = [EthicalFact(**f) for f in data.get("ethical_facts", [])]
    return CompilerIR(
        document=document,
        stakeholders=stakeholders,
        events=events,
        commitments=commitments,
        ethical_facts=ethical_facts,
    )
