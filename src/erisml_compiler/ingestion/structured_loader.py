"""Load a Tier-1 structured-input file (JSON, or ErisML source) into a CompilerIR skeleton.

ErisML source (FR-19) is the YAML form `erisml_backend.codegen.render_erisml` emits, with the
IR's own keys (`stakeholders`, `relations`, `events`, `commitments`, `norms`, `ethical_facts`,
`extra`); files ending in .erisml, .yaml or .yml are read as YAML.

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

import yaml

from erisml_compiler.ir.schemas import (
    Commitment,
    CompilerIR,
    Document,
    EthicalFact,
    Event,
    Norm,
    Relation,
    Stakeholder,
)

ERISML_SOURCE_SUFFIXES = {".erisml", ".yaml", ".yml"}


def load_structured_input(path: str | Path) -> CompilerIR:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) if p.suffix.lower() in ERISML_SOURCE_SUFFIXES else json.loads(raw)
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    doc = data.get("document") or {}
    document = Document(
        doc_id=doc.get("doc_id", data.get("doc_id", p.stem)),
        title=doc.get("title", data.get("title", p.stem)),
        source=str(p),
        domain=doc.get("domain", data.get("domain")),
        language=doc.get("language", data.get("language", "en")),
        timestamp=datetime.now(timezone.utc).isoformat(),
        sha256=sha,
        raw_text=doc.get("raw_text") or json.dumps(data, indent=2, default=str),
    )
    # `agents` is the original Tier-1 key; ErisML source uses the IR's `stakeholders`
    stakeholders = [Stakeholder(**a) for a in data.get("stakeholders", data.get("agents", []))]
    return CompilerIR(
        document=document,
        stakeholders=stakeholders,
        relations=[Relation(**r) for r in data.get("relations", [])],
        events=[Event(**e) for e in data.get("events", [])],
        commitments=[Commitment(**c) for c in data.get("commitments", [])],
        norms=[Norm(**n) for n in data.get("norms", [])],
        ethical_facts=[EthicalFact(**f) for f in data.get("ethical_facts", [])],
        extra=data.get("extra", {}),
    )
