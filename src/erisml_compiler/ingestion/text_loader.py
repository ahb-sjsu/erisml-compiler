"""Load a natural-language text document and attach metadata."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from erisml_compiler.ir.schemas import Document


def load_text_document(path: str | Path, doc_id: str | None = None) -> Document:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return Document(
        doc_id=doc_id or p.stem,
        title=p.stem,
        source=str(p),
        domain=None,
        language="en",
        timestamp=datetime.now(timezone.utc).isoformat(),
        sha256=sha,
        raw_text=raw,
    )
