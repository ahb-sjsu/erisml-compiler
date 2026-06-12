"""Hash chain over the compiled IR.

Every compile produces a SHA-256 hash of the canonical-JSON IR (excluding
the audit record itself, which contains this very hash). The hash chain is
deterministic given the same input + same dependency versions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from erisml_compiler import __version__
from erisml_compiler.ir.schemas import AuditRecord, CompilerIR


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_ir_hash(ir: CompilerIR) -> str:
    """Compute SHA-256 of the IR with `audit` and per-run metadata excluded.

    Per-run metadata: `document.timestamp`, `document.source` (file path).
    These are run-environment details, not substantive content. The
    document identity is preserved via `document.sha256` (text hash) which
    IS included in the IR hash.

    Canonicalised via Pydantic's `model_dump(mode='json')` then
    JSON-serialised with sorted keys for cross-platform determinism.
    """
    data = ir.model_dump(
        mode="json",
        exclude={
            "audit": True,
            "document": {"timestamp": True, "source": True},
        },
    )
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finalize_audit(
    ir: CompilerIR,
    *,
    tier: str,
    extractor: str,
    em_profile: str,
    passes: list,
    model_version: str | None = None,
) -> AuditRecord:
    """Build the audit record. Call after the IR is otherwise complete; the
    returned record is then assigned to `ir.audit`."""
    return AuditRecord(
        ir_hash=compute_ir_hash(ir),
        source_text_hash=compute_text_hash(ir.document.raw_text),
        compiler_version=__version__,
        tier=tier,
        extractor=extractor,
        model_version=model_version,
        em_profile=em_profile,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        passes=passes,
    )
