"""Audit: hash chain + provenance + artifact bundling."""

from erisml_compiler.audit.artifact import bundle_artifact
from erisml_compiler.audit.hash_chain import (
    compute_ir_hash,
    compute_text_hash,
    finalize_audit,
)
from erisml_compiler.audit.provenance import record_pass

__all__ = [
    "bundle_artifact",
    "compute_ir_hash",
    "compute_text_hash",
    "finalize_audit",
    "record_pass",
]
