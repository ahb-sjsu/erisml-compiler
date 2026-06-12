"""Render IR as an ErisML-like textual source.

This is a minimal, human-readable rendering of the compiled IR. The format
is YAML-like to keep the MVP simple; Phase 2 will replace this with the
formal ErisML grammar.
"""
from __future__ import annotations

import yaml

from erisml_compiler.ir.schemas import CompilerIR


def render_erisml(ir: CompilerIR) -> str:
    """Render the IR as YAML-style ErisML source text."""
    data = ir.model_dump(mode="json", exclude_none=True)
    # Move large/derived fields to the end for readability.
    ordered_keys = [
        "schema_version", "document", "canonical_form",
        "stakeholders", "commitments", "events", "ethical_facts",
        "conflicts", "norms", "moral_vectors", "timeline",
        "em_outputs", "deme_verdict", "audit",
    ]
    ordered = {k: data[k] for k in ordered_keys if k in data}
    extra = {k: v for k, v in data.items() if k not in ordered}
    ordered.update(extra)
    return yaml.dump(ordered, sort_keys=False, default_flow_style=False, width=100)
