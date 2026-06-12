"""Tests for the --strict-v3 / strict_v3 flag (item 4 of release-planning-01).

When `strict_v3=True`, the orchestrator must NOT silently fall back to the
Phase 2 V2-migration builder. Bridge ImportErrors or exceptions must
re-raise as `StrictV3Error` so a research / production run fails loudly
instead of producing a downgraded result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from erisml_compiler.pipeline.orchestrator import (
    CompileOptions,
    StrictV3Error,
    compile_document,
)
from erisml_compiler.tiers import CompilerTier

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_strict_v3_default_false_in_compile_options():
    """The non-strict default must remain the silent-fallback behaviour
    we relied on through Phases 1-6."""
    opts = CompileOptions(tier=CompilerTier.RULES, extractor="mock")
    assert opts.strict_v3 is False


def test_strict_v3_raises_on_bridge_import_error(monkeypatch):
    """When erisml-lib (or any bridge import) raises ImportError and
    strict_v3 is True, compile_document must surface StrictV3Error."""
    from erisml_compiler.pipeline import orchestrator as orch

    def fake_compile_to_v3_tensor(*args, **kwargs):
        raise ImportError("simulated missing erisml-lib")

    # Inject the import error inside the bridge module so the
    # orchestrator's lazy import sees it.
    import erisml_compiler.erisml_backend.v3_bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "compile_to_v3_tensor", fake_compile_to_v3_tensor)

    with pytest.raises(StrictV3Error, match="strict_v3=True"):
        compile_document(
            EXAMPLES_DIR / "nazi_attic.txt",
            CompileOptions(
                tier=CompilerTier.RULES,
                extractor="mock",
                canonicalizer=None,
                tensor_rank=2,
                strict_v3=True,
            ),
        )


def test_strict_v3_raises_on_generic_bridge_exception(monkeypatch):
    """A non-ImportError exception from the V3 bridge must also raise
    StrictV3Error in strict mode (not just ImportError)."""
    import erisml_compiler.erisml_backend.v3_bridge as bridge_mod

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic bridge failure")

    monkeypatch.setattr(bridge_mod, "compile_to_v3_tensor", boom)

    with pytest.raises(StrictV3Error, match="V3 bridge raised RuntimeError"):
        compile_document(
            EXAMPLES_DIR / "nazi_attic.txt",
            CompileOptions(
                tier=CompilerTier.RULES,
                extractor="mock",
                canonicalizer=None,
                tensor_rank=2,
                strict_v3=True,
            ),
        )


def test_non_strict_still_falls_back_to_phase2(monkeypatch):
    """Default behaviour: bridge failure falls back to the Phase 2
    fanout builder and the IR still has a moral_tensor_v3."""
    import erisml_compiler.erisml_backend.v3_bridge as bridge_mod

    def fake_import_error(*args, **kwargs):
        raise ImportError("simulated")

    monkeypatch.setattr(bridge_mod, "compile_to_v3_tensor", fake_import_error)

    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="mock",
            canonicalizer=None,
            tensor_rank=2,
            strict_v3=False,  # explicit default
        ),
    )
    assert ir.moral_tensor_v3 is not None
    assert ir.moral_tensor_v3.metadata.get("build_strategy") == "phase2_fanout_from_rank1"
