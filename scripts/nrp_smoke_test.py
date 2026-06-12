"""NRP-LLM smoke test.

Reads ERISML_LLM_API_KEY (do not hardcode tokens). Issues one
chat-completion call to verify wiring, then runs LLMExtractor against the
Nazi-attic example, then compiles end-to-end with the LLM as the primary
extractor and the rule extractor as the critic.

Run:
    export ERISML_LLM_API_KEY=...    # set in your shell, not in code
    export ERISML_LLM_MODEL=qwen3    # optional, default qwen3
    python scripts/nrp_smoke_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    if not os.environ.get("ERISML_LLM_API_KEY"):
        print("[!] ERISML_LLM_API_KEY not set. Refusing to run.", file=sys.stderr)
        return 2

    print("[#] Step 1: trivial chat-completion call")
    from erisml_compiler.annotation.llm_extractor import NRPOpenAIAdapter
    adapter = NRPOpenAIAdapter()
    print(f"    base_url = {adapter.base_url}")
    print(f"    model    = {adapter.model}")
    response = adapter.call(
        system="You are a helpful assistant. Respond in one short sentence.",
        user="What is the value of 2+2? Respond ONLY with the integer answer.",
    )
    print(f"    response (truncated): {response[:200]!r}")

    print()
    print("[#] Step 2: LLMExtractor on examples/nazi_attic.txt")
    from erisml_compiler.annotation.llm_extractor import LLMExtractor
    from erisml_compiler.ingestion.text_loader import load_text_document
    from erisml_compiler.segmentation.segmenter import segment_paragraphs

    here = Path(__file__).resolve().parent.parent
    nazi_path = here / "examples" / "nazi_attic.txt"
    doc = load_text_document(nazi_path)
    segments = segment_paragraphs(doc.raw_text)
    extractor = LLMExtractor(adapter=adapter)
    result = extractor.extract(doc, segments)
    print(f"    stakeholders ({len(result.stakeholders)}): {[s.id for s in result.stakeholders]}")
    print(f"    commitments ({len(result.commitments)}): {[c.type for c in result.commitments]}")
    print(f"    ethical facts ({len(result.ethical_facts)}): {[f.kind for f in result.ethical_facts]}")
    print(f"    canonical form: {result.canonical_form}")

    print()
    print("[#] Step 3: full compile with LLM extractor + rule critic")
    from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
    from erisml_compiler.tiers import CompilerTier
    ir = compile_document(
        nazi_path,
        CompileOptions(
            tier=CompilerTier.LLM,
            extractor="llm",
            critic="rule",
            llm_adapter=adapter,
        ),
    )
    print(f"    canonical form: {ir.canonical_form}")
    print(f"    verdict: {ir.deme_verdict.verdict} (confidence {ir.deme_verdict.confidence:.2f})")
    if ir.audit:
        print(f"    IR hash: {ir.audit.ir_hash}")
    cr = ir.extra.get("extractor_metadata", {}).get("critic_report")
    if cr:
        print(f"    critic agreement overall: {cr.get('overall_agreement', 0):.2f}")
        print(f"    stakeholder overlap: {cr.get('stakeholder_overlap', 0):.2f}")
        print(f"    fact-kind overlap:   {cr.get('fact_kind_overlap', 0):.2f}")

    print()
    print("[+] NRP smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
