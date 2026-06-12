"""The 12-pass pipeline orchestrator (per spec section 12).

Pass map:
   0. Ingestion              -> ingestion.text_loader / structured_loader
   1. Segmentation           -> segmentation.segmenter (text tiers only)
   2. Entity extraction      -> extractor.extract (Tier 2/3) or pass-through (Tier 1)
   3. Stakeholder class.     -> same
   4. Event extraction       -> same
   5. Norm/commitment ext.   -> same
   6. Ethical-fact extract.  -> same
   7. Canonicalization       -> extractor sets ir.canonical_form
   8. Tensorization          -> evaluation.tensor_builder.build_timeline (uses EM-DAG)
   9. ErisML IR generation   -> codegen.render_erisml (on demand)
  10. DEME evaluation         -> erisml_backend.deme_bridge.DEMEBridge.evaluate
  11. Contraction & residue   -> embedded in DEMEVerdict
  12. Audit artifact          -> audit.hash_chain.finalize_audit + audit.artifact.bundle
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from erisml_compiler.annotation.base import Extractor, ExtractorResult
from erisml_compiler.annotation.critic import CriticExtractor
from erisml_compiler.annotation.mock_extractor import MockExtractor
from erisml_compiler.annotation.rule_extractor import RuleExtractor
from erisml_compiler.audit.hash_chain import finalize_audit
from erisml_compiler.audit.provenance import record_pass
from erisml_compiler.canonicalizer.base import Canonicalizer, auto_canonicalizer
from erisml_compiler.em_dag import EMDAG, load_profile
from erisml_compiler.erisml_backend.deme_bridge import DEMEBridge
from erisml_compiler.evaluation.conflict_detector import detect_conflicts
from erisml_compiler.evaluation.moral_vector import build_moral_vector_from_em_outputs
from erisml_compiler.evaluation.tensor_builder import build_timeline
from erisml_compiler.evaluation.tensor_builder_v3 import build_moral_tensor_v3
from erisml_compiler.ingestion.structured_loader import load_structured_input
from erisml_compiler.ingestion.text_loader import load_text_document
from erisml_compiler.ir.schemas import CompilerIR, PassRecord
from erisml_compiler.segmentation.segmenter import segment_paragraphs
from erisml_compiler.tiers import CompilerTier

import logging as _logging

_v3_log = _logging.getLogger(__name__)


def _produce_v3_tensor(ir, em_outputs, dag, options):
    """V3 tensor dispatch:
    - rank 1-2 → Phase 3-4 bridge (per-party V3 facts → DEME V3 modules)
    - rank 3-6 → Phase 5 higher-rank builder (stacks rank-2 slices over
      time / action / coalition / uncertainty axes)

    Falls back to the Phase 2 V2-migration tensor builder on ImportError
    (erisml-lib not installed) or any module exception.
    """
    rank = options.tensor_rank
    try:
        if rank <= 2:
            from erisml_compiler.erisml_backend.v3_bridge import compile_to_v3_tensor

            return compile_to_v3_tensor(ir, rank=rank)
        else:
            from erisml_compiler.erisml_backend.v3_higher_rank import (
                HigherRankConfig,
                build_moral_tensor_v3_rank3plus,
            )

            cfg = HigherRankConfig(
                n_actions=options.tensor_n_actions,
                n_coalitions=options.tensor_n_coalitions,
                n_samples=options.tensor_n_samples,
                sample_noise_std=options.tensor_sample_noise_std,
                sample_seed=options.tensor_sample_seed,
            )
            return build_moral_tensor_v3_rank3plus(ir, rank=rank, config=cfg)
    except ImportError:
        _v3_log.info("erisml-lib not installed; using Phase 2 fanout path")
        return build_moral_tensor_v3(ir, em_outputs, dag, rank=min(rank, 2))
    except Exception as e:  # noqa: BLE001
        _v3_log.warning(
            "V3 bridge raised %s; falling back to Phase 2 fanout: %s",
            type(e).__name__,
            e,
        )
        return build_moral_tensor_v3(ir, em_outputs, dag, rank=min(rank, 2))


@dataclass
class CompileOptions:
    tier: CompilerTier
    extractor: str = "rule"  # "mock" | "rule" | "probe" | "llm"
    critic: str | None = None  # same set or None
    em_profile: str | Path | None = None  # path to YAML; default if None
    canonicalizer: Canonicalizer | None = None  # default: auto-select
    llm_adapter: object | None = None  # for tier="llm" or critic="llm"
    probe_config: object | None = None  # ProbeExtractorConfig for tier="probe"
    fail_unknown_mock: bool = True
    tensor_rank: int = 2  # DEME V3 rank for ir.moral_tensor_v3 (1-6)
    # Higher-rank axes (rank >= 3 only). a/c are stub axes today; s is
    # real Monte Carlo over fact.confidence.
    tensor_n_actions: int = 1
    tensor_n_coalitions: int = 1
    tensor_n_samples: int = 1
    tensor_sample_noise_std: float = 0.05
    tensor_sample_seed: int = 0


def _resolve_em_profile(em_profile: str | Path | None) -> EMDAG:
    if em_profile is None:
        # Use the bundled default profile.
        here = Path(__file__).parent.parent / "em_dag" / "profiles" / "default.yaml"
        return load_profile(here)
    return load_profile(em_profile)


def _resolve_extractor(
    name: str, llm_adapter: object | None = None, probe_config=None
) -> Extractor:
    if name == "mock":
        return MockExtractor()
    if name == "rule":
        return RuleExtractor()
    if name == "probe":
        from erisml_compiler.annotation.probe_extractor import (
            ProbeExtractor,
            ProbeExtractorConfig,
        )

        return ProbeExtractor(config=probe_config or ProbeExtractorConfig())
    if name == "llm":
        from erisml_compiler.annotation.llm_extractor import (
            LLMExtractor,
            NRPOpenAIAdapter,
        )

        adapter = llm_adapter or NRPOpenAIAdapter()
        return LLMExtractor(adapter=adapter)
    raise ValueError(f"Unknown extractor: {name!r} (expected mock/rule/probe/llm)")


def _load_known_forms() -> dict[str, str]:
    """Load the canonical-forms registry from the bundled ontology YAML."""
    path = Path(__file__).parent.parent / "ontology" / "canonical_forms.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    forms = data.get("canonical_forms", {}) or {}
    return {tag: entry.get("description", "") for tag, entry in forms.items()}


def _build_situation_summary(ir: CompilerIR) -> str:
    """Build a one-paragraph summary of the situation, used for
    canonicalizer input. Combines title, the first segment's text (if
    any), the canonical-form-relevant subset of ethical facts, and the
    commitment content."""
    parts: list[str] = []
    if ir.document.title:
        parts.append(ir.document.title)
    parts.append(ir.document.raw_text[:400])
    for fact in ir.ethical_facts[:8]:
        parts.append(fact.description)
    for c in ir.commitments[:4]:
        parts.append(f"{c.type} by {c.holder}: {c.content}")
    return " | ".join(p for p in parts if p)


def compile_document(
    input_path: str | Path,
    options: CompileOptions,
) -> CompilerIR:
    """Run the full 12-pass pipeline. Returns a fully-populated IR."""
    input_path = Path(input_path)
    passes: list[PassRecord] = []
    tier_value = options.tier.value
    dag = _resolve_em_profile(options.em_profile)

    # Pass 0: Ingestion (and pass-through extraction for Tier 1).
    if options.tier == CompilerTier.GEOMETRIC:
        with record_pass(passes, 0, "ingestion_structured", tier_value):
            ir = load_structured_input(input_path)
        extractor_name = "structured_pass_through"
    else:
        with record_pass(passes, 0, "ingestion_text", tier_value):
            document = load_text_document(input_path)

        # Pass 1: Segmentation.
        with record_pass(passes, 1, "segmentation", tier_value):
            segments = segment_paragraphs(document.raw_text)

        # Passes 2-7: Extraction (entity, stakeholder, event, norm, ethical-fact, canonicalization).
        primary_extractor = _resolve_extractor(
            options.extractor, options.llm_adapter, options.probe_config
        )
        extractor_name = primary_extractor.name

        if options.critic:
            critic_extractor = _resolve_extractor(
                options.critic, options.llm_adapter, options.probe_config
            )
            composed = CriticExtractor(primary=primary_extractor, critic=critic_extractor)
            extractor_name = composed.name
            with record_pass(passes, 2, "extraction_with_critic", tier_value):
                result: ExtractorResult = composed.extract(document, segments)
        else:
            with record_pass(passes, 2, "extraction_passes_2_through_7", tier_value):
                result = primary_extractor.extract(document, segments)

        ir = CompilerIR(
            document=document,
            segments=segments,
            stakeholders=result.stakeholders,
            relations=result.relations,
            events=result.events,
            commitments=result.commitments,
            norms=result.norms,
            ethical_facts=result.ethical_facts,
            conflicts=result.conflicts,
            canonical_form=result.canonical_form,
            extra={"extractor_metadata": result.extractor_metadata},
        )

        # Pass 7 (canonicalization): augment / replace the extractor's
        # canonical_form using the configured Canonicalizer. The
        # extractor's tag is preserved if the canonicalizer cannot match.
        with record_pass(passes, 7, "canonicalization", tier_value):
            canonicalizer = options.canonicalizer or auto_canonicalizer()
            known_forms = _load_known_forms()
            summary = _build_situation_summary(ir)
            canon_result = canonicalizer.canonicalize(summary, known_forms)
            if canon_result.tag is not None:
                ir.canonical_form = canon_result.tag
            ir.extra["canonicalization"] = {
                "backend": canon_result.backend,
                "tag": canon_result.tag,
                "confidence": canon_result.confidence,
                "matched_known_form": canon_result.matched_known_form,
                "evidence": canon_result.evidence,
            }

    # Pass 8: Tensorisation (build MoralVector timeline by walking the EM-DAG).
    with record_pass(passes, 8, "tensorisation", tier_value):
        timeline = build_timeline(ir, dag)
        ir.timeline = timeline
        # Final-state EM outputs (re-evaluated on the full IR).
        em_outputs = dag.evaluate(ir)
        ir.em_outputs = em_outputs
        final_vector = build_moral_vector_from_em_outputs(em_outputs, dag)
        ir.moral_vectors = [final_vector]
        # DEME V3 alignment (Phase 2): produce the rank-N tensor
        # alongside the V2 moral_vectors. Phase 4 will make V3 the only
        # producer; for now both ship.
        ir.moral_tensor_v3 = _produce_v3_tensor(ir, em_outputs, dag, options)

    # Pass 9-10: ErisML IR is the in-memory `ir` itself; DEME evaluation.
    with record_pass(passes, 10, "deme_evaluation", tier_value):
        bridge = DEMEBridge(profile_name=dag.name)
        verdict = bridge.evaluate(ir, final_vector)
        ir.deme_verdict = verdict

    # Pass 11: Detect conflicts at the vector level (augments extractor's).
    with record_pass(passes, 11, "conflict_detection", tier_value):
        ir.conflicts = detect_conflicts(ir, final_vector)

    # Pass 12: Audit.
    with record_pass(passes, 12, "audit_finalisation", tier_value):
        audit = finalize_audit(
            ir,
            tier=tier_value,
            extractor=extractor_name,
            em_profile=dag.name,
            passes=passes.copy(),
        )
        ir.audit = audit

    return ir
