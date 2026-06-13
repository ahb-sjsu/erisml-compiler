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

import hashlib
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
from erisml_compiler.em_dag.dag import load_ethos_profile
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


class StrictV3Error(RuntimeError):
    """Raised when --strict-v3 is set and the V3 dispatch path fails.

    The fallback to the Phase 2 V2-migration builder is suppressed in
    strict mode so a silent downgrade cannot invalidate research /
    production results.
    """


def _produce_v3_tensor(ir, em_outputs, dag, options, *, ethos_weights=None):
    """V3 tensor dispatch:
    - rank 1-2 → Phase 3-4 bridge (per-party V3 facts → DEME V3 modules)
    - rank 3-6 → Phase 5 higher-rank builder (stacks rank-2 slices over
      time / action / coalition / uncertainty axes)

    Falls back to the Phase 2 V2-migration tensor builder on ImportError
    (erisml-lib not installed) or any module exception — UNLESS
    `options.strict_v3` is True, in which case the failure re-raises as
    `StrictV3Error` (or the original ImportError) so callers can fail
    loudly instead of silently downgrading.
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
                coalition_mode=options.coalition_mode,
            )
            return build_moral_tensor_v3_rank3plus(ir, rank=rank, config=cfg)
    except ImportError as e:
        if options.strict_v3:
            raise StrictV3Error(
                f"strict_v3=True but erisml-lib (V3 modules) is not installed. "
                f"Install with `pip install 'erisml-compiler[deme-v3]'` or set "
                f"strict_v3=False to allow the Phase 2 fallback. Underlying: {e}"
            ) from e
        _v3_log.info("erisml-lib not installed; using Phase 2 fanout path")
        return build_moral_tensor_v3(
            ir, em_outputs, dag, rank=min(rank, 2), ethos_weights=ethos_weights
        )
    except Exception as e:  # noqa: BLE001
        if options.strict_v3:
            raise StrictV3Error(
                f"strict_v3=True but V3 bridge raised {type(e).__name__}: {e}. "
                f"Set strict_v3=False to allow the Phase 2 fallback."
            ) from e
        _v3_log.warning(
            "V3 bridge raised %s; falling back to Phase 2 fanout: %s",
            type(e).__name__,
            e,
        )
        return build_moral_tensor_v3(
            ir, em_outputs, dag, rank=min(rank, 2), ethos_weights=ethos_weights
        )


@dataclass
class CompileOptions:
    tier: CompilerTier
    extractor: str = "rule"  # "mock" | "rule" | "probe" | "llm"
    critic: str | None = None  # same set or None
    em_profile: str | Path | None = None  # path to YAML; default if None
    projections: tuple[str, ...] = (
        "consequentialist_distributive",
        "deontic_kantian",
        "virtue_aristotelian",
        "care_ethics_relational",
    )
    """Which framework projections to run in Pass 8. The default runs
    both — the consequentialist (which populates the legacy moral_*
    fields for backward compat) and the deontic-Kantian
    (universalizability + mere-means + consent + legitimacy gates).
    Set to ('consequentialist_distributive',) to suppress deontic
    analysis. See `docs/plans/release-planning-06-framework-pluralist-architecture.md`."""
    ethos_profile: str | Path | None = None
    """Path to a fitted ethos YAML (e.g.
    `src/erisml_compiler/em_dag/profiles/dear_abby_socialchem_v0.1.yaml`).
    When set, per-module weights are applied at MoralVector projection
    time. Audit record carries the profile name + YAML sha256. Defaults
    to None — no ethos, all modules at equal weight."""
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
    # Phase 6 knobs.
    emit_strategic_analysis: bool = True
    emit_decision_proof: bool = True
    coalition_mode: str = "all_subsets"  # for strategic Shapley
    # When True, the V3 dispatch path (rank 1-6 bridge + higher-rank
    # builder) must succeed. Any ImportError (erisml-lib missing) or
    # exception in the bridge re-raises instead of silently falling back
    # to the Phase 2 V2-migration builder. For research / production
    # workloads where a silent downgrade would invalidate results.
    strict_v3: bool = False


def _resolve_em_profile(em_profile: str | Path | None) -> EMDAG:
    if em_profile is None:
        # Use the bundled default profile.
        here = Path(__file__).parent.parent / "em_dag" / "profiles" / "default.yaml"
        return load_profile(here)
    return load_profile(em_profile)


def _projection_result_to_dict(res) -> dict[str, Any]:
    """Convert a ProjectionResult into a JSON-serialisable dict.

    The `framework_specific` block may carry non-serialisable objects
    (MoralTensorV3, EMOutput, etc.) which already live on the top-level
    IR; we drop them here to keep ir.projections small + JSON-clean.
    """
    fs = res.framework_specific
    fs_clean: dict[str, Any] = {}
    for k, v in fs.items():
        if k in ("moral_tensor_v3", "moral_vector", "timeline", "em_outputs", "deme_verdict"):
            continue
        fs_clean[k] = v
    return {
        "framework": res.framework,
        "verdict": res.verdict,
        "polarity": res.polarity,
        "confidence": res.confidence,
        "findings": [f.model_dump() for f in res.findings],
        "framework_specific": fs_clean,
        "metadata": res.metadata,
    }


def _resolve_ethos_profile(
    ethos_profile: str | Path | None,
) -> tuple[dict[str, float] | None, str | None, str | None]:
    """Returns (weights_dict, profile_name, profile_sha256)."""
    if ethos_profile is None:
        return None, None, None
    p = Path(ethos_profile)
    data = load_ethos_profile(p)
    name = str(data.get("name") or p.stem)
    weights_raw = data.get("weights") or {}
    weights = {str(k): float(v) for k, v in weights_raw.items()}
    sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
    return (weights or None), name, sha256


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
    ethos_weights, ethos_name, ethos_sha256 = _resolve_ethos_profile(options.ethos_profile)
    _ir_graph_hash: str | None = None

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
            graph=getattr(result, "graph", None),
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

    # Pass 7.5: Graph identity. If the extractor emitted a graph
    # directly (graph-native extractor), use it. Otherwise lift the
    # flat extractor output into a typed MoralGraph. The graph is the
    # *primary* descriptive representation; the flat fields stay
    # populated for backward compat. See release-planning-06.
    with record_pass(passes, 75, "graph_identity", tier_value):
        from erisml_compiler.ir.graph import graph_from_flat, graph_hash as _graph_hash

        if ir.graph is None:
            ir.graph = graph_from_flat(ir)
        # else: extractor was graph-native; trust its emission.
        _ir_graph_hash = _graph_hash(ir.graph)

    # Pass 8: Projection (two-layer IR). For each enabled projection,
    # build framework-bound output from the descriptive substrate.
    # The consequentialist projection's output back-fills the legacy
    # ir.moral_tensor_v3 / moral_vectors / timeline / em_outputs fields
    # for backward compat with downstream consumers.
    with record_pass(passes, 8, "projection_pass", tier_value):
        from erisml_compiler.projections import (
            ConsequentialistProjection,
            DeonticProjection,
            substrate_from_ir,
        )

        substrate = substrate_from_ir(ir)
        projections_run: dict[str, Any] = {}

        if "consequentialist_distributive" in options.projections:
            cp = ConsequentialistProjection(
                dag=dag, ethos_weights=ethos_weights, tensor_rank=options.tensor_rank,
            )
            # Pre-compute the tensor via the orchestrator's V3 dispatcher
            # (honours strict_v3 + bridge vs Phase 2 fallback). The
            # projection accepts a tensor override so it doesn't always
            # use the fallback path.
            _em_for_tensor = dag.evaluate(ir)
            _tensor_for_projection = _produce_v3_tensor(
                ir, _em_for_tensor, dag, options, ethos_weights=ethos_weights,
            )
            cres = cp.project(substrate, ir=ir, tensor_override=_tensor_for_projection)
            projections_run[cp.framework] = cres

            # Back-fill legacy fields for backward compat.
            fs = cres.framework_specific
            ir.timeline = fs.get("timeline", [])
            ir.em_outputs = fs.get("em_outputs", {})
            ir.moral_vectors = [fs["moral_vector"]] if "moral_vector" in fs else []
            ir.deme_verdict = fs.get("deme_verdict")
            ir.moral_tensor_v3 = fs.get("moral_tensor_v3")
            ir.per_party_verdicts = fs.get("per_party_verdicts") or None
            ir.fairness_metrics = fs.get("fairness_metrics") or None

        if "deontic_kantian" in options.projections:
            dp = DeonticProjection()
            dres = dp.project(substrate, graph=ir.graph)
            projections_run[dp.framework] = dres

        if "virtue_aristotelian" in options.projections:
            from erisml_compiler.projections import VirtueProjection

            vp = VirtueProjection()
            vres = vp.project(substrate, graph=ir.graph)
            projections_run[vp.framework] = vres

        if "care_ethics_relational" in options.projections:
            from erisml_compiler.projections import CareEthicsProjection

            cep = CareEthicsProjection()
            ceres = cep.project(substrate, graph=ir.graph)
            projections_run[cep.framework] = ceres

        # Store projection results as JSON-serialisable dicts; the rich
        # in-memory MoralTensorV3 etc. stay accessible via ir.moral_tensor_v3.
        ir.projections = {
            framework: _projection_result_to_dict(res)
            for framework, res in projections_run.items()
        }

        # Surface verdict disagreement explicitly — no silent aggregation.
        # Compare normalised *polarity* across frameworks, not the
        # framework-native verdict strings (those differ even when the
        # frameworks agree — "permitted" / "permissible" / "virtuous" /
        # "caring" all mean "the act is OK in this framework's terms").
        verdicts = {fw: res.verdict for fw, res in projections_run.items()}
        polarities = {fw: res.polarity for fw, res in projections_run.items()}
        if len({p for p in polarities.values() if p != "neutral"}) > 1:
            ir.cross_projection_disagreement = {
                "verdicts": verdicts,
                "polarities": polarities,
                "note": (
                    "Frameworks disagree on this case. The compiler "
                    "surfaces all verdicts without aggregating; "
                    "choosing between them is itself a metaethical "
                    "decision the compiler defers to the caller."
                ),
            }

    # Pass 9-10: DEME evaluation now happens inside
    # ConsequentialistProjection during Pass 8; ir.deme_verdict is
    # already populated. We keep the pass-record for audit-chain
    # continuity but the work is a no-op here.
    with record_pass(passes, 10, "deme_evaluation_via_projection", tier_value):
        pass

    # Pass 11: Detect conflicts at the vector level (augments extractor's).
    with record_pass(passes, 11, "conflict_detection", tier_value):
        if ir.moral_vectors:
            ir.conflicts = detect_conflicts(ir, ir.moral_vectors[0])

    # Pass 12: Audit.
    with record_pass(passes, 12, "audit_finalisation", tier_value):
        audit = finalize_audit(
            ir,
            tier=tier_value,
            extractor=extractor_name,
            em_profile=dag.name,
            passes=passes.copy(),
            ethos_profile=ethos_name,
            ethos_profile_sha256=ethos_sha256,
            graph_hash=_ir_graph_hash,
        )
        ir.audit = audit

    # DEME V3 Phase 6: strategic analysis + decision proof. Best-effort;
    # silently skipped when erisml-lib isn't installed or when there
    # aren't enough stakeholders for meaningful analysis.
    if options.emit_strategic_analysis and ir.moral_tensor_v3 is not None:
        try:
            from erisml_compiler.erisml_backend.v3_phase6 import compute_strategic_analysis

            ir.strategic_analysis = compute_strategic_analysis(
                ir,
                ir.moral_tensor_v3,
                coalition_mode=options.coalition_mode,
            )
        except Exception as e:  # noqa: BLE001
            _v3_log.warning("Strategic analysis raised %s; skipping", e)

    if options.emit_decision_proof and ir.moral_tensor_v3 is not None:
        try:
            from erisml_compiler.erisml_backend.v3_phase6 import build_decision_proof

            ir.decision_proof = build_decision_proof(
                ir,
                ir.moral_tensor_v3,
                strategic_analysis=ir.strategic_analysis,
            )
        except Exception as e:  # noqa: BLE001
            _v3_log.warning("Decision proof emission raised %s; skipping", e)

    return ir
