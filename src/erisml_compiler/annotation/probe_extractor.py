"""ProbeExtractor: Tier 2.5 extractor.

Frozen LaBSE backbone + trained probe heads. Deterministic, multilingual,
no LLM API needed at inference. Falls between RuleExtractor (Tier 2,
deterministic but limited) and LLMExtractor (Tier 3, expressive but
non-deterministic and network-dependent).

Phase 3 ships:
    - The class wiring
    - Inference path: text -> LaBSE -> probe head -> stakeholder roles,
      commitment status, ethical-fact kinds
    - Checkpoint loading via torch
    - Graceful fallback when torch / sentence-transformers / checkpoint
      is unavailable

What it does NOT do without a trained checkpoint:
    - Produce useful predictions. With random-init probe heads it returns
      noise. The whole point of the calibration loop (calibrate command)
      is to produce checkpoints with meaningful predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from erisml_compiler.annotation.base import Extractor, ExtractorResult
from erisml_compiler.ir.schemas import (
    EthicalFact,
    EthicalFactKind,
    Stakeholder,
    StakeholderRole,
)

# Default class vocabularies. These are dataset-dependent: a real
# checkpoint would record its own vocabularies in its `config`.
DEFAULT_ROLE_CLASSES: tuple[StakeholderRole, ...] = (
    "agent",
    "patient",
    "victim",
    "authority",
    "coercer",
    "protector",
    "bystander",
    "nonconsenting_third_party",
)
DEFAULT_FACT_KIND_CLASSES: tuple[EthicalFactKind, ...] = (
    "coercion",
    "consent",
    "harm",
    "externality",
    "legitimacy",
    "care",
    "truth",
    "role_duty",
    "deception",
)


@dataclass
class ProbeExtractorConfig:
    role_classes: tuple[str, ...] = DEFAULT_ROLE_CLASSES
    fact_kind_classes: tuple[EthicalFactKind, ...] = DEFAULT_FACT_KIND_CLASSES
    role_checkpoint: str | None = None
    fact_kind_checkpoint: str | None = None
    labse_model: str = "sentence-transformers/LaBSE"
    device: str = "cpu"


class ProbeExtractor(Extractor):
    """Tier 2.5 probe-based extractor."""

    name = "probe"

    def __init__(self, config: ProbeExtractorConfig | None = None):
        self.config = config or ProbeExtractorConfig()
        self._role_backbone: Any | None = None
        self._fact_backbone: Any | None = None
        self._load_attempted = False

    def _lazy_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from erisml_compiler.calibration.train import load_checkpoint
        except ImportError:
            return
        if self.config.role_checkpoint and Path(self.config.role_checkpoint).exists():
            self._role_backbone = load_checkpoint(
                self.config.role_checkpoint,
                labse_model=self.config.labse_model,
                device=self.config.device,
            )
        if self.config.fact_kind_checkpoint and Path(self.config.fact_kind_checkpoint).exists():
            self._fact_backbone = load_checkpoint(
                self.config.fact_kind_checkpoint,
                labse_model=self.config.labse_model,
                device=self.config.device,
            )

    def _predict_role(self, text: str) -> tuple[str, float]:
        if self._role_backbone is None:
            return ("agent", 0.1)
        import torch

        emb = self._role_backbone.encode_texts([text])
        with torch.no_grad():
            logits, _, _, _ = self._role_backbone(emb)
            probs = torch.softmax(logits, dim=-1)
            cls = int(probs.argmax(dim=-1).item())
            conf = float(probs.max().item())
        role = self.config.role_classes[min(cls, len(self.config.role_classes) - 1)]
        return (role, conf)

    def _predict_fact_kind(self, text: str) -> tuple[EthicalFactKind, float]:
        if self._fact_backbone is None:
            return ("coercion", 0.1)
        import torch

        emb = self._fact_backbone.encode_texts([text])
        with torch.no_grad():
            logits, _, _, _ = self._fact_backbone(emb)
            probs = torch.softmax(logits, dim=-1)
            cls = int(probs.argmax(dim=-1).item())
            conf = float(probs.max().item())
        kind = self.config.fact_kind_classes[min(cls, len(self.config.fact_kind_classes) - 1)]
        return (kind, conf)

    def extract(self, document, segments) -> ExtractorResult:
        self._lazy_load()

        result = ExtractorResult(
            extractor_metadata={
                "extractor": "probe",
                "role_checkpoint": self.config.role_checkpoint,
                "fact_kind_checkpoint": self.config.fact_kind_checkpoint,
                "labse_model": self.config.labse_model,
                "no_checkpoints_loaded": (
                    self._role_backbone is None and self._fact_backbone is None
                ),
            }
        )
        if not segments:
            return result

        # Self (narrator) stakeholder, role predicted from the first segment.
        first_text = segments[0].text
        role, role_conf = self._predict_role(first_text)
        result.stakeholders.append(
            Stakeholder(
                id="self",
                label="Document narrator/subject",
                type="individual",
                roles=[role],
                vulnerability="moderate",
                consent_status="n/a",
                source_spans=[f"{segments[0].segment_id}:0-{len(first_text)}"],
                confidence=role_conf,
                requires_review=(role_conf < 0.5),
            )
        )

        # One fact per segment, kind predicted by the fact probe.
        for i, seg in enumerate(segments):
            kind, conf = self._predict_fact_kind(seg.text)
            result.ethical_facts.append(
                EthicalFact(
                    id=f"probe_fact_{i+1:03d}",
                    kind=kind,
                    subjects=["self"],
                    description=(
                        f"Probe-detected {kind} signal in {seg.segment_id} "
                        f"(text begins: {seg.text[:60]!r})"
                    ),
                    severity="moderate",
                    confidence=conf,
                    source_spans=[f"{seg.segment_id}:0-{len(seg.text)}"],
                )
            )

        # Canonical form left None at this layer; the pipeline's
        # canonicalizer pass picks it up from the situation summary.
        return result
