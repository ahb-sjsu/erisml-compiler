"""xbse-backed dimension scoring — plug validated `*-BSE` feeders into the compiler.

Each DEME-9 dimension has a cross-dataset-validated `xbse` encoder whose signed valence axis maps a
text to a scalar in [-1, 1] (`xbse.scorer.DimensionScorer`). This module wraps those into the
compiler's `DimensionScore` type, behind a `DimensionScoringBackend` protocol so it is a drop-in
alternative to the rubric backend.

Discipline: a feeder may only be used if its checkpoint PASSED the gate. `from_checkpoints` builds
each per-dimension scorer through `xbse.DimensionScorer.from_pairsource`, which calls `require_pass`
— so the compiler cannot score with an unvalidated encoder.

`xbse` is an optional dependency (`pip install erisml-compiler[scorers]`); it is imported lazily so
this module (and its type/protocol) import even where `xbse` and the checkpoints are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from erisml_compiler.ir.schemas import DimensionScore
from erisml_compiler.ir.v3.dimensions import MORAL_DIMENSIONS_V3

if TYPE_CHECKING:  # avoid a hard xbse import at module load
    from xbse.scorer import DimensionScorer as XBSEScorer, Valence

# DEME-9 dimension -> the joint feeder that scores it. Checkpoint / report file names are derived:
#   checkpoint = f"{feeder}.pt"          report = f"{feeder}_report.json"
DEME9_REGISTRY: dict[str, str] = {
    "physical_harm": "physharm_joint",
    "rights_respect": "rights_joint",
    "fairness_equity": "fairness_joint",
    "autonomy_respect": "autonomy_joint",
    "privacy_protection": "privacy_joint",
    "societal_environmental": "environmental_joint",
    "virtue_care": "care_joint",
    "legitimacy_trust": "legitimacy_joint",
    "epistemic_quality": "epistemic_joint",
}

_DEFAULT_CKPT_DIR = "/home/claude/xbse_ckpt"  # training host; override in from_checkpoints


class DimensionScoringBackend(Protocol):
    """Anything that can score one dimension of a text into a DimensionScore."""

    def score(self, text: str, dimension: str) -> DimensionScore: ...


def valence_to_dimension_score(valence: Valence, dimension: str) -> DimensionScore:
    """Convert an xbse signed Valence into the compiler's DimensionScore."""
    value = max(-1.0, min(1.0, float(valence.value)))
    conf = max(0.0, min(1.0, float(valence.confidence)))
    return DimensionScore(
        value=value,
        confidence=conf,
        uncertainty=max(0.0, 1.0 - conf),
        direction=valence.direction,
        source_spans=[],
        explanation=f"xbse:{dimension} cross-dataset feeder",
    )


def _neutral(dimension: str, why: str) -> DimensionScore:
    return DimensionScore(
        value=0.0, confidence=0.0, uncertainty=1.0, direction="neutral",
        source_spans=[], explanation=f"xbse:{dimension} — {why}",
    )


class XBSEDimensionScorer:
    """Scores MoralVector dimensions from validated xbse feeders (one per DEME-9 dimension)."""

    def __init__(self, scorers: dict[str, XBSEScorer], reports: dict | None = None):
        # dimension name (DEME-9) -> xbse.DimensionScorer
        self._scorers = dict(scorers)
        # dimension name -> xbse.Report (kept so scored dimensions carry validation provenance)
        self._reports = dict(reports or {})

    @property
    def dimensions(self) -> list[str]:
        return list(self._scorers)

    def validation_records(self) -> list[dict]:
        """Per-dimension validation provenance for the audit artifact.

        Each dict matches erisml.ethics.decision_proof.FeederValidationRecord, so the DEMEv3
        DecisionProof can bind every xbse-scored dimension to the encoder + pre-registered bar that
        produced it (identity, bar, cross-dataset AUROC, lexical-control margin, PASS/FAIL)."""
        out = []
        for dimension, report in self._reports.items():
            m = getattr(report, "metrics", {}) or {}
            thr = getattr(report, "thresholds", {}) or {}
            out.append(
                {
                    "dimension": dimension,
                    "feeder_name": report.instance,
                    "checkpoint_hash": report.checkpoint_hash,
                    "bar_auroc_min": float(thr.get("auroc>", 0.0)),
                    "bar_source": getattr(report, "bar_source", ""),
                    "bar_derivation": getattr(report, "bar_derivation", ""),
                    "bar_registered": getattr(report, "bar_registered", ""),
                    "structure_auroc": float(m.get("structure_auroc", 0.0)),
                    "bow_auroc": float(m.get("bow_auroc", 0.0)),
                    "lexical_margin": float(m.get("lexical_margin", 0.0)),
                    "validated": bool(getattr(report, "passed", False)),
                }
            )
        return out

    def score(self, text: str, dimension: str) -> DimensionScore:
        scorer = self._scorers.get(dimension)
        if scorer is None:
            return _neutral(dimension, "no validated feeder loaded")
        return valence_to_dimension_score(scorer.score(text), dimension)

    def score_all(self, text: str) -> dict[str, DimensionScore]:
        """Score every DEME-9 dimension; dimensions without a loaded feeder come back neutral."""
        return {dim: self.score(text, dim) for dim in MORAL_DIMENSIONS_V3}

    @classmethod
    def from_checkpoints(
        cls,
        registry: dict[str, str] | None = None,
        ckpt_dir: str = _DEFAULT_CKPT_DIR,
        device: str = "cuda",
        base_model: str = "BAAI/bge-m3",
    ) -> XBSEDimensionScorer:
        """Load each dimension's validated encoder + PASS report and build its scorer (gated).

        Runs on the training host where the checkpoints live. Any dimension whose checkpoint or
        report is missing, or whose report is not a PASS, is skipped (scored neutral at inference).
        """
        import json
        import os

        import torch

        from xbse.encoder import BSEEncoder
        from xbse.instances.joint_builders import BUILDERS
        from xbse.report import Report
        from xbse.scorer import DimensionScorer as XBSEScorer

        registry = registry or DEME9_REGISTRY
        scorers: dict[str, XBSEScorer] = {}
        reports: dict = {}
        for dimension, feeder in registry.items():
            ckpt = os.path.join(ckpt_dir, f"{feeder}.pt")
            report_path = os.path.join(ckpt_dir, f"{feeder}_report.json")
            if not (os.path.exists(ckpt) and os.path.exists(report_path)):
                continue
            with open(report_path) as fh:
                report = Report(**json.load(fh))
            src = BUILDERS[feeder]()
            enc = BSEEncoder(base_model=base_model, max_len=src.max_len, device=device)
            enc.load_state_dict(torch.load(ckpt, map_location=device))
            scorers[dimension] = XBSEScorer.from_pairsource(
                enc, src, report, report.checkpoint_hash
            )
            reports[dimension] = report
        return cls(scorers, reports)
