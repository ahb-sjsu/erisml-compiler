"""xbse-backed dimension scoring — plug validated `*-BSE` feeders into the compiler.

Each DEME-9 dimension has a cross-dataset-validated `xbse` encoder whose signed valence axis maps a
text to a scalar in [-1, 1] (`xbse.scorer.DimensionScorer`). This module wraps those into the
compiler's `DimensionScore` type, behind a `DimensionScoringBackend` protocol so it is a drop-in
alternative to the rubric backend.

Discipline: a feeder may only be used if its checkpoint PASSED the gate. `from_checkpoints` builds
each per-dimension scorer through `xbse.DimensionScorer.from_pairsource`, which calls `require_pass`
— so the compiler cannot score with an unvalidated encoder.

Two review-driven refinements (xbse XBSE_REVIEW_1, closed 2026-07-23):

  * **Calibrated authority (R2).** Production reports carry a calibration block (split-honest ECE +
    the registered ``reliability_weight = max(0, 2*AUROC - 1)``). The scorer multiplies each
    feeder's confidence by its weight, so unequal reliabilities reach MoralVector's per-dimension
    uncertainty instead of being laundered into equal authority by the binary PASS bit
    (physical_harm enters at weight 0.26, privacy at 0.71). `reliability_records()` exposes the
    block for the audit artifact.
  * **Specificity dispositions (R1).** The registered 12x12 gate matrix demotes feeders that cannot
    beat their trained siblings (or the general-valence channel G) on their own held-out pairs:
    care / fairness / legitimacy / epistemic scores are real signal but substantially *general
    moral valence*; their DimensionScores say so (`SPECIFICITY_DISPOSITIONS`).

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

# Specificity dispositions from the registered 12x12 gate matrix (xbse
# experiments/specificity_verdicts.json, run 2026-07-23, margin 0.05): an "own-axis" feeder beats
# every trained sibling AND the validated general-valence channel G on its own held-out pairs; a
# "DEMOTE-to-G" feeder cannot — its cross-dataset AUROC is real signal that is substantially
# general moral valence, not its named dimension. Scores from demoted feeders remain usable but
# carry the disposition in their provenance; consumers must not read them as axis-specific.
SPECIFICITY_DISPOSITIONS: dict[str, str] = {
    "physical_harm": "own-axis",
    "privacy_protection": "own-axis",
    "autonomy_respect": "own-axis",
    "societal_environmental": "own-axis",
    "virtue_care": "DEMOTE-to-G",
    "fairness_equity": "DEMOTE-to-G",
    "legitimacy_trust": "DEMOTE-to-G",
    "epistemic_quality": "DEMOTE-to-G",
    # rights_respect: no validated feeder. The discriminating CourtListener run (2026-07-23)
    # FAILED — corpus-choice hypothesis refuted, method-failure branch open (xbse README).
}


class DimensionScoringBackend(Protocol):
    """Anything that can score one dimension of a text into a DimensionScore."""

    def score(self, text: str, dimension: str) -> DimensionScore: ...


def valence_to_dimension_score(
    valence: Valence,
    dimension: str,
    reliability: float | None = None,
    disposition: str | None = None,
) -> DimensionScore:
    """Convert an xbse signed Valence into the compiler's DimensionScore.

    ``reliability`` is the registered per-feeder ``reliability_weight = max(0, 2*AUROC - 1)`` from
    the report's calibration block (XBSE_REVIEW_1 R2): confidence is multiplied by it, so a
    0.63-AUROC feeder (weight 0.26) enters MoralVector uncertainty at ~4x the uncertainty of a
    0.91-AUROC feeder (weight 0.82) instead of with equal authority. ``disposition`` is the
    specificity verdict; a DEMOTE-to-G score is flagged in its explanation."""
    value = max(-1.0, min(1.0, float(valence.value)))
    conf = max(0.0, min(1.0, float(valence.confidence)))
    explanation = f"xbse:{dimension} cross-dataset feeder"
    if reliability is not None:
        conf *= max(0.0, min(1.0, float(reliability)))
        explanation += f" [reliability_weight={float(reliability):.3f}]"
    if disposition == "DEMOTE-to-G":
        explanation += (
            " [specificity: DEMOTE-to-G — reads general moral valence, not axis-specific]"
        )
    return DimensionScore(
        value=value,
        confidence=conf,
        uncertainty=max(0.0, 1.0 - conf),
        direction=valence.direction,
        source_spans=[],
        explanation=explanation,
    )


def _neutral(dimension: str, why: str) -> DimensionScore:
    return DimensionScore(
        value=0.0,
        confidence=0.0,
        uncertainty=1.0,
        direction="neutral",
        source_spans=[],
        explanation=f"xbse:{dimension} — {why}",
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

    def _reliability(self, dimension: str) -> float | None:
        """The registered reliability weight from the report's calibration block, if wired."""
        report = self._reports.get(dimension)
        block = getattr(report, "calibration", None) or {}
        w = block.get("reliability_weight")
        return float(w) if w is not None else None

    def reliability_records(self) -> list[dict]:
        """Per-dimension calibration + specificity provenance for the audit artifact.

        Kept separate from `validation_records()` (whose field set is pinned to
        erisml.ethics.decision_proof.FeederValidationRecord): these are the R2/R1 closure fields —
        split-honest ECE, the reliability weight the scores were multiplied by, and the 12x12
        specificity disposition."""
        out = []
        for dimension, report in self._reports.items():
            block = getattr(report, "calibration", None) or {}
            out.append(
                {
                    "dimension": dimension,
                    "reliability_weight": self._reliability(dimension),
                    "calibration_method": block.get("calibration_method"),
                    "calibration_ece": block.get("calibration_ece"),
                    "raw_ece": block.get("raw_ece"),
                    "n_calibration_pairs": block.get("n_calibration_pairs"),
                    "specificity_disposition": SPECIFICITY_DISPOSITIONS.get(dimension),
                }
            )
        return out

    def score(self, text: str, dimension: str) -> DimensionScore:
        scorer = self._scorers.get(dimension)
        if scorer is None:
            return _neutral(dimension, "no validated feeder loaded")
        return valence_to_dimension_score(
            scorer.score(text),
            dimension,
            reliability=self._reliability(dimension),
            disposition=SPECIFICITY_DISPOSITIONS.get(dimension),
        )

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
