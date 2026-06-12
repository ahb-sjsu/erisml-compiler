"""Calibration: train probe heads on a frozen LaBSE backbone.

Adapts the BIP probe architecture from `ahb-sjsu/sqnd-probe` (v10.16.9):

  - Frozen multilingual encoder (LaBSE by default)
  - Multi-head probe(s) for downstream extraction (stakeholder roles,
    commitment status, ethical-fact kinds)
  - Adversarial heads (language and period) discouraging the encoder from
    using nuisance shortcuts
  - Spectral decoupling: minimises cross-covariance between embedding and
    nuisance variables (sqnd-probe v10.16.9)
  - Variational Information Bottleneck (VIB) for compressed representations
  - Confusion loss: encodes "no classifier can predict the nuisance"
  - Bond Index: aggregate cross-cultural / cross-lingual generalisation score

Phase 3 ships:
  - The architecture and the training loop (verified on synthetic data)
  - A `ProbeExtractor` that loads probe checkpoints and runs inference
  - An integration path for real sqnd-probe v10.16.9 checkpoints (drop in
    the .pt file and configure the head schema)

Phase 3 does NOT ship trained weights -- those require a real corpus and
GPU compute. The loop runs; the dataset class needs your corrected-IR
corpus and source texts to produce useful weights.
"""

from erisml_compiler.calibration.adversarial_heads import (
    AdversarialHead,
    MultiHeadAdversarial,
)
from erisml_compiler.calibration.bond_index import compute_bond_index
from erisml_compiler.calibration.dataset import (
    ProbeBatch,
    ProbeTrainingDataset,
)
from erisml_compiler.calibration.losses import (
    confusion_loss,
    spectral_decoupling_loss,
    vib_kl_loss,
)
from erisml_compiler.calibration.probe_head import ProbeHead, ProbeBackbone
from erisml_compiler.calibration.train import CalibrationConfig, train_probe

__all__ = [
    "AdversarialHead",
    "CalibrationConfig",
    "MultiHeadAdversarial",
    "ProbeBackbone",
    "ProbeBatch",
    "ProbeHead",
    "ProbeTrainingDataset",
    "compute_bond_index",
    "confusion_loss",
    "spectral_decoupling_loss",
    "train_probe",
    "vib_kl_loss",
]
