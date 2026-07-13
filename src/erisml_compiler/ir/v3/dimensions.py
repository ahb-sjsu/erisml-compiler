"""The MoralVector ontology — the SINGLE SOURCE OF TRUTH for DEME's moral dimensions.

This module is the canonical definition of the moral vector's channels: the 9
frozen k-axis dimensions plus the validated extension channels. Every other
package references THIS module rather than keeping its own copy:

  - `erisml.ethics.moral_tensor` (erisml-lib / DEME engine) imports these names.
  - `gtc-prototype` (`gtc.__init__`) imports `MORAL_DIMENSIONS_V3` + the
    extension channels from here.
  - `agi.safety.erisml.moral_tensor` (agi-hpc safety gateway) is a separate,
    safety-critical system; it carries a guarded copy checked against this
    module by `tests/test_dimension_consistency.py`.

Why here: `erisml-compiler` is the base of the dependency graph — consumers
hard-depend on it, and it stands alone (it only *optionally* touches erisml-lib
via a lazy import), so it is the one node every consumer can reach without a
dependency cycle.

The 9 k-axis dimensions derive from the "Nine Dimensions of Ethical Assessment"
3×3 matrix (frozen — see `MORAL_DIMENSIONS_V3`):

|                | What Matters       | Who Decides            | What We Know          |
|----------------|--------------------|------------------------|------------------------|
| Individual     | Autonomy/Agency    | Rights/Duties          | Privacy/Data           |
| Relational     | Virtue/Care        | Consequences/Welfare   | Epistemic Status       |
| Collective     | Justice/Fairness   | Procedural Legitimacy  | Societal/Environmental |

Extension channels (validated moral foundations OUTSIDE the frozen k-axis) are
defined below as `MORAL_EXTENSION_CHANNELS`; the full ordered vocabulary is
`MORAL_VECTOR_CHANNELS`.
"""

from __future__ import annotations

from enum import Enum

# Canonical k-axis ordering. MUST match
# `erisml.ethics.moral_tensor.MORAL_DIMENSION_NAMES` exactly.
MORAL_DIMENSIONS_V3: tuple[str, ...] = (
    "physical_harm",  # k=0: Relational / Consequences-and-Welfare
    "rights_respect",  # k=1: Individual / Rights-and-Duties
    "fairness_equity",  # k=2: Collective / Justice-and-Fairness
    "autonomy_respect",  # k=3: Individual / Autonomy-and-Agency
    "privacy_protection",  # k=4: Individual / Privacy-and-Data
    "societal_environmental",  # k=5: Collective / Societal-and-Environmental
    "virtue_care",  # k=6: Relational / Virtue-and-Care
    "legitimacy_trust",  # k=7: Collective / Procedural-Legitimacy
    "epistemic_quality",  # k=8: Relational / Epistemic-Status
)


class DimensionAxis(str, Enum):
    """Names for tensor axes 2..6 of a MoralTensorV3.

    Axis 0 is always `k` (moral dimensions, fixed length 9). Axes
    1..5 are optional and conventionally named. (Inherits from `str`
    + `Enum` rather than `StrEnum` for Python 3.10 compatibility.)
    """

    K = "k"  # moral dimensions (always axis 0, always length 9)
    N = "n"  # parties / stakeholders
    TAU = "tau"  # time
    A = "a"  # action choices
    C = "c"  # coalition configurations
    S = "s"  # uncertainty samples (Monte Carlo)


# Matrix coordinates: 3 levels × 3 framings.
LEVELS: tuple[str, ...] = ("Individual", "Relational", "Collective")
FRAMINGS: tuple[str, ...] = ("What Matters", "Who Decides", "What We Know")


DIMENSION_MATRIX_3X3: dict[str, tuple[str, str]] = {
    "autonomy_respect": ("Individual", "What Matters"),
    "rights_respect": ("Individual", "Who Decides"),
    "privacy_protection": ("Individual", "What We Know"),
    "virtue_care": ("Relational", "What Matters"),
    "physical_harm": ("Relational", "Who Decides"),
    "epistemic_quality": ("Relational", "What We Know"),
    "fairness_equity": ("Collective", "What Matters"),
    "legitimacy_trust": ("Collective", "Who Decides"),
    "societal_environmental": ("Collective", "What We Know"),
}

# Lossy V2 → V3 dimension mapping.
#
# Six V2 dimensions map directly (sometimes renamed):
#   physical_harm        -> physical_harm
#   rights_respect       -> rights_respect
#   fairness_equity      -> fairness_equity
#   autonomy_consent     -> autonomy_respect           (renamed)
#   legitimacy_trust     -> legitimacy_trust
#   epistemic_quality    -> epistemic_quality
#   care_protection      -> virtue_care                (renamed)
#
# Three V2 dimensions do NOT have direct V3 dims and need
# special handling on migration:
#   vow_fidelity         -> contributes to legitimacy_trust + virtue_care
#                            (averaged); flagged in tensor metadata.
#   third_party_externality -> mapped to societal_environmental;
#                            flagged in tensor metadata.
#   repair_residue       -> NOT a V3 dimension. Surfaced as a
#                            *tensor operation* in DEME V3 (residue is the
#                            shortfall after collapse). In migration we
#                            attach it to tensor.metadata["repair_residue"].
#
# One V3 dimension has no V2 source:
#   privacy_protection   -> defaults to 0.0 (neutral) on migration; flagged.
V2_TO_V3_DIMENSION_MAP: dict[str, list[str]] = {
    "physical_harm": ["physical_harm"],
    "rights_respect": ["rights_respect"],
    "fairness_equity": ["fairness_equity"],
    "autonomy_consent": ["autonomy_respect"],
    "legitimacy_trust": ["legitimacy_trust"],
    "epistemic_quality": ["epistemic_quality"],
    "care_protection": ["virtue_care"],
    "vow_fidelity": ["legitimacy_trust", "virtue_care"],  # split
    "third_party_externality": ["societal_environmental"],
    # "repair_residue":         intentionally absent — stored in metadata
}


# ---------------------------------------------------------------------------
# Extension channels (validated moral foundations OUTSIDE the frozen k-axis)
# ---------------------------------------------------------------------------
#
# The canonical k-axis is frozen at 9 (the 3×3 "Nine Dimensions" matrix), and
# is duplicated byte-for-byte across three packages
# (`test_dimension_consistency.py`). New moral foundations therefore do NOT
# widen the k-axis; following the `repair_residue` precedent, they ride as
# named keys in `MoralTensorV3.metadata["extension_channels"]` while
# `shape[0]` stays 9.
#
# `purity` and `loyalty` are the two MFT "binding" foundations absent from the
# DEME-9. Both were validated through the SAME pre-registered cross-dataset
# gate as every k-axis feeder (xbse; see `experiments/b1_results.json` and
# `experiments/foundation_presence_findings.md`):
#
#   - a signed VALENCE feeder supplies the cell value in [-1, 1]
#     (loyalty AUROC 0.911, purity 0.811 — both robust to the domain adversary)
#   - a valence-agnostic PRESENCE feeder supplies the engagement gate: whether
#     the foundation is *engaged* at all (purity 0.719, robust; loyalty 0.661,
#     lam=0 config only — see EXTENSION_CHANNEL_PROVENANCE)
#
# This supersedes the earlier `moralvector_reference.md` note that retired
# purity/sanctity as "off-target, not a feeder": a validated feeder now exists.
MORAL_EXTENSION_CHANNELS: tuple[str, ...] = (
    "purity",
    "loyalty",
)

# Per-channel provenance: the validated xbse feeders behind each extension
# channel. `valence` feeds the signed cell value; `presence` gates engagement.
# `checkpoint_hash` is the sha256[:16] a downstream consumer's `require_pass`
# binds against. `presence_caveat` records the domain-adversarial (lam) result.
EXTENSION_CHANNEL_PROVENANCE: dict[str, dict[str, object]] = {
    "purity": {
        "foundation": "MFT sanctity/degradation",
        "valence": {"auroc": 0.811, "checkpoint_hash": "001506fc21518a5e", "gate": "PASS"},
        "presence": {"auroc": 0.719, "checkpoint_hash": "1997cb0b3ac9d12a", "gate": "PASS"},
        "presence_caveat": "robust - passes at lam=0 AND lam=1 (0.699)",
    },
    "loyalty": {
        "foundation": "MFT loyalty/betrayal",
        "valence": {"auroc": 0.911, "checkpoint_hash": "23d54d10fac7ea90", "gate": "PASS"},
        "presence": {"auroc": 0.661, "checkpoint_hash": "377f1bc8977fbf35", "gate": "PASS"},
        "presence_caveat": "lam=0 joint-contrastive only - the adversary (lam=1) strips it to 0.641 (fail)",
    },
}

# Sanity: extension channels are disjoint from the frozen k-axis.
assert not (set(MORAL_EXTENSION_CHANNELS) & set(MORAL_DIMENSIONS_V3)), (
    "extension channels must not collide with the canonical k-axis"
)

# The full ordered MoralVector vocabulary: the 9 frozen k-axis dimensions
# followed by the validated extension channels. Downstream consumers that need
# "every channel the vector can carry" (readouts, decision membership, audit)
# reference THIS rather than concatenating the two tuples themselves.
MORAL_VECTOR_CHANNELS: tuple[str, ...] = (*MORAL_DIMENSIONS_V3, *MORAL_EXTENSION_CHANNELS)


def is_extension_channel(name: str) -> bool:
    """True if `name` is a validated extension channel (rides in tensor metadata)."""
    return name in MORAL_EXTENSION_CHANNELS


def is_canonical_dimension(name: str) -> bool:
    """True if `name` is one of the 9 frozen k-axis dimensions."""
    return name in MORAL_DIMENSIONS_V3


__all__ = [
    "MORAL_DIMENSIONS_V3",
    "MORAL_EXTENSION_CHANNELS",
    "EXTENSION_CHANNEL_PROVENANCE",
    "MORAL_VECTOR_CHANNELS",
    "DimensionAxis",
    "DIMENSION_MATRIX_3X3",
    "LEVELS",
    "FRAMINGS",
    "V2_TO_V3_DIMENSION_MAP",
    "is_extension_channel",
    "is_canonical_dimension",
]
