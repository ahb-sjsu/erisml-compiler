"""The 9 moral dimensions of DEME V3.

Derived from the "Nine Dimensions of Ethical Assessment" 3×3 matrix:

|                | What Matters       | Who Decides            | What We Know          |
|----------------|--------------------|------------------------|------------------------|
| Individual     | Autonomy/Agency    | Rights/Duties          | Privacy/Data           |
| Relational     | Virtue/Care        | Consequences/Welfare   | Epistemic Status       |
| Collective     | Justice/Fairness   | Procedural Legitimacy  | Societal/Environmental |

The canonical ordering used by the tensor's k-axis matches
`erisml.ethics.moral_tensor.MORAL_DIMENSION_NAMES` for direct
interoperability.
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
