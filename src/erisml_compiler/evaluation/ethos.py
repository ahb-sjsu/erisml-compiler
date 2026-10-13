"""Fitted ethos profiles, read as DEME dimension weights.

An ethos profile (e.g. ``profiles/dear_abby_socialchem_v0.1.yaml``, fitted by
``erisml_compiler.social_chem`` to the Dear Abby slice of Social Chemistry 101) weighs the
compiler's ten moral channels, named by the module that once scored each. Since 0.10 DEME
judges, and its nine dimensions are the canonical core (erisml-lib
docs/moralvector_v2_architecture.md section 3), so a profile's weights are carried onto DEME's
``DimensionWeights`` through the shipped compiler-10 to DEME-9 map
(``ir.v3.dimensions.V2_TO_V3_DIMENSION_MAP``): a channel's weight goes to the DEME dimension it
maps to; vow fidelity is split evenly between legitimacy and virtue/care, as the map splits
it; repair has no DEME dimension. A DEME dimension no weighted channel reaches keeps the
library's default weight. The profile's file hash is recorded in the audit, as before.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from erisml.ethics.governance.config_v2 import DimensionWeights

from erisml_compiler.ir.v3.dimensions import V2_TO_V3_DIMENSION_MAP

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

# The channel each profile key names: the compiler-10 dimension its module scored.
CHANNEL_DIMENSION: dict[str, str] = {
    "harm": "physical_harm",
    "rights": "rights_respect",
    "fairness": "fairness_equity",
    "autonomy": "autonomy_consent",
    "legitimacy": "legitimacy_trust",
    "epistemic": "epistemic_quality",
    "care": "care_protection",
    "fidelity": "vow_fidelity",
    "externality": "third_party_externality",
    "repair": "repair_residue",
}


def load_ethos_profile(path: str | Path) -> dict[str, Any]:
    """Load a fitted ethos profile from YAML, with its provenance (ethos_description,
    bias_notes, corpus) verbatim, so a caller can surface the bias it carries."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "weights" not in data:
        raise ValueError(f"Ethos profile at {p} has no 'weights' section.")
    return data


def ethos_dimension_weights(channel_weights: dict[str, float]) -> DimensionWeights:
    """A profile's channel weights as DEME dimension weights (see the module docstring)."""
    totals: dict[str, float] = {}
    for channel, weight in channel_weights.items():
        targets = V2_TO_V3_DIMENSION_MAP.get(CHANNEL_DIMENSION.get(channel, channel), [])
        for dim in targets:
            totals[dim] = totals.get(dim, 0.0) + float(weight) / len(targets)
    return DimensionWeights(**totals)


def resolve_ethos_profile(
    path: str | Path | None,
) -> tuple[DimensionWeights | None, str | None, str | None]:
    """(DEME dimension weights, profile name, file SHA-256), or Nones without a profile."""
    if path is None:
        return None, None, None
    p = Path(path)
    data = load_ethos_profile(p)
    weights = {str(k): float(v) for k, v in (data.get("weights") or {}).items()}
    sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
    name = str(data.get("name") or p.stem)
    return (ethos_dimension_weights(weights) if weights else None), name, sha256


__all__ = [
    "CHANNEL_DIMENSION",
    "PROFILES_DIR",
    "ethos_dimension_weights",
    "load_ethos_profile",
    "resolve_ethos_profile",
]
