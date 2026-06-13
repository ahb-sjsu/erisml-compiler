"""Serialise a ProfileFitResult to the EM-DAG profile YAML format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from erisml_compiler.social_chem.schema import ProfileFitResult


def profile_to_dict(profile: ProfileFitResult) -> dict[str, Any]:
    """YAML-serialisable dict for one fitted profile."""
    return {
        "name": profile.name,
        "description": profile.description,
        "ethos_description": profile.ethos_description,
        "bias_notes": list(profile.bias_notes),
        "corpus": {
            "source": profile.corpus.source,
            "license": profile.corpus.license,
            "citation": profile.corpus.citation,
            "n_rows": profile.corpus.n_rows,
            "n_situations": profile.corpus.n_situations,
            "canonical_sha256": profile.corpus.canonical_sha256,
            "foundation_distribution": dict(profile.corpus.foundation_distribution),
            "judgment_distribution": dict(profile.corpus.judgment_distribution),
            "schema_tsv_columns": list(profile.corpus.schema_tsv_columns),
        },
        "fit_method": profile.fit_method,
        "fitted_date": profile.fitted_date,
        "mft_to_em_mapping": {
            k: {m: round(float(w), 4) for m, w in v.items()}
            for k, v in profile.mft_to_em_mapping.items()
        },
        "weights": {k: round(float(v), 6) for k, v in sorted(profile.weights.items())},
        "priors": {k: round(float(v), 6) for k, v in sorted(profile.priors.items())},
        "coverage": {k: round(float(v), 6) for k, v in sorted(profile.coverage.items())},
        "metadata": dict(profile.metadata),
    }


def write_profile(profile: ProfileFitResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            profile_to_dict(profile),
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return out
