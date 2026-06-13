"""Fit dear_abby_socialchem_v0.1 and aita_socialchem_v0.1 from the
Social Chem 101 corpus slices (cached locally under data/social-chem-101/).

Run: python scripts/fit_socialchem_profiles.py
Output: src/erisml_compiler/em_dag/profiles/*.yaml
"""
from __future__ import annotations

import time
from pathlib import Path

from erisml_compiler.social_chem import (
    fingerprint_corpus,
    fit_profile,
    load_situations,
    project_situation,
    write_profile,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "social-chem-101"
PROFILES_DIR = (
    Path(__file__).resolve().parent.parent
    / "src" / "erisml_compiler" / "em_dag" / "profiles"
)


DEAR_ABBY_ETHOS = (
    "Per-EM-module weights fit to AI2 crowd-workers' rule-of-thumb tags on the "
    "Dear Abby slice of Social Chem 101 (Forbes et al., EMNLP 2020). "
    "Amplifies the moral channels Dear Abby's audience-and-editor jointly "
    "surface as load-bearing: care/harm, fidelity/loyalty, fairness, and "
    "legitimacy/authority. EM modules with no MFT mapping (epistemic, "
    "autonomy, externality, repair) receive the floor weight; that's a real "
    "coverage gap in the source corpus, not a bug in the fit. Verdict signs "
    "come from AI2 workers reading the original letters, NOT from the "
    "columnist's published replies."
)

AITA_ETHOS = (
    "Per-EM-module weights fit to AI2 crowd-workers' rule-of-thumb tags on the "
    "r/AmItheAsshole slice of Social Chem 101 (Forbes et al., EMNLP 2020). "
    "Reddit AITA is verdict-supervised by the community (YTA/NTA/etc.), and "
    "AI2 workers extracted moral-foundation-tagged RoTs from each post. The "
    "result amplifies the channels Reddit's AITA community surfaces "
    "(strongly fairness + fidelity, less harm/care than Dear Abby). "
    "EM modules with no MFT mapping take the floor weight."
)

SHARED_BIAS = [
    "AI2 MTurk worker demographics (US, 2020-skewed). Verdict directions "
    "reflect what those workers thought the situation's moral judgment was, "
    "not necessarily what the original source's audience would say.",
    "Moral Foundations Theory's 5 channels do not partition the EM-DAG's 10 "
    "modules. Epistemic, autonomy, externality, and repair receive no MFT "
    "signal — their fitted weight is the floor, not a fitted score. The "
    "`coverage:` field per module in this YAML makes that gap explicit.",
    "Hand-curated MFT -> EM mapping (see `mft_to_em_mapping:` field). Tuning "
    "those weights changes the fitted profile. Treat the mapping as a "
    "transparent default that callers can override.",
]

DEAR_ABBY_BIAS = [
    "Source self-selection: Dear Abby's letter-writers (and her newspaper "
    "editor) skew toward certain concerns — family/relational — and silent "
    "concerns are underrepresented.",
    "Time slice: 1985-2017 letters. Cultural mores from that window, "
    "particularly U.S. middle-class.",
    "No columnist verdicts: the Kaggle questions-only data was supplanted by "
    "Social Chem because the latter has explicit verdict labels — but those "
    "labels still come from AI2 workers, not from Abby herself.",
] + SHARED_BIAS

AITA_BIAS = [
    "Source self-selection: r/AmItheAsshole posts come from people who chose "
    "to publicly relitigate a moral dilemma on Reddit. The population skews "
    "young, U.S.-American, internet-literate.",
    "Community verdict bias: r/AITA's voting community has its own norms "
    "(strong egalitarianism, low tolerance for partner/family overreach) "
    "that AI2 workers' RoTs partially encode.",
] + SHARED_BIAS


def fit_one(
    *,
    tsv_path: Path,
    out_yaml: Path,
    area: str,
    name: str,
    description: str,
    ethos_description: str,
    bias_notes: list[str],
) -> None:
    t0 = time.time()
    print(f"--- {name}: loading {tsv_path.name} (area={area}) ---")
    situations = load_situations(tsv_path, area=area)
    t1 = time.time()
    print(f"    loaded {len(situations)} situations in {t1 - t0:.1f}s")

    aggregates = [project_situation(s) for s in situations]
    t2 = time.time()
    print(f"    projected {len(aggregates)} aggregates in {t2 - t1:.1f}s")

    corpus = fingerprint_corpus(
        situations,
        source=f"social-chem-101.v1.0.tsv :: area={area}",
    )
    profile = fit_profile(
        aggregates,
        corpus=corpus,
        name=name,
        description=description,
        ethos_description=ethos_description,
        bias_notes=bias_notes,
    )
    written = write_profile(profile, out_yaml)
    t3 = time.time()
    print(f"    fit + wrote {written.name} in {t3 - t2:.1f}s")
    print(f"    weights (normalised):")
    for m, w in sorted(profile.weights.items(), key=lambda x: -x[1]):
        cov = profile.coverage.get(m, 0.0)
        pri = profile.priors.get(m, 0.0)
        print(f"      {m:15s}  weight={w:6.3f}  prior={pri:+5.2f}  coverage={cov:5.2%}")
    print(f"    sha256={profile.corpus.canonical_sha256[:16]}...")


def main() -> int:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    fit_one(
        tsv_path=DATA_DIR / "social-chem-101.dearabby.tsv",
        out_yaml=PROFILES_DIR / "dear_abby_socialchem_v0.1.yaml",
        area="dearabby",
        name="dear_abby_socialchem_v0.1",
        description=(
            "Per-EM-module weights derived from the Dear Abby slice of "
            "Social Chem 101 (Forbes et al., EMNLP 2020)."
        ),
        ethos_description=DEAR_ABBY_ETHOS,
        bias_notes=DEAR_ABBY_BIAS,
    )
    fit_one(
        tsv_path=DATA_DIR / "social-chem-101.amitheasshole.tsv",
        out_yaml=PROFILES_DIR / "aita_socialchem_v0.1.yaml",
        area="amitheasshole",
        name="aita_socialchem_v0.1",
        description=(
            "Per-EM-module weights derived from the r/AmItheAsshole slice of "
            "Social Chem 101 (Forbes et al., EMNLP 2020)."
        ),
        ethos_description=AITA_ETHOS,
        bias_notes=AITA_BIAS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
