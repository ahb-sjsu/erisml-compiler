"""Social Chemistry 101 corpus reader + ethos-profile fitter.

The Social Chem 101 corpus (Forbes et al., EMNLP 2020) provides 355k
worker-curated rules-of-thumb across four `area` sources:

    dearabby       50,300 rows   /  11,729 unique situations
    amitheasshole  96,082 rows
    rocstories    101,791 rows
    confessions   107,749 rows

Each row carries:
  - rot-moral-foundations: pipe-separated subset of Haidt's MFT-5
    {care-harm, fairness-cheating, loyalty-betrayal,
     authority-subversion, sanctity-degradation}
  - action-moral-judgment: signed integer in [-2, +2]
  - rot-agree, action-agree: ordinal agreement scales

This subpackage projects those MFT labels onto the ErisML EM-DAG's
10 modules, aggregates per-situation signals, and emits a fitted ethos
profile YAML usable via the compiler's --em-profile flag.

This replaces the deleted `dear_abby` subpackage's title-polarity
weak-supervision path. The Kaggle questions-only corpus is no longer
needed; Social Chem 101 already contains AI2-worker-extracted RoTs
from the same Dear Abby source with proper moral-foundation tags.

What's emitted:

  - em_dag/profiles/dear_abby_socialchem_v0.1.yaml
  - em_dag/profiles/aita_socialchem_v0.1.yaml
    Each: per-module weights + priors fit from one SocialChem area,
    plus ethos_description, bias_notes, and corpus fingerprint.

What this is NOT:

  - Not a moral authority. Each profile is one named ethos among many.
  - Not the columnist's verdicts. The action-moral-judgment values
    reflect AI2 crowd-workers reading the original posts/letters and
    typing what THEY think the moral judgment is. That is its own
    bias source (US MTurk worker demographics, 2020).
  - Not foundation-complete. Several EM modules (epistemic, autonomy,
    externality, repair) have no direct MFT channel; they default to
    floor weight and are documented as such in the profile YAML.
"""

from erisml_compiler.social_chem.fitting import (
    aggregate_situations,
    fit_profile,
    normalise_weights,
)
from erisml_compiler.social_chem.loader import (
    EXPECTED_COLUMNS,
    fingerprint_corpus,
    group_by_situation,
    iter_rows,
    load_situations,
)
from erisml_compiler.social_chem.profile_writer import (
    profile_to_dict,
    write_profile,
)
from erisml_compiler.social_chem.projection import (
    DEFAULT_MFT_TO_EM_DAG,
    MFT_FOUNDATIONS,
    project_situation,
)
from erisml_compiler.social_chem.schema import (
    CorpusFingerprint,
    ProfileFitResult,
    Situation,
    SituationAggregate,
    SituationRoT,
)

__all__ = [
    "CorpusFingerprint",
    "DEFAULT_MFT_TO_EM_DAG",
    "EXPECTED_COLUMNS",
    "MFT_FOUNDATIONS",
    "ProfileFitResult",
    "Situation",
    "SituationAggregate",
    "SituationRoT",
    "aggregate_situations",
    "fingerprint_corpus",
    "fit_profile",
    "group_by_situation",
    "iter_rows",
    "load_situations",
    "normalise_weights",
    "profile_to_dict",
    "project_situation",
    "write_profile",
]
