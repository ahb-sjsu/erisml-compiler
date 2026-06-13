"""Tests for the social_chem corpus reader + ethos profile fitter."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import yaml

from erisml_compiler.social_chem import (
    CorpusFingerprint,
    DEFAULT_MFT_TO_EM_DAG,
    EXPECTED_COLUMNS,
    MFT_FOUNDATIONS,
    ProfileFitResult,
    Situation,
    SituationRoT,
    aggregate_situations,
    fingerprint_corpus,
    fit_profile,
    group_by_situation,
    iter_rows,
    load_situations,
    normalise_weights,
    profile_to_dict,
    project_situation,
    write_profile,
)
from erisml_compiler.social_chem.fitting import EM_DAG_MODULES_DEFAULT


# ------------------------------------------------------------------ helpers


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EXPECTED_COLUMNS), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({col: r.get(col, "") for col in EXPECTED_COLUMNS})


def _row(**kw: str) -> dict[str, str]:
    base = {col: "" for col in EXPECTED_COLUMNS}
    base.update(kw)
    return base


@pytest.fixture
def tiny_tsv(tmp_path: Path) -> Path:
    rows = [
        # situation A — 3 RoT rows tagging care-harm, fairness-cheating, loyalty-betrayal
        _row(
            area="dearabby",
            situation="MOTHER LIES TO DAUGHTER",
            **{
                "situation-short-id": "dearabby/title/mother-lies",
                "rot": "It is bad to lie to your children.",
                "rot-moral-foundations": "care-harm",
                "rot-agree": "5",
                "rot-categorization": "morality-ethics",
                "action": "lying to your children.",
                "action-moral-judgment": "-2",
                "action-agree": "5",
            },
        ),
        _row(
            area="dearabby",
            situation="MOTHER LIES TO DAUGHTER",
            **{
                "situation-short-id": "dearabby/title/mother-lies",
                "rot": "It is wrong to deceive a family member.",
                "rot-moral-foundations": "loyalty-betrayal|fairness-cheating",
                "rot-agree": "4",
                "rot-categorization": "morality-ethics",
                "action-moral-judgment": "-1",
                "action-agree": "4",
            },
        ),
        _row(
            area="dearabby",
            situation="MOTHER LIES TO DAUGHTER",
            **{
                "situation-short-id": "dearabby/title/mother-lies",
                "rot": "It is good to be honest with family.",
                "rot-moral-foundations": "loyalty-betrayal",
                "rot-agree": "3",
                "rot-categorization": "social-norms",
                "action-moral-judgment": "1",
                "action-agree": "3",
            },
        ),
        # situation B — single row tagging authority-subversion only
        _row(
            area="dearabby",
            situation="EMPLOYEE TALKS BACK TO BOSS",
            **{
                "situation-short-id": "dearabby/title/employee-talks-back",
                "rot": "It is bad to disrespect authority.",
                "rot-moral-foundations": "authority-subversion",
                "rot-agree": "4",
                "rot-categorization": "morality-ethics",
                "action-moral-judgment": "-1",
                "action-agree": "4",
            },
        ),
        # AITA row that should be filtered out when area='dearabby'
        _row(
            area="amitheasshole",
            situation="REDDIT POST",
            **{
                "situation-short-id": "amitheasshole/post/123",
                "rot": "It is good to apologize.",
                "rot-moral-foundations": "care-harm",
                "rot-agree": "5",
                "action-moral-judgment": "2",
            },
        ),
        # Row with no foundations / no judgment (should be loaded but inert)
        _row(
            area="dearabby",
            situation="NEUTRAL TITLE",
            **{
                "situation-short-id": "dearabby/title/neutral",
                "rot": "It is normal to enjoy weekends.",
                "rot-categorization": "description",
            },
        ),
    ]
    path = tmp_path / "tiny.tsv"
    _write_tsv(path, rows)
    return path


# ------------------------------------------------------------------ loader


def test_iter_rows_filters_by_area(tiny_tsv: Path) -> None:
    dearabby = list(iter_rows(tiny_tsv, area="dearabby"))
    aita = list(iter_rows(tiny_tsv, area="amitheasshole"))
    assert len(dearabby) == 5
    assert len(aita) == 1
    assert dearabby[0].rot_moral_foundations == ("care-harm",)
    assert dearabby[1].rot_moral_foundations == ("loyalty-betrayal", "fairness-cheating")


def test_iter_rows_no_filter(tiny_tsv: Path) -> None:
    all_rows = list(iter_rows(tiny_tsv))
    assert len(all_rows) == 6


def test_iter_rows_limit(tiny_tsv: Path) -> None:
    assert len(list(iter_rows(tiny_tsv, limit=2))) == 2


def test_iter_rows_raises_for_unexpected_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("foo\tbar\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected TSV columns"):
        list(iter_rows(bad))


def test_iter_rows_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_rows(tmp_path / "nope.tsv"))


def test_load_situations_groups_correctly(tiny_tsv: Path) -> None:
    sits = load_situations(tiny_tsv, area="dearabby")
    sids = {s.situation_short_id for s in sits}
    assert sids == {
        "dearabby/title/mother-lies",
        "dearabby/title/employee-talks-back",
        "dearabby/title/neutral",
    }
    by_sid = {s.situation_short_id: s for s in sits}
    assert len(by_sid["dearabby/title/mother-lies"].rots) == 3
    assert len(by_sid["dearabby/title/employee-talks-back"].rots) == 1


def test_fingerprint_deterministic(tiny_tsv: Path) -> None:
    sits1 = load_situations(tiny_tsv, area="dearabby")
    sits2 = load_situations(tiny_tsv, area="dearabby")
    fp1 = fingerprint_corpus(sits1, source="t")
    fp2 = fingerprint_corpus(list(reversed(sits2)), source="t")
    assert fp1.canonical_sha256 == fp2.canonical_sha256
    assert fp1.n_situations == 3
    assert fp1.foundation_distribution["care-harm"] == 1
    assert fp1.foundation_distribution["loyalty-betrayal"] == 2


# ------------------------------------------------------------------ projection


def test_default_mft_to_em_mapping_covers_all_five() -> None:
    for f in MFT_FOUNDATIONS:
        assert f in DEFAULT_MFT_TO_EM_DAG, f"{f} missing from default mapping"


def test_projection_signal_sign_matches_judgment() -> None:
    rot = SituationRoT(
        area="dearabby",
        situation_short_id="x",
        situation="x",
        rot="It is bad to lie.",
        rot_categorization="morality-ethics",
        rot_moral_foundations=("care-harm",),
        rot_agree=5,
        action="lying.",
        action_moral_judgment=-2,
    )
    sit = Situation(situation_short_id="x", situation="x", area="dearabby", rots=(rot,))
    agg = project_situation(sit)
    # care-harm -> harm (negative judgment), care (slightly less weight)
    assert agg.per_module_value["harm"] < 0
    assert agg.per_module_value["care"] < 0
    assert agg.per_module_value["harm"] == pytest.approx(-1.0, abs=0.01)


def test_projection_neutral_row_yields_zero_value() -> None:
    rot = SituationRoT(
        area="dearabby",
        situation_short_id="x",
        situation="x",
        rot="It is normal.",
        rot_categorization="description",
        rot_moral_foundations=("care-harm",),
        rot_agree=3,
        action_moral_judgment=0,
    )
    sit = Situation(situation_short_id="x", situation="x", area="dearabby", rots=(rot,))
    agg = project_situation(sit)
    assert agg.per_module_value["harm"] == pytest.approx(0.0, abs=0.01)


def test_projection_unmapped_module_absent() -> None:
    rot = SituationRoT(
        area="dearabby",
        situation_short_id="x",
        situation="x",
        rot="It is good to read books.",
        rot_categorization="description",
        rot_moral_foundations=("care-harm",),
        rot_agree=3,
        action_moral_judgment=1,
    )
    sit = Situation(situation_short_id="x", situation="x", area="dearabby", rots=(rot,))
    agg = project_situation(sit)
    # epistemic has no MFT mapping, so it should not appear in the aggregate
    assert "epistemic" not in agg.per_module_value


def test_projection_agreement_weighted_mean() -> None:
    rot_hi = SituationRoT(
        area="dearabby", situation_short_id="x", situation="x",
        rot="bad", rot_categorization="morality-ethics",
        rot_moral_foundations=("care-harm",), rot_agree=5, action_moral_judgment=-2,
    )
    rot_lo = SituationRoT(
        area="dearabby", situation_short_id="x", situation="x",
        rot="good", rot_categorization="morality-ethics",
        rot_moral_foundations=("care-harm",), rot_agree=1, action_moral_judgment=2,
    )
    sit = Situation(situation_short_id="x", situation="x", area="dearabby", rots=(rot_hi, rot_lo))
    agg = project_situation(sit)
    # Mean weighted by agree: (-1.0 * 1.0 + 1.0 * 0.2) / 1.2 = -0.667
    assert agg.per_module_value["harm"] < -0.5


def test_projection_falls_back_to_rot_text_when_judgment_missing() -> None:
    rot = SituationRoT(
        area="dearabby",
        situation_short_id="x",
        situation="x",
        rot="It is bad to abandon your friends.",
        rot_categorization="social-norms",
        rot_moral_foundations=("loyalty-betrayal",),
        rot_agree=4,
        action_moral_judgment=None,
    )
    sit = Situation(situation_short_id="x", situation="x", area="dearabby", rots=(rot,))
    agg = project_situation(sit)
    assert agg.per_module_value["fidelity"] < 0


# ------------------------------------------------------------------ fitting


def test_aggregate_situations_includes_all_modules(tiny_tsv: Path) -> None:
    sits = load_situations(tiny_tsv, area="dearabby")
    aggs = [project_situation(s) for s in sits]
    weights_raw, priors, coverage, n = aggregate_situations(aggs)
    assert set(weights_raw.keys()) == set(EM_DAG_MODULES_DEFAULT)
    # Modules with MFT signal in the tiny corpus
    assert coverage["harm"] > 0
    assert coverage["fidelity"] > 0
    assert coverage["legitimacy"] > 0
    # Modules without MFT signal
    assert coverage["epistemic"] == 0.0
    assert coverage["autonomy"] == 0.0


def test_normalise_weights_sum_equals_n_modules() -> None:
    raw = {"a": 0.4, "b": 0.6, "c": 1.0}
    normed = normalise_weights(raw)
    assert math.isclose(sum(normed.values()), len(raw), abs_tol=1e-6)


def test_normalise_weights_degenerate_zero_total() -> None:
    raw = {"a": 0.0, "b": 0.0}
    normed = normalise_weights(raw)
    assert all(v == 1.0 for v in normed.values())


def test_fit_profile_end_to_end(tiny_tsv: Path, tmp_path: Path) -> None:
    sits = load_situations(tiny_tsv, area="dearabby")
    aggs = [project_situation(s) for s in sits]
    fp = fingerprint_corpus(sits, source="test/dearabby_tiny")

    profile = fit_profile(
        aggs,
        corpus=fp,
        name="dear_abby_socialchem_v0.1_test",
        description="test fit",
        ethos_description="test ethos description",
        bias_notes=["note 1", "note 2"],
    )
    # Sum-to-n normalisation across modules that actually received signal
    assert math.isclose(sum(profile.weights.values()), len(profile.weights), abs_tol=1e-6)
    # Unmapped modules are dropped from `weights` (so the loader defaults to 1.0)
    # but kept in `coverage` for the audit trail.
    assert set(profile.weights.keys()).issubset(set(EM_DAG_MODULES_DEFAULT))
    assert set(profile.coverage.keys()) == set(EM_DAG_MODULES_DEFAULT)
    assert profile.coverage["epistemic"] == 0.0
    assert profile.corpus.canonical_sha256 == fp.canonical_sha256
    assert profile.fit_method == "mft_to_em_via_agreement_weighted_means"
    assert profile.mft_to_em_mapping == DEFAULT_MFT_TO_EM_DAG

    out = tmp_path / "p.yaml"
    write_profile(profile, out)
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["name"] == profile.name
    assert loaded["fit_method"] == profile.fit_method
    assert "mft_to_em_mapping" in loaded
    assert "coverage" in loaded
    assert "bias_notes" in loaded
    assert isinstance(loaded["weights"], dict)


def test_profile_to_dict_round_trip(tiny_tsv: Path) -> None:
    sits = load_situations(tiny_tsv, area="dearabby")
    aggs = [project_situation(s) for s in sits]
    fp = fingerprint_corpus(sits, source="t")
    profile = fit_profile(
        aggs, corpus=fp,
        name="x", description="d",
        ethos_description="e", bias_notes=["b"],
    )
    d = profile_to_dict(profile)
    assert d["name"] == profile.name
    assert d["corpus"]["citation"] == fp.citation
    assert d["weights"]["harm"] == round(profile.weights["harm"], 6)
