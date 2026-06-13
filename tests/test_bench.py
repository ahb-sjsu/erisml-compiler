"""Tests for the MoralTensor-Bench harness."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from erisml_compiler.bench.runner import (
    corpus_hash,
    load_scenarios,
    render_report_markdown,
    run_bench,
)
from erisml_compiler.bench.schema import (
    BenchAggregate,
    ExpectedCommitment,
    ExpectedScenario,
    ExpectedStakeholder,
    ScenarioGold,
    ScenarioScore,
)
from erisml_compiler.bench.scoring import (
    aggregate_score,
    load_weights,
    score_commitment_f1,
    score_ethical_fact_kind_recall,
    score_overall_verdict,
    score_premature_contraction,
    score_scenario,
    score_stakeholder_recall,
    score_stakeholder_role_f1,
    weighted_score,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "src" / "erisml_compiler" / "bench" / "v0.1"


# ---------------------------------------------------- schema + loading


def test_load_scenarios_finds_three_seeds() -> None:
    scenarios = load_scenarios(BENCH_DIR)
    ids = {s.scenario_id for s in scenarios}
    assert ids == {"nazi_attic_001", "medical_confidentiality_001", "whistleblower_001"}


def test_each_scenario_has_required_fields() -> None:
    for s in load_scenarios(BENCH_DIR):
        assert s.category
        assert s.raw_text.strip()
        assert s.expected.stakeholders, f"{s.scenario_id} has no expected stakeholders"
        assert s.expected.commitments, f"{s.scenario_id} has no expected commitments"


def test_corpus_hash_deterministic() -> None:
    scenarios = load_scenarios(BENCH_DIR)
    h1 = corpus_hash(scenarios)
    h2 = corpus_hash(list(reversed(scenarios)))
    assert h1 == h2
    assert len(h1) == 64


def test_load_scenarios_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scenarios(tmp_path / "nope")


# ---------------------------------------------------- per-metric scoring


class _StubIR:
    """Minimal CompilerIR-shaped object for scoring tests."""

    def __init__(self, **kw):
        self.stakeholders = kw.get("stakeholders", [])
        self.commitments = kw.get("commitments", [])
        self.ethical_facts = kw.get("ethical_facts", [])
        self.canonical_form = kw.get("canonical_form")
        self.deme_verdict = kw.get("deme_verdict")


class _StubSH:
    def __init__(self, sid, roles=()):
        self.id = sid
        self.roles = list(roles)


class _StubCommit:
    def __init__(self, holder, beneficiary, type_):
        self.holder = holder
        self.beneficiary = beneficiary
        self.type = type_


class _StubFact:
    def __init__(self, kind, subjects=()):
        class _Kind:
            def __init__(self, v):
                self.value = v

        self.kind = _Kind(kind)
        self.subjects = list(subjects)


class _StubVerdict:
    def __init__(self, verdict, per_party=None):
        self.verdict = verdict
        self.per_party = per_party or {}


def test_stakeholder_recall_full_match() -> None:
    ex = [ExpectedStakeholder(id="alice", roles=["agent"]),
          ExpectedStakeholder(id="bob", roles=["patient"])]
    ir = _StubIR(stakeholders=[_StubSH("alice"), _StubSH("bob")])
    assert score_stakeholder_recall(ex, ir) == 1.0


def test_stakeholder_recall_partial() -> None:
    ex = [ExpectedStakeholder(id="alice"),
          ExpectedStakeholder(id="bob")]
    ir = _StubIR(stakeholders=[_StubSH("alice")])
    assert score_stakeholder_recall(ex, ir) == 0.5


def test_stakeholder_recall_fuzzy_match_handles_underscores() -> None:
    ex = [ExpectedStakeholder(id="hidden_refugees")]
    ir = _StubIR(stakeholders=[_StubSH("hiddenRefugees")])
    assert score_stakeholder_recall(ex, ir) == 1.0


def test_stakeholder_role_f1_exact_match() -> None:
    ex = [ExpectedStakeholder(id="a", roles=["agent", "vow_holder"])]
    ir = _StubIR(stakeholders=[_StubSH("a", roles=["agent", "vow_holder"])])
    assert score_stakeholder_role_f1(ex, ir) == 1.0


def test_stakeholder_role_f1_partial() -> None:
    ex = [ExpectedStakeholder(id="a", roles=["agent", "vow_holder"])]
    ir = _StubIR(stakeholders=[_StubSH("a", roles=["agent"])])
    f1 = score_stakeholder_role_f1(ex, ir)
    assert 0.0 < f1 < 1.0


def test_commitment_f1() -> None:
    ex = [ExpectedCommitment(holder="alice", beneficiary="bob", type="vow")]
    ir = _StubIR(commitments=[_StubCommit("alice", "bob", "vow")])
    assert score_commitment_f1(ex, ir) == 1.0


def test_ethical_fact_kind_recall() -> None:
    ir = _StubIR(ethical_facts=[_StubFact("coercion"), _StubFact("care")])
    assert score_ethical_fact_kind_recall(["coercion", "care"], ir) == 1.0
    assert score_ethical_fact_kind_recall(["coercion"], ir) == 1.0
    assert score_ethical_fact_kind_recall(["legitimacy"], ir) == 0.0


def test_overall_verdict_match() -> None:
    ir = _StubIR(deme_verdict=_StubVerdict("permitted"))
    assert score_overall_verdict("permitted", ir) == 1.0
    assert score_overall_verdict("tragic_conflict_escalate", ir) == 0.0
    assert math.isnan(score_overall_verdict(None, ir))


def test_premature_contraction_only_penalised_when_expected() -> None:
    ir_clean = _StubIR(deme_verdict=_StubVerdict("permitted"))
    ir_review = _StubIR(deme_verdict=_StubVerdict("requires_human_review"))
    assert score_premature_contraction(False, ir_clean) == 0.0
    assert score_premature_contraction(True, ir_review) == 0.0  # good — escalated as expected
    assert score_premature_contraction(True, ir_clean) == 1.0   # bad — collapsed prematurely


# ---------------------------------------------------- aggregate + weights


def test_load_default_weights_sums_to_unity_or_returns_dict() -> None:
    w = load_weights()
    assert isinstance(w, dict)
    assert "stakeholder_recall" in w


def test_weighted_score_combines_metrics() -> None:
    agg = {
        "mean_stakeholder_recall": 1.0,
        "mean_stakeholder_role_f1": 1.0,
        "mean_commitment_f1": 1.0,
        "mean_canonical_form_match": 1.0,
        "mean_ethical_fact_kind_recall": 1.0,
        "mean_per_party_verdict_accuracy": 1.0,
        "mean_overall_verdict_match": 1.0,
        "premature_contraction_rate": 0.0,
    }
    w = load_weights()
    score = weighted_score(agg, w)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_aggregate_score_emits_bench_aggregate() -> None:
    s = ScenarioScore(
        scenario_id="t1", category="c", stakeholder_recall=0.5,
        stakeholder_role_f1=0.5, commitment_f1=0.5,
        canonical_form_match=0.0, ethical_fact_kind_recall=0.5,
        per_party_verdict_accuracy=0.5, overall_verdict_match=0.0,
        premature_contraction=0.0,
    )
    agg = aggregate_score([s], load_weights())
    assert isinstance(agg, BenchAggregate)
    assert agg.n_scenarios == 1
    assert 0.0 < agg.moral_tensor_bench_score < 1.0


# ---------------------------------------------------- end-to-end run_bench


def test_run_bench_on_seed_corpus_produces_report() -> None:
    report = run_bench(BENCH_DIR, extractor="rule")
    assert report.aggregate.n_scenarios == 3
    assert report.aggregate.n_failed_compile == 0
    assert report.compiler_version
    assert len(report.corpus_hash) == 64
    assert report.bench_version.startswith("v0.1")
    # The seed scenarios are *aspirational* gold — the current rule
    # extractor uses generic stakeholder IDs (`self`, `collective_*_seg_*`)
    # so semantic IDs (`speaker`, `gestapo`) don't fuzzy-match. The bench
    # honestly surfaces that gap. We only assert the aggregate is a valid
    # number in [0, 1].
    assert 0.0 <= report.aggregate.mean_stakeholder_recall <= 1.0
    assert 0.0 <= report.aggregate.moral_tensor_bench_score <= 1.0
    # One scenario (medical_confidentiality) does match canonical_form.
    assert report.aggregate.mean_canonical_form_match > 0.0


def test_render_markdown_includes_per_scenario_rows() -> None:
    report = run_bench(BENCH_DIR, extractor="rule")
    md = render_report_markdown(report)
    assert "MoralTensor-Bench" in md
    for s in report.per_scenario:
        assert f"`{s.scenario_id}`" in md
