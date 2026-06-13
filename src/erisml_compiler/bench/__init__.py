"""MoralTensor-Bench harness.

A named benchmark that evaluates whether the compiler produces the
*right* structural output (stakeholders, commitments, verdicts,
canonical forms) on a hand-curated scenario corpus. See
`docs/plans/release-planning-03-moraltensor-bench.md`.

v0.1 ships with 3 seed scenarios (recast bundled examples). Future
curation expands toward the 80-scenario target documented in the
design note.
"""
from erisml_compiler.bench.runner import BenchReport, BenchRun, run_bench
from erisml_compiler.bench.schema import (
    ExpectedCommitment,
    ExpectedScenario,
    ExpectedStakeholder,
    ScenarioGold,
    ScenarioScore,
)
from erisml_compiler.bench.scoring import (
    aggregate_score,
    score_scenario,
    weighted_score,
)

__all__ = [
    "BenchReport",
    "BenchRun",
    "ExpectedCommitment",
    "ExpectedScenario",
    "ExpectedStakeholder",
    "ScenarioGold",
    "ScenarioScore",
    "aggregate_score",
    "run_bench",
    "score_scenario",
    "weighted_score",
]
