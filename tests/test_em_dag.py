"""EM-DAG layer tests: cycle detection, topological order, profile loading."""

from pathlib import Path

import pytest

from erisml_compiler.em_dag import EMDAG, load_profile
from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.ir.schemas import (
    CompilerIR,
    DimensionScore,
    Document,
    EMOutput,
)

PROFILE_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "erisml_compiler"
    / "em_dag"
    / "profiles"
    / "default.yaml"
)


def test_default_profile_loads_and_validates():
    dag = load_profile(PROFILE_PATH)
    assert dag.name == "default_em_dag_v0.1"
    assert len(dag) == 10
    # Topological order: legitimacy must come before autonomy.
    order = dag.topological_order
    assert order.index("legitimacy") < order.index("autonomy")
    assert order.index("legitimacy") < order.index("fidelity")
    assert order.index("harm") < order.index("externality")
    assert order.index("harm") < order.index("care")
    # Repair is the latest (depends on harm, externality, fidelity).
    for upstream in ("harm", "externality", "fidelity"):
        assert order.index(upstream) < order.index("repair")


def test_dag_rejects_cycles():
    """Construct a synthetic cycle and ensure detection."""

    class A(EthicalModule):
        name, dimension, dependencies = "a", "physical_harm", ("b",)

        def evaluate(self, ir, upstream):
            return EMOutput(
                module_name="a",
                score=DimensionScore(value=0.0, direction="neutral"),
            )

    class B(EthicalModule):
        name, dimension, dependencies = "b", "rights_respect", ("a",)

        def evaluate(self, ir, upstream):
            return EMOutput(
                module_name="b",
                score=DimensionScore(value=0.0, direction="neutral"),
            )

    with pytest.raises(ValueError, match="cycle"):
        EMDAG([A(), B()])


def test_dag_rejects_missing_dependency():
    class C(EthicalModule):
        name, dimension, dependencies = "c", "physical_harm", ("nonexistent",)

        def evaluate(self, ir, upstream):
            return EMOutput(
                module_name="c",
                score=DimensionScore(value=0.0, direction="neutral"),
            )

    with pytest.raises(ValueError, match="dependency"):
        EMDAG([C()])


def test_evaluate_empty_ir_returns_neutral_scores():
    dag = load_profile(PROFILE_PATH)
    ir = CompilerIR(document=Document(doc_id="empty"))
    outputs = dag.evaluate(ir)
    assert len(outputs) == 10
    for output in outputs.values():
        # Empty IR -> no facts -> neutral 0.0 scores.
        assert output.score.value == 0.0
