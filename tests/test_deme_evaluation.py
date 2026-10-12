"""The compiler's evaluation is erisml-lib's DEME (0.10): the EM-DAG is retired.

The IR's moral vector and timeline are DEME's nine-dimension vector; a fitted ethos profile
becomes DEME's dimension weights; conflicts are read on DEME's vector; the I-EIP comparison
puts the probes' ten channels in DEME's nine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from erisml.ethics.moral_vector import MoralVector as DemeMoralVector

from erisml_compiler.delta import compare_morals
from erisml_compiler.delta.deme_space import DEME_DIMENSIONS, activation_lens, text_lens
from erisml_compiler.erisml_backend.deme_bridge import DEMEBridge
from erisml_compiler.evaluation.conflict_detector import detect_conflicts, signed
from erisml_compiler.evaluation.ethos import (
    PROFILES_DIR,
    ethos_dimension_weights,
    resolve_ethos_profile,
)
from erisml_compiler.ir.schemas import (
    MORAL_DIMENSIONS,
    CompilerIR,
    DimensionScore,
    Document,
    MoralVector,
)
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_ethos_channel_weights_become_deme_dimension_weights():
    w = ethos_dimension_weights({"harm": 2.0, "care": 1.5, "fidelity": 1.0, "repair": 9.0})
    assert w.physical_harm == 2.0
    assert w.virtue_care == 1.5 + 0.5  # care, and half of fidelity
    assert w.legitimacy_trust == 0.5  # the other half of fidelity
    assert w.privacy_protection == 1.0  # no channel reaches it: the library default


def test_the_bundled_ethos_profiles_load_with_their_hash():
    weights, name, sha = resolve_ethos_profile(PROFILES_DIR / "dear_abby_socialchem_v0.1.yaml")
    assert name == "dear_abby_socialchem_v0.1" and len(sha) == 64
    assert weights.physical_harm > 1.0  # care/harm is the profile's strongest channel


def test_a_compile_carries_deme_s_vector_and_timeline():
    ir = compile_document(
        EXAMPLES / "nazi_attic.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="mock"),
    )
    assert isinstance(ir.moral_vectors[0], DemeMoralVector)
    assert ir.timeline and all(isinstance(t.vector, DemeMoralVector) for t in ir.timeline)
    assert ir.audit.em_profile == "erisml_compiler_default"  # the DEME profile now
    again = CompilerIR.model_validate_json(ir.model_dump_json())
    assert again.moral_vectors[0] == ir.moral_vectors[0]


def test_an_ethos_profile_reaches_deme_and_the_audit():
    path = PROFILES_DIR / "dear_abby_socialchem_v0.1.yaml"
    ir = compile_document(
        EXAMPLES / "nazi_attic.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="mock", ethos_profile=path),
    )
    assert ir.audit.ethos_profile == "dear_abby_socialchem_v0.1"
    assert ir.projections["consequentialist_distributive"]["metadata"]["ethos_applied"] is True


def test_the_retired_em_profile_option_is_refused():
    with pytest.raises(ValueError, match="retired with the EM-DAG"):
        compile_document(
            EXAMPLES / "nazi_attic.txt",
            CompileOptions(tier=CompilerTier.RULES, extractor="mock", em_profile="x.yaml"),
        )


def test_without_facts_deme_has_no_vector():
    verdict, vector = DEMEBridge().judge(CompilerIR(document=Document(doc_id="empty")))
    assert verdict.verdict == "indeterminate" and vector is None


def test_conflicts_are_read_on_deme_s_signed_scale():
    v = DemeMoralVector(physical_harm=0.95, autonomy_respect=1.0)
    s = signed(v)
    assert s["physical_harm"] == pytest.approx(-0.9) and s["autonomy_respect"] == 1.0
    conflicts = detect_conflicts(CompilerIR(document=Document(doc_id="d")), v)
    assert any(c.dimensions == ["rights_respect", "physical_harm"] for c in conflicts)
    assert all(c.requires_escalation for c in conflicts if "physical_harm" in c.dimensions)


def test_the_probes_ten_channels_are_compared_in_deme_s_nine():
    probes = MoralVector(
        **{d: DimensionScore(value=0.8, direction="positive") for d in MORAL_DIMENSIONS}
    )
    lens = activation_lens(probes)
    assert lens.virtue_care.value == 1.0  # care 0.8 and half of fidelity 0.4, clamped
    assert lens.privacy_protection.uncertainty == 1.0  # no channel reaches it
    agree = compare_morals(
        text_lens(DemeMoralVector(physical_harm=0.1)), lens, space=DEME_DIMENSIONS
    )
    assert len(agree.per_dimension) == 9
