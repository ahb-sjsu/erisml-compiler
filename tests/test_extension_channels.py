"""Extension-channel wiring: helpers + the audit-hash binding.

Covers the `repair_residue`-style extension channels (purity, loyalty) added to
the MoralVector SSOT: the tensor helpers validate them, and build_decision_proof
must bind them into the tensor_hash (the gap this fix closes — metadata-only
channels were previously invisible to the decision proof).
"""

from __future__ import annotations

import pytest

from erisml_compiler.ir.v3.dimensions import (
    MORAL_EXTENSION_CHANNELS,
    MORAL_VECTOR_CHANNELS,
)
from erisml_compiler.ir.v3.tensor import MoralTensorV3


def test_registry_is_axis_plus_extensions():
    assert MORAL_EXTENSION_CHANNELS == ("purity", "loyalty")
    assert MORAL_VECTOR_CHANNELS[-2:] == ("purity", "loyalty")
    assert len(MORAL_VECTOR_CHANNELS) == 11


def test_set_and_get_extension_channel():
    t = MoralTensorV3.zeros(shape=(9,))
    t.set_extension_channel("purity", 0.4, presence=0.72)
    rec = t.get_extension_channel("purity")
    assert rec["value"] == 0.4
    assert rec["presence"] == 0.72
    assert rec["provenance"]["valence"]["checkpoint_hash"] == "001506fc21518a5e"
    # tensor stays frozen at 9 on the k-axis
    assert t.shape[0] == 9


def test_rejects_unregistered_and_out_of_range():
    t = MoralTensorV3.zeros(shape=(9,))
    with pytest.raises(ValueError):
        t.set_extension_channel("sanctity_not_registered", 0.1)
    with pytest.raises(ValueError):
        t.set_extension_channel("purity", 1.5)
    with pytest.raises(ValueError):
        t.set_extension_channel("loyalty", 0.2, presence=2.0)


def _proof_hash(tensor):
    from erisml_compiler.erisml_backend.v3_phase6 import build_decision_proof

    class _Audit:
        ir_hash = "seed"

    class _IR:
        ethical_facts: list = []
        canonical_form = "test"
        schema_version = "v3"
        stakeholders: list = []
        per_party_verdicts: dict = {}
        fairness_metrics: dict = {}
        deme_verdict = None
        audit = _Audit()

    return build_decision_proof(_IR(), tensor)["moral_vector_summary"]["tensor_hash"]


def test_decision_proof_binds_extension_channels():
    """Two tensors identical except for an extension channel MUST get different
    tensor hashes — otherwise the channel is unauditable."""
    base = MoralTensorV3.zeros(shape=(9,))
    with_purity = MoralTensorV3.zeros(shape=(9,))
    with_purity.set_extension_channel("purity", -0.6, presence=0.72)

    assert _proof_hash(base) != _proof_hash(with_purity)

    # And a change to the extension VALUE changes the hash (not just presence).
    a = MoralTensorV3.zeros(shape=(9,))
    a.set_extension_channel("loyalty", 0.3)
    b = MoralTensorV3.zeros(shape=(9,))
    b.set_extension_channel("loyalty", -0.3)
    assert _proof_hash(a) != _proof_hash(b)
