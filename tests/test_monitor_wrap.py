"""HuggingFaceActivationSource can wrap an already-loaded model.

A caller that also generates (the twin's I-EIP replay captures states and then the
classification from the same text) can then hold one copy of the weights. The capture must
be the same whichever way the source got its model; a tiny random Qwen2 saved to disk
checks that without any download.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from erisml_compiler.monitor.huggingface_source import HuggingFaceActivationSource


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        AutoModelForCausalLM,
        PreTrainedTokenizerFast,
        Qwen2Config,
    )

    d = tmp_path_factory.mktemp("tiny-qwen2")
    words = ["[UNK]", "the", "robot", "sees", "margaret", "fall", "smoke", "dog"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(words),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(d)
    return str(d)


def test_a_wrapped_model_captures_what_a_loaded_one_does(tiny):
    from transformers import AutoModelForCausalLM

    text = "the robot sees margaret fall"
    loaded = HuggingFaceActivationSource(tiny, device="cpu", dtype="float32", layers=[1, 3])
    lm = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32).eval()
    wrapped = HuggingFaceActivationSource(
        tiny, device="cpu", layers=[1, 3], model=lm.model, tokenizer=loaded._tokenizer
    )
    a, b = loaded.capture(text), wrapped.capture(text)
    assert a.layer_indices() == b.layer_indices() == [1, 3]
    for la, lb in zip(a.layers, b.layers):
        assert torch.equal(la.hidden, lb.hidden)


def test_closing_a_wrapper_leaves_the_model_usable_and_hook_free(tiny):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny)
    lm = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32).eval()
    src = HuggingFaceActivationSource(tiny, device="cpu", layers=[0], model=lm.model, tokenizer=tok)
    src.capture("smoke dog")
    src.close()
    assert all(not m._forward_hooks for m in lm.model.layers)
    ids = tok("the dog", return_tensors="pt")
    with torch.no_grad():
        out = lm.generate(**ids, max_new_tokens=2, do_sample=False)
    assert out.shape[1] == ids["input_ids"].shape[1] + 2
