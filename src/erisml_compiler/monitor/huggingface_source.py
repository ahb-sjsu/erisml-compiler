"""HuggingFace-backed activation source.

Loads a causal LM (default Qwen/Qwen2.5-7B-Instruct) and registers forward
hooks on a chosen subset of `model.model.layers` (or the equivalent
`transformer.h.*` path on older architectures). On each `.capture(text)`
call, runs a forward pass and returns the layer outputs.

Hook registration is idempotent — repeated `capture()` calls reuse the
same hooks. `close()` removes them.

Memory footprint: hooks store full (T, D) hidden states for the selected
layers per forward pass. For Qwen2.5-7B (D=3584) at T=128 tokens, a
single capture is ~1.8 MB per layer. We default to capturing every 4th
layer plus the final layer to keep this bounded.
"""

from __future__ import annotations

from typing import Sequence

from erisml_compiler.monitor.base import (
    ActivationCapture,
    ActivationSource,
    LayerActivation,
)

# Per-architecture candidate paths to the transformer block list. We try
# in order; first one that resolves wins. The duplicate (`model.layers`
# vs `model.model.layers`) is because `AutoModel` returns the base model
# directly for Qwen/LLaMA-family architectures, whereas
# `AutoModelForCausalLM` wraps it under `.model`. We accept both.
_LAYER_PATH_BY_ARCH = {
    "qwen2": [("model", "layers"), ("layers",)],
    "qwen3": [("model", "layers"), ("layers",)],
    "llama": [("model", "layers"), ("layers",)],
    "mistral": [("model", "layers"), ("layers",)],
    "gpt2": [("transformer", "h"), ("h",)],
    "bert": [("encoder", "layer"), ("bert", "encoder", "layer")],
    "roberta": [("encoder", "layer"), ("roberta", "encoder", "layer")],
}


def _resolve_layers(model, model_type: str):
    """Return the layer ModuleList for the given model.

    Tries the candidate paths for the architecture first, then falls
    back to a recursive search if none resolve.
    """
    candidates = _LAYER_PATH_BY_ARCH.get(model_type, [])
    for path in candidates:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok:
            return node

    # Fallback: scan for the first ModuleList containing modules with a
    # name matching "*Layer*" or "*Block*". This is best-effort.
    import torch

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            first = mod[0].__class__.__name__.lower()
            if "layer" in first or "block" in first:
                return mod
    raise ValueError(
        f"Could not locate transformer layers for model_type={model_type!r}; "
        f"add a mapping to _LAYER_PATH_BY_ARCH."
    )


def _default_layer_subset(total_layers: int) -> list[int]:
    """Every 4th layer plus the final layer. Bounded and deterministic."""
    layers = list(range(0, total_layers, 4))
    if (total_layers - 1) not in layers:
        layers.append(total_layers - 1)
    return layers


class HuggingFaceActivationSource(ActivationSource):
    """HF transformers-backed source.

    Args:
        model_id: HF repo id (e.g. "Qwen/Qwen2.5-7B-Instruct"). Must be
            pre-downloaded; this source does NOT auto-fetch from the hub
            in order to keep the trust boundary clear (probe poisoning).
        device: "cuda", "cuda:1", "cpu", etc.
        dtype: torch dtype string ("bfloat16", "float16", "float32").
        layers: which layer indices to capture. If None, every 4th + final.
        max_tokens: truncate inputs above this length.
        trust_remote_code: passed through; default False.
    """

    name = "huggingface"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        layers: Sequence[int] | None = None,
        max_tokens: int = 512,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens

        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self._model.eval()

        cfg = self._model.config
        model_type = getattr(cfg, "model_type", "").lower()
        self._layer_modules = _resolve_layers(self._model, model_type)
        total_layers = len(self._layer_modules)
        self._total_layers = total_layers

        if layers is None:
            self._selected = _default_layer_subset(total_layers)
        else:
            for li in layers:
                if li < 0 or li >= total_layers:
                    raise ValueError(f"layer {li} outside [0, {total_layers}) for {model_id}")
            self._selected = list(layers)

        # Hidden dim — fields differ slightly across architectures.
        hidden_dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(cfg, "d_model", None)
        )
        if hidden_dim is None:
            raise ValueError(f"Could not infer hidden_dim for {model_id}")
        self.hidden_dim = int(hidden_dim)

        self._captured: dict[int, "torch.Tensor"] = {}
        self._hook_handles: list = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        torch = self._torch

        def make_hook(idx: int):
            def hook(module, inputs, output):
                # Some layers return (hidden, present_kv); take the hidden.
                hs = output[0] if isinstance(output, tuple) else output
                # Squeeze batch dim — we forward one sequence at a time.
                self._captured[idx] = hs.detach().squeeze(0).to("cpu", torch.float32)

            return hook

        for idx in self._selected:
            handle = self._layer_modules[idx].register_forward_hook(make_hook(idx))
            self._hook_handles.append(handle)

    def available_layers(self) -> list[int]:
        return list(range(self._total_layers))

    def capture(self, text: str, *, layers: Sequence[int] | None = None) -> ActivationCapture:
        torch = self._torch
        if layers is not None and set(layers) - set(self._selected):
            raise ValueError(
                f"Requested layers {layers} not in pre-registered hook set "
                f"{self._selected}; reinstantiate with `layers=` to change."
            )

        target_layers = list(layers) if layers is not None else self._selected
        self._captured.clear()

        assert self._tokenizer is not None  # set in __init__; only cleared by close()
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        ).to(self.device)
        with torch.no_grad():
            self._model(**enc, use_cache=False)

        layer_acts: list[LayerActivation] = []
        for idx in target_layers:
            hidden = self._captured.get(idx)
            if hidden is None:
                raise RuntimeError(
                    f"Layer {idx} hook did not fire — model_type may not match "
                    f"the resolved layer path."
                )
            pooled = hidden.mean(dim=0)
            layer_acts.append(
                LayerActivation(
                    layer_index=idx,
                    layer_name=f"{self.model_id}.layer.{idx}",
                    hidden=hidden,
                    pooled=pooled,
                )
            )

        return ActivationCapture(
            text=text,
            source_name=self.name,
            model_id=self.model_id,
            hidden_dim=int(self.hidden_dim),
            layers=layer_acts,
            metadata={
                "device": self.device,
                "tokenizer": self._tokenizer.__class__.__name__,
                "n_input_tokens": int(enc["input_ids"].shape[1]),
            },
        )

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        # Drop the model + tokenizer references to free GPU memory.
        del self._model
        del self._tokenizer
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
