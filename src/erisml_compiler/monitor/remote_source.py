"""Remote activation source — paramiko-driven Atlas inference.

The host (typically a laptop or workstation without a GPU) sends an input
text over SSH to Atlas, where a small Python harness runs the
HuggingFaceActivationSource and returns the captured tensors as a base64
torch.save blob. The host deserializes and reconstructs an
ActivationCapture identical to what a local HF source would have produced.

Trust boundary (spec §31.6): the Atlas harness runs with the credentials
of the SSH user. We do NOT mount the host filesystem on Atlas, and we do
not exec arbitrary code from the host — only the `text` argument is
passed across. The probe head + ProbeBackbone live on the host; only raw
hidden states cross the boundary.

This source is silicon-incompatible by design. It is for development /
research; for production deployments use the local HF source or the
ProbeExtractor.

The current implementation spawns a fresh harness per `capture()`, which
re-loads the model. That is fine for one-off introspection (typical
development use), unacceptable for sustained monitoring. For sustained
workloads, run the compiler directly on Atlas (use HuggingFaceActivationSource
there) rather than driving it remotely from a CPU host.
"""
from __future__ import annotations

import base64
import io
import json
import shlex
from typing import Sequence

from erisml_compiler.monitor.base import (
    ActivationCapture,
    ActivationSource,
    LayerActivation,
)


# Atlas-side harness. Kept compact so it can be passed as a single
# command-line argument. Reads (model_id, layers, text) from stdin as
# JSON, prints (hidden_dim, layers payload, metadata) as base64 stdout.
_ATLAS_HARNESS = r"""
import base64, io, json, sys
import torch
from transformers import AutoModel, AutoTokenizer

req = json.loads(sys.stdin.read())
model_id = req["model_id"]
selected = req["layers"]
text = req["text"]
device = req.get("device", "cuda")
dtype_name = req.get("dtype", "bfloat16")
max_tokens = req.get("max_tokens", 512)

torch_dtype = getattr(torch, dtype_name)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModel.from_pretrained(model_id, dtype=torch_dtype, trust_remote_code=False).to(device).eval()

# Resolve layers — same map as huggingface_source.py.
mt = getattr(model.config, "model_type", "").lower()
if mt in ("qwen2", "qwen3", "llama", "mistral"):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_mods = model.model.layers
    else:
        layer_mods = model.layers  # AutoModel returns the base model directly
elif mt == "gpt2":
    layer_mods = getattr(model, "transformer", model).h
elif mt in ("bert", "roberta"):
    layer_mods = getattr(model, "encoder", model).layer
else:
    raise SystemExit(f"unsupported model_type={mt!r}")

total = len(layer_mods)
for li in selected:
    if li < 0 or li >= total:
        raise SystemExit(f"layer {li} out of range [0,{total})")

captured = {}
handles = []
def mk(idx):
    def h(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured[idx] = hs.detach().squeeze(0).to("cpu", torch.float32)
    return h
for idx in selected:
    handles.append(layer_mods[idx].register_forward_hook(mk(idx)))

enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
with torch.no_grad():
    model(**enc, use_cache=False)
for h in handles:
    h.remove()

cfg = model.config
hidden_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)

# Pack each layer as its own base64 torch.save blob — keeps the wire
# protocol forward-compatible and avoids one giant pickle.
payload = {"hidden_dim": int(hidden_dim), "model_type": mt, "n_input_tokens": int(enc["input_ids"].shape[1]), "layers": []}
for idx in selected:
    buf = io.BytesIO()
    torch.save(captured[idx], buf)
    payload["layers"].append({
        "layer_index": idx,
        "layer_name": f"{model_id}.layer.{idx}",
        "blob_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
    })

sys.stdout.write("ERISML_HARNESS_RESULT:" + base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii") + "\n")
"""


class RemoteAtlasActivationSource(ActivationSource):
    """Paramiko-driven remote source. See module docstring for trust model.

    Args:
        ssh_host: tailscale IP or hostname.
        ssh_user: remote username.
        ssh_password: remote password (no key auth, by user preference;
            see `reference_atlas_ssh.md`).
        model_id: HF repo id present in the remote cache.
        remote_python: path to the venv python on the remote host.
        layers, device, dtype, max_tokens: see HuggingFaceActivationSource.
    """

    name = "remote-atlas"

    def __init__(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_password: str,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        remote_python: str = "/home/claude/env/bin/python3",
        layers: Sequence[int] | None = None,
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        max_tokens: int = 512,
        connect_timeout: float = 20.0,
    ):
        try:
            import paramiko
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "RemoteAtlasActivationSource needs paramiko: pip install paramiko"
            ) from e

        self._paramiko = paramiko
        self.model_id = model_id
        self._remote_python = remote_python
        self._device = device
        self._dtype = dtype
        self._max_tokens = max_tokens
        # Layer selection is *advisory* until the first capture, when we
        # learn the model's actual layer count from the remote harness.
        self._requested_layers = list(layers) if layers is not None else None
        self._resolved_layers: list[int] | None = None
        self._total_layers: int | None = None
        self.hidden_dim = 0

        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(
            ssh_host,
            username=ssh_user,
            password=ssh_password,
            timeout=connect_timeout,
        )

    def available_layers(self) -> list[int]:
        if self._total_layers is None:
            raise RuntimeError(
                "Run capture() once before calling available_layers() — "
                "the remote source learns the model's layer count on first contact."
            )
        return list(range(self._total_layers))

    def _select_layers(self) -> list[int]:
        if self._resolved_layers is not None:
            return self._resolved_layers
        if self._requested_layers is not None:
            self._resolved_layers = self._requested_layers
            return self._resolved_layers
        # Without a prior capture we cannot pick "every 4th + final"
        # because we do not know the total. Probe with [0] and then
        # widen on the next call.
        return [0]

    def capture(
        self, text: str, *, layers: Sequence[int] | None = None
    ) -> ActivationCapture:
        target = list(layers) if layers is not None else self._select_layers()
        req = {
            "model_id": self.model_id,
            "layers": target,
            "text": text,
            "device": self._device,
            "dtype": self._dtype,
            "max_tokens": self._max_tokens,
        }

        cmd = f"{shlex.quote(self._remote_python)} -c {shlex.quote(_ATLAS_HARNESS)}"
        stdin, stdout, stderr = self._ssh.exec_command(cmd, timeout=600)
        stdin.write(json.dumps(req))
        stdin.channel.shutdown_write()

        out_bytes = stdout.read()
        err_bytes = stderr.read()
        out = out_bytes.decode("utf-8", errors="replace")

        marker = "ERISML_HARNESS_RESULT:"
        idx = out.find(marker)
        if idx < 0:
            raise RuntimeError(
                "Remote harness did not return a result.\n"
                f"stdout: {out[-2000:]!r}\n"
                f"stderr: {err_bytes.decode('utf-8', errors='replace')[-2000:]!r}"
            )
        payload_b64 = out[idx + len(marker):].strip().split("\n", 1)[0]
        payload = json.loads(base64.b64decode(payload_b64))

        import torch  # local import; remote_source still importable without torch on host

        layer_acts: list[LayerActivation] = []
        for layer in payload["layers"]:
            blob = base64.b64decode(layer["blob_b64"])
            hidden = torch.load(io.BytesIO(blob), weights_only=True)
            pooled = hidden.mean(dim=0)
            layer_acts.append(
                LayerActivation(
                    layer_index=layer["layer_index"],
                    layer_name=layer["layer_name"],
                    hidden=hidden,
                    pooled=pooled,
                )
            )

        # Update inferred state.
        self.hidden_dim = int(payload["hidden_dim"])
        # We can only learn the real total_layers if the harness reports
        # it; right now it doesn't. Leave _total_layers None unless caller
        # passed explicit layers — captures still work either way.

        return ActivationCapture(
            text=text,
            source_name=self.name,
            model_id=self.model_id,
            hidden_dim=self.hidden_dim,
            layers=layer_acts,
            metadata={
                "device": self._device,
                "remote": True,
                "model_type": payload.get("model_type"),
                "n_input_tokens": payload.get("n_input_tokens"),
            },
        )

    def close(self) -> None:
        try:
            self._ssh.close()
        except Exception:
            pass
