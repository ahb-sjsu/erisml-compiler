"""Phase 4 paper experiment — activation lens on Qwen2.5-7B-Instruct, run on Atlas.

What this does, from the host (Windows laptop):

  1. Connect to Atlas via Tailscale + paramiko.
  2. Send a self-contained harness over SSH. The harness:
       a) Loads Qwen2.5-7B-Instruct on Atlas GPU 1 in bfloat16.
       b) Registers forward hooks on every-4th + final transformer layer.
       c) For each of the 3 bundled scenarios:
            - capture per-layer (T,D) activations under the original text,
            - capture per-layer activations under each surface-form rewrite,
            - emit a JSON record per scenario with pooled (D,) vectors only
              (full (T,D) tensors are too big to ship back; pooled is what
              the probe consumes anyway).
  3. Host side: reconstruct ActivationCaptures, run IEIPMonitor with random
     probes (calibration is a separate paper), run compare_morals against
     the IR's first MoralVector, run check_equivariance, run
     detect_failure_modes.
  4. Emit a per-scenario summary table + a combined JSON report.

Interpretation: probes are random, so the *content* of the moral vectors
is structural noise. The thing this experiment demonstrates is that the
full Phase 4 pipeline runs end-to-end on a real 7B model: layer
resolution, hook registration, activation transport, audit-chain hashing,
equivariance computation, and failure-mode detection. Calibrated probes
+ a moral-language corpus are a separate paper.

Usage:
    python scripts/experiments/atlas_phase4_experiment.py --out experiments/phase4/

Requires that you've run `eris-compile compile` on the three bundled
scenarios first (the experiment reads the resulting IR JSONs to anchor
the text-lens MoralVectors).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import paramiko

ATLAS_HOST = "100.68.134.21"
USER = "claude"
PASSWORD = "roZes9090!~"
VENV_PY = "/home/claude/env/bin/python3"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SCENARIOS = [
    {
        "name": "nazi_attic",
        "text_path": "examples/nazi_attic.txt",
        "ir_path": "out/nazi_attic.ir.json",
        "rewrites": [
            {"name": "identity", "fn": "s"},
            {"name": "lowercase", "fn": "s.lower()"},
            {"name": "trim_punct", "fn": "s.rstrip('.').strip()"},
        ],
    },
    {
        "name": "medical_confidentiality",
        "text_path": "examples/medical_confidentiality.txt",
        "ir_path": "out/medical_confidentiality.ir.json",
        "rewrites": [
            {"name": "identity", "fn": "s"},
            {"name": "lowercase", "fn": "s.lower()"},
        ],
    },
    {
        "name": "whistleblower",
        "text_path": "examples/whistleblower.txt",
        "ir_path": "out/whistleblower.ir.json",
        "rewrites": [
            {"name": "identity", "fn": "s"},
            {"name": "lowercase", "fn": "s.lower()"},
        ],
    },
]


# Atlas-side harness. Loads the model ONCE, processes all scenarios, returns
# a single JSON payload with pooled activations for each (scenario, rewrite,
# layer) triple.
_ATLAS_HARNESS = r"""
import base64, io, json, sys
import torch
from transformers import AutoModel, AutoTokenizer

req = json.loads(sys.stdin.read())
model_id = req["model_id"]
device = req["device"]
max_tokens = req["max_tokens"]
items = req["items"]  # list of {scenario, text, rewrites}

torch_dtype = torch.bfloat16
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
print("HARNESS: tokenizer loaded", file=sys.stderr, flush=True)

model = AutoModel.from_pretrained(model_id, dtype=torch_dtype, trust_remote_code=False).to(device).eval()
print("HARNESS: model loaded onto", device, file=sys.stderr, flush=True)

# AutoModel returns Qwen2Model (no LM head), so layers are at .layers.
# AutoModelForCausalLM wraps under .model.layers. Try both.
if hasattr(model, "model") and hasattr(model.model, "layers"):
    layer_mods = model.model.layers
elif hasattr(model, "layers"):
    layer_mods = model.layers
else:
    raise SystemExit("could not resolve transformer.layers")
total_layers = len(layer_mods)
selected = list(range(0, total_layers, 4))
if (total_layers - 1) not in selected:
    selected.append(total_layers - 1)

hidden_dim = model.config.hidden_size

# Hook setup
captured = {}
handles = []
def mk(idx):
    def h(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        # Pool to (D,) immediately to keep memory bounded.
        captured[idx] = hs.detach().squeeze(0).mean(dim=0).to("cpu", torch.float32)
    return h
for idx in selected:
    handles.append(layer_mods[idx].register_forward_hook(mk(idx)))

print(f"HARNESS: hooked {len(selected)} layers out of {total_layers}", file=sys.stderr, flush=True)

results = []
for item in items:
    scenario = item["scenario"]
    rewrite_outputs = []
    for rw in item["rewrites"]:
        captured.clear()
        text_rewritten = rw["text"]
        enc = tok(text_rewritten, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        with torch.no_grad():
            model(**enc, use_cache=False)
        layer_vecs = {}
        for idx in selected:
            v = captured[idx]
            buf = io.BytesIO()
            torch.save(v, buf)
            layer_vecs[str(idx)] = base64.b64encode(buf.getvalue()).decode("ascii")
        rewrite_outputs.append({
            "name": rw["name"],
            "text_preview": text_rewritten[:120],
            "n_tokens": int(enc["input_ids"].shape[1]),
            "layer_vecs_b64": layer_vecs,
        })
        print(f"HARNESS: {scenario}/{rw['name']} done ({enc['input_ids'].shape[1]} tokens)", file=sys.stderr, flush=True)
    results.append({"scenario": scenario, "rewrites": rewrite_outputs})

for h in handles:
    h.remove()
del model
torch.cuda.empty_cache()
print("HARNESS: done, model dropped", file=sys.stderr, flush=True)

payload = {
    "model_id": model_id,
    "hidden_dim": int(hidden_dim),
    "total_layers": total_layers,
    "selected_layers": selected,
    "results": results,
}
sys.stdout.write("ERISML_EXP_RESULT:" + base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii") + "\n")
"""


def run_atlas(req: dict) -> dict:
    print(f"[host] connecting to {ATLAS_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ATLAS_HOST, username=USER, password=PASSWORD, timeout=30)
    print("[host] connected. dispatching harness...")

    import shlex
    cmd = f"{shlex.quote(VENV_PY)} -c {shlex.quote(_ATLAS_HARNESS)}"
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=1800)
    stdin.write(json.dumps(req))
    stdin.channel.shutdown_write()

    # Stream stderr live to the host console so the user sees model-load progress.
    err_chan = stderr.channel
    out_chan = stdout.channel
    out_chunks = []
    err_chunks = []
    while True:
        if out_chan.recv_ready():
            out_chunks.append(out_chan.recv(65536))
        if err_chan.recv_stderr_ready():
            data = err_chan.recv_stderr(65536)
            err_chunks.append(data)
            sys.stderr.write(data.decode("utf-8", errors="replace"))
            sys.stderr.flush()
        if out_chan.exit_status_ready():
            # Drain remaining
            while out_chan.recv_ready():
                out_chunks.append(out_chan.recv(65536))
            while err_chan.recv_stderr_ready():
                data = err_chan.recv_stderr(65536)
                err_chunks.append(data)
                sys.stderr.write(data.decode("utf-8", errors="replace"))
            break

    out_bytes = b"".join(out_chunks)
    out = out_bytes.decode("utf-8", errors="replace")
    ssh.close()

    marker = "ERISML_EXP_RESULT:"
    idx = out.find(marker)
    if idx < 0:
        raise RuntimeError(f"Harness did not return a result.\nstdout:\n{out[-2000:]!r}")
    payload_b64 = out[idx + len(marker):].strip().split("\n", 1)[0]
    return json.loads(base64.b64decode(payload_b64))


def decode_layer_vec(b64: str):
    import torch
    blob = base64.b64decode(b64)
    return torch.load(io.BytesIO(blob), weights_only=True)


def host_analyze(remote_payload: dict, scenarios: list[dict], out_dir: Path) -> dict:
    """Reconstruct ActivationCaptures, run IEIPMonitor + Delta + equivariance."""
    import torch  # noqa
    from erisml_compiler.delta import compare_morals, detect_failure_modes
    from erisml_compiler.delta.equivariance import (
        LayerEquivarianceResult,
        EquivarianceReport,
        _cosine_sim,
        _l2,
    )
    from erisml_compiler.ir.schemas import MoralVector
    from erisml_compiler.monitor.activation_probe import ActivationProbe
    from erisml_compiler.monitor.base import ActivationCapture, LayerActivation
    from erisml_compiler.monitor.ieip_monitor import IEIPMonitor, MonitorTrace

    hidden_dim = remote_payload["hidden_dim"]
    selected_layers = remote_payload["selected_layers"]
    model_id = remote_payload["model_id"]

    # One probe per layer, deterministic seeds so re-runs reproduce.
    probes = {idx: ActivationProbe(hidden_dim=hidden_dim, seed=idx) for idx in selected_layers}

    per_scenario_summary: list[dict] = []

    for scenario_payload, scenario_cfg in zip(remote_payload["results"], scenarios):
        scenario_name = scenario_payload["scenario"]
        # Build captures keyed by rewrite name.
        captures: dict[str, ActivationCapture] = {}
        for rw in scenario_payload["rewrites"]:
            layer_acts: list[LayerActivation] = []
            for idx in selected_layers:
                pooled = decode_layer_vec(rw["layer_vecs_b64"][str(idx)])
                # We pool on the device already, so .hidden == .pooled here.
                # That's fine — the probe consumes pooled. For real
                # equivariance against (T,D) we'd want the full tensor.
                layer_acts.append(
                    LayerActivation(
                        layer_index=idx,
                        layer_name=f"{model_id}.layer.{idx}",
                        hidden=pooled.unsqueeze(0),  # (1, D) — token-pooled already
                        pooled=pooled,
                    )
                )
            captures[rw["name"]] = ActivationCapture(
                text=rw["text_preview"],
                source_name="remote-atlas",
                model_id=model_id,
                hidden_dim=hidden_dim,
                layers=layer_acts,
                metadata={"n_tokens": rw["n_tokens"]},
            )

        # IEIPMonitor on the identity capture.
        identity = captures["identity"]
        monitor = IEIPMonitor.__new__(IEIPMonitor)
        monitor.source = None  # type: ignore[assignment]
        monitor._probes_in = probes
        monitor._seed = 0
        monitor._probes = {}
        trace = monitor.monitor_capture(identity)

        # Load text-lens MoralVector from the pre-compiled IR.
        ir_path = Path(scenario_cfg["ir_path"])
        if not ir_path.exists():
            raise FileNotFoundError(
                f"IR not found: {ir_path}. Run `eris-compile compile {scenario_cfg['text_path']}` first."
            )
        ir_dict = json.loads(ir_path.read_text(encoding="utf-8"))
        text_mv_dict = ir_dict.get("global_moral_vector") or ir_dict["moral_vectors"][0]
        text_mv = MoralVector.model_validate(text_mv_dict)

        delta = compare_morals(text_mv, trace.aggregated)

        # Hand-built equivariance report from the rewrites.
        per_layer_per_rewrite = []
        failed_set: set[int] = set()
        base_probes = {
            la.layer_index: (la.pooled, probes[la.layer_index].probe_layer(la).logits)
            for la in identity.layers
        }
        for rw_name, cap in captures.items():
            if rw_name == "identity":
                continue
            for la in cap.layers:
                base_pooled, base_logits = base_probes[la.layer_index]
                pr = probes[la.layer_index].probe_layer(la)
                pc = _cosine_sim(base_pooled, la.pooled)
                pl = _l2(base_pooled, la.pooled)
                lc = _cosine_sim(base_logits, pr.logits)
                ll = _l2(base_logits, pr.logits)
                passed = pc >= 0.95 and lc >= 0.9
                if not passed:
                    failed_set.add(la.layer_index)
                per_layer_per_rewrite.append(
                    LayerEquivarianceResult(
                        layer_index=la.layer_index,
                        rewrite_name=rw_name,
                        pooled_cosine_sim=pc,
                        pooled_l2=pl,
                        probe_logits_cosine_sim=lc,
                        probe_logits_l2=ll,
                        passed=passed,
                    )
                )
        eq = EquivarianceReport(
            text=identity.text,
            rewrites_applied=[k for k in captures if k != "identity"],
            layer_indices=sorted(probes.keys()),
            per_layer_per_rewrite=per_layer_per_rewrite,
            failed_layers=sorted(failed_set),
            config={"pooled_cosine_threshold": 0.95, "probe_cosine_threshold": 0.9},
        )

        fm_report = detect_failure_modes(delta=delta, trace=trace, equivariance=eq)

        per_scenario_summary.append({
            "scenario": scenario_name,
            "trace_hash": trace.trace_hash(),
            "n_layers_probed": len(trace.per_layer),
            "activation_norms": trace.activation_norms,
            "delta_divergence": delta.divergence,
            "delta_direction_breaks": delta.direction_break_count,
            "delta_flag_for_review": delta.flag_for_review,
            "equivariance_failed_layers": eq.failed_layers,
            "failure_modes_fired": [m.value for m in fm_report.fired],
            "requires_human_review": fm_report.requires_human_review,
        })

        scenario_out = out_dir / f"{scenario_name}.report.json"
        scenario_out.write_text(
            json.dumps(
                {
                    "summary": per_scenario_summary[-1],
                    "trace": trace.to_dict(),
                    "delta": delta.to_dict(),
                    "equivariance": eq.to_dict(),
                    "failure_modes": fm_report.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[host] {scenario_name}: hash={per_scenario_summary[-1]['trace_hash'][:16]}... "
              f"divergence={delta.divergence:.3f} eq_failed={eq.failed_layers} "
              f"modes={per_scenario_summary[-1]['failure_modes_fired']}")

    combined_out = out_dir / "phase4_atlas_summary.json"
    combined = {
        "model_id": remote_payload["model_id"],
        "hidden_dim": remote_payload["hidden_dim"],
        "total_layers": remote_payload["total_layers"],
        "selected_layers": remote_payload["selected_layers"],
        "scenarios": per_scenario_summary,
    }
    combined_out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase4", help="Output dir.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda:1", help="Atlas GPU 1 by default.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the request: each scenario contributes its original + rewrites.
    items = []
    for sc in SCENARIOS:
        text = Path(sc["text_path"]).read_text(encoding="utf-8").strip()
        # Compute each rewrite locally so we don't need to ship lambdas.
        s = text
        rewrites = []
        for r in sc["rewrites"]:
            rewritten = eval(r["fn"], {}, {"s": text})  # safe: literal expressions
            rewrites.append({"name": r["name"], "text": rewritten})
        items.append({"scenario": sc["name"], "rewrites": rewrites})

    req = {
        "model_id": MODEL_ID,
        "device": args.device,
        "max_tokens": args.max_tokens,
        "items": items,
    }

    print(f"[host] dispatching {len(items)} scenarios x {sum(len(i['rewrites']) for i in items)} captures to Atlas")
    remote_payload = run_atlas(req)
    print(f"[host] received remote payload: model_id={remote_payload['model_id']} "
          f"hidden_dim={remote_payload['hidden_dim']} layers={len(remote_payload['selected_layers'])}")

    summary = host_analyze(remote_payload, SCENARIOS, out_dir)
    print(f"[host] wrote summary to {out_dir / 'phase4_atlas_summary.json'}")
    print(json.dumps(summary["scenarios"], indent=2))


if __name__ == "__main__":
    main()
