"""Atlas recon: enumerate HuggingFace / GGUF / vLLM-compatible models present.

Run from the host machine (Windows). Connects to Atlas via Tailscale, queries
common model-cache locations and the venv Python's HF cache, and prints a
structured report.

Used by Phase 4 to pick the activation-lens model.
"""
from __future__ import annotations

import base64
import json
import sys
import paramiko

from erisml_compiler.atlas_creds import atlas_credentials

ATLAS_HOST, USER, PASSWORD = atlas_credentials()  # env or ~/.atlas_creds; never hardcoded
VENV_PY = "/home/claude/env/bin/python3"


def run(ssh: paramiko.SSHClient, cmd: str, timeout: float = 60.0) -> str:
    """Run a shell command, return decoded stdout+stderr via base64 stream."""
    stdin, stdout, stderr = ssh.exec_command(f"({cmd}) 2>&1 | base64", timeout=timeout)
    raw = stdout.read()
    return base64.b64decode(raw).decode("utf-8", errors="replace")


def main() -> int:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ATLAS_HOST, username=USER, password=PASSWORD, timeout=20)

    report: dict[str, str] = {}

    # 1. Identity + GPUs + Python env
    report["uname"] = run(ssh, "uname -a; hostname")
    report["nvidia-smi"] = run(ssh, "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader")
    report["python_versions"] = run(ssh, f"{VENV_PY} --version; {VENV_PY} -c 'import torch, transformers, sys; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 || true")

    # 2. HuggingFace caches in conventional locations
    report["hf_cache_default"] = run(ssh, "ls -1 ~/.cache/huggingface/hub/ 2>/dev/null | head -100")
    report["hf_cache_archive"] = run(ssh, "ls -1 /archive/huggingface/hub/ 2>/dev/null | head -100")
    report["hf_models_dir"] = run(ssh, "ls -1 /archive/models/ 2>/dev/null | head -100")

    # 3. llama.cpp / GGUF
    report["gguf_archive"] = run(ssh, "find /archive -maxdepth 4 -name '*.gguf' 2>/dev/null | head -50")
    report["gguf_home"] = run(ssh, "find ~ -maxdepth 4 -name '*.gguf' 2>/dev/null | head -50")

    # 4. Running model servers
    report["listening_ports"] = run(ssh, "ss -ltnp 2>/dev/null | head -40 || netstat -ltnp 2>/dev/null | head -40")
    report["llama_servers"] = run(ssh, "pgrep -af 'llama-server|vllm|ollama|text-generation|tgi|llama_cpp' 2>/dev/null || echo '(none)'")

    # 5. Ollama models if installed
    report["ollama"] = run(ssh, "which ollama && ollama list 2>&1 || echo '(no ollama)'")

    # 6. Disk space — we'll be writing checkpoints
    report["df_archive"] = run(ssh, "df -h /archive 2>/dev/null; df -h ~")

    # 7. Probe LaBSE specifically — used by Phase 3 calibration; the probe
    # extractor relies on this too. If it's already cached, we save the
    # 500MB download.
    report["labse_cached"] = run(
        ssh,
        f"{VENV_PY} -c 'from huggingface_hub import scan_cache_dir; "
        f"s = scan_cache_dir(); "
        f"print(\"\\n\".join(sorted(r.repo_id for r in s.repos)))' 2>&1 | head -200 || true",
    )

    ssh.close()

    # Render
    for key, val in report.items():
        print("=" * 72)
        print(f"  {key}")
        print("=" * 72)
        print(val.rstrip())
        print()

    # Also dump JSON next to this script for downstream Phase 4 tooling
    out = {k: v.strip() for k, v in report.items()}
    print("---JSON---")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
