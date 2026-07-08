"""Host-side driver for the Phase-5 calibration Atlas job (paramiko).

Subcommands (run from the repo root):
  preflight  — check env (torch/transformers/requests), GPU-1 free, NRP token
  put        — sftp the texts + scripts to the Atlas working dir
  smoke      — run label+extract on the first 3 texts, synchronously
  launch     — start the full run detached (nohup), print PID
  poll       — tail run.log; report running / done
  fetch      — sftp the per-layer npz + labels back to --local

Atlas: GPU-1 only (GPU-0 = artemis-avatar). Never kills/reboots anything.
"""
from __future__ import annotations

import argparse
import os
import sys

import paramiko

# Credentials come from a gitignored sibling `_atlas.py` (preferred) or env.
# Never hardcode the password in this committed file.
try:
    from _atlas import HOST, PASSWORD, USER
except ImportError:  # pragma: no cover
    HOST = os.environ.get("ATLAS_HOST", "100.68.134.21")
    USER = os.environ.get("ATLAS_USER", "claude")
    PASSWORD = os.environ["ATLAS_PASSWORD"]
VENV_PY = "/home/claude/env/bin/python3"
REMOTE_DIR = "/home/claude/erisml_calib"
FILES = ["experiments/calibration/calib_texts.jsonl",
         "experiments/calibration/signed_rubric.py",
         "experiments/calibration/atlas_label_and_extract.py"]


def _ssh() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return c


def _run(ssh, cmd, echo=True) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=1800)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if echo:
        if out:
            print(out, end="")
        if err:
            print(err, end="", file=sys.stderr)
    return rc, out, err


def preflight(ssh):
    _run(ssh, f"mkdir -p {REMOTE_DIR}")
    print("== python env ==")
    _run(ssh, f"{VENV_PY} -c \"import torch,transformers,requests as r,numpy; "
              f"print('torch',torch.__version__,'cuda',torch.cuda.is_available(),"
              f"'ndev',torch.cuda.device_count()); print('transformers',transformers.__version__)\"")
    print("== GPUs ==")
    _run(ssh, "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader")
    print("== NRP token ==")
    _run(ssh, "test -f ~/.llmtoken && wc -c ~/.llmtoken || echo 'NO TOKEN'")


def put(ssh):
    _run(ssh, f"mkdir -p {REMOTE_DIR}")
    sftp = ssh.open_sftp()
    for f in FILES:
        remote = f"{REMOTE_DIR}/{f.split('/')[-1]}"
        sftp.put(f, remote)
        print(f"put {f} -> {remote}")
    sftp.close()


def smoke(ssh):
    _run(ssh, f"cd {REMOTE_DIR} && head -3 calib_texts.jsonl > smoke_texts.jsonl && wc -l smoke_texts.jsonl")
    print("== running 3-text label+extract (synchronous) ==")
    rc, out, err = _run(
        ssh, f"cd {REMOTE_DIR} && {VENV_PY} atlas_label_and_extract.py "
             f"--texts smoke_texts.jsonl --out-dir smoke_out --device cuda:1 --min-judges 1")
    print(f"== smoke exit={rc} ==")
    _run(ssh, f"cd {REMOTE_DIR} && ls -la smoke_out/ 2>/dev/null && "
              f"{VENV_PY} -c \"import numpy as np,glob; "
              f"f=sorted(glob.glob('smoke_out/layer_*.npz'))[0]; d=np.load(f,allow_pickle=True); "
              f"print('layer file',f,'X',d['X'].shape,'Y',d['Y'].shape); "
              f"print('Y[0]',d['Y'][0].round(2))\"")


def launch(ssh, n_min_judges=1):
    cmd = (f"cd {REMOTE_DIR} && nohup {VENV_PY} atlas_label_and_extract.py "
           f"--texts calib_texts.jsonl --out-dir calib_features --device cuda:1 "
           f"--min-judges {n_min_judges} > run.log 2>&1 & echo PID=$!")
    _run(ssh, cmd)
    print(f"launched; poll with: python {__file__} poll")


def poll(ssh):
    _run(ssh, f"cd {REMOTE_DIR} && tail -n 8 run.log 2>/dev/null; "
              f"echo '---'; ls calib_features/*.npz 2>/dev/null | wc -l | "
              f"xargs echo 'layer files so far:'; "
              f"pgrep -f atlas_label_and_extract.py >/dev/null && echo STATUS=RUNNING || echo STATUS=DONE")


def fetch(ssh, local):
    import os
    os.makedirs(local, exist_ok=True)
    sftp = ssh.open_sftp()
    remote_feat = f"{REMOTE_DIR}/calib_features"
    for name in sftp.listdir(remote_feat):
        if name.endswith(".npz") or name in ("labels.jsonl", "extract_meta.json"):
            sftp.get(f"{remote_feat}/{name}", f"{local}/{name}")
            print(f"fetched {name}")
    sftp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "put", "smoke", "launch", "poll", "fetch"])
    ap.add_argument("--local", default="experiments/calibration/calib_features")
    ap.add_argument("--min-judges", type=int, default=1)
    a = ap.parse_args()
    ssh = _ssh()
    try:
        if a.cmd == "preflight":
            preflight(ssh)
        elif a.cmd == "put":
            put(ssh)
        elif a.cmd == "smoke":
            smoke(ssh)
        elif a.cmd == "launch":
            launch(ssh, a.min_judges)
        elif a.cmd == "poll":
            poll(ssh)
        elif a.cmd == "fetch":
            fetch(ssh, a.local)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
