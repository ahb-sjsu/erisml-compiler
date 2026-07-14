"""Download Social Chem 101 + Scruples to /archive/ethics-corpora on Atlas.

One-shot caching script. Run from local Windows; the actual downloads happen
on Atlas, so files land at /archive (not pushed back over the WAN).

Usage:  python scripts/atlas_cache_aic.py
"""
from __future__ import annotations

import base64

import paramiko

from erisml_compiler.atlas_creds import atlas_credentials

ATLAS_HOST, ATLAS_USER, ATLAS_PASS = atlas_credentials()  # env or ~/.atlas_creds; never hardcoded

BASH_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
SC=/archive/ethics-corpora/social-chem-101
SCR=/archive/ethics-corpora/scruples
mkdir -p "$SC" "$SCR"
echo === downloading socialchem
curl -fsL https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip -o "$SC/social-chem-101.zip"
echo === downloading scruples anecdotes
curl -fsL https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz -o "$SCR/anecdotes.tar.gz"
echo === downloading scruples dilemmas
curl -fsL https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/dilemmas.tar.gz -o "$SCR/dilemmas.tar.gz"
echo === extracting socialchem
cd "$SC" && unzip -o -q social-chem-101.zip
echo === extracting scruples anecdotes
cd "$SCR" && tar xzf anecdotes.tar.gz
echo === extracting scruples dilemmas
cd "$SCR" && tar xzf dilemmas.tar.gz
echo === DONE, listing socialchem
ls -la "$SC"
echo === listing scruples
ls -la "$SCR"
echo === socialchem inner
find "$SC" -maxdepth 3 -type f | head -40
echo === sha256
sha256sum "$SC"/social-chem-101.zip "$SCR"/anecdotes.tar.gz "$SCR"/dilemmas.tar.gz
"""


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ATLAS_HOST, username=ATLAS_USER, password=ATLAS_PASS, timeout=15)

    sftp = c.open_sftp()
    with sftp.open("/tmp/atlas_cache_aic.sh", "w") as f:
        f.write(BASH_SCRIPT)
    sftp.chmod("/tmp/atlas_cache_aic.sh", 0o755)
    sftp.close()

    stdin, stdout, stderr = c.exec_command(
        "bash /tmp/atlas_cache_aic.sh 2>&1 | base64", timeout=600
    )
    out = stdout.read().decode()
    rc = stdout.channel.recv_exit_status()
    c.close()

    decoded = base64.b64decode(out).decode("utf-8", errors="replace")
    print(decoded)
    print(f"--- exit {rc} ---")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
