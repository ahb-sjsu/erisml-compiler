"""Extract dearabby + AITA slices from Social Chem 101 on Atlas, pull local."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import paramiko

ATLAS_HOST = "100.68.134.21"
ATLAS_USER = "claude"
ATLAS_PASS = "roZes9090!~"

# Carve slices on Atlas (preserve header), then sha256.
EXTRACT = r"""#!/usr/bin/env bash
set -euo pipefail
TSV=/archive/ethics-corpora/social-chem-101/social-chem-101/social-chem-101.v1.0.tsv
OUT=/archive/ethics-corpora/social-chem-101/slices
mkdir -p "$OUT"

extract_area () {
  local area="$1"
  local outfile="$OUT/social-chem-101.${area}.tsv"
  awk -F'\t' -v area="$area" 'NR==1{for(i=1;i<=NF;i++)if($i=="area")a=i; print; next} $a==area' "$TSV" > "$outfile"
  echo "$outfile $(wc -l < "$outfile") rows"
  sha256sum "$outfile"
}

extract_area dearabby
extract_area amitheasshole
extract_area rocstories
extract_area confessions

echo === sizes
ls -la "$OUT"
"""


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ATLAS_HOST, username=ATLAS_USER, password=ATLAS_PASS, timeout=15)

    sftp = c.open_sftp()
    with sftp.open("/tmp/atlas_extract_slices.sh", "w") as f:
        f.write(EXTRACT)
    sftp.chmod("/tmp/atlas_extract_slices.sh", 0o755)

    stdin, stdout, stderr = c.exec_command(
        "bash /tmp/atlas_extract_slices.sh 2>&1 | base64", timeout=600
    )
    out = stdout.read().decode()
    rc = stdout.channel.recv_exit_status()
    print(base64.b64decode(out).decode("utf-8", errors="replace"))
    print(f"--- extract exit {rc} ---")
    if rc != 0:
        c.close()
        return rc

    # Pull the two slices we'll actively use down to local Windows.
    local_dir = Path("C:/source/erisml-compiler/data/social-chem-101")
    local_dir.mkdir(parents=True, exist_ok=True)
    for area in ("dearabby", "amitheasshole"):
        remote = f"/archive/ethics-corpora/social-chem-101/slices/social-chem-101.{area}.tsv"
        local = local_dir / f"social-chem-101.{area}.tsv"
        print(f"=== sftp.get {remote} -> {local}")
        sftp.get(remote, str(local))
        size = os.path.getsize(local)
        print(f"    {size:,} bytes")

    sftp.close()
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
