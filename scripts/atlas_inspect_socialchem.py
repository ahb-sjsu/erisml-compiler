"""Inspect the Social Chem 101 TSV on Atlas and pull the dearabby slice locally."""
from __future__ import annotations

import base64

import paramiko

ATLAS_HOST = "100.68.134.21"
ATLAS_USER = "claude"
ATLAS_PASS = "roZes9090!~"


INSPECT = r"""#!/usr/bin/env bash
set -euo pipefail
TSV=/archive/ethics-corpora/social-chem-101/social-chem-101/social-chem-101.v1.0.tsv
echo === SIZE
ls -la "$TSV"
echo === HEADER
head -1 "$TSV" | tr '\t' '\n' | nl
echo === ROW COUNT
wc -l "$TSV"
echo === AREA DISTRIBUTION
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)if($i=="area")a=i} NR>1{print $a}' "$TSV" | sort | uniq -c | sort -rn
echo === DEARABBY HEAD
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)if($i=="area")a=i; print; next} $a=="dearabby"' "$TSV" | head -3
echo === SAMPLE DEARABBY ROW pretty
awk -F'\t' 'NR==1{split($0,h,"\t"); next} NR==1{next}' "$TSV"
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){h[i]=$i}} NR>1{for(i=1;i<=NF;i++)if(h[i]=="area"&&$i=="dearabby"){for(j=1;j<=NF;j++)printf "%s=%s\n",h[j],$j; print "---"; exit}}' "$TSV"
echo === DEARABBY COUNT
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)if($i=="area")a=i} NR>1 && $a=="dearabby"' "$TSV" | wc -l
echo === AITA COUNT
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)if($i=="area")a=i} NR>1 && $a=="amitheasshole"' "$TSV" | wc -l
echo === MORAL FOUNDATIONS DIST DEARABBY
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i=="area")a=i;if($i=="rot-moral-foundations")m=i}} NR>1 && $a=="dearabby"{print $m}' "$TSV" | sort | uniq -c | sort -rn | head -20
"""


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ATLAS_HOST, username=ATLAS_USER, password=ATLAS_PASS, timeout=15)

    sftp = c.open_sftp()
    with sftp.open("/tmp/atlas_inspect_sc.sh", "w") as f:
        f.write(INSPECT)
    sftp.chmod("/tmp/atlas_inspect_sc.sh", 0o755)
    sftp.close()

    stdin, stdout, stderr = c.exec_command(
        "bash /tmp/atlas_inspect_sc.sh 2>&1 | base64", timeout=120
    )
    out = stdout.read().decode()
    rc = stdout.channel.recv_exit_status()
    c.close()
    print(base64.b64decode(out).decode("utf-8", errors="replace"))
    print(f"--- exit {rc} ---")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
