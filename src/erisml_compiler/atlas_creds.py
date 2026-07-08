"""Atlas SSH credential loader.

NEVER hardcode credentials in committed source. This resolves them at runtime:

  1. Environment: ATLAS_HOST / ATLAS_USER / ATLAS_PASSWORD
  2. A gitignored file `~/.atlas_creds` — first line is the password
     (optionally `host`/`user`/`password` on their own `key=value` lines).
  3. Otherwise raise with instructions.

The host/user default to the known non-secret values; only the password must
come from env or the file.
"""

from __future__ import annotations

import os
from pathlib import Path


def atlas_credentials() -> tuple[str, str, str]:
    """Return (host, user, password). Raises RuntimeError if no password source."""
    host = os.environ.get("ATLAS_HOST", "100.68.134.21")
    user = os.environ.get("ATLAS_USER", "claude")
    password = os.environ.get("ATLAS_PASSWORD")

    if not password:
        path = Path(os.path.expanduser("~/.atlas_creds"))
        if path.exists():
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            kv = {}
            for ln in lines:
                if "=" in ln and ln.split("=", 1)[0].strip().lower() in {"host", "user", "password"}:
                    k, v = ln.split("=", 1)
                    kv[k.strip().lower()] = v.strip()
            host = kv.get("host", host)
            user = kv.get("user", user)
            password = kv.get("password") or (lines[0] if lines and "=" not in lines[0] else None)

    if not password:
        raise RuntimeError(
            "Atlas password not found. Set the ATLAS_PASSWORD environment variable, "
            "or create a gitignored ~/.atlas_creds file (chmod 600) whose first line "
            "is the password. Never hardcode credentials in source."
        )
    return host, user, password
