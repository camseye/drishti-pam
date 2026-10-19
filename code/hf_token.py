"""Shared Hugging Face token resolution + login helper.

Token resolution order (first match wins):
  1. The `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variable.
  2. A `.hf_token` file at the project root (a single line containing the
     token, no other content). Treated as a secret; do not commit it.
  3. None — anonymous downloads. Hugging Face rate-limits anonymous traffic
     and gates some models, but flan-t5-base and distilbert-base-uncased
     are public and will work without a token, just slower.

Once resolved, calling `setup_token()` does two things:
  * Calls `huggingface_hub.login(token=token, add_to_git_credential=False)`
    so every subsequent `from_pretrained()` call inherits the auth header.
    With auth, downloads pull from authenticated endpoints which are both
    faster and not rate-limited.
  * Re-exports the token into the process's HF_TOKEN env var so child
    subprocesses (e.g. those spawned by smoke.py) inherit it too.

This module is imported by generator/llm_augment.py, llm_detector/train.py,
and llm_detector/detect.py. It has no hard dependency on huggingface_hub —
if the package is missing the helper degrades to a no-op and a warning,
because the mock pipeline must still run on a machine that has never
heard of Hugging Face.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / ".hf_token"

_LOGGED_IN = False


def resolve_token() -> str | None:
    """Look up the token without touching huggingface_hub. Pure I/O."""
    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(env_var)
        if v and v.strip():
            return v.strip()
    if TOKEN_FILE.exists():
        try:
            t = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if t and not t.startswith("#"):
                return t
        except OSError:
            pass
    return None


def setup_token(verbose: bool = True) -> str | None:
    """Resolve, log in if possible, and propagate to env. Returns the token
    string (or None) so callers can pass it directly to `from_pretrained(
    token=...)` if they prefer explicit threading over the global login."""
    global _LOGGED_IN
    token = resolve_token()
    if token is None:
        if verbose:
            print("[hf-token] No token found (env HF_TOKEN / .hf_token file). "
                  "Downloads will use anonymous endpoints (rate-limited).",
                  flush=True)
        return None

    # Make sure subprocesses see the token even if it came from .hf_token.
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

    if _LOGGED_IN:
        return token

    try:
        from huggingface_hub import login                                   # type: ignore[import-not-found]
        login(token=token, add_to_git_credential=False)
        _LOGGED_IN = True
        if verbose:
            # Show only the prefix so a leaked log line doesn't leak the key.
            redacted = token[:4] + "…" + token[-4:] if len(token) > 8 else "(short)"
            print(f"[hf-token] Authenticated to Hugging Face Hub (token={redacted}). "
                  f"Downloads will be faster.", flush=True)
    except ImportError:
        if verbose:
            print("[hf-token] huggingface_hub not installed; cannot log in. "
                  "Token will still be available via env vars to any "
                  "transformers download that reads them.", flush=True)
    except Exception as e:                                                  # noqa: BLE001
        # login() can fail on revoked / typo'd tokens. Don't crash — fall
        # back to anonymous and let the user notice from the slower download.
        if verbose:
            print(f"[hf-token] login() failed: {e}. Falling back to "
                  f"anonymous downloads.", flush=True)
    return token


def save_token(token: str) -> Path:
    """Persist a token to `.hf_token` for future runs. Used by the
    interactive prompts in smoke.py / setup_gpu.ps1."""
    TOKEN_FILE.write_text(token.strip() + "\n", encoding="utf-8")
    try:
        # Best-effort POSIX permission tighten — Windows ignores this.
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass
    return TOKEN_FILE
