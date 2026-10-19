"""Diagnose stage-3 access-violation crashes (`exit 3221225477`,
`0xC0000005 STATUS_ACCESS_VIOLATION`) on Windows.

Walks the same import chain that `llm_detector/train.py` follows when
the HF backend is selected, but in isolation — one import at a time,
each preceded by an indented "Step N starting…" line that survives
even if the import segfaults. The LAST printed line tells you exactly
which library faulted.

If a step crashes:
  * Python's faulthandler (enabled at the very top, before any heavy
    import) prints a native C stack trace to stderr — usually pinpointing
    the offending DLL by name (libiomp5md.dll, mkl_avx2.dll, libtorch_cpu.dll).
  * Re-run with `python -X dev doctor.py` for extra runtime checks.
  * The "What this means + how to fix" table at the bottom of this file
    catches the common Windows variants.

The script never downloads anything from Hugging Face; it only checks
the local install. To verify HF download too, pass `--with-download` and
it will attempt to load `distilbert-base-uncased` from the configured
HF cache (downloading if missing).
"""
from __future__ import annotations

# --- 0. Windows-safety preamble (mirror of train.py / detect.py top) ---- #
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import faulthandler
faulthandler.enable()

import argparse
import platform
import time
from pathlib import Path


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"

# Disable colour on legacy Windows consoles where ANSI isn't honoured —
# detection is best-effort; the worst-case is escape-code noise.
if sys.platform == "win32" and not os.environ.get("WT_SESSION") \
        and not os.environ.get("TERM_PROGRAM"):
    GREEN = RED = DIM = RESET = ""


_FAILURES: list[str] = []
_VERSIONS: dict[str, str] = {}


def step(n: int, total: int, what: str) -> None:
    """Print a 'starting' marker that is durable to a native crash on the
    next line — flush=True forces stderr+stdout to the OS before the
    fault handler kicks in."""
    print(f"  [{n}/{total}] {what} ...", end=" ", flush=True)


def ok(detail: str = "", t0: float | None = None) -> None:
    elapsed = f" ({time.time() - t0:.2f}s)" if t0 is not None else ""
    print(f"{GREEN}OK{RESET}{elapsed}  {DIM}{detail}{RESET}", flush=True)


def fail(reason: str, fix: str = "") -> None:
    """Mark the current step as failed but DON'T abort the whole probe.
    Native crashes (access violation) bypass this — they're caught by
    faulthandler and kill the process, which is what we want when
    pinpointing a faulting DLL."""
    _FAILURES.append(reason)
    print(f"{RED}FAIL{RESET}  {DIM}{reason}{RESET}", flush=True)
    if fix:
        print(f"        → {fix}", flush=True)


def _safe_import(name: str, n: int, total: int, label: str,
                 fix_hint: str = "") -> "object | None":
    """Wrap an import in try/except so a missing module doesn't abort the
    whole probe. ImportError → fail+continue; any non-ImportError
    exception (incl. native crash → no exception, faulthandler triggers)
    propagates."""
    step(n, total, label)
    t0 = time.time()
    try:
        mod = __import__(name)
    except ImportError as e:
        fail(f"ModuleNotFoundError: {e}", fix=fix_hint)
        return None
    ver = getattr(mod, "__version__", "?")
    _VERSIONS[name] = ver
    ok(f"{name}={ver}", t0)
    return mod


def section(title: str) -> None:
    print()
    print("─" * 72)
    print(f"  {title}")
    print("─" * 72, flush=True)


def main(with_download: bool = False) -> int:
    section("Environment")
    print(f"  python      = {sys.version.splitlines()[0]}")
    print(f"  executable  = {sys.executable}")
    print(f"  platform    = {platform.platform()}")
    print(f"  cwd         = {Path.cwd()}")
    print(f"  guards      = KMP_DUPLICATE_LIB_OK={os.environ['KMP_DUPLICATE_LIB_OK']}, "
          f"OMP={os.environ['OMP_NUM_THREADS']}, "
          f"TOKENIZERS_PARALLELISM={os.environ['TOKENIZERS_PARALLELISM']}",
          flush=True)

    # Total step count tracked for the [n/total] indicator.
    total = 12 if with_download else 9

    section("Step-by-step import probe")
    print("  If any step crashes, the LAST printed 'starting...' line tells",
          flush=True)
    print("  you which library is faulting. Native stack trace will follow.",
          flush=True)
    print()

    # 1. numpy — the ABI base layer for everything else.
    np = _safe_import("numpy", 1, total, "import numpy",
                      fix_hint="pip install --force-reinstall 'numpy<2' "
                               "(if torch was built against numpy 1.x)")

    # 2. torch — the heavy DLL base. Most Windows access violations come from
    # this import or the first kernel launched after it.
    torch = _safe_import("torch", 2, total, "import torch",
                         fix_hint="install matching wheel via setup_gpu.ps1 "
                                  "(GPU) or `pip install torch` (CPU). If "
                                  "the import CRASHES with no traceback, "
                                  "the wheel is mismatched against your "
                                  "CUDA driver — re-run setup_gpu.ps1 to "
                                  "auto-detect and reinstall.")

    if torch is not None:
        # 3. torch.cuda probe — does NOT actually allocate, just queries.
        step(3, total, "torch.cuda.is_available()")
        t0 = time.time()
        try:
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                n = torch.cuda.device_count()
                names = ", ".join(torch.cuda.get_device_name(i)
                                  for i in range(n))
                ok(f"CUDA available: {n} GPU(s) — {names}", t0)
            else:
                ok("CPU-only torch (no CUDA)", t0)
        except Exception as e:
            fail(f"{type(e).__name__}: {e}",
                 fix="GPU driver / CUDA install mismatch. Try `nvidia-smi` "
                     "first; if that's also broken, reinstall the GeForce / "
                     "Quadro driver.")

        # 4. tiny tensor matmul — exercises BLAS / OMP. This is where MKL
        # double-init crashes on Windows if the guards above didn't hold.
        step(4, total, "torch tensor matmul (exercises MKL/OMP)")
        t0 = time.time()
        try:
            x = torch.zeros(64, 64)
            y = x @ x
            _ = y.sum().item()
            ok(f"matmul on {x.device}", t0)
        except Exception as e:
            fail(f"{type(e).__name__}: {e}",
                 fix="External KMP/OMP env vars are interfering. Run "
                     "smoke.py instead of stages directly — it sets "
                     "KMP_DUPLICATE_LIB_OK=TRUE and OMP_NUM_THREADS=1 "
                     "before spawning subprocesses.")
    else:
        # If torch is missing the rest of the probe is moot but we still
        # report the high-level package state so the user sees the full picture.
        print("  [3-4] skipped (torch missing)", flush=True)

    # 5. tokenizers — Rust runtime, separate DLL surface from torch.
    _safe_import("tokenizers", 5, total, "import tokenizers",
                 fix_hint="pip install -U 'tokenizers'")

    # 6. transformers — pulls in tokenizers + safetensors + many sub-libs.
    _safe_import("transformers", 6, total, "import transformers",
                 fix_hint="pip install --force-reinstall --upgrade "
                          "'transformers>=5.5,<6'")

    # 7. datasets — relies on pyarrow, another DLL with known Windows
    # quirks (missing OneDrive support, file-locking on AV-scanned dirs).
    _safe_import("datasets", 7, total, "import datasets",
                 fix_hint="pip install --force-reinstall --upgrade 'datasets>=3'")

    # 8. accelerate — Trainer requires accelerate >= 0.34 for v5+.
    _safe_import("accelerate", 8, total, "import accelerate",
                 fix_hint="pip install -U 'accelerate>=1'")

    # 9. huggingface_hub — token resolution path.
    _safe_import("huggingface_hub", 9, total, "import huggingface_hub",
                 fix_hint="pip install -U 'huggingface_hub>=1'")

    if with_download:
        section("HF cache + small model load (downloads if missing)")
        if "transformers" not in _VERSIONS or "torch" not in _VERSIONS:
            print("  [10-12] skipped (transformers/torch missing — fix above first)",
                  flush=True)
        else:
            from transformers import (AutoTokenizer,
                                      AutoModelForSequenceClassification)

            step(10, total, "HF cache directory")
            cache = Path(os.environ.get("HF_HOME") or
                         Path.home() / ".cache" / "huggingface")
            if cache.exists():
                try:
                    size_mb = sum(f.stat().st_size for f in cache.rglob("*")
                                  if f.is_file()) / 1e6
                    ok(f"{cache} ({size_mb:.0f} MB cached)")
                except OSError as e:
                    fail(f"{type(e).__name__}: {e}",
                         fix="HF cache directory unreadable — could be a "
                             "Windows AV scanning lock.")
            else:
                ok(f"{cache} (empty — fresh install)")

            step(11, total, "AutoTokenizer.from_pretrained(distilbert-base-uncased)")
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                ok(f"vocab={len(tok)}", t0)

                step(12, total,
                     "AutoModelForSequenceClassification + 1-token forward")
                t0 = time.time()
                model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2)
                model.eval()
                with torch.no_grad():                                  # type: ignore[union-attr]
                    ids = tok("hello", return_tensors="pt")
                    out = model(**ids)
                ok(f"logits shape={tuple(out.logits.shape)}", t0)
            except Exception as e:
                fail(f"{type(e).__name__}: {e}",
                     fix="If this is a CRASH (no traceback) it's almost "
                         "always Windows Defender intercepting the model "
                         "mmap. Add an exclusion for the HF cache "
                         "directory (~/.cache/huggingface/) and retry.")

    # ------------------------------------------------------------------ #
    # Compatibility checks — flag known-bad version combos even when the
    # imports themselves succeeded.
    # ------------------------------------------------------------------ #
    section("Version compatibility")
    advisories: list[str] = []
    if "transformers" in _VERSIONS:
        try:
            major = int(_VERSIONS["transformers"].split(".")[0])
            minor = int(_VERSIONS["transformers"].split(".")[1])
            if (major, minor) < (5, 5):
                advisories.append(
                    f"transformers={_VERSIONS['transformers']} is older "
                    "than the required >=5.5. The Trainer API changed in "
                    "v5.0 (`processing_class` rename) and the "
                    "`pipeline('text2text-generation')` factory was removed. "
                    "Fix: pip install --force-reinstall --upgrade "
                    "'transformers>=5.5,<6'")
        except (ValueError, IndexError):
            pass
    if "torch" in _VERSIONS and "numpy" in _VERSIONS:
        try:
            np_major = int(_VERSIONS["numpy"].split(".")[0])
            t_major = int(_VERSIONS["torch"].split(".")[0])
            t_minor = int(_VERSIONS["torch"].split(".")[1])
            # torch < 2.3 was built against numpy 1.x; pairing it with
            # numpy 2.x produces sporadic ABI-related access violations.
            if np_major >= 2 and (t_major, t_minor) < (2, 3):
                advisories.append(
                    f"torch={_VERSIONS['torch']} (built for numpy 1.x) + "
                    f"numpy={_VERSIONS['numpy']} (2.x) is a KNOWN ABI "
                    "MISMATCH. Fix: pip install --force-reinstall "
                    "'numpy<2' OR upgrade torch to >=2.3.")
        except (ValueError, IndexError):
            pass
    if "datasets" not in _VERSIONS:
        advisories.append(
            "`datasets` is missing entirely — `llm_detector/train.py` "
            "stage 3 imports `from datasets import Dataset` and will fail "
            "on first run. Fix: pip install --force-reinstall --upgrade "
            "'datasets>=3'.")

    if advisories:
        for a in advisories:
            print(f"  {RED}WARN{RESET}  {a}", flush=True)
    else:
        print(f"  {GREEN}All known incompatible combos: clear.{RESET}",
              flush=True)

    # ------------------------------------------------------------------ #
    section("Summary")
    if _FAILURES:
        print(f"  {RED}{len(_FAILURES)} step(s) failed.{RESET} See the FAIL "
              "lines above for the fix command per row.", flush=True)
        return 1
    if advisories:
        print(f"  {RED}{len(advisories)} version advisory(ies).{RESET} "
              "Imports succeeded but the runtime will likely crash. "
              "Apply the fixes above before re-running smoke.py.",
              flush=True)
        return 1
    print(f"  {GREEN}Your install is healthy.{RESET} If smoke.py still crashes",
          flush=True)
    print("  at stage 3, run the heavyweight check (downloads ~270 MB):",
          flush=True)
    print("    python doctor.py --with-download", flush=True)
    print("  Or capture a native trace from train.py directly:", flush=True)
    print("    python -X faulthandler -u llm_detector/train.py "
          "--backend huggingface", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-download", action="store_true",
                   help="Also exercise model download + a 1-token forward "
                        "pass (uses ~270 MB of HF cache on first run).")
    args = p.parse_args()
    try:
        sys.exit(main(with_download=args.with_download))
    except Exception:
        # We don't suppress the traceback; we just want to make sure the
        # process exits with a non-zero code so the parent shell can react.
        import traceback
        traceback.print_exc()
        sys.exit(1)
