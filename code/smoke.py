"""End-to-end smoke harness for the PAM behavioural anomaly project.

Runs the data + detector pipeline in order so a fresh checkout can be
reproduced with a single command:

    python smoke.py                              # interactive: asks about backend, GPU, token
    python smoke.py --backend mock               # local mock build (~15 s, no prompts)
    python smoke.py --backend huggingface \\
                    --expand-factor 2            # cloud build (Kaggle, ~15 min, no prompts)

Each stage is a separate subprocess call so failures land in the stage that
caused them.

Stages:

    1. Generate the synthetic dataset            generator/generate.py
    2. Score the rule detector                   rule_detector/detect.py
    3. Train the trained-LLM detector            llm_detector/train.py
    4. Score the trained-LLM detector on test    llm_detector/detect.py
    5. Compare on the test split                 compare.py

Choosing `--backend huggingface` switches stages 1 and 3 to the HuggingFace
code path: Flan-T5-base paraphrasing in the generator, DistilBERT
fine-tuning in the trainer. Stage 4 auto-detects from the saved model.

After `compare.py` finishes, the comparison and per-session prediction
files land under `code/data/`. Run `python docs/build_dashboard.py` to
refresh the dashboard with the new run.

Interactive mode (the default when no `--backend` is supplied AND stdin is a
TTY) prompts for:
  * mock vs huggingface backend
  * number of sessions (default 10000)
  * GPU vs CPU (only if huggingface AND a GPU is detected)
  * Hugging Face token (only if huggingface AND no token is currently set)
  * expand-factor (only if huggingface)

Pass `--non-interactive` to suppress all prompts even when stdin is a TTY
(useful for CI / unattended runs); the script then uses the supplied flags
or falls back to the documented defaults.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Interactive prompt helpers
# --------------------------------------------------------------------------- #

def _ask(prompt: str, default: str, choices: list[str] | None = None) -> str:
    """Prompt the user with a default; loop on invalid input. `choices`
    (when given) restricts answers to a closed set, case-insensitive."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if choices is None or raw.lower() in [c.lower() for c in choices]:
            return raw
        print(f"  Please answer one of: {', '.join(choices)}")


def _ask_yes_no(prompt: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{d}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def _ask_int(prompt: str, default: int, min_value: int = 1) -> int:
    """Free-form numeric prompt with a minimum bound and a default."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            n = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if n < min_value:
            print(f"  Value must be >= {min_value} (got {n}).")
            continue
        return n


def _detect_gpu() -> tuple[bool, str]:
    """Return (has_gpu, message). Never raises — torch may not be installed."""
    try:
        import torch                                                # type: ignore[import-not-found]
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
            return True, f"GPU detected: {names}"
        return False, f"GPU not available (torch {torch.__version__} CPU-only)"
    except ImportError:
        return False, "torch is not installed"


def interactive_setup() -> dict:
    """Run the interactive prompt sequence. Returns the chosen options as a
    dict suitable for merging back onto argparse defaults."""
    print()
    print("=" * 72)
    print("  Drishti-PAM — interactive setup")
    print("=" * 72)
    print("Press Enter at any prompt to accept the [default].")
    print()

    backend = _ask("Backend? mock / huggingface", "mock",
                   choices=["mock", "huggingface"]).lower()

    chosen: dict = {"backend": backend}

    # Session count — small numbers are useful for smoke tests / quick
    # iteration on Kaggle GPU minutes; default 10 000 matches the
    # published reference run. The trainer needs at least ~50 sessions
    # to produce a non-degenerate train/val split, so anything below
    # that is rejected.
    chosen["num_sessions"] = _ask_int(
        "Number of sessions (10000 = published reference run; "
        "lower = faster iteration; bare-min 50)",
        default=10_000, min_value=50)

    if backend == "huggingface":
        has_gpu, msg = _detect_gpu()
        print(f"  {msg}")
        if has_gpu:
            use_gpu = _ask_yes_no("  Use GPU?", default=True)
            chosen["force_cpu"] = not use_gpu
        else:
            print("  (Running on CPU. The mock backend is ~1000x faster; "
                  "ctrl-C to switch.)")
            chosen["force_cpu"] = True

        # Token prompt — only ask if no token is currently visible.
        try:
            import hf_token
            tok = hf_token.resolve_token()
        except ImportError:
            tok = None
        if tok:
            redacted = tok[:4] + "…" + tok[-4:] if len(tok) > 8 else "(short)"
            print(f"  Hugging Face token already configured ({redacted}).")
        else:
            print("  No Hugging Face token found.")
            print("  Get one at https://huggingface.co/settings/tokens "
                  "(read access is enough).")
            print("  A token avoids anonymous-download rate limits and is "
                  "much faster on Kaggle.")
            entered = input("  Paste token (or Enter to skip): ").strip()
            if entered:
                save = _ask_yes_no("  Save to .hf_token for future runs?",
                                   default=True)
                if save:
                    try:
                        import hf_token
                        path = hf_token.save_token(entered)
                        print(f"  Saved to {path}")
                    except ImportError:
                        # Manual fallback if hf_token missing for some reason.
                        (ROOT / ".hf_token").write_text(entered + "\n",
                                                        encoding="utf-8")
                # Make it visible to the rest of this process.
                os.environ["HF_TOKEN"] = entered
                os.environ["HUGGINGFACE_HUB_TOKEN"] = entered

        ef = _ask("  Expand-factor (1 = light, 2 = wider command pool)",
                  "1", choices=["1", "2", "3"])
        chosen["expand_factor"] = int(ef)

    print()
    print("Setup complete:")
    print(f"  backend       = {chosen['backend']}")
    print(f"  num-sessions  = {chosen['num_sessions']}")
    if backend == "huggingface":
        print(f"  expand-factor = {chosen['expand_factor']}")
        print(f"  force-cpu     = {chosen.get('force_cpu', False)}")
    print("=" * 72)
    print()
    return chosen


# --------------------------------------------------------------------------- #
# Pipeline construction + execution
# --------------------------------------------------------------------------- #

def build_pipeline(backend: str, expand_factor: int,
                   num_sessions: int | None = None) -> list[list[str]]:
    """Compose the per-stage subprocess argv list. The HF flags are only
    threaded into the stages that consume them (generator + trainer); the
    rule detector, the inference stage, and the comparison are
    backend-agnostic.

    `num_sessions` (when not None) overrides the generator's built-in
    10 000 default. The downstream stages all read from `data/sessions.json`
    so the count flows through naturally — no need to pass it again."""

    gen_cmd = [PYTHON, "-u", "generator/generate.py"]
    train_cmd = [PYTHON, "-u", "llm_detector/train.py"]
    if num_sessions is not None:
        gen_cmd += ["--num-sessions", str(num_sessions)]
    if backend == "huggingface":
        gen_cmd += ["--llm-backend", "huggingface",
                    "--expand-factor", str(expand_factor)]
        train_cmd += ["--backend", "huggingface"]

    return [
        gen_cmd,
        [PYTHON, "-u", "rule_detector/detect.py"],
        train_cmd,
        [PYTHON, "-u", "llm_detector/detect.py"],
        [PYTHON, "-u", "compare.py"],
    ]


# Friendly stage labels for the progression header.
STAGE_LABELS = [
    "Generate synthetic dataset",
    "Score rule detector",
    "Train trained-LLM detector",
    "Score trained-LLM detector",
    "Compare detectors on test split",
]


def run(stage: list[str], idx: int, total: int, label: str) -> int:
    pretty = " ".join(shlex.quote(s) for s in stage)
    bar = "─" * 72
    print(f"\n{bar}")
    print(f"  Stage {idx}/{total}: {label}")
    print(f"  $ {pretty}")
    print(bar, flush=True)
    t0 = time.time()
    # `python -u` (set in build_pipeline) keeps subprocess stdout unbuffered
    # so tqdm bars and progress messages show in real time even when piped.
    res = subprocess.run(stage, cwd=str(ROOT))
    elapsed = time.time() - t0
    status = "ok" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
    print(f"\n  Stage {idx}/{total} {status} in {elapsed:.1f}s",
          flush=True)
    return res.returncode


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mock", "huggingface"], default=None,
                   help="LLM backend for both data augmentation and detector "
                        "training. Default 'mock' is offline; 'huggingface' "
                        "downloads Flan-T5-base + DistilBERT. When omitted "
                        "and stdin is a TTY, an interactive prompt asks.")
    p.add_argument("--expand-factor", type=int, default=1,
                   help="Number of variants per command in the generator's "
                        "command-pool expansion. The cloud notebook runs "
                        "this with 2 to widen the augmented vocabulary.")
    p.add_argument("--num-sessions", type=int, default=None,
                   help="Number of synthetic sessions to generate. When "
                        "omitted, the generator's built-in default (10000) "
                        "is used. Lower values are useful for quick "
                        "iteration / Kaggle-GPU budget. Minimum 50 (the "
                        "trainer needs a non-degenerate train/val split).")
    p.add_argument("--non-interactive", action="store_true",
                   help="Suppress interactive prompts even when stdin is a "
                        "TTY. Use the supplied flags / defaults instead.")
    p.add_argument("--force-cpu", action="store_true",
                   help="Force CPU execution by hiding all CUDA devices "
                        "(sets CUDA_VISIBLE_DEVICES=). Useful when a GPU "
                        "is present but you don't want to use it.")
    args = p.parse_args(argv)

    # Decide whether to prompt:
    # - explicit --non-interactive: never prompt
    # - explicit --backend on the CLI: don't prompt (assume scripted run)
    # - stdin is not a TTY: don't prompt (e.g. piped, CI, Kaggle notebook)
    interactive = (not args.non_interactive
                   and args.backend is None
                   and sys.stdin.isatty())

    if interactive:
        chosen = interactive_setup()
        backend = chosen["backend"]
        expand_factor = chosen.get("expand_factor", 1)
        force_cpu = chosen.get("force_cpu", False)
        num_sessions = chosen.get("num_sessions")
    else:
        backend = args.backend or "mock"
        expand_factor = args.expand_factor
        force_cpu = args.force_cpu
        num_sessions = args.num_sessions

    if num_sessions is not None and num_sessions < 50:
        print(f"[smoke] --num-sessions={num_sessions} is below the safe "
              f"minimum (50). The trainer needs at least 50 to produce a "
              f"non-degenerate train/val split. Aborting.", file=sys.stderr)
        return 2

    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[smoke] CUDA_VISIBLE_DEVICES set to empty — running on CPU.",
              flush=True)

    pipeline = build_pipeline(backend, expand_factor,
                              num_sessions=num_sessions)
    n_str = f"{num_sessions}" if num_sessions is not None else "default(10000)"
    print(f"Backend: {backend}   num-sessions: {n_str}   "
          f"expand-factor: {expand_factor}   stages: {len(pipeline)}")

    # Pre-flight check for the HF backend: if torch is missing or CPU-only,
    # the run will still complete but Flan-T5 + DistilBERT fine-tune on CPU
    # is roughly 30-40 minutes. Warn loudly so the user knows what's coming
    # and can ctrl-C to install GPU torch instead.
    if backend == "huggingface":
        # OMP-safety env vars — set BEFORE the first subprocess starts so all
        # children inherit them. Each child also calls setdefault() at its
        # own top, which catches the case where someone runs the child
        # directly without going through smoke.py. See llm_detector/train.py
        # for the full explanation; in short, this prevents the
        # 0xC0000005 ACCESS_VIOLATION crash (Windows exit 3221225477) that
        # fires when torch's MKL libiomp5md.dll collides with the OMP
        # runtime pulled in by tokenizers / safetensors / numpy.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        # Tokenizers' Rust runtime spawns its own thread pool; pinning to
        # 1 prevents a separate Windows DLL race that the OMP env vars
        # alone don't catch. See llm_detector/train.py preamble.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("RAYON_NUM_THREADS", "1")
        # Force Python's fault handler ON in every child subprocess. If
        # any HF stage segfaults / hits an access violation, we'll get a
        # native C stack trace on stderr instead of a silent
        # `exit 3221225477` with no clue which DLL faulted.
        os.environ.setdefault("PYTHONFAULTHANDLER", "1")
        # Default to single-GPU. Kaggle exposes T4×2 by default; HF
        # Trainer wraps the model in `torch.nn.DataParallel` whenever
        # `torch.cuda.device_count() > 1`, which (a) emits a confusing
        # "tied weights" warning for DistilBERT, (b) doubles the
        # effective batch size at the cost of halving the optimizer
        # step count (so `--num-train-epochs=3` runs 75 steps instead
        # of 150 on a 1k-session dataset and the model fails to
        # converge — Stage 5's llm-trained F1 collapses to 0.0), and
        # (c) brings no real speedup on this scale (data-loading and
        # tokenisation dominate, not GPU forward). Pin to one GPU so
        # the step count matches `--num-train-epochs * (n_train /
        # batch_size)`. Override by setting CUDA_VISIBLE_DEVICES
        # explicitly (e.g. "0,1") before invoking smoke.py.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        print(f"[smoke] Windows-safety env: KMP_DUPLICATE_LIB_OK="
              f"{os.environ['KMP_DUPLICATE_LIB_OK']}, "
              f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}, "
              f"TOKENIZERS_PARALLELISM={os.environ['TOKENIZERS_PARALLELISM']}, "
              f"PYTHONFAULTHANDLER={os.environ['PYTHONFAULTHANDLER']}, "
              f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
              flush=True)

        try:
            import torch  # type: ignore[import-not-found]
            if not torch.cuda.is_available():
                print("", flush=True)
                print("=" * 72, flush=True)
                print("  WARNING: --backend huggingface requested but torch has no GPU.",
                      flush=True)
                print(f"  torch version: {torch.__version__}", flush=True)
                print("  This run will use CPU. Expected wall-clock: ~30-40 minutes.",
                      flush=True)
                print("  To enable GPU on Windows, ctrl-C now and run:",
                      flush=True)
                print("    .\\setup_gpu.ps1", flush=True)
                print("  Or use the Kaggle notebook at notebooks/cloud_run.ipynb",
                      flush=True)
                print("  (free GPU T4 x2, ~12 minutes end-to-end).", flush=True)
                print("", flush=True)
                print("  If stage 3 dies immediately with exit 3221225477,",
                      flush=True)
                print("  that's a Windows OpenMP DLL conflict. The OMP guards",
                      flush=True)
                print("  above silence it. Override the train batch size on",
                      flush=True)
                print("  small-RAM machines with: PAM_TRAIN_BATCH_SIZE=2",
                      flush=True)
                print("=" * 72, flush=True)
                print("", flush=True)
        except ImportError:
            print("", flush=True)
            print("=" * 72, flush=True)
            print("  ERROR: --backend huggingface requested but torch is not installed.",
                  flush=True)
            print("  Install with one of:", flush=True)
            print("    .\\setup_gpu.ps1                                 # Windows, auto-detect",
                  flush=True)
            print("    pip install -r requirements.txt                 # Linux / macOS / CPU only",
                  flush=True)
            print("=" * 72, flush=True)
            return 2

    # Resolve and log in with HF token (no-op if no token). We do this in the
    # parent process AND set HF_TOKEN in the env so child subprocesses (each
    # stage) inherit the credential without re-prompting.
    if backend == "huggingface":
        try:
            import hf_token
            hf_token.setup_token(verbose=True)
        except ImportError:
            pass

    t0 = time.time()
    total = len(pipeline)
    for idx, stage in enumerate(pipeline, start=1):
        label = STAGE_LABELS[idx - 1] if idx - 1 < len(STAGE_LABELS) else "stage"
        rc = run(stage, idx, total, label)
        if rc != 0:
            print(f"\n*** Pipeline halted at stage {idx} ({label}) — exit {rc}",
                  file=sys.stderr)
            # Windows access-violation exit code (0xC0000005). The OMP
            # guards catch the most common case; if the user is still
            # hitting it the cause is somewhere deeper (Defender mmap
            # interception, MSVC redist, faulty wheel, numpy ABI break).
            # Point them at doctor.py which probes each import in
            # isolation.
            if rc in (3221225477, -1073741819):
                print("", file=sys.stderr)
                print("=" * 72, file=sys.stderr)
                print("  exit 3221225477 (== 0xC0000005 STATUS_ACCESS_VIOLATION).",
                      file=sys.stderr)
                print("  This is a native crash inside one of the HF / torch",
                      file=sys.stderr)
                print("  DLLs. The OMP guards above silence the most common",
                      file=sys.stderr)
                print("  case, but yours is happening elsewhere. To pinpoint",
                      file=sys.stderr)
                print("  which import is faulting, run the doctor:",
                      file=sys.stderr)
                print("", file=sys.stderr)
                print("    python doctor.py", file=sys.stderr)
                print("", file=sys.stderr)
                print("  That walks every torch / transformers / datasets",
                      file=sys.stderr)
                print("  import in isolation and prints the LAST step that",
                      file=sys.stderr)
                print("  succeeded before the crash. The fix is almost",
                      file=sys.stderr)
                print("  always one of: re-run setup_gpu.ps1 to reinstall a",
                      file=sys.stderr)
                print("  matching torch wheel; pip install -U datasets;",
                      file=sys.stderr)
                print("  switch to --backend mock for local CI.",
                      file=sys.stderr)
                print("=" * 72, file=sys.stderr)
            return rc
    elapsed = time.time() - t0
    print(f"\nAll {total} stages completed in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
