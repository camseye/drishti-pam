"""LLM-driven data augmentation for the Drishti-PAM generator (Stage B).

Two roles:
  1. Justification paraphraser — rewrites a ticket's business_justification
     so that two tickets built from the same template do not share verbatim
     text. Real change records are written by humans; every author phrases
     things slightly differently, and detectors should not be allowed to
     latch onto a closed set of canonical sentences.
  2. Command-pool expander — given a base list of commands for a surface,
     produces additional plausible variants. Widens the command vocabulary so
     the trained-LLM detector (Stage D) cannot just memorise a closed token
     set.

Two backends:
  * "mock"        (default) — deterministic, runs offline, no model loaded.
  * "huggingface" — real Flan-T5-base via `transformers`. Requires `torch`
                    and ~1 GB model download. The code is shipped verbatim
                    so it can be lifted into the Kaggle notebook unchanged,
                    but it is DISABLED in the local mock-only build.

The mock backend is intentionally simple: synonym substitution and sentence
reordering for justifications, surface-only mutations (inline comments,
whitespace, echo wrappers) for commands. It is enough to demonstrate the
augmentation hook is wired correctly and to introduce real variation into
the dataset. For genuinely fluent paraphrases use the HF backend; that
runs in `notebooks/cloud_run.ipynb` (Step 3 on Kaggle), not on the local
machine.

Determinism: the mock paraphraser seeds its RNG from sha256(text + key), so
two calls with the same arguments are byte-identical, and two calls with the
same template but different `key` (ticket id) produce different output.
"""

from __future__ import annotations

# Windows + CPU-torch safety: silence the OpenMP DLL collision that surfaces as
# 0xC0000005 ACCESS_VIOLATION (Windows exit 3221225477) the first time torch
# enters a parallel region. Must precede any torch / transformers / numpy /
# tokenizers / safetensors import. The mock backend never loads torch so this
# is irrelevant to the local default build, but the HF backend lazy-loads
# torch inside `_hf_setup()` and would crash without these env vars on a
# Windows + CPU torch box.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
# Single-GPU default for the paraphrase model (Flan-T5-base / large).
# Inference doesn't suffer the DataParallel step-count bug, but keeping
# it pinned to one GPU matches the training-stage default and avoids
# confusing per-stage device-count divergence in the logs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import faulthandler
faulthandler.enable()

import hashlib
import random
from typing import Callable

# Default backend. Local development uses "mock"; the Kaggle notebook flips
# this to "huggingface". Tests can also pass a backend to Augmenter() to
# override the module-level default.
BACKEND = "mock"


# --------------------------------------------------------------------------- #
# Mock JUSTIFICATION paraphraser
# --------------------------------------------------------------------------- #

# Small synonym map. Pure surface-form swaps that don't change meaning. The
# list per key includes the original word so swap-with-self is allowed (which
# leaves a phrase unchanged with non-zero probability).
SYNONYMS: dict[str, list[str]] = {
    "Routine":     ["Scheduled", "Standard", "Planned", "Routine"],
    "routine":     ["scheduled", "standard", "planned", "routine"],
    "Investigate": ["Triage", "Inspect", "Review", "Investigate"],
    "investigate": ["triage", "inspect", "review", "investigate"],
    "Restart":     ["Recycle", "Bounce", "Cycle", "Restart"],
    "restart":     ["recycle", "bounce", "cycle", "restart"],
    "Update":      ["Refresh", "Apply", "Rotate", "Update"],
    "update":      ["refresh", "apply", "rotate", "update"],
    "Renew":       ["Rotate", "Reissue", "Refresh", "Renew"],
    "renew":       ["rotate", "reissue", "refresh", "renew"],
    "review":      ["audit", "inspect", "examine", "review"],
    "Review":      ["Audit", "Inspect", "Examine", "Review"],
    "fix":         ["resolve", "correct", "remediate", "fix"],
    "Fix":         ["Resolve", "Correct", "Remediate", "Fix"],
    "Rollback":    ["Backout", "Revert", "Rollback"],
    "Backout":     ["Rollback", "Revert", "Backout"],
    "Window":      ["Change window", "Approved window", "Window"],
    "approved":    ["sanctioned", "authorised", "approved"],
    "Approved":    ["Sanctioned", "Authorised", "Approved"],
    "limited":     ["scoped", "restricted", "limited"],
    "verified":    ["validated", "confirmed", "verified"],
    "after":       ["following", "post", "after"],
    "before":      ["prior to", "ahead of", "before"],
    "Add":         ["Provision", "Create", "Add"],
    "Provision":   ["Add", "Create", "Provision"],
    "Diagnostic":  ["Investigative", "Read-only", "Diagnostic"],
    "diagnostic":  ["investigative", "read-only", "diagnostic"],
}

# Optional opener / closer prepended or appended to the paraphrased text.
# Empty strings are deliberately included so most justifications stay short.
OPENERS = ["", "", "", "Per runbook, ", "As approved by CAB, ",
           "Per change record, ", "Per the change ticket, "]
CLOSERS = ["", "", "", " End of justification.", " Refer to the linked runbook.",
           " Backout plan attached."]


def _content_seed(*parts: str) -> int:
    """Deterministic 32-bit seed derived from concatenated string parts."""
    h = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def _mock_paraphrase(text: str, key: str,
                     max_length: int | None = None) -> str:
    """Deterministic surface-level paraphrase. Not fluent, but varied enough
    that two callers passing the same template produce different output as
    long as their `key` (ticket_id) differs."""

    rng = random.Random(_content_seed(text, key))

    # Synonym substitution, word-by-word. Punctuation is preserved by
    # splitting trailing dot / comma and re-attaching after the lookup.
    out_words: list[str] = []
    for w in text.split(" "):
        suffix = ""
        base = w
        while base and base[-1] in ".,;:":
            suffix = base[-1] + suffix
            base = base[:-1]
        candidates = SYNONYMS.get(base) or SYNONYMS.get(base.lower())
        if candidates and rng.random() < 0.6:
            chosen = rng.choice(candidates)
            # Preserve title-case of the original.
            if base[:1].isupper():
                chosen = chosen[:1].upper() + chosen[1:]
            out_words.append(chosen + suffix)
        else:
            out_words.append(w)
    body = " ".join(out_words)

    # Optional sentence reordering: swap two adjacent sentences with prob 0.25.
    # Strip trailing period from each sentence first so re-joining with ". "
    # cannot produce double-periods when the input already terminated each
    # clause with one.
    sents = [s.strip().rstrip(".") for s in body.split(". ") if s.strip()]
    if len(sents) >= 2 and rng.random() < 0.25:
        i = rng.randrange(len(sents) - 1)
        sents[i], sents[i + 1] = sents[i + 1], sents[i]
    body = ". ".join(sents) + "."

    opener = rng.choice(OPENERS)
    closer = rng.choice(CLOSERS)
    out = (opener + body + closer).strip()
    # Same defensive cap as the HF backend — fall back to the original input
    # rather than violating a length invariant.
    if max_length is not None and len(out) > max_length:
        return text
    return out


# --------------------------------------------------------------------------- #
# Mock COMMAND-POOL expander
# --------------------------------------------------------------------------- #

def _mock_expand_commands(commands: list[str], factor: int,
                          seed: int) -> list[str]:
    """Procedural expansion: for each input command, emit `factor` variants
    that preserve semantics. The original command tokens are preserved so
    destructive / sensitive / lateral / escalation signals survive untouched.

    Returns a deduplicated list, original commands first."""

    if factor < 1:
        return list(commands)
    rng = random.Random(seed)
    out: list[str] = list(commands)
    for cmd in commands:
        for _ in range(factor):
            variant = _vary_command(cmd, rng)
            if variant != cmd:
                out.append(variant)
    seen, dedup = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def _vary_command(cmd: str, rng: random.Random) -> str:
    """Produce one surface-only variant of a command. Mutations are limited to
    inline comments, echo wrappers, and whitespace; the original command text
    is never modified, so detectors keying off keywords (`DROP`, `/etc/shadow`,
    `nmap`, `sudo su -`) still see them."""

    # All mutations preserve the original command verbatim; they only ADD
    # wrapping context (inline comments, echo prefix, env-preserving sudo).
    # That keeps detector signals (`DROP`, `/etc/shadow`, `nmap`, `sudo su -`)
    # intact under substring matching.
    flavour = rng.random()
    if flavour < 0.30:
        step = rng.randint(1, 9)
        return f"{cmd}  # step {step}"
    if flavour < 0.55:
        check = rng.choice(["pre-check", "post-check", "approved", "validated"])
        return f"{cmd}  # {check}"
    if flavour < 0.80:
        msg = rng.choice(["running step", "executing", "applying change",
                          "starting"])
        return f"echo '{msg}' && {cmd}"
    # Preserve-environment sudo wrapper. Skip if the command doesn't start with
    # plain "sudo " — wrapping unrelated commands with -E would be nonsense.
    if cmd.startswith("sudo ") and " -E " not in cmd[:10]:
        return cmd.replace("sudo ", "sudo -E ", 1)
    return f"{cmd}  # routine"


# --------------------------------------------------------------------------- #
# HuggingFace backend (real model, lazy-loaded)
#
# Active when BACKEND == "huggingface" or when Augmenter(backend="huggingface")
# is constructed. Uses `AutoModelForSeq2SeqLM` directly — the older
# `pipeline("text2text-generation", ...)` API was removed in transformers v5.0
# (Jan 2026). Direct model usage is forwards-compatible with v5+ and works
# unchanged on v4.x.
#
# Local install:
#   pip install -U "transformers>=5.5" "torch>=2.7" "datasets>=3" "accelerate>=1"
# For GPU (Windows / CUDA 12.6):
#   pip uninstall -y torch
#   pip install torch --index-url https://download.pytorch.org/whl/cu126
# --------------------------------------------------------------------------- #

_HF_TOKENIZER = None
_HF_MODEL = None
_HF_DEVICE = None
# One-shot flag: print the decode strategy on the first batched call only.
# Resets per process (subprocess.run children get their own).
_DECODE_STRATEGY_LOGGED = False


def _hf_setup() -> None:
    """Lazy-load the paraphrase model on the best available device.
    Idempotent.

    Model selection: defaults to ``google/flan-t5-base`` (250M params,
    ~990 MB download). Override via ``PAM_HF_PARAPHRASE_MODEL`` if base
    proves too small for your justifications — Flan-T5-large (770M)
    fits comfortably on a T4 at fp16 and produces noticeably more
    fluent paraphrases. The download size scales linearly with the
    parameter count.
    """
    global _HF_TOKENIZER, _HF_MODEL, _HF_DEVICE
    if _HF_MODEL is not None:
        return

    import time
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore[import-not-found]
    import torch                                                    # type: ignore[import-not-found]

    # Hugging Face login (if a token is available) — speeds up downloads
    # and avoids anonymous rate limits. No-op if the user has no token.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    try:
        import hf_token                                              # type: ignore[import-not-found]
        _hf_tok = hf_token.setup_token(verbose=True)
    except ImportError:
        _hf_tok = None

    model_name = os.environ.get("PAM_HF_PARAPHRASE_MODEL",
                                "google/flan-t5-base")

    if torch.cuda.is_available():
        _HF_DEVICE = torch.device("cuda")
        dtype = torch.float16
        print(f"[hf] {model_name}: GPU = {torch.cuda.get_device_name(0)}, "
              f"dtype = float16", flush=True)
    else:
        _HF_DEVICE = torch.device("cpu")
        dtype = torch.float32
        print(f"[hf] {model_name}: CPU only — torch has no CUDA support. "
              "Generator will be ~10-30x slower than on GPU. To enable GPU: "
              "pip uninstall -y torch && pip install torch --index-url "
              "https://download.pytorch.org/whl/cu126", flush=True)

    print(f"[hf] [1/2] Downloading / loading {model_name} tokenizer "
          "(~few seconds, first run pulls ~5 MB)...", flush=True)
    t0 = time.time()
    tok_kwargs = {"token": _hf_tok} if _hf_tok else {}
    _HF_TOKENIZER = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    print(f"[hf]      tokenizer ready in {time.time() - t0:.1f}s", flush=True)

    print(f"[hf] [2/2] Downloading / loading {model_name} model "
          "(first run pulls ~990 MB for flan-t5-base; subsequent runs "
          "reuse the HF cache)...", flush=True)
    t0 = time.time()
    # `dtype=` is the modern kwarg (transformers >= 4.50); on older builds
    # it falls through to the legacy `torch_dtype=` via aliasing.
    model_kwargs: dict = {**tok_kwargs}
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype, **model_kwargs)
    _HF_MODEL = model.to(_HF_DEVICE).eval()
    print(f"[hf]      model ready in {time.time() - t0:.1f}s "
          f"(on {_HF_DEVICE})", flush=True)


def _hf_generate(prompt: str, max_new_tokens: int = 60) -> str:
    """Single-prompt convenience wrapper around `_hf_generate_batch`. Kept
    for back-compat; new call sites should batch."""
    return _hf_generate_batch([prompt], max_new_tokens=max_new_tokens)[0]


def _hf_generate_batch(prompts: list[str], max_new_tokens: int = 60,
                       batch_size: int | None = None) -> list[str]:
    """Decode from the paraphrase model in batched mode.

    On a Kaggle T4 with ~7 500 unique justifications, batching is the
    difference between the pipeline finishing in ~20 minutes vs ~6 hours
    — single-prompt calls leave the GPU 95 % idle waiting for Python
    overhead and per-call CUDA sync. Default batch_size of 16 fits
    comfortably in T4 memory at max_new_tokens=80; override with the
    `PAM_HF_BATCH_SIZE` env var if needed (smaller for OOM, larger for
    Ampere+).

    Decode strategy
    ---------------
    Defaults to **nucleus sampling** (``do_sample=True, top_p=0.9,
    temperature=0.8``). Greedy decoding (``do_sample=False, num_beams=1``)
    on Flan-T5-family models has a strong "copy the input" attractor for
    out-of-distribution domains like IT change-management text — at
    every position the highest-probability next token is the input
    token at that position, so the model deterministically lands on a
    verbatim copy. We saw this in production: ~56 % of paraphrases
    fell into the ``echoed`` / ``substring_copy`` / ``prefix_copy``
    buckets even with the T5-native ``paraphrase: <text>`` prompt and
    even after switching to ``flan-t5-large``. Sampling forces the
    decoder off the highest-probability path; alternative paraphrase
    tokens with non-trivial probability now have a real chance of being
    selected, breaking the copy attractor.

    Reproducibility: ``torch.manual_seed`` is set before every batched
    invocation, so two runs of the same dataset on the same hardware /
    library versions produce byte-identical paraphrases. Across hardware
    or torch versions, exact byte-equality is not guaranteed but the
    bucket distribution stays stable.

    Tuning knobs (env vars):
      * ``PAM_HF_SAMPLE``       — ``true`` / ``false`` (default ``true``).
        Set to ``false`` to revert to greedy decoding.
      * ``PAM_HF_TOP_P``        — nucleus cut-off (default ``0.9``). Use
        ``0.95`` for more variation, ``0.85`` for more conservative.
      * ``PAM_HF_TEMPERATURE``  — softmax shape (default ``0.8``). Higher
        = more diverse / risk of hallucination; lower = closer to greedy.
      * ``PAM_HF_SEED``         — RNG seed (default ``42``).

    Returns one decoded string per input prompt, in the same order.
    """
    import os
    import torch                                                    # type: ignore[import-not-found]
    try:
        from tqdm import tqdm                                       # type: ignore[import-not-found]
    except ImportError:
        def tqdm(x, **_): return x                                  # type: ignore[no-redef]

    _hf_setup()

    if batch_size is None:
        batch_size = int(os.environ.get("PAM_HF_BATCH_SIZE", "16"))

    # Decode-strategy configuration. Read once per call so env-var changes
    # at runtime (e.g. between two notebook cells) are honoured.
    do_sample = os.environ.get("PAM_HF_SAMPLE", "true").lower() in (
        "1", "true", "yes", "on")
    top_p = float(os.environ.get("PAM_HF_TOP_P", "0.9"))
    temperature = float(os.environ.get("PAM_HF_TEMPERATURE", "0.8"))
    seed = int(os.environ.get("PAM_HF_SEED", "42"))

    # Validate sampling knobs — bad values silently break decoding.
    if do_sample:
        if temperature <= 0:
            print(f"[hf] WARNING: PAM_HF_TEMPERATURE={temperature} ≤ 0 is "
                  f"invalid; clamping to 0.8.", flush=True)
            temperature = 0.8
        if not (0.0 < top_p <= 1.0):
            print(f"[hf] WARNING: PAM_HF_TOP_P={top_p} out of (0, 1]; "
                  f"clamping to 0.9.", flush=True)
            top_p = 0.9

    # One-shot decode-strategy log line. Useful when comparing two runs
    # (the user can grep `[hf] decode strategy:` in the logs).
    global _DECODE_STRATEGY_LOGGED
    if not _DECODE_STRATEGY_LOGGED:
        if do_sample:
            print(f"[hf] decode strategy: sampling(top_p={top_p}, "
                  f"temperature={temperature}, seed={seed})", flush=True)
        else:
            print(f"[hf] decode strategy: greedy (do_sample=False, "
                  f"num_beams=1) — note: greedy decoding on Flan-T5 is "
                  f"prone to verbatim-copy outputs on out-of-distribution "
                  f"text. Set PAM_HF_SAMPLE=true to enable nucleus "
                  f"sampling (recommended).", flush=True)
        _DECODE_STRATEGY_LOGGED = True

    # Seed BOTH CPU and CUDA generators. CPU seed governs tokenizer non-
    # determinism (none currently, but safe); CUDA seed governs the
    # actual sampling RNG. Seeding once per call guarantees determinism
    # across re-runs of the same call AND across two calls in the same
    # process (so the deferred-flush pass produces identical output if
    # invoked twice with the same prompts).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out: list[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    iterator = range(0, len(prompts), batch_size)
    if n_batches > 1:
        iterator = tqdm(list(iterator), desc=f"[hf] generate (bs={batch_size})",
                        unit="batch", leave=False)

    for i in iterator:
        chunk = prompts[i:i + batch_size]
        # `padding=True` pads to the longest prompt in this chunk; the
        # tokenizer returns BatchEncoding which `.to(device)` ports en bloc.
        inputs = _HF_TOKENIZER(chunk, return_tensors="pt", truncation=True,
                               padding=True, max_length=512).to(_HF_DEVICE)
        # Build kwargs conditionally — passing top_p / temperature when
        # do_sample=False makes transformers v5 emit a confusing warning.
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_p"] = top_p
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            ids = _HF_MODEL.generate(**inputs, **gen_kwargs)
        decoded = _HF_TOKENIZER.batch_decode(ids, skip_special_tokens=True)
        out.extend(s.strip() for s in decoded)
    return out


def _build_paraphrase_prompt(text: str, max_length: int | None) -> str:
    """Single source of truth for the paraphrase prompt template.
    Used by both the single-call path and the deferred-batch flush so
    the cached HF outputs are byte-comparable across the two modes.

    T5-native prefix
    ----------------
    The previous "Paraphrase this... Original: <text> Paraphrase:" form
    triggered prompt-echo behaviour on Flan-T5-base: the model was
    instruction-tuned to recognise "Original:" / "Paraphrase:" as
    labels, so given a line that ended in "Paraphrase:" it copied the
    prior "Original: ..." line back verbatim as if completing the
    structure. On a Kaggle T4 this produced near-zero useful paraphrases
    and corrupted business_justification across the entire dataset —
    the trainer then learned nothing and Stage 5's llm-trained F1
    collapsed to 0.0.

    The T5 family was pretrained on the canonical "paraphrase: <text>"
    prefix (see Raffel et al. 2020 §3 — task prefixes). Sticking to that
    form removes the labels the model can echo, and Flan-T5-base
    paraphrases naturally without needing a multi-line instruction.
    `max_length` is no longer encoded in the prompt (the model ignored
    it anyway); it is enforced post-hoc in `_classify_paraphrase`.
    """
    return f"paraphrase: {text}"


def _classify_paraphrase(paraphrase: str, original: str,
                         max_length: int | None) -> str:
    """Classify a Flan-T5 paraphrase output into a quality bucket.

    Returns one of:
      * ``kept``           — usable; substitute for the original.
      * ``empty``          — model returned nothing (decode bug / over-
        truncation).
      * ``too_short``      — < 10 chars; not enough signal.
      * ``too_long``       — exceeds the optional ``max_length`` cap.
      * ``echoed``         — exact case-insensitive copy of the input.
      * ``prompt_leak``    — output starts with a label like
        ``Original:`` / ``Paraphrase:`` / ``Rewrite:``. The model copied
        our prompt structure instead of paraphrasing. The T5-native
        prompt should never trigger this, but smaller / older Flan-T5
        builds occasionally still do — flagging it as garbage is the
        canary that catches the v1 zero-F1 bug if it returns.
      * ``substring_copy`` — output contains the input verbatim (model
        wrapped a prefix / suffix around the original), OR output is
        itself a substring of the input (model returned just a
        fragment). Either way the dataset gets no real variation.
      * ``prefix_copy``    — first 30 chars of output match the first
        30 chars of input. Catches the truncated-copy case where
        ``max_new_tokens`` cuts the output before either-direction
        containment can match.

    Every non-``kept`` bucket falls back to the original template at
    the call site, so a fully-failed Flan-T5 run produces the same
    dataset shape as the mock backend (no variation, but also no
    corruption).
    """
    p = paraphrase.strip()
    o = original.strip()
    if not p:
        return "empty"
    if len(p) < 10:
        return "too_short"
    if max_length is not None and len(p) > max_length:
        return "too_long"

    p_lower = p.lower()
    o_lower = o.lower()

    # Prompt-structure leakage — the v1 zero-F1 bug.
    for marker in ("original:", "paraphrase:", "rewrite:", "rephrase:",
                   "text:", "input:", "output:", "answer:"):
        if p_lower.startswith(marker):
            return "prompt_leak"

    if p_lower == o_lower:
        return "echoed"

    # Either-direction containment.
    if o_lower in p_lower or p_lower in o_lower:
        return "substring_copy"

    # Truncated-prefix copy: 30 chars (~8 T5 tokens) is well past chance
    # of accidental overlap on a real paraphrase.
    HEAD = 30
    if len(p_lower) >= HEAD and len(o_lower) >= HEAD:
        if p_lower[:HEAD] == o_lower[:HEAD]:
            return "prefix_copy"

    return "kept"


def _hf_paraphrase(text: str, key: str, max_length: int | None = None) -> str:
    """Real Flan-T5-base paraphrase. Active when BACKEND == 'huggingface'.

    `max_length`, when set, both (a) hints the model in the prompt and (b)
    enforces a post-hoc fallback to the original text if the model
    overshoots. This future-proofs any scenario that adds a length
    invariant — formal English tends to expand short text and Flan-T5 is
    no exception."""
    max_new_tokens = int(os.environ.get("PAM_HF_MAX_NEW_TOKENS", "80"))
    para = _hf_generate(_build_paraphrase_prompt(text, max_length),
                        max_new_tokens=max_new_tokens)
    if max_length is not None and len(para) > max_length:
        return text
    return para


def _hf_expand_commands(commands: list[str], factor: int,
                        seed: int) -> list[str]:
    """Real Flan-T5-base command-pool expansion.

    The mock backend `_vary_command` ALWAYS preserves the original command
    verbatim (it only adds wrappers like `# step 3` or `echo '...' && ...`).
    The HF backend provides the same guarantee or the generator's
    self-check fires for any scenario whose invariant is "the command
    stream contains token X" (S7 destructive, S12 sensitive, S13 lateral,
    S14 escalation).

    Approach: ask Flan-T5 for a short *purpose phrase* describing the
    command, then wrap it as `echo '<phrase>' && <original cmd>`. The
    original command tokens (DROP, /etc/shadow, nmap, sudo su -, ...) are
    preserved verbatim in every variant, and the rule detector's
    `_command_strip_wrappers` already knows how to peel `echo '...' && ...`
    off when matching against the in-scope command pool.

    Batched: a single call to `_hf_generate_batch` covers every (cmd,
    variant) pair, turning ~200 sequential GPU calls into ~13 batched
    forward passes."""
    expanded: list[str] = list(commands)
    pairs = [(cmd, i) for cmd in commands for i in range(factor)]
    if not pairs:
        return expanded
    prompts = [
        ("Write one short phrase (under 8 words) describing the IT "
         "operations purpose of this command. Reply with just the "
         f"phrase, no quotes, no commands.\n\nCommand: {cmd}\nPhrase:")
        for cmd, _ in pairs
    ]
    print(f"[hf] expanding command pool: {len(prompts)} variants in batches",
          flush=True)
    phrases = _hf_generate_batch(prompts, max_new_tokens=20)

    for (cmd, _i), phrase in zip(pairs, phrases):
        phrase = phrase.rstrip(".").strip()
        # Defang quotes / newlines so the wrapped echo is shell-safe.
        phrase = phrase.replace("'", "").replace('"', "")
        phrase = " ".join(phrase.split())[:80]
        # Scrub destructive / data-exfil verbs from the purpose phrase
        # before wrapping. Without this, Flan-T5 phrases like "delete
        # temporary files" leaked tokens into the command stream of
        # CLEAN sessions, firing rule_detector's S7 destructive check
        # (caused 27 false positives on the May-10 cloud run) and
        # adding noise to the LLM's training signal. The replacement
        # word "manage" is intentionally vague — it preserves textual
        # variety for DistilBERT without sneaking destructive keywords
        # into otherwise-clean command text.
        phrase = _DESTRUCTIVE_VERBS_RE.sub("manage", phrase)
        if not phrase:
            phrase = "operational task"
        variant = f"echo '{phrase}' && {cmd}"
        if variant not in expanded:
            expanded.append(variant)
    return expanded


# Verbs that the rule_detector keys off when it scans command text for
# destructive / mass-export operations. If Flan-T5's purpose phrase
# contains any of these, the wrap (echo '...' && cmd) leaks the verb
# into the command stream and triggers S7 false positives in clean
# sessions. Used by `_hf_expand_commands` only.
_DESTRUCTIVE_VERBS_RE = __import__("re").compile(
    r"\b(delete|drop|truncate|remove|removing|removes|removed|"
    r"purge|purges|purged|kill|kills|killed|"
    r"dump|dumps|dumping|outfile|"
    r"export|exports|exporting|exported|"
    r"alter|alters|altering|altered|"
    r"erase|erases|erasing|erased|"
    r"destroy|destroys|destroying|destroyed|"
    r"wipe|wipes|wiping|wiped)\b",
    __import__("re").IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

class Augmenter:
    """Single entry point for the generator. Caches paraphrases by
    (text, key, max_length) so repeated lookups for the same ticket are
    byte-identical, and repeated runs of the generator with the same seed
    produce the same dataset.

    HuggingFace deferred mode
    -------------------------
    The HF backend is roughly 5 000× slower per call than the mock backend
    (greedy decode of ~120 tokens on Flan-T5-base, batch=1, plus Python
    overhead). Calling it inline once per ticket would mean ~7 500
    sequential GPU calls and a wall-clock of 4-6 hours on a Kaggle T4.

    To avoid that, the HF backend operates in DEFERRED mode:
      * `paraphrase_justification(text, key)` records the request and
        returns the original text immediately. Sessions are emitted with
        the raw template as their `business_justification`.
      * After emission, the generator calls
        `flush_pending_paraphrases()`. That runs every recorded request
        through `_hf_generate_batch` with batch_size=16 (override via
        the `PAM_HF_BATCH_SIZE` env var) and populates the cache.
      * The generator then re-walks `sessions` and substitutes each
        `business_justification` from the cache.

    Batching turns 7 500 sequential GPU calls into ~470 batched forward
    passes — wall-clock drops from 4-6 hours to ~15-25 minutes on T4.

    The mock backend has no equivalent — `_mock_paraphrase` runs in
    microseconds, so deferring would be pointless and adds nothing to
    the byte-identical mock-pipeline reproducibility guarantee."""

    def __init__(self, backend: str | None = None):
        self.backend = backend or BACKEND
        self._cache: dict[tuple[str, str, int | None], str] = {}
        # HF-only deferred-call queue. Mock backend never appends.
        self._hf_pending: list[tuple[str, str, int | None]] = []

    def paraphrase_justification(self, text: str, key: str,
                                 max_length: int | None = None) -> str:
        """Paraphrase one justification.

        `max_length`, when set, guarantees the output is no longer than
        `max_length` characters. Both backends fall back to the original
        text if the model overshoots, so callers that care about a length
        invariant (e.g. the S2 vague-justification rule's 40-char cutoff)
        can rely on this contract regardless of which backend is active.

        HF backend: deferred — see class docstring."""
        ck = (text, key, max_length)
        if ck in self._cache:
            return self._cache[ck]
        if self.backend == "huggingface":
            # Deferred: record the call, return the original text. The
            # generator must call flush_pending_paraphrases() and then
            # rewrite session justifications from the cache to actually
            # see HF output.
            self._hf_pending.append(ck)
            self._cache[ck] = text
            return text
        out = _mock_paraphrase(text, key, max_length=max_length)
        self._cache[ck] = out
        return out

    def flush_pending_paraphrases(self) -> int:
        """Run all deferred HF paraphrase calls in batched mode and
        replace each pending cache entry with the real HF paraphrase.
        No-op for the mock backend (the queue is always empty there).

        max_new_tokens defaults to 80 (down from 120). Justifications
        average ~50-100 chars, ~15-30 T5 tokens, so 80 is generous;
        but each batched forward pass blocks on the LONGEST sequence,
        so capping the upper bound shaves ~20-30 % off Stage 1 wall-
        clock when the GPU is already saturated (the 99 %-utilised local
        case). Override with `PAM_HF_MAX_NEW_TOKENS` if you ever see
        truncated outputs (minimum enforced: 16).

        Defensive output filtering
        --------------------------
        Flan-T5 occasionally produces empty / whitespace-only / echo-the-
        prompt outputs, especially on (a) older transformers (<5.0) that
        miss decode-time fixes, (b) PAM_HF_MAX_NEW_TOKENS too low, or
        (c) Kaggle's preinstalled-transformers fallback path. If we
        accept those silently, every session's business_justification
        becomes garbage — the trainer learns nothing and Stage 5's
        llm-trained F1 collapses to zero. We therefore validate every
        paraphrase: empty, < 10 chars, or "echoes the original" → fall
        back to the original template (cache key stays at the provisional
        original, so the substitute pass in generate.py is a no-op for
        those records). Counts of each fallback path are logged so the
        operator can see the data-quality picture in one line.

        Returns the number of unique HF generations performed (NOT the
        number of substitutions kept; see logs for the difference)."""
        if self.backend != "huggingface" or not self._hf_pending:
            return 0

        max_new_tokens = int(os.environ.get("PAM_HF_MAX_NEW_TOKENS", "80"))
        if max_new_tokens < 16:
            print(f"[hf] WARNING: PAM_HF_MAX_NEW_TOKENS={max_new_tokens} is "
                  f"too low (minimum 16). Bumping to 16. Justifications "
                  f"average ~25 T5 tokens; below 16 you'll truncate every "
                  f"sentence mid-word.", flush=True)
            max_new_tokens = 16
        elif max_new_tokens < 32:
            print(f"[hf] NOTE: PAM_HF_MAX_NEW_TOKENS={max_new_tokens} is "
                  f"aggressive — expect frequent mid-sentence truncation. "
                  f"Recommended: 64-128.", flush=True)

        # Sort for determinism — same input order across machines + runs.
        unique = sorted(set(self._hf_pending))
        prompts = [_build_paraphrase_prompt(text, max_length)
                   for text, _key, max_length in unique]
        print(f"[hf] flushing {len(unique)} deferred paraphrases "
              f"(batched, max_new_tokens={max_new_tokens})...", flush=True)
        results = _hf_generate_batch(prompts, max_new_tokens=max_new_tokens)

        # Eight quality buckets — see `_classify_paraphrase`. Each non-
        # "kept" bucket falls back to the original template, so the
        # dataset stays clean even if Flan-T5 returns garbage for every
        # call.
        buckets = {"kept": 0, "empty": 0, "too_short": 0, "too_long": 0,
                   "echoed": 0, "prompt_leak": 0, "substring_copy": 0,
                   "prefix_copy": 0}
        sample: list[tuple[str, str, str]] = []  # (status, original, paraphrase)
        for (text, key, max_length), paraphrase in zip(unique, results):
            status = _classify_paraphrase(paraphrase, text, max_length)
            buckets[status] += 1
            if status == "kept":
                self._cache[(text, key, max_length)] = paraphrase.strip()
            else:
                # Every non-kept bucket falls back to the original
                # template. The substitute pass in `generate.py` then
                # finds cache[ck] == text and skips the substitution
                # (counted as `n_skip_same`).
                self._cache[(text, key, max_length)] = text
            # Sample up to 3 failures + 3 successes so the operator can
            # eyeball quality without scrolling through 7500 lines.
            n_fail_in_sample = sum(1 for s, _, _ in sample if s != "kept")
            n_kept_in_sample = sum(1 for s, _, _ in sample if s == "kept")
            if status != "kept" and n_fail_in_sample < 3:
                sample.append((status, text, paraphrase))
            elif status == "kept" and n_kept_in_sample < 3:
                sample.append((status, text, paraphrase))

        n_kept = buckets["kept"]
        total = len(unique)
        kept_pct = (n_kept / total * 100.0) if total else 0.0
        # One-line stats — tells you in 120 chars whether the run is healthy.
        print(f"[hf]   kept={n_kept} ({kept_pct:.1f}%)  "
              f"empty={buckets['empty']}  too_short={buckets['too_short']}  "
              f"too_long={buckets['too_long']}  echoed={buckets['echoed']}  "
              f"prompt_leak={buckets['prompt_leak']}  "
              f"substring_copy={buckets['substring_copy']}  "
              f"prefix_copy={buckets['prefix_copy']}  total={total}",
              flush=True)

        # If MORE than half the paraphrases were rejected, something is
        # very wrong — Kaggle's stale transformers, MAX_NEW_TOKENS too low,
        # or the model crashed silently. Print a sample so the user can
        # diagnose, and emit a loud warning. We don't raise — the
        # generator can still continue with original templates and produce
        # a usable (if less varied) dataset.
        if n_kept < total // 2:
            print(f"[hf] WARNING: {total - n_kept}/{total} paraphrases "
                  f"rejected as bad. Falling back to original templates "
                  f"for those records. The dataset will lack paraphrase "
                  f"variation — DistilBERT will overfit to the small set "
                  f"of distinct templates and Stage 5's llm-trained F1 "
                  f"may be poor. Most likely causes:",
                  flush=True)
            # Targeted advice based on which bucket dominates.
            if buckets["prompt_leak"] > total // 4:
                print(f"[hf]   * prompt_leak={buckets['prompt_leak']} — "
                      f"Flan-T5 is echoing the prompt structure. This "
                      f"should not happen with the T5-native "
                      f"`paraphrase: <text>` prefix; check that "
                      f"_build_paraphrase_prompt was not patched and "
                      f"that transformers is >= 5.5.",
                      flush=True)
            if (buckets["substring_copy"] + buckets["prefix_copy"]
                    + buckets["echoed"]) > total // 3:
                # Most common cause of high copy buckets is greedy
                # decoding hitting the "copy the input" attractor — see
                # _hf_generate_batch docstring. The shipped default is
                # sampling, but if PAM_HF_SAMPLE was set to false (or
                # an old branch reverted to greedy), this is what you
                # see. Switching paraphrase model size does NOT fix
                # this — flan-t5-large under greedy decoding produced
                # the same ~34 % kept rate as flan-t5-base in
                # production.
                print(f"[hf]   * substring/prefix/echo copies dominate — "
                      f"this is the classic 'greedy-decoding copy "
                      f"attractor'. Verify your earlier `[hf] decode "
                      f"strategy:` log line says `sampling(...)`. If "
                      f"it says `greedy`, set "
                      f"`os.environ['PAM_HF_SAMPLE'] = 'true'` (or "
                      f"unset PAM_HF_SAMPLE) and re-run. If it ALREADY "
                      f"says sampling, raise PAM_HF_TEMPERATURE to 0.9 "
                      f"or 1.0 for more diversity, or switch to "
                      f"`PAM_HF_PARAPHRASE_MODEL=google/flan-t5-large`.",
                      flush=True)
            print(f"[hf]   * transformers < 5.0 on Kaggle (`pip install "
                  f"--upgrade transformers>=5.5,<6` in the notebook)",
                  flush=True)
            print(f"[hf]   * PAM_HF_MAX_NEW_TOKENS too low (current="
                  f"{max_new_tokens}; default is 80, raise to 128 if your "
                  f"justifications are long)", flush=True)
            print(f"[hf]   * `python doctor.py --with-download` to verify "
                  f"the model loads and runs end-to-end", flush=True)

        # Always print 3 samples so the user can sanity-check what HF is
        # producing. This is the single most useful debug aid we have.
        if sample:
            print(f"[hf] sample paraphrases (status | original -> output):",
                  flush=True)
            for status, orig, para in sample[:6]:
                o = (orig[:60] + "...") if len(orig) > 60 else orig
                p = (para[:60] + "...") if len(para) > 60 else para
                print(f"[hf]   [{status:9s}] {o!r}  ->  {p!r}", flush=True)

        self._hf_pending.clear()
        return len(unique)

    def expand_command_pool(self, commands: list[str], factor: int,
                            seed: int) -> list[str]:
        if self.backend == "huggingface":
            return _hf_expand_commands(commands, factor, seed)
        return _mock_expand_commands(commands, factor, seed)


# Module-level convenience for callers that don't want to construct an object.
DEFAULT = Augmenter()


def paraphrase(text: str, key: str) -> str:
    return DEFAULT.paraphrase_justification(text, key)


def expand_commands(commands: list[str], factor: int = 1,
                    seed: int = 0) -> list[str]:
    return DEFAULT.expand_command_pool(commands, factor, seed)
