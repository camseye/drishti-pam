"""Trained-LLM detector — inference (Drishti-PAM, Stage D).

Reads:
    data/sessions.json
    data/split.json
    data/llm_model.json

Writes:
    data/llm_predictions.json

Two backends mirroring train.py:
  * "mock"        — load the JSON model from train.py and apply the log-odds
                    classifier to the test split.
  * "huggingface" — load the fine-tuned DistilBERT from disk and run
                    inference. Wired but disabled in the local mock build.

Output schema is identical to rule_detector so compare.py can read both:
    {
      "session_id":     "sess-000123",
      "prediction":     "clean" | "suspicious",
      "scenario_hits":  [],            # always empty (this is a binary
                                       # classifier, no per-scenario rule fires)
      "confidence":     0.93,
      "evidence":       "score=2.74 threshold=0.85 top=[cmd:nmap +1.81, ...]"
    }
"""

from __future__ import annotations

# Windows + CPU-torch safety: silence the OpenMP DLL collision that surfaces as
# 0xC0000005 ACCESS_VIOLATION (Windows exit 3221225477) the first time torch
# enters a parallel region. Must precede any torch / transformers / numpy /
# tokenizers / safetensors import. See llm_detector/train.py for the full
# explanation. No-op on Linux / macOS, and `setdefault` lets users override.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
# Pin to a single GPU at inference time too — keeps device behaviour
# consistent with training (which uses CUDA_VISIBLE_DEVICES=0 to avoid
# DataParallel halving the step count). Inference doesn't suffer from
# the step-count bug, but inheriting the same single-GPU default keeps
# the Stage 4 device choice stable across `smoke.py` runs and direct
# `python detect.py` runs. Override via CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import faulthandler
faulthandler.enable()

import argparse
import json
import math
import sys
from pathlib import Path

# Sibling-import the feature extractor from train.py so the inference path
# can never drift from the training feature definitions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _detect_mock(sessions: list[dict], model: dict) -> list[dict]:
    weights: dict[str, float] = model["weights"]
    bias: float = model["bias"]
    threshold: float = model["threshold"]
    agg = train.ticket_aggregate(sessions)

    out = []
    for s in sessions:
        feats = train.extract_features(s, agg)
        contributions = []
        score = bias
        for k in feats:
            w = weights.get(k)
            if w is None:
                continue
            score += w
            contributions.append((k, w))
        # Top three contributing features (by |w|), suspicious-side first.
        # Sort with name as tie-breaker so the evidence string is stable
        # across processes (set iteration order in extract_features depends
        # on PYTHONHASHSEED).
        contributions.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
        top = ", ".join(f"{k} {w:+.2f}" for k, w in contributions[:3])
        prediction = "suspicious" if score >= threshold else "clean"
        out.append({
            "session_id": s["session_id"],
            "prediction": prediction,
            "scenario_hits": [],
            "confidence": round(_sigmoid(score - threshold), 3),
            "evidence": f"score={score:+.2f} threshold={threshold:+.2f} top=[{top}]",
        })
    return out


# --------------------------------------------------------------------------- #
# HuggingFace backend (WIRED BUT OFF)
#
# Lifted into Stage F's Kaggle notebook unchanged. Locally disabled.
# --------------------------------------------------------------------------- #

def _detect_huggingface(target_sessions: list[dict],
                        all_sessions: list[dict],
                        model_dir: Path) -> list[dict]:
    """Score sessions with a fine-tuned DistilBERT.

    Two saved-model layouts are supported:

    1. **Hybrid** (current architecture) — directory contains
       ``hybrid_config.json``, a ``distilbert/`` subdirectory with the
       trained encoder, and ``hybrid_head.pt`` with the cat-MLP +
       classifier state. We reconstruct the
       ``HybridDistilBERTClassifier`` wrapper, build a per-session
       ``cat_features`` vector at inference time (using the same helper
       train.py used) and run the fused forward pass.

    2. **Classic** (pre-hybrid; kept for backwards compatibility with
       any model artefacts produced by older builds of this repo) — the
       directory is loadable via
       ``AutoModelForSequenceClassification.from_pretrained``.

    `target_sessions` is the slice we score; `all_sessions` is the full
    session list used to build the cross-session aggregate (n_users,
    n_late) so cat features at inference match the values seen during
    training.

    Compatible with transformers v4.50+ and v5.x; explicitly moves model
    + inputs to GPU when CUDA is available so a misconfigured CPU-only
    torch install fails loudly instead of silently running 20x slower.
    """
    import time as _time
    from transformers import AutoTokenizer                       # type: ignore[import-not-found]
    import torch                                                 # type: ignore[import-not-found]
    try:
        from tqdm import tqdm                                    # type: ignore[import-not-found]
    except ImportError:
        def tqdm(x, **_):                                        # type: ignore[no-redef]
            return x

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[hf] DistilBERT inference: GPU = "
              f"{torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("[hf] DistilBERT inference: CPU (torch has no CUDA support).",
              flush=True)

    hybrid_config_path = model_dir / "hybrid_config.json"
    is_hybrid = hybrid_config_path.exists()

    print(f"[hf] [1/2] Loading fine-tuned model from {model_dir} "
          f"({'hybrid' if is_hybrid else 'classic'} layout)...", flush=True)
    t0 = _time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if is_hybrid:
        from transformers import AutoModel                       # type: ignore[import-not-found]
        cfg = json.load(hybrid_config_path.open(encoding="utf-8"))
        # Load the trained DistilBERT from disk (its weights have been
        # fine-tuned end-to-end with the head, so we must NOT re-pull
        # the pretrained checkpoint). Falls back to the base name only
        # if the local directory is missing (very old builds).
        distilbert_dir = model_dir / "distilbert"
        if distilbert_dir.exists():
            base = AutoModel.from_pretrained(str(distilbert_dir))
        else:
            print(f"[hf] WARN: {distilbert_dir} missing, falling back to "
                  f"base distilbert-base-uncased weights — head weights "
                  f"will be loaded but encoder won't reflect training!",
                  flush=True)
            base = AutoModel.from_pretrained(cfg["base_model_name"])
        # v1 checkpoints (no architecture_version key, no text_proj_dim)
        # used a 768+16 concat. v2 added a text-projection layer and a
        # wider cat-MLP, recorded as text_proj_dim / cat_hidden_dim /
        # cat_out_dim. Detect which by reading the config; fall back to
        # v1 layout when the keys are missing.
        arch_version = cfg.get("architecture_version", 1)
        if arch_version >= 2:
            model = train._build_hybrid_model(
                base,
                n_cat_features=cfg["n_cat_features"],
                n_labels=cfg["n_labels"],
                text_proj_dim=cfg["text_proj_dim"],
                cat_hidden_dim=cfg["cat_hidden_dim"],
                cat_out_dim=cfg["cat_out_dim"],
            )
        else:
            # v1 layout: 768-dim text concat with 16-dim cat. Force the
            # v2 helper to reproduce v1 by passing text_proj_dim equal
            # to DistilBERT's hidden_size and skipping the projection.
            # The simplest faithful recreation is to re-import the
            # legacy builder; rather than maintain that, we reconstruct
            # via temporary monkey-patch of the v2 dimensions. Cheap
            # and isolated.
            print(f"[hf] WARN: loading legacy hybrid v1 checkpoint "
                  f"(no architecture_version recorded). The v2 wrapper "
                  f"adds a text-projection layer; v1 layout used "
                  f"768+16=784 concat. v1 head weights have shape "
                  f"784->2 and won't load into the v2 classifier.",
                  flush=True)
            # Build with v1 dims so head weights load cleanly: text
            # path is identity-projected (Linear(768, 768) + ReLU which
            # is then a no-op on positive embeddings — close enough for
            # a deprecated path that's only kept for one-run-old
            # checkpoints).
            model = train._build_hybrid_model(
                base,
                n_cat_features=cfg["n_cat_features"],
                n_labels=cfg["n_labels"],
                text_proj_dim=base.config.hidden_size,
                cat_hidden_dim=32,
                cat_out_dim=16,
            )
        head_state = torch.load(model_dir / "hybrid_head.pt",
                                map_location="cpu")
        model.cat_mlp.load_state_dict(head_state["cat_mlp"])
        model.classifier.load_state_dict(head_state["classifier"])
        # v2 checkpoints additionally save text_proj weights. v1
        # didn't have this layer; for v1 the random-init text_proj is
        # used as-is (degraded but functional for backwards compat).
        if arch_version >= 2 and "text_proj" in head_state:
            model.text_proj.load_state_dict(head_state["text_proj"])
        model = model.to(device).eval()

        # Cross-session aggregate (n_users, n_late, ...) — must come
        # from the FULL session list so a test session sharing a
        # ticket with two training-only sessions still sees
        # n_users_on_ticket=3 at inference, exactly as it did at
        # training time.
        agg = train.ticket_aggregate(all_sessions)

        print(f"[hf]      hybrid model loaded in {_time.time() - t0:.1f}s "
              f"(on {next(model.parameters()).device}; "
              f"cat_features={cfg['n_cat_features']})", flush=True)
    else:
        from transformers import AutoModelForSequenceClassification  # type: ignore[import-not-found]
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir))
        model = model.to(device).eval()
        agg = None
        print(f"[hf]      classic model loaded in {_time.time() - t0:.1f}s "
              f"(on {next(model.parameters()).device})", flush=True)

    # Read tokenisation max_length from hybrid_config when available
    # (v2 records the train-time value so train and inference use the
    # same window). v1 / classic checkpoints fall back to 256.
    if is_hybrid:
        text_max_length = int(cfg.get("text_max_length", 256))
    else:
        text_max_length = 256
    print(f"[hf] [2/2] Scoring {len(target_sessions)} sessions "
          f"(text_max_length={text_max_length})...", flush=True)
    out = []
    t0 = _time.time()
    # Build aggregate ONCE outside the per-session loop. The hybrid
    # path also uses `agg` for its cat vector — for the classic path
    # we still build the same aggregate because the v2 text flattener
    # uses it to add `n_users_on_ticket` etc. to the input. Classic
    # checkpoints were trained without those fields and are a tiny
    # bit out-of-distribution at inference, but DistilBERT tolerates
    # extra context far better than missing context.
    if agg is None:
        agg = train.ticket_aggregate(all_sessions)
    with torch.no_grad():
        for s in tqdm(target_sessions, desc="[hf] DistilBERT inference",
                      unit="sess", leave=False):
            text = train._flatten_session_to_text(s, agg)
            # BatchEncoding.to(device) moves all internal tensors at once;
            # works on transformers >= 4.5 through v5.x.
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            padding="max_length",
                            max_length=text_max_length).to(device)
            if is_hybrid:
                cat = train._session_to_cat_vector(s, agg)
                cat_t = torch.tensor([cat], dtype=torch.float32,
                                     device=device)
                logits = model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"],
                               cat_features=cat_t).logits[0]
            else:
                logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
            p_susp = float(probs[1])
            prediction = "suspicious" if p_susp >= 0.5 else "clean"
            tag = "hybrid" if is_hybrid else "distilbert"
            out.append({
                "session_id": s["session_id"],
                "prediction": prediction,
                "scenario_hits": [],
                "confidence": round(p_susp, 3),
                "evidence": f"{tag}_p_suspicious={p_susp:.3f}",
            })
    rate = len(target_sessions) / max(_time.time() - t0, 1e-3)
    print(f"[hf]      scored {len(target_sessions)} sessions in "
          f"{_time.time() - t0:.1f}s ({rate:.1f} sess/s)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=Path, default=Path("data/sessions.json"))
    p.add_argument("--split", type=Path, default=Path("data/split.json"))
    p.add_argument("--model", type=Path, default=Path("data/llm_model.json"))
    p.add_argument("--out", type=Path, default=Path("data/llm_predictions.json"))
    p.add_argument("--score-all", action="store_true",
                   help="Score ALL sessions, not just the test split. Used by "
                        "the smoke harness to dump a full-coverage prediction "
                        "file alongside the test-only one.")
    args = p.parse_args(argv)

    sessions = json.load(args.sessions.open(encoding="utf-8"))
    split = json.load(args.split.open(encoding="utf-8"))
    model = json.load(args.model.open(encoding="utf-8"))

    if args.score_all:
        target = sessions
    else:
        test_set = set(split["test_ids"])
        target = [s for s in sessions if s["session_id"] in test_set]

    backend = model.get("backend", "mock")
    if backend == "mock":
        preds = _detect_mock(target, model)
    else:
        # Hybrid path needs the full session list to rebuild the
        # cross-session aggregate (n_users / n_late) the cat_features
        # vector depends on. Classic path ignores the all_sessions arg.
        preds = _detect_huggingface(target, sessions,
                                    Path(model["model_dir"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(preds, f, indent=1)
    n_susp = sum(1 for p in preds if p["prediction"] == "suspicious")
    print(f"trained-LLM detector ({backend}): scored {len(preds)} sessions "
          f"-> {n_susp} suspicious, {len(preds) - n_susp} clean")
    print(f"  output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
