"""Trained-LLM detector — training pipeline (Drishti-PAM, Stage D).

Two backends, same I/O contract:

  * "mock"        (default) — a deterministic, log-odds bag-of-features
                   classifier that runs in seconds on a CPU. No model
                   weights are downloaded; nothing leaves the machine.
                   Designed to be a believable "trained" competitor to the
                   rule detector with characteristically different errors:
                   it picks up on continuous signals like "two distinct
                   users on a shared ticket" that the rules deliberately
                   ignore.

  * "huggingface" — fine-tunes `distilbert-base-uncased` on the same
                    train split. The code is shipped verbatim so it can be
                    lifted into Stage F's Kaggle notebook unchanged. It is
                    DISABLED in the local mock build.

Inputs:
    data/sessions.json
    data/labels.json

Outputs:
    data/split.json          {"train_ids": [...], "test_ids": [...]}
    data/llm_model.json      (mock backend) trained model artefact
    data/llm_model/          (huggingface backend) saved DistilBERT weights

The split is stratified by the (label, scenario) tuple so every scenario is
proportionally represented in both train and test, with a deterministic seed.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Windows + CPU-torch safety preamble. Must run BEFORE any torch / transformers
# / datasets / numpy / tokenizers / safetensors import. Stdlib imports above
# (argparse, json, math, re, sys, collections, datetime, pathlib) do NOT
# load torch, so this position is still early enough.
#
# `KMP_DUPLICATE_LIB_OK=TRUE` silences the "OMP: Error #15" / 0xC0000005
# ACCESS_VIOLATION crash that fires when torch's bundled `libiomp5md.dll`
# collides with the OMP runtime pulled in by tokenizers / safetensors /
# numpy on Windows. Symptom: stage 3 dies immediately with exit code
# 3221225477 (== 0xC0000005). Intel-recommended workaround. No-op on
# Linux / macOS.
#
# `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` further reduce the risk of
# mixed-runtime contention on Windows. They cap intra-op parallelism, but
# for a few-hundred-example fine-tune the wall-clock difference is < 30 s.
# All three are `setdefault()` so an external user who has already set
# them can still override.
# --------------------------------------------------------------------------- #
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Tokenizers' Rust runtime (`rayon`) maintains its own thread pool and on
# Windows the worker init can race with the MKL OMP load, producing the
# same 0xC0000005 ACCESS_VIOLATION. Pinning to single-threaded eliminates
# the race without any measurable wall-clock impact for our workload sizes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
# Default to single-GPU. HF Trainer wraps the model in `torch.nn.DataParallel`
# whenever `torch.cuda.device_count() > 1` (true on Kaggle T4×2). DataParallel
# halves the optimizer step count for a fixed `--num-train-epochs`, so a 1k-
# session dataset trains for 75 steps instead of 150 and the model fails to
# converge — Stage 5's llm-trained F1 collapses to 0.0. Pin to one GPU so the
# step count matches `epochs * (n_train / per_device_batch_size)`. Set
# CUDA_VISIBLE_DEVICES=0,1 explicitly to opt back into multi-GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# faulthandler installs a Windows Structured Exception Handler so any future
# access violation (incl. the ones the OMP guards above don't catch — e.g.
# Defender intercepting model-weight mmaps, MSVC redistributable mismatch,
# numpy 2.x ABI break) prints a native C stack trace to stderr instead of
# silently dying with `exit 3221225477`. Cheap to enable; vital to debug.
import faulthandler
faulthandler.enable()

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, time
from pathlib import Path

BUSINESS_HOURS = (time(8, 0), time(18, 0))
TOKEN_RE = re.compile(r"[a-z0-9_/.\-]+")  # lowercased word splitter


# --------------------------------------------------------------------------- #
# Stratified 80/20 split
# --------------------------------------------------------------------------- #

def stratified_split(labels: list[dict], train_ratio: float = 0.8,
                     seed: int = 42) -> tuple[list[str], list[str]]:
    """Group sessions by (label, scenario) and split each group at the same
    ratio. Deterministic given `seed`. Returns sorted (train_ids, test_ids)."""

    import random
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for l in labels:
        key = l["scenario"] if l["label"] == "suspicious" else "clean"
        by_class[key].append(l["session_id"])
    train_ids: list[str] = []
    test_ids: list[str] = []
    for k, ids in sorted(by_class.items()):
        ids_shuffled = list(ids)
        rng.shuffle(ids_shuffled)
        n_train = int(round(len(ids_shuffled) * train_ratio))
        train_ids.extend(ids_shuffled[:n_train])
        test_ids.extend(ids_shuffled[n_train:])
    return sorted(train_ids), sorted(test_ids)


# --------------------------------------------------------------------------- #
# Feature extraction. Returns a dict {feature_name: 1} of "active" features.
# --------------------------------------------------------------------------- #

CAT_FEATURE_NAMES = [
    "off_hours", "asset_pivot", "approver_self",
    "many_fetches", "very_many_fetches",
    "short_justification", "very_short_justification",
    "shared_ticket_2", "shared_ticket_3plus", "shared_ticket_5plus",
    "late_session", "high_command_count",
    "rdp_surface", "sql_surface", "cloud_surface",
]


def extract_features(session: dict, agg: dict) -> dict[str, int]:
    """Active categorical features for one session, plus a bag of tokens
    drawn from the command and justification text."""
    feats: dict[str, int] = {}
    just = (session["ticket"]["business_justification"] or "").lower()
    cmds = session["session"]["commands"]
    cmd_text = " ".join(c.get("command", "") for c in cmds).lower()
    surfaces = {c.get("surface", "") for c in cmds}

    sess_start = datetime.fromisoformat(session["session"]["start_time"])
    if not (BUSINESS_HOURS[0] <= sess_start.time() <= BUSINESS_HOURS[1]):
        feats["off_hours"] = 1
    if session["session"]["host"] not in session["ticket"]["named_assets"]:
        feats["asset_pivot"] = 1
    if session["user"] == session["ticket"].get("approver"):
        feats["approver_self"] = 1

    nf = len(session.get("vault_fetches") or [])
    if nf >= 5:  feats["many_fetches"] = 1
    if nf >= 10: feats["very_many_fetches"] = 1

    if len(just) < 50: feats["short_justification"] = 1
    if len(just) < 25: feats["very_short_justification"] = 1

    info = agg.get(session["ticket_id"], {})
    n_users = info.get("n_users", 1)
    if n_users == 2:   feats["shared_ticket_2"] = 1
    if n_users >= 3:   feats["shared_ticket_3plus"] = 1
    if n_users >= 5:   feats["shared_ticket_5plus"] = 1
    if info.get("n_late", 0) >= 1: feats["late_session"] = 1

    if len(cmds) >= 8: feats["high_command_count"] = 1
    if "rdp" in surfaces:   feats["rdp_surface"] = 1
    if "sql" in surfaces:   feats["sql_surface"] = 1
    if "cloud" in surfaces: feats["cloud_surface"] = 1

    # Token bag from command + justification text
    cmd_tokens = set(TOKEN_RE.findall(cmd_text))
    just_tokens = set(TOKEN_RE.findall(just))
    for t in cmd_tokens:
        feats[f"cmd:{t}"] = 1
    for t in just_tokens:
        feats[f"just:{t}"] = 1

    return feats


def ticket_aggregate(sessions: list[dict]) -> dict[str, dict]:
    """Same algorithm as rule_detector but kept local so the LLM detector is
    self-contained."""
    by_ticket: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        by_ticket[s["ticket_id"]].append(s)
    agg = {}
    for tid, group in by_ticket.items():
        users = {s["user"] for s in group}
        we = datetime.fromisoformat(group[0]["ticket"]["change_window_end"])
        n_late = sum(1 for s in group
                     if datetime.fromisoformat(s["session"]["start_time"]) > we)
        agg[tid] = {"n_sessions": len(group), "n_users": len(users),
                    "n_late": n_late}
    return agg


# --------------------------------------------------------------------------- #
# Hybrid HF detector — fixed-length categorical vector
#
# The mock log-odds detector reaches F1 ~0.81 by combining the 15 binary
# `CAT_FEATURE_NAMES` flags above with a token bag from text. The HF
# detector by contrast only saw `_flatten_session_to_text(session)` and
# was blind to:
#   * timing signals      (start_hour, window_offset)
#   * count signals       (n_fetches, n_users_on_ticket, n_commands)
#   * cross-session info  (shared-ticket flags, late_session)
#
# `_session_to_cat_vector` produces a fixed-length numeric vector that
# the hybrid model fuses with DistilBERT's [CLS] embedding before the
# classifier head. The 15 binary features come from `extract_features`
# (so the hybrid stays in lock-step with the mock and can never drift
# in feature definitions); the 4 continuous tail values are normalised
# raw counts the binary thresholds throw away.
# --------------------------------------------------------------------------- #

CAT_VECTOR_NAMES = list(CAT_FEATURE_NAMES) + [
    "n_commands_norm",   # len(commands) / 30, capped at 2.0
    "n_fetches_norm",    # len(vault_fetches) / 20, capped at 2.0
    "n_users_norm",      # n_users_on_ticket / 10, capped at 2.0
    "start_hour_norm",   # session start hour / 24
]
N_CAT_FEATURES = len(CAT_VECTOR_NAMES)   # 19


def _session_to_cat_vector(session: dict, agg: dict) -> list[float]:
    """Fixed-length numeric vector per session for the hybrid HF model.

    Length is `N_CAT_FEATURES` = 19. Order follows `CAT_VECTOR_NAMES`
    exactly so train and inference stay in lock-step. Returns a plain
    Python list of floats (HF Datasets stores it as a 2-D tensor column,
    no dtype dance required)."""
    feats = extract_features(session, agg)
    v: list[float] = [float(feats.get(name, 0)) for name in CAT_FEATURE_NAMES]

    n_cmds = len(session["session"]["commands"])
    n_fetches = len(session.get("vault_fetches") or [])
    n_users = agg.get(session["ticket_id"], {}).get("n_users", 1)
    try:
        start_hour = datetime.fromisoformat(
            session["session"]["start_time"]).hour
    except Exception:
        # Defensive: malformed timestamp shouldn't crash inference.
        start_hour = 12

    v.extend([
        min(n_cmds / 30.0, 2.0),
        min(n_fetches / 20.0, 2.0),
        min(n_users / 10.0, 2.0),
        start_hour / 24.0,
    ])
    assert len(v) == N_CAT_FEATURES, (
        f"cat vector length mismatch: got {len(v)}, expected "
        f"{N_CAT_FEATURES} (names={CAT_VECTOR_NAMES})")
    return v


# --------------------------------------------------------------------------- #
# Mock backend: log-odds Naive Bayes
# --------------------------------------------------------------------------- #

# Smoothing constant — Laplace add-one on co-occurrence counts.
ALPHA = 0.5
# Cap on the number of token features we keep (highest |log-odds| first).
TOKEN_FEATURE_CAP = 250
# Minimum corpus document-frequency for a token to be considered.
MIN_TOKEN_DF = 6
# Maximum fraction of CLEAN training sessions a token can appear in before we
# discard it as too-common-to-be-discriminative. Tokens like "the", "ssh",
# "sudo" appear in nearly every session and add noise without signal.
MAX_CLEAN_DF_RATIO = 0.20
# Hold out this fraction of training for threshold calibration so the model
# doesn't pick a threshold that overfits its own decision surface.
VAL_HOLDOUT = 0.20


def _train_mock(sessions: list[dict], labels: list[dict],
                train_ids: set[str], seed: int = 42) -> dict:
    """Compute per-feature log-odds weights from the training partition.

    Three quality gates over and above plain log-odds:
      * tokens that appear in more than MAX_CLEAN_DF_RATIO of clean training
        sessions are treated as background noise and dropped;
      * the top TOKEN_FEATURE_CAP tokens by |log-odds| are kept;
      * a 20% validation slice is held out to calibrate the decision
        threshold without leaking the test set.
    """

    import random
    rng = random.Random(seed)

    label_map = {l["session_id"]: l["label"] for l in labels}
    agg = ticket_aggregate(sessions)
    train_sessions = [s for s in sessions if s["session_id"] in train_ids]

    # Validation hold-out for threshold calibration only — weights are still
    # learned from the FULL training partition. (The mock is a closed-form
    # estimator, no SGD steps to reserve a val slice for; we only need the
    # held-out scores.)
    shuffled = list(train_sessions)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_HOLDOUT)
    val_sessions = shuffled[:n_val]
    val_ids = {s["session_id"] for s in val_sessions}

    n_pos = sum(1 for s in train_sessions
                if label_map[s["session_id"]] == "suspicious")
    n_neg = len(train_sessions) - n_pos
    n_neg_safe = max(n_neg, 1)

    # Co-occurrence counts on the FULL training partition.
    feat_pos: Counter = Counter()
    feat_neg: Counter = Counter()

    feats_per_session: dict[str, dict[str, int]] = {}
    for s in train_sessions:
        f = extract_features(s, agg)
        feats_per_session[s["session_id"]] = f
        target = feat_pos if label_map[s["session_id"]] == "suspicious" else feat_neg
        for k in f:
            target[k] += 1

    # Token feature shortlist:
    #   * appears in at least MIN_TOKEN_DF training sessions (any class)
    #   * appears in at most MAX_CLEAN_DF_RATIO of CLEAN training sessions
    df = Counter()
    for f in feats_per_session.values():
        for k in f:
            df[k] += 1
    token_feats = [
        k for k in df
        if k.startswith(("cmd:", "just:"))
        and df[k] >= MIN_TOKEN_DF
        and feat_neg[k] / n_neg_safe <= MAX_CLEAN_DF_RATIO
    ]

    weights: dict[str, float] = {}
    for k in token_feats:
        p_pos = (feat_pos[k] + ALPHA) / (n_pos + 2 * ALPHA)
        p_neg = (feat_neg[k] + ALPHA) / (n_neg + 2 * ALPHA)
        weights[k] = math.log(p_pos / p_neg)
    # (-|weight|, token) is a stable, fully-deterministic ordering that ties
    # alphabetically when two tokens carry the same magnitude.
    ranked_tokens = sorted(token_feats,
                            key=lambda k: (-abs(weights[k]), k))[:TOKEN_FEATURE_CAP]

    # Sort the kept-features list so the JSON serialisation order is stable
    # across processes (Python's set iteration order depends on PYTHONHASHSEED
    # and would otherwise drift run-to-run).
    keep = sorted(set(CAT_FEATURE_NAMES) | set(ranked_tokens))
    kept_weights: dict[str, float] = {}
    for k in keep:
        p_pos = (feat_pos[k] + ALPHA) / (n_pos + 2 * ALPHA)
        p_neg = (feat_neg[k] + ALPHA) / (n_neg + 2 * ALPHA)
        kept_weights[k] = round(math.log(p_pos / p_neg), 6)

    bias = round(math.log(max(n_pos, 1) / n_neg_safe), 6)

    # Threshold calibration on the held-out validation slice. We sweep over
    # all unique scores in the slice (capped at 500) and pick the one with
    # the highest F1.
    val_scores = []
    val_truth = []
    for s in val_sessions:
        f = feats_per_session[s["session_id"]]
        score = bias + sum(kept_weights.get(k, 0.0) for k in f)
        val_scores.append(score)
        val_truth.append(1 if label_map[s["session_id"]] == "suspicious" else 0)

    best_threshold = 0.0
    best_f1 = -1.0
    candidates = sorted(set(val_scores))
    if len(candidates) > 500:
        step = len(candidates) // 500
        candidates = candidates[::step]
    for thr in candidates:
        tp = fp = fn = 0
        for sc, tr in zip(val_scores, val_truth):
            pred = 1 if sc >= thr else 0
            if pred == 1 and tr == 1: tp += 1
            elif pred == 1 and tr == 0: fp += 1
            elif pred == 0 and tr == 1: fn += 1
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    return {
        "backend": "mock",
        "weights": kept_weights,
        "bias": bias,
        "threshold": round(best_threshold, 6),
        "metadata": {
            "n_train": len(train_sessions),
            "n_validation_holdout": n_val,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_features": len(kept_weights),
            "calibration_validation_f1": round(best_f1, 4),
            "alpha": ALPHA,
            "min_token_df": MIN_TOKEN_DF,
            "max_clean_df_ratio": MAX_CLEAN_DF_RATIO,
            "token_feature_cap": TOKEN_FEATURE_CAP,
        },
    }


# --------------------------------------------------------------------------- #
# HuggingFace backend — hybrid DistilBERT + cat-feature MLP
#
# Lifted verbatim into Stage F's Kaggle notebook. To enable locally, install
# transformers + torch + datasets and pass --llm-backend huggingface.
#
# Architecture v2 (forward pass) — REBALANCED for cat-feature signal:
#
#   text  ─► DistilBERT encoder ─► [CLS] (768-dim) ─► Linear(768→128) ─► ReLU ┐
#                                                                             ├─ concat (160) ─► Dropout ─► Linear ─► 2 logits
#   cat_vector (19) ─► Linear(19→64) ─► ReLU ─► Dropout ─► Linear(64→32) ─► ReLU ┘
#
# Why the rebalance vs v1? v1 concatenated 768-dim text with 16-dim cat,
# which made the cat path 2 % of the input. The optimizer ignored cat
# features for low-frequency scenarios (S5 / S10 / S15 stuck at 0 %
# recall) because moving their weights barely changed global accuracy.
# v2 projects text down to 128 dim and expands cat to 32 dim, so the
# cat path is now 20 % of the classifier input — matching its share of
# the discriminative signal in the data. Total params unchanged within
# 1 % (still ~66 M, dominated by DistilBERT).
#
# Backwards compat: detect.py reads `architecture_version` from the
# saved hybrid_config.json. v1 checkpoints from earlier runs still load
# (they hit the legacy 768+16 path). New runs always save v2.
# --------------------------------------------------------------------------- #

# Architecture v2 hyperparameters. Kept as module constants so detect.py
# can import the same numbers and reconstruct the wrapper without a
# local copy.
TEXT_PROJ_DIM = 128         # DistilBERT [CLS] 768 -> 128 before concat
CAT_HIDDEN_DIM = 64         # cat-MLP first hidden layer width
CAT_OUT_DIM = 32            # cat-MLP output, concat partner for text proj
ARCHITECTURE_VERSION = 2    # bumped whenever the wrapper layout changes


def _build_hybrid_model(base_distilbert, n_cat_features: int = N_CAT_FEATURES,
                        n_labels: int = 2,
                        text_proj_dim: int = TEXT_PROJ_DIM,
                        cat_hidden_dim: int = CAT_HIDDEN_DIM,
                        cat_out_dim: int = CAT_OUT_DIM):
    """Construct a HybridDistilBERTClassifier from a pre-loaded DistilBERT.

    The class is defined inside this function to avoid importing torch /
    transformers at module load time — the mock backend never touches HF
    code, so we keep its import cost free."""
    import torch
    from torch import nn
    from transformers.modeling_outputs import SequenceClassifierOutput  # type: ignore[import-not-found]

    class HybridDistilBERTClassifier(nn.Module):
        """DistilBERT [CLS] embedding fused with a small MLP over the
        per-session categorical / numeric vector. HF Trainer-compatible:
        forward returns a SequenceClassifierOutput and self.config is set
        to the underlying DistilBERT's config so internal HF utilities
        (problem_type detection, save_pretrained inference, etc.) keep
        working."""

        def __init__(self, distilbert, n_cat_features: int, n_labels: int,
                     text_proj_dim: int, cat_hidden_dim: int,
                     cat_out_dim: int):
            super().__init__()
            self.distilbert = distilbert
            self.config = distilbert.config
            hidden = distilbert.config.hidden_size  # 768 on distilbert-base
            # Project DistilBERT's [CLS] embedding down so the text path
            # doesn't dominate the concat. Without this, cat features
            # for low-frequency scenarios (S5/S10/S15) get gradient-
            # starved and end up at 0 % recall.
            self.text_proj = nn.Sequential(
                nn.Linear(hidden, text_proj_dim),
                nn.ReLU(),
            )
            self.cat_mlp = nn.Sequential(
                nn.Linear(n_cat_features, cat_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cat_hidden_dim, cat_out_dim),
                nn.ReLU(),
            )
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(text_proj_dim + cat_out_dim, n_labels)
            # Loss config — populated by the Trainer subclass via
            # `set_loss_config`. Defaults reproduce plain CE for any
            # caller that bypasses the wrapper (e.g. legacy unit tests).
            self.loss_class_weights: torch.Tensor | None = None
            self.loss_focal_gamma: float = 0.0
            self.n_cat_features = n_cat_features
            self.n_labels = n_labels
            self.text_proj_dim = text_proj_dim
            self.cat_hidden_dim = cat_hidden_dim
            self.cat_out_dim = cat_out_dim

        def set_loss_config(self, class_weights=None, focal_gamma: float = 0.0):
            """Configure the per-call loss. Called once by `_train_huggingface`
            after model construction. `class_weights` is a 1-D tensor of
            length n_labels; `focal_gamma` >= 0, with 0 reducing to plain
            (weighted) cross-entropy."""
            self.loss_class_weights = class_weights
            self.loss_focal_gamma = float(focal_gamma)

        def _compute_loss(self, logits, labels):
            """Focal cross-entropy with optional class weights. With
            gamma=0 and weights=None this is plain CE — gives bit-exact
            backwards compat with v1 checkpoints when set_loss_config
            isn't called."""
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(
                1, labels.unsqueeze(1)).squeeze(1)
            target_probs = target_log_probs.exp()
            # Focal modulating term: (1 - p_t)^gamma. With gamma=0 this
            # is identically 1 and the loss collapses to plain (weighted)
            # cross-entropy.
            if self.loss_focal_gamma > 0:
                focal_weight = (1.0 - target_probs).pow(self.loss_focal_gamma)
            else:
                focal_weight = torch.ones_like(target_probs)
            losses = -focal_weight * target_log_probs
            if self.loss_class_weights is not None:
                w = self.loss_class_weights.to(losses.device)
                losses = losses * w[labels]
            return losses.mean()

        def forward(self, input_ids=None, attention_mask=None,
                    cat_features=None, labels=None, **_unused):
            outputs = self.distilbert(input_ids=input_ids,
                                      attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            if cat_features is None:
                # Defensive — should never fire in normal flow because
                # HybridCollator always supplies cat_features. If a caller
                # ever passes plain HF batches (e.g. an evaluation harness
                # written before this refactor), fall back to a zero
                # vector so we degrade gracefully to text-only behaviour.
                batch = cls_emb.size(0)
                cat_features = torch.zeros(batch, self.n_cat_features,
                                           device=cls_emb.device)
            text_emb = self.text_proj(cls_emb)
            cat_emb = self.cat_mlp(cat_features.float())
            combined = torch.cat([text_emb, cat_emb], dim=-1)
            combined = self.dropout(combined)
            logits = self.classifier(combined)
            loss = None
            if labels is not None:
                loss = self._compute_loss(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return HybridDistilBERTClassifier(base_distilbert, n_cat_features,
                                      n_labels, text_proj_dim,
                                      cat_hidden_dim, cat_out_dim)


def _build_hybrid_collator(tokenizer):
    """Custom data collator: pads the variable-length text fields via
    `tokenizer.pad`, stacks the fixed-length `cat_features` vector and
    the labels into tensors. Returns a dict matching the model's forward
    signature."""
    import torch

    class HybridCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            cat = torch.tensor(
                [f["cat_features"] for f in features],
                dtype=torch.float32,
            )
            labels = torch.tensor([f["label"] for f in features],
                                  dtype=torch.long)
            text_features = [
                {"input_ids": f["input_ids"],
                 "attention_mask": f["attention_mask"]}
                for f in features
            ]
            batch = self.tokenizer.pad(text_features, return_tensors="pt")
            batch["cat_features"] = cat
            batch["labels"] = labels
            return batch

    return HybridCollator(tokenizer)


def _train_huggingface(sessions: list[dict], labels: list[dict],
                       train_ids: set[str], out_dir: Path) -> dict:
    """Fine-tune a hybrid DistilBERT + cat-feature classifier on the
    train partition.

    Architecture:
      * DistilBERT encoder over `_flatten_session_to_text(session)` →
        [CLS] embedding (768-dim).
      * Small MLP over `_session_to_cat_vector(session, agg)` (19-dim,
        15 binary + 4 normalised counts) → 16-dim.
      * Concatenate [CLS] (768) + cat_emb (16) → linear → 2 logits.

    Compatible with transformers v4.50+ and v5.x (latest stable v5.7 as
    of May 2026). Earlier transformers versions miss the
    `processing_class` rename and the `text2text-generation` pipeline
    removal; pin transformers >= 5.5 for current Python (3.13 / 3.14)."""
    import inspect                                               # std-lib
    import time as _time                                         # std-lib
    from transformers import (AutoTokenizer,                    # type: ignore[import-not-found]
                              AutoModel,
                              Trainer, TrainingArguments)
    from datasets import Dataset                                 # type: ignore[import-not-found]
    import torch                                                 # type: ignore[import-not-found]

    # Hugging Face login (if a token is available) — speeds up downloads
    # and avoids anonymous rate limits. No-op if the user has no token.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        import hf_token                                          # type: ignore[import-not-found]
        _hf_tok = hf_token.setup_token(verbose=True)
    except ImportError:
        _hf_tok = None
    tok_kwargs: dict = {"token": _hf_tok} if _hf_tok else {}

    # Device pick + report — Trainer auto-detects but we want a loud signal
    # that GPU is or isn't being used so misconfigured environments don't
    # silently fall back to a 30-minute CPU run.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus))
        print(f"[hf] DistilBERT trainer: GPU x{n_gpus} = {gpu_names}",
              flush=True)
        train_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_fp16 = train_dtype is torch.float16
        use_bf16 = train_dtype is torch.bfloat16
    else:
        device = torch.device("cpu")
        print("[hf] DistilBERT trainer: CPU only — torch has no CUDA support. "
              "Fine-tuning will take ~25-40 min. Install GPU torch to "
              "accelerate: pip uninstall -y torch && pip install torch "
              "--index-url https://download.pytorch.org/whl/cu126", flush=True)
        use_fp16 = use_bf16 = False

    label_map = {l["session_id"]: 1 if l["label"] == "suspicious" else 0
                 for l in labels}
    train_sessions = [s for s in sessions if s["session_id"] in train_ids]
    # Aggregator built from the FULL session list (not just the train
    # split): cross-session signals like `n_users_on_ticket` need to see
    # every session sharing a ticket, including ones that ended up in
    # the test partition. This matches what the rule_detector does.
    agg = ticket_aggregate(sessions)
    examples = [{"text": _flatten_session_to_text(s, agg),
                 "cat_features": _session_to_cat_vector(s, agg),
                 "label": label_map[s["session_id"]]}
                for s in train_sessions]
    print(f"[hf] Built {len(examples)} training examples "
          f"(text + {N_CAT_FEATURES}-dim cat vector + label).",
          flush=True)

    # Sanity-check the cat vector class balance — if it's accidentally
    # all-zeros (e.g. agg is empty) the cat head will learn nothing and
    # we'd be back to text-only. Print a tiny histogram so misconfigured
    # runs fail loudly instead of silently regressing to baseline.
    pos_cat = [ex["cat_features"] for ex in examples if ex["label"] == 1]
    neg_cat = [ex["cat_features"] for ex in examples if ex["label"] == 0]
    if pos_cat and neg_cat:
        for i, name in enumerate(CAT_VECTOR_NAMES[:6]):
            p_mean = sum(v[i] for v in pos_cat) / len(pos_cat)
            n_mean = sum(v[i] for v in neg_cat) / len(neg_cat)
            print(f"[hf]   cat[{name:>20}] suspicious={p_mean:.2f} "
                  f"clean={n_mean:.2f}", flush=True)

    print("[hf] [1/3] Downloading / loading distilbert-base-uncased "
          "tokenizer (first run pulls a few MB)...", flush=True)
    t0 = _time.time()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                               **tok_kwargs)
    print(f"[hf]      tokenizer ready in {_time.time() - t0:.1f}s",
          flush=True)

    print("[hf] [2/3] Downloading / loading distilbert-base-uncased base "
          "encoder (first run pulls ~270 MB; subsequent runs reuse the HF "
          "cache)...", flush=True)
    t0 = _time.time()
    base_distilbert = AutoModel.from_pretrained("distilbert-base-uncased",
                                                **tok_kwargs)
    model = _build_hybrid_model(base_distilbert,
                                n_cat_features=N_CAT_FEATURES, n_labels=2)
    # Configure focal loss + class weights so the model weighs minority-class
    # (suspicious) examples and hard examples (the ones whose target prob
    # is currently low) more heavily. With the default 70/30 clean/suspicious
    # split, alpha = [1.0, 1/0.30 ≈ 2.33] balances the per-class gradient.
    # gamma = 2.0 is the canonical Lin et al. (2017) focal value — it's
    # large enough to push hard examples but small enough not to destabilise
    # training. Both knobs are env-overridable for ablation.
    n_pos_train = sum(1 for ex in examples if ex["label"] == 1)
    n_neg_train = len(examples) - n_pos_train
    pos_weight = (n_neg_train / max(n_pos_train, 1)) if n_pos_train else 1.0
    pos_weight = float(os.environ.get("PAM_LOSS_POS_WEIGHT", pos_weight))
    focal_gamma = float(os.environ.get("PAM_LOSS_FOCAL_GAMMA", 2.0))
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    model.set_loss_config(class_weights=class_weights,
                          focal_gamma=focal_gamma)
    print(f"[hf]      loss config: focal_gamma={focal_gamma}, "
          f"class_weights=[1.0, {pos_weight:.2f}] "
          f"(n_pos={n_pos_train}, n_neg={n_neg_train})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_head = (sum(p.numel() for p in model.cat_mlp.parameters())
              + sum(p.numel() for p in model.text_proj.parameters())
              + sum(p.numel() for p in model.classifier.parameters()))
    print(f"[hf]      hybrid v{ARCHITECTURE_VERSION} model ready in "
          f"{_time.time() - t0:.1f}s "
          f"({n_params:,} total params, {n_head:,} in "
          f"text-proj + cat-MLP + classifier)", flush=True)

    # Bump max_length 256 -> 512 so the richer text representation
    # (now includes start_hour, window timing, fetch counts, all
    # commands instead of top 12) fits without truncating discriminative
    # signal for long sessions. 512 is DistilBERT's hard ceiling.
    text_max_length = int(os.environ.get("PAM_TEXT_MAX_LENGTH", 512))
    print(f"[hf] [3/3] Tokenising training examples "
          f"(padding to {text_max_length} tokens)...", flush=True)
    t0 = _time.time()
    ds = Dataset.from_list(examples)
    # Map tokenization but preserve `cat_features` and `label` columns.
    # Datasets keeps unspecified columns by default after `.map`.
    def _tokenize(batch):
        tok = tokenizer(batch["text"], truncation=True,
                        padding="max_length", max_length=text_max_length)
        return tok
    ds = ds.map(_tokenize, batched=True, remove_columns=["text"])
    print(f"[hf]      tokenised in {_time.time() - t0:.1f}s "
          f"({len(ds)} rows; columns={list(ds.column_names)})", flush=True)

    # CPU fine-tunes use a smaller batch — 16 × 512-token sequences in fp32
    # has a working set of roughly 8-10 GB, which OOMs Windows boxes with 8 GB
    # RAM and a browser open. Batch 4 brings it back under 3 GB while only
    # adding ~30 % wall-clock on a typical laptop CPU. GPU batches stay at
    # 16 because T4 / Ampere have plenty of memory and benefit from larger
    # batches. Override with the `PAM_TRAIN_BATCH_SIZE` env var if needed.
    # Note: doubled max_length means CUDA memory roughly doubles too; if
    # T4 OOMs at batch=16 try `PAM_TRAIN_BATCH_SIZE=8`.
    if device.type == "cuda":
        default_batch = 16
    else:
        default_batch = 4
    train_batch = int(os.environ.get("PAM_TRAIN_BATCH_SIZE", default_batch))

    # Training schedule v2: 10 epochs (was 3) with cosine LR + 10 % warmup.
    # The May-10 cloud run showed train_loss still falling at end of epoch 3
    # (0.61 -> 0.43, not converged). 10 epochs lets the model fully exploit
    # the cat-feature + focal-loss combination. Cosine schedule with warmup
    # is the standard transformer-fine-tune recipe and produces a smoother
    # loss curve than the default linear-decay-from-step-0.
    n_epochs = int(os.environ.get("PAM_NUM_TRAIN_EPOCHS", 10))
    warmup_ratio = float(os.environ.get("PAM_WARMUP_RATIO", 0.1))
    lr_scheduler = os.environ.get("PAM_LR_SCHEDULER", "cosine")

    # Build TrainingArguments defensively — `report_to` and a few others
    # changed accepted-value rules across v4 -> v5. Pass only what every
    # supported version accepts and let Trainer fill in sensible defaults.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs: dict = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=train_batch,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        # `report_to="none"` is the canonical "disable all integrations"
        # value that's been stable from v4.20 through v5.7. The empty list
        # `[]` is also accepted but emits a deprecation warning on v5.
        report_to="none",
    )
    # Cosine schedule + warmup are post-v4.0 features; gate behind a
    # signature check for safety.
    if "lr_scheduler_type" in ta_params:
        ta_kwargs["lr_scheduler_type"] = lr_scheduler
    if "warmup_ratio" in ta_params:
        ta_kwargs["warmup_ratio"] = warmup_ratio
    # Mixed precision is opt-in per parameter and only valid on GPU. Older
    # TrainingArguments builds may not have the bf16 flag at all.
    if use_fp16 and "fp16" in ta_params:
        ta_kwargs["fp16"] = True
    if use_bf16 and "bf16" in ta_params:
        ta_kwargs["bf16"] = True
    args = TrainingArguments(**ta_kwargs)

    # transformers 4.41 renamed `tokenizer` -> `processing_class` on Trainer.
    # v5.0 (Jan 2026) REMOVED the legacy name entirely. Inspect the live
    # signature so we work on every supported version without a hard pin.
    collator = _build_hybrid_collator(tokenizer)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": ds,
        "data_collator": collator,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    n_steps = max(1, len(examples) // ta_kwargs["per_device_train_batch_size"]) \
              * ta_kwargs["num_train_epochs"]
    print(f"[hf] starting fine-tune (HYBRID model): {len(examples)} examples, "
          f"{ta_kwargs['num_train_epochs']} epochs, "
          f"batch={ta_kwargs['per_device_train_batch_size']}, "
          f"~{n_steps} steps total, "
          f"fp16={use_fp16}, bf16={use_bf16}", flush=True)
    print("[hf] Trainer prints its own tqdm progress bar below; if nothing "
          "appears within ~30 s the GPU may be stalled — ctrl-C and check "
          "`nvidia-smi`.", flush=True)
    t0 = _time.time()
    trainer.train()
    elapsed = _time.time() - t0

    # Verify the model ended up on the device we expected.
    final_device = next(model.parameters()).device
    print(f"[hf] fine-tune complete in {elapsed:.1f}s; "
          f"model on device={final_device}", flush=True)

    # HF's Trainer keeps log_history in memory but never persists it.
    # Emit the five-column schema build_figures.py expects (step,
    # epoch, loss, grad_norm, learning_rate); rows without a "loss"
    # key (eval rows, final summary row) are skipped so column types
    # stay uniform for the CSV reader at build_figures.py:298-303.
    import csv as _csv
    log_path = out_dir.parent / "training_log.csv"
    log_fields = ["step", "epoch", "loss", "grad_norm", "learning_rate"]
    log_rows = [
        {k: r.get(k, "") for k in log_fields}
        for r in trainer.state.log_history
        if "loss" in r
    ]
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    print(f"[hf] training log: {len(log_rows)} step rows -> {log_path}",
          flush=True)

    # Saving strategy: the hybrid wrapper isn't a PreTrainedModel, so
    # `save_pretrained` doesn't apply. We split the artefacts into three
    # pieces so detect.py can load each with the appropriate API:
    #   * tokenizer/  → tokenizer.save_pretrained (HF native)
    #   * distilbert/ → model.distilbert.save_pretrained (HF native; the
    #                   trained encoder, no need to re-pull from HF cache
    #                   on a different machine)
    #   * hybrid_head.pt   → torch.save of cat_mlp + classifier state
    #   * hybrid_config.json → metadata so detect.py can reconstruct the
    #                          wrapper (n_cat_features, label count,
    #                          cat-vector field names)
    print("[hf] Saving hybrid model to disk (tokenizer + distilbert + "
          "hybrid head)...", flush=True)
    t0 = _time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "distilbert").mkdir(parents=True, exist_ok=True)
    model.distilbert.save_pretrained(out_dir / "distilbert")
    head_state = {
        "cat_mlp": {k: v.detach().cpu()
                    for k, v in model.cat_mlp.state_dict().items()},
        "text_proj": {k: v.detach().cpu()
                      for k, v in model.text_proj.state_dict().items()},
        "classifier": {k: v.detach().cpu()
                       for k, v in model.classifier.state_dict().items()},
    }
    torch.save(head_state, out_dir / "hybrid_head.pt")
    with (out_dir / "hybrid_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "architecture": "HybridDistilBERTClassifier",
            "architecture_version": ARCHITECTURE_VERSION,
            "base_model_name": "distilbert-base-uncased",
            "n_cat_features": N_CAT_FEATURES,
            "n_labels": 2,
            "cat_vector_names": CAT_VECTOR_NAMES,
            # v2 architecture hyperparameters; detect.py reads these to
            # reconstruct the wrapper for any checkpoint that records
            # them. v1 checkpoints (no `architecture_version` key, no
            # text_proj_dim) are auto-detected and rebuilt with the
            # legacy 768+16 layout for backwards compat.
            "text_proj_dim": TEXT_PROJ_DIM,
            "cat_hidden_dim": CAT_HIDDEN_DIM,
            "cat_out_dim": CAT_OUT_DIM,
            "text_max_length": text_max_length,
            "training": {
                "num_train_epochs": n_epochs,
                "warmup_ratio": warmup_ratio,
                "lr_scheduler": lr_scheduler,
                "focal_gamma": focal_gamma,
                "class_weights": [1.0, pos_weight],
            },
        }, f, indent=1)
    print(f"[hf]   saved in {_time.time() - t0:.1f}s -> {out_dir}",
          flush=True)

    return {
        "backend": "huggingface",
        "model_dir": str(out_dir),
        "hybrid": True,
        "metadata": {
            "n_train": len(train_sessions),
            "device": str(device),
            "fp16": use_fp16,
            "bf16": use_bf16,
            "n_cat_features": N_CAT_FEATURES,
            "architecture": "HybridDistilBERTClassifier",
            "architecture_version": ARCHITECTURE_VERSION,
            "text_proj_dim": TEXT_PROJ_DIM,
            "cat_hidden_dim": CAT_HIDDEN_DIM,
            "cat_out_dim": CAT_OUT_DIM,
            "text_max_length": text_max_length,
            "n_epochs": n_epochs,
            "lr_scheduler": lr_scheduler,
            "warmup_ratio": warmup_ratio,
            "focal_gamma": focal_gamma,
            "pos_class_weight": pos_weight,
        },
    }


def _flatten_session_to_text(session: dict,
                             agg: dict | None = None) -> str:
    """Produce a single text blob per session for DistilBERT input.

    v2 representation, designed to expose every signal the rule detector
    uses so DistilBERT has a fighting chance on structural-only
    scenarios (S5 off-hours, S10 asset pivot, S11 zombie ticket, etc.).

    Includes:
      * category, user, approver
      * full business justification (no truncation)
      * session host vs ticket-named-asset (for S10 asset pivot)
      * session start hour, day-of-week, and whether it falls inside
        business hours / the change window (for S5, S11)
      * explicit user==approver flag (for S15)
      * vault-fetch count (for S3)
      * cross-session aggregate: number of distinct users on the same
        ticket, late-session count (for S1, S11) — only when an
        aggregate dict is supplied (it always is in train + inference)
      * ALL commands (was top-12); per-command 80-char cap is kept
        because no realistic command line exceeds that, and bounded
        per-line length stops a single 4 KB SQL blob from eating the
        whole context window.

    Tokeniser truncates at `max_length` (512 in v2), so very long
    sessions still fit; commands beyond ~30 lines start getting
    truncated, which is fine because S7/S12/S13/S14 signals appear in
    the first few commands by construction."""
    t = session["ticket"]
    s = session["session"]
    just = (t.get("business_justification") or "").strip()
    cmds = s.get("commands", [])
    cmd_blob = " | ".join(
        (c.get("command", "") or "")[:80] for c in cmds
    )
    user = session.get("user", "?")
    approver = t.get("approver", "?")

    # Timing — DistilBERT can read these as text strings. The
    # categorical cat-MLP path will see normalised numerics; this
    # text path is defence-in-depth.
    try:
        start_dt = datetime.fromisoformat(s["start_time"])
        start_hour = start_dt.hour
        dow = start_dt.strftime("%a")
        in_business_hours = (BUSINESS_HOURS[0]
                             <= start_dt.time() <= BUSINESS_HOURS[1])
    except Exception:
        start_hour = -1
        dow = "?"
        in_business_hours = True
    try:
        we = datetime.fromisoformat(t["change_window_end"])
        late = ("yes"
                if start_dt > we else "no") if start_hour >= 0 else "?"
    except Exception:
        late = "?"

    host = s.get("host", "?")
    named_assets = t.get("named_assets") or []
    asset_match = "yes" if host in named_assets else "no"

    n_fetches = len(session.get("vault_fetches") or [])
    n_cmds = len(cmds)

    if agg is not None:
        info = agg.get(session.get("ticket_id", ""), {})
        n_users = info.get("n_users", 1)
        n_late_on_ticket = info.get("n_late", 0)
    else:
        n_users = 1
        n_late_on_ticket = 0

    approver_self = "yes" if user == approver else "no"

    # Field order matters slightly: put the most discriminative
    # numeric / structural facts up front so they survive truncation
    # if the justification or command blob is long.
    return (
        f"category: {t.get('category','?')} . "
        f"user: {user} . approver: {approver} . approver_self: {approver_self} . "
        f"start_hour: {start_hour:02d} . day: {dow} . "
        f"in_business_hours: {'yes' if in_business_hours else 'no'} . "
        f"started_after_window_end: {late} . "
        f"host: {host} . host_in_named_assets: {asset_match} . "
        f"n_vault_fetches: {n_fetches} . n_commands: {n_cmds} . "
        f"n_users_on_ticket: {n_users} . "
        f"n_late_sessions_on_ticket: {n_late_on_ticket} . "
        f"justification: {just} . "
        f"commands: {cmd_blob}"
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=Path, default=Path("data/sessions.json"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.json"))
    p.add_argument("--split-out", type=Path, default=Path("data/split.json"))
    p.add_argument("--model-out", type=Path, default=Path("data/llm_model.json"))
    p.add_argument("--backend", choices=["mock", "huggingface"], default="mock")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    sessions = json.load(args.sessions.open(encoding="utf-8"))
    labels = json.load(args.labels.open(encoding="utf-8"))

    train_ids, test_ids = stratified_split(labels, args.train_ratio, args.seed)
    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    with args.split_out.open("w", encoding="utf-8") as f:
        json.dump({"train_ids": train_ids, "test_ids": test_ids,
                   "train_ratio": args.train_ratio, "seed": args.seed}, f, indent=1)
    print(f"Split:   train={len(train_ids)}  test={len(test_ids)}  "
          f"-> {args.split_out}")

    if args.backend == "mock":
        model = _train_mock(sessions, labels, set(train_ids), seed=args.seed)
        with args.model_out.open("w", encoding="utf-8") as f:
            json.dump(model, f, indent=1)
        print(f"Mock model: {model['metadata']['n_features']} features, "
              f"threshold={model['threshold']}, "
              f"val_F1={model['metadata']['calibration_validation_f1']}")
        print(f"  output: {args.model_out}")
    else:
        model = _train_huggingface(sessions, labels, set(train_ids),
                                    args.model_out.with_suffix(""))
        # For HF backend we save a tiny pointer JSON; the actual weights live
        # under model_out.with_suffix("") (a directory).
        with args.model_out.open("w", encoding="utf-8") as f:
            json.dump(model, f, indent=1)
        print(f"DistilBERT saved to {model['model_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
