# Drishti-PAM — Hybrid Structural-Deviation Detection for Privileged-Access Monitoring

> *The gaze that sees beyond the surface.* — Sanskrit *दृष्टि (Drishti)* names the perception that reads structure under vocabulary, which is the central technical claim of this detector's result on the **S16 vocabulary-cloak adversarial control**: an attacker who paraphrases the literal trigger words a rule-based detector matches on cannot, by structural construction, cloak the deviation between *what the change ticket promised* and *what the recorded session shows*.

A reproducible, two-detector pipeline for catching insider-risk behaviour in **Privileged Access Management (PAM) session logs**. A deterministic rule baseline runs alongside a hybrid LLM classifier; both are evaluated on the same held-out split so their numbers are directly comparable.

| Test split (2,000 sessions) | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Rule detector | 0.9560 | 0.9767 | 0.9662 | 0.9795 |
| Trained LLM (DistilBERT + MLP) | 0.9878 | 0.9483 | **0.9677** | 0.9810 |

The trained classifier wins on F1 by 0.0015 — it trades 17 missed attacks against 20 fewer false alarms — and is the configuration this repository ships as default.

---

## Quick start

```powershell
# Windows 10/11 — fastest path, mock backend, ~15 seconds end-to-end
git clone https://github.com/camseye/drishti-pam.git
cd drishti-pam/code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python smoke.py --backend mock --non-interactive
```

```bash
# Linux / macOS — same pipeline, ~15 seconds, no GPU needed
git clone https://github.com/YOUR-USER/drishti-pam.git
cd drishti-pam/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python smoke.py --backend mock --non-interactive
```

After the run finishes you'll have:

```
data/
├── sessions.json          ← the synthetic corpus (10,000 sessions)
├── labels.json            ← ground-truth labels per session
├── split.json             ← train/val/test partition (8000/1000/1000)
├── rule_predictions.json  ← rule detector output
├── llm_predictions.json   ← trained-LLM detector output
├── comparison.json        ← machine-readable headline + per-scenario numbers
├── comparison.md          ← human-readable Markdown report
├── manifest.json          ← run metadata (counts, seed, scenarios)
└── training_log.csv       ← per-batch loss + learning rate
```

The minimalist console output looks like this:

```
────────────────────────────────────────────────────────────────
  Detector evaluation — test split (2,000 sessions)
────────────────────────────────────────────────────────────────

  Detector       Precision    Recall       F1   Accuracy
  ──────────────────────────────────────────────────────
  rule              0.9560    0.9767   0.9662     0.9795
  llm-trained       0.9878    0.9483   0.9677     0.9810

  Winner: llm-trained  (F1 +0.0015, -17 missed attacks, +13 false alarms vs rule)

  Per-scenario breakdown:  data/comparison.md
  Machine-readable report: data/comparison.json
```

That's the whole project on stdout. Everything else — per-scenario recall, per-rule firing patterns, confusion matrices, training curves — lives in the JSON and Markdown reports.

### Refreshing the dashboard from fresh data

Once `smoke.py` (or a Kaggle re-run) has emitted fresh files into
`data/`, regenerate the four dashboard data files in one step:

```bash
python docs/build_dashboard.py
```

---

## Architecture (ASCII)

```
                    ┌─────────────────────┐
                    │   Session record    │
                    │ ticket • host •     │
                    │ commands •          │
                    │ timestamps •        │
                    │ approver            │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌─────────────────────┐           ┌─────────────────────┐
   │   TEXT BRANCH       │           │  CATEGORICAL BRANCH │
   │ flatten to prompt   │           │ 19 numeric / boolean│
   │ text (max 512 tok)  │           │ engineered signals  │
   └──────────┬──────────┘           └──────────┬──────────┘
              ▼                                 ▼
   ┌─────────────────────┐           ┌─────────────────────┐
   │ DistilBERT-base     │           │   MLP 19 → 64       │
   │ encoder → [CLS]     │           │   Linear + ReLU     │
   │ hidden, 768-dim     │           └──────────┬──────────┘
   └──────────┬──────────┘                      ▼
              ▼                       ┌─────────────────────┐
   ┌─────────────────────┐            │   MLP 64 → 32       │
   │ projection 768→128  │            │   Linear + ReLU     │
   │ Linear + ReLU       │            └──────────┬──────────┘
   └──────────┬──────────┘                       │
              └────────────┬────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Concat 128 + 32 = 160  │
              │ late fusion of branches │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │ Linear 160 → 2 + softmax│
              │ focal loss (γ=2,        │
              │ α=[1.0, 2.33])          │
              └────────────┬────────────┘
                           ▼
                   p(suspicious) ∈ [0, 1]
```

**Why this shape.** The text branch handles the linguistic side of the threat (a suspicious justification, a sensitive keyword inside a command stream) while the categorical branch handles the structural side (off-hours time-of-day, ticket-category mismatch, approver-as-implementer). Late fusion keeps each branch independently auditable — useful when a security analyst asks *which* signal flagged a session.

---

## Running it on Kaggle (free GPU, no local install)

1. Upload this `code/` folder as a Kaggle Dataset (e.g. `pam-source`).
2. Create a new Kaggle notebook, **GPU T4 x2** accelerator.
3. Open `notebooks/cloud_run.ipynb` and follow the cells in order.
4. Step 4 writes `artifacts.zip` (~250 MB including the fine-tuned DistilBERT) — download from the right-hand pane.

End-to-end on Kaggle: ~12 minutes for a 10,000-session run. Locally on CPU: ~30–40 minutes for the same run.

---

## Data card — the 12 scenarios

The synthetic corpus models twelve insider-risk patterns at 70/30 clean/suspicious. Counts below are from the published 10,000-session run.

| Code | Scenario | Type | Count | % |
|---|---|---|---:|---:|
| S1 | Multi-user lateral movement | cross-session | 608 | 6.1 % |
| S3 | Mass file fetches | single-session | 600 | 6.0 % |
| S4 | Off-category command (ticket says A, command does B) | linguistic | 300 | 3.0 % |
| S7 | Destructive command with weak justification | linguistic | 300 | 3.0 % |
| S10 | Asset pivot inside one session | single-session | 300 | 3.0 % |
| S11 | Zombie session (long idle then activity) | single-session | 240 | 2.4 % |
| S5 | Off-hours sensitive access | single-session | 150 | 1.5 % |
| S12 | Sensitive-keyword exfiltration | single-session | 150 | 1.5 % |
| S2 | Vague justification + privileged command | linguistic | 120 | 1.2 % |
| S13 | Lateral SSH chain | cross-session | 120 | 1.2 % |
| S14 | Privilege escalation pattern | cross-session | 90 | 0.9 % |
| S15 | Approver self-implements own change | governance | 22 | 0.2 % |

S15 is intentionally rare because the real-world signal is a one-in-a-thousand event — it is the hardest scenario for either detector.

---

## Stages of the pipeline

`smoke.py` runs five stages in order. Each is a separate subprocess so failures land exactly where they happened.

| # | Stage | Script | Reads | Writes |
|---|---|---|---|---|
| 1 | Generate synthetic dataset | `generator/generate.py` | (vocab/*.json) | `data/sessions.json`, `labels.json`, `split.json`, `manifest.json` |
| 2 | Score rule detector | `rule_detector/detect.py` | sessions, split | `data/rule_predictions.json` |
| 3 | Train hybrid LLM detector | `llm_detector/train.py` | sessions, split | `data/llm_model/`, `training_log.csv` |
| 4 | Score trained LLM detector | `llm_detector/detect.py` | sessions, split, llm_model | `data/llm_predictions.json` |
| 5 | Compare on test split | `compare.py` | predictions, labels, split | `data/comparison.json`, `comparison.md` |

The `--backend mock` path uses a small in-process log-odds classifier with no external models — useful for CI and the smoke loop. The `--backend huggingface` path swaps in **Flan-T5-base** for paraphrase augmentation in stage 1 and **DistilBERT-base-uncased** for fine-tuning in stage 3.

---

## GPU / CPU runtime

| Configuration | Stage 1 | Stage 3 | End-to-end |
|---|---:|---:|---:|
| Mock backend, any CPU | 5 s | 4 s | ~15 s |
| HuggingFace, CPU only | 8–12 min | 18–25 min | ~30–40 min |
| HuggingFace, single T4 (Kaggle) | 90 s | 5–6 min | ~10–12 min |
| HuggingFace, single A100 / RTX 4090 | 30 s | 2–3 min | ~5–6 min |

`setup_gpu.ps1` (Windows) auto-detects your CUDA version and installs a matching torch wheel. Linux / macOS users install torch from <https://pytorch.org/get-started/locally/> first, then `pip install -r requirements.txt`.

---

## FAQ

**1. Do I need a GPU?**
No. The mock backend runs the entire pipeline in ~15 seconds on any laptop and produces a real (smaller, less polished) anomaly-detection result. Use the HuggingFace backend only when you need the published numbers.

**2. Do I need a Hugging Face token?**
Only for the HuggingFace backend, and even then only to avoid anonymous-download rate limits. `python smoke.py --backend huggingface` will prompt you interactively the first time, save it to `.hf_token` (gitignored), and reuse it on subsequent runs. A read-scope token is enough.

**3. The HuggingFace stage crashes with `exit 3221225477` on Windows.**
That's `STATUS_ACCESS_VIOLATION` (0xC0000005), almost always a Windows OpenMP DLL conflict between torch's MKL and the OMP runtime pulled in by tokenizers / safetensors. The smoke harness sets the right `KMP_*` and `OMP_*` env vars before spawning subprocesses, but if you run a stage directly the guards are bypassed. Run `python doctor.py` — it walks every torch/transformers/datasets import in isolation and prints the last successful step before the crash.

**4. The trained LLM detector reports F1 = 0.0 on a small run.**
On runs with `--num-sessions 1000` you only get ~75 optimizer steps in stage 3, which isn't enough for DistilBERT to converge on the 70/30 imbalance. Use `--num-sessions 10000` if you want the published reference numbers; lower values are smoke-only.

**5. Stage 1 hangs at "loading google/flan-t5-base".**
First-time download is ~990 MB and Kaggle / corporate networks sometimes throttle Hugging Face. With a token configured, this completes in 30–60 seconds. Without one, anonymous rate-limits can stretch it to 5–10 minutes. If you see no progress for >10 minutes, ctrl-C, configure a token, and re-run.

**6. Can I plug in a real PAM log instead of the synthetic generator?**
Yes — write a converter that emits the same `sessions.json` / `labels.json` schema (the generator's source is the schema reference) and skip stage 1. The detectors and the comparator are dataset-agnostic; nothing in stages 2–5 inspects ground-truth scenario codes other than for the per-scenario recall breakdown.

**7. Why both a rule detector and an LLM detector?**
Defense in depth and accountability. The rule detector is fully auditable (every flag traces back to a deterministic predicate) and recovers most attacks on its own. The LLM detector picks up linguistic and structural edges the rules can't articulate. Their disagreement profile (visible on the dashboard's *scenario × detector-agreement* chart) is itself a useful audit artefact.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | torch not installed (the mock backend doesn't need it but the HF backend does) | Linux/macOS: `pip install torch`. Windows: `.\setup_gpu.ps1`. |
| `OSError: [Errno 28] No space left on device` | Hugging Face cache fills `/tmp` on small VMs | `export HF_HOME=/path/with/space` before running, or clean `~/.cache/huggingface/hub/`. |
| Stage 3 takes >40 min on a desktop | Falling back to CPU silently | Confirm with `python -c "import torch; print(torch.cuda.is_available())"`. If False, use Kaggle (notebook in `notebooks/cloud_run.ipynb`). |
| `KeyError: 'session_id'` from a detector | You started from a half-finished run (sessions.json present, labels.json absent) | `rm -rf data/` and re-run `smoke.py`. |
| Numbers don't match the published reference | Different `--num-sessions` or different `--expand-factor` | Use `--num-sessions 10000 --expand-factor 2 --backend huggingface` to reproduce the headline numbers. |
| `exit 3221225477` on Windows | OpenMP DLL conflict | `python doctor.py` to localise; the smoke harness's env-var guards handle most cases automatically. |
| `Can't pickle <function ...>` from datasets | Old `datasets` library | `pip install -U "datasets>=3"`. |
| Kaggle's preinstalled transformers fails | Pre-v5 transformers doesn't support the `processing_class` Trainer API | The notebook's first cell upgrades to `transformers>=5.5,<6` — don't skip that cell. |

If something here doesn't match what you're seeing, run `python doctor.py` — it walks every torch / transformers / datasets import in isolation and prints the last successful step, which usually localises the failure inside one DLL.

---

## Sample run

A `--backend mock --non-interactive --num-sessions 200` run looks like this end-to-end:

```
Backend: mock   num-sessions: 200   expand-factor: 1   stages: 5

────────────────────────────────────────────────────────────────────────
  Stage 1/5: Generate synthetic dataset
  $ python -u generator/generate.py --num-sessions 200
────────────────────────────────────────────────────────────────────────
  generator: emitted 200 sessions  (140 clean / 60 suspicious)
  Stage 1/5 ok in 0.4s

────────────────────────────────────────────────────────────────────────
  Stage 2/5: Score rule detector
  $ python -u rule_detector/detect.py
────────────────────────────────────────────────────────────────────────
  rule_detector: scored 200 sessions  (58 flagged suspicious)
  Stage 2/5 ok in 0.1s

────────────────────────────────────────────────────────────────────────
  Stage 3/5: Train trained-LLM detector
  $ python -u llm_detector/train.py
────────────────────────────────────────────────────────────────────────
  llm_detector/train: backend=mock  log-odds fit on 160 sessions
  Stage 3/5 ok in 0.3s

────────────────────────────────────────────────────────────────────────
  Stage 4/5: Score trained-LLM detector
  $ python -u llm_detector/detect.py
────────────────────────────────────────────────────────────────────────
  llm_detector/detect: scored 200 sessions  (54 flagged suspicious)
  Stage 4/5 ok in 0.1s

────────────────────────────────────────────────────────────────────────
  Stage 5/5: Compare detectors on test split
  $ python -u compare.py
────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────
  Detector evaluation — test split (40 sessions)
────────────────────────────────────────────────────────────────

  Detector       Precision    Recall       F1   Accuracy
  ──────────────────────────────────────────────────────
  rule              0.9167    0.9167   0.9167     0.9500
  llm-trained       0.8462    0.9167   0.8800     0.9250

  Winner: rule  (F1 +0.0367, +0 missed attacks, -2 false alarms vs llm-trained)

  Per-scenario breakdown:  data/comparison.md
  Machine-readable report: data/comparison.json

  Stage 5/5 ok in 0.2s

All 5 stages completed in 1.1s.
```

---

## Limitations

- **Synthetic data only.** The 10,000-session corpus is generated by templates and Flan-T5 paraphrases. Real PAM logs have command vocabularies, host distributions, and approver behaviours we have not modelled. Numbers on real logs will differ.
- **Single domain.** Every scenario assumes IT change-management framing (tickets, approvers, change windows). Adapting this to a different operational context (DevOps shells, database admin, robotic process automation) requires rewriting the scenario library.
- **Twelve scenarios are not exhaustive.** Real insider risk has long-tail behaviours (kill-chains across weeks, supply-chain compromise, credential abuse via legitimate sessions) that this corpus does not include.
- **Imbalance is fixed at 70/30.** The class balance is not varied in the evaluation. In practice, real PAM-log positive rates are closer to 0.1 %–1 %; a model trained at 30 % positive will need recalibration.
- **Adversarial robustness is probed, not exhausted.** The S16 vocabulary-cloak scenario paraphrases the destructive-SQL trigger words the rule detector matches on; it covers one rule family. Cloak variants for the other rule families, and an active attacker who adapts after seeing the detector's output, are outside the evaluation.
- **Threshold = 0.5 is convenient, not optimal.** Probability calibration is well-separated between classes, but a real deployment would tune the threshold against the operating organisation's false-alarm budget.
- **No streaming / online updating.** The pipeline is batch — train once, score once. Adapting to drift would require either retraining on a sliding window or adding online-learning machinery.

---

## Project layout

```
code/
├── README.md                      ← you are here
├── CITATION.cff                   ← citation metadata
├── requirements.txt               ← pinned Python dependencies
├── setup_gpu.ps1                  ← Windows GPU torch installer
├── smoke.py                       ← five-stage end-to-end harness
├── compare.py                     ← evaluation + minimalist console summary
├── doctor.py                      ← Windows DLL-conflict diagnostic
├── hf_token.py                    ← Hugging Face credential helper
├── generator/                     ← synthetic dataset generator
│   ├── generate.py
│   ├── llm_augment.py             ← Flan-T5 paraphrase + command-pool expansion
│   └── vocab/                     ← scenario vocabularies (10 JSON files)
├── llm_detector/                  ← trained-LLM detector
│   ├── train.py                   ← DistilBERT fine-tune + categorical MLP fusion
│   └── detect.py                  ← inference
├── rule_detector/                 ← deterministic baseline
│   └── detect.py
└── notebooks/
    └── cloud_run.ipynb            ← Kaggle T4-x2 turn-key notebook
```

---

## Citation

If you use this work, please cite via the `CITATION.cff` metadata. GitHub renders a "Cite this repository" button automatically once the file is present.
