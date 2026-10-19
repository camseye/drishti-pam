# Drishti-PAM
*The gaze that sees beyond the surface — hybrid structural-deviation detection for privileged-access monitoring.*

**Live dashboard**: <https://camseye.github.io/drishti-pam/>
*(rebuilt and redeployed automatically on every push to `main` — see [GitHub Actions workflow](./.github/workflows/pages.yml))*

> Sanskrit *दृष्टि (Drishti)* names the perception that reads structure
> under vocabulary — the quality of seeing past the surface words to the
> structural relationship beneath them. That is the central technical
> claim of this detector's result on the **S16 vocabulary-cloak
> adversarial control**: an attacker who paraphrases the literal trigger
> words a rule-based detector matches on cannot, by structural
> construction, cloak the deviation between *what the change ticket
> promised* and *what the recorded session shows*. Drishti-PAM is the
> detector that reads the deviation, not the surface.

## AI Use Disclosure

Parts of this project were developed with the assistance of AI tools, including large language models such as [e.g., ChatGPT / Claude / GitHub Copilot / Perplexity]. AI was used for [e.g., brainstorming, code suggestions, debugging, documentation drafting].

All AI-generated output was reviewed, tested, and edited by the author, who is responsible for the final code, design decisions, and documentation. The project's research ideas, methodology, and conclusions are the author's own.

This software is provided "as is", without warranty. Because AI-assisted code may contain errors or security flaws, review and test it before using it in any production or security-critical environment. Do not input sensitive or confidential data (credentials, logs, or personal information) into any AI-assisted component without proper safeguards.

## What this repository contains

A reproducible, two-detector pipeline for catching insider-risk
behaviour in **Privileged Access Management (PAM) session logs**.
A deterministic rule baseline runs alongside a hybrid LLM classifier;
both are evaluated on the same held-out split so their numbers are
directly comparable.

| Component | Path | Purpose |
|---|---|---|
| Synthetic corpus generator | [`code/generator/`](./code/generator/) | Twelve organic insider-risk scenarios + one adversarial vocabulary-cloak control (S16), generated as PAM session telemetry with curated vocabulary files and LLM-augmented justifications. |
| Rule detector | [`code/rule_detector/`](./code/rule_detector/) | Twelve plain-Python rules, fifty lines each, auditable end-to-end. |
| Hybrid LLM detector | [`code/llm_detector/`](./code/llm_detector/) | Fine-tuned `distilbert-base-uncased` encoder fused with engineered categorical features, trained with focal loss + class-weighted sampling. |
| SOAR / SIEM integration sketches | [`code/integrations/`](./code/integrations/) | Splunk SPL, Microsoft Sentinel KQL, and Cortex XSOAR playbook artefacts plus a multi-stage Dockerfile. |
| Dashboard | [`code/docs/`](./code/docs/) | Static-HTML triage console with seven KPI tiles, risk distribution donut, scenario × detector-agreement chart, threat-coverage tile, and per-session drawer. |

## How to run the project

The complete flow takes one terminal session. Each step writes JSON
files into `code/data/`; the final step regenerates the dashboard's
data layer from whatever is present and serves the static site.
Every input is optional — when something is missing, the dashboard
renders a graceful placeholder rather than failing.

### 1. Clone and install

```bash
git clone https://github.com/camseye/drishti-pam.git
cd drishti-pam/code
python -m venv .venv
source .venv/bin/activate    # on Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate prediction data

Pick one of the two paths below. The smoke run is a 90-second demo
suitable for verifying the install; the Kaggle run produces the
canonical reference numbers shown on the dashboard's hero KPIs.

**Path A — Smoke run** (90 s, mock LLM backend, 500 sessions):

```bash
python smoke.py --backend mock --non-interactive
```

**Path B — Kaggle full run** (~12 min on a free-tier T4, real
HuggingFace backend, 2,000-session test partition):

Open [`code/notebooks/cloud_run.ipynb`](./code/notebooks/cloud_run.ipynb)
on Kaggle, run all cells, download the output zip, and unzip its
contents into `code/data/`. Then seed the dashboard's hero-KPI
reference snapshot:

```bash
cp data/comparison.json data/reference_comparison.json
```

Commit `data/reference_comparison.json` to make the canonical Kaggle
numbers appear on the live GitHub Pages dashboard's hero KPIs (the
file is allowlisted in `.gitignore`; everything else under `data/`
stays gitignored). Without this commit, the live dashboard's hero
falls back to a placeholder while the rest of the deployment uses
the CI smoke-run numbers.

Either path produces the four prediction files the dashboard reads:
`comparison.json`, `rule_predictions.json`, `llm_predictions.json`,
`labels.json`.

### 3. Build the dashboard

One command reads everything in `code/data/` and writes the
dashboard's data layer into `code/docs/data/`:

```bash
python docs/build_dashboard.py
```

Re-run this whenever any input file changes.

### 4. Serve the dashboard

**Local — Python http.server**:

```bash
python -m http.server 8000 -d docs
```

Open <http://localhost:8000> in any browser.

**Local — Docker** (no Python install required on the host):

```bash
docker build -t drishti-pam:latest .
docker run --rm -p 8000:8000 drishti-pam:latest
```

See [`code/integrations/README.md`](./code/integrations/README.md)
for the multi-stage Dockerfile details and SOAR/SIEM integration
sketches.

**Public — GitHub Pages**: any push to `main` triggers
[`.github/workflows/pages.yml`](./.github/workflows/pages.yml),
which runs the smoke pipeline + dashboard build in CI and publishes
`code/docs/` to GitHub Pages. After enabling Pages once
(*Settings → Pages → Source = "GitHub Actions"*), the live URL is
`https://<owner>.github.io/<repo>/` and refreshes ~2 minutes after
each push.

### What refreshes automatically

| When you change … | Re-run | The dashboard updates … |
|---|---|---|
| `code/data/comparison.json` (smoke or Kaggle output) | `python docs/build_dashboard.py` | Combined F1 KPI · scenario × detector-agreement chart · operational FP triage cost tile · numeric tables |
| `code/data/reference_comparison.json` (frozen Kaggle snapshot) | `python docs/build_dashboard.py` | Hero KPI numbers (combined F1 · detector agreement · analyst-hour cost reduction) |
| `code/threat_model.json` (STRIDE × ATT&CK × kill-chain map) | `python docs/build_dashboard.py` | Threat-coverage tile |
| `code/regulatory_map.json` (PCI-DSS / NIST 800-53 / EU AI Act mapping) | `python docs/build_dashboard.py` | Regulatory mapping tile |

## Headline numbers (test partition · 2,000 sessions · seed = 17)

| Detector | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Rule | 0.9560 | 0.9767 | 0.9662 | 0.9795 |
| Hybrid LLM | 0.9878 | 0.9483 | **0.9677** | 0.9810 |

The two detectors disagree on only thirty-one of two thousand test
sessions. The disagreement pattern, not the headline F1 difference,
is the most informative signal: open the dashboard's *scenario × detector-
agreement* chart to see exactly where each detector wins and what
that implies for production deployment.

## Where to start reading

Start with [`code/README.md`](./code/README.md) for the long-form
architecture, scenario data card, FAQ, and troubleshooting; see
[`code/integrations/README.md`](./code/integrations/README.md) for
the SOAR/SIEM artefacts and Dockerfile; open
[`code/docs/`](./code/docs/) (or the GitHub Pages mirror) for the
live dashboard.

## License

Released under the [MIT License](./LICENSE).

## Citation

Citation metadata lives in [`code/CITATION.cff`](./code/CITATION.cff).
