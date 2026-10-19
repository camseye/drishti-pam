# Drishti-PAM dashboard

This folder contains the static HTML dashboard that ships with
**Drishti-PAM** on GitHub Pages.

The dashboard has two tiers. The top-of-page **hero** section
communicates the project to an executive reader in five seconds: project
name, one-line pitch, three headline KPIs (combined F1, detector
agreement, analyst-hours saved), and a link to the GitHub repository.
Below the hero, the **operator console** preserves the seven-KPI triage
view, the filter
sidebar, the risk distribution donut, the scenario × detector-agreement
chart, the threat-coverage tile, and the per-session drawer — built
around a composite Tier-2 SOC analyst's daily triage workflow.

```
docs/
├── index.html              dashboard markup (2-tab SPA + threat tile)
├── styles.css              cyber-dark-neon theme + light override
├── app.js                  Triage tab — filtering, sorting, donut, drawer, threat coverage, router
├── data.js                 Data Source tab — raw corpus table (lazy-loaded)
├── build_dashboard.py      ETL: ../data/*.json + ../threat_model.json -> ./data/*.json
├── data/                   built artefacts (regenerated, not edited by hand)
│   ├── summary.json              aggregate KPI counts + threat_coverage block (~2 KB)
│   ├── sessions_summary.json     lean table rows for triage (~3 MB at 10K)
│   ├── sessions_details.json     full per-session payload, lazy on drawer open (~25 MB at 10K)
│   ├── sessions_raw.json         raw corpus rows for Data Source tab, lazy on first reveal (~7 MB at 10K)
│   └── threat_model.json         STRIDE × ATT&CK × kill-chain mapping for the Threat Coverage tile + drawer chips (~3 KB)
└── README.md               this file
```

## Two views, one page

The dashboard is a tiny single-page application with two routes:

| Tab | URL | What it answers |
|-----|-----|-----------------|
| **Triage Console** (default) | `#/triage` | "Of the sessions on the test split, which ones do I need to look at?" — model output, risk + compliance per session |
| **Data Source** | `#/data` | "What did the model actually see?" — every generated session, with its split badge and ground-truth label, before any model touched it |

The router lives in `app.js` (`wireRouter` / `applyRoute`) and is
hash-based, so deep links like `https://…/#/data?` survive page
reloads and Pages caching.

## Features

- Header (title, GitHub link, dark/light theme toggle)
- Tab strip — Triage / Data Source
- KPI strip (7 cards: sessions / risky / non-compliant / top scenario / combined F1 / deployment / FP triage cost)
- Triage filter panel (risk · compliance · scenario · detector agreement · user/asset search)
- Risk-distribution donut + 5-row legend
- Detector-agreement-per-suspicious-scenario stacked bar chart
- Triage session table (sortable, paginated 50 rows/page, colour-coded badges + pills)
- Plain-English scenario labels (with `S1`/`S2`/… tag for cross-reference)
- Inline collapsible methodology explainers — risk + compliance algorithms
- Session detail drawer (click any row)
- Data Source filter panel (split · label · scenario · ticket category · user/asset search)
- Data Source raw table (15 columns, sortable, sticky session ID column, justification preview)
- Threat Coverage tile (STRIDE / MITRE ATT&CK / Kill Chain stats + top-6 techniques bar list)
- Drawer ATT&CK chips (per-scenario technique IDs + names)
- Deep links: `?session=sess-000042` opens that session's drawer · `#explain` opens both methodology explainers · `#/data` lands on Data Source tab

## Quick start (local preview)

After `code/smoke.py` has populated `code/data/`:

```powershell
cd code
python docs/build_dashboard.py            # produces docs/data/*.json
python -m http.server 8000 --directory docs
# open http://localhost:8000/ in your browser
```

The page reads `data/summary.json` and `data/sessions_summary.json`
relative to wherever it is served. Opening `index.html` directly with
a `file://` URL will fail because most browsers block fetch requests
to local JSON. Always serve via `python -m http.server` or any other
static server.

## Refresh after a Kaggle run

The dashboard reads the four per-session prediction files in
`code/data/`. After dropping fresh Kaggle artefacts into `code/data/`
(or running `code/smoke.py`):

```
code/data/sessions.json            (10K sessions)
code/data/labels.json              (10K labels)
code/data/rule_predictions.json    (10K records)
code/data/llm_predictions.json     (the test split, ~2K records)
```

Then:

```powershell
cd code
python docs/build_dashboard.py
```

The script joins the four files, computes a 5-level risk rating and a
3-level policy compliance flag for every session that appears in
**both** the rule and LLM prediction files (this is naturally the
test split), and emits the two compact JSON files the dashboard
loads at runtime.

## Enabling GitHub Pages

The workflow at `.github/workflows/pages.yml` rebuilds the dashboard
data and publishes this folder on every push to `main`. One-time
setup in the repository settings → Pages:

1. Source: **GitHub Actions**
2. Save.

GitHub will publish the page at
`https://<your-user>.github.io/drishti-pam/` about two minutes after
each push. The `data/` folder is gitignored because the workflow
regenerates it.

## Design contract

The dashboard answers one question for anyone who opens it:

> "Of the N sessions on the test split, which ones do I need to
> look at?"

It answers across four lenses:

| Lens | Component |
|------|-----------|
| Aggregate | KPI strip (7 cards: sessions / risky / non-compliant / top scenario / combined F1 / deployment / FP triage cost) |
| Visual | Risk-distribution donut |
| Tabular | Filterable, sortable, paginated session table |
| Operational | Filter panel: risk rating · compliance · scenario · detector agreement · user/asset search |

## Risk-rating algorithm

The page itself ships an inline collapsible explainer that walks
analysts through the algorithm in plain English. The reference
table:

| Rule says | LLM says | Extra condition | Risk |
|-----------|----------|-----------------|------|
| Suspicious | Suspicious | LLM ≥ 0.92 AND ≥ 2 distinct rules fired | **CRITICAL** |
| Suspicious | Suspicious | (any other) | **HIGH** |
| Suspicious | Clean | — | **MEDIUM** |
| Clean | Suspicious | LLM ≥ 0.70 | **MEDIUM** |
| Clean | Suspicious | LLM < 0.70 | **LOW** |
| Clean | Clean | — | **MINIMAL** |

The thresholds (0.92 LLM softmax, 2 distinct rules, 0.70 LLM
softmax) were picked to match the disagreement structure visible
on the *scenario × detector-agreement* chart: a 'CRITICAL' label
is reserved for high-confidence joint detections so it can drive
an analyst escalation without false alarms.

## Detector agreement per suspicious scenario

The bar chart between the donut and the table shows, for each
ground-truth suspicious scenario, how the two detectors voted.
Bars are stacked by the four agreement buckets:

| Stack colour | Meaning | Triage value |
|--------------|---------|--------------|
| Red — *Both flag* | Rule and LLM both said suspicious | High-confidence true positives |
| Orange — *Rule only* | Rule caught it, LLM missed | LLM blind spot — investigate why |
| Yellow — *LLM only* | LLM caught it, rule missed | Rule gap — may need a new rule |
| Grey — *Both clean* | Neither detector flagged it | **Missed detections — highest-value triage signal** |

Genuinely clean ground-truth sessions are excluded so the
suspicious bars stay readable (clean dwarfs everything ~30:1 on
realistic data). The chart respects every active filter, so
narrowing the panel to e.g. *Compliance = Non-compliant* re-draws
the chart against just those sessions.

## Plain-English scenario mapping

The dashboard never shows raw scenario IDs in isolation — every
`S1, S2, ...` code is rendered alongside (or replaced by) a
plain-English short label. Long descriptions appear as hover
tooltips. Full scenario reference:

| ID | Short label | Long description |
|----|-------------|------------------|
| S1 | Multi-user ticket | Several users sharing one change ticket within an hour. |
| S2 | Vague justification | Change explanation is too generic to defend. |
| S3 | Off-window session | Work happens before or after the approved change window. |
| S4 | Wrong-category commands | Commands don't match the ticket's stated category. |
| S5 | Off-hours work | Session starts outside business hours. |
| S7 | Destructive on read-only | DELETE/DROP/TRUNCATE on a non-destructive ticket. |
| S10 | Out-of-scope asset | Commands target a host not on the named-assets list. |
| S11 | Zombie ticket | Same ticket reused over many days, often outside its window. |
| S12 | Sensitive-data probing | Queries against PII columns or secret files. |
| S13 | Lateral movement | Jumping or probing other hosts beyond the ticket scope. |
| S14 | Privilege escalation | sudo/su without ticket authorisation. |
| S15 | Approver-as-implementer | Requester equals approver — segregation-of-duties break. |
| S16 | Vocabulary-cloak adversarial | Destructive SQL paraphrased (CALL purge_*, EXEC sp_purgehistory, ALTER … RENAME TO _orphaned_*) so it bypasses the rule detector's literal trigger phrases. Adversarial control-evaluation scenario; rule detector cannot catch S16 by construction. |

## Compliance algorithm

For every session, five behavioural signals are checked:

| Signal | Pass condition |
|--------|----------------|
| s1 justification length | ≥ 12 words |
| s2 command count | ≤ 50 |
| s3 business hours | session start within Mon–Fri 09:00–18:00 |
| s4 sensitive keywords | no command matches a token from `generator/vocab/sensitive_keywords.json` |
| s5 justification ↔ command overlap | ≥ 1 shared keyword (after stopword removal) |

| Pass count | Label |
|-----------:|-------|
| 5 | fully_compliant |
| 4 | partially_compliant |
| 0–3 | non_compliant |

The "non-compliant" threshold is deliberately strict — calibrated
against the synthetic corpus, "more than one signal failing" is the
right bar for a triage queue. Loosening it to "≤2 signals fail"
collapses almost the whole corpus into 'fully compliant' and the
KPI loses signal.

Signals are deliberately independent of the suspicious-or-not
ground truth. Even ground-truth-clean sessions can be partially or
non-compliant if (for example) they run outside business hours or
have a thin justification — exactly the kind of finding an analyst
would want surfaced in a real PAM review.

## Data Source view

The second tab presents the corpus the detectors were fed — every
session that came out of `code/generator/`, before either detector
touched it. Use it to:

* Sanity-check that the synthetic generator produced what you
  expected (split balance, scenario coverage, ticket-category
  distribution).
* Look up the raw inputs of any session you spot in the Triage
  Console — copy the `sess-…` ID from the triage view and search
  here.
* Audit the train/test split before retraining.

| Column | Meaning |
|--------|---------|
| **Session** | Stable session ID (`sess-NNNNNN`). Sticky on horizontal scroll. |
| **Split** | `train` / `test` / `unknown` — from `code/data/split.json`. |
| **Label** | Ground-truth label assigned by the generator (`clean` / `suspicious`). |
| **Scenario** | Plain-English scenario name + S-tag (e.g. *Multi-user ticket (S1)*). |
| **User**, **Asset** | The privileged user and target host. |
| **Ticket** | Change-request ID, **Category**, and **Approver**. |
| **Start**, **Change window** | Session start time + ticket-approved window. |
| **# Cmd**, **# Assets**, **# Vault** | Counts of executed commands, named assets on the ticket, vault fetches. |
| **Justification** | The free-text reason on the ticket — truncated, full text on hover. |

The view is read-only. No row-click drawer (the drawer is a
triage-side concept that surfaces what the model did with a
session, not what the session contains). Filters are scoped to
data-quality concerns: split, label, scenario, ticket category,
and a free-text user/asset search.

`sessions_raw.json` is **lazy-loaded** — it doesn't ship on
initial page paint. The fetch only happens the first time the user
clicks the Data Source tab (or lands directly on `#/data` via deep
link), so users who only ever look at the Triage Console don't
download the larger payload.

## Browser support

The dashboard targets evergreen Chrome / Firefox / Safari / Edge
released within the last 24 months. It uses CSS Grid, custom
properties, `aspect-ratio`, `<dialog>`, and `fetch`. No polyfills
shipped. Works on a phone in either orientation.

## Capturing a screenshot

Run `python -m http.server 8000 --directory code/docs` after
`code/docs/build_dashboard.py`, open `http://localhost:8000/` in a
modern browser, and capture the live page. The dashboard is a
single-page application so a single screenshot covers the hero +
operator console; for the per-session drawer or the *Data Source*
tab, navigate first then capture.
