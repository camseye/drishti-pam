# SOAR / SIEM integration sketches

This folder ships four small reference artefacts that show how
**Drishti-PAM**'s detector output plugs into a SOC stack: a Splunk
search, a Microsoft Sentinel query, a Cortex XSOAR playbook, and a
Dockerfile for the dashboard.

The artefacts are reference sketches, not production rules. They are
syntactically valid (verified by `_validate.py`) but have not been run
against a live Splunk, Sentinel, or XSOAR tenant. Adapt and test them
in your own environment before production use.

## What ships in this folder

```
integrations/
├── __init__.py               package marker
├── splunk_detection.spl      Splunk SPL search over rule_predictions.json
├── sentinel_rule.kql         Microsoft Sentinel KQL over the same shape
├── xsoar_playbook.json       Cortex XSOAR v6 playbook stub (5-task spine)
├── _validate.py              stdlib syntactic-check script
└── README.md                 this file
```

The companion `Dockerfile` at the repo root packages the static
dashboard so a SOC engineer can `docker build` + `docker run` the
project without installing Python — see "Docker" below.

## What does NOT ship

* **Live integrations.** No Splunk app package, no Sentinel solution
  package, no XSOAR pack. Each artefact is a single file that a
  customer would drop into their existing tenant and adapt.
* **A CyberArk PSM REST adapter.** The XSOAR playbook stubs out the
  "fetch session details" call as a generic HTTP GET against
  `${pam_api_base}/API/Recordings/${incident.session_id}`; point it
  at your own CyberArk PSM tenant.
* **Authentication and secret management.** All three sketches
  assume the upstream platform's standard auth path (Splunk role-
  based access, Sentinel managed identity, XSOAR vault) is already
  configured by the customer's platform team.

## What each artefact does

### `splunk_detection.spl`

A single Splunk search-processing-language query that ingests
`code/data/rule_predictions.json` (and joins on
`code/data/llm_predictions.json` for the agreement column) and
emits a 200-row triage table sorted by risk. Risk thresholds
(`CRITICAL` ≥ 0.92 + ≥ 2 rules, `HIGH` ≥ 0.85, `MEDIUM` ≥ 0.70,
`LOW` otherwise) mirror the dashboard's risk-rating algorithm
verbatim, so the SPL output and the dashboard output rank sessions
identically.

### `sentinel_rule.kql`

The same logic in Microsoft Sentinel's Kusto Query Language against
two custom-log tables: `PAMRulePredictions_CL` and
`PAMLlmPredictions_CL`. KQL has different idioms than SPL (pipe
stages as functions, type-strict joins, `parse_json` for nested
fields) so this is a re-expression rather than a transliteration of
the SPL. Includes a `parse_json(scenario_hits_s)`
stage showing how the JSON-blob `evidence` field would be flattened
into Sentinel's columnar schema.

### `xsoar_playbook.json`

A Cortex XSOAR v6-conformant playbook stub that stitches the SIEM
alert into an analyst-handoff workflow:

```
[Alert: PAM CRITICAL detected]
   → [task 1: HttpV2  — fetch full session via PAM REST]
       → [task 2: ADGetUser  — enrich user from HR]
           → [task 3: condition  — risk == CRITICAL?]
               ├─ yes  → [task 4a: Tier-2 SOC, severity HIGH]
               └─ else → [task 4b: Tier-1 SOC, severity MEDIUM]
                   → [task 5: title  — wait for analyst verdict]
```

The stub captures the 5-task spine. A production playbook would
extend task 5 into three branches (auto-revoke vault credential on
confirmed-true-positive, append to retrain feedback queue on false-
positive, follow-up case on inconclusive), but that surface is
customer-specific and out of scope for this sketch.

### `_validate.py`

A stdlib-only validator that runs:

* SPL → at least 5 pipe stages, balanced quotes / parens / brackets,
  references the expected sourcetype, computes a risk rating, ends
  with an analyst-readable output stage.
* KQL → at least 5 pipe stages, balanced parens / brackets, uses at
  least one `extend` / `where` / `project` / `order`, references both
  custom-log tables, computes a risk rating, ends with a `take` or
  `order` so the output is bounded.
* XSOAR JSON → parses, has the four required top-level keys
  (`id`, `name`, `starttaskid`, `tasks`), the `starttaskid` points at
  a real task, every `nexttasks` edge points at a real task, and the
  playbook spans `start` + `regular` + `condition` task types.

Run with:

```powershell
cd code
python -m integrations._validate
```

It exits 0 on success and 1 on any artefact failing. The smoke
harness in `code/smoke.py` does **not** call this script (that would
couple the data pipeline to the integrations folder); the dashboard
builder in `code/docs/build_dashboard.py` does **not** call it
either. Run it yourself before committing changes to any artefact
in this folder.

## Docker — packaging the static dashboard

The companion `Dockerfile` at the repo root (sibling of `code/`)
packages the static dashboard so a SOC analyst or demo audience can
run the project without a Python toolchain:

```bash
# from the repo root (the directory that contains code/ AND Dockerfile)
docker build -t drishti-pam:latest .
docker run --rm -p 8000:8000 drishti-pam:latest
# open http://localhost:8000 in a browser
```

The build is multi-stage:

* **stage 1 — builder** (`python:3.13-slim`) — installs the project's
  Python deps from `code/requirements.txt`, runs the smoke harness in
  mock-backend mode (~15 s, no GPU), then runs `docs/build_dashboard.py`
  to materialise the dashboard data files.
* **stage 2 — runtime** (`python:3.13-alpine`) — copies *only* the
  rendered `code/docs/` folder out of the builder image, runs
  `python -m http.server 8000 --directory docs` on port 8000.

The resulting runtime image is approximately 50 MB compressed against
~600 MB for a single-stage build of the same content. Build context is
constrained by the `.dockerignore` next to the Dockerfile so
generated data and the trained-model checkpoints (when present) are
not copied into the build context.
