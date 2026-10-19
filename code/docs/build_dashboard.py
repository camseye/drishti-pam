"""ETL for the Drishti-PAM dashboard.

Reads from `code/data/`:
    sessions.json            10,000 generated PAM sessions
    labels.json              ground-truth scenario + clean/suspicious label
    rule_predictions.json    rule detector output (all sessions)
    llm_predictions.json     hybrid LLM output (test split only)
    split.json               train/test partition (seed + id lists)

Reads from `code/generator/vocab/`:
    sensitive_keywords.json  three categories: pii_columns,
                             secret_files, sensitive_command_examples

Reads from `code/`:
    threat_model.json        STRIDE × MITRE ATT&CK × Cyber Kill Chain
                             mapping for the 12 risk scenarios.
                             Optional — when absent the dashboard's
                             Threat Coverage tile renders an empty
                             state.

Emits to `code/docs/data/`:
    summary.json             aggregate KPIs for the header strip.
                             Also ships a compact ``threat_coverage``
                             block (technique / tactic / phase counts +
                             top techniques) so the Threat Coverage
                             tile can render without fetching the full
                             threat_model.json.
    sessions_summary.json    one record per session in the intersection
                             of rule_predictions and llm_predictions —
                             naturally the test split where both
                             detectors have produced scores. Lean — just
                             the fields the table and filters need.
                             ~250 B per session, ~2.5 MB at 10K.
    sessions_details.json    full per-session data for the drawer:
                             justification, command stream, rule
                             evidence, LLM evidence, compliance
                             signals, etc. Lazy-loaded by the front-end
                             on first drawer click. ~1.5 KB per session,
                             ~15 MB at 10K. Keyed by session id.
    sessions_raw.json        ALL generated sessions (10K) in a flat,
                             input-only view: identity, ticket metadata,
                             counts, ground-truth label + scenario, and
                             the train/test split badge. Powers the
                             dashboard's second view (the "Data Source"
                             tab). Lazy-loaded by the front-end only on
                             first navigation to that tab.
                             ~500 B per session, ~5 MB at 10K.
    threat_model.json        Full per-scenario STRIDE / ATT&CK / kill-
                             chain mapping plus the same coverage
                             summary. Always emitted (a stub when the
                             source is absent) so the front-end can
                             render a stable schema. ~3 KB.

The dashboard reads the same source of truth that the smoke-test
pipeline emits into ``code/data/``. After a Kaggle run:

    cd drishti-pam/code
    # (drop the four data/*.json files into code/data/)
    python docs/build_dashboard.py

Risk rating uses the hybrid algorithm (option C):
    - Both detectors flag         -> HIGH (or CRITICAL if
                                     llm_score >= 0.92 AND >= 2 distinct
                                     rules fired)
    - Rule only flags             -> MEDIUM (rule has interpretability)
    - LLM only flags              -> MEDIUM if llm_score >= 0.70 else LOW
    - Both call clean             -> MINIMAL

Compliance uses the composite γ algorithm — five behavioural signals,
each independent of the suspicious-or-not label:
    s1  justification length (words) >= 12
    s2  command count <= 50
    s3  session start within business hours (Mon-Fri, 09:00-18:00)
    s4  no command contains a sensitive-keyword token
    s5  >=1 keyword overlap between justification and command stream

    5/5 pass        -> fully_compliant
    4/5 pass        -> partially_compliant
    0-3 pass        -> non_compliant

Standard library only — no extra dependencies beyond what code/ already
needs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent           # code/docs/
CODE_ROOT = HERE.parent                           # code/
DEFAULT_DATA_DIR = CODE_ROOT / "data"
DEFAULT_VOCAB_DIR = CODE_ROOT / "generator" / "vocab"
DEFAULT_THREAT_MODEL_PATH = CODE_ROOT / "threat_model.json"
DEFAULT_OUT_DIR = HERE / "data"
# Frozen reference comparison (full Kaggle run, 2,000-session test
# partition). The Drishti-PAM dashboard hero binds its three headline
# KPIs (combined F1, detector agreement, analyst-hour cost
# reduction) to this file rather than to the live
# `code/data/comparison.json` so the hero shows the canonical headline
# numbers regardless of whether the local data dir holds a smoke run
# (500 sessions) or a full Kaggle re-run. Optional: when absent, the
# hero block emits a placeholder rather than failing.
#
# Default path lives inside the public code/data/ tree. To populate it,
# either copy a frozen comparison.json from a full Kaggle run or pass
# --reference-comparison <path> to override.
DEFAULT_REFERENCE_COMPARISON_PATH = (
    DEFAULT_DATA_DIR / "reference_comparison.json")

# Deployment-sketch artefacts. Detected at build time so the
# dashboard's Deployment KPI tile can show "Containerised" or
# "Local-only" depending on whether the SOAR/SIEM sketches and the
# Dockerfile actually exist next to this build. Optional: when absent
# the tile renders the "Local-only" branch with a sensible explanation.
DEFAULT_DOCKERFILE_PATH = CODE_ROOT / "Dockerfile"
DEFAULT_INTEGRATIONS_DIR = CODE_ROOT / "integrations"

# How many words count as a "long-enough" justification.
JUSTIFICATION_MIN_WORDS = 12

# Maximum command count a "tidy" session is allowed to have.
COMMAND_COUNT_MAX = 50

# Business-hours window (inclusive start, exclusive end).
BUSINESS_HOUR_START = 9
BUSINESS_HOUR_END = 18

# Stopwords stripped before the justification<->command keyword overlap
# check (signal s5). Kept short on purpose — the more aggressive the
# stopword list, the more "compliant" the dashboard reports the corpus,
# which is the wrong direction for a triage view.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
    "per", "that", "the", "this", "to", "was", "were", "will", "with",
    "would", "should", "could", "do", "does", "doing", "i", "we",
    "you", "they", "their", "his", "her", "your", "our", "us",
    "ticket", "request", "session", "approved", "pending", "approval",
    "via", "into", "onto", "over", "under", "after", "before", "during",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")
            if len(t) > 2 and t.lower() not in _STOPWORDS}


# --------------------------------------------------------------------------- #
# Risk rating + compliance logic
# --------------------------------------------------------------------------- #

def risk_rating(rule_call: str, llm_call: str, llm_score: float,
                distinct_rules: int) -> str:
    """Hybrid risk rating: agreement picks bucket, score positions within."""
    both_flag = rule_call == "suspicious" and llm_call == "suspicious"
    rule_only = rule_call == "suspicious" and llm_call == "clean"
    llm_only = rule_call == "clean" and llm_call == "suspicious"

    if both_flag:
        if llm_score >= 0.92 and distinct_rules >= 2:
            return "CRITICAL"
        return "HIGH"
    if rule_only:
        return "MEDIUM"
    if llm_only:
        return "MEDIUM" if llm_score >= 0.70 else "LOW"
    return "MINIMAL"


def compliance_check(session: dict, sensitive_terms: set[str]) -> dict:
    """Compute the five γ-compliance signals for one session.

    Returns a dict with the five booleans plus a three-level overall
    label and the count of passing signals. Each signal also gets a
    short human-readable explanation so the detail drawer can show
    'why is this session non-compliant?' without re-running the
    logic in JS.
    """
    ticket = session.get("ticket") or {}
    inner = session.get("session") or {}
    commands = inner.get("commands") or []
    justification = ticket.get("business_justification", "") or ""
    start_time = inner.get("start_time") or ""

    # s1: justification length
    just_word_count = len(justification.split())
    s1 = just_word_count >= JUSTIFICATION_MIN_WORDS

    # s2: command count cap
    cmd_count = len(commands)
    s2 = cmd_count <= COMMAND_COUNT_MAX

    # s3: business hours (Mon-Fri, 09:00-18:00 local time as recorded)
    s3 = False
    parsed_time = None
    if start_time:
        try:
            parsed_time = datetime.fromisoformat(start_time)
            in_hours = BUSINESS_HOUR_START <= parsed_time.hour < BUSINESS_HOUR_END
            in_weekday = parsed_time.weekday() < 5  # Mon=0, Sun=6
            s3 = in_hours and in_weekday
        except (ValueError, TypeError):
            s3 = False

    # s4: no command contains a sensitive-keyword token
    s4_offenders: list[str] = []
    for cmd in commands:
        cmd_text = (cmd.get("command") or "").lower()
        for term in sensitive_terms:
            if term in cmd_text:
                s4_offenders.append(term)
                break  # one hit per command is enough
    s4 = len(s4_offenders) == 0

    # s5: keyword overlap between justification and command stream
    just_tokens = _tokens(justification)
    cmd_tokens: set[str] = set()
    for cmd in commands:
        cmd_tokens |= _tokens(cmd.get("command") or "")
    overlap = just_tokens & cmd_tokens
    s5 = len(overlap) >= 1

    pass_count = sum([s1, s2, s3, s4, s5])
    # Calibrated against the synthetic data: most clean sessions
    # naturally pass 3-5 signals, so "fully compliant" requires all 5
    # and "non-compliant" means more than one signal failed. This
    # gives the triage view a meaningful non-compliant bucket without
    # being so strict that even ground-truth-clean sessions all fail.
    if pass_count == 5:
        label = "fully_compliant"
    elif pass_count == 4:
        label = "partially_compliant"
    else:
        label = "non_compliant"

    return {
        "label": label,
        "pass_count": pass_count,
        "signals": {
            "s1_justification_length": {
                "passed": s1,
                "value": just_word_count,
                "threshold": JUSTIFICATION_MIN_WORDS,
                "label": "justification ≥ 12 words",
            },
            "s2_command_count": {
                "passed": s2,
                "value": cmd_count,
                "threshold": COMMAND_COUNT_MAX,
                "label": "command count ≤ 50",
            },
            "s3_business_hours": {
                "passed": s3,
                "value": (parsed_time.isoformat() if parsed_time else None),
                # Pre-formatted display string keeps the JS from
                # round-tripping the naive timestamp through `Date`,
                # which would convert it to the browser's local
                # timezone and confuse the analyst (the synthetic
                # data uses naive ISO timestamps that should be read
                # as wall-clock).
                "display": (parsed_time.strftime("%a %d %b %Y, %H:%M")
                            if parsed_time else None),
                "label": "session within Mon–Fri 09:00–18:00",
            },
            "s4_sensitive_keywords": {
                "passed": s4,
                "offenders": sorted(set(s4_offenders))[:5],
                "label": "no commands contain sensitive keywords",
            },
            "s5_justification_command_overlap": {
                "passed": s5,
                "shared_terms": sorted(overlap)[:8],
                "label": "justification & commands share ≥ 1 keyword",
            },
        },
    }


# --------------------------------------------------------------------------- #
# Sensitive-keyword vocab loader
# --------------------------------------------------------------------------- #

def load_sensitive_terms(vocab_dir: Path) -> set[str]:
    """Flatten `sensitive_keywords.json` into a single lowercased set.

    The vocab file ships three categories: pii_columns, secret_files,
    sensitive_command_examples. For the s4 substring check we just
    union them and lowercase. Single-character entries are dropped so
    we don't false-positive on common letters.
    """
    path = vocab_dir / "sensitive_keywords.json"
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return set()
    terms: set[str] = set()
    for key in ("pii_columns", "secret_files", "sensitive_command_examples"):
        for term in raw.get(key, []) or []:
            t = (term or "").strip().lower()
            if len(t) >= 3:
                terms.add(t)
    return terms


# --------------------------------------------------------------------------- #
# Threat model (STRIDE × MITRE ATT&CK × Cyber Kill Chain)
# --------------------------------------------------------------------------- #

def load_threat_model(path: Path) -> dict | None:
    """Load the scenario-to-threat-model mapping if present.

    The file is the source of truth for the dashboard's "Threat
    coverage" tile. When absent, the dashboard tile is rendered
    empty (front-end shows a "missing" notice).
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: threat_model.json failed to load: {exc}",
              file=sys.stderr)
        return None


def _build_hero_summary(reference_comparison_path: Path) -> dict:
    """Build the ``hero`` block embedded in summary.json that powers
    the Drishti-PAM dashboard's hero section.

    Three headline KPIs:

    1. **Combined F1** \u2014 the better of {rule_f1, llm_f1} from the
       frozen reference run, with the other detector's F1 reported
       in the sub-line so the gap (or lack thereof) is visible.
    2. **Detector agreement** \u2014 share of test sessions on which the
       rule and hybrid LLM detectors return the same call (both flag
       or both clear), from the reference run's ``agreement`` block.
    3. **Analyst-hour cost reduction** \u2014 derived from the operational
       metrics block: rule's hi-baseline FP-hours minus llm-trained's
       hi-baseline FP-hours = analyst-hours saved per audit window
       at the 18-min/triage operational baseline.

    Reads the frozen reference comparison.json (path supplied by
    caller; defaults to the project's reference data directory).
    Falls back gracefully if the file is missing \u2014 emits
    ``available: False`` with a placeholder message and the hero
    binds default labels instead of numbers."""
    if not reference_comparison_path.exists():
        return {
            "available": False,
            "message": ("Reference comparison.json not found at "
                        f"{reference_comparison_path}; hero KPIs "
                        "unavailable. Hero will render placeholder text."),
        }
    try:
        with reference_comparison_path.open("r", encoding="utf-8") as f:
            ref = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "message": f"Failed to read reference comparison: {exc}",
        }

    # Headline F1 \u2014 better of the two detectors.
    reports = ref.get("reports", []) or []
    rule_rep = next((r for r in reports if r.get("detector") == "rule"), {})
    llm_rep = next((r for r in reports
                    if r.get("detector") in ("llm-trained", "llm_trained",
                                             "llm")), {})
    rule_f1 = ((rule_rep.get("binary") or {}).get("f1") or 0.0)
    llm_f1 = ((llm_rep.get("binary") or {}).get("f1") or 0.0)
    if llm_f1 >= rule_f1:
        best_label, best_f1, other_label, other_f1 = (
            "LLM", llm_f1, "Rule", rule_f1)
    else:
        best_label, best_f1, other_label, other_f1 = (
            "Rule", rule_f1, "LLM", llm_f1)
    f1_value_str = f"{best_f1:.3f}"
    f1_sub_str = (f"{best_label} {best_f1:.3f} \u00b7 "
                  f"{other_label} {other_f1:.3f} \u2014 within "
                  f"{abs(best_f1 - other_f1):.4f}")

    # Detector agreement \u2014 both_flag + both_clean over the test split.
    agr = ref.get("agreement", {}) or {}
    test_size = agr.get("test_size") or 0
    agree = (agr.get("both_flag") or 0) + (agr.get("both_clean") or 0)
    if test_size:
        agreement_value_str = f"{100 * agree / test_size:.1f}%"
        agreement_sub_str = (f"{agree:,} of {test_size:,} test sessions "
                             "\u00b7 rule vs hybrid LLM")
    else:
        agreement_value_str = "\u2014"
        agreement_sub_str = "agreement data unavailable"

    # Operational FP-cost saving \u2014 rule_hi - llm_hi at 18-min baseline.
    op = ref.get("operational_metrics", {}) or {}
    fp_cost = op.get("false_positive_cost", {}) or {}
    rule_fp = fp_cost.get("rule", {}) or {}
    llm_fp = fp_cost.get("llm-trained", {}) or {}
    rule_hours_hi = rule_fp.get("fp_hours_hi_baseline")
    llm_hours_hi = llm_fp.get("fp_hours_hi_baseline")
    fp_minutes_hi = ((op.get("assumptions") or {})
                     .get("fp_triage_minutes_hi_baseline", 18.0))
    if (rule_hours_hi is not None and llm_hours_hi is not None
            and rule_hours_hi >= llm_hours_hi):
        savings_hours = round(rule_hours_hi - llm_hours_hi, 1)
        opcost_value_str = f"{savings_hours} hr / window"
        opcost_sub_str = (
            f"Rule {rule_hours_hi:.1f} hr \u2192 LLM {llm_hours_hi:.1f} hr "
            f"at {int(fp_minutes_hi)}-min/triage")
    else:
        opcost_value_str = "\u2014"
        opcost_sub_str = "operational metrics unavailable"

    return {
        "available": True,
        "f1_value": f1_value_str,
        "f1_sub": f1_sub_str,
        "agreement_value": agreement_value_str,
        "agreement_sub": agreement_sub_str,
        "opcost_value": opcost_value_str,
        "opcost_sub": opcost_sub_str,
        "_source": str(reference_comparison_path.name),
        "_doc": ("Hero KPIs. Sourced from the frozen "
                 "reference comparison.json so the hero renders the "
                 "headline numbers regardless of whether the local "
                 "data dir holds a smoke run."),
    }


def _build_deployment_summary(dockerfile_path: Path,
                              integrations_dir: Path) -> dict:
    """Build the ``deployment`` block embedded in summary.json.
    Inspects the repo for a Dockerfile + the integrations/ folder;
    emits an ``available: True`` block when both exist, an
    ``available: False`` block otherwise. The front-end renders the
    Deployment KPI tile from this block; the field is purely
    informational and never affects detector / triage output."""
    dockerfile_present = dockerfile_path.exists()
    integration_files: list[str] = []
    if integrations_dir.exists() and integrations_dir.is_dir():
        for ext in ("*.spl", "*.kql", "*.json"):
            integration_files.extend(
                sorted(p.name for p in integrations_dir.glob(ext)))
    if not (dockerfile_present and integration_files):
        return {
            "available": False,
            "dockerfile_present": dockerfile_present,
            "integrations_count": len(integration_files),
            "message": "Dockerfile + integrations not detected",
        }
    dockerfile_lines = 0
    try:
        dockerfile_lines = sum(
            1 for _ in dockerfile_path.read_text(
                encoding="utf-8").splitlines())
    except OSError:
        dockerfile_lines = 0
    return {
        "available": True,
        "dockerfile_present": True,
        "dockerfile_lines": dockerfile_lines,
        "integrations_count": len(integration_files),
        "integrations": integration_files,
        "image_name": "drishti-pam:latest",
        "build_command": "docker build -t drishti-pam:latest .",
        "run_command": "docker run --rm -p 8000:8000 drishti-pam:latest",
    }


def summarise_threat_model(threat_model: dict | None) -> dict:
    """Build the small KPI block shown on the dashboard's Threat tile.

    Returns a stable schema even when the source file is missing so
    the front-end can render a sensible empty state without branching.
    Keys mirror the names already used in
    ``threat_model.coverage_summary`` so the JS can fall back to either
    source. ``top_techniques`` is a small ranked list of the most
    frequently-mapped techniques so the tile can fit a compact bar
    list without re-counting in JS.
    """
    if not threat_model:
        return {
            "available": False,
            "total_scenarios": 0,
            "governance_only_count": 0,
            "unique_techniques_covered": 0,
            "unique_tactics_covered": 0,
            "kill_chain_phases_covered": 0,
            "stride_categories_present": [],
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "top_techniques": [],
            "version": None,
            "attck_version": None,
        }

    scenarios = threat_model.get("scenarios", {}) or {}

    # Recompute the top-techniques ranking here (rather than trusting
    # the static coverage_summary block) so we always emit a usable
    # ranked list even if the JSON only ships aggregate counts.
    tech_counts: dict[tuple[str, str], int] = {}
    for sdata in scenarios.values():
        for tech in sdata.get("attck", {}).get("techniques", []) or []:
            key = (tech.get("id", "?"), tech.get("name", "?"))
            tech_counts[key] = tech_counts.get(key, 0) + 1
    top = sorted(tech_counts.items(),
                 key=lambda kv: (-kv[1], kv[0][0]))[:6]
    top_techniques = [
        {"id": k[0], "name": k[1], "scenario_count": v}
        for k, v in top
    ]

    cov = threat_model.get("coverage_summary") or {}
    metadata = threat_model.get("metadata") or {}
    return {
        "available": True,
        "total_scenarios": cov.get("total_scenarios", len(scenarios)),
        "governance_only_count": cov.get("governance_only_count", 0),
        "governance_only_ids": cov.get("governance_only_ids", []),
        "unique_techniques_covered": cov.get(
            "unique_techniques_covered", len(tech_counts)),
        "unique_tactics_covered": cov.get("unique_tactics_covered", 0),
        "kill_chain_phases_covered": cov.get("kill_chain_phases_covered", 0),
        "stride_categories_present": cov.get(
            "stride_categories_present", []),
        "confidence_distribution": cov.get(
            "confidence_distribution", {}),
        "top_techniques": top_techniques,
        "version": metadata.get("version"),
        "attck_version": metadata.get("attck_version"),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Source directory containing sessions.json, "
                        "labels.json, rule_predictions.json, "
                        "llm_predictions.json. Defaults to ../data/.")
    p.add_argument("--vocab-dir", type=Path, default=DEFAULT_VOCAB_DIR,
                   help="Directory holding sensitive_keywords.json. "
                        "Defaults to ../generator/vocab/.")
    p.add_argument("--threat-model", type=Path,
                   default=DEFAULT_THREAT_MODEL_PATH,
                   help="Path to threat_model.json (STRIDE × ATT&CK × "
                        "kill-chain mapping). Defaults to "
                        "../threat_model.json. Optional — if absent, "
                        "the dashboard's threat-coverage tile renders "
                        "an empty state.")
    p.add_argument("--dockerfile", type=Path,
                   default=DEFAULT_DOCKERFILE_PATH,
                   help="Path to the project Dockerfile. Defaults to "
                        "../Dockerfile. Used purely to populate the "
                        "dashboard's Deployment KPI tile; absence "
                        "flips the tile to 'Local-only'.")
    p.add_argument("--integrations-dir", type=Path,
                   default=DEFAULT_INTEGRATIONS_DIR,
                   help="Directory containing the SOAR/SIEM integration "
                        "sketches (splunk_detection.spl, "
                        "sentinel_rule.kql, xsoar_playbook.json). "
                        "Defaults to ../integrations. Used for the "
                        "Deployment KPI tile only.")
    p.add_argument("--reference-comparison", type=Path,
                   default=DEFAULT_REFERENCE_COMPARISON_PATH,
                   help="Path to the frozen reference comparison.json "
                        "that powers the Drishti-PAM dashboard hero "
                        "KPIs (combined F1, detector agreement, analyst-"
                        "hour cost reduction). Optional: when absent "
                        "the hero falls back to placeholder labels.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output directory for sessions_summary.json + "
                        "summary.json. Defaults to ./data/ next to "
                        "this script.")
    args = p.parse_args(argv)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  data dir : {data_dir}")
    print(f"  vocab dir: {args.vocab_dir}")
    print(f"  out dir  : {out_dir}")

    # 1. Load source files. The four data files are required; the
    #    sensitive-keywords vocab is optional (an empty set just means
    #    s4 always passes).
    required = ["sessions.json", "labels.json",
                "rule_predictions.json", "llm_predictions.json"]
    for fn in required:
        if not (data_dir / fn).exists():
            print(f"ERROR: missing {data_dir / fn}", file=sys.stderr)
            print("       Run code/smoke.py first to populate code/data/.",
                  file=sys.stderr)
            return 2

    sessions = {s["session_id"]: s for s in
                json.loads((data_dir / "sessions.json").read_text(encoding="utf-8"))}
    labels = {l["session_id"]: l for l in
              json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))}
    rule_preds = {p["session_id"]: p for p in
                  json.loads((data_dir / "rule_predictions.json").read_text(encoding="utf-8"))}
    llm_preds = {p["session_id"]: p for p in
                 json.loads((data_dir / "llm_predictions.json").read_text(encoding="utf-8"))}

    # split.json is optional (older pipelines didn't emit it). When
    # missing, every session's split-badge defaults to "unknown" and
    # the data-source page's split filter still works (it just shows
    # 'unknown' as the only choice).
    split_path = data_dir / "split.json"
    if split_path.exists():
        split_doc = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = set(split_doc.get("train_ids") or [])
        test_ids = set(split_doc.get("test_ids") or [])
    else:
        train_ids = set()
        test_ids = set()

    sensitive_terms = load_sensitive_terms(args.vocab_dir)
    threat_model = load_threat_model(args.threat_model)
    threat_summary = summarise_threat_model(threat_model)
    print(f"  loaded {len(sessions):,} sessions, "
          f"{len(rule_preds):,} rule preds, "
          f"{len(llm_preds):,} LLM preds, "
          f"{len(sensitive_terms):,} sensitive terms, "
          f"split={len(train_ids):,}/{len(test_ids):,}")
    if threat_summary["available"]:
        print(f"  threat model: "
              f"{threat_summary['unique_techniques_covered']} techniques · "
              f"{threat_summary['unique_tactics_covered']} tactics · "
              f"{threat_summary['kill_chain_phases_covered']} kill-chain phases "
              f"(ATT&CK {threat_summary['attck_version']})")
    else:
        print("  threat model: not loaded — Threat Coverage tile will be empty")
    deployment_summary = _build_deployment_summary(
        args.dockerfile, args.integrations_dir)
    if deployment_summary["available"]:
        print(f"  deployment: Containerised — Dockerfile "
              f"({deployment_summary['dockerfile_lines']} lines) + "
              f"{deployment_summary['integrations_count']} SOAR/SIEM "
              f"artefact(s) detected")
    else:
        print(f"  deployment: Local-only — Dockerfile present="
              f"{deployment_summary['dockerfile_present']}, "
              f"integrations count="
              f"{deployment_summary['integrations_count']}")

    # Operational metrics print line is emitted later in main(), after
    # the per-detector FP counts have been computed against the joined
    # records. See the op_metrics_summary block.

    # 2. The dashboard renders the intersection of rule and LLM
    #    predictions. In a healthy pipeline this equals the test
    #    partition (since the LLM detector only scores test sessions);
    #    if a future detect.py change scores everything, the dashboard
    #    naturally widens with no code change here.
    paired_ids = sorted(set(rule_preds) & set(llm_preds) & set(sessions))
    n_paired = len(paired_ids)
    if n_paired == 0:
        print("ERROR: no sessions appear in BOTH rule and LLM "
              "prediction files. Did the LLM detector produce empty "
              "output?", file=sys.stderr)
        return 2

    # 3. Build per-session records.
    #
    # Two parallel structures share the same `session_id` keyspace:
    #   `summary_records`  — list of slim rows for the table + filters
    #                        (~10 fields each).
    #   `detail_records`   — dict keyed by session_id holding everything
    #                        the drawer needs (justification, command
    #                        stream, rule/llm evidence, compliance
    #                        signals, etc.). Lazy-loaded by the JS on
    #                        first drawer click.
    summary_records: list[dict] = []
    detail_records: dict[str, dict] = {}
    risk_counts: dict[str, int] = {
        "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MINIMAL": 0}
    compliance_counts: dict[str, int] = {
        "fully_compliant": 0, "partially_compliant": 0, "non_compliant": 0}
    scenario_counts: dict[str, int] = {}
    agreement_counts = {
        "both_flag": 0, "both_clean": 0,
        "rule_only_flag": 0, "llm_only_flag": 0,
    }

    for sid in paired_ids:
        sess = sessions[sid]
        lab = labels.get(sid, {})
        rp = rule_preds[sid]
        lp = llm_preds[sid]

        rule_call = rp.get("prediction", "clean")
        llm_call = lp.get("prediction", "clean")
        llm_score = float(lp.get("confidence", 0.0))
        rule_fires = rp.get("scenario_hits") or []
        distinct_rules = len(set(rule_fires))

        # Detector agreement bucket
        if rule_call == "suspicious" and llm_call == "suspicious":
            agreement = "both_flag"
        elif rule_call == "clean" and llm_call == "clean":
            agreement = "both_clean"
        elif rule_call == "suspicious" and llm_call == "clean":
            agreement = "rule_only_flag"
        else:
            agreement = "llm_only_flag"
        agreement_counts[agreement] += 1

        rating = risk_rating(rule_call, llm_call, llm_score, distinct_rules)
        risk_counts[rating] += 1

        compliance = compliance_check(sess, sensitive_terms)
        compliance_counts[compliance["label"]] += 1

        scenario = lab.get("scenario") or "clean"
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        ticket = sess.get("ticket") or {}
        inner = sess.get("session") or {}
        commands = inner.get("commands") or []
        vault_fetches = sess.get("vault_fetches") or []

        # Slim row for the table + filters. Every byte here ships
        # with the initial page load, so we only include what the
        # table renders or the filters key on.
        summary_records.append({
            "id": sid,
            "user": sess.get("user", "?"),
            "ticket_id": sess.get("ticket_id", "?"),
            "host": inner.get("host", "?"),
            "scenario_gt": scenario,
            "label_gt": lab.get("label", "clean"),
            "rule_call": rule_call,
            "llm_call": llm_call,
            "llm_score": round(llm_score, 4),
            "agreement": agreement,
            "risk_rating": rating,
            "compliance": compliance["label"],
            "compliance_pass_count": compliance["pass_count"],
        })

        # Full per-session record for the drawer. Lazy-loaded by JS
        # only on first drawer click — payload size doesn't affect
        # initial page load at all.
        detail_records[sid] = {
            "id": sid,
            "user": sess.get("user", "?"),
            "ticket_id": sess.get("ticket_id", "?"),
            "ticket_category": ticket.get("category", "?"),
            "ticket_approver": ticket.get("approver", "?"),
            "named_assets": ticket.get("named_assets") or [],
            "change_window_start": ticket.get("change_window_start", ""),
            "change_window_end": ticket.get("change_window_end", ""),
            "host": inner.get("host", "?"),
            "start_time": inner.get("start_time", ""),
            "scenario_gt": scenario,
            "label_gt": lab.get("label", "clean"),
            "rule_call": rule_call,
            "rule_fires": list(set(rule_fires)),
            "rule_evidence": str(rp.get("evidence") or ""),
            "rule_confidence": round(float(rp.get("confidence", 0.0)), 4),
            "llm_call": llm_call,
            "llm_score": round(llm_score, 4),
            "llm_evidence": str(lp.get("evidence") or ""),
            "agreement": agreement,
            "risk_rating": rating,
            "compliance": compliance["label"],
            "compliance_pass_count": compliance["pass_count"],
            "compliance_signals": compliance["signals"],
            "justification": ticket.get("business_justification", ""),
            "commands": [
                {
                    "ts": c.get("timestamp", ""),
                    "surface": c.get("surface", ""),
                    "command": c.get("command", ""),
                }
                # Drawer caps the displayed list at 50; everything beyond
                # is summarised as "+N more". Synthetic sessions rarely
                # exceed 30 so 50 is a safe cap.
                for c in commands[:50]
            ],
            "command_count_total": len(commands),
            "vault_fetches": [
                {
                    "ts": v.get("timestamp", ""),
                    "user": v.get("user", "?"),
                }
                for v in vault_fetches
            ],
        }

    # 3b. Raw records — every generated session, not just the paired
    #     subset. Powers the second view (the "Data Source" tab).
    #     Lean fields only; no model output, no full command stream.
    #     The split-badge resolution logic:
    #         in train_ids -> "train"
    #         in test_ids  -> "test"
    #         else         -> "unknown"  (only happens if split.json
    #                                     was missing or out of sync
    #                                     with sessions.json)
    raw_records: list[dict] = []
    split_counts: dict[str, int] = {"train": 0, "test": 0, "unknown": 0}
    for sid in sorted(sessions.keys()):
        sess = sessions[sid]
        ticket = sess.get("ticket") or {}
        inner = sess.get("session") or {}
        commands = inner.get("commands") or []
        vault_fetches = sess.get("vault_fetches") or []
        named_assets = ticket.get("named_assets") or []
        lab = labels.get(sid, {})

        if sid in train_ids:
            split = "train"
        elif sid in test_ids:
            split = "test"
        else:
            split = "unknown"
        split_counts[split] += 1

        raw_records.append({
            "id": sid,
            "split": split,
            "label_gt": lab.get("label", "clean"),
            "scenario_gt": lab.get("scenario") or "clean",
            "user": sess.get("user", "?"),
            "host": inner.get("host", "?"),
            "ticket_id": sess.get("ticket_id", "?"),
            "ticket_category": ticket.get("category", "?"),
            "ticket_approver": ticket.get("approver", "?"),
            "start_time": inner.get("start_time", ""),
            "change_window_start": ticket.get("change_window_start", ""),
            "change_window_end": ticket.get("change_window_end", ""),
            "command_count": len(commands),
            "named_assets_count": len(named_assets),
            "vault_fetches_count": len(vault_fetches),
            "justification": ticket.get("business_justification", ""),
        })

    # 4. Aggregate summary for the KPI strip.
    n_risky = risk_counts["CRITICAL"] + risk_counts["HIGH"] + risk_counts["MEDIUM"]
    n_noncompliant = compliance_counts["non_compliant"]

    # Combined-detector F1 — recompute from this dataset so the KPI
    # always matches what's on screen, even if the user has rerun
    # smoke.py with a different session count.
    tp = sum(1 for r in summary_records
             if r["risk_rating"] != "MINIMAL"
             and r["label_gt"] == "suspicious")
    fp = sum(1 for r in summary_records
             if r["risk_rating"] != "MINIMAL"
             and r["label_gt"] == "clean")
    fn = sum(1 for r in summary_records
             if r["risk_rating"] == "MINIMAL"
             and r["label_gt"] == "suspicious")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    # Operational FP cost — picks the *cheaper* detector under the
    # 18-min/triage high baseline (Tier-2 analyst-shift mechanistic
    # estimate). The full operational_metrics block lives in
    # data/comparison.json; this dashboard summary surfaces only the
    # headline number the KPI tile needs.
    rule_fp = sum(1 for r in summary_records
                  if r["rule_call"] == "suspicious"
                  and r["label_gt"] == "clean")
    llm_fp = sum(1 for r in summary_records
                 if r["llm_call"] == "suspicious"
                 and r["label_gt"] == "clean")
    fp_minutes = 18.0
    op_metrics_summary: dict
    if rule_fp == 0 and llm_fp == 0:
        op_metrics_summary = {
            "available": False,
            "_note": ("Both detectors recorded zero false positives in "
                      "this dataset — operational FP cost is undefined."),
        }
    else:
        if (llm_fp <= rule_fp) and llm_fp > 0:
            best_name, best_fp = "llm-trained", llm_fp
        elif rule_fp > 0:
            best_name, best_fp = "rule", rule_fp
        else:
            best_name, best_fp = "llm-trained", llm_fp
        op_metrics_summary = {
            "available": True,
            "best_detector": best_name,
            "best_fp": best_fp,
            "best_hours": round(best_fp * fp_minutes / 60.0, 2),
            "fp_minutes": int(fp_minutes),
            "rule_fp": rule_fp,
            "llm_fp": llm_fp,
        }

    if op_metrics_summary["available"]:
        print(f"  operational FP cost: {op_metrics_summary['best_detector']} "
              f"\u2014 {op_metrics_summary['best_fp']} FP \u00d7 "
              f"{op_metrics_summary['fp_minutes']} min = "
              f"{op_metrics_summary['best_hours']} hr "
              f"(rule_fp={op_metrics_summary['rule_fp']}, "
              f"llm_fp={op_metrics_summary['llm_fp']})")
    else:
        print("  operational FP cost: undefined (zero FPs in this dataset)")

    top_scenario = max(
        ((s, n) for s, n in scenario_counts.items() if s != "clean"),
        key=lambda x: x[1], default=(None, 0))

    hero_summary = _build_hero_summary(args.reference_comparison)

    summary = {
        "_doc": "Aggregate KPIs rendered into the dashboard's header strip. "
                "Recomputed every time build_dashboard.py runs so the "
                "screen always matches the current data.",
        "n_sessions": n_paired,
        "n_sessions_total": len(sessions),
        "n_risky": n_risky,
        "n_risky_pct": round(100 * n_risky / n_paired, 1),
        "n_noncompliant": n_noncompliant,
        "n_noncompliant_pct": round(100 * n_noncompliant / n_paired, 1),
        "top_scenario": top_scenario[0],
        "top_scenario_count": top_scenario[1],
        "combined_f1": round(f1, 4),
        "combined_precision": round(precision, 4),
        "combined_recall": round(recall, 4),
        "risk_counts": risk_counts,
        "compliance_counts": compliance_counts,
        "agreement_counts": agreement_counts,
        "scenario_counts": scenario_counts,
        "split_counts": split_counts,
        "threat_coverage": threat_summary,
        "deployment": deployment_summary,
        "operational_metrics": op_metrics_summary,
        "hero": hero_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }

    # 5. Write outputs.
    #
    # The aggregate summary is small (~1 KB) and pretty-printed.
    # The lean session table is compact and ships with the initial
    # page load. The drawer details file is compact too but lazy-
    # loaded only on first drawer click, so its size is decoupled
    # from page-paint time.
    summary_path = out_dir / "summary.json"
    sessions_path = out_dir / "sessions_summary.json"
    details_path = out_dir / "sessions_details.json"
    raw_path = out_dir / "sessions_raw.json"
    threat_path = out_dir / "threat_model.json"

    summary_path.write_text(json.dumps(summary, indent=2),
                            encoding="utf-8")
    sessions_path.write_text(json.dumps(summary_records),
                             encoding="utf-8")
    details_path.write_text(json.dumps(detail_records),
                            encoding="utf-8")
    raw_path.write_text(json.dumps(raw_records),
                        encoding="utf-8")
    # The threat model is small (~3 KB) and rarely changes — pretty-
    # print it so a curious viewer can `cat` it without piping through
    # `jq`. Always emit a stub when the source is missing so the
    # front-end always sees a stable schema.
    threat_payload = {
        "_doc": "Scenario -> STRIDE / MITRE ATT&CK / Cyber Kill Chain "
                "mapping. Source of truth: code/threat_model.json. "
                "Coverage statistics summarised in summary.threat_coverage.",
        "summary": threat_summary,
        "scenarios": (threat_model or {}).get("scenarios", {}),
        "metadata": (threat_model or {}).get("metadata", {}),
    }
    threat_path.write_text(json.dumps(threat_payload, indent=2),
                           encoding="utf-8")

    sz_summary = summary_path.stat().st_size
    sz_sessions = sessions_path.stat().st_size
    sz_details = details_path.stat().st_size
    sz_raw = raw_path.stat().st_size
    sz_threat = threat_path.stat().st_size
    print()
    print(f"  wrote {summary_path.relative_to(HERE)}  "
          f"({sz_summary:,} bytes)")
    print(f"  wrote {sessions_path.relative_to(HERE)}  "
          f"({sz_sessions:,} bytes, ships on initial load)")
    print(f"  wrote {details_path.relative_to(HERE)}  "
          f"({sz_details:,} bytes, lazy-loaded on first drawer click)")
    print(f"  wrote {raw_path.relative_to(HERE)}  "
          f"({sz_raw:,} bytes, lazy-loaded on first 'Data Source' tab)")
    print(f"  wrote {threat_path.relative_to(HERE)}  "
          f"({sz_threat:,} bytes, ships on initial load)")
    print()
    print(f"  {n_paired:,} sessions ready for the triage console")
    print(f"  {len(sessions):,} sessions ready for the data-source view")
    print(f"    split:       "
          + ", ".join(f"{k}={v}" for k, v in split_counts.items() if v))
    print(f"    risk:        "
          + ", ".join(f"{k}={v}" for k, v in risk_counts.items() if v))
    print(f"    compliance:  "
          + ", ".join(f"{k.split('_')[0]}={v}"
                      for k, v in compliance_counts.items()))
    print(f"    combined F1: {f1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
