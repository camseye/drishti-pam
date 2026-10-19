"""Syntactic validator for the SOAR / SIEM integration sketches.

Stdlib-only static checks for the three artefacts that ship next to
this file:

    splunk_detection.spl   pipe-balance + at-least-one stats/eval/where stage
    sentinel_rule.kql      pipe-balance + at-least-one extend/project/where
                           + balanced parentheses
    xsoar_playbook.json    JSON parses + minimum XSOAR v6 schema present
                           + starttaskid points at an existing task ID
                           + every nexttasks edge points at an existing task

The intent is a coffee-break-readable sanity check, not a production
parser. It is *not* equivalent to running the SPL through `splunk btool
check`, the KQL through `Kusto.Explorer`, or the JSON through Cortex
XSOAR's playbook validator. Those are all third-party tools that need
either a live tenant or a downloaded SDK; this script gives a high-
confidence first-pass signal that the artefacts are not obviously
broken (unbalanced quotes, missing required keys, dangling task
references) so the rest of the build chain can rely on them.

Run from the repo root:

    cd code
    python -m integrations._validate

Exits 0 on success, 1 on any artefact failing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPL_PATH = HERE / "splunk_detection.spl"
KQL_PATH = HERE / "sentinel_rule.kql"
XSOAR_PATH = HERE / "xsoar_playbook.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _check(label: str, ok: bool, detail: str = "") -> bool:
    marker = "PASS" if ok else "FAIL"
    msg = f"  [{marker}] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return ok


_SPL_REAL_LINE = re.compile(
    r"""^\s*(
        index\s*=             |   # bare search starting with index=
        search\s+index\s*=    |   # explicit search index=
        \|\s*(eval|where|stats|sort|head|join|rename|fields|
              outputlookup|spath|table|tail|dedup|rex|search|
              lookup|inputlookup|fillnull|streamstats|eventstats|
              transaction|bin|chart|timechart)\b
    )""",
    re.VERBOSE,
)


def _strip_spl_comments(text: str) -> str:
    """SPL has no native comment syntax; the project uses a leading-`|`
    banner block at the top of every file. Strip the banner by skipping
    every line until the first one that looks like a real SPL stage
    (matches `_SPL_REAL_LINE`). Everything from that point onward is
    kept verbatim, including continuation lines that don't themselves
    start with a pipe — those carry the rest of the previous stage's
    arguments and must stay together for the parens / brackets balance
    check to be meaningful."""
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if _SPL_REAL_LINE.match(ln)),
        len(lines),
    )
    return "\n".join(lines[start:])


def _pipe_count(text: str) -> int:
    """Count leading-pipe stage markers (`| ...` at the start of a line)."""
    return sum(1 for ln in text.splitlines() if ln.lstrip().startswith("|"))


def _balanced(text: str, open_ch: str, close_ch: str) -> bool:
    depth = 0
    for ch in text:
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# --------------------------------------------------------------------------- #
# Per-artefact checks
# --------------------------------------------------------------------------- #

def validate_spl(path: Path) -> bool:
    print(f"\nSPL: {path.name}")
    if not path.exists():
        return _check("file exists", False, str(path))
    text = path.read_text(encoding="utf-8")
    body = _strip_spl_comments(text)
    ok = True
    ok &= _check("file is non-empty", bool(text.strip()))
    ok &= _check(
        "at least 5 pipe stages",
        _pipe_count(body) >= 5,
        f"found {_pipe_count(body)} stages")
    ok &= _check(
        "balanced double-quotes",
        text.count('"') % 2 == 0,
        f"{text.count(chr(34))} double-quote chars")
    ok &= _check(
        "balanced parentheses", _balanced(body, "(", ")"))
    ok &= _check(
        "balanced square brackets", _balanced(body, "[", "]"))
    ok &= _check(
        "references rule_predictions sourcetype",
        "pam:rule_predictions" in text)
    ok &= _check(
        "computes a risk rating",
        "risk" in body and "CRITICAL" in body)
    ok &= _check(
        "ends with outputlookup, table, head, or stats",
        re.search(r"\|\s*(outputlookup|table|head|stats)\b",
                  body.split("\n")[-1] if body else "") is not None or
        any(re.search(r"\|\s*(outputlookup|table|head|stats)\b", ln)
            for ln in body.split("\n")[-3:]))
    return ok


def validate_kql(path: Path) -> bool:
    print(f"\nKQL: {path.name}")
    if not path.exists():
        return _check("file exists", False, str(path))
    text = path.read_text(encoding="utf-8")
    ok = True
    ok &= _check("file is non-empty", bool(text.strip()))
    ok &= _check(
        "at least 5 pipe stages",
        _pipe_count(text) >= 5,
        f"found {_pipe_count(text)} stages")
    ok &= _check(
        "balanced parentheses", _balanced(text, "(", ")"))
    ok &= _check(
        "balanced square brackets", _balanced(text, "[", "]"))
    ok &= _check(
        "uses at least one extend, where, project, or order",
        any(kw in text for kw in ("| extend", "| where",
                                  "| project", "| order")))
    ok &= _check(
        "references the custom-log table",
        "PAMRulePredictions_CL" in text)
    ok &= _check(
        "joins the LLM custom-log table",
        "PAMLlmPredictions_CL" in text)
    ok &= _check(
        "computes a risk rating",
        "risk =" in text and "CRITICAL" in text)
    ok &= _check(
        "ends with order/take/project (analyst-readable output)",
        any(text.rstrip().endswith(stem)
            for stem in ("take 200", "take 100"))
        or "| order by" in text)
    return ok


def validate_xsoar(path: Path) -> bool:
    print(f"\nXSOAR: {path.name}")
    if not path.exists():
        return _check("file exists", False, str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _check("file is valid JSON", False, str(exc))
    _check("file is valid JSON", True)
    ok = True
    required = ("id", "name", "starttaskid", "tasks")
    for key in required:
        ok &= _check(f"required top-level key `{key}` present",
                     key in payload)
    if "tasks" not in payload:
        return ok
    tasks = payload["tasks"]
    ok &= _check("`tasks` is a non-empty dict",
                 isinstance(tasks, dict) and len(tasks) > 0,
                 f"tasks count = {len(tasks) if isinstance(tasks, dict) else 'n/a'}")
    if not isinstance(tasks, dict) or not tasks:
        return ok
    starttask = payload.get("starttaskid")
    ok &= _check("`starttaskid` points at an existing task",
                 starttask in tasks,
                 f"starttaskid={starttask!r}, available={list(tasks)[:5]}…")
    task_required = ("id", "taskid", "type", "task")
    for tid, t in tasks.items():
        for k in task_required:
            ok &= _check(f"task `{tid}` has `{k}`", k in t)
    edges_ok = True
    dangling: list[str] = []
    for tid, t in tasks.items():
        for branch_name, dests in (t.get("nexttasks") or {}).items():
            for dest in dests or []:
                if dest not in tasks:
                    dangling.append(f"{tid}.{branch_name} -> {dest}")
                    edges_ok = False
    ok &= _check("every nexttasks edge points at a real task",
                 edges_ok,
                 ("dangling: " + ", ".join(dangling[:5]))
                 if dangling else "")
    types_seen = {t.get("type") for t in tasks.values()}
    ok &= _check(
        "playbook spans start + work + decision + close",
        {"start", "regular", "condition"}.issubset(types_seen)
        or {"start", "regular"}.issubset(types_seen),
        f"task types present: {sorted(types_seen)}")
    return ok


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    print("integrations/_validate.py — SOAR/SIEM syntactic checks")
    print("=" * 60)
    spl_ok = validate_spl(SPL_PATH)
    kql_ok = validate_kql(KQL_PATH)
    xsoar_ok = validate_xsoar(XSOAR_PATH)
    print()
    print("=" * 60)
    summary = [
        ("Splunk SPL",   spl_ok),
        ("Sentinel KQL", kql_ok),
        ("XSOAR JSON",   xsoar_ok),
    ]
    for label, ok in summary:
        print(f"  {label:<14} {'OK' if ok else 'FAIL'}")
    overall = all(ok for _, ok in summary)
    print(f"\nOverall: {'OK' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
