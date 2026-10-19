"""Rule-based PAM session detector for Drishti-PAM (Stage C).

Twelve plain Python rules covering twelve of the thirteen ground-truth
scenarios (S1, S2, S3, S4, S5, S7, S10, S11, S12, S13, S14, S15). Reads
`data/sessions.json`, scores each session, and writes
`data/rule_predictions.json` in a flat shape so downstream comparison code
can stay simple:

    {
      "session_id":     "sess-000123",
      "prediction":     "clean" | "suspicious",
      "scenario_hits":  ["S2", "S7"],   # may be empty
      "confidence":     0.85,           # 0.0 - 1.0
      "evidence":       "vague justification (4 chars); destructive SQL"
    }

S16 (vocabulary-cloak adversarial) is INTENTIONALLY NOT COVERED by any
rule. By construction, S16 sessions paraphrase destructive commands to
avoid the literal trigger phrases that S7 (and every other rule) matches
on. This is the headline finding of the project's adversarial-scenario
analysis: rule-based detection cannot scale to keyword-cloak attacks
because its detection surface IS the literal token vocabulary. Adding a
no-op `_rule_s16` would be misleading; the absence of a rule is the result.

Implementation notes:
  * Two scenarios (S1, S11) require cross-session aggregation by
    ticket_id, so a one-pass per-session loop is preceded by a build-the-
    ticket-index pass.
  * S7 reads command TEXT for destructive tokens (sessions carry no
    is_destructive flag).
  * S2 matches a list of known vague phrases as well as length, so
    the paraphraser's optional opener/closer doesn't blind the rule to a
    short core ("Per runbook, as agreed Refer to the linked runbook." still
    contains "as agreed").

Standard library only. No regex packages required (one stdlib `re` for the
vague-phrase patterns), no ML, no HTTP.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants — kept on-screen so a beginner can audit thresholds quickly.
# --------------------------------------------------------------------------- #

BUSINESS_HOURS = (time(8, 0), time(18, 0))

# S1 — fire on tickets with at least this many DISTINCT users. v2 datasets
# include legitimate pair-work clean tickets with two users on one ticket,
# so threshold is 3 (deliberately misses the S1-small bucket; that intentional
# false-negative is part of the story the trained-LLM detector will tell).
S1_USER_THRESHOLD = 3

# S2 — vague phrases scanned in the justification body. Length fallback at
# the end catches anything else that's just short.
S2_VAGUE_PHRASES = (
    "as agreed", "see ticket", "as discussed", "per slack", "per email",
    "per chat", "per usual", "see notes", "tbd", "asap", "n/a",
    "fix it",
)
S2_VAGUE_LENGTH = 30   # characters

# S3 — repeat-fetch threshold and minimum window length to avoid firing on
# legitimately long change windows that have a few extra fetches.
S3_FETCH_THRESHOLD = 5

# S4 — fraction of session commands that must overlap the ticket-category's
# expected pool. Anything below this fires.
S4_OVERLAP_THRESHOLD = 0.30

# S7 — destructive tokens in command text. Each entry must be a substring
# that appears in destructive commands but NOT in legitimate DDL or readonly
# operations. We deliberately spell out the object kind ("drop table", not
# "drop ") to avoid firing on legitimate `DROP INDEX IF EXISTS` maintenance,
# and exclude bare "update " because shell `yum update` / `dnf update` are
# legitimate OS-patching commands. SQL UPDATE is detected separately via
# the "update ... set" two-token check inside _has_destructive_text().
DESTRUCTIVE_TOKENS = (
    "delete from",
    "drop table",
    "drop database",
    "drop schema",
    "truncate table",
    "into outfile",
    "pg_dump",
    "mysqldump",
)
DESTRUCTIVE_OK_CATEGORIES = {"incident_remediation"}

# S11 — zombie ticket: same ticket reused after the change window closed.
S11_MIN_SESSIONS = 3

# S12 / S13 / S14 — keyword scans. Lists are deliberately on-screen.
S12_SENSITIVE_TOKENS = (
    "ssn", "social_security", "credit_card", "cc_number", "card_number",
    "cvv", "passport", "drivers_license", "salary", "compensation",
    "bonus", "dob", "date_of_birth",
    "/etc/shadow", "/etc/passwd", "id_rsa", ".ssh/id_rsa", "id_ed25519",
    "vault.json", "secrets.txt", ".aws/credentials", ".kube/config",
    "wallet.dat",
)

S13_LATERAL_TOKENS = (
    # Discovery — keep only scanners that are out of scope for a normal
    # network_config ticket. ping/traceroute/arp are used by legitimate
    # network admins and would generate too many false positives.
    "nmap", "masscan",
    "showmount -e", "rpcinfo -p",
    "smbclient -l", "enum4linux",
    # Lateral execution
    "psexec.exe", "wmiexec",
    "winrm", "powershell -computername",
    "invoke-command -computername",
    "rsync -avz", "scp /tmp",
)

S14_ESCALATION_TOKENS = (
    "sudo su -", "sudo -i", "sudo -s", "su -", "su root",
    "setuid", "chmod u+s", "chmod 4755", "chmod 4777",
    "usermod -ag sudo", "usermod -ag wheel", "usermod -ag admin",
    "gpasswd -a", "visudo", "/etc/sudoers.d/",
    "set-localgroupmember", "net localgroup administrators",
    "add-localgroupmember -group administrators",
)


# --------------------------------------------------------------------------- #
# S4 vocabulary loader — same files the generator uses
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
VOCAB_DIR = HERE.parent / "generator" / "vocab"


def _load_vocab() -> dict:
    out: dict = {}
    for p in sorted(VOCAB_DIR.glob("*.json")):
        with p.open(encoding="utf-8") as fp:
            data = json.load(fp)
        data.pop("_doc", None)
        out[p.stem] = data
    return out


VOCAB = _load_vocab()


def _category_command_pool(category: str) -> set[str]:
    """All commands that the *expected* surface and ticket category would
    legitimately produce. Used by the divergence check (S4)."""

    bag: set[str] = set()
    if category == "database_maintenance":
        for kind in ("readonly", "ddl"):
            for entry in VOCAB["sql_commands"].get(kind, []):
                # Pool entries are either bare strings (v2 normalisation) or
                # legacy [stmt, flag] pairs (v1 vocab files left in place).
                stmt = entry[0] if isinstance(entry, list) else entry
                bag.add(stmt)
    elif category == "windows_admin":
        for entry in VOCAB["rdp_blocks"].get("legitimate", []):
            _, app, keys = entry
            bag.add(f"{app}: {keys}")
    elif category == "cloud_resource_change":
        bag.update(VOCAB["cloud_commands"].get("legitimate", []))
    else:
        bag.update(VOCAB["ssh_commands"].get(category, []))
    return bag


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _command_text(commands: list[dict]) -> str:
    """Lowercased concatenation of command + window_title (RDP only). Used by
    every keyword-based rule (S7, S12, S13, S14)."""
    parts: list[str] = []
    for c in commands:
        parts.append(c.get("command", ""))
        if "window_title" in c:
            parts.append(c["window_title"])
    return " ".join(parts).lower()


def _command_strip_wrappers(c: str) -> str:
    """Strip mock-augmenter wrappers so command-pool overlap (S4) still
    matches when the augmenter has prefixed an echo or appended a comment.

    Examples:
      "echo 'starting' && sudo X"   -> "sudo X"
      "sudo X  # step 2"            -> "sudo X"
      "sudo -E X"                   -> "sudo X"        (preserve-env wrapper)
    """
    out = c
    # Strip leading echo wrapper "echo '...' && CMD"
    m = re.match(r"^echo '[^']*' && (.+)$", out)
    if m:
        out = m.group(1)
    # Strip trailing inline comment "  # ..."
    out = re.split(r"\s+#\s", out, maxsplit=1)[0]
    # Strip preserve-env sudo wrapper "sudo -E " -> "sudo "
    out = out.replace("sudo -E ", "sudo ", 1)
    return out.strip()


def _ticket_aggregate(sessions: list[dict]) -> dict[str, dict]:
    """Group sessions by ticket_id and pre-compute aggregate features needed
    by S1 (multi-user) and S11 (zombie ticket)."""

    by_ticket: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        by_ticket[s["ticket_id"]].append(s)

    agg: dict[str, dict] = {}
    for tid, group in by_ticket.items():
        users = {s["user"] for s in group}
        we = _parse(group[0]["ticket"]["change_window_end"])
        n_late = sum(1 for s in group
                     if _parse(s["session"]["start_time"]) > we)
        agg[tid] = {
            "n_sessions": len(group),
            "n_users": len(users),
            "n_late": n_late,
            "users": users,
        }
    return agg


# --------------------------------------------------------------------------- #
# The twelve rules
# --------------------------------------------------------------------------- #

def _rule_s1_multi_user(session: dict, agg: dict) -> tuple[bool, str]:
    info = agg[session["ticket_id"]]
    if info["n_users"] >= S1_USER_THRESHOLD:
        return True, (f"{info['n_users']} distinct users on ticket "
                      f"{session['ticket_id']}")
    return False, ""


def _rule_s2_vague_justification(session: dict, agg: dict) -> tuple[bool, str]:
    text = (session["ticket"]["business_justification"] or "").strip().lower()
    for phrase in S2_VAGUE_PHRASES:
        if phrase in text and len(text) < 100:
            return True, f"vague phrase {phrase!r} in justification (len={len(text)})"
    if len(text) < S2_VAGUE_LENGTH:
        return True, f"justification too short ({len(text)} chars)"
    return False, ""


def _rule_s3_repeated_fetch(session: dict, agg: dict) -> tuple[bool, str]:
    fetches = session.get("vault_fetches") or []
    if len(fetches) >= S3_FETCH_THRESHOLD:
        return True, f"{len(fetches)} vault fetches on ticket {session['ticket_id']}"
    return False, ""


def _rule_s4_commands_diverge(session: dict, agg: dict) -> tuple[bool, str]:
    cat = session["ticket"]["category"]
    pool = _category_command_pool(cat)
    if not pool:
        return False, ""
    cmds = [c.get("command", "") for c in session["session"]["commands"]]
    if not cmds:
        return False, ""
    # Strip augmenter wrappers before pool lookup.
    overlap = sum(1 for c in cmds if _command_strip_wrappers(c) in pool)
    overlap_ratio = overlap / len(cmds)
    if overlap_ratio < S4_OVERLAP_THRESHOLD:
        return True, (f"{overlap}/{len(cmds)} commands match the "
                      f"'{cat}' pool (overlap {overlap_ratio:.0%})")
    return False, ""


def _rule_s5_off_hours(session: dict, agg: dict) -> tuple[bool, str]:
    """Off-hours: only fires when the session is BOTH outside business hours
    AND outside its own ticket's change window. A legitimate evening change
    that happens to start at 18:15 (still inside the approved window) does
    not trip this rule."""
    sess_start = _parse(session["session"]["start_time"])
    win_start = _parse(session["ticket"]["change_window_start"])
    win_end = _parse(session["ticket"]["change_window_end"])
    inside_window = win_start <= sess_start <= win_end
    inside_business = BUSINESS_HOURS[0] <= sess_start.time() <= BUSINESS_HOURS[1]
    if inside_window or inside_business:
        return False, ""
    return True, (f"session start {sess_start.isoformat(timespec='minutes')} "
                  f"is outside both ticket window and business hours")


def _has_destructive_text(text: str) -> str | None:
    """Return the matched destructive token, or None. SQL UPDATE is detected
    via the two-token combination 'update ' + ' set ' in the same blob; this
    avoids matching shell commands like `yum update` or `apt update`."""
    for tok in DESTRUCTIVE_TOKENS:
        if tok in text:
            return tok
    if "update " in text and " set " in text:
        return "update ... set"
    return None


def _rule_s7_scope_creep(session: dict, agg: dict) -> tuple[bool, str]:
    cat = session["ticket"]["category"]
    if cat in DESTRUCTIVE_OK_CATEGORIES:
        return False, ""
    text = _command_text(session["session"]["commands"])
    hit = _has_destructive_text(text)
    if hit is None:
        return False, ""
    # database_maintenance always fires when destructive ops appear: legitimate
    # DDL only uses CREATE / DROP INDEX / VACUUM / REINDEX, never DELETE FROM
    # or DROP TABLE. A migration ticket would be a different category.
    return True, f"destructive token {hit!r} on '{cat}' ticket"


def _rule_s10_asset_pivot(session: dict, agg: dict) -> tuple[bool, str]:
    named = set(session["ticket"].get("named_assets") or [])
    host = session["session"].get("host")
    if named and host and host not in named:
        return True, (f"session host {host!r} not in ticket named_assets "
                      f"{sorted(named)}")
    return False, ""


def _rule_s11_zombie(session: dict, agg: dict) -> tuple[bool, str]:
    info = agg[session["ticket_id"]]
    if info["n_sessions"] >= S11_MIN_SESSIONS and info["n_late"] >= 1:
        return True, (f"ticket {session['ticket_id']} has "
                      f"{info['n_sessions']} sessions, {info['n_late']} past "
                      "the change window")
    return False, ""


def _rule_s12_sensitive_data(session: dict, agg: dict) -> tuple[bool, str]:
    text = _command_text(session["session"]["commands"])
    hit = next((tok for tok in S12_SENSITIVE_TOKENS if tok in text), None)
    if hit:
        return True, f"sensitive token {hit!r} in command text"
    return False, ""


def _rule_s13_lateral_movement(session: dict, agg: dict) -> tuple[bool, str]:
    text = _command_text(session["session"]["commands"])
    hit = next((tok for tok in S13_LATERAL_TOKENS if tok in text), None)
    if hit:
        return True, f"lateral-movement token {hit!r} in command text"
    return False, ""


def _rule_s14_privilege_escalation(session: dict, agg: dict) -> tuple[bool, str]:
    text = _command_text(session["session"]["commands"])
    hit = next((tok for tok in S14_ESCALATION_TOKENS if tok in text), None)
    if hit:
        return True, f"escalation token {hit!r} in command text"
    return False, ""


def _rule_s15_approver_implementer(session: dict, agg: dict) -> tuple[bool, str]:
    user = session["user"]
    approver = session["ticket"].get("approver")
    if approver and user == approver:
        return True, f"requester {user} == approver {approver}"
    return False, ""


_RULES = (
    ("S1",  _rule_s1_multi_user),
    ("S2",  _rule_s2_vague_justification),
    ("S3",  _rule_s3_repeated_fetch),
    ("S4",  _rule_s4_commands_diverge),
    ("S5",  _rule_s5_off_hours),
    ("S7",  _rule_s7_scope_creep),
    ("S10", _rule_s10_asset_pivot),
    ("S11", _rule_s11_zombie),
    ("S12", _rule_s12_sensitive_data),
    ("S13", _rule_s13_lateral_movement),
    ("S14", _rule_s14_privilege_escalation),
    ("S15", _rule_s15_approver_implementer),
)


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #

def detect_one(session: dict, agg: dict) -> dict:
    hits: list[str] = []
    evidence: list[str] = []
    for sc, fn in _RULES:
        fired, why = fn(session, agg)
        if fired:
            hits.append(sc)
            evidence.append(f"[{sc}] {why}")
    if not hits:
        return {
            "session_id": session["session_id"],
            "prediction": "clean",
            "scenario_hits": [],
            "confidence": 0.95,
            "evidence": "no rule fired",
        }
    confidence = min(0.99, 0.70 + 0.05 * len(hits))
    return {
        "session_id": session["session_id"],
        "prediction": "suspicious",
        "scenario_hits": hits,
        "confidence": round(confidence, 3),
        "evidence": " ; ".join(evidence),
    }


def detect_all(sessions: list[dict]) -> list[dict]:
    agg = _ticket_aggregate(sessions)
    return [detect_one(s, agg) for s in sessions]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=Path, default=Path("data/sessions.json"))
    p.add_argument("--out", type=Path, default=Path("data/rule_predictions.json"))
    args = p.parse_args(argv)

    sessions = json.load(args.sessions.open(encoding="utf-8"))
    predictions = detect_all(sessions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=1)

    n_susp = sum(1 for p in predictions if p["prediction"] == "suspicious")
    n_clean = len(predictions) - n_susp
    print(f"rule detector: scored {len(predictions)} sessions "
          f"-> {n_susp} suspicious, {n_clean} clean")
    print(f"  output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
