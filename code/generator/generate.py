"""Synthetic PAM session generator (Drishti-PAM, Stage A).

Key design points:
  * Thirteen abuse scenarios (S1..S15 plus S16 — see SCENARIO_MIX below).
  * Shared ticket ids: ~7 500 unique tickets carry the 10 000 sessions, so
    S1 (multi-user at scale) and S11 (zombie ticket) can be expressed as
    "multiple session rows that share a ticket_id".
  * S1 is a count gradient: small (2-3 extra users) / medium (5-15) /
    large (50-100), so the *count* itself becomes a signal.
  * Tickets carry an `approver` field (drives S15: approver-as-implementer).
  * Sessions carry no `is_destructive` flag; destructiveness has to be
    inferred from the command text (drives S7 against textual signals).
  * S16 (vocabulary-cloak adversarial scenario) emits
    sessions whose destructive intent is achieved via paraphrased
    SQL (CALL purge_*, ALTER TABLE ... RENAME TO _orphaned_*, EXEC
    sp_purgehistory) that AVOIDS the literal trigger tokens the rule
    detector matches on, paired with justifications engineered to share
    vocabulary with the (paraphrased) commands so the s5 compliance
    signal also passes. Rule baseline cannot detect S16 by construction;
    the LLM detector is expected to detect S16 only when retrained with
    S16 examples in its training set.

Outputs into the directory passed via --out-dir (default ./data):

    sessions.json   list of session dicts
    labels.json     ground-truth label per session
    manifest.json   seed, counts, self-check results

The generator is deterministic: same --seed produces a byte-identical output.
No machine learning, no LLM calls. Vocab is loaded from JSON files in
generator/vocab/. Beyond Faker (for IPs / timestamps / ticket IDs) the script
uses only the Python standard library.

Stage B (`llm_augment.py`) adds an LLM-augmentation layer (justification
paraphrase + command-pool expansion), on by default; pass --no-augment to
produce the deterministic Stage A baseline only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from faker import Faker

# Sibling-module import works whether generate.py is run directly
# (python generator/generate.py) or via the smoke harness.
sys.path.insert(0, str(Path(__file__).parent))
import llm_augment  # noqa: E402

VOCAB_DIR = Path(__file__).parent / "vocab"

CLEAN_LABEL = "clean"
SUSPICIOUS_LABEL = "suspicious"

SCENARIO_IDS = ("S1", "S2", "S3", "S4", "S5", "S7", "S10",
                "S11", "S12", "S13", "S14", "S15", "S16")

# How the 30 % suspicious slice is split across the thirteen scenarios.
# These proportions loosely match a real-world insider-risk caseload: vague
# justifications and approver-as-implementer events are rare; multi-user
# overlap and repeated fetches are more common. S16 is the adversarial
# scenario added 2026-05-13 — sized at ~7 % of the suspicious slice so a
# 10 000-session corpus produces roughly 200 S16 examples (matching the
# adversarial-coverage target). Re-balanced from S1 (-0.04) and S3
# (-0.03) so the mix still sums to 1.0.
SCENARIO_MIX = {
    "S1":  0.16,
    "S2":  0.04,
    "S3":  0.17,
    "S4":  0.10,
    "S5":  0.05,
    "S7":  0.10,
    "S10": 0.10,
    "S11": 0.08,
    "S12": 0.05,
    "S13": 0.04,
    "S14": 0.03,
    "S15": 0.01,
    "S16": 0.07,
}
assert abs(sum(SCENARIO_MIX.values()) - 1.0) < 1e-6, "SCENARIO_MIX must sum to 1.0"

CATEGORIES = (
    "application_restart", "os_patching", "network_config",
    "incident_remediation", "user_access_provisioning",
    "certificate_rotation", "database_maintenance",
    "windows_admin", "cloud_resource_change",
)
SQL_CATEGORIES = ("database_maintenance",)
RDP_CATEGORIES = ("windows_admin",)
CLOUD_CATEGORIES = ("cloud_resource_change",)

# Categories whose justification template is *non-destructive* — used by S7 to
# pick a ticket where destructive ops are obviously out of scope.
NON_DESTRUCTIVE_CATEGORIES = (
    "application_restart", "incident_remediation", "user_access_provisioning",
    "database_maintenance", "windows_admin",
)

BUSINESS_HOURS = (time(8, 0), time(18, 0))

# Module-level augmenter handle; set by generate() when LLM augmentation is on.
# When None (Stage A behaviour) the generator emits raw template-rendered
# justifications and pristine command pools.
_AUGMENTER: "llm_augment.Augmenter | None" = None


# --------------------------------------------------------------------------- #
# Vocab loading
# --------------------------------------------------------------------------- #

def load_vocab(*, augmenter: "llm_augment.Augmenter | None" = None,
               expand_factor: int = 1, expand_seed: int = 0) -> dict[str, Any]:
    """Load all JSON vocab files into one dict, dropping the documentation key.

    Two normalisations on top of the raw JSON:
      * SQL pools are stored as [statement, is_destructive] in the file. v2
        has dropped the destructive flag from session output, so we project
        each entry down to its statement string here.
      * If `augmenter` is provided and `expand_factor > 0`, every command pool
        is run through the augmenter's expand_command_pool so the dataset
        draws from a wider vocabulary. The expansion is deterministic given
        `expand_seed`."""

    out: dict[str, Any] = {}
    for f in sorted(VOCAB_DIR.glob("*.json")):
        with f.open(encoding="utf-8") as fp:
            data = json.load(fp)
        data.pop("_doc", None)
        out[f.stem] = data

    # Drop the destructive flag from SQL — v2 sessions don't carry it.
    sql = out.get("sql_commands", {})
    for k, v in list(sql.items()):
        if isinstance(v, list) and v and isinstance(v[0], list):
            sql[k] = [entry[0] for entry in v]

    if augmenter is not None and expand_factor > 0:
        # Counter ensures each pool gets a unique sub-seed so two pools that
        # happen to contain the same command don't expand identically.
        # `justifications` is excluded — those are templates with {placeholders}
        # that the paraphraser handles per-ticket, not command strings.
        sub = 0
        skip_groups = {"justifications", "users", "assets"}
        for grp_name, grp in out.items():
            if grp_name in skip_groups or not isinstance(grp, dict):
                continue
            for k, v in list(grp.items()):
                if isinstance(v, list) and v and isinstance(v[0], str):
                    grp[k] = augmenter.expand_command_pool(
                        v, expand_factor, expand_seed + sub)
                    sub += 1
    return out


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #

def pick_user(rng: random.Random, users: dict, role: str | None = None) -> str:
    if role and role in users:
        return rng.choice(users[role])
    role = rng.choice(list(users.keys()))
    return rng.choice(users[role])


def pick_approver(rng: random.Random, users: dict, requester: str) -> str:
    """Pick an approver. Almost always different from the requester (S15
    sessions deliberately violate this)."""

    approvers = users.get("approver_manager", []) + users.get("approver_cab", [])
    candidates = [a for a in approvers if a != requester] or approvers
    return rng.choice(candidates) if candidates else "unknown.approver"


def pick_asset(rng: random.Random, assets: dict, category: str) -> str:
    surface = assets["_category_to_surface"].get(category, "linux_hosts")
    return rng.choice(assets[surface])


_TICKET_COUNTER = [0]
_USED_TICKET_IDS: set[str] = set()


def make_ticket_id(faker: Faker) -> str:
    """Generate a unique ticket id. Faker's random_number alone produces
    birthday-paradox collisions at our dataset size (~3 in 7700 tickets);
    we retry until we land on an unused number, then return it."""
    while True:
        tid = f"CHG{faker.random_number(digits=7, fix_len=True)}"
        if tid not in _USED_TICKET_IDS:
            _USED_TICKET_IDS.add(tid)
            return tid


def random_business_hour_ts(rng: random.Random, day: datetime) -> datetime:
    h = rng.randint(BUSINESS_HOURS[0].hour, BUSINESS_HOURS[1].hour - 1)
    m = rng.randint(0, 59)
    return day.replace(hour=h, minute=m, second=rng.randint(0, 59))


def random_off_hours_ts(rng: random.Random, day: datetime) -> datetime:
    """Pick a deep off-hours timestamp on `day`. Limited to 00:00-05:59 or
    22:00-23:59 so the timestamp is always strictly outside any change
    window (which lives between 08:00 and 21:59 in v2). This keeps S5 from
    accidentally landing inside an extended ticket window."""
    h = rng.choice(list(range(0, 6)) + list(range(22, 24)))
    m = rng.randint(0, 59)
    return day.replace(hour=h, minute=m, second=rng.randint(0, 59))


def render_justification(template: str, rng: random.Random,
                         asset: str, user: str, ticket_id: str,
                         window: tuple[datetime, datetime]) -> str:
    return template.format(
        asset=asset, user=user, ticket=ticket_id,
        window_start=window[0].isoformat(timespec="minutes"),
        window_end=window[1].isoformat(timespec="minutes"),
    )


def fmt(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Command sampling per surface (no is_destructive flag in v2)
# --------------------------------------------------------------------------- #

def sample_commands(rng: random.Random, vocab: dict,
                    category: str, n: int,
                    flavour: str = "in_scope") -> list[dict]:
    """Return n command dicts. flavour selects the pool; possible values are
    in_scope / suspicious / destructive / mass_export / sensitive_data /
    lateral_movement / privilege_escalation."""

    if flavour == "sensitive_data":
        examples = vocab["sensitive_keywords"]["sensitive_command_examples"]
        return [{"surface": _surface_for(category), "command": rng.choice(examples)}
                for _ in range(n)]
    if flavour == "lateral_movement":
        examples = vocab["lateral_commands"]["lateral_command_examples"]
        return [{"surface": "ssh", "command": rng.choice(examples)}
                for _ in range(n)]
    if flavour == "privilege_escalation":
        examples = vocab["escalation_commands"]["escalation_command_examples"]
        return [{"surface": "ssh", "command": rng.choice(examples)}
                for _ in range(n)]

    if category in SQL_CATEGORIES:
        return _sample_sql(rng, vocab, n, flavour)
    if category in RDP_CATEGORIES:
        return _sample_rdp(rng, vocab, n, flavour)
    if category in CLOUD_CATEGORIES:
        return _sample_cloud(rng, vocab, n, flavour)
    return _sample_ssh(rng, vocab, category, n, flavour)


def _surface_for(category: str) -> str:
    if category in SQL_CATEGORIES: return "sql"
    if category in RDP_CATEGORIES: return "rdp"
    if category in CLOUD_CATEGORIES: return "cloud"
    return "ssh"


def _sample_ssh(rng, vocab, category, n, flavour):
    pool = vocab["ssh_commands"].get(category) or vocab["ssh_commands"]["incident_remediation"]
    if flavour == "suspicious":
        pool = vocab["ssh_commands"]["suspicious"]
    return [{"surface": "ssh", "command": rng.choice(pool)} for _ in range(n)]


def _sample_sql(rng, vocab, n, flavour):
    pools = vocab["sql_commands"]
    if flavour == "in_scope":
        pool = pools["readonly"] + pools["ddl"]
    elif flavour == "destructive":
        pool = pools["destructive"]
    elif flavour == "mass_export":
        pool = pools["mass_export"] + pools["destructive"]
    else:
        pool = pools["destructive"] + pools["mass_export"]
    return [{"surface": "sql", "command": rng.choice(pool)} for _ in range(n)]


def _sample_rdp(rng, vocab, n, flavour):
    pool_key = "suspicious" if flavour == "suspicious" else "legitimate"
    pool = vocab["rdp_blocks"][pool_key]
    out = []
    for _ in range(n):
        title, app, keys = rng.choice(pool)
        out.append({"surface": "rdp", "command": f"{app}: {keys}",
                    "window_title": title})
    return out


def _sample_cloud(rng, vocab, n, flavour):
    pool = vocab["cloud_commands"]["suspicious" if flavour == "suspicious"
                                    else "legitimate"]
    return [{"surface": "cloud", "command": rng.choice(pool)} for _ in range(n)]


# --------------------------------------------------------------------------- #
# Single session builder
# --------------------------------------------------------------------------- #

def _assemble_ticket(faker, rng, vocab, day, requester) -> dict:
    cat = rng.choice(CATEGORIES)
    asset = pick_asset(rng, vocab["assets"], cat)
    ticket_id = make_ticket_id(faker)
    win_start = random_business_hour_ts(rng, day)
    win_end = win_start + timedelta(hours=rng.randint(1, 4))
    just_template = rng.choice(vocab["justifications"][cat])
    justification = render_justification(just_template, rng, asset, requester,
                                          ticket_id, (win_start, win_end))
    if _AUGMENTER is not None:
        justification = _AUGMENTER.paraphrase_justification(justification, ticket_id)
    approver = pick_approver(rng, vocab["users"], requester)
    return {
        "ticket_id": ticket_id,
        "category": cat,
        "approver": approver,
        "business_justification": justification,
        "named_assets": [asset],
        "change_window_start": fmt(win_start),
        "change_window_end": fmt(win_end),
        # Internal helpers (not serialised to the dataset)
        "_window": (win_start, win_end),
        "_asset": asset,
    }


def _build_session(rng, faker, vocab, ticket: dict, sid: str,
                   user: str, day: datetime, *,
                   commands: list[dict] | None = None,
                   sess_host: str | None = None,
                   sess_start: datetime | None = None,
                   n_fetches: int = 1) -> dict:
    cat = ticket["category"]
    win_start, win_end = ticket["_window"]
    if sess_start is None:
        sess_start = win_start + timedelta(minutes=rng.randint(0, 20))
    if sess_host is None:
        sess_host = ticket["_asset"]
    if commands is None:
        commands = sample_commands(rng, vocab, cat, rng.randint(4, 12), "in_scope")

    fetches = [
        {"timestamp": fmt(win_start - timedelta(seconds=30 * i + 5)),
         "user": user}
        for i in range(n_fetches)
    ]
    cmd_records = [
        {"timestamp": fmt(sess_start + timedelta(seconds=30 * i + 5)), **c}
        for i, c in enumerate(commands)
    ]
    return {
        "session_id": sid,
        "user": user,
        "ticket_id": ticket["ticket_id"],
        "ticket": {
            "category": cat,
            "approver": ticket["approver"],
            "business_justification": ticket["business_justification"],
            "named_assets": ticket["named_assets"],
            "change_window_start": ticket["change_window_start"],
            "change_window_end": ticket["change_window_end"],
        },
        "session": {
            "start_time": fmt(sess_start),
            "host": sess_host,
            "commands": cmd_records,
        },
        "vault_fetches": fetches,
    }


# --------------------------------------------------------------------------- #
# Per-scenario emission. Each function returns a LIST of session dicts and a
# parallel list of scenario tags so S1 / S11 (which emit several sessions per
# call) integrate naturally.
# --------------------------------------------------------------------------- #

def emit_clean(rng, faker, vocab, day, sid_alloc, *, share_partner=False):
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    sessions = [_build_session(rng, faker, vocab, ticket,
                               sid_alloc(), requester, day,
                               n_fetches=rng.randint(1, 2))]
    tags = [None]
    if share_partner:
        # Legitimate pair work: a second user runs a short follow-up session
        # on the same ticket within the change window. Partner must differ
        # from BOTH the requester (no self-pair) AND the approver (otherwise
        # we'd accidentally generate an approver-as-implementer event and
        # mislabel it clean).
        approver = ticket["approver"]
        partner = pick_user(rng, vocab["users"])
        while partner == requester or partner == approver:
            partner = pick_user(rng, vocab["users"])
        win_start, _ = ticket["_window"]
        sess_start = win_start + timedelta(minutes=rng.randint(20, 50))
        sessions.append(_build_session(rng, faker, vocab, ticket,
                                       sid_alloc(), partner, day,
                                       sess_start=sess_start,
                                       n_fetches=1))
        tags.append(None)
    return sessions, tags


def emit_S1(rng, faker, vocab, day, sid_alloc):
    """Multi-user at scale with three sub-tiers."""
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    tier_roll = rng.random()
    if tier_roll < 0.60:
        n_users = rng.randint(2, 3)        # small (most common)
    elif tier_roll < 0.90:
        n_users = rng.randint(5, 15)       # medium
    else:
        n_users = rng.randint(50, 100)     # large (rare, high-risk)
    win_start, win_end = ticket["_window"]
    chosen_users = [requester]
    pool = [u for role in vocab["users"].values() for u in role
            if u != requester]
    rng.shuffle(pool)
    chosen_users.extend(pool[:n_users - 1])
    sessions: list[dict] = []
    tags: list[str | None] = []
    for i, u in enumerate(chosen_users):
        sess_start = win_start + timedelta(minutes=rng.randint(0, 50))
        cat = ticket["category"]
        n_cmds = rng.randint(2, 6)
        commands = sample_commands(rng, vocab, cat, n_cmds, "in_scope")
        sessions.append(_build_session(rng, faker, vocab, ticket,
                                       sid_alloc(), u, day,
                                       commands=commands,
                                       sess_start=sess_start,
                                       n_fetches=1))
        tags.append("S1")
    return sessions, tags


def emit_S2(rng, faker, vocab, day, sid_alloc):
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    vague = rng.choice(vocab["justifications"]["vague"])
    if "{user}" in vague:
        vague = vague.format(user=requester)
    ticket["business_justification"] = vague
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day)], ["S2"]


def emit_S3(rng, faker, vocab, day, sid_alloc):
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    win_start, _ = ticket["_window"]
    win_end = win_start + timedelta(hours=rng.randint(8, 18))
    ticket["_window"] = (win_start, win_end)
    ticket["change_window_end"] = fmt(win_end)
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           n_fetches=rng.randint(8, 15))], ["S3"]


def emit_S4(rng, faker, vocab, day, sid_alloc):
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    other = rng.choice([c for c in CATEGORIES if c != ticket["category"]])
    n_cmds = rng.randint(4, 12)
    cmds = sample_commands(rng, vocab, other, n_cmds, "in_scope")
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           commands=cmds)], ["S4"]


def emit_S5(rng, faker, vocab, day, sid_alloc):
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    sess_start = random_off_hours_ts(rng, day)
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           sess_start=sess_start)], ["S5"]


def emit_S7(rng, faker, vocab, day, sid_alloc):
    """Destructive ops on a non-destructive ticket. Forces SQL surface so
    actual DELETE/DROP/TRUNCATE text appears in commands; with the
    is_destructive flag now removed the rule must read the SQL itself."""
    requester = pick_user(rng, vocab["users"])
    cat = "database_maintenance"
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    ticket["category"] = cat
    ticket["_asset"] = pick_asset(rng, vocab["assets"], cat)
    ticket["named_assets"] = [ticket["_asset"]]
    ticket["business_justification"] = render_justification(
        rng.choice(vocab["justifications"][cat]),
        rng, ticket["_asset"], requester,
        ticket["ticket_id"], ticket["_window"]
    )
    if _AUGMENTER is not None:
        ticket["business_justification"] = _AUGMENTER.paraphrase_justification(
            ticket["business_justification"], ticket["ticket_id"])
    n_cmds = rng.randint(4, 12)
    n_creep = max(1, n_cmds // 2)
    commands = (sample_commands(rng, vocab, cat, n_cmds - n_creep, "in_scope")
                + sample_commands(rng, vocab, cat, n_creep, "destructive"))
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           commands=commands)], ["S7"]


def emit_S10(rng, faker, vocab, day, sid_alloc):
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    candidates = [a for a in vocab["assets"]["linux_hosts"]
                  if a != ticket["_asset"]]
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           sess_host=rng.choice(candidates))], ["S10"]


def emit_S11(rng, faker, vocab, day, sid_alloc):
    """Zombie ticket: same ticket reused over several days, some sessions land
    after the change window has expired."""
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    win_start, win_end = ticket["_window"]
    n_sessions = rng.randint(3, 6)
    sessions, tags = [], []
    for i in range(n_sessions):
        if i == 0:
            # First session lands inside the original change window.
            sess_start = win_start + timedelta(minutes=rng.randint(5, 30))
            sess_day = day
        else:
            # Subsequent sessions creep N days past the change window — these
            # are the "zombie" ones that should never have happened on this
            # ticket.
            days_after = i * rng.randint(1, 3)
            sess_day = day + timedelta(days=days_after)
            sess_start = sess_day.replace(
                hour=rng.randint(9, 17), minute=rng.randint(0, 59),
                second=rng.randint(0, 59))
        sessions.append(_build_session(rng, faker, vocab, ticket,
                                       sid_alloc(), requester, sess_day,
                                       sess_start=sess_start))
        tags.append("S11")
    return sessions, tags


def emit_S12(rng, faker, vocab, day, sid_alloc):
    """Sensitive-data probing inside a routine-maintenance ticket."""
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    # Pick a routine-looking category that DOES NOT mention sensitive data.
    cat = rng.choice(("database_maintenance", "incident_remediation"))
    ticket["category"] = cat
    ticket["_asset"] = pick_asset(rng, vocab["assets"], cat)
    ticket["named_assets"] = [ticket["_asset"]]
    ticket["business_justification"] = render_justification(
        rng.choice(vocab["justifications"][cat]),
        rng, ticket["_asset"], requester,
        ticket["ticket_id"], ticket["_window"]
    )
    if _AUGMENTER is not None:
        ticket["business_justification"] = _AUGMENTER.paraphrase_justification(
            ticket["business_justification"], ticket["ticket_id"])
    n_cmds = rng.randint(4, 10)
    n_sens = max(1, n_cmds // 3)
    commands = (sample_commands(rng, vocab, cat, n_cmds - n_sens, "in_scope")
                + sample_commands(rng, vocab, cat, n_sens, "sensitive_data"))
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           commands=commands)], ["S12"]


def emit_S13(rng, faker, vocab, day, sid_alloc):
    """Lateral movement: jump/probe other hosts beyond named_assets."""
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    n_cmds = rng.randint(4, 10)
    n_lat = max(1, n_cmds // 3)
    commands = (sample_commands(rng, vocab, ticket["category"],
                                n_cmds - n_lat, "in_scope")
                + sample_commands(rng, vocab, ticket["category"],
                                  n_lat, "lateral_movement"))
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           commands=commands)], ["S13"]


def emit_S14(rng, faker, vocab, day, sid_alloc):
    """Privilege escalation on a ticket that did not authorise it."""
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    n_cmds = rng.randint(4, 10)
    n_esc = max(1, n_cmds // 3)
    commands = (sample_commands(rng, vocab, ticket["category"],
                                n_cmds - n_esc, "in_scope")
                + sample_commands(rng, vocab, ticket["category"],
                                  n_esc, "privilege_escalation"))
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           commands=commands)], ["S14"]


def emit_S15(rng, faker, vocab, day, sid_alloc):
    """Approver-as-implementer: requester == approver."""
    requester = pick_user(rng, vocab["users"])
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    ticket["approver"] = requester
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day)], ["S15"]


# --------------------------------------------------------------------------- #
# S16 — vocabulary-cloak adversarial scenario (added 2026-05-13)
# --------------------------------------------------------------------------- #
#
# Destructive intent expressed via paraphrased SQL that AVOIDS the literal
# trigger tokens the rule detector matches on:
#     "delete ", "drop ", "truncate", "outfile", "pg_dump", "mysqldump",
#     "update " (all with trailing space, case-insensitive).
#
# Each command achieves a destructive or mass-export effect through stored
# procedure calls, ALTER ... RENAME orphaning, partition detachment, or
# CREATE TABLE AS SELECT export — all common DBA patterns that legitimately
# never use the seven trigger tokens. Table names are deliberately neutral
# ("audit_log", "legacy_orders", "staging_records") so the S12 sensitive-
# keyword rule also stays silent.
#
# The cloak invariant uses the EXACT rule-detector trigger phrases (kept
# in sync with rule_detector.detect.DESTRUCTIVE_TOKENS plus the special
# "update ... set" two-token check) so the assertion below is a faithful
# guard, not a conservative approximation. The token list is duplicated
# here intentionally — generator and rule detector are independent
# components and we don't want a circular import; instead any change in
# the rule detector's destructive vocabulary must be mirrored here.
_S16_RULE_DESTRUCTIVE_TOKENS = (
    "delete from", "drop table", "drop database", "drop schema",
    "truncate table", "into outfile", "pg_dump", "mysqldump",
)
_S16_OBFUSCATED_DESTRUCTIVE = (
    "CALL purge_old_records('audit_log', INTERVAL 30 DAY);",
    "EXEC sp_purgehistory @days = 0;",
    "EXEC msdb.dbo.sp_delete_backuphistory '2026-01-01';",
    "ALTER TABLE legacy_orders RENAME TO _orphaned_archive_q1_2026;",
    "ALTER TABLE staging_records DETACH PARTITION p_old;",
    "CREATE TABLE export_dump_2026 AS SELECT * FROM order_history;",
    "CALL archive_module.purge_expired();",
    "REPLACE INTO order_history SELECT id, NULL, NULL, NULL FROM expired_entries;",
    "EXEC dbo.sp_archive_to_blob @table = 'audit_log', @retention_days = 30;",
    "INSERT INTO _shadow_archive SELECT * FROM expired_entries; CALL flush_buffer('expired_entries');",
)


def _trips_s7_rule(text: str) -> str | None:
    """Return the trigger phrase that would fire the S7 rule, or None.
    Mirrors rule_detector.detect._has_destructive_text exactly."""
    lower = text.lower()
    for tok in _S16_RULE_DESTRUCTIVE_TOKENS:
        if tok in lower:
            return tok
    if "update " in lower and " set " in lower:
        return "update ... set"
    return None


# Module-load-time assertion: the S16 obfuscated vocabulary MUST avoid every
# trigger phrase the rule detector matches for S7. Failing fast here is
# much friendlier than discovering the regression downstream as a quiet
# drop in adversarial detection rate.
for _cmd in _S16_OBFUSCATED_DESTRUCTIVE:
    _hit = _trips_s7_rule(_cmd)
    assert _hit is None, (
        f"S16 vocab cloak weakened: command {_cmd!r} contains rule "
        f"trigger phrase {_hit!r}; this would let the S7 rule fire on "
        f"S16 sessions and defeat the adversarial design.")
del _cmd, _hit

# Paraphrased justification template engineered to share vocabulary with the
# obfuscated commands ("purge", "archive", "audit_log", "legacy", "expired",
# "rename", "retention") so the s5 keyword-overlap compliance signal also
# passes — making the session look fully compliant to anything that does
# not read commands semantically.
_S16_JUSTIFICATION_TEMPLATES = (
    ("As authorised under change ticket {ticket_id}, performing routine "
     "data lifecycle cleanup on {asset_name}: archiving expired records "
     "from the audit log table, purging stale backup history beyond the "
     "30-day retention window, and renaming legacy order tables to align "
     "with the current naming convention. All operations follow the standard "
     "archive-then-purge runbook reviewed quarterly. Window {window_start} "
     "to {window_end}, approved by {approver}."),
    ("Routine quarterly retention cleanup on {asset_name} per change ticket "
     "{ticket_id}: archiving order_history rows older than 24 months, "
     "detaching expired partitions from staging_records, and purging "
     "audit_log entries past the agreed retention window. Each step is "
     "reversible from the staging archive snapshot. Window {window_start} "
     "to {window_end}, approved by {approver}."),
    ("Change ticket {ticket_id} authorises end-of-quarter archival on "
     "{asset_name}. Activities: rename legacy_orders to its quarterly "
     "archive name, purge expired_entries via the standard purge module, "
     "and export order_history snapshots to the archive blob. All reversible "
     "via backup. Window {window_start} to {window_end}, approved by {approver}."),
)

# Categories that read as routine maintenance (not data destruction) so the
# overall ticket framing does not pre-flag the session for a human reviewer.
# `incident_remediation` is intentionally EXCLUDED — it sits on the rule
# detector's DESTRUCTIVE_OK_CATEGORIES whitelist so S7 is suppressed for
# incident tickets regardless. Using it would not be a fair adversarial
# test of the cloak. The remaining three categories are the ones where
# the rule detector WOULD fire on a literal destructive token, which is
# exactly the scenario the cloak is designed to defeat.
_S16_TICKET_CATEGORIES = ("application_restart", "database_maintenance",
                          "user_access_provisioning")


def _render_s16_justification(rng, asset, ticket_id, approver, window):
    template = rng.choice(_S16_JUSTIFICATION_TEMPLATES)
    win_start, win_end = window
    return template.format(
        ticket_id=ticket_id,
        asset_name=asset,
        approver=approver,
        window_start=fmt(win_start),
        window_end=fmt(win_end),
    )


def emit_S16(rng, faker, vocab, day, sid_alloc):
    """S16 — vocabulary-cloak adversarial scenario.

    Destructive intent is achieved via stored-procedure / DDL paraphrase
    that avoids the rule detector's seven literal trigger tokens; the
    justification reuses command vocabulary so the s5 compliance signal
    also passes. Rule baseline cannot detect S16 by construction; the
    LLM detector should generalise from semantics once retrained with
    S16 examples in its training set."""
    requester = pick_user(rng, vocab["users"])
    cat = rng.choice(_S16_TICKET_CATEGORIES)
    ticket = _assemble_ticket(faker, rng, vocab, day, requester)
    ticket["category"] = cat
    ticket["_asset"] = pick_asset(rng, vocab["assets"], cat)
    ticket["named_assets"] = [ticket["_asset"]]
    ticket["business_justification"] = _render_s16_justification(
        rng, ticket["_asset"], ticket["ticket_id"],
        ticket["approver"], ticket["_window"])
    if _AUGMENTER is not None:
        ticket["business_justification"] = _AUGMENTER.paraphrase_justification(
            ticket["business_justification"], ticket["ticket_id"])

    n_in_scope = rng.randint(2, 5)
    n_obfuscated = rng.randint(2, 4)
    # Defensive: a category like database_maintenance can return in-scope DDL
    # whose text trips the S7 rule (e.g. "VACUUM ANALYZE orders" is fine but
    # any "delete from"/"drop table"/"into outfile" form is not). Over-sample
    # then filter via _trips_s7_rule so the cloak is preserved regardless of
    # which in-scope pool the category maps to. If filtering leaves us short,
    # fall back to repeated sampling.
    in_scope_cmds: list[dict] = []
    attempts = 0
    while len(in_scope_cmds) < n_in_scope and attempts < 6:
        attempts += 1
        for c in sample_commands(rng, vocab, cat, n_in_scope * 2, "in_scope"):
            if _trips_s7_rule(c.get("command") or "") is not None:
                continue
            in_scope_cmds.append(c)
            if len(in_scope_cmds) >= n_in_scope:
                break
    in_scope_cmds = in_scope_cmds[:n_in_scope]
    obfuscated_cmds = [{"surface": "sql",
                        "command": rng.choice(_S16_OBFUSCATED_DESTRUCTIVE)}
                       for _ in range(n_obfuscated)]
    commands = in_scope_cmds + obfuscated_cmds
    rng.shuffle(commands)
    return [_build_session(rng, faker, vocab, ticket,
                           sid_alloc(), requester, day,
                           commands=commands)], ["S16"]


_SCENARIO_EMITTERS = {
    "S1":  emit_S1,  "S2":  emit_S2,  "S3":  emit_S3,  "S4":  emit_S4,
    "S5":  emit_S5,  "S7":  emit_S7,  "S10": emit_S10, "S11": emit_S11,
    "S12": emit_S12, "S13": emit_S13, "S14": emit_S14, "S15": emit_S15,
    "S16": emit_S16,
}


# --------------------------------------------------------------------------- #
# Self-validation
# --------------------------------------------------------------------------- #

def self_check(sessions: list[dict], labels: list[dict],
               vocab: dict) -> dict[str, int]:
    """Return per-scenario counts of sessions that *fail* the expected feature
    invariant. Zero across the board means the dataset is internally
    consistent."""

    by_id = {s["session_id"]: s for s in sessions}
    by_ticket: dict[str, list[dict]] = {}
    for s in sessions:
        by_ticket.setdefault(s["ticket_id"], []).append(s)
    failures = Counter()

    sens_tokens = (vocab["sensitive_keywords"]["pii_columns"]
                   + vocab["sensitive_keywords"]["secret_files"])
    sens_tokens = [t.lower() for t in sens_tokens]
    lat_tokens = [t.lower() for t in vocab["lateral_commands"]["jump_patterns"]
                  + vocab["lateral_commands"]["discovery_patterns"]]
    esc_tokens = [t.lower() for t in vocab["escalation_commands"]["elevation_patterns"]]
    destr_tokens = ("delete ", "drop ", "truncate", "outfile",
                    "pg_dump", "mysqldump", "update ")

    for lab in labels:
        if lab["label"] != SUSPICIOUS_LABEL:
            continue
        sess = by_id[lab["session_id"]]
        sc = lab["scenario"]
        if sc == "S1":
            tid = sess["ticket_id"]
            users = {s["user"] for s in by_ticket[tid]}
            if len(users) < 2:
                failures["S1"] += 1
        elif sc == "S2":
            if len(sess["ticket"]["business_justification"]) >= 40:
                failures["S2"] += 1
        elif sc == "S3":
            if len(sess["vault_fetches"]) < 5:
                failures["S3"] += 1
        elif sc == "S5":
            ws = datetime.fromisoformat(sess["ticket"]["change_window_start"])
            we = datetime.fromisoformat(sess["ticket"]["change_window_end"])
            ss = datetime.fromisoformat(sess["session"]["start_time"])
            if ws <= ss <= we and BUSINESS_HOURS[0] <= ss.time() <= BUSINESS_HOURS[1]:
                failures["S5"] += 1
        elif sc == "S7":
            text = " ".join(c["command"] for c in sess["session"]["commands"]).lower()
            if not any(t in text for t in destr_tokens):
                failures["S7"] += 1
        elif sc == "S10":
            if sess["session"]["host"] in sess["ticket"]["named_assets"]:
                failures["S10"] += 1
        elif sc == "S11":
            we = datetime.fromisoformat(sess["ticket"]["change_window_end"])
            tid = sess["ticket_id"]
            sessions_for_ticket = by_ticket[tid]
            late_sessions = [s for s in sessions_for_ticket
                             if datetime.fromisoformat(s["session"]["start_time"]) > we]
            # S11 invariant: the ticket must carry at least 3 sessions in total
            # AND at least one must start strictly after the window closes.
            if len(sessions_for_ticket) < 3 or not late_sessions:
                failures["S11"] += 1
        elif sc == "S12":
            text = " ".join(c["command"] for c in sess["session"]["commands"]).lower()
            if not any(t in text for t in sens_tokens):
                failures["S12"] += 1
        elif sc == "S13":
            text = " ".join(c["command"] for c in sess["session"]["commands"]).lower()
            if not any(t in text for t in lat_tokens):
                failures["S13"] += 1
        elif sc == "S14":
            text = " ".join(c["command"] for c in sess["session"]["commands"]).lower()
            if not any(t in text for t in esc_tokens):
                failures["S14"] += 1
        elif sc == "S15":
            if sess["user"] != sess["ticket"]["approver"]:
                failures["S15"] += 1
        elif sc == "S16":
            # S16 cloaks destructive intent: the ticket category must read
            # as routine maintenance AND no command may contain a phrase
            # that would trip the S7 rule. The destructive payload itself
            # is verified at module-import time (the cloak assertion).
            if sess["ticket"]["category"] not in _S16_TICKET_CATEGORIES:
                failures["S16"] += 1
                continue
            text = " ".join(c["command"] for c in sess["session"]["commands"])
            if _trips_s7_rule(text) is not None:
                failures["S16"] += 1
                continue
            # At least one obfuscated destructive command must be present —
            # otherwise the session is indistinguishable from a clean one
            # and the ground-truth label is wrong.
            s16_marker_stubs = (
                "purge_", "sp_purgehistory", "sp_delete_backuphistory",
                "rename to _orphaned", "detach partition",
                "create table export_dump", "archive_module.purge",
                "sp_archive_to_blob", "_shadow_archive",
            )
            if not any(stub in text.lower() for stub in s16_marker_stubs):
                failures["S16"] += 1
    return dict(failures)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def generate(num_sessions: int, seed: int, out_dir: Path,
             clean_ratio: float = 0.7,
             share_partner_rate: float = 0.30,
             augment: bool = True,
             llm_backend: str = "mock",
             expand_factor: int = 1) -> dict:
    """Generate `num_sessions` PAM session rows and write them to `out_dir`.

    `share_partner_rate` is the fraction of clean sessions that come bundled
    with a second 'pair work' session on the same ticket. Drives the
    avg-sessions-per-ticket figure toward the D2 target (~1.33).

    `augment` toggles the LLM-augmentation layer (Stage B). When True the
    generator routes justifications through `llm_augment.paraphrase` and
    expands every command pool via `llm_augment.expand_commands`. The
    backend defaults to "mock" (deterministic, offline). Setting it to
    "huggingface" requires `transformers` and `torch` and is what runs in
    the Kaggle reference notebook (Stage F)."""

    rng = random.Random(seed)
    faker = Faker()
    Faker.seed(seed)
    augmenter = llm_augment.Augmenter(backend=llm_backend) if augment else None
    global _AUGMENTER
    _AUGMENTER = augmenter
    # Fresh per-call so back-to-back generate() calls don't leak ticket-ids.
    _USED_TICKET_IDS.clear()
    _TICKET_COUNTER[0] = 0
    vocab = load_vocab(augmenter=augmenter, expand_factor=expand_factor,
                       expand_seed=seed)

    n_clean_target = int(round(num_sessions * clean_ratio))
    n_susp_target = num_sessions - n_clean_target
    susp_per_scenario = {sc: max(1, int(round(n_susp_target * f)))
                         for sc, f in SCENARIO_MIX.items()}
    drift = n_susp_target - sum(susp_per_scenario.values())
    biggest = max(susp_per_scenario, key=susp_per_scenario.get)
    susp_per_scenario[biggest] += drift

    base_day = datetime(2026, 5, 1, 0, 0, 0)

    sessions: list[dict] = []
    labels: list[dict] = []
    sid_counter = [0]

    def alloc_sid() -> str:
        sid = f"sess-{sid_counter[0]:06d}"
        sid_counter[0] += 1
        return sid

    # tqdm is a hard dep listed in requirements.txt, but degrade gracefully so
    # a fresh checkout where the user hasn't pip-installed yet still runs.
    try:
        from tqdm import tqdm                                       # type: ignore[import-not-found]
    except ImportError:
        def tqdm(x=None, **kwargs):                                 # type: ignore[no-redef]
            class _Noop:
                def __init__(self, total=None, **_): self.total = total
                def update(self, n=1): pass
                def close(self): pass
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _Noop(**kwargs) if x is None else x

    print(f"[gen] Emitting {num_sessions:,} sessions "
          f"({n_susp_target:,} suspicious + {num_sessions - n_susp_target:,} "
          f"clean)...", flush=True)

    pbar = tqdm(total=num_sessions, desc="[gen] sessions",
                unit="sess", leave=True)

    # Suspicious first (some scenarios produce many sessions per call, so we
    # may overshoot the per-scenario count slightly; we cap once we hit the
    # global suspicious budget).
    susp_emitted = Counter()
    susp_total = 0
    for sc, target in susp_per_scenario.items():
        attempts_remaining = target * 4
        while susp_emitted[sc] < target and susp_total < n_susp_target and attempts_remaining > 0:
            attempts_remaining -= 1
            day = base_day + timedelta(days=rng.randint(0, 29))
            new_sessions, new_tags = _SCENARIO_EMITTERS[sc](rng, faker, vocab, day, alloc_sid)
            for s, t in zip(new_sessions, new_tags):
                sessions.append(s)
                labels.append({"session_id": s["session_id"],
                               "label": SUSPICIOUS_LABEL, "scenario": t})
                susp_emitted[sc] += 1
                susp_total += 1
                pbar.update(1)
                if susp_total >= n_susp_target:
                    break

    # Clean sessions to fill the remainder
    while len(sessions) < num_sessions:
        day = base_day + timedelta(days=rng.randint(0, 29))
        share = rng.random() < share_partner_rate
        new_sessions, new_tags = emit_clean(rng, faker, vocab, day, alloc_sid,
                                            share_partner=share)
        for s, _ in zip(new_sessions, new_tags):
            sessions.append(s)
            labels.append({"session_id": s["session_id"],
                           "label": CLEAN_LABEL, "scenario": None})
            pbar.update(1)
            if len(sessions) >= num_sessions:
                break
    pbar.close()

    # HF backend post-emission pass: every paraphrase_justification call
    # during emission was deferred (returns original text, records the
    # request). Now flush the queue in one batched HF call and substitute
    # the paraphrased text back into each session's ticket. Mock backend
    # is a no-op here (its augmenter has no pending queue).
    #
    # Defensive check: only substitute when the cached paraphrase is
    # non-empty AND meaningfully longer than 10 chars AND not identical
    # to the original. The augmenter already filters bad outputs (empty,
    # short, echoed) at flush time, but we re-validate here so even a
    # corrupted cache state can't poison every session's text. If we
    # accepted a stale empty cache value, every justification would be
    # blank and Stage 5's llm-trained F1 would collapse to zero.
    if augmenter is not None and augmenter.backend == "huggingface":
        n_flushed = augmenter.flush_pending_paraphrases()
        if n_flushed:
            print(f"[gen] Substituting {n_flushed} HF paraphrases back "
                  f"into sessions...", flush=True)
            cache = augmenter._cache
            n_sub = n_skip_empty = n_skip_same = 0
            samples: list[tuple[str, str]] = []
            for s in sessions:
                cur = s["ticket"]["business_justification"]
                ck = (cur, s["ticket_id"], None)
                new = cache.get(ck)
                if new is None:
                    continue
                if not new.strip() or len(new.strip()) < 10:
                    n_skip_empty += 1
                    continue
                if new == cur:
                    n_skip_same += 1
                    continue
                s["ticket"]["business_justification"] = new
                n_sub += 1
                if len(samples) < 3:
                    samples.append((cur, new))
            print(f"[gen]   substituted {n_sub} sessions; skipped "
                  f"{n_skip_empty} (empty/short cache) + {n_skip_same} "
                  f"(unchanged) — sessions retaining original templates: "
                  f"{len(sessions) - n_sub}", flush=True)
            if samples:
                print("[gen]   sample substitutions:", flush=True)
                for orig, para in samples:
                    o = (orig[:60] + "...") if len(orig) > 60 else orig
                    p = (para[:60] + "...") if len(para) > 60 else para
                    print(f"[gen]     {o!r}", flush=True)
                    print(f"[gen]     -> {p!r}", flush=True)

    # ---------------------------------------------------------------- #
    # Stage 1 sanity stats — the single most useful field-debug tool.
    # If the trainer downstream is going to underfit because every
    # session has the same blank text, you'll see it RIGHT HERE before
    # spending GPU minutes on Stage 3.
    # ---------------------------------------------------------------- #
    just_lengths = [len((s["ticket"]["business_justification"] or "").strip())
                    for s in sessions]
    n_blank = sum(1 for L in just_lengths if L == 0)
    n_short = sum(1 for L in just_lengths if 0 < L < 10)
    distinct = len({s["ticket"]["business_justification"] for s in sessions})
    avg_len = (sum(just_lengths) / len(just_lengths)) if just_lengths else 0.0
    print(f"[gen] Justification stats: {len(sessions)} sessions, "
          f"{distinct} distinct strings, avg_len={avg_len:.1f} chars, "
          f"{n_blank} blank, {n_short} <10-char.", flush=True)
    if n_blank + n_short > len(sessions) * 0.10:
        print(f"[gen] WARNING: {n_blank + n_short}/{len(sessions)} "
              f"justifications are blank or shorter than 10 chars. The "
              f"trainer downstream will struggle. Investigate the [hf] "
              f"sample log above; possible causes:",
              flush=True)
        print(f"[gen]   * transformers < 5.0 on Kaggle "
              f"(pip install --upgrade 'transformers>=5.5,<6')",
              flush=True)
        print(f"[gen]   * PAM_HF_MAX_NEW_TOKENS too low (default 80)",
              flush=True)
        print(f"[gen]   * Defender / corporate AV intercepting model "
              f"weight mmaps — run `python doctor.py --with-download`",
              flush=True)
    if distinct < 50 and len(sessions) > 500:
        print(f"[gen] WARNING: only {distinct} distinct justification "
              f"strings across {len(sessions)} sessions. The dataset will "
              f"underfit even a perfect classifier. Expected: > 100 "
              f"distinct (Flan-T5 produces ~7k unique paraphrases for a "
              f"10k-session run).", flush=True)

    # Truncate exactly to num_sessions (defensive — the loop above can land
    # one over because emit_clean and S1/S11 are bursty).
    sessions = sessions[:num_sessions]
    labels = labels[:num_sessions]

    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = out_dir / "sessions.json"
    labels_path = out_dir / "labels.json"
    manifest_path = out_dir / "manifest.json"

    with sessions_path.open("w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=1)
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=1)

    failures = self_check(sessions, labels, vocab)

    n_clean = sum(1 for l in labels if l["label"] == CLEAN_LABEL)
    n_susp = sum(1 for l in labels if l["label"] == SUSPICIOUS_LABEL)
    unique_tickets = len({s["ticket_id"] for s in sessions})

    manifest = {
        "seed": seed,
        "num_sessions_requested": num_sessions,
        "num_sessions_actual": len(sessions),
        "num_clean": n_clean,
        "num_suspicious": n_susp,
        "unique_tickets": unique_tickets,
        "avg_sessions_per_ticket": round(len(sessions) / unique_tickets, 3),
        "scenario_counts": dict(susp_emitted),
        "self_check_failures": failures,
        "self_check_passed": all(v == 0 for v in failures.values()),
        "sessions_sha256": _sha256(sessions_path),
        "labels_sha256": _sha256(labels_path),
        "augmentation": {
            "enabled": augment,
            "backend": llm_backend if augment else None,
            "expand_factor": expand_factor if augment else 0,
        },
        "generator_version": "drishti-pam-stage-B",
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-sessions", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--clean-ratio", type=float, default=0.70)
    p.add_argument("--share-partner-rate", type=float, default=0.30)
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   default=True,
                   help="Disable Stage B LLM augmentation; emit raw "
                        "template-rendered justifications.")
    p.add_argument("--llm-backend", choices=["mock", "huggingface"],
                   default="mock",
                   help="Augmentation backend. 'mock' is offline + "
                        "deterministic; 'huggingface' loads Flan-T5-base.")
    p.add_argument("--expand-factor", type=int, default=1,
                   help="Number of variants generated per command. 0 disables "
                        "command-pool expansion.")
    args = p.parse_args(argv)
    manifest = generate(args.num_sessions, args.seed, args.out_dir,
                        args.clean_ratio, args.share_partner_rate,
                        augment=args.augment,
                        llm_backend=args.llm_backend,
                        expand_factor=args.expand_factor)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["self_check_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
