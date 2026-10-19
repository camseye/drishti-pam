"""Compare the rule detector and the trained-LLM detector on the test split.

Reads:
    data/labels.json
    data/split.json
    data/rule_predictions.json
    data/llm_predictions.json

Writes:
    data/comparison.json     machine-readable per-detector report
    data/comparison.md       human-readable Markdown report

Both detectors are evaluated on the *same test partition* defined in
data/split.json, so the headline numbers are directly comparable. Per-scenario
recall is the binary metric "did the detector mark this session as suspicious",
which works for both the rule detector (which also exposes per-rule hits) and
the binary LLM detector.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S7", "S10",
             "S11", "S12", "S13", "S14", "S15", "S16")


def _load(path: Path) -> dict[str, dict]:
    return {r["session_id"]: r for r in json.load(path.open(encoding="utf-8"))}


def compute_agreement(rule_preds: dict, llm_preds: dict, labels: dict,
                      test_ids: set[str]) -> dict:
    """Compute pairwise agreement between rule and llm-trained on the test split.

    Returns a dict with four mutually-exclusive aggregate counts that sum to
    test_size, plus a by_ground_truth breakdown that splits each of those four
    cells across (ground_truth_suspicious, ground_truth_clean). The split is
    what the dashboard's *scenario × detector-agreement* chart renders.

    Counts are computed only on session ids that appear in BOTH prediction
    files AND in the test partition; sessions present in only one prediction
    file are silently skipped (this should not happen in a healthy pipeline
    but defends against partial reruns).
    """
    paired_ids = (set(rule_preds) & set(llm_preds)) & test_ids
    n = len(paired_ids)

    both_flag = both_clean = rule_only = llm_only = 0
    both_tp = both_fn = rule_tp_only = llm_tp_only = 0
    both_tn = both_fp = rule_fp_only = llm_fp_only = 0

    for sid in paired_ids:
        r_susp = rule_preds[sid]["prediction"] == "suspicious"
        l_susp = llm_preds[sid]["prediction"] == "suspicious"
        gt_susp = labels[sid]["label"] == "suspicious"

        if r_susp and l_susp:
            both_flag += 1
            if gt_susp:
                both_tp += 1
            else:
                both_fp += 1
        elif (not r_susp) and (not l_susp):
            both_clean += 1
            if gt_susp:
                both_fn += 1
            else:
                both_tn += 1
        elif r_susp and not l_susp:
            rule_only += 1
            if gt_susp:
                rule_tp_only += 1
            else:
                rule_fp_only += 1
        else:
            llm_only += 1
            if gt_susp:
                llm_tp_only += 1
            else:
                llm_fp_only += 1

    n_susp = sum(1 for sid in paired_ids
                 if labels[sid]["label"] == "suspicious")
    n_clean = n - n_susp

    return {
        "_doc": ("Pairwise agreement between rule and llm-trained on the "
                 "test split. The four aggregate counts are mutually "
                 "exclusive and sum to test_size. by_ground_truth splits "
                 "each cell across the suspicious / clean ground-truth "
                 "label so downstream consumers can refer to specific "
                 "sub-counts (e.g. 'rule-only true positives')."),
        "test_size": n,
        "both_flag": both_flag,
        "both_clean": both_clean,
        "rule_only_flag": rule_only,
        "llm_only_flag": llm_only,
        "by_ground_truth": {
            "ground_truth_suspicious": {
                "n": n_susp,
                "both_TP": both_tp,
                "rule_TP_only": rule_tp_only,
                "llm_TP_only": llm_tp_only,
                "both_FN": both_fn,
            },
            "ground_truth_clean": {
                "n": n_clean,
                "both_TN": both_tn,
                "rule_FP_only": rule_fp_only,
                "llm_FP_only": llm_fp_only,
                "both_FP": both_fp,
            },
        },
    }


def compute_operational_metrics(
    reports: list[dict],
    agreement: dict | None,
    *,
    fp_triage_minutes: float = 18.0,
    fp_triage_minutes_lower: float = 8.0,
    batch_cadence_minutes: float = 5.0,
    audit_window_hours: float = 24.0,
) -> dict:
    """Derive security-operational metrics from existing test-split reports.

    Five sub-blocks:

      1. ``assumptions`` — the operational constants every other sub-block
         depends on, named explicitly so an analyst can adjust them and
         re-run without hunting through code. The eighteen-minute FP
         triage cost is a mechanistic high-baseline drawn from a Tier-2
         analyst-shift walkthrough; the eight-minute lower bound matches
         the productivity-gain bracket Gartner reports for UEBA platforms
         with integrated triage views and is reported alongside as the
         conservative alternative.
      2. ``time_to_detect`` — *structural* TTD per detector under a
         five-minute batch-scoring cadence. The dataset has no
         per-event timestamps to support a finer estimate, so this is
         a budget rather than a measurement; the field name
         ``ttd_median_minutes_structural`` is verbose on purpose.
      3. ``false_positive_cost`` — per-detector FP cost converted to
         analyst-hours under both the eighteen- and eight-minute
         baselines, plus a delta showing the saving each detector
         offers against a hypothetical "every alert is reviewed for
         the high baseline" pessimistic ceiling.
      4. ``alert_volume`` — three configurations (rule-only,
         disjunctive ensemble, conjunctive ensemble) with the
         volume reduction percentage against the rule-only baseline.
         Disjunctive grows volume slightly (catches one more attack);
         conjunctive shrinks it materially.
      5. ``mean_time_between_false_positives`` — per-detector
         MTBF-FP under the assumed 24-hour audit window. The
         assumption itself is explicit: the test partition is
         a stratified random subset, not 24 hours of contiguous
         sessions, so the metric is a structural budget under the
         documented stationarity assumption rather than a measured
         inter-arrival time.

    Returns ``None`` only if the input ``reports`` list is empty;
    otherwise the function always returns a populated dict so
    downstream consumers can render its values unconditionally.
    """
    if not reports:
        return {
            "_doc": "No detector reports available — operational metrics skipped.",
            "available": False,
        }

    rule = next((r for r in reports if r["detector"] == "rule"), None)
    llm = next((r for r in reports
                if r["detector"] == "llm-trained"), None)

    def _ttd_block(report: dict) -> dict:
        # Structural TTD = half the batch cadence + (a tiny amortised
        # bump for the predict-pass itself, which is already implicit
        # in the cadence). Reported as the same number for every
        # detector because both run inside the same batch loop in this
        # pipeline. A future event-stream pipeline that scored sessions
        # incrementally would give per-detector numbers diverging here.
        ttd_min = batch_cadence_minutes / 2.0
        return {
            "ttd_median_minutes_structural": round(ttd_min, 2),
            "_note": (f"Half the assumed {batch_cadence_minutes:g}-minute "
                      "batch cadence; structural budget under the "
                      "synchronous-batch assumption, not measured."),
        }

    def _fp_cost_block(report: dict) -> dict:
        fp = report["binary"]["fp"]
        hi_h = round(fp * fp_triage_minutes / 60.0, 2)
        lo_h = round(fp * fp_triage_minutes_lower / 60.0, 2)
        return {
            "fp_count": fp,
            "fp_minutes_hi_baseline": round(fp * fp_triage_minutes, 1),
            "fp_minutes_lo_baseline": round(
                fp * fp_triage_minutes_lower, 1),
            "fp_hours_hi_baseline": hi_h,
            "fp_hours_lo_baseline": lo_h,
        }

    def _mtbf_block(report: dict) -> dict:
        fp = report["binary"]["fp"]
        if fp == 0:
            return {
                "fp_count": 0,
                "mtbf_fp_hours": None,
                "_note": ("Zero false positives in the test partition; "
                          "MTBF-FP is undefined."),
            }
        return {
            "fp_count": fp,
            "mtbf_fp_hours": round(audit_window_hours / fp, 2),
            "mtbf_fp_minutes": round(audit_window_hours * 60.0 / fp, 1),
        }

    # ---- Alert volume sub-block (only meaningful when both detectors
    # ran AND the agreement block is available, since disjunctive /
    # conjunctive volumes are derived from agreement counts).
    alert_volume_block: dict
    if rule is None or llm is None or agreement is None:
        alert_volume_block = {
            "available": False,
            "_note": ("Alert-volume reduction requires both detectors "
                      "and the agreement block. One of those is missing."),
        }
    else:
        rule_only_volume = rule["predicted_suspicious"]
        llm_only_volume = llm["predicted_suspicious"]
        disjunctive_volume = (agreement["both_flag"]
                              + agreement["rule_only_flag"]
                              + agreement["llm_only_flag"])
        conjunctive_volume = agreement["both_flag"]

        def _delta_pct(after: int, before: int) -> float:
            if not before:
                return 0.0
            return round((before - after) / before * 100.0, 1)

        alert_volume_block = {
            "available": True,
            "configurations": {
                "rule_only_baseline": {
                    "alert_count": rule_only_volume,
                    "reduction_vs_rule_pct": 0.0,
                },
                "llm_only": {
                    "alert_count": llm_only_volume,
                    "reduction_vs_rule_pct": _delta_pct(
                        llm_only_volume, rule_only_volume),
                },
                "disjunctive_ensemble": {
                    "alert_count": disjunctive_volume,
                    "reduction_vs_rule_pct": _delta_pct(
                        disjunctive_volume, rule_only_volume),
                    "_note": ("Flag if EITHER detector fires. Slightly "
                              "higher volume than rule-only by the "
                              "llm_only_flag count; trades a small "
                              "false-alarm rate increase for a small "
                              "true-positive recall increase."),
                },
                "conjunctive_ensemble": {
                    "alert_count": conjunctive_volume,
                    "reduction_vs_rule_pct": _delta_pct(
                        conjunctive_volume, rule_only_volume),
                    "_note": ("Flag only if BOTH detectors fire. Lower "
                              "volume than rule-only by the rule_only_flag "
                              "count; trades some recall for very high "
                              "precision; suitable for automated "
                              "remediation pipelines."),
                },
            },
        }

    return {
        "_doc": ("Security-operational metrics derived from the test-"
                 "split reports + agreement block. Every metric rests on "
                 "the operational constants in `assumptions`; an analyst "
                 "who disagrees with any of those constants can re-run "
                 "with `--fp-triage-minutes` / `--batch-cadence-minutes` "
                 "/ `--audit-window-hours` to substitute their own."),
        "available": True,
        "assumptions": {
            "fp_triage_minutes_hi_baseline": fp_triage_minutes,
            "fp_triage_minutes_lo_baseline": fp_triage_minutes_lower,
            "batch_cadence_minutes": batch_cadence_minutes,
            "audit_window_hours": audit_window_hours,
            "_note": ("hi_baseline is a Tier-2 analyst-shift mechanistic "
                      "estimate; lo_baseline is Gartner UEBA Market "
                      "Guide's lower-bracket figure. batch_cadence and "
                      "audit_window are illustrative budgets; the "
                      "synthetic corpus does not let either be "
                      "measured."),
        },
        "time_to_detect": {
            "rule": _ttd_block(rule) if rule else None,
            "llm-trained": _ttd_block(llm) if llm else None,
        },
        "false_positive_cost": {
            "rule": _fp_cost_block(rule) if rule else None,
            "llm-trained": _fp_cost_block(llm) if llm else None,
        },
        "alert_volume": alert_volume_block,
        "mean_time_between_false_positives": {
            "rule": _mtbf_block(rule) if rule else None,
            "llm-trained": _mtbf_block(llm) if llm else None,
        },
    }


def score(detector_name: str, preds: dict, labels: dict,
          test_ids: set[str]) -> dict:
    """Compute binary + per-scenario metrics on the test partition only."""
    preds_test = {sid: p for sid, p in preds.items() if sid in test_ids}
    labels_test = {sid: l for sid, l in labels.items() if sid in test_ids}

    n_clean = sum(1 for l in labels_test.values() if l["label"] == "clean")
    n_susp = sum(1 for l in labels_test.values() if l["label"] == "suspicious")

    n_pred_susp = sum(1 for p in preds_test.values()
                      if p["prediction"] == "suspicious")
    n_pred_clean = len(preds_test) - n_pred_susp

    tp = sum(1 for sid, p in preds_test.items()
             if p["prediction"] == "suspicious"
             and labels_test[sid]["label"] == "suspicious")
    fp = n_pred_susp - tp
    fn = n_susp - tp
    tn = n_clean - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    # Per-scenario binary recall: of the test sessions whose ground-truth
    # scenario is X, how many did the detector call suspicious?
    per_scenario: dict[str, dict] = {}
    for sc in SCENARIOS:
        gt_sids = [sid for sid, l in labels_test.items()
                   if l.get("scenario") == sc]
        gt = len(gt_sids)
        caught = sum(1 for sid in gt_sids
                     if preds_test[sid]["prediction"] == "suspicious")
        per_scenario[sc] = {
            "gt_count": gt,
            "caught": caught,
            "missed": gt - caught,
            "recall": (caught / gt) if gt else 0.0,
        }

    # Optional: per-rule firing pattern (only meaningful for the rule detector)
    rule_fires: Counter = Counter()
    for sid, p in preds_test.items():
        for r in p.get("scenario_hits") or []:
            rule_fires[r] += 1

    return {
        "detector": detector_name,
        "split": "test",
        "n_predictions": len(preds_test),
        "binary": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        },
        "predicted_suspicious": n_pred_susp,
        "predicted_clean": n_pred_clean,
        "ground_truth_suspicious": n_susp,
        "ground_truth_clean": n_clean,
        "per_scenario": per_scenario,
        "rule_fires": dict(rule_fires),  # empty for binary detectors
    }


# --------------------------------------------------------------------------- #
# Console + Markdown rendering
# --------------------------------------------------------------------------- #

def render_console(reports: list[dict], test_size: int) -> str:
    """Minimalist headline view for stdout.

    The full per-scenario / per-rule numbers go into ``comparison.json``
    and ``comparison.md`` — stdout shows only the four headline metrics
    and a one-line summary of which detector won. Anyone wanting the
    deeper breakdown opens the markdown file."""
    lines: list[str] = []
    bar = "─" * 64
    lines.append("")
    lines.append(bar)
    lines.append(f"  Detector evaluation — test split ({test_size:,} sessions)")
    lines.append(bar)
    lines.append("")
    lines.append(f"  {'Detector':<14}{'Precision':>11}{'Recall':>10}"
                 f"{'F1':>9}{'Accuracy':>11}")
    lines.append(f"  {'─' * 14}{'─' * 11}{'─' * 10}{'─' * 9}{'─' * 11}")
    for r in reports:
        b = r["binary"]
        lines.append(
            f"  {r['detector']:<14}{b['precision']:>11.4f}"
            f"{b['recall']:>10.4f}{b['f1']:>9.4f}{b['accuracy']:>11.4f}"
        )

    # One-line summary that names the winner on F1 (the metric most often
    # used to compare imbalanced binary classifiers). Skip the summary if
    # only one detector ran — there's nothing to compare against.
    if len(reports) >= 2:
        ranked = sorted(reports, key=lambda r: -r["binary"]["f1"])
        top, runner = ranked[0], ranked[1]
        delta = top["binary"]["f1"] - runner["binary"]["f1"]
        fn_diff = runner["binary"]["fn"] - top["binary"]["fn"]
        fp_diff = runner["binary"]["fp"] - top["binary"]["fp"]
        lines.append("")
        lines.append(f"  Winner: {top['detector']}  "
                     f"(F1 +{delta:.4f}, "
                     f"{fn_diff:+d} missed attacks, "
                     f"{fp_diff:+d} false alarms vs {runner['detector']})")
    lines.append("")
    return "\n".join(lines)


def render_markdown(reports: list[dict]) -> str:
    out: list[str] = []
    out.append("# Detector comparison (test split)\n")
    out.append("## Headline metrics — binary (clean vs suspicious)\n")
    out.append("| Detector | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |")
    out.append("|----------|----------:|-------:|---:|---------:|---:|---:|---:|---:|")
    for r in reports:
        b = r["binary"]
        out.append(f"| {r['detector']} | {b['precision']:.4f} | {b['recall']:.4f} "
                   f"| {b['f1']:.4f} | {b['accuracy']:.4f} | {b['tp']} | {b['fp']} "
                   f"| {b['fn']} | {b['tn']} |")
    out.append("")
    out.append("## Per-scenario recall on TEST split\n")
    cols = " | ".join(SCENARIOS)
    out.append(f"| Detector | {cols} |")
    out.append("|----------|" + "|".join(["---:" for _ in SCENARIOS]) + "|")
    for r in reports:
        cells = " | ".join(f"{r['per_scenario'][sc]['recall']:.0%}"
                           for sc in SCENARIOS)
        out.append(f"| {r['detector']} | {cells} |")
    out.append("")
    out.append("## Per-scenario ground-truth count, caught, missed\n")
    out.append(f"| Detector | metric | {cols} |")
    out.append("|----------|--------|" + "|".join(["---:" for _ in SCENARIOS]) + "|")
    for r in reports:
        for key, label in (("gt_count", "ground truth"),
                           ("caught", "correctly caught"),
                           ("missed", "missed")):
            cells = " | ".join(str(r["per_scenario"][sc][key])
                               for sc in SCENARIOS)
            out.append(f"| {r['detector']} | {label} | {cells} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=Path("data/labels.json"))
    p.add_argument("--split", type=Path, default=Path("data/split.json"))
    p.add_argument("--rule", type=Path, default=Path("data/rule_predictions.json"))
    p.add_argument("--llm", type=Path, default=Path("data/llm_predictions.json"))
    p.add_argument("--out-json", type=Path, default=Path("data/comparison.json"))
    p.add_argument("--out-md", type=Path, default=Path("data/comparison.md"))
    p.add_argument("--llm-threshold", type=float, default=None,
                   help=("Decision threshold the LLM detector used at "
                         "inference time. Surfaced under each LLM "
                         "report's 'decision_threshold' field so "
                         "downstream consumers can read the calibrated "
                         "value. Optional; if omitted, the field is "
                         "set to null."))
    p.add_argument("--fp-triage-minutes", type=float, default=18.0,
                   help=("Per-false-positive analyst-triage cost in "
                         "minutes (high baseline; default 18 reflects "
                         "a Tier-2 analyst-shift mechanistic estimate)."))
    p.add_argument("--fp-triage-minutes-lower", type=float, default=8.0,
                   help=("Lower-bound per-FP triage cost in minutes "
                         "(default 8 matches Gartner UEBA Market Guide's "
                         "lower-bracket integrated-triage figure)."))
    p.add_argument("--batch-cadence-minutes", type=float, default=5.0,
                   help=("Assumed batch-scoring cadence used to compute "
                         "structural time-to-detect (default 5 min)."))
    p.add_argument("--audit-window-hours", type=float, default=24.0,
                   help=("Assumed audit window in hours used to compute "
                         "mean-time-between-false-positives (default 24)."))
    args = p.parse_args(argv)

    labels = _load(args.labels)
    split = json.load(args.split.open(encoding="utf-8"))
    test_ids = set(split["test_ids"])

    detectors: list[tuple[str, Path]] = []
    if args.rule.exists():
        detectors.append(("rule", args.rule))
    if args.llm.exists():
        detectors.append(("llm-trained", args.llm))

    if not detectors:
        print("ERROR: no detector prediction files found", file=sys.stderr)
        return 2

    reports: list[dict] = []
    raw_preds: dict[str, dict] = {}
    for name, path in detectors:
        preds = _load(path)
        raw_preds[name] = preds
        unknown = [sid for sid in preds if sid not in labels]
        if unknown:
            print(f"WARNING: {len(unknown)} predictions in {name} reference "
                  f"unknown session ids; first: {unknown[:3]}", file=sys.stderr)
        report = score(name, preds, labels, test_ids)
        if name == "llm-trained" and args.llm_threshold is not None:
            report["decision_threshold"] = args.llm_threshold
        reports.append(report)

    # Pairwise agreement is only meaningful when BOTH detectors ran.
    agreement: dict | None = None
    if "rule" in raw_preds and "llm-trained" in raw_preds:
        agreement = compute_agreement(
            raw_preds["rule"], raw_preds["llm-trained"], labels, test_ids)

    # Operational metrics always emit (single-detector or paired) so
    # downstream consumers can render their values unconditionally.
    op_metrics = compute_operational_metrics(
        reports, agreement,
        fp_triage_minutes=args.fp_triage_minutes,
        fp_triage_minutes_lower=args.fp_triage_minutes_lower,
        batch_cadence_minutes=args.batch_cadence_minutes,
        audit_window_hours=args.audit_window_hours,
    )

    print(render_console(reports, len(test_ids)))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "reports": reports,
        "scenario_order": list(SCENARIOS),
        "test_size": len(test_ids),
        "split_seed": split.get("seed"),
        "train_ratio": split.get("train_ratio"),
    }
    if agreement is not None:
        payload["agreement"] = agreement
    payload["operational_metrics"] = op_metrics
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with args.out_md.open("w", encoding="utf-8") as f:
        f.write(render_markdown(reports))

    print(f"  Per-scenario breakdown:  {args.out_md}")
    print(f"  Machine-readable report: {args.out_json}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
