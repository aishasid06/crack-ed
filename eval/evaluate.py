"""Evaluate a pipeline run against seed/labels.json (SPEC §9 Phase 10).

Test split only by default. Labels are right-joined to the run's cases, so listings the pipeline never
captured count: uncaptured `ignore` listings are correct ignores, uncaptured actionable listings are
discovery misses. Discovery recall is reported separately from classification (captured listings)
and end-to-end. All headline metrics carry bootstrap 95% CIs and are broken down per site and per
difficulty. Writes metrics.json (schema read by the UI's Evaluation page) and report.md.

    python -m eval.evaluate [--run ID] [--split test|dev] [--out DIR] [--bootstrap 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.config import AppConfig, get_config
from bh.db import Case, Category, Listing, Run, get_engine, utcnow
from bh.trace import describe_task_statuses
from eval import metrics as M

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1


def latest_run_with_cases(engine: Engine) -> int:
    with Session(engine) as s:
        ids = sorted({c.run_id for c in s.exec(select(Case)) if c.run_id is not None})
    if not ids:
        raise RuntimeError("no cases in the database — run the pipeline first (make run)")
    return ids[-1]


def load_predictions(engine: Engine, run_id: int) -> tuple[dict[str, M.Prediction], dict[str, Any]]:
    """Cases of ``run_id`` keyed by listing URL path, plus that run's totals."""
    preds: dict[str, M.Prediction] = {}
    with Session(engine) as s:
        run = s.get(Run, run_id)
        if run is None:
            raise KeyError(f"run {run_id} not found")
        for c in s.exec(select(Case).where(Case.run_id == run_id)):
            listing = s.get(Listing, c.listing_id)
            if listing is not None:
                preds[urlparse(listing.url).path] = M.Prediction(str(c.category), str(c.action), float(c.confidence))
        info = {"run_id": run_id, "status": str(run.status), "started_at": run.started_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None, "totals": run.totals}
    return preds, info


def evaluate(run_id: int | None = None, split: str = "test", labels_path: Path = ROOT / "seed" / "labels.json",
             engine: Engine | None = None, cfg: AppConfig | None = None, n_boot: int = 2000) -> dict[str, Any]:
    eng = engine or get_engine()
    cfg = cfg or get_config()
    rid = run_id if run_id is not None else latest_run_with_cases(eng)
    labels = json.loads(labels_path.read_text())
    preds, run_info = load_predictions(eng, rid)
    rows = M.join(labels, preds, split)
    unmatched = len(set(preds) - {v["url_path"] for v in labels.values()})
    categories = [c.value for c in Category]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utcnow().isoformat(),
        "run": run_info,
        "split": split,
        "split_note": "held-out test split" if split == "test" else "fitted on (thresholds and scorer) — optimistic",
        "definitions": {
            "report_precision": "predicted platform_report that should be reported / predicted platform_report",
            "report_recall": "should_report listings predicted platform_report / should_report listings "
                             "(uncaptured = not reported)",
            "escalate_recall_lenient": "expected escalate predicted escalate or platform_report (reaches a lawyer)",
            "escalate_recall_strict": "expected escalate predicted exactly escalate",
            "false_alarm_rate": "expected ignore but reported or escalated (sent to a lawyer) / expected ignore",
            "e2e_action_accuracy": "predicted action == expected action over all listings (uncaptured = ignore)",
            "discovery_recall": "captured / actionable (expected_action != ignore)",
            "category_accuracy_captured": "predicted category == label category over captured listings",
            "ci": f"bootstrap percentile 95% CI over listings, {n_boot} resamples, seed 0 (ci_low/ci_high); "
                  "Wilson score 95% interval for proportions (wilson_low/wilson_high)",
        },
        "counts": {**M.counts(rows), "predictions_without_label": unmatched},
        "headline": M.headline(rows, n_boot),
        "by_site": M.breakdown(rows, "site", n_boot),
        "by_difficulty": M.breakdown(rows, "difficulty", n_boot),
        "classification": M.classification(rows, categories),
        "pr_curve": M.pr_curve(rows, cfg.thresholds.act_threshold),
        "misses": [{"key": r.key, "category": r.category, "expected_action": r.expected_action,
                    "predicted_action": r.pred_action, "predicted_category": r.pred_category,
                    "difficulty": r.difficulty} for r in rows
                   if r.e2e_action != r.expected_action or (r.captured and r.pred_category != r.category)],
    }


# ---------------------------------------------------------------- report


def _fmt(e: dict[str, Any]) -> str:
    """value [Wilson 95%] for proportions; value [bootstrap 95%] otherwise."""
    if e["value"] is None:
        return "n/a"
    if e.get("wilson_low") is not None:
        return f"{e['value']:.2f} [{e['wilson_low']:.2f}–{e['wilson_high']:.2f}]"
    ci = f" [{e['ci_low']:.2f}–{e['ci_high']:.2f}]" if e.get("ci_low") is not None else ""
    return f"{e['value']:.2f}{ci}"


def _metric_table(groups: dict[str, Any], names: list[str]) -> list[str]:
    head = "| | n | " + " | ".join(names) + " |"
    lines = [head, "|" + "---|" * (len(names) + 2)]
    for g, v in groups.items():
        lines.append(f"| {g} | {v['counts']['listings']} | " +
                     " | ".join(_fmt(v["metrics"][n]) for n in names) + " |")
    return lines


def _matrix(cm: dict[str, Any]) -> list[str]:
    lines = ["| label \\ predicted | " + " | ".join(cm["cols"]) + " |", "|" + "---|" * (len(cm["cols"]) + 1)]
    for r, vals in zip(cm["rows"], cm["matrix"], strict=True):
        lines.append(f"| {r} | " + " | ".join(str(v) if v else "·" for v in vals) + " |")
    return lines


def report_markdown(res: dict[str, Any]) -> str:
    h, c = res["headline"], res["counts"]
    names = ["report_precision", "report_recall", "escalate_recall_lenient", "false_alarm_rate", "discovery_recall",
             "e2e_action_accuracy"]
    tot = res["run"].get("totals") or {}
    llm = tot.get("llm", {})
    out = [f"# Evaluation — run {res['run']['run_id']} ({res['split']} split)", "",
           f"*{res['split_note']}.* Decision support for a lawyer — not legal advice.", "",
           "## Headline", "",
           "Intervals are 95%: Wilson score intervals for proportions (they stay informative at 0/n and n/n, "
           "where a percentile bootstrap collapses), bootstrap percentile intervals for the accuracies. "
           "metrics.json carries both.", "",
           f"- **Precision on should_report:** {_fmt(h['report_precision'])} "
           f"({c['true_report']}/{c['predicted_report']})",
           f"- **Recall on escalate** (reaches a lawyer): {_fmt(h['escalate_recall_lenient'])} "
           f"({c['escalate_reaching_lawyer']}/{c['expected_escalate']}); strictly escalate: "
           f"{_fmt(h['escalate_recall_strict'])} ({c['escalate_exact']}/{c['expected_escalate']})",
           f"- Recall on should_report: {_fmt(h['report_recall'])} ({c['true_report']}/{c['should_report']})",
           f"- **False alarms** (should be ignored, sent to a lawyer): {_fmt(h['false_alarm_rate'])} "
           f"({c['false_alarms']}/{c['expected_ignore']})",
           f"- **Discovery recall:** {_fmt(h['discovery_recall'])} ({c['actionable_captured']}/{c['actionable']} "
           f"actionable captured; {c['uncaptured_ignore_correct']} uncaptured ignore listings counted correct)",
           f"- End-to-end action accuracy: {_fmt(h['e2e_action_accuracy'])} · category accuracy on captured: "
           f"{_fmt(h['category_accuracy_captured'])}", "",
           "## By site", "", *_metric_table(res["by_site"], names), "",
           "## By difficulty", "", *_metric_table(res["by_difficulty"], names), "",
           "## Actions (end to end, uncaptured as its own column)", "",
           *_matrix(res["classification"]["action_confusion_end_to_end"]), "",
           "## Categories (captured listings)", "", *_matrix(res["classification"]["category_confusion_captured"]), ""]
    pr = res["pr_curve"]
    if pr.get("average_precision") is not None:
        a = pr["act_threshold"]
        out += ["## PR curve (should_report vs confidence)", "",
                f"Average precision {pr['average_precision']:.3f}. At act_threshold {a['threshold']}: precision "
                f"{a['precision']:.2f}, recall {a['recall']:.2f}. Actual decisions (rules + downgrades): precision "
                f"{_fmt({'value': pr['decisions']['precision'], 'ci_low': None})}, recall "
                f"{_fmt({'value': pr['decisions']['recall'], 'ci_low': None})}. Points in metrics.json.", ""]
    if res["misses"]:
        out += ["## Errors", "", "| listing | label | expected | predicted action | predicted category |",
                "|---|---|---|---|---|"]
        out += [f"| {m['key']} | {m['category']} ({m['difficulty']}) | {m['expected_action']} | "
                f"{m['predicted_action']} | {m['predicted_category']} |" for m in res["misses"]]
        out.append("")
    out += ["## Cost and time (run totals)", "",
            f"- Wall time: {tot.get('wall_time_s', 'n/a')} s · tasks: {tot.get('tasks', {}).get('total', 'n/a')} "
            f"({describe_task_statuses(tot.get('tasks', {}).get('by_status', {}))}) "
            f"· max parallelism: {tot.get('max_parallelism', 'n/a')}",
            f"- LLM calls: {llm.get('calls', 0)} · input tokens: {llm.get('input_tokens', 0)} · output tokens: "
            f"{llm.get('output_tokens', 0)}"]
    for model, v in (llm.get("by_model") or {}).items():
        out.append(f"  - {model}: {v['calls']} calls, {v['input_tokens']} in / {v['output_tokens']} out")
    return "\n".join(out) + "\n"


def write(res: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j, md = out_dir / "metrics.json", out_dir / "report.md"
    j.write_text(json.dumps(res, indent=2))
    md.write_text(report_markdown(res))
    return j, md


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=None, help="run whose cases to evaluate (default: latest with cases)")
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--labels", type=Path, default=ROOT / "seed" / "labels.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    res = evaluate(args.run, args.split, args.labels, n_boot=args.bootstrap)
    out = args.out or cfg.paths.resolve(cfg.paths.data_dir) / "eval" / f"run_{res['run']['run_id']}_{args.split}"
    j, md = write(res, out)
    print(report_markdown(res).split("## By site")[0])
    print(f"wrote {j} and {md}")


if __name__ == "__main__":
    main()
