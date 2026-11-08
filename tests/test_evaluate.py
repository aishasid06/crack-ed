"""Evaluation metrics and report (SPEC §9 Phase 10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bh.db import Action, Case, Category, Listing, Run, session_scope
from eval import metrics as M
from eval.evaluate import evaluate, report_markdown, write


def _lab(i: int, cat: str, action: str, site: str = "mockbay", difficulty: str = "easy", split: str = "test") -> tuple[str, dict[str, Any]]:
    return f"{site}:{i}", {"category": cat, "expected_action": action, "should_report": action == "platform_report",
                           "difficulty": difficulty, "split": split, "site": site, "url_path": f"/{site}/item/{i}"}


LABELS = dict([
    _lab(1, "counterfeit_identical", "platform_report"),
    _lab(2, "counterfeit_identical", "platform_report"),
    _lab(3, "copied_photos", "platform_report", "tradeloft", "hard"),
    _lab(4, "grey_import", "escalate"),
    _lab(5, "lookalike", "escalate", "tradeloft"),
    _lab(6, "lookalike", "escalate", "tradeloft", "hard"),
    _lab(7, "unrelated", "ignore"),             # uncaptured → correct ignore
    _lab(8, "genuine_resale", "monitor"),
    _lab(9, "authorised", "ignore", split="dev"),  # excluded by split
])
PREDS = {
    "/mockbay/item/1": M.Prediction("counterfeit_identical", "platform_report", 0.95),
    "/mockbay/item/2": M.Prediction("counterfeit_identical", "escalate", 0.55),        # downgraded: report miss
    # 3 uncaptured: report discovery miss
    "/mockbay/item/4": M.Prediction("grey_import", "platform_report", 0.75),          # escalate reached via report
    "/tradeloft/item/5": M.Prediction("lookalike", "escalate", 0.2),
    # 6 uncaptured: escalate discovery miss
    "/mockbay/item/8": M.Prediction("genuine_resale", "monitor", 0.1),
}


@pytest.fixture
def rows() -> list[M.Row]:
    return M.join(LABELS, PREDS, "test")


def test_right_join_keeps_uncaptured(rows: list[M.Row]) -> None:
    assert len(rows) == 8  # dev row excluded
    by = {r.key: r for r in rows}
    assert not by["tradeloft:3"].captured and by["tradeloft:3"].pred_action == M.NOT_CAPTURED
    assert by["mockbay:7"].e2e_action == "ignore" == by["mockbay:7"].expected_action  # correct ignore


def test_headline_counts_and_values(rows: list[M.Row]) -> None:
    c = M.counts(rows)
    assert c == {"listings": 8, "captured": 5, "actionable": 7, "actionable_captured": 5, "predicted_report": 2,
                 "true_report": 1, "should_report": 3, "expected_escalate": 3, "escalate_reaching_lawyer": 2,
                 "escalate_exact": 1, "uncaptured_ignore_correct": 1, "expected_ignore": 1, "false_alarms": 0}
    assert M.report_precision(rows) == 0.5  # 1 of 2 reports correct (grey import reported)
    assert M.report_recall(rows) == pytest.approx(1 / 3)  # uncaptured copy counts as a miss
    assert M.escalate_recall_lenient(rows) == pytest.approx(2 / 3)
    assert M.escalate_recall_strict(rows) == pytest.approx(1 / 3)
    assert M.discovery_recall(rows) == pytest.approx(5 / 7)
    assert M.e2e_action_accuracy(rows) == pytest.approx(4 / 8)  # 1, 5, 7, 8
    assert M.category_accuracy_captured(rows) == 1.0


def test_false_alarm_rate_counts_ignores_sent_to_a_lawyer() -> None:
    labels = dict([_lab(1, "unrelated", "ignore"), _lab(2, "unrelated", "ignore"), _lab(3, "unrelated", "ignore"),
                   _lab(4, "authorised", "ignore"), _lab(5, "counterfeit_identical", "platform_report")])
    preds = {"/mockbay/item/1": M.Prediction("lookalike", "escalate", 0.3),          # false alarm
             "/mockbay/item/2": M.Prediction("copied_photos", "platform_report", 0.8),  # false alarm
             "/mockbay/item/4": M.Prediction("authorised", "ignore", 0.1),
             "/mockbay/item/5": M.Prediction("counterfeit_identical", "platform_report", 0.9)}
    rows = M.join(labels, preds, "test")  # 3 uncaptured → not sent anywhere
    assert M.false_alarm_rate(rows) == pytest.approx(2 / 4)
    c = M.counts(rows)
    assert (c["false_alarms"], c["expected_ignore"]) == (2, 4)
    assert M.wilson_intervals(rows)["false_alarm_rate"]["num"] == 2


def test_metric_undefined_without_denominator() -> None:
    only_ignore = M.join(dict([_lab(1, "unrelated", "ignore")]), {}, "test")
    assert M.report_precision(only_ignore) is None and M.discovery_recall(only_ignore) is None


def test_bootstrap_is_deterministic_and_brackets_point(rows: list[M.Row]) -> None:
    a = M.bootstrap(rows, M.discovery_recall, n=500, seed=3)
    b = M.bootstrap(rows, M.discovery_recall, n=500, seed=3)
    assert a == b and a.ci_low <= a.value <= a.ci_high and a.n == 8 and 0 < a.valid_resamples <= 500
    perfect = [r for r in rows if r.captured]
    e = M.bootstrap(perfect, M.discovery_recall, n=200)
    assert e.value == e.ci_low == e.ci_high == 1.0


def test_breakdowns(rows: list[M.Row]) -> None:
    site = M.breakdown(rows, "site", n=200)
    assert set(site) == {"mockbay", "tradeloft"}
    assert site["tradeloft"]["counts"]["listings"] == 3 and site["tradeloft"]["counts"]["captured"] == 1
    assert site["tradeloft"]["metrics"]["discovery_recall"]["value"] == pytest.approx(1 / 3)
    diff = M.breakdown(rows, "difficulty", n=200)
    assert diff["hard"]["counts"]["actionable_captured"] == 0


def test_confusions_and_pr(rows: list[M.Row]) -> None:
    cls = M.classification(rows, [c.value for c in Category])
    act = cls["action_confusion_end_to_end"]
    i = act["rows"].index("platform_report")
    assert dict(zip(act["cols"], act["matrix"][i], strict=True)) == {
        "platform_report": 1, "escalate": 1, "monitor": 0, "ignore": 0, M.NOT_CAPTURED: 1}
    assert sum(map(sum, cls["category_confusion_captured"]["matrix"])) == 5
    pr = M.pr_curve(rows, 0.7)
    assert 0 < pr["average_precision"] <= 1 and pr["points"]
    assert pr["act_threshold"] == {"threshold": 0.7, "precision": 0.5, "recall": pytest.approx(1 / 3)}
    assert pr["decisions"]["precision"] == 0.5


def test_evaluate_end_to_end_from_db(engine: Any, tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(LABELS))
    with session_scope(engine) as s:
        run = Run()
        s.add(run)
        s.flush()
        rid = run.id
        run.totals = {"wall_time_s": 12.5, "tasks": {"total": 40}, "max_parallelism": 3,
                      "llm": {"calls": 2, "input_tokens": 900, "output_tokens": 80,
                              "by_model": {"claude-opus-5-5": {"calls": 2, "input_tokens": 900, "output_tokens": 80}}}}
        for path, p in PREDS.items():
            site = path.split("/")[1]
            l = Listing(platform=site, url=f"http://127.0.0.1:8000{path}", title="x")
            s.add(l)
            s.flush()
            s.add(Case(listing_id=l.id, run_id=rid, category=Category(p.category), confidence=p.confidence,
                       rule_id="r", action=Action(p.action)))
        other = Run()
        s.add(other)  # a later run without cases must not be picked by default
    res = evaluate(None, "test", labels_path, engine, n_boot=200)
    assert res["run"]["run_id"] == rid and res["split"] == "test"
    assert res["headline"]["report_precision"]["value"] == 0.5 and res["counts"]["predictions_without_label"] == 0
    assert {"mockbay", "tradeloft"} == set(res["by_site"]) and {"easy", "hard"} == set(res["by_difficulty"])
    assert {m["key"] for m in res["misses"]} == {"mockbay:2", "tradeloft:3", "mockbay:4", "tradeloft:6"}
    j, md = write(res, tmp_path / "out")
    assert json.loads(j.read_text())["schema_version"] == 1
    text = md.read_text()
    assert "Precision on should_report:** 0.50 [" in text and "Recall on escalate" in text
    assert "claude-opus-5-5: 2 calls" in text and "## By difficulty" in text
    assert report_markdown(evaluate(rid, "dev", labels_path, engine, n_boot=50)).count("optimistic") == 1


def test_wilson_interval_informative_at_extremes() -> None:
    lo, hi = M.wilson(16, 16)
    assert hi == 1.0 and 0.75 < lo < 0.85  # bootstrap would say [1, 1]
    lo0, hi0 = M.wilson(0, 10)
    assert lo0 == 0.0 and 0.2 < hi0 < 0.35
    assert M.wilson(0, 0) == (None, None)
    lo5, hi5 = M.wilson(5, 10)
    assert lo5 < 0.5 < hi5


def test_headline_carries_both_intervals(rows: list[M.Row]) -> None:
    h = M.headline(rows, n=200)
    rp = h["report_precision"]
    assert rp["num"] == 1 and rp["den"] == 2 and rp["wilson_low"] < 0.5 < rp["wilson_high"]
    assert rp["ci_low"] is not None and "wilson_low" not in h["e2e_action_accuracy"]
