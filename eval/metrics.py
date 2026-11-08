"""Evaluation metrics (SPEC §9 Phase 10). Pure functions over joined rows; no database access.

Semantics:
- Rows are labelled listings right-joined to predictions, so listings the pipeline never captured
  still count. Uncaptured `ignore` listings are correct ignores; uncaptured actionable listings
  (report / escalate / monitor) are discovery misses.
- Discovery recall = captured / actionable. Classification metrics use captured rows only.
  End-to-end metrics treat an uncaptured listing as "not reported, not escalated" (ignore).
- Headline: precision on should_report; recall on escalate (lenient = the listing reaches a lawyer,
  predicted escalate or platform_report; strict = predicted exactly escalate).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

NOT_CAPTURED = "not_captured"
ACTIONS = ["platform_report", "escalate", "monitor", "ignore"]


@dataclass(frozen=True)
class Row:
    key: str
    site: str
    difficulty: str
    category: str
    expected_action: str
    should_report: bool
    captured: bool
    pred_category: str  # NOT_CAPTURED when uncaptured
    pred_action: str  # NOT_CAPTURED when uncaptured
    confidence: float  # P(should_report); 0 when uncaptured

    @property
    def actionable(self) -> bool:
        return self.expected_action != "ignore"

    @property
    def pred_report(self) -> bool:
        return self.pred_action == "platform_report"

    @property
    def e2e_action(self) -> str:
        return self.pred_action if self.captured else "ignore"


@dataclass(frozen=True)
class Prediction:
    category: str
    action: str
    confidence: float


def join(labels: dict[str, dict[str, Any]], predictions: dict[str, Prediction], split: str | None) -> list[Row]:
    """One row per labelled listing (right join). ``predictions`` is keyed by label ``url_path``."""
    rows = []
    for key, lab in sorted(labels.items()):
        if split and lab["split"] != split:
            continue
        p = predictions.get(lab["url_path"])
        rows.append(Row(key=key, site=lab["site"], difficulty=lab["difficulty"], category=lab["category"],
                        expected_action=lab["expected_action"], should_report=bool(lab["should_report"]),
                        captured=p is not None, pred_category=p.category if p else NOT_CAPTURED,
                        pred_action=p.action if p else NOT_CAPTURED, confidence=p.confidence if p else 0.0))
    return rows


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


# ---------------------------------------------------------------- metric functions (rows → value | None)


def report_precision(rows: list[Row]) -> float | None:
    pred = [r for r in rows if r.pred_report]
    return _ratio(sum(r.should_report for r in pred), len(pred))


def report_recall(rows: list[Row]) -> float | None:
    pos = [r for r in rows if r.should_report]
    return _ratio(sum(r.pred_report for r in pos), len(pos))


def escalate_recall_lenient(rows: list[Row]) -> float | None:
    pos = [r for r in rows if r.expected_action == "escalate"]
    return _ratio(sum(r.pred_action in ("escalate", "platform_report") for r in pos), len(pos))


def escalate_recall_strict(rows: list[Row]) -> float | None:
    pos = [r for r in rows if r.expected_action == "escalate"]
    return _ratio(sum(r.pred_action == "escalate" for r in pos), len(pos))


def e2e_action_accuracy(rows: list[Row]) -> float | None:
    return _ratio(sum(r.e2e_action == r.expected_action for r in rows), len(rows))


def discovery_recall(rows: list[Row]) -> float | None:
    act = [r for r in rows if r.actionable]
    return _ratio(sum(r.captured for r in act), len(act))


def category_accuracy_captured(rows: list[Row]) -> float | None:
    cap = [r for r in rows if r.captured]
    return _ratio(sum(r.pred_category == r.category for r in cap), len(cap))


HEADLINE: dict[str, Callable[[list[Row]], float | None]] = {
    "report_precision": report_precision,
    "report_recall": report_recall,
    "escalate_recall_lenient": escalate_recall_lenient,
    "escalate_recall_strict": escalate_recall_strict,
    "e2e_action_accuracy": e2e_action_accuracy,
    "discovery_recall": discovery_recall,
    "category_accuracy_captured": category_accuracy_captured,
}


def counts(rows: list[Row]) -> dict[str, int]:
    """The raw numerators/denominators behind the headline metrics."""
    return {
        "listings": len(rows), "captured": sum(r.captured for r in rows),
        "actionable": sum(r.actionable for r in rows),
        "actionable_captured": sum(r.actionable and r.captured for r in rows),
        "predicted_report": sum(r.pred_report for r in rows),
        "true_report": sum(r.pred_report and r.should_report for r in rows),
        "should_report": sum(r.should_report for r in rows),
        "expected_escalate": sum(r.expected_action == "escalate" for r in rows),
        "escalate_reaching_lawyer": sum(r.expected_action == "escalate" and r.pred_action in
                                        ("escalate", "platform_report") for r in rows),
        "escalate_exact": sum(r.expected_action == "escalate" and r.pred_action == "escalate" for r in rows),
        "uncaptured_ignore_correct": sum(not r.captured and r.expected_action == "ignore" for r in rows),
    }


# ---------------------------------------------------------------- intervals


def wilson(num: int, den: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score 95% interval for a proportion. Unlike the percentile bootstrap it stays informative
    at 0/n and n/n (a bootstrap of 16/16 is always 1.0)."""
    if den == 0:
        return None, None
    p = num / den
    centre = (p + z * z / (2 * den)) / (1 + z * z / den)
    half = z * ((p * (1 - p) / den + z * z / (4 * den * den)) ** 0.5) / (1 + z * z / den)
    return max(0.0, centre - half), min(1.0, centre + half)


# numerator/denominator count keys behind each proportion metric (see ``counts``)
FRACTIONS = {
    "report_precision": ("true_report", "predicted_report"),
    "report_recall": ("true_report", "should_report"),
    "escalate_recall_lenient": ("escalate_reaching_lawyer", "expected_escalate"),
    "escalate_recall_strict": ("escalate_exact", "expected_escalate"),
    "discovery_recall": ("actionable_captured", "actionable"),
}


def wilson_intervals(rows: list["Row"]) -> dict[str, dict[str, Any]]:
    c = counts(rows)
    out = {}
    for name, (n, d) in FRACTIONS.items():
        lo, hi = wilson(c[n], c[d])
        out[name] = {"num": c[n], "den": c[d], "wilson_low": lo, "wilson_high": hi}
    return out


# ---------------------------------------------------------------- bootstrap


@dataclass(frozen=True)
class Estimate:
    value: float | None
    ci_low: float | None
    ci_high: float | None
    n: int  # rows in the denominator population
    valid_resamples: int


def bootstrap(rows: list[Row], fn: Callable[[list[Row]], float | None], n: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> Estimate:
    """Percentile CI from resampling listings with replacement. Resamples where the metric is undefined
    (zero denominator) are skipped and counted out of ``valid_resamples``."""
    point = fn(rows)
    if not rows:
        return Estimate(point, None, None, 0, 0)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    vals = []
    for _ in range(n):
        v = fn([rows[i] for i in rng.choice(idx, size=len(rows), replace=True)])
        if v is not None:
            vals.append(v)
    if not vals:
        return Estimate(point, None, None, len(rows), 0)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), len(rows), len(vals))


def headline(rows: list[Row], n: int = 2000, seed: int = 0) -> dict[str, Any]:
    out = {name: asdict(bootstrap(rows, fn, n, seed)) for name, fn in HEADLINE.items()}
    for name, w in wilson_intervals(rows).items():
        out[name].update(w)
    return out


def breakdown(rows: list[Row], key: str, n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Headline metrics with CIs and counts for each value of ``key`` (``site`` or ``difficulty``)."""
    out = {}
    for value in sorted({getattr(r, key) for r in rows}):
        sub = [r for r in rows if getattr(r, key) == value]
        out[value] = {"counts": counts(sub), "metrics": headline(sub, n, seed)}
    return out


# ---------------------------------------------------------------- confusion matrices, PR curve


def confusion(rows: list[Row], truth: Callable[[Row], str], pred: Callable[[Row], str],
              truth_labels: list[str], pred_labels: list[str]) -> dict[str, Any]:
    m = {t: {p: 0 for p in pred_labels} for t in truth_labels}
    for r in rows:
        m[truth(r)][pred(r)] += 1
    return {"rows": truth_labels, "cols": pred_labels, "matrix": [[m[t][p] for p in pred_labels] for t in truth_labels]}


def classification(rows: list[Row], categories: list[str]) -> dict[str, Any]:
    """Category and action confusion on captured rows; end-to-end action confusion on all rows
    (uncaptured shown as its own column)."""
    cap = [r for r in rows if r.captured]
    per_cat = {}
    for c in categories:
        tp = sum(r.category == c and r.pred_category == c for r in cap)
        predicted = sum(r.pred_category == c for r in cap)
        actual = sum(r.category == c for r in cap)
        per_cat[c] = {"precision": _ratio(tp, predicted), "recall": _ratio(tp, actual), "support": actual}
    return {
        "category_confusion_captured": confusion(cap, lambda r: r.category, lambda r: r.pred_category,
                                                 categories, categories),
        "action_confusion_end_to_end": confusion(rows, lambda r: r.expected_action, lambda r: r.pred_action,
                                                 ACTIONS, ACTIONS + [NOT_CAPTURED]),
        "per_category_captured": per_cat,
    }


def pr_curve(rows: list[Row], act_threshold: float) -> dict[str, Any]:
    """should_report vs confidence (uncaptured listings score 0). Includes the act_threshold point and the
    operating point of the actual decisions (which also reflect rule elements and downgrades)."""
    y = np.array([r.should_report for r in rows], dtype=int)
    s = np.array([r.confidence for r in rows], dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return {"points": [], "average_precision": None, "note": "needs both classes"}
    prec, rec, thr = precision_recall_curve(y, s)
    points = [{"threshold": float(t), "precision": float(p), "recall": float(r)}
              for p, r, t in zip(prec[:-1], rec[:-1], thr, strict=True)]
    at = s >= act_threshold
    tp = int((at & (y == 1)).sum())
    return {
        "points": points,
        "average_precision": float(average_precision_score(y, s)),
        "act_threshold": {"threshold": act_threshold, "precision": _ratio(tp, int(at.sum())),
                          "recall": _ratio(tp, int(y.sum()))},
        "decisions": {"precision": report_precision(rows), "recall": report_recall(rows)},
    }
