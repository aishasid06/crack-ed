"""Read-side queries for the UI. Streamlit only reads the DB and writes approval decisions (via
``bh.enforce.approvals`` and ``bh.trace.resume_task``); nothing here mutates state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from bh.config import PROJECT_ROOT
from bh.db import Brand, Case, CaseStatus, Features, Listing, LLMCall, Product, Run, Task, TaskStatus, TradeMark, \
    get_engine

QUEUE_STATUSES = (CaseStatus.awaiting_approval, CaseStatus.drafted)


def _eng(engine: Engine | None) -> Engine:
    return engine or get_engine()


def resolve_path(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


@dataclass
class CaseBundle:
    case: Case
    listing: Listing
    features: Features | None
    product: Product | None

    @property
    def listing_image(self) -> Path | None:
        for p in self.listing.local_image_paths:
            if (r := resolve_path(p)) is not None:
                return r
        return None

    @property
    def gold_image(self) -> Path | None:
        for p in (self.product.image_paths if self.product else []):
            if (r := resolve_path(p)) is not None:
                return r
        return None

    @property
    def screenshot(self) -> Path | None:
        return resolve_path(self.listing.screenshot_path)


def _bundles(s: Session, cases: list[Case]) -> list[CaseBundle]:
    if not cases:
        return []
    lids = {c.listing_id for c in cases}
    listings = {l.id: l for l in s.exec(select(Listing).where(col(Listing.id).in_(lids)))}
    feats = {f.listing_id: f for f in s.exec(select(Features).where(col(Features.listing_id).in_(lids)))}
    pids = {f.matched_product_id for f in feats.values() if f.matched_product_id}
    products = {p.id: p for p in s.exec(select(Product).where(col(Product.id).in_(pids)))} if pids else {}
    out = []
    for c in cases:
        f = feats.get(c.listing_id)
        out.append(CaseBundle(c, listings[c.listing_id], f, products.get(f.matched_product_id) if f else None))
    return out


def _latest_per_listing(cases: list[Case]) -> list[Case]:
    """A listing re-classified in a later run gets a new case; only the newest one is actionable."""
    best: dict[int, Case] = {}
    for c in cases:
        if c.listing_id not in best or (c.id or 0) > (best[c.listing_id].id or 0):
            best[c.listing_id] = c
    return list(best.values())


def review_queue(engine: Engine | None = None) -> list[CaseBundle]:
    with Session(_eng(engine)) as s:
        all_cases = list(s.exec(select(Case)))
        newest = {c.id for c in _latest_per_listing(all_cases)}
        cases = [c for c in all_cases if c.id in newest and c.status in QUEUE_STATUSES]
        cases.sort(key=lambda c: (-c.expected_value, c.id or 0))
        return _bundles(s, cases)


def cases_with_status(status: CaseStatus, engine: Engine | None = None) -> list[CaseBundle]:
    with Session(_eng(engine)) as s:
        cases = list(s.exec(select(Case).where(Case.status == status).order_by(col(Case.id))))
        return _bundles(s, cases)


def awaiting_final_confirm(engine: Engine | None = None) -> list[CaseBundle]:
    return [b for b in cases_with_status(CaseStatus.approved, engine)
            if b.case.review_verified_at is not None and not b.case.final_confirmed]


def case_bundle(case_id: int, engine: Engine | None = None) -> CaseBundle | None:
    with Session(_eng(engine)) as s:
        c = s.get(Case, case_id)
        return _bundles(s, [c])[0] if c else None


def all_cases(engine: Engine | None = None) -> list[CaseBundle]:
    with Session(_eng(engine)) as s:
        return _bundles(s, list(s.exec(select(Case).order_by(col(Case.id).desc()))))


def waiting_tasks(engine: Engine | None = None) -> list[Task]:
    with Session(_eng(engine)) as s:
        return list(s.exec(select(Task).where(Task.status == TaskStatus.waiting_human)
                           .order_by(col(Task.finished_at).desc())))


def brands(engine: Engine | None = None) -> list[Brand]:
    with Session(_eng(engine)) as s:
        return list(s.exec(select(Brand)))


def trademarks(engine: Engine | None = None) -> list[TradeMark]:
    with Session(_eng(engine)) as s:
        return list(s.exec(select(TradeMark)))


# ---------------------------------------------------------------- runs / trace


def stage_of(run: Run) -> str:
    snap = run.config_snapshot or {}
    if "stage" in snap:
        return str(snap["stage"])
    return "pipeline" if snap.get("pipeline") else "discover"


def runs(limit: int = 50, since: datetime | None = None, engine: Engine | None = None) -> list[Run]:
    with Session(_eng(engine)) as s:
        q = select(Run).order_by(col(Run.id).desc()).limit(limit)
        rows = list(s.exec(q))
    if since is not None:
        rows = [r for r in rows if _aware(r.started_at) >= _aware(since)]
    return rows


def _aware(dt: datetime) -> datetime:
    from datetime import UTC

    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def tasks_for_runs(run_ids: list[int], engine: Engine | None = None) -> list[Task]:
    if not run_ids:
        return []
    with Session(_eng(engine)) as s:
        return list(s.exec(select(Task).where(col(Task.run_id).in_(run_ids)).order_by(col(Task.id))))


def llm_calls_for_runs(run_ids: list[int], engine: Engine | None = None) -> list[LLMCall]:
    if not run_ids:
        return []
    with Session(_eng(engine)) as s:
        return list(s.exec(select(LLMCall).where(col(LLMCall.run_id).in_(run_ids)).order_by(col(LLMCall.id))))


def task_status_counts(tasks: list[Task]) -> dict[str, int]:
    out = {str(s): 0 for s in TaskStatus}
    for t in tasks:
        out[str(t.status)] += 1
    return out


def tokens_by_model(calls: list[LLMCall]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for c in calls:
        m = by.setdefault(c.model, {"model": c.model, "calls": 0, "input_tokens": 0, "output_tokens": 0,
                                    "latency_ms": 0})
        m["calls"] += 1
        m["input_tokens"] += c.input_tokens
        m["output_tokens"] += c.output_tokens
        m["latency_ms"] += c.latency_ms
    return sorted(by.values(), key=lambda m: -(m["input_tokens"] + m["output_tokens"]))


def task_tree(tasks: list[Task]) -> list[tuple[int, Task]]:
    """Depth-first (depth, task) order: parents before children, siblings by start time."""
    ids = {t.id for t in tasks}
    children: dict[int | None, list[Task]] = {}
    for t in tasks:
        parent = t.parent_id if t.parent_id in ids else None
        children.setdefault(parent, []).append(t)
    for kids in children.values():
        kids.sort(key=lambda t: t.id or 0)  # ids are assigned in start order
    out: list[tuple[int, Task]] = []

    def walk(pid: int | None, depth: int) -> None:
        for t in children.get(pid, []):
            out.append((depth, t))
            walk(t.id, depth + 1)

    walk(None, 0)
    return out


def case_counts(engine: Engine | None = None) -> dict[str, int]:
    with Session(_eng(engine)) as s:
        cases = _latest_per_listing(list(s.exec(select(Case))))
    out: dict[str, int] = {}
    for c in cases:
        out[str(c.status)] = out.get(str(c.status), 0) + 1
    return out
