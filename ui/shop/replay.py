"""Replay a saved shopper search instead of running the pipeline again.

Every real search is already saved in the shopper database: its run, every traced step with its timing,
the listings it captured and the verdicts. When a shopper repeats a search (same words, same photo) that an
earlier real search finished, the app plays that run back: a new run is created and the saved steps are
re-created on their original timeline, compressed to about ``shopper.replay_seconds``, and the saved verdicts
are attached when the steps finish. The shopper UI reads the database exactly as for a live search.

Kept truthful underneath: the new run is marked ``replay_of`` in its snapshot, and the saved run's LLM calls
are not copied, so token and timing figures (trace, run totals, evaluation) only ever count real work.

    python -m ui.shop.replay list
    python -m ui.shop.replay --search-id ID --source RUN_ID [--seconds 30]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from bh.db import Case, Run, RunStatus, Task, TaskStatus, session_scope, utcnow


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _sha(photo: Path | None) -> str | None:
    import hashlib

    return hashlib.sha256(photo.read_bytes()).hexdigest() if photo is not None and photo.exists() else None


def is_saved_search(run: Run) -> bool:
    snap = run.config_snapshot or {}
    return snap.get("stage") == "shopper" and run.status == RunStatus.finished and not snap.get("replay_of")


def find_source(text: str | None, photo: Path | None, eng: Engine) -> int | None:
    """The newest finished real search with the same words and the same photo (or no photo on both)."""
    want_text, want_photo = norm(text), _sha(photo)
    with Session(eng) as s:
        for r in s.exec(select(Run).order_by(col(Run.id).desc()).limit(500)):
            snap = r.config_snapshot or {}
            if not is_saved_search(r) or norm(snap.get("query")) != want_text:
                continue
            if want_photo is None and not snap.get("has_photo"):
                return r.id
            if want_photo is not None and snap.get("photo_sha256") == want_photo:
                return r.id
    return None


def saved_searches(eng: Engine) -> list[dict[str, Any]]:
    with Session(eng) as s:
        runs = [r for r in s.exec(select(Run).order_by(col(Run.id).desc())) if is_saved_search(r)]
        out = []
        for r in runs:
            n = len(s.exec(select(Case.id).where(Case.run_id == r.id)).all())
            snap = r.config_snapshot or {}
            secs = (r.finished_at - r.started_at).total_seconds() if r.finished_at else None
            out.append({"run": r.id, "query": snap.get("query"), "photo": bool(snap.get("has_photo")),
                        "sites": snap.get("sites"), "listings": n, "seconds": round(secs or 0)})
    return out


def _plan(tasks: list[Task], run_start: Any, seconds: float) -> tuple[list[tuple[float, str, Task]], float]:
    """Start/finish events at offsets from the run start, scaled so the whole run lasts about ``seconds``."""
    def off(dt: Any) -> float:
        a = dt if dt.tzinfo else dt.replace(tzinfo=run_start.tzinfo)
        return max(0.0, (a - run_start).total_seconds())

    ends = [off(t.finished_at) for t in tasks if t.finished_at] + [off(t.started_at) for t in tasks if t.started_at]
    total = max(ends, default=0.0)
    scale = min(1.0, seconds / total) if total > 0 else 1.0
    events = []
    for t in tasks:
        if t.started_at is None:
            continue
        events.append((off(t.started_at) * scale, "start", t))
        events.append(((off(t.finished_at) if t.finished_at else off(t.started_at)) * scale, "finish", t))
    events.sort(key=lambda e: (e[0], e[1] == "finish"))  # a step's start always comes before its finish
    return events, total * scale


def replay(search_id: str, source_run_id: int, eng: Engine, seconds: float = 30.0, tick: float = 0.2) -> int:
    """Re-create ``source_run_id`` as a new run for ``search_id``, step by step in (compressed) real time."""
    with Session(eng) as s:
        src = s.get(Run, source_run_id)
        if src is None or not is_saved_search(src):
            raise ValueError(f"run {source_run_id} is not a finished shopper search")
        tasks = list(s.exec(select(Task).where(Task.run_id == source_run_id).order_by(col(Task.id))))
        cases = list(s.exec(select(Case).where(Case.run_id == source_run_id)))
        src_snapshot, src_start, brand_id = dict(src.config_snapshot or {}), src.started_at, src.brand_id
    events, _ = _plan(tasks, src_start, seconds)

    start = utcnow()
    with session_scope(eng) as s:
        run = Run(brand_id=brand_id, status=RunStatus.running, started_at=start,
                  config_snapshot={**src_snapshot, "search_id": search_id, "replay_of": source_run_id})
        s.add(run)
        s.flush()
        rid = run.id
    new_id: dict[int, int] = {}
    t0 = time.monotonic()
    i = 0
    while i < len(events):
        due = time.monotonic() - t0
        batch = []
        while i < len(events) and events[i][0] <= due:
            batch.append(events[i])
            i += 1
        if batch:
            with session_scope(eng) as s:
                for _, kind, t in batch:
                    now = utcnow()
                    if kind == "start":
                        row = Task(run_id=rid, type=t.type, site=t.site, dedup_key=t.dedup_key,
                                   status=TaskStatus.running, parent_id=new_id.get(t.parent_id or -1),
                                   attempts=t.attempts, started_at=now, payload=dict(t.payload or {}))
                        s.add(row)
                        s.flush()
                        new_id[t.id] = row.id  # type: ignore[index,assignment]
                    elif t.id in new_id:
                        row = s.get(Task, new_id[t.id])
                        if row is not None:
                            row.status, row.finished_at, row.error = t.status, now, t.error
                            row.waiting_url = t.waiting_url
        if i < len(events):
            time.sleep(min(tick, max(0.0, events[i][0] - (time.monotonic() - t0))))

    from bh.trace import compute_run_totals

    with session_scope(eng) as s:
        by_new = {v: k for k, v in new_id.items()}
        original = {t.id: t for t in tasks}
        for row in s.exec(select(Task).where(Task.run_id == rid, Task.status == TaskStatus.running)):
            src_t = original.get(by_new.get(row.id))  # type: ignore[arg-type]
            row.status, row.finished_at = (src_t.status if src_t else TaskStatus.succeeded), utcnow()
        for c in cases:
            data = c.model_dump(exclude={"id", "run_id"})
            s.add(Case(**data, run_id=rid))
        r = s.get(Run, rid)
        assert r is not None
        r.finished_at, r.status = start + timedelta(seconds=time.monotonic() - t0), RunStatus.finished
    totals = compute_run_totals(rid, eng)
    with session_scope(eng) as s:
        r = s.get(Run, rid)
        assert r is not None
        r.totals = {**totals, "replay_of": source_run_id}
    return rid  # type: ignore[return-value]


def main() -> int:
    from bh.config import get_config
    from bh.shopper import use_shopper_db

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["list"], help="list the saved searches that can be replayed")
    ap.add_argument("--search-id")
    ap.add_argument("--source", type=int)
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()
    cfg = get_config()
    use_shopper_db(cfg)
    from bh.db import get_engine

    eng = get_engine()
    if args.command == "list":
        for row in saved_searches(eng):
            print(row)
        return 0
    if not args.search_id or args.source is None:
        ap.error("give --search-id and --source, or the list command")
    try:
        rid = replay(args.search_id, args.source, eng, args.seconds or cfg.shopper.replay_seconds)
        print(f"[bh] search {args.search_id} finished as run {rid} (replay of run {args.source})", flush=True)
        code = 0
    except Exception as e:  # noqa: BLE001
        print(f"[bh] replay failed: {e}", flush=True)
        code = 1
    print(f"[bh] exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
