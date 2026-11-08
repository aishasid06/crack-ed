"""Timing / token tracing (SPEC §7.5).

``run_span`` creates a Run row; ``task_span`` creates (or resumes) a Task row keyed by
``(run_id, dedup_key)``. Both set context variables so nested spans pick up their
parent and ``bh.llm`` attributes every LLM call to the current task and run.

DB writes run in a worker thread (``asyncio.to_thread``) so the event loop never blocks.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from bh.db import LLMCall, Run, RunStatus, Task, TaskStatus, get_engine, session_scope, utcnow
from bh.errors import HumanRequired
from bh.security import redact

_current_run: ContextVar[int | None] = ContextVar("bh_current_run", default=None)
_current_task: ContextVar[int | None] = ContextVar("bh_current_task", default=None)


def current_run_id() -> int | None:
    return _current_run.get()


def current_task_id() -> int | None:
    return _current_task.get()


@dataclass
class TaskHandle:
    id: int
    run_id: int
    type: str
    site: str | None
    dedup_key: str
    parent_id: int | None
    attempts: int
    already_done: bool = False  # a task with this dedup_key already succeeded (or is running) in this run
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHandle:
    id: int
    brand_id: int | None


# ---------------------------------------------------------------- sync DB helpers


def _start_task(
    engine: Engine,
    run_id: int,
    type_: str,
    site: str | None,
    dedup_key: str,
    parent_id: int | None,
    payload: dict[str, Any] | None,
) -> TaskHandle:
    for _ in range(2):  # second pass handles a concurrent insert of the same dedup_key
        try:
            with session_scope(engine) as s:
                task = s.exec(select(Task).where(Task.run_id == run_id, Task.dedup_key == dedup_key)).first()
                if task is not None and task.status in (TaskStatus.succeeded, TaskStatus.running):
                    assert task.id is not None
                    return TaskHandle(task.id, run_id, task.type, task.site, dedup_key, task.parent_id,
                                      task.attempts, already_done=True, payload=dict(task.payload))
                if task is None:
                    task = Task(run_id=run_id, type=type_, site=site, dedup_key=dedup_key, parent_id=parent_id,
                                payload=payload or {})
                    s.add(task)
                else:
                    if payload:
                        task.payload = payload
                    if task.id is not None:
                        _supersede_waiting_steps(s, run_id, task.id)
                task.status = TaskStatus.running
                task.attempts += 1
                task.started_at = utcnow()
                task.finished_at = None
                task.error = None
                task.waiting_url = None
                s.flush()
                assert task.id is not None
                return TaskHandle(task.id, run_id, task.type, task.site, dedup_key, task.parent_id,
                                  task.attempts, payload=dict(task.payload))
        except IntegrityError:
            continue
    raise RuntimeError(f"could not create task {dedup_key!r} for run {run_id}")


SUPERSEDED_NOTE = "superseded: parent task retried after human hand-off"


def _supersede_waiting_steps(s: Session, run_id: int, task_id: int) -> None:
    """A retried task's step spans (one-off dedup keys, never re-entered) that recorded the challenge would
    otherwise stay waiting_human for good. Mark them superseded: the retry replaced them; nothing failed."""
    frontier = [task_id]
    while frontier:
        children = list(s.exec(select(Task).where(Task.run_id == run_id, Task.parent_id.in_(frontier))))  # type: ignore[union-attr]
        for c in children:
            if c.status == TaskStatus.waiting_human:
                c.status, c.waiting_url = TaskStatus.superseded, None
                c.finished_at = c.finished_at or utcnow()
                c.error = (c.error + "; " if c.error else "") + SUPERSEDED_NOTE
        frontier = [c.id for c in children if c.id is not None]


_STATUS_PHRASE = {"superseded": "superseded by human hand-off", "waiting_human": "waiting for a human"}
_STATUS_ORDER = ["succeeded", "superseded", "running", "pending", "waiting_human", "failed"]


def describe_task_statuses(by_status: dict[str, int]) -> str:
    """Human-readable task status summary. Superseded steps are reported neutrally, never as errors,
    e.g. "1045 succeeded · 2 superseded by human hand-off"."""
    keys = sorted(by_status, key=lambda k: (_STATUS_ORDER.index(k) if k in _STATUS_ORDER else 99, k))
    return " · ".join(f"{by_status[k]} {_STATUS_PHRASE.get(k, k)}" for k in keys if by_status[k]) or "no tasks"


def _finish_task(engine: Engine, task_id: int, status: TaskStatus, error: str | None = None,
                 waiting_url: str | None = None, payload: dict[str, Any] | None = None) -> None:
    with session_scope(engine) as s:
        task = s.get(Task, task_id)
        if task is None:
            return
        task.status = status
        task.finished_at = utcnow()
        task.error = error
        task.waiting_url = waiting_url
        if payload is not None:
            task.payload = dict(payload)


def resume_task(task_id: int, engine: Engine | None = None) -> None:
    """Human finished the challenge: move waiting_human → pending so the worker picks it up again."""
    with session_scope(engine) as s:
        task = s.get(Task, task_id)
        if task is None:
            raise KeyError(f"task {task_id} not found")
        if task.status != TaskStatus.waiting_human:
            raise ValueError(f"task {task_id} is {task.status}, not waiting_human")
        task.status = TaskStatus.pending
        task.waiting_url = None


# Units of work (one listing captured, one search run, one notice drafted...). Stage and browser-step
# spans nest inside them, so counting every open span would overstate parallelism.
WORK_TYPES = frozenset({"discovery", "capture", "draft", "submit"})


def _max_parallelism(tasks: list[Task]) -> int:
    events: list[tuple[Any, int]] = []
    now = utcnow()
    for t in tasks:
        if t.started_at is None:
            continue
        events.append((t.started_at, 1))
        events.append((t.finished_at or now, -1))
    events.sort(key=lambda e: (e[0], e[1]))  # ends sort before starts at equal timestamps
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


def compute_run_totals(run_id: int, engine: Engine | None = None) -> dict[str, Any]:
    """Aggregate tokens by model, task counts by type/status, sites, max parallelism, wall time."""
    with Session(engine or get_engine()) as s:
        run = s.get(Run, run_id)
        if run is None:
            raise KeyError(f"run {run_id} not found")
        tasks = list(s.exec(select(Task).where(Task.run_id == run_id)))
        calls = list(s.exec(select(LLMCall).where(LLMCall.run_id == run_id)))

    by_type: dict[str, dict[str, int]] = {}
    by_status: dict[str, int] = {}
    for t in tasks:
        st = str(t.status)
        by_type.setdefault(t.type, {}).setdefault(st, 0)
        by_type[t.type][st] += 1
        by_status[st] = by_status.get(st, 0) + 1

    by_model: dict[str, dict[str, int]] = {}
    for c in calls:
        m = by_model.setdefault(c.model, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                          "latency_ms": 0, "errors": 0})
        m["calls"] += 1
        m["input_tokens"] += c.input_tokens
        m["output_tokens"] += c.output_tokens
        m["latency_ms"] += c.latency_ms
        m["errors"] += 1 if c.error else 0

    end = run.finished_at or utcnow()
    return {
        "tasks": {"total": len(tasks), "by_status": by_status, "by_type": by_type},
        "sites": sorted({t.site for t in tasks if t.site}),
        "llm": {
            "calls": len(calls),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "by_model": by_model,
        },
        "max_parallelism": _max_parallelism([t for t in tasks if t.type in WORK_TYPES]),
        "max_concurrent_spans": _max_parallelism(tasks),
        "wall_time_s": round((end - run.started_at).total_seconds(), 3),
    }


def run_tokens_used(run_id: int, engine: Engine | None = None) -> int:
    with Session(engine or get_engine()) as s:
        calls = s.exec(select(LLMCall).where(LLMCall.run_id == run_id))
        return sum(c.input_tokens + c.output_tokens for c in calls)


# ---------------------------------------------------------------- async spans


def _reopen_run(engine: Engine, run_id: int) -> int | None:
    """Mark a run running again and requeue tasks a crashed worker left ``running`` (their span never
    closed; left as is they would count as in progress and be skipped forever). Returns its brand_id."""
    with session_scope(engine) as s:
        run = s.get(Run, run_id)
        if run is None:
            raise KeyError(f"run {run_id} not found")
        run.status, run.finished_at = RunStatus.running, None
        for t in s.exec(select(Task).where(Task.run_id == run_id, Task.status == TaskStatus.running)):
            t.status, t.error = TaskStatus.pending, "interrupted; requeued on resume"
        return run.brand_id


@asynccontextmanager
async def run_span(brand_id: int | None, config_snapshot: dict[str, Any],
                   engine: Engine | None = None, resume_run_id: int | None = None) -> AsyncIterator[RunHandle]:
    """Trace one run. With ``resume_run_id`` the existing run is reopened instead of creating a new one:
    tasks keyed by the same dedup_key that already succeeded are then skipped by ``task_span``."""
    eng = engine or get_engine()

    def _create() -> int:
        with session_scope(eng) as s:
            run = Run(brand_id=brand_id, config_snapshot=config_snapshot)
            s.add(run)
            s.flush()
            assert run.id is not None
            return run.id

    if resume_run_id is not None:
        stored_brand = await asyncio.to_thread(_reopen_run, eng, resume_run_id)
        run_id, brand_id = resume_run_id, brand_id if brand_id is not None else stored_brand
    else:
        run_id = await asyncio.to_thread(_create)
    token = _current_run.set(run_id)
    status = RunStatus.failed
    try:
        yield RunHandle(run_id, brand_id)
        status = RunStatus.finished
    finally:
        _current_run.reset(token)

        def _close() -> None:
            with session_scope(eng) as s:
                run = s.get(Run, run_id)
                assert run is not None
                run.finished_at = utcnow()
                run.status = status
            totals = compute_run_totals(run_id, eng)
            with session_scope(eng) as s:
                run = s.get(Run, run_id)
                assert run is not None
                run.totals = totals

        await asyncio.shield(asyncio.to_thread(_close))


@asynccontextmanager
async def task_span(run_id: int | None, type: str, site: str | None, dedup_key: str, *,
                    parent_id: int | None = None, payload: dict[str, Any] | None = None,
                    engine: Engine | None = None) -> AsyncIterator[TaskHandle]:
    """Trace one unit of work. Callers must check ``handle.already_done`` and skip if set.

    Outcome mapping: normal exit → succeeded; ``HumanRequired`` → waiting_human (re-raised);
    any other exception (incl. cancellation) → failed with a redacted reason (re-raised).
    Changes to ``handle.payload`` made inside the span are persisted when it closes.
    """
    eng = engine or get_engine()
    rid = run_id if run_id is not None else current_run_id()
    if rid is None:
        raise RuntimeError("task_span needs a run_id (pass one or open a run_span first)")
    parent = parent_id if parent_id is not None else current_task_id()
    handle = await asyncio.to_thread(_start_task, eng, rid, type, site, dedup_key, parent, payload)

    run_token = _current_run.set(rid)
    task_token = _current_task.set(handle.id)
    try:
        yield handle
    except HumanRequired as e:
        await asyncio.shield(asyncio.to_thread(
            _finish_task, eng, handle.id, TaskStatus.waiting_human, redact(str(e)), e.url, handle.payload))
        raise
    except BaseException as e:
        if not handle.already_done:
            msg = redact(f"{type_name(e)}: {e}")[:2000]
            await asyncio.shield(asyncio.to_thread(_finish_task, eng, handle.id, TaskStatus.failed, msg,
                                                   None, handle.payload))
        raise
    else:
        if not handle.already_done:
            await asyncio.to_thread(_finish_task, eng, handle.id, TaskStatus.succeeded, None, None, handle.payload)
    finally:
        _current_task.reset(task_token)
        _current_run.reset(run_token)


def type_name(e: BaseException) -> str:
    return type(e).__name__
