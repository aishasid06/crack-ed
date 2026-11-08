from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.db import Run, RunStatus, Task, TaskStatus
from bh.errors import HumanRequired
from bh.trace import current_task_id, resume_task, run_span, task_span
from tests.conftest import TEST_PASS


def _tasks(engine: Engine) -> dict[str, Task]:
    with Session(engine) as s:
        return {t.dedup_key: t for t in s.exec(select(Task))}


async def test_task_success_and_nesting(engine: Engine) -> None:
    async with run_span(None, {"k": 1}) as run:
        async with task_span(run.id, "discovery", "mockbay", "mockbay:q1") as parent:
            assert current_task_id() == parent.id
            async with task_span(None, "capture", "mockbay", "mockbay:item1") as child:
                assert child.parent_id == parent.id
            assert current_task_id() == parent.id
    assert current_task_id() is None
    t = _tasks(engine)
    assert t["mockbay:q1"].status == TaskStatus.succeeded
    assert t["mockbay:item1"].parent_id == t["mockbay:q1"].id
    assert t["mockbay:q1"].started_at and t["mockbay:q1"].finished_at


async def test_failure_is_recorded_and_redacted(engine: Engine) -> None:
    async with run_span(None, {}) as run:
        with pytest.raises(ValueError):
            async with task_span(run.id, "draft", "portal", "draft:1"):
                raise ValueError(f"login failed with password={TEST_PASS} Cookie: session=abc123")
    task = _tasks(engine)["draft:1"]
    assert task.status == TaskStatus.failed
    assert TEST_PASS not in (task.error or "") and "abc123" not in (task.error or "")
    assert "ValueError" in task.error


async def test_human_required_pauses_then_resumes(engine: Engine) -> None:
    async with run_span(None, {}) as run:
        with pytest.raises(HumanRequired):
            async with task_span(run.id, "draft", "portal", "draft:2"):
                raise HumanRequired("challenge page", url="http://127.0.0.1:8000/portal/verify")
        task = _tasks(engine)["draft:2"]
        assert task.status == TaskStatus.waiting_human
        assert task.waiting_url.endswith("/portal/verify")

        resume_task(task.id)
        assert _tasks(engine)["draft:2"].status == TaskStatus.pending
        async with task_span(run.id, "draft", "portal", "draft:2") as again:
            assert not again.already_done and again.attempts == 2
    assert _tasks(engine)["draft:2"].status == TaskStatus.succeeded


async def test_resume_rejects_non_waiting(engine: Engine) -> None:
    async with run_span(None, {}) as run:
        async with task_span(run.id, "x", None, "x:1") as h:
            pass
    with pytest.raises(ValueError):
        resume_task(h.id)


async def test_dedup_marks_already_done(engine: Engine) -> None:
    async with run_span(None, {}) as run:
        async with task_span(run.id, "capture", "mockbay", "url:1") as first:
            assert not first.already_done
        async with task_span(run.id, "capture", "mockbay", "url:1") as second:
            assert second.already_done and second.id == first.id
    assert len(_tasks(engine)) == 1


async def test_concurrent_duplicate_is_deduped(engine: Engine) -> None:
    async with run_span(None, {}) as run:
        async def work() -> bool:
            async with task_span(run.id, "capture", "mockbay", "url:dup") as h:
                await asyncio.sleep(0.05)
                return h.already_done

        flags = await asyncio.gather(work(), work(), work())
    assert sorted(flags) == [False, True, True]
    assert _tasks(engine)["url:dup"].attempts == 1


async def test_run_totals(engine: Engine) -> None:
    async with run_span(None, {"brand": "vaelo"}) as run:
        async def work(i: int, site: str) -> None:
            async with task_span(run.id, "discovery", site, f"{site}:{i}"):
                await asyncio.sleep(0.1)

        await asyncio.gather(*(work(i, "mockbay") for i in range(3)), work(0, "tradeloft"))
        with pytest.raises(RuntimeError):
            async with task_span(run.id, "capture", "mockbay", "bad"):
                raise RuntimeError("x")

    with Session(engine) as s:
        r = s.get(Run, run.id)
    assert r.status == RunStatus.finished and r.finished_at is not None
    totals = r.totals
    assert totals["tasks"]["total"] == 5
    assert totals["tasks"]["by_type"]["discovery"] == {"succeeded": 4}
    assert totals["tasks"]["by_status"]["failed"] == 1
    assert totals["sites"] == ["mockbay", "tradeloft"]
    assert totals["max_parallelism"] == 4
    assert totals["wall_time_s"] >= 0.1


async def test_run_marked_failed_on_exception(engine: Engine) -> None:
    with pytest.raises(KeyError):
        async with run_span(None, {}) as run:
            raise KeyError("x")
    with Session(engine) as s:
        assert s.get(Run, run.id).status == RunStatus.failed


async def test_task_span_requires_run(engine: Engine) -> None:
    with pytest.raises(RuntimeError):
        async with task_span(None, "x", None, "k"):
            pass


async def test_handoff_steps_are_superseded_not_failed(engine: Engine) -> None:
    from bh.trace import describe_task_statuses

    async with run_span(None, {}) as run:
        with pytest.raises(HumanRequired):
            async with task_span(run.id, "draft", "portal", "draft:case:1") as parent:
                async with task_span(None, "browser:login", "portal", "login:1:portal"):
                    raise HumanRequired("challenge", url="http://x/portal/verify")
        resume_task(parent.id)
        async with task_span(run.id, "draft", "portal", "draft:case:1"):  # retry after the human verified
            async with task_span(None, "browser:login", "portal", "login:2:portal"):
                pass
    t = _tasks(engine)
    assert t["login:1:portal"].status == TaskStatus.superseded and t["login:1:portal"].finished_at
    assert "superseded" in t["login:1:portal"].error and t["draft:case:1"].status == TaskStatus.succeeded
    with Session(engine) as s:
        by_status = s.get(Run, run.id).totals["tasks"]["by_status"]
    assert by_status == {"succeeded": 2, "superseded": 1}
    assert describe_task_statuses(by_status) == "2 succeeded · 1 superseded by human hand-off"
    assert describe_task_statuses({"failed": 1, "succeeded": 3}) == "3 succeeded · 1 failed"
