from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.config import AppConfig, Secrets
from bh.db import LLMCall
from bh.errors import BudgetExceeded, LLMError, LLMRefusal, SecretLeakError
from bh.llm import FALLBACK_BETA, LLMClient
from bh.trace import run_span, task_span
from tests.conftest import TEST_PASS


class Card(BaseModel):
    title: str
    price: float


def _resp(text: str = "ok", *, model: str = "claude-haiku-5-5", stop: str = "end_turn",
          inp: int = 100, out: int = 20, parsed: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_input_tokens=5,
                              cache_creation_input_tokens=0),
        model=model, stop_reason=stop, parsed_output=parsed,
    )


class FakeMessages:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response, self.exc = response, exc
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def _do(self, kind: str, kw: dict[str, Any]) -> Any:
        self.calls.append((kind, kw))
        if self.exc:
            raise self.exc
        return self.response

    async def create(self, **kw: Any) -> Any:
        return await self._do("create", kw)

    async def parse(self, **kw: Any) -> Any:
        return await self._do("parse", kw)


def _client(cfg: AppConfig, secrets: Secrets, engine: Engine, msgs: FakeMessages) -> LLMClient:
    fake = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    return LLMClient(cfg=cfg, secrets=secrets, client=fake, engine=engine)


def _calls(engine: Engine) -> list[LLMCall]:
    with Session(engine) as s:
        return list(s.exec(select(LLMCall)))


async def test_small_call_recorded_against_task(cfg, secrets, engine) -> None:
    msgs = FakeMessages(_resp("hello"))
    llm = _client(cfg, secrets, engine, msgs)
    async with run_span(None, {}) as run:
        async with task_span(run.id, "extract", "echobazaar", "k") as task:
            r = await llm.complete(role="small", purpose="extract", messages=[{"role": "user", "content": "hi"}])
    assert r.text == "hello" and r.input_tokens == 100 and r.output_tokens == 20
    kind, kw = msgs.calls[0]
    assert kind == "create" and kw["model"] == "claude-haiku-5-5"
    assert kw["output_config"] == {"effort": "low"}
    assert "fallbacks" not in kw and "betas" not in kw and "thinking" not in kw
    (c,) = _calls(engine)
    assert (c.task_id, c.run_id, c.model, c.purpose) == (task.id, run.id, "claude-haiku-5-5", "extract")
    assert c.cache_read_input_tokens == 5 and c.latency_ms >= 0 and c.error is None


async def test_strong_uses_fallbacks_and_records_served_model(cfg, secrets, engine) -> None:
    msgs = FakeMessages(_resp(model="claude-opus-4-8"))  # pretend the fallback served it
    llm = _client(cfg, secrets, engine, msgs)
    r = await llm.complete(role="strong", purpose="classify", messages=[{"role": "user", "content": "x"}],
                           system="sys")
    _, kw = msgs.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert kw["system"] == "sys"
    assert r.model == "claude-opus-4-8" and r.requested_model == "claude-opus-5-5"
    (c,) = _calls(engine)
    assert c.model == "claude-opus-4-8" and c.requested_model == "claude-opus-5-5"


async def test_complete_json_parses(cfg, secrets, engine) -> None:
    msgs = FakeMessages(_resp('{"title":"t","price":1.5}', parsed=Card(title="t", price=1.5)))
    llm = _client(cfg, secrets, engine, msgs)
    r = await llm.complete_json(role="small", purpose="extract", schema=Card,
                                messages=[{"role": "user", "content": "x"}])
    assert r.parsed == Card(title="t", price=1.5)
    kind, kw = msgs.calls[0]
    assert kind == "parse" and kw["output_format"] is Card


async def test_complete_json_missing_parse_raises(cfg, secrets, engine) -> None:
    llm = _client(cfg, secrets, engine, FakeMessages(_resp("not json", parsed=None)))
    with pytest.raises(LLMError):
        await llm.complete_json(role="small", purpose="extract", schema=Card,
                                messages=[{"role": "user", "content": "x"}])
    assert len(_calls(engine)) == 1  # still logged


async def test_complete_json_truncated_raises(cfg, secrets, engine) -> None:
    llm = _client(cfg, secrets, engine, FakeMessages(_resp(stop="max_tokens", parsed=Card(title="t", price=1))))
    with pytest.raises(LLMError):
        await llm.complete_json(role="small", purpose="extract", schema=Card,
                                messages=[{"role": "user", "content": "x"}])


@pytest.mark.parametrize("where", ["system", "message", "nested"])
async def test_secret_never_sent(cfg, secrets, engine, where) -> None:
    msgs = FakeMessages(_resp())
    llm = _client(cfg, secrets, engine, msgs)
    system = f"pw {TEST_PASS}" if where == "system" else None
    if where == "nested":
        content: Any = [{"type": "text", "text": f"field value {TEST_PASS}"}]
    else:
        content = f"login with {TEST_PASS}" if where == "message" else "clean"
    with pytest.raises(SecretLeakError):
        await llm.complete(role="small", purpose="x", system=system, messages=[{"role": "user", "content": content}])
    assert msgs.calls == [] and _calls(engine) == []


async def test_image_blocks_allowed(cfg, secrets, engine) -> None:
    msgs = FakeMessages(_resp())
    llm = _client(cfg, secrets, engine, msgs)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K"}},
               {"type": "text", "text": "compare"}]
    await llm.complete(role="strong", purpose="vision", messages=[{"role": "user", "content": content}])
    assert len(msgs.calls) == 1


async def test_budget_exceeded(cfg, secrets, engine) -> None:
    budget = cfg.budgets.max_llm_tokens_per_run
    msgs = FakeMessages(_resp(inp=budget, out=1))
    llm = _client(cfg, secrets, engine, msgs)
    async with run_span(None, {}):
        await llm.complete(role="small", purpose="a", messages=[{"role": "user", "content": "x"}])
        with pytest.raises(BudgetExceeded):
            await llm.complete(role="small", purpose="b", messages=[{"role": "user", "content": "x"}])
    assert len(msgs.calls) == 1


async def test_refusal_recorded_and_raised(cfg, secrets, engine) -> None:
    llm = _client(cfg, secrets, engine, FakeMessages(_resp(stop="refusal")))
    with pytest.raises(LLMRefusal):
        await llm.complete(role="strong", purpose="draft", messages=[{"role": "user", "content": "x"}])
    (c,) = _calls(engine)
    assert c.stop_reason == "refusal" and c.error == "refusal" and c.input_tokens == 100


async def test_api_error_recorded_redacted(cfg, secrets, engine) -> None:
    llm = _client(cfg, secrets, engine, FakeMessages(exc=ConnectionError(f"x-api-key: {secrets.anthropic_api_key.get_secret_value()}")))
    with pytest.raises(ConnectionError):
        await llm.complete(role="small", purpose="x", messages=[{"role": "user", "content": "x"}])
    (c,) = _calls(engine)
    assert c.error and "sk-ant-test" not in c.error and c.input_tokens == 0


async def test_run_totals_aggregate_tokens_by_model(cfg, secrets, engine) -> None:
    from bh.db import Run

    small = _client(cfg, secrets, engine, FakeMessages(_resp(model="claude-haiku-5-5", inp=10, out=2)))
    strong = _client(cfg, secrets, engine, FakeMessages(_resp(model="claude-opus-5-5", inp=50, out=7)))
    async with run_span(None, {}) as run:
        async with task_span(run.id, "classify", None, "c1"):
            await small.complete(role="small", purpose="a", messages=[{"role": "user", "content": "x"}])
            await small.complete(role="small", purpose="a", messages=[{"role": "user", "content": "x"}])
            await strong.complete(role="strong", purpose="b", messages=[{"role": "user", "content": "x"}])
    with Session(engine) as s:
        llm = s.get(Run, run.id).totals["llm"]
    assert llm["input_tokens"] == 70 and llm["output_tokens"] == 11 and llm["calls"] == 3
    assert llm["by_model"]["claude-haiku-5-5"]["calls"] == 2
    assert llm["by_model"]["claude-opus-5-5"]["input_tokens"] == 50
