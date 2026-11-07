"""Anthropic wrapper that records every call (tokens, model, latency) to ``LLMCall``.

Guarantees enforced here, not by callers:
- Prompts are scanned for configured secrets before sending (fail closed).
- The run's token budget is checked before each call.
- Every call — successful or not — produces an ``LLMCall`` row attributed to the
  current task/run from ``bh.trace`` context.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from bh.config import AppConfig, ModelRoleConfig, Secrets, get_config, get_secrets
from bh.db import LLMCall, get_engine, session_scope
from bh.errors import BudgetExceeded, LLMError, LLMRefusal
from bh.security import assert_no_secrets, redact
from bh.trace import current_run_id, current_task_id, run_tokens_used

Role = Literal["small", "strong"]
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class LLMResult(Generic[T]):
    text: str
    model: str
    requested_model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    stop_reason: str | None
    call_id: int | None
    parsed: T | None = None


class LLMClient:
    def __init__(self, cfg: AppConfig | None = None, secrets: Secrets | None = None,
                 client: Any | None = None, engine: Engine | None = None) -> None:
        self.cfg = cfg or get_config()
        self.secrets = secrets or get_secrets()
        self._client = client
        self._engine = engine

    @property
    def client(self) -> Any:
        if self._client is None:
            key = self.secrets.anthropic_api_key
            # None → SDK resolves ANTHROPIC_API_KEY / auth token / `ant auth login` profile.
            self._client = AsyncAnthropic(
                api_key=key.get_secret_value() if key else None,
                timeout=self.cfg.llm.timeout_s,
                max_retries=self.cfg.llm.max_retries,
            )
        return self._client

    def role(self, role: Role) -> ModelRoleConfig:
        return self.cfg.llm.small if role == "small" else self.cfg.llm.strong

    # ------------------------------------------------------------ public API

    async def complete(self, *, role: Role, purpose: str, messages: list[dict[str, Any]],
                       system: str | None = None, max_tokens: int | None = None) -> LLMResult[Any]:
        """Free-text completion."""
        return await self._call(role, purpose, messages, system, max_tokens, None)

    async def complete_json(self, *, role: Role, purpose: str, messages: list[dict[str, Any]],
                            schema: type[T], system: str | None = None,
                            max_tokens: int | None = None) -> LLMResult[T]:
        """Structured output validated against a Pydantic schema (``result.parsed``)."""
        result = await self._call(role, purpose, messages, system, max_tokens, schema)
        if result.parsed is None:
            raise LLMError(f"{purpose}: no parsed output (stop_reason={result.stop_reason})",
                           result.stop_reason)
        return result

    # ------------------------------------------------------------ internals

    async def _call(self, role: Role, purpose: str, messages: list[dict[str, Any]], system: str | None,
                    max_tokens: int | None, schema: type[BaseModel] | None) -> LLMResult[Any]:
        rc = self.role(role)
        assert_no_secrets(system, messages, secrets=self.secrets.secret_values())

        run_id, task_id = current_run_id(), current_task_id()
        if run_id is not None:
            used = await asyncio.to_thread(run_tokens_used, run_id, self._engine)
            budget = self.cfg.budgets.max_llm_tokens_per_run
            if used >= budget:
                raise BudgetExceeded(f"run {run_id} used {used} tokens (budget {budget})")

        kwargs: dict[str, Any] = {
            "model": rc.model,
            "max_tokens": max_tokens or rc.max_tokens,
            "messages": messages,
            "output_config": {"effort": rc.effort},
        }
        if system:
            kwargs["system"] = system
        if rc.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        if schema is not None:
            kwargs["output_format"] = schema

        api = self.client.beta.messages
        t0 = time.perf_counter()
        try:
            resp = await (api.parse(**kwargs) if schema is not None else api.create(**kwargs))
        except Exception as e:
            latency = int((time.perf_counter() - t0) * 1000)
            await asyncio.to_thread(self._record, LLMCall(
                task_id=task_id, run_id=run_id, model=rc.model, requested_model=rc.model, purpose=purpose,
                latency_ms=latency, error=redact(f"{type(e).__name__}: {e}")[:2000],
            ))
            raise
        latency = int((time.perf_counter() - t0) * 1000)

        usage = resp.usage
        served = getattr(resp, "model", None) or rc.model
        call = LLMCall(
            task_id=task_id, run_id=run_id, model=served, requested_model=rc.model, purpose=purpose,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            latency_ms=latency, stop_reason=resp.stop_reason,
        )
        if resp.stop_reason == "refusal":
            call.error = "refusal"
        call_id = await asyncio.to_thread(self._record, call)

        if resp.stop_reason == "refusal":
            raise LLMRefusal(f"{purpose}: model declined the request", "refusal")

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        parsed = getattr(resp, "parsed_output", None) if schema is not None else None
        if schema is not None and resp.stop_reason == "max_tokens":
            raise LLMError(f"{purpose}: structured output truncated at max_tokens", "max_tokens")
        return LLMResult(text=text, model=served, requested_model=rc.model,
                         input_tokens=call.input_tokens, output_tokens=call.output_tokens,
                         latency_ms=latency, stop_reason=resp.stop_reason, call_id=call_id, parsed=parsed)

    def _record(self, call: LLMCall) -> int | None:
        with session_scope(self._engine or get_engine()) as s:
            s.add(call)
            s.flush()
            return call.id
