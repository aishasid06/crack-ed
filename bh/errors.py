"""Shared exception types (kept dependency-free so any layer can import them)."""

from __future__ import annotations


class BountyHunterError(Exception):
    """Base class for project errors."""


class HumanRequired(BountyHunterError):
    """A challenge page (or other human-only step) was detected. The task pauses as
    ``waiting_human``; the agent must never attempt to solve it."""

    def __init__(self, message: str, url: str | None = None) -> None:
        super().__init__(message)
        self.url = url


class ReviewMismatch(HumanRequired):
    """The portal review page differs from the approved case; submission is blocked until a human acts."""


class BudgetExceeded(BountyHunterError):
    """The run's LLM token budget (config.budgets.max_llm_tokens_per_run) is used up."""


class SecretLeakError(BountyHunterError):
    """A secret value was found in content about to be sent to an LLM."""


class LLMError(BountyHunterError):
    """LLM call returned an unusable response (refusal, truncation, schema failure)."""

    def __init__(self, message: str, stop_reason: str | None = None) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason


class LLMRefusal(LLMError):
    """stop_reason == 'refusal' (after any server-side fallback)."""


class SelectorError(BountyHunterError):
    """A site adapter's selectors did not match the page (layout changed or wrong page)."""


class ExtractionFailed(BountyHunterError):
    """Adapter retries and the generic fallback extractor both failed."""


class LoginFailed(BountyHunterError):
    """Credentials were rejected or the login form was not found. Never includes the password."""
