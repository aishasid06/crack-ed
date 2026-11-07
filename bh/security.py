"""Secret hygiene: redaction for logs/DB and a fail-closed guard for LLM prompts."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from bh.errors import SecretLeakError

_MIN_SECRET_LEN = 4

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-[REDACTED]"),
    (re.compile(r"(?i)((?:set-)?cookie\s*[:=]\s*)[^\r\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)[^\r\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-api-key\s*[:=]\s*)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:password|passwd|pass)\s*[:=]\s*)\S+"), r"\1[REDACTED]"),
]


def _known_secrets() -> list[str]:
    from bh.config import get_secrets

    return [s for s in get_secrets().secret_values() if len(s) >= _MIN_SECRET_LEN]


def redact(text: str | None, secrets: Iterable[str] | None = None) -> str:
    """Remove known secret values and common credential patterns from text."""
    if not text:
        return text or ""
    out = text
    for s in sorted(secrets if secrets is not None else _known_secrets(), key=len, reverse=True):
        if len(s) >= _MIN_SECRET_LEN:
            out = out.replace(s, "[REDACTED]")
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return out


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # base64 image payloads cannot carry typed secrets; skip to keep the scan cheap
            if k == "data" and isinstance(v, str) and obj.get("type") == "base64":
                continue
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def assert_no_secrets(*payloads: Any, secrets: Iterable[str] | None = None) -> None:
    """Raise SecretLeakError if any known secret appears in the payloads (fail closed)."""
    values = [s for s in (secrets if secrets is not None else _known_secrets()) if len(s) >= _MIN_SECRET_LEN]
    if not values:
        return
    for payload in payloads:
        for text in _iter_strings(payload):
            for s in values:
                if s in text:
                    raise SecretLeakError("refusing to send a prompt containing a configured secret")
