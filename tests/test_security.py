from __future__ import annotations

import pytest

from bh.errors import SecretLeakError
from bh.security import assert_no_secrets, redact
from tests.conftest import TEST_KEY, TEST_PASS


@pytest.mark.parametrize("raw", [
    f"typed {TEST_PASS} into field",
    "Cookie: session=abcdef; other=1",
    "Set-Cookie: sid=zzz; HttpOnly",
    "Authorization: Bearer abc.def",
    "password=letmein",
    "key sk-ant-api03-ABCDEFGHIJKLMNOP",
])
def test_redact(raw: str) -> None:
    out = redact(raw)
    assert "[REDACTED]" in out
    for leaked in (TEST_PASS, "abcdef", "zzz", "abc.def", "letmein", "ABCDEFGHIJKLMNOP"):
        assert leaked not in out


def test_redact_leaves_clean_text() -> None:
    assert redact("price £6.99 = 25% of RRP") == "price £6.99 = 25% of RRP"


def test_assert_no_secrets_detects_in_nested_structures() -> None:
    with pytest.raises(SecretLeakError):
        assert_no_secrets({"messages": [{"content": [{"type": "text", "text": f"k={TEST_KEY}"}]}]})
    assert_no_secrets({"messages": [{"content": "clean"}]})


def test_assert_no_secrets_noop_without_secrets() -> None:
    assert_no_secrets("anything", secrets=[])
