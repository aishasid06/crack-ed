from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy.engine import Engine

from bh.config import AppConfig, Secrets, load_config
from bh.db import init_db, make_engine, set_engine

TEST_PASS = "hunter2-portal-pass"
TEST_KEY = "sk-ant-test-0123456789abcdef"


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(eng)
    set_engine(eng)
    yield eng
    set_engine(None)
    eng.dispose()


@pytest.fixture
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def secrets() -> Secrets:
    return Secrets(
        mock_portal_user=SecretStr("lawyer@vaelo.example"),
        mock_portal_pass=SecretStr(TEST_PASS),
        anthropic_api_key=SecretStr(TEST_KEY),
    )


@pytest.fixture(autouse=True)
def _isolated_secrets(monkeypatch: pytest.MonkeyPatch, secrets: Secrets) -> None:
    """Make redaction use test secrets, never a developer's real .env."""
    import bh.security

    monkeypatch.setattr(bh.security, "_known_secrets", lambda: secrets.secret_values())
