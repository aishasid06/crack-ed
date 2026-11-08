"""The report portal's own persistence (separate DB + SQLModel registry from ``bh``)."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column
from sqlalchemy.engine import Engine
from sqlalchemy.orm import registry
from sqlmodel import Field, Session, SQLModel, create_engine


def utcnow() -> datetime:
    return datetime.now(UTC)


class PortalBase(SQLModel, registry=registry()):
    pass


class PortalSession(PortalBase, table=True):
    token: str = Field(primary_key=True)
    username: str
    created_at: datetime = Field(default_factory=utcnow)


class ReportDraft(PortalBase, table=True):
    id: str = Field(primary_key=True)
    username: str = Field(index=True)
    steps: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    evidence: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    submitted_ref: str | None = None


class Submission(PortalBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ref: str = Field(unique=True, index=True)
    draft_id: str = Field(unique=True)  # one submission per draft: re-submits are idempotent
    username: str
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    evidence: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    submitted_at: datetime = Field(default_factory=utcnow)
    status: str = "received"


class ListingReport(PortalBase, table=True):
    """A buyer's report of a listing, made through the marketplace's own "Report this listing" form."""
    id: int | None = Field(default=None, primary_key=True)
    ref: str = Field(unique=True, index=True)
    site: str = Field(index=True)
    listing_id: str
    listing_url: str
    listing_title: str
    seller_name: str | None = None
    reason: str
    details: str
    reporter_name: str
    reporter_email: str
    relationship: str
    evidence: str | None = None  # stored file name under <uploads_dir>/listing_reports/<ref>/
    declaration: bool = False
    submitted_at: datetime = Field(default_factory=utcnow)
    status: str = "received"


def make_store_engine(url: str) -> Engine:
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30}
                           if url.startswith("sqlite") else {})
    PortalBase.metadata.create_all(engine)
    return engine


@contextmanager
def store_session(engine: Engine) -> Iterator[Session]:
    with Session(engine, expire_on_commit=False) as s:
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise


def new_token() -> str:
    return secrets.token_urlsafe(32)


def new_draft_id() -> str:
    return "D" + secrets.token_hex(6)


def new_listing_report_ref() -> str:
    return "LR-" + secrets.token_hex(4).upper()


def new_ref() -> str:
    return "IPR-" + secrets.token_hex(4).upper()
