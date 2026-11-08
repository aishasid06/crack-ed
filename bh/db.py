"""SQLModel data model (SPEC §4) and session helpers.

All timestamps are timezone-aware UTC. JSON columns hold lists/dicts. Enum-valued
columns store the lowercase value string.

Additive fields beyond SPEC §4 are marked ``# extra:`` with a reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, UniqueConstraint, event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine


def utcnow() -> datetime:
    return datetime.now(UTC)


def _json_list() -> Any:
    return Field(default_factory=list, sa_column=Column(JSON, nullable=False))


def _json_dict() -> Any:
    return Field(default_factory=dict, sa_column=Column(JSON, nullable=False))


# ---------------------------------------------------------------- enums


class MarkType(StrEnum):
    word = "word"
    figurative = "figurative"


class Category(StrEnum):
    authorised = "authorised"
    genuine_resale = "genuine_resale"
    counterfeit_identical = "counterfeit_identical"
    counterfeit_similar = "counterfeit_similar"
    copied_photos = "copied_photos"
    grey_import = "grey_import"
    modified_goods = "modified_goods"
    lookalike = "lookalike"
    unrelated = "unrelated"


class Action(StrEnum):
    ignore = "ignore"
    monitor = "monitor"
    platform_report = "platform_report"
    escalate = "escalate"
    refer_trading_standards = "refer_trading_standards"


class CaseStatus(StrEnum):
    classified = "classified"
    awaiting_approval = "awaiting_approval"
    approved = "approved"
    rejected = "rejected"
    drafted = "drafted"
    submitted = "submitted"
    waiting_human = "waiting_human"
    failed = "failed"


class TaskStatus(StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    waiting_human = "waiting_human"
    superseded = "superseded"  # step span replaced by its parent's retry after a human hand-off (not an error)


class RunStatus(StrEnum):
    running = "running"
    finished = "finished"
    failed = "failed"


class ExtractionMethod(StrEnum):
    adapter = "adapter"
    generic_llm = "generic_llm"
    recipe = "recipe"  # learned recipe replayed by code (no LLM), adapter-less sites


# ---------------------------------------------------------------- brand assets


class Brand(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    slug: str = Field(index=True, unique=True)  # extra: CLI selects the brand by slug
    is_fictional: bool = True  # extra: UI labels the brand/registration as fictional


class TradeMark(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    brand_id: int = Field(foreign_key="brand.id", index=True)
    mark_text: str
    mark_type: MarkType
    registration_number: str = Field(unique=True)
    classes: list[int] = _json_list()
    territory: str = "UK"
    status: str = "registered"
    logo_path: str | None = None


class Product(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    brand_id: int = Field(foreign_key="brand.id", index=True)
    name: str
    rrp: float
    goods_class: int | None = None  # extra: products.json carries the goods class (SPEC §6)
    image_paths: list[str] = _json_list()
    phashes: list[str] = _json_list()
    clip_embeddings: str | None = None  # path to .npy


class AuthorisedSeller(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("brand_id", "platform", "seller_name"),)
    id: int | None = Field(default=None, primary_key=True)
    brand_id: int = Field(foreign_key="brand.id", index=True)
    platform: str
    seller_name: str


# ---------------------------------------------------------------- discovery


class Listing(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    platform: str = Field(index=True)
    url: str = Field(unique=True, index=True)
    external_id: str | None = None
    title: str = ""
    description: str = ""
    price: float | None = None
    currency: str | None = None
    seller_name: str | None = Field(default=None, index=True)
    seller_location: str | None = None
    seller_age_days: int | None = None
    rating: float | None = None
    quantity: int | None = None
    image_urls: list[str] = _json_list()
    local_image_paths: list[str] = _json_list()
    screenshot_path: str | None = None
    discovered_by_query: str | None = None
    first_seen_at: datetime = Field(default_factory=utcnow)
    run_id: int | None = Field(default=None, foreign_key="run.id", index=True)  # extra: run that first saw it
    extraction_method: ExtractionMethod = ExtractionMethod.adapter  # extra: SPEC §7.2 "record that the fallback was used"


class Features(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    listing_id: int = Field(foreign_key="listing.id", unique=True, index=True)
    matched_product_id: int | None = Field(default=None, foreign_key="product.id")
    hash_dist: int | None = None
    clip_sim: float | None = None
    price_ratio: float | None = None
    authorised: bool = False
    brand_in_text: bool = False
    misspelt_brand: bool = False
    replica_kw: list[str] = _json_list()  # matched keywords (evidence), empty = none
    inspired_kw: list[str] = _json_list()
    seller_age_days: int | None = None
    modification_kw: list[str] = _json_list()  # extra: needed by modified_goods / exhausted_resale rules
    origin_outside_uk_eea: bool | None = None  # extra: needed by grey_import rule; None = unknown
    origin_evidence: str | None = None  # extra: e.g. "seller location: Osaka, Japan"
    # extras for the three own-photo routes (SPEC §9 Phase 6)
    dhash_dist: int | None = None
    orb_inliers: int | None = None
    own_photo: bool | None = None  # None = no images to compare
    own_photo_routes: list[str] = _json_list()  # e.g. ["hash", "orb"]
    # extras for text / seller signals
    brand_score: float | None = None
    mark_referential_only: bool = False  # mark only used as "not X", "inspired by X", "X-style"
    official_kw: list[str] = _json_list()
    goods_terms: list[str] = _json_list()  # product nouns from the brand's own product data found in the text
    seller_cluster: str | None = None
    cluster_size: int = 1
    cluster_suspect: bool = False
    bulk_quantity: bool = False
    image_count: int = 0
    matched_by: str | None = None  # how matched_product_id was chosen
    computed_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------- cases


class Case(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("run_id", "listing_id"),)
    id: int | None = Field(default=None, primary_key=True)
    listing_id: int = Field(foreign_key="listing.id", index=True)
    run_id: int | None = Field(default=None, foreign_key="run.id", index=True)  # extra: one case per listing per run
    category: Category
    confidence: float
    rule_id: str
    legal_basis: list[dict[str, Any]] = _json_list()  # [{statute, section, summary}] copied from rules.yaml
    legal_elements: dict[str, Any] = _json_dict()  # {element: {status: met|unmet|unknown, evidence}}
    action: Action
    secondary_actions: list[str] = _json_list()  # extra: Trading Standards overlay (SPEC §7.3)
    expected_value: float = 0.0
    status: CaseStatus = Field(default=CaseStatus.classified, index=True)
    reasoning: str = ""
    llm_used: bool = False  # extra: borderline vision call was made
    # extra: the borderline model's assessment contradicts the rules' outcome, which is kept; shown in the review
    # queue as "model disagrees — review"
    model_disagrees: bool = False
    model_disagreement: str | None = None
    notice_text: str | None = None  # extra: factual notice preview shown in review queue
    portal_draft_id: str | None = None  # extra: resume point for submit-on-approval
    submission_ref: str | None = None  # extra: portal submission id once submitted
    error: str | None = None  # extra: reason when status=failed
    created_at: datetime = Field(default_factory=utcnow)
    decided_at: datetime | None = None
    decided_by: str | None = None
    declared_by: str | None = None  # portal declarations signatory (= decided_by), set after approval
    declared_at: datetime | None = None
    review_diff: dict[str, Any] = _json_dict()  # portal review page vs approved case mismatches (blocks submit)
    review_rank: int | None = None  # extra: position in the EV-ordered review queue (None = outside review budget)
    in_review_budget: bool = False  # extra: selected by greedy EV/cost within review_budget_minutes
    review_verified_at: datetime | None = None  # extra: review page matched the approved case
    final_confirmed: bool = False  # extra: second confirmation (only used when enforce.require_final_confirm)
    submitted_at: datetime | None = None  # extra


# ---------------------------------------------------------------- tracing


class Run(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    brand_id: int | None = Field(default=None, foreign_key="brand.id")
    status: RunStatus = RunStatus.running  # extra: UI shows live state
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    config_snapshot: dict[str, Any] = _json_dict()
    totals: dict[str, Any] = _json_dict()


class Task(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("run_id", "dedup_key"),)
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    type: str = Field(index=True)
    site: str | None = None
    dedup_key: str
    status: TaskStatus = Field(default=TaskStatus.pending, index=True)
    parent_id: int | None = Field(default=None, foreign_key="task.id", index=True)
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    waiting_url: str | None = None  # extra: link shown on "Human required" page
    payload: dict[str, Any] = _json_dict()  # extra: inputs needed to resume the task


class LLMCall(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    task_id: int | None = Field(default=None, foreign_key="task.id", index=True)
    run_id: int | None = Field(default=None, foreign_key="run.id", index=True)  # extra: per-run token budget
    model: str  # model that actually served the call (may differ under refusal fallback)
    requested_model: str | None = None  # extra
    purpose: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0  # extra
    cache_creation_input_tokens: int = 0  # extra
    latency_ms: int = 0
    stop_reason: str | None = None  # extra
    error: str | None = None  # extra: failed calls are logged too
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------- engine / sessions


def _sqlite_path(url: str) -> Path | None:
    prefix = "sqlite:///"
    if url.startswith(prefix) and ":memory:" not in url:
        return Path(url[len(prefix):])
    return None


def make_engine(url: str) -> Engine:
    """Create an engine. SQLite gets WAL + busy timeout so the worker and the UI can share it."""
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 30}
        p = _sqlite_path(url)
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args=connect_args)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn: Any, _record: Any) -> None:  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            if _sqlite_path(url) is not None:
                cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Lightweight forward migration for SQLite: add columns that new model fields introduced.
    Added columns are nullable (SQLite cannot add NOT NULL columns without a default)."""
    from sqlalchemy import inspect, text

    if engine.dialect.name != "sqlite":
        return
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    ddl = col.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


def _default_url() -> str:
    from bh.config import get_config

    cfg = get_config()
    url = cfg.paths.db_url
    p = _sqlite_path(url)
    if p is not None and not p.is_absolute():
        url = f"sqlite:///{cfg.paths.resolve(str(p))}"
    return url


_engine_override: Engine | None = None


@lru_cache(maxsize=1)
def _default_engine() -> Engine:
    engine = make_engine(_default_url())
    init_db(engine)
    return engine


def get_engine() -> Engine:
    return _engine_override if _engine_override is not None else _default_engine()


def set_engine(engine: Engine | None) -> None:
    """Point the app at a different engine (tests, alternate DB). None restores the default."""
    global _engine_override
    _engine_override = engine


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    with Session(engine or get_engine(), expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
