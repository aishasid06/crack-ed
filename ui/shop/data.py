"""Shopper searches: start one, find its run, read its activity and results. Searches live in the shopper
database (``config.shopper.db_url``), never in the enforcement pipeline's."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from bh.brand import load_brand_assets
from bh.config import get_config
from bh.db import Case, Run, Task, init_db, make_engine
from ui.data import CaseBundle, _bundles
from ui.shop.verdicts import CATEGORY_GROUP


@lru_cache(maxsize=1)
def engine() -> Engine:
    cfg = get_config()
    url = cfg.shopper.db_url
    prefix = "sqlite:///"
    if url.startswith(prefix) and not Path(url[len(prefix):]).is_absolute():
        url = prefix + str(cfg.paths.resolve(url[len(prefix):]))
    eng = make_engine(url)
    init_db(eng)
    return eng


_engine_override: Engine | None = None


def get_engine() -> Engine:
    return _engine_override or engine()


def set_engine(e: Engine | None) -> None:
    global _engine_override
    _engine_override = e


@lru_cache(maxsize=1)
def brand_name() -> str:
    cfg = get_config()
    try:
        return load_brand_assets(cfg.paths.resolve(cfg.paths.brand_dir)).meta.name
    except Exception:
        return cfg.brand.capitalize()


def uploads_dir() -> Path:
    cfg = get_config()
    d = cfg.paths.resolve(cfg.shopper.uploads_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class SearchRequest:
    id: str
    text: str | None
    photo: Path | None


def new_search(text: str | None, photo_name: str | None, photo_bytes: bytes | None) -> SearchRequest:
    sid = uuid.uuid4().hex[:12]
    photo = None
    if photo_bytes:
        ext = Path(photo_name or "photo.jpg").suffix.lower()
        photo = uploads_dir() / f"{sid}{ext if ext in ('.jpg', '.jpeg', '.png') else '.jpg'}"
        photo.write_bytes(photo_bytes)
    req = SearchRequest(sid, (text or "").strip() or None, photo)
    (uploads_dir() / f"{sid}.json").write_text(json.dumps({"text": req.text, "photo": str(photo) if photo else None}),
                                               encoding="utf-8")
    return req


def load_search(sid: str) -> SearchRequest | None:
    if not re.fullmatch(r"[0-9a-f]{12}", sid or ""):
        return None
    try:
        d: dict[str, Any] = json.loads((uploads_dir() / f"{sid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return SearchRequest(sid, d.get("text"), Path(d["photo"]) if d.get("photo") else None)


def find_run(sid: str, eng: Engine | None = None) -> Run | None:
    with Session(eng or get_engine()) as s:
        for r in s.exec(select(Run).order_by(col(Run.id).desc()).limit(200)):
            if (r.config_snapshot or {}).get("search_id") == sid:
                return r
    return None


def tasks(run_id: int, eng: Engine | None = None) -> list[Task]:
    with Session(eng or get_engine()) as s:
        return list(s.exec(select(Task).where(Task.run_id == run_id).order_by(col(Task.id))))


def results(run_id: int, eng: Engine | None = None) -> list[CaseBundle]:
    """Classified listings worth showing a shopper (unrelated ones are left out)."""
    with Session(eng or get_engine()) as s:
        cases = [c for c in s.exec(select(Case).where(Case.run_id == run_id))
                 if CATEGORY_GROUP.get(str(c.category), "hidden") != "hidden"]
        return _bundles(s, cases)
