"""FastAPI app serving MockBay, TradeLoft and the rights-holder report portal.

Run:  uvicorn --factory mocksites.app:create_default_app --port 8000

Everything here is synthetic. The catalog contains no ground-truth labels.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.engine import Engine

from mocksites.catalog import Catalog, load_catalog
from mocksites.store import make_store_engine
from mocksites.visual import ClipEmbedder, ImageEmbedder, VisualIndex

HERE = Path(__file__).resolve().parent
VERIFY_PATH = "/portal/verify"


@dataclass
class MockSettings:
    catalog_dir: Path
    db_url: str
    portal_user: str
    portal_pass: str
    uploads_dir: Path
    challenge_probability: float = 0.0
    challenge_rapid_requests: int = 30
    challenge_window_s: float = 10.0
    challenge_cooldown_s: float = 300.0
    max_upload_mb: float = 5.0
    seed: int | None = None
    embedder: ImageEmbedder | None = None  # None → CLIP (clip-ViT-B-32)
    allowed_listing_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")


class CatalogHolder:
    """Reloads catalog.json when it changes on disk, so seeding doesn't need a restart."""

    def __init__(self, catalog_dir: Path) -> None:
        self.dir = catalog_dir
        self._mtime: float | None = None
        self._catalog = Catalog(listings=[])
        self._lock = threading.Lock()

    def get(self) -> Catalog:
        path = self.dir / "catalog.json"
        mtime = path.stat().st_mtime if path.exists() else None
        with self._lock:
            if mtime != self._mtime:
                self._catalog = load_catalog(self.dir)
                self._mtime = mtime
            return self._catalog


@dataclass
class ChallengeGate:
    """Decides when /portal interposes the human check. Keyed by client IP so a human
    completing the check in their own browser releases the paused agent on the same machine."""

    probability: float
    rapid_requests: int
    window_s: float
    cooldown_s: float
    rng: random.Random
    _hits: dict[str, deque[float]] = field(default_factory=dict)
    _verified_until: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def should_challenge(self, key: str, is_get: bool, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            hits.append(now)
            while hits and hits[0] < now - self.window_s:
                hits.popleft()
            if self._verified_until.get(key, 0.0) > now:
                return False
            if not is_get:
                return False  # never drop a form POST; the next page view is challenged instead
            if len(hits) > self.rapid_requests:
                return True
            return self.rng.random() < self.probability

    def mark_verified(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._verified_until[key] = now + self.cooldown_s
            self._hits.pop(key, None)


def _money(v: float, currency: str = "GBP") -> str:
    sym = {"GBP": "£", "EUR": "€", "USD": "$"}.get(currency, currency + " ")
    return f"{sym}{v:,.2f}"


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def create_app(settings: MockSettings) -> FastAPI:
    app = FastAPI(title="Bounty Hunter mock sites (synthetic)", docs_url=None, redoc_url=None, openapi_url=None)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    (settings.catalog_dir / "images").mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters["money"] = _money
    templates.env.globals["quote"] = quote

    engine: Engine = make_store_engine(settings.db_url)
    app.state.settings = settings
    app.state.templates = templates
    app.state.store = engine
    app.state.catalog = CatalogHolder(settings.catalog_dir)
    app.state.visual = VisualIndex(settings.embedder or ClipEmbedder(), settings.catalog_dir / "images",
                                   settings.catalog_dir / "clip_cache.npz")
    app.state.gate = ChallengeGate(settings.challenge_probability, settings.challenge_rapid_requests,
                                   settings.challenge_window_s, settings.challenge_cooldown_s,
                                   random.Random(settings.seed))

    @app.middleware("http")
    async def portal_challenge(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if path.startswith("/portal") and not path.startswith((VERIFY_PATH, "/portal/static")):
            gate: ChallengeGate = request.app.state.gate
            if gate.should_challenge(client_key(request), request.method == "GET"):
                target = path + (f"?{request.url.query}" if request.url.query else "")
                return RedirectResponse(f"{VERIFY_PATH}?next={quote(target, safe='')}", status_code=303)
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    app.mount("/media", StaticFiles(directory=str(settings.catalog_dir / "images")), name="media")

    from mocksites import routes_mockbay, routes_portal, routes_tradeloft

    app.include_router(routes_mockbay.router)
    app.include_router(routes_tradeloft.router)
    app.include_router(routes_portal.router)
    routes_portal.install(app)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        cat = request.app.state.catalog.get()
        return templates.TemplateResponse(request, "index.html", {
            "n_mockbay": len(cat.for_site("mockbay")), "n_tradeloft": len(cat.for_site("tradeloft"))})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def settings_from_config() -> MockSettings:
    from bh.config import get_config, get_secrets

    cfg = get_config()
    user, password = get_secrets().require_portal()
    m = cfg.mocksites
    db_url = m.db_url
    if db_url.startswith("sqlite:///") and not Path(db_url[len("sqlite:///"):]).is_absolute():
        db_url = f"sqlite:///{cfg.paths.resolve(db_url[len('sqlite:///'):])}"
    return MockSettings(
        catalog_dir=cfg.paths.resolve(m.catalog_dir),
        db_url=db_url,
        portal_user=user,
        portal_pass=password,
        uploads_dir=cfg.paths.resolve(cfg.paths.data_dir) / "portal_uploads",
        challenge_probability=m.challenge_probability,
        challenge_rapid_requests=m.challenge_rapid_requests,
        challenge_window_s=m.challenge_window_s,
        challenge_cooldown_s=m.challenge_cooldown_s,
        max_upload_mb=m.max_upload_mb,
        seed=m.seed,
        allowed_listing_hosts=(m.host, "localhost", "127.0.0.1"),
    )


def create_default_app() -> FastAPI:
    return create_app(settings_from_config())
