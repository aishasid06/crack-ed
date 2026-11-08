"""Bazaarly: the "unseen" marketplace (SPEC §9 Phase 11 item 2). No adapter exists for it.

Deliberately unlike MockBay / TradeLoft:
- a cookie modal that blocks the page until a choice is made;
- search only through a form (POST); the results URL is an opaque search id, never a query string;
- result tiles paged with a "Next" button (JavaScript navigation, not a link);
- listing details split across tabs (Overview / Details / Seller); hidden panels are not in the
  accessibility tree until their tab is opened.
"""

from __future__ import annotations

import math
import secrets
import threading

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from mocksites.catalog import Catalog
from mocksites.search import search

router = APIRouter(prefix="/bazaarly")
PAGE_SIZE = 10
CONSENT_COOKIE = "bz_consent"
_MAX_SEARCHES = 5000


class SearchStore:
    """Server-side search sessions: id → term (keeps search terms out of URLs)."""

    def __init__(self) -> None:
        self._terms: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, term: str) -> str:
        sid = secrets.token_hex(6)
        with self._lock:
            if len(self._terms) >= _MAX_SEARCHES:
                self._terms.pop(next(iter(self._terms)))
            self._terms[sid] = term
        return sid

    def get(self, sid: str) -> str | None:
        with self._lock:
            return self._terms.get(sid)


def _store(request: Request) -> SearchStore:
    st = getattr(request.app.state, "bazaarly_searches", None)
    if st is None:
        st = request.app.state.bazaarly_searches = SearchStore()
    return st


def _catalog(request: Request) -> Catalog:
    return request.app.state.catalog.get()


def _ctx(request: Request, **kw: object) -> dict[str, object]:
    here = request.url.path
    return {"consent_needed": CONSENT_COOKIE not in request.cookies, "here": here, **kw}


def _render(request: Request, name: str, status_code: int = 200, **kw: object) -> Response:
    return request.app.state.templates.TemplateResponse(request, f"bazaarly/{name}", _ctx(request, **kw),
                                                        status_code=status_code)


@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> Response:
    return _render(request, "home.html", total=len(_catalog(request).for_site("bazaarly")))


@router.post("/consent")
def consent(choice: str = Form(...), next: str = Form("/bazaarly/")) -> Response:
    target = next if next.startswith("/bazaarly") else "/bazaarly/"
    resp = RedirectResponse(target, status_code=303)
    resp.set_cookie(CONSENT_COOKIE, "all" if choice == "all" else "necessary", max_age=60 * 60 * 24 * 180,
                    samesite="lax", path="/bazaarly")
    return resp


@router.get("/search", response_class=HTMLResponse)
def search_get(request: Request) -> Response:
    """There is no query-string search on Bazaarly: use the form."""
    return _render(request, "message.html", status_code=405, title="Use the search box",
                   message="Search from the box at the top of the page.")


@router.post("/search")
def search_post(request: Request, term: str = Form("")) -> Response:
    sid = _store(request).put(term.strip()[:120])
    return RedirectResponse(f"/bazaarly/results/{sid}/1", status_code=303)


@router.get("/results/{sid}/{page}", response_class=HTMLResponse)
def results(request: Request, sid: str, page: int) -> Response:
    term = _store(request).get(sid)
    if term is None:
        return _render(request, "message.html", status_code=404, title="Search expired",
                       message="That search has expired. Please search again.")
    found = search(_catalog(request).for_site("bazaarly"), term, "relevance")
    pages = max(1, math.ceil(len(found) / PAGE_SIZE))
    page = min(max(page, 1), pages)
    tiles = found[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    nxt = f"/bazaarly/results/{sid}/{page + 1}" if page < pages else None
    return _render(request, "results.html", term=term, tiles=tiles, total=len(found), page=page, pages=pages,
                   next_url=nxt, first=(page - 1) * PAGE_SIZE + 1, last=(page - 1) * PAGE_SIZE + len(tiles))


@router.get("/p/{item_id}", response_class=HTMLResponse)
def product(request: Request, item_id: str) -> Response:
    listing = _catalog(request).get("bazaarly", item_id)
    if listing is None:
        raise HTTPException(404, "This item is no longer available.")
    return _render(request, "product.html", l=listing)
