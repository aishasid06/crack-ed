"""MockBay: grid search with page numbers, item pages, blocking cookie-consent banner."""

from __future__ import annotations

import math
from typing import Literal

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from mocksites.catalog import Catalog
from mocksites.search import search

router = APIRouter(prefix="/mockbay")
PAGE_SIZE = 12
CONSENT_COOKIE = "mb_consent"


def _ctx(request: Request, **kw: object) -> dict[str, object]:
    return {"consent_needed": CONSENT_COOKIE not in request.cookies, "path": str(request.url.path)
            + (f"?{request.url.query}" if request.url.query else ""), **kw}


def _catalog(request: Request) -> Catalog:
    return request.app.state.catalog.get()


@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> Response:
    cat = _catalog(request)
    featured = sorted(cat.for_site("mockbay"), key=lambda l: (l.listed_days_ago, l.id))[:8]
    return request.app.state.templates.TemplateResponse(request, "mockbay/home.html",
                                                        _ctx(request, featured=featured))


@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", sort: Literal["relevance", "price_asc"] = "relevance",
                page: int = Query(1, ge=1)) -> Response:
    results = search(_catalog(request).for_site("mockbay"), q, sort)
    pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    items = results[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return request.app.state.templates.TemplateResponse(request, "mockbay/search.html", _ctx(
        request, q=q, sort=sort, page=page, pages=pages, total=len(results), items=items))


@router.get("/item/{item_id}", response_class=HTMLResponse)
def item(request: Request, item_id: str) -> Response:
    listing = _catalog(request).get("mockbay", item_id)
    if listing is None:
        raise HTTPException(404, "This listing has ended or does not exist.")
    return request.app.state.templates.TemplateResponse(request, "mockbay/item.html", _ctx(request, l=listing))


@router.post("/consent")
def consent(choice: Literal["all", "essential"] = Form(...), next: str = Form("/mockbay/")) -> Response:
    target = next if next.startswith("/mockbay") else "/mockbay/"
    resp = RedirectResponse(target, status_code=303)
    resp.set_cookie(CONSENT_COOKIE, choice, max_age=60 * 60 * 24 * 180, samesite="lax", path="/mockbay")
    return resp
