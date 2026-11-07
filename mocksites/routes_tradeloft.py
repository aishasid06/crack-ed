"""TradeLoft: form search with category/price filters, "Load more" pagination via
an HTML fragment endpoint, spec-table listing pages, and search-by-photo (CLIP)."""

from __future__ import annotations

import io
from urllib.parse import urlencode

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image, UnidentifiedImageError

from mocksites.catalog import Catalog, MockListing
from mocksites.search import search

router = APIRouter(prefix="/tradeloft")
PAGE_SIZE = 8


def _catalog(request: Request) -> Catalog:
    return request.app.state.catalog.get()


def _filtered(request: Request, q: str, category: str, min_price: float | None,
              max_price: float | None) -> list[MockListing]:
    return search(_catalog(request).for_site("tradeloft"), q, "relevance",
                  min_price=min_price, max_price=max_price, site_category=category or None)


def _price(v: str | None) -> float | None:
    if v is None or v.strip() == "":
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        raise HTTPException(422, "Price filter must be a number") from None


def _query_string(q: str, category: str, min_price: str | None, max_price: str | None) -> str:
    return urlencode({"q": q, "category": category, "min_price": min_price or "", "max_price": max_price or ""})


@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> Response:
    return request.app.state.templates.TemplateResponse(request, "tradeloft/home.html", {
        "categories": _catalog(request).site_categories("tradeloft"), "q": "", "category": "",
        "min_price": "", "max_price": ""})


@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", category: str = "", min_price: str | None = None,
                max_price: str | None = None) -> Response:
    results = _filtered(request, q, category, _price(min_price), _price(max_price))
    return request.app.state.templates.TemplateResponse(request, "tradeloft/results.html", {
        "categories": _catalog(request).site_categories("tradeloft"), "q": q, "category": category,
        "min_price": min_price or "", "max_price": max_price or "", "total": len(results),
        "rows": results[:PAGE_SIZE], "next_offset": PAGE_SIZE, "has_more": len(results) > PAGE_SIZE,
        "qs": _query_string(q, category, min_price, max_price)})


@router.get("/search/more", response_class=HTMLResponse)
def search_more(request: Request, q: str = "", category: str = "", min_price: str | None = None,
                max_price: str | None = None, offset: int = Query(0, ge=0)) -> Response:
    results = _filtered(request, q, category, _price(min_price), _price(max_price))
    rows = results[offset: offset + PAGE_SIZE]
    nxt = offset + PAGE_SIZE
    return request.app.state.templates.TemplateResponse(request, "tradeloft/rows_fragment.html", {
        "rows": rows, "next_offset": nxt, "has_more": len(results) > nxt})


@router.get("/listing/{listing_id}", response_class=HTMLResponse)
def listing(request: Request, listing_id: str) -> Response:
    l = _catalog(request).get("tradeloft", listing_id)
    if l is None:
        raise HTTPException(404, "Listing not found")
    return request.app.state.templates.TemplateResponse(request, "tradeloft/listing.html", {"l": l})


@router.get("/visual-search", response_class=HTMLResponse)
def visual_form(request: Request) -> Response:
    return request.app.state.templates.TemplateResponse(request, "tradeloft/visual.html",
                                                        {"rows": None, "error": None})


@router.post("/visual-search", response_class=HTMLResponse)
def visual_search(request: Request, photo: UploadFile = File(...)) -> Response:
    # sync def → runs in the threadpool, so CLIP inference doesn't block the event loop
    templates = request.app.state.templates
    max_bytes = int(request.app.state.settings.max_upload_mb * 1024 * 1024)
    raw = photo.file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        return templates.TemplateResponse(request, "tradeloft/visual.html",
                                          {"rows": None, "error": "Photo is too large."}, status_code=413)
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        return templates.TemplateResponse(request, "tradeloft/visual.html",
                                          {"rows": None, "error": "Upload a JPEG or PNG photo."}, status_code=422)
    ranked = request.app.state.visual.rank(img, _catalog(request).for_site("tradeloft"))
    return templates.TemplateResponse(request, "tradeloft/visual.html",
                                      {"rows": [l for l, _ in ranked], "error": None})
