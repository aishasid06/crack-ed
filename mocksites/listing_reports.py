"""Each marketplace's own "Report this listing" form: how a buyer tells MockBay or TradeLoft that a listing is
counterfeit or misleading. Served for both sites with the same field ids, inside each site's own layout.

    GET/POST /mockbay/item/{id}/report          GET/POST /tradeloft/listing/{id}/report
    GET      /mockbay/reports/{ref}             GET      /tradeloft/reports/{ref}       (the submitted report)

The declaration is a buyer's ("accurate to the best of my knowledge"), not a rights owner's: rights owners use
the IP report portal (/portal).
"""

from __future__ import annotations

import io
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from PIL import Image, UnidentifiedImageError
from sqlmodel import select
from starlette.datastructures import UploadFile

from mocksites.listing_report_forms import DECLARATION, REASONS, RELATIONSHIPS
from mocksites.store import ListingReport, new_listing_report_ref, store_session

router = APIRouter()

SITES = {
    "mockbay": {"name": "MockBay", "base": "mockbay/base.html", "listing": "/mockbay/item/{id}"},
    "tradeloft": {"name": "TradeLoft", "base": "tradeloft/base.html", "listing": "/tradeloft/listing/{id}"},
}
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _listing(request: Request, site: str, listing_id: str) -> Any:
    listing = request.app.state.catalog.get().get(site, listing_id)
    if listing is None:
        raise HTTPException(404, "This listing has ended or does not exist.")
    return listing


def _ctx(request: Request, site: str, **kw: Any) -> dict[str, Any]:
    from mocksites.routes_mockbay import CONSENT_COOKIE

    meta = SITES[site]
    return {"site": site, "site_name": meta["name"], "base": meta["base"], "reasons": REASONS,
            "relationships": RELATIONSHIPS, "declaration": DECLARATION,
            "consent_needed": site == "mockbay" and CONSENT_COOKIE not in request.cookies,
            "path": str(request.url.path), **kw}


def validate(form: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    d = {k: str(form.get(k, "") or "").strip() for k in ("reason", "details", "reporter_name", "reporter_email",
                                                           "relationship")}
    d["declaration"] = form.get("declaration") in ("on", "yes", "true", "1")
    e: dict[str, str] = {}
    if d["reason"] not in REASONS:
        e["reason"] = "Choose what's wrong with the listing."
    if len(d["details"]) < 20:
        e["details"] = "Tell us what's wrong in at least 20 characters."
    elif len(d["details"]) > 2000:
        e["details"] = "Keep the description under 2,000 characters."
    if not d["reporter_name"]:
        e["reporter_name"] = "Enter your name."
    if not _EMAIL.match(d["reporter_email"]):
        e["reporter_email"] = "Enter a valid email address."
    if d["relationship"] not in RELATIONSHIPS:
        e["relationship"] = "Tell us how you came across the listing."
    if not d["declaration"]:
        e["declaration"] = "Confirm the information is accurate."
    return d, e


async def _read_evidence(request: Request, f: Any) -> tuple[bytes | None, str | None, str | None]:
    """(bytes, extension, error) for an optional PNG/JPEG upload."""
    if not isinstance(f, UploadFile) or not f.filename:
        return None, None, None
    max_bytes = int(request.app.state.settings.max_upload_mb * 1024 * 1024)
    raw = await f.read(max_bytes + 1)
    if len(raw) > max_bytes:
        return None, None, f"The file is larger than {request.app.state.settings.max_upload_mb:g} MB."
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = (im.format or "").upper()
    except (UnidentifiedImageError, OSError):
        return None, None, "Upload a PNG or JPEG image."
    if fmt not in ("PNG", "JPEG"):
        return None, None, "Upload a PNG or JPEG image."
    return raw, ".png" if fmt == "PNG" else ".jpg", None


def _form(request: Request, site: str, listing_id: str, values: dict[str, Any] | None = None,
          errors: dict[str, str] | None = None, status: int = 200) -> Response:
    listing = _listing(request, site, listing_id)
    return request.app.state.templates.TemplateResponse(
        request, "listing_report/form.html",
        _ctx(request, site, l=listing, values=values or {"relationship": "shopper"}, errors=errors or {},
             listing_url=SITES[site]["listing"].format(id=listing.id)), status_code=status)


async def _submit(request: Request, site: str, listing_id: str) -> Response:
    listing = _listing(request, site, listing_id)
    form = await request.form()
    d, errors = validate(dict(form))
    raw, ext, ev_error = await _read_evidence(request, form.get("evidence"))
    if ev_error:
        errors["evidence"] = ev_error
    if errors:
        return _form(request, site, listing_id, d, errors, status=422)
    ref = new_listing_report_ref()
    evidence = None
    if raw is not None:
        folder = request.app.state.settings.uploads_dir / "listing_reports" / ref
        folder.mkdir(parents=True, exist_ok=True)
        evidence = f"evidence{ext}"
        (folder / evidence).write_bytes(raw)
    path = SITES[site]["listing"].format(id=listing.id)
    with store_session(request.app.state.store) as s:
        s.add(ListingReport(ref=ref, site=site, listing_id=listing.id, listing_url=str(request.base_url).rstrip("/")
                            + path, listing_title=listing.title, seller_name=listing.seller_name,
                            reason=d["reason"], details=d["details"], reporter_name=d["reporter_name"],
                            reporter_email=d["reporter_email"], relationship=d["relationship"], evidence=evidence,
                            declaration=True))
    return RedirectResponse(f"/{site}/reports/{ref}", status_code=303)


def _received(request: Request, site: str, ref: str) -> Response:
    with store_session(request.app.state.store) as s:
        r = s.exec(select(ListingReport).where(ListingReport.ref == ref, ListingReport.site == site)).first()
    if r is None:
        raise HTTPException(404, "Report not found")
    return request.app.state.templates.TemplateResponse(request, "listing_report/received.html", _ctx(request, site, r=r))


def _evidence(request: Request, site: str, ref: str) -> Response:
    with store_session(request.app.state.store) as s:
        r = s.exec(select(ListingReport).where(ListingReport.ref == ref, ListingReport.site == site)).first()
    if r is None or not r.evidence:
        raise HTTPException(404, "No evidence attached")
    path = request.app.state.settings.uploads_dir / "listing_reports" / _SAFE.sub("_", ref) / _SAFE.sub("_", r.evidence)
    if not path.exists():
        raise HTTPException(404, "No evidence attached")
    return FileResponse(path)


@router.get("/mockbay/item/{listing_id}/report", response_class=HTMLResponse)
def mockbay_form(request: Request, listing_id: str) -> Response:
    return _form(request, "mockbay", listing_id)


@router.post("/mockbay/item/{listing_id}/report", response_class=HTMLResponse)
async def mockbay_submit(request: Request, listing_id: str) -> Response:
    return await _submit(request, "mockbay", listing_id)


@router.get("/tradeloft/listing/{listing_id}/report", response_class=HTMLResponse)
def tradeloft_form(request: Request, listing_id: str) -> Response:
    return _form(request, "tradeloft", listing_id)


@router.post("/tradeloft/listing/{listing_id}/report", response_class=HTMLResponse)
async def tradeloft_submit(request: Request, listing_id: str) -> Response:
    return await _submit(request, "tradeloft", listing_id)


@router.get("/{site}/reports/{ref}", response_class=HTMLResponse)
def received(request: Request, site: str, ref: str) -> Response:
    if site not in SITES:
        raise HTTPException(404)
    return _received(request, site, ref)


@router.get("/{site}/reports/{ref}/evidence")
def evidence(request: Request, site: str, ref: str) -> Response:
    if site not in SITES:
        raise HTTPException(404)
    return _evidence(request, site, ref)
