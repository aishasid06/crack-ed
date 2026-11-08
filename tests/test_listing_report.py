"""The agent files a shopper's report through each marketplace's own "Report this listing" form, in a real
browser against the live mock sites, and the submitted report page shows exactly what the shopper confirmed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from sqlmodel import Session

from bh.browser.session import BrowserManager
from bh.db import ShopperReport, session_scope
from bh.enforce.listing_report import run_report, site_for
from tests.test_browser import LiveServer, _cfg_for, live, seeded_catalog, secrets  # noqa: F401 (fixtures)


def _report(engine: Any, url: str, evidence: Path | None) -> int:
    with session_scope(engine) as s:
        r = ShopperReport(listing_url=url, listing_title="A listing", site="x", reason="counterfeit",
                          details="Priced at 14% of the official price and described as a 1:1 copy.",
                          reporter_name="Sam Shopper", reporter_email="sam@example.com", relationship="shopper",
                          evidence_path=str(evidence) if evidence else None)
        s.add(r)
        s.flush()
        return r.id  # type: ignore[return-value]


@pytest.mark.parametrize("site", ["mockbay", "tradeloft"])
async def test_agent_files_the_report_on_the_marketplace(live: LiveServer, tmp_path: Path, engine: Any,
                                                         secrets: Any, site: str) -> None:
    listing = live.catalog.for_site(site)[0]
    path = f"/mockbay/item/{listing.id}" if site == "mockbay" else f"/tradeloft/listing/{listing.id}"
    shot = tmp_path / "shot.png"
    Image.new("RGB", (40, 30), (200, 80, 60)).save(shot)
    rid = _report(engine, live.base + path, shot)
    cfg = _cfg_for(live.base)
    assert site_for(cfg, live.base + path) == site
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        assert await run_report(rid, cfg, browser=b) == "submitted"

    with Session(engine) as s:
        r = s.get(ShopperReport, rid)
    assert r is not None and r.status == "submitted" and r.report_ref and r.report_ref.startswith("LR-")
    assert r.report_url == f"{live.base}/{site}/reports/{r.report_ref}" and r.submitted_at is not None
    async with httpx.AsyncClient() as c:
        page = (await c.get(r.report_url)).text
        evidence = await c.get(f"{r.report_url}/evidence")
    for text in ("Counterfeit or fake item", "1:1 copy", "Sam Shopper", "sam@example.com", listing.title):
        assert text in page
    assert evidence.status_code == 200 and evidence.headers["content-type"] == "image/png"
    assert ("POST", f"{path}/report") in live.requests


async def test_report_that_the_form_rejects_is_marked_failed(live: LiveServer, tmp_path: Path, engine: Any,
                                                             secrets: Any) -> None:
    listing = live.catalog.for_site("mockbay")[0]
    rid = _report(engine, f"{live.base}/mockbay/item/{listing.id}", None)
    with session_scope(engine) as s:
        s.get(ShopperReport, rid).reporter_email = "not-an-email"  # type: ignore[union-attr]
    cfg = _cfg_for(live.base)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        with pytest.raises(Exception, match="rejected the report"):
            await run_report(rid, cfg, browser=b)
    with Session(engine) as s:
        r = s.get(ShopperReport, rid)
    assert r is not None and r.status == "failed" and "valid email" in (r.error or "") and r.report_ref is None


async def test_report_on_a_read_only_site_is_refused_before_any_page_opens(live: LiveServer, tmp_path: Path,
                                                                          engine: Any, secrets: Any) -> None:
    from bh.errors import EnforcementBlocked

    listing = live.catalog.for_site("mockbay")[0]
    rid = _report(engine, f"{live.base}/mockbay/item/{listing.id}", None)
    # read_only sites must also stay within the hard limits (bh/config.py), or the browser refuses to start
    cfg = _cfg_for(live.base, mockbay={"mode": "read_only", "max_concurrency": 1, "max_pages": 2,
                                       "min_delay_ms": 2000})
    before = len(live.requests)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        with pytest.raises(EnforcementBlocked):
            await run_report(rid, cfg, browser=b)
    with Session(engine) as s:
        assert s.get(ShopperReport, rid).status == "failed"  # type: ignore[union-attr]
    assert live.requests[before:] == []  # refused before the browser opened anything
