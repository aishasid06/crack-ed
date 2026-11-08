"""Enforcement allowlist and read_only sites (SPEC §1): refused, logged, and no browser navigation."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from bh.browser.session import BrowserManager
from bh.config import ENFORCEMENT_HOSTS, PROJECT_ROOT, AppConfig, load_config
from bh.enforce.guard import check_site, check_url, effective_hosts
from bh.enforce.portal import PortalClient, draft_cases, run_drafting
from bh.enforce.submit import process_approved, run_submit_once
from bh.errors import EnforcementBlocked


class RecordingBrowser:
    """Stands in for BrowserManager; records any attempt to open a page, navigate or act."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.calls: list[str] = []

    @asynccontextmanager
    async def page(self, site: str) -> Any:
        self.calls.append(f"page:{site}")
        yield object()

    async def goto(self, site: str, page: Any, url: str) -> None:
        self.calls.append(f"goto:{url}")

    async def login(self, site: str, page: Any) -> None:
        self.calls.append(f"login:{site}")

    def site_url(self, site: str, path: str = "") -> str:
        return self.cfg.site(site).base_url + path

    @asynccontextmanager
    async def action(self, site: str, name: str) -> Any:
        self.calls.append(f"action:{name}")
        yield


def _with_portal(cfg: AppConfig, **portal: Any) -> AppConfig:
    sites = {**cfg.sites, "portal": cfg.sites["portal"].model_copy(update=portal)}
    return cfg.model_copy(update={"sites": sites})


def _narrowed(cfg: AppConfig, hosts: list[str]) -> AppConfig:
    return cfg.model_copy(update={"enforce": cfg.enforce.model_copy(update={"allowed_hosts": hosts})})


BAD = [("non-allowlisted host", {"base_url": "http://evil.example/portal"}, "not in the enforcement allowlist"),
       ("read_only site", {"mode": "read_only", "max_concurrency": 1, "max_pages": 1, "min_delay_ms": 2000},
        "read_only")]


def test_default_allowlist(cfg: AppConfig) -> None:
    assert ENFORCEMENT_HOSTS == {"127.0.0.1", "localhost"} and effective_hosts(cfg) == ENFORCEMENT_HOSTS
    check_site(cfg, "portal")  # the configured mock portal is allowed


def test_config_may_not_widen_the_allowlist(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    raw["enforce"]["allowed_hosts"] = ["127.0.0.1", "evil.example"]
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="may only narrow"):
        load_config(p)


def test_config_may_narrow_the_allowlist(cfg: AppConfig) -> None:
    narrow = _narrowed(_with_portal(cfg, base_url="http://localhost:8000/portal"), ["127.0.0.1"])
    with pytest.raises(EnforcementBlocked):
        check_site(narrow, "portal")
    check_site(_narrowed(cfg, ["127.0.0.1"]), "portal")  # 127.0.0.1 still allowed


@pytest.mark.parametrize("name,portal,msg", BAD, ids=[b[0] for b in BAD])
async def test_draft_refused_without_navigation(cfg: AppConfig, engine: Any, caplog: pytest.LogCaptureFixture,
                                                name: str, portal: dict[str, Any], msg: str) -> None:
    bad = _with_portal(cfg, **portal)
    browser = RecordingBrowser(bad)
    caplog.set_level(logging.ERROR, logger="bh.enforce.guard")
    with pytest.raises(EnforcementBlocked, match=msg):
        await draft_cases(1, bad, browser, engine)  # type: ignore[arg-type]
    with pytest.raises(EnforcementBlocked, match=msg):
        await process_approved(1, bad, browser, engine)  # type: ignore[arg-type]
    with pytest.raises(EnforcementBlocked, match=msg):
        PortalClient(browser, bad)  # type: ignore[arg-type]
    assert browser.calls == []  # no page opened, nothing navigated or clicked
    assert sum("enforcement blocked" in r.getMessage() for r in caplog.records) == 3


@pytest.mark.parametrize("name,portal,msg", BAD, ids=[b[0] for b in BAD])
async def test_cli_entry_points_refuse_before_launching_a_browser(cfg: AppConfig, engine: Any, name: str,
                                                                  portal: dict[str, Any], msg: str,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import bh.enforce.portal as portal_mod

    launched: list[str] = []
    monkeypatch.setattr(portal_mod, "BrowserManager", lambda *a, **k: launched.append("browser"))
    bad = _with_portal(cfg, **portal)
    with pytest.raises(EnforcementBlocked, match=msg):
        await run_drafting(bad, engine)
    with pytest.raises(EnforcementBlocked, match=msg):
        await run_submit_once(bad, engine)
    assert launched == []


@pytest.mark.parametrize("name,portal,msg", BAD, ids=[b[0] for b in BAD])
async def test_login_refused_before_navigation(cfg: AppConfig, secrets: Any, tmp_path: Path, name: str,
                                               portal: dict[str, Any], msg: str) -> None:
    bad = _with_portal(cfg, **portal)
    b = BrowserManager(bad, secrets, sessions_dir=tmp_path / "s")  # not started: any navigation would fail

    class NoPage:
        def __getattr__(self, item: str) -> Any:
            raise AssertionError(f"page.{item} used before the guard refused")

    with pytest.raises(EnforcementBlocked, match=msg):
        await b.login("portal", NoPage())  # type: ignore[arg-type]
    assert not (tmp_path / "s").exists()  # no browser context was created


async def test_off_host_page_blocks_form_submission(cfg: AppConfig) -> None:
    browser = RecordingBrowser(cfg)
    client = PortalClient(browser, cfg)  # type: ignore[arg-type]

    class RedirectedPage:
        url = "https://phishing.example/portal/report/D1/step/1"

        async def click(self, sel: str) -> None:
            raise AssertionError("clicked on an off-host page")

    with pytest.raises(EnforcementBlocked, match="phishing.example"):
        await client._next(RedirectedPage(), "step1_owner")  # type: ignore[arg-type]
    with pytest.raises(EnforcementBlocked):
        await client.submit(RedirectedPage())  # type: ignore[arg-type]
    assert browser.calls == []
    with pytest.raises(EnforcementBlocked):
        check_url(cfg, "http://127.0.0.1.evil.example/portal")  # suffix tricks do not pass
