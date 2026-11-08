"""Enforcement guard (SPEC §1: submissions only go to our own mock sites).

Every enforcement step — portal login, drafting, review verification, submission — calls this guard
before any browser page is opened or any control is clicked. It refuses:
  - a site whose registry ``mode`` is ``read_only``;
  - a case about a listing on a read_only site or a real host (``check_listing_target``);
  - a case from a real-brand demo run or about a non-fictional brand (``check_case_claimable``);
  - a site or live page whose host is not in the effective allowlist =
    ``ENFORCEMENT_HOSTS`` (code) ∩ ``enforce.allowed_hosts`` (config may only narrow it).
Refusals are logged and raise ``EnforcementBlocked``.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from bh.config import ENFORCEMENT_HOSTS, AppConfig
from bh.errors import EnforcementBlocked

log = logging.getLogger(__name__)


def effective_hosts(cfg: AppConfig) -> frozenset[str]:
    return ENFORCEMENT_HOSTS & {h.lower() for h in cfg.enforce.allowed_hosts}


def _block(message: str) -> None:
    log.error("enforcement blocked: %s", message)
    raise EnforcementBlocked(message)


def check_url(cfg: AppConfig, url: str, what: str = "page") -> None:
    host = (urlparse(url).hostname or "").lower()
    if host not in effective_hosts(cfg):
        _block(f"{what} host {host or '(none)'!r} is not in the enforcement allowlist "
               f"{sorted(effective_hosts(cfg))} ({url})")


def check_listing_target(cfg: AppConfig, platform: str, url: str) -> None:
    """A report may only concern a listing on one of our own mock marketplaces (SPEC §1: never target real
    sellers). Refuses listings from read_only (real) sites and listings whose URL is on a real host."""
    sc = cfg.sites.get(platform)
    if sc is not None and sc.mode == "read_only":
        _block(f"listing on read_only site {platform!r} ({url}); real sellers are never reported")
    check_url(cfg, url, what=f"listing on {platform!r}")


REAL_DEMO_BLOCK = ("real-brand demo: we don't represent this brand and no claim is made against any seller; "
                   "deciding, drafting, approval and submission are refused")


def check_case_claimable(cfg: AppConfig, session: Any, case: Any) -> None:
    """No claim from a real-brand demo: refuses if this config is a demo run, the case's run was one
    (``Run.config_snapshot.real_brand_demo``), or the run's brand is not fictional (``Brand.is_fictional``)."""
    from bh.db import Brand, Run

    if cfg.real_brand_demo:
        _block(f"case {case.id}: {REAL_DEMO_BLOCK}")
    run = session.get(Run, case.run_id) if case.run_id is not None else None
    if run is None:
        return
    if (run.config_snapshot or {}).get("real_brand_demo"):
        _block(f"case {case.id} (run {run.id}): {REAL_DEMO_BLOCK}")
    brand = session.get(Brand, run.brand_id) if run.brand_id is not None else None
    if brand is not None and not brand.is_fictional:
        _block(f"case {case.id}: brand {brand.slug!r} is not fictional; {REAL_DEMO_BLOCK}")


def check_site(cfg: AppConfig, site: str) -> None:
    sc = cfg.site(site)
    if sc.mode == "read_only":
        _block(f"site {site!r} is read_only; login, drafting and submission are refused")
    check_url(cfg, sc.base_url, what=f"site {site!r}")
