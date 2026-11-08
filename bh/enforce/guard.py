"""Enforcement guard (SPEC §1: submissions only go to our own mock sites).

Every enforcement step — portal login, drafting, review verification, submission — calls this guard
before any browser page is opened or any control is clicked. It refuses:
  - a site whose registry ``mode`` is ``read_only``;
  - a site or live page whose host is not in the effective allowlist =
    ``ENFORCEMENT_HOSTS`` (code) ∩ ``enforce.allowed_hosts`` (config may only narrow it).
Refusals are logged and raise ``EnforcementBlocked``.
"""

from __future__ import annotations

import logging
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


def check_site(cfg: AppConfig, site: str) -> None:
    sc = cfg.site(site)
    if sc.mode == "read_only":
        _block(f"site {site!r} is read_only; login, drafting and submission are refused")
    check_url(cfg, sc.base_url, what=f"site {site!r}")
