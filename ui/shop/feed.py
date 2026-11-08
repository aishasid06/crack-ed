"""Live activity for a shopper search, built from the traced tasks of its run: which site the agent is on
right now, which step of the search it has reached, and a plain-language log of what it has done."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from bh.db import Task, TaskStatus

SITE_NAMES = {"echobazaar": "EchoBazaar", "tradeloft": "TradeLoft", "bazaarly": "Bazaarly", "portal": "the report portal"}
NAV_TASKS = {"agent:task", "recipe:task"}  # navigator / recipe replay on an adapter-less site: payload task = search|listing

STEPS = [
    ("Understanding your search", {"load_brand", "identify_photo", "plan_search"}),
    ("Searching the marketplaces", {"discovery", "photo_search", "download:thumbnails", "prefilter"}),
    ("Opening the closest matches", {"capture_all", "capture", "download:images"}),
    ("Checking each listing", {"features", "classify_all", "classify"}),
]
STEP_OF = {t: i for i, (_, types) in enumerate(STEPS) for t in types}


def site_name(site: str | None) -> str:
    if site and site not in SITE_NAMES:
        try:
            from bh.config import get_config

            cfg = get_config()
            if site in cfg.sites:
                return cfg.sites[site].label(site)
        except Exception:  # noqa: BLE001 - a name is cosmetic
            pass
    return SITE_NAMES.get(site or "", (site or "a site").capitalize())


def _step_of(t: Task) -> int | None:
    if t.type in NAV_TASKS:
        return 1 if (t.payload or {}).get("task") == "search" else 2
    return STEP_OF.get(t.type)


_INDEXED = re.compile(r'^\[\d+\]\s*(?P<role>[\w-]+)?\s*"(?P<name>[^"]*)"')  # agent target: [6] tab "Details"
_CSS = re.compile(r"[#.\[\]=>:]")  # a CSS selector, never shown to shoppers


def _target(target: str) -> tuple[str, str]:
    """(kind, readable name) for an agent or recipe target; selectors and element numbers are dropped."""
    target = target.strip()
    if m := _INDEXED.match(target):
        return (m.group("role") or "").lower(), m.group("name").strip()
    if target.startswith("#tab-"):
        return "tab", target[5:].replace("-", " ").capitalize()
    if not target or _CSS.search(target):
        return "", ""
    return "", target


def _verb(action: str, target: str) -> str:
    """A browsing step in plain words, e.g. ``click`` + ``[6] tab "Details"`` → ``Opening the “Details” tab``."""
    kind, name = _target(target)
    if action == "click":
        if kind == "tab" or (name and "tab" in target.lower()):
            return f"Opening the “{name}” tab" if name else "Opening a tab"
        if name.lower() in ("next", "next page", "›", "»"):
            return "Going to the next page"
        return f"Clicking “{name}”" if name else "Clicking through"
    return {
        "type": "Typing in the search box", "back": "Going back a page",
        "scroll": f"Scrolling {target if target in ('up', 'down') else 'down'}",
        "extract_results": "Reading the search results", "read_results": "Reading the search results",
        "extract_listing": "Reading the listing details", "read_fields": "Reading the listing details",
        "read_images": "Looking at the photos", "done": "Finishing up", "open": "Opening the site",
        "consent": "Answering the cookie banner", "submit": "Searching",
    }.get(action, "Browsing the site")


@dataclass
class SiteNow:
    site: str
    doing: str
    url: str | None


@dataclass
class Feed:
    steps: list[tuple[str, str]]  # (label, done | active | pending)
    now: list[SiteNow] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    opened: int = 0
    to_open: int = 0
    checked: int = 0
    to_check: int = 0


def _ts(t: Task) -> datetime:
    dt = t.started_at or datetime.min.replace(tzinfo=UTC)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _doing(t: Task) -> str | None:
    p = t.payload or {}
    # Agent steps and replayed recipe steps read the same: either way the agent is working on the live site.
    if t.type in ("agent:action", "recipe:step") and p.get("action"):
        return _verb(str(p["action"]), str(p.get("target") or ""))
    if t.type == "agent:model":
        return "Working out what to do next on this site"
    if t.type == "agent:consent":
        return "Answering the cookie banner"
    if t.type in NAV_TASKS:
        subject = str(p.get("subject") or "")
        if p.get("task") == "search":
            return f"Searching for “{subject}”" if subject else "Searching"
        return "Reading a listing"
    if t.type == "discovery":
        return f"Searching for “{p.get('query', '')}”"
    if t.type == "photo_search":
        return "Searching with your photo"
    if t.type == "capture":
        return "Reading a listing"
    if t.type == "browser:action" and str(p.get("action", "")).startswith("click:load_more"):
        return "Loading more results"
    return None


def _event(t: Task, brand: str) -> str | None:
    p, site = t.payload or {}, site_name(t.site)
    if t.status == TaskStatus.waiting_human:
        return f"{site} asked us to prove we're human. We never do that automatically, so we skipped it."
    if t.status == TaskStatus.failed:
        if t.type in ("discovery", "photo_search"):
            return f"Couldn't search {site} this time, so we carried on without it."
        return None
    if t.status != TaskStatus.succeeded:
        return None
    if t.type == "agent:model" and p.get("action"):
        why = str(p.get("reason") or "").strip().rstrip(".")
        step = _verb(str(p["action"]), str(p.get("target") or ""))
        return f"On {site}, the agent decided: {step}" + (f", because {why[0].lower() + why[1:]}." if why else ".")
    if t.type == "agent:consent":
        return f"Answered {site}'s cookie banner, choosing “{p.get('button', 'reject')}”."
    # recipe:learn / recipe:task are internal bookkeeping: the steps above already say what happened on the site
    if t.type == "load_brand":
        return "Loaded the brand's official product photos to compare against."
    if t.type == "identify_photo":
        if p.get("product"):
            return f"Your photo looks like the {p['product']}."
        return "Your photo didn't closely match an official product, so we'll search by name too."
    if t.type == "plan_search":
        qs = p.get("queries") or []
        return "Searching for " + ", ".join(f"“{q}”" for q in qs) + "." if qs else None
    if t.type == "discovery":
        n, keep = p.get("cards", 0), p.get("passing", 0)
        return f"Searched {site} for “{p.get('query', '')}”: {n} results, {keep} worth a closer look."
    if t.type == "photo_search":
        return f"Searched {site} by photo: {p.get('cards', 0)} similar items."
    if t.type == "prefilter":
        return f"{p.get('kept', 0)} listings look like a match. Opening them now."
    if t.type == "classify_all":
        return "Finished checking every listing."
    return None


def build(tasks: list[Task], brand: str, finished: bool) -> Feed:
    started = [t for t in tasks if t.started_at is not None]
    reached = max((s for t in started if (s := _step_of(t)) is not None), default=-1)
    steps = []
    for i, (label, _) in enumerate(STEPS):
        state = "done" if i < reached or (finished and i <= reached) else "active" if i == reached else "pending"
        steps.append((label, state))
    if reached == -1 and not finished:
        steps[0] = (steps[0][0], "active")

    feed = Feed(steps)
    if not finished:
        running = sorted((t for t in started if t.status == TaskStatus.running), key=_ts, reverse=True)
        by_site: dict[str, SiteNow] = {}
        for site in dict.fromkeys(t.site for t in running if t.site):
            here = [t for t in running if t.site == site]  # newest first
            doing = next((d for t in here if (d := _doing(t))), None)
            goto = next((t for t in here if t.type == "browser:goto"), None)
            if doing is None and goto is None:
                continue
            by_site[site] = SiteNow(site, doing or "Opening a page", (goto.payload or {}).get("url") if goto else None)
        feed.now = sorted(by_site.values(), key=lambda s: s.site)

    events = [(t, e) for t in started if (e := _event(t, brand))]
    events.sort(key=lambda te: te[0].finished_at or _ts(te[0]), reverse=True)
    feed.events = [e for _, e in events[:8]]
    caps = [t for t in tasks if t.type == "capture"]
    feed.opened = sum(t.status == TaskStatus.succeeded for t in caps)
    cap_all = next((t for t in tasks if t.type == "capture_all"), None)
    feed.to_open = int((cap_all.payload or {}).get("n", len(caps))) if cap_all else len(caps)
    cls = [t for t in tasks if t.type == "classify"]
    feed.checked = sum(t.status == TaskStatus.succeeded for t in cls)
    cls_all = next((t for t in tasks if t.type == "classify_all"), None)
    feed.to_check = int((cls_all.payload or {}).get("n", len(cls))) if cls_all else len(cls)
    return feed


def short_url(url: str | None) -> str:
    if not url:
        return ""
    u = urlparse(url)
    return (u.path or "/") + (f"?{u.query}" if u.query else "")
