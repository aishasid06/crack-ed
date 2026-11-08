"""Live activity for a shopper search, built from the traced tasks of its run: which site the agent is on
right now, which step of the search it has reached, and a plain-language log of what it has done."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from bh.db import Task, TaskStatus

SITE_NAMES = {"mockbay": "MockBay", "tradeloft": "TradeLoft", "portal": "the report portal"}

STEPS = [
    ("Understanding your search", {"load_brand", "identify_photo", "plan_search"}),
    ("Searching the marketplaces", {"discovery", "photo_search", "download:thumbnails", "prefilter"}),
    ("Opening the closest matches", {"capture_all", "capture", "download:images"}),
    ("Checking each listing", {"features", "classify_all", "classify"}),
]
STEP_OF = {t: i for i, (_, types) in enumerate(STEPS) for t in types}


def site_name(site: str | None) -> str:
    return SITE_NAMES.get(site or "", (site or "a site").capitalize())


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
    reached = max((STEP_OF[t.type] for t in started if t.type in STEP_OF), default=-1)
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
