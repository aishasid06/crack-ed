"""Validation for the 5-step IP report form. Each ``parse_stepN`` returns
``(data, errors)``; ``errors`` maps field name → message and is empty on success."""

from __future__ import annotations

import re
from datetime import date
from typing import Any
from urllib.parse import urlparse

TERRITORIES = ["UK", "EU", "US", "Other"]
RELATIONSHIPS = {"owner": "Rights owner", "authorised_agent": "Authorised agent (e.g. law firm)"}
RIGHT_TYPES = {"trade_mark": "Trade mark", "copyright": "Copyright", "both": "Trade mark and copyright"}
STATUTES = {
    "Trade Marks Act 1994": "trade_mark",
    "Copyright, Designs and Patents Act 1988": "copyright",
    "Registered Designs Act 1949": "design",
    "Common law (passing off)": "other",
}
DECLARATIONS = {
    "decl_good_faith": "I have a good-faith belief that use of the material described is not authorised "
                       "by the rights owner, its agent or the law.",
    "decl_accurate": "The information in this notice is accurate.",
    "decl_authorised": "I am the rights owner or am authorised to act on the owner's behalf.",
}
LISTING_PATH_PREFIXES = ("/echobazaar/", "/tradeloft/")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UK_REG = re.compile(r"^UK\d{11}$")

Errors = dict[str, str]


def _s(form: dict[str, Any], key: str) -> str:
    v = form.get(key, "")
    return v.strip() if isinstance(v, str) else ""


def parse_step1(form: dict[str, Any]) -> tuple[dict[str, Any], Errors]:
    d = {k: _s(form, k) for k in ("owner_name", "owner_email", "owner_address", "relationship", "agent_name")}
    e: Errors = {}
    if not d["owner_name"]:
        e["owner_name"] = "Rights owner name is required."
    if not _EMAIL.match(d["owner_email"]):
        e["owner_email"] = "A valid contact email is required."
    if not d["owner_address"]:
        e["owner_address"] = "Registered address is required."
    if d["relationship"] not in RELATIONSHIPS:
        e["relationship"] = "Choose your relationship to the rights owner."
    if d["relationship"] == "authorised_agent" and not d["agent_name"]:
        e["agent_name"] = "Agent name is required when filing as an agent."
    return d, e


def parse_step2(form: dict[str, Any]) -> tuple[dict[str, Any], Errors]:
    rt = _s(form, "right_type")
    d: dict[str, Any] = {"right_type": rt}
    e: Errors = {}
    if rt not in RIGHT_TYPES:
        e["right_type"] = "Choose the type of right."
        return d, e
    if rt in ("trade_mark", "both"):
        reg = _s(form, "registration_number").upper()
        d["registration_number"] = reg
        reg_ok = bool(_UK_REG.match(reg)) if _s(form, "territory") == "UK" else bool(reg)
        if not reg_ok:
            e["registration_number"] = "Enter the registration number (UK format: UK followed by 11 digits)."
        raw_classes = _s(form, "classes")
        try:
            classes = sorted({int(c) for c in re.split(r"[,\s]+", raw_classes) if c})
            if not classes or any(not 1 <= c <= 45 for c in classes):
                raise ValueError
            d["classes"] = classes
        except ValueError:
            d["classes"] = raw_classes
            e["classes"] = "Enter Nice classes between 1 and 45, comma separated."
        terr = _s(form, "territory")
        d["territory"] = terr
        if terr not in TERRITORIES:
            e["territory"] = "Choose the territory of registration."
    if rt in ("copyright", "both"):
        d["work_description"] = _s(form, "work_description")
        if len(d["work_description"]) < 10:
            e["work_description"] = "Describe the copyright work (at least 10 characters)."
        raw_date = _s(form, "first_publication_date")
        d["first_publication_date"] = raw_date
        try:
            if date.fromisoformat(raw_date) > date.today():
                e["first_publication_date"] = "First publication date cannot be in the future."
        except ValueError:
            e["first_publication_date"] = "Enter the first publication date (YYYY-MM-DD)."
    return d, e


def parse_step3(form_lists: dict[str, list[str]], right_type: str,
                allowed_hosts: tuple[str, ...]) -> tuple[dict[str, Any], Errors]:
    urls = form_lists.get("url", [])
    statutes = form_lists.get("statute", [])
    sections = form_lists.get("section", [])
    details = form_lists.get("details", [])
    n = max(len(urls), len(statutes), len(sections), len(details))
    items: list[dict[str, str]] = []
    e: Errors = {}
    seen: set[str] = set()

    def at(lst: list[str], i: int) -> str:
        return lst[i].strip() if i < len(lst) else ""

    for i in range(n):
        it = {"url": at(urls, i), "statute": at(statutes, i), "section": at(sections, i), "details": at(details, i)}
        if not any(it.values()):
            continue  # blank row from "add more"
        idx = len(items)
        items.append(it)
        p = urlparse(it["url"])
        if p.scheme not in ("http", "https") or p.hostname not in allowed_hosts \
                or not p.path.startswith(LISTING_PATH_PREFIXES):
            e[f"url_{idx}"] = "Enter the full URL of a listing on EchoBazaar or TradeLoft."
        elif it["url"] in seen:
            e[f"url_{idx}"] = "This URL is already listed."
        seen.add(it["url"])
        kind = STATUTES.get(it["statute"])
        if kind is None:
            e[f"statute_{idx}"] = "Choose the statute relied on."
        elif kind == "trade_mark" and right_type not in ("trade_mark", "both"):
            e[f"statute_{idx}"] = "A trade mark ground needs a trade mark right on step 2."
        elif kind == "copyright" and right_type not in ("copyright", "both"):
            e[f"statute_{idx}"] = "A copyright ground needs a copyright right on step 2."
        if not it["section"] or len(it["section"]) > 60:
            e[f"section_{idx}"] = "Give the specific section relied on (e.g. s.10(1))."
        if len(it["details"]) < 10:
            e[f"details_{idx}"] = "Describe the specific infringement (at least 10 characters)."
    if not items:
        e["items"] = "Add at least one infringing listing URL."
    return {"items": items}, e


def parse_step5(form: dict[str, Any]) -> tuple[dict[str, Any], Errors]:
    d: dict[str, Any] = {k: form.get(k) in ("on", "yes", "true", "1") for k in DECLARATIONS}
    d["signature"] = _s(form, "signature")
    e: Errors = {k: "This declaration is required." for k in DECLARATIONS if not d[k]}
    if not d["signature"]:
        e["signature"] = "Type your full name as a signature."
    return d, e
