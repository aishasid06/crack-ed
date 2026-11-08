"""Learned recipes for adapter-less sites (SPEC §9 Phase 11 item 2).

After the agentic navigator succeeds, code (not the model) turns what it did into a recipe, stored in
``data/recipes/<site>.json``:

- search: the search input, the submit control (Enter or a button), the results-page URL pattern, the result
  card pattern (container / link / title / price selectors) and the next-page control;
- listing: where each field sits (which tab to open, which element, and a text pattern around the value), and
  where the photos are.

Replay runs the recipe with no LLM. It goes through the navigator's own guardrails (same host, only the search
form, consent answered by code) because the recipe file is data, not trusted code. If replay fails (an element
is missing, the results page is not reached, required fields are not found), the caller falls back to the
agent loop, and a successful agent run updates the recipe (version + 1). Every replay is traced:
``recipe:task`` (mode "replay", recipe version, steps, 0 tokens) with one ``recipe:step`` span per code step.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bh.browser.adapters.base import Card, ListingData
from bh.config import AppConfig
from bh.db import ExtractionMethod
from bh.errors import ExtractionFailed, HumanRequired
from bh.trace import current_run_id, task_span

if TYPE_CHECKING:
    from bh.browser.agent_nav import NavOutcome, Navigator

log = logging.getLogger(__name__)

# 2: comma-decimal prices and single-result card patterns (2026-11-08); recipes learned before are ignored
# (load() returns None) and re-learned by the agent on the next visit.
RECIPE_SCHEMA = 2
LISTING_FIELDS = ("title", "description", "price", "seller_name", "seller_location", "seller_age_days", "rating",
                  "quantity", "condition")
_NUMERIC = {"price": float, "rating": float, "seller_age_days": int, "quantity": int}
_NUM_GROUP = r"(\d[\d,]*(?:\.\d+)?)"
_PRICE_GROUP = r"(\d[\d.,]*\d|\d)"  # "12.99", "1,234.50", "89,99", "1.234,56"


def parse_price(text: str | None) -> float | None:
    """A price in either convention: "£1,234.50" / "89,99 €" / "1.234,56 €" / "12"."""
    m = re.search(_PRICE_GROUP, text or "")
    if not m:
        return None
    t = m.group(1)
    if "," in t and "." in t:  # the last separator is the decimal point
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:  # "89,99" decimal comma; "1,234" thousands
        t = t.replace(",", ".") if re.fullmatch(r"\d+,\d{1,2}", t) else t.replace(",", "")
    elif t.count(".") > 1 or re.fullmatch(r"\d{1,3}(\.\d{3})+", t):  # "1.234.567" / "1.234" thousands
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None
_CURRENCY = {"£": "GBP", "$": "USD", "€": "EUR", "GBP": "GBP", "USD": "USD", "EUR": "EUR"}
_NEXT = re.compile(r"^\s*(next|next page|more results|show more|›|»|→)\b", re.I)


class ReplayFailed(ExtractionFailed):
    """The recipe no longer matches the site (the caller falls back to the agent loop)."""


# ---------------------------------------------------------------- recipe model


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Locator(_M):
    css: str
    name: str = ""
    tag: str = ""
    role: str = ""


class CardPattern(_M):
    container: str
    link: str | None = None  # None: the container is the link
    href: str  # regex on the link's path
    title: str | None = None
    title_label_prefix: str | None = None  # title = aria-label minus this prefix (when no title element)
    price: str | None = None


class SearchRecipe(_M):
    input: Locator
    submit: Locator | None = None  # None: press Enter in the input
    results_url: str  # regex on the results page's path
    cards: CardPattern
    next: Locator | None = None
    next_learned: Literal["agent", "heuristic", "none"] = "none"


class FieldLoc(_M):
    tab: Locator | None = None  # tab to open first (None: shown on arrival)
    css: str
    pattern: str | None = None  # regex with one group; None: the element's whole text


class ListingRecipe(_M):
    url: str  # regex on the listing path
    fields: dict[str, FieldLoc]
    images: FieldLoc | None = None  # css of the <img> elements
    currency: str | None = None


class Recipe(_M):
    schema_version: int = RECIPE_SCHEMA
    site: str
    version: int = 0
    updated_at: str = ""
    search: SearchRecipe | None = None
    listing: ListingRecipe | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------- store


_LOCKS: dict[Path, asyncio.Lock] = {}


class RecipeStore:
    """``<recipes_dir>/<site>.json``. Writes are atomic and serialised per file."""

    def __init__(self, cfg: AppConfig) -> None:
        self.dir = cfg.paths.resolve(cfg.navigator.recipes_dir)

    def path(self, site: str) -> Path:
        if not re.fullmatch(r"[a-z0-9_\-]+", site):
            raise ValueError(f"bad site name {site!r}")
        return self.dir / f"{site}.json"

    def load(self, site: str) -> Recipe | None:
        p = self.path(site)
        if not p.exists():
            return None
        try:
            r = Recipe.model_validate_json(p.read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as e:
            log.warning("ignoring unreadable recipe %s: %s", p, e)
            return None
        if r.site != site or r.schema_version != RECIPE_SCHEMA:
            log.warning("ignoring recipe %s (site %r, schema %s)", p, r.site, r.schema_version)
            return None
        return r

    async def save_part(self, site: str, part: Literal["search", "listing"], value: BaseModel, reason: str) -> int:
        """Store a learned part; bumps the version only if it changed. Returns the recipe version."""
        p = self.path(site)
        lock = _LOCKS.setdefault(p, asyncio.Lock())
        async with lock:
            cur = self.load(site) or Recipe(site=site)
            if getattr(cur, part) == value:
                return cur.version
            now = datetime.now(UTC).isoformat(timespec="seconds")
            new = cur.model_copy(update={part: value, "version": cur.version + 1, "updated_at": now,
                                         "history": [*cur.history, {"version": cur.version + 1, "part": part,
                                                                    "at": now, "reason": reason[:300],
                                                                    "run_id": current_run_id()}][-50:]})
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(new.model_dump(mode="json"), indent=2), encoding="utf-8")
            os.replace(tmp, p)
            return new.version


# ---------------------------------------------------------------- DOM helpers (JavaScript)

# Stable CSS selector for an element: id, a distinctive attribute, its classes, or a short path from the nearest
# identifiable ancestor. Ids / classes containing long digit runs are treated as generated and skipped.
CSS_FOR_JS = r"""
  const bhCss = (el) => {
    const q = s => { try { return document.querySelectorAll(s).length === 1; } catch (e) { return false; } };
    const stable = c => !/\d{3,}/.test(c) && !/^(active|selected|open|is-|has-|focus|hover)/.test(c);
    const own = e => {
      const tag = e.tagName.toLowerCase();
      if (e.id && !/\d{3,}/.test(e.id) && q('#' + CSS.escape(e.id))) return '#' + CSS.escape(e.id);
      for (const a of ['name', 'aria-label', 'data-testid', 'placeholder', 'title']) {
        const v = e.getAttribute(a);
        if (v && v.length < 80 && !/\d{3,}/.test(v)) {
          const s = `${tag}[${a}="${v.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"]`;
          if (q(s)) return s;
        }
      }
      const cls = [...e.classList].filter(stable).map(c => '.' + CSS.escape(c)).join('');
      if (cls && q(tag + cls)) return tag + cls;
      return null;
    };
    const s = own(el);
    if (s) return s;
    const parts = [];
    let e = el;
    for (let depth = 0; e && e !== document.body && depth < 6; depth++) {
      const tag = e.tagName.toLowerCase();
      const sib = e.parentElement ? [...e.parentElement.children].filter(x => x.tagName === e.tagName) : [e];
      parts.unshift(sib.length > 1 ? `${tag}:nth-of-type(${sib.indexOf(e) + 1})` : tag);
      e = e.parentElement;
      const anc = e && e !== document.body ? own(e) : null;
      if (anc && q(anc + ' > ' + parts.join(' > '))) return anc + ' > ' + parts.join(' > ');
    }
    const sel = 'body > ' + parts.join(' > ');
    return q(sel) ? sel : null;
  };
  const bhNorm = s => (s || '').replace(/\s+/g, ' ').trim();
  const bhSig = e => e.tagName.toLowerCase()
    + [...e.classList].filter(c => !/\d{3,}/.test(c)).map(c => '.' + CSS.escape(c)).join('');
"""

_PROBE_CARDS_JS = "(cards) => {" + CSS_FOR_JS + r"""
  const PRICE_RE = /[£$€]\s?\d|\d[.,]\d{2}\s?([£$€]|GBP|EUR|USD\b)/;
  const abs = h => { try { return new URL(h, location.href).href; } catch (e) { return ''; } };
  const want = new Map(cards.map(c => [c.url, c.title]));
  const links = [...document.querySelectorAll('a[href]')].filter(a => want.has(abs(a.getAttribute('href'))));
  if (!links.length) return null;
  const title0 = bhNorm(want.get(abs(links[0].getAttribute('href'))));
  let c = links[0];
  if (links.length >= 2) {  // the largest ancestor holding exactly one result link
    while (c.parentElement && c.parentElement !== document.body
           && links.filter(l => c.parentElement.contains(l)).length === 1) c = c.parentElement;
  } else {  // one result: the smallest ancestor that holds the title and a price (or a photo)
    const holds = e => [...e.querySelectorAll('*')].some(x => x.children.length === 0 && bhNorm(x.innerText) === title0)
      && ([...e.querySelectorAll('*')].some(x => x.children.length === 0 && PRICE_RE.test(x.innerText || ''))
          || !!e.querySelector('img'));
    while (c && c !== document.body && !holds(c)) c = c.parentElement;
    if (!c || c === document.body) return null;
  }
  const container = bhSig(c);
  const conts = [...document.querySelectorAll(container)];
  const covered = links.filter(l => conts.some(k => k.contains(l))).length;
  const link = c === links[0] ? null : bhSig(links[0]);
  const title = bhNorm(want.get(abs(links[0].getAttribute('href'))));
  let titleSel = null, labelPrefix = null, priceSel = null;
  for (const e of c.querySelectorAll('*')) {
    if (!titleSel && title && bhNorm(e.innerText) === title && e.children.length === 0) titleSel = bhSig(e);
    if (!priceSel && e.children.length === 0 && PRICE_RE.test(e.innerText || ''))
      priceSel = bhSig(e);
  }
  const label = bhNorm(links[0].getAttribute('aria-label'));
  if (!titleSel && title && label.endsWith(title)) labelPrefix = label.slice(0, label.length - title.length);
  return {container, link, title: titleSel, title_label_prefix: labelPrefix, price: priceSel, covered,
          total: links.length, paths: links.map(l => new URL(abs(l.getAttribute('href'))).pathname)};
}
"""

_READ_CARDS_JS = r"""(p) => {
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const out = [];
  for (const c of document.querySelectorAll(p.container)) {
    const a = p.link ? c.querySelector(p.link) : (c.matches('a[href]') ? c : c.querySelector('a[href]'));
    if (!a || !a.getAttribute('href')) continue;
    let title = p.title ? norm((c.querySelector(p.title) || {}).innerText) : '';
    const label = norm(a.getAttribute('aria-label'));
    if (!title && p.title_label_prefix !== null && label.startsWith(p.title_label_prefix))
      title = label.slice(p.title_label_prefix.length).trim();
    if (!title) title = norm(a.innerText) || label;
    const price = p.price ? norm((c.querySelector(p.price) || {}).innerText) : '';
    out.push({url: new URL(a.getAttribute('href'), location.href).href, title, price});
  }
  return out;
}"""

_PROBE_FIELDS_JS = "(values) => {" + CSS_FOR_JS + r"""
  const vis = e => !e.closest('[hidden]') && e.getBoundingClientRect().width > 0;
  const all = [...document.body.querySelectorAll('*')]
    .filter(e => !['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE'].includes(e.tagName) && vis(e));
  const texts = all.map(e => bhNorm(e.innerText));
  const out = {};
  for (const [field, cands] of Object.entries(values)) {
    let best = null;
    all.forEach((e, k) => {
      const t = texts[k];
      if (!t) return;
      for (const v of cands) {
        const i = t.indexOf(v);
        if (i < 0) continue;
        const before = i > 0 ? t[i - 1] : ' ', after = t[i + v.length] || ' ';
        if (/\w/.test(v[0]) && /[\w.]/.test(before)) continue;
        if (/\w/.test(v[v.length - 1]) && /\w/.test(after)) continue;
        if (!best || t.length < best.text.length) best = {k, text: t, value: v};
      }
    });
    if (best) { const css = bhCss(all[best.k]); if (css) out[field] = {css, text: best.text, value: best.value}; }
  }
  return out;
}
"""

_PROBE_IMAGES_JS = "(urls) => {" + CSS_FOR_JS + r"""
  const want = new Set(urls);
  const imgs = [...document.querySelectorAll('img')].filter(i => want.has(i.currentSrc || i.src) || want.has(i.src));
  if (!imgs.length) return null;
  const parent = imgs[0].parentElement;
  const pc = parent && parent !== document.body ? bhCss(parent) : null;
  const sel = pc ? pc + ' img' : 'img';
  return {css: sel, n: document.querySelectorAll(sel).length, matched: imgs.length};
}
"""

_READ_FIELDS_JS = r"""(fields) => {
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const out = {};
  for (const [name, css] of Object.entries(fields)) {
    let e = null;
    try { e = document.querySelector(css); } catch (err) { e = null; }
    out[name] = e && !e.closest('[hidden]') ? norm(e.innerText) : null;
  }
  return out;
}"""

_COUNT_LINKS_JS = r"""(pattern) => {
  const re = new RegExp(pattern);
  return [...document.querySelectorAll('a[href]')].filter(a => {
    try { const u = new URL(a.getAttribute('href'), location.href); return u.host === location.host && re.test(u.pathname); }
    catch (e) { return false; }
  }).length;
}"""

_READ_IMAGES_JS = r"""(css) => [...document.querySelectorAll(css)].map(i => i.currentSrc || i.src).filter(Boolean)"""


async def probe_cards(page: Page, cards: list[Card]) -> dict[str, Any] | None:
    """Card pattern for the cards the agent just extracted from this page (code, no LLM)."""
    try:
        return await page.evaluate(_PROBE_CARDS_JS, [{"url": c.url, "title": c.title} for c in cards])
    except Exception as e:  # noqa: BLE001 - learning is best effort
        log.debug("card probe failed: %s", e)
        return None


def next_candidate(elements: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A visible "Next" control on a results page (heuristic, used if the agent never needed one)."""
    for e in elements:
        if not e.get("consent_dialog") and e.get("css") and _NEXT.match(e.get("name") or ""):
            return e
    return None


# ---------------------------------------------------------------- learning (code, from the agent's trace)


def _locator(el: dict[str, Any] | None) -> Locator | None:
    if not el or not el.get("css"):
        return None
    return Locator(css=el["css"], name=el.get("name") or "", tag=el.get("tag") or "", role=el.get("role") or "")


def path_pattern(paths: list[str]) -> str:
    """Regex for a family of URL paths: segments that differ (or contain digits) become ``[^/]+``."""
    split = [p.rstrip("/").split("/") for p in paths if p]
    if not split:
        return r"^/.*$"
    n = len(split[0])
    if any(len(s) != n for s in split):
        common = os.path.commonprefix(["/".join(s) for s in split])
        return "^" + re.escape(common.rsplit("/", 1)[0]) + r"/.+$"
    segs = []
    for i in range(n):
        vals = {s[i] for s in split}
        v = next(iter(vals))
        segs.append(re.escape(v) if len(vals) == 1 and not re.search(r"\d", v) else r"[^/]+")
    return "^" + "/".join(segs) + "/?$"


def learn_search(outcome: NavOutcome) -> SearchRecipe | None:
    """Search recipe from a successful agent search, or None if the trace does not show enough."""
    trace = outcome.trace
    typed = [t for t in trace if t["action"] == "type" and t["outcome"] == "ok"]
    pages = [t for t in trace if t["action"] == "extract_results" and t.get("cards")]
    if not typed or not pages:
        return None
    t = typed[-1]
    inp = _locator(t["el"])
    if inp is None:
        return None
    submit: Locator | None = None
    if not t["args"].get("submit"):  # typed, then clicked a submit button
        after = [x for x in trace[trace.index(t) + 1:] if x["action"] == "click" and x["outcome"] == "ok"]
        if not after or not (after[0]["el"] or {}).get("submits"):
            return None
        submit = _locator(after[0]["el"])
    probe = next((p["probe"] for p in pages if p.get("probe") and p["probe"]["covered"] == p["probe"]["total"]),
                 None)
    if probe is None:
        return None
    hrefs = [path for p in pages if p.get("probe") for path in p["probe"]["paths"]]
    cards = CardPattern(container=probe["container"], link=probe["link"], href=path_pattern(hrefs),
                        title=probe["title"], title_label_prefix=probe["title_label_prefix"], price=probe["price"])
    results_paths = [urlparse(p["url"]).path for p in pages]
    nxt, how = None, "none"
    result_urls = {p["url"] for p in pages}
    for x in trace:
        if x["action"] == "click" and x["outcome"] == "ok" and x["url_before"] in result_urls \
                and x["url_after"] in result_urls and x["url_after"] != x["url_before"]:
            nxt, how = _locator(x["el"]), "agent"
            break
    if nxt is None:
        cand = next((p["next"] for p in pages if p.get("next")), None)
        if cand is not None:
            nxt, how = _locator(cand), "heuristic"
    return SearchRecipe(input=inp, submit=submit, results_url=path_pattern(results_paths), cards=cards, next=nxt,
                        next_learned=how)


def _candidates(field: str, value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if field == "price":
        v = float(value)
        dot = [f"{v:,.2f}", f"{v:.2f}"]
        comma = [x.replace(",", "_").replace(".", ",").replace("_", ".") for x in dot]  # "1.234,56", "89,99"
        return list(dict.fromkeys(dot + comma + ([f"{int(v):,}", str(int(v))] if v.is_integer() else [])))
    if field == "rating":
        v = float(value)
        return list(dict.fromkeys([f"{v:.1f}", f"{v:g}", f"{v:.2f}"]))
    if field in ("seller_age_days", "quantity"):
        return list(dict.fromkeys([f"{int(value):,}", str(int(value))]))
    text = re.sub(r"\s+", " ", str(value)).strip()
    if field == "description" and len(text) > 60:
        return [text[:60].rsplit(" ", 1)[0]]  # whole words (the probe checks word boundaries)
    return [text]


def field_pattern(field: str, text: str, value: str) -> str | None:
    """Regex capturing ``value`` from ``text`` by the words around it; None means "the whole text"."""
    if field == "description" or text == value:
        return None
    i = text.find(value)
    pre = text[max(0, i - 24):i]
    if i > 24 and " " in pre:
        pre = pre.split(" ", 1)[1]
    rest = text[i + len(value):]
    suf = rest.split(" ")[1] if rest.startswith(" ") and len(rest.split(" ")) > 1 else rest.split(" ")[0]
    suf = suf[:12]
    group = _PRICE_GROUP if field == "price" else _NUM_GROUP if field in _NUMERIC else ("(.+?)" if suf else "(.+)")
    sep = r"\s*" if rest.startswith(" ") else ""
    return re.escape(pre) + group + (sep + re.escape(suf) if suf else r"\s*$")


def parse_field(field: str, text: str | None, loc: FieldLoc) -> Any:
    if not text:
        return None
    raw = text
    if loc.pattern:
        m = re.search(loc.pattern, text)
        if not m:
            return None
        raw = m.group(1).strip()
    if field == "price":
        return parse_price(raw)
    if field in _NUMERIC:
        m = re.search(_NUM_GROUP, raw)
        if not m:
            return None
        try:
            return _NUMERIC[field](float(m.group(1).replace(",", "")))
        except ValueError:
            return None
    return raw


def _same(field: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if field in ("price", "rating"):
        return abs(float(a) - float(b)) < 0.005
    if field == "description":
        x, y = re.sub(r"\s+", " ", str(a)).strip(), re.sub(r"\s+", " ", str(b)).strip()
        return bool(x) and (x.startswith(y[:60]) or y.startswith(x[:60]))
    return str(a).strip() == str(b).strip()


# ---------------------------------------------------------------- spans


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _span(type_: str, site: str, key: str, payload: dict[str, Any]) -> Any:
    from contextlib import nullcontext

    if current_run_id() is None:
        return nullcontext(None)
    return task_span(None, type_, site, key, payload=payload)


# ---------------------------------------------------------------- replay (no LLM)


class Replayer:
    """Runs a recipe through the navigator's guardrails. Never calls a model."""

    def __init__(self, nav: Navigator, recipe: Recipe, traced: bool = True) -> None:
        self.nav, self.r, self.site = nav, recipe, nav.site
        self.steps = 0
        self.traced = traced  # False while verifying a freshly learned recipe
        self._key = ""

    async def _step(self, page: Page, action: str, target: str, fn: Any) -> Any:
        self.steps += 1
        if not self.traced:
            return await fn()
        async with _span("recipe:step", self.site, f"{self._key}:{self.steps}",
                         {"kind": "code step", "mode": "replay", "step": self.steps, "action": action,
                          "target": target, "url_before": page.url, "recipe_version": self.r.version}) as sp:
            result = "failed"
            try:
                out = await fn()
                result = "ok"
                return out
            except ReplayFailed as e:
                result = f"failed: {e}"
                raise
            finally:
                if sp is not None:
                    sp.payload = {**sp.payload, "outcome": result, "url_after": page.url}

    async def _element(self, page: Page, loc: Locator, what: str) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = await self.nav.observe(page)
        el = next((e for e in obs["elements"] if e.get("css") == loc.css), None)
        if el is None and loc.name:
            el = next((e for e in obs["elements"] if e["name"] == loc.name and e["tag"] == loc.tag), None)
        if el is None:
            raise ReplayFailed(f"{what} not found ({loc.css})")
        return el, obs

    async def _act(self, page: Page, action: str, loc: Locator, what: str, **args: Any) -> None:
        from bh.browser.agent_nav import GuardrailBlocked

        el, obs = await self._element(page, loc, what)
        args = {"index": el["index"], **args}
        try:
            self.nav._check(action, args, el, page.url, obs["elements"])
            await self.nav._robots_before(page, action, args, el)
        except GuardrailBlocked as e:
            raise ReplayFailed(f"{what}: guardrail refused the recipe step: {e}") from e
        try:
            await self.nav._execute(page, action, args)
        except Exception as e:  # noqa: BLE001
            raise ReplayFailed(f"{what}: {type(e).__name__}: {str(e).splitlines()[0][:160]}") from e
        await self.nav._settle(page)
        from bh.browser.agent_nav import NavigationFailed

        try:
            await self.nav.after_navigation(page)  # same site, robots.txt (read_only sites)
        except NavigationFailed as e:
            raise ReplayFailed(f"{what}: {e}") from e

    async def _consent(self, page: Page, outcome: NavOutcome) -> None:
        obs = await self.nav.observe(page)
        await self.nav._consent(page, obs, outcome, self._key)

    @staticmethod
    def _card(site: str, raw: dict[str, Any]) -> Card | None:
        if not raw.get("title"):
            return None
        price, currency = parse_price(raw.get("price")), None
        if price is not None:
            sym = next((s for s in _CURRENCY if s in (raw.get("price") or "")), None)
            currency = _CURRENCY.get(sym) if sym else None
        return Card(site=site, url=raw["url"], title=raw["title"], price=price, currency=currency)

    async def search(self, page: Page, query: str, max_pages: int, outcome: NavOutcome) -> AsyncIterator[list[Card]]:
        rec = self.r.search
        if rec is None:
            raise ReplayFailed("recipe has no search part")
        self._key = f"recipe:{self.site}:search:{self.nav._seq}:{_h(query)}"
        self.nav._seq += 1
        outcome.mode, outcome.recipe_version = "replay", self.r.version
        seen: set[str] = set()
        async with _span("recipe:task", self.site, self._key,
                         {"kind": "code step", "mode": "replay", "task": "search", "subject": query[:200],
                          "recipe_version": self.r.version}) as sp:
            try:
                await self._step(page, "open", "home", lambda: self.nav.b.goto(
                    self.site, page, self.nav.b.site_url(self.site, "/")))
                await self._consent(page, outcome)
                await self._step(page, "type", rec.input.css, lambda: self._act(
                    page, "type", rec.input, "search input", text=query, submit=rec.submit is None))
                if rec.submit is not None:
                    await self._step(page, "click", rec.submit.css,
                                     lambda: self._act(page, "click", rec.submit, "search button"))
                for n in range(1, max_pages + 1):
                    if not re.search(rec.results_url, urlparse(page.url).path):
                        raise ReplayFailed(f"not on a results page ({urlparse(page.url).path} !~ {rec.results_url})")

                    async def read() -> list[Card]:
                        raw = await page.evaluate(_READ_CARDS_JS, rec.cards.model_dump())
                        out = []
                        for r in raw:
                            if not self.nav._same_host(r["url"]) or \
                                    not re.search(rec.cards.href, urlparse(r["url"]).path):
                                continue
                            c = self._card(self.site, r)
                            if c is not None:
                                out.append(c)
                        if raw and not out:
                            raise ReplayFailed("result cards found but none could be read")
                        if not raw:  # "no results", or a card pattern that no longer matches?
                            n = await page.evaluate(_COUNT_LINKS_JS, rec.cards.href)
                            if n:
                                raise ReplayFailed(f"card pattern matched nothing but the page links to {n} "
                                                   "listing(s)")
                        return out

                    cards = await self._step(page, "read_results", rec.cards.container, read)
                    outcome.pages_read += 1
                    new = [c for c in cards if c.url not in seen]
                    seen |= {c.url for c in cards}
                    if new:
                        yield new
                    if not cards or n == max_pages:
                        break
                    obs = await self.nav.observe(page)
                    nxt = rec.next if rec.next and any(e.get("css") == rec.next.css for e in obs["elements"]) \
                        else None
                    if nxt is None and rec.next is None:
                        # learned from a one-page search: look for a visible "Next" control (code, no LLM)
                        nxt = _locator(next_candidate(obs["elements"]))  # type: ignore[arg-type]
                    if nxt is None:
                        break  # no next control: last page
                    before = page.url
                    await self._step(page, "click", nxt.css,
                                     lambda: self._act(page, "click", nxt, "next-page control"))
                    if page.url == before:
                        break
                outcome.success, outcome.reason = True, f"replayed recipe v{self.r.version}"
            except GeneratorExit:  # the caller stopped reading pages (e.g. the page cap): not a failure
                outcome.success = True
                outcome.reason = (f"replayed recipe v{self.r.version}; stopped by the caller after "
                                  f"{outcome.pages_read} result page(s)")
            except ReplayFailed as e:
                outcome.reason = str(e)
                raise
            finally:
                outcome.steps = self.steps
                if sp is not None:
                    sp.payload = {**sp.payload, "success": outcome.success, "reason": outcome.reason,
                                  "steps": self.steps, "pages_read": outcome.pages_read, "tokens": 0,
                                  "listings_found": len(seen), "consent": outcome.consent}

    async def _read_fields(self, page: Page, recipe: ListingRecipe, outcome: NavOutcome) -> dict[str, Any]:
        groups: dict[str | None, list[str]] = {}
        for f, loc in recipe.fields.items():
            groups.setdefault(loc.tab.css if loc.tab else None, []).append(f)
        img_tab = recipe.images.tab.css if recipe.images and recipe.images.tab else None
        values: dict[str, Any] = {}
        images: list[str] = []
        order = [None] + [k for k in groups if k is not None]
        if recipe.images and img_tab not in order:
            order.append(img_tab)
        for tab in order:
            if tab is not None:
                loc = next(l.tab for l in [*recipe.fields.values(), *( [recipe.images] if recipe.images else [])]
                           if l.tab and l.tab.css == tab)
                await self._step(page, "click", tab, lambda loc=loc: self._act(page, "click", loc, f"tab {loc.name}"))
            names = groups.get(tab, [])
            if names:
                raw = await self._step(page, "read_fields", ", ".join(names), lambda names=names: page.evaluate(
                    _READ_FIELDS_JS, {f: recipe.fields[f].css for f in names}))
                for f in names:
                    values[f] = parse_field(f, raw.get(f), recipe.fields[f])
                    if f == "price" and raw.get(f):
                        sym = next((s for s in _CURRENCY if s in raw[f]), None)
                        values["_currency"] = _CURRENCY.get(sym) if sym else None
            if recipe.images and img_tab == tab:
                images = await self._step(page, "read_images", recipe.images.css,
                                          lambda: page.evaluate(_READ_IMAGES_JS, recipe.images.css))
        values["_images"] = images
        return values

    async def listing(self, page: Page, url: str, outcome: NavOutcome) -> ListingData:
        rec = self.r.listing
        if rec is None:
            raise ReplayFailed("recipe has no listing part")
        if not re.search(rec.url, urlparse(url).path):
            raise ReplayFailed(f"{urlparse(url).path} does not look like a listing page ({rec.url})")
        self._key = f"recipe:{self.site}:listing:{self.nav._seq}:{_h(url)}"
        self.nav._seq += 1
        outcome.mode, outcome.recipe_version = "replay", self.r.version
        async with _span("recipe:task", self.site, self._key,
                         {"kind": "code step", "mode": "replay", "task": "listing", "subject": url[:200],
                          "recipe_version": self.r.version}) as sp:
            found: list[str] = []
            try:
                await self._step(page, "open", url, lambda: self.nav.b.goto(self.site, page, url))
                await self._consent(page, outcome)
                v = await self._read_fields(page, rec, outcome)
                found = [f for f in rec.fields if v.get(f) not in (None, "")]
                missing = [f for f in rec.fields if f not in found]
                if not v.get("title") or len(missing) * 2 > len(rec.fields):
                    raise ReplayFailed(f"listing fields not found: {', '.join(missing)}")
                data = ListingData(
                    site=self.site, url=url, title=v["title"], description=v.get("description") or "",
                    price=v.get("price"), currency=v.get("_currency") or (rec.currency if v.get("price") else None),
                    seller_name=v.get("seller_name"), seller_location=v.get("seller_location"),
                    seller_age_days=v.get("seller_age_days"),
                    rating=v.get("rating") if v.get("rating") is None or 0 <= v["rating"] <= 5 else None,
                    quantity=v.get("quantity"), condition=v.get("condition"),
                    image_urls=[u for u in v["_images"] if self.nav._same_host(u)],
                    extraction_method=ExtractionMethod.recipe,
                    fallback_reason=f"recipe v{self.r.version}" + (f"; missing {', '.join(missing)}" if missing else ""))
                outcome.success, outcome.reason = True, f"replayed recipe v{self.r.version}"
                return data
            except ReplayFailed as e:
                outcome.reason = str(e)
                raise
            except HumanRequired:
                raise
            finally:
                outcome.steps = self.steps
                if sp is not None:
                    sp.payload = {**sp.payload, "success": outcome.success, "reason": outcome.reason,
                                  "steps": self.steps, "tokens": 0, "fields_found": found,
                                  "consent": outcome.consent}


# ---------------------------------------------------------------- learning a listing recipe (code, no LLM)


def choose_locations(hits_by_state: list[tuple[Locator | None, dict[str, dict[str, Any]]]]) -> dict[str, FieldLoc]:
    """Pick one location per field from the probe hits of every page state (arrival, then each tab).

    The value found anywhere is not enough: "Pre-owned" also appears in a "Pre-owned … bottle" title. Prefer
    the element the value fills most of ("Condition: Pre-owned" over a long title), never an element another
    field occupies whole, and the earlier state on a tie. Title is placed first (it is the anchor field).
    """
    cands: dict[str, list[tuple[float, int, Locator | None, dict[str, Any]]]] = {}
    for i, (tab, hits) in enumerate(hits_by_state):
        for f, h in hits.items():
            cover = len(h["value"]) / max(len(h["text"]), 1)
            cands.setdefault(f, []).append((cover, i, tab, h))
    out: dict[str, FieldLoc] = {}
    whole: set[str] = set()  # elements a field occupies entirely
    for f in sorted(cands, key=lambda f: (f != "title", f)):
        for cover, _, tab, h in sorted(cands[f], key=lambda c: (-c[0], c[1])):
            if h["css"] in whole:
                continue
            out[f] = FieldLoc(tab=tab, css=h["css"], pattern=field_pattern(f, h["text"], h["value"]))
            if cover >= 0.999:
                whole.add(h["css"])
            break
    return out


async def learn_listing(nav: Navigator, page: Page, url: str, data: ListingData,
                        outcome: NavOutcome) -> ListingRecipe | None:
    """Find where the agent's values sit on the page (opening the tabs it opened), then replay the result on
    the same page and keep only the fields that read back the agent's values."""
    from bh.browser.agent_nav import GuardrailBlocked, NavOutcome as _NO

    values = {f: _candidates(f, getattr(data, f)) for f in LISTING_FIELDS}
    values = {f: c for f, c in values.items() if c}
    if "title" not in values:
        return None
    tabs: list[Locator] = []
    for t in outcome.trace:
        el = t.get("el") or {}
        if t["action"] == "click" and t["outcome"] == "ok" and el.get("role") == "tab":
            loc = _locator(el)
            if loc and loc.css not in {x.css for x in tabs}:
                tabs.append(loc)
    await nav.b.goto(nav.site, page, url)
    await nav._consent(page, await nav.observe(page), _NO(), "learn")
    hits_by_state: list[tuple[Locator | None, dict[str, dict[str, Any]]]] = []
    images: FieldLoc | None = None
    for tab in [None, *tabs]:
        if tab is not None:
            obs = await nav.observe(page)
            el = next((e for e in obs["elements"] if e.get("css") == tab.css), None)
            if el is None:
                continue
            try:
                nav._check("click", {"index": el["index"]}, el, page.url, obs["elements"])
                await nav._execute(page, "click", {"index": el["index"]})
            except (GuardrailBlocked, Exception):  # noqa: BLE001 - best effort
                continue
            await nav._settle(page)
        hits_by_state.append((tab, await page.evaluate(_PROBE_FIELDS_JS, values)))
        if images is None and data.image_urls:
            img = await page.evaluate(_PROBE_IMAGES_JS, data.image_urls)
            if img:
                images = FieldLoc(tab=tab, css=img["css"])
    found = choose_locations(hits_by_state)
    if "title" not in found:
        return None
    candidate = ListingRecipe(url=path_pattern([urlparse(url).path]), fields=found, images=images,
                              currency=data.currency)
    # verify on the same page with the replayer (no LLM); keep fields that read back the agent's values
    probe = Replayer(nav, Recipe(site=nav.site, listing=candidate), traced=False)
    await nav.b.goto(nav.site, page, url)
    try:
        got = await probe._read_fields(page, candidate, _NO())
    except ReplayFailed:
        return None
    ok = {f: loc for f, loc in found.items() if _same(f, got.get(f), getattr(data, f))}
    if "title" not in ok:
        return None
    return candidate.model_copy(update={"fields": ok})
