"""Generate the synthetic marketplace: 120 listings across MockBay and TradeLoft, their
images, and the ground-truth ``seed/labels.json`` (SPEC §6). Writes the mock sites'
catalog (``<catalog_dir>/catalog.json`` + ``images/``), which they hot-reload.

Ground truth is written ONLY to labels.json. Catalog entries and image filenames carry
no category information.

    python -m seed.generate [--seed 42] [--llm-text]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel

from bh.brand import BrandAssets, ProductAsset, load_brand_assets
from bh.rules import RuleSet, load_rules
from mocksites.catalog import MockListing, write_catalog
from seed import assets as A
from seed.mark import Placement, composite_mark, sloppy_placement
from seed.variants import apply_random

SEED_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = SEED_DIR / "labels.json"

CATEGORY_COUNTS: dict[str, int] = {
    "authorised": 15, "genuine_resale": 15, "counterfeit_identical": 15, "counterfeit_similar": 15,
    "copied_photos": 15, "grey_import": 10, "modified_goods": 10, "lookalike": 15, "unrelated": 10,
}
HARD_FRACTION = 0.2
DEV_FRACTION = 0.6
# Bulk counterfeit seller cluster (Trading Standards overlay): one trader using the same name,
# images and text on both sites; 8 listings ≥ config thresholds.bulk_listing_threshold (5).
CLUSTER_QUOTA: dict[str, tuple[str, int]] = {"counterfeit_identical": ("mockbay", 6),
                                             "counterfeit_similar": ("tradeloft", 2)}

CRIMINAL_BASIS_RULE = "trading_standards"  # overlay in rules.yaml applied to the bulk cluster
# expected_action / legal_basis come from the category's rule in rules.yaml (labels.json).

UK = ["London, UK", "Manchester, UK", "Leeds, UK", "Bristol, UK", "Glasgow, UK", "Cardiff, UK", "Birmingham, UK"]
EEA = ["Dublin, Ireland", "Lyon, France", "Berlin, Germany", "Rotterdam, Netherlands", "Madrid, Spain"]
FAR_EAST = ["Shenzhen, China", "Guangzhou, China", "Yiwu, China", "Hong Kong", "Istanbul, Turkey"]
GREY_ORIGINS = [("Austin, USA", "USA"), ("Portland, USA", "USA"), ("Osaka, Japan", "Japan"),
                ("Tokyo, Japan", "Japan"), ("Toronto, Canada", "Canada"), ("Sydney, Australia", "Australia")]

REPLICA_KW = ["replica", "1:1", "AAA quality", "mirror quality"]
MISSPELLINGS = ["Vaello", "Vaeloo", "Vae1o", "V-aelo", "Vaelow"]
SUBTLE_MISSPELLINGS = ["Vaelø", "VAEL0", "Vaéló"]
INSPIRED_KW = ["inspired by Vaelo", "Vaelo dupe", "Vaelo-style", "similar to Vaelo"]
MODIFY_KW = ["customised", "relabelled", "refurbished", "repackaged"]

PRIVATE_HANDLES = ["jess_k_82", "tom.collects", "declutter_dan", "mia_sells_stuff", "rob_the_runner", "kat_h",
                   "secondlife_sam", "priya_preloved", "old_kit_olly", "the_attic_finds", "lucy_m_uk", "ben_b",
                   "hannah.h", "greenhome_gwen", "nick_clearout"]
TRADER_HANDLES = ["bright_home_deals", "sunrise_trading_co", "best_value_hub", "dragon_gate_goods",
                  "happyshop2024", "global_sip_supply", "luckystar_mall", "cheapcups_direct", "q_mart_online",
                  "wholesale_world88", "starlight_bazaar", "fastship_store"]
MAKER_HANDLES = ["engrave_it_emma", "custom_cups_co", "giftcraft_studio", "refresh_and_resell", "the_gift_box_co"]
# Grey importers resell genuine goods bought abroad; they are not the counterfeit traders above.
GREY_HANDLES = ["import_direct_store", "worldwide_kitchen_imports", "overseas_finds", "pacific_goods_hub",
                "transatlantic_home"]
DESIGN_HANDLES = ["nordic_living_store", "simple_sip", "urban_outdoor_gear", "hydrate_hub", "minimal_home_co",
                  "trailready", "daily_drinkware"]
GENERAL_HANDLES = ["home_and_more", "bits_and_bobs_uk", "gadget_garage", "the_corner_shop", "pagebound_books",
                   "greenfingers_supplies", "stepright_shoes", "daily_deals_depot"]

BULK_SELLER = {"mockbay": "megadeal_factory", "tradeloft": "MegaDeal Factory"}



@dataclass
class Spec:
    """One generated listing before site ids are assigned."""

    category: str
    hard: bool
    site: str
    title: str
    description: str
    price: float
    seller_name: str
    seller_location: str
    seller_age_days: int
    seller_rating: float
    seller_feedback_count: int
    quantity: int
    condition: str
    site_category: str
    images: list[Image.Image]
    specs: dict[str, str]
    product_slug: str | None = None
    bulk_cluster: bool = False
    signals: list[str] = field(default_factory=list)  # words the LLM rewrite must keep
    image_sources: list[str] = field(default_factory=list)  # provenance, for tests/debugging only


@dataclass
class PoolItem:
    path: Path
    meta: dict[str, Any]


@dataclass
class Pools:
    """Non-gold images: lookalike get-up, unrelated goods, and unbranded blanks for counterfeit photos."""

    lookalike: list[PoolItem]
    unrelated: list[PoolItem]
    blank: list[PoolItem]

    def blanks_for(self, slug: str) -> list[PoolItem]:
        return [b for b in self.blank if b.meta.get("for_product") == slug] or self.blank


def load_pools(other_dir: Path) -> Pools:
    def pool(name: str) -> list[PoolItem]:
        d = other_dir / name
        idx = d / "index.json"
        if not idx.exists():
            raise FileNotFoundError(f"{idx} missing — run `python -m seed.assets` or `python -m seed.import_abo build`")
        items = [PoolItem(d / e["file"], e) for e in json.loads(idx.read_text())["items"]]
        missing = [i.path for i in items if not i.path.exists()]
        if missing or not items:
            raise FileNotFoundError(f"{d}: missing or empty pool ({missing[:3]})")
        return items

    return Pools(pool("lookalike"), pool("unrelated"), pool("blank"))


@dataclass
class Ctx:
    assets: BrandAssets
    rng: random.Random
    pools: Pools
    gold_cache: dict[str, Image.Image] = field(default_factory=dict)

    def gold(self, rel: str) -> Image.Image:
        if rel not in self.gold_cache:
            with Image.open(self.assets.image_path(rel)) as im:
                self.gold_cache[rel] = im.convert("RGB")
        return self.gold_cache[rel].copy()


# ---------------------------------------------------------------- helpers


def _price(rrp: float, lo: float, hi: float, rng: random.Random) -> float:
    p = rrp * rng.uniform(lo, hi)
    return max(0.99, round(p) - 0.01 if p > 3 and rng.random() < 0.6 else round(p, 2))


def _short(p: ProductAsset) -> str:
    return p.name.replace("Vaelo ", "")


def _capacity(p: ProductAsset) -> str:
    m = re.search(r"(\d+(?:ml|L))", p.name)
    return m.group(1) if m else ""


GENERIC_BY_TYPE = {"bottle": "Insulated stainless steel water bottle", "tall_bottle": "Sports water bottle",
                   "mug": "Drinking mug", "flask": "Vacuum flask"}


def _generic(p: ProductAsset) -> str:
    return p.generic_name or GENERIC_BY_TYPE.get(p.product_type or "", "Reusable drinks bottle")


def _genuine_photos(ctx: Ctx, p: ProductAsset, n: int) -> tuple[list[Image.Image], list[str]]:
    picks = ctx.rng.sample(p.images, k=min(n, len(p.images)))
    out = []
    for rel in picks:
        im = ctx.gold(rel)
        if ctx.rng.random() < 0.5:
            im = im.resize((480, 480), Image.LANCZOS)
        out.append(im)
    return out, [f"gold:{r}" for r in picks]


def _copied_photos(ctx: Ctx, p: ProductAsset, n: int, tmin: int, tmax: int) -> tuple[list[Image.Image], list[str]]:
    picks = ctx.rng.sample(p.images, k=min(n, len(p.images)))
    out, src = [], []
    for rel in picks:
        im, names = apply_random(ctx.gold(rel), ctx.rng, tmin, tmax)
        out.append(im)
        src.append(f"gold:{rel}+{'+'.join(names)}")
    return out, src


_GLYPH_FALLBACK = str.maketrans({"Ø": "0", "É": "E", "Ó": "O", "ø": "0", "é": "e", "ó": "o"})


def _fake_render(ctx: Ctx, p: ProductAsset, mark: str, n: int) -> tuple[list[Image.Image], list[str]]:
    """Counterfeit photo: a different (unbranded) product of the same kind with the mark printed on, sloppily."""
    mark = mark.translate(_GLYPH_FALLBACK)  # the bundled font lacks some accented glyphs
    blanks = ctx.pools.blanks_for(p.slug)
    picks = ctx.rng.sample(blanks, k=min(n, len(blanks)))
    out, src = [], []
    for b in picks:
        with Image.open(b.path) as im:
            base = im.convert("RGB")
        out.append(composite_mark(base, mark, ctx.rng.random() < 0.5, sloppy_placement(
            ctx.rng, Placement(anchor_y=0.55, rel_width=0.55))))
        src.append(f"blank:{b.path.parent.name}/{b.path.name}+mark:{mark}")
    return out, src


def _engrave(im: Image.Image, text: str) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    f = A.font(max(12, im.width // 22))
    w = d.textlength(text, font=f)
    d.text((im.width / 2 - w / 2, im.height * 0.72), text, font=f, fill=(250, 250, 250))
    return im


def _specs(brand: str, p: ProductAsset | None, colour: str = "") -> dict[str, str]:
    s = {"Brand": brand}
    if p is not None:
        s["Capacity"] = _capacity(p)
        s["Material"] = p.material or "Stainless steel"
    if colour:
        s["Colour"] = colour
    return s




# ---------------------------------------------------------------- per-category builders


def build_authorised(ctx: Ctx, site: str, hard: bool) -> Spec:
    r = ctx.rng
    p = r.choice(ctx.assets.products)
    seller = r.choice([s.seller_name for s in ctx.assets.authorised_sellers if s.platform == site])
    imgs, src = _genuine_photos(ctx, p, r.randint(2, 3))
    if hard:  # deep discount + large stock from an authorised retailer: looks suspicious, is lawful
        price, qty = _price(p.rrp, 0.55, 0.7, r), r.randint(150, 400)
        title = f"{p.name} - CLEARANCE, limited stock"
        desc = (f"End-of-line clearance on the {p.name}. Genuine stock from an authorised Vaelo retailer.\n"
                f"Double-walled stainless steel, keeps drinks cold for 24 hours.")
    else:
        price, qty = _price(p.rrp, 0.85, 1.0, r), r.randint(10, 120)
        title = r.choice([f"{p.name} - Official Store", f"{p.name} | Free UK delivery", f"Genuine {p.name}"])
        desc = (f"The {p.name} from the official Vaelo range. Double-walled stainless steel, BPA-free lid.\n"
                f"Sold by an authorised Vaelo retailer. 2-year manufacturer warranty.")
    return Spec("authorised", hard, site, title, desc, price, seller, r.choice(UK[:3]), r.randint(1500, 3200),
                round(r.uniform(4.7, 5.0), 1), r.randint(800, 9000), qty, "New", "Drinkware", imgs,
                _specs("Vaelo", p, p.colour_name or ""), p.slug, image_sources=src)


def build_genuine_resale(ctx: Ctx, site: str, hard: bool) -> Spec:
    r = ctx.rng
    p = r.choice(ctx.assets.products)
    imgs, src = _genuine_photos(ctx, p, r.randint(1, 2))
    if hard:  # new, near-RRP, unauthorised UK seller: lawful resale (exhaustion) despite looking "official"
        price, cond = _price(p.rrp, 0.9, 1.0, r), "New"
        title = f"{p.name} - brand new, unwanted gift"
        desc = (f"Brand new {p.name}, received as a gift but I already have one. Still in the original "
                f"Vaelo box. Collection or post.")
        signals = ["Vaelo"]
    else:
        price, cond = _price(p.rrp, 0.6, 0.85, r), r.choice(["Used", "Pre-owned"])
        title = r.choice([f"Vaelo {_short(p)} - used, good condition", f"Pre-owned {p.name}",
                          f"{p.name} used a handful of times"])
        desc = (f"Selling my {p.name}. Pre-owned, used for a few months, light scuffs on the base.\n"
                f"Genuine Vaelo, bought from a high-street shop. No box.")
        signals = ["Vaelo", "used" if "used" in title else "Pre-owned"]
    return Spec("genuine_resale", hard, site, title, desc, price, r.choice(PRIVATE_HANDLES), r.choice(UK + EEA[:2]),
                r.randint(200, 3000), round(r.uniform(4.4, 5.0), 1), r.randint(3, 300), 1, cond, "Drinkware",
                imgs, _specs("Vaelo", p), p.slug, signals=signals, image_sources=src)


def build_counterfeit_identical(ctx: Ctx, site: str, hard: bool, cluster: dict[str, Any] | None = None) -> Spec:
    r = ctx.rng
    p = cluster["product"] if cluster else r.choice(ctx.assets.products)
    if cluster:
        imgs, src = cluster["images"], cluster["sources"]
    elif r.random() < 0.6:
        imgs, src = _copied_photos(ctx, p, r.randint(1, 2), 1, 2)
    else:
        imgs, src = _fake_render(ctx, p, "VAELO", r.randint(1, 2))
    if hard:  # no replica words, price just around the threshold; needs the combined signals
        price = _price(p.rrp, 0.42, 0.55, r)
        title = f"{p.name} - new, fast dispatch"
        desc = (f"{p.name}. Brand new. Wholesale quantities available, message for bulk pricing.\n"
                f"Ships from our warehouse within 24h.")
        signals = ["Vaelo", "Wholesale"]
    else:
        kw = r.choice(REPLICA_KW)
        price = _price(p.rrp, 0.1, 0.4, r)
        title = r.choice([f"Vaelo {_short(p)} {kw}", f"{kw.upper()} Vaelo {_short(p)} best price",
                          f"Vaelo {_short(p)} - {kw} - bulk available"])
        desc = (f"High quality Vaelo {_short(p).lower()} {kw}. Same as original, nobody can tell the difference.\n"
                f"Bulk orders welcome. Free shipping on 10+.")
        signals = ["Vaelo", kw]
    if cluster:
        title, desc = cluster["title"], cluster["desc"]
        signals = cluster["signals"]
    loc = r.choice(FAR_EAST) if not (hard and r.random() < 0.5) else r.choice(UK)
    return Spec("counterfeit_identical", hard, site, title, desc, price,
                cluster["seller"] if cluster else r.choice(TRADER_HANDLES), cluster["location"] if cluster else loc,
                r.randint(3, 150), round(r.uniform(3.0, 4.4), 1), r.randint(0, 120), r.randint(40, 600), "New",
                "Drinkware", imgs, _specs("Vaelo", p), p.slug, bulk_cluster=bool(cluster), signals=signals,
                image_sources=src)


def build_counterfeit_similar(ctx: Ctx, site: str, hard: bool, cluster: dict[str, Any] | None = None) -> Spec:
    r = ctx.rng
    p = cluster["product"] if cluster else r.choice(ctx.assets.products)
    if hard:
        mis = r.choice(SUBTLE_MISSPELLINGS)
        price = _price(p.rrp, 0.42, 0.58, r)
        imgs, src = _fake_render(ctx, p, mis.upper(), r.randint(1, 2))
        title = f"{mis} {_short(p)} - official design"
        desc = (f"{mis} {_short(p).lower()}, official design, double-walled steel.\n"
                f"Same factory finish, gift ready.")
        signals = [mis, "official"]
    else:
        mis = r.choice(MISSPELLINGS)
        price = _price(p.rrp, 0.1, 0.42, r)
        if r.random() < 0.5:
            imgs, src = _copied_photos(ctx, p, r.randint(1, 2), 1, 3)
        else:
            imgs, src = _fake_render(ctx, p, mis.upper(), r.randint(1, 2))
        extra = r.choice(["official", "original", r.choice(REPLICA_KW)])
        title = f"{mis} {_short(p)} {extra}"
        desc = (f"{mis} {_short(p).lower()} - {extra} quality. Looks exactly like the real thing.\n"
                f"Wholesale and retail. Fast shipping.")
        signals = [mis, extra]
    if cluster:
        imgs, src = cluster["images"], cluster["sources"]
        title = f"{cluster['misspelling']} {_short(p)} - {cluster['kw']}"
        desc = cluster["desc"].replace("Vaelo", cluster["misspelling"])
        signals = [cluster["misspelling"], cluster["kw"]]
    return Spec("counterfeit_similar", hard, site, title, desc, price,
                cluster["seller"] if cluster else r.choice(TRADER_HANDLES),
                cluster["location"] if cluster else r.choice(FAR_EAST + UK[:1]), r.randint(3, 200),
                round(r.uniform(3.0, 4.5), 1), r.randint(0, 150), r.randint(20, 500), "New", "Drinkware", imgs,
                _specs(mis if not cluster else cluster["misspelling"], p), p.slug, bulk_cluster=bool(cluster),
                signals=signals, image_sources=src)



def build_copied_photos(ctx: Ctx, site: str, hard: bool) -> Spec:
    r = ctx.rng
    p = r.choice(ctx.assets.products)
    generic = _generic(p)
    if hard:
        imgs, src = _copied_photos(ctx, p, 1, 3, 4)
        title = f"Premium {generic.lower()} {_capacity(p)} - gift idea"
        desc = (f"Premium {generic.lower()}, {_capacity(p)}. Leak-proof lid, keeps drinks hot or cold.\n"
                f"Unbranded. Colour as pictured.")
    else:
        imgs, src = _copied_photos(ctx, p, r.randint(1, 2), 1, 3)
        title = f"{generic} {_capacity(p)}"
        desc = f"{generic}, {_capacity(p)}. Double wall, BPA free. Generic product, ships in plain packaging."
    return Spec("copied_photos", hard, site, title, desc, _price(p.rrp, 0.15, 0.5, r), r.choice(TRADER_HANDLES),
                r.choice(FAR_EAST + UK), r.randint(5, 400), round(r.uniform(3.4, 4.8), 1), r.randint(0, 400),
                r.randint(5, 300), "New", "Drinkware", imgs, _specs("Unbranded", p), p.slug,
                signals=["Unbranded"] if hard else ["Generic"], image_sources=src)


def build_grey_import(ctx: Ctx, site: str, hard: bool) -> Spec:
    r = ctx.rng
    p = r.choice(ctx.assets.products)
    imgs, src = _genuine_photos(ctx, p, r.randint(1, 2))
    loc, country = r.choice(GREY_ORIGINS)
    if hard:  # UK-based seller; foreign origin only in the wording
        title = f"{p.name} - new in box"
        desc = (f"Genuine {p.name}, new in box. Imported from {country} - {country} retail packaging, "
                f"instructions not in English.\nDispatched from our UK unit.")
        loc = r.choice(UK)
        signals = ["Vaelo", f"Imported from {country}"]
    else:
        title = r.choice([f"{p.name} ({country} version)", f"{p.name} - {country} edition, international shipping"])
        desc = (f"Genuine {p.name} bought from a {country} retailer. Ships internationally from {loc}.\n"
                f"Import duties may apply.")
        signals = ["Vaelo", country]
    return Spec("grey_import", hard, site, title, desc, _price(p.rrp, 0.8, 1.05, r), r.choice(GREY_HANDLES),
                loc, r.randint(300, 2500), round(r.uniform(4.3, 4.9), 1), r.randint(40, 900), r.randint(1, 12),
                "New", "Drinkware", imgs, _specs("Vaelo", p), p.slug, signals=signals, image_sources=src)


def build_modified_goods(ctx: Ctx, site: str, hard: bool) -> Spec:
    r = ctx.rng
    p = r.choice(ctx.assets.products)
    imgs, src = _genuine_photos(ctx, p, r.randint(1, 2))
    name = r.choice(["EMMA", "JAKE", "MUM", "TEAM 7", "ALEX"])
    if hard:
        title = f"{p.name} gift set"
        desc = (f"Genuine {p.name}, repackaged by us into a luxury gift box with ribbon and card.\n"
                f"Original Vaelo packaging removed.")
        signals = ["Vaelo", "repackaged"]
    else:
        kw = r.choice(MODIFY_KW)
        imgs = [_engrave(im, name) for im in imgs]
        src = [s + "+engraved" for s in src]
        title = f"{kw.capitalize()} Vaelo {_short(p)} - personalised with your name"
        desc = (f"Genuine Vaelo {_short(p).lower()}, {kw} in our workshop with laser engraving.\n"
                f"Send us the name you want after purchase.")
        signals = ["Vaelo", kw]
    return Spec("modified_goods", hard, site, title, desc, _price(p.rrp, 0.75, 1.3, r), r.choice(MAKER_HANDLES),
                r.choice(UK + EEA), r.randint(100, 2000), round(r.uniform(4.4, 5.0), 1), r.randint(10, 600),
                r.randint(1, 30), r.choice(["New other", "Refurbished"]) if not hard else "New other", "Drinkware",
                imgs, _specs("Vaelo (modified)", p), p.slug, signals=signals, image_sources=src)


def build_lookalike(ctx: Ctx, site: str, hard: bool) -> Spec:
    r = ctx.rng
    picks = r.sample(ctx.pools.lookalike, k=r.randint(1, 2))
    imgs = [Image.open(it.path).convert("RGB") for it in picks]
    if hard:  # mentions the brand only to disclaim it
        title = "Double-wall steel bottle - NOT Vaelo but just as good"
        desc = "Not affiliated with Vaelo. Our own design with the same capacity and finish at half the price."
        signals = ["Vaelo"]
    elif r.random() < 0.6:
        kw = r.choice(INSPIRED_KW)
        title = r.choice([f"Steel bottle {kw}", f"Travel mug - {kw}", f"{kw.capitalize()} insulated flask"])
        desc = f"Insulated drinkware, {kw}. Minimalist look, matte finish, own brand."
        signals = [kw]
    else:
        title = r.choice(["Minimalist insulated bottle, Scandinavian style", "Matte travel mug with lid",
                          "Everyday steel flask, 1L"])
        desc = "Our own-brand insulated drinkware. Clean minimalist design, powder-coated finish."
        signals = []
    return Spec("lookalike", hard, site, title, desc, round(r.uniform(9, 22), 2), r.choice(DESIGN_HANDLES),
                r.choice(UK + EEA + FAR_EAST), r.randint(50, 2500), round(r.uniform(4.0, 4.9), 1),
                r.randint(5, 900), r.randint(5, 200), "New", "Drinkware", imgs, _specs("Own brand", None),
                None, signals=signals, image_sources=[f"other:lookalike/{it.path.name}" for it in picks])


def build_unrelated(ctx: Ctx, site: str, hard: bool) -> Spec:
    r = ctx.rng
    it = r.choice(ctx.pools.unrelated)
    cat = it.meta["site_category"]
    # hard: brand-like word in the title (Vale / Velo / Vaal) → fuzzy brand false positive
    title = it.meta["title_hard"] if hard else it.meta["title"]
    desc = f"{title}. Good quality, quick dispatch."
    return Spec("unrelated", hard, site, title, desc, round(r.uniform(6, 80), 2), r.choice(GENERAL_HANDLES),
                r.choice(UK + EEA), r.randint(30, 3000), round(r.uniform(4.0, 5.0), 1), r.randint(0, 2000),
                r.randint(1, 50), r.choice(["New", "Used"]), cat, [Image.open(it.path).convert("RGB")],
                {"Brand": "Unbranded"}, None, image_sources=[f"other:unrelated/{it.path.name}"])


BUILDERS = {
    "authorised": build_authorised, "genuine_resale": build_genuine_resale,
    "counterfeit_identical": build_counterfeit_identical, "counterfeit_similar": build_counterfeit_similar,
    "copied_photos": build_copied_photos, "grey_import": build_grey_import, "modified_goods": build_modified_goods,
    "lookalike": build_lookalike, "unrelated": build_unrelated,
}


# ---------------------------------------------------------------- assembly


def _bulk_cluster(ctx: Ctx) -> dict[str, Any]:
    """Shared images/text for the bulk counterfeit seller cluster (Trading Standards overlay)."""
    r = ctx.rng
    p = ctx.assets.products[0]
    imgs, src = _copied_photos(ctx, p, 2, 1, 2)
    kw = "1:1"
    return {"product": p, "images": imgs, "sources": [s + "+cluster" for s in src], "kw": kw,
            "misspelling": "Vaello", "location": "Yiwu, China",
            "title": f"Vaelo {_short(p)} {kw} - factory direct, bulk",
            "desc": (f"Vaelo {_short(p).lower()} {kw}. Factory direct, 500+ units in stock.\n"
                     f"Wholesale price for orders over 50. Fast shipping worldwide."),
            "signals": ["Vaelo", kw]}


def plan(ctx: Ctx) -> list[Spec]:
    r = ctx.rng
    specs: list[Spec] = []
    cluster = _bulk_cluster(ctx)
    for cat, n in CATEGORY_COUNTS.items():
        n_hard = round(n * HARD_FRACTION)
        hard_flags = [True] * n_hard + [False] * (n - n_hard)
        r.shuffle(hard_flags)
        sites = ["mockbay", "tradeloft"] * (n // 2 + 1)
        r.shuffle(sites)
        quota_site, quota = CLUSTER_QUOTA.get(cat, ("", 0))
        for i, hard in enumerate(hard_flags):
            if not hard and quota > 0:
                quota -= 1
                c = {**cluster, "seller": BULK_SELLER[quota_site]}
                specs.append(BUILDERS[cat](ctx, quota_site, hard, c))  # type: ignore[operator]
            else:
                specs.append(BUILDERS[cat](ctx, sites[i], hard))
    return specs


def _assign_ids(specs: list[Spec], rng: random.Random) -> list[str]:
    used: set[str] = set()
    ids = []
    for s in specs:
        while True:
            lid = str(rng.randint(200000, 899999)) if s.site == "mockbay" else f"TL-{rng.randint(10000, 99999)}"
            if lid not in used:
                used.add(lid)
                ids.append(lid)
                break
    return ids


def _split(keys_by_stratum: dict[tuple[str, str], list[str]], rng: random.Random) -> dict[str, str]:
    out: dict[str, str] = {}
    for stratum in sorted(keys_by_stratum):
        keys = sorted(keys_by_stratum[stratum])
        rng.shuffle(keys)
        n_dev = int(len(keys) * DEV_FRACTION + 0.5)
        for i, k in enumerate(keys):
            out[k] = "dev" if i < n_dev else "test"
    return out


def listing_path(site: str, lid: str) -> str:
    return f"/mockbay/item/{lid}" if site == "mockbay" else f"/tradeloft/listing/{lid}"


# ---------------------------------------------------------------- optional LLM text


class _RewriteItem(BaseModel):
    index: int
    title: str
    description: str


class _Rewrite(BaseModel):
    items: list[_RewriteItem]


REWRITE_SYSTEM = (
    "You write realistic, varied marketplace listing text for a synthetic test dataset about a fictional "
    "drinkware brand. Rewrite each listing's title and description in a different seller voice while keeping "
    "the same facts, price-free. Every phrase listed under must_keep must appear verbatim in the title or "
    "description. Do not add new brand names, claims of authenticity or legal wording that is not in the input.")


async def rewrite_texts(specs: list[Spec], llm: Any) -> int:
    """One small-model call per category; rewritten text is kept only if all signal phrases survive."""
    changed = 0
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(specs):
        by_cat[s.category].append(i)
    for cat, idxs in by_cat.items():
        payload = [{"index": j, "title": specs[i].title, "description": specs[i].description,
                    "must_keep": specs[i].signals} for j, i in enumerate(idxs)]
        res = await llm.complete_json(role="small", purpose=f"seed_text:{cat}", schema=_Rewrite,
                                      system=REWRITE_SYSTEM,
                                      messages=[{"role": "user", "content": json.dumps(payload)}])
        for item in res.parsed.items:
            if not 0 <= item.index < len(idxs):
                continue
            s = specs[idxs[item.index]]
            text = f"{item.title}\n{item.description}".lower()
            if all(sig.lower() in text for sig in s.signals) and item.title.strip():
                s.title, s.description = item.title.strip()[:120], item.description.strip()[:1500]
                changed += 1
    return changed


# ---------------------------------------------------------------- main entry


@dataclass
class GenerateResult:
    catalog_path: Path
    labels_path: Path
    n_listings: int
    counts: dict[str, int]
    n_hard: int
    split_counts: dict[str, int]
    llm_rewritten: int = 0


def generate(brand_dir: Path, other_dir: Path, catalog_dir: Path, labels_path: Path, seed: int = 42,
             llm: Any | None = None, rules: RuleSet | None = None) -> GenerateResult:
    if rules is None:
        from bh.rules import get_rules

        rules = get_rules()
    A.build_brand(brand_dir)  # no-ops when assets exist
    A.build_other(other_dir)
    assets = load_brand_assets(brand_dir)
    pools = load_pools(other_dir)

    rng = random.Random(seed)
    ctx = Ctx(assets, rng, pools)
    specs = plan(ctx)
    llm_changed = asyncio.run(rewrite_texts(specs, llm)) if llm is not None else 0
    ids = _assign_ids(specs, rng)

    images_dir = catalog_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")):
        old.unlink()
    (catalog_dir / "clip_cache.npz").unlink(missing_ok=True)

    listings: list[MockListing] = []
    labels: dict[str, dict[str, Any]] = {}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s, lid in zip(specs, ids, strict=True):
        names = []
        for k, im in enumerate(s.images):
            fn = f"{lid.lower()}_{k + 1}.jpg"
            im.convert("RGB").save(images_dir / fn, "JPEG", quality=88)
            names.append(fn)
        listings.append(MockListing(
            id=lid, site=s.site, title=s.title, description=s.description, price=s.price, currency="GBP",  # type: ignore[arg-type]
            seller_name=s.seller_name, seller_location=s.seller_location, seller_age_days=s.seller_age_days,
            seller_rating=s.seller_rating, seller_feedback_count=s.seller_feedback_count, quantity=s.quantity,
            site_category=s.site_category, condition=s.condition, images=names, specs=s.specs,
            listed_days_ago=rng.randint(0, 90)))
        key = f"{s.site}:{lid}"
        difficulty = "hard" if s.hard else "easy"
        strata[(s.category, difficulty)].append(key)
        rule = rules.rule_for_category(s.category)
        basis = [rule.legal_basis.citation()] if rule.legal_basis else []
        secondary: list[str] = []
        if s.bulk_cluster:
            overlay = next(o for o in rules.overlays if o.id == CRIMINAL_BASIS_RULE)
            basis.append(overlay.legal_basis.citation())
            secondary.append(str(overlay.action))
        expected = str(rule.action)
        labels[key] = {"category": s.category, "rule_id": rule.id, "legal_basis": basis,
                       "expected_action": expected, "should_report": expected == "platform_report",
                       "expected_secondary_actions": secondary, "difficulty": difficulty, "split": "", "site": s.site, "url_path": listing_path(s.site, lid),
                       "product": s.product_slug, "bulk_cluster": s.bulk_cluster,
                       "image_sources": s.image_sources}
    for key, split in _split(strata, rng).items():
        labels[key]["split"] = split

    rng.shuffle(listings)  # catalog order carries no category information
    catalog_path = write_catalog(catalog_dir, listings)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(dict(sorted(labels.items())), indent=2), encoding="utf-8")
    return GenerateResult(catalog_path, labels_path, len(listings), dict(Counter(s.category for s in specs)),
                          sum(s.hard for s in specs), dict(Counter(v["split"] for v in labels.values())), llm_changed)


def main() -> None:
    from bh.config import get_config

    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand-dir", type=Path, default=SEED_DIR / "brand")
    ap.add_argument("--other-dir", type=Path, default=SEED_DIR / "other_images")
    ap.add_argument("--out", type=Path, default=cfg.paths.resolve(cfg.mocksites.catalog_dir))
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm-text", action="store_true", default=cfg.llm.enable_llm_seed_text,
                    help="rewrite titles/descriptions with the small model (one call per category)")
    args = ap.parse_args()
    llm = None
    if args.llm_text:
        from bh.llm import LLMClient

        llm = LLMClient(cfg)
    res = generate(args.brand_dir, args.other_dir, args.out, args.labels, args.seed, llm)
    print(f"wrote {res.n_listings} listings → {res.catalog_path}")
    print(f"labels → {res.labels_path}  (hard={res.n_hard}, split={res.split_counts})")
    print("mix:", res.counts)
    if args.llm_text:
        print(f"LLM-rewritten listings: {res.llm_rewritten}")


if __name__ == "__main__":
    main()
