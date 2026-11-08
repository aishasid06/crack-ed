"""Mix real product listings into the mock marketplaces as background noise.

Real drinkware listings from the Amazon Berkeley Objects dataset (ABO, CC BY 4.0) are added to the seeded
catalog. None of them is a Vaelo product, so the correct outcome for every one is "ignore". Running the
pipeline and the evaluation on them measures false alarms on data the system was not built around.

    python -m seed.real_listings [--n 300] [--seed 11]

- Real: title, description (ABO bullet points) and photos.
- Generated: price, seller, location, account age, rating, quantity. ABO has none of these, so they are drawn
  from one neutral range for every real listing, not from the synthetic categories' ranges.
- The 120 synthetic listings are kept as they are. Rerunning replaces earlier real listings.

Writes the catalog (the mock sites hot-reload it) and ``seed/labels_real.json``: the labels from
``seed/labels.json`` plus one ``unrelated`` / ``ignore`` label per real listing (difficulty ``real``, split
60/40 dev/test). ``seed/labels.json`` is left untouched as the synthetic-only baseline.

Evaluate with ``python -m eval.evaluate --labels seed/labels_real.json``. Do not refit the scorer on these
labels before the first comparison, or the before/after numbers stop being comparable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from mocksites.catalog import MockListing, load_catalog, write_catalog
from seed import import_abo as I
from seed.generate import DEFAULT_LABELS, EEA, FAR_EAST, UK, _split, listing_path

SEED_DIR = Path(__file__).resolve().parent
REAL_LABELS = SEED_DIR / "labels_real.json"
IMAGES_PER_LISTING = 2

# Drinkware product types and names. Coffee pods ("K-Cups") are filed under COFFEE and left out.
TYPES = {"BOTTLE", "DRINKING_CUP", "THERMOS", "KITCHEN", "JAR"}
NAME_RE = re.compile(r"bottle|tumbler|\bmugs?\b|flask|thermos|\bcups?\b|glass(es)?\b|beaker|carafe|\bjug\b|"
                     r"sipper|stein|goblet", re.I)
EXCLUDE_RE = re.compile(r"lid only|kadhai|lunch box|casserole|container|canister|\bjar\b|cooker|\bpan\b|kettle|"
                        r"grinder|dispenser|brush|rack|opener|k-cup|coffee pods?", re.I)

# Neutral seller handles, distinct from the synthetic generator's handle pools and the authorised sellers.
HANDLE_A = ["northside", "harbour", "maple", "riverside", "oakwood", "daily", "brightwater", "kingsway", "meadow",
            "summit", "copperfield", "westgate", "lakeside", "parkview", "highfield"]
HANDLE_B = ["home_goods", "kitchen_supply", "living", "traders", "market", "essentials", "emporium", "store",
            "housewares", "outlet"]
US = ["Seattle, USA", "Chicago, USA", "Toronto, Canada"]


@dataclass(frozen=True)
class RealItem:
    item_id: str
    product_type: str
    brand: str
    name: str
    description: str
    image_ids: list[str]


def _en_values(rec: dict[str, Any], key: str) -> list[str]:
    return [str(v.get("value", "")).strip() for v in rec.get(key) or []
            if str(v.get("language_tag", "")).startswith("en") and str(v.get("value", "")).strip()]


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0].rstrip(",;:-") + "…"


def candidates(cache: Path, exclude_items: set[str]) -> list[RealItem]:
    """English-titled ABO drinkware listings, one per product name, sorted by item id."""
    seen: dict[str, RealItem] = {}
    for shard in I.LISTING_SHARDS:
        with gzip.open(cache / "metadata" / shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pt = (r.get("product_type") or [{}])[0].get("value", "")
                if r["item_id"] in exclude_items or not r.get("main_image_id") or pt not in TYPES:
                    continue
                names = _en_values(r, "item_name")
                if not names or not NAME_RE.search(names[0]) or EXCLUDE_RE.search(names[0]):
                    continue
                key = re.sub(r"\(renewed\)|\W+", " ", names[0].lower()).strip()[:50]
                if key in seen:
                    continue
                bullets = _en_values(r, "bullet_point")
                brand = (_en_values(r, "brand") or [""])[0]
                seen[key] = RealItem(r["item_id"], pt, brand, _clip(names[0], 140),
                                     _clip(" ".join(bullets[:4]) or names[0], 600),
                                     [r["main_image_id"], *r.get("other_image_id", [])][:IMAGES_PER_LISTING])
    return sorted(seen.values(), key=lambda i: i.item_id)


def _price(rng: random.Random) -> float:
    p = rng.uniform(4, 45)  # one neutral retail range for every real listing
    return round(p) - 0.01 if rng.random() < 0.6 else round(p, 2)


def _location(rng: random.Random) -> str:
    pool = rng.choices([UK, EEA, FAR_EAST, US], weights=[60, 15, 15, 10])[0]
    return rng.choice(pool)


def _new_id(site: str, used: set[str], rng: random.Random) -> str:
    while True:
        lid = str(rng.randint(200000, 899999)) if site == "mockbay" else f"TL-{rng.randint(10000, 99999)}"
        if lid not in used:
            used.add(lid)
            return lid


def add_real_listings(catalog_dir: Path, base_labels: Path = DEFAULT_LABELS, out_labels: Path = REAL_LABELS,
                      n: int = 300, seed: int = 11, cache: Path | None = None) -> dict[str, Any]:
    cache = cache or I.cache_dir()
    I.ensure_metadata(cache)
    sel = I.load_selection()
    used_abo = {g["abo_item_id"] for g in sel["gold"]} | {e["abo_item_id"] for p in I.POOLS for e in sel[p]}
    rng = random.Random(seed)
    pool = candidates(cache, used_abo)
    rng.shuffle(pool)
    index = I.load_image_index(cache)
    chosen = [it for it in pool if any(i in index for i in it.image_ids)][:n]
    paths = I.download_images(cache, index, [i for it in chosen for i in it.image_ids], "original")

    labels: dict[str, dict[str, Any]] = json.loads(base_labels.read_text(encoding="utf-8"))
    synthetic_paths = {v["url_path"] for v in labels.values()}
    catalog = load_catalog(catalog_dir)
    # Bazaarly listings belong to seed/bazaarly.py (their own labels group); keep them as they are.
    keep = [l for l in catalog.listings if listing_path(l.site, l.id) in synthetic_paths or l.site == "bazaarly"]
    images_dir = catalog_dir / "images"
    for old in catalog.listings:  # real listings from an earlier run of this script
        if listing_path(old.site, old.id) not in synthetic_paths and old.site != "bazaarly":
            for img in old.images:
                (images_dir / img).unlink(missing_ok=True)
    used_ids = {l.id for l in keep}

    from bh.rules import get_rules
    from seed.mark import prepare_square

    rule = get_rules().rule_for_category("unrelated")
    added: list[MockListing] = []
    real_keys: list[str] = []
    types: Counter[str] = Counter()
    for k, it in enumerate(chosen):
        site = "mockbay" if k % 2 == 0 else "tradeloft"
        lid = _new_id(site, used_ids, rng)
        names = []
        for j, iid in enumerate(i for i in it.image_ids if i in paths):
            fn = f"{lid.lower()}_{j + 1}.jpg"
            with Image.open(paths[iid]) as raw:
                prepare_square(raw, I.OTHER_SIZE).save(images_dir / fn, "JPEG", quality=88)
            names.append(fn)
        if not names:
            continue
        specs = {"Brand": it.brand or "Unbranded", "Type": it.product_type.replace("_", " ").title()}
        added.append(MockListing(
            id=lid, site=site, title=it.name, description=it.description, price=_price(rng),  # type: ignore[arg-type]
            currency="GBP", seller_name=f"{rng.choice(HANDLE_A)}_{rng.choice(HANDLE_B)}",
            seller_location=_location(rng), seller_age_days=rng.randint(60, 4000),
            seller_rating=round(rng.uniform(3.9, 5.0), 1), seller_feedback_count=rng.randint(5, 5000),
            quantity=rng.randint(1, 60), site_category="Drinkware",
            condition="New" if rng.random() < 0.9 else "Used", images=names, specs=specs,
            listed_days_ago=rng.randint(0, 90)))
        key = f"{site}:{lid}"
        real_keys.append(key)
        types[it.product_type] += 1
        labels[key] = {"category": "unrelated", "rule_id": rule.id, "legal_basis": [],
                       "expected_action": str(rule.action), "should_report": False, "expected_secondary_actions": [],
                       "difficulty": "real", "split": "", "site": site, "url_path": listing_path(site, lid),
                       "product": None, "bulk_cluster": False, "image_sources": [f"abo:{i}" for i in it.image_ids],
                       "source": "abo", "abo_item_id": it.item_id}
    for key, split in _split({("unrelated", "real"): real_keys}, rng).items():
        labels[key]["split"] = split

    listings = keep + added
    rng.shuffle(listings)  # catalog order carries no information
    write_catalog(catalog_dir, listings)
    (catalog_dir / "clip_cache.npz").unlink(missing_ok=True)  # visual search re-embeds the new catalog
    out_labels.write_text(json.dumps(dict(sorted(labels.items())), indent=2), encoding="utf-8")
    return {"candidates": len(pool), "added": len(added), "synthetic_kept": len(keep),
            "by_site": dict(Counter(l.site for l in added)),
            "split": dict(Counter(labels[k]["split"] for k in real_keys)),
            "by_type": dict(types),
            "labels": str(out_labels)}


def main() -> None:
    from bh.config import get_config

    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="real listings to add (fewer if ABO has fewer)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--catalog", type=Path, default=cfg.paths.resolve(cfg.mocksites.catalog_dir))
    args = ap.parse_args()
    print(json.dumps(add_real_listings(args.catalog, n=args.n, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
