"""Populate Bazaarly, the "unseen" marketplace browsed by the agentic navigator (SPEC Phase 11 item 2).

- Real ABO drinkware listings (via seed/real_listings.py's candidate selection), all ``unrelated`` / ``ignore``.
  Items already used for seed photos or for the real listings on EchoBazaar / TradeLoft are skipped.
  Without the ABO cache, only the Vaelo cases are added.
- A handful of seeded Vaelo cases built with the synthetic generator's builders (counterfeits, copied photos,
  a lookalike, a genuine resale, a grey import).

Labels go to a separate group, ``seed/labels_bazaarly.json`` (``group: "bazaarly"``), so the existing
``seed/labels.json`` / ``seed/labels_real.json`` metrics are unchanged. Other sites' listings are kept;
re-running replaces only Bazaarly's.

    python -m seed.bazaarly [--real 40] [--seed 23]
    python -m eval.evaluate --labels seed/labels_bazaarly.json --split test
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from bh.brand import load_brand_assets
from bh.rules import get_rules
from mocksites.catalog import MockListing, load_catalog, write_catalog
from seed import generate as G
from seed import real_listings as RL

SEED_DIR = Path(__file__).resolve().parent
LABELS_OUT = SEED_DIR / "labels_bazaarly.json"
SITE = "bazaarly"
# Seeded Vaelo cases on Bazaarly (no authorised sellers exist for this site).
VAELO_CASES = ["counterfeit_identical", "counterfeit_identical", "counterfeit_similar", "copied_photos",
               "copied_photos", "lookalike", "genuine_resale", "grey_import"]


def _new_id(used: set[str], rng: random.Random) -> str:
    while True:
        lid = f"BZ-{rng.randint(10000, 99999)}"
        if lid not in used:
            used.add(lid)
            return lid


def _label(category: str, rule: Any, lid: str, difficulty: str, extra: dict[str, Any]) -> dict[str, Any]:
    basis = [rule.legal_basis.citation()] if rule.legal_basis else []
    return {"category": category, "rule_id": rule.id, "legal_basis": basis, "expected_action": str(rule.action),
            "should_report": str(rule.action) == "platform_report", "expected_secondary_actions": [],
            "difficulty": difficulty, "split": "", "site": SITE, "url_path": G.listing_path(SITE, lid),
            "group": "bazaarly", **extra}


def _abo_used_elsewhere() -> set[str]:
    used: set[str] = set()
    real = SEED_DIR / "labels_real.json"
    if real.exists():
        used |= {v["abo_item_id"] for v in json.loads(real.read_text()).values() if v.get("abo_item_id")}
    return used


def add_bazaarly(catalog_dir: Path, brand_dir: Path = SEED_DIR / "brand", other_dir: Path = SEED_DIR / "other_images",
                 labels_out: Path = LABELS_OUT, n_real: int = 40, seed: int = 23,
                 cache: Path | None = None, cases: list[str] | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    rules = get_rules()
    catalog = load_catalog(catalog_dir)
    images_dir = catalog_dir / "images"
    keep = [l for l in catalog.listings if l.site != SITE]
    for old in catalog.listings:  # Bazaarly listings from an earlier run of this script
        if old.site == SITE:
            for img in old.images:
                (images_dir / img).unlink(missing_ok=True)
    used_ids = {l.id for l in catalog.listings}
    added: list[MockListing] = []
    labels: dict[str, dict[str, Any]] = {}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)

    # seeded Vaelo cases, built exactly like the synthetic catalog's
    ctx = G.Ctx(load_brand_assets(brand_dir), rng, G.load_pools(other_dir))
    cases = list(cases or VAELO_CASES)
    for cat in cases:
        s = G.BUILDERS[cat](ctx, SITE, False)
        lid = _new_id(used_ids, rng)
        names = []
        for k, im in enumerate(s.images):
            fn = f"{lid.lower()}_{k + 1}.jpg"
            im.convert("RGB").save(images_dir / fn, "JPEG", quality=88)
            names.append(fn)
        added.append(MockListing(
            id=lid, site=SITE, title=s.title, description=s.description, price=s.price, currency="GBP",
            seller_name=s.seller_name, seller_location=s.seller_location, seller_age_days=s.seller_age_days,
            seller_rating=s.seller_rating, seller_feedback_count=s.seller_feedback_count, quantity=s.quantity,
            site_category=s.site_category, condition=s.condition, images=names, specs=s.specs,
            listed_days_ago=rng.randint(0, 90)))
        key = f"{SITE}:{lid}"
        labels[key] = _label(cat, rules.rule_for_category(cat), lid, "easy",
                             {"product": s.product_slug, "bulk_cluster": False, "image_sources": s.image_sources,
                              "source": "seeded"})
        strata[(cat, "easy")].append(key)

    # real ABO listings as background
    n_abo = 0
    try:
        from seed import import_abo as I
        from seed.mark import prepare_square

        cache = cache or I.cache_dir()
        if n_real > 0 and (cache / "metadata" / "images.csv.gz").exists():
            sel = I.load_selection()
            exclude = ({g["abo_item_id"] for g in sel["gold"]} | {e["abo_item_id"] for p in I.POOLS for e in sel[p]}
                       | _abo_used_elsewhere())
            pool = RL.candidates(cache, exclude)
            rng.shuffle(pool)
            index = I.load_image_index(cache)
            chosen = [it for it in pool if any(i in index for i in it.image_ids)][:n_real]
            paths = I.download_images(cache, index, [i for it in chosen for i in it.image_ids], "original")
            rule = rules.rule_for_category("unrelated")
            for it in chosen:
                lid = _new_id(used_ids, rng)
                names = []
                for j, iid in enumerate(i for i in it.image_ids if i in paths):
                    fn = f"{lid.lower()}_{j + 1}.jpg"
                    with Image.open(paths[iid]) as raw:
                        prepare_square(raw, I.OTHER_SIZE).save(images_dir / fn, "JPEG", quality=88)
                    names.append(fn)
                if not names:
                    continue
                added.append(MockListing(
                    id=lid, site=SITE, title=it.name, description=it.description, price=RL._price(rng),
                    currency="GBP", seller_name=f"{rng.choice(RL.HANDLE_A)}_{rng.choice(RL.HANDLE_B)}",
                    seller_location=RL._location(rng), seller_age_days=rng.randint(60, 4000),
                    seller_rating=round(rng.uniform(3.9, 5.0), 1), seller_feedback_count=rng.randint(5, 5000),
                    quantity=rng.randint(1, 60), site_category="Drinkware",
                    condition="New" if rng.random() < 0.9 else "Used", images=names,
                    specs={"Brand": it.brand or "Unbranded"}, listed_days_ago=rng.randint(0, 90)))
                key = f"{SITE}:{lid}"
                labels[key] = _label("unrelated", rule, lid, "real",
                                     {"product": None, "bulk_cluster": False,
                                      "image_sources": [f"abo:{i}" for i in it.image_ids], "source": "abo",
                                      "abo_item_id": it.item_id})
                strata[("unrelated", "real")].append(key)
                n_abo += 1
    except FileNotFoundError:
        pass  # no ABO assets: Vaelo cases only

    for key, split in G._split(strata, rng).items():
        labels[key]["split"] = split
    listings = keep + added
    rng.shuffle(listings)
    write_catalog(catalog_dir, listings)
    (catalog_dir / "clip_cache.npz").unlink(missing_ok=True)
    labels_out.write_text(json.dumps(dict(sorted(labels.items())), indent=2), encoding="utf-8")
    return {"added": len(added), "vaelo_cases": len(cases), "real": n_abo, "other_sites_kept": len(keep),
            "split": dict(Counter(v["split"] for v in labels.values())),
            "categories": dict(Counter(v["category"] for v in labels.values())), "labels": str(labels_out)}


def main() -> None:
    from bh.config import get_config

    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=int, default=40, help="real ABO listings to add (0 = Vaelo cases only)")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--catalog", type=Path, default=cfg.paths.resolve(cfg.mocksites.catalog_dir))
    args = ap.parse_args()
    print(json.dumps(add_bazaarly(args.catalog, n_real=args.real, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
