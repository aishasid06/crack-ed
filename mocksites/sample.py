"""Write a small synthetic demo catalog so the mock sites can be browsed before
Phase 3's seed generator exists. Not used by the pipeline; overwritten by ``seed/generate.py``.

    python -m mocksites.sample [--out data/mocksites] [--n 30]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from mocksites.catalog import MockListing, write_catalog

PRODUCTS = [("Insulated bottle 500ml", 28.0), ("Travel mug 350ml", 22.0), ("Lunch flask 750ml", 34.0)]
SELLERS = [("homegoods_uk", "Leeds, UK"), ("bottle_bargains", "Shenzhen, China"), ("kitchen_corner", "Dublin, Ireland"),
           ("eco_sip_store", "Bristol, UK"), ("global_traders88", "Osaka, Japan")]


def _image(path: Path, rgb: tuple[int, int, int], label: str) -> None:
    im = Image.new("RGB", (400, 400), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((140, 60, 260, 360), radius=40, fill=rgb)
    d.rectangle((160, 30, 240, 70), fill=(60, 60, 60))
    d.text((20, 370), label, fill=(90, 90, 90))
    im.save(path, "JPEG", quality=85)


def build(out: Path, n: int = 30, seed: int = 7) -> int:
    rng = random.Random(seed)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    listings: list[MockListing] = []
    for i in range(n):
        site = "mockbay" if i % 2 == 0 else "tradeloft"
        name, rrp = rng.choice(PRODUCTS)
        seller, loc = rng.choice(SELLERS)
        colour = (rng.randrange(40, 220), rng.randrange(40, 220), rng.randrange(40, 220))
        imgs = []
        for k in range(rng.randint(1, 3)):
            fn = f"demo_{i:03d}_{k}.jpg"
            _image(images / fn, colour, f"demo {i}")
            imgs.append(fn)
        price = round(rrp * rng.uniform(0.2, 1.05), 2)
        lid = f"{100200 + i}" if site == "mockbay" else f"TL-{5000 + i}"
        brand = rng.choice(["Vaelo", "Vaelo", "Vaello", "Generic"])
        listings.append(MockListing(
            id=lid, site=site, title=f"{brand} {name}", currency="GBP", price=price,
            description=f"{brand} {name.lower()}. Demo listing generated for UI testing.\nShips in 2-3 days.",
            seller_name=seller, seller_location=loc, seller_age_days=rng.randint(5, 3000),
            seller_rating=round(rng.uniform(3.2, 5.0), 1), seller_feedback_count=rng.randint(0, 900),
            quantity=rng.choice([1, 2, 5, 10, 100]), site_category=rng.choice(["Drinkware", "Kitchen", "Outdoor"]),
            condition=rng.choice(["New", "Used", "New other"]), images=imgs,
            specs={"Brand": brand, "Capacity": name.split()[-1], "Material": "Stainless steel"},
            listed_days_ago=rng.randint(0, 60)))
    write_catalog(out, listings)
    return len(listings)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    if args.out:
        out = Path(args.out)
    else:
        from bh.config import get_config

        cfg = get_config()
        out = cfg.paths.resolve(cfg.mocksites.catalog_dir)
    print(f"wrote {build(out, args.n)} demo listings to {out}")


if __name__ == "__main__":
    main()
