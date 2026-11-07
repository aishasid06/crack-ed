"""Import real catalogue photos from the Amazon Berkeley Objects (ABO) dataset.

ABO: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Credit: Collins et al.,
"ABO: Dataset and Benchmarks for Real-World 3D Object Understanding", CVPR 2022.

Workflow (only metadata + the selected images are downloaded; everything is cached under data/abo/):

    python -m seed.import_abo candidates   # metadata + small thumbnails → data/abo/review/*.jpg contact sheets
    # review the sheets, record choices in seed/abo_selection.yaml (gold items must show NO real branding)
    python -m seed.import_abo build        # download selected originals → seed/brand/ + seed/other_images/

Gold assets get the fictional brand's mark composited on (seed/mark.py). Without the ABO subset,
seed/assets.py renders synthetic stand-ins instead.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from PIL import Image, ImageDraw

from seed import assets as A

ABO_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
ABO_HOME = f"{ABO_BASE}/index.html"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CITATION = ('Collins, J., Goel, S., Deng, K., Luthra, A., Xu, L., Gundogdu, E., Zhang, X., Yago Vicente, T., '
            'Dideriksen, T., Arora, H., Guillaumin, M., Malik, J. "ABO: Dataset and Benchmarks for Real-World '
            '3D Object Understanding." CVPR 2022.')
LISTING_SHARDS = [f"listings_{c}.json.gz" for c in "0123456789abcdef"]
SEED_DIR = Path(__file__).resolve().parent
SELECTION = SEED_DIR / "abo_selection.yaml"
GOLD_SIZE = 800
OTHER_SIZE = 640

DRINKWARE_TYPES = {"BOTTLE", "DRINKING_CUP", "THERMOS", "KITCHEN"}
DRINKWARE_RE = re.compile(r"water bottle|sports bottle|bottle|tumbler|travel mug|flask|thermos|\bmug\b", re.I)
DRINKWARE_EXCLUDE = re.compile(
    r"vacuum (cleaner|dust|seal|bag)|casserole|cooler|lunch bag|beer|baby|rack|opener|straw|coaster|measuring|"
    r"pitcher|set of|pack|count|piece|\d+-pc|glasses|bags?\b|hot cups", re.I)
UNRELATED_TYPES = {"SHOES": "Fashion", "LAMP": "Home", "CHAIR": "Furniture", "HANDBAG": "Fashion", "HAT": "Fashion",
                   "SUITCASE": "Travel", "CELLULAR_PHONE_CASE": "Electronics", "RUG": "Home", "WALL_ART": "Home",
                   "OFFICE_PRODUCTS": "Office", "PET_SUPPLIES": "Pets", "SPORTING_GOODS": "Sports",
                   "TABLE": "Furniture", "LIGHT_FIXTURE": "Home"}


def cache_dir() -> Path:
    from bh.config import get_config

    cfg = get_config()
    return cfg.paths.resolve(cfg.paths.data_dir) / "abo"


# ---------------------------------------------------------------- downloads


def fetch(client: httpx.Client, url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with client.stream("GET", url, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
                tmp.replace(dest)
                return dest
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return dest


def ensure_metadata(cache: Path) -> None:
    with httpx.Client() as c:
        fetch(c, f"{ABO_BASE}/LICENSE-CC-BY-4.0.txt", cache / "LICENSE-CC-BY-4.0.txt")
        fetch(c, f"{ABO_BASE}/images/metadata/images.csv.gz", cache / "metadata" / "images.csv.gz")
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda s: fetch(c, f"{ABO_BASE}/listings/metadata/{s}", cache / "metadata" / s),
                        LISTING_SHARDS))


# ---------------------------------------------------------------- metadata


@dataclass
class Item:
    item_id: str
    product_type: str
    name: str
    brand: str
    image_ids: list[str] = field(default_factory=list)


def _en(rec: dict[str, Any], key: str) -> str:
    vals = rec.get(key) or []
    for v in vals:
        if str(v.get("language_tag", "")).startswith("en"):
            return str(v.get("value", ""))
    return str(vals[0].get("value", "")) if vals else ""


def load_items(cache: Path) -> dict[str, Item]:
    items: dict[str, Item] = {}
    for shard in LISTING_SHARDS:
        with gzip.open(cache / "metadata" / shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["item_id"] in items or not r.get("main_image_id"):
                    continue
                items[r["item_id"]] = Item(r["item_id"], (r.get("product_type") or [{}])[0].get("value", ""),
                                           _en(r, "item_name"), _en(r, "brand"),
                                           [r["main_image_id"], *r.get("other_image_id", [])])
    return items


def load_image_index(cache: Path) -> dict[str, tuple[int, int, str]]:
    with gzip.open(cache / "metadata" / "images.csv.gz", "rt") as fh:
        return {row["image_id"]: (int(row["height"]), int(row["width"]), row["path"]) for row in csv.DictReader(fh)}


def image_url(path: str, size: str = "original") -> str:
    return f"{ABO_BASE}/images/{size}/{path}"


def download_images(cache: Path, index: dict[str, tuple[int, int, str]], ids: list[str], size: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with httpx.Client() as c, ThreadPoolExecutor(12) as ex:
        def one(iid: str) -> tuple[str, Path | None]:
            if iid not in index:
                return iid, None
            path = index[iid][2]
            return iid, fetch(c, image_url(path, size), cache / "images" / size / path)

        for iid, p in ex.map(one, ids):
            if p is not None:
                out[iid] = p
    return out


# ---------------------------------------------------------------- candidates (review sheets)


def drinkware_candidates(items: dict[str, Item]) -> list[Item]:
    seen_names: set[str] = set()
    out = []
    for it in items.values():
        if it.product_type not in DRINKWARE_TYPES or not DRINKWARE_RE.search(it.name) \
                or DRINKWARE_EXCLUDE.search(it.name) or len(it.image_ids) < 3:
            continue
        norm = re.sub(r"\(renewed\)\s*", "", it.name.lower())[:60]
        if norm in seen_names:
            continue
        seen_names.add(norm)
        out.append(it)
    return sorted(out, key=lambda i: (i.product_type, i.name))


def unrelated_candidates(items: dict[str, Item], per_type: int, seed: int = 5) -> list[Item]:
    rng = random.Random(seed)
    by_type: dict[str, list[Item]] = {}
    for it in items.values():
        if it.product_type in UNRELATED_TYPES:
            by_type.setdefault(it.product_type, []).append(it)
    out = []
    for t in sorted(by_type):
        out += rng.sample(sorted(by_type[t], key=lambda i: i.item_id), k=min(per_type, len(by_type[t])))
    return out


def _sheet(rows: list[tuple[str, list[tuple[str, Path]]]], path: Path, thumb: int = 150) -> None:
    cols = max(len(r[1]) for r in rows)
    sheet = Image.new("RGB", (cols * thumb, len(rows) * (thumb + 18)), "white")
    d = ImageDraw.Draw(sheet)
    f = A.font(11)
    for ri, (label, imgs) in enumerate(rows):
        y = ri * (thumb + 18)
        d.text((2, y + 1), label[:150], fill=(200, 0, 0), font=f)
        for ci, (iid, p) in enumerate(imgs):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb - 4, thumb - 22))
                sheet.paste(im, (ci * thumb + 2, y + 16))
            d.text((ci * thumb + 2, y + thumb - 6), iid, fill=(0, 0, 160), font=f)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "JPEG", quality=85)


def cmd_candidates(cache: Path, per_sheet: int = 8, unrelated_per_type: int = 6) -> None:
    ensure_metadata(cache)
    items, index = load_items(cache), load_image_index(cache)
    drink = drinkware_candidates(items)
    unrel = unrelated_candidates(items, unrelated_per_type)
    ids = [i for it in drink for i in it.image_ids[:8]] + [it.image_ids[0] for it in unrel]
    thumbs = download_images(cache, index, ids, "small")
    review = cache / "review"
    shutil.rmtree(review, ignore_errors=True)
    for n in range(0, len(drink), per_sheet):
        rows = [(f"{it.item_id} [{it.product_type}] {it.brand} | {it.name}",
                 [(i, thumbs[i]) for i in it.image_ids[:8] if i in thumbs]) for it in drink[n:n + per_sheet]]
        _sheet(rows, review / f"drinkware_{n // per_sheet:02d}.jpg")
    rows = []
    for t in sorted(UNRELATED_TYPES):
        its = [it for it in unrel if it.product_type == t]
        if its:
            rows.append((t, [(f"{it.item_id}:{it.image_ids[0]}", thumbs[it.image_ids[0]]) for it in its
                             if it.image_ids[0] in thumbs]))
    _sheet(rows, review / "unrelated.jpg")
    (review / "candidates.json").write_text(json.dumps(
        {"drinkware": [it.__dict__ for it in drink], "unrelated": [it.__dict__ for it in unrel]}, indent=1))
    print(f"{len(drink)} drinkware candidates, {len(unrel)} unrelated → {review}")


# ---------------------------------------------------------------- build


POOLS = ("lookalike", "unrelated", "blank")


def load_selection(path: Path = SELECTION) -> dict[str, Any]:
    sel = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = [g["abo_item_id"] for g in sel["gold"]] + [e["abo_item_id"] for p in POOLS for e in sel[p]]
    gold_items = {g["abo_item_id"] for g in sel["gold"]}
    for pool in POOLS:
        clash = gold_items & {e["abo_item_id"] for e in sel[pool]}
        if clash:
            raise ValueError(f"gold items must not be reused in {pool}: {sorted(clash)}")
    pool_items = {pool: {e["abo_item_id"] for e in sel[pool]} for pool in POOLS}
    if pool_items["blank"] & pool_items["lookalike"]:
        raise ValueError("an ABO item cannot be both a counterfeit blank and a lookalike")
    if not items:
        raise ValueError("empty selection")
    return sel


def selected_image_ids(sel: dict[str, Any]) -> list[str]:
    ids = [i for g in sel["gold"] for i in g["images"]] + [e["image"] for p in POOLS for e in sel[p]]
    return list(dict.fromkeys(ids))


def _placement(base: dict[str, Any], override: dict[str, Any] | None) -> Any:
    from seed.mark import Placement

    kw = {**base, **(override or {})}
    if kw.get("box") is not None:
        kw["box"] = tuple(kw["box"])
    if kw.get("ink") is not None:
        kw["ink"] = tuple(kw["ink"])
    return Placement(**kw)


def _attribution_lines(entries: list[tuple[str, str, str]]) -> list[str]:
    return [f"| {fn} | {item} | {img} |" for fn, item, img in entries]


def cmd_build(cache: Path, selection: Path = SELECTION, brand_dir: Path = A.BRAND_DIR,
              other_dir: Path = A.OTHER_DIR) -> dict[str, int]:
    from seed.mark import composite_mark, prepare_square

    ensure_metadata(cache)
    sel = load_selection(selection)
    index = load_image_index(cache)
    ids = selected_image_ids(sel)
    missing = [i for i in ids if i not in index]
    if missing:
        raise KeyError(f"image ids not in ABO index: {missing}")
    paths = download_images(cache, index, ids, "original")

    # gold assets: unbranded ABO photos + the fictional mark
    photos = brand_dir / "photos"
    shutil.rmtree(photos, ignore_errors=True)
    photos.mkdir(parents=True)
    products, credits = [], []
    for g in sel["gold"]:
        rels = []
        for k, iid in enumerate(g["images"]):
            with Image.open(paths[iid]) as raw:
                im = prepare_square(raw, GOLD_SIZE)
            place = _placement(g.get("placement", {}), (g.get("overrides") or {}).get(iid))
            fn = f"{g['slug']}_{k + 1}.jpg"
            composite_mark(im, A.MARK, True, place).save(photos / fn, "JPEG", quality=92)
            rels.append(f"photos/{fn}")
            credits.append((f"brand/photos/{fn}", g["abo_item_id"], iid))
        products.append({
            "slug": g["slug"], "name": g["name"], "rrp": g["rrp"], "goods_class": g["goods_class"],
            "product_type": g["product_type"], "generic_name": g["generic_name"], "material": g["material"],
            "colour_name": g["colour_name"], "images": rels,
            "source": {"dataset": "ABO", "abo_item_id": g["abo_item_id"], "abo_image_ids": g["images"],
                       "licence": LICENSE_NAME, "modified": "resized/padded; fictional mark composited"}})
    A.render_logo(brand_dir / "logo.png")
    (brand_dir / "brand.json").write_text(json.dumps(
        {"name": "Vaelo", "slug": "vaelo", "fictional": True, "image_source": "ABO", "owner": A.OWNER,
         "copyright": A.COPYRIGHT,
         "note": "Fictional brand and registrations for a hackathon demo. Product photos from ABO (CC BY 4.0) "
                 "with the fictional mark composited."}, indent=2))
    (brand_dir / "products.json").write_text(json.dumps(products, indent=2))
    (brand_dir / "authorised_sellers.json").write_text(json.dumps(A.AUTHORISED_SELLERS, indent=2))
    (brand_dir / "trademark.json").write_text(json.dumps(A.TRADEMARKS, indent=2))

    # other pools
    for old in other_dir.glob("*.jpg"):  # legacy flat synthetic layout
        old.unlink()
    counts = {"gold_photos": len(credits)}
    for pool in POOLS:
        d = other_dir / pool
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        entries = []
        for n, e in enumerate(sel[pool], 1):
            fn = f"{pool}_{n:02d}.jpg"
            with Image.open(paths[e["image"]]) as raw:
                prepare_square(raw, OTHER_SIZE).save(d / fn, "JPEG", quality=90)
            entries.append({"file": fn, **{k: v for k, v in e.items() if k != "image"}, "abo_image_id": e["image"]})
            credits.append((f"other_images/{pool}/{fn}", e["abo_item_id"], e["image"]))
        (d / "index.json").write_text(json.dumps({"source": "ABO", "items": entries}, indent=2))
        counts[pool] = len(entries)

    text = "\n".join([
        "# Image attribution", "",
        f"Product photos are from the Amazon Berkeley Objects (ABO) dataset ({ABO_HOME}),",
        f"licensed under {LICENSE_NAME} ({LICENSE_URL}). {CITATION}", "",
        "Changes: images were resized and padded/cropped to squares; gold photos and, at generation time,",
        "counterfeit listing photos have the FICTIONAL 'VAELO' mark composited onto them. The products shown",
        "are not Vaelo products and Amazon does not endorse this project.", "",
        "| file | ABO item | ABO image |", "|---|---|---|", *_attribution_lines(credits), ""])
    (brand_dir / "ATTRIBUTION.md").write_text(text)
    (other_dir / "ATTRIBUTION.md").write_text(text)
    print(f"built from ABO: {counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["candidates", "build"])
    ap.add_argument("--cache", type=Path, default=None)
    args = ap.parse_args()
    cache = args.cache or cache_dir()
    if args.command == "candidates":
        cmd_candidates(cache)
    else:
        cmd_build(cache)


if __name__ == "__main__":
    main()
