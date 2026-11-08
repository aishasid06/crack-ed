"""Procedurally render the fictional brand's gold assets and the "other" images.

SPEC §6 expects team-taken photos in ``seed/brand/`` and ``seed/other_images/``. When those
folders are empty this module renders synthetic stand-ins so the project runs end to end.
Existing files are never overwritten.

    python -m seed.assets [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SEED_DIR = Path(__file__).resolve().parent
BRAND_DIR = SEED_DIR / "brand"
OTHER_DIR = SEED_DIR / "other_images"
SIZE = 512
MARK = "VAELO"

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class ProductSpec:
    slug: str
    name: str
    rrp: float
    goods_class: int
    shape: str  # bottle | mug | flask
    body: RGB
    accent: RGB


PRODUCTS: list[ProductSpec] = [
    ProductSpec("arc-500", "Vaelo Arc Bottle 500ml", 28.00, 21, "bottle", (38, 70, 83), (233, 196, 106)),
    ProductSpec("arc-750", "Vaelo Arc Bottle 750ml", 34.00, 21, "tall_bottle", (42, 157, 143), (244, 241, 222)),
    ProductSpec("commuter-350", "Vaelo Commuter Mug 350ml", 24.00, 21, "mug", (231, 111, 81), (38, 70, 83)),
    ProductSpec("trail-1l", "Vaelo Trail Flask 1L", 42.00, 21, "flask", (70, 70, 74), (231, 111, 81)),
]

# Fictional rights owner, used to fill the report portal (never a real company).
OWNER = {"name": "Vaelo Ltd (fictional)", "email": "legal@vaelo.example",
         "address": "1 Fictional Way, London EC1A 1AA, United Kingdom"}
COPYRIGHT = {"work_description": "Vaelo product photographs (catalogue and lifestyle images of the Vaelo range)",
             "first_published": "2024-03-01"}

AUTHORISED_SELLERS = [
    {"platform": "echobazaar", "seller_name": "vaelo_official"},
    {"platform": "echobazaar", "seller_name": "kitchenhaus_uk"},
    {"platform": "tradeloft", "seller_name": "Vaelo Store"},
    {"platform": "tradeloft", "seller_name": "KitchenHaus"},
]

TRADEMARKS = [
    {"mark_text": MARK, "mark_type": "word", "registration_number": "UK00004123456", "classes": [21],
     "territory": "UK", "status": "registered", "logo_path": None},
    {"mark_text": MARK, "mark_type": "figurative", "registration_number": "UK00004123457", "classes": [21],
     "territory": "UK", "status": "registered", "logo_path": "logo.png"},
]

BACKGROUNDS: list[tuple[RGB, RGB]] = [
    ((246, 244, 240), (214, 210, 202)), ((232, 238, 243), (186, 198, 210)),
    ((250, 240, 228), (222, 196, 168)), ((236, 242, 232), (188, 204, 180)),
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


# Seller-shot settings: counterfeiters / lookalike sellers photograph their own goods,
# so their framing and backdrops differ from the brand's studio photos.
SELLER_BACKGROUNDS: list[tuple[RGB, RGB]] = [
    ((92, 84, 78), (190, 176, 160)), ((60, 66, 80), (150, 160, 175)), ((120, 98, 72), (220, 205, 180)),
    ((48, 52, 48), (140, 150, 136)), ((200, 200, 205), (90, 92, 100)),
]


def _gradient_bg(top: RGB, bottom: RGB, size: int = SIZE, horizontal: bool = False) -> Image.Image:
    im = Image.new("RGB", (size, size), top)
    d = ImageDraw.Draw(im)
    for i in range(size):
        t = i / (size - 1)
        c = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom))
        d.line([(i, 0), (i, size)] if horizontal else [(0, i), (size, i)], fill=c)
    return im


def seller_scene(rng: random.Random) -> dict[str, object]:
    """render_object kwargs for a non-studio, seller-taken photo."""
    side = rng.choice([-1, 1])
    return dict(bg=rng.choice(SELLER_BACKGROUNDS), horizontal=True, angle=rng.uniform(-10, 10),
                scale=rng.uniform(0.62, 0.8), dx=side * rng.randint(70, 120), dy=rng.randint(-40, 10),
                scene=rng.choice(["shelf", "box", "shelf_box"]))


def _shade(body: RGB, f: float) -> RGB:
    return tuple(max(0, min(255, int(c * f))) for c in body)  # type: ignore[return-value]


def _shaded_rect(im: Image.Image, box: tuple[int, int, int, int], body: RGB, radius: int) -> None:
    """Rounded rect with a left-to-right highlight → shadow gradient (cylinder look)."""
    x0, y0, x1, y1 = box
    layer = Image.new("RGB", im.size, body)
    d = ImageDraw.Draw(layer)
    for x in range(x0, x1 + 1):
        t = (x - x0) / max(1, x1 - x0)
        f = 1.25 - 0.55 * t + 0.25 * math.exp(-((t - 0.22) ** 2) / 0.004)
        d.line([(x, y0), (x, y1)], fill=_shade(body, f))
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    im.paste(layer, (0, 0), mask)


def emblem(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, colour: RGB) -> None:
    """The figurative mark: three stacked wave arcs inside a ring."""
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=colour, width=max(2, r // 6))
    for k in range(3):
        y = cy - r // 3 + k * r // 3
        d.arc((cx - r * 2 // 3, y - r // 4, cx, y + r // 4), 200, 340, fill=colour, width=max(2, r // 8))
        d.arc((cx, y - r // 4, cx + r * 2 // 3, y + r // 4), 20, 160, fill=colour, width=max(2, r // 8))


def render_object(shape: str, body: RGB, accent: RGB, *, mark: str | None = MARK, with_emblem: bool = True,
                  bg: tuple[RGB, RGB] = BACKGROUNDS[0], angle: float = 0.0, scale: float = 1.0,
                  dx: int = 0, dy: int = 0, props: bool = False, horizontal: bool = False,
                  scene: str | None = None) -> Image.Image:
    """Render a drinkware-like product photo. ``mark=None`` renders it unbranded."""
    im = _gradient_bg(*bg, horizontal=horizontal)
    d = ImageDraw.Draw(im)
    if scene:  # seller shot: wall/floor split and a cardboard box on the far side
        floor = int(SIZE * 0.6)
        d.rectangle((0, floor, SIZE, SIZE), fill=_shade(bg[0], 1.5))
        if "box" in scene:
            bx = 30 if dx > 0 else SIZE - 230
            d.rectangle((bx, floor - 120, bx + 200, floor + 90), fill=(176, 136, 92))
            d.line((bx, floor - 60, bx + 200, floor - 60), fill=(140, 104, 66), width=6)
    if props:  # lifestyle shot: table edge + plant
        d.rectangle((0, int(SIZE * 0.78), SIZE, SIZE), fill=_shade(bg[1], 0.85))
        d.rounded_rectangle((40, 300, 110, 400), radius=8, fill=(160, 110, 80))
        for k in range(6):
            a = math.radians(-150 + k * 24)
            d.ellipse((75 + 70 * math.cos(a) - 18, 300 + 70 * math.sin(a) - 30,
                       75 + 70 * math.cos(a) + 18, 300 + 70 * math.sin(a) + 30), fill=(70, 130, 80))

    cx, base = SIZE // 2 + dx, int(SIZE * 0.86) + dy
    dims = {"bottle": (0.24, 0.62), "tall_bottle": (0.24, 0.72), "mug": (0.34, 0.44), "flask": (0.32, 0.66)}
    wf, hf = dims.get(shape, (0.26, 0.6))
    w, h = int(SIZE * wf * scale), int(SIZE * hf * scale)
    x0, y0, x1, y1 = cx - w // 2, base - h, cx + w // 2, base

    shadow = Image.new("L", im.size, 0)
    ImageDraw.Draw(shadow).ellipse((x0 - 20, base - 14, x1 + 30, base + 18), fill=110)
    im.paste((0, 0, 0), (0, 0, SIZE, SIZE), shadow.filter(ImageFilter.GaussianBlur(10)).point(lambda v: v // 2))

    if shape == "mug":
        d.arc((x1 - w // 6, y0 + h // 5, x1 + w // 3, y0 + h * 3 // 4), 270, 90, fill=_shade(body, 0.8),
              width=max(6, w // 9))
    _shaded_rect(im, (x0, y0, x1, y1), body, radius=max(8, w // 5))
    if shape in ("bottle", "tall_bottle"):
        nw = w * 3 // 5
        _shaded_rect(im, (cx - nw // 2, y0 - h // 10, cx + nw // 2, y0 + 10), body, radius=nw // 4)
        _shaded_rect(im, (cx - nw // 2 - 3, y0 - h // 10 - h // 9, cx + nw // 2 + 3, y0 - h // 10 + 4),
                     accent, radius=8)
    elif shape == "flask":
        _shaded_rect(im, (x0 + 6, y0 - h // 8, x1 - 6, y0 + 8), accent, radius=12)
        ImageDraw.Draw(im).arc((cx - w // 4, y0 - h // 4, cx + w // 4, y0 - h // 10), 180, 360,
                               fill=_shade(accent, 0.8), width=8)
    else:  # mug lid
        _shaded_rect(im, (x0 - 4, y0 - h // 9, x1 + 4, y0 + 6), accent, radius=10)

    d = ImageDraw.Draw(im)
    if mark:
        fsize = max(14, int(w * 0.28))
        f = font(fsize)
        tw = d.textlength(mark, font=f)
        ty = y0 + h // 2
        d.text((cx - tw / 2, ty), mark, font=f, fill=accent)
        if with_emblem:
            emblem(d, cx, ty - int(fsize * 1.1), max(10, int(w * 0.16)), accent)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, fillcolor=bg[0])
    return im


def render_logo(path: Path) -> None:
    """1050x400 PNG: brand-folder images must be at least 300 px on the shorter side."""
    im = Image.new("RGBA", (1050, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    emblem(d, 175, 200, 137, (38, 70, 83))
    d.text((362, 112), MARK, font=font(180), fill=(38, 70, 83))
    im.save(path)


# ---------------------------------------------------------------- unrelated objects


def render_unrelated(kind: str, rng: random.Random) -> Image.Image:
    bg = rng.choice(BACKGROUNDS)
    im = _gradient_bg(*bg)
    d = ImageDraw.Draw(im)
    c: RGB = (rng.randrange(40, 220), rng.randrange(40, 220), rng.randrange(40, 220))
    if kind == "lamp":
        d.polygon([(186, 120), (326, 120), (380, 250), (132, 250)], fill=c)
        d.rectangle((248, 250, 264, 420), fill=(60, 60, 60))
        d.ellipse((180, 410, 332, 450), fill=(60, 60, 60))
    elif kind == "shoe":
        d.polygon([(90, 360), (120, 260), (230, 250), (300, 320), (430, 340), (440, 390), (90, 400)], fill=c)
        d.rectangle((90, 395, 440, 415), fill=(240, 240, 240))
    elif kind == "books":
        for k in range(4):
            cc = (rng.randrange(40, 220), rng.randrange(40, 220), rng.randrange(40, 220))
            d.rectangle((120 + k * 8, 380 - k * 50, 400 - k * 6, 425 - k * 50), fill=cc)
    elif kind == "plant":
        d.polygon([(190, 300), (322, 300), (300, 430), (212, 430)], fill=(170, 100, 70))
        for k in range(9):
            a = math.radians(-170 + k * 20)
            d.ellipse((256 + 110 * math.cos(a) - 25, 290 + 110 * math.sin(a) - 45,
                       256 + 110 * math.cos(a) + 25, 290 + 110 * math.sin(a) + 45), fill=(60, 140, 70))
    elif kind == "phone_case":
        d.rounded_rectangle((176, 90, 336, 430), radius=30, fill=c)
        d.rounded_rectangle((196, 110, 256, 190), radius=14, fill=(30, 30, 30))
    elif kind == "umbrella":
        d.pieslice((96, 120, 416, 400), 180, 360, fill=c)
        d.line((256, 260, 256, 430), fill=(50, 50, 50), width=8)
        d.arc((226, 410, 256, 450), 0, 180, fill=(50, 50, 50), width=8)
    elif kind == "headphones":
        d.arc((150, 100, 362, 340), 180, 360, fill=c, width=22)
        d.rounded_rectangle((128, 220, 188, 330), radius=20, fill=c)
        d.rounded_rectangle((324, 220, 384, 330), radius=20, fill=c)
    else:  # backpack
        d.rounded_rectangle((156, 130, 356, 430), radius=50, fill=c)
        d.rounded_rectangle((186, 290, 326, 400), radius=20, fill=_shade(c, 0.8))
    return im.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC, fillcolor=bg[0])


UNRELATED_KINDS = ["lamp", "shoe", "books", "plant", "phone_case", "umbrella", "headphones", "backpack"]


# ---------------------------------------------------------------- writers


def _photo_params(k: int) -> dict[str, object]:
    return [dict(bg=BACKGROUNDS[0], angle=0.0, scale=1.0, dx=0, dy=0, props=False),
            dict(bg=BACKGROUNDS[1], angle=-7.0, scale=0.92, dx=-30, dy=-10, props=False),
            dict(bg=BACKGROUNDS[2], angle=5.0, scale=0.85, dx=60, dy=0, props=True),
            dict(bg=BACKGROUNDS[3], angle=0.0, scale=1.1, dx=10, dy=20, props=False)][k]


SYNTHETIC_PRODUCT_META = {  # generic names / materials for the synthetic stand-in products
    "arc-500": ("Insulated stainless steel water bottle", "Stainless steel", "Deep teal"),
    "arc-750": ("Sports water bottle, vacuum insulated", "Stainless steel", "Sea green"),
    "commuter-350": ("Travel coffee mug with lid", "Stainless steel", "Terracotta"),
    "trail-1l": ("Large vacuum flask with carry handle", "Stainless steel", "Graphite"),
}

# Unrelated items: (kind, title, title_hard with a brand-like word, site category)
UNRELATED_META = {
    "lamp": ("Modern bedside table lamp", "Vale Park reading lamp, cone shade", "Home"),
    "shoe": ("Canvas trainers, unisex", "Velo running shoes, lightweight", "Fashion"),
    "books": ("Bundle of 4 paperback novels", "Cookbook collection, set of 4", "Books"),
    "plant": ("Potted house plant with ceramic pot", "Vale garden fern in terracotta pot", "Garden"),
    "phone_case": ("Slim phone case, matte finish", "Velo shockproof phone case", "Electronics"),
    "umbrella": ("Compact windproof umbrella", "Golf umbrella, large canopy", "Fashion"),
    "headphones": ("Over-ear wireless headphones", "Velo studio headphones", "Electronics"),
    "backpack": ("Everyday backpack 20L", "Vaal hiking daypack 25L", "Outdoor"),
}


def build_brand(brand_dir: Path = BRAND_DIR, force: bool = False) -> bool:
    """Write synthetic gold assets. Returns False (and does nothing) if assets already exist."""
    if (brand_dir / "products.json").exists() and not force:
        return False
    photos = brand_dir / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    products = []
    for p in PRODUCTS:
        paths = []
        for k in range(4):
            fn = f"{p.slug}_{k + 1}.jpg"
            render_object(p.shape, p.body, p.accent, **_photo_params(k)).save(photos / fn, "JPEG", quality=92)  # type: ignore[arg-type]
            paths.append(f"photos/{fn}")
        generic, material, colour = SYNTHETIC_PRODUCT_META[p.slug]
        products.append({"slug": p.slug, "name": p.name, "rrp": p.rrp, "goods_class": p.goods_class,
                         "product_type": p.shape, "generic_name": generic, "material": material,
                         "colour_name": colour, "images": paths, "source": {"dataset": "synthetic"}})
    render_logo(brand_dir / "logo.png")
    meta = {"name": "Vaelo", "slug": "vaelo", "fictional": True, "image_source": "synthetic",
            "note": "Fictional brand and registrations for a hackathon demo.", "owner": OWNER, "copyright": COPYRIGHT}
    (brand_dir / "brand.json").write_text(json.dumps(meta, indent=2))
    (brand_dir / "products.json").write_text(json.dumps(products, indent=2))
    (brand_dir / "authorised_sellers.json").write_text(json.dumps(AUTHORISED_SELLERS, indent=2))
    (brand_dir / "trademark.json").write_text(json.dumps(TRADEMARKS, indent=2))
    return True


def _write_pool(d: Path, entries: list[tuple[Image.Image, dict[str, object]]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    items = []
    for n, (im, meta) in enumerate(entries, 1):
        fn = f"{d.name}_{n:02d}.jpg"
        im.save(d / fn, "JPEG", quality=90)
        items.append({"file": fn, **meta})
    (d / "index.json").write_text(json.dumps({"source": "synthetic", "items": items}, indent=2))


def other_pools_exist(other_dir: Path) -> bool:
    return all((other_dir / p / "index.json").exists() for p in ("lookalike", "unrelated", "blank"))


def build_other(other_dir: Path = OTHER_DIR, force: bool = False, seed: int = 11) -> bool:
    """Write the lookalike / unrelated / blank pools (each a folder with index.json)."""
    if other_pools_exist(other_dir) and not force:
        return False
    rng = random.Random(seed)
    look, blank, unrel = [], [], []
    for k in range(12):
        p = PRODUCTS[k % len(PRODUCTS)]
        body = tuple(max(0, min(255, c + rng.randint(-35, 35))) for c in p.body)
        accent = tuple(max(0, min(255, c + rng.randint(-40, 40))) for c in p.accent)
        # lookalike: same get-up, no mark; a third carry a different, non-confusing word
        mark = rng.choice([None, None, "AQUA", "TERRA", "NOMAD"])
        look.append((render_object(p.shape, body, accent, mark=mark, with_emblem=False,  # type: ignore[arg-type]
                                   **seller_scene(rng)), {"imitates": p.slug}))
    for p in PRODUCTS:  # unbranded bases for counterfeit photos (mark composited at generation time)
        for _ in range(2):
            body = tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in p.body)
            accent = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in p.accent)
            blank.append((render_object(p.shape, body, accent, mark=None, **seller_scene(rng)),  # type: ignore[arg-type]
                          {"for_product": p.slug}))
    for k in range(16):
        kind = UNRELATED_KINDS[k % len(UNRELATED_KINDS)]
        title, title_hard, cat = UNRELATED_META[kind]
        unrel.append((render_unrelated(kind, rng), {"kind": kind, "title": title, "title_hard": title_hard,
                                                    "site_category": cat}))
    for old in other_dir.glob("*.jpg"):  # legacy flat layout
        old.unlink()
    _write_pool(other_dir / "lookalike", look)
    _write_pool(other_dir / "blank", blank)
    _write_pool(other_dir / "unrelated", unrel)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing assets")
    args = ap.parse_args()
    print("brand assets:", "written" if build_brand(force=args.force) else "kept existing")
    print("other images:", "written" if build_other(force=args.force) else "kept existing")


if __name__ == "__main__":
    main()
