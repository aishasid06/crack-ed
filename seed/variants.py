"""Image variants applied to copied / counterfeit photos (SPEC §6)."""

from __future__ import annotations

import io
import random
from collections.abc import Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from seed.assets import font

Variant = Callable[[Image.Image, random.Random], Image.Image]


def resize(im: Image.Image, rng: random.Random) -> Image.Image:
    f = rng.uniform(0.45, 0.8)
    return im.resize((max(64, int(im.width * f)), max(64, int(im.height * f))), Image.LANCZOS)


def crop(im: Image.Image, rng: random.Random) -> Image.Image:
    w, h = im.size
    l, t = int(w * rng.uniform(0.02, 0.09)), int(h * rng.uniform(0.02, 0.09))
    r, b = int(w * rng.uniform(0.02, 0.09)), int(h * rng.uniform(0.02, 0.09))
    return im.crop((l, t, w - r, h - b))


def mirror(im: Image.Image, rng: random.Random) -> Image.Image:
    return ImageOps.mirror(im)


def brightness(im: Image.Image, rng: random.Random) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(rng.choice([rng.uniform(0.7, 0.88), rng.uniform(1.12, 1.3)]))


def jpeg_q25(im: Image.Image, rng: random.Random) -> Image.Image:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=25)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def watermark(im: Image.Image, rng: random.Random) -> Image.Image:
    base = im.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    text = rng.choice(["BEST DEALS", "shop4less", "TOP SELLER", "fastship", "SALE"])
    f = font(max(14, base.width // 12))
    for k in range(3):
        y = int(base.height * (0.2 + 0.3 * k))
        d.text((int(base.width * rng.uniform(0.05, 0.4)), y), text, font=f, fill=(255, 255, 255, 120))
    return Image.alpha_composite(base, layer).convert("RGB")


VARIANTS: dict[str, Variant] = {
    "resize": resize, "crop": crop, "mirror": mirror, "brightness": brightness,
    "jpeg_q25": jpeg_q25, "watermark": watermark,
}


def apply_random(im: Image.Image, rng: random.Random, n_min: int, n_max: int,
                 allow: tuple[str, ...] = tuple(VARIANTS)) -> tuple[Image.Image, list[str]]:
    names = rng.sample(list(allow), k=rng.randint(n_min, min(n_max, len(allow))))
    # JPEG compression last so it degrades the final image like a re-upload would
    names.sort(key=lambda n: n == "jpeg_q25")
    out = im.convert("RGB")
    for n in names:
        out = VARIANTS[n](out, rng)
    return out.convert("RGB"), names
