"""Composite the fictional brand's mark onto product photos so it looks printed.

Used for (a) gold assets built from unbranded ABO photos and (b) counterfeit listing photos
(the mark — or a misspelt one — applied, more sloppily, to a different product).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from seed.assets import emblem, font

RGB = tuple[int, int, int]
DARK_INK: RGB = (28, 52, 64)
LIGHT_INK: RGB = (244, 236, 214)


def prepare_square(im: Image.Image, size: int) -> Image.Image:
    """Square image: pad studio shots (white border) with white, centre-crop lifestyle shots."""
    im = im.convert("RGB")
    a = np.asarray(im, dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    studio = float((border.min(axis=1) > 235).mean()) > 0.9
    w, h = im.size
    if studio:
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(im, ((side - w) // 2, (side - h) // 2))
        im = canvas
    else:
        side = min(w, h)
        im = im.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
    return im.resize((size, size), Image.LANCZOS)


def object_mask(im: Image.Image, tol: int = 22) -> Image.Image:
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    m = ((255 - a).max(axis=2) > tol).astype(np.uint8) * 255
    return Image.fromarray(m, "L").filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5))


def object_box(im: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the product on a light background (projection-based, ignores specks)."""
    m = np.asarray(object_mask(im)) > 0
    h, w = m.shape
    cols, rows = np.where(m.mean(axis=0) > 0.02)[0], np.where(m.mean(axis=1) > 0.02)[0]
    if cols.size == 0 or rows.size == 0:
        return (0, 0, w, h)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def render_mark(text: str, with_emblem: bool, ink: RGB, height_px: int = 220) -> Image.Image:
    """Mark as an RGBA image (ink colour, alpha = shape), tightly cropped."""
    f = font(height_px // 2)
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    tw = int(tmp.textlength(text, font=f))
    w, h = max(tw + 20, height_px), height_px + (height_px // 2 if with_emblem else 0)
    alpha = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(alpha)
    if with_emblem:
        emblem(d, w // 2, height_px // 4 + 4, height_px // 4, 255)  # type: ignore[arg-type]
    d.text(((w - tw) / 2, h - height_px // 2 - height_px // 6), text, font=f, fill=255)
    box = alpha.getbbox() or (0, 0, w, h)
    alpha = alpha.crop(box)
    out = Image.new("RGBA", alpha.size, ink + (0,))
    out.putalpha(alpha)
    return out


@dataclass(frozen=True)
class Placement:
    """Where the mark goes, relative to the object's box (or an explicit box in 0-1 image coords)."""

    anchor_x: float = 0.5
    anchor_y: float = 0.5
    rel_width: float = 0.5
    rotate: float = 0.0
    perspective: float = 0.05
    opacity: float = 0.9
    box: tuple[float, float, float, float] | None = None  # override the detected object box
    ink: RGB | None = None


def _perspective(mark: Image.Image, k: float) -> Image.Image:
    """Narrow the top edge slightly (camera a little above the product)."""
    if k <= 0:
        return mark
    w, h = mark.size
    dx = int(w * k)
    return mark.transform((w, h), Image.QUAD, (-dx, 0, 0, h, w, h, w + dx, 0), resample=Image.BICUBIC)


def composite_mark(im: Image.Image, text: str = "VAELO", with_emblem: bool = True,
                   place: Placement = Placement()) -> Image.Image:
    base = im.convert("RGB")
    W, H = base.size
    if place.box:
        x0, y0, x1, y1 = (int(place.box[0] * W), int(place.box[1] * H), int(place.box[2] * W), int(place.box[3] * H))
        clip = Image.new("L", base.size, 0)
        ImageDraw.Draw(clip).rectangle((x0, y0, x1, y1), fill=255)
    else:
        x0, y0, x1, y1 = object_box(base)
        clip = object_mask(base)
    bw = max(8, x1 - x0)
    cx, cy = x0 + place.anchor_x * bw, y0 + place.anchor_y * (y1 - y0)

    target_w = max(12, int(bw * place.rel_width))
    probe = render_mark(text, with_emblem, (0, 0, 0))
    target_h = max(8, int(probe.height * target_w / probe.width))
    region = base.crop((int(cx - target_w / 2), int(cy - target_h / 2), int(cx + target_w / 2), int(cy + target_h / 2)))
    lum = float(np.asarray(region.convert("L")).mean()) if region.size[0] and region.size[1] else 128.0
    ink = place.ink or (DARK_INK if lum > 135 else LIGHT_INK)

    mark = render_mark(text, with_emblem, ink).resize((target_w, target_h), Image.LANCZOS)
    mark = _perspective(mark, place.perspective)
    if place.rotate:
        mark = mark.rotate(place.rotate, resample=Image.BICUBIC, expand=True)
    ox, oy = int(cx - mark.width / 2), int(cy - mark.height / 2)

    # printed look: follow the product's shading, then soften edges slightly
    under = np.asarray(base.crop((ox, oy, ox + mark.width, oy + mark.height)).convert("L"), dtype=np.float32)
    shade = np.clip(under / max(1.0, float(under.mean())), 0.65, 1.25)[..., None]
    rgba = np.asarray(mark, dtype=np.float32)
    rgba[..., :3] = np.clip(rgba[..., :3] * shade, 0, 255)
    rgba[..., 3] *= place.opacity
    mark = Image.fromarray(rgba.astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(0.6))
    alpha = Image.fromarray(np.minimum(np.asarray(mark.getchannel("A")),
                                       np.asarray(clip.crop((ox, oy, ox + mark.width, oy + mark.height)))), "L")
    out = base.copy()
    out.paste(mark.convert("RGB"), (ox, oy), alpha)
    return out


def sloppy_placement(rng: random.Random, base: Placement = Placement()) -> Placement:
    """Counterfeit print: off-centre, wrong size, slightly rotated."""
    return Placement(anchor_x=base.anchor_x + rng.uniform(-0.08, 0.08),
                     anchor_y=base.anchor_y + rng.uniform(-0.1, 0.1),
                     rel_width=base.rel_width * rng.uniform(0.75, 1.25), rotate=rng.uniform(-7, 7),
                     perspective=rng.uniform(0.0, 0.08), opacity=rng.uniform(0.8, 1.0), box=base.box)
