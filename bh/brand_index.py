"""``load_brand`` (SPEC §7.1 node 1): load the configured brand's gold assets, upsert them into
the DB, and compute / cache perceptual hashes and CLIP embeddings of every gold photo.

Brand words (name, marks, product names) come only from data — never from code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.brand import BrandAssets, load_brand_assets, upsert_brand
from bh.config import AppConfig
from bh.db import AuthorisedSeller, Brand, Product, TradeMark, get_engine, session_scope
from bh.vision import ImageEncoder, hashes, open_rgb


@dataclass(frozen=True)
class GoldImage:
    product_id: int
    path: Path
    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash
    phash_m: imagehash.ImageHash
    dhash_m: imagehash.ImageHash


@dataclass(frozen=True)
class ProductInfo:
    id: int
    slug: str
    name: str
    short_name: str
    rrp: float
    goods_class: int | None
    generic_name: str | None
    image_paths: list[Path]


@dataclass
class BrandContext:
    brand_id: int
    slug: str
    name: str
    marks: list[str]
    registrations: list[dict[str, object]]
    products: list[ProductInfo]
    authorised: set[tuple[str, str]]
    gold: list[GoldImage]
    gold_vecs: np.ndarray  # (n_gold, d), rows aligned with ``gold``
    encoder_name: str
    assets: BrandAssets = field(repr=False)

    def product(self, product_id: int | None) -> ProductInfo | None:
        return next((p for p in self.products if p.id == product_id), None)

    def is_authorised(self, platform: str, seller: str | None) -> bool:
        return bool(seller) and (platform, seller.strip().lower()) in self.authorised  # type: ignore[union-attr]

    def nearest_hash(self, im: Image.Image) -> tuple[int, int | None]:
        """Smallest pHash distance to any gold photo (or its mirror image) and that photo's product."""
        h = imagehash.phash(im)
        best, pid = 10**6, None
        for g in self.gold:
            d = min(h - g.phash, h - g.phash_m)
            if d < best:
                best, pid = int(d), g.product_id
        return best, pid

    def nearest_clip(self, vec: np.ndarray) -> tuple[float, int | None]:
        if self.gold_vecs.size == 0:
            return 0.0, None
        sims = self.gold_vecs @ vec
        i = int(np.argmax(sims))
        return float(sims[i]), self.gold[i].product_id


def _short_name(product_name: str, brand_name: str, marks: list[str]) -> str:
    out = product_name
    for w in [brand_name, *marks]:
        out = re.sub(rf"^\s*{re.escape(w)}\s+", "", out, flags=re.I)
    return out.strip()


def _sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def _clip_cache(cfg: AppConfig, slug: str, encoder: ImageEncoder, paths: list[Path]) -> np.ndarray:
    """Gold embeddings cached at <embeddings_dir>/<slug>/gold.npy, invalidated by image content or model."""
    d = cfg.paths.resolve(cfg.vision.embeddings_dir) / slug
    npy, meta = d / "gold.npy", d / "gold.json"
    key = {"model": encoder.name, "images": [str(p) for p in paths], "sha256": _sha(paths)}
    if npy.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        return np.load(npy)
    vecs = encoder.encode([open_rgb(p) for p in paths]) if paths else np.zeros((0, 512), np.float32)
    d.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    meta.write_text(json.dumps(key))
    return vecs


def load_brand_sync(cfg: AppConfig, encoder: ImageEncoder, engine: Engine | None = None) -> BrandContext:
    eng = engine or get_engine()
    assets = load_brand_assets(cfg.paths.resolve(cfg.paths.brand_dir))
    if assets.meta.slug != cfg.brand:
        raise ValueError(f"config brand {cfg.brand!r} does not match brand assets slug {assets.meta.slug!r}")
    brand_id = upsert_brand(assets, eng)

    with Session(eng) as s:
        brand = s.get(Brand, brand_id)
        assert brand is not None
        marks = sorted({t.mark_text for t in s.exec(select(TradeMark).where(TradeMark.brand_id == brand_id))})
        regs = [t.model_dump(include={"mark_text", "mark_type", "registration_number", "classes", "territory"})
                for t in s.exec(select(TradeMark).where(TradeMark.brand_id == brand_id))]
        rows = {p.name: p for p in s.exec(select(Product).where(Product.brand_id == brand_id))}
        authorised = {(a.platform, a.seller_name.strip().lower())
                      for a in s.exec(select(AuthorisedSeller).where(AuthorisedSeller.brand_id == brand_id))}

    products: list[ProductInfo] = []
    gold: list[GoldImage] = []
    for pa in assets.products:
        row = rows[pa.name]
        assert row.id is not None
        paths = [assets.image_path(i) for i in pa.images]
        products.append(ProductInfo(row.id, pa.slug, pa.name, _short_name(pa.name, brand.name, marks), pa.rrp,
                                    pa.goods_class, pa.generic_name, paths))
        for p in paths:
            hs = hashes(open_rgb(p))
            gold.append(GoldImage(row.id, p, hs["phash"], hs["dhash"], hs["phash_m"], hs["dhash_m"]))

    all_paths = [g.path for g in gold]
    vecs = _clip_cache(cfg, assets.meta.slug, encoder, all_paths)
    npy = cfg.paths.resolve(cfg.vision.embeddings_dir) / assets.meta.slug / "gold.npy"
    with session_scope(eng) as s:  # persist per-product hashes + embeddings path (SPEC §4 Product)
        for info in products:
            row = s.get(Product, info.id)
            assert row is not None
            row.phashes = [str(g.phash) for g in gold if g.product_id == info.id]
            row.clip_embeddings = str(npy)

    return BrandContext(brand_id, assets.meta.slug, brand.name, marks, regs, products, authorised, gold,
                        np.asarray(vecs, dtype=np.float32), encoder.name, assets)


async def load_brand(cfg: AppConfig, encoder: ImageEncoder, engine: Engine | None = None) -> BrandContext:
    return await asyncio.to_thread(load_brand_sync, cfg, encoder, engine)
