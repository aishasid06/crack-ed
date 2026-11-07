"""Load the brand's gold assets from ``seed/brand/`` and upsert them into the DB.

Perceptual hashes and CLIP embeddings are computed later (pipeline ``load_brand`` node,
Phase 6/8); this module only validates and persists the declared assets.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.engine import Engine
from sqlmodel import select

from bh.db import AuthorisedSeller, Brand, MarkType, Product, TradeMark, session_scope


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class OwnerInfo(_M):
    name: str
    email: str
    address: str


class CopyrightInfo(_M):
    work_description: str
    first_published: str  # ISO date of first publication of the brand's product photos


class BrandMeta(_M):
    name: str
    slug: str
    fictional: bool = True
    owner: OwnerInfo | None = None
    copyright: CopyrightInfo | None = None


class ProductAsset(_M):
    slug: str
    name: str
    rrp: float = Field(gt=0)
    goods_class: int = Field(ge=1, le=45)
    images: list[str] = Field(min_length=1)
    product_type: str | None = None
    generic_name: str | None = None
    material: str | None = None
    colour_name: str | None = None


class TradeMarkAsset(_M):
    mark_text: str
    mark_type: MarkType
    registration_number: str = Field(pattern=r"^UK\d{11}$")
    classes: list[int] = Field(min_length=1)
    territory: str = "UK"
    status: str = "registered"
    logo_path: str | None = None


class SellerAsset(_M):
    platform: str
    seller_name: str


class BrandAssets(_M):
    root: Path
    meta: BrandMeta
    products: list[ProductAsset] = Field(min_length=1)
    trademarks: list[TradeMarkAsset] = Field(min_length=1)
    authorised_sellers: list[SellerAsset]

    @field_validator("products")
    @classmethod
    def _unique_slugs(cls, v: list[ProductAsset]) -> list[ProductAsset]:
        if len({p.slug for p in v}) != len(v):
            raise ValueError("product slugs must be unique")
        return v

    def image_path(self, rel: str) -> Path:
        return self.root / rel

    def is_authorised(self, platform: str, seller_name: str | None) -> bool:
        if not seller_name:
            return False
        name = seller_name.strip().lower()
        return any(s.platform == platform and s.seller_name.strip().lower() == name
                   for s in self.authorised_sellers)


def load_brand_assets(brand_dir: Path) -> BrandAssets:
    def read(name: str) -> object:
        p = brand_dir / name
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `python -m seed.assets` or add the brand's assets")
        return json.loads(p.read_text(encoding="utf-8"))

    tm_raw = read("trademark.json")
    assets = BrandAssets(
        root=brand_dir,
        meta=BrandMeta.model_validate(read("brand.json")),
        products=[ProductAsset.model_validate(p) for p in read("products.json")],  # type: ignore[union-attr]
        trademarks=[TradeMarkAsset.model_validate(t) for t in (tm_raw if isinstance(tm_raw, list) else [tm_raw])],
        authorised_sellers=[SellerAsset.model_validate(s) for s in read("authorised_sellers.json")],  # type: ignore[union-attr]
    )
    for p in assets.products:
        for img in p.images:
            if not assets.image_path(img).exists():
                raise FileNotFoundError(f"product {p.slug}: missing image {img}")
    for t in assets.trademarks:
        if t.logo_path and not assets.image_path(t.logo_path).exists():
            raise FileNotFoundError(f"trade mark {t.registration_number}: missing logo {t.logo_path}")
    return assets


def upsert_brand(assets: BrandAssets, engine: Engine | None = None) -> int:
    """Insert or update Brand, TradeMark, Product and AuthorisedSeller rows. Returns brand id."""
    with session_scope(engine) as s:
        brand = s.exec(select(Brand).where(Brand.slug == assets.meta.slug)).first()
        if brand is None:
            brand = Brand(name=assets.meta.name, slug=assets.meta.slug, is_fictional=assets.meta.fictional)
            s.add(brand)
            s.flush()
        assert brand.id is not None
        bid = brand.id

        for t in assets.trademarks:
            row = s.exec(select(TradeMark).where(TradeMark.registration_number == t.registration_number)).first()
            logo = str(assets.image_path(t.logo_path)) if t.logo_path else None
            if row is None:
                row = TradeMark(brand_id=bid, registration_number=t.registration_number, mark_text=t.mark_text,
                                mark_type=t.mark_type)
                s.add(row)
            row.mark_text, row.mark_type, row.classes = t.mark_text, t.mark_type, list(t.classes)
            row.territory, row.status, row.logo_path = t.territory, t.status, logo

        for p in assets.products:
            row = s.exec(select(Product).where(Product.brand_id == bid, Product.name == p.name)).first()
            if row is None:
                row = Product(brand_id=bid, name=p.name, rrp=p.rrp)
                s.add(row)
            row.rrp, row.goods_class = p.rrp, p.goods_class
            row.image_paths = [str(assets.image_path(i)) for i in p.images]

        existing = {(a.platform, a.seller_name) for a in s.exec(
            select(AuthorisedSeller).where(AuthorisedSeller.brand_id == bid))}
        for a in assets.authorised_sellers:
            if (a.platform, a.seller_name) not in existing:
                s.add(AuthorisedSeller(brand_id=bid, platform=a.platform, seller_name=a.seller_name))
    return bid
