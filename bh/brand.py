"""Load a brand's gold assets from a brand folder (``seed/brand/`` or e.g. ``data/brands/<slug>/``)
and upsert them into the DB.

Folder layout: ``brand.json``, ``products.json``, ``authorised_sellers.json``, optional
``trademark.json``, optional ``logo.png``, product photos (e.g. ``photos/``).

``validate_brand_dir`` reports *every* problem without raising (for the UI); ``load_brand_assets``
refuses an invalid folder with all problems listed. Perceptual hashes and CLIP embeddings are computed
later (pipeline ``load_brand`` node); this module only validates and persists the declared assets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.config import DEMO_BANNER  # noqa: F401  (re-exported: the UI shows it for real-brand demo runs)
from bh.db import AuthorisedSeller, Brand, MarkType, Product, Run, TradeMark, get_engine, session_scope

if TYPE_CHECKING:
    from bh.config import AppConfig

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
IMAGE_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_SIDE = 300
FICTIONAL_ERROR = "brand must be marked fictional: true (this project only runs on fictional brands)"
# Real-brand demo (narrow exception to SPEC §1): "fictional": false only together with "demo_real_brand": true,
# and only run with --allow-real-brand-demo (classify only, read_only sites only, no claims).
REAL_DEMO_WARNING = ("real-brand demo: this brand is not fictional. It can only run with --allow-real-brand-demo, "
                     "up to classify, on read_only sites; no case is ever decided, drafted, approved or submitted")
REAL_DEMO_FLAG_ERROR = ("brand.json marks a real brand (demo_real_brand: true): run it only with "
                        "--allow-real-brand-demo")


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
    demo_real_brand: bool = False  # with fictional: false only; see REAL_DEMO_WARNING
    owner: OwnerInfo | None = None
    copyright: CopyrightInfo | None = None


class ProductAsset(_M):
    slug: str
    name: str
    rrp: float = Field(gt=0)
    currency: str = Field(default="GBP", pattern=r"^[A-Z]{3}$")  # currency of rrp (ISO 4217)
    goods_class: int = Field(ge=1, le=45)
    images: list[str] = Field(default_factory=list)  # ≥ 1 required, except in a real-brand demo (text only)
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
    trademarks: list[TradeMarkAsset] = Field(default_factory=list)  # optional: copyright-only brands
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


class InvalidBrandDir(ValueError):
    """A brand folder failed validation; ``errors`` lists every problem."""

    def __init__(self, brand_dir: Path | str, errors: list[str]) -> None:
        super().__init__(f"brand folder {brand_dir} is invalid:\n" + "\n".join(f"  - {e}" for e in errors))
        self.brand_dir, self.errors = str(brand_dir), list(errors)


class BrandCheck(BaseModel):
    """Result of ``validate_brand_dir``: errors block a run, warnings need confirmation."""

    ok: bool
    brand_dir: str
    slug: str | None = None
    name: str | None = None
    products: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    real_brand_demo: bool = False  # fictional: false + demo_real_brand: true


def _has_dotdot(path: str | Path) -> bool:
    return ".." in Path(path).parts


def resolve_brand_dir(path: str | Path) -> Path:
    """Resolve a brand folder path safely: no ``..`` components; must resolve (after symlinks) to a directory."""
    if _has_dotdot(path):
        raise InvalidBrandDir(path, [f"path must not contain '..': {path}"])
    p = Path(path).expanduser()
    try:
        resolved = p.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError):
        raise InvalidBrandDir(path, [f"folder not found: {path}"]) from None
    if not resolved.is_dir():
        raise InvalidBrandDir(path, [f"not a directory: {path}"])
    return resolved


def _field_errors(file: str, idx: int | None, e: ValidationError) -> list[str]:
    where = f"{file}[{idx}]" if idx is not None else file
    out = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        out.append(f"{where}{'.' + loc if loc else ''}: {err['msg']}")
    return out


def _is_rooted(rel: str) -> bool:
    """Absolute on any OS: '/etc/x' is not ``is_absolute()`` on Windows (no drive) but still escapes the folder."""
    return (Path(rel).is_absolute() or PurePosixPath(rel).is_absolute() or bool(PureWindowsPath(rel).drive)
            or rel.startswith(("/", "\\")))


def _check_image(root: Path, rel: str, label: str) -> list[str]:
    """Path must stay inside the brand folder; file must be a JPEG/PNG under 10 MB, ≥ 300 px shorter side."""
    if _is_rooted(rel) or _has_dotdot(rel):
        return [f"{label}: path must be relative to the brand folder without '..': {rel}"]
    p = root / rel
    try:
        resolved = p.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return [f"{label}: file not found: {rel}"]
    if not resolved.is_relative_to(root):
        return [f"{label}: path escapes the brand folder (symlink): {rel}"]
    size = resolved.stat().st_size
    if size > MAX_IMAGE_BYTES:
        return [f"{label}: {rel} is {size / 1024 / 1024:.1f} MB (max 10 MB)"]
    try:
        with Image.open(resolved) as im:
            fmt, (w, h) = im.format, im.size
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return [f"{label}: {rel} is not a readable image ({type(e).__name__})"]
    errs = []
    if fmt not in IMAGE_FORMATS:
        errs.append(f"{label}: {rel} is {fmt}; must be JPEG or PNG")
    if min(w, h) < MIN_IMAGE_SIDE:
        errs.append(f"{label}: {rel} is {w}x{h}; shorter side must be at least {MIN_IMAGE_SIDE} px")
    return errs


def validate_brand_dir(path: str | Path) -> BrandCheck:
    """Check a brand folder and report every problem (never raises). ``ok`` is False if any error."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        root = resolve_brand_dir(path)
    except InvalidBrandDir as e:
        return BrandCheck(ok=False, brand_dir=str(path), errors=e.errors)

    def read(name: str, required: bool) -> Any:
        f = root / name
        if not f.exists():
            if required:
                errors.append(f"{name}: missing")
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            errors.append(f"{name}: invalid JSON ({e})")
            return None

    check = BrandCheck(ok=False, brand_dir=str(root))
    raw_meta = read("brand.json", True)
    if raw_meta is not None:
        if not isinstance(raw_meta, dict):
            errors.append("brand.json: must be a JSON object")
        else:
            demo = raw_meta.get("fictional") is False and raw_meta.get("demo_real_brand") is True
            if demo:
                check.real_brand_demo = True
                warnings.append(f"brand.json: {REAL_DEMO_WARNING}")
            elif raw_meta.get("fictional") is not True:
                errors.append(f"brand.json.fictional: {FICTIONAL_ERROR}")
            elif raw_meta.get("demo_real_brand"):
                errors.append("brand.json.demo_real_brand: only allowed together with fictional: false")
            try:
                meta = BrandMeta.model_validate(raw_meta)
                check.slug, check.name = meta.slug, meta.name
                if not SLUG_RE.match(meta.slug):
                    errors.append(f"brand.json.slug: {meta.slug!r} must be lowercase letters, digits and hyphens")
                elif meta.slug != root.name:
                    warnings.append(f"brand.json.slug {meta.slug!r} differs from the folder name {root.name!r}")
                if meta.owner is None:
                    warnings.append("brand.json: no 'owner' details — portal reports cannot be drafted")
                if meta.copyright is None:
                    warnings.append("brand.json: no 'copyright' details — copied-photo reports cannot be drafted")
            except ValidationError as e:
                errors += _field_errors("brand.json", None, e)

    raw_products = read("products.json", True)
    if raw_products is not None:
        if not isinstance(raw_products, list) or not raw_products:
            errors.append("products.json: must be a non-empty JSON list")
        else:
            slugs: list[str] = []
            for i, rp in enumerate(raw_products):
                try:
                    prod = ProductAsset.model_validate(rp)
                except ValidationError as e:
                    errors += _field_errors("products.json", i, e)
                    continue
                slugs.append(prod.slug)
                if not prod.images:
                    if check.real_brand_demo:
                        warnings.append(f"products.json[{i}] ({prod.slug}): no photos — text-only demo product: "
                                        "image evidence (own photos, CLIP) cannot apply")
                    else:
                        errors.append(f"products.json[{i}].images: List should have at least 1 item after "
                                      "validation, not 0")
                    continue
                if len(prod.images) < 3:
                    warnings.append(f"products.json[{i}] ({prod.slug}): only {len(prod.images)} photo(s); 3-4 "
                                    "recommended")
                for img in prod.images:
                    errors += _check_image(root, img, f"products.json[{i}] ({prod.slug}) photo")
            dupes = sorted({x for x in slugs if slugs.count(x) > 1})
            if dupes:
                errors.append(f"products.json: duplicate product slugs {dupes}")
            check.products = len(slugs)

    raw_sellers = read("authorised_sellers.json", True)
    if raw_sellers is not None:
        if not isinstance(raw_sellers, list):
            errors.append("authorised_sellers.json: must be a JSON list")
        else:
            if not raw_sellers:
                warnings.append("authorised_sellers.json: no authorised sellers — every seller is treated as "
                                "unauthorised")
            for i, rs in enumerate(raw_sellers):
                try:
                    SellerAsset.model_validate(rs)
                except ValidationError as e:
                    errors += _field_errors("authorised_sellers.json", i, e)

    raw_tm = read("trademark.json", False)
    if raw_tm is None and not (root / "trademark.json").exists():
        warnings.append("no trademark.json — copyright-only brand: trade mark rules cannot apply")
    elif raw_tm is not None:
        tms = raw_tm if isinstance(raw_tm, list) else [raw_tm]
        if not tms:
            warnings.append("trademark.json is empty — copyright-only brand: trade mark rules cannot apply")
        for i, rt in enumerate(tms):
            try:
                tm = TradeMarkAsset.model_validate(rt)
            except ValidationError as e:
                errors += _field_errors("trademark.json", i, e)
                continue
            if tm.logo_path:
                errors += _check_image(root, tm.logo_path, f"trademark.json[{i}] logo")
    if (root / "logo.png").exists() and not any("logo.png" in e for e in errors):
        errors += [e for e in _check_image(root, "logo.png", "logo") if e not in errors]

    check.errors, check.warnings, check.ok = errors, warnings, not errors
    return check


def load_brand_assets(brand_dir: Path) -> BrandAssets:
    """Load a validated brand folder; raises ``InvalidBrandDir`` listing every problem otherwise."""
    check = validate_brand_dir(brand_dir)
    if not check.ok:
        raise InvalidBrandDir(brand_dir, check.errors)
    root = Path(check.brand_dir)

    def read(name: str) -> Any:
        f = root / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    tm_raw = read("trademark.json") or []
    return BrandAssets(
        root=root,
        meta=BrandMeta.model_validate(read("brand.json")),
        products=[ProductAsset.model_validate(p) for p in read("products.json")],
        trademarks=[TradeMarkAsset.model_validate(t) for t in (tm_raw if isinstance(tm_raw, list) else [tm_raw])],
        authorised_sellers=[SellerAsset.model_validate(x) for x in read("authorised_sellers.json")],
    )


def config_for_brand_dir(cfg: "AppConfig", path: str | Path, allow_real_brand_demo: bool = False) -> "AppConfig":
    """``cfg`` set up to protect the brand in ``path``: validated, absolute ``paths.brand_dir``, ``brand`` = slug.
    A real-brand demo folder needs ``allow_real_brand_demo`` and marks the config ``real_brand_demo``."""
    check = validate_brand_dir(path)
    if not check.ok:
        raise InvalidBrandDir(path, check.errors)
    if check.real_brand_demo and not allow_real_brand_demo:
        raise InvalidBrandDir(path, [REAL_DEMO_FLAG_ERROR])
    return cfg.model_copy(update={"brand": check.slug, "real_brand_demo": check.real_brand_demo,
                                  "paths": cfg.paths.model_copy(update={"brand_dir": check.brand_dir})})


def config_for_run(cfg: "AppConfig", run_id: int | None, engine: Engine | None = None,
                   allow_real_brand_demo: bool = False) -> "AppConfig":
    """The configuration a run was started with, as far as the brand goes (``brand`` + ``paths.brand_dir``
    from ``Run.config_snapshot``). Falls back to ``cfg`` when the run has no snapshot of them."""
    if run_id is None:
        return cfg
    with Session(engine or get_engine()) as s:
        run = s.get(Run, run_id)
    snap = (run.config_snapshot if run else None) or {}
    brand_dir = (snap.get("paths") or {}).get("brand_dir")
    if not brand_dir:
        return cfg
    resolved = cfg.paths.resolve(brand_dir)
    if snap.get("brand") == cfg.brand and resolved == cfg.paths.resolve(cfg.paths.brand_dir) \
            and bool(snap.get("real_brand_demo")) == cfg.real_brand_demo:
        return cfg
    return config_for_brand_dir(cfg, resolved, allow_real_brand_demo)


def assets_for_run(cfg: "AppConfig", run_id: int | None, engine: Engine | None = None) -> BrandAssets:
    c = config_for_run(cfg, run_id, engine)
    return load_brand_assets(c.paths.resolve(c.paths.brand_dir))


def upsert_brand(assets: BrandAssets, engine: Engine | None = None) -> int:
    """Insert or update Brand, TradeMark, Product and AuthorisedSeller rows. Returns brand id."""
    with session_scope(engine) as s:
        brand = s.exec(select(Brand).where(Brand.slug == assets.meta.slug)).first()
        if brand is None:
            brand = Brand(name=assets.meta.name, slug=assets.meta.slug, is_fictional=assets.meta.fictional)
            s.add(brand)
            s.flush()
        brand.is_fictional = assets.meta.fictional  # the claim guard reads this (bh/enforce/guard.py)
        assert brand.id is not None
        bid = brand.id

        for t in assets.trademarks:
            row = s.exec(select(TradeMark).where(TradeMark.registration_number == t.registration_number)).first()
            logo = str(assets.image_path(t.logo_path)) if t.logo_path else None
            if row is not None and row.brand_id != bid:
                raise ValueError(f"trade mark registration {t.registration_number} already belongs to another brand "
                                 f"(brand id {row.brand_id}); a registration can only protect one brand")
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
