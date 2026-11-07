"""Marketplace catalog served by the mock sites.

The catalog is what a seller would publish — it contains NO ground truth. Seed
ground-truth labels live only in ``seed/labels.json``; ``load_catalog`` refuses any
entry carrying label-like keys so they cannot leak into rendered pages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Site = Literal["mockbay", "tradeloft"]

# Keys that would carry ground truth. Never allowed in the catalog, at any level.
FORBIDDEN_KEYS = frozenset({
    "category_label", "label", "labels", "ground_truth", "truth", "difficulty", "should_act",
    "split", "legal_basis", "is_counterfeit", "infringement_type", "gt",
})


class CatalogError(ValueError):
    pass


class MockListing(BaseModel):
    """A listing as published on a mock marketplace. ``site_category`` is the
    marketplace's own shop category (e.g. "Drinkware"), not an infringement label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9\-]+$")
    site: Site
    title: str
    description: str
    price: float = Field(ge=0)
    currency: str = "GBP"
    seller_name: str
    seller_location: str
    seller_age_days: int = Field(ge=0)
    seller_rating: float = Field(ge=0, le=5)
    seller_feedback_count: int = Field(default=0, ge=0)
    quantity: int = Field(default=1, ge=0)
    site_category: str = "Home & Kitchen"
    condition: str = "New"
    images: list[str] = Field(min_length=1)  # filenames under <catalog_dir>/images/
    specs: dict[str, str] = Field(default_factory=dict)  # shown as a table on TradeLoft
    listed_days_ago: int = Field(default=0, ge=0)

    @field_validator("images")
    @classmethod
    def _safe_names(cls, v: list[str]) -> list[str]:
        for name in v:
            if "/" in name or "\\" in name or name.startswith("."):
                raise ValueError(f"image name must be a bare filename: {name!r}")
        return v


class Catalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    listings: list[MockListing]

    def for_site(self, site: Site) -> list[MockListing]:
        return [l for l in self.listings if l.site == site]

    def get(self, site: Site, listing_id: str) -> MockListing | None:
        for l in self.listings:
            if l.site == site and l.id == listing_id:
                return l
        return None

    def site_categories(self, site: Site) -> list[str]:
        return sorted({l.site_category for l in self.for_site(site)})


def _check_no_labels(obj: object, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise CatalogError(f"catalog must not contain ground-truth key {k!r} at {path}")
            _check_no_labels(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_labels(v, f"{path}[{i}]")


def load_catalog(catalog_dir: Path) -> Catalog:
    path = catalog_dir / "catalog.json"
    if not path.exists():
        return Catalog(listings=[])
    raw = json.loads(path.read_text(encoding="utf-8"))
    _check_no_labels(raw)
    cat = Catalog.model_validate(raw)
    seen: set[tuple[str, str]] = set()
    for l in cat.listings:
        key = (l.site, l.id)
        if key in seen:
            raise CatalogError(f"duplicate listing id {l.id} on {l.site}")
        seen.add(key)
        for img in l.images:
            if not (catalog_dir / "images" / img).exists():
                raise CatalogError(f"listing {l.id}: missing image file {img}")
    return cat


def write_catalog(catalog_dir: Path, listings: list[MockListing]) -> Path:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "catalog.json"
    data = Catalog(listings=listings).model_dump(mode="json")
    _check_no_labels(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
