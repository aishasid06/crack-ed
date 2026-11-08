"""Delete what was collected from a real (read_only) marketplace once the measurement is done.

    python -m bh.discovery.purge --site <name>          # shows what would be deleted
    python -m bh.discovery.purge --site <name> --yes    # deletes it
    python -m bh.discovery.purge --brand-dir data/brands/<slug> [--yes]   # a real-brand demo brand

A real-brand demo brand (``demo_real_brand: true``) is purged with its Brand, Product, TradeMark and
AuthorisedSeller rows, its cached CLIP embeddings and its folder (which must be under ``data/brands/``); its runs
are kept but unlinked from it. Fictional brands are never purged this way.

Removes the site's Listing rows with their Features and Case rows, its listing folders under
``data/listings/<site>/`` (photos, screenshots), and redacts the payloads of its trace tasks (run totals and
timings stay). Only read_only sites can be purged, so mock data is never removed by accident.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.config import AppConfig, get_config
from bh.db import AuthorisedSeller, Brand, Case, Features, Listing, Product, Run, Task, TradeMark, get_engine, session_scope


def purge_site(cfg: AppConfig, site: str, *, apply: bool, engine: Engine | None = None) -> dict[str, Any]:
    if site not in cfg.sites:
        raise ValueError(f"site {site!r} is not configured")
    if cfg.site(site).mode != "read_only":
        raise ValueError(f"site {site!r} is not read_only; only real-site data can be purged")
    eng = engine or get_engine()
    folder = cfg.paths.resolve(cfg.paths.data_dir) / "listings" / site
    with Session(eng) as s:
        ids = [l.id for l in s.exec(select(Listing).where(Listing.platform == site))]
        n_cases = len(list(s.exec(select(Case).where(Case.listing_id.in_(ids))))) if ids else 0  # type: ignore[union-attr]
        n_tasks = len(list(s.exec(select(Task).where(Task.site == site))))
    out = {"site": site, "listings": len(ids), "cases": n_cases, "trace_tasks_redacted": n_tasks,
           "folder": str(folder) if folder.exists() else None, "applied": apply}
    if not apply:
        return out
    with session_scope(eng) as s:
        if ids:
            for model in (Case, Features):
                for row in s.exec(select(model).where(model.listing_id.in_(ids))):  # type: ignore[attr-defined]
                    s.delete(row)
            s.flush()
            for row in s.exec(select(Listing).where(Listing.platform == site)):
                s.delete(row)
        for t in s.exec(select(Task).where(Task.site == site)):
            t.payload = {"purged": True}
            t.waiting_url = None
            t.error = "purged" if t.error else None
            t.dedup_key = f"purged:{t.id}"
    if folder.exists():
        shutil.rmtree(folder)
    return out


def purge_brand(cfg: AppConfig, folder: Path, *, apply: bool, engine: Engine | None = None) -> dict[str, Any]:
    from bh.brand import validate_brand_dir

    check = validate_brand_dir(folder)
    if not check.real_brand_demo:
        raise ValueError(f"{folder} is not a real-brand demo brand (demo_real_brand: true); not purged")
    root = Path(check.brand_dir).resolve()
    brands_root = (cfg.paths.resolve(cfg.paths.data_dir) / "brands").resolve()
    if brands_root not in root.parents:
        raise ValueError(f"{root} is not under {brands_root}; not purged")
    slug = check.slug or root.name
    eng = engine or get_engine()
    emb = cfg.paths.resolve(cfg.vision.embeddings_dir) / slug
    with Session(eng) as s:
        brand = s.exec(select(Brand).where(Brand.slug == slug)).first()
        runs = list(s.exec(select(Run).where(Run.brand_id == brand.id))) if brand else []
    out = {"brand": slug, "db_brand": brand is not None, "runs_unlinked": len(runs), "folder": str(root),
           "embeddings": str(emb) if emb.exists() else None, "applied": apply}
    if not apply:
        return out
    if brand is not None:
        with session_scope(eng) as s:
            for model in (AuthorisedSeller, Product, TradeMark):
                for row in s.exec(select(model).where(model.brand_id == brand.id)):  # type: ignore[attr-defined]
                    s.delete(row)
            for run in s.exec(select(Run).where(Run.brand_id == brand.id)):
                run.brand_id = None
            s.flush()
            row = s.get(Brand, brand.id)
            if row is not None:
                s.delete(row)
    if emb.exists():
        shutil.rmtree(emb)
    shutil.rmtree(root)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--site", help="a read_only (real) site")
    what.add_argument("--brand-dir", type=Path, help="a real-brand demo brand folder under data/brands/")
    ap.add_argument("--yes", action="store_true", help="really delete (default: show what would be deleted)")
    args = ap.parse_args()
    try:
        cfg = get_config()
        out = (purge_site(cfg, args.site, apply=args.yes) if args.site
               else purge_brand(cfg, args.brand_dir, apply=args.yes))
        print(json.dumps(out, indent=2))
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
