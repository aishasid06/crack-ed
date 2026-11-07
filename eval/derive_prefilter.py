"""Derive the prefilter CLIP gate from the DEV split only (no test-split leakage).

Method: for every dev listing whose expected_action is not ``ignore``, take its search-result
thumbnail (first listing image) and check the two non-CLIP prefilter routes (pHash to any gold
variant ≤ prefilter_hash, brand fuzzy score ≥ prefilter_brand_fuzzy). The CLIP gate must keep every
actionable dev listing those routes miss, so:

    gate = floor_to_0.01( min CLIP(thumbnail, nearest gold) over those listings − margin )

The margin (default 0.02) absorbs embedding noise. Also reports how many dev ``ignore`` listings
pass the gate (the capture cost).

    python -m eval.derive_prefilter [--margin 0.02]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from bh.brand_index import load_brand_sync
from bh.config import get_config
from bh.discovery.matching import BrandMatcher
from bh.vision import ClipEncoder, open_rgb

ROOT = Path(__file__).resolve().parent.parent


def derive(labels_path: Path, catalog_dir: Path, margin: float = 0.02) -> dict[str, Any]:
    cfg = get_config()
    th = cfg.thresholds
    labels = json.loads(labels_path.read_text())
    catalog = {f"{l['site']}:{l['id']}": l for l in json.loads((catalog_dir / "catalog.json").read_text())["listings"]}
    ctx = load_brand_sync(cfg, ClipEncoder(cfg.vision.clip_model))
    matcher = BrandMatcher([*ctx.marks, ctx.name])
    encoder = ClipEncoder(cfg.vision.clip_model)

    dev = {k: v for k, v in labels.items() if v["split"] == "dev"}
    keys = sorted(dev)
    thumbs = [open_rgb(catalog_dir / "images" / catalog[k]["images"][0]) for k in keys]
    vecs = encoder.encode(thumbs)
    rows = []
    for k, im, v in zip(keys, thumbs, vecs, strict=True):
        hash_d, _ = ctx.nearest_hash(im)
        clip, _ = ctx.nearest_clip(v)
        rows.append({"key": k, "category": dev[k]["category"], "actionable": dev[k]["expected_action"] != "ignore",
                     "clip": clip, "hash": hash_d, "brand": matcher.score(catalog[k]["title"])})
    other_routes = lambda r: r["hash"] <= th.prefilter_hash or r["brand"] >= th.prefilter_brand_fuzzy  # noqa: E731
    need_clip = [r for r in rows if r["actionable"] and not other_routes(r)]
    basis = need_clip or [r for r in rows if r["actionable"]]
    lowest = min(basis, key=lambda r: r["clip"])
    gate = math.floor((lowest["clip"] - margin) * 100) / 100
    ignore = [r for r in rows if not r["actionable"]]
    return {
        "method": "min dev-actionable thumbnail CLIP among listings missed by hash/brand routes, minus margin",
        "split": "dev", "n_dev": len(rows), "margin": margin, "gate": gate,
        "n_actionable_needing_clip": len(need_clip),
        "lowest": {k: lowest[k] for k in ("key", "category", "clip", "hash", "brand")},
        "dev_ignore_passing_gate": sum(r["clip"] >= gate or other_routes(r) for r in ignore),
        "n_dev_ignore": len(ignore),
        "actionable_recall_at_gate": sum(r["clip"] >= gate or other_routes(r) for r in rows if r["actionable"])
        / max(1, sum(r["actionable"] for r in rows)),
    }


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "seed" / "labels.json")
    ap.add_argument("--catalog", type=Path, default=cfg.paths.resolve(cfg.mocksites.catalog_dir))
    ap.add_argument("--margin", type=float, default=0.02)
    args = ap.parse_args()
    print(json.dumps(derive(args.labels, args.catalog, args.margin), indent=2))


if __name__ == "__main__":
    main()
