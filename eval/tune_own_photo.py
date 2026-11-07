"""Tune the own-photo route thresholds and lookalike_clip on the DEV split only.

Positives: dev listing images derived from a gold photo (genuine, copied, modified — label
image_sources starting with "gold:"). Hard negatives: dev lookalike images. For each route,
choose the threshold that maximises positive recall subject to ZERO lookalike false matches on dev:

  hash  — grid over (pHash ≤ p, dHash ≤ d) on the same gold variant; ties → smaller thresholds
  clip  — max lookalike similarity + 0.01
  orb   — max inliers over lookalike and unrelated images + 1 (at least 12)

False matches on counterfeit photos (other products with the mark printed on) and unrelated images
are reported for information; own-photo is never decisive for counterfeit rules.

    python -m eval.tune_own_photo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bh.brand_index import load_brand_sync
from bh.config import get_config
from bh.features import OwnPhotoMatcher
from bh.vision import ClipEncoder, open_rgb

ROOT = Path(__file__).resolve().parent.parent


def kind_of(src: str, category: str) -> str:
    if src.startswith("gold:"):
        return "positive"
    if category == "lookalike":
        return "lookalike"
    if src.startswith("blank:") or src.startswith("render:"):
        return "counterfeit_photo"
    return "unrelated"


def tune(labels_path: Path, catalog_dir: Path) -> dict[str, Any]:
    cfg = get_config()
    labels = json.loads(labels_path.read_text())
    catalog = {f"{l['site']}:{l['id']}": l for l in json.loads((catalog_dir / "catalog.json").read_text())["listings"]}
    encoder = ClipEncoder(cfg.vision.clip_model)
    ctx = load_brand_sync(cfg, encoder)
    m = OwnPhotoMatcher(ctx, cfg)

    rows = []
    for key, lab in sorted(labels.items()):
        if lab["split"] != "dev":
            continue
        for fn, src in zip(catalog[key]["images"], lab["image_sources"], strict=False):
            im = open_rgb(catalog_dir / "images" / fn)
            rows.append({"kind": kind_of(src, lab["category"]), "img": im, "src": src, "key": key,
                         "category": lab["category"],
                         "pairs": m.variant_distances(im), "orb": m.orb_inliers(im)})
    vecs = encoder.encode([r["img"] for r in rows])
    for r, v in zip(rows, vecs, strict=True):
        r["clip"] = ctx.nearest_clip(v)[0]

    pos = [r for r in rows if r["kind"] == "positive"]
    look = [r for r in rows if r["kind"] == "lookalike"]
    hits = lambda r, p, d: any(a <= p and b <= d for a, b in r["pairs"])  # noqa: E731

    best = None
    for p in range(0, 21):
        for d in range(0, 25):
            if any(hits(r, p, d) for r in look):
                continue
            rec = sum(hits(r, p, d) for r in pos)
            key = (rec, -(p + d))
            if best is None or key > best[0]:
                best = (key, p, d)
    _, hp, hd = best
    clip_min = round(max(r["clip"] for r in look) + 0.01, 3)
    orb_min = max(12, max(r["orb"] for r in rows if r["kind"] in ("lookalike", "unrelated")) + 1)

    def route_stats(kind: str) -> dict[str, int]:
        rs = [r for r in rows if r["kind"] == kind]
        h = [hits(r, hp, hd) for r in rs]
        c = [r["clip"] >= clip_min for r in rs]
        o = [r["orb"] >= orb_min for r in rs]
        return {"n": len(rs), "hash": sum(h), "clip": sum(c), "orb": sum(o),
                "any": sum(a or b or x for a, b, x in zip(h, c, o, strict=True))}

    # lookalike_clip (listing level): separate dev lookalike listings from dev unrelated listings by the
    # best CLIP similarity of any listing image; threshold halfway between the classes when separable.
    by_listing: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = by_listing.setdefault(r["key"], {"category": r["category"], "clip": 0.0})
        d["clip"] = max(d["clip"], r["clip"])
    look_l = [d["clip"] for d in by_listing.values() if d["category"] == "lookalike"]
    unrel_l = [d["clip"] for d in by_listing.values() if d["category"] == "unrelated"]
    lo, hi = max(unrel_l), min(look_l)
    lookalike_clip = round((lo + hi) / 2, 3) if hi > lo else round(hi, 3)

    return {
        "split": "dev", "n_images": len(rows),
        "thresholds": {"phash_max": hp, "dhash_max": hd, "clip_min": clip_min, "orb_min_inliers": orb_min},
        "lookalike_clip": {"value": lookalike_clip, "dev_unrelated_max": round(lo, 3), "dev_lookalike_min": round(hi, 3),
                           "separable": hi > lo},
        "matches": {k: route_stats(k) for k in ("positive", "lookalike", "counterfeit_photo", "unrelated")},
        "positive_clip_median": sorted(r["clip"] for r in pos)[len(pos) // 2],
        "lookalike_clip_max": max(r["clip"] for r in look),
    }


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "seed" / "labels.json")
    ap.add_argument("--catalog", type=Path, default=cfg.paths.resolve(cfg.mocksites.catalog_dir))
    args = ap.parse_args()
    res = tune(args.labels, args.catalog)
    print(json.dumps(res, indent=2))
    t = res["thresholds"]
    print("\nconfig.yaml own_photo:\n" + "\n".join(f"  {k}: {v}" for k, v in t.items()))
    print(f"config.yaml thresholds.lookalike_clip: {res['lookalike_clip']['value']}")


if __name__ == "__main__":
    main()
