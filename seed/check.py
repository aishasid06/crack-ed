"""Measure how separable the seeded images are (Phase 3 checks).

Reports, per label category and image provenance:
- perceptual-hash distance to the nearest gold photo (pHash, gold also mirrored)
- CLIP cosine similarity to the nearest gold photo
and CLIP top-10 retrieval with each gold photo as the query over all listing images.

    python -m seed.check [--labels seed/labels.json] [--catalog data/mocksites] [--json out.json]

Uses ground-truth labels, so it is an offline dataset check — never part of the agent.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image, ImageOps

from bh.brand import load_brand_assets

SEED_DIR = Path(__file__).resolve().parent


def provenance(src: str) -> str:
    if src.startswith("gold:"):
        return "gold-variant" if "+" in src else "gold"
    if src.startswith("blank:") or src.startswith("render:"):
        return "counterfeit-photo"
    return "other"


def _stats(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "p50": float(np.median(a)), "max": float(a.max())}


def run(labels_path: Path, catalog_dir: Path, brand_dir: Path, top_k: int = 10) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    labels = json.loads(labels_path.read_text())
    catalog = {f"{l['site']}:{l['id']}": l for l in json.loads((catalog_dir / "catalog.json").read_text())["listings"]}
    assets = load_brand_assets(brand_dir)
    gold = [(p.slug, assets.image_path(rel)) for p in assets.products for rel in p.images]
    gold_imgs = [Image.open(path).convert("RGB") for _, path in gold]
    gold_hash = [imagehash.phash(im) for im in gold_imgs] + [imagehash.phash(ImageOps.mirror(im)) for im in gold_imgs]

    model = SentenceTransformer("clip-ViT-B-32")
    emb = lambda ims: model.encode(ims, convert_to_numpy=True, normalize_embeddings=True,  # noqa: E731
                                   show_progress_bar=False, batch_size=32)
    gold_vec = emb(gold_imgs)

    rows: list[dict[str, Any]] = []
    for key, lab in labels.items():
        l = catalog[key]
        for fn, src in zip(l["images"], lab["image_sources"], strict=False):
            rows.append({"key": key, "category": lab["category"], "prov": provenance(src), "src": src,
                         "product": lab.get("product"), "img": Image.open(catalog_dir / "images" / fn).convert("RGB")})
    vecs = emb([r["img"] for r in rows])
    for r, v in zip(rows, vecs, strict=True):
        h = imagehash.phash(r["img"])
        r["hash"] = int(min(h - g for g in gold_hash))
        r["clip"] = float((gold_vec @ v).max())

    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = f"{r['category']} / {r['prov']}"
        by[k]["hash"].append(r["hash"])
        by[k]["clip"].append(r["clip"])
    per_group = {k: {"hash": _stats(v["hash"]), "clip": _stats(v["clip"]),
                     "hash_le_8": sum(x <= 8 for x in v["hash"])} for k, v in sorted(by.items())}

    # CLIP top-k retrieval: gold photo → listings (max over each listing's images)
    keys = sorted({r["key"] for r in rows})
    idx = {k: i for i, k in enumerate(keys)}
    lst = np.full((len(gold), len(keys)), -1.0)
    sims = gold_vec @ vecs.T
    for j, r in enumerate(rows):
        col = idx[r["key"]]
        lst[:, col] = np.maximum(lst[:, col], sims[:, j])
    same_product_gold, lookalike_or_unrelated, any_actionable = [], [], []
    for qi, (slug, _) in enumerate(gold):
        top = [keys[c] for c in np.argsort(-lst[qi])[:top_k]]
        same_product_gold.append(sum(
            labels[k].get("product") == slug and any(s.startswith(f"gold:") for s in labels[k]["image_sources"])
            for k in top) / top_k)
        lookalike_or_unrelated.append(sum(labels[k]["category"] in ("lookalike", "unrelated") for k in top))
        any_actionable.append(sum(labels[k]["expected_action"] != "ignore" for k in top) / top_k)
    gold_derived = [r["clip"] for r in rows if r["prov"] in ("gold", "gold-variant")]
    look = [r["clip"] for r in rows if r["category"] == "lookalike"]
    return {
        "per_group": per_group,
        "retrieval": {"queries": len(gold), "k": top_k,
                      "mean_precision_same_product_gold_derived": float(np.mean(same_product_gold)),
                      "mean_lookalike_or_unrelated_in_topk": float(np.mean(lookalike_or_unrelated)),
                      "mean_fraction_not_ignore": float(np.mean(any_actionable))},
        "clip_margin": {"gold_derived_min": float(min(gold_derived)), "gold_derived_p10": float(np.percentile(gold_derived, 10)),
                        "lookalike_max": float(max(look)), "lookalike_p90": float(np.percentile(look, 90))},
    }


def print_report(res: dict[str, Any]) -> None:
    print(f"{'category / provenance':40s} {'n':>3s}  {'pHash min/p50/max':>17s}  ≤8   {'CLIP min/p50/max':>17s}")
    for k, v in res["per_group"].items():
        h, c = v["hash"], v["clip"]
        print(f"{k:40s} {h['n']:3d}  {h['min']:5.0f}/{h['p50']:4.0f}/{h['max']:4.0f}  {v['hash_le_8']:3d}  "
              f"{c['min']:.3f}/{c['p50']:.3f}/{c['max']:.3f}")
    print("retrieval:", json.dumps(res["retrieval"]))
    print("clip margin:", json.dumps(res["clip_margin"]))


def main() -> None:
    from bh.config import get_config

    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=SEED_DIR / "labels.json")
    ap.add_argument("--catalog", type=Path, default=cfg.paths.resolve(cfg.mocksites.catalog_dir))
    ap.add_argument("--brand-dir", type=Path, default=SEED_DIR / "brand")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    res = run(args.labels, args.catalog, args.brand_dir)
    print_report(res)
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
