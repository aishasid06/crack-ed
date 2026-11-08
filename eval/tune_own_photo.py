"""Tune the own-photo route thresholds and lookalike_clip on the DEV split only.

Positives: dev listing images derived from a gold photo (genuine, copied, modified — label
image_sources starting with "gold:"). Hard negatives: dev lookalike images. For each route,
choose the threshold that maximises positive recall subject to ZERO lookalike false matches on dev:

  hash  — grid over (pHash ≤ p, dHash ≤ d) on the same gold variant; ties → smaller thresholds
  clip  — max lookalike similarity + 0.01
  orb   — max inliers over lookalike and unrelated images + 1 (at least 12)

False matches on counterfeit photos (other products with the mark printed on) and unrelated images
are reported for information; own-photo is never decisive for counterfeit rules.

With real listings in the labels (``--labels seed/labels_real.json``, see seed/real_listings.py), two more
values are tuned against the dev split's real listings, which the thresholds above never saw:

  orb_clip_min          — ORB counts only when the same image's CLIP similarity reaches this: max CLIP of
                          any negative image (lookalike, unrelated, real) that passes the ORB inlier
                          threshold + 0.01
  lookalike_silent_clip — listing level, for lookalikes WITHOUT copying wording: halfway between the
                          best CLIP of dev non-lookalike listings without such wording and the lowest CLIP of
                          dev lookalikes without it

The original thresholds are computed on synthetic listings only, exactly as before.

    python -m eval.tune_own_photo [--labels seed/labels_real.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from bh.brand_index import load_brand_sync
from bh.config import get_config
from bh.features import OwnPhotoMatcher, phrase_hits
from bh.rules import get_rules
from bh.vision import ClipEncoder, open_rgb

ROOT = Path(__file__).resolve().parent.parent


def kind_of(src: str, category: str) -> str:
    if src.startswith("abo:"):
        return "real"  # real non-brand listing (seed/real_listings.py)
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

    inspired = get_rules().vocabulary.get("inspired", [])
    rows = []
    wording: dict[str, bool] = {}
    for key, lab in sorted(labels.items()):
        if lab["split"] != "dev":
            continue
        entry = catalog[key]
        wording[key] = bool(phrase_hits(f"{entry['title']}\n{entry['description']}", inspired))
        for fn, src in zip(entry["images"], lab["image_sources"], strict=False):
            im = open_rgb(catalog_dir / "images" / fn)
            rows.append({"kind": kind_of(src, lab["category"]), "img": im, "src": src, "key": key,
                         "category": lab["category"], "real": lab.get("source") == "abo",
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
    # ORB corroboration: no negative image (incl. real listings) may pass ORB together with this CLIP level
    orb_neg = [r["clip"] for r in rows if r["kind"] in ("lookalike", "unrelated", "real") and r["orb"] >= orb_min]
    orb_clip_min = round(max(orb_neg) + 0.01, 3) if orb_neg else 0.0

    def route_stats(kind: str) -> dict[str, int]:
        rs = [r for r in rows if r["kind"] == kind]
        h = [hits(r, hp, hd) for r in rs]
        c = [r["clip"] >= clip_min for r in rs]
        o = [r["orb"] >= orb_min and r["clip"] >= orb_clip_min for r in rs]
        return {"n": len(rs), "hash": sum(h), "clip": sum(c), "orb": sum(o),
                "orb_uncorroborated": sum(r["orb"] >= orb_min for r in rs),
                "any": sum(a or b or x for a, b, x in zip(h, c, o, strict=True))}

    # lookalike_clip (listing level): separate dev lookalike listings from dev unrelated listings by the
    # best CLIP similarity of any listing image; threshold halfway between the classes when separable.
    # Synthetic listings only, as originally tuned.
    by_listing: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = by_listing.setdefault(r["key"], {"category": r["category"], "clip": 0.0, "real": r["real"],
                                             "wording": wording[r["key"]]})
        d["clip"] = max(d["clip"], r["clip"])
    look_l = [d["clip"] for d in by_listing.values() if d["category"] == "lookalike"]
    unrel_l = [d["clip"] for d in by_listing.values() if d["category"] == "unrelated" and not d["real"]]
    lo, hi = max(unrel_l), min(look_l)
    lookalike_clip = round((lo + hi) / 2, 3) if hi > lo else round(hi, 3)

    # lookalike_silent_clip: only listings that can reach the lookalike rule matter: no own-photo match on any
    # image (no_hash_match) and no copying wording (wording already meets the element)
    for r in rows:
        own = hits(r, hp, hd) or r["clip"] >= clip_min or (r["orb"] >= orb_min and r["clip"] >= orb_clip_min)
        by_listing[r["key"]]["own"] = by_listing[r["key"]].get("own", False) or own
    candidates = [d for d in by_listing.values() if not d["own"] and not d["wording"]]
    silent_look = [d["clip"] for d in candidates if d["category"] == "lookalike"]
    silent_neg = [d["clip"] for d in candidates if d["category"] == "unrelated"]
    s_lo = max(silent_neg) if silent_neg else 0.0
    s_hi = min(silent_look) if silent_look else None
    if s_hi is None:
        silent_clip = round(s_lo + 0.01, 3)
    elif s_hi > s_lo:
        silent_clip = round((s_lo + s_hi) / 2, 3)
    else:  # not separable: keep every dev lookalike (round DOWN, or the lowest one falls just below)
        silent_clip = math.floor(s_hi * 1000) / 1000

    return {
        "split": "dev", "n_images": len(rows),
        "thresholds": {"phash_max": hp, "dhash_max": hd, "clip_min": clip_min, "orb_min_inliers": orb_min,
                       "orb_clip_min": orb_clip_min},
        "orb_clip_min": {"value": orb_clip_min, "negative_images_passing_orb": len(orb_neg),
                         "negative_clip_max": round(max(orb_neg), 3) if orb_neg else None},
        "lookalike_clip": {"value": lookalike_clip, "dev_unrelated_max": round(lo, 3), "dev_lookalike_min": round(hi, 3),
                           "separable": hi > lo},
        "lookalike_silent_clip": {"value": silent_clip, "dev_unrelated_without_wording_max": round(s_lo, 3),
                                  "dev_lookalikes_without_wording": len(silent_look),
                                  "dev_lookalike_without_wording_min": round(s_hi, 3) if s_hi is not None else None,
                                  "separable": s_hi is None or s_hi > s_lo},
        "matches": {k: route_stats(k) for k in ("positive", "lookalike", "counterfeit_photo", "unrelated", "real")},
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
    print(f"config.yaml thresholds.lookalike_silent_clip: {res['lookalike_silent_clip']['value']}")


if __name__ == "__main__":
    main()
