"""ABO-based seed assets: selection integrity, attribution, mark compositing, data separation.

Tests that need the built ABO assets (seed/brand from ABO) skip when they are absent, so the
synthetic fallback alone still passes the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import imagehash
import numpy as np
import pytest
from PIL import Image, ImageOps

from seed import generate as G
from seed.import_abo import POOLS, SELECTION, load_selection, selected_image_ids
from seed.mark import Placement, composite_mark, object_box, prepare_square

ROOT = Path(__file__).resolve().parent.parent
BRAND, OTHER = ROOT / "seed" / "brand", ROOT / "seed" / "other_images"


def _abo_built() -> bool:
    p = BRAND / "brand.json"
    return p.exists() and json.loads(p.read_text()).get("image_source") == "ABO"


needs_abo = pytest.mark.skipif(not _abo_built(), reason="ABO assets not built (python -m seed.import_abo build)")


# ---------------------------------------------------------------- selection file (no network)


def test_selection_pools_are_disjoint() -> None:
    sel = load_selection(SELECTION)
    gold_items = {g["abo_item_id"] for g in sel["gold"]}
    for pool in POOLS:
        assert not gold_items & {e["abo_item_id"] for e in sel[pool]}, pool
    assert not {e["abo_item_id"] for e in sel["blank"]} & {e["abo_item_id"] for e in sel["lookalike"]}
    assert 4 <= len(sel["gold"]) <= 5 and all(3 <= len(g["images"]) <= 4 for g in sel["gold"])
    assert len(selected_image_ids(sel)) == len(set(selected_image_ids(sel)))
    slugs = {g["slug"] for g in sel["gold"]}
    assert all(b["for_product"] in slugs for b in sel["blank"])


def test_selection_titles_have_no_real_brand_names() -> None:
    raw = SELECTION.read_text().lower()
    titles = " ".join(e["title"] + " " + e["title_hard"] for e in load_selection(SELECTION)["unrelated"]).lower()
    for brand in ("amazon", "solimo", "pinzon", "umi", "eono", "rivet", "stone & beam"):
        assert brand not in titles
    assert "cc by 4.0" in raw


def test_load_selection_rejects_reuse(tmp_path: Path) -> None:
    import yaml

    sel = yaml.safe_load(SELECTION.read_text())
    sel["lookalike"].append({"abo_item_id": sel["gold"][0]["abo_item_id"], "image": "x"})
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(sel))
    with pytest.raises(ValueError):
        load_selection(p)


# ---------------------------------------------------------------- mark compositing


def _studio(size: int = 400) -> Image.Image:
    im = Image.new("RGB", (size, size), (255, 255, 255))
    im.paste((184, 115, 51), (150, 60, 250, 380))  # copper-ish bottle body
    return im


def test_prepare_square_pads_studio_and_crops_lifestyle() -> None:
    studio = Image.new("RGB", (300, 600), (255, 255, 255))
    studio.paste((0, 0, 0), (100, 100, 200, 500))
    out = prepare_square(studio, 200)
    assert out.size == (200, 200) and out.getpixel((5, 100)) == (255, 255, 255)  # padded, not cropped
    life = Image.new("RGB", (600, 300), (30, 90, 30))
    assert prepare_square(life, 200).getpixel((5, 5)) == (30, 90, 30)  # cropped, no white padding


def test_object_box_and_mark_stay_on_product() -> None:
    im = _studio()
    assert object_box(im) == pytest.approx((150, 60, 250, 380), abs=4)
    out = composite_mark(im, "VAELO", True, Placement(anchor_y=0.5, rel_width=0.7))
    diff = np.abs(np.asarray(out, dtype=int) - np.asarray(im, dtype=int)).sum(axis=2) > 30
    ys, xs = np.where(diff)
    assert diff.sum() > 200  # mark is visible
    assert xs.min() >= 148 and xs.max() <= 252 and ys.min() >= 58 and ys.max() <= 382  # clipped to the product


def test_ink_contrasts_with_surface() -> None:
    light = Image.new("RGB", (300, 300), (255, 255, 255))
    light.paste((200, 200, 200), (100, 50, 200, 250))
    dark = Image.new("RGB", (300, 300), (255, 255, 255))
    dark.paste((30, 40, 50), (100, 50, 200, 250))
    for im, want_dark in ((light, True), (dark, False)):
        out = np.asarray(composite_mark(im, "VAELO", False, Placement(rel_width=0.8)), dtype=int)
        changed = np.abs(out - np.asarray(im, dtype=int)).sum(axis=2) > 30
        mean = out[changed].mean()
        assert (mean < 150) if want_dark else (mean > 120)


def test_synthetic_fallback_writes_pools(tmp_path: Path) -> None:
    from seed.assets import build_other

    assert build_other(tmp_path / "other")
    pools = G.load_pools(tmp_path / "other")
    assert len(pools.lookalike) == 12 and len(pools.unrelated) == 16 and len(pools.blank) == 8
    assert all({"title", "title_hard", "site_category"} <= set(u.meta) for u in pools.unrelated)


# ---------------------------------------------------------------- built ABO assets


@needs_abo
def test_abo_assets_and_attribution() -> None:
    sel = load_selection(SELECTION)
    products = json.loads((BRAND / "products.json").read_text())
    assert [p["slug"] for p in products] == [g["slug"] for g in sel["gold"]]
    for p, g in zip(products, sel["gold"], strict=True):
        assert p["source"] == {"dataset": "ABO", "abo_item_id": g["abo_item_id"], "abo_image_ids": g["images"],
                               "licence": "CC BY 4.0", "modified": "resized/padded; fictional mark composited"}
    for d in (BRAND, OTHER):
        text = (d / "ATTRIBUTION.md").read_text()
        assert "CC BY 4.0" in text and "creativecommons.org/licenses/by/4.0" in text and "Collins" in text
        for g in sel["gold"]:
            assert g["abo_item_id"] in text
    for pool in POOLS:
        idx = json.loads((OTHER / pool / "index.json").read_text())
        assert idx["source"] == "ABO" and len(idx["items"]) == len(sel[pool])


@needs_abo
def test_abo_gold_photos_carry_the_mark() -> None:
    from seed.import_abo import cache_dir, load_image_index

    cache = cache_dir()
    if not (cache / "metadata" / "images.csv.gz").exists():
        pytest.skip("ABO cache not present")
    index = load_image_index(cache)
    sel = load_selection(SELECTION)
    for g in sel["gold"]:
        for k, iid in enumerate(g["images"]):
            raw = cache / "images" / "original" / index[iid][2]
            if not raw.exists():
                pytest.skip("ABO originals not cached")
            with Image.open(raw) as r:
                base = np.asarray(prepare_square(r, 800), dtype=int)
            gold = np.asarray(Image.open(BRAND / "photos" / f"{g['slug']}_{k + 1}.jpg").convert("RGB"), dtype=int)
            changed = (np.abs(gold - base).sum(axis=2) > 60).mean()
            assert 0.001 < changed < 0.08, (g["slug"], iid, changed)  # a printed mark: visible but small


@needs_abo
def test_abo_generated_data_separation(tmp_path: Path) -> None:
    """Regenerate from the repo's ABO assets and re-check the Phase 3 image properties."""
    res = G.generate(BRAND, OTHER, tmp_path / "catalog", tmp_path / "labels.json", seed=42)
    labels = json.loads(res.labels_path.read_text())
    assert res.counts == G.CATEGORY_COUNTS and res.split_counts == {"dev": 72, "test": 48}
    for v in labels.values():
        assert {"rule_id", "expected_action", "should_report", "expected_secondary_actions"} <= set(v)
    catalog = {f"{l['site']}:{l['id']}": l for l in json.loads(res.catalog_path.read_text())["listings"]}
    gold_imgs = [Image.open(p).convert("RGB") for p in sorted((BRAND / "photos").glob("*.jpg"))]
    hashes = {f: [f(im) for im in gold_imgs] + [f(ImageOps.mirror(im)) for im in gold_imgs]
              for f in (imagehash.phash, imagehash.dhash, imagehash.whash)}

    def dists(fn: str) -> dict[str, int]:
        im = Image.open(tmp_path / "catalog" / "images" / fn).convert("RGB")
        return {f.__name__: int(min(f(im) - g for g in hs)) for f, hs in hashes.items()}

    agree = 0
    for key, v in labels.items():
        for fn in catalog[key]["images"]:
            d = dists(fn)
            if v["category"] in ("authorised", "genuine_resale", "grey_import"):
                assert d["phash"] <= 4, (key, d)
            elif v["category"] == "unrelated":
                assert d["phash"] > 10 and d["dhash"] > 10, (key, d)  # wHash is unreliable on white studio shots
            elif v["category"] == "lookalike" and all(x <= 8 for x in d.values()):
                agree += 1
    # Lookalikes shot on the same backdrop as gold photos fool every hash type; only CLIP separates them.
    # Keep that set small so own-photo matching stays meaningful.
    assert agree <= 6
