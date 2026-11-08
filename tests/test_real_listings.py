"""Real ABO listings mixed into the mock catalog (offline: a tiny fake ABO cache stands in for the download)."""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from mocksites.catalog import load_catalog
from seed import import_abo as I
from seed import real_listings as R
from tests.test_browser import seeded_catalog  # noqa: F401 (fixture)


def _rec(item_id: str, ptype: str, name: str, lang: str = "en_GB", images: int = 2) -> dict[str, Any]:
    return {"item_id": item_id, "product_type": [{"value": ptype}], "item_name": [{"language_tag": lang, "value": name}],
            "brand": [{"language_tag": lang, "value": "TestBrand"}],
            "bullet_point": [{"language_tag": lang, "value": f"{name} bullet"}],
            "main_image_id": f"{item_id}a", "other_image_id": [f"{item_id}{c}" for c in "bcd"[: images - 1]]}


@pytest.fixture
def fake_abo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gold_id = I.load_selection()["gold"][0]["abo_item_id"]
    records = [
        _rec("R1", "BOTTLE", "Steel Water Bottle 750ml"),
        _rec("R2", "DRINKING_CUP", "Highball Glasses, Set of 6"),
        _rec("R3", "DRINKING_CUP", "Highball Glasses, Set of 6"),     # duplicate name → dropped
        _rec("R4", "COFFEE", "Dark Roast Coffee Cups 80 Ct"),         # coffee pods → dropped
        _rec("R5", "JAR", "Glass Storage Jar with Lid"),              # not drinkware → dropped
        _rec("R6", "BOTTLE", "Wasserflasche Edelstahl", lang="de_DE"),  # no English title → dropped
        _rec(gold_id, "BOTTLE", "Gold Item Bottle"),                  # used for seed photos → dropped
    ]
    cache = tmp_path / "abo"
    (cache / "metadata").mkdir(parents=True)
    for k, shard in enumerate(I.LISTING_SHARDS):
        with gzip.open(cache / "metadata" / shard, "wt", encoding="utf-8") as fh:
            for r in (records if k == 0 else []):
                fh.write(json.dumps(r) + "\n")
    buf = io.StringIO()
    w = csv.DictWriter(buf, ["image_id", "height", "width", "path"])
    w.writeheader()
    paths: dict[str, Path] = {}
    for r in records:
        for iid in [r["main_image_id"], *r["other_image_id"]]:
            p = cache / "img" / f"{iid}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (300, 200), (90, 120, 150)).save(p)
            paths[iid] = p
            w.writerow({"image_id": iid, "height": 200, "width": 300, "path": f"x/{iid}.jpg"})
    with gzip.open(cache / "metadata" / "images.csv.gz", "wt") as fh:
        fh.write(buf.getvalue())
    monkeypatch.setattr(I, "ensure_metadata", lambda c: None)
    monkeypatch.setattr(I, "download_images", lambda c, idx, ids, size: {i: paths[i] for i in ids if i in paths})
    return cache


def test_adds_real_listings_as_ignore_without_touching_synthetic(seeded_catalog: Path, fake_abo: Path,  # noqa: F811
                                                                tmp_path: Path) -> None:
    catalog_dir = tmp_path / "catalog"
    shutil.copytree(seeded_catalog, catalog_dir)
    base_labels = seeded_catalog.parent / "labels.json"
    base_before = base_labels.read_text()
    synthetic = {(l.site, l.id): l for l in load_catalog(catalog_dir).listings}
    out = tmp_path / "labels_real.json"

    res = R.add_real_listings(catalog_dir, base_labels, out, n=10, cache=fake_abo)
    assert res["candidates"] == 2 and res["added"] == 2 and res["synthetic_kept"] == len(synthetic)
    cat = load_catalog(catalog_dir)  # validates: no ground-truth keys, every image file exists
    assert len(cat.listings) == len(synthetic) + 2
    assert all(synthetic[(l.site, l.id)] == l for l in cat.listings if (l.site, l.id) in synthetic)
    real = [l for l in cat.listings if (l.site, l.id) not in synthetic]
    assert {l.title for l in real} == {"Steel Water Bottle 750ml", "Highball Glasses, Set of 6"}
    assert {l.site for l in real} == {"echobazaar", "tradeloft"}

    labels = json.loads(out.read_text())
    assert base_labels.read_text() == base_before  # synthetic-only baseline left as it is
    added = {k: v for k, v in labels.items() if v.get("source") == "abo"}
    assert len(labels) == len(json.loads(base_before)) + 2 and len(added) == 2
    for v in added.values():
        assert (v["category"], v["expected_action"], v["should_report"], v["difficulty"]) == \
            ("unrelated", "ignore", False, "real")
        assert v["split"] in ("dev", "test") and v["abo_item_id"] in ("R1", "R2")
    raw = json.loads((catalog_dir / "catalog.json").read_text())["listings"]
    raw_real = [l for l in raw if (l["site"], l["id"]) not in synthetic]
    assert all(set(l) == set(raw[0]) for l in raw_real)  # same published fields as any listing, nothing extra
    blob = json.dumps(raw_real)
    assert not any(s in blob for s in ("R1", "R2", "abo:", '"source"', '"difficulty"', "unrelated"))

    # rerun replaces the earlier real listings (and their images) instead of adding more
    old_images = {img for l in real for img in l.images}
    R.add_real_listings(catalog_dir, base_labels, out, n=1, seed=3, cache=fake_abo)
    cat2 = load_catalog(catalog_dir)
    assert len(cat2.listings) == len(synthetic) + 1
    still = {img for l in cat2.listings for img in l.images}
    assert not any((catalog_dir / "images" / img).exists() for img in old_images - still)
    assert sum(v.get("source") == "abo" for v in json.loads(out.read_text()).values()) == 1
