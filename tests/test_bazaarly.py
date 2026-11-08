"""Bazaarly mock site and its seeding (SPEC Phase 11 item 2)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mocksites.app import MockSettings, create_app
from mocksites.catalog import load_catalog
from seed.bazaarly import VAELO_CASES, add_bazaarly
from tests.test_mocksites import ColourEmbedder

CASES = VAELO_CASES * 2  # 16 listings: two result pages of 10


@pytest.fixture(scope="module")
def bz_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from seed import generate as G

    base = tmp_path_factory.mktemp("bz")
    G.generate(base / "brand", base / "other", base / "catalog", base / "labels.json", seed=5)
    add_bazaarly(base / "catalog", base / "brand", base / "other", base / "labels_bazaarly.json", n_real=0, seed=3,
                 cases=CASES)
    return base / "catalog"


@pytest.fixture
def client(bz_catalog: Path, tmp_path: Path) -> TestClient:
    return TestClient(create_app(MockSettings(catalog_dir=bz_catalog, db_url=f"sqlite:///{tmp_path / 'm.db'}",
                                              portal_user="u@x.example", portal_pass="pw-1234",
                                              uploads_dir=tmp_path / "up", embedder=ColourEmbedder())))


def test_seeding_keeps_other_sites_and_separate_labels(bz_catalog: Path) -> None:
    base = bz_catalog.parent
    cat = load_catalog(bz_catalog)
    bz = cat.for_site("bazaarly")
    assert len(bz) == len(CASES) and len(cat.for_site("echobazaar")) + len(cat.for_site("tradeloft")) == 120
    labels = json.loads((base / "labels_bazaarly.json").read_text())
    assert len(labels) == len(CASES) and all(v["group"] == "bazaarly" for v in labels.values())
    assert all(v["url_path"].startswith("/bazaarly/p/BZ-") for v in labels.values())
    assert {v["split"] for v in labels.values()} <= {"dev", "test"}
    base_labels = json.loads((base / "labels.json").read_text())
    assert not any(k.startswith("bazaarly:") for k in base_labels)  # existing groups unchanged
    raw = (bz_catalog / "catalog.json").read_text()
    assert "bazaarly" in raw and '"group"' not in raw and "counterfeit" not in raw  # no labels in the catalog


def test_reseeding_replaces_only_bazaarly(bz_catalog: Path, tmp_path: Path) -> None:
    import shutil

    d = tmp_path / "cat"
    shutil.copytree(bz_catalog, d)
    base = bz_catalog.parent
    before = {(l.site, l.id) for l in load_catalog(d).listings if l.site != "bazaarly"}
    add_bazaarly(d, base / "brand", base / "other", tmp_path / "l.json", n_real=0, seed=99)
    cat = load_catalog(d)
    assert {(l.site, l.id) for l in cat.listings if l.site != "bazaarly"} == before
    assert len(cat.for_site("bazaarly")) == len(VAELO_CASES)


def test_cookie_modal_blocks_until_choice(client: TestClient) -> None:
    r = client.get("/bazaarly/")
    assert r.status_code == 200 and 'role="dialog"' in r.text and "Cookie choices" in r.text
    r = client.post("/bazaarly/consent", data={"choice": "necessary", "next": "/bazaarly/"}, follow_redirects=False)
    assert r.status_code == 303 and "bz_consent" in r.cookies
    client.cookies.set("bz_consent", "necessary", path="/bazaarly")
    assert 'role="dialog"' not in client.get("/bazaarly/").text


def test_search_only_via_form_and_opaque_results(client: TestClient) -> None:
    assert client.get("/bazaarly/search", params={"term": "vaelo"}).status_code == 405
    r = client.post("/bazaarly/search", data={"term": "vaelo"}, follow_redirects=False)
    loc = r.headers["location"]
    assert r.status_code == 303 and re.fullmatch(r"/bazaarly/results/[0-9a-f]{12}/1", loc) and "vaelo" not in loc
    page = client.get(loc).text
    assert page.count('class="bz-tile"') >= 1 and "/bazaarly/p/BZ-" in page
    assert client.get("/bazaarly/results/000000000000/1").status_code == 404


def test_next_button_pages_results(client: TestClient) -> None:
    r = client.post("/bazaarly/search", data={"term": ""}, follow_redirects=True)  # everything
    assert 'class="bz-next"' in r.text and "<a" not in r.text.split('class="bz-pager"')[1].split("</nav>")[0]
    nxt = re.search(r'data-go="([^"]+)"', r.text).group(1)
    assert client.get(nxt).status_code == 200


def test_product_details_in_tabs(client: TestClient, bz_catalog: Path) -> None:
    l = load_catalog(bz_catalog).for_site("bazaarly")[0]
    html = client.get(f"/bazaarly/p/{l.id}").text
    assert 'role="tablist"' in html and html.count('role="tab"') == 3
    assert re.search(r'id="panel-details"[^>]*hidden', html) and re.search(r'id="panel-seller"[^>]*hidden', html)
    assert l.seller_name in html  # in the DOM, but hidden until the Seller tab is opened
    assert client.get("/bazaarly/p/BZ-00000").status_code == 404


def test_index_lists_bazaarly(client: TestClient) -> None:
    assert "Bazaarly" in client.get("/").text
