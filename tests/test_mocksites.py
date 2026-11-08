"""Hits every mock-site route (SPEC §9 Phase 2)."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mocksites.app import ChallengeGate, MockSettings, create_app
from mocksites.catalog import CatalogError, MockListing, load_catalog, write_catalog

USER, PASS = "lawyer@vaelo.example", "portal-test-pass"
RED, GREEN, BLUE = (220, 30, 30), (30, 200, 30), (30, 30, 220)


class ColourEmbedder:
    """Deterministic test embedder (mean colour), standing in for CLIP so tests don't download a model."""

    name = "test-mean-colour"

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        v = np.array([np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) for im in images])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _jpeg(rgb: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), rgb).save(buf, "JPEG")
    return buf.getvalue()


def _png(rgb: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), rgb).save(buf, "PNG")
    return buf.getvalue()


def _listing(i: int, site: str, **kw: object) -> MockListing:
    base = dict(id=f"{1000 + i}" if site == "mockbay" else f"TL-{i}", site=site, title=f"Vaelo bottle {i}",
                description="Steel bottle", price=10.0 + i, seller_name=f"seller{i % 3}", seller_location="Leeds, UK",
                seller_age_days=100, seller_rating=4.5, images=[f"img_{site}_{i}.jpg"], site_category="Drinkware")
    base.update(kw)
    return MockListing(**base)  # type: ignore[arg-type]


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    d = tmp_path / "catalog"
    (d / "images").mkdir(parents=True)
    listings = [_listing(i, "mockbay") for i in range(30)]
    listings.append(_listing(99, "mockbay", title="Garden hose", description="Green hose", price=5.0))
    colours = [RED, GREEN, BLUE]
    tl = [_listing(i, "tradeloft", site_category="Outdoor" if i % 4 == 0 else "Drinkware") for i in range(20)]
    listings += tl
    for l in listings:
        rgb = colours[int(re.sub(r"\D", "", l.id)) % 3] if l.site == "tradeloft" else (100, 100, 100)
        (d / "images" / l.images[0]).write_bytes(_jpeg(rgb))
    write_catalog(d, listings)
    return d


def _settings(tmp_path: Path, catalog_dir: Path, **kw: object) -> MockSettings:
    s = MockSettings(catalog_dir=catalog_dir, db_url=f"sqlite:///{tmp_path / 'mock.db'}", portal_user=USER,
                     portal_pass=PASS, uploads_dir=tmp_path / "uploads", challenge_probability=0.0,
                     challenge_rapid_requests=10_000, embedder=ColourEmbedder(), seed=1)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@pytest.fixture
def client(tmp_path: Path, catalog_dir: Path) -> TestClient:
    return TestClient(create_app(_settings(tmp_path, catalog_dir)))


def _login(c: TestClient) -> None:
    r = c.post("/portal/login", data={"username": USER, "password": PASS, "next": "/portal/"}, follow_redirects=False)
    assert r.status_code == 303 and "portal_session" in r.cookies


# ---------------------------------------------------------------- generic


def test_index_and_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/")
    assert r.status_code == 200 and "31 listings" in r.text and "20 listings" in r.text


def test_static_and_media(client: TestClient) -> None:
    assert client.get("/static/mockbay.css").status_code == 200
    assert client.get("/static/tradeloft.js").status_code == 200
    r = client.get("/media/img_mockbay_0.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


# ---------------------------------------------------------------- MockBay


def test_mockbay_consent_banner_until_dismissed(client: TestClient) -> None:
    r = client.get("/mockbay/")
    assert r.status_code == 200 and 'id="cookie-consent"' in r.text
    r = client.post("/mockbay/consent", data={"choice": "all", "next": "/mockbay/search?q=vaelo"},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/mockbay/search?q=vaelo"
    client.cookies.set("mb_consent", "all", path="/mockbay")
    assert 'id="cookie-consent"' not in client.get("/mockbay/").text


def test_mockbay_consent_rejects_offsite_redirect(client: TestClient) -> None:
    r = client.post("/mockbay/consent", data={"choice": "essential", "next": "https://evil.example/"},
                    follow_redirects=False)
    assert r.headers["location"] == "/mockbay/"


def test_mockbay_search_pagination_and_sort(client: TestClient) -> None:
    r1 = client.get("/mockbay/search", params={"q": "vaelo", "page": 1})
    assert r1.status_code == 200
    assert r1.text.count('class="s-item"') == 12
    assert '<span class="srp-count__n">30</span>' in r1.text  # hose excluded
    assert 'class="pagination__next"' in r1.text
    r3 = client.get("/mockbay/search", params={"q": "vaelo", "page": 3})
    assert r3.text.count('class="s-item"') == 6 and 'pagination__next' not in r3.text
    cheap = client.get("/mockbay/search", params={"q": "vaelo", "sort": "price_asc"})
    prices = [float(p) for p in re.findall(r'class="s-item__price">£([\d.]+)', cheap.text)]
    assert prices == sorted(prices) and prices[0] == 10.0


def test_mockbay_fuzzy_match(client: TestClient) -> None:
    r = client.get("/mockbay/search", params={"q": "vaello"})
    assert '<span class="srp-count__n">30</span>' in r.text


def test_mockbay_empty_query_and_bad_sort(client: TestClient) -> None:
    assert '<span class="srp-count__n">31</span>' in client.get("/mockbay/search").text
    assert client.get("/mockbay/search", params={"sort": "bogus"}).status_code == 422


def test_mockbay_item(client: TestClient) -> None:
    r = client.get("/mockbay/item/1003")
    assert r.status_code == 200
    for frag in ('class="x-item-title">Vaelo bottle 3<', 'class="x-seller-name">seller0<', "£13.00",
                 'class="x-seller-age">100<', "img_mockbay_3.jpg"):
        assert frag in r.text
    assert client.get("/mockbay/item/nope").status_code == 404


# ---------------------------------------------------------------- TradeLoft


def test_tradeloft_home_form(client: TestClient) -> None:
    r = client.get("/tradeloft/")
    assert r.status_code == 200 and 'id="tl-search-form"' in r.text
    assert '<option value="Outdoor"' in r.text and '<option value="Drinkware"' in r.text


def test_tradeloft_search_and_load_more(client: TestClient) -> None:
    r = client.get("/tradeloft/search", params={"q": "vaelo"})
    assert r.text.count('class="tl-row"') == 8 and "<strong>20</strong>" in r.text
    assert 'id="tl-load-more"' in r.text and 'data-next-offset="8"' in r.text
    more = client.get("/tradeloft/search/more", params={"q": "vaelo", "offset": 8})
    assert more.text.count('class="tl-row"') == 8 and 'data-has-more="true"' in more.text
    last = client.get("/tradeloft/search/more", params={"q": "vaelo", "offset": 16})
    assert last.text.count('class="tl-row"') == 4 and 'data-has-more="false"' in last.text


def test_tradeloft_filters(client: TestClient) -> None:
    r = client.get("/tradeloft/search", params={"q": "", "category": "Outdoor"})
    assert "<strong>5</strong>" in r.text
    r = client.get("/tradeloft/search", params={"q": "vaelo", "min_price": "15", "max_price": "20"})
    assert "<strong>6</strong>" in r.text  # prices 15..20
    r = client.get("/tradeloft/search", params={"q": "vaelo", "min_price": "", "max_price": ""})
    assert "<strong>20</strong>" in r.text
    assert client.get("/tradeloft/search", params={"min_price": "abc"}).status_code == 422


def test_tradeloft_listing(client: TestClient) -> None:
    r = client.get("/tradeloft/listing/TL-4")
    assert r.status_code == 200
    assert '<table class="tl-specs">' in r.text and 'class="tl-seller-name">seller1<' in r.text
    assert "Trading on TradeLoft for 100 days" in r.text
    assert client.get("/tradeloft/listing/TL-999").status_code == 404


def test_tradeloft_visual_search_ranks_by_similarity(client: TestClient) -> None:
    assert 'id="tl-visual-form"' in client.get("/tradeloft/visual-search").text
    r = client.post("/tradeloft/visual-search", files={"photo": ("q.jpg", _jpeg(BLUE), "image/jpeg")})
    assert r.status_code == 200
    ids = re.findall(r'href="/tradeloft/listing/(TL-\d+)" class="tl-row-thumb"', r.text)
    assert ids, "no results"
    blue = {f"TL-{i}" for i in range(20) if i % 3 == 2}
    assert set(ids[: len(blue)]) == blue  # all blue listings ranked first


def test_tradeloft_visual_search_rejects_non_image(client: TestClient) -> None:
    r = client.post("/tradeloft/visual-search", files={"photo": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_visual_cache_written(client: TestClient, catalog_dir: Path) -> None:
    client.post("/tradeloft/visual-search", files={"photo": ("q.jpg", _jpeg(RED), "image/jpeg")})
    assert (catalog_dir / "clip_cache.npz").exists()


# ---------------------------------------------------------------- portal


def test_portal_requires_login(client: TestClient) -> None:
    r = client.get("/portal/submissions", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/portal/login?next=%2Fportal%2Fsubmissions"
    assert client.post("/portal/report/new", follow_redirects=False).status_code == 303


def test_portal_bad_login(client: TestClient) -> None:
    r = client.post("/portal/login", data={"username": USER, "password": "wrong"})
    assert r.status_code == 401 and "Incorrect email or password" in r.text
    assert 'id="login-form"' in client.get("/portal/login").text


def test_portal_login_next_is_sanitised(client: TestClient) -> None:
    r = client.post("/portal/login", data={"username": USER, "password": PASS, "next": "https://evil.example"},
                    follow_redirects=False)
    assert r.headers["location"] == "/portal/"


def _new_draft(c: TestClient) -> str:
    r = c.post("/portal/report/new", follow_redirects=False)
    m = re.match(r"/portal/report/(D[0-9a-f]+)/step/1", r.headers["location"])
    assert m
    return m.group(1)


STEP1 = {"owner_name": "Vaelo Ltd (fictional)", "owner_email": "legal@vaelo.example",
         "owner_address": "1 Fictional Way, London", "relationship": "owner"}
STEP2 = {"right_type": "trade_mark", "registration_number": "UK00004123456", "classes": "21", "territory": "UK"}
STEP5 = {"decl_good_faith": "on", "decl_accurate": "on", "decl_authorised": "on", "signature": "A. Lawyer"}


def _step3(urls: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for u in urls:
        out += [("url", u), ("statute", "Trade Marks Act 1994"), ("section", "s.10(1)"),
                ("details", "Identical sign VAELO used for identical goods (class 21).")]
    return out


def _complete_draft(c: TestClient, urls: list[str]) -> str:
    did = _new_draft(c)
    base = f"/portal/report/{did}"
    assert c.post(f"{base}/step/1", data=STEP1, follow_redirects=False).headers["location"] == f"{base}/step/2"
    assert c.post(f"{base}/step/2", data=STEP2, follow_redirects=False).headers["location"] == f"{base}/step/3"
    r = c.post(f"{base}/step/3", content=_urlenc(_step3(urls)),
               headers={"content-type": "application/x-www-form-urlencoded"}, follow_redirects=False)
    assert r.headers["location"] == f"{base}/step/4"
    r = c.post(f"{base}/step/4", files=[("evidence", ("shot1.png", _png(), "image/png"))], follow_redirects=False)
    assert r.headers["location"] == f"{base}/step/5"
    assert c.post(f"{base}/step/5", data=STEP5, follow_redirects=False).headers["location"] == f"{base}/review"
    return did


def _urlenc(pairs: list[tuple[str, str]]) -> str:
    from urllib.parse import urlencode

    return urlencode(pairs)


LISTING_URLS = ["http://testserver/mockbay/item/1003", "http://127.0.0.1:8000/tradeloft/listing/TL-4"]


def test_portal_full_flow_submit_and_list(client: TestClient) -> None:
    _login(client)
    assert "Start a new IP report" in client.get("/portal/").text
    did = _complete_draft(client, LISTING_URLS)

    review = client.get(f"/portal/report/{did}/review")
    assert review.status_code == 200 and 'id="final-submit"' in review.text
    assert "UK00004123456" in review.text and LISTING_URLS[1] in review.text and "s.10(1)" in review.text
    # nothing submitted until the final click
    assert "No reports submitted yet" in client.get("/portal/submissions").text

    r = client.post(f"/portal/report/{did}/submit", follow_redirects=False)
    ref_url = r.headers["location"]
    assert r.status_code == 303 and ref_url.startswith("/portal/submissions/IPR-")
    detail = client.get(ref_url)
    assert 'id="submit-confirmation"' in detail.text
    ref = re.search(r'id="submission-ref">(IPR-[0-9A-F]+)<', detail.text).group(1)

    listing = client.get("/portal/submissions").text
    assert ref in listing and LISTING_URLS[0] in listing and listing.count('class="pt-submission-row"') == 1

    # idempotent re-submit, and the draft can no longer be edited
    again = client.post(f"/portal/report/{did}/submit", follow_redirects=False)
    assert again.headers["location"] == f"/portal/submissions/{ref}"
    assert client.get(f"/portal/report/{did}/step/1", follow_redirects=False).headers["location"] == \
        f"/portal/submissions/{ref}"
    assert client.get("/portal/submissions").text.count('class="pt-submission-row"') == 1
    # evidence is retrievable
    assert client.get(f"/portal/evidence/{did}/01_shot1.png").status_code == 200


def test_portal_step_guard_and_review_requires_completion(client: TestClient) -> None:
    _login(client)
    did = _new_draft(client)
    r = client.get(f"/portal/report/{did}/step/4", follow_redirects=False)
    assert r.headers["location"] == f"/portal/report/{did}/step/1"
    r = client.get(f"/portal/report/{did}/review", follow_redirects=False)
    assert r.headers["location"] == f"/portal/report/{did}/step/1"
    r = client.post(f"/portal/report/{did}/submit", follow_redirects=False)
    assert r.headers["location"] == f"/portal/report/{did}/step/1"
    assert client.get(f"/portal/report/{did}/step/9").status_code == 404
    assert client.get("/portal/report/Dnope/step/1").status_code == 404


def test_portal_validation_errors(client: TestClient) -> None:
    _login(client)
    did = _new_draft(client)
    base = f"/portal/report/{did}"
    r = client.post(f"{base}/step/1", data={**STEP1, "owner_email": "bad", "relationship": "authorised_agent"})
    assert r.status_code == 422 and 'id="err-owner_email"' in r.text and 'id="err-agent_name"' in r.text
    client.post(f"{base}/step/1", data=STEP1)
    r = client.post(f"{base}/step/2", data={**STEP2, "registration_number": "123", "classes": "99"})
    assert r.status_code == 422 and 'id="err-registration_number"' in r.text and 'id="err-classes"' in r.text
    r = client.post(f"{base}/step/2", data={"right_type": "copyright", "work_description": "x",
                                            "first_publication_date": "nope"})
    assert 'id="err-work_description"' in r.text and 'id="err-first_publication_date"' in r.text
    client.post(f"{base}/step/2", data=STEP2)

    bad3 = [("url", "https://realmarket.example/item/1"), ("statute", "Copyright, Designs and Patents Act 1988"),
            ("section", ""), ("details", "short")]
    r = client.post(f"{base}/step/3", content=_urlenc(bad3), headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 422
    for f in ("url_0", "statute_0", "section_0", "details_0"):
        assert f'id="err-{f}"' in r.text
    dup = _step3([LISTING_URLS[0], LISTING_URLS[0]])
    r = client.post(f"{base}/step/3", content=_urlenc(dup), headers={"content-type": "application/x-www-form-urlencoded"})
    assert 'id="err-url_1"' in r.text
    r = client.post(f"{base}/step/3", content=_urlenc([("url", ""), ("statute", ""), ("section", ""), ("details", "")]),
                    headers={"content-type": "application/x-www-form-urlencoded"})
    assert 'id="err-items"' in r.text
    client.post(f"{base}/step/3", content=_urlenc(_step3(LISTING_URLS[:1])),
                headers={"content-type": "application/x-www-form-urlencoded"})

    r = client.post(f"{base}/step/4", files=[("evidence", ("x.txt", b"not an image", "text/plain"))])
    assert r.status_code == 422 and 'id="err-evidence"' in r.text
    r = client.post(f"{base}/step/4", data={})
    assert r.status_code == 422
    client.post(f"{base}/step/4", files=[("evidence", ("s.png", _png(), "image/png"))])

    r = client.post(f"{base}/step/5", data={"signature": ""})
    assert r.status_code == 422 and 'id="err-decl_good_faith"' in r.text and 'id="err-signature"' in r.text


def test_portal_changing_right_type_invalidates_step3(client: TestClient) -> None:
    _login(client)
    did = _complete_draft(client, LISTING_URLS[:1])
    client.post(f"/portal/report/{did}/step/2", data={"right_type": "copyright",
                "work_description": "Product photographs of the bottle", "first_publication_date": "2024-01-01"})
    r = client.get(f"/portal/report/{did}/review", follow_redirects=False)
    assert r.headers["location"] == f"/portal/report/{did}/step/3"


def test_portal_evidence_delete(client: TestClient) -> None:
    _login(client)
    did = _new_draft(client)
    base = f"/portal/report/{did}"
    client.post(f"{base}/step/1", data=STEP1)
    client.post(f"{base}/step/2", data=STEP2)
    client.post(f"{base}/step/3", content=_urlenc(_step3(LISTING_URLS[:1])),
                headers={"content-type": "application/x-www-form-urlencoded"})
    client.post(f"{base}/step/4", files=[("evidence", ("a.png", _png(), "image/png"))])
    assert "01_a.png" in client.get(f"{base}/step/4").text
    client.post(f"{base}/evidence/01_a.png/delete")
    assert "01_a.png" not in client.get(f"{base}/step/4").text
    assert client.get(f"{base}/step/5", follow_redirects=False).headers["location"] == f"{base}/step/4"


def test_portal_dashboard_lists_drafts_and_logout(client: TestClient) -> None:
    _login(client)
    did = _new_draft(client)
    assert did in client.get("/portal/").text
    r = client.post("/portal/logout", follow_redirects=False)
    assert r.headers["location"] == "/portal/login"
    client.cookies.clear()
    assert client.get("/portal/", follow_redirects=False).status_code == 303


def test_portal_users_cannot_see_each_others_drafts(client: TestClient, tmp_path: Path) -> None:
    from mocksites.store import ReportDraft, store_session

    with store_session(client.app.state.store) as s:
        s.add(ReportDraft(id="Dother", username="someone@else"))
    _login(client)
    assert client.get("/portal/report/Dother/step/1").status_code == 404


# ---------------------------------------------------------------- challenge


def test_challenge_random_and_verify(tmp_path: Path, catalog_dir: Path) -> None:
    c = TestClient(create_app(_settings(tmp_path, catalog_dir, challenge_probability=1.0)))
    r = c.get("/portal/login", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/portal/verify?next=%2Fportal%2Flogin"
    page = c.get("/portal/verify", params={"next": "/portal/login"})
    assert 'id="human-verification"' in page.text and "I am human" in page.text
    # not ticking the box is refused
    assert c.post("/portal/verify", data={"next": "/portal/login"}).status_code == 400
    r = c.post("/portal/verify", data={"confirm_human": "yes", "next": "/portal/login"}, follow_redirects=False)
    assert r.headers["location"] == "/portal/login"
    assert c.get("/portal/login", follow_redirects=False).status_code == 200  # cooldown applies
    # marketplaces are never challenged
    assert c.get("/mockbay/", follow_redirects=False).status_code == 200


def test_challenge_never_drops_a_post(tmp_path: Path, catalog_dir: Path) -> None:
    c = TestClient(create_app(_settings(tmp_path, catalog_dir, challenge_probability=1.0)))
    r = c.post("/portal/login", data={"username": USER, "password": PASS}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/portal/"


def test_challenge_rapid_requests(tmp_path: Path, catalog_dir: Path) -> None:
    c = TestClient(create_app(_settings(tmp_path, catalog_dir, challenge_rapid_requests=5)))
    codes = [c.get("/portal/login", follow_redirects=False).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5] == 303


def test_challenge_gate_window_expires() -> None:
    import random

    g = ChallengeGate(0.0, 2, 10.0, 60.0, random.Random(0))
    assert not g.should_challenge("k", True, now=0) and not g.should_challenge("k", True, now=1)
    assert g.should_challenge("k", True, now=2)
    assert not g.should_challenge("k", True, now=20)  # old hits fell out of the window
    g.mark_verified("k", now=21)
    assert not any(g.should_challenge("k", True, now=21 + i * 0.01) for i in range(10))


# ---------------------------------------------------------------- ground truth never served


def test_catalog_rejects_label_keys(tmp_path: Path) -> None:
    d = tmp_path / "c"
    (d / "images").mkdir(parents=True)
    (d / "images" / "a.jpg").write_bytes(_jpeg(RED))
    good = _listing(1, "mockbay", images=["a.jpg"]).model_dump()
    for bad_key in ("difficulty", "should_act", "split", "label"):
        (d / "catalog.json").write_text(json.dumps({"listings": [{**good, bad_key: "x"}]}))
        with pytest.raises(CatalogError):
            load_catalog(d)
    (d / "catalog.json").write_text(json.dumps({"listings": [{**good, "specs": {"Difficulty": "hard"}}]}))
    with pytest.raises(CatalogError):
        load_catalog(d)


def test_catalog_rejects_missing_image_and_duplicates(tmp_path: Path) -> None:
    d = tmp_path / "c"
    (d / "images").mkdir(parents=True)
    (d / "catalog.json").write_text(json.dumps({"listings": [_listing(1, "mockbay").model_dump(mode="json")]}))
    with pytest.raises(CatalogError):
        load_catalog(d)
    (d / "images" / "img_mockbay_1.jpg").write_bytes(_jpeg(RED))
    dup = _listing(1, "mockbay").model_dump(mode="json")
    (d / "catalog.json").write_text(json.dumps({"listings": [dup, dup]}))
    with pytest.raises(CatalogError):
        load_catalog(d)


LABEL_WORDS = ["counterfeit_identical", "counterfeit_similar", "copied_photos", "grey_import", "modified_goods",
               "genuine_resale", "lookalike", "should_act", "difficulty", "labels.json"]


def test_no_label_vocabulary_in_rendered_pages(client: TestClient) -> None:
    pages = ["/", "/mockbay/", "/mockbay/search?q=vaelo", "/mockbay/item/1003", "/tradeloft/",
             "/tradeloft/search?q=vaelo", "/tradeloft/search/more?q=vaelo&offset=8", "/tradeloft/listing/TL-4",
             "/tradeloft/visual-search", "/portal/login"]
    for p in pages:
        html = client.get(p).text.lower()
        assert "<!--" not in html, f"HTML comment on {p}"
        for w in LABEL_WORDS:
            assert w not in html, f"{w!r} leaked on {p}"


def test_catalog_hot_reload(client: TestClient, catalog_dir: Path) -> None:
    import os
    import time

    cat = load_catalog(catalog_dir)
    extra = _listing(500, "mockbay", title="Brand new reload item")
    (catalog_dir / "images" / extra.images[0]).write_bytes(_jpeg(RED))
    write_catalog(catalog_dir, [*cat.listings, extra])
    later = time.time() + 5
    os.utime(catalog_dir / "catalog.json", (later, later))
    assert client.get("/mockbay/item/1500").status_code == 200


# ---------------------------------------------------------------- marketplace "Report this listing" forms

_REPORT = {"reason": "counterfeit", "details": "Sold as genuine but priced at a fraction of the official price.",
           "reporter_name": "Sam Shopper", "reporter_email": "sam@example.com", "relationship": "shopper",
           "declaration": "on"}


@pytest.mark.parametrize("listing_path", ["/mockbay/item/1001", "/tradeloft/listing/TL-1"])
def test_listing_page_links_to_its_report_form(client: TestClient, listing_path: str) -> None:
    page = client.get(listing_path)
    assert page.status_code == 200 and f'href="{listing_path}/report"' in page.text
    form = client.get(f"{listing_path}/report")
    assert form.status_code == 200
    for field_id in ("lr-reason", "lr-details", "lr-name", "lr-email", "lr-relationship-shopper", "lr-evidence",
                     "lr-declaration", "lr-submit"):
        assert f'id="{field_id}"' in form.text
    assert "rights owner" not in form.text.lower()  # a buyer's declaration, not a rights owner's


@pytest.mark.parametrize("site,listing_path", [("mockbay", "/mockbay/item/1001"), ("tradeloft", "/tradeloft/listing/TL-1")])
def test_report_submission_shows_every_input(client: TestClient, site: str, listing_path: str) -> None:
    r = client.post(f"{listing_path}/report", data=_REPORT, files={"evidence": ("shot.png", _png(), "image/png")},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith(f"/{site}/reports/LR-")
    page = client.get(r.headers["location"])
    assert page.status_code == 200
    for text in ("Counterfeit or fake item", _REPORT["details"], "Sam Shopper", "sam@example.com",
                 "I came across it while shopping", "accurate and complete", "Received, awaiting review"):
        assert text in page.text
    ref = r.headers["location"].rsplit("/", 1)[1]
    ev = client.get(f"/{site}/reports/{ref}/evidence")
    assert ev.status_code == 200 and ev.headers["content-type"] == "image/png"
    assert client.get(f"/{'tradeloft' if site == 'mockbay' else 'mockbay'}/reports/{ref}").status_code == 404


def test_report_form_rejects_incomplete_reports(client: TestClient) -> None:
    bad = {**_REPORT, "reason": "", "details": "fake", "reporter_email": "nope", "declaration": ""}
    r = client.post("/mockbay/item/1001/report", data=bad, follow_redirects=False)
    assert r.status_code == 422
    for msg in ("Choose what", "at least 20 characters", "valid email", "Confirm the information"):
        assert msg in r.text
    r = client.post("/mockbay/item/1001/report", data=_REPORT,
                    files={"evidence": ("x.txt", b"not an image", "text/plain")}, follow_redirects=False)
    assert r.status_code == 422 and "PNG or JPEG" in r.text
    assert client.get("/mockbay/item/nope/report").status_code == 404
