"""Seed generator (SPEC §6, Phase 3)."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imagehash
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps
from sqlmodel import Session, select

from bh.brand import load_brand_assets, upsert_brand
from bh.config import load_config
from bh.db import AuthorisedSeller, Brand, Product, TradeMark
from mocksites.catalog import FORBIDDEN_KEYS, load_catalog
from seed import generate as G
from seed.variants import VARIANTS, apply_random

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gen(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("seed")
    res = G.generate(base / "brand", base / "other", base / "catalog", base / "labels.json", seed=42)
    labels = json.loads(res.labels_path.read_text())
    catalog = load_catalog(base / "catalog")
    return {"base": base, "res": res, "labels": labels, "catalog": catalog,
            "assets": load_brand_assets(base / "brand")}


def _by_key(gen: dict[str, Any]) -> dict[str, Any]:
    return {f"{l.site}:{l.id}": l for l in gen["catalog"].listings}


# ---------------------------------------------------------------- mix, split, difficulty


def test_counts_and_mix(gen: dict[str, Any]) -> None:
    labels = gen["labels"]
    assert len(labels) == 120 == len(gen["catalog"].listings)
    assert Counter(v["category"] for v in labels.values()) == Counter(G.CATEGORY_COUNTS)
    sites = Counter(v["site"] for v in labels.values())
    assert sites["mockbay"] >= 40 and sites["tradeloft"] >= 40


def test_labels_match_catalog_one_to_one(gen: dict[str, Any]) -> None:
    listings = _by_key(gen)
    assert set(listings) == set(gen["labels"])
    for key, v in gen["labels"].items():
        site, lid = key.split(":")
        assert v["url_path"] == G.listing_path(site, lid)
        assert set(v) >= {"category", "legal_basis", "expected_action", "should_report", "difficulty", "split"}


def test_hard_cases_about_20_percent(gen: dict[str, Any]) -> None:
    labels = gen["labels"]
    hard = [v for v in labels.values() if v["difficulty"] == "hard"]
    assert len(hard) == 24
    assert {v["category"] for v in hard} == set(G.CATEGORY_COUNTS)


def test_split_60_40_stratified(gen: dict[str, Any]) -> None:
    labels = gen["labels"]
    assert Counter(v["split"] for v in labels.values()) == {"dev": 72, "test": 48}
    strata: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for v in labels.values():
        strata[(v["category"], v["difficulty"])][v["split"]] += 1
    for (cat, diff), c in strata.items():
        n = c["dev"] + c["test"]
        assert c["dev"] == int(n * 0.6 + 0.5), (cat, diff, c)
    assert {v["category"] for v in labels.values() if v["split"] == "test"} == set(G.CATEGORY_COUNTS)


def test_expected_action_from_rules(gen: dict[str, Any]) -> None:
    from bh.rules import get_rules

    rules = get_rules()
    expected = {"authorised": "ignore", "genuine_resale": "monitor", "counterfeit_identical": "platform_report",
                "counterfeit_similar": "platform_report", "copied_photos": "platform_report",
                "grey_import": "escalate", "modified_goods": "escalate", "lookalike": "escalate",
                "unrelated": "ignore"}
    for v in gen["labels"].values():
        rule = rules.rule_for_category(v["category"])
        assert v["rule_id"] == rule.id and v["expected_action"] == str(rule.action) == expected[v["category"]]
        assert v["should_report"] == (v["expected_action"] == "platform_report")
        assert "should_act" not in v
        if rule.legal_basis:
            assert v["legal_basis"][0] == rule.legal_basis.citation()
        else:
            assert v["legal_basis"] == []
    counts = Counter(v["expected_action"] for v in gen["labels"].values())
    assert counts == {"platform_report": 45, "escalate": 35, "monitor": 15, "ignore": 25}


def test_deterministic(gen: dict[str, Any], tmp_path: Path) -> None:
    base = gen["base"]
    res2 = G.generate(base / "brand", base / "other", tmp_path / "catalog", tmp_path / "labels.json", seed=42)
    assert json.loads(res2.labels_path.read_text()) == gen["labels"]
    assert res2.catalog_path.read_text() == gen["res"].catalog_path.read_text()


# ---------------------------------------------------------------- bulk cluster


def test_bulk_counterfeit_cluster(gen: dict[str, Any]) -> None:
    threshold = load_config().thresholds.bulk_listing_threshold
    listings = _by_key(gen)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    per_seller: Counter[str] = Counter()
    for key, v in gen["labels"].items():
        if v["category"].startswith("counterfeit_"):
            per_seller[norm(listings[key].seller_name)] += 1
    seller, n = per_seller.most_common(1)[0]
    assert n >= threshold and seller == "megadealfactory"
    cluster = [k for k, v in gen["labels"].items() if v["bulk_cluster"]]
    assert len(cluster) == 8 and {gen["labels"][k]["site"] for k in cluster} == {"mockbay", "tradeloft"}
    assert all("Trade Marks Act 1994 s.92" in gen["labels"][k]["legal_basis"] for k in cluster)
    assert all(gen["labels"][k]["expected_secondary_actions"] == ["refer_trading_standards"] for k in cluster)
    # cluster listings share images
    hashes = [str(imagehash.phash(Image.open(gen["base"] / "catalog" / "images" / listings[k].images[0])))
              for k in cluster]
    assert len(set(hashes)) <= 2


# ---------------------------------------------------------------- category signals


def _rrp(gen: dict[str, Any], slug: str | None) -> float | None:
    return next((p.rrp for p in gen["assets"].products if p.slug == slug), None)


def _items(gen: dict[str, Any], cat: str, difficulty: str | None = None) -> list[tuple[Any, dict[str, Any]]]:
    listings = _by_key(gen)
    return [(listings[k], v) for k, v in gen["labels"].items()
            if v["category"] == cat and (difficulty is None or v["difficulty"] == difficulty)]


def test_price_bands(gen: dict[str, Any]) -> None:
    bands = {"authorised": (0.55, 1.0), "genuine_resale": (0.6, 1.0), "counterfeit_identical": (0.1, 0.55),
             "counterfeit_similar": (0.1, 0.58), "copied_photos": (0.15, 0.5), "grey_import": (0.8, 1.05),
             "modified_goods": (0.75, 1.3)}
    for cat, (lo, hi) in bands.items():
        for l, v in _items(gen, cat):
            ratio = l.price / _rrp(gen, v["product"])
            assert lo - 0.05 <= ratio <= hi + 0.05, (cat, l.title, ratio)


def test_text_signals(gen: dict[str, Any]) -> None:
    for l, _ in _items(gen, "counterfeit_identical", "easy"):
        assert "Vaelo" in l.title and any(k.lower() in (l.title + l.description).lower() for k in G.REPLICA_KW)
    for l, _ in _items(gen, "counterfeit_similar"):
        assert any(m in l.title for m in G.MISSPELLINGS + G.SUBTLE_MISSPELLINGS) and "Vaelo " not in l.title
    for l, _ in _items(gen, "genuine_resale", "easy"):
        assert re.search(r"used|pre-owned", l.title, re.I) and l.quantity == 1
    for l, _ in _items(gen, "copied_photos"):
        assert "vaelo" not in (l.title + l.description).lower()
    for l, _ in _items(gen, "modified_goods"):
        assert any(k in (l.title + l.description).lower() for k in G.MODIFY_KW)
    for l, _ in _items(gen, "grey_import"):
        far = not l.seller_location.endswith(("UK", "Ireland", "France", "Germany", "Netherlands", "Spain"))
        assert far or "imported from" in l.description.lower()
    for l, _ in _items(gen, "lookalike", "easy"):
        text = (l.title + " " + l.description).lower()
        assert "vaelo" not in text or any(k.lower() in text for k in G.INSPIRED_KW)
    for l, _ in _items(gen, "unrelated"):
        assert "vaelo" not in (l.title + l.description).lower() and l.site_category != "Drinkware"


def test_authorised_sellers_used(gen: dict[str, Any]) -> None:
    assets = gen["assets"]
    for l, _ in _items(gen, "authorised"):
        assert assets.is_authorised(l.site, l.seller_name)
    for cat in G.CATEGORY_COUNTS:
        if cat != "authorised":
            for l, _ in _items(gen, cat):
                assert not assets.is_authorised(l.site, l.seller_name)


def test_hard_resale_is_near_rrp_unauthorised_uk(gen: dict[str, Any]) -> None:
    for l, v in _items(gen, "genuine_resale", "hard"):
        assert l.price / _rrp(gen, v["product"]) >= 0.85 and l.condition == "New"
        assert not any(k in l.title.lower() for k in ("used", "pre-owned"))


# ---------------------------------------------------------------- images


def test_image_hash_separation(gen: dict[str, Any]) -> None:
    assets = gen["assets"]
    gold = []
    for p in assets.products:
        for rel in p.images:
            im = Image.open(assets.image_path(rel))
            gold += [imagehash.phash(im), imagehash.phash(ImageOps.mirror(im))]
    images = gen["base"] / "catalog" / "images"

    def min_dist(fn: str) -> int:
        h = imagehash.phash(Image.open(images / fn))
        return min(h - g for g in gold)

    for cat in ("authorised", "genuine_resale", "grey_import"):
        for l, _ in _items(gen, cat):
            assert all(min_dist(f) <= 4 for f in l.images), (cat, l.id)
    for cat in ("lookalike", "unrelated"):
        for l, _ in _items(gen, cat):
            assert all(min_dist(f) > 10 for f in l.images), (cat, l.id)
    # crop+resize moves phash for ~1/3 of copies; CLIP similarity covers those in Phase 6
    copied = [min_dist(f) for l, _ in _items(gen, "copied_photos", "easy") for f in l.images]
    assert sum(d <= 8 for d in copied) / len(copied) >= 0.6


@pytest.mark.parametrize("name", list(VARIANTS))
def test_each_variant_runs(name: str) -> None:
    import random

    im = Image.new("RGB", (256, 256), (40, 90, 120))
    out = VARIANTS[name](im, random.Random(0))
    assert out.size[0] >= 64


def test_apply_random_puts_jpeg_last() -> None:
    import random

    im = Image.new("RGB", (256, 256), (40, 90, 120))
    for seed in range(20):
        _, names = apply_random(im, random.Random(seed), 2, 4)
        if "jpeg_q25" in names:
            assert names[-1] == "jpeg_q25"


# ---------------------------------------------------------------- no ground truth leaks


LABEL_WORDS = [c for c in G.CATEGORY_COUNTS if "_" in c] + ["lookalike", "should_report", "expected_action",
                                                               "difficulty", "tm_identical", "tm_similar"]


def test_catalog_carries_no_labels(gen: dict[str, Any]) -> None:
    raw = gen["res"].catalog_path.read_text().lower()
    for w in LABEL_WORDS:
        assert w not in raw, w
    for k in FORBIDDEN_KEYS:
        assert f'"{k}":' not in raw
    for l in gen["catalog"].listings:
        for fn in l.images:
            assert re.fullmatch(rf"{re.escape(l.id.lower())}_\d+\.jpg", fn)


def test_catalog_order_is_not_grouped_by_category(gen: dict[str, Any]) -> None:
    labels = gen["labels"]
    cats = [labels[f"{l.site}:{l.id}"]["category"] for l in gen["catalog"].listings]
    runs = sum(1 for a, b in zip(cats, cats[1:]) if a == b)
    assert runs < 40


def test_mocksites_code_never_reads_labels() -> None:
    """No mock-site code imports the seed package or refers to labels.json (docstrings excepted)."""
    import ast

    for p in (ROOT / "mocksites").rglob("*.py"):
        tree = ast.parse(p.read_text())
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("seed"), p
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith("seed") for a in node.names), p
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert "labels.json" not in node.value, p
    for p in (ROOT / "mocksites" / "templates").rglob("*.html"):
        assert "labels" not in p.read_text(), p


# ---------------------------------------------------------------- browsable via the mock sites


def test_all_listings_browsable(gen: dict[str, Any], tmp_path: Path) -> None:
    from mocksites.app import MockSettings, create_app
    from tests.test_mocksites import ColourEmbedder

    app = create_app(MockSettings(catalog_dir=gen["base"] / "catalog", db_url=f"sqlite:///{tmp_path / 'm.db'}",
                                  portal_user="u@x.example", portal_pass="pw-test-1", uploads_dir=tmp_path / "up",
                                  embedder=ColourEmbedder()))
    c = TestClient(app)
    for key, v in gen["labels"].items():
        r = c.get(v["url_path"])
        assert r.status_code == 200, key
        l = _by_key(gen)[key]
        assert l.title.replace("&", "&amp;") in r.text or l.title in r.text
        assert c.get(f"/media/{l.images[0]}").status_code == 200
    r = c.get("/mockbay/search", params={"q": "vaelo"})
    assert int(re.search(r'srp-count__n">(\d+)<', r.text).group(1)) >= 24 and "pagination__next" in r.text
    r = c.get("/tradeloft/search", params={"q": "vaelo"})
    assert 'id="tl-load-more"' in r.text and "hidden" not in r.text.split('id="tl-load-more"')[1].split(">")[0]


# ---------------------------------------------------------------- brand loader


def test_brand_loader_and_upsert(gen: dict[str, Any], engine: Any) -> None:
    assets = gen["assets"]
    assert len(assets.products) == 4 and all(len(p.images) == 4 for p in assets.products)
    assert {t.mark_type for t in assets.trademarks} == {"word", "figurative"}
    assert assets.is_authorised("mockbay", "VAELO_OFFICIAL ") and not assets.is_authorised("tradeloft", "vaelo_official")
    bid = upsert_brand(assets, engine)
    assert upsert_brand(assets, engine) == bid  # idempotent
    with Session(engine) as s:
        assert len(list(s.exec(select(Brand)))) == 1
        assert len(list(s.exec(select(Product)))) == 4
        assert len(list(s.exec(select(TradeMark)))) == 2
        assert len(list(s.exec(select(AuthorisedSeller)))) == 4
        tm = s.exec(select(TradeMark).where(TradeMark.mark_type == "figurative")).one()
        assert tm.classes == [21] and Path(tm.logo_path).exists()
        assert all(Path(p).exists() for prod in s.exec(select(Product)) for p in prod.image_paths)


def test_brand_loader_missing_files(tmp_path: Path, gen: dict[str, Any]) -> None:
    import shutil

    from bh.brand import InvalidBrandDir

    with pytest.raises(InvalidBrandDir):
        load_brand_assets(tmp_path / "nope")
    d = tmp_path / "brand"
    shutil.copytree(gen["base"] / "brand", d)
    next((d / "photos").glob("*.jpg")).unlink()
    with pytest.raises(InvalidBrandDir, match="file not found"):
        load_brand_assets(d)


def test_build_assets_never_overwrites(gen: dict[str, Any]) -> None:
    from seed.assets import build_brand, build_other

    assert build_brand(gen["base"] / "brand") is False
    assert build_other(gen["base"] / "other") is False


# ---------------------------------------------------------------- optional LLM text


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete_json(self, *, role: str, purpose: str, schema: Any, system: str,
                            messages: list[dict[str, Any]]) -> Any:
        assert role == "small"
        self.calls.append(purpose)
        payload = json.loads(messages[0]["content"])
        items = []
        for it in payload:
            if it["index"] % 2 == 0:  # keeps every signal phrase → accepted
                items.append({"index": it["index"], "title": "Rewritten: " + it["title"],
                              "description": it["description"] + " Thanks for looking."})
            else:  # drops signals → rejected unless there were none
                items.append({"index": it["index"], "title": "Totally different", "description": "nothing"})
        return SimpleNamespace(parsed=schema.model_validate({"items": items}))


def test_llm_rewrite_keeps_signals(gen: dict[str, Any]) -> None:
    import random

    ctx = G.Ctx(gen["assets"], random.Random(1), G.load_pools(gen["base"] / "other"))
    specs = G.plan(ctx)
    before = {id(s): (s.title, s.signals) for s in specs}
    llm = FakeLLM()
    changed = asyncio.run(G.rewrite_texts(specs, llm))
    assert len(llm.calls) == len(G.CATEGORY_COUNTS) and changed > 0
    for s in specs:
        text = (s.title + s.description).lower()
        assert all(sig.lower() in text for sig in before[id(s)][1])


# ---------------------------------------------------------------- rules.yaml


def test_rules_yaml_structure() -> None:
    from bh.rules import get_rules

    rules = get_rules()
    assert [r.id for r in rules.rules] == ["authorised", "unrelated", "copied_photos", "tm_identical", "tm_similar",
                                           "modified_goods", "grey_import", "exhausted_resale", "lookalike", "fallback"]
    assert rules.rule("tm_identical").legal_basis.citation() == "Trade Marks Act 1994 s.10(1)"
    assert rules.rule("tm_identical").requires == ["mark_exact_in_text_or_logo_match", "goods_in_registered_class",
                                                   "not_authorised_seller", "not_exhausted"]
    assert {"replica", "inspired", "modification"} <= set(rules.vocabulary)


def test_rules_yaml_validation(tmp_path: Path) -> None:
    import yaml
    from pydantic import ValidationError

    from bh.rules import load_rules

    raw = yaml.safe_load((ROOT / "rules.yaml").read_text())
    bad = {**raw, "rules": raw["rules"][:-1]}  # no fallback
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_rules(tmp_path / "r.yaml")
    bad = {**raw, "rules": [{**raw["rules"][0], "action": "sue_them"}] + raw["rules"][1:]}
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_rules(tmp_path / "r.yaml")
