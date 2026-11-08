"""Features, element predicates, rule ordering, scorer and borderline LLM check (SPEC §7.3, Phase 6)."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pytest
from PIL import Image, ImageOps
from sqlmodel import Session, select

from bh.brand_index import ProductInfo, load_brand_sync
from bh.classify import (PREDICATES, EvalCtx, LLMVerdict, Scorer, apply_rules, borderline_check, classify_listings,
                         evaluate_elements)
from bh.config import AppConfig
from bh.db import Case, Features, Listing, session_scope
from bh.features import (FEATURE_NAMES, FeatureExtractor, ImageMatch, ListingFeatures, ListingInput, OwnPhotoMatcher,
                         assign_clusters, mark_occurrences, origin, phrase_hits, vector)
from bh.llm import LLMClient
from bh.rules import get_rules
from tests.test_discovery import TinyImageEncoder

RULES = get_rules()
PRODUCT = ProductInfo(1, "arc", "Brand Arc Bottle 500ml", "Arc Bottle 500ml", 28.0, 21, "Insulated water bottle",
                      [])


def _ctx(authorised: bool = False) -> Any:
    return SimpleNamespace(marks=["VAELO"], name="Vaelo", products=[PRODUCT],
                           registrations=[{"registration_number": "UK00004123456", "classes": [21]}],
                           product=lambda pid: PRODUCT if pid == 1 else None)


def _li(**kw: Any) -> ListingInput:
    base = dict(listing_id=1, platform="echobazaar", title="t", description="", price=27.0, currency="GBP",
                seller_name="someone", seller_location="Leeds, UK", seller_age_days=500, rating=4.8, quantity=1,
                image_paths=[])
    base.update(kw)
    return ListingInput(**base)


def _f(**kw: Any) -> ListingFeatures:
    base = dict(listing_id=1, matched_product_id=1, matched_by="own_photo", hash_dist=0, dhash_dist=0, clip_sim=0.97,
                orb_inliers=200, own_photo=True, own_photo_routes=["hash", "orb"], price_ratio=0.96, image_count=2,
                brand_in_text=True, origin_outside_uk_eea=False, origin_evidence="seller location Leeds, UK (UK/EEA)",
                goods_terms=["bottle"])
    base.update(kw)
    return ListingFeatures(**base)


def _classify(cfg: AppConfig, f: ListingFeatures, li: ListingInput | None = None) -> tuple[str, dict[str, Any], Any]:
    e = EvalCtx(f, li or _li(), _ctx(), cfg)
    els = evaluate_elements(e, RULES)
    out = apply_rules(els, RULES)
    return out.rule.id, els, out


# ---------------------------------------------------------------- rule ordering


def test_near_rrp_genuine_from_unauthorised_uk_seller_is_resale_never_counterfeit(cfg: AppConfig) -> None:
    rule, els, _ = _classify(cfg, _f(price_ratio=0.96))
    assert rule == "exhausted_resale"
    assert els["not_exhausted"].status == "unmet" and "96% of RRP" in els["not_exhausted"].evidence
    assert els["not_authorised_seller"].status == "met"  # unauthorised, yet lawful resale


def test_counterfeit_identical_with_evidence(cfg: AppConfig) -> None:
    rule, els, out = _classify(cfg, _f(price_ratio=0.25, replica_kw=["1:1"]), _li(price=7.0))
    assert rule == "tm_identical"
    ev = els["not_exhausted"].evidence
    assert "£7.00 = 25% of RRP £28.00" in ev and "'1:1'" in ev
    assert els["goods_in_registered_class"].status == "met" and "UK00004123456" in els["goods_in_registered_class"].evidence


def test_own_photo_is_not_decisive_for_counterfeit_rules(cfg: AppConfig) -> None:
    """Same counterfeit signals without own photos still classify as tm_identical."""
    rule, _, _ = _classify(cfg, _f(own_photo=False, own_photo_routes=[], hash_dist=30, orb_inliers=3, clip_sim=0.8,
                                   matched_by="clip", price_ratio=0.2))
    assert rule == "tm_identical"
    for r in RULES.rules:
        if r.category.startswith("counterfeit"):
            assert "own_photo" not in r.requires


def test_misspelt_mark_is_tm_similar(cfg: AppConfig) -> None:
    rule, els, _ = _classify(cfg, _f(brand_in_text=False, misspelt_brand=True, misspellings=["Vaello"],
                                     price_ratio=0.3, official_kw=["official"]))
    assert rule == "tm_similar" and "Vaello" in els["similar_sign"].evidence


def test_copied_photos_without_mark(cfg: AppConfig) -> None:
    rule, _, _ = _classify(cfg, _f(brand_in_text=False, price_ratio=0.3))
    assert rule == "copied_photos"


def test_modified_goods_before_grey_import(cfg: AppConfig) -> None:
    rule, _, _ = _classify(cfg, _f(modification_kw=["customised"], origin_outside_uk_eea=True))
    assert rule == "modified_goods"


def test_grey_import(cfg: AppConfig) -> None:
    rule, els, _ = _classify(cfg, _f(origin_outside_uk_eea=True, origin_evidence='listing text: "imported from Japan"'))
    assert rule == "grey_import" and "Japan" in els["origin_outside_uk_eea"].evidence


def test_lookalike_with_referential_mark(cfg: AppConfig) -> None:
    rule, els, _ = _classify(cfg, _f(own_photo=False, own_photo_routes=[], hash_dist=20, orb_inliers=4, clip_sim=0.82,
                                     matched_by="clip", brand_in_text=False, mark_referential_only=True,
                                     inspired_kw=["inspired by"], price_ratio=0.5))
    assert rule == "lookalike" and els["mark_exact_in_text_or_logo_match"].status == "unmet"


def _silent(**kw: Any) -> ListingFeatures:
    """A listing that looks like a brand product, with no own-photo match, no mark and no copying wording."""
    base = dict(own_photo=False, own_photo_routes=[], hash_dist=20, orb_inliers=4, matched_by="clip",
                brand_in_text=False, inspired_kw=[], price_ratio=0.5)
    return _f(**{**base, **kw})


def test_resemblance_alone_is_not_a_lookalike(cfg: AppConfig) -> None:
    silent = cfg.thresholds.lookalike_silent_clip
    assert silent is not None and cfg.thresholds.lookalike_clip < silent
    below = round((cfg.thresholds.lookalike_clip + silent) / 2, 3)  # similar, but not a close copy
    rule, els, _ = _classify(cfg, _silent(clip_sim=below))
    assert rule == "fallback" and els["inspired_kw_or_close_copy"].status == "unmet"
    assert "resemblance only" in els["inspired_kw_or_close_copy"].evidence
    rule, els, _ = _classify(cfg, _silent(clip_sim=silent + 0.01))  # very close copy, no wording needed
    assert rule == "lookalike" and els["inspired_kw_or_close_copy"].status == "met"
    rule, _, _ = _classify(cfg, _silent(clip_sim=below, inspired_kw=["dupe"]))  # wording is evidence of copying
    assert rule == "lookalike"


def test_original_lookalike_rule_when_silent_threshold_unset(cfg: AppConfig) -> None:
    c = cfg.model_copy(update={"thresholds": cfg.thresholds.model_copy(update={"lookalike_silent_clip": None})})
    rule, _, _ = _classify(c, _silent(clip_sim=cfg.thresholds.lookalike_clip + 0.01))
    assert rule == "lookalike"


def test_orb_needs_clip_corroboration(cfg: AppConfig) -> None:
    m = OwnPhotoMatcher.__new__(OwnPhotoMatcher)  # routes() only reads the config
    m.cfg = cfg
    op = cfg.own_photo
    assert op.orb_clip_min > 0
    keypoints_only = ImageMatch(orb_inliers=op.orb_min_inliers + 20, clip=op.orb_clip_min - 0.05)
    assert m.routes(keypoints_only) == []  # e.g. a real studio shot of a similar bottle
    corroborated = ImageMatch(orb_inliers=op.orb_min_inliers, clip=op.orb_clip_min)
    assert m.routes(corroborated) == ["orb"]
    m.cfg = cfg.model_copy(update={"own_photo": op.model_copy(update={"orb_clip_min": 0.0})})
    assert m.routes(keypoints_only) == ["orb"]  # 0 restores ORB-alone matching


def test_authorised_seller_wins_even_when_cheap(cfg: AppConfig) -> None:
    rule, _, _ = _classify(cfg, _f(authorised=True, price_ratio=0.6))
    assert rule == "authorised"


def test_unrelated_and_brandlike_word_on_other_goods(cfg: AppConfig) -> None:
    rule, _, _ = _classify(cfg, _f(own_photo=False, own_photo_routes=[], clip_sim=0.6, hash_dist=24, orb_inliers=2,
                                   matched_product_id=None, matched_by=None, brand_in_text=False, price_ratio=None,
                                   goods_terms=[]))
    assert rule == "unrelated"
    rule2, els2, _ = _classify(cfg, _f(own_photo=False, own_photo_routes=[], clip_sim=0.6, hash_dist=24,
                                       orb_inliers=2, matched_product_id=None, matched_by=None, brand_in_text=False,
                                       misspelt_brand=True, misspellings=["Velo"], price_ratio=None, goods_terms=[]))
    assert els2["goods_same_or_similar"].status == "unmet" and RULES.rule(rule2).category == "unrelated"


def test_unknown_element_blocks_and_is_reported(cfg: AppConfig) -> None:
    _, _, out = _classify(cfg, _f(price_ratio=None, matched_product_id=None, matched_by=None))
    assert "tm_identical" in [r.id for r in out.blocked_by_unknown]


def test_trading_standards_overlay(cfg: AppConfig) -> None:
    _, els, out = _classify(cfg, _f(price_ratio=0.2, cluster_suspect=True, cluster_size=8, seller_cluster="mega"),
                            _li(quantity=500))
    assert out.rule.id == "tm_identical" and out.secondary_actions == ["refer_trading_standards"]
    assert "bulk_seller_or_quantity" in out.overlay_elements
    _, _, out2 = _classify(cfg, _f(price_ratio=0.96))
    assert out2.secondary_actions == []


def test_every_element_has_a_predicate() -> None:
    assert RULES.elements() <= set(PREDICATES)


# ---------------------------------------------------------------- text features


def test_mark_occurrences_referential() -> None:
    marks = ["VAELO"]
    assert mark_occurrences("Genuine Vaelo bottle", marks) == (1, 0)
    assert mark_occurrences("Steel bottle inspired by Vaelo", marks) == (1, 1)
    assert mark_occurrences("NOT Vaelo but just as good. Not affiliated with Vaelo.", marks) == (2, 2)
    assert mark_occurrences("Vaelo-style insulated flask", marks) == (1, 1)
    assert mark_occurrences("Vaelø bottle", marks) == (0, 0)  # accent variant is a misspelling, not the mark


def test_phrase_hits_word_boundaries() -> None:
    assert phrase_hits("AAA quality replica", ["aaa", "replica"]) == ["aaa", "replica"]
    assert phrase_hits("aaah lovely bottle", ["aaa"]) == []
    assert phrase_hits("Vaelo 1:1 copy", ["1:1"]) == ["1:1"]


def test_origin(cfg: AppConfig) -> None:
    assert origin("Leeds, UK", "", cfg)[0] is False
    assert origin("Osaka, Japan", "", cfg)[0] is True
    out, ev = origin("London, UK", "Imported from Japan - Japan retail packaging", cfg)
    assert out is True and "Imported from Japan" in ev
    assert origin("Atlantis", "", cfg)[0] is None
    assert origin("Toronto, Canada", "Limited edition colour", cfg)[0] is True  # 'Limited' is not a country


# ---------------------------------------------------------------- image features on the synthetic brand


@pytest.fixture(scope="module")
def brand_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from seed.assets import build_brand, build_other

    d = tmp_path_factory.mktemp("cls")
    build_brand(d / "brand")
    build_other(d / "other")
    return d


def _cfg_brand(cfg: AppConfig, d: Path, tmp: Path) -> AppConfig:
    paths = cfg.paths.model_copy(update={"brand_dir": str(d / "brand"), "data_dir": str(tmp / "data")})
    return cfg.model_copy(update={"paths": paths, "vision": cfg.vision.model_copy(update={"embeddings_dir": str(tmp)})})


def test_own_photo_routes(cfg: AppConfig, brand_env: Path, tmp_path: Path, engine: Any) -> None:
    c = _cfg_brand(cfg, brand_env, tmp_path)
    # ORB inlier counts are data-dependent: on the synthetic renders lookalikes reach ~41 inliers (flat shading,
    # identical drawn shapes) vs ~160 for a cropped gold photo, so the ABO-tuned 16 does not transfer here.
    # No CLIP vectors here (match(..., None)), so the CLIP corroboration of ORB is switched off to test ORB itself;
    # test_orb_needs_clip_corroboration covers the corroboration.
    c = c.model_copy(update={"own_photo": c.own_photo.model_copy(update={"orb_min_inliers": 80, "orb_clip_min": 0.0})})
    ctx = load_brand_sync(c, TinyImageEncoder(), engine)
    m = OwnPhotoMatcher(ctx, c)
    gold = Image.open(ctx.gold[0].path).convert("RGB")
    assert "hash" in m.routes(m.match(ImageOps.mirror(gold), None))
    w, h = gold.size
    cropped = gold.crop((int(w * .15), int(h * .1), int(w * .9), int(h * .95))).resize((300, 300))
    assert "orb" in m.routes(m.match(cropped, None))
    look = Image.open(sorted((brand_env / "other" / "lookalike").glob("*.jpg"))[0]).convert("RGB")
    r = m.routes(m.match(look, None))
    assert "hash" not in r and "orb" not in r


def test_feature_extractor_text_and_price(cfg: AppConfig, brand_env: Path, tmp_path: Path, engine: Any) -> None:
    c = _cfg_brand(cfg, brand_env, tmp_path)
    ctx = load_brand_sync(c, TinyImageEncoder(), engine)
    fx = FeatureExtractor(ctx, c, TinyImageEncoder(), RULES.vocabulary)
    p = ctx.products[0]
    f = fx.extract(_li(title=f"Vaelø {p.short_name} 1:1 replica", description="AAA quality, official",
                       price=round(p.rrp * 0.25, 2), image_paths=[p.image_paths[0]], seller_location="Yiwu, China"))
    assert f.own_photo and f.matched_product_id == p.id and f.price_ratio == pytest.approx(0.25, abs=0.01)
    assert f.misspelt_brand and "Vaelø" in f.misspellings and not f.brand_in_text
    assert {"1:1", "replica"} <= set(f.replica_kw) and f.official_kw == ["official"]
    assert f.origin_outside_uk_eea is True
    assert len(vector(f, c.thresholds.counterfeit_price_ratio)) == len(FEATURE_NAMES)
    f2 = fx.extract(_li(title="Pre-owned Vaelo bottle", image_paths=[]))
    assert f2.brand_in_text and f2.own_photo is None and f2.matched_by == "text" or f2.matched_product_id is None


def test_clusters_ignore_shared_brand_photos(cfg: AppConfig) -> None:
    gold_copy = "0" * 16
    a = _f(listing_id=1, seller_cluster="alice", image_hashes=[])  # gold copies are not recorded
    b = _f(listing_id=2, seller_cluster="bob", image_hashes=[])
    edits = [_f(listing_id=10 + i, seller_cluster=f"s{i}", image_hashes=["ffff0000ffff0000"], replica_kw=["1:1"])
             for i in range(5)]
    assign_clusters([a, b, *edits], cfg)
    assert a.seller_cluster != b.seller_cluster and a.cluster_size == 1
    assert len({e.seller_cluster for e in edits}) == 1 and edits[0].cluster_size == 5 and edits[0].cluster_suspect
    assert gold_copy  # documented: unmodified gold copies never reach image_hashes


# ---------------------------------------------------------------- scorer


def test_scorer_load_and_validation(cfg: AppConfig, tmp_path: Path) -> None:
    from eval.fit_scorer import make_model

    rng = np.random.default_rng(0)
    X = rng.random((60, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0.5).astype(int)
    p = tmp_path / "m.joblib"
    joblib.dump({"model": make_model().fit(X, y), "feature_names": FEATURE_NAMES, "meta": {}}, p)
    s = Scorer(p)
    probs = s.predict([_f(clip_sim=0.95), _f(clip_sim=0.05)], 0.5)
    assert 0 <= probs[1] < probs[0] <= 1
    joblib.dump({"model": None, "feature_names": ["x"], "meta": {}}, p)
    with pytest.raises(ValueError):
        Scorer(p)
    with pytest.raises(FileNotFoundError):
        Scorer(tmp_path / "missing.joblib")


# ---------------------------------------------------------------- borderline LLM check


def _fake_llm(cfg: AppConfig, secrets: Any, engine: Any, verdict: dict[str, Any]) -> tuple[LLMClient, list]:
    seen: list[dict[str, Any]] = []

    class Msgs:
        async def parse(self, **kw: Any) -> Any:
            seen.append(kw)
            return SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=3000, output_tokens=200),
                                   model="claude-opus-5-5", stop_reason="end_turn",
                                   parsed_output=LLMVerdict.model_validate(verdict))

    return LLMClient(cfg=cfg, secrets=secrets, client=SimpleNamespace(beta=SimpleNamespace(messages=Msgs())),
                     engine=engine), seen


def test_borderline_llm_can_resolve_unknowns(cfg: AppConfig, secrets: Any, engine: Any, brand_env: Path) -> None:
    img = sorted((brand_env / "brand" / "photos").glob("*.jpg"))[0]
    prod = replace(PRODUCT, image_paths=[img])
    ctx = _ctx()
    ctx.products, ctx.product = [prod], (lambda pid: prod if pid == 1 else None)
    f = _f(price_ratio=None)
    e = EvalCtx(f, _li(image_paths=[img]), ctx, cfg)
    els = evaluate_elements(e, RULES)
    out = apply_rules(els, RULES)
    verdict = {"rule_id": "tm_identical", "reasoning": "Mark printed on bottle; listed at a fraction of RRP.",
               "elements": [{"name": "not_exhausted", "status": "met", "evidence": "price far below RRP"}]}
    llm, seen = _fake_llm(cfg, secrets, engine, verdict)
    out2, els2, note = asyncio.run(borderline_check(llm, e, els, out, 0.55, RULES))
    assert out2.rule.id == "tm_identical" and els2["not_exhausted"].evidence.startswith("[LLM]")
    req = seen[0]
    assert req["model"] == "claude-opus-5-5" and req["output_format"] is LLMVerdict
    blocks = req["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in blocks) == 2
    payload = json.loads(blocks[-1]["text"])
    assert {c["rule_id"] for c in payload["candidate_rules"]} == {r.id for r in RULES.rules}


@pytest.mark.parametrize("verdict,why", [
    ({"rule_id": "sue_everyone", "reasoning": "x", "elements": []}, "unknown rule"),
    ({"rule_id": "copied_photos", "reasoning": "x",
      "elements": [{"name": "not_exhausted", "status": "met", "evidence": "x"}]}, "outside"),
])
def test_borderline_llm_cannot_invent(cfg: AppConfig, secrets: Any, engine: Any, verdict: dict[str, Any],
                                      why: str) -> None:
    e = EvalCtx(_f(price_ratio=None), _li(), _ctx(), cfg)
    els = evaluate_elements(e, RULES)
    out = apply_rules(els, RULES)
    llm, _ = _fake_llm(cfg, secrets, engine, verdict)
    out2, els2, note = asyncio.run(borderline_check(llm, e, els, out, 0.5, RULES))
    assert out2.rule.id == out.rule.id and els2 == els and why in note


# ---------------------------------------------------------------- end to end on stored listings


async def test_classify_listings_writes_features_and_cases(cfg: AppConfig, brand_env: Path, tmp_path: Path,
                                                           engine: Any, secrets: Any) -> None:
    from eval.fit_scorer import make_model

    c = _cfg_brand(cfg, brand_env, tmp_path)
    c = c.model_copy(update={"thresholds": c.thresholds.model_copy(update={"borderline_low": 0.0,
                                                                            "borderline_high": 0.0})})
    ctx = load_brand_sync(c, TinyImageEncoder(), engine)
    p = ctx.products[0]
    specs = [  # (title, price ratio, seller, location, image) — near-RRP genuine resale vs counterfeit
        ("Pre-owned Vaelo bottle", 0.95, "jess_k_82", "Leeds, UK", p.image_paths[0]),
        ("Vaelo bottle 1:1 replica", 0.2, "cheap_stuff", "Yiwu, China", p.image_paths[1]),
        ("Genuine Vaelo bottle", 0.9, "vaelo_official", "London, UK", p.image_paths[2]),
    ]
    ids = []
    with session_scope(engine) as s:
        for i, (t, r, seller, loc, img) in enumerate(specs):
            row = Listing(platform="echobazaar", url=f"http://x/echobazaar/item/{i}", title=t, description="",
                          price=round(p.rrp * r, 2), currency="GBP", seller_name=seller, seller_location=loc,
                          quantity=1, local_image_paths=[str(img)])
            s.add(row)
            s.flush()
            ids.append(row.id)
    rng = np.random.default_rng(0)
    X = rng.random((40, len(FEATURE_NAMES)))
    path = tmp_path / "s.joblib"
    joblib.dump({"model": make_model().fit(X, (X[:, 0] > .5).astype(int)), "feature_names": FEATURE_NAMES,
                 "meta": {}}, path)
    from bh.trace import run_span

    async with run_span(None, {}) as run:
        res = await classify_listings(run.id, ids, c, TinyImageEncoder(), None, engine, Scorer(path), RULES)
    assert len(res.case_ids) == 3 and res.llm_calls == 0
    with Session(engine) as s:
        cases = {cs.listing_id: cs for cs in s.exec(select(Case))}
        feats = list(s.exec(select(Features)))
    assert len(feats) == 3
    assert cases[ids[0]].rule_id == "exhausted_resale" and str(cases[ids[0]].action) == "monitor"
    assert cases[ids[1]].rule_id == "tm_identical" and cases[ids[1]].legal_basis[0]["citation"] == \
        "Trade Marks Act 1994 s.10(1)"
    assert cases[ids[2]].rule_id == "authorised"
    for cs in cases.values():
        rule = RULES.rule(cs.rule_id)
        assert set(rule.requires) <= set(cs.legal_elements)
        assert all(v["status"] in ("met", "unmet", "unknown") and v["evidence"] for v in cs.legal_elements.values())
        assert cs.reasoning.startswith(rule.id)
