"""Prices compared and quoted in their own currency, and the "model disagrees" flag (from real-brand demo run 28).

Run 28: EUR listings against an RRP the code assumed was GBP left price_ratio unknown, which sent every listing to
the strong model; the model then quoted "£86.99" for a EUR price. Mario Kart Wii stayed modified_goods on a keyword
from the publisher's blurb although the model disagreed, and nothing recorded that."""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
from sqlmodel import Session, select

from bh.brand_index import load_brand_sync
from bh.classify import EvalCtx, Scorer, _price_ev, apply_rules, borderline_check, classify_listings, evaluate_elements
from bh.config import AppConfig
from bh.db import Case, CaseStatus, Features, Listing, session_scope
from bh.decide import DISAGREE_LABEL, review_queue
from bh.features import FEATURE_NAMES
from bh.money import money
from bh.notice import _details
from bh.show_cases import _text, cases_for_run
from tests.test_classify import PRODUCT, RULES, _cfg_brand, _ctx, _f, _fake_llm, _li, brand_env  # noqa: F401
from tests.test_discovery import TinyImageEncoder


def test_money() -> None:
    assert (money(86.99, "EUR"), money(1234.5, "GBP"), money(5, "usd"), money(12, "CHF"), money(3, None)) == \
        ("€86.99", "£1,234.50", "$5.00", "CHF 12.00", "£3.00")
    assert money(None, "EUR") == "unknown"


def test_price_evidence_uses_each_currency(cfg: AppConfig) -> None:
    eur = replace(PRODUCT, currency="EUR", rrp=86.99)
    ctx = _ctx()
    ctx.products, ctx.product = [eur], (lambda pid: eur if pid == 1 else None)
    same = EvalCtx(_f(price_ratio=1.0), _li(price=86.99, currency="EUR"), ctx, cfg)
    assert _price_ev(same) == "price €86.99 = 100% of RRP €86.99 (Arc Bottle 500ml)"
    other = EvalCtx(_f(price_ratio=None), _li(price=86.99, currency="EUR"), _ctx(), cfg)  # RRP in GBP
    assert _price_ev(other) == "price €86.99; not compared with RRP £28.00 (different currency)"


def test_borderline_payload_carries_both_currencies(cfg: AppConfig, secrets: Any, engine: Any) -> None:
    eur = replace(PRODUCT, currency="EUR")
    ctx = _ctx()
    ctx.products, ctx.product = [eur], (lambda pid: eur if pid == 1 else None)
    e = EvalCtx(_f(price_ratio=None), _li(price=86.99, currency="EUR"), ctx, cfg)
    els = evaluate_elements(e, RULES)
    llm, seen = _fake_llm(cfg, secrets, engine, {"rule_id": "fallback", "reasoning": "x", "elements": []})
    asyncio.run(borderline_check(llm, e, els, apply_rules(els, RULES), 0.5, RULES))
    payload = json.loads(seen[0]["messages"][0]["content"][-1]["text"])
    assert payload["listing"]["price"] == {"amount": 86.99, "currency": "EUR"}
    assert payload["client"]["rrp"] == {"amount": 28.0, "currency": "EUR"}
    system = seen[0]["system"]
    text = system if isinstance(system, str) else " ".join(b.get("text", "") for b in system)
    assert "never convert or change a currency" in text


def test_notice_quotes_the_listing_currency() -> None:
    assets = SimpleNamespace(trademarks=[])
    case = SimpleNamespace(rule_id="copied_photos")
    eur = SimpleNamespace(title="Kart", seller_name="s", price=86.99, currency="EUR")
    gbp = SimpleNamespace(title="Kart", seller_name="s", price=86.99, currency="GBP")
    assert "at €86.99" in _details(case, eur, assets, None)  # type: ignore[arg-type]
    assert "at £86.99" in _details(case, gbp, assets, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- end to end: EUR RRP, no strong-model call


def _eur_brand(brand_env: Path, tmp: Path, currency: str) -> Path:  # noqa: F811
    d = tmp / "brand_eur"
    shutil.copytree(brand_env / "brand", d)
    prods = json.loads((d / "products.json").read_text())
    for p in prods:
        p["currency"] = currency
    (d / "products.json").write_text(json.dumps(prods))
    return d


async def _classify_eur_listing(cfg: AppConfig, brand_env: Path, tmp: Path, engine: Any, secrets: Any,  # noqa: F811
                                rrp_currency: str) -> tuple[Any, Case, Features, list]:
    from eval.fit_scorer import make_model
    from bh.trace import run_span

    c = _cfg_brand(cfg, brand_env, tmp)
    c = c.model_copy(update={"paths": c.paths.model_copy(update={"brand_dir": str(_eur_brand(brand_env, tmp, rrp_currency))}),
                             "thresholds": c.thresholds.model_copy(update={"borderline_low": 0.0, "borderline_high": 0.0})})
    ctx = load_brand_sync(c, TinyImageEncoder(), engine)
    p = ctx.products[0]
    with session_scope(engine) as s:
        row = Listing(platform="mockbay", url="http://x/mockbay/item/eur", title="Pre-owned Vaelo bottle",
                      description="", price=round(p.rrp * 0.95, 2), currency="EUR", seller_name="jess_k_82",
                      seller_location="Leeds, UK", quantity=1, local_image_paths=[str(p.image_paths[0])])
        s.add(row)
        s.flush()
        lid = row.id
    X = np.random.default_rng(0).random((40, len(FEATURE_NAMES)))
    path = tmp / "s.joblib"
    joblib.dump({"model": make_model().fit(X, (X[:, 0] > .5).astype(int)), "feature_names": FEATURE_NAMES,
                 "meta": {}}, path)
    llm, seen = _fake_llm(c, secrets, engine, {"rule_id": "fallback", "reasoning": "x", "elements": []})
    async with run_span(None, {}) as run:
        res = await classify_listings(run.id, [lid], c, TinyImageEncoder(), llm, engine, Scorer(path), RULES)
    with Session(engine) as s:
        case = s.exec(select(Case).where(Case.listing_id == lid)).one()
        feats = s.exec(select(Features).where(Features.listing_id == lid)).one()
    return res, case, feats, seen


async def test_eur_listing_vs_eur_rrp_gets_a_ratio_and_no_strong_call(cfg: AppConfig, brand_env: Path,  # noqa: F811
                                                                      tmp_path: Path, engine: Any, secrets: Any) -> None:
    res, case, feats, seen = await _classify_eur_listing(cfg, brand_env, tmp_path, engine, secrets, "EUR")
    assert feats.price_ratio == 0.95 and case.rule_id == "exhausted_resale"
    assert res.llm_calls == 0 and seen == [] and not case.llm_used  # nothing unknown, P outside the band
    assert "€" in case.legal_elements["price_ratio_ge_resale_min"]["evidence"]
    assert "£" not in json.dumps(case.legal_elements, ensure_ascii=False)


async def test_eur_listing_vs_gbp_rrp_stays_unknown(cfg: AppConfig, brand_env: Path, tmp_path: Path,  # noqa: F811
                                                    engine: Any, secrets: Any) -> None:
    res, case, feats, seen = await _classify_eur_listing(cfg, brand_env, tmp_path, engine, secrets, "GBP")
    assert feats.price_ratio is None and res.llm_calls == 1 and case.llm_used  # never converted; the unknown asks
    payload = json.loads(seen[0]["messages"][0]["content"][-1]["text"])
    ev = payload["computed_evidence"]["price_ratio_ge_resale_min"]["evidence"]
    assert "not compared with RRP £" in ev and "(different currency)" in ev and ev.startswith("price €")


# ---------------------------------------------------------------- model disagrees


def _check(cfg: AppConfig, secrets: Any, engine: Any, f: Any, verdict: dict[str, Any]) -> tuple[Any, Any, str | None]:
    e = EvalCtx(f, _li(), _ctx(), cfg)
    els = evaluate_elements(e, RULES)
    out = apply_rules(els, RULES)
    llm, _ = _fake_llm(cfg, secrets, engine, verdict)
    out2, _, _, disagreement = asyncio.run(borderline_check(llm, e, els, out, 0.5, RULES))
    return out, out2, disagreement


def test_model_choosing_another_rule_is_flagged_not_applied(cfg: AppConfig, secrets: Any, engine: Any) -> None:
    out, out2, why = _check(cfg, secrets, engine, _f(price_ratio=None), {
        "rule_id": "exhausted_resale", "reasoning": "Official blurb; 'personalized' describes in-game features.",
        "elements": [{"name": "price_ratio_ge_resale_min", "status": "unknown", "evidence": "no RRP"}]})
    assert out.rule.id != "exhausted_resale" and out2.rule.id == out.rule.id  # rules' category kept
    assert why and why.startswith("model chose exhausted_resale") and "in-game features" in why


def test_model_marking_a_kept_element_unmet_is_flagged(cfg: AppConfig, secrets: Any, engine: Any) -> None:
    out, out2, why = _check(cfg, secrets, engine, _f(price_ratio=0.95), {
        "rule_id": "exhausted_resale", "reasoning": "The title says replica.",
        "elements": [{"name": "no_replica_kw", "status": "unmet", "evidence": "'replica' in title"}]})
    assert out.rule.id == out2.rule.id == "exhausted_resale"
    assert why and "marks no_replica_kw unmet" in why


def test_model_rule_fully_met_is_adopted_without_flag(cfg: AppConfig, secrets: Any, engine: Any) -> None:
    out, out2, why = _check(cfg, secrets, engine, _f(price_ratio=None), {
        "rule_id": "tm_identical", "reasoning": "Mark on the goods at a fraction of RRP.",
        "elements": [{"name": "not_exhausted", "status": "met", "evidence": "price far below RRP"}]})
    assert out2.rule.id == "tm_identical" and why is None


def test_flag_is_stored_and_shown(engine: Any) -> None:
    from bh.classify import save_case
    from bh.db import Run

    with session_scope(engine) as s:
        run = Run()
        s.add(run)
        s.flush()
        l = Listing(platform="mockbay", url="http://x/mockbay/item/wii", title="Mario Kart Wii", price=84.99,
                    currency="EUR")
        s.add(l)
        s.flush()
        rid, lid = run.id, l.id
    rule = RULES.rule("modified_goods")
    from bh.classify import Element, RuleOutcome

    els = {n: Element("met", "x") for n in rule.requires}
    why = "model chose exhausted_resale (rules kept modified_goods): 'personalized' describes in-game features."
    cid = save_case(engine, rid, lid, RuleOutcome(rule, [], [], {}), els, 0.01, True, "r", RULES, why)  # type: ignore[arg-type]
    with session_scope(engine) as s:
        c = s.get(Case, cid)
        assert c.model_disagrees and c.model_disagreement == why  # type: ignore[union-attr]
        c.status, c.review_rank = CaseStatus.awaiting_approval, 1  # type: ignore[union-attr]
    text = _text(cases_for_run(rid, engine))  # type: ignore[arg-type]
    assert f"{DISAGREE_LABEL}: {why}" in text and "(mockbay, €84.99)" in text
    q = review_queue(engine, rid)
    assert q[0]["model_disagrees"] and q[0]["label"] == DISAGREE_LABEL and q[0]["model_disagreement"] == why
    from bh.enforce.approvals import _queue_text

    assert DISAGREE_LABEL in _queue_text(rid, engine)
