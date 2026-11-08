"""A small, realistic DB state for UI tests (and for previewing the UI without running the pipeline):

    python -m tests.ui_fixture sqlite:///path/to/demo.db
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from sqlalchemy.engine import Engine

from bh.config import PROJECT_ROOT
from bh.db import Action, Case, CaseStatus, Category, Features, Listing, LLMCall, Run, RunStatus, Task, TaskStatus, \
    init_db, make_engine, session_scope, utcnow
from bh.brand import load_brand_assets, upsert_brand
from bh.db import Product
from sqlmodel import select

PHOTOS = PROJECT_ROOT / "seed" / "brand" / "photos"
OTHER = PROJECT_ROOT / "seed" / "other_images"


def _el(status: str, evidence: str) -> dict[str, str]:
    return {"status": status, "evidence": evidence}


def populate(engine: Engine) -> dict[str, int]:
    assets = load_brand_assets(PROJECT_ROOT / "seed" / "brand")
    bid = upsert_brand(assets, engine)
    now = utcnow()
    other = sorted((OTHER / "lookalike").glob("*.jpg"))[:1] or sorted(PHOTOS.glob("*.jpg"))[4:5]
    with session_scope(engine) as s:
        product = s.exec(select(Product).where(Product.brand_id == bid)).first()
        assert product is not None
        disc = Run(brand_id=bid, status=RunStatus.finished, started_at=now - timedelta(minutes=6),
                   finished_at=now - timedelta(minutes=3), config_snapshot={},
                   totals={"wall_time_s": 182.4, "tasks": {"total": 4}, "llm": {"calls": 1, "input_tokens": 1830,
                                                                                 "output_tokens": 240},
                           "max_parallelism": 3, "sites": ["mockbay", "tradeloft"]})
        s.add(disc)
        s.flush()
        root = Task(run_id=disc.id, type="discovery_stage", dedup_key="fan_out", status=TaskStatus.succeeded,
                    started_at=now - timedelta(minutes=6), finished_at=now - timedelta(minutes=4))
        s.add(root)
        s.flush()
        for i, (site, st_) in enumerate([("mockbay", TaskStatus.succeeded), ("tradeloft", TaskStatus.succeeded),
                                         ("mockbay", TaskStatus.failed)]):
            s.add(Task(run_id=disc.id, type="discovery", site=site, dedup_key=f"discovery:{site}:q{i}",
                       status=st_, parent_id=root.id, attempts=1, started_at=now - timedelta(minutes=6, seconds=-i * 9),
                       finished_at=now - timedelta(minutes=5, seconds=-i * 14),
                       error="TimeoutError: selector .card" if st_ == TaskStatus.failed else None))
        challenge = Task(run_id=disc.id, type="draft", site="portal", dedup_key="draft:case:3",
                         status=TaskStatus.waiting_human, started_at=now - timedelta(minutes=3),
                         finished_at=now - timedelta(minutes=2), payload={"case_id": 3, "stage": "draft"},
                         waiting_url="http://127.0.0.1:8000/portal/verify",
                         error="HumanRequired: challenge page at /portal/verify")
        s.add(challenge)
        s.flush()
        s.add(LLMCall(task_id=root.id, run_id=disc.id, model="claude-haiku-5-5", purpose="generic_extract",
                      input_tokens=1830, output_tokens=240, latency_ms=1420))

        def listing(n: int, title: str, price: float, seller: str, img: Path, platform: str = "mockbay",
                    **kw: object) -> Listing:
            l = Listing(platform=platform, url=f"http://127.0.0.1:8000/{platform}/item/{1000 + n}",
                        external_id=str(1000 + n), title=title, price=price, currency="GBP", seller_name=seller,
                        local_image_paths=[str(img)], screenshot_path=str(img), run_id=disc.id,
                        discovered_by_query="vaelo bottle", description=f"{title}. Fast UK dispatch.", **kw)
            s.add(l)
            s.flush()
            return l

        rows = [
            (listing(1, "VAELO Arc Copper Bottle 900ml — 1:1 replica, bulk lots", 7.5, "cheapbottles_88", other[0],
                     seller_location="Shenzhen, China", quantity=120, seller_age_days=12),
             Category.counterfeit_identical, "tm_identical", 0.93, CaseStatus.drafted, 210.4,
             {"mark_exact_in_text_or_logo_match": _el("met", "title contains 'VAELO'"),
              "goods_in_registered_class": _el("met", "bottle → class 21"),
              "not_authorised_seller": _el("met", "cheapbottles_88 not on the authorised list"),
              "not_exhausted": _el("met", "price £7.50 = 23% of RRP £32.00; '1:1', 'replica'"),
              "bulk_seller_or_quantity": _el("met", "quantity 120 ≥ 50")},
             ["refer_trading_standards"], "D1a2b"),
            (listing(2, "Copper water bottle 900ml, keeps cold 24h", 14.0, "homeware_hub",
                     PHOTOS / "arc-copper_2.jpg", platform="tradeloft", seller_location="Leeds, UK"),
             Category.copied_photos, "copied_photos", 0.86, CaseStatus.awaiting_approval, 38.2,
             {"own_photo": _el("met", "pHash 4 / dHash 3 to Arc Copper photo 2"),
              "mark_not_used_for_goods": _el("met", "no mark in title or description")}, [], None),
            (listing(3, "Vaelo-style glass tumbler (not genuine)", 9.0, "glassworks", other[0],
                     seller_location="Bristol, UK"),
             Category.counterfeit_similar, "tm_similar", 0.74, CaseStatus.waiting_human, 12.0,
             {"similar_sign": _el("met", "'Vaelo-style'"), "goods_same_or_similar": _el("met", "tumbler"),
              "confusion_indicator": _el("unknown", "no own photos; get-up unclear"),
              "not_authorised_seller": _el("met", "not authorised"),
              "not_exhausted": _el("met", "28% of RRP")}, [], None),
            (listing(4, "Pre-owned Vaelo Arc bottle, lightly used", 26.0, "jo_sells",
                     PHOTOS / "arc-copper_1.jpg", seller_location="Manchester, UK"),
             Category.genuine_resale, "exhausted_resale", 0.12, CaseStatus.classified, -1.0,
             {"own_photo_or_brand": _el("met", "brand in title"),
              "price_ratio_ge_resale_min": _el("met", "81% of RRP")}, [], None),
            (listing(5, "VAELO Arc Copper Bottle (replica)", 8.0, "cheapbottles_88", other[0]),
             Category.counterfeit_identical, "tm_identical", 0.91, CaseStatus.approved, 95.0,
             {"mark_exact_in_text_or_logo_match": _el("met", "'VAELO'")}, [], "D9f9f"),
        ]
        ids = {}
        for i, (l, cat, rule, conf, status, ev, elements, sec, draft) in enumerate(rows, 1):
            rule_obj = {"tm_identical": ("Trade Marks Act 1994", "s.10(1)",
                                         "Identical sign used for identical goods; likelihood of confusion need "
                                         "not be shown"),
                        "tm_similar": ("Trade Marks Act 1994", "s.10(2)", "Similar sign, likelihood of confusion"),
                        "copied_photos": ("Copyright, Designs and Patents Act 1988", "ss.16-17, 20",
                                          "Copying the rights owner's product photographs"),
                        "exhausted_resale": ("Trade Marks Act 1994", "s.12 (UK+ exhaustion)", "Rights exhausted")}
            st, sec_, summ = rule_obj[rule]
            s.add(Features(listing_id=l.id, matched_product_id=product.id, clip_sim=0.9 - i * 0.03, hash_dist=4 + i,
                           price_ratio=(l.price or 0) / product.rrp, own_photo_routes=["hash"] if i == 2 else []))
            c = Case(listing_id=l.id, run_id=disc.id, category=cat, confidence=conf, rule_id=rule,
                     legal_basis=[{"statute": st, "section": sec_, "summary": summ}], legal_elements=elements,
                     action=Action.platform_report if status != CaseStatus.classified else Action.monitor,
                     secondary_actions=sec, expected_value=ev, status=status, portal_draft_id=draft,
                     reasoning=f"Rule {rule} matched: every required element is met.",
                     notice_text=(f"Notice of infringing listing\n\nRights owner: Vaelo Ltd (fictional).\nListing: "
                                  f"{l.url}\nProvision: {st} {sec_}.\nFacts: The listing '{l.title}' is offered "
                                  f"by seller '{l.seller_name}' at £{l.price:.2f}.")
                     if status != CaseStatus.classified else None,
                     in_review_budget=status in (CaseStatus.drafted, CaseStatus.awaiting_approval),
                     review_diff={"registration_number": {"expected": "UK00004123456", "actual": "UK00004123457"}}
                     if status == CaseStatus.waiting_human else {},
                     decided_by="A. Lawyer" if status in (CaseStatus.approved, CaseStatus.waiting_human) else None,
                     decided_at=now if status in (CaseStatus.approved, CaseStatus.waiting_human) else None)
            s.add(c)
            s.flush()
            ids[rule if rule not in ids else f"{rule}_{i}"] = c.id  # type: ignore[assignment]
        ids["challenge_task"] = challenge.id  # type: ignore[assignment]
        ids["run"] = disc.id  # type: ignore[assignment]
    return ids


if __name__ == "__main__":
    eng = make_engine(sys.argv[1])
    init_db(eng)
    print(populate(eng))
