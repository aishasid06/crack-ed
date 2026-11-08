"""Real-brand demo mode: the narrow exception to SPEC §1 "fictional brand only".

A brand folder may set "fictional": false only together with "demo_real_brand": true, and it runs only with
--allow-real-brand-demo: up to classify, on read_only sites, with explicit --query text. No case from such a run
is ever decided, drafted, approved or submitted. Tests use the mock server and a temp database; no real site
is contacted."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, select

from bh.brand import (FICTIONAL_ERROR, REAL_DEMO_FLAG_ERROR, InvalidBrandDir, config_for_brand_dir,
                      config_for_run, validate_brand_dir)
from bh.brand_index import load_brand_sync
from bh.config import DEMO_BANNER, AppConfig, Secrets
from bh.db import Brand, Case, CaseStatus, Listing, Product, Run
from bh.decide import decide_cases
from bh.discovery.purge import purge_brand
from bh.enforce.approvals import approve_case
from bh.enforce.guard import check_case_claimable
from bh.enforce.portal import load_case
from bh.errors import EnforcementBlocked
from bh.orchestrator.run import PreflightError, preflight
from bh.orchestrator.run import main as run_main
from bh.show_cases import _text, cases_for_run
from tests.test_agent_nav import _cfg, bz_live  # noqa: F401 (fixture)
from tests.test_bazaarly import bz_catalog  # noqa: F401 (fixture)
from tests.test_browser import LiveServer, secrets  # noqa: F401 (fixture)
from tests.test_discovery import TinyImageEncoder
from tests.test_readonly import _read_only


def _write(d: Path, name: str, data: Any) -> None:
    (d / name).write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    """A text-only real-brand demo folder at <tmp>/data/brands/kart-demo (no photos, no trademark.json)."""
    d = tmp_path / "data" / "brands" / "kart-demo"
    d.mkdir(parents=True)
    _write(d, "brand.json", {"name": "Kart", "slug": "kart-demo", "fictional": False, "demo_real_brand": True,
                             "owner": {"name": "Not represented (demo)", "email": "n/a", "address": "n/a"}})
    _write(d, "products.json", [{"slug": "kart-8", "name": "Kart 8", "rrp": 86.99, "goods_class": 9, "images": []}])
    _write(d, "authorised_sellers.json", [])
    return d


# ---------------------------------------------------------------- brand folder rules


def test_demo_brand_validates_with_a_warning(demo_dir: Path) -> None:
    c = validate_brand_dir(demo_dir)
    assert c.ok and c.real_brand_demo and not c.errors
    assert any("real-brand demo" in w for w in c.warnings)
    assert any("no photos — text-only demo product" in w for w in c.warnings)


@pytest.mark.parametrize("meta,error", [
    ({"fictional": False}, FICTIONAL_ERROR),                          # real brand without the demo flag
    ({"fictional": False, "demo_real_brand": False}, FICTIONAL_ERROR),
    ({"fictional": True, "demo_real_brand": True}, "only allowed together with fictional: false"),
])
def test_real_brand_needs_both_flags(demo_dir: Path, meta: dict[str, Any], error: str) -> None:
    raw = json.loads((demo_dir / "brand.json").read_text())
    raw.pop("demo_real_brand")
    _write(demo_dir, "brand.json", {**raw, **meta})
    c = validate_brand_dir(demo_dir)
    assert not c.ok and any(error in e for e in c.errors)


def test_photos_still_required_for_fictional_brands(demo_dir: Path) -> None:
    raw = json.loads((demo_dir / "brand.json").read_text())
    _write(demo_dir, "brand.json", {**raw, "fictional": True, "demo_real_brand": False})
    assert "products.json[0].images: List should have at least 1 item" in " ".join(validate_brand_dir(demo_dir).errors)


def test_demo_brand_needs_the_flag(cfg: AppConfig, demo_dir: Path) -> None:
    with pytest.raises(InvalidBrandDir, match="--allow-real-brand-demo"):
        config_for_brand_dir(cfg, demo_dir)
    on = config_for_brand_dir(cfg, demo_dir, allow_real_brand_demo=True)
    assert on.real_brand_demo and on.brand == "kart-demo"
    assert on.snapshot()["demo_banner"] == DEMO_BANNER and "demo_banner" not in cfg.snapshot()
    assert REAL_DEMO_FLAG_ERROR


def test_text_only_brand_loads(cfg: AppConfig, demo_dir: Path, engine: Any, tmp_path: Path) -> None:
    on = config_for_brand_dir(cfg, demo_dir, allow_real_brand_demo=True)
    on = on.model_copy(update={"vision": on.vision.model_copy(update={"embeddings_dir": str(tmp_path / "emb")})})
    ctx = load_brand_sync(on, TinyImageEncoder(), engine)
    assert ctx.gold == [] and ctx.gold_vecs.size == 0 and [p.name for p in ctx.products] == ["Kart 8"]
    assert ctx.nearest_clip(ctx.gold_vecs.reshape(-1)[:0]) == (0.0, None)
    with Session(engine) as s:
        assert s.exec(select(Brand).where(Brand.slug == "kart-demo")).one().is_fictional is False


# ---------------------------------------------------------------- run rules (CLI + preflight)


@pytest.mark.parametrize("extra,msg", [
    ([], "--allow-real-brand-demo"),                                          # no flag
    (["--allow-real-brand-demo", "--until", "draft"], "classify only"),       # flag, but beyond classify
    (["--allow-real-brand-demo", "--until", "decide"], "classify only"),
])
def test_cli_refuses(demo_dir: Path, capsys: pytest.CaptureFixture[str], extra: list[str], msg: str) -> None:
    assert run_main(["--brand-dir", str(demo_dir), "--sites", "realsite", "--query", "Kart", *extra]) == 2
    assert msg in capsys.readouterr().err


def test_flag_is_refused_for_fictional_brands(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_main(["--allow-real-brand-demo", "--until", "discovery", "--sites", "mockbay"]) == 2
    assert "only for a brand folder marked demo_real_brand" in capsys.readouterr().err


async def test_preflight_limits_demo_runs(bz_live: LiveServer, secrets: Secrets, demo_dir: Path) -> None:  # noqa: F811
    base = _read_only(_cfg(bz_live), "mockbay", min_delay_ms=2000)
    cfg = config_for_brand_dir(base, demo_dir, allow_real_brand_demo=True)
    with pytest.raises(PreflightError, match="classify only"):
        await preflight(cfg, "decide", secrets, ["mockbay"], None, queries=["Kart"])
    with pytest.raises(PreflightError, match="only use read_only sites"):
        await preflight(cfg, "discovery", secrets, ["mockbay", "tradeloft"], None, queries=["Kart"])
    with pytest.raises(PreflightError, match="explicit --query"):
        await preflight(cfg, "discovery", secrets, ["mockbay"], None)
    assert await preflight(cfg, "discovery", secrets, ["mockbay"], None, queries=["Kart"]) is None


# ---------------------------------------------------------------- no claims (decide, draft, submit, approve)


def _demo_case(engine: Any, *, snapshot_flag: bool, fictional: bool) -> int:
    with Session(engine) as s:
        b = Brand(name="Kart", slug=f"kart-{snapshot_flag}-{fictional}", is_fictional=fictional)
        s.add(b)
        s.flush()
        run = Run(brand_id=b.id, config_snapshot={"real_brand_demo": True} if snapshot_flag else {})
        s.add(run)
        s.flush()
        l = Listing(platform="mockbay", url=f"http://127.0.0.1:8000/mockbay/listing/{run.id}", title="Kart 8")
        s.add(l)
        s.flush()
        c = Case(listing_id=l.id, run_id=run.id, category="counterfeit_identical", confidence=0.9, rule_id="r",
                 action="platform_report", status=CaseStatus.awaiting_approval)
        s.add(c)
        s.commit()
        return c.id  # type: ignore[return-value]


@pytest.mark.parametrize("snapshot_flag,fictional", [(True, True), (False, False), (True, False)],
                         ids=["demo run", "non-fictional brand", "both"])
def test_no_claim_from_a_demo_run(cfg: AppConfig, engine: Any, snapshot_flag: bool, fictional: bool) -> None:
    cid = _demo_case(engine, snapshot_flag=snapshot_flag, fictional=fictional)
    with pytest.raises(EnforcementBlocked, match="no claim is made"):
        load_case(engine, cid, cfg)              # drafting and submission load cases here
    with pytest.raises(EnforcementBlocked, match="no claim is made"):
        approve_case(cid, "lawyer1", engine)
    with pytest.raises(EnforcementBlocked, match="no claim is made"):
        decide_cases([cid], cfg, engine)
    with Session(engine) as s:
        c = s.get(Case, cid)
        assert c.status == CaseStatus.awaiting_approval and c.decided_by is None  # type: ignore[union-attr]


def test_demo_config_blocks_any_case(cfg: AppConfig, engine: Any) -> None:
    cid = _demo_case(engine, snapshot_flag=False, fictional=True)  # an ordinary case ...
    load_case(engine, cid, cfg)  # ... is fine normally
    with Session(engine) as s, pytest.raises(EnforcementBlocked):
        check_case_claimable(cfg.model_copy(update={"real_brand_demo": True}), s, s.get(Case, cid))  # not in a demo


# ---------------------------------------------------------------- show cases, purge


def test_show_cases_prints_banner_and_evidence(engine: Any) -> None:
    with Session(engine) as s:
        run = Run(config_snapshot={"real_brand_demo": True})
        s.add(run)
        s.flush()
        l = Listing(platform="realsite", url="https://www.realmarket.co.uk/products/100", title="Kart 8 Deluxe",
                    price=86.99, currency="EUR")
        s.add(l)
        s.flush()
        s.add(Case(listing_id=l.id, run_id=run.id, category="genuine_resale", confidence=0.7, rule_id="genuine_resale",
                   action="ignore", legal_elements={"seller_not_authorised": {"status": "met", "evidence": "no list"}}))
        s.commit()
        rid = run.id
    out = cases_for_run(rid, engine)  # type: ignore[arg-type]
    assert out["banner"] == DEMO_BANNER and out["cases"][0]["category"] == "genuine_resale"
    text = _text(out)
    assert text.startswith(f"*** {DEMO_BANNER} ***") and "seller_not_authorised: met — no list" in text


def test_purge_demo_brand(cfg: AppConfig, demo_dir: Path, engine: Any, tmp_path: Path) -> None:
    paths = cfg.paths.model_copy(update={"data_dir": str(tmp_path / "data")})
    on = config_for_brand_dir(cfg, demo_dir, allow_real_brand_demo=True)
    on = on.model_copy(update={"paths": on.paths.model_copy(update={"data_dir": str(tmp_path / "data")}),
                               "vision": on.vision.model_copy(update={"embeddings_dir": str(tmp_path / "emb")})})
    load_brand_sync(on, TinyImageEncoder(), engine)
    with Session(engine) as s:
        bid = s.exec(select(Brand).where(Brand.slug == "kart-demo")).one().id
        s.add(Run(brand_id=bid))
        s.commit()
    dry = purge_brand(on, demo_dir, apply=False, engine=engine)
    assert (dry["db_brand"], dry["runs_unlinked"], dry["applied"]) == (True, 1, False) and demo_dir.exists()
    purge_brand(on, demo_dir, apply=True, engine=engine)
    with Session(engine) as s:
        assert s.exec(select(Brand).where(Brand.slug == "kart-demo")).first() is None
        assert s.exec(select(Product).where(Product.brand_id == bid)).first() is None
        assert all(r.brand_id is None for r in s.exec(select(Run)))
    assert not demo_dir.exists() and not (tmp_path / "emb" / "kart-demo").exists()
    fictional = tmp_path / "data" / "brands" / "fict"
    shutil.copytree(Path(__file__).resolve().parents[1] / "seed" / "brand", fictional)
    with pytest.raises(ValueError, match="not a real-brand demo"):
        purge_brand(cfg.model_copy(update={"paths": paths}), fictional, apply=True, engine=engine)
    assert fictional.exists()


def test_resume_of_a_demo_run_needs_the_flag(cfg: AppConfig, demo_dir: Path, engine: Any) -> None:
    on = config_for_brand_dir(cfg, demo_dir, allow_real_brand_demo=True)
    with Session(engine) as s:
        run = Run(config_snapshot=on.snapshot())
        s.add(run)
        s.commit()
        rid = run.id
    with pytest.raises(InvalidBrandDir, match="--allow-real-brand-demo"):
        config_for_run(cfg, rid, engine)
    assert config_for_run(cfg, rid, engine, allow_real_brand_demo=True).real_brand_demo
