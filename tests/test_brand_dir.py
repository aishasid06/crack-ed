"""Brand folders other than seed/brand (UI-created brands): validation, --brand-dir, per-run brand data."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from bh.brand import (FICTIONAL_ERROR, BrandAssets, InvalidBrandDir, config_for_brand_dir, config_for_run,
                      load_brand_assets, validate_brand_dir)
from bh.config import AppConfig
from bh.db import Action, Case, Category, Listing, Run, session_scope
from bh.enforce.portal import BrandsByRun
from bh.notice import NotAReportableCase, build_plan


@pytest.fixture(scope="module")
def synthetic_brand(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from seed.assets import build_brand

    d = tmp_path_factory.mktemp("src") / "brand"
    build_brand(d)
    return d


@pytest.fixture
def acme(synthetic_brand: Path, tmp_path: Path) -> Path:
    """A valid brand folder at <tmp>/data/brands/acme (the layout the UI writes)."""
    d = tmp_path / "data" / "brands" / "acme"
    shutil.copytree(synthetic_brand, d)
    meta = json.loads((d / "brand.json").read_text())
    meta.update(name="Acme Drinkware", slug="acme", owner={**meta["owner"], "name": "Acme Ltd (fictional)"})
    (d / "brand.json").write_text(json.dumps(meta))
    return d


def _edit(d: Path, name: str, fn: Any) -> None:
    data = json.loads((d / name).read_text())
    fn(data)
    (d / name).write_text(json.dumps(data))


def test_valid_folder(acme: Path) -> None:
    c = validate_brand_dir(acme)
    assert c.ok and c.slug == "acme" and c.name == "Acme Drinkware" and c.products == 4
    assert c.errors == [] and c.warnings == []
    assets = load_brand_assets(acme)
    assert assets.meta.slug == "acme" and assets.root == acme.resolve()


def test_every_problem_is_listed(acme: Path, tmp_path: Path) -> None:
    _edit(acme, "brand.json", lambda m: m.update(fictional=False))
    _edit(acme, "products.json", lambda ps: (ps[0].update(rrp=0), ps[1].update(goods_class=99)))
    _edit(acme, "trademark.json", lambda ts: ts[0].update(registration_number="UK123"))
    (acme / "authorised_sellers.json").write_text("[{not json")
    products = json.loads((acme / "products.json").read_text())
    (acme / products[2]["images"][0]).unlink()                                     # missing photo
    Image.new("RGB", (120, 400), "white").save(acme / products[2]["images"][1])    # too small
    Image.new("RGB", (400, 400), "white").save(acme / "photos" / "anim.gif")
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (400, 400), "white").save(outside)
    os.symlink(outside, acme / "photos" / "escape.jpg")
    _edit(acme, "products.json", lambda ps: ps[3].update(images=["photos/anim.gif", "../secret.jpg",
                                                                 "/etc/hosts", "photos/escape.jpg"]))
    c = validate_brand_dir(acme)
    text = "\n".join(c.errors)
    assert not c.ok
    expected = [
        f"brand.json.fictional: {FICTIONAL_ERROR}",
        "products.json[0].rrp:", "products.json[1].goods_class:",
        "trademark.json[0].registration_number:",
        "authorised_sellers.json: invalid JSON",
        f"file not found: {products[2]['images'][0]}",
        "is 120x400; shorter side must be at least 300 px",
        "photos/anim.gif is GIF; must be JPEG or PNG",
        "without '..': ../secret.jpg", "without '..': /etc/hosts",
        "path escapes the brand folder (symlink): photos/escape.jpg",
    ]
    for e in expected:
        assert e in text, (e, text)
    with pytest.raises(InvalidBrandDir) as ei:
        load_brand_assets(acme)
    assert ei.value.errors == c.errors


def test_fictional_must_be_explicit(acme: Path) -> None:
    _edit(acme, "brand.json", lambda m: m.pop("fictional"))
    assert validate_brand_dir(acme).errors == [f"brand.json.fictional: {FICTIONAL_ERROR}"]


def test_oversized_image(acme: Path) -> None:
    products = json.loads((acme / "products.json").read_text())
    big = acme / products[0]["images"][0]
    with big.open("ab") as fh:
        fh.write(b"\0" * (10 * 1024 * 1024 + 1))
    assert any("(max 10 MB)" in e for e in validate_brand_dir(acme).errors)


def test_trade_mark_is_optional(acme: Path) -> None:
    (acme / "trademark.json").unlink()
    c = validate_brand_dir(acme)
    assert c.ok and any("copyright-only" in w for w in c.warnings)
    assets = load_brand_assets(acme)
    assert assets.trademarks == []
    case = Case(id=1, listing_id=1, category=Category.counterfeit_identical, confidence=0.9, rule_id="tm_identical",
                action=Action.platform_report, legal_basis=[{"statute": "Trade Marks Act 1994", "section": "s.10(1)"}])
    listing = Listing(platform="mockbay", url="http://127.0.0.1:8000/mockbay/item/1", title="x", screenshot_path="s")
    with pytest.raises(NotAReportableCase, match="no trade mark registration"):
        build_plan(case, listing, assets)


def test_warnings_do_not_block(acme: Path) -> None:
    _edit(acme, "brand.json", lambda m: (m.pop("owner"), m.pop("copyright")))
    (acme / "authorised_sellers.json").write_text("[]")
    c = validate_brand_dir(acme)
    assert c.ok and len(c.warnings) == 3


@pytest.mark.parametrize("bad", ["data/brands/../brand", "/definitely/not/here"])
def test_unsafe_or_missing_paths(bad: str) -> None:
    c = validate_brand_dir(bad)
    assert not c.ok and len(c.errors) == 1


def test_symlinked_folder_and_file_path(acme: Path, tmp_path: Path) -> None:
    link = tmp_path / "linked-acme"
    os.symlink(acme, link)
    assert validate_brand_dir(link).ok  # resolves to a real directory
    assert not validate_brand_dir(acme / "brand.json").ok  # a file, not a folder


def test_config_for_brand_dir(cfg: AppConfig, acme: Path) -> None:
    c = config_for_brand_dir(cfg, acme)
    assert c.brand == "acme" and Path(c.paths.brand_dir) == acme.resolve() and cfg.brand == "vaelo"


# ---------------------------------------------------------------- orchestrator CLI


def test_cli_rejects_invalid_folder_before_anything(acme: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    import bh.orchestrator.run as run_mod

    called: list[Any] = []
    monkeypatch.setattr(run_mod, "run_pipeline", lambda *a, **k: called.append(1))
    _edit(acme, "products.json", lambda ps: ps[0].update(rrp=-1))
    assert run_mod.main(["--brand-dir", str(acme)]) == 2
    err = capsys.readouterr().err
    assert "cannot start: brand folder" in err and "products.json[0].rrp" in err and called == []


def test_cli_runs_with_folder_config(acme: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bh.orchestrator.run as run_mod

    seen: dict[str, Any] = {}

    async def fake(cfg: AppConfig, **kw: Any) -> dict[str, Any]:
        seen.update(cfg=cfg, **kw)
        return {"run_id": 1}

    monkeypatch.setattr(run_mod, "run_pipeline", fake)
    monkeypatch.setattr(run_mod, "summarise", lambda state: state)
    assert run_mod.main(["--brand-dir", str(acme), "--until", "discovery"]) == 0
    assert seen["cfg"].brand == "acme" and Path(seen["cfg"].paths.brand_dir) == acme.resolve()


async def test_brand_flag_must_match_folder(acme: Path, cfg: AppConfig) -> None:
    from bh.orchestrator.run import PreflightError, preflight

    with pytest.raises(PreflightError, match="'vaelo'"):
        await preflight(config_for_brand_dir(cfg, acme), "discovery", None, None, "vaelo")  # type: ignore[arg-type]


# ---------------------------------------------------------------- later stages use the run's brand


def test_per_run_brand_for_drafting_and_submission(acme: Path, cfg: AppConfig, engine: Any) -> None:
    acme_cfg = config_for_brand_dir(cfg, acme)
    with session_scope(engine) as s:
        r_acme, r_default, r_none = Run(config_snapshot=acme_cfg.snapshot()), Run(config_snapshot=cfg.snapshot()), Run()
        s.add_all([r_acme, r_default, r_none])
        s.flush()
        ids = []
        for r in (r_acme, r_default, r_none):
            l = Listing(platform="mockbay", url=f"http://127.0.0.1:8000/mockbay/item/{r.id}", title="t")
            s.add(l)
            s.flush()
            c = Case(listing_id=l.id, run_id=r.id, category=Category.counterfeit_identical, confidence=0.9,
                     rule_id="tm_identical", action=Action.platform_report)
            s.add(c)
            s.flush()
            ids.append(c.id)
        run_ids = (r_acme.id, r_default.id, r_none.id)
    assert config_for_run(cfg, run_ids[0], engine).brand == "acme"
    assert config_for_run(cfg, run_ids[1], engine) is cfg and config_for_run(cfg, run_ids[2], engine) is cfg
    brands = BrandsByRun(cfg, engine)  # the submit loop runs with the default config
    a, d = brands.for_case(ids[0]), brands.for_case(ids[1])
    assert isinstance(a, BrandAssets) and a.meta.owner.name == "Acme Ltd (fictional)" and a.meta.slug == "acme"
    assert d.meta.slug == "vaelo"


def test_registration_number_cannot_be_shared(acme: Path, synthetic_brand: Path, engine: Any) -> None:
    from bh.brand import upsert_brand

    upsert_brand(load_brand_assets(synthetic_brand), engine)  # vaelo owns UK00004123456
    with pytest.raises(ValueError, match="already belongs to another brand"):
        upsert_brand(load_brand_assets(acme), engine)  # acme copied the same registrations
    _edit(acme, "trademark.json", lambda ts: [t.update(registration_number=f"UK0000999000{i}") for i, t in enumerate(ts)])
    assert upsert_brand(load_brand_assets(acme), engine)
