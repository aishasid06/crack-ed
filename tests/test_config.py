from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bh.config import PROJECT_ROOT, AppConfig, load_config, load_secrets
from tests.conftest import TEST_PASS


def test_loads_repo_config(cfg: AppConfig) -> None:
    assert cfg.brand == "vaelo"
    assert cfg.llm.small.model == "claude-haiku-5-5"
    assert cfg.llm.strong.model == "claude-opus-5-5"
    assert cfg.site("mockbay").max_concurrency >= 1
    assert {"mockbay", "tradeloft", "portal"} <= set(cfg.sites)


def test_unknown_site_raises(cfg: AppConfig) -> None:
    with pytest.raises(KeyError):
        cfg.site("realmarketplace")


def test_p_success_wildcards(cfg: AppConfig) -> None:
    d = cfg.decision
    assert d.p_success_for("mockbay", "tm_identical") == 0.90
    assert d.p_success_for("tradeloft", "grey_import") == 0.40  # *|rule
    assert d.p_success_for("other", "unknown_rule") == 0.30  # *|*


def test_cost_for_action(cfg: AppConfig) -> None:
    assert cfg.decision.cost_for("platform_report") == pytest.approx(6 * 2.5)
    assert cfg.decision.cost_for("ignore") == 0


def _write_variant(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    mutate(raw)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_invalid_borderline_band_rejected(tmp_path: Path) -> None:
    def bad(raw: dict) -> None:
        raw["thresholds"]["borderline_low"] = 0.8

    with pytest.raises(ValidationError):
        load_config(_write_variant(tmp_path, bad))


def test_unknown_key_rejected(tmp_path: Path) -> None:
    def bad(raw: dict) -> None:
        raw["llm"]["small"]["api_key"] = "nope"  # secrets must not live in config.yaml

    with pytest.raises(ValidationError):
        load_config(_write_variant(tmp_path, bad))


def test_secrets_from_env_file_never_in_repr_or_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         cfg: AppConfig) -> None:
    for k in ("MOCK_PORTAL_USER", "MOCK_PORTAL_PASS", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"MOCK_PORTAL_USER=u@example\nMOCK_PORTAL_PASS={TEST_PASS}\n")
    s = load_secrets(env)
    assert s.require_portal() == ("u@example", TEST_PASS)
    assert TEST_PASS not in repr(s)
    assert TEST_PASS not in str(cfg.snapshot())


def test_require_portal_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for k in ("MOCK_PORTAL_USER", "MOCK_PORTAL_PASS"):
        monkeypatch.delenv(k, raising=False)
    s = load_secrets(tmp_path / "missing.env")
    with pytest.raises(RuntimeError):
        s.require_portal()


def test_duplicate_yaml_keys_rejected(tmp_path: Path) -> None:
    text = (PROJECT_ROOT / "config.yaml").read_text() + "\nbrand: other\n"
    p = tmp_path / "config.yaml"
    p.write_text(text)
    with pytest.raises(yaml.constructor.ConstructorError):
        load_config(p)


def test_new_sections_loaded(cfg: AppConfig) -> None:
    assert cfg.enforce.require_final_confirm is False
    assert cfg.discovery.max_queries > 0 and cfg.vision.clip_model and cfg.paths.brand_dir
