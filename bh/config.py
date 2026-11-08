"""Configuration loading.

Non-secret settings come from ``config.yaml`` (validated with Pydantic).
Secrets come only from the environment / ``.env`` and are held as ``SecretStr``
so they never appear in reprs, logs or config snapshots.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Effort = Literal["low", "medium", "high", "xhigh", "max"]
ActionName = Literal["ignore", "monitor", "platform_report", "escalate", "refer_trading_standards"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    db_url: str
    data_dir: str
    sessions_dir: str
    rules_file: str
    brand_dir: str = "seed/brand"

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


class ModelRoleConfig(_Strict):
    model: str
    effort: Effort = "medium"
    max_tokens: int = Field(gt=0, le=128_000)
    fallbacks: bool = False


class LLMConfig(_Strict):
    small: ModelRoleConfig
    strong: ModelRoleConfig
    timeout_s: float = 120
    max_retries: int = Field(default=2, ge=0)
    enable_query_extension: bool = False
    enable_llm_seed_text: bool = False


class BudgetsConfig(_Strict):
    max_llm_tokens_per_run: int = Field(gt=0)
    max_listings_per_run: int = Field(gt=0)
    review_budget_minutes: float = Field(ge=0)


class MockSitesConfig(_Strict):
    host: str
    port: int
    challenge_probability: float = Field(ge=0, le=1)
    challenge_rapid_requests: int = Field(gt=0)
    challenge_window_s: float = Field(gt=0)
    challenge_cooldown_s: float = Field(ge=0)
    db_url: str
    catalog_dir: str
    max_upload_mb: float = Field(gt=0)
    seed: int | None = None


# Hosts that enforcement (portal login, drafting, verification, submission) may ever touch (SPEC §1:
# submissions only go to our own mock sites). Widening this set requires a code change; config may only
# narrow it (enforce.allowed_hosts).
ENFORCEMENT_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost"})


class SiteConfig(_Strict):
    base_url: str
    mode: Literal["full", "read_only"] = "full"  # read_only: search/extract only; bh/enforce refuses it
    max_concurrency: int = Field(gt=0)
    max_pages: int = Field(gt=0)
    min_delay_ms: int = Field(ge=0)
    adapter: str | None = None  # None → generic LLM extractor
    navigator: bool = False  # adapter-less site browsed by the agentic navigator (needs an LLM)
    search_url: str | None = None  # for sites without an adapter: "{base}/search?q={query}&page={page}"


class NavigatorConfig(_Strict):
    """Agentic navigator (bh/browser/agent_nav.py). Opt-in: navigator sites are searched only when
    ``enabled`` (or ``--navigator``) and an LLM is available (``--llm``)."""

    enabled: bool = False
    max_steps: int = Field(default=30, gt=0)  # agent steps per task (search or listing)
    max_tokens_per_task: int = Field(default=60000, gt=0)
    max_elements: int = Field(default=80, gt=0)  # indexed interactive elements shown to the model
    settle_ms: int = Field(default=400, ge=0)  # wait after an action before re-observing
    action_timeout_ms: int = Field(default=3000, gt=0)
    max_result_pages: int = Field(default=3, gt=0)
    recipes_dir: str = "data/recipes"  # learned recipes, one JSON per site (replayed without an LLM)


class ChallengeConfig(_Strict):
    url_patterns: list[str] = Field(default_factory=list)
    selectors: list[str] = Field(default_factory=list)
    title_patterns: list[str] = Field(default_factory=list)


class BrowserConfig(_Strict):
    headless: bool = True
    navigation_timeout_ms: int = 15000
    retries: int = Field(default=1, ge=0)
    selector_timeout_ms: int = 3000
    snapshot_max_chars: int = Field(default=12000, gt=500)
    max_image_candidates: int = Field(default=24, gt=0)
    challenge: ChallengeConfig = Field(default_factory=ChallengeConfig)
    demo_mode: bool = False  # headed browser + highlight each element before the agent acts on it
    demo_highlight_ms: int = Field(default=700, ge=0)


class ThresholdsConfig(_Strict):
    prefilter_clip: float
    prefilter_hash: int
    prefilter_brand_fuzzy: int
    hash_match: int
    lookalike_clip: float
    # a lookalike without copying wording ("inspired by", "dupe", ...) needs CLIP >= this; None = resemblance
    # plus "no mark" suffices (the original rule)
    lookalike_silent_clip: float | None = None
    counterfeit_price_ratio: float
    resale_min_ratio: float
    act_threshold: float
    bulk_listing_threshold: int
    bulk_quantity: int
    borderline_low: float
    borderline_high: float

    @model_validator(mode="after")
    def _check_band(self) -> "ThresholdsConfig":
        if not 0 <= self.borderline_low < self.borderline_high <= 1:
            raise ValueError("borderline band must satisfy 0 <= low < high <= 1")
        if self.counterfeit_price_ratio > self.resale_min_ratio:
            raise ValueError("counterfeit_price_ratio must not exceed resale_min_ratio")
        return self


class DecisionConfig(_Strict):
    currency: str = "GBP"
    lawyer_rate_per_minute: float = Field(gt=0)
    lawyer_minutes: dict[ActionName, float]
    p_success: dict[str, float]

    def p_success_for(self, platform: str, rule_id: str) -> float:
        """Lookup with wildcard fallback: exact → platform|* → *|rule → *|*."""
        for key in (f"{platform}|{rule_id}", f"{platform}|*", f"*|{rule_id}", "*|*"):
            if key in self.p_success:
                return self.p_success[key]
        raise KeyError(f"no p_success prior for {platform}|{rule_id} and no '*|*' default")

    def cost_for(self, action: str) -> float:
        return self.lawyer_minutes.get(action, 0.0) * self.lawyer_rate_per_minute  # type: ignore[call-overload]


class VisionConfig(_Strict):
    clip_model: str = "clip-ViT-B-32"
    embeddings_dir: str = "data/embeddings"


class DiscoveryConfig(_Strict):
    max_queries: int = Field(default=14, gt=0)
    misspelling_variants: int = Field(default=4, ge=0)
    thumb_concurrency: int = Field(default=8, gt=0)
    image_concurrency: int = Field(default=8, gt=0)
    max_images_per_listing: int = Field(default=6, gt=0)
    max_image_mb: float = Field(default=8, gt=0)
    screenshot_full_page: bool = True
    stop_after_empty_pages: int = Field(default=2, gt=0)  # adaptive paging; site max_pages is the hard cap


class OwnPhotoConfig(_Strict):
    phash_max: int = Field(default=8, ge=0)
    dhash_max: int = Field(default=10, ge=0)
    clip_min: float = Field(default=0.93, gt=0, le=1)
    orb_min_inliers: int = Field(default=25, gt=0)
    # ORB counts only when the same image is also CLIP-similar to a gold photo (0 = ORB alone suffices)
    orb_clip_min: float = Field(default=0.0, ge=0, le=1)
    crops: list[float] = Field(default_factory=lambda: [0.9, 0.8])


class TextConfig(_Strict):
    misspelling_min_score: float = Field(default=80, gt=0, le=100)
    non_uk_eea_countries: list[str] = Field(default_factory=list)


class ModelsConfig(_Strict):
    scorer_path: str = "data/models/scorer.joblib"


class EnforceConfig(_Strict):
    require_final_confirm: bool = False
    max_evidence_mb: float = Field(default=4, gt=0)
    draft_only_in_budget: bool = True
    allowed_hosts: list[str] = Field(default_factory=lambda: sorted(ENFORCEMENT_HOSTS))

    @model_validator(mode="after")
    def _narrow_only(self) -> "EnforceConfig":
        extra = sorted({h.lower() for h in self.allowed_hosts} - ENFORCEMENT_HOSTS)
        if extra:
            raise ValueError(f"enforce.allowed_hosts may only narrow the code allowlist {sorted(ENFORCEMENT_HOSTS)}; "
                             f"not allowed: {extra} (widening requires a code change)")
        return self


class AppConfig(_Strict):
    brand: str
    paths: PathsConfig
    llm: LLMConfig
    budgets: BudgetsConfig
    mocksites: MockSitesConfig
    sites: dict[str, SiteConfig]
    browser: BrowserConfig
    thresholds: ThresholdsConfig
    decision: DecisionConfig
    uk_eea_countries: list[str]
    vision: VisionConfig = Field(default_factory=VisionConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    enforce: EnforceConfig = Field(default_factory=EnforceConfig)
    own_photo: OwnPhotoConfig = Field(default_factory=OwnPhotoConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    navigator: NavigatorConfig = Field(default_factory=NavigatorConfig)

    def site(self, name: str) -> SiteConfig:
        try:
            return self.sites[name]
        except KeyError:
            raise KeyError(f"site {name!r} not configured in config.yaml") from None

    def snapshot(self) -> dict:
        """JSON-safe snapshot for Run.config_snapshot (contains no secrets by construction)."""
        return self.model_dump(mode="json")


class Secrets(BaseModel):
    """Credentials from the environment. Values are SecretStr; never pass to an LLM."""

    model_config = ConfigDict(frozen=True)

    mock_portal_user: SecretStr | None = None
    mock_portal_pass: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    def secret_values(self) -> list[str]:
        """Raw secret strings, for redaction / leak detection only."""
        out: list[str] = []
        for v in (self.mock_portal_pass, self.anthropic_api_key):
            if v is not None and v.get_secret_value():
                out.append(v.get_secret_value())
        return out

    def require_portal(self) -> tuple[str, str]:
        if not (self.mock_portal_user and self.mock_portal_pass):
            raise RuntimeError("MOCK_PORTAL_USER / MOCK_PORTAL_PASS not set (see .env.example)")
        return self.mock_portal_user.get_secret_value(), self.mock_portal_pass.get_secret_value()


def _config_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("BH_CONFIG", PROJECT_ROOT / "config.yaml"))


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (YAML would silently keep the last one)."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[object] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml_strict(text: str) -> object:
    return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass


def load_config(path: str | Path | None = None) -> AppConfig:
    p = _config_path(path)
    raw = load_yaml_strict(p.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


def load_secrets(env_file: str | Path | None = None) -> Secrets:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    def _get(name: str) -> SecretStr | None:
        v = os.environ.get(name)
        return SecretStr(v) if v else None

    return Secrets(
        mock_portal_user=_get("MOCK_PORTAL_USER"),
        mock_portal_pass=_get("MOCK_PORTAL_PASS"),
        anthropic_api_key=_get("ANTHROPIC_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return load_secrets()
