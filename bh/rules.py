"""Load and validate ``rules.yaml`` (legal rules are data, SPEC §1/§7.3).

Element predicates are implemented in ``bh/classify.py`` (Phase 6); this module only
parses the rule set and answers structural questions about it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bh.db import Action, Category


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LegalBasis(_M):
    statute: str
    section: str
    summary: str

    def citation(self) -> str:
        return f"{self.statute} {self.section}"


class Rule(_M):
    id: str
    category: Category
    legal_basis: LegalBasis | None
    requires: list[str]
    action: Action
    min_confidence: float | None = Field(default=None, ge=0, le=1)


class Overlay(_M):
    id: str
    applies_to_categories: list[Category]
    legal_basis: LegalBasis
    requires: list[str] = Field(min_length=1)
    action: Action


class Background(_M):
    area: str
    law: str
    summary: str


class RuleSet(_M):
    background: list[Background]
    rules: list[Rule] = Field(min_length=2)
    overlays: list[Overlay] = Field(default_factory=list)
    vocabulary: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "RuleSet":
        ids = [r.id for r in self.rules] + [o.id for o in self.overlays]
        if len(ids) != len(set(ids)):
            raise ValueError("rule/overlay ids must be unique")
        if self.rules[-1].requires:
            raise ValueError("the last rule must be an unconditional fallback (empty requires)")
        if any(not r.requires for r in self.rules[:-1]):
            raise ValueError("only the last rule may have empty requires")
        missing = set(Category) - {r.category for r in self.rules}
        if missing:
            raise ValueError(f"no rule produces categories: {sorted(missing)}")
        return self

    def rule(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(rule_id)

    def rule_for_category(self, category: Category | str) -> Rule:
        """The first (non-fallback) rule that yields this category."""
        cat = Category(category)
        for r in self.rules:
            if r.category == cat and r.requires:
                return r
        return self.rules[-1]

    def elements(self) -> set[str]:
        return {e for r in self.rules for e in r.requires} | {e for o in self.overlays for e in o.requires}


def load_rules(path: Path) -> RuleSet:
    return RuleSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def get_rules() -> RuleSet:
    from bh.config import get_config

    cfg = get_config()
    return load_rules(cfg.paths.resolve(cfg.paths.rules_file))
