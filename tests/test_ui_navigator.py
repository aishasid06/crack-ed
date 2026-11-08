"""UI integration of the navigator-branch features (after merging main's UI): real sites stay out of the UI,
refused approvals show a message, and the case file shows the demo banner and "model disagrees"."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session

pytest.importorskip("streamlit")

from bh.config import DEMO_BANNER, get_config  # noqa: E402
from bh.db import Case, CaseStatus, Run  # noqa: E402
from bh.decide import DISAGREE_LABEL  # noqa: E402
from bh.enforce.approvals import ClaimRefused, DecisionError, approve_case  # noqa: E402
from bh.errors import EnforcementBlocked  # noqa: E402
from tests.test_ui import _app, ids  # noqa: E402,F401 (fixture)
from ui import data  # noqa: E402
from ui.views.dossier import extraction_label  # noqa: E402

REASON = "model chose exhausted_resale (rules kept modified_goods): 'personalized' describes in-game features."


def _mark(engine: Engine, case_id: int, *, demo: bool = False, disagrees: bool = False) -> None:
    with Session(engine) as s:
        c = s.get(Case, case_id)
        assert c is not None
        if disagrees:
            c.model_disagrees, c.model_disagreement = True, REASON
        if demo:
            run = s.get(Run, c.run_id)
            assert run is not None
            run.config_snapshot = {"real_brand_demo": True, "demo_banner": DEMO_BANNER}
            s.add(run)
        s.add(c)
        s.commit()


def test_run_page_never_offers_real_sites(ids: dict[str, int]) -> None:  # noqa: F811
    real = [s for s, c in get_config().sites.items() if c.mode == "read_only"]
    assert real, "config should register a read_only (real) site"
    at = _app("run", "A. Lawyer")
    sites = next(m for m in at.multiselect if m.label == "Sites to search")
    assert not set(real) & set(sites.options) and not set(real) & set(sites.value)


def test_review_shows_model_disagreement(engine: Engine, ids: dict[str, int]) -> None:  # noqa: F811
    top = data.review_queue()[0].case.id
    _mark(engine, top, disagrees=True)  # type: ignore[arg-type]
    at = _app("review", "A. Lawyer")
    assert any(DISAGREE_LABEL.capitalize() in w.value and "in-game features" in w.value for w in at.warning)


def test_refused_approval_shows_a_message_not_a_crash(engine: Engine, ids: dict[str, int]) -> None:  # noqa: F811
    top = data.review_queue()[0].case.id
    _mark(engine, top, demo=True)  # type: ignore[arg-type]
    at = _app("review", "Priya Shah")
    assert any(DEMO_BANNER in i.value for i in at.info)  # the case file carries the banner
    next(b for b in at.button if b.label == "Approve and submit").click().run()
    assert not at.exception and any("no claim is made" in e.value for e in at.error)
    with Session(engine) as s:
        c = s.get(Case, top)
        assert c is not None and c.status != CaseStatus.approved and c.decided_by is None


def test_cases_page_shows_the_demo_banner(engine: Engine, ids: dict[str, int]) -> None:  # noqa: F811
    _mark(engine, data.review_queue()[0].case.id, demo=True)  # type: ignore[arg-type]
    at = _app("cases", "A. Lawyer")
    assert any(DEMO_BANNER in i.value for i in at.info)


def test_claim_refused_is_both_kinds_of_error(engine: Engine, ids: dict[str, int]) -> None:  # noqa: F811
    top = data.review_queue()[0].case.id
    _mark(engine, top, demo=True)  # type: ignore[arg-type]
    with pytest.raises(ClaimRefused) as ei:
        approve_case(top, "Priya Shah", engine)  # type: ignore[arg-type]
    assert isinstance(ei.value, DecisionError) and isinstance(ei.value, EnforcementBlocked)


@pytest.mark.parametrize("platform,method,label", [
    ("mockbay", "adapter", "site adapter"),
    ("mockbay", "generic_llm", "LLM fallback extractor"),
    ("bazaarly", "generic_llm", "agentic navigator"),
    ("bazaarly", "recipe", "learned recipe (replayed, no LLM)"),
])
def test_extraction_labels(platform: str, method: str, label: str) -> None:
    assert extraction_label(NS(platform=platform, extraction_method=method)) == label

