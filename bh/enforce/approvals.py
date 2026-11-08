"""Human decisions on cases. These are the ONLY functions that move a case to approved / rejected;
the UI and this CLI call them. Nothing in the pipeline approves automatically.

"Approve and submit" is the single human authorisation (SPEC §7.1): it also authorises signing the
portal declarations in the approver's name.

    python -m bh.enforce.approvals approve <case_id> --by "<name>"
    python -m bh.enforce.approvals reject  <case_id> --by "<name>"
    python -m bh.enforce.approvals confirm <case_id> --by "<name>"   # only with enforce.require_final_confirm
    python -m bh.enforce.approvals resume  <case_id>                  # after a human handled waiting_human
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy.engine import Engine
from sqlmodel import Session

from bh.db import Case, CaseStatus, get_engine, session_scope, utcnow

APPROVABLE = {CaseStatus.awaiting_approval, CaseStatus.drafted}
REJECTABLE = {CaseStatus.awaiting_approval, CaseStatus.drafted, CaseStatus.approved, CaseStatus.waiting_human,
              CaseStatus.classified}


class DecisionError(ValueError):
    pass


def _user(user: str) -> str:
    u = (user or "").strip()
    if not u:
        raise DecisionError("a named human decision-maker is required")
    return u


def approve_case(case_id: int, user: str, engine: Engine | None = None) -> None:
    """Approve and submit: the submit loop may now sign declarations as ``user`` and submit."""
    u = _user(user)
    with session_scope(engine) as s:
        c = s.get(Case, case_id)
        if c is None:
            raise DecisionError(f"case {case_id} not found")
        if c.status not in APPROVABLE:
            raise DecisionError(f"case {case_id} is {c.status}; only awaiting_approval/drafted cases can be approved")
        c.status, c.decided_by, c.decided_at = CaseStatus.approved, u, utcnow()


def reject_case(case_id: int, user: str, engine: Engine | None = None) -> None:
    u = _user(user)
    with session_scope(engine) as s:
        c = s.get(Case, case_id)
        if c is None:
            raise DecisionError(f"case {case_id} not found")
        if c.status not in REJECTABLE or c.submission_ref:
            raise DecisionError(f"case {case_id} is {c.status}; it can no longer be rejected")
        c.status, c.decided_by, c.decided_at = CaseStatus.rejected, u, utcnow()


def confirm_final(case_id: int, user: str, engine: Engine | None = None) -> None:
    """Second confirmation (only when enforce.require_final_confirm): after the review page was verified."""
    u = _user(user)
    with session_scope(engine) as s:
        c = s.get(Case, case_id)
        if c is None or c.status != CaseStatus.approved or c.review_verified_at is None:
            raise DecisionError(f"case {case_id} has no verified review page awaiting confirmation")
        if c.decided_by != u:
            raise DecisionError("the final confirmation must come from the approving lawyer")
        c.final_confirmed = True


def resume_case(case_id: int, engine: Engine | None = None) -> CaseStatus:
    """After a human dealt with a challenge or a review-page mismatch: put the case back in the queue.
    A mismatch discards the portal draft so the next attempt drafts afresh from the approved case."""
    with session_scope(engine) as s:
        c = s.get(Case, case_id)
        if c is None or c.status != CaseStatus.waiting_human:
            raise DecisionError(f"case {case_id} is not waiting for a human")
        if c.review_diff:
            c.portal_draft_id, c.review_verified_at = None, None
        if c.decided_by:
            c.status = CaseStatus.approved
        elif c.portal_draft_id:
            c.status = CaseStatus.drafted
        else:
            c.status = CaseStatus.awaiting_approval
        return c.status


def _describe(case_id: int, engine: Engine | None) -> str:
    with Session(engine or get_engine()) as s:
        c = s.get(Case, case_id)
    assert c is not None
    who = f" by {c.decided_by} at {c.decided_at:%Y-%m-%d %H:%M} UTC" if c.decided_by and c.decided_at else ""
    return f"case {case_id}: {c.status}{who} (rule {c.rule_id}, portal draft {c.portal_draft_id or '-'})"


def main(argv: list[str] | None = None, engine: Engine | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bh.enforce.approvals", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_ in (("approve", "Approve and submit: the submit loop may sign as --by and submit"),
                        ("reject", "Reject: never signed, never submitted"),
                        ("confirm", "Second confirmation after review-page verification (require_final_confirm)")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("case_id", type=int)
        p.add_argument("--by", required=True, help="name of the lawyer making the decision (declaration signatory)")
    p = sub.add_parser("resume", help="put a waiting_human case back in the queue")
    p.add_argument("case_id", type=int)
    args = ap.parse_args(argv)
    try:
        if args.command == "approve":
            approve_case(args.case_id, args.by, engine)
        elif args.command == "reject":
            reject_case(args.case_id, args.by, engine)
        elif args.command == "confirm":
            confirm_final(args.case_id, args.by, engine)
        else:
            resume_case(args.case_id, engine)
    except DecisionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(_describe(args.case_id, engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

