"""Rights-holder IP report portal: login, 5-step report draft, review + submit,
submissions list, and the human-verification page (which agents must never solve)."""

from __future__ import annotations

import hmac
import io
import re
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from PIL import Image, UnidentifiedImageError
from sqlmodel import select
from starlette.datastructures import UploadFile

from mocksites import portal_forms as pf
from mocksites.app import client_key
from mocksites.store import (
    PortalSession, ReportDraft, Submission, new_draft_id, new_ref, new_token, store_session, utcnow,
)

router = APIRouter(prefix="/portal")
SESSION_COOKIE = "portal_session"
STEP_TITLES = {1: "Rights owner details", 2: "Right relied on", 3: "Infringing listings",
               4: "Evidence", 5: "Declarations"}
MAX_EVIDENCE_FILES = 10


class LoginRequired(Exception):
    def __init__(self, next_url: str) -> None:
        self.next_url = next_url


def install(app: FastAPI) -> None:
    async def _redirect_login(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, LoginRequired)
        return RedirectResponse(f"/portal/login?next={quote(exc.next_url, safe='')}", status_code=303)

    app.add_exception_handler(LoginRequired, _redirect_login)


# ---------------------------------------------------------------- helpers


def _t(request: Request, name: str, ctx: dict[str, Any], status_code: int = 200) -> Response:
    return request.app.state.templates.TemplateResponse(request, name, ctx, status_code=status_code)


def _safe_next(nxt: str | None, default: str = "/portal/") -> str:
    if nxt and nxt.startswith("/portal") and not nxt.startswith("/portal/verify"):
        return nxt
    return default


def _user(request: Request) -> str:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with store_session(request.app.state.store) as s:
            sess = s.get(PortalSession, token)
            if sess is not None:
                return sess.username
    nxt = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    raise LoginRequired(nxt)


def _draft(request: Request, draft_id: str, user: str) -> ReportDraft:
    with store_session(request.app.state.store) as s:
        d = s.get(ReportDraft, draft_id)
    if d is None or d.username != user:
        raise HTTPException(404, "Report not found")
    return d


def _complete_steps(d: ReportDraft) -> set[int]:
    done = {int(k) for k in d.steps if k in {"1", "2", "3", "5"}}
    if d.evidence:
        done.add(4)
    return done


def _first_incomplete(d: ReportDraft) -> int:
    done = _complete_steps(d)
    for n in range(1, 6):
        if n not in done:
            return n
    return 6  # all complete → review


def _step_url(d: ReportDraft, n: int) -> str:
    return f"/portal/report/{d.id}/review" if n == 6 else f"/portal/report/{d.id}/step/{n}"


def _save_step(request: Request, d: ReportDraft, n: int, data: dict[str, Any]) -> None:
    with store_session(request.app.state.store) as s:
        db = s.get(ReportDraft, d.id)
        assert db is not None
        steps = dict(db.steps)
        steps[str(n)] = data
        # changing the right type can invalidate step-3 grounds → step 3 must be redone
        if n == 2 and "3" in steps and _step3_conflicts(steps["3"], data["right_type"]):
            del steps["3"]
        db.steps = steps
        db.updated_at = utcnow()


def _step3_conflicts(step3: dict[str, Any], right_type: str) -> bool:
    for it in step3.get("items", []):
        kind = pf.STATUTES.get(it.get("statute", ""))
        if kind == "trade_mark" and right_type not in ("trade_mark", "both"):
            return True
        if kind == "copyright" and right_type not in ("copyright", "both"):
            return True
    return False


def _step_ctx(d: ReportDraft, n: int, values: dict[str, Any], errors: dict[str, str]) -> dict[str, Any]:
    return {"d": d, "n": n, "titles": STEP_TITLES, "values": values, "errors": errors,
            "done": _complete_steps(d), "pf": pf}


# ---------------------------------------------------------------- auth


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/portal/") -> Response:
    return _t(request, "portal/login.html", {"next": _safe_next(next), "error": None})


@router.post("/login")
def login(request: Request, username: str = Form(""), password: str = Form(""),
          next: str = Form("/portal/")) -> Response:
    st = request.app.state.settings
    ok_user = hmac.compare_digest(username.encode(), st.portal_user.encode())
    ok_pass = hmac.compare_digest(password.encode(), st.portal_pass.encode())
    if not (ok_user and ok_pass):
        return _t(request, "portal/login.html", {"next": _safe_next(next), "error": "Incorrect email or password."},
                  status_code=401)
    token = new_token()
    with store_session(request.app.state.store) as s:
        s.add(PortalSession(token=token, username=username))
    resp = RedirectResponse(_safe_next(next), status_code=303)
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", path="/portal")
    return resp


@router.post("/logout")
def logout(request: Request) -> Response:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with store_session(request.app.state.store) as s:
            sess = s.get(PortalSession, token)
            if sess:
                s.delete(sess)
    resp = RedirectResponse("/portal/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE, path="/portal")
    return resp


# ---------------------------------------------------------------- human verification


@router.get("/verify", response_class=HTMLResponse)
def verify_form(request: Request, next: str = "/portal/") -> Response:
    return _t(request, "portal/verify.html", {"next": _safe_next(next), "error": None})


@router.post("/verify")
def verify(request: Request, confirm_human: str = Form(""), next: str = Form("/portal/")) -> Response:
    if confirm_human != "yes":
        return _t(request, "portal/verify.html", {"next": _safe_next(next),
                  "error": "Please confirm you are a human to continue."}, status_code=400)
    request.app.state.gate.mark_verified(client_key(request))
    return RedirectResponse(_safe_next(next), status_code=303)


# ---------------------------------------------------------------- dashboard / drafts


@router.get("", include_in_schema=False)
@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> Response:
    user = _user(request)
    with store_session(request.app.state.store) as s:
        drafts = list(s.exec(select(ReportDraft).where(ReportDraft.username == user,
                                                       ReportDraft.submitted_ref.is_(None))  # type: ignore[union-attr]
                             .order_by(ReportDraft.updated_at.desc())))  # type: ignore[attr-defined]
        n_sub = len(list(s.exec(select(Submission).where(Submission.username == user))))
    return _t(request, "portal/dashboard.html", {"user": user, "drafts": drafts, "n_submissions": n_sub,
                                                 "first_incomplete": _first_incomplete, "step_url": _step_url})


@router.post("/report/new")
def new_report(request: Request) -> Response:
    user = _user(request)
    d = ReportDraft(id=new_draft_id(), username=user)
    with store_session(request.app.state.store) as s:
        s.add(d)
    return RedirectResponse(f"/portal/report/{d.id}/step/1", status_code=303)


def _guard(d: ReportDraft, n: int) -> Response | None:
    if d.submitted_ref:
        return RedirectResponse(f"/portal/submissions/{d.submitted_ref}", status_code=303)
    if not 1 <= n <= 5:
        raise HTTPException(404, "No such step")
    first = _first_incomplete(d)
    if n > first:
        return RedirectResponse(_step_url(d, first), status_code=303)
    return None


@router.get("/report/{draft_id}/step/{n}", response_class=HTMLResponse)
def step_form(request: Request, draft_id: str, n: int) -> Response:
    d = _draft(request, draft_id, _user(request))
    if (r := _guard(d, n)) is not None:
        return r
    values = d.steps.get(str(n), {})
    return _t(request, f"portal/step{n}.html", _step_ctx(d, n, values, {}))


@router.post("/report/{draft_id}/step/{n}", response_class=HTMLResponse)
async def step_submit(request: Request, draft_id: str, n: int) -> Response:
    d = _draft(request, draft_id, _user(request))
    if (r := _guard(d, n)) is not None:
        return r
    form = await request.form()
    flat = {k: v for k, v in form.items() if isinstance(v, str)}
    errors: dict[str, str]
    if n == 1:
        data, errors = pf.parse_step1(flat)
    elif n == 2:
        data, errors = pf.parse_step2(flat)
    elif n == 3:
        lists = {k: [v for v in form.getlist(k) if isinstance(v, str)] for k in ("url", "statute", "section", "details")}
        data, errors = pf.parse_step3(lists, d.steps["2"]["right_type"],
                                      request.app.state.settings.allowed_listing_hosts)
    elif n == 4:
        files = [f for f in form.getlist("evidence") if isinstance(f, UploadFile) and f.filename]
        errors = await _save_evidence(request, d, files)
        data = {}
        d = _draft(request, draft_id, d.username)
        if not errors and not d.evidence:
            errors = {"evidence": "Upload at least one screenshot as evidence."}
        if errors:
            return _t(request, "portal/step4.html", _step_ctx(d, n, {}, errors), status_code=422)
        return RedirectResponse(_step_url(d, 5), status_code=303)
    else:
        data, errors = pf.parse_step5(flat)
    if errors:
        return _t(request, f"portal/step{n}.html", _step_ctx(d, n, data, errors), status_code=422)
    _save_step(request, d, n, data)
    d = _draft(request, draft_id, d.username)
    nxt = n + 1 if n < 5 else _first_incomplete(d)  # after step 5 → review (or first gap)
    return RedirectResponse(_step_url(d, nxt), status_code=303)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


async def _save_evidence(request: Request, d: ReportDraft, files: list[UploadFile]) -> dict[str, str]:
    settings = request.app.state.settings
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(d.evidence) + len(files) > MAX_EVIDENCE_FILES:
        return {"evidence": f"At most {MAX_EVIDENCE_FILES} evidence files per report."}
    accepted: list[tuple[str, bytes]] = []
    for f in files:
        raw = await f.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return {"evidence": f"{f.filename}: file exceeds {settings.max_upload_mb:g} MB."}
        try:
            with Image.open(io.BytesIO(raw)) as im:
                fmt = im.format
                im.verify()
        except (UnidentifiedImageError, OSError):
            return {"evidence": f"{f.filename}: not a valid PNG or JPEG image."}
        if fmt not in ("PNG", "JPEG"):
            return {"evidence": f"{f.filename}: only PNG or JPEG screenshots are accepted."}
        accepted.append((_SAFE_NAME.sub("_", f.filename or "evidence")[-80:], raw))
    if not accepted:
        return {}
    folder = settings.uploads_dir / d.id
    folder.mkdir(parents=True, exist_ok=True)
    with store_session(request.app.state.store) as s:
        db = s.get(ReportDraft, d.id)
        assert db is not None
        names = list(db.evidence)
        for name, raw in accepted:
            stored = f"{len(names) + 1:02d}_{name}"
            (folder / stored).write_bytes(raw)
            names.append(stored)
        db.evidence = names
        db.updated_at = utcnow()
    return {}


@router.post("/report/{draft_id}/evidence/{name}/delete")
def delete_evidence(request: Request, draft_id: str, name: str) -> Response:
    d = _draft(request, draft_id, _user(request))
    if d.submitted_ref:
        raise HTTPException(409, "Report already submitted")
    with store_session(request.app.state.store) as s:
        db = s.get(ReportDraft, d.id)
        assert db is not None
        if name in db.evidence:
            db.evidence = [e for e in db.evidence if e != name]
            (request.app.state.settings.uploads_dir / d.id / name).unlink(missing_ok=True)
    return RedirectResponse(f"/portal/report/{d.id}/step/4", status_code=303)


@router.get("/evidence/{draft_id}/{name}")
def evidence_file(request: Request, draft_id: str, name: str) -> Response:
    d = _draft(request, draft_id, _user(request))
    if name not in d.evidence:
        raise HTTPException(404, "Not found")
    return FileResponse(request.app.state.settings.uploads_dir / d.id / name)


# ---------------------------------------------------------------- review / submit


@router.get("/report/{draft_id}/review", response_class=HTMLResponse)
def review(request: Request, draft_id: str) -> Response:
    d = _draft(request, draft_id, _user(request))
    if d.submitted_ref:
        return RedirectResponse(f"/portal/submissions/{d.submitted_ref}", status_code=303)
    first = _first_incomplete(d)
    if first != 6:
        return RedirectResponse(_step_url(d, first), status_code=303)
    return _t(request, "portal/review.html", {"d": d, "s": d.steps, "pf": pf, "titles": STEP_TITLES,
                                              "n": 6, "done": _complete_steps(d)})


@router.post("/report/{draft_id}/submit")
def submit(request: Request, draft_id: str) -> Response:
    user = _user(request)
    d = _draft(request, draft_id, user)
    if d.submitted_ref:  # idempotent
        return RedirectResponse(f"/portal/submissions/{d.submitted_ref}", status_code=303)
    if _first_incomplete(d) != 6:
        return RedirectResponse(_step_url(d, _first_incomplete(d)), status_code=303)
    ref = new_ref()
    with store_session(request.app.state.store) as s:
        db = s.get(ReportDraft, d.id)
        assert db is not None
        s.add(Submission(ref=ref, draft_id=d.id, username=user, data=dict(db.steps), evidence=list(db.evidence)))
        db.submitted_ref = ref
        db.updated_at = utcnow()
    return RedirectResponse(f"/portal/submissions/{ref}?new=1", status_code=303)


@router.get("/submissions", response_class=HTMLResponse)
def submissions(request: Request) -> Response:
    user = _user(request)
    with store_session(request.app.state.store) as s:
        subs = list(s.exec(select(Submission).where(Submission.username == user)
                           .order_by(Submission.submitted_at.desc())))  # type: ignore[attr-defined]
    return _t(request, "portal/submissions.html", {"subs": subs, "pf": pf})


@router.get("/submissions/{ref}", response_class=HTMLResponse)
def submission_detail(request: Request, ref: str, new: int = 0) -> Response:
    user = _user(request)
    with store_session(request.app.state.store) as s:
        sub = s.exec(select(Submission).where(Submission.ref == ref)).first()
    if sub is None or sub.username != user:
        raise HTTPException(404, "Submission not found")
    return _t(request, "portal/submission.html", {"sub": sub, "s": sub.data, "pf": pf, "new": bool(new)})
