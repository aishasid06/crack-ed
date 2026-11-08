"""Background processes the UI can start and stop. The pipeline never runs inside Streamlit (SPEC §2):
each worker is a separate OS process whose pid, command and log live in ``data/ui/<name>.json``.

Workers:
  pipeline — the full run (``bh.orchestrator.run`` once Phase 8 exists, otherwise ``ui.pipeline``,
             which chains the stage CLIs: discover → classify → decide → draft)
  submit   — the submit-on-approval loop (``bh.enforce.submit --poll``)
  mock     — the local mock marketplaces + report portal (uvicorn)
  eval     — ``eval.evaluate`` (Phase 10), when present
  draft    — one drafting pass (``bh.enforce.portal``), e.g. after a human cleared a challenge
  search-<id> — one shopper search (``bh.shopper``)
  report-<id> — file one shopper report through a marketplace's report form (``bh.enforce.listing_report``)
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bh.config import PROJECT_ROOT, get_config

IS_WINDOWS = os.name == "nt"


def state_dir() -> Path:
    d = get_config().paths.resolve(get_config().paths.data_dir) / "ui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def pipeline_command(brand: str, use_llm: bool) -> list[str]:
    if has_module("bh.orchestrator.run"):
        cmd = [sys.executable, "-m", "bh.orchestrator.run", "--brand", brand]
    else:
        cmd = [sys.executable, "-m", "ui.pipeline", "--brand", brand]
    return cmd + (["--llm"] if use_llm else [])


def command_for(name: str, **kw: Any) -> list[str]:
    cfg = get_config()
    if name.startswith("report-"):  # a shopper's listing report (bh.enforce.listing_report)
        return [sys.executable, "-m", "bh.enforce.listing_report", "--report-id", name.removeprefix("report-")]
    if name.startswith("search-"):  # a shopper search (bh.shopper), one process per search
        cmd = [sys.executable, "-m", "bh.shopper", "--search-id", name.removeprefix("search-")]
        if kw.get("text"):
            cmd += ["--text", str(kw["text"])]
        if kw.get("photo"):
            cmd += ["--photo", str(kw["photo"])]
        return cmd
    if name == "pipeline":
        return pipeline_command(kw.get("brand", cfg.brand), bool(kw.get("use_llm")))
    if name == "submit":
        return [sys.executable, "-m", "bh.enforce.submit", "--poll", str(kw.get("interval", 5))]
    if name == "mock":
        return [sys.executable, "-m", "uvicorn", "--factory", "mocksites.app:create_default_app",
                "--host", cfg.mocksites.host, "--port", str(cfg.mocksites.port)]
    if name == "eval":
        return [sys.executable, "-m", "eval.evaluate"]
    if name == "draft":
        return [sys.executable, "-m", "bh.enforce.portal"]
    raise KeyError(name)


@dataclass
class WorkerState:
    name: str
    pid: int | None
    cmd: list[str]
    started_at: datetime | None
    log_path: Path
    alive: bool
    exit_code: int | None
    extra: dict[str, Any]

    @property
    def label(self) -> str:
        return "running" if self.alive else ("finished" if self.exit_code == 0 else
                                             ("failed" if self.started_at else "not started"))


def _pid_alive(pid: int) -> tuple[bool, int | None]:
    """(alive, exit_code if known). Never sends a signal on Windows (os.kill would terminate)."""
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False, None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False, None
            return (code.value == 259, None if code.value == 259 else int(code.value))  # 259 = STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False, os.waitstatus_to_exitcode(status)
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True, None
    except OSError:
        return False, None


def _read(name: str) -> dict[str, Any]:
    p = state_dir() / f"{name}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(name: str, data: dict[str, Any]) -> None:
    (state_dir() / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def status(name: str) -> WorkerState:
    d = _read(name)
    pid = d.get("pid")
    alive, code = _pid_alive(pid) if pid else (False, None)
    if not alive and pid and d.get("exit_code") is None:
        d["exit_code"] = code if code is not None else _exit_from_log(Path(d.get("log", "")))
        _write(name, d)
    started = datetime.fromisoformat(d["started_at"]) if d.get("started_at") else None
    return WorkerState(name, pid, d.get("cmd", []), started, Path(d.get("log", state_dir() / f"{name}.log")),
                       alive, None if alive else d.get("exit_code"), d.get("extra", {}))


def _exit_from_log(log: Path) -> int | None:
    """Exit code of a finished non-child process is unknown on POSIX; ui.pipeline writes a final marker."""
    try:
        tail = log.read_text(encoding="utf-8", errors="replace")[-400:]
    except OSError:
        return None
    if "[bh] exit 0" in tail:
        return 0
    return 1 if "[bh] exit" in tail or "Traceback" in tail else None


def start(name: str, extra: dict[str, Any] | None = None, **kw: Any) -> WorkerState:
    current = status(name)
    if current.alive:
        return current
    cmd = command_for(name, **kw)
    log_path = state_dir() / f"{name}.log"
    log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - handed to the child process
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    opts: dict[str, Any] = {}
    if IS_WINDOWS:
        opts["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    else:
        opts["start_new_session"] = True
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, **opts)
    log.close()
    _write(name, {"pid": proc.pid, "cmd": cmd, "started_at": datetime.now(UTC).isoformat(), "log": str(log_path),
                  "exit_code": None, "extra": extra or {}})
    return status(name)


def stop(name: str) -> None:
    st = status(name)
    if not st.alive or not st.pid:
        return
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(st.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(os.getpgid(st.pid), signal.SIGTERM)
        except OSError:
            pass
    d = _read(name)
    d["exit_code"] = -1
    d.setdefault("extra", {})["stopped_by_user"] = True
    _write(name, d)


def tail(name: str, lines: int = 40) -> str:
    p = status(name).log_path
    try:
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def progress(name: str = "pipeline") -> dict[str, Any]:
    """Stage progress written by ui.pipeline (``data/ui/<name>.progress.json``)."""
    try:
        return json.loads((state_dir() / f"{name}.progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mock_sites_up(timeout: float = 0.6) -> bool:
    cfg = get_config()
    try:
        with urllib.request.urlopen(f"http://{cfg.mocksites.host}:{cfg.mocksites.port}/", timeout=timeout) as r:
            return 200 <= r.status < 500
    except Exception:
        return False
