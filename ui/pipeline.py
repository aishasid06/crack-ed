"""Stage-chain pipeline the Run page launches while the LangGraph orchestrator (SPEC §9 Phase 8,
``bh.orchestrator.run``) does not exist yet. It runs the existing stage CLIs one after another in child
processes, exactly as the Makefile targets do:

    discover → (fit scorer if missing) → classify → decide → draft

Each stage opens its own Run row (traced as usual). Progress goes to ``data/ui/pipeline.progress.json``
so the UI can show which stage is running. Once ``bh.orchestrator.run`` exists the UI launches that instead.

    python -m ui.pipeline [--brand SLUG] [--llm]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any

from bh.config import PROJECT_ROOT, get_config
from ui.workers import state_dir

STAGES: list[tuple[str, str, list[str]]] = [
    ("discover", "Search the marketplaces and capture listings", ["-m", "bh.discovery.worker"]),
    ("fit_scorer", "Fit the calibrated scorer (dev split)", ["-m", "eval.fit_scorer"]),
    ("classify", "Classify listings against the legal rules", ["-m", "bh.classify"]),
    ("decide", "Rank by expected value and prepare notices", ["-m", "bh.decide"]),
    ("draft", "Draft portal reports (steps 1–4, no declarations)", ["-m", "bh.enforce.portal"]),
]
LLM_STAGES = {"discover", "classify"}


def _save(progress: dict[str, Any]) -> None:
    (state_dir() / "pipeline.progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", default=None)
    ap.add_argument("--llm", action="store_true", help="enable LLM fallback extraction and borderline checks")
    args = ap.parse_args()
    cfg = get_config()
    if args.brand and args.brand != cfg.brand:
        print(f"[bh] brand {args.brand!r} is not the configured brand {cfg.brand!r}; point BH_CONFIG at a "
              f"config for that brand", flush=True)
        print("[bh] exit 2", flush=True)
        return 2
    scorer_exists = cfg.paths.resolve(cfg.models.scorer_path).exists()
    progress: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(), "brand": cfg.brand, "llm": args.llm,
        "stages": [{"id": sid, "label": label, "status": "skipped" if sid == "fit_scorer" and scorer_exists
                    else "pending"} for sid, label, _ in STAGES],
    }
    _save(progress)
    code = 0
    for i, (sid, label, argv) in enumerate(STAGES):
        stage = progress["stages"][i]
        if stage["status"] == "skipped":
            continue
        cmd = [sys.executable, *argv] + (["--llm"] if args.llm and sid in LLM_STAGES else [])
        stage.update(status="running", started_at=datetime.now(UTC).isoformat())
        _save(progress)
        print(f"[bh] stage {sid}: {label}", flush=True)
        t0 = time.monotonic()
        code = subprocess.call(cmd, cwd=PROJECT_ROOT)
        stage.update(status="succeeded" if code == 0 else "failed", seconds=round(time.monotonic() - t0, 1),
                     exit_code=code)
        _save(progress)
        if code != 0:
            print(f"[bh] stage {sid} failed with exit code {code}", flush=True)
            break
    progress["finished_at"] = datetime.now(UTC).isoformat()
    _save(progress)
    print(f"[bh] exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
