"""Fit the calibrated should_report scorer on the DEV split only.

Computes features (bh.features, exactly as classification does) for listings captured by a discovery
run, joins them to seed/labels.json by URL path, keeps dev-split rows, and fits
CalibratedClassifierCV(LogisticRegression, sigmoid, 3-fold). Saves {model, feature_names, meta} to
config.models.scorer_path and reports 3-fold cross-validated metrics on dev.

    python -m eval.fit_scorer [--run latest|<id>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sqlmodel import Session

from bh.brand_index import load_brand
from bh.classify import compute_features, listings_of_run
from bh.config import AppConfig, get_config
from bh.db import Listing, get_engine, utcnow
from bh.features import FEATURE_NAMES, vector
from bh.rules import get_rules
from bh.trace import run_span
from bh.vision import ClipEncoder, ImageEncoder

ROOT = Path(__file__).resolve().parent.parent


def make_model() -> CalibratedClassifierCV:
    base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


async def build_dataset(source_run: int | None, cfg: AppConfig, encoder: ImageEncoder,
                        labels: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    eng = get_engine()
    ids = listings_of_run(source_run, eng)
    if not ids:
        raise RuntimeError("no captured listings — run discovery first (python -m bh.discovery.worker)")
    by_path = {v["url_path"]: (k, v) for k, v in labels.items()}
    async with run_span(None, {"stage": "fit_scorer", "source_run": source_run}) as run:
        ctx = await load_brand(cfg, encoder, eng)
        feats = await compute_features(run.id, ids, ctx, cfg, encoder, eng, get_rules())
    X, y, keys, splits = [], [], [], []
    with Session(eng) as s:
        for lid, (_, f) in feats.items():
            row = s.get(Listing, lid)
            hit = by_path.get(urlparse(row.url).path) if row else None
            if hit is None:
                continue
            key, lab = hit
            X.append(vector(f, cfg.thresholds.counterfeit_price_ratio))
            y.append(int(lab["should_report"]))
            keys.append(key)
            splits.append(lab["split"])
    return np.asarray(X, float), np.asarray(y, int), keys, splits


def fit(source_run: int | None = None, labels_path: Path = ROOT / "seed" / "labels.json",
        encoder: ImageEncoder | None = None, cfg: AppConfig | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    labels = json.loads(labels_path.read_text())
    X, y, keys, splits = asyncio.run(build_dataset(source_run, cfg, encoder or ClipEncoder(cfg.vision.clip_model),
                                                   labels))
    dev = np.asarray([s == "dev" for s in splits])
    Xd, yd = X[dev], y[dev]
    if len(set(yd)) < 2:
        raise RuntimeError("dev split needs both classes to fit the scorer")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    oof = cross_val_predict(make_model(), Xd, yd, cv=cv, method="predict_proba")[:, 1]
    model = make_model().fit(Xd, yd)
    meta = {"trained_at": utcnow().isoformat(), "split": "dev", "n": int(dev.sum()), "positives": int(yd.sum()),
            "cv_auc": round(float(roc_auc_score(yd, oof)), 4), "cv_brier": round(float(brier_score_loss(yd, oof)), 4),
            "source_run": source_run, "model": "CalibratedClassifierCV(LogisticRegression, sigmoid, cv=3)"}
    path = cfg.paths.resolve(cfg.models.scorer_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "meta": meta}, path)
    coefs = model.calibrated_classifiers_[0].estimator.coef_[0]
    top = sorted(zip(FEATURE_NAMES, coefs, strict=True), key=lambda t: -abs(t[1]))[:8]
    return {**meta, "path": str(path), "n_captured_labelled": len(y),
            "top_coefficients_fold0": {k: round(float(v), 3) for k, v in top}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest")
    args = ap.parse_args()
    print(json.dumps(fit(None if args.run == "latest" else int(args.run)), indent=2))


if __name__ == "__main__":
    main()
