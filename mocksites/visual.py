"""Server-side visual search for TradeLoft: rank listings by CLIP image similarity."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from mocksites.catalog import MockListing


class ImageEmbedder(Protocol):
    name: str

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Return an (n, d) array of L2-normalised embeddings."""
        ...


class ClipEmbedder:
    """sentence-transformers ``clip-ViT-B-32``; the model loads lazily on first use."""

    name = "clip-ViT-B-32"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _get(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name)
            return self._model

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        model = self._get()
        vecs = model.encode([im.convert("RGB") for im in images], convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class VisualIndex:
    """Embeddings for every image of every listing, computed once and cached on disk."""

    def __init__(self, embedder: ImageEmbedder, images_dir: Path, cache_path: Path | None) -> None:
        self.embedder = embedder
        self.images_dir = images_dir
        self.cache_path = cache_path
        self._vecs: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as data:
                if str(data["__embedder__"]) == self.embedder.name:
                    self._vecs = {k: data[k] for k in data.files if k != "__embedder__"}

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp.npz")
            np.savez(tmp, __embedder__=np.array(self.embedder.name), **self._vecs)
            tmp.replace(self.cache_path)

    def _ensure(self, names: list[str]) -> None:
        with self._lock:
            missing = sorted({n for n in names if n not in self._vecs})
            if not missing:
                return
            imgs = []
            for n in missing:
                with Image.open(self.images_dir / n) as im:
                    imgs.append(im.convert("RGB"))
            for n, v in zip(missing, self.embedder.embed(imgs), strict=True):
                self._vecs[n] = v
            self._save_cache()

    def rank(self, query: Image.Image, listings: list[MockListing], top_k: int = 24) -> list[tuple[MockListing, float]]:
        if not listings:
            return []
        self._ensure([n for l in listings for n in l.images])
        q = self.embedder.embed([query.convert("RGB")])[0]
        scored = []
        for l in listings:
            sim = max(float(np.dot(q, self._vecs[n])) for n in l.images)
            scored.append((l, sim))
        scored.sort(key=lambda t: (-t[1], t[0].id))
        return scored[:top_k]
