"""Image helpers: perceptual hashes and a lazily-loaded CLIP encoder.

All functions are synchronous and CPU-bound; async callers wrap them in ``asyncio.to_thread``.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Protocol

import imagehash
import numpy as np
from PIL import Image, ImageOps


class ImageEncoder(Protocol):
    name: str

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """(n, d) L2-normalised embeddings."""
        ...


class ClipEncoder:
    """sentence-transformers CLIP; loads on first use, thread-safe."""

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name)
            vecs = self._model.encode([im.convert("RGB") for im in images], convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return np.asarray(vecs, dtype=np.float32)


def open_rgb(src: Path | bytes) -> Image.Image:
    with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
        return im.convert("RGB")


def hashes(im: Image.Image) -> dict[str, imagehash.ImageHash]:
    """pHash and dHash of the image and of its mirror image."""
    m = ImageOps.mirror(im)
    return {"phash": imagehash.phash(im), "dhash": imagehash.dhash(im),
            "phash_m": imagehash.phash(m), "dhash_m": imagehash.dhash(m)}


def hex_to_hash(h: str) -> imagehash.ImageHash:
    return imagehash.hex_to_hash(h)
