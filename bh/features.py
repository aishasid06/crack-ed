"""Listing features vs the brand's gold assets (SPEC §4 Features, §7.3).

Image: own-photo matching with three routes (thresholds tuned on the dev split only):
  1. pHash AND dHash both within threshold on the same gold variant (original, mirrored, centre crops)
  2. CLIP similarity to a gold photo
  3. ORB keypoint matches that survive a RANSAC homography (robust to crops and resizes)
wHash is deliberately not used (unreliable on white studio shots).

Text: exact mark, misspelt mark, referential-only use ("not X", "inspired by X"), keyword lists from
rules.yaml, product nouns taken from the brand's own product data. Commercial: price / RRP of the
matched product, origin (seller location + "imported from" wording), seller clusters.

All functions here are synchronous and CPU-bound; async callers use ``asyncio.to_thread``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

from bh.brand_index import BrandContext, ProductInfo
from bh.config import AppConfig
from bh.discovery.matching import BrandMatcher, normalize
from bh.vision import ImageEncoder, open_rgb

ORB_SIZE = 640
GOLD_COPY_MAX = 2  # pHash distance at which an image is an unmodified copy of a gold photo
_STOP = {"with", "and", "the", "for", "steel", "stainless", "water", "drinking", "insulated", "vacuum"}


# ---------------------------------------------------------------- own-photo matching


@dataclass
class _GoldVariant:
    product_id: int
    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash


@dataclass
class ImageMatch:
    """Best evidence that one listing image is a copy of a gold photo."""

    phash: int = 64  # smallest pHash over all gold variants
    pair_phash: int = 64  # (pHash, dHash) on the single gold variant where both are best together
    pair_dhash: int = 64
    clip: float = 0.0
    orb_inliers: int = 0
    product_hash: int | None = None
    product_clip: int | None = None
    product_orb: int | None = None


def _orb_ready(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L"))
    scale = ORB_SIZE / max(g.shape)
    if scale < 1:
        g = cv2.resize(g, (int(g.shape[1] * scale), int(g.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return g


class OwnPhotoMatcher:
    def __init__(self, ctx: BrandContext, cfg: AppConfig) -> None:
        self.ctx, self.cfg = ctx, cfg
        self.variants: list[_GoldVariant] = []
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.gold_orb: list[tuple[int, list[cv2.KeyPoint], np.ndarray]] = []
        for g in ctx.gold:
            im = open_rgb(g.path)
            for base in (im, ImageOps.mirror(im)):
                for v in [base, *(self._crop(base, f) for f in cfg.own_photo.crops)]:
                    self.variants.append(_GoldVariant(g.product_id, imagehash.phash(v), imagehash.dhash(v)))
                kp, des = self.orb.detectAndCompute(_orb_ready(base), None)
                if des is not None and len(kp) >= 8:
                    self.gold_orb.append((g.product_id, kp, des))

    @staticmethod
    def _crop(im: Image.Image, f: float) -> Image.Image:
        w, h = im.size
        dx, dy = int(w * (1 - f) / 2), int(h * (1 - f) / 2)
        return im.crop((dx, dy, w - dx, h - dy))

    def _orb_inliers(self, im: Image.Image) -> tuple[int, int | None]:
        kp, des = self.orb.detectAndCompute(_orb_ready(im), None)
        if des is None or len(kp) < 8:
            return 0, None
        best, pid = 0, None
        for product_id, gkp, gdes in self.gold_orb:
            pairs = self.bf.knnMatch(des, gdes, k=2)
            good = [m for m, *rest in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * rest[0].distance]
            if len(good) < 8:
                continue
            src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([gkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            n = int(mask.sum()) if mask is not None else 0
            if n > best:
                best, pid = n, product_id
        return best, pid

    def variant_distances(self, im: Image.Image) -> list[tuple[int, int]]:
        """(pHash, dHash) distance to every gold variant — used to tune the hash route."""
        ph, dh = imagehash.phash(im), imagehash.dhash(im)
        return [(int(ph - v.phash), int(dh - v.dhash)) for v in self.variants]

    def orb_inliers(self, im: Image.Image) -> int:
        return self._orb_inliers(im)[0]

    def match(self, im: Image.Image, vec: np.ndarray | None) -> ImageMatch:
        m = ImageMatch()
        ph, dh = imagehash.phash(im), imagehash.dhash(im)
        best = 10.0**6
        for v in self.variants:
            p, d = int(ph - v.phash), int(dh - v.dhash)
            m.phash = min(m.phash, p)
            score = max(p / max(1, self.cfg.own_photo.phash_max), d / max(1, self.cfg.own_photo.dhash_max))
            if score < best:
                best, m.pair_phash, m.pair_dhash, m.product_hash = score, p, d, v.product_id
        if vec is not None:
            m.clip, m.product_clip = self.ctx.nearest_clip(vec)
        m.orb_inliers, m.product_orb = self._orb_inliers(im)
        return m

    def routes(self, m: ImageMatch) -> list[str]:
        op = self.cfg.own_photo
        out = []
        if m.pair_phash <= op.phash_max and m.pair_dhash <= op.dhash_max:
            out.append("hash")
        if m.clip >= op.clip_min:
            out.append("clip")
        if m.orb_inliers >= op.orb_min_inliers:
            out.append("orb")
        return out


# ---------------------------------------------------------------- text


def phrase_hits(text: str, phrases: list[str]) -> list[str]:
    t = " " + text.lower() + " "
    hits = []
    for ph in phrases:
        p = ph.lower()
        if re.fullmatch(r"[a-z0-9 ]+", p.strip()):
            if re.search(rf"(?<![a-z0-9]){re.escape(p.strip())}(?![a-z0-9])", t):
                hits.append(ph.strip())
        elif p in t:
            hits.append(ph.strip())
    return hits


_REFERENTIAL_BEFORE = re.compile(
    r"(?:\bnot(?:\s+affiliated\s+with)?|\binspired\s+by|\bsimilar\s+to|\balternative\s+to|\blike|\bcompatible\s+with"
    r"|\bcompared\s+to|\bvs\.?)\s+(?:a\s+|an\s+|the\s+)?$")
_REFERENTIAL_AFTER = re.compile(r"^(?:-|\s)?(?:style|dupe|inspired|alternative)\b")


def mark_occurrences(text: str, marks: list[str]) -> tuple[int, int]:
    """(total, referential) counts of exact mark occurrences; referential = 'not X', 'inspired by X',
    'X-style', 'X dupe', ... (use of the mark to refer to the brand rather than as a sign for goods).
    Exact means case-insensitive only: accented or digit variants are misspellings, not the mark."""
    norm = text.lower()
    total = referential = 0
    for mk in {m.lower() for m in marks}:
        for mt in re.finditer(rf"(?<![a-z0-9]){re.escape(mk)}(?![a-z0-9])", norm):
            total += 1
            before, after = norm[max(0, mt.start() - 30): mt.start()], norm[mt.end(): mt.end() + 15]
            if _REFERENTIAL_BEFORE.search(before) or _REFERENTIAL_AFTER.search(after):
                referential += 1
    return total, referential


def goods_vocabulary(products: list[ProductInfo]) -> set[str]:
    words: set[str] = set()
    for p in products:
        for text in (p.short_name, p.generic_name or ""):
            for w in re.findall(r"[a-z]+", text.lower()):
                if len(w) >= 3 and w not in _STOP:
                    words.add(w)
    return words


# ---------------------------------------------------------------- origin


def _country_of(location: str | None) -> str | None:
    if not location:
        return None
    return location.split(",")[-1].strip() or None


def origin(location: str | None, text: str, cfg: AppConfig) -> tuple[bool | None, str]:
    """(outside_uk_eea, evidence). Text wording ('imported from Japan', 'USA edition') overrides location."""
    inside = {c.lower() for c in cfg.uk_eea_countries}
    outside = {c.lower() for c in cfg.text.non_uk_eea_countries}
    names = "|".join(sorted((re.escape(c) for c in outside), key=len, reverse=True))
    if names:
        m = re.search(rf"\b(?:imported\s+from|import\s+from|bought\s+from\s+an?)\s+({names})\b", text, re.I) or \
            re.search(rf"\b({names})\s+(?:version|edition|import|retail\s+packaging|retailer)\b", text, re.I)
        if m:
            return True, f"listing text: \"{m.group(0)}\""
    country = _country_of(location)
    if country is None:
        return None, "no seller location"
    if country.lower() in inside:
        return False, f"seller location {location} (UK/EEA)"
    if country.lower() in outside:
        return True, f"seller location {location} (outside UK/EEA)"
    return None, f"seller location {location} (country not recognised)"


# ---------------------------------------------------------------- listing features


@dataclass
class ListingInput:
    listing_id: int
    platform: str
    title: str
    description: str
    price: float | None
    currency: str | None
    seller_name: str | None
    seller_location: str | None
    seller_age_days: int | None
    rating: float | None
    quantity: int | None
    image_paths: list[Path]


@dataclass
class ListingFeatures:
    listing_id: int
    matched_product_id: int | None = None
    matched_by: str | None = None
    hash_dist: int | None = None
    dhash_dist: int | None = None
    clip_sim: float | None = None
    orb_inliers: int | None = None
    own_photo: bool | None = None
    own_photo_routes: list[str] = field(default_factory=list)
    price_ratio: float | None = None
    authorised: bool = False
    brand_in_text: bool = False
    mark_referential_only: bool = False
    misspelt_brand: bool = False
    misspellings: list[str] = field(default_factory=list)
    brand_score: float = 0.0
    replica_kw: list[str] = field(default_factory=list)
    inspired_kw: list[str] = field(default_factory=list)
    modification_kw: list[str] = field(default_factory=list)
    official_kw: list[str] = field(default_factory=list)
    goods_terms: list[str] = field(default_factory=list)
    origin_outside_uk_eea: bool | None = None
    origin_evidence: str = ""
    seller_age_days: int | None = None
    quantity: int | None = None
    bulk_quantity: bool = False
    rating: float | None = None
    image_count: int = 0
    seller_cluster: str | None = None
    cluster_size: int = 1
    cluster_suspect: bool = False
    image_hashes: list[str] = field(default_factory=list, repr=False)

    def counterfeit_indicator(self, counterfeit_price_ratio: float) -> bool:
        return bool(self.replica_kw) or self.misspelt_brand or (
            self.price_ratio is not None and self.price_ratio < counterfeit_price_ratio)


class FeatureExtractor:
    def __init__(self, ctx: BrandContext, cfg: AppConfig, encoder: ImageEncoder,
                 vocabulary: dict[str, list[str]]) -> None:
        self.ctx, self.cfg, self.encoder, self.vocab = ctx, cfg, encoder, vocabulary
        self.photos = OwnPhotoMatcher(ctx, cfg)
        self.matcher = BrandMatcher([*ctx.marks, ctx.name])
        self.goods_words = goods_vocabulary(ctx.products)

    def _text_product(self, text: str) -> ProductInfo | None:
        t = normalize(text)
        best, prod = 0, None
        for p in self.ctx.products:
            words = [w for w in re.findall(r"[a-z]+", p.short_name.lower()) if len(w) >= 3 and w not in _STOP]
            hit = sum(bool(re.search(rf"\b{re.escape(w)}\b", t)) for w in words)
            if words and hit / len(words) >= 0.67 and hit > best:
                best, prod = hit, p
        return prod

    def extract(self, li: ListingInput) -> ListingFeatures:
        f = ListingFeatures(li.listing_id, seller_age_days=li.seller_age_days, quantity=li.quantity, rating=li.rating)
        text = f"{li.title}\n{li.description}"
        th = self.cfg.thresholds

        # images
        imgs = []
        for p in li.image_paths:
            try:
                imgs.append(open_rgb(p))
            except (OSError, ValueError):
                continue
        f.image_count = len(imgs)
        if imgs:
            vecs = self.encoder.encode(imgs)
            routes: set[str] = set()
            votes: dict[int, float] = {}
            best_clip, best_hash, best_d, best_orb = 0.0, 64, 64, 0
            for im, v in zip(imgs, vecs, strict=True):
                m = self.photos.match(im, v)
                r = self.photos.routes(m)
                routes.update(r)
                best_clip, best_hash = max(best_clip, m.clip), min(best_hash, m.phash)
                best_d, best_orb = min(best_d, m.pair_dhash), max(best_orb, m.orb_inliers)
                for pid, w in ((m.product_orb, 3.0 if "orb" in r else 0), (m.product_hash, 2.0 if "hash" in r else 0),
                               (m.product_clip, m.clip)):
                    if pid is not None:
                        votes[pid] = votes.get(pid, 0) + w
                if m.phash > GOLD_COPY_MAX:  # unmodified brand photos are shared legitimately; not a seller link
                    f.image_hashes.append(str(imagehash.phash(im)))
            f.clip_sim, f.hash_dist, f.dhash_dist, f.orb_inliers = best_clip, best_hash, best_d, best_orb
            f.own_photo_routes = sorted(routes)
            f.own_photo = bool(routes)
            if f.own_photo or best_clip >= th.lookalike_clip:
                f.matched_product_id = max(votes, key=votes.get) if votes else None
                f.matched_by = "own_photo" if f.own_photo else "clip"

        # text
        total, referential = mark_occurrences(text, [*self.ctx.marks, self.ctx.name])
        f.brand_in_text = total > referential
        f.mark_referential_only = total > 0 and total == referential
        f.brand_score = self.matcher.score(text)
        exact = {m.lower() for m in [*self.ctx.marks, self.ctx.name]}
        f.misspellings = sorted({w for w in (re.sub(r"^\W+|\W+$", "", tok) for tok in re.findall(r"\S+", text))
                                 if w and w.lower() not in exact and not any(w.lower().startswith(e + "-") for e in exact)
                                 and self.matcher.score(w) >= self.cfg.text.misspelling_min_score})
        f.misspelt_brand = bool(f.misspellings)
        f.replica_kw = phrase_hits(text, self.vocab.get("replica", []))
        f.inspired_kw = phrase_hits(text, self.vocab.get("inspired", []))
        f.modification_kw = phrase_hits(text, self.vocab.get("modification", []))
        f.official_kw = phrase_hits(text, self.vocab.get("official", []))
        f.goods_terms = sorted(w for w in self.goods_words if re.search(rf"\b{re.escape(w)}\b", text.lower()))
        if f.matched_product_id is None and (prod := self._text_product(text)):
            f.matched_product_id, f.matched_by = prod.id, "text"

        # commercial / seller
        prod = self.ctx.product(f.matched_product_id)
        if prod and li.price is not None and (li.currency or "GBP").upper() == "GBP":
            f.price_ratio = round(li.price / prod.rrp, 4)
        f.authorised = self.ctx.is_authorised(li.platform, li.seller_name)
        f.origin_outside_uk_eea, f.origin_evidence = origin(li.seller_location, text, self.cfg)
        f.bulk_quantity = (li.quantity or 0) >= th.bulk_quantity
        f.seller_cluster = re.sub(r"[^a-z0-9]", "", (li.seller_name or "").lower()) or None
        return f


def assign_clusters(feats: list[ListingFeatures], cfg: AppConfig, max_hash: int = 2) -> None:
    """Union sellers with the same normalised name or sharing (near-)identical *seller-specific* images
    (unmodified copies of the brand's photos are excluded — resellers share those legitimately); flag
    clusters with ≥ bulk_listing_threshold listings carrying counterfeit indicators."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    keyed = [(f, f.seller_cluster or f"listing-{f.listing_id}") for f in feats]
    for _, k in keyed:
        find(k)
    hashes = [(k, imagehash.hex_to_hash(h)) for f, k in keyed for h in f.image_hashes]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i][0] != hashes[j][0] and hashes[i][1] - hashes[j][1] <= max_hash:
                union(hashes[i][0], hashes[j][0])
    groups: dict[str, list[ListingFeatures]] = {}
    for f, k in keyed:
        groups.setdefault(find(k), []).append(f)
    for root, members in groups.items():
        suspect = sum(m.counterfeit_indicator(cfg.thresholds.counterfeit_price_ratio) for m in members) \
            >= cfg.thresholds.bulk_listing_threshold
        for m in members:
            m.seller_cluster, m.cluster_size, m.cluster_suspect = root, len(members), suspect


def vector(f: ListingFeatures, counterfeit_price_ratio: float) -> list[float]:
    """Numeric feature vector for the calibrated scorer (order = FEATURE_NAMES)."""
    pr = f.price_ratio
    return [
        f.clip_sim or 0.0, float(bool(f.own_photo)), 1.0 - min(f.hash_dist if f.hash_dist is not None else 32, 32) / 32,
        min(f.orb_inliers or 0, 100) / 100, float(f.brand_in_text), float(f.mark_referential_only),
        float(f.misspelt_brand), float(bool(f.replica_kw)), float(bool(f.inspired_kw)),
        float(bool(f.modification_kw)), float(bool(f.official_kw)), min(pr, 2.0) if pr is not None else 1.0,
        float(pr is None), float(pr is not None and pr < counterfeit_price_ratio), float(f.authorised),
        float(f.origin_outside_uk_eea is True), float(f.origin_outside_uk_eea is None),
        math.log1p(f.seller_age_days or 0) / 8, math.log1p(f.quantity or 0) / 7, float(f.cluster_suspect),
        float(f.bulk_quantity), (f.rating if f.rating is not None else 4.5) / 5, float(bool(f.goods_terms)),
    ]


FEATURE_NAMES = ["clip_sim", "own_photo", "hash_close", "orb", "brand_in_text", "mark_referential_only",
                 "misspelt_brand", "replica_kw", "inspired_kw", "modification_kw", "official_kw", "price_ratio",
                 "price_missing", "price_below_counterfeit_ratio", "authorised", "origin_outside", "origin_unknown",
                 "log_seller_age", "log_quantity", "cluster_suspect", "bulk_quantity", "rating", "goods_terms"]
