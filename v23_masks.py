#!/usr/bin/env python3
"""V23 — precise alpha / stroke mask sets (patch plan §Patch 2).

The broad ``logo_fallback`` mask is fine for *confirming* a watermark footprint,
but far too coarse to drive product repair. V23 derives a set of precise masks
from the solved Sunsky alpha and the local high-pass response, and splits the
footprint into a product fragment and a background fragment so a repair can use
the right tool on each.

Hard rule (patch plan §Patch 2): a ``logo_fallback`` mask with a non-empty
product fragment may NEVER drive a destructive / semi-destructive fill — only
reverse-alpha or conservative stroke-only cleanup. ``destructive_allowed`` is
forced ``False`` in that case.

This module is additive and side-effect free; every builder degrades gracefully
to empty masks rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Area caps for the safe micro mask (patch plan §Patch 2.4).
MICRO_MAX_FOOTPRINT_FRAC = 0.20   # <= 20% of the alpha footprint
MICRO_MAX_BOX_FRAC = 0.04         # <= 4% of the mark box
MICRO_MAX_COMPONENT_FRAC = 0.015  # <= 1.5% of the mark box per component

# Alpha bands (mirror v22_patch AF_GLYPH_CORE / AF_HALO_LO).
ALPHA_CORE_MIN = 0.10
ALPHA_HALO_LO = 0.02

NEAR_WHITE_LUMA = 235.0


@dataclass
class V23MaskSet:
    alpha_core: np.ndarray            # alpha >= 0.10
    alpha_halo: np.ndarray            # 0.02 <= alpha < 0.10
    alpha_footprint: np.ndarray       # alpha >= 0.02 (full overlay footprint)
    detected_stroke: np.ndarray       # footprint & high-pass / CLAHE response
    safe_micro_mask: np.ndarray       # small components inside footprint (capped)
    background_fragment_mask: np.ndarray   # footprint pixels NOT product-like
    product_fragment_mask: np.ndarray      # footprint pixels product-like
    mask_type: str = "logo_fallback"
    confidence: float = 0.0
    destructive_allowed: bool = False
    footprint_area: int = 0

    def to_record(self) -> dict:
        return {
            "v23_mask_type": self.mask_type,
            "v23_mask_confidence": round(float(self.confidence), 4),
            "v23_destructive_allowed": bool(self.destructive_allowed),
            "v23_alpha_footprint_area": int(self.footprint_area),
            "v23_product_fragment_area": int((self.product_fragment_mask > 0).sum()),
            "v23_background_fragment_area":
                int((self.background_fragment_mask > 0).sum()),
            "v23_safe_micro_area": int((self.safe_micro_mask > 0).sum()),
        }


def _as_tuple(bbox):
    if isinstance(bbox, dict):
        return (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
    return tuple(int(v) for v in bbox)


def _empty_set(shape, mask_type="logo_fallback"):
    H, W = shape[:2]
    z = np.zeros((H, W), np.uint8)
    return V23MaskSet(
        alpha_core=z.copy(), alpha_halo=z.copy(), alpha_footprint=z.copy(),
        detected_stroke=z.copy(), safe_micro_mask=z.copy(),
        background_fragment_mask=z.copy(), product_fragment_mask=z.copy(),
        mask_type=mask_type, confidence=0.0, destructive_allowed=False,
        footprint_area=0)


def _alpha_buffer(image, bbox):
    """Full-image float32 alpha buffer from the solved Sunsky asset, or ``None``."""
    try:
        import sunsky_reverse_alpha as _sra
    except Exception:
        return None
    if not _sra.alpha_asset_available():
        return None
    asset = _sra.load_alpha_asset()
    if asset is None:
        return None
    bx, by, bw, bh = _as_tuple(bbox)
    H, W = image.shape[:2]
    if bw < 4 or bh < 4:
        return None
    crop = cv2.resize(asset["alpha"], (bw, bh), interpolation=cv2.INTER_AREA)
    buf = np.zeros((H, W), np.float32)
    x2, y2 = min(W, bx + bw), min(H, by + bh)
    bx, by = max(0, bx), max(0, by)
    if x2 <= bx or y2 <= by:
        return None
    buf[by:y2, bx:x2] = crop[: y2 - by, : x2 - bx]
    return buf


def _mask_type(image, bbox, watermark_mask) -> tuple:
    """Return ``(mask_type, coverage)``. Mirrors v18.stroke_mask_confidence
    without importing it (keeps this module light)."""
    bx, by, bw, bh = _as_tuple(bbox)
    H, W = image.shape[:2]
    box_area = float(max(1, bw * bh))
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        m = (watermark_mask > 0)
        in_box = int(m[by:by + bh, bx:bx + bw].sum())
        if in_box >= 12:
            cov = in_box / box_area
            return ("stroke" if cov < 0.45 else "widened_text", cov)
    return ("logo_fallback", 0.0)


def build_v23_mask_set(image, bbox, watermark_mask=None, product_mask=None,
                       roi_class="") -> V23MaskSet:
    """Build the V23 mask set for one watermark box.

    ``alpha_core`` / ``alpha_halo`` / ``alpha_footprint`` come from the solved
    Sunsky alpha (so they are smaller and more precise than the bounding box).
    The footprint is split into a product fragment and a background fragment, and
    a capped ``safe_micro_mask`` is computed for residue cleanup.
    """
    bx, by, bw, bh = _as_tuple(bbox)
    H, W = image.shape[:2]
    mask_type, coverage = _mask_type(image, bbox, watermark_mask)

    buf = _alpha_buffer(image, bbox)
    if buf is None:
        # No solved alpha: fall back to the watermark mask crop as footprint.
        ms = _empty_set(image.shape, mask_type)
        if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
            foot = (watermark_mask > 0).astype(np.uint8) * 255
            ms.alpha_footprint = foot
            ms.detected_stroke = foot.copy()
            ms.footprint_area = int((foot > 0).sum())
            _split_fragments(image, bbox, ms, product_mask)
        _resolve_destructive(ms)
        return ms

    core = (buf >= ALPHA_CORE_MIN)
    halo = (buf >= ALPHA_HALO_LO) & (buf < ALPHA_CORE_MIN)
    foot = (buf >= ALPHA_HALO_LO)

    ms = V23MaskSet(
        alpha_core=(core.astype(np.uint8) * 255),
        alpha_halo=(halo.astype(np.uint8) * 255),
        alpha_footprint=(foot.astype(np.uint8) * 255),
        detected_stroke=np.zeros((H, W), np.uint8),
        safe_micro_mask=np.zeros((H, W), np.uint8),
        background_fragment_mask=np.zeros((H, W), np.uint8),
        product_fragment_mask=np.zeros((H, W), np.uint8),
        mask_type=mask_type,
        confidence=float(np.clip(buf.max(), 0.0, 1.0)),
        footprint_area=int(foot.sum()))

    # detected_stroke: footprint intersected with the local high-pass / CLAHE
    # response (patch plan §Patch 2.3) — the pixels that actually carry the glyph.
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray).astype(np.float32)
        hp = np.abs(eq - cv2.GaussianBlur(eq, (0, 0), 2.0))
        resp = (hp > 6.0)
        ms.detected_stroke = ((foot & resp).astype(np.uint8) * 255)
    except Exception:
        ms.detected_stroke = ms.alpha_core.copy()

    _split_fragments(image, bbox, ms, product_mask)
    _build_safe_micro(image, bbox, ms)
    _resolve_destructive(ms)
    return ms


def _split_fragments(image, bbox, ms: V23MaskSet, product_mask):
    """Split the footprint into product-like and background fragments. The two
    are mutually exclusive by construction (patch plan §Patch 2.5/2.6)."""
    H, W = image.shape[:2]
    bx, by, bw, bh = _as_tuple(bbox)
    foot = (ms.alpha_footprint > 0)
    if not foot.any():
        return
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    product_like = (g < NEAR_WHITE_LUMA)
    if product_mask is not None and product_mask.shape[:2] == (H, W):
        product_like = product_like | (product_mask > 0)
    prod = foot & product_like
    bg = foot & (~product_like)
    ms.product_fragment_mask = (prod.astype(np.uint8) * 255)
    ms.background_fragment_mask = (bg.astype(np.uint8) * 255)


def _build_safe_micro(image, bbox, ms: V23MaskSet):
    """Small low-contrast components inside the footprint only, area-capped
    (patch plan §Patch 2.4)."""
    bx, by, bw, bh = _as_tuple(bbox)
    box_area = float(max(1, bw * bh))
    foot_area = float(max(1, ms.footprint_area))
    foot = (ms.alpha_footprint > 0).astype(np.uint8)
    if not foot.any():
        return
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0))
        cand = ((hp >= 3.0) & (hp <= 60.0) & (foot > 0)).astype(np.uint8)
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(cand, 8)
        keep = np.zeros_like(cand)
        total = 0
        for i in range(1, n):
            a = float(stats[i, cv2.CC_STAT_AREA])
            if a / box_area > MICRO_MAX_COMPONENT_FRAC:
                continue
            keep[_lab == i] = 255
            total += a
        # Enforce footprint and box caps on the aggregate.
        if total / foot_area <= MICRO_MAX_FOOTPRINT_FRAC and \
                total / box_area <= MICRO_MAX_BOX_FRAC:
            ms.safe_micro_mask = keep
    except Exception:
        pass


def _resolve_destructive(ms: V23MaskSet):
    """Hard rule: logo_fallback + non-empty product fragment => not allowed.
    A precise stroke mask on a pure-background footprint may allow it."""
    product_present = bool((ms.product_fragment_mask > 0).any())
    if ms.mask_type == "logo_fallback" and product_present:
        ms.destructive_allowed = False
        return
    ms.destructive_allowed = bool(
        ms.mask_type == "stroke" and not product_present)
