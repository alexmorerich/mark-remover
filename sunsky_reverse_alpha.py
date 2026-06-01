#!/usr/bin/env python3
"""V19 — Sunsky reverse-alpha recovery engine.

The ``sunsky-online.com`` watermark is a *fixed, semi-transparent* visible text
overlay. Every watermarked pixel is an alpha blend of the overlay (a fixed logo
colour) and the real product pixel underneath::

    watermarked = alpha * logo + (1 - alpha) * original

Given a solved per-pixel ``alpha`` map and the ``logo`` colour, the original
pixel can be *recovered* by inverting the blend::

    original = (watermarked - alpha * logo) / (1 - alpha)

This is fundamentally different from inpainting or covering: it does NOT paint a
rectangle, clone a neighbour, or hallucinate a fill. It *subtracts the overlay*
and keeps the real underlying product pixels. That is exactly why it is allowed
to run where destructive generators are banned — over product text, flex cables,
dark surfaces and metallic gradients — because it preserves whatever was beneath
the mark instead of replacing it.

Borrowed from the ``remove-ai-watermarks`` Doubao/Jimeng visible-watermark
engines (patch plan §1.2, §1.4):
  * a solved alpha asset (``assets/sunsky_alpha.png`` + ``sunsky_alpha_meta.json``)
  * fixed placement AND NCC-aligned placement, keep the lower-residual one
  * thin residual cleanup over ONLY the glyph footprint (never a full bbox)

This module is purely *additive*: every public entry point returns ``None`` /
``pass_local_safety=False`` on any doubt or missing asset, so the caller simply
falls back to the existing V18 candidates. Nothing here can ship an output the
V17/V19 final audit would reject — every reverse-alpha candidate still passes
through the unchanged P0 gate in the pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "assets"
ALPHA_PNG = ASSET_DIR / "sunsky_alpha.png"
ALPHA_META = ASSET_DIR / "sunsky_alpha_meta.json"

# Inversion guards (patch plan §4.2).
MIN_ALPHA = 0.002            # below this the overlay is negligible — leave pixel
MAX_ALPHA = 0.85             # clamp so the denominator never collapses
DENOM_FLOOR = 0.25           # (1 - alpha) floor; recovery is unstable beyond this

# NCC alignment search window (patch plan §1.4).
SCALE_LO, SCALE_HI, SCALE_STEPS = 0.88, 1.12, 7
X_SHIFT, Y_SHIFT = 6, 4      # px search window around the detected box

# Thin residual cleanup (patch plan §1.5).
RESIDUAL_DILATE_PX = 2
RESIDUAL_INPAINT_RADIUS = 2

# Local safety thresholds (patch plan §4.1). These are a CHEAP pre-screen; the
# authoritative gate is still the V17/V19 final audit in the pipeline.
MAX_CHANGED_PRODUCT_RATIO = 0.92   # reverse-alpha may touch product, but not all
DARK_BLOB_MAX = 26.0               # luma jump on dark stock that reads as a blob
RESIDUAL_IMPROVE_MIN = 0.04        # residual confidence must drop by at least this


# ---------------------------------------------------------------------------
# Alpha asset (patch plan §1.3). Loaded once and cached.
# ---------------------------------------------------------------------------
_ASSET_CACHE: Optional[dict] = None


def load_alpha_asset(force: bool = False) -> Optional[dict]:
    """Load the solved alpha asset. Returns a dict with keys ``alpha`` (float32
    HxW in [0, 1]), ``logo_bgr`` (tuple) and ``meta`` (dict), or ``None`` when
    the asset is missing / unreadable / empty (patch plan §6 — missing-asset
    failure is graceful)."""
    global _ASSET_CACHE
    if _ASSET_CACHE is not None and not force:
        return _ASSET_CACHE
    if not ALPHA_PNG.exists():
        return None
    raw = cv2.imread(str(ALPHA_PNG), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.size == 0:
        return None
    alpha = raw.astype(np.float32) / 255.0
    # An all-zero / near-empty map is not a usable asset.
    if float(alpha.max()) < 0.02 or int((alpha > 0.05).sum()) < 8:
        return None
    meta = {}
    if ALPHA_META.exists():
        try:
            meta = json.loads(ALPHA_META.read_text())
        except Exception:
            meta = {}
    logo = meta.get("logo_bgr") or [200.0, 200.0, 200.0]
    logo_bgr = (float(logo[0]), float(logo[1]), float(logo[2]))
    _ASSET_CACHE = {"alpha": alpha, "logo_bgr": logo_bgr, "meta": meta}
    return _ASSET_CACHE


def alpha_asset_available() -> bool:
    return load_alpha_asset() is not None


# ---------------------------------------------------------------------------
# Placement / result dataclasses (patch plan §4.1).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlphaPlacement:
    alpha_map: np.ndarray                 # cropped alpha (float32 [0,1]) at bbox size
    logo_bgr: Tuple[float, float, float]
    bbox: Tuple[int, int, int, int]       # (x, y, w, h) where it applies
    source: str                           # "fixed" | "aligned"
    ncc_score: float
    scale: float
    x_offset: int
    y_offset: int


@dataclass
class ReverseAlphaResult:
    image: Optional[np.ndarray] = None
    placement: Optional[AlphaPlacement] = None
    residual_confidence_before_cleanup: float = 1.0
    residual_confidence_after_cleanup: float = 1.0
    residual_confidence_original: float = 1.0
    changed_product_ratio: float = 0.0
    product_damage_score: float = 0.0
    protected_text_overlap: float = 0.0
    thin_residual_cleanup_used: bool = False
    thin_residual_cleanup_mask_overlap_text: float = 0.0
    pass_local_safety: bool = False
    source: str = "none"
    reject_reason: Optional[str] = None

    def to_record(self) -> dict:
        return {
            "reverse_alpha_attempted": True,
            "reverse_alpha_passed": bool(self.pass_local_safety),
            "reverse_alpha_source": self.source,
            "reverse_alpha_ncc": round(
                float(self.placement.ncc_score) if self.placement else 0.0, 4),
            "reverse_alpha_residual_before": round(
                self.residual_confidence_before_cleanup, 4),
            "reverse_alpha_residual_after": round(
                self.residual_confidence_after_cleanup, 4),
            "reverse_alpha_reject_reason": self.reject_reason or "",
            "alpha_bbox": list(self.placement.bbox) if self.placement else None,
            "alpha_scale": round(float(self.placement.scale), 4)
                if self.placement else 1.0,
            "thin_residual_cleanup_used": bool(self.thin_residual_cleanup_used),
            "thin_residual_cleanup_mask_overlap_text": round(
                self.thin_residual_cleanup_mask_overlap_text, 4),
        }


# ---------------------------------------------------------------------------
# Vectorised safe alpha inversion (patch plan §4.2).
# ---------------------------------------------------------------------------
def apply_reverse_alpha(image, alpha, logo_bgr, min_alpha=MIN_ALPHA,
                        max_alpha=MAX_ALPHA, denominator_floor=DENOM_FLOOR):
    """Invert the alpha blend over the masked pixels ONLY. ``alpha`` is a full
    image-sized float map in [0, 1]; pixels with ``alpha < min_alpha`` are left
    byte-identical (patch plan §1.5: never touch pixels outside the alpha mask).
    """
    a = np.clip(alpha, 0.0, max_alpha).astype(np.float32)
    mask = a >= min_alpha
    a3 = a[:, :, None]
    logo = np.array(logo_bgr, np.float32).reshape(1, 1, 3)
    restored = (image.astype(np.float32) - a3 * logo) / \
        np.clip(1.0 - a3, denominator_floor, 1.0)
    restored = np.clip(restored, 0, 255)
    out = image.copy()
    out[mask] = restored[mask].astype(np.uint8)
    return out


def _full_alpha_buffer(shape, placement: AlphaPlacement) -> np.ndarray:
    """Place the cropped alpha map into a full-image float buffer at its bbox."""
    H, W = shape[:2]
    buf = np.zeros((H, W), np.float32)
    bx, by, bw, bh = placement.bbox
    bx2, by2 = min(W, bx + bw), min(H, by + bh)
    bx, by = max(0, bx), max(0, by)
    if bx2 <= bx or by2 <= by:
        return buf
    crop = placement.alpha_map[: by2 - by, : bx2 - bx]
    buf[by:by2, bx:bx2] = crop
    return buf


def apply_placement(image, placement: AlphaPlacement) -> np.ndarray:
    buf = _full_alpha_buffer(image.shape, placement)
    return apply_reverse_alpha(image, buf, placement.logo_bgr)


# ---------------------------------------------------------------------------
# Residual confidence — how watermark-like is the structure left in the band.
# A cheap NCC of the glyph alpha shape against the local high-pass response;
# higher = more residual overlay structure remaining (patch plan §4.1 step 5).
# ---------------------------------------------------------------------------
def _local_contrast(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    hp = np.abs(g - blur)
    m = float(hp.max())
    return hp / m if m > 1e-6 else hp


def residual_confidence(image, bbox, alpha_crop) -> float:
    """Correlation between the glyph alpha shape and the high-pass response of
    the band. Returns 0..1 (higher = stronger residual)."""
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    bx, by = max(0, bx), max(0, by)
    bw, bh = min(bw, W - bx), min(bh, H - by)
    if bw < 4 or bh < 4:
        return 0.0
    band = image[by:by + bh, bx:bx + bw]
    if band.ndim == 3:
        band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hp = _local_contrast(band)
    a = cv2.resize(alpha_crop, (bw, bh), interpolation=cv2.INTER_AREA)
    a = a / (a.max() + 1e-6)
    aw = float((a > 0.05).sum())
    if aw < 4:
        return 0.0
    # mean high-pass energy inside the glyph footprint vs the whole band.
    inside = float((hp * (a > 0.05)).sum()) / aw
    overall = float(hp.mean()) + 1e-6
    return float(np.clip(inside / (overall * 3.0), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Placement generation (patch plan §1.4, §4.1).
# ---------------------------------------------------------------------------
def fixed_alpha_map(image, mark_box, asset=None) -> Optional[AlphaPlacement]:
    """Resize the canonical alpha asset to the detected box and place it there
    with no shift (patch plan: try fixed geometry first)."""
    asset = asset or load_alpha_asset()
    if asset is None:
        return None
    bx, by, bw, bh = mark_box
    if bw < 4 or bh < 4:
        return None
    crop = cv2.resize(asset["alpha"], (bw, bh), interpolation=cv2.INTER_AREA)
    ncc = residual_confidence(image, (bx, by, bw, bh), crop)
    return AlphaPlacement(alpha_map=crop, logo_bgr=asset["logo_bgr"],
                          bbox=(bx, by, bw, bh), source="fixed",
                          ncc_score=ncc, scale=1.0, x_offset=0, y_offset=0)


def aligned_alpha_map(image, mark_box, ctx=None, asset=None) \
        -> Optional[AlphaPlacement]:
    """Search scale / x-shift / y-shift for the alpha placement whose glyph shape
    best correlates with the watermark structure in the band (TM_CCOEFF_NORMED
    over the center band). Mirrors the Doubao/Jimeng aligned-placement step."""
    asset = asset or load_alpha_asset()
    if asset is None:
        return None
    bx, by, bw, bh = mark_box
    if bw < 8 or bh < 6:
        return None
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hp = _local_contrast(gray)

    best = None
    best_score = -1.0
    scales = np.linspace(SCALE_LO, SCALE_HI, SCALE_STEPS)
    for sc in scales:
        sw, sh = int(round(bw * sc)), int(round(bh * sc))
        if sw < 6 or sh < 4 or sw >= W or sh >= H:
            continue
        glyph = cv2.resize(asset["alpha"], (sw, sh), interpolation=cv2.INTER_AREA)
        glyph = (glyph / (glyph.max() + 1e-6)).astype(np.float32)
        # Search a small window around the centered fixed position.
        cx = bx + (bw - sw) // 2
        cy = by + (bh - sh) // 2
        for dx in range(-X_SHIFT, X_SHIFT + 1, 2):
            for dy in range(-Y_SHIFT, Y_SHIFT + 1, 2):
                px, py = cx + dx, cy + dy
                if px < 0 or py < 0 or px + sw > W or py + sh > H:
                    continue
                region = hp[py:py + sh, px:px + sw]
                if region.shape != glyph.shape:
                    continue
                # Normalised correlation between glyph shape and high-pass band.
                rv = float((region * glyph).sum())
                norm = float(np.sqrt((region ** 2).sum()) *
                             np.sqrt((glyph ** 2).sum())) + 1e-6
                score = rv / norm
                if score > best_score:
                    best_score = score
                    best = (glyph, (px, py, sw, sh), float(sc), dx, dy)
    if best is None:
        return None
    glyph, box, sc, dx, dy = best
    # Restore the asset's true alpha amplitude (we normalised it for matching).
    glyph_alpha = glyph * float(asset["alpha"].max())
    ncc = residual_confidence(image, box, glyph_alpha)
    return AlphaPlacement(alpha_map=glyph_alpha, logo_bgr=asset["logo_bgr"],
                          bbox=box, source="aligned",
                          ncc_score=float(np.clip(best_score, 0.0, 1.0)),
                          scale=sc, x_offset=dx, y_offset=dy)


# ---------------------------------------------------------------------------
# Thin residual cleanup over ONLY the glyph footprint (patch plan §1.5). Never
# expands into a full-band / full-bbox fill, and skips protected product text.
# ---------------------------------------------------------------------------
def _glyph_residual_mask(shape, placement: AlphaPlacement,
                         dilate_px=RESIDUAL_DILATE_PX) -> np.ndarray:
    buf = _full_alpha_buffer(shape, placement)
    m = (buf > 0.05).astype(np.uint8)
    if dilate_px > 0:
        k = np.ones((dilate_px * 2 + 1,) * 2, np.uint8)
        m = cv2.dilate(m, k)
    return (m * 255).astype(np.uint8)


def thin_residual_cleanup(image, placement, *, protected_text_mask=None,
                          flex_mask=None, dilate_px=RESIDUAL_DILATE_PX,
                          method=cv2.INPAINT_NS):
    """Run a thin inpaint over the dilated glyph footprint only. Returns
    ``(cleaned_image, mask_overlap_text)``. Where the footprint overlaps
    protected product text, those pixels are removed from the cleanup mask (patch
    plan §1.5 restriction: never destructively inpaint over product labels).

    ``dilate_px`` may be widened ONLY by the caller on a provably pure-background
    region (no product overlap), where a slightly wider, texture-preserving
    inpaint fully clears the watermark footprint without any product risk."""
    mask = _glyph_residual_mask(image.shape, placement, dilate_px=dilate_px)
    overlap_text = 0.0
    total = float((mask > 0).sum()) + 1e-6
    if protected_text_mask is not None and \
            protected_text_mask.shape[:2] == image.shape[:2]:
        pt = (protected_text_mask > 0)
        overlap_text = float((pt & (mask > 0)).sum()) / total
        mask[pt] = 0
    if flex_mask is not None and flex_mask.shape[:2] == image.shape[:2]:
        # On flex cables, keep the cleanup stroke-thin (no dilation bleed).
        mask[(flex_mask > 0)] = 0
    if int((mask > 0).sum()) < 4:
        return image, overlap_text
    cleaned = cv2.inpaint(image, mask, RESIDUAL_INPAINT_RADIUS, method)
    return cleaned, overlap_text


# ---------------------------------------------------------------------------
# Local safety probes.
# ---------------------------------------------------------------------------
def _changed_product_ratio(original, candidate, bbox, product_mask) -> float:
    bx, by, bw, bh = bbox
    o = original[by:by + bh, bx:bx + bw].astype(np.float32)
    c = candidate[by:by + bh, bx:bx + bw].astype(np.float32)
    if o.size == 0 or o.shape != c.shape:
        return 0.0
    changed = (np.abs(o - c).max(axis=2) > 5)
    n = int(changed.sum())
    if n < 4:
        return 0.0
    if product_mask is not None and product_mask.shape[:2] == original.shape[:2]:
        pm = product_mask[by:by + bh, bx:bx + bw] > 0
        return float((changed & pm).sum()) / float(n)
    return 0.0


def _dark_blob(original, candidate, bbox) -> float:
    bx, by, bw, bh = bbox
    o = original[by:by + bh, bx:bx + bw]
    c = candidate[by:by + bh, bx:bx + bw]
    if o.size == 0 or o.shape != c.shape:
        return 0.0
    og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = og < 80
    if int(dark.sum()) < 20:
        return 0.0
    return float(np.abs(cg[dark] - og[dark]).mean())


# ---------------------------------------------------------------------------
# Main entry (patch plan §4.1). Build fixed + aligned placements, apply, keep
# the lower-residual one, optionally clean thin residue, verify local safety.
# ---------------------------------------------------------------------------
def repair_sunsky_reverse_alpha(image, mark_box, *, watermark_mask=None,
                                product_mask=None, protected_text_mask=None,
                                ctx=None, allow_thin_cleanup=True,
                                experimental_estimated_alpha=False) \
        -> ReverseAlphaResult:
    """Recover the product pixels under the Sunsky watermark by inverting the
    alpha blend. Returns a :class:`ReverseAlphaResult`; ``pass_local_safety`` is
    True only when the candidate cleared every cheap local check. The pipeline
    still runs the authoritative V17/V19 audit on the bytes."""
    res = ReverseAlphaResult()
    if isinstance(mark_box, dict):
        mark_box = (mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"])
    bx, by, bw, bh = mark_box

    asset = load_alpha_asset()
    if asset is None and not experimental_estimated_alpha:
        res.reject_reason = "alpha_asset_missing"
        return res
    if asset is None:
        res.reject_reason = "alpha_asset_missing"
        return res
    if bw < 6 or bh < 4:
        res.reject_reason = "box_too_small"
        return res

    orig_residual = residual_confidence(
        image, (bx, by, bw, bh),
        cv2.resize(asset["alpha"], (bw, bh), interpolation=cv2.INTER_AREA))
    res.residual_confidence_original = orig_residual

    placements = []
    fp = fixed_alpha_map(image, (bx, by, bw, bh), asset)
    if fp is not None:
        placements.append(fp)
    ap = aligned_alpha_map(image, (bx, by, bw, bh), ctx, asset)
    if ap is not None:
        placements.append(ap)
    if not placements:
        res.reject_reason = "no_placement"
        return res

    # Apply each placement, score residual on the result, keep the lowest.
    best = None
    best_resid = 2.0
    for pl in placements:
        cand = apply_placement(image, pl)
        rc = residual_confidence(cand, pl.bbox, pl.alpha_map)
        if rc < best_resid:
            best_resid = rc
            best = (pl, cand, rc)
    placement, candidate, resid_before = best
    res.placement = placement
    res.source = placement.source
    res.residual_confidence_before_cleanup = resid_before

    # Is this a provably pure-background box (no product overlap)? On background
    # the residual cleanup may be wider and texture-preserving without any
    # product risk — this is what clears the watermark footprint the narrow
    # alpha asset under-covers on textured backgrounds.
    ctx_overlap = float(getattr(ctx, "product_overlap", 1.0)) if ctx else 1.0
    measured_overlap = _changed_product_ratio(
        image, candidate, placement.bbox, product_mask)
    pure_background = (ctx_overlap < 0.03 and measured_overlap < 0.02 and
                       (product_mask is None or
                        float((product_mask[by:by + bh, bx:bx + bw] > 0).mean()
                              if (by + bh <= image.shape[0] and
                                  bx + bw <= image.shape[1]) else 1.0) < 0.03))

    # Thin residual cleanup over the glyph footprint only.
    resid_after = resid_before
    if allow_thin_cleanup:
        flex_mask = None
        dpx = 4 if pure_background else RESIDUAL_DILATE_PX
        meth = cv2.INPAINT_TELEA if pure_background else cv2.INPAINT_NS
        cleaned, overlap_text = thin_residual_cleanup(
            candidate, placement, protected_text_mask=protected_text_mask,
            flex_mask=flex_mask, dilate_px=dpx, method=meth)
        rc2 = residual_confidence(cleaned, placement.bbox, placement.alpha_map)
        if rc2 <= resid_before + 1e-6:
            candidate = cleaned
            resid_after = rc2
            res.thin_residual_cleanup_used = True
            res.thin_residual_cleanup_mask_overlap_text = overlap_text
    res.residual_confidence_after_cleanup = resid_after
    res.image = candidate

    # ---- Local safety (cheap pre-screen; the audit is authoritative). -------
    res.changed_product_ratio = _changed_product_ratio(
        image, candidate, placement.bbox, product_mask)
    res.product_damage_score = _dark_blob(image, candidate, placement.bbox)
    try:
        import v17_final_audit
        res.protected_text_overlap = float(
            v17_final_audit._protected_text_overlap(
                image, placement.bbox, watermark_mask))
    except Exception:
        res.protected_text_overlap = 0.0

    reasons = []
    if res.changed_product_ratio > MAX_CHANGED_PRODUCT_RATIO:
        reasons.append("changed_too_much_product")
    if res.product_damage_score > DARK_BLOB_MAX:
        reasons.append("dark_surface_blob")
    # Residual must actually improve vs the untouched original band.
    if resid_after > orig_residual - RESIDUAL_IMPROVE_MIN and \
            orig_residual > 0.05:
        reasons.append("residual_did_not_improve")
    res.pass_local_safety = (len(reasons) == 0)
    res.reject_reason = None if res.pass_local_safety else ",".join(reasons)
    return res
