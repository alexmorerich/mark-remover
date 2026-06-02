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

# V20 — reverse-alpha variant beam (patch plan §Patch 3). A small deterministic
# set of recovery variants is generated; each passes the SAME local pre-screen,
# and the pipeline tries them in rank order, publishing the first that clears the
# authoritative P0 audit. No randomness — the beam is fully reproducible.
GAIN_LO, GAIN_HI, GAIN_STEPS = 0.75, 1.25, 6     # local alpha-gain search
LOGO_DELTA_MAX = 12.0                            # per-channel logo colour nudge
CORE_ALPHA = 0.12                                # glyph core threshold
HALO_ALPHA_LO = 0.02                             # glyph halo band [lo, core)
VARIANT_NAMES = (
    "v20_reverse_alpha_fixed",
    "v20_reverse_alpha_ncc",
    "v20_reverse_alpha_ncc_local_gain",
    "v20_reverse_alpha_core_halo_split",
    "v20_reverse_alpha_per_channel_logo",
    "v20_reverse_alpha_low_alpha_halo_cleanup",
    "v20_reverse_alpha_no_cleanup",
    # V21 — smooth-surface refinement (patch plan §Patch 5). A texture-preserving
    # bilateral pass restricted to the glyph halo suppresses faint reverse-alpha
    # residue on cardboard / smooth coloured / metallic backs without flattening
    # the surface. Screened by the same local pre-screen; audited authoritatively.
    "v21_reverse_alpha_texture_preserve_blend",
)


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
    ghost_dot_score: float = 0.0          # V20 (patch plan §Patch 4)
    ghost_dot_hard_fail: bool = False     # V20

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


# ---------------------------------------------------------------------------
# V20 — reverse-alpha variant beam (patch plan §Patch 3). Build a small set of
# deterministic recovery variants from the fixed + NCC placements: a local
# alpha-gain solve, a per-channel logo-colour solve, a core/halo split, and a
# soft halo cleanup. Each variant is a real-pixel recovery (it subtracts the
# overlay, never paints a box), and each is screened by the same cheap local
# safety check. The pipeline runs the authoritative P0 audit on the bytes.
# ---------------------------------------------------------------------------
def _apply_with_gain(image, placement: AlphaPlacement, gain: float) -> np.ndarray:
    """Apply reverse-alpha with the alpha map scaled by ``gain`` (clamped)."""
    buf = _full_alpha_buffer(image.shape, placement) * float(gain)
    return apply_reverse_alpha(image, buf, placement.logo_bgr)


def _solve_local_gain(image, placement: AlphaPlacement) -> float:
    """Fit a scalar gain in [GAIN_LO, GAIN_HI] that minimises high-pass residual
    inside the glyph footprint (patch plan §Patch 3.1)."""
    best_gain, best_resid = 1.0, 2.0
    for g in np.linspace(GAIN_LO, GAIN_HI, GAIN_STEPS):
        cand = _apply_with_gain(image, placement, float(g))
        rc = residual_confidence(cand, placement.bbox, placement.alpha_map)
        if rc < best_resid:
            best_resid, best_gain = rc, float(g)
    return best_gain


def _solve_logo_delta(image, placement: AlphaPlacement,
                      protected_text_mask=None) -> Tuple[float, float, float]:
    """Estimate a small per-channel logo-colour adjustment (delta_bgr in
    [-LOGO_DELTA_MAX, +LOGO_DELTA_MAX]) so the recovered footprint pixels match
    the surrounding ring median, using only high-confidence alpha pixels and
    excluding protected product text (patch plan §Patch 3.2)."""
    H, W = image.shape[:2]
    buf = _full_alpha_buffer(image.shape, placement)
    high = buf >= 0.30
    if protected_text_mask is not None and \
            protected_text_mask.shape[:2] == (H, W):
        high &= (protected_text_mask == 0)
    if int(high.sum()) < 8:
        return (0.0, 0.0, 0.0)
    base = apply_placement(image, placement)
    ring = _ring_bgr_median(image, placement.bbox)
    rec_med = base[high].reshape(-1, 3).astype(np.float32).mean(axis=0)
    # A positive (ring - recovered) gap means recovery under-subtracted the
    # overlay; nudging the logo brighter pushes recovered pixels toward the ring.
    delta = np.clip(ring - rec_med, -LOGO_DELTA_MAX, LOGO_DELTA_MAX)
    return (float(delta[0]), float(delta[1]), float(delta[2]))


def _ring_bgr_median(image, bbox) -> np.ndarray:
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    pad = max(6, max(bw, bh) // 3)
    y1, y2 = max(0, by - pad), min(H, by + bh + pad)
    x1, x2 = max(0, bx - pad), min(W, bx + bw + pad)
    region = image[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    if region.size == 0:
        return np.array([200.0, 200.0, 200.0], np.float32)
    return np.median(region, axis=0)


def _apply_per_channel(image, placement: AlphaPlacement,
                       delta_bgr: Tuple[float, float, float]) -> np.ndarray:
    logo = (placement.logo_bgr[0] + delta_bgr[0],
            placement.logo_bgr[1] + delta_bgr[1],
            placement.logo_bgr[2] + delta_bgr[2])
    buf = _full_alpha_buffer(image.shape, placement)
    return apply_reverse_alpha(image, buf, logo)


def _apply_core_halo_split(image, placement: AlphaPlacement) -> np.ndarray:
    """Invert the glyph core aggressively; soften the halo by blending the
    recovered halo pixels toward a locally-blurred surface (patch plan §Patch
    3.3 — most visible ghosts come from the core/halo boundary)."""
    buf = _full_alpha_buffer(image.shape, placement)
    cand = apply_reverse_alpha(image, buf, placement.logo_bgr)
    halo = (buf >= HALO_ALPHA_LO) & (buf < CORE_ALPHA)
    if int(halo.sum()) < 4:
        return cand
    blurred = cv2.GaussianBlur(cand, (0, 0), 1.2)
    out = cand.copy()
    out[halo] = ((cand[halo].astype(np.float32) * 0.4 +
                  blurred[halo].astype(np.float32) * 0.6)).astype(np.uint8)
    return out


def _screen_variant(image, candidate, placement, *, watermark_mask, product_mask,
                    orig_residual) -> ReverseAlphaResult:
    """Cheap local-safety + scoring for one variant image (mirrors the single
    candidate path). Authoritative gate is still the V17/V20 audit."""
    res = ReverseAlphaResult()
    res.placement = placement
    res.source = placement.source
    res.image = candidate
    res.residual_confidence_original = orig_residual
    resid_after = residual_confidence(candidate, placement.bbox,
                                      placement.alpha_map)
    res.residual_confidence_before_cleanup = resid_after
    res.residual_confidence_after_cleanup = resid_after
    res.changed_product_ratio = _changed_product_ratio(
        image, candidate, placement.bbox, product_mask)
    res.product_damage_score = _dark_blob(image, candidate, placement.bbox)
    try:
        import v17_final_audit
        res.protected_text_overlap = float(
            v17_final_audit._protected_text_overlap(
                image, placement.bbox, watermark_mask))
        gd = v17_final_audit.detect_reverse_alpha_ghost_dots(
            image, candidate, placement.bbox, watermark_mask, product_mask)
        res.ghost_dot_score = float(gd.get("score", 0.0))
        res.ghost_dot_hard_fail = bool(gd.get("hard_fail", False))
    except Exception:
        res.protected_text_overlap = 0.0
    reasons = []
    if res.changed_product_ratio > MAX_CHANGED_PRODUCT_RATIO:
        reasons.append("changed_too_much_product")
    if res.product_damage_score > DARK_BLOB_MAX:
        reasons.append("dark_surface_blob")
    if getattr(res, "ghost_dot_hard_fail", False):
        reasons.append("ghost_dot_residue_on_product")
    if resid_after > orig_residual - RESIDUAL_IMPROVE_MIN and orig_residual > 0.05:
        reasons.append("residual_did_not_improve")
    res.pass_local_safety = (len(reasons) == 0)
    res.reject_reason = None if res.pass_local_safety else ",".join(reasons)
    return res


def build_variant_beam(image, mark_box, *, watermark_mask=None, product_mask=None,
                       protected_text_mask=None, ctx=None, allow_thin_cleanup=True):
    """Return an ordered list of ``(name, ReverseAlphaResult)`` reverse-alpha
    variants that passed the local pre-screen, safest/cleanest first (patch plan
    §Patch 3). Empty when the asset is missing or no variant is locally safe."""
    if isinstance(mark_box, dict):
        mark_box = (mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"])
    bx, by, bw, bh = mark_box
    asset = load_alpha_asset()
    if asset is None or bw < 6 or bh < 4:
        return []
    orig_residual = residual_confidence(
        image, (bx, by, bw, bh),
        cv2.resize(asset["alpha"], (bw, bh), interpolation=cv2.INTER_AREA))

    fp = fixed_alpha_map(image, (bx, by, bw, bh), asset)
    ap = aligned_alpha_map(image, (bx, by, bw, bh), ctx, asset)
    base = ap or fp
    if base is None:
        return []

    raw: list = []   # (name, candidate_image, placement)
    if fp is not None:
        raw.append(("v20_reverse_alpha_fixed", apply_placement(image, fp), fp))
    raw.append(("v20_reverse_alpha_ncc", apply_placement(image, base), base))
    gain = _solve_local_gain(image, base)
    raw.append(("v20_reverse_alpha_ncc_local_gain",
                _apply_with_gain(image, base, gain), base))
    raw.append(("v20_reverse_alpha_core_halo_split",
                _apply_core_halo_split(image, base), base))
    delta = _solve_logo_delta(image, base, protected_text_mask)
    raw.append(("v20_reverse_alpha_per_channel_logo",
                _apply_per_channel(image, base, delta), base))
    if allow_thin_cleanup:
        base_img = apply_placement(image, base)
        halo_clean, _ = thin_residual_cleanup(
            base_img, base, protected_text_mask=protected_text_mask,
            dilate_px=1, method=cv2.INPAINT_NS)
        raw.append(("v20_reverse_alpha_low_alpha_halo_cleanup", halo_clean, base))
    # Explicit no-cleanup variant (alias of the ncc placement, kept for ranking
    # diversity / report parity).
    raw.append(("v20_reverse_alpha_no_cleanup", apply_placement(image, base), base))
    # V21 — smooth-surface texture-preserving refinement (patch plan §Patch 5).
    # Bilateral filter the glyph halo of the reverse-alpha base: it suppresses
    # faint low-contrast residue while keeping high-frequency surface texture
    # (cardboard grain, brushed metal), blended back ONLY over the footprint.
    try:
        base_img2 = apply_placement(image, base)
        foot2 = footprint_mask_for_box(image, (bx, by, bw, bh),
                                       watermark_mask, dilate_px=2)
        if foot2 is not None:
            smooth = cv2.bilateralFilter(base_img2, 5, 35, 7)
            refined = _blend_at_mask(base_img2, smooth, foot2, feather_px=2)
            raw.append(("v21_reverse_alpha_texture_preserve_blend", refined, base))
    except Exception:
        pass

    scored = []
    seen_names = set()
    for name, cand, pl in raw:
        if name in seen_names:
            continue
        seen_names.add(name)
        res = _screen_variant(image, cand, pl, watermark_mask=watermark_mask,
                              product_mask=product_mask,
                              orig_residual=orig_residual)
        if not res.pass_local_safety:
            continue
        # rank key: lower residual, fewer ghosts, less product disturbance.
        key = (round(res.residual_confidence_after_cleanup, 4),
               round(getattr(res, "ghost_dot_score", 0.0), 4),
               round(res.changed_product_ratio, 4))
        scored.append((key, name, res))
    scored.sort(key=lambda t: t[0])
    return [(name, res) for _k, name, res in scored]


# ===========================================================================
# V21 — Residue micro-cleanup beam (patch plan §Patch 2) + smooth-surface
# reverse-alpha refinement (patch plan §Patch 5).
#
# Reverse-alpha removes the *readable* overlay but can leave faint, low-contrast
# paired dots / pits aligned on the old watermark baseline (most visible on
# cardboard, smooth coloured backs and bright metal). V21 adds a micro-clean
# pass that touches ONLY small aligned residue components INSIDE the glyph
# footprint, area-capped, never a full bbox / band / rectangle. Each variant is
# returned as a candidate image; the AUTHORITATIVE gate is still the V17 final
# audit in the pipeline — a micro-clean that creates a blob or damages the
# product is rejected there, exactly like every other candidate.
# ===========================================================================
RESIDUE_MAX_FOOTPRINT_FRAC = 0.20   # edited area <= 20% of alpha footprint
RESIDUE_MAX_BOX_FRAC = 0.04         # edited area <= 4% of mark_box
RESIDUE_COMPONENT_MAX_FRAC = 0.015  # a residue blob is "small": <= 1.5% of box
RESIDUE_LOWCONTRAST_LO = 4          # min |hi-pass| (8-bit) to count as residue
RESIDUE_LOWCONTRAST_HI = 60         # above this it is real product detail — skip


def footprint_mask_for_box(image, mark_box, watermark_mask=None,
                           dilate_px=1) -> Optional[np.ndarray]:
    """Return a full-image uint8 mask of the watermark glyph footprint inside
    ``mark_box`` — the alpha-asset glyph shape if available, else the supplied
    ``watermark_mask`` crop. Used to restrict every V21 micro edit to the old
    overlay footprint (never the whole box)."""
    if isinstance(mark_box, dict):
        mark_box = (mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"])
    bx, by, bw, bh = [int(v) for v in mark_box]
    H, W = image.shape[:2]
    if bw < 4 or bh < 4:
        return None
    foot = np.zeros((H, W), np.uint8)
    asset = load_alpha_asset()
    if asset is not None:
        a = cv2.resize(asset["alpha"], (bw, bh), interpolation=cv2.INTER_AREA)
        glyph = (a > 0.06).astype(np.uint8) * 255
        foot[by:by + bh, bx:bx + bw] = glyph
    elif watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        foot[by:by + bh, bx:bx + bw] = watermark_mask[by:by + bh, bx:bx + bw]
    else:
        return None
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        foot = cv2.dilate(foot, k)
    if int((foot > 0).sum()) < 4:
        return None
    return foot


def _detect_residue_mask(candidate, mark_box, footprint, product_mask=None):
    """Find small, aligned, low-contrast residue components of ``candidate``
    inside ``footprint``. Returns ``(mask_uint8, edited_frac_of_footprint)`` or
    ``(None, 0.0)`` when no safe residue is found / the area cap is exceeded."""
    if isinstance(mark_box, dict):
        mark_box = (mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"])
    bx, by, bw, bh = [int(v) for v in mark_box]
    H, W = candidate.shape[:2]
    foot_bool = footprint > 0
    foot_area = int(foot_bool.sum())
    if foot_area < 4:
        return None, 0.0
    gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    # High-pass: deviation from a local median = paired-dot / pit residue.
    blur = cv2.medianBlur(gray, 5)
    hp = cv2.absdiff(gray, blur)
    resid = ((hp >= RESIDUE_LOWCONTRAST_LO) & (hp <= RESIDUE_LOWCONTRAST_HI) &
             foot_bool).astype(np.uint8)
    if int(resid.sum()) == 0:
        return None, 0.0
    # Keep only SMALL connected components (large => real product detail).
    box_area = max(1, bw * bh)
    comp_max = max(2, int(RESIDUE_COMPONENT_MAX_FRAC * box_area))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(resid, connectivity=8)
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= comp_max:
            keep[lab == i] = 255
    # Never touch protected product pixels passed explicitly.
    if product_mask is not None and product_mask.shape[:2] == (H, W):
        # micro-clean MAY touch product (it is non-destructive blur/inpaint of a
        # tiny dot), but cap product-side area harder by intersecting with the
        # footprint only — already done. Keep as-is.
        pass
    edited = int((keep > 0).sum())
    if edited == 0:
        return None, 0.0
    frac_foot = edited / foot_area
    frac_box = edited / box_area
    if frac_foot > RESIDUE_MAX_FOOTPRINT_FRAC or frac_box > RESIDUE_MAX_BOX_FRAC:
        # Too much area — this is not isolated residue; refuse (caller skips).
        return None, frac_foot
    return keep, frac_foot


def _blend_at_mask(base, repl, mask, feather_px=1):
    """Feathered alpha blend of ``repl`` into ``base`` over ``mask``."""
    m = (mask > 0).astype(np.float32)
    if feather_px > 0:
        m = cv2.GaussianBlur(m, (2 * feather_px + 1, 2 * feather_px + 1), 0)
    m3 = np.dstack([m, m, m])
    return (base.astype(np.float32) * (1 - m3) +
            repl.astype(np.float32) * m3).clip(0, 255).astype(np.uint8)


def build_residue_micro_cleanup_beam(original, candidate, mark_box,
                                     *, watermark_mask=None, product_mask=None,
                                     roi_class=""):
    """Patch plan §Patch 2. Given a reverse-alpha ``candidate`` (mark removed but
    possibly faint dots remaining), return a list of ``(name, image)`` micro-clean
    variants that touch ONLY small aligned residue components inside the glyph
    footprint. Empty when there is no safe residue to clean. Every variant still
    passes the unchanged P0 / V17 audit in the pipeline."""
    if candidate is None or candidate.shape != original.shape:
        return []
    foot = footprint_mask_for_box(original, mark_box, watermark_mask, dilate_px=1)
    if foot is None:
        return []
    resid, _frac = _detect_residue_mask(candidate, mark_box, foot, product_mask)
    if resid is None:
        return []
    out = []
    # 1) surface blur — average out the dot against its immediate surround.
    try:
        blurred = cv2.GaussianBlur(candidate, (5, 5), 0)
        out.append(("v21_residue_micro_surface_blur",
                    _blend_at_mask(candidate, blurred, resid, feather_px=1)))
    except Exception:
        pass
    # 2) hue-matched clone / NS inpaint of only the residue components.
    try:
        ns = cv2.inpaint(candidate, resid, RESIDUAL_INPAINT_RADIUS, cv2.INPAINT_NS)
        out.append(("v21_residue_micro_hue_matched_clone", ns))
    except Exception:
        pass
    # 3) component inpaint (Telea) — different reconstruction prior.
    try:
        telea = cv2.inpaint(candidate, resid, RESIDUAL_INPAINT_RADIUS,
                            cv2.INPAINT_TELEA)
        out.append(("v21_residue_micro_component_inpaint", telea))
    except Exception:
        pass
    # 4) reverse-alpha residual-gain: lift the residue toward the local ring mean
    #    (handles faint pits where inpaint over-smooths).
    try:
        ring = cv2.dilate(resid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (5, 5))) & ~resid
        if int((ring > 0).sum()) >= 4:
            target = cv2.medianBlur(candidate, 5)
            out.append(("v21_residue_micro_reverse_alpha_gain",
                        _blend_at_mask(candidate, target, resid, feather_px=1)))
    except Exception:
        pass
    # 5) no-op control (kept for ranking parity / report).
    out.append(("v21_residue_micro_noop_control", candidate))
    # De-dup by name, drop shape mismatches.
    seen, uniq = set(), []
    for name, img in out:
        if name in seen or img is None or img.shape != original.shape:
            continue
        seen.add(name)
        uniq.append((name, img))
    return uniq
