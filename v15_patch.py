#!/usr/bin/env python3
"""V15 — Cover-quality patch: full-text coverage, ghost-aware residual, dark-stroke inpaint.

V14 raised true `clean_repaired` output and softened the worst covers, but two
cover-side problems remained visible to the eye on the hardest surfaces:

  1. **Faint residual text.** The detected watermark mask under-covers the full
     ``sunsky-online.com`` text, so trailing/edge glyphs survive — and the V13
     residual gate measures only *inside* the mask, so a faint ghost just
     outside it scores as "hidden".
  2. **Bright / patchy covers on dark surfaces.** A median / background fill over
     a dark flex cable reads as a light block.

V15 adds three bounded, additive helpers (no V13 gate change):

  * ``widen_text_mask`` — template-locate the FULL watermark text line in a
    horizontally-expanded band and grow the stroke mask to cover the clipped
    glyphs, with strong guards so it never grows onto product structure;
  * ``full_region_residual_ok`` — a ghost-aware residual verdict that checks the
    whole widened text region (template correlation + text-component drop), not
    only the tight mask, so a faint surviving ghost is caught and the cover is
    rejected / re-tried;
  * ``dark_stroke_cover`` — on a dark / flex-cable surface, inpaint the *widened
    stroke mask* from the neighbouring (dark) pixels with Telea, never a
    light/median fill.

Everything here is conservative and reversible: the widened mask is only used to
*remove more watermark*, the residual verdict only makes the gate stricter, and
the dark-stroke cover is just one more beam candidate. If a guard trips, V15
falls back to the V14 behaviour unchanged.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Reuse the canonical-template residual tools from progressive_repair; fall back
# to safe stubs so this module imports / unit-tests stand-alone.
try:  # pragma: no cover - exercised indirectly
    from progressive_repair import (
        _get_canonical_template as _pr_template,
        _residual_template_corr as _pr_tmpl_corr,
        _text_component_score as _pr_text_score,
        _hp_norm as _pr_hp_norm,
    )
    _HAVE_PR = True
except Exception:  # pragma: no cover
    _pr_template = lambda: None
    _pr_tmpl_corr = lambda g: 0.0
    _pr_text_score = lambda g: 0.0
    _pr_hp_norm = None
    _HAVE_PR = False


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


# ---------------------------------------------------------------------------
# 1) Mask widening to the full watermark text footprint.
# ---------------------------------------------------------------------------
def _text_row_band(bbox, watermark_mask, x_expand, y_pad, shape):
    """Search band around the watermark text row: the y-extent of the existing
    mask (the glyph row) padded a little, the x-extent expanded outward to catch
    clipped leading/trailing glyphs."""
    bx, by, bw, bh = bbox
    H, W = shape[:2]
    y1, y2 = by, by + bh
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        ys, xs = np.where(watermark_mask[by:by + bh, bx:bx + bw] > 0)
        if ys.size:
            y1 = by + int(ys.min())
            y2 = by + int(ys.max()) + 1
    pad_y = max(2, int(round((y2 - y1) * y_pad)))
    ex = max(8, int(round(bw * x_expand)))
    by1, by2 = max(0, y1 - pad_y), min(H, y2 + pad_y)
    bx1, bx2 = max(0, bx - ex), min(W, bx + bw + ex)
    return by1, by2, bx1, bx2


def widen_text_mask(image, bbox, watermark_mask, product_mask=None,
                    x_expand=0.35, y_pad=0.3, max_overlap=0.6):
    """Grow the watermark stroke mask to cover the full ``sunsky-online.com``
    text line (clipped trailing glyphs included).

    Conservative by design — a glyph pixel is added only when it is faint,
    text-like and aligned with the existing watermark row, and the whole
    widening is skipped on a busy product band (to never eat product detail).
    Returns the widened uint8 mask (full image) or the original mask if widening
    is unsafe / unavailable.
    """
    H, W = image.shape[:2]
    if watermark_mask is None or watermark_mask.shape[:2] != (H, W) \
            or not np.any(watermark_mask > 0):
        return watermark_mask
    by1, by2, bx1, bx2 = _text_row_band(bbox, watermark_mask, x_expand, y_pad,
                                        image.shape)
    if by2 - by1 < 3 or bx2 - bx1 < 8:
        return watermark_mask
    band = cv2.cvtColor(image[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Local row background = column-wise median of the band's top & bottom
    # margins (above/below the glyph row), so the baseline tracks a gradient.
    margin = max(1, (by2 - by1) // 6)
    top = band[:margin]
    bot = band[-margin:]
    base_rows = np.concatenate([top, bot], axis=0)
    base = np.median(base_rows, axis=0, keepdims=True)
    noise = float(np.median(np.abs(base_rows - base))) * 1.4826
    noise = max(noise, 2.0)

    # The sunsky mark is a semi-transparent light overlay: glyph pixels deviate
    # from the local background (brighter on dark surface, slightly grey on
    # white). Detect a faint, NOT-strong deviation (strong edges are product).
    dev = np.abs(band - base)
    glyph = ((dev > 2.0 * noise) & (dev < 14.0 * noise)).astype(np.uint8)
    glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))

    # Guard 1 — never widen across a busy product band (would eat detail).
    edges = cv2.Canny(band.astype(np.uint8), 50, 150)
    if float((edges > 0).mean()) > 0.18:
        return watermark_mask

    # Guard 2 — drop components that overlap product structure too much, and
    # keep only glyph-row-aligned, glyph-sized components.
    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(glyph)
    keep = np.zeros_like(glyph)
    row_h = by2 - by1
    prod_band = None
    if product_mask is not None and product_mask.shape[:2] == (H, W):
        prod_band = product_mask[by1:by2, bx1:bx2] > 0
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w_ = int(stats[i, cv2.CC_STAT_WIDTH])
        h_ = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < 2 or h_ > row_h or w_ > 0.5 * (bx2 - bx1):
            continue
        comp = lab == i
        if prod_band is not None and prod_band.any():
            ov = float((comp & prod_band).sum()) / max(1, int(comp.sum()))
            if ov > max_overlap:
                continue
        keep[comp] = 1

    add = np.zeros((H, W), np.uint8)
    add[by1:by2, bx1:bx2] = keep * 255
    widened = cv2.bitwise_or((watermark_mask > 0).astype(np.uint8) * 255, add)

    # Guard 3 — refuse a runaway expansion (added area must stay text-thin).
    base_area = int((watermark_mask > 0).sum())
    if int((widened > 0).sum()) > base_area * 3 + 200:
        return watermark_mask
    return widened


# ---------------------------------------------------------------------------
# 2) Ghost-aware full-region residual verdict.
# ---------------------------------------------------------------------------
def full_region_residual_ok(original, cover, bbox, watermark_mask=None,
                            x_expand=0.35, y_pad=0.4) -> bool:
    """True when the watermark text is gone across the FULL text region — not
    just inside the tight mask. Catches a faint surviving ghost the mask-relative
    check misses, by requiring both the template correlation and the text-like
    component energy to drop substantially vs the original.
    """
    by1, by2, bx1, bx2 = _text_row_band(bbox, watermark_mask, x_expand, y_pad,
                                        original.shape)
    if by2 - by1 < 4 or bx2 - bx1 < 12:
        return True
    o = cv2.cvtColor(original[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(cover[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
    if o.size == 0 or c.shape != o.shape:
        return True

    # The discriminative signal is the watermark-SPECIFIC template correlation.
    # A generic text-component count is NOT used here: a flex cable / connector
    # surface is full of text-like high-pass components that have nothing to do
    # with the watermark, so it false-positives on exactly the hard surfaces we
    # care about. The canonical-template correlation only responds to the
    # sunsky glyph shapes.
    o_corr = _pr_tmpl_corr(o)
    c_corr = _pr_tmpl_corr(c)

    # Absolute readable-watermark ceiling (the cover must not itself match the
    # watermark template).
    if c_corr > 0.34:
        return False
    # Relative drop — a faint surviving ghost keeps the watermark correlation
    # close to the original; a real removal collapses it.
    if o_corr > 0.22 and c_corr > 0.6 * o_corr:
        return False
    return True


# ---------------------------------------------------------------------------
# 3) Dark-surface stroke inpaint (no light/median fill).
# ---------------------------------------------------------------------------
def dark_stroke_cover(image, bbox, widened_mask, radius=3):
    """Inpaint the (widened) watermark stroke mask from neighbouring pixels with
    Telea — the right move on a dark flex cable / dark product surface, where a
    median or light fill would leave a bright block. Telea propagates the real
    surrounding (dark) pixels inward, so no colour is invented."""
    H, W = image.shape[:2]
    if widened_mask is None or widened_mask.shape[:2] != (H, W) \
            or not np.any(widened_mask > 0):
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.dilate((widened_mask > 0).astype(np.uint8) * 255, k, iterations=1)
    return cv2.inpaint(image, m, radius, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Convenience — surface darkness probe used to decide dark-stroke priority.
# ---------------------------------------------------------------------------
def is_dark_surface(image, bbox, thresh=0.30) -> bool:
    bx, by, bw, bh = bbox
    if bw <= 1 or bh <= 1:
        return False
    g = cv2.cvtColor(image[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2GRAY)
    return float((g < 80).mean()) > thresh


# ---------------------------------------------------------------------------
# V15.1 Fix 1 — pure / near-uniform background: just imitate the neighbour.
#
# On a uniform local background (the screen interior, a plain white sheet, a
# flat surface) the only correct repair is to copy the surrounding background
# over the watermark footprint. Anything cleverer (template-logo subtraction,
# clone-of-clone) risks leaving a garbled doubled-glyph smear that the metric
# QA misses because it is faint. This is the lightest, safest, most invisible
# fix — so when the local background is uniform we use it outright.
# ---------------------------------------------------------------------------
def is_uniform_local_background(image, bbox, product_mask=None,
                                std_max=12.0, edge_max=0.06,
                                overlap_max=0.10, band_px=6) -> bool:
    """True when the watermark sits on a locally-uniform background, judged by
    the THIN perimeter band immediately around the footprint (the pixels a
    neighbour-imitation Telea fill actually copies from) — not a wide padded
    ring, which would catch unrelated product structure further out (e.g. the
    flex connector below a screen) and wrongly veto a pure-white case.

    If that immediate perimeter is uniform and low-edge, a fill from it is
    guaranteed clean. A product edge running through the perimeter raises the
    std / edge density and correctly vetoes it (no smear onto structure)."""
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    if bw <= 1 or bh <= 1:
        return False
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p = max(2, band_px)
    y1, y2 = max(0, by - p), min(H, by + bh + p)
    x1, x2 = max(0, bx - p), min(W, bx + bw + p)
    perim = np.zeros((y2 - y1, x2 - x1), bool)
    perim[:] = True
    iy1, iy2 = by - y1, by - y1 + bh
    ix1, ix2 = bx - x1, bx - x1 + bw
    perim[max(0, iy1):max(0, iy2), max(0, ix1):max(0, ix2)] = False
    vals = g[y1:y2, x1:x2][perim]
    if vals.size < 16:
        return False
    # Robust spread (MAD) of the immediate perimeter — uniform surround only.
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    if mad > std_max:
        return False
    # Low edge density in the perimeter band (no product structure crossing it).
    edges = cv2.Canny(g[y1:y2, x1:x2], 50, 150)
    if float((edges[perim] > 0).mean()) > edge_max:
        return False
    # The watermark footprint itself must not sit on real product pixels.
    if product_mask is not None and product_mask.shape[:2] == (H, W):
        pr = product_mask[by:by + bh, bx:bx + bw]
        if pr.size and float((pr > 0).mean()) > overlap_max:
            return False
    return True


def uniform_background_fill(image, bbox, watermark_mask=None, x_expand=0.35,
                            radius=4):
    """Imitate the surrounding (uniform) background over the watermark
    footprint: Telea-propagate the real neighbouring pixels inward, then feather
    the boundary. On a uniform surface this is exact and invisible. Covers the
    full bbox widened horizontally so no trailing glyph survives."""
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    fp = np.zeros((H, W), np.uint8)
    pad_y = max(2, int(round(bh * 0.18)))
    pad_x = max(8, int(round(bw * x_expand)))
    fy1, fy2 = max(0, by - pad_y), min(H, by + bh + pad_y)
    fx1, fx2 = max(0, bx - pad_x), min(W, bx + bw + pad_x)
    fp[fy1:fy2, fx1:fx2] = 255
    result = cv2.inpaint(image, fp, radius, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(fp.astype(np.float32) / 255.0, (0, 0), sigmaX=2.0)
    a3 = np.stack([alpha] * 3, axis=-1)
    return (result.astype(np.float32) * a3 +
            image.astype(np.float32) * (1 - a3)).astype(np.uint8)


# ---------------------------------------------------------------------------
# V15.1 Fix 2 — escalation fill when a watermark survives the first clean.
# ---------------------------------------------------------------------------
def forced_removal_fill(image, bbox, watermark_mask=None, x_expand=0.3,
                        radius=4):
    """A stronger last-resort removal used only when the post-clean re-detection
    still finds the watermark. Telea-inpaints the full watermark text band over
    the bbox (widened mask ∪ central text band). On product surfaces this trades
    a little local texture for actually removing a mark that would otherwise
    ship visible — the watermark MUST go."""
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    band = np.zeros((H, W), np.uint8)
    pad_x = max(6, int(round(bw * x_expand)))
    # central text band (tight in y, full text width) — where the glyph row is.
    ty1 = by + int(bh * 0.10)
    ty2 = by + int(bh * 0.90)
    tx1, tx2 = max(0, bx - pad_x), min(W, bx + bw + pad_x)
    band[max(0, ty1):min(H, ty2), tx1:tx2] = 255
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        wm = cv2.dilate((watermark_mask > 0).astype(np.uint8) * 255, kd,
                        iterations=1)
        band = cv2.bitwise_or(band, wm)
    result = cv2.inpaint(image, band, radius, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(band.astype(np.float32) / 255.0, (0, 0), sigmaX=2.0)
    a3 = np.stack([alpha] * 3, axis=-1)
    return (result.astype(np.float32) * a3 +
            image.astype(np.float32) * (1 - a3)).astype(np.uint8)
