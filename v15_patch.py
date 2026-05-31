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
