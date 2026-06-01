#!/usr/bin/env python3
"""V17 regression tests — truthful final audit + product-aware fill gating.

These lock the failure modes found in the V16 50-image compare run (patch plan
§8/§10): published residual watermark, and product damage from a cosmetic seam
that lands on product pixels. They use synthetic fixtures so they run anywhere
without the bench assets.

Run:  python3 -m pytest test_v17_patch.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

import v17_final_audit as v17


# ---------------------------------------------------------------------------
# Fixtures — synthetic ROIs with a known watermark band and product structure.
# ---------------------------------------------------------------------------
def _white(h=120, w=240):
    return np.full((h, w, 3), 245, np.uint8)


def _bbox(img, frac_w=0.5, frac_h=0.25):
    H, W = img.shape[:2]
    bw, bh = int(W * frac_w), int(H * frac_h)
    bx, by = (W - bw) // 2, (H - bh) // 2
    return (bx, by, bw, bh)


def _draw_text_band(img, bbox, luma=70):
    """Draw a readable dot-chain (watermark-like glyph row) inside bbox."""
    bx, by, bw, bh = bbox
    out = img.copy()
    cy = by + bh // 2
    x = bx + 4
    while x < bx + bw - 6:
        out[cy - 4:cy + 4, x:x + 4] = luma   # a glyph
        x += 10                               # gap -> dot-chain
    return out


def _watermark_mask(img, bbox):
    m = np.zeros(img.shape[:2], np.uint8)
    bx, by, bw, bh = bbox
    m[by:by + bh, bx:bx + bw] = 255
    return m


def _thin_wm_mask(img, bbox):
    """Realistic watermark stroke mask: a thin glyph row, not the whole box.
    (The real pipeline passes the stroke mask, never the full bbox.)"""
    m = np.zeros(img.shape[:2], np.uint8)
    bx, by, bw, bh = bbox
    cy = by + bh // 2
    m[cy - 4:cy + 4, bx + 2:bx + bw - 2] = 255
    return m


# ===========================================================================
# 1. Residual watermark on a published output must be a P0 hard fail (§1.1).
# ===========================================================================
def test_readable_residual_is_p0_fail():
    img = _white()
    bbox = _bbox(img)
    original = _draw_text_band(img, bbox, luma=60)
    # "Cleaned" output still has a faint readable glyph row (broken clean).
    output = _draw_text_band(img, bbox, luma=120)
    wm = _watermark_mask(img, bbox)
    res = v17.audit_final_output(original, output, bbox, None, wm,
                                 still_present=True)
    assert res.pass_p0 is False
    assert res.has_readable_residual is True
    assert "published_residual_watermark" in res.hard_fail_reasons


def test_clean_white_fill_passes():
    img = _white()
    bbox = _bbox(img)
    original = _draw_text_band(img, bbox, luma=60)
    output = _white()   # perfectly filled to background
    wm = _watermark_mask(img, bbox)
    res = v17.audit_final_output(original, output, bbox, None, wm,
                                 still_present=False)
    assert res.pass_p0 is True
    assert res.has_readable_residual is False


# ===========================================================================
# 2. A cosmetic seam ON PRODUCT is product damage, ON BACKGROUND is allowed
#    (§1.2 / §4.1).
# ===========================================================================
def test_visible_patch_on_background_is_cosmetic():
    img = _white()
    bbox = _bbox(img)
    original = _draw_text_band(img, bbox, luma=60)
    # Output: faint uniform seam on pure white background, no product.
    output = _white()
    output[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 240
    res = v17.audit_final_output(
        original, output, bbox, None, _watermark_mask(img, bbox),
        still_present=False, visible_patch_failed=True, visible_band_failed=True)
    # No product under the seam -> cosmetic, not a hard fail.
    assert res.pure_background_change is True
    assert "visible_patch_on_product" not in res.hard_fail_reasons
    assert "visible_band_on_product" not in res.hard_fail_reasons


def test_visible_patch_on_product_is_hard_fail():
    # Product = textured dark region under the whole bbox.
    img = _white()
    bbox = _bbox(img)
    bx, by, bw, bh = bbox
    original = img.copy()
    # Lay down product structure (edges + dark pixels) across the box.
    for x in range(bx, bx + bw, 6):
        original[by:by + bh, x:x + 2] = 30
    product_mask = np.zeros(img.shape[:2], np.uint8)
    product_mask[by:by + bh, bx:bx + bw] = 255
    # Output: a flat bright band wiped across the product (destructive fill).
    output = original.copy()
    output[by:by + bh, bx:bx + bw] = 240
    res = v17.audit_final_output(
        original, output, bbox, product_mask, _thin_wm_mask(img, bbox),
        still_present=False, visible_patch_failed=True, visible_band_failed=True)
    assert res.pure_background_change is False
    assert res.changed_product_ratio > v17.PRODUCT_OVERLAP_HARD
    assert ("visible_patch_on_product" in res.hard_fail_reasons or
            "visible_band_on_product" in res.hard_fail_reasons)
    assert res.pass_p0 is False


# ===========================================================================
# 3. Thin flex cable broken by the fill must hard-fail (§5.2 / §10.2).
# ===========================================================================
def test_thin_flex_cable_break_is_hard_fail():
    img = _white(h=120, w=300)
    bbox = _bbox(img, frac_w=0.6, frac_h=0.2)
    bx, by, bw, bh = bbox
    original = img.copy()
    # A continuous thin black cable crossing the box horizontally.
    cy = by + bh // 2
    original[cy - 1:cy + 2, 0:img.shape[1]] = 10
    product_mask = np.zeros(img.shape[:2], np.uint8)
    product_mask[cy - 3:cy + 4, :] = 255
    # Output: a white band cuts the cable inside the box.
    output = original.copy()
    output[by:by + bh, bx:bx + bw] = 245
    res = v17.audit_final_output(
        original, output, bbox, product_mask, _thin_wm_mask(img, bbox),
        still_present=False, roi_class="thin_flex_cable",
        visible_patch_failed=True, visible_band_failed=True)
    assert res.pass_p0 is False
    assert ("changed_thin_flex_structure" in res.hard_fail_reasons or
            "visible_band_on_product" in res.hard_fail_reasons or
            "changed_product_silhouette" in res.hard_fail_reasons)


# ===========================================================================
# 4. uniform_background_fill is allowed only on strict pure background (§1.3).
# ===========================================================================
def test_uniform_fill_allowed_on_pure_white():
    img = _white()
    bbox = _bbox(img)
    img2 = _draw_text_band(img, bbox, luma=70)   # only the watermark itself
    wm = _watermark_mask(img, bbox)
    assert v17.allow_uniform_background_fill(img2, bbox, None, wm) is True


def test_uniform_fill_blocked_on_product():
    img = _white()
    bbox = _bbox(img)
    bx, by, bw, bh = bbox
    original = img.copy()
    for x in range(bx, bx + bw, 5):           # dense product edges in the box
        original[by:by + bh, x:x + 2] = 20
    product_mask = np.zeros(img.shape[:2], np.uint8)
    product_mask[by:by + bh, bx:bx + bw] = 255
    wm = _watermark_mask(img, bbox)
    assert v17.allow_uniform_background_fill(
        original, bbox, product_mask, wm) is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
