#!/usr/bin/env python3
"""V20 — mixed product/background segmented repair + union product mask
(patch plan §Patch 5, §Patch 8)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v18_patch  # noqa: E402


def _mixed_scene():
    """Left half white background, right half dark product, watermark crossing."""
    img = np.full((140, 400, 3), 240, np.uint8)
    img[:, 200:] = 30                       # dark product on the right
    box = (120, 50, 180, 36)
    wm = np.zeros(img.shape[:2], np.uint8)
    bx, by, bw, bh = box
    for i in range(9):                      # watermark strokes across the seam
        x0 = bx + 6 + i * (bw // 10)
        cv2.rectangle(wm, (x0, by + bh // 3), (x0 + 3, by + 2 * bh // 3), 255, -1)
    # apply the strokes to the image (darken slightly = overlay)
    img[wm > 0] = (200, 200, 200)
    product_mask = np.zeros(img.shape[:2], np.uint8)
    product_mask[:, 200:] = 255
    return img, box, wm, product_mask


# --------------------------------------------------------------------------
# Patch 8 — union product-safe mask
# --------------------------------------------------------------------------
def test_union_mask_covers_dark_and_colored_product():
    img, box, wm, product_mask = _mixed_scene()
    safe = v18_patch.build_product_mask_safe(img, box, product_mask, wm)
    assert safe.shape[:2] == img.shape[:2]
    # The dark product half inside the box registers as safe (product) pixels.
    bx, by, bw, bh = box
    right = safe[by:by + bh, 200:bx + bw]
    assert int((right > 0).sum()) > 0


def test_union_mask_excludes_watermark_strokes():
    img, box, wm, product_mask = _mixed_scene()
    safe = v18_patch.build_product_mask_safe(img, box, None, wm)
    # Watermark stroke pixels must not be counted as product (they are excluded).
    assert int((safe[wm > 0] > 0).sum()) == 0


def test_ban_destructive_when_union_overlap_high():
    img, box, wm, product_mask = _mixed_scene()
    ctx = v18_patch.compute_product_context(img, box, product_mask, wm)
    assert ctx.product_mask_safe_overlap > 0.0
    # A mixed/product scene must ban destructive fills.
    assert v18_patch.should_ban_destructive(ctx) is True


def test_union_mask_low_on_pure_white_background():
    # The union product-safe mask must NOT over-flag a pure white background: the
    # watermark's own strokes are excluded, so its overlap stays near zero (the
    # legacy non-white product_overlap heuristic is a separate signal).
    img = np.full((140, 400, 3), 245, np.uint8)
    box = (120, 50, 180, 36)
    wm = np.zeros(img.shape[:2], np.uint8)
    bx, by, bw, bh = box
    for i in range(9):
        x0 = bx + 6 + i * (bw // 10)
        cv2.rectangle(wm, (x0, by + bh // 3), (x0 + 3, by + 2 * bh // 3), 255, -1)
    img[wm > 0] = (205, 205, 205)
    ctx = v18_patch.compute_product_context(img, box, None, wm)
    assert ctx.product_mask_safe_overlap <= v18_patch.PRODUCT_MASK_SAFE_BAN


# --------------------------------------------------------------------------
# Patch 5 — segmented mixed repair
# --------------------------------------------------------------------------
def test_segmented_mixed_repair_returns_valid_or_none():
    img, box, wm, product_mask = _mixed_scene()
    out = v18_patch.segmented_reverse_alpha_background_clone(
        img, box, wm, product_mask)
    # Either a valid same-shape image or None (skipped) — never a crash.
    assert out is None or (out.shape == img.shape and out.dtype == img.dtype)


def test_segmented_mixed_repair_does_not_fill_product_with_background():
    img, box, wm, product_mask = _mixed_scene()
    out = v18_patch.segmented_reverse_alpha_background_clone(
        img, box, wm, product_mask)
    if out is None:
        return
    # The dark product region must stay dark (not flooded with white background).
    bx, by, bw, bh = box
    prod_after = cv2.cvtColor(out[by:by + bh, 210:bx + bw], cv2.COLOR_BGR2GRAY)
    assert float(prod_after.mean()) < 160, "product pixels were lightened too much"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
