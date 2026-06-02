#!/usr/bin/env python3
"""V20 — product-side cover publishing must be a HARD failure (patch plan §Patch 1).

A clean_covered output whose changed pixels land on product (more than 1%) and
which shows a visible cover shape / boundary / silhouette crossing must NOT
publish. Covers are allowed only on pure background or as stroke-shaped changes.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v17_final_audit as audit  # noqa: E402


def _textured_product():
    rng = np.random.RandomState(11)
    return rng.randint(40, 210, (140, 420, 3)).astype(np.uint8)


def _flat_colored_product(value=(60, 90, 160)):
    img = np.zeros((140, 420, 3), np.uint8)
    img[:] = value
    return img


# --------------------------------------------------------------------------
# Patch 1 — rectangular slab / box cover on product is a hard fail.
# --------------------------------------------------------------------------
def test_v20_rectangular_slab_on_product_hard_fails():
    original = _textured_product()
    box = (130, 50, 170, 40)
    output = original.copy()
    bx, by, bw, bh = box
    output[by:by + bh, bx:bx + bw] = 150          # flat gray slab over product
    product_mask = np.full(original.shape[:2], 255, np.uint8)
    changed = audit._changed_mask(original, output, box)
    cav = audit.detect_cover_shape_artifact_v20(
        original, output, changed, product_mask, box)
    assert cav["changed_on_product_fraction"] > 0.01
    assert cav["rectangularity"] > audit.COVER_RECTANGULARITY_MAX
    assert cav["hard_fail_reason"] is not None


def test_v20_clean_covered_on_product_slab_fails_audit():
    original = _textured_product()
    box = (130, 50, 170, 40)
    output = original.copy()
    bx, by, bw, bh = box
    output[by:by + bh, bx:bx + bw] = 150
    product_mask = np.full(original.shape[:2], 255, np.uint8)
    res = audit.audit_final_output(
        original, output, box, product_mask, None,
        final_status="clean_covered", still_present=False)
    assert res.pass_p0 is False
    assert res.cover_artifact_v20.get("hard_fail_reason") is not None


def test_v20_clean_covered_on_pure_background_passes():
    original = np.full((140, 420, 3), 245, np.uint8)
    box = (130, 50, 170, 40)
    output = original.copy()
    bx, by, bw, bh = box
    output[by:by + bh, bx:bx + bw] = 243          # imperceptible seam on white
    product_mask = np.zeros(original.shape[:2], np.uint8)
    res = audit.audit_final_output(
        original, output, box, product_mask, None,
        final_status="clean_covered", still_present=False)
    assert res.pass_p0 is True
    assert res.cover_artifact_v20.get("hard_fail_reason") is None


def test_v20_stroke_shaped_change_on_product_allowed():
    # A thin stroke-shaped change (low rectangularity, tone-matched) is not a slab.
    original = _flat_colored_product()
    box = (130, 50, 170, 40)
    output = original.copy()
    bx, by, bw, bh = box
    # paint a few 2px tall thin strokes matched closely to surface tone
    for i in range(6):
        x0 = bx + 8 + i * 26
        output[by + bh // 2: by + bh // 2 + 2, x0:x0 + 14] = (58, 88, 158)
    product_mask = np.full(original.shape[:2], 255, np.uint8)
    changed = audit._changed_mask(original, output, box)
    cav = audit.detect_cover_shape_artifact_v20(
        original, output, changed, product_mask, box)
    # stroke-shaped: low rectangularity, low color delta => no hard fail.
    assert cav["hard_fail_reason"] is None


# --------------------------------------------------------------------------
# Patch 4 — reverse-alpha ghost dots on product are a hard fail.
# --------------------------------------------------------------------------
def test_v20_ghost_dots_on_product_hard_fail():
    original = _flat_colored_product()
    box = (130, 50, 200, 36)
    output = original.copy()
    bx, by, bw, bh = box
    # faint paired dots aligned on the baseline (low contrast vs surface).
    y = by + bh // 2
    for i in range(8):
        x = bx + 10 + i * 22
        cv2.circle(output, (x, y), 2, (70, 100, 170), -1)       # +~10 luma
        cv2.circle(output, (x + 6, y), 2, (70, 100, 170), -1)   # paired
    product_mask = np.full(original.shape[:2], 255, np.uint8)
    gd = audit.detect_reverse_alpha_ghost_dots(
        original, output, box, None, product_mask)
    assert gd["component_count"] >= audit.GHOST_DOT_MIN_COMPONENTS
    assert gd["hard_fail"] is True


def test_v20_ghost_dots_on_background_not_hard_fail():
    original = np.full((140, 420, 3), 245, np.uint8)
    box = (130, 50, 200, 36)
    output = original.copy()
    bx, by, bw, bh = box
    y = by + bh // 2
    for i in range(8):
        x = bx + 10 + i * 22
        cv2.circle(output, (x, y), 2, (236, 236, 236), -1)
    product_mask = np.zeros(original.shape[:2], np.uint8)
    gd = audit.detect_reverse_alpha_ghost_dots(
        original, output, box, None, product_mask)
    # On pure background: never a hard fail (routed to micro-cleanup instead).
    assert gd["hard_fail"] is False


def test_v20_clean_surface_no_ghost_dots():
    original = _flat_colored_product()
    output = original.copy()             # perfectly clean
    box = (130, 50, 200, 36)
    product_mask = np.full(original.shape[:2], 255, np.uint8)
    gd = audit.detect_reverse_alpha_ghost_dots(
        original, output, box, None, product_mask)
    assert gd["hard_fail"] is False
    assert gd["score"] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
