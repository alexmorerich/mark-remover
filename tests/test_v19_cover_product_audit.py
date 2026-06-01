#!/usr/bin/env python3
"""V19 — stricter cover-on-product audit tests (patch plan §6.1)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v17_final_audit as audit  # noqa: E402


def _textured_product():
    rng = np.random.RandomState(7)
    return rng.randint(40, 210, (120, 400, 3)).astype(np.uint8)


def test_rectangular_slab_cover_on_product_is_hard_fail():
    original = _textured_product()
    box = (120, 40, 160, 36)
    output = original.copy()
    # Paint a flat gray rectangular slab over product pixels (a destructive cover).
    bx, by, bw, bh = box
    output[by:by + bh, bx:bx + bw] = 150
    product_mask = np.full(original.shape[:2], 255, np.uint8)
    changed = audit._changed_mask(original, output, box)
    art = audit.detect_cover_shape_artifact_v19(
        original, output, changed, product_mask, box)
    assert art["has_artifact"] is True
    assert "wedge_or_slab_shape" in art["artifact_reasons"] or \
        "visible_patch_on_product" in art["artifact_reasons"]


def test_clean_covered_on_product_slab_fails_audit():
    original = _textured_product()
    box = (120, 40, 160, 36)
    output = original.copy()
    bx, by, bw, bh = box
    output[by:by + bh, bx:bx + bw] = 150
    product_mask = np.full(original.shape[:2], 255, np.uint8)
    res = audit.audit_final_output(
        original, output, box, product_mask, None,
        final_status="clean_covered", still_present=False)
    assert res.pass_p0 is False
    assert len(res.hard_fail_reasons) > 0


def test_clean_covered_on_pure_background_passes():
    # A cover on a uniform white background (no product) is allowed.
    original = np.full((120, 400, 3), 245, np.uint8)
    box = (120, 40, 160, 36)
    output = original.copy()
    bx, by, bw, bh = box
    output[by:by + bh, bx:bx + bw] = 243   # imperceptible seam on white
    product_mask = np.zeros(original.shape[:2], np.uint8)
    res = audit.audit_final_output(
        original, output, box, product_mask, None,
        final_status="clean_covered", still_present=False)
    assert res.pass_p0 is True
