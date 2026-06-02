#!/usr/bin/env python3
"""V20 — thin flex cable continuity protection (patch plan §Patch 6)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v13_gates  # noqa: E402
import v18_patch  # noqa: E402


def _cable_scene():
    """A white canvas with a long horizontal dark cable line through the bbox."""
    img = np.full((120, 400, 3), 240, np.uint8)
    cv2.line(img, (20, 60), (380, 60), (20, 20, 20), 4)   # the cable
    box = (120, 44, 180, 32)
    wm = np.zeros(img.shape[:2], np.uint8)
    bx, by, bw, bh = box
    for i in range(8):
        x0 = bx + 8 + i * 20
        cv2.rectangle(wm, (x0, by + 4), (x0 + 3, by + 12), 255, -1)
    return img, box, wm


def test_continuity_intact_passes():
    img, box, wm = _cable_scene()
    # an identical output (no damage) preserves continuity.
    tf = v13_gates.detect_thin_flex_continuity_v20(img, img.copy(), box)
    assert tf["hard_fail"] is False
    assert tf["continuity_drop"] <= v13_gates.THIN_FLEX_CONTINUITY_DROP_MAX


def test_cut_cable_hard_fails():
    img, box, wm = _cable_scene()
    out = img.copy()
    # cut a white notch through the cable inside the bbox.
    out[55:66, 190:215] = 240
    tf = v13_gates.detect_thin_flex_continuity_v20(img, out, box)
    assert tf["hard_fail"] is True


def test_flex_reverse_alpha_returns_valid_or_none():
    img, box, wm = _cable_scene()
    out = v18_patch.thin_flex_reverse_alpha_line_preserve(img, box, wm, None)
    assert out is None or (out.shape == img.shape and out.dtype == img.dtype)


def test_flex_reverse_alpha_preserves_cable_when_published():
    img, box, wm = _cable_scene()
    out = v18_patch.thin_flex_reverse_alpha_line_preserve(img, box, wm, None)
    if out is None:
        return  # asset-dependent; skipped if no candidate
    tf = v13_gates.detect_thin_flex_continuity_v20(img, out, box)
    assert tf["hard_fail"] is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
