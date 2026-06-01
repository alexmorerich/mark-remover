#!/usr/bin/env python3
"""V19 — alpha alignment tests (patch plan §6.1)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunsky_reverse_alpha as ra  # noqa: E402


def _synthetic_watermarked(shift_x=0, scale=1.0):
    """Render the canonical glyph alpha onto a textured background, optionally
    shifted / scaled, so alignment has something real to lock onto."""
    asset = ra.load_alpha_asset(force=True)
    alpha = asset["alpha"]
    H, W = 120, 600
    rng = np.random.RandomState(3)
    bg = rng.randint(60, 200, (H, W, 3)).astype(np.uint8)
    gw, gh = int(240 * scale), int(36 * scale)
    glyph = cv2.resize(alpha, (gw, gh))
    x0 = (W - gw) // 2 + shift_x
    y0 = (H - gh) // 2
    a = np.zeros((H, W), np.float32)
    a[y0:y0 + gh, x0:x0 + gw] = glyph
    logo = np.array([180.0, 180.0, 180.0], np.float32)
    a3 = np.clip(a, 0, ra.MAX_ALPHA)[:, :, None]
    wm = np.clip(a3 * logo + (1 - a3) * bg, 0, 255).astype(np.uint8)
    # The detected box is the centered nominal location (no shift knowledge).
    box = ((W - 240) // 2, (H - 36) // 2, 240, 36)
    return wm, box


def test_fixed_placement_inside_detected_box():
    wm, box = _synthetic_watermarked()
    pl = ra.fixed_alpha_map(wm, box)
    assert pl is not None
    assert pl.source == "fixed"
    bx, by, bw, bh = pl.bbox
    assert (bx, by, bw, bh) == box


def test_aligned_beats_or_equals_fixed_on_shift():
    # A shifted watermark: the aligned placement should not be worse at locating
    # the real glyph structure than the naive fixed placement.
    wm, box = _synthetic_watermarked(shift_x=5)
    fixed = ra.fixed_alpha_map(wm, box)
    aligned = ra.aligned_alpha_map(wm, box)
    assert fixed is not None and aligned is not None
    # Applying each, the aligned candidate's residual should be <= fixed + slack.
    f_resid = ra.residual_confidence(
        ra.apply_placement(wm, fixed), fixed.bbox, fixed.alpha_map)
    a_resid = ra.residual_confidence(
        ra.apply_placement(wm, aligned), aligned.bbox, aligned.alpha_map)
    assert a_resid <= f_resid + 0.15


def test_reverse_alpha_picks_lower_residual_placement():
    wm, box = _synthetic_watermarked(shift_x=4)
    res = ra.repair_sunsky_reverse_alpha(wm, box, allow_thin_cleanup=True)
    assert res.image is not None
    # Residual after recovery must improve on the untouched band.
    assert res.residual_confidence_after_cleanup <= \
        res.residual_confidence_original + 1e-6
    assert res.source in ("fixed", "aligned")
