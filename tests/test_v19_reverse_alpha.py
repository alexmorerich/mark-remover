#!/usr/bin/env python3
"""V19 — reverse-alpha math + asset tests (patch plan §6.1)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunsky_reverse_alpha as ra  # noqa: E402


def test_alpha_asset_loads_with_nonempty_glyph():
    asset = ra.load_alpha_asset(force=True)
    assert asset is not None, "solve the asset first: scripts/sunsky_alpha_solve.py"
    alpha = asset["alpha"]
    assert alpha.dtype == np.float32
    assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0
    # Must carry a real glyph + halo, not a blank map.
    assert int((alpha > 0.05).sum()) >= 8
    assert float(alpha.max()) > 0.1


def test_apply_reverse_alpha_only_touches_masked_pixels():
    img = np.full((40, 200, 3), 230, np.uint8)
    alpha = np.zeros((40, 200), np.float32)
    alpha[10:20, 30:120] = 0.4            # a band of overlay
    out = ra.apply_reverse_alpha(img, alpha, (180.0, 180.0, 180.0))
    # Pixels with alpha < MIN_ALPHA are byte-identical.
    untouched = alpha < ra.MIN_ALPHA
    assert np.array_equal(out[untouched], img[untouched])
    # Masked pixels did change.
    touched = alpha >= ra.MIN_ALPHA
    assert not np.array_equal(out[touched], img[touched])


def test_reverse_alpha_recovers_known_blend():
    # Build a synthetic watermarked image from a known original + overlay.
    rng = np.random.RandomState(0)
    original = rng.randint(40, 200, (36, 240, 3)).astype(np.uint8)
    asset = ra.load_alpha_asset(force=True)
    alpha = asset["alpha"]
    logo = np.array([180.0, 180.0, 180.0], np.float32)
    a3 = np.clip(alpha, 0, ra.MAX_ALPHA)[:, :, None]
    watermarked = (a3 * logo + (1 - a3) * original.astype(np.float32))
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    recovered = ra.apply_reverse_alpha(watermarked, alpha, (180.0, 180.0, 180.0))
    # Recovery should be much closer to the original than the watermarked image.
    err_before = np.abs(watermarked.astype(int) - original.astype(int)).mean()
    err_after = np.abs(recovered.astype(int) - original.astype(int)).mean()
    assert err_after < err_before


def test_missing_asset_returns_graceful_result(monkeypatch):
    monkeypatch.setattr(ra, "load_alpha_asset", lambda force=False: None)
    img = np.full((40, 200, 3), 230, np.uint8)
    res = ra.repair_sunsky_reverse_alpha(img, (30, 10, 120, 20))
    assert res.pass_local_safety is False
    assert res.reject_reason == "alpha_asset_missing"
    assert res.image is None


def test_thin_cleanup_mask_is_small_not_full_bbox():
    asset = ra.load_alpha_asset(force=True)
    img = np.full((60, 300, 3), 230, np.uint8)
    pl = ra.fixed_alpha_map(img, (40, 15, 220, 30), asset)
    assert pl is not None
    mask = ra._glyph_residual_mask(img.shape, pl)
    box_area = 220 * 30
    glyph_area = int((mask > 0).sum())
    # The cleanup footprint is the dilated glyph, far smaller than the full bbox.
    assert glyph_area < box_area * 0.85
    assert glyph_area > 0
