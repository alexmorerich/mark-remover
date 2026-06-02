#!/usr/bin/env python3
"""V20 — reverse-alpha variant beam (patch plan §Patch 3)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunsky_reverse_alpha as ra  # noqa: E402
import v18_patch  # noqa: E402


def _white_with_overlay():
    """A white-background crop with a faint semi-transparent gray band that
    mimics the sunsky overlay footprint the alpha asset targets."""
    img = np.full((120, 400, 3), 240, np.uint8)
    asset = ra.load_alpha_asset(force=True)
    if asset is None:
        return img, None
    box = (60, 40, 280, 36)
    bx, by, bw, bh = box
    a = cv2.resize(asset["alpha"], (bw, bh), interpolation=cv2.INTER_AREA)
    logo = np.array(asset["logo_bgr"], np.float32)
    region = img[by:by + bh, bx:bx + bw].astype(np.float32)
    a3 = a[:, :, None]
    blended = a3 * logo.reshape(1, 1, 3) + (1 - a3) * region
    img[by:by + bh, bx:bx + bw] = np.clip(blended, 0, 255).astype(np.uint8)
    return img, box


def test_variant_names_are_stable_and_deterministic():
    assert len(ra.VARIANT_NAMES) == 7
    img, box = _white_with_overlay()
    if box is None:
        return
    beam1 = ra.build_variant_beam(img, box)
    beam2 = ra.build_variant_beam(img, box)
    # Deterministic: same names in same order across two calls.
    assert [n for n, _ in beam1] == [n for n, _ in beam2]


def test_beam_variants_are_valid_recovery_images():
    img, box = _white_with_overlay()
    if box is None:
        return
    beam = ra.build_variant_beam(img, box)
    for name, res in beam:
        assert name in ra.VARIANT_NAMES
        assert res.image is not None
        assert res.image.shape == img.shape and res.image.dtype == img.dtype
        # A recovery removes overlay structure -> residual no worse than original.
        assert res.residual_confidence_after_cleanup <= \
            res.residual_confidence_original + 1e-6


def test_beam_recovers_overlay_on_white_background():
    img, box = _white_with_overlay()
    if box is None:
        return
    beam = ra.build_variant_beam(img, box)
    assert len(beam) >= 1, "at least one variant should pass the local pre-screen"
    bx, by, bw, bh = box
    name, res = beam[0]
    before = cv2.cvtColor(img[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(res.image[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2GRAY)
    # The recovered footprint is brighter (closer to white background) than the
    # darkened overlay band.
    assert float(after.mean()) >= float(before.mean()) - 1.0


def test_beam_does_not_modify_all_product_pixels():
    # Reverse-alpha must touch only the overlay footprint, not the whole bbox.
    img, box = _white_with_overlay()
    if box is None:
        return
    beam = ra.build_variant_beam(img, box)
    for name, res in beam:
        assert res.changed_product_ratio <= ra.MAX_CHANGED_PRODUCT_RATIO


def test_v18_beam_wrapper_returns_candidates_and_record():
    img, box = _white_with_overlay()
    if box is None:
        return
    ctx = v18_patch.compute_product_context(img, box, None, None)
    cands, rec = v18_patch.reverse_alpha_variant_beam(img, box, None, None, ctx)
    assert "reverse_alpha_variants_attempted" in rec
    assert rec["reverse_alpha_variants_attempted"] == len(ra.VARIANT_NAMES)
    assert rec["reverse_alpha_variants_passed"] == len(cands)
    for name, im in cands:
        assert im is not None and im.shape == img.shape


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
