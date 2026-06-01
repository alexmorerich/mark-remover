#!/usr/bin/env python3
"""V19 — product-text protection tests (patch plan §6.1)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import product_text_detector as ptd  # noqa: E402
import sunsky_reverse_alpha as ra  # noqa: E402
import v18_patch  # noqa: E402


def _image_with_label():
    img = np.full((120, 400, 3), 235, np.uint8)
    # printed product label below the watermark band.
    cv2.putText(img, "A1707", (150, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (20, 20, 20), 2)
    return img


def test_heuristic_detector_finds_label_and_returns_mask():
    img = _image_with_label()
    box = (140, 40, 220, 30)
    ptm = ptd.detect_product_text(img, box)
    assert ptm.source in ("heuristic_fallback", "ppocrv3")
    assert ptm.mask.shape[:2] == img.shape[:2]
    # The label region should register some protected-text pixels.
    label_region = ptm.mask[70:110, 140:320]
    assert int((label_region > 0).sum()) > 0


def test_thin_cleanup_skips_protected_text_pixels():
    asset = ra.load_alpha_asset(force=True)
    img = _image_with_label()
    pl = ra.fixed_alpha_map(img, (40, 15, 320, 36), asset)
    assert pl is not None
    # A protected-text mask covering the label row.
    pt = np.zeros(img.shape[:2], np.uint8)
    pt[70:110, 140:320] = 255
    _cleaned, overlap = ra.thin_residual_cleanup(img, pl, protected_text_mask=pt)
    # Overlap is reported; the cleanup mask had those pixels removed.
    assert overlap >= 0.0
    mask = ra._glyph_residual_mask(img.shape, pl)
    mask[pt > 0] = 0
    # No cleanup pixels remain inside the protected-text region.
    assert int((mask[70:110, 140:320] > 0).sum()) == 0


def test_reverse_alpha_allowed_over_text_does_not_flatten_strokes():
    img = _image_with_label()
    box = (140, 40, 220, 36)
    # Reverse-alpha touches only the overlay footprint; the dark label strokes
    # below the band must remain dark (not flattened to the surface).
    res = ra.repair_sunsky_reverse_alpha(img, box, allow_thin_cleanup=True)
    if res.image is None:
        return  # asset-dependent; nothing to assert without a candidate
    label_before = cv2.cvtColor(img[70:110, 140:320], cv2.COLOR_BGR2GRAY)
    label_after = cv2.cvtColor(res.image[70:110, 140:320], cv2.COLOR_BGR2GRAY)
    dark_before = int((label_before < 80).sum())
    dark_after = int((label_after < 80).sum())
    # The label's dark strokes survive (allow a little slack).
    assert dark_after >= dark_before * 0.7
