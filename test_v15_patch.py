"""V15 — cover-quality patch unit tests.

Lock the three V15 primitives: mask widening (covers more, never runs away or
eats product), the ghost-aware full-region residual verdict (catches a faint
surviving watermark via the canonical template, ignores generic product
texture), and the dark-stroke cover (hides without a light fill)."""
import cv2
import numpy as np

import v15_patch


def _white(h=120, w=400, val=242):
    return np.full((h, w, 3), val, np.uint8)


def _text(img, s, org, scale=0.8, color=(150, 150, 150), thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)
    return img


def test_widen_text_mask_grows_but_bounded():
    img = _white()
    _text(img, "sunsky-online.com", (60, 70))
    bbox = (60, 48, 240, 34)
    # A tight mask that covers only the first few glyphs.
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[52:78, 64:140] = 255
    base = int((mask > 0).sum())
    wide = v15_patch.widen_text_mask(img, bbox, mask)
    assert wide is not None
    wide_area = int((wide > 0).sum())
    # Widening must add coverage…
    assert wide_area >= base
    # …but never run away (Guard 3: <= 3x + 200).
    assert wide_area <= base * 3 + 200


def test_widen_skips_busy_product_band():
    # A high-edge-density band (busy product) must NOT be widened (Guard 1).
    img = _white()
    rng = np.random.RandomState(0)
    img[40:90, 40:360] = rng.randint(0, 255, (50, 320, 3), dtype=np.uint8)
    bbox = (40, 40, 320, 50)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[55:75, 60:120] = 255
    wide = v15_patch.widen_text_mask(img, bbox, mask)
    # Returns the original mask unchanged on a busy band.
    assert int((wide > 0).sum()) == int((mask > 0).sum())


def test_full_region_residual_passes_clean_cover():
    img = _white()
    _text(img, "sunsky-online.com", (60, 70))
    bbox = (60, 48, 240, 34)
    # A clean cover: the text region replaced by plain background.
    cover = img.copy()
    cover[40:82, 50:312] = 242
    assert v15_patch.full_region_residual_ok(img, cover, bbox) is True


def test_full_region_residual_catches_ghost():
    img = _white()
    _text(img, "sunsky-online.com", (60, 70))
    bbox = (60, 48, 240, 34)
    # A "cover" that only faded the text (a ghost remains) — must be rejected.
    cover = img.copy()
    roi = cover[40:82, 50:312].astype(np.float32)
    bg = 242.0
    cover[40:82, 50:312] = np.clip(bg + (roi - bg) * 0.5, 0, 255).astype(np.uint8)
    assert v15_patch.full_region_residual_ok(img, cover, bbox) is False


def test_full_region_residual_ignores_product_texture():
    # A textured (cable-like) surface with NO watermark must pass — the verdict
    # keys on the watermark template, not generic high-pass texture.
    rng = np.random.RandomState(1)
    img = _white(val=40)
    img[50:74, 40:360] = rng.randint(20, 90, (24, 320, 3), dtype=np.uint8)
    bbox = (40, 50, 320, 24)
    cover = img.copy()  # identical: no watermark, nothing removed
    assert v15_patch.full_region_residual_ok(img, cover, bbox) is True


def test_dark_stroke_cover_hides_mark_on_dark():
    img = _white(val=30)  # dark surface
    _text(img, "sunsky", (60, 70), color=(200, 200, 200))
    bbox = (60, 48, 180, 34)
    mask = np.zeros(img.shape[:2], np.uint8)
    # mask the bright glyph pixels
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask[(g > 120)] = 255
    out = v15_patch.dark_stroke_cover(img, bbox, mask)
    assert out is not None
    # The bright strokes should be gone (region now near the dark surface).
    region = cv2.cvtColor(out[48:82, 60:240], cv2.COLOR_BGR2GRAY)
    assert float((region > 120).mean()) < 0.05


def test_is_dark_surface():
    assert v15_patch.is_dark_surface(_white(val=20), (10, 10, 80, 40)) is True
    assert v15_patch.is_dark_surface(_white(val=240), (10, 10, 80, 40)) is False
