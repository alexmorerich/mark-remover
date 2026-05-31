"""V13 — unit tests for the new detectors (dot-chain v2, visible patch shape,
product silhouette, product overlap, protected text)."""
import numpy as np
import cv2
import v13_gates as v


def _plain(h=160, w=360, val=240):
    return np.full((h, w, 3), val, np.uint8)


# --- dot-chain v2 -----------------------------------------------------------
def test_dot_chain_clean_passes():
    o = _plain(); c = o.copy()
    bbox = (120, 60, 120, 24)
    r = v.detect_residual_dot_chain_v2(o, c, bbox)
    assert r["passed"]
    assert r["dot_chain_score"] == 0.0


def test_dot_chain_aligned_fragments_flagged():
    o = _plain()
    c = o.copy()
    # a horizontal line of aligned dark fragments = readable residue
    for i, x in enumerate(range(125, 235, 12)):
        cv2.rectangle(c, (x, 70), (x + 5, 78), (60, 60, 60), -1)
    bbox = (120, 60, 120, 24)
    r = v.detect_residual_dot_chain_v2(o, c, bbox)
    assert r["component_count"] >= 3
    assert not r["passed"]


# --- visible patch shape ----------------------------------------------------
def test_visible_patch_clean_passes():
    o = _plain(); c = o.copy()
    bbox = (120, 60, 120, 24)
    m = v.detect_visible_patch_shape_v13(o, c, bbox)
    ok, _ = v.visible_patch_gate(m, repaired=True)
    assert ok


def test_visible_patch_rectangle_flagged():
    o = _plain(); c = o.copy()
    cv2.rectangle(c, (120, 60), (240, 84), (150, 150, 150), -1)
    bbox = (120, 60, 120, 24)
    m = v.detect_visible_patch_shape_v13(o, c, bbox)
    ok, _ = v.visible_patch_gate(m, repaired=True)
    assert not ok


# --- product silhouette -----------------------------------------------------
def test_silhouette_bright_blob_on_dark_flagged():
    o = np.full((160, 360, 3), 25, np.uint8)
    c = o.copy()
    cv2.rectangle(c, (130, 64), (180, 90), (230, 230, 230), -1)
    bbox = (120, 60, 120, 32)
    pm = np.zeros((160, 360), np.uint8)
    pm[60:92, 120:240] = 255
    r = v.detect_product_silhouette_damage_v13(o, c, pm, bbox)
    assert not r["passed"]


def test_silhouette_clean_passes():
    o = np.full((160, 360, 3), 25, np.uint8)
    c = o.copy()
    bbox = (120, 60, 120, 32)
    pm = np.zeros((160, 360), np.uint8)
    pm[60:92, 120:240] = 255
    r = v.detect_product_silhouette_damage_v13(o, c, pm, bbox)
    assert r["passed"]


# --- product overlap routing ------------------------------------------------
def test_overlap_plain_white_no_override():
    o = _plain()
    bbox = (120, 60, 120, 24)
    r = v.estimate_product_overlap_v13(o, bbox, None)
    assert r["override_class"] is None


def test_overlap_dark_surface_override():
    o = np.full((160, 360, 3), 30, np.uint8)
    bbox = (120, 60, 120, 24)
    r = v.estimate_product_overlap_v13(o, bbox, None)
    assert r["override_class"] in ("dark_product_surface", "thin_flex_cable",
                                   "complex_product_detail", "text_or_label_area")
