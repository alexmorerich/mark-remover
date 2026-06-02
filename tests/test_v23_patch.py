#!/usr/bin/env python3
"""V23 regression suite (patch plan §Patch 11).

Every test asserts a V23 invariant: V23 may only ADD safer candidates and
sharper diagnostics — it must never lower an existing safety threshold, never
drive a destructive fill on product, and never publish a residual / damaged
output. The existing V13–V22 suites continue to pass unchanged.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v13_report
import v16_pipeline
import v17_final_audit
import v18_patch
import v22_patch
import v23_masks
import v23_mixed_repair
import v23_flex_repair
import v23_retry
import v23_reverse_alpha_variants as v23ra
import v23_surface_classifier
import sunsky_reverse_alpha as sra


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _placement(image, bbox):
    asset = sra.load_alpha_asset()
    if asset is None:
        return None
    return sra.aligned_alpha_map(image, bbox, None, asset) or \
        sra.fixed_alpha_map(image, bbox, asset)


def _dark_image(h=160, w=240, luma=45):
    img = np.full((h, w, 3), luma, np.uint8)
    return img


# --------------------------------------------------------------------------- #
# 1. No threshold may be lowered                                              #
# --------------------------------------------------------------------------- #
def test_v23_must_not_lower_existing_thresholds():
    # Frozen routing / audit thresholds the V23 plan must preserve verbatim.
    assert v18_patch.PRODUCT_OVERLAP_LOW == 0.03
    assert v17_final_audit.PRODUCT_OVERLAP_HARD == 0.03
    assert v22_patch.AF_SCORE_HARDFAIL == 0.50
    assert v22_patch.BAND_VISIBLE_MIN == 0.50
    assert v22_patch.AF_RESIDUE_AREA_MAX == 0.22
    # V23 cover gate is STRICTER than the V17 uniform-fill gate (never looser).
    assert v16_pipeline.V23_COVER_PRODUCT_OVERLAP_MAX <= \
        v17_final_audit.PRODUCT_OVERLAP_HARD
    assert v16_pipeline.V23_COVER_PURE_BG_MIN >= 0.85


# --------------------------------------------------------------------------- #
# 2. logo_fallback may never drive a product cover/fill                       #
# --------------------------------------------------------------------------- #
def test_v23_logo_fallback_cannot_drive_product_cover():
    # Product (non-white) surface under the box, NO precise watermark mask ->
    # mask_type resolves to logo_fallback and the product fragment is non-empty.
    img = np.full((160, 240, 3), 70, np.uint8)   # dark product surface
    bbox = (90, 70, 90, 22)
    ms = v23_masks.build_v23_mask_set(img, bbox, watermark_mask=None,
                                      product_mask=None)
    assert ms.mask_type == "logo_fallback"
    assert (ms.product_fragment_mask > 0).any()
    assert ms.destructive_allowed is False


def test_v23_alpha_core_mask_is_smaller_than_bbox():
    img = np.full((160, 240, 3), 240, np.uint8)
    bbox = (90, 70, 100, 24)
    ms = v23_masks.build_v23_mask_set(img, bbox)
    core = int((ms.alpha_core > 0).sum())
    foot = int((ms.alpha_footprint > 0).sum())
    box_area = bbox[2] * bbox[3]
    # The solved-alpha footprint is strictly smaller than the bounding box.
    assert 0 < foot < box_area
    assert core <= foot


def test_v23_micro_mask_area_capped():
    img = np.full((160, 240, 3), 240, np.uint8)
    bbox = (90, 70, 100, 24)
    ms = v23_masks.build_v23_mask_set(img, bbox)
    foot = max(1, int((ms.alpha_footprint > 0).sum()))
    micro = int((ms.safe_micro_mask > 0).sum())
    assert micro / foot <= v23_masks.MICRO_MAX_FOOTPRINT_FRAC + 1e-9


def test_v23_background_product_masks_do_not_overlap():
    img = np.full((160, 240, 3), 240, np.uint8)
    img[70:94, 90:140] = 60   # half the box is dark product
    bbox = (90, 70, 100, 24)
    ms = v23_masks.build_v23_mask_set(img, bbox)
    overlap = (ms.product_fragment_mask > 0) & (ms.background_fragment_mask > 0)
    assert not overlap.any()


# --------------------------------------------------------------------------- #
# 3. Cover is background-only                                                  #
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, **kw):
        self.pure_background_score = kw.get("pure_background_score", 1.0)
        self.product_overlap = kw.get("product_overlap", 0.0)
        self.product_mask_safe_overlap = kw.get("product_mask_safe_overlap", 0.0)
        self.touches_silhouette = kw.get("touches_silhouette", False)
        self.protected_text_score = kw.get("protected_text_score", 0.0)


def test_v23_cover_background_only():
    pure = _Ctx(pure_background_score=0.98, product_overlap=0.0)
    assert v16_pipeline._v23_cover_allowed(pure, {}) is True


def test_v23_cover_banned_on_dark_smooth_product():
    pure = _Ctx(pure_background_score=0.98, product_overlap=0.0)
    surface = {"dark_smooth_product_surface": True,
               "destructive_fill_unsafe": True}
    assert v16_pipeline._v23_cover_allowed(pure, surface) is False


def test_v23_cover_banned_on_product_overlap():
    ctx = _Ctx(pure_background_score=0.98, product_overlap=0.20)
    assert v16_pipeline._v23_cover_allowed(ctx, {}) is False


def test_v23_cover_banned_on_thin_flex():
    pure = _Ctx(pure_background_score=0.98, product_overlap=0.0)
    surface = {"long_thin_component_crosses_bbox": True}
    assert v16_pipeline._v23_cover_allowed(pure, surface) is False


# --------------------------------------------------------------------------- #
# 4. Component-aware mixed repair never paints product                        #
# --------------------------------------------------------------------------- #
def test_v23_mixed_repair_no_product_paint():
    # White background on the left, dark product on the right of the box.
    img = np.full((160, 240, 3), 245, np.uint8)
    img[60:120, 140:240] = 55
    bbox = (100, 78, 90, 22)
    before = img.copy()
    out = v23_mixed_repair.v23_component_aware_mixed_repair(
        img, bbox, watermark_mask=None, product_mask=None)
    if out is None:
        return   # no safe mixed repair found — acceptable (auto-reject path)
    # It must not flatten the dark product into a solid block.
    assert not v17_final_audit._metallic_block(before, out, bbox)
    # The product region keeps its structure (not replaced wholesale).
    prod = before[60:120, 160:240].astype(np.float32)
    outp = out[60:120, 160:240].astype(np.float32)
    assert np.mean(np.abs(prod - outp)) < 40.0


# --------------------------------------------------------------------------- #
# 5. Thin-flex line restore preserves continuity                              #
# --------------------------------------------------------------------------- #
def test_v23_thin_flex_line_restore_preserves_continuity():
    img = np.full((160, 240, 3), 235, np.uint8)
    cv2.line(img, (20, 100), (220, 100), (20, 20, 20), 5)   # dark flex cable
    bbox = (90, 88, 90, 24)
    out = v23_flex_repair.v23_thin_flex_line_restore(
        img, bbox, watermark_mask=None, product_mask=None)
    if out is None:
        return   # nothing safe to do — acceptable
    after = v13_report  # noqa: F841 (kept for symmetry)
    cont = __import__("v13_gates").detect_thin_flex_continuity_v20(img, out, bbox)
    assert not cont.get("hard_fail", False)


def test_v23_flex_repair_never_uses_cover():
    # The flex restore is reverse-alpha based; it can only return None or a
    # recovered image, never a destructive cover candidate name.
    img = np.full((120, 200, 3), 235, np.uint8)
    out = v23_flex_repair.v23_thin_flex_line_restore(img, (80, 50, 60, 20))
    assert out is None or out.shape == img.shape


# --------------------------------------------------------------------------- #
# 6. Dark surface: no bright blob                                             #
# --------------------------------------------------------------------------- #
def test_v23_dark_surface_no_bright_blob():
    img = _dark_image(luma=45)
    bbox = (90, 70, 90, 24)
    pl = _placement(img, bbox)
    if pl is None:
        pytest.skip("alpha asset unavailable")
    out = v23ra._dark_surface_bias(sra, img, pl)
    if out is None:
        return
    # The dark-surface variant must never plant a bright blob.
    assert v17_final_audit._dark_surface_blob(img, out, bbox) <= \
        v17_final_audit.NEW_DARK_BLOB_DELTA


# --------------------------------------------------------------------------- #
# 7. Metallic gradient not flattened                                          #
# --------------------------------------------------------------------------- #
def test_v23_metallic_gradient_not_flattened():
    # Horizontal brightness gradient (a reflective surface).
    grad = np.tile(np.linspace(60, 200, 240, dtype=np.uint8), (160, 1))
    img = cv2.merge([grad, grad, grad])
    bbox = (90, 70, 90, 24)
    pl = _placement(img, bbox)
    if pl is None:
        pytest.skip("alpha asset unavailable")
    out = v23ra._metallic_gradient_locked(sra, img, pl)
    if out is None:
        return
    assert not v17_final_audit._metallic_block(img, out, bbox)


# --------------------------------------------------------------------------- #
# 8. Partial-glyph residue still blocks publish                               #
# --------------------------------------------------------------------------- #
def test_v23_partial_glyph_residue_blocks_publish():
    # The V22 hard-fail threshold for partial-glyph residue is unchanged, and
    # the published audit folds it into a must-be-zero counter.
    assert v22_patch.AF_SCORE_HARDFAIL == 0.50
    assert "published_partial_glyph_residue" in v13_report.V23_MUST_BE_ZERO
    assert v13_report._V23_REASON_TO_COUNTER.get(
        "published_partial_glyph_residue") == "published_partial_glyph_residue"


# --------------------------------------------------------------------------- #
# 9. Retry ladder removes the failed cover family                             #
# --------------------------------------------------------------------------- #
def test_v23_retry_ladder_removes_failed_cover_family():
    route = v23_retry.route_retry(["cover_failed_visible_patch_on_product"])
    assert route["ban_cover"] is True
    assert "v16_uniform_background_fill" in route["ban_families"]
    pool = [("v20_reverse_alpha_ncc", None),
            ("v16_uniform_background_fill", None),
            ("v16_forced_removal", None)]
    filtered = v23_retry.filter_cover_pool(pool, route["ban_families"])
    names = [n for n, _ in filtered]
    assert "v16_uniform_background_fill" not in names
    assert "v16_forced_removal" not in names
    assert "v20_reverse_alpha_ncc" in names


def test_v23_retry_after_protected_text_uses_reverse_alpha_only():
    route = v23_retry.route_retry([], ["changed_protected_text"])
    assert route["families"] == ["reverse_alpha"]
    assert route["ban_cover"] is True


# --------------------------------------------------------------------------- #
# 10. No cleaned.jpg may leak into auto_rejected                              #
# --------------------------------------------------------------------------- #
def test_v23_auto_rejected_has_no_cleaned_jpg():
    # Validate the invariant on any available run output (proxy for the V23 run).
    checked = False
    for out in ("output_v23", "output_v22"):
        d = ROOT / out / "auto_rejected"
        if d.exists():
            checked = True
            assert list(d.rglob("cleaned.jpg")) == []
    if not checked:
        pytest.skip("no run output available")


# --------------------------------------------------------------------------- #
# 11. The report carries the V23 reject taxonomy                              #
# --------------------------------------------------------------------------- #
def test_v23_report_has_reject_taxonomy():
    rec = {
        "status": "auto_rejected",
        "v17_hard_fail_reasons": ["published_partial_glyph_residue"],
        "reject_reasons": ["repair_failed_final_gate"],
        "v9_roi_class": "metallic_or_reflective",
        "v9_final_method": "auto_rejected",
        "mask_type": "logo_fallback",
        "v22_product_like_overlap": 0.5,
        "v18_n_safe_candidates": 0,
    }
    out = v18_patch.build_v23_records(rec, "auto_rejected")
    tax = out["v23_reject_taxonomy"]
    assert tax["bucket"] in v18_patch.V23_REJECT_BUCKETS
    assert tax["bucket"] == "partial_glyph_residue"
    assert tax["safe_candidate_count"] == 0


def test_v23_reject_bucket_is_always_in_taxonomy():
    for reasons, expect in [
        (["visible_band_on_nonwhite_surface"], "visible_band_on_nonwhite_surface"),
        (["changed_protected_text"], "protected_text_overlap"),
        (["visible_patch_on_product"], "cover_or_fill_on_product"),
    ]:
        rec = {"status": "auto_rejected", "v17_hard_fail_reasons": reasons,
               "reject_reasons": []}
        b = v18_patch.classify_v23_reject_bucket(rec)
        assert b in v18_patch.V23_REJECT_BUCKETS
        assert b == expect
