#!/usr/bin/env python3
"""V22 — safety-preserving recovery after V21.

These tests pin the V22 contract (patch plan §Patch 1, §Patch 2, §Patch 3,
§Patch 4, §Patch 9):

  * a visible BAND on a non-white / product-like surface is a HARD P0 fail, but a
    faint band on PURE WHITE background stays cosmetic (allowed);
  * faint partial-glyph residue inside the solved Sunsky footprint is a HARD P0
    fail, while a genuinely clean output passes;
  * the V22 cleanup beam edits only small residue components inside the glyph
    footprint and never a full bbox / band;
  * low-texture product surfaces (dark-smooth / translucent-stack) are classified
    as product so destructive covers are banned on them;
  * the report exposes the V22 must-be-zero counters and stays clean.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import v13_report          # noqa: E402
import v17_final_audit      # noqa: E402
import v18_patch            # noqa: E402
import v22_patch            # noqa: E402
import sunsky_reverse_alpha as sra   # noqa: E402


# ---------------------------------------------------------------------------
# Patch 1 — visible band on a non-white / product-like surface.
# ---------------------------------------------------------------------------
def test_band_on_grey_surface_hard_fails():
    orig = np.full((200, 200, 3), 150, np.uint8)
    cv2.putText(orig, "sunsky", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (200, 200, 200), 2)
    out = orig.copy()
    cv2.rectangle(out, (40, 90), (160, 122), (185, 185, 185), -1)
    bbox = (40, 88, 120, 36)
    r = v22_patch.detect_visible_band_on_nonwhite_surface_v22(
        orig, out, bbox, None, None, "low_texture_background")
    assert r["hard_fail"] is True
    assert r["surface_is_nonwhite"] is True
    assert r["band_like"] is True


def test_band_on_pure_white_is_cosmetic_not_hard_fail():
    orig = np.full((200, 200, 3), 255, np.uint8)
    out = orig.copy()
    cv2.rectangle(out, (40, 90), (160, 122), (250, 250, 250), -1)
    r = v22_patch.detect_visible_band_on_nonwhite_surface_v22(
        orig, out, (40, 88, 120, 36), None, None, "plain_white")
    assert r["hard_fail"] is False
    assert r["surface_is_nonwhite"] is False


def test_glyph_shaped_change_on_grey_does_not_trip_band():
    orig = np.full((200, 200, 3), 150, np.uint8)
    out = orig.copy()
    cv2.putText(out, "..", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (140, 140, 140), 1)
    r = v22_patch.detect_visible_band_on_nonwhite_surface_v22(
        orig, out, (40, 88, 120, 36), None, None, "low_texture_background")
    assert r["hard_fail"] is False
    assert r["band_like"] is False


def test_audit_blocks_published_band_on_nonwhite():
    orig = np.full((200, 200, 3), 150, np.uint8)
    out = orig.copy()
    cv2.rectangle(out, (30, 90), (170, 124), (185, 185, 185), -1)
    pm = np.full((200, 200), 255, np.uint8)
    res = v17_final_audit.audit_final_output(
        orig, out, (30, 88, 140, 38), pm, None,
        final_status="clean_repaired", roi_class="low_texture_background")
    assert res.pass_p0 is False
    assert "published_visible_band_on_nonwhite_surface" in res.hard_fail_reasons


# ---------------------------------------------------------------------------
# Patch 2 — alpha-footprint partial-glyph residue.
# ---------------------------------------------------------------------------
def test_clean_output_has_no_partial_glyph_residue():
    orig = np.full((120, 240, 3), 200, np.uint8)
    out = orig.copy()                       # perfectly clean
    bbox = (40, 40, 160, 40)
    r = v22_patch.detect_alpha_footprint_residue_v22(orig, out, bbox, None, None)
    assert r["hard_fail"] is False
    assert r["score"] == 0.0


def test_partial_glyph_residue_scores_inside_footprint():
    # Build an output where faint dots survive INSIDE the alpha footprint.
    asset = sra.load_alpha_asset()
    if asset is None:
        pytest.skip("alpha asset unavailable")
    bbox = (40, 40, 160, 40)
    bx, by, bw, bh = bbox
    orig = np.full((120, 240, 3), 205, np.uint8)
    out = orig.copy()
    foot = sra.footprint_mask_for_box(orig, bbox, None, dilate_px=0)
    assert foot is not None
    ys, xs = np.where(foot[by:by + bh, bx:bx + bw] > 0)
    # Sprinkle faint low-contrast dots on the footprint baseline.
    rng = np.random.RandomState(0)
    idx = rng.choice(len(xs), size=min(40, len(xs)), replace=False)
    for k in idx:
        yy, xx = by + int(ys[k]), bx + int(xs[k])
        out[yy, xx] = (193, 193, 193)        # ~12 luma delta, low-contrast
    r = v22_patch.detect_alpha_footprint_residue_v22(orig, out, bbox, None, None)
    # The residue is detectable (non-zero score / component count).
    assert r["component_count"] >= 1


# ---------------------------------------------------------------------------
# Patch 3 — cleanup beam edits only inside the footprint, never a full bbox.
# ---------------------------------------------------------------------------
def test_v22_cleanup_beam_is_footprint_capped():
    asset = sra.load_alpha_asset()
    if asset is None:
        pytest.skip("alpha asset unavailable")
    bbox = (40, 40, 160, 40)
    bx, by, bw, bh = bbox
    orig = np.full((120, 240, 3), 205, np.uint8)
    cand = orig.copy()
    foot = sra.footprint_mask_for_box(orig, bbox, None, dilate_px=0)
    ys, xs = np.where(foot[by:by + bh, bx:bx + bw] > 0)
    for k in range(0, len(xs), 3):
        cand[by + int(ys[k]), bx + int(xs[k])] = (193, 193, 193)
    beam = sra.build_alpha_footprint_residue_cleanup_beam_v22(
        orig, cand, bbox, watermark_mask=None, product_mask=None)
    for name, img in beam:
        assert img.shape == orig.shape
        changed = (cv2.absdiff(cand, img).max(axis=2) > 3)
        # Every edit must land inside the (slightly dilated) glyph footprint.
        foot_d = cv2.dilate((foot > 0).astype(np.uint8),
                            np.ones((5, 5), np.uint8)) > 0
        outside = changed & (~foot_d)
        assert int(outside.sum()) == 0, f"{name} edited outside footprint"
        # And never a full band: edited area is a small fraction of the box.
        assert float(changed[by:by + bh, bx:bx + bw].mean()) < 0.30


# ---------------------------------------------------------------------------
# Patch 4 — low-texture product-surface classification + destructive ban.
# ---------------------------------------------------------------------------
def test_dark_smooth_surface_classified_and_bans_destructive():
    img = np.full((200, 200, 3), 30, np.uint8)      # dark smooth back cover
    bbox = (40, 80, 120, 40)
    surf = v22_patch.classify_surface_v22(img, bbox, None, None, "")
    assert surf["dark_smooth_product_surface"] is True
    ctx = v18_patch.ProductContext(dark_smooth_product_surface=True)
    assert v18_patch.should_ban_destructive(ctx) is True


def test_pure_white_surface_does_not_ban_destructive():
    img = np.full((200, 200, 3), 255, np.uint8)
    bbox = (40, 80, 120, 40)
    surf = v22_patch.classify_surface_v22(img, bbox, None, None, "plain_white")
    assert surf["dark_smooth_product_surface"] is False
    assert surf["translucent_stack_surface"] is False
    ctx = v18_patch.ProductContext()          # all-zero / pure background
    assert v18_patch.should_ban_destructive(ctx) is False


# ---------------------------------------------------------------------------
# Patch 9 — report exposes the V22 must-be-zero counters and stays clean.
# ---------------------------------------------------------------------------
def test_report_emits_v22_counters_and_stays_clean(tmp_path):
    root = tmp_path / "out"
    rep = root / "clean_repaired" / "a"
    rej = root / "auto_rejected" / "b"
    rep.mkdir(parents=True)
    rej.mkdir(parents=True)
    (rep / "qa.json").write_text(json.dumps({
        "status": "clean_repaired",
        "v9_final_method": "v22_residue_alpha_footprint_median_blend",
        "p0_gates": {}, "v17_hard_fail_reasons": [],
        "v22_surface_class": "low_texture_background",
        "v22_residue_cleanup_attempted": 1,
        "v22_alpha_footprint": {"hard_fail": False},
    }))
    (rej / "qa.json").write_text(json.dumps({
        "status": "auto_rejected",
        "reject_reasons": ["v17_published_visible_band_on_nonwhite_surface"],
        "v17_hard_fail_reasons": ["published_visible_band_on_nonwhite_surface"],
        "v22_surface_class": "translucent_stack_surface",
    }))
    report = v13_report.build_report(root)
    assert report["version"] == "V22_PATCH"
    assert "v22_published_audit_failures" in report
    for k in v13_report.V22_MUST_BE_ZERO:
        assert report["v22_published_audit_failures"][k] == 0
    assert report["all_clean"] is True
    assert report["v22"]["residue_cleanup_published"] == 1


def test_report_flags_published_v22_leak(tmp_path):
    root = tmp_path / "out"
    rep = root / "clean_repaired" / "a"
    rep.mkdir(parents=True)
    # A (synthetic) published output that still carries a V22 hard-fail signal —
    # CI must flag this as not-clean.
    (rep / "qa.json").write_text(json.dumps({
        "status": "clean_repaired",
        "v9_final_method": "x", "p0_gates": {},
        "v17_hard_fail_reasons": ["published_partial_glyph_residue"],
    }))
    report = v13_report.build_report(root)
    assert report["v22_published_audit_failures"][
        "published_partial_glyph_residue"] == 1
    assert report["v22_published_audit_failures"][
        "clean_repaired_with_any_p0_fail"] == 1
    assert report["all_clean"] is False
