#!/usr/bin/env python3
"""V21 — residue micro-cleanup beam tests (patch plan §Patch 2, §Patch 11).

The micro-cleanup beam may ONLY touch small aligned residue components inside
the glyph footprint, area-capped, and must never paint a full bbox / band /
rectangle. These tests assert the area discipline and the safety property that
the beam cannot produce a candidate the final audit would have to reject for
touching too much of the box.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunsky_reverse_alpha as ra  # noqa: E402


def _synthetic_residue(box, n_dots=8, dot_val=18):
    """A clean smooth gray background with a few tiny low-contrast paired dots
    placed along the watermark baseline inside ``box`` — exactly the faint
    residue reverse-alpha leaves behind."""
    H, W = 120, 600
    img = np.full((H, W, 3), 200, np.uint8)
    bx, by, bw, bh = box
    rng = np.random.RandomState(7)
    cy = by + bh // 2
    for i in range(n_dots):
        cx = bx + int((i + 0.5) / n_dots * bw)
        img[cy - 1:cy + 1, cx - 1:cx + 1] = 200 - dot_val
    return img


def test_residue_beam_produces_area_capped_variants():
    box = (180, 42, 240, 36)
    img = _synthetic_residue(box)
    variants = ra.build_residue_micro_cleanup_beam(img, img.copy(), box)
    names = [n for n, _ in variants]
    # We always get at least the noop control + one real cleaner when residue is
    # present, and every name is a v21 residue variant.
    assert any(n.startswith("v21_residue_micro") for n in names)
    assert all(n.startswith("v21_residue_micro") for n in names)


def test_residue_edited_area_is_small_and_inside_footprint():
    box = (180, 42, 240, 36)
    img = _synthetic_residue(box)
    foot = ra.footprint_mask_for_box(img, box, dilate_px=1)
    assert foot is not None
    mask, frac = ra._detect_residue_mask(img, box, foot)
    assert mask is not None
    # Area discipline: <= 20% of footprint AND inside the footprint.
    assert frac <= ra.RESIDUE_MAX_FOOTPRINT_FRAC
    edited = (mask > 0)
    assert np.all(foot[edited] > 0)   # never edits outside the glyph footprint


def test_large_uniform_difference_is_not_treated_as_residue():
    # A whole-box dark slab is NOT isolated residue — the detector must refuse it
    # (returns None) so the beam cannot smuggle a full-box edit through.
    box = (180, 42, 240, 36)
    img = np.full((120, 600, 3), 200, np.uint8)
    bx, by, bw, bh = box
    img[by:by + bh, bx:bx + bw] = 120   # large solid block, not small dots
    foot = ra.footprint_mask_for_box(img, box, dilate_px=1)
    mask, frac = ra._detect_residue_mask(img, box, foot)
    assert mask is None   # too much area -> refused


def test_clean_surface_yields_no_residue_candidates():
    # A perfectly clean smooth surface has nothing to micro-clean.
    box = (180, 42, 240, 36)
    img = np.full((120, 600, 3), 200, np.uint8)
    variants = ra.build_residue_micro_cleanup_beam(img, img.copy(), box)
    # No residue -> empty (the noop control is only emitted when residue exists).
    assert variants == []


def test_noop_control_excluded_from_pipeline_wrapper():
    import v18_patch
    box = (180, 42, 240, 36)
    img = _synthetic_residue(box)
    ctx = v18_patch.compute_product_context(img, box, None, None)
    base = [("v20_reverse_alpha_ncc", img.copy())]
    cands, rec = v18_patch.residue_micro_cleanup_beam(
        img, box, None, None, ctx, base)
    names = [n for n, _ in cands]
    assert "v21_residue_micro_noop_control" not in names
    assert rec["v21_residue_micro_attempted"] >= 1
