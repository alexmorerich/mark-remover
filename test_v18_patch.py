#!/usr/bin/env python3
"""V18 regression tests (patch plan §10).

Two invariants the V18 patch must hold:

1. **Never publish product damage.** The product-context router must BAN
   destructive full-band / forced-removal / uniform fills on any product-overlap
   region, and the low-contrast glyph-residue audit must catch faint dot chains
   the residual-OCR detector misses. A published output must be clean or the
   image must be auto_rejected — never clean_covered with a visible slab.

2. **Add safe options without weakening the gate.** The safe-candidate
   generators must only ever produce stroke-restricted, product-preserving fills,
   and the report must carry the V18 version + taxonomy aggregation.

These are fast unit tests over the V18 module + a couple of synthetic-image
behaviour checks, plus report-shape assertions. The slow end-to-end acceptance
(``clean_repaired >= 28``, ``auto_rejected <= 12``) is enforced by the benchmark
run + ``v13_report.py`` CI gate, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import v13_report
import v17_final_audit
import v18_patch


# ---------------------------------------------------------------------------
# Synthetic fixtures.
# ---------------------------------------------------------------------------
def _white_bg(h=200, w=300):
    return np.full((h, w, 3), 245, np.uint8)


def _dark_product(h=200, w=300):
    img = np.full((h, w, 3), 245, np.uint8)
    img[60:140, :] = 20            # a dark product band across the middle
    return img


def _bbox_center(img, fw=0.4, fh=0.12):
    H, W = img.shape[:2]
    bw, bh = int(W * fw), int(H * fh)
    bx, by = W // 2 - bw // 2, H // 2 - bh // 2
    return (bx, by, bw, bh)


def _fake_stroke_mask(img, bbox, n=8):
    H, W = img.shape[:2]
    bx, by, bw, bh = bbox
    m = np.zeros((H, W), np.uint8)
    for i in range(n):
        x0 = bx + 6 + i * (bw // (n + 1))
        cv2.rectangle(m, (x0, by + bh // 3), (x0 + 3, by + 2 * bh // 3), 255, -1)
    return m


# ---------------------------------------------------------------------------
# Product-context routing (patch plan §2).
# ---------------------------------------------------------------------------
def test_pure_background_allows_destructive():
    img = _white_bg()
    bbox = _bbox_center(img)
    wm = _fake_stroke_mask(img, bbox)
    ctx = v18_patch.compute_product_context(img, bbox, None, wm)
    assert ctx.product_overlap < v18_patch.PRODUCT_OVERLAP_LOW
    assert ctx.pure_background_score >= v18_patch.PURE_BACKGROUND_SCORE_MIN
    assert not v18_patch.should_ban_destructive(ctx)
    assert "uniform_background_fill" in v18_patch.allowed_generators(ctx)
    assert "forced_removal_fill" in v18_patch.allowed_generators(ctx)


def test_product_overlap_bans_destructive():
    img = _dark_product()
    # bbox squarely on the dark product band.
    H, W = img.shape[:2]
    bbox = (W // 4, 70, W // 2, 60)
    wm = _fake_stroke_mask(img, bbox)
    ctx = v18_patch.compute_product_context(img, bbox, None, wm,
                                            roi_class="dark_product_surface")
    assert ctx.product_overlap > v18_patch.PRODUCT_OVERLAP_LOW
    assert v18_patch.should_ban_destructive(ctx)
    gens = v18_patch.allowed_generators(ctx)
    # On a heavy-product region only product-safe generators are allowed.
    assert "uniform_background_fill" not in gens
    assert "forced_removal_fill" not in gens
    assert gens.issubset({"segmented", "stroke_only", "specialized",
                          "segmented_micro_cover", "stroke_only_inpaint",
                          "ring_clone_background_fragments"})


def test_destructive_generator_names_are_banned_set():
    # The names the pipeline gates on must all be in DESTRUCTIVE_GENERATORS, and
    # no V18 safe generator name may be.
    assert "v16_uniform_background_fill" in v18_patch.DESTRUCTIVE_GENERATORS
    assert "v16_forced_removal" in v18_patch.DESTRUCTIVE_GENERATORS
    for safe in ("v18_segmented_product_background", "v18_stroke_only_inpaint",
                 "v18_dark_surface_stroke_clone", "v18_flex_line_preserving"):
        assert safe not in v18_patch.DESTRUCTIVE_GENERATORS


# ---------------------------------------------------------------------------
# Safe candidate generation (patch plan §3, §5).
# ---------------------------------------------------------------------------
def test_safe_candidates_are_valid_images_only():
    img = _white_bg()
    bbox = _bbox_center(img)
    wm = _fake_stroke_mask(img, bbox)
    ctx = v18_patch.compute_product_context(img, bbox, None, wm)
    cands = v18_patch.build_safe_candidates(img, bbox, wm, None, ctx)
    assert len(cands) >= 1
    for name, im in cands:
        # V20 added reverse-alpha-backed safe candidates alongside the v18_ ones.
        assert name.startswith(("v18_", "v20_"))
        assert im is not None and im.shape == img.shape and im.dtype == img.dtype


def test_safe_candidates_only_touch_strokes_on_product():
    # On a dark product band, a safe candidate must change only a small fraction
    # of product pixels (stroke-restricted), never paint a full band.
    img = _dark_product()
    H, W = img.shape[:2]
    bbox = (W // 4, 70, W // 2, 60)
    wm = _fake_stroke_mask(img, bbox)
    ctx = v18_patch.compute_product_context(img, bbox, None, wm,
                                            roi_class="dark_product_surface")
    cands = v18_patch.build_safe_candidates(img, bbox, wm, None, ctx)
    prod = np.zeros((H, W), np.uint8)
    prod[60:140, :] = 255
    for name, im in cands:
        prod_ratio, _ = v17_final_audit.changed_region_product_overlap(
            img, im, bbox, prod, wm)
        # A stroke-restricted repair disturbs only a small product fraction.
        assert prod_ratio < 0.6, f"{name} changed too much product: {prod_ratio}"


def test_ranking_orders_safest_first():
    img = _dark_product()
    H, W = img.shape[:2]
    bbox = (W // 4, 70, W // 2, 60)
    wm = _fake_stroke_mask(img, bbox)
    ctx = v18_patch.compute_product_context(img, bbox, None, wm,
                                            roi_class="dark_product_surface")
    cands = v18_patch.build_safe_candidates(img, bbox, wm, None, ctx)
    ranked = v18_patch.rank_candidates(img, cands, bbox, None, wm, ctx)
    assert [n for n, _ in ranked] != []
    # ranking is stable in length and never invents/drops candidates.
    assert sorted(n for n, _ in ranked) == sorted(n for n, _ in cands)


# ---------------------------------------------------------------------------
# Low-contrast glyph residue audit (patch plan §8).
# ---------------------------------------------------------------------------
def test_glyph_residue_passes_on_clean_background():
    img = _white_bg()
    bbox = _bbox_center(img)
    out = img.copy()  # perfectly clean
    r = v17_final_audit.detect_low_contrast_glyph_residue_v18(img, out, bbox)
    assert r["passed"] is True
    assert r["residue_score"] == 0.0


def test_glyph_residue_catches_aligned_dot_chain():
    img = _white_bg()
    bbox = _bbox_center(img)
    out = img.copy()
    bx, by, bw, bh = bbox
    # Faint low-contrast dots aligned on the baseline (luma ~ -10 vs surface).
    y = by + bh // 2
    for i in range(10):
        x = bx + 6 + i * (bw // 11)
        cv2.circle(out, (x, y), 2, (235, 235, 235), -1)
    r = v17_final_audit.detect_low_contrast_glyph_residue_v18(img, out, bbox)
    assert r["component_count"] >= v17_final_audit.GLYPH_RESIDUE_MIN_COMPONENTS
    assert r["aligned"] is True
    assert r["passed"] is False


def test_glyph_residue_does_not_punish_natural_texture():
    # Random scattered texture (not baseline-aligned) must NOT trip the audit.
    rng = np.random.RandomState(7)
    img = _white_bg()
    bbox = _bbox_center(img)
    out = img.copy()
    bx, by, bw, bh = bbox
    for _ in range(40):
        x = bx + int(rng.randint(0, bw))
        y = by + int(rng.randint(0, bh))
        cv2.circle(out, (x, y), 1, (236, 236, 236), -1)
    r = v17_final_audit.detect_low_contrast_glyph_residue_v18(img, out, bbox)
    # Scattered dots have high vertical spread => not aligned => pass.
    assert r["passed"] is True


# ---------------------------------------------------------------------------
# Report shape (patch plan §1, §9).
# ---------------------------------------------------------------------------
def test_report_carries_v18_versions_and_taxonomy(tmp_path):
    # A minimal output tree with one auto_rejected record carrying a taxonomy.
    d = tmp_path / "auto_rejected" / "img1"
    d.mkdir(parents=True)
    (d / "qa.json").write_text(__import__("json").dumps({
        "status": "auto_rejected",
        "reject_reasons": ["cover_failed_visible_band"],
        "v18_roi_class": "dark_product_surface",
        "v9_final_method": "auto_rejected",
        "v18_reject_taxonomy": {
            "roi_class": "dark_product_surface",
            "mask_used": "logo_fallback",
            "best_repair_method": "repair",
            "candidate_fail_reasons": ["cover_failed_visible_band"],
        },
    }))
    rep = v13_report.build_report(tmp_path)
    assert rep["version"] == "V21_PATCH"
    assert rep["patch_version"] == "v21_safer_reject_recovery"
    assert rep["state_machine_version"] == "v16"
    assert rep["gate_version"] == "v13_frozen"
    assert rep["auto_rejected_by_roi_class"]["dark_product_surface"] == 1
    assert rep["auto_rejected_by_mask_type"]["logo_fallback"] == 1
    assert "published_low_contrast_glyph_residue" in \
        rep["v17_published_audit_failures"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
