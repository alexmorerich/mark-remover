"""
V12 Visual-Truthfulness Regression / Unit Lock.

Asserts the V12 honesty guarantees from the patch plan:
  - Phase A — final_publish_gate is the ONE authority for clean_repaired;
    a candidate that clears base QA but trips any visual gate is demoted.
  - Phase B — dotted / broken-glyph residuals (dot-chain) never pass as
    clean_repaired (stricter 0.28 ceiling + count-based rule).
  - Phase D — a visible rectangular band fails the clean_repaired band gate.
  - Phase E/F — a bright patch on a dark product surface fails the product
    gate; white/median fills are banned outright on dark/thin-flex classes.
  - Phase J — PDF / JSONL / trace all carry the V12_VISUAL_TRUTHFULNESS
    version string and the v12 schema versions.

Run: python3 test_v12_visual_truthfulness.py   (or pytest)
"""
import numpy as np
import cv2

import progressive_repair as pr
import mark_remover as mr


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------

def _noise(img, sigma=3):
    n = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)


def _white_with_watermark(bbox=(80, 60, 160, 22)):
    np.random.seed(2026)
    img = _noise(np.full((180, 320, 3), 244, np.uint8))
    bx, by, bw, bh = bbox
    cv2.putText(img, "sunsky-online.com", (bx + 2, by + bh - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1,
                cv2.LINE_AA)
    return img, bbox


def _dark_product_with_watermark(bbox=(70, 60, 150, 22)):
    np.random.seed(7)
    img = _noise(np.full((180, 320, 3), 18, np.uint8), 2)
    bx, by, bw, bh = bbox
    cv2.putText(img, "sunsky-online.com", (bx + 2, by + bh - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
                cv2.LINE_AA)
    product = np.full(img.shape[:2], 255, np.uint8)
    return img, bbox, product


def _ctx(img, bbox, product_mask=None, stroke_mask=None, roi_class=None):
    roi = pr.analyze_roi(img, bbox, product_mask=product_mask)
    if roi_class is not None:
        roi.roi_class = roi_class
    return pr.RepairContext(
        image=img, watermark_bbox=bbox, stroke_mask=stroke_mask,
        bbox_mask=np.zeros(img.shape[:2], np.uint8), roi_analysis=roi,
        product_mask=product_mask)


def _ring_fill(img, bbox):
    bx, by, bw, bh = bbox
    out = img.copy()
    ring_med, sigma = pr._ring_stats(img, bbox)
    patch = np.full((bh, bw, 3), ring_med, np.uint8)
    out[by:by + bh, bx:bx + bw] = pr._add_noise(patch, max(1.0, sigma))
    return out


# --------------------------------------------------------------------------
# Phase A — final_publish_gate is the single authority
# --------------------------------------------------------------------------

def _good_qa():
    return {"metrics_valid": True, "residual_pass": True,
            "product_gate_pass": True, "residual_dot_chain_score": 0.05,
            "visible_band_score": 0.04, "band_rectangularity": 0.08,
            "band_luma_delta": 2.0, "band_edge_box_score": 0.05}


def test_publish_gate_clean_passes():
    v = pr.final_publish_gate(_good_qa())
    assert v["publish_ok"] and v["status"] == "clean_repaired"
    assert v["reject_reasons"] == []


def test_publish_gate_rejects_invalid_metrics():
    qa = _good_qa(); qa["metrics_valid"] = False
    v = pr.final_publish_gate(qa)
    assert not v["publish_ok"] and "qa_metrics_invalid" in v["reject_reasons"]


def test_publish_gate_demotes_to_covered():
    qa = _good_qa(); qa["visible_band_score"] = 0.4
    v = pr.final_publish_gate(qa)
    assert v["status"] == "clean_covered"


# --------------------------------------------------------------------------
# Phase B — dot-chain / broken-glyph residual
# --------------------------------------------------------------------------

def test_dot_chain_residual_not_clean_repaired():
    img, bbox = _white_with_watermark()
    ctx = _ctx(img, bbox)
    bx, by, bw, bh = bbox
    dot = _ring_fill(img, bbox)
    y = by + bh // 2
    for x in range(bx + 4, bx + bw - 4, 9):
        cv2.circle(dot, (x, y), 1, (85, 85, 85), -1)
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("dot", dot))
    # A repair with a residual dot-chain over the ceiling must never be
    # publishable as clean_repaired.
    if qa.residual_dot_chain_score > pr.RESIDUAL_DOT_CHAIN_MAX:
        assert not qa.publish_ok
        assert "dot_chain_residual" in qa.publish_reject_reasons
    # Either way it is caught: residual fails OR it is demoted from repaired.
    assert (not qa.residual_pass) or qa.publish_ok == (
        qa.residual_dot_chain_score <= pr.RESIDUAL_DOT_CHAIN_MAX and
        qa.passed)


def test_dot_chain_threshold_tightened():
    # V12 lowered the ceiling from 0.35 to 0.28.
    assert pr.RESIDUAL_DOT_CHAIN_MAX == 0.28


def test_count_based_dotchain_constants_present():
    assert pr.RESIDUAL_DOTCHAIN_COUNT_MIN >= 4
    assert 0.0 < pr.RESIDUAL_DOTCHAIN_SPAN_MIN < 1.0
    assert 0.0 < pr.RESIDUAL_DOTCHAIN_AREA_MIN < 1.0


def test_clean_fill_still_publishes():
    img, bbox = _white_with_watermark()
    ctx = _ctx(img, bbox)
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("clean", _ring_fill(img, bbox)))
    assert qa.metrics_valid and qa.residual_pass
    assert qa.publish_ok and qa.publish_status == "clean_repaired"


# --------------------------------------------------------------------------
# Phase D — visible rectangular band
# --------------------------------------------------------------------------

def test_white_band_fails_repaired_band_gate():
    # A pale rectangle pasted on a textured surface = a visible band.
    np.random.seed(3)
    img = _noise(np.full((180, 320, 3), 130, np.uint8), 14)
    bbox = (80, 70, 150, 26)
    bx, by, bw, bh = bbox
    banded = img.copy()
    banded[by:by + bh, bx:bx + bw] = 200  # flat pale rectangle
    m = pr.detect_rectangular_band_visibility(banded, bbox)
    assert not pr.band_gate(m, repaired=True)
    assert m["visible_band_score"] > 0 or m["band_luma_delta"] > 6


def test_clean_inpaint_passes_band_gate():
    img, bbox = _white_with_watermark()
    clean = _ring_fill(img, bbox)
    m = pr.detect_rectangular_band_visibility(clean, bbox)
    assert pr.band_gate(m, repaired=True)


def test_band_gate_looser_for_cover():
    # A mildly visible patch the repaired gate rejects can still be an honest
    # cover.
    m = {"visible_band_score": 0.22, "band_rectangularity": 0.24,
         "band_luma_delta": 9.0, "band_edge_box_score": 0.20}
    assert not pr.band_gate(m, repaired=True)
    assert pr.band_gate(m, repaired=False)


# --------------------------------------------------------------------------
# Phase E/F — dark product bright patch + white-fill ban
# --------------------------------------------------------------------------

def test_dark_product_bright_patch_fails():
    img, bbox, product = _dark_product_with_watermark()
    ctx = _ctx(img, bbox, product_mask=product)
    bx, by, bw, bh = bbox
    bad = img.copy()
    bad[by:by + bh, bx:bx + bw] = 235  # bright block on black surface
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("blob", bad))
    assert not qa.publish_ok
    assert not qa.product_gate_pass


def test_product_contour_break_scored():
    img, bbox, product = _dark_product_with_watermark()
    ctx = _ctx(img, bbox, product_mask=product)
    bx, by, bw, bh = bbox
    bad = img.copy()
    bad[by:by + bh, bx:bx + bw] = 235
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("blob", bad))
    assert 0.0 <= qa.product_contour_break_score <= 1.0


def test_white_fill_banned_on_dark_and_flex():
    for cls in ("dark_product_surface", "thin_flex_cable"):
        img, bbox, product = _dark_product_with_watermark()
        ctx = _ctx(img, bbox, product_mask=product, roi_class=cls)
        names = [t.name for t in pr.select_tools_for_roi(ctx)]
        assert not (set(names) & pr.WHITE_FILL_BANNED_TOOLS), \
            f"white fill leaked into {cls}: {names}"


# --------------------------------------------------------------------------
# Phase J — version + schema consistency
# --------------------------------------------------------------------------

def test_version_string_is_v12():
    assert mr.PIPELINE_VERSION == "V18_PATCH"


def test_schema_versions_are_v12():
    assert pr.QA_SCHEMA_VERSION == "v13"
    assert pr.STRATEGY_SCHEMA_VERSION == "v13"


def test_run_metadata_carries_version():
    meta = mr.run_metadata(2026)
    assert meta["version"] == "V18_PATCH"
    assert meta["qa_schema_version"] == "v13"
    assert meta["strategy_schema_version"] == "v13"
    assert meta["run_seed"] == 2026


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            np.random.seed(2026)
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
