"""
V11 Regression / Unit Lock.

Asserts the V11 honesty guarantees from the improvement plan:
  - zero / missing / NaN QA metrics can never pass (Phase 1)
  - dotted / broken-glyph residuals fail the residual gate (Phase 2)
  - product-overlap damage (bright blob, colour shift) fails closed (Phase 3)
  - plain-white backgrounds prefer ring/exact fill over Telea (Phase 4)
  - full-bbox repair is blocked when product overlap is high (Phase 3/8)
  - legitimate (non-watermark) product structure is protected
  - clean_repaired requires metrics_valid AND residual AND product gates

Run: python3 test_v11_regression.py    (or pytest)
"""
import numpy as np
import cv2

import progressive_repair as pr


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


def _ctx(img, bbox, product_mask=None, stroke_mask=None):
    roi = pr.analyze_roi(img, bbox, product_mask=product_mask)
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
# Phase 1 — QA truthfulness
# --------------------------------------------------------------------------

def test_zero_metric_qa_fails_closed():
    qa = {k: 0.0 for k in pr.REQUIRED_QA_METRICS}
    valid, reasons = pr.validate_qa_metrics(qa)
    assert not valid
    assert "qa_metrics_probably_not_computed" in reasons


def test_missing_metric_fails_closed():
    valid, reasons = pr.validate_qa_metrics({"ssim": 0.9})
    assert not valid
    assert any(r.startswith("missing_required_qa_metric") for r in reasons)


def test_nan_metric_fails_closed():
    qa = {k: 0.5 for k in pr.REQUIRED_QA_METRICS}
    qa["ssim"] = float("nan")
    valid, reasons = pr.validate_qa_metrics(qa)
    assert not valid


def test_good_metrics_pass_validation():
    qa = {"ssim": 0.98, "color_delta": 3.0, "boundary_jump": 2.0,
          "seam_delta": 0.01, "cover_visibility": 0.02, "product_damage": 0.0,
          "residual_template_corr": 0.05, "residual_text_component": 0.02,
          "residual_improvement": 0.9}
    valid, reasons = pr.validate_qa_metrics(qa)
    assert valid and reasons == []


# --------------------------------------------------------------------------
# Phase 2 — dotted / broken-glyph residual
# --------------------------------------------------------------------------

def test_dot_chain_residual_fails():
    img, bbox = _white_with_watermark()
    ctx = _ctx(img, bbox)
    bx, by, bw, bh = bbox
    # "repair" that leaves a horizontal chain of gray dots / fragments
    dot = _ring_fill(img, bbox)
    y = by + bh // 2
    for x in range(bx + 4, bx + bw - 4, 9):
        cv2.circle(dot, (x, y), 1, (90, 90, 90), -1)
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("dot", dot))
    assert not qa.passed
    assert not qa.residual_pass


def test_clean_fill_passes_residual():
    img, bbox = _white_with_watermark()
    ctx = _ctx(img, bbox)
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("clean", _ring_fill(img, bbox)))
    assert qa.metrics_valid
    assert qa.residual_pass
    assert qa.passed


# --------------------------------------------------------------------------
# Phase 3 — product damage / bright blob
# --------------------------------------------------------------------------

def _dark_product_with_watermark(bbox=(70, 60, 150, 22)):
    np.random.seed(7)
    img = _noise(np.full((180, 320, 3), 18, np.uint8), 2)  # dark product
    bx, by, bw, bh = bbox
    cv2.putText(img, "sunsky-online.com", (bx + 2, by + bh - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
                cv2.LINE_AA)
    product = np.full(img.shape[:2], 255, np.uint8)  # all product
    return img, bbox, product


def test_product_bright_blob_fails():
    img, bbox, product = _dark_product_with_watermark()
    ctx = _ctx(img, bbox, product_mask=product)
    bx, by, bw, bh = bbox
    # Damage: paste a bright white patch onto the dark product surface.
    bad = img.copy()
    bad[by:by + bh, bx:bx + bw] = 235
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("blob", bad))
    assert not qa.passed
    assert not qa.product_gate_pass


def test_product_metrics_detect_overlap():
    img, bbox, product = _dark_product_with_watermark()
    bad = img.copy()
    bx, by, bw, bh = bbox
    bad[by:by + bh, bx:bx + bw] = 235
    m = pr.compute_product_damage_metrics(img, bad, bbox, product)
    assert m["product_overlap"] > 0.5
    assert m["product_color_delta_lab"] > pr.PRODUCT_COLOR_DELTA_MAX
    ok, reason = pr.product_gate(m)
    assert not ok and reason


# --------------------------------------------------------------------------
# Phase 4 — plain white prefers ring fill over Telea
# --------------------------------------------------------------------------

def test_plain_white_prefers_ring_fill_over_telea():
    img, bbox = _white_with_watermark()
    ctx = _ctx(img, bbox)
    assert ctx.roi_analysis.roi_class in pr.PLAIN_BG_CLASSES
    cand, trace = pr.repair_image_progressively(ctx)
    method = cand.metadata.get("final_method")
    # A real-pixel / statistical fill must win on plain white — never Telea
    # or an adaptive cover.
    assert method != "opencv_telea_inpaint"
    assert method != "final_adaptive_cover"
    assert cand.metadata.get("qa", {}).get("passed")


def test_plain_white_fast_path_applicable():
    img, bbox = _white_with_watermark()
    ctx = _ctx(img, bbox)
    tool = pr._TOOL_INDEX["plain_white_fast_path"]
    assert tool.is_applicable(ctx)
    out = tool.apply(ctx)
    assert out is not None
    qa = pr.run_local_qa(ctx, out)
    assert qa.passed


# --------------------------------------------------------------------------
# Phase 3/8 — full-bbox repair blocked on high product overlap
# --------------------------------------------------------------------------

def test_full_bbox_cover_blocked_when_product_overlap_high():
    # Textured dark product fully under the watermark.
    img = _noise(np.full((180, 320, 3), 30, np.uint8), 12)
    bbox = (70, 60, 150, 22)
    bx, by, bw, bh = bbox
    cv2.putText(img, "sunsky-online.com", (bx + 2, by + bh - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    product = np.full(img.shape[:2], 255, np.uint8)
    ctx = _ctx(img, bbox, product_mask=product)
    blocked, ov, ed, lls = pr._check_product_overlap_guard(ctx)
    assert ov > 0.5
    assert blocked  # high overlap + structure => block non-stroke tools


# --------------------------------------------------------------------------
# Phase 6 — legitimate product text protected (changed-area gate)
# --------------------------------------------------------------------------

def test_repair_area_too_large_fails_on_product():
    img, bbox, product = _dark_product_with_watermark()
    bx, by, bw, bh = bbox
    # Repair that changes far more than the watermark footprint, with a small
    # but real luma shift (18 -> 30, diff 12 > 6) so the change registers
    # everywhere it is applied, while staying under the colour-delta (~4.7
    # Lab) and bright-blob (|12| < 25) gates — so ONLY the area-ratio gate
    # should trip.
    bad = img.copy()
    bad[by - 15:by + bh + 15, bx - 15:bx + bw + 15] = 30
    m = pr.compute_product_damage_metrics(img, bad, bbox, product)
    # The changed area spills well past the watermark footprint on a product
    # surface — the Phase 8 area-ratio metric must register it and the gate
    # must reject (exact reason depends on which threshold trips first).
    assert m["changed_area_ratio"] > pr.CHANGED_AREA_RATIO_MAX
    ok, reason = pr.product_gate(m)
    assert not ok and reason


# --------------------------------------------------------------------------
# Integration — clean_repaired requires all gates
# --------------------------------------------------------------------------

def test_clean_repaired_requires_metrics_residual_and_product_gates():
    img, bbox = _white_with_watermark()
    ctx = _ctx(img, bbox)
    qa = pr.run_local_qa(ctx, pr.RepairCandidate("clean", _ring_fill(img, bbox)))
    assert qa.passed == (qa.metrics_valid and qa.residual_pass and
                         qa.product_gate_pass)


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
