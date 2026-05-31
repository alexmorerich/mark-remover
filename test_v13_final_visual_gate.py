"""V13 — the final visual publish gate must accept clean output and reject
visually-bad output, for BOTH repaired and covered."""
import numpy as np
import cv2
import v13_gates as v


def _plain(h=200, w=400, val=240):
    return np.full((h, w, 3), val, np.uint8)


GOOD_QA = {"metrics_valid": True, "residual_pass": True,
           "product_gate_pass": True}


def test_clean_output_passes_repaired():
    o = _plain()
    c = o.copy()  # nothing changed → nothing visible
    bbox = (150, 80, 150, 30)
    verdict = v.final_visual_publish_gate_v13(o, c, bbox, None, GOOD_QA,
                                              repaired=True)
    assert verdict.publish_ok, verdict.reject_reasons
    assert verdict.status == "clean_repaired"


def test_hard_rectangular_patch_rejected():
    o = _plain()
    c = o.copy()
    cv2.rectangle(c, (150, 80), (300, 110), (120, 120, 120), -1)
    bbox = (150, 80, 150, 30)
    verdict = v.final_visual_publish_gate_v13(o, c, bbox, None, GOOD_QA,
                                              repaired=True)
    assert not verdict.publish_ok
    assert verdict.status == "clean_covered"


def test_gate_applies_to_covered_too():
    # A covered output with a strong bright block on a dark surface must fail
    # even under the looser covered thresholds.
    o = np.full((200, 400, 3), 30, np.uint8)
    c = o.copy()
    cv2.rectangle(c, (150, 80), (300, 110), (220, 220, 220), -1)
    bbox = (150, 80, 150, 30)
    verdict = v.final_visual_publish_gate_v13(o, c, bbox, None, GOOD_QA,
                                              repaired=False)
    assert not verdict.publish_ok


def test_invalid_metrics_block_publish():
    o = _plain(); c = o.copy()
    bbox = (150, 80, 150, 30)
    verdict = v.final_visual_publish_gate_v13(
        o, c, bbox, None, {"metrics_valid": False, "residual_pass": True,
                           "product_gate_pass": True}, repaired=True)
    assert not verdict.publish_ok
    assert "metrics_invalid" in verdict.reject_reasons


def test_verdict_record_has_all_subgates():
    o = _plain(); c = o.copy()
    bbox = (150, 80, 150, 30)
    rec = v.final_visual_publish_gate_v13(o, c, bbox, None, GOOD_QA,
                                          repaired=True).to_record()
    for k in ("v13_dot_chain_pass", "v13_visible_patch_pass",
              "v13_rectangular_band_pass", "v13_product_damage_pass",
              "v13_silhouette_pass", "v13_protected_text_pass"):
        assert k in rec
