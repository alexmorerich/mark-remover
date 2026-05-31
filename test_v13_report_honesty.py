"""V13 — the honesty counters must increment whenever a bad artifact survives
into a published output, and stay zero for clean output."""
import v13_gates as v
import numpy as np
import cv2


def _plain(h=160, w=360, val=240):
    return np.full((h, w, 3), val, np.uint8)


GOOD_QA = {"metrics_valid": True, "residual_pass": True,
           "product_gate_pass": True}


def test_counters_stay_zero_for_clean():
    o = _plain(); c = o.copy()
    bbox = (120, 60, 120, 24)
    verdict = v.final_visual_publish_gate_v13(o, c, bbox, None, GOOD_QA,
                                              repaired=True)
    counters = {k: 0 for k in v.HONESTY_COUNTERS}
    v.update_honesty_counters(counters, verdict.status, verdict)
    assert all(val == 0 for val in counters.values()), counters


def test_counter_increments_for_bad_repaired():
    o = _plain(); c = o.copy()
    cv2.rectangle(c, (120, 60), (240, 84), (140, 140, 140), -1)
    bbox = (120, 60, 120, 24)
    verdict = v.final_visual_publish_gate_v13(o, c, bbox, None, GOOD_QA,
                                              repaired=True)
    counters = {k: 0 for k in v.HONESTY_COUNTERS}
    # If the gate did its job it demoted to covered; force-count as if it had
    # still been published as repaired to prove the counter wiring works.
    v.update_honesty_counters(counters, "clean_repaired", verdict)
    assert counters["clean_repaired_with_visible_patch"] == 1
    assert counters["final_publish_failures"] == 1


def test_all_counter_keys_present():
    assert "final_publish_failures" in v.HONESTY_COUNTERS
    assert len(v.HONESTY_COUNTERS) == 9
