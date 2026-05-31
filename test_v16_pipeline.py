"""V16 — auto-decision state-machine tests (patch plan "Required Tests").

Drive `v16_pipeline.decide_final_status` with stubbed gate / re-check functions
so the state transitions are deterministic, and check the `v13_report.py` CI gate
fails on a dirty published output.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

import v16_pipeline
import v13_report


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------
class _Verdict:
    """Minimal FinalVisualVerdict stand-in."""
    def __init__(self, publish_ok, reject_reasons=None, **gates):
        self.publish_ok = publish_ok
        self.reject_reasons = reject_reasons or []
        defaults = dict(
            residual_pass=publish_ok, dot_chain_pass=publish_ok,
            visible_patch_pass=publish_ok, rectangular_band_pass=publish_ok,
            polygon_patch_pass=publish_ok, product_damage_pass=publish_ok,
            silhouette_pass=publish_ok, protected_text_pass=publish_ok)
        defaults.update(gates)
        for k, v in defaults.items():
            setattr(self, k, v)

    def to_record(self):
        return {}


def PASS():
    return _Verdict(True)


def FAIL_HARD():
    # a hard reason ⇒ classify_failure returns hard_fail ⇒ no near-miss rescue
    return _Verdict(False, ["product_damage"], product_damage_pass=False)


class GateStub:
    """Returns queued verdicts, then a default for everything after."""
    def __init__(self, queue, default):
        self.q = list(queue)
        self.default = default
        self.calls = 0

    def __call__(self, image, repaired, residual_pass=None):
        self.calls += 1
        return self.q.pop(0) if self.q else self.default()


def _clean_recheck(image):
    return False, {}


def _img():
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, (200, 320, 3), dtype=np.uint8)


def _mask():
    m = np.zeros((200, 320), np.uint8)
    m[96:108, 130:210] = 255
    return m


BBOX = (120, 90, 100, 28)


def _decide(gate, recheck=_clean_recheck, **kw):
    img = _img()
    return v16_pipeline.decide_final_status(
        img, BBOX, None, _mask(), {"metrics_valid": True, "residual_pass": True,
                                   "product_gate_pass": True},
        loop_repair_image=img.copy(), loop_repair_method="repair",
        roi_class="metallic_or_reflective",
        gate_fn=gate, recheck_fn=recheck, **kw)


# --------------------------------------------------------------------------
# 1 — failed repair + failed cover -> auto_rejected (not clean_covered)
# --------------------------------------------------------------------------
def test_failed_repair_and_cover_become_auto_rejected():
    res = _decide(GateStub([], FAIL_HARD))
    assert res.status == "auto_rejected"
    assert res.publish_ok is False
    assert res.candidate_publish_failures > 0


# --------------------------------------------------------------------------
# 2 — failed repair + passing cover -> clean_covered, all P0 gates green
# --------------------------------------------------------------------------
def test_passing_cover_becomes_clean_covered():
    # first call (repair) fails hard; every later call (cover) passes
    res = _decide(GateStub([FAIL_HARD()], PASS))
    assert res.status == "clean_covered"
    assert res.publish_ok is True
    assert all(res.p0_gates.values())


# --------------------------------------------------------------------------
# 4 — candidate failures allowed; a later candidate passes
# --------------------------------------------------------------------------
def test_candidate_failures_allowed_when_one_passes():
    res = _decide(GateStub([FAIL_HARD(), FAIL_HARD(), FAIL_HARD()], PASS))
    assert res.status == "clean_covered"
    assert res.candidate_publish_failures >= 3
    assert res.publish_ok is True


# --------------------------------------------------------------------------
# 5 — auto_rejected is a final automated decision, NOT manual review
# --------------------------------------------------------------------------
def test_auto_rejected_is_not_manual_review():
    res = _decide(GateStub([], FAIL_HARD))
    assert res.status == "auto_rejected"
    assert res.publish_ok is False
    # The result object carries no manual flag; mark_remover sets
    # manual_required=False for it (verified in the e2e regression).


# --------------------------------------------------------------------------
# A passing repair short-circuits to clean_repaired
# --------------------------------------------------------------------------
def test_passing_repair_becomes_clean_repaired():
    res = _decide(GateStub([], PASS))
    assert res.status == "clean_repaired"
    assert res.publish_ok is True
    assert all(res.p0_gates.values())


# --------------------------------------------------------------------------
# 3 — report CI fails when a published output violates a P0 gate
# --------------------------------------------------------------------------
def _write_rec(root, stem, status, p0, **extra):
    d = root / status / stem
    d.mkdir(parents=True, exist_ok=True)
    rec = {"status": status, "product_id": stem, "image": stem + ".jpg",
           "p0_gates": p0, "publish_ok": status.startswith("clean"),
           "final_output_publish_failure": extra.get("fop", False),
           "candidate_publish_failures": extra.get("cpf", 0),
           "reject_reasons": extra.get("reasons", []),
           "v9_final_method": "x"}
    (d / "qa.json").write_text(json.dumps(rec))


_ALL_PASS = {k: True for k in (
    "residual_ocr_pass", "template_residual_pass", "dot_chain_pass",
    "visible_patch_pass", "visible_band_pass", "product_damage_pass",
    "silhouette_pass", "protected_text_pass")}


def test_report_ci_passes_for_clean_published_and_auto_rejected():
    tmp = Path(tempfile.mkdtemp())
    _write_rec(tmp, "a", "clean_repaired", dict(_ALL_PASS))
    _write_rec(tmp, "b", "clean_covered", dict(_ALL_PASS), cpf=2)
    _write_rec(tmp, "c", "auto_rejected", {}, reasons=["cover_failed_visible_patch"])
    rep = v13_report.build_report(tmp)
    assert rep["all_clean"] is True
    assert v13_report.main(["x", str(tmp)]) == 0
    assert rep["final_auto_rejected"] == 1
    assert rep["candidate_publish_failures"] == 2


def test_report_ci_fails_for_dirty_published_output():
    tmp = Path(tempfile.mkdtemp())
    dirty = dict(_ALL_PASS)
    dirty["visible_patch_pass"] = False     # a published output with a patch
    _write_rec(tmp, "a", "clean_covered", dirty)
    rep = v13_report.build_report(tmp)
    assert rep["all_clean"] is False
    assert rep["must_be_zero"]["published_with_visible_patch"] == 1
    assert v13_report.main(["x", str(tmp)]) == 1
