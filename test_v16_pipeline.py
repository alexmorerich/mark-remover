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


def FAIL_COSMETIC():
    # only a cosmetic gate fails (visible band/patch) — watermark gone, product
    # safe ⇒ V16.1 publishes as clean_covered with cosmetic_seam=True.
    return _Verdict(False, ["visible_band"], residual_pass=True,
                    dot_chain_pass=True, product_damage_pass=True,
                    silhouette_pass=True, protected_text_pass=True,
                    visible_patch_pass=False, rectangular_band_pass=False,
                    polygon_patch_pass=True)


class GateStub:
    """Returns queued verdicts, then a default for everything after."""
    def __init__(self, queue, default):
        self.q = list(queue)
        self.default = default
        self.calls = 0

    def __call__(self, image, repaired, residual_pass=None):
        self.calls += 1
        return self.q.pop(0) if self.q else self.default()


class RepairAwareGate:
    """V18 adds product-safe candidates to the REPAIR path, so the number of
    repair candidates is no longer fixed. To test "every repair fails, a cover
    passes -> clean_covered" robustly, fail on ``repaired=True`` and pass on
    ``repaired=False`` (after an optional number of initial cover failures)."""
    def __init__(self, cover_fails_before_pass=0):
        self.cover_calls = 0
        self.cover_fails = cover_fails_before_pass
        self.calls = 0

    def __call__(self, image, repaired, residual_pass=None):
        self.calls += 1
        if repaired:
            return FAIL_HARD()
        self.cover_calls += 1
        if self.cover_calls <= self.cover_fails:
            return FAIL_HARD()
        return PASS()


def _clean_recheck(image):
    return False, {}


def _img():
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, (200, 320, 3), dtype=np.uint8)


def _white_img():
    # Pure-background canvas: under V17 a cosmetic seam here is confirmed to be
    # on background, so it stays cosmetic (publishable) rather than reading as
    # product damage on the random-noise canvas.
    return np.full((200, 320, 3), 245, np.uint8)


def _mask():
    m = np.zeros((200, 320), np.uint8)
    m[96:108, 130:210] = 255
    return m


BBOX = (120, 90, 100, 28)


def _decide(gate, recheck=_clean_recheck, img=None, **kw):
    img = _img() if img is None else img
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
    # every repair candidate fails hard; the first cover candidate passes.
    res = _decide(RepairAwareGate(cover_fails_before_pass=0))
    assert res.status == "clean_covered"
    assert res.publish_ok is True
    assert all(res.p0_gates.values())


# --------------------------------------------------------------------------
# 4 — candidate failures allowed; a later candidate passes
# --------------------------------------------------------------------------
def test_candidate_failures_allowed_when_one_passes():
    # every repair fails + the first two cover candidates fail, then a cover
    # passes -> clean_covered with several recorded candidate failures.
    res = _decide(RepairAwareGate(cover_fails_before_pass=2), img=_white_img())
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
# V16.1 — mark removed + product-safe but a cosmetic seam -> clean_covered
# (published with cosmetic_seam flag), NOT auto_rejected and NOT mark-present.
# --------------------------------------------------------------------------
def test_cosmetic_seam_publishes_as_clean_covered():
    # V17: a cosmetic seam publishes only when the change is on background.
    res = _decide(GateStub([], FAIL_COSMETIC), img=_white_img())
    assert res.status == "clean_covered"
    assert res.publish_ok is True
    assert res.cosmetic_seam is True
    # hard-safety gates all green; only the cosmetic gates fail
    p0 = res.p0_gates
    assert p0["residual_ocr_pass"] and p0["product_damage_pass"]
    assert p0["silhouette_pass"] and p0["protected_text_pass"]
    assert p0["visible_band_pass"] is False


def test_v17_cosmetic_seam_on_product_is_auto_rejected():
    # V17 contract: the SAME cosmetic-fail verdict that publishes on a clean
    # background must be auto_rejected when the change lands on product pixels
    # (the random-noise canvas reads as textured product). Never ship a seam on
    # product. (patch plan §1.2 / §4.1)
    res = _decide(GateStub([], FAIL_COSMETIC), img=_img())
    assert res.status == "auto_rejected"
    assert res.publish_ok is False


def test_mark_still_present_is_not_published():
    # gate would pass cosmetically, but the detector says the mark is STILL
    # there ⇒ residual_ocr fails ⇒ not safe ⇒ auto_rejected (never published).
    res = _decide(GateStub([], FAIL_COSMETIC),
                  recheck=lambda im: (True, {"confidence": 0.7}))
    assert res.status == "auto_rejected"
    assert res.publish_ok is False


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


def test_report_ci_allows_cosmetic_seam_published():
    # A published output with a visible band/patch but all hard-safety gates
    # green is a cosmetic seam — allowed, CI passes.
    tmp = Path(tempfile.mkdtemp())
    seam = dict(_ALL_PASS)
    seam["visible_patch_pass"] = False
    seam["visible_band_pass"] = False
    d = tmp / "clean_covered" / "a"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"status": "clean_covered", "product_id": "a", "image": "a.jpg",
           "p0_gates": seam, "publish_ok": True, "cosmetic_seam": True,
           "final_output_publish_failure": False}
    (d / "qa.json").write_text(json.dumps(rec))
    rep = v13_report.build_report(tmp)
    assert rep["all_clean"] is True
    assert rep["cosmetic"]["published_with_visible_patch"] == 1
    assert rep["published_with_cosmetic_seam"] == 1
    assert v13_report.main(["x", str(tmp)]) == 0


def test_report_ci_fails_for_dirty_published_output():
    # A HARD-safety failure on a published output (mark still present) must fail.
    tmp = Path(tempfile.mkdtemp())
    dirty = dict(_ALL_PASS)
    dirty["residual_ocr_pass"] = False     # watermark still detectable
    _write_rec(tmp, "a", "clean_covered", dirty, fop=True)
    rep = v13_report.build_report(tmp)
    assert rep["all_clean"] is False
    assert rep["must_be_zero"]["published_with_residual_ocr"] == 1
    assert v13_report.main(["x", str(tmp)]) == 1
