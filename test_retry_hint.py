#!/usr/bin/env python3
"""test_retry_hint.py — consume side of the audit `retry_hint` (region-targeted re-inpaint).

The audit-side emit is additive and present ONLY on retry_repair (== REJECT_RESIDUAL_WATERMARK).
These tests simulate that emit (it is not yet committed on the audit branch) and prove the
orchestrator + owner consume it correctly, with first-pass behaviour byte-identical and the
whole feature dormant when the field is absent. Heavy models are mocked — no OCR/LaMa runs.

  python3 test_retry_hint.py
"""
import json, os, sys, tempfile
import numpy as np

import owner_agent as ow
import orchestrator as orch

HINT = [100, 200, 300, 260]          # a residual region the Audit still sees (full-image px)
IMG_H, IMG_W = 600, 800


def _contains(outer, inner):
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


# ── owner: process_one hint branch ──────────────────────────────────────────────
def test_owner_hint_reinpaints_even_when_detector_misses():
    """The leak case: first pass left residual BECAUSE the detector under-read the mark,
    so on retry lf.find returns NO_WATERMARK. Without the hint that copies the dirty
    original through; WITH the hint we must still re-inpaint the hinted region."""
    calls = {}
    ow.rb._imread = lambda p: np.zeros((IMG_H, IMG_W, 3), np.uint8)
    ow.lf.find = lambda bgr, reader: {"presence": "NO_WATERMARK", "candidates": []}

    def fake_clean(bgr, bbox_p=None):
        calls["bbox_p"] = list(bbox_p)
        return np.ones_like(bgr), {"status": "cleaned", "residual_after": 0}
    ow.rb.clean_image = fake_clean

    rec, img, dest, mask = ow.process_one("/x/a.jpg", "/job", reader=None,
                                          attempt=1, fail_reason="FAIL_RESIDUAL_WATERMARK",
                                          retry_box=HINT)

    assert "bbox_p" in calls, "clean_image was NOT called — dirty original would be copied through"
    assert _contains(calls["bbox_p"], HINT), f"mask {calls['bbox_p']} must cover hint {HINT}"
    assert dest == "final" and img is not None and not isinstance(img, str)
    assert rec["owner_status"] == "cleaned" and rec["method"] == "repair_hint"
    assert rec["retry_hint_box"] == HINT
    assert rec["bbox_padded"] == calls["bbox_p"]
    print("PASS  owner re-inpaints hint region even when detector misses (acceptance #1)")


def test_owner_hint_unions_with_detection():
    """When the detector DOES fire on retry, the mask covers union(detection, hint)."""
    det = [50, 180, 320, 270]            # extends beyond the hint on every side but x2
    calls = {}
    ow.rb._imread = lambda p: np.zeros((IMG_H, IMG_W, 3), np.uint8)
    ow.lf.find = lambda bgr, reader: {"presence": "CONFIRMED_WATERMARK",
                                      "candidates": [{"_box": det, "roi_class": "plain_white"}]}
    ow.rb.clean_image = lambda bgr, bbox_p=None: (calls.__setitem__("bbox_p", list(bbox_p))
                                                  or (np.ones_like(bgr), {"residual_after": 0}))
    rec, img, dest, mask = ow.process_one("/x/a.jpg", "/job", reader=None,
                                          attempt=1, fail_reason="FAIL_RESIDUAL_WATERMARK",
                                          retry_box=HINT)
    union = [min(det[0], HINT[0]), min(det[1], HINT[1]), max(det[2], HINT[2]), max(det[3], HINT[3])]
    assert _contains(calls["bbox_p"], union), f"mask {calls['bbox_p']} must cover union {union}"
    print("PASS  owner mask covers union(detection, hint)")


def test_owner_first_pass_unaffected():
    """retry_box=None (first pass): the hint branch is inert; a clean image still copies
    through and clean_image is never reached."""
    seen = {"clean": 0}
    ow.rb._imread = lambda p: np.zeros((IMG_H, IMG_W, 3), np.uint8)
    ow.lf.find = lambda bgr, reader: {"presence": "NO_WATERMARK", "candidates": []}
    ow.rb.clean_image = lambda *a, **k: seen.__setitem__("clean", seen["clean"] + 1) or (None, {})
    rec, img, dest, mask = ow.process_one("/x/a.jpg", "/job", reader=None)
    assert img == "COPY" and dest == "final" and rec["method"] == "copy"
    assert seen["clean"] == 0, "first pass must not enter the hint/clean_image branch"
    print("PASS  first-pass (retry_box=None) path unchanged — copy-through, no re-inpaint")


# ── owner: handle_feedback threads retry_box → process_one ────────────────────────
def test_handle_feedback_threads_retry_box():
    captured = {}
    ow.rb._init_worker = lambda device: None
    ow.rb._get_reader = lambda: None
    ow._save = lambda *a, **k: None
    ow.process_one = (lambda *a, **k:
                      (captured.update(retry_box=k.get("retry_box"), fail=k.get("fail_reason"))
                       or ({"id": "x", "owner_status": "cleaned", "attempt": 1}, None, "final", None)))
    with tempfile.TemporaryDirectory() as job:
        d = ow.job_dirs(job)
        with open(d["manifest"], "w") as f:
            f.write(json.dumps({"id": "x", "original": "originals/x.jpg", "attempt": 0}) + "\n")
        with open(d["feedback"], "w") as f:
            f.write(json.dumps({"id": "x", "verdict": "FAIL_RESIDUAL_WATERMARK", "retry_box": HINT}) + "\n")
        ow.handle_feedback(job, "cpu")
    assert captured.get("retry_box") == HINT, captured
    assert captured.get("fail") == "FAIL_RESIDUAL_WATERMARK"
    print("PASS  handle_feedback reads retry_box and passes it to process_one")


# ── orchestrator: stash + thread on FAIL→RETRY, dormant when field absent ─────────
def _run_orchestrator(audit_emits_hint):
    captured = {}
    audit_calls = [0]

    def fake_run_owner(job, device, feedback):
        if feedback:                                   # capture round-1 feedback before round 2 overwrites it
            fb = orch.jp(job, "audit_feedback.jsonl")
            captured["fb"] = ([json.loads(l) for l in open(fb) if l.strip()]
                              if os.path.exists(fb) else [])
        with open(orch.jp(job, "owner_manifest.jsonl"), "a") as f:
            for fn in os.listdir(orch.jp(job, "originals")):
                iid = os.path.splitext(fn)[0]
                with open(orch.jp(job, "finals", fn), "wb") as g:
                    g.write(b"final-bytes")
                f.write(json.dumps({"id": iid, "original": os.path.join("originals", fn),
                                    "final": os.path.join("finals", fn),
                                    "owner_status": "cleaned",
                                    "attempt": 1 if feedback else 0}) + "\n")

    def fake_run_audit(job, items, device):
        audit_calls[0] += 1
        first = audit_calls[0] == 1
        with open(orch.jp(job, "audit_results.jsonl"), "a") as f:
            for it in items:
                if first:
                    rec = {"id": it["id"], "audit_decision": "REJECT_RESIDUAL_WATERMARK",
                           "publish_allowed": False, "recommended_next_action": "retry_repair"}
                    if audit_emits_hint:
                        rec["retry_hint"] = {"box": HINT, "reason": "resid", "basis": "canonical_centre_band"}
                else:
                    rec = {"id": it["id"], "audit_decision": "PASS",
                           "publish_allowed": True, "recommended_next_action": "publish"}
                f.write(json.dumps(rec) + "\n")

    orch.run_owner, orch.run_audit = fake_run_owner, fake_run_audit
    job = tempfile.mkdtemp()
    os.makedirs(orch.jp(job, "originals"))
    with open(orch.jp(job, "originals", "x.jpg"), "wb") as f:
        f.write(b"orig")
    rep = orch.orchestrate(job, "cpu")
    st = json.load(open(orch.jp(job, "state.json")))
    return captured, st, rep


def test_orchestrator_threads_hint():
    captured, st, rep = _run_orchestrator(audit_emits_hint=True)
    assert captured["fb"][0].get("retry_box") == HINT, "feedback to owner must carry retry_box"
    assert st["images"]["x"].get("retry_box") == HINT, "state must stash retry_box (resumable)"
    assert rep["published"] == 1 and rep["rejected"] == 0
    print("PASS  orchestrator stashes retry_box on state + threads it to the owner; retry → PASS")


def test_orchestrator_dormant_without_hint():
    """Today's reality: audit does NOT emit retry_hint. The retry still runs (blind), and
    no retry_box leaks into the feedback — proving the consume side is inert until audit ships."""
    captured, st, rep = _run_orchestrator(audit_emits_hint=False)
    assert "retry_box" not in captured["fb"][0], "no hint → feedback must not carry retry_box"
    assert "retry_box" not in st["images"]["x"], "no hint → nothing stashed on state"
    assert rep["published"] == 1, "blind retry still functions unchanged"
    print("PASS  no audit hint → feature dormant, blind retry unchanged (forward-compatible)")


if __name__ == "__main__":
    tests = [test_owner_hint_reinpaints_even_when_detector_misses,
             test_owner_hint_unions_with_detection,
             test_owner_first_pass_unaffected,
             test_handle_feedback_threads_retry_box,
             test_orchestrator_threads_hint,
             test_orchestrator_dormant_without_hint]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} retry_hint consume-side tests passed")
