#!/usr/bin/env python3
"""Contract tests — the integration boundary every agent must keep green (agents/README.md §3.3).

Two layers:
  • Orchestration logic with INJECTED FAKES (no torch/cv2/easyocr) — this is itself the proof of the
    interface-isolation rule: the orchestrator depends only on `shared.contract`, so fakes drop in.
  • Boundary tests over the REAL agent adapters (guarded by an engine-availability skip) asserting the
    §2 red lines: cleaners never mutate the original; the validator returns only a QAReport.

    python3 tests/contract_tests/test_contract.py        # or: python3 -m pytest tests/contract_tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.contract import (CleanRequest, CleanResult, Cleaner, DetectionResult, QAReport,
                             Status, WatermarkType)
from agents.orchestrator import Orchestrator, PipelineConfig, build_default_pipeline


# ───────────────────────── fakes (no heavy backends) ─────────────────────────
class FakeCleaner(Cleaner):
    def __init__(self, tier, name=None):
        self.tier = tier
        self.name = name or f"fake-{tier}"
        self.calls = 0

    def clean(self, req):
        self.calls += 1
        return CleanResult(req.image, self.tier, meta={"fake": True})


class FakeDetector:
    def __init__(self, result):
        self._r = result

    def detect(self, image):
        return self._r


class FakeValidator:
    def __init__(self, pass_on_call=None, fail_score=0.5, retryable=True):
        self.calls = 0
        self.pass_on_call = pass_on_call
        self.fail_score = fail_score
        self.retryable = retryable

    def validate(self, original, cleaned, mask):
        self.calls += 1
        passed = self.pass_on_call is not None and self.calls == self.pass_on_call
        s = 1.0 if passed else self.fail_score
        return QAReport(passed, s, s, s, retryable=self.retryable)


class _AlwaysPass:
    def __init__(self, score):
        self.score = score

    def validate(self, original, cleaned, mask):
        return QAReport(True, self.score, self.score, self.score)


def _det(has=True, roi="white_bg", score=0.9, risk="low", shape=(40, 40)):
    mask = np.zeros(shape, np.uint8)
    mask[10:20, 10:20] = 255
    return DetectionResult(has, mask if has else None, WatermarkType.SEMI_TRANSPARENT_TEXT,
                           score, bbox=[10, 10, 20, 20], roi_type=roi, risk=risk)


def _orch(n_tiers=3, validator=None, **cfg):
    cleaners = {t: FakeCleaner(t) for t in range(1, n_tiers + 1)}
    o = Orchestrator(FakeDetector(_det()), cleaners,
                     validator or FakeValidator(pass_on_call=1), PipelineConfig(**cfg))
    return o, cleaners


def _img(shape=(40, 40, 3)):
    return np.full(shape, 255, np.uint8)


# ───────────────────────── orchestration / DoD ─────────────────────────
class TestSkipAndPass(unittest.TestCase):
    def test_skip_is_passthrough_zero_cleaners(self):
        o, cleaners = _orch()
        o.detector = FakeDetector(_det(has=False))
        img = _img()
        out = o.process(img)
        self.assertIs(out.status, Status.SKIP)
        self.assertEqual(out.attempts, 0)
        self.assertEqual(sum(c.calls for c in cleaners.values()), 0)
        self.assertIs(out.image, img)

    def test_easy_watermark_passes_at_tier1(self):
        o, cleaners = _orch(validator=FakeValidator(pass_on_call=1))
        out = o.process(_img())
        self.assertIs(out.status, Status.PASS)
        self.assertEqual(out.attempts, 1)
        self.assertEqual([t["tier"] for t in out.trace], [1])
        self.assertEqual(cleaners[2].calls + cleaners[3].calls, 0)


class TestEscalation(unittest.TestCase):
    def test_escalates_to_tier3_then_passes(self):
        o, _ = _orch(validator=FakeValidator(pass_on_call=3))
        out = o.process(_img())
        self.assertIs(out.status, Status.PASS)
        self.assertEqual([t["tier"] for t in out.trace], [1, 2, 3])
        self.assertEqual(out.attempts, 3)

    def test_cap_to_manual_review(self):
        o, cleaners = _orch(validator=FakeValidator(pass_on_call=None))
        out = o.process(_img())
        self.assertIs(out.status, Status.MANUAL_REVIEW)
        self.assertEqual([t["tier"] for t in out.trace], [1, 2, 3])
        self.assertEqual(sum(c.calls for c in cleaners.values()), 3)

    def test_escalate_never_repeats_a_tier(self):
        o, _ = _orch(validator=FakeValidator(pass_on_call=None))
        tiers = [t["tier"] for t in o.process(_img()).trace]
        self.assertEqual(tiers, sorted(set(tiers)))

    def test_nonretryable_verdict_exits_immediately(self):
        o, cleaners = _orch(validator=FakeValidator(pass_on_call=None, retryable=False))
        out = o.process(_img())
        self.assertIs(out.status, Status.MANUAL_REVIEW)
        self.assertEqual(out.attempts, 1)
        self.assertEqual(cleaners[2].calls + cleaners[3].calls, 0)

    def test_complex_detection_starts_higher(self):
        o, _ = _orch(validator=FakeValidator(pass_on_call=1))
        o.detector = FakeDetector(_det(roi="metallic_or_reflective", risk="high"))
        self.assertEqual(o.process(_img()).trace[0]["tier"], 2)


class TestSelectTier(unittest.TestCase):
    def setUp(self):
        self.o, _ = _orch()

    def test_simple_starts_tier1(self):
        self.assertEqual(self.o.select_tier(_det(roi="white_bg", score=0.9), None, 0), 1)

    def test_complex_starts_tier2(self):
        self.assertEqual(self.o.select_tier(_det(roi="metal", risk="medium"), None, 0), 2)

    def test_low_confidence_starts_tier2(self):
        self.assertEqual(self.o.select_tier(_det(roi="white_bg", score=0.40, risk="low"), None, 0), 2)

    def test_retry_escalates_one_rung(self):
        d, mid = _det(roi="white_bg", score=0.9), QAReport(False, 0.5, 0.5, 0.5)
        self.assertEqual(self.o.select_tier(d, mid, 1), 2)
        self.assertEqual(self.o.select_tier(d, mid, 2), 3)

    def test_very_low_score_jumps_to_top(self):
        d, low = _det(roi="white_bg", score=0.9), QAReport(False, 0.10, 0.10, 0.10)
        self.assertEqual(self.o.select_tier(d, low, 1), 3)

    def test_clamps_at_top_tier(self):
        d, mid = _det(roi="white_bg", score=0.9), QAReport(False, 0.5, 0.5, 0.5)
        self.assertEqual(self.o.select_tier(d, mid, 9), 3)


class TestPerTierThreshold(unittest.TestCase):
    def test_per_tier_threshold_only_tightens(self):
        o, _ = _orch(validator=_AlwaysPass(0.5), max_retries=3, tier_qa_threshold={1: 0.80})
        out = o.process(_img())
        self.assertIs(out.status, Status.PASS)
        self.assertEqual(out.trace[0]["verdict"], "fail")          # validator OK at 0.5, tier-1 gate 0.80 rejects
        self.assertEqual([t["tier"] for t in out.trace], [1, 2])


class TestOpenClosed(unittest.TestCase):
    def test_new_tier_via_registration_only(self):
        o, _ = _orch(n_tiers=3, validator=FakeValidator(pass_on_call=4), max_retries=4)
        o.register_cleaner(FakeCleaner(4, name="fake-diffusion-2"))
        out = o.process(_img())
        self.assertIs(out.status, Status.PASS)
        self.assertEqual([t["tier"] for t in out.trace], [1, 2, 3, 4])


class TestStatusAliasAndFactory(unittest.TestCase):
    def test_manual_review_aliases_auto_rejected(self):
        self.assertEqual(Status.MANUAL_REVIEW.terminal_label, "auto_rejected")
        self.assertEqual(Status.PASS.terminal_label, "published")
        self.assertEqual(Status.SKIP.terminal_label, "clean")

    def test_single_failure_terminal(self):
        self.assertEqual([s for s in Status if s not in (Status.PASS, Status.SKIP)], [Status.MANUAL_REVIEW])

    def test_build_default_pipeline_is_light_and_named(self):
        pipe = build_default_pipeline(use_audit=False)
        self.assertEqual(sorted(pipe.cleaners), [1, 2, 3])
        self.assertEqual(pipe.cleaners[1].name, "classic-cleaner")
        self.assertEqual(pipe.cleaners[2].name, "neural-cleaner")
        self.assertEqual(pipe.cleaners[3].name, "diffusion-cleaner")


# ───────────────────────── validator heuristic (invariant #6) ─────────────────────────
class TestValidatorHeuristic(unittest.TestCase):
    def setUp(self):
        from agents.validator import Validator
        self.v = Validator(qa_threshold=0.70)
        H, W = 64, 64
        self.white = np.full((H, W, 3), 255, np.uint8)
        self.mask = np.zeros((H, W), np.uint8)
        self.mask[24:40, 8:56] = 255
        self.orig = self.white.copy()
        for c in range(8, 56, 4):
            self.orig[24:40, c:c + 1] = 120

    def test_clean_removal_passes(self):
        r = self.v.validate(self.orig, self.white.copy(), self.mask)
        self.assertTrue(r.passed)
        self.assertEqual(r.verdict, "pass")
        self.assertGreater(r.removal_score, 0.7)

    def test_removed_but_visible_patch_fails(self):
        cleaned = self.white.copy()
        cleaned[24:40, 8:56] = 180
        r = self.v.validate(self.orig, cleaned, self.mask)
        self.assertFalse(r.passed)
        self.assertGreater(r.removal_score, 0.7)
        self.assertLess(r.fidelity_score, 0.5)
        self.assertIn("visible_patch", r.failed_checks)

    def test_still_watermarked_fails(self):
        r = self.v.validate(self.orig, self.orig.copy(), self.mask)
        self.assertFalse(r.passed)
        self.assertIn("residual_watermark", r.failed_checks)


# ───────────────────────── boundary tests over REAL adapters (§2 red lines) ─────────────────────────
try:
    import cv2  # noqa: F401
    import product_preserve_clean  # noqa: F401
    _ENGINES = True
except Exception:
    _ENGINES = False


@unittest.skipUnless(_ENGINES, "engine deps (cv2 / product_preserve_clean) unavailable")
class TestCleanerBoundary(unittest.TestCase):
    def _white_with_mark(self):
        img = np.full((80, 80, 3), 255, np.uint8)
        img[34:46, 16:64] = 120
        mask = np.zeros((80, 80), np.uint8)
        mask[34:46, 16:64] = 255
        return img, mask

    def test_classic_cleaner_does_not_mutate_original(self):
        from agents.classic_cleaner import ClassicCleaner
        img, mask = self._white_with_mark()
        before = img.copy()
        req = CleanRequest(img, mask, WatermarkType.SEMI_TRANSPARENT_TEXT, tier=1, attempt=0,
                           bbox=[16, 34, 64, 46], roi_type="white_bg")
        res = ClassicCleaner().clean(req)
        self.assertTrue(np.array_equal(img, before), "cleaner mutated the input image")
        self.assertEqual(res.tier_used, 1)
        self.assertIsInstance(res, CleanResult)

    def test_classic_cleaner_telea_fallback_on_textured_roi(self):
        # forces the built-in cv2 Telea path (committed-safe; no dependency on uncommitted glyph_clean)
        from agents.classic_cleaner import ClassicCleaner
        img, mask = self._white_with_mark()
        before = img.copy()
        c = ClassicCleaner()
        c._engine()
        if hasattr(c._ppc, "glyph_clean"):
            self.skipTest("glyph_clean present — fallback path not exercised")
        req = CleanRequest(img, mask, WatermarkType.SEMI_TRANSPARENT_TEXT, tier=1, attempt=0,
                           bbox=[16, 34, 64, 46], roi_type="metallic_or_reflective")
        res = c.clean(req)
        self.assertTrue(np.array_equal(img, before))
        self.assertEqual(res.meta["method"], "telea")


if __name__ == "__main__":
    unittest.main(verbosity=2)
