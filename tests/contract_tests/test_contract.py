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

_REPO_ROOT_T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT_T)

from shared.contract import (CleanRequest, CleanResult, Cleaner, DetectionResult, FailureReason,
                             QAReport, Status, WatermarkType)
from agents.orchestrator import Orchestrator, PipelineConfig
from integration import build_default_pipeline


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
        # No max_retries override: the escalation budget derives from the ladder length, so registering
        # a 4th tier makes it reachable with NO config edit (truly registration-only / open-closed).
        o, _ = _orch(n_tiers=3, validator=FakeValidator(pass_on_call=4))
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

    def test_classic_refuses_textured_roi_without_glyph(self):
        # P0 fix: textured surface + no texture-safe engine ⇒ REFUSE (escalate to neural), never the
        # product-scarring plain-Telea path. Skips when glyph_clean is present (texture-safe path covers it).
        from agents.classic_cleaner import ClassicCleaner
        img, mask = self._white_with_mark()
        before = img.copy()
        c = ClassicCleaner()
        c._engine()
        if hasattr(c._ppc, "glyph_clean"):
            self.skipTest("glyph_clean present — texture-safe path covers textured ROIs")
        req = CleanRequest(img, mask, WatermarkType.SEMI_TRANSPARENT_TEXT, tier=1, attempt=0,
                           bbox=[16, 34, 64, 46], roi_type="metallic_or_reflective")
        res = c.clean(req)
        self.assertEqual(res.status, "refused")
        self.assertEqual(res.meta["failure_reason"], FailureReason.BACKGROUND_TOO_COMPLEX)
        self.assertTrue(np.array_equal(img, before), "refusal must not mutate the input")

    def test_classic_telea_on_simple_surface(self):
        # low-texture / simple non-white surface still takes the plain-Telea fallback (Telea-safe)
        from agents.classic_cleaner import ClassicCleaner
        img, mask = self._white_with_mark()
        before = img.copy()
        c = ClassicCleaner()
        c._engine()
        if hasattr(c._ppc, "glyph_clean"):
            self.skipTest("glyph_clean present — fallback path not exercised")
        req = CleanRequest(img, mask, WatermarkType.SEMI_TRANSPARENT_TEXT, tier=1, attempt=0,
                           bbox=[16, 34, 64, 46], roi_type="simple_product_surface")
        res = c.clean(req)
        self.assertEqual(res.status, "cleaned")
        self.assertEqual(res.meta["method"], "telea")
        self.assertTrue(np.array_equal(img, before))


# ───────────────────────── FailureReason taxonomy ─────────────────────────
class TestFailureReasonTaxonomy(unittest.TestCase):
    """Boundary tests for the FailureReason enum added in contract.py.

    These guard the integration surface: if a member is renamed or removed,
    agent code that references it by name will break at the import boundary —
    this test catches that before any engine runs.
    """

    # ── presence: every chartered member must exist ───────────────────────
    def test_classic_tier_reasons_present(self):
        self.assertIs(FailureReason.MASK_OVER_PRODUCT_DETAIL,
                      FailureReason("mask_over_product_detail"))
        self.assertIs(FailureReason.BACKGROUND_TOO_COMPLEX,
                      FailureReason("background_too_complex"))
        self.assertIs(FailureReason.UNSAFE_FOR_CLASSIC,
                      FailureReason("unsafe_for_classic"))

    def test_neural_tier_reasons_present(self):
        self.assertIs(FailureReason.SEMANTIC_RISK, FailureReason("semantic_risk"))
        self.assertIs(FailureReason.PRODUCT_DAMAGE, FailureReason("product_damage"))
        self.assertIs(FailureReason.MASK_TOO_SMALL, FailureReason("mask_too_small"))

    def test_diffusion_tier_reasons_present(self):
        self.assertIs(FailureReason.SEMANTIC_PRODUCT_RISK,
                      FailureReason("semantic_product_risk"))
        self.assertIs(FailureReason.PRODUCT_IDENTITY_CHANGE,
                      FailureReason("product_identity_change"))
        self.assertIs(FailureReason.HALLUCINATION_RISK,
                      FailureReason("hallucination_risk"))

    def test_shared_mask_too_large_present(self):
        """MASK_TOO_LARGE is shared between Classic (Tier 1) and Neural (Tier 2)."""
        self.assertIs(FailureReason.MASK_TOO_LARGE, FailureReason("mask_too_large"))

    # ── str subtype: values are JSON-safe strings ─────────────────────────
    def test_all_members_are_str_instances(self):
        for reason in FailureReason:
            self.assertIsInstance(reason, str,
                                  f"{reason!r} must be a str for JSON/JSONL serialisation")

    def test_value_equals_name_lower(self):
        """Snake-case values must match the lower-cased member name (no typos)."""
        for reason in FailureReason:
            self.assertEqual(reason.value, reason.name.lower())

    # ── meta dict round-trip: the pattern cleaners will use ───────────────
    def test_embeds_in_clean_result_meta(self):
        import numpy as np
        img = np.zeros((4, 4, 3), np.uint8)
        result = CleanResult(img, tier_used=1,
                             meta={"failure_reason": FailureReason.MASK_TOO_LARGE})
        stored = result.meta["failure_reason"]
        self.assertEqual(stored, "mask_too_large")
        self.assertIsInstance(stored, FailureReason)

    # ── no duplicate values (aliases would silently hide a member) ────────
    def test_no_alias_members(self):
        all_values = [r.value for r in FailureReason]
        self.assertEqual(len(all_values), len(set(all_values)),
                         "FailureReason must not contain alias members (duplicate values)")


# ───────────────────────── region-targeted escalation (the unified retry semantics) ─────────────────────────
class BoxValidator:
    """Fake validator that returns a retry_box (a retryable residual) on the first ``box_calls``
    validate()s, then either passes on ``pass_on_call`` or keeps failing. Lets a test drive the
    orchestrator's Phase-1 (intra-tier) vs Phase-2 (escalation) split deterministically."""
    def __init__(self, box=(10, 10, 8, 8), box_calls=0, pass_on_call=None):
        self.calls = 0
        self.box = list(box)
        self.box_calls = box_calls
        self.pass_on_call = pass_on_call

    def validate(self, original, cleaned, mask):
        self.calls += 1
        if self.pass_on_call is not None and self.calls == self.pass_on_call:
            return QAReport(True, 1.0, 1.0, 1.0)
        return QAReport(False, 0.5, 0.5, 0.5,
                        retry_box=list(self.box) if self.calls <= self.box_calls else None)


class RecordingCleaner(Cleaner):
    """Records the retry_box seen on every clean() call; optionally refuses with a fixed reason.
    Box-aware by default (supports_retry_box=True) so it exercises the orchestrator's Phase-1 path;
    a test may set the instance attr False to model a region-blind tier (classic)."""
    supports_retry_box = True

    def __init__(self, tier, name=None, refuse_reason=None):
        self.tier = tier
        self.name = name or f"rec-{tier}"
        self.boxes = []
        self.refuse_reason = refuse_reason

    def clean(self, req):
        self.boxes.append(req.retry_box)
        if self.refuse_reason is not None:
            return CleanResult(req.image, self.tier, status="refused",
                               meta={"failure_reason": self.refuse_reason})
        return CleanResult(req.image, self.tier, meta={})


def _orch_rec(validator, **cfg):
    cleaners = {t: RecordingCleaner(t) for t in range(1, 4)}
    return Orchestrator(FakeDetector(_det()), cleaners, validator, PipelineConfig(**cfg)), cleaners


class TestRegionTargetedRetry(unittest.TestCase):
    def test_retry_box_re_runs_same_tier_and_is_dispatched(self):
        # two residual-with-box failures, then pass → all on tier 1; the box rides the retries
        o, cs = _orch_rec(BoxValidator(box=(10, 10, 8, 8), box_calls=2, pass_on_call=3))
        out = o.process(_img())
        self.assertIs(out.status, Status.PASS)
        self.assertEqual([t["tier"] for t in out.trace], [1, 1, 1])         # never escalated
        self.assertEqual(cs[1].boxes, [None, [10, 10, 8, 8], [10, 10, 8, 8]])  # 1st None, retries carry box
        self.assertEqual(cs[2].boxes + cs[3].boxes, [])                      # stronger tiers untouched

    def test_intra_tier_budget_exhausts_then_escalates(self):
        # every fail carries a box → 1 initial + 2 local retries per tier, then climb the ladder
        o, cs = _orch_rec(BoxValidator(box_calls=99, pass_on_call=None), max_intra_tier_retries=2)
        out = o.process(_img())
        self.assertIs(out.status, Status.MANUAL_REVIEW)
        self.assertEqual([t["tier"] for t in out.trace], [1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.assertEqual(len(cs[1].boxes), 3)                               # initial + 2 local

    def test_intra_tier_retry_count_is_configurable(self):
        o, cs = _orch_rec(BoxValidator(box_calls=99, pass_on_call=None), max_intra_tier_retries=1)
        out = o.process(_img())
        self.assertEqual([t["tier"] for t in out.trace], [1, 1, 2, 2, 3, 3])  # initial + 1 local per tier

    def test_no_box_is_pure_escalation(self):
        o, cs = _orch_rec(BoxValidator(box_calls=0, pass_on_call=None))      # never a box → legacy ladder
        out = o.process(_img())
        self.assertEqual([t["tier"] for t in out.trace], [1, 2, 3])
        self.assertEqual(cs[1].boxes, [None])                               # never asked to focus


class TestCleanerRefusal(unittest.TestCase):
    def test_nonretryable_refusal_terminates_without_validate_or_escalation(self):
        v = FakeValidator(pass_on_call=None)                                # must NOT be consulted
        cleaners = {1: RecordingCleaner(1, refuse_reason=FailureReason.PRODUCT_DAMAGE),
                    2: RecordingCleaner(2), 3: RecordingCleaner(3)}
        out = Orchestrator(FakeDetector(_det()), cleaners, v, PipelineConfig()).process(_img())
        self.assertIs(out.status, Status.MANUAL_REVIEW)
        self.assertEqual(out.attempts, 1)
        self.assertEqual(v.calls, 0)                                        # refusal skips QA
        self.assertEqual(len(cleaners[2].boxes) + len(cleaners[3].boxes), 0)  # no escalation
        self.assertEqual(out.trace[0]["verdict"], "refused")
        self.assertEqual(out.trace[0]["failure_reason"], FailureReason.PRODUCT_DAMAGE)

    def test_retryable_refusal_escalates(self):
        cleaners = {1: RecordingCleaner(1, refuse_reason=FailureReason.BACKGROUND_TOO_COMPLEX),
                    2: RecordingCleaner(2), 3: RecordingCleaner(3)}
        out = Orchestrator(FakeDetector(_det()), cleaners, _AlwaysPass(0.9), PipelineConfig()).process(_img())
        self.assertIs(out.status, Status.PASS)
        self.assertEqual([t["tier"] for t in out.trace], [1, 2])           # skipped the dead tier
        self.assertEqual(out.trace[0]["verdict"], "refused")

    def test_operational_refusal_at_top_tier_is_manual_review(self):
        # diffusion-style infra refusal: escalatable, but nothing above it → MANUAL_REVIEW
        cleaners = {1: RecordingCleaner(1, refuse_reason=FailureReason.BACKGROUND_TOO_COMPLEX),
                    2: RecordingCleaner(2, refuse_reason=FailureReason.MASK_TOO_LARGE),
                    3: RecordingCleaner(3, refuse_reason="infrastructure_missing")}
        out = Orchestrator(FakeDetector(_det()), cleaners, _AlwaysPass(0.9), PipelineConfig()).process(_img())
        self.assertIs(out.status, Status.MANUAL_REVIEW)
        self.assertEqual([t["tier"] for t in out.trace], [1, 2, 3])


class TestRetryabilityTaxonomy(unittest.TestCase):
    """The authoritative branch source the orchestrator consults (shared/contract.py)."""
    def test_product_and_semantic_risks_are_non_retryable(self):
        from shared.contract import NON_RETRYABLE_FAILURE_REASONS
        for r in (FailureReason.PRODUCT_DAMAGE, FailureReason.MASK_OVER_PRODUCT_DETAIL,
                  FailureReason.SEMANTIC_RISK, FailureReason.SEMANTIC_PRODUCT_RISK,
                  FailureReason.PRODUCT_IDENTITY_CHANGE, FailureReason.HALLUCINATION_RISK):
            self.assertIn(r, NON_RETRYABLE_FAILURE_REASONS)
            self.assertFalse(r.is_retryable)

    def test_capability_limits_are_retryable(self):
        for r in (FailureReason.BACKGROUND_TOO_COMPLEX, FailureReason.UNSAFE_FOR_CLASSIC,
                  FailureReason.MASK_TOO_SMALL, FailureReason.MASK_TOO_LARGE):
            self.assertTrue(r.is_retryable)

    def test_operational_reason_is_not_in_nonretryable_set(self):
        from shared.contract import NON_RETRYABLE_FAILURE_REASONS
        self.assertNotIn("infrastructure_missing", NON_RETRYABLE_FAILURE_REASONS)

    def test_contract_additive_fields_default_none(self):
        self.assertIsNone(QAReport(True, 1.0, 1.0, 1.0).retry_box)
        req = CleanRequest(_img(), _det().mask, WatermarkType.UNKNOWN, tier=1, attempt=0)
        self.assertIsNone(req.retry_box)


class TestNeuralFocusMask(unittest.TestCase):
    """Pure-numpy: the neural cleaner's box restriction (no LaMa needed)."""
    def test_restricts_strictly_inside_box(self):
        from agents.neural_cleaner.cleaner import _focus_mask
        mask = np.zeros((40, 40), np.uint8); mask[10:20, 4:36] = 255
        r = _focus_mask(mask, [20, 10, 16, 10]); ys, xs = np.where(r > 0)
        self.assertTrue(xs.min() >= 20 and xs.max() < 36 and ys.min() >= 10 and ys.max() < 20)

    def test_falls_back_to_box_rect_when_disjoint(self):
        from agents.neural_cleaner.cleaner import _focus_mask
        mask = np.zeros((40, 40), np.uint8); mask[10:20, 20:30] = 255
        r = _focus_mask(mask, [0, 0, 5, 5]); ys, xs = np.where(r > 0)
        self.assertEqual((int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())), (0, 4, 0, 4))


class TestDiffusionRefusalAdapter(unittest.TestCase):
    """The real diffusion adapter must refuse EXPLICITLY (never an unchanged-image no-op) when no
    backend is installed — the configuration here and on CI."""
    def test_refuses_infrastructure_missing_without_backend(self):
        from agents.diffusion_cleaner import DiffusionCleaner
        img = _img((40, 40, 3)); before = img.copy()
        mask = np.zeros((40, 40), np.uint8); mask[10:20, 10:20] = 255
        req = CleanRequest(img, mask, WatermarkType.UNKNOWN, tier=3, attempt=0, bbox=[10, 10, 20, 20])
        res = DiffusionCleaner().clean(req)
        try:
            import diffusers  # noqa: F401
            import torch
            backend = torch.cuda.is_available() or torch.backends.mps.is_available()
        except Exception:
            backend = False
        if not backend:
            self.assertEqual(res.status, "refused")
            self.assertEqual(res.meta["failure_reason"], "infrastructure_missing")
        self.assertTrue(np.array_equal(img, before), "diffusion cleaner mutated the input")


class TestManualReviewImage(unittest.TestCase):
    """MANUAL_REVIEW must surface the ORIGINAL image (auto_rejected ≡ original restored), never the
    last failed/damaged attempt — so a caller that writes outcome.image cannot publish damage."""
    def test_manual_review_returns_original_not_damaged(self):
        class DamagingCleaner(Cleaner):
            def __init__(self, tier):
                self.tier = tier
                self.name = f"dmg-{tier}"
            def clean(self, req):
                return CleanResult(np.zeros_like(req.image), self.tier)   # "damaged" output
        cleaners = {t: DamagingCleaner(t) for t in (1, 2, 3)}
        o = Orchestrator(FakeDetector(_det()), cleaners, FakeValidator(pass_on_call=None), PipelineConfig())
        img = _img()
        out = o.process(img)
        self.assertIs(out.status, Status.MANUAL_REVIEW)
        self.assertTrue(np.array_equal(out.image, img), "must return the original")
        self.assertFalse(np.array_equal(out.image, np.zeros_like(img)), "must not return the damaged attempt")


class TestRegionBlindCleaner(unittest.TestCase):
    """A region-blind tier (supports_retry_box=False, e.g. classic) must NOT consume intra-tier
    retries — a focused retry would reproduce the same result. The orchestrator escalates instead."""
    def test_region_blind_tier_skips_phase1(self):
        cleaners = {t: RecordingCleaner(t) for t in range(1, 4)}
        for c in cleaners.values():
            c.supports_retry_box = False
        v = BoxValidator(box_calls=99, pass_on_call=None)        # every fail carries a box
        out = Orchestrator(FakeDetector(_det()), cleaners, v, PipelineConfig()).process(_img())
        self.assertIs(out.status, Status.MANUAL_REVIEW)
        self.assertEqual([t["tier"] for t in out.trace], [1, 2, 3])   # pure escalation, no [1,1,1]
        self.assertEqual(cleaners[1].boxes, [None])                   # tier 1 tried once, never re-focused


class TestNeuralBoundary(unittest.TestCase):
    """The neural cleaner must enforce the no-mutation red line ITSELF (defensive copy), not trust the
    engine — proven with a hostile fake engine that tries to scribble on its input in place. No LaMa."""
    def test_neural_cleaner_does_not_mutate_original(self):
        from agents.neural_cleaner.cleaner import NeuralCleaner
        c = NeuralCleaner()

        class _HostileRB:
            @staticmethod
            def _lama_crop_inpaint(bgr, mask):
                bgr[:] = 0                                   # try to mutate the input in place
                return np.zeros_like(bgr), True
        c._rb = _HostileRB()                                 # inject; bypasses the lazy run_bulk import
        img = _img((40, 40, 3))
        before = img.copy()
        mask = np.zeros((40, 40), np.uint8)
        mask[10:20, 10:20] = 255
        req = CleanRequest(img, mask, WatermarkType.UNKNOWN, tier=2, attempt=0)
        res = c.clean(req)
        self.assertTrue(np.array_equal(img, before), "neural cleaner mutated the input")
        self.assertEqual(res.tier_used, 2)


class TestRoiVocabulary(unittest.TestCase):
    """The detector is the single producer of roi_class; shared/roi.py is the single consumer vocab.
    These pin them together so they can never drift (the bug this refactor fixed)."""
    def test_finder_emitted_rois_all_classified(self):
        from shared.roi import WHITE_ROIS, TEXTURED_ROIS, TELEA_SAFE_ROIS, FINDER_EMITTED_ROIS
        classified = WHITE_ROIS | TEXTURED_ROIS | TELEA_SAFE_ROIS
        self.assertEqual(FINDER_EMITTED_ROIS - classified, set(),
                         "every roi_class the finder emits must be classified by the agents layer")

    def test_finder_source_returns_are_registered(self):
        import re
        from shared.roi import FINDER_EMITTED_ROIS
        path = os.path.join(_REPO_ROOT_T, "logo_finder.py")
        if not os.path.exists(path):
            self.skipTest("logo_finder.py not present")
        with open(path) as fh:
            src = fh.read()
        i = src.find("def _roi_class")
        j = src.find("\ndef ", i + 1)
        body = src[i:j] if j != -1 else src[i:]
        returned = set(re.findall(r'return "([a-z_]+)"', body))
        self.assertTrue(returned, "could not parse _roi_class returns")
        self.assertEqual(returned - FINDER_EMITTED_ROIS, set(),
                         f"logo_finder emits roi_class not registered in shared/roi.py: "
                         f"{returned - FINDER_EMITTED_ROIS}")


class TestAuditValidatorDefaults(unittest.TestCase):
    """A publish gate must FAIL SAFE on audit schema drift: a missing sub-score reads as max risk
    (score 0.0), never as 'clean'. Driven with injected fakes — no real audit / cv2 / OCR."""
    def test_missing_scores_default_pessimistic(self):
        from agents.validator import AuditValidator
        v = AuditValidator()

        class _FakeAudit:
            @staticmethod
            def audit_pair(op, fp, meta=None, mask_path=None, reader=None):
                return {"scores": {}, "publish_allowed": False, "evidence": [],
                        "recommended_next_action": "auto_reject"}

        class _FakeCv2:
            @staticmethod
            def imwrite(p, im):
                return True
        v._audit, v._cv2 = _FakeAudit(), _FakeCv2()          # inject; bypasses the lazy audit import
        img = np.full((8, 8, 3), 255, np.uint8)
        mask = np.zeros((8, 8), np.uint8)
        mask[2:6, 2:6] = 255
        r = v.validate(img, img, mask)
        self.assertEqual(r.removal_score, 0.0)
        self.assertEqual(r.fidelity_score, 0.0)
        self.assertFalse(r.passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
