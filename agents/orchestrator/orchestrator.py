#!/usr/bin/env python3
"""orchestrator agent — control loop, routing, the single retry owner (charter: agents/orchestrator/AGENT.md).

Owns the whole loop: detect → skip/route → clean → validate → pass / escalate / manual-review. It is
the ONLY place the attempt counter lives, and the ONLY place routing policy lives (select_tier — there
is no separate Router). It depends on the `Cleaner` interface and a {tier: Cleaner} registry, never on a
concrete tier, so a new tier is registered, not wired in (open/closed).

Failure has exactly one exit: MANUAL_REVIEW. On disk that is the frozen CONTRACT_v1 `auto_rejected`
terminal (original restored from backup); see Status.terminal_label. No second FAIL sink.

Boundaries (contract tests): does not touch pixels, detect, clean, or score; it only routes / retries.
"""
from __future__ import annotations

import numpy as np

from shared.contract import CleanRequest, DetectionResult, PipelineOutcome, QAReport, Status

from .config import PipelineConfig


def _mask_area_frac(mask) -> float:
    if mask is None:
        return 0.0
    m = np.asarray(mask)
    return float((m > 0).mean()) if m.size else 0.0


class Orchestrator:
    """``process(image) -> PipelineOutcome``. Holds a {tier: Cleaner} registry, MAX_RETRIES, and the
    tier-selection policy."""

    def __init__(self, detector, cleaners, validator, config: PipelineConfig | None = None):
        self.detector = detector
        self.cleaners = dict(cleaners)          # {tier(int): Cleaner}
        self.validator = validator
        self.config = config or PipelineConfig()

    def register_cleaner(self, cleaner) -> None:   # open/closed: extend the ladder, no process() edit
        self.cleaners[cleaner.tier] = cleaner

    @property
    def MAX_RETRIES(self) -> int:
        return self.config.max_retries

    def _tiers(self) -> list:
        return sorted(self.cleaners)

    # ── routing policy: the one place it lives; pure and unit-testable ──
    def _is_complex(self, det: DetectionResult) -> bool:
        cfg = self.config
        return ((det.roi_type or "").lower() in cfg.complex_rois
                or det.risk in ("high", "medium")
                or det.score < cfg.simple_score_floor
                or _mask_area_frac(det.mask) > cfg.large_area_frac)

    def _start_tier(self, det: DetectionResult, tiers: list) -> int:
        target = self.config.start_tier_complex if self._is_complex(det) else self.config.start_tier_simple
        eligible = [t for t in tiers if t <= target]
        return max(eligible) if eligible else tiers[0]

    def select_tier(self, det: DetectionResult, qa: QAReport | None, attempt: int) -> int:
        """First pass routes by type/score; retries escalate one rung up the sorted ladder; a very
        low qa_score jumps straight to the strongest tier. Pure function of (det, qa, attempt)."""
        tiers = self._tiers()
        idx = tiers.index(self._start_tier(det, tiers)) + attempt
        if attempt > 0 and qa is not None and qa.qa_score < self.config.qa_band_jump:
            idx = len(tiers) - 1
        return tiers[min(idx, len(tiers) - 1)]

    # ── the control loop ──
    def process(self, image) -> PipelineOutcome:
        det = self.detector.detect(image)
        if not det.has_watermark:                       # skip = pass-through, zero cleaners
            return PipelineOutcome(Status.SKIP, image, 0, None, [])

        qa, result, trace, prev_tier = None, None, [], None
        for attempt in range(self.MAX_RETRIES):
            tier = self.select_tier(det, qa, attempt)
            if prev_tier is not None and tier <= prev_tier:
                break                                   # escalation exhausted — never re-run a tier
            result = self.cleaners[tier].clean(
                CleanRequest(image, det.mask, det.watermark_type, tier, attempt, qa,
                             bbox=det.bbox, roi_type=det.roi_type))
            qa = self.validator.validate(image, result.image, det.mask)
            # per-tier threshold can only TIGHTEN — never resurrects a validator fail.
            thr = self.config.tier_qa_threshold.get(tier)
            passed = qa.passed and (thr is None or qa.qa_score >= thr)
            trace.append({"attempt": attempt, "tier": tier, "cleaner": self.cleaners[tier].name,
                          "qa_score": round(qa.qa_score, 3), "removal_score": round(qa.removal_score, 3),
                          "fidelity_score": round(qa.fidelity_score, 3), "verdict": "pass" if passed else "fail",
                          "failed_checks": qa.failed_checks, "decision": qa.decision})
            if passed:
                return PipelineOutcome(Status.PASS, result.image, attempt + 1, qa, trace)
            if not qa.retryable:                        # terminal-damage verdict — don't escalate into damage
                break
            prev_tier = tier

        return PipelineOutcome(Status.MANUAL_REVIEW,
                               result.image if result is not None else image, len(trace), qa, trace)
