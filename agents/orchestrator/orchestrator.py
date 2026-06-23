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

from shared.contract import (CleanRequest, DetectionResult, NON_RETRYABLE_FAILURE_REASONS,
                             PipelineOutcome, QAReport, Status)

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
        """Escalation budget in ladder rungs. Defaults to the full ladder (len(tiers)) when the config
        leaves it None, so registering a tier extends reach with no config edit (open/closed)."""
        return self.config.max_retries if self.config.max_retries is not None else len(self._tiers())

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

    def select_tier(self, det: DetectionResult, qa: QAReport | None, escalation: int) -> int:
        """Map an escalation level to a tier. Level 0 is the type/score-routed start tier; each
        rung climbs one tier up the sorted ladder; a very low qa_score on a retry jumps straight to
        the strongest tier. Pure function of (det, qa, escalation).

        Note: only Phase-2 *tier escalation* advances ``escalation``. Phase-1 intra-tier local
        retries re-run the same tier and do NOT bump it, so this stays a clean ladder selector."""
        tiers = self._tiers()
        idx = tiers.index(self._start_tier(det, tiers)) + escalation
        if escalation > 0 and qa is not None and qa.qa_score < self.config.qa_band_jump:
            idx = len(tiers) - 1
        return tiers[min(idx, len(tiers) - 1)]

    @property
    def MAX_INTRA_TIER_RETRIES(self) -> int:
        return self.config.max_intra_tier_retries

    # ── the control loop: unified Region-Targeted Escalation ──
    def process(self, image) -> PipelineOutcome:
        """detect → skip/route → (clean → validate)* → pass / escalate / manual-review.

        Two retry semantics, unified into one loop:
          • Phase 1 — intra-tier local retry. A *retryable* validator failure that carries a
            ``retry_box`` re-runs the SAME tier focused strictly on that region, up to
            MAX_INTRA_TIER_RETRIES. Local repair is preferred before spending a stronger tier.
          • Phase 2 — tier escalation. When the local budget is spent (or there is no box), climb
            one rung of the sorted ladder.

        Failure has exactly one exit, MANUAL_REVIEW, reached when: the ladder/budget is exhausted; a
        validator verdict is non-retryable (terminal product/text damage — escalating would only add
        damage); or a cleaner refuses with a non-retryable ``failure_reason``. A cleaner refusal whose
        reason IS retryable (a tier-capability limit, or an operational miss like a missing backend)
        skips that tier and escalates instead of wasting a validate pass on an unchanged image.
        """
        det = self.detector.detect(image)
        if not det.has_watermark:                       # skip = pass-through, zero cleaners
            return PipelineOutcome(Status.SKIP, image, 0, None, [])

        top_tier = max(self._tiers())
        qa, result, trace, prev_tier = None, None, [], None
        escalation = 0          # tier-ladder rung (drives select_tier); ONLY Phase-2 advances it
        intra_retries = 0       # Phase-1 local re-inpaints already spent at the current tier
        retry_box = None        # region hint handed to the cleaner for a local retry

        # hard safety bound on cleaner invocations: every rung may burn its full local budget
        attempt_cap = self.MAX_RETRIES * (1 + self.MAX_INTRA_TIER_RETRIES) + 1
        for attempt in range(attempt_cap):
            tier = self.select_tier(det, qa, escalation)
            if prev_tier is not None and tier < prev_tier:     # ladder never regresses (defensive)
                break
            result = self.cleaners[tier].clean(
                CleanRequest(image, det.mask, det.watermark_type, tier, attempt, qa,
                             bbox=det.bbox, roi_type=det.roi_type, retry_box=retry_box))

            # ── a cleaner may REFUSE (diffusion w/ no backend, or a product-risk bail) ──
            if result.status == "refused":
                reason = result.meta.get("failure_reason")
                trace.append({"attempt": attempt, "tier": tier, "cleaner": self.cleaners[tier].name,
                              "verdict": "refused", "failure_reason": reason,
                              "note": result.meta.get("note", "")})
                if reason in NON_RETRYABLE_FAILURE_REASONS:    # product/semantic risk → terminal
                    break
                if tier >= top_tier or escalation + 1 >= self.MAX_RETRIES:
                    break                                      # nowhere left to escalate → MANUAL_REVIEW
                escalation += 1; intra_retries = 0; retry_box = None; prev_tier = tier
                continue

            # ── a cleaner may report NOOP (engine ran, changed nothing — empty / degenerate mask) ──
            # Validating an unchanged image is pointless (removal must fail); escalate straight away.
            if result.status == "noop":
                trace.append({"attempt": attempt, "tier": tier, "cleaner": self.cleaners[tier].name,
                              "verdict": "noop", "note": result.meta.get("note", "")})
                if tier >= top_tier or escalation + 1 >= self.MAX_RETRIES:
                    break
                escalation += 1; intra_retries = 0; retry_box = None; prev_tier = tier
                continue

            qa = self.validator.validate(image, result.image, det.mask)
            # per-tier threshold can only TIGHTEN — never resurrects a validator fail.
            thr = self.config.tier_qa_threshold.get(tier)
            passed = qa.passed and (thr is None or qa.qa_score >= thr)
            trace.append({"attempt": attempt, "tier": tier, "cleaner": self.cleaners[tier].name,
                          "qa_score": round(qa.qa_score, 3), "removal_score": round(qa.removal_score, 3),
                          "fidelity_score": round(qa.fidelity_score, 3), "verdict": "pass" if passed else "fail",
                          "failed_checks": qa.failed_checks, "decision": qa.decision,
                          "retry_box": qa.retry_box})
            if passed:
                return PipelineOutcome(Status.PASS, result.image, len(trace), qa, trace)
            if not qa.retryable:                        # terminal-damage verdict — don't escalate into damage
                break

            # ── Phase 1: local re-inpaint on the SAME tier when the validator pointed at a region ──
            # Only for cleaners that ACT on the box — a region-blind tier (classic) would re-run
            # identically, so spending the intra-tier budget there is guaranteed-wasted; escalate instead.
            supports_box = getattr(self.cleaners[tier], "supports_retry_box", False)
            if qa.retry_box is not None and supports_box and intra_retries < self.MAX_INTRA_TIER_RETRIES:
                intra_retries += 1
                retry_box = list(qa.retry_box)
                prev_tier = tier                        # stay on this tier; escalation unchanged
                continue

            # ── Phase 2: escalate one rung; stop when the ladder or the escalation budget is spent ──
            if tier >= top_tier or escalation + 1 >= self.MAX_RETRIES:
                break
            escalation += 1; intra_retries = 0; retry_box = None; prev_tier = tier

        # MANUAL_REVIEW ≡ on-disk `auto_rejected` = ORIGINAL restored from backup (Status.terminal_label).
        # Return the original, never the last failed/damaged attempt — so a caller that writes
        # outcome.image cannot publish a damaged image. The failed attempt's scores live in `trace`.
        return PipelineOutcome(Status.MANUAL_REVIEW, image, len(trace), qa, trace)
