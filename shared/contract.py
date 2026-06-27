#!/usr/bin/env python3
"""shared/contract.py — the ONE coupling point between the six agents.

Every agent (detector · orchestrator · classic/neural/diffusion cleaner · validator) talks
to the rest of the system ONLY through the types and interfaces defined here. No agent
imports another agent's implementation. Changing this file affects everyone, so it changes
only on a dedicated `contract/*` branch with the contract tests green (see agents/README.md §4).

Two roles in one file:
  • the typed handoff record  — DetectionResult · CleanRequest · CleanResult · QAReport · PipelineOutcome
  • the agent interfaces       — Cleaner (ABC) · Detector / Validator (Protocols)

Naming follows the design-v2 charters (`watermark_type`, `evidence`, `failed_checks`, `verdict`).
The record may also be persisted as JSONL; these dataclasses are its in-process typed form.

Status note — MANUAL_REVIEW is the single failure terminal. On disk / in the frozen CONTRACT_v1
manifest it is written `auto_rejected` (original restored from backup); `Status.terminal_label`
gives that label. There is no second FAIL state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

# BGR uint8 image and a region mask, matching the cv2 convention used across the repo.
Image = np.ndarray
Mask = np.ndarray


class WatermarkType(str, Enum):
    """What KIND of mark it is. This is explainable metadata carried on the record, NOT a routing key —
    the orchestrator routes on roi_type / risk / score, so a cleaner never branches on this. The detector
    currently distinguishes the three it can tell apart by signal; CORNER_STAMP and TILED are part of the
    domain taxonomy but not yet emitted (they stay defined so the detector can adopt them additively)."""
    SEMI_TRANSPARENT_TEXT = "semi_transparent_text"   # the sunsky-online.com mark
    OPAQUE_LOGO = "opaque_logo"
    CORNER_STAMP = "corner_stamp"                      # forward taxonomy — not yet emitted by the detector
    TILED = "tiled"                                    # forward taxonomy — not yet emitted by the detector
    UNKNOWN = "unknown_mark"


class Status(str, Enum):
    """Closed set of terminal pipeline outcomes. PASS · SKIP · MANUAL_REVIEW.
    (`retry` is an internal loop state, never a terminal.)"""
    PASS = "PASS"
    SKIP = "SKIP"
    MANUAL_REVIEW = "MANUAL_REVIEW"

    @property
    def terminal_label(self) -> str:
        """CONTRACT_v1 on-disk terminal this status maps to (manifest compatibility)."""
        return {"PASS": "published", "SKIP": "clean", "MANUAL_REVIEW": "auto_rejected"}[self.value]


class FailureReason(str, Enum):
    """Centralized taxonomy of cleaner-tier failure codes.

    When a cleaner cannot safely complete a request it sets
    ``CleanResult.meta["failure_reason"]`` to one of these values.
    The Orchestrator inspects this to decide whether to escalate to the next
    tier, bypass a tier, or terminate to MANUAL_REVIEW.

    Being a ``str`` subclass makes every member JSON-serialisable and safe to
    embed in ``CleanResult.meta`` or JSONL trace records without conversion.
    """

    # ── Tier 1 · Classic Cleaner ─────────────────────────────────────────────
    MASK_OVER_PRODUCT_DETAIL = "mask_over_product_detail"
    """Classic (Tier 1): mask intersects a labeled product-detail region; inpainting
    would erase content the QA gate would flag as product damage."""

    BACKGROUND_TOO_COMPLEX = "background_too_complex"
    """Classic (Tier 1): background is textured, gradient, or metallic — Telea/NS
    inpainting is unreliable and the result would not pass the fidelity check."""

    UNSAFE_FOR_CLASSIC = "unsafe_for_classic"
    """Classic (Tier 1): general-purpose bail-out; Classic declines the request
    without a more specific reason (e.g. unusual watermark geometry)."""

    # ── Tier 2 · Neural Cleaner ──────────────────────────────────────────────
    SEMANTIC_RISK = "semantic_risk"
    """Neural (Tier 2): the inpaint region overlaps semantic product features;
    completing the fill would alter the product's perceived meaning or category."""

    PRODUCT_DAMAGE = "product_damage"
    """Neural (Tier 2): prior-tier output or the current attempt has introduced
    (or is predicted to introduce) visible structural damage to the product."""

    MASK_TOO_SMALL = "mask_too_small"
    """Neural (Tier 2): the mask region is too narrow for the Neural diffuser's
    minimum receptive field; result would be artefact-prone."""

    # ── Tiers 1 + 2 · Classic and Neural ────────────────────────────────────
    MASK_TOO_LARGE = "mask_too_large"
    """Classic (Tier 1) + Neural (Tier 2): mask covers an unsafe proportion of the
    image area — Classic would lose too much background context; Neural risks
    hallucinating replacement content over a large region."""

    # ── Tier 3 · Diffusion Cleaner ───────────────────────────────────────────
    SEMANTIC_PRODUCT_RISK = "semantic_product_risk"
    """Diffusion (Tier 3): the diffusion model may hallucinate product-specific
    features (labels, ports, connectors) that were not present in the original."""

    PRODUCT_IDENTITY_CHANGE = "product_identity_change"
    """Diffusion (Tier 3): completing the inpaint would alter the product's visual
    identity or brand markings beyond the watermark region."""

    HALLUCINATION_RISK = "hallucination_risk"
    """Diffusion (Tier 3): high probability that the diffusion model invents new
    background or product content rather than faithfully restoring the scene."""

    @property
    def is_retryable(self) -> bool:
        """Whether the orchestrator may keep trying (escalate to a stronger tier) after a
        cleaner refuses with this reason.

        Capability/complexity limits (a weaker tier simply cannot do this job) ARE retryable —
        a stronger tier may succeed. Product- and semantic-integrity risks are NOT: any further
        automated inpainting would risk irreversible product damage, so they terminate straight
        to MANUAL_REVIEW. See ``NON_RETRYABLE_FAILURE_REASONS`` for the authoritative set.
        """
        return self not in NON_RETRYABLE_FAILURE_REASONS

    # Implementation status (2026-06): the deterministic cleaners emit only the reasons they can
    # decide WITHOUT ML — Classic: BACKGROUND_TOO_COMPLEX (textured surface, no texture-safe engine),
    # MASK_TOO_LARGE; Neural: MASK_TOO_LARGE; Diffusion: the operational "infrastructure_missing"
    # string. The semantic / product-damage reasons (SEMANTIC_RISK, PRODUCT_DAMAGE,
    # SEMANTIC_PRODUCT_RISK, PRODUCT_IDENTITY_CHANGE, HALLUCINATION_RISK, MASK_OVER_PRODUCT_DETAIL,
    # MASK_TOO_SMALL, UNSAFE_FOR_CLASSIC) are taxonomy the VALIDATOR catches post-hoc today; they
    # stay defined so a cleaner can adopt them later with no contract change (open/closed).


# The authoritative branch source the Orchestrator consults: reasons on which it must NOT escalate
# (escalating a generative tier into a product-integrity risk only deepens the damage). Everything
# NOT listed here is a tier-capability limit a stronger tier may still clear. Operational refusals
# that are not enum members (e.g. a cleaner's "infrastructure_missing") are likewise absent here, so
# they remain escalatable — the orchestrator skips the dead tier and terminates only if none is left.
NON_RETRYABLE_FAILURE_REASONS: frozenset = frozenset({
    FailureReason.MASK_OVER_PRODUCT_DETAIL,   # Tier 1: mask sits on labeled product detail
    FailureReason.SEMANTIC_RISK,              # Tier 2: fill would alter product meaning
    FailureReason.PRODUCT_DAMAGE,             # Tier 2: damage present / predicted
    FailureReason.SEMANTIC_PRODUCT_RISK,      # Tier 3: may hallucinate product-specific features
    FailureReason.PRODUCT_IDENTITY_CHANGE,    # Tier 3: would alter visual identity / branding
    FailureReason.HALLUCINATION_RISK,         # Tier 3: likely invents new content
})


@dataclass
class DetectionResult:
    """detector → orchestrator. ``has_watermark is False`` short-circuits to SKIP."""
    has_watermark: bool
    mask: Optional[Mask]                 # glyph + halo region; None when has_watermark is False
    watermark_type: WatermarkType
    score: float                         # normalized confidence, 0..1
    bbox: Optional[list] = None          # [x1,y1,x2,y2] px, original image space
    roi_type: str = "unknown"            # what is UNDER the mark — drives routing
    risk: str = "low"                    # low · medium · high
    evidence: list = field(default_factory=list)   # explainable signals (ocr/template/center…)


@dataclass
class CleanRequest:
    """orchestrator → cleaner. Everything a tier needs; no tier-specific calls leak out."""
    image: Image
    mask: Mask
    watermark_type: WatermarkType
    tier: int
    attempt: int                         # 0-based
    prior_qa: Optional["QAReport"] = None
    bbox: Optional[list] = None
    roi_type: str = "unknown"
    retry_box: Optional[list] = None     # [x, y, w, h] region hint for an intra-tier local re-inpaint;
    #                                      None on the first attempt at a tier. When present, the cleaner
    #                                      MUST focus repair strictly inside this box (region-targeted retry).


@dataclass
class CleanResult:
    """cleaner → orchestrator. The cleaner returns a result; it never self-publishes.

    ``status="refused"`` is a first-class outcome: the cleaner declined the job (no backend, or a
    capability / product-risk bail-out) and set ``meta["failure_reason"]``. The orchestrator branches
    on it — escalate on a retryable reason, terminate to MANUAL_REVIEW on a non-retryable one — without
    wasting a validate pass on an unchanged image. ``status="noop"`` means the engine ran but changed
    nothing (e.g. empty mask); the orchestrator escalates rather than re-validating an identical image."""
    image: Image
    tier_used: int
    status: str = "cleaned"              # cleaned · covered · noop · refused
    meta: dict = field(default_factory=dict)   # agent, method, engine, timing; failure_reason when refused


@dataclass
class QAReport:
    """validator → orchestrator. Carries NO retry/loop state (the validator is stateless).

    ``retryable`` describes the failure (a terminal product/text-damage verdict sets it False so
    the orchestrator escalates to MANUAL_REVIEW instead of into more damage); it never counts attempts.
    """
    passed: bool
    qa_score: float                      # combined, 0..1
    removal_score: float                 # is the watermark gone inside the mask?
    fidelity_score: float                # are the surroundings intact?
    failed_checks: list = field(default_factory=list)   # residual_watermark · visible_patch · …
    notes: str = ""
    retryable: bool = True
    decision: str = ""                   # production audit_decision when available
    retry_box: Optional[list] = None     # [x, y, w, h] of the residual region to re-target on a local
    #                                      retry; set ONLY on a retryable failure the same tier could still
    #                                      clear with a focused re-inpaint. None ⇒ no local hint (escalate).

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"


@dataclass
class PipelineOutcome:
    """orchestrator.process result."""
    status: Status
    image: Image
    attempts: int
    final_qa: Optional[QAReport]
    trace: list = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "terminal": self.status.terminal_label,
                "attempts": self.attempts, "trace": self.trace,
                "final_qa": None if self.final_qa is None else {
                    "verdict": self.final_qa.verdict, "qa_score": round(self.final_qa.qa_score, 3),
                    "removal_score": round(self.final_qa.removal_score, 3),
                    "fidelity_score": round(self.final_qa.fidelity_score, 3),
                    "failed_checks": self.final_qa.failed_checks,
                    "decision": self.final_qa.decision, "notes": self.final_qa.notes}}


# ───────────────────────────── agent interfaces ─────────────────────────────
class Cleaner(ABC):
    """The one cleaner interface. Every tier implements it; the orchestrator depends on this
    ABC + a {tier: Cleaner} registry, never on a concrete tier — so adding a tier is
    registration, not an orchestrator edit (open/closed)."""
    tier: int
    name: str
    supports_retry_box: bool = False
    """Whether this cleaner acts on ``CleanRequest.retry_box`` (region-targeted local re-inpaint).
    A region-blind cleaner (e.g. the deterministic classic tier, which re-runs identically whatever
    the box) leaves this False, so the orchestrator does NOT spend Phase-1 intra-tier retries on it —
    a focused retry would only reproduce the same result. Cleaners that honor the box set it True."""

    @abstractmethod
    def clean(self, req: CleanRequest) -> CleanResult:
        ...


@runtime_checkable
class DetectorAPI(Protocol):
    """detector agent interface (the concrete class lives in agents/detector/)."""
    def detect(self, image: Image) -> DetectionResult: ...


@runtime_checkable
class ValidatorAPI(Protocol):
    """validator agent interface — stateless QA. ``validate`` gets original + cleaned + mask."""
    def validate(self, original: Image, cleaned: Image, mask: Mask) -> QAReport: ...
