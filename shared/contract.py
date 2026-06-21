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
    SEMI_TRANSPARENT_TEXT = "semi_transparent_text"   # the sunsky-online.com mark
    OPAQUE_LOGO = "opaque_logo"
    CORNER_STAMP = "corner_stamp"
    TILED = "tiled"
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


@dataclass
class CleanResult:
    """cleaner → orchestrator. The cleaner returns a result; it never self-publishes."""
    image: Image
    tier_used: int
    status: str = "cleaned"              # cleaned · covered · noop
    meta: dict = field(default_factory=dict)   # agent, method, engine, timing


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
