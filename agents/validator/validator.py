#!/usr/bin/env python3
"""validator agent — independent QA gate (charter: agents/validator/AGENT.md).

Scores cleaned-vs-original WITHIN the mask and emits a QAReport with a `removal_score` (is the
watermark gone inside the mask?) and a `fidelity_score` (are the surroundings intact — no visible
patch, no over-blur, no tampering outside the mask?). "removed but blurred/patched" fails QA. The
validator is STATELESS w.r.t. retries: it returns a report and nothing else, and never repairs pixels.

Two implementations, same interface:
  • Validator      — numpy-only heuristic. Dependency-light, deterministic, the default for tests and
                     quick runs. Sound on flat/white backgrounds (the dominant real case); a proxy,
                     not the semantic OCR check.
  • AuditValidator — delegates to the production `audit.audit_pair` (OCR/template residual + damage /
                     patch / protected-text gates). The real publish gate.

Boundaries (contract tests): returns only a QAReport; does not modify pixels, call cleaners, or count retries.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from shared.contract import Image, Mask, QAReport

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EPS = 1e-6


def _gray(img: Image) -> np.ndarray:
    a = np.asarray(img, np.float64)
    return 0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2] if a.ndim == 3 else a


def _boolmask(mask: Mask) -> np.ndarray:
    return np.asarray(mask) > 0


def _dilate(m: np.ndarray, k: int = 6) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        out[:-1, :] |= out[1:, :]; out[1:, :] |= out[:-1, :]
        out[:, :-1] |= out[:, 1:]; out[:, 1:] |= out[:, :-1]
    return out


def _contrast(gray: np.ndarray, m: np.ndarray) -> float:
    if m.sum() < 4:
        return 0.0
    gy, gx = np.gradient(gray)
    return float((np.abs(gx) + np.abs(gy))[m].mean())


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class Validator:
    """Heuristic QA gate. ``validate(original, cleaned, mask) -> QAReport``."""

    OUT_REF = 12.0       # gray-levels of mean change outside the mask that = full fidelity loss
    PATCH_REF = 30.0     # in-mask vs ring brightness gap that = a fully visible patch
    TEX_FLOOR = 8.0      # ring gradient below this ⇒ surroundings smooth, no blur penalty

    def __init__(self, qa_threshold: float = 0.70):
        self.qa_threshold = qa_threshold

    def validate(self, original: Image, cleaned: Image, mask: Mask) -> QAReport:
        go, gc, m = _gray(original), _gray(cleaned), _boolmask(mask)

        removal = _clamp01(1.0 - _contrast(gc, m) / (_contrast(go, m) + _EPS))

        outside = ~m
        out_diff = float(np.abs(go - gc)[outside].mean()) if outside.any() else 0.0
        outside_fid = _clamp01(1.0 - out_diff / self.OUT_REF)
        ring = _dilate(m, 6) & ~m
        ring_c, in_c = _contrast(gc, ring), _contrast(gc, m)
        tex_fid = 1.0 if ring_c < self.TEX_FLOOR else _clamp01(in_c / (ring_c + _EPS))
        ring_mean = float(gc[ring].mean()) if ring.any() else 0.0
        in_mean = float(gc[m].mean()) if m.any() else 0.0
        patch_fid = _clamp01(1.0 - abs(in_mean - ring_mean) / self.PATCH_REF)
        fidelity = min(outside_fid, tex_fid, patch_fid)

        qa = min(removal, fidelity)
        failed = []
        if removal < self.qa_threshold:
            failed.append("residual_watermark")
        if patch_fid < self.qa_threshold:
            failed.append("visible_patch")
        if tex_fid < self.qa_threshold:
            failed.append("structure_damage")
        if outside_fid < self.qa_threshold:
            failed.append("color_shift")
        return QAReport(passed=qa >= self.qa_threshold, qa_score=qa,
                        removal_score=removal, fidelity_score=fidelity, failed_checks=failed,
                        notes=f"heuristic removal={removal:.2f} fidelity={fidelity:.2f}")


class AuditValidator(Validator):
    """Production gate: delegates to `audit.audit_pair` via temp files, maps its rich decision into
    a QAReport. ``passed``/``decision``/``failed_checks`` come from audit; the two scores are derived
    from audit's named sub-scores so the orchestrator can band-route on retries.

    Non-retryable audit verdicts (product / protected-text / unnatural damage) set ``retryable=False``
    so the orchestrator exits to MANUAL_REVIEW rather than escalating into worse damage.
    """
    _RETRYABLE_ACTIONS = {"retry_repair", "try_cover"}

    def __init__(self, reader=None, qa_threshold: float = 0.70, tmp_dir: str | None = None):
        super().__init__(qa_threshold)
        self._reader = reader
        self._tmp = tmp_dir
        self._audit = None
        self._cv2 = None

    def _engine(self):
        if self._audit is None:
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
            import audit
            import cv2
            self._audit, self._cv2 = audit, cv2
        return self._audit

    def validate(self, original: Image, cleaned: Image, mask: Mask) -> QAReport:
        import tempfile
        audit, cv2 = self._engine(), self._cv2
        d = self._tmp or tempfile.mkdtemp(prefix="contract_qa_")
        op, fp, mp = (os.path.join(d, n) for n in ("orig.png", "final.png", "mask.png"))
        cv2.imwrite(op, original)
        cv2.imwrite(fp, cleaned)
        cv2.imwrite(mp, np.asarray(mask, np.uint8))
        rec = audit.audit_pair(op, fp, meta=None, mask_path=mp, reader=self._reader)

        s = rec.get("scores", {})
        removal = _clamp01(1.0 - max(s.get("residual_logo_score", 0.0), s.get("dot_chain_score", 0.0),
                                     s.get("ghost_text_score", 0.0)))
        fidelity = _clamp01(1.0 - max(s.get("patch_visibility_score", 0.0), s.get("product_damage_score", 0.0),
                                      s.get("texture_mismatch_score", 0.0), s.get("color_delta_score", 0.0),
                                      s.get("edge_damage_score", 0.0), s.get("protected_text_damage_score", 0.0)))
        action = rec.get("recommended_next_action", "auto_reject")
        failed = [e.get("type", "") for e in rec.get("evidence", []) if e.get("type")]
        return QAReport(passed=bool(rec.get("publish_allowed")), qa_score=min(removal, fidelity),
                        removal_score=removal, fidelity_score=fidelity, failed_checks=failed,
                        notes=rec.get("notes", ""), decision=rec.get("audit_decision", ""),
                        retryable=action in self._RETRYABLE_ACTIONS)
