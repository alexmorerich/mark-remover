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


def _mask_xywh(m: np.ndarray):
    """Bounding box of the truthy region as the agents-layer retry_box ``[x, y, w, h]``; None if empty."""
    ys, xs = np.where(np.asarray(m) > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _xyxy_to_xywh(box):
    """audit emits a residual hint box as [x1,y1,x2,y2]; the agents-layer retry_box is [x,y,w,h]."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


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
        # Region hint for a Phase-1 intra-tier retry: emit whenever residual remains (removal failed) —
        # that residual IS what a focused re-inpaint on the same tier should re-attack. Withhold it for a
        # PURE fidelity failure (mark removed but surroundings damaged): re-running this tier there would
        # only re-damage, so the orchestrator escalates instead. The local retry budget (≤2) bounds it.
        retry_box = _mask_xywh(m) if removal < self.qa_threshold else None
        return QAReport(passed=qa >= self.qa_threshold, qa_score=qa,
                        removal_score=removal, fidelity_score=fidelity, failed_checks=failed,
                        notes=f"heuristic removal={removal:.2f} fidelity={fidelity:.2f}",
                        retry_box=retry_box)


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
        # One fresh, auto-removed temp dir PER CALL (rooted at self._tmp when given): no leaked dirs
        # over a bulk run, and no shared-filename race between concurrent validations.
        if self._tmp:
            os.makedirs(self._tmp, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="contract_qa_", dir=self._tmp) as d:
            op, fp, mp = (os.path.join(d, n) for n in ("orig.png", "final.png", "mask.png"))
            cv2.imwrite(op, original)
            cv2.imwrite(fp, cleaned)
            cv2.imwrite(mp, np.asarray(mask, np.uint8))
            rec = audit.audit_pair(op, fp, meta=None, mask_path=mp, reader=self._reader)

        # Missing sub-scores default to MAX RISK (1.0 → score 0.0): a publish gate must fail SAFE on
        # audit schema drift, never silently treat an unreported risk as "clean".
        s = rec.get("scores", {})
        removal = _clamp01(1.0 - max(s.get("residual_logo_score", 1.0), s.get("dot_chain_score", 1.0),
                                     s.get("ghost_text_score", 1.0)))
        fidelity = _clamp01(1.0 - max(s.get("patch_visibility_score", 1.0), s.get("product_damage_score", 1.0),
                                      s.get("texture_mismatch_score", 1.0), s.get("color_delta_score", 1.0),
                                      s.get("edge_damage_score", 1.0), s.get("protected_text_damage_score", 1.0)))
        action = rec.get("recommended_next_action", "auto_reject")
        failed = [e.get("type", "") for e in rec.get("evidence", []) if e.get("type")]
        retryable = action in self._RETRYABLE_ACTIONS
        # Region-targeted retry hint: prefer audit's own residual box (it pinpoints where the mark still
        # is — emitted only on a retry_repair verdict); otherwise, on any retryable residual failure with
        # intact surroundings, fall back to the mask bbox. None ⇒ the orchestrator escalates the tier.
        hint = (rec.get("retry_hint") or {}).get("box")
        if hint:
            retry_box = _xyxy_to_xywh(hint)
        elif retryable and removal < self.qa_threshold:
            retry_box = _mask_xywh(_boolmask(mask))
        else:
            retry_box = None
        return QAReport(passed=bool(rec.get("publish_allowed")), qa_score=min(removal, fidelity),
                        removal_score=removal, fidelity_score=fidelity, failed_checks=failed,
                        notes=rec.get("notes", ""), decision=rec.get("audit_decision", ""),
                        retryable=retryable, retry_box=retry_box)
