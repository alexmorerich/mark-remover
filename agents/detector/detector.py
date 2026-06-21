#!/usr/bin/env python3
"""detector agent — finds the watermark, emits mask + type + score (charter: agents/detector/AGENT.md).

Wraps the recall-max `logo_finder.find` ensemble and adapts its {presence, candidates} output
to a `DetectionResult`. ``has_watermark is False`` ⇒ the orchestrator routes to SKIP.

Boundaries (enforced by contract tests): does NOT modify pixels, route, retry, or pass/fail.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from shared.contract import DetectionResult, WatermarkType

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# logo_finder presence tiers that mean "treat as watermarked" (aggressive recall).
_PRESENT = ("CONFIRMED_WATERMARK", "LIKELY_WATERMARK", "UNCERTAIN_WATERMARK")


def _xywh_to_xyxy(box) -> list:
    x, y, w, h = box
    return [int(x), int(y), int(x + w), int(y + h)]


def _infer_type(cand) -> WatermarkType:
    ev = cand.get("evidence", {}) or {}
    if float(ev.get("alpha_gray_score", 0.0)) >= 0.3 or cand.get("text"):
        return WatermarkType.SEMI_TRANSPARENT_TEXT
    if cand.get("matched_template") not in (None, "", "canon-matte"):
        return WatermarkType.OPAQUE_LOGO
    return WatermarkType.UNKNOWN


class Detector:
    """``detect(image) -> DetectionResult``. Lazily imports logo_finder so the package loads
    without easyocr/torch; an OCR ``reader`` may be injected for full recall."""

    def __init__(self, reader=None, present_levels=_PRESENT):
        self._reader = reader
        self._present = tuple(present_levels)
        self._lf = None

    def _engine(self):
        if self._lf is None:
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
            import logo_finder as lf
            self._lf = lf
        return self._lf

    def _box_mask(self, shape, box) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(shape[1], x2), min(shape[0], y2)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 255
        return m

    def detect(self, image) -> DetectionResult:
        lf = self._engine()
        out = lf.find(image, self._reader)
        presence = out.get("presence", "NO_WATERMARK")
        cands = out.get("candidates") or []
        if presence not in self._present or not cands:
            return DetectionResult(False, None, WatermarkType.UNKNOWN, 0.0)

        best = max(cands, key=lambda c: float(c.get("score", 0.0)))
        bbox = _xywh_to_xyxy(best["bbox"])
        evidence = list((best.get("evidence", {}) or {}).keys())
        if best.get("text"):
            evidence = [f"ocr:{best['text']}"] + evidence
        return DetectionResult(
            has_watermark=True, mask=self._box_mask(image.shape, bbox),
            watermark_type=_infer_type(best), score=float(best.get("score", 0.0)),
            bbox=bbox, roi_type=best.get("roi_class", "unknown"),
            risk=best.get("risk_level", "low"), evidence=evidence[:6])
