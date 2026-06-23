#!/usr/bin/env python3
"""neural-cleaner agent — tier 2, LaMa single-pass inpaint (charter: agents/neural_cleaner/AGENT.md).

Wraps `run_bulk._lama_crop_inpaint`: inpaints only the mask's bounding region and hard-pastes the
unmasked pixels back, so a stray detection can only ever touch its own box. Handles textured /
larger regions the classic tier smears, while preserving product detail outside the mask.

Region-targeted retry: when the orchestrator passes a ``retry_box`` ([x, y, w, h]) — the validator's
residual region on an intra-tier retry — the cleaner restricts the inpaint mask to STRICTLY inside
that box, so the re-inpaint re-attacks only the residual instead of re-processing the whole image.

Boundaries (contract tests): returns a cleaned result and NEVER mutates the input; no detect/route/QA.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from shared.contract import CleanRequest, CleanResult, Cleaner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _focus_mask(mask: np.ndarray, box) -> np.ndarray:
    """Restrict ``mask`` to strictly inside the retry box ``[x, y, w, h]``.

    Keeps the repair glyph-tight (mask ∧ box) so a focused re-inpaint never repaints product detail
    that merely happens to fall inside the box. If the box and the mask do not overlap (e.g. the
    first pass left faint residual the detector mask never covered), fall back to the box rectangle
    itself — the validator is telling us the residual is there. An empty/degenerate box yields an
    all-zero mask, which `_lama_crop_inpaint` treats as a safe no-op."""
    x, y, w, h = (int(round(float(v))) for v in box)
    H, W = mask.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    focus = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        focus[y1:y2, x1:x2] = 255
    restricted = np.where(focus > 0, mask, np.uint8(0)).astype(np.uint8)
    return restricted if restricted.max() > 0 else focus


class NeuralCleaner(Cleaner):
    tier = 2
    name = "neural-cleaner"

    def __init__(self, device: str = "mps"):
        self._device = device
        self._rb = None

    def _engine(self):
        if self._rb is None:
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
            import run_bulk as rb
            rb._init_worker(self._device)           # pin the LaMa device for this worker
            self._rb = rb
        return self._rb

    def clean(self, req: CleanRequest) -> CleanResult:
        rb = self._engine()
        t0 = time.time()
        mask = np.asarray(req.mask)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        method = "lama_crop_inpaint"
        if req.retry_box is not None:                       # intra-tier retry: focus on the residual
            mask = _focus_mask(mask, req.retry_box)
            method = "lama_crop_inpaint_box"
        out, did = rb._lama_crop_inpaint(req.image, mask)   # returns a fresh array; input untouched
        return CleanResult(out, self.tier, status="cleaned",
                           meta={"agent": self.name, "method": method,
                                 "engine": "run_bulk", "did_inpaint": bool(did),
                                 "retry_box": list(req.retry_box) if req.retry_box is not None else None,
                                 "device": self._device, "ms": round((time.time() - t0) * 1000, 1)})
