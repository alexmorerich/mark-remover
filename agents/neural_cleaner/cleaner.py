#!/usr/bin/env python3
"""neural-cleaner agent — tier 2, LaMa single-pass inpaint (charter: agents/neural_cleaner/AGENT.md).

Wraps `run_bulk._lama_crop_inpaint`: inpaints only the mask's bounding region and hard-pastes the
unmasked pixels back, so a stray detection can only ever touch its own box. Handles textured /
larger regions the classic tier smears, while preserving product detail outside the mask.

Boundaries (contract tests): returns a cleaned result and NEVER mutates the input; no detect/route/QA.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from shared.contract import CleanRequest, CleanResult, Cleaner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        out, did = rb._lama_crop_inpaint(req.image, mask)   # returns a fresh array; input untouched
        return CleanResult(out, self.tier, status="cleaned",
                           meta={"agent": self.name, "method": "lama_crop_inpaint",
                                 "engine": "run_bulk", "did_inpaint": bool(did),
                                 "device": self._device, "ms": round((time.time() - t0) * 1000, 1)})
