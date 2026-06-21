#!/usr/bin/env python3
"""classic-cleaner agent — tier 1, deterministic OpenCV / Telea (charter: agents/classic_cleaner/AGENT.md).

White / near-white backgrounds → structure-preserving recovery (`product_preserve_clean.clean`).
Any other surface → glyph-tight Telea inpaint when the texture-safe `glyph_clean` is available,
otherwise a built-in `cv2.INPAINT_TELEA` over the detector mask (pure OpenCV — always present).
Cheap, fast, no model. Best for small / simple / semi-transparent marks.

Boundaries (contract tests): returns a cleaned result and NEVER mutates the input image; does not
detect, route, or judge. The input is copied defensively before any engine call.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from shared.contract import CleanRequest, CleanResult, Cleaner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WHITE_ROIS = ("white_bg", "white_background", "plain_white", "near_white",
               "near_white_background", "pure_background", "low_texture_background")


def _mask_bbox(mask):
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class ClassicCleaner(Cleaner):
    tier = 1
    name = "classic-cleaner"

    def __init__(self):
        self._ppc = None
        self._cv2 = None

    def _engine(self):
        if self._ppc is None:
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
            import product_preserve_clean as ppc
            import cv2
            self._ppc, self._cv2 = ppc, cv2
        return self._ppc

    def _telea(self, bgr, mask):
        cv2 = self._cv2
        m = np.asarray(mask, np.uint8)
        if m.max() == 0:
            return bgr
        return cv2.inpaint(bgr, (m > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)

    def clean(self, req: CleanRequest) -> CleanResult:
        ppc = self._engine()
        bgr = req.image.copy()                      # never mutate the original (contract red line)
        roi = (req.roi_type or "").lower()
        t0 = time.time()
        if roi in _WHITE_ROIS:
            out, method = ppc.clean(bgr), "structure_preserve"
        else:
            glyph = getattr(ppc, "glyph_clean", None)   # texture-safe path, if the damage-fix has landed
            box = req.bbox or _mask_bbox(req.mask)
            if glyph is not None and box is not None:
                out, method = glyph(bgr, box), "glyph_telea"
            else:
                out, method = self._telea(bgr, req.mask), "telea"
        return CleanResult(out, self.tier, status="cleaned",
                           meta={"agent": self.name, "method": method,
                                 "engine": "product_preserve_clean/cv2",
                                 "ms": round((time.time() - t0) * 1000, 1)})
