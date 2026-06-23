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

from shared.contract import CleanRequest, CleanResult, Cleaner, FailureReason
from shared.roi import TEXTURED_ROIS, WHITE_ROIS

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mask_bbox(mask):
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class ClassicCleaner(Cleaner):
    tier = 1
    name = "classic-cleaner"
    supports_retry_box = False           # deterministic: a focused re-inpaint reproduces the same
    #                                      result, so the orchestrator must not spend intra-tier retries here
    MAX_AREA_FRAC = 0.50                  # mask over half the image → decline (too little context to recover)

    def __init__(self):
        self._ppc = None
        self._cv2 = None

    def _refuse(self, image, reason, note) -> CleanResult:
        return CleanResult(image, self.tier, status="refused",
                           meta={"agent": self.name, "failure_reason": reason, "note": note})

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

        # ── mask-too-large bail-out (retryable → orchestrator escalates to a stronger tier) ──
        m = np.asarray(req.mask)
        if m.size and float((m > 0).mean()) > self.MAX_AREA_FRAC:
            return self._refuse(req.image, FailureReason.MASK_TOO_LARGE,
                                f"mask covers >{self.MAX_AREA_FRAC:.0%} of the image — too much for classic")

        if roi in WHITE_ROIS:
            out, method = ppc.clean(bgr), "structure_preserve"
        else:
            glyph = getattr(ppc, "glyph_clean", None)   # texture-safe path, if the damage-fix has landed
            box = req.bbox or _mask_bbox(req.mask)
            if glyph is not None and box is not None:
                out, method = glyph(bgr, box), "glyph_telea"
            elif roi in TEXTURED_ROIS:
                # ── textured / dark / specular surface and no texture-safe engine in this build: plain
                # Telea would SCAR the product (the historical damage path). Decline → escalate to neural
                # (LaMa), which is texture-safe. Retryable reason, so the orchestrator climbs the ladder. ──
                return self._refuse(req.image, FailureReason.BACKGROUND_TOO_COMPLEX,
                                    f"textured roi '{roi}' needs a texture-safe engine (glyph_clean) "
                                    "absent in this build — escalate to neural")
            else:
                out, method = self._telea(bgr, req.mask), "telea"   # low-texture / simple / unknown surface
        return CleanResult(out, self.tier, status="cleaned",
                           meta={"agent": self.name, "method": method,
                                 "engine": "product_preserve_clean/cv2",
                                 "ms": round((time.time() - t0) * 1000, 1)})
