#!/usr/bin/env python3
"""V19 — Product-text protection (patch plan §1.7).

Printed product labels (SKU strings, "Original", connector pin numbers, colour
names) must never be flattened by a cover or a destructive inpaint. The legacy
``_protected_text_overlap`` heuristic in ``v17_final_audit`` is a coarse
high-contrast-stroke fraction; this module adds an optional PP-OCRv3 ONNX text
detector that returns precise text quads, with a graceful fall back to the
heuristic when ONNX / the model is unavailable.

The resulting :class:`ProductTextMask` is used to:
  * ban destructive covers / residual inpaint over product text,
  * classify ``protected_text_overlap``,
  * ALLOW reverse-alpha over product text (it subtracts the overlay rather than
    repainting the underlying glyphs), while still forbidding a destructive
    residual cleanup there.

Pure-stdlib + OpenCV; onnxruntime is imported lazily and never required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PPOCR_MODEL = SCRIPT_DIR / "assets" / "ppocrv3_det.onnx"

Quad = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


@dataclass
class ProductTextMask:
    mask: np.ndarray                       # uint8 (0/255), full image size
    boxes: List[Quad] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "heuristic_fallback"     # "ppocrv3" | "heuristic_fallback"

    def overlap_in_box(self, bbox) -> float:
        bx, by, bw, bh = bbox
        sub = self.mask[by:by + bh, bx:bx + bw]
        if sub.size == 0:
            return 0.0
        return float((sub > 0).mean())


_SESSION = None
_SESSION_TRIED = False


def _load_ppocr_session():
    global _SESSION, _SESSION_TRIED
    if _SESSION_TRIED:
        return _SESSION
    _SESSION_TRIED = True
    if not PPOCR_MODEL.exists():
        return None
    try:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(
            str(PPOCR_MODEL), providers=["CPUExecutionProvider"])
    except Exception:
        _SESSION = None
    return _SESSION


def _ppocr_detect(image, bbox) -> Optional[ProductTextMask]:
    sess = _load_ppocr_session()
    if sess is None:
        return None
    try:
        H, W = image.shape[:2]
        inp_h, inp_w = 480, 480
        blob = cv2.resize(image, (inp_w, inp_h)).astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5
        blob = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        prob = out[0, 0]
        seg = (prob > 0.3).astype(np.uint8) * 255
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros((H, W), np.uint8)
        boxes: List[Quad] = []
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 12:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(int)
            cv2.fillPoly(mask, [box], 255)
            boxes.append(tuple(map(tuple, box)))
        return ProductTextMask(mask=mask, boxes=boxes,
                               confidence=float(prob.max()), source="ppocrv3")
    except Exception:
        return None


def _heuristic_detect(image, bbox, watermark_mask=None) -> ProductTextMask:
    """High-contrast non-watermark strokes inside the band — the legacy product
    text proxy, but returned as a full-image mask so callers can subtract it."""
    H, W = image.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    bx, by, bw, bh = bbox
    # Look a little wider than the watermark box — labels often sit beside it.
    px = int(bw * 0.25)
    py = int(bh * 0.6)
    x0, y0 = max(0, bx - px), max(0, by - py)
    x1, y1 = min(W, bx + bw + px), min(H, by + bh + py)
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return ProductTextMask(mask=mask)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    surface = float(np.median(g))
    high = (np.abs(g - surface) > 45).astype(np.uint8) * 255
    # Morphological close into text-like blobs; drop large product structures.
    high = cv2.morphologyEx(high, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(high, connectivity=8)
    keep = np.zeros_like(high)
    roi_area = roi.shape[0] * roi.shape[1]
    boxes: List[Quad] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w_i = int(stats[i, cv2.CC_STAT_WIDTH])
        h_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        # text-glyph sized: not a speck, not a huge product region.
        if a < 6 or a > roi_area * 0.18:
            continue
        if h_i > roi.shape[0] * 0.8 or w_i > roi.shape[1] * 0.9:
            continue
        keep[labels == i] = 255
        gx, gy = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        boxes.append((((x0 + gx, y0 + gy), (x0 + gx + w_i, y0 + gy),
                       (x0 + gx + w_i, y0 + gy + h_i), (x0 + gx, y0 + gy + h_i))))
    # Exclude the watermark's own strokes.
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        wm = (watermark_mask[y0:y1, x0:x1] > 0)
        keep[wm] = 0
    mask[y0:y1, x0:x1] = keep
    conf = float((keep > 0).mean())
    return ProductTextMask(mask=mask, boxes=boxes, confidence=conf,
                           source="heuristic_fallback")


def detect_product_text(image, bbox, watermark_mask=None) -> ProductTextMask:
    """Return a :class:`ProductTextMask`. Uses PP-OCRv3 ONNX when available,
    else the high-contrast-stroke heuristic (patch plan §1.7)."""
    if isinstance(bbox, dict):
        bbox = (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
    ppocr = _ppocr_detect(image, bbox)
    if ppocr is not None and ppocr.boxes:
        return ppocr
    return _heuristic_detect(image, bbox, watermark_mask)


def ppocr_available() -> bool:
    return _load_ppocr_session() is not None
