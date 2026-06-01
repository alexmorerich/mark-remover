#!/usr/bin/env python3
"""V19 — optional CPU LaMA-ONNX crop/paste backend (patch plan §1.6).

A *candidate generator*, never a publish shortcut. It is used only for hard
stroke-only cases on a product surface where reverse-alpha and the segmented
repairs leave a residual, and ONLY when:

    mask_type == "stroke"
    product_overlap <= safe threshold
    protected_text_overlap == 0
    the candidate is crop-local

Behaviour (patch plan §1.6):
    1. crop a padded region around the mask,
    2. resize the crop to the model input size,
    3. run LaMA,
    4. resize the result back,
    5. paste ONLY the masked pixels back,
    6. leave every unmasked pixel byte-identical.

onnxruntime + the model file are optional. When unavailable, ``available()``
returns False and ``run_lama_crop_paste`` returns ``None`` so the caller simply
skips this generator. Every output it does produce still passes through the
unchanged V17/V19 final audit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAMA_MODEL = SCRIPT_DIR / "assets" / "lama_fp32.onnx"
MODEL_INPUT = 512
MASK_PAD = 24

_SESSION = None
_SESSION_TRIED = False


def _load_session():
    global _SESSION, _SESSION_TRIED
    if _SESSION_TRIED:
        return _SESSION
    _SESSION_TRIED = True
    if not LAMA_MODEL.exists():
        return None
    try:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(
            str(LAMA_MODEL), providers=["CPUExecutionProvider"])
    except Exception:
        _SESSION = None
    return _SESSION


def available() -> bool:
    return _load_session() is not None


def _bbox_of_mask(mask, pad, shape):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    H, W = shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def run_lama_crop_paste(image, stroke_mask, *, mask_type="stroke",
                        product_overlap=0.0, protected_text_overlap=0.0,
                        product_overlap_safe=0.10) -> Optional[np.ndarray]:
    """Crop around the mask, inpaint with LaMA, paste back ONLY masked pixels.
    Returns ``None`` (skip) unless every precondition holds and the model is
    available."""
    sess = _load_session()
    if sess is None:
        return None
    if mask_type != "stroke":
        return None
    if product_overlap > product_overlap_safe or protected_text_overlap > 1e-6:
        return None
    if stroke_mask is None or stroke_mask.shape[:2] != image.shape[:2]:
        return None
    crop_box = _bbox_of_mask(stroke_mask, MASK_PAD, image.shape)
    if crop_box is None:
        return None
    x0, y0, x1, y1 = crop_box
    crop = image[y0:y1, x0:x1]
    cmask = (stroke_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    ch, cw = crop.shape[:2]
    if ch < 4 or cw < 4:
        return None
    try:
        img_in = cv2.resize(crop, (MODEL_INPUT, MODEL_INPUT)).astype(np.float32) / 255.0
        msk_in = cv2.resize(cmask, (MODEL_INPUT, MODEL_INPUT),
                            interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))[None, ...]
        msk_in = msk_in[None, None, ...]
        names = [i.name for i in sess.get_inputs()]
        feed = {names[0]: img_in.astype(np.float32)}
        if len(names) > 1:
            feed[names[1]] = msk_in.astype(np.float32)
        out = sess.run(None, feed)[0]
        res = out[0]
        if res.shape[0] in (1, 3):
            res = np.transpose(res, (1, 2, 0))
        if res.max() <= 1.5:
            res = res * 255.0
        res = np.clip(res, 0, 255).astype(np.uint8)
        res = cv2.resize(res, (cw, ch))
    except Exception:
        return None
    out_img = image.copy()
    region = out_img[y0:y1, x0:x1]
    m = (cmask > 0)
    region[m] = res[m]
    out_img[y0:y1, x0:x1] = region
    return out_img
