#!/usr/bin/env python3
"""V23 — thin-flex / black-frame line-preserving repair (patch plan §Patch 4).

Thin flex cables and black frames are a common auto-reject source: a fill either
leaves residue or cuts the line / opens a notch. V23 detects the dark line / frame
geometry BEFORE repair, recovers the glyph footprint with non-destructive
reverse-alpha (which cannot cut the line because it keeps the real pixels), does a
capped micro cleanup only OUTSIDE the line skeleton, and verifies the line
continuity afterwards with the frozen V20 continuity detector. It returns
``None`` whenever continuity drops, an endpoint shifts, or a notch appears.

Every output still passes through the unchanged P0 audit downstream.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import v13_gates

try:
    import sunsky_reverse_alpha as _sra
except Exception:  # pragma: no cover
    _sra = None


def _as_tuple(bbox):
    if isinstance(bbox, dict):
        return (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
    return tuple(int(v) for v in bbox)


def _line_skeleton(image, bbox) -> Optional[np.ndarray]:
    """Return a bool mask (full image) of the dark line / frame pixels inside the
    box, or ``None`` if no strong line is present."""
    bx, by, bw, bh = _as_tuple(bbox)
    H, W = image.shape[:2]
    inner = image[by:by + bh, bx:bx + bw]
    if inner.size == 0:
        return None
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(inner, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=max(8, bw // 2), maxLineGap=4)
    if lines is None:
        return None
    sk = np.zeros((H, W), np.uint8)
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = ln
        cv2.line(sk, (bx + x1, by + y1), (bx + x2, by + y2), 255, 3)
    if int(sk.sum()) < 6:
        return None
    return sk > 0


def v23_thin_flex_line_restore(image, bbox, watermark_mask=None,
                               product_mask=None, mask_set=None) -> Optional[np.ndarray]:
    """Return a line-preserving repaired image, or ``None`` if no safe repair
    exists (or no line is present — let the generic candidates handle it)."""
    if _sra is None or not _sra.alpha_asset_available():
        return None
    bx, by, bw, bh = _as_tuple(bbox)
    if bw < 6 or bh < 4:
        return None
    skeleton = _line_skeleton(image, (bx, by, bw, bh))
    if skeleton is None:
        return None     # not a thin-flex / frame case

    before = v13_gates.detect_thin_flex_continuity_v20(image, image, (bx, by, bw, bh))

    # Reverse-alpha recovery (non-destructive — cannot cut the line).
    try:
        res = _sra.repair_sunsky_reverse_alpha(
            image, bbox, watermark_mask=watermark_mask, product_mask=product_mask,
            allow_thin_cleanup=True)
        out = res.image if (res is not None and res.image is not None) else None
    except Exception:
        out = None
    if out is None or out.shape != image.shape:
        return None

    # Capped micro cleanup ONLY outside the line skeleton, inside the footprint.
    if mask_set is not None:
        micro = (mask_set.safe_micro_mask > 0) & (~skeleton)
        if int(micro.sum()) >= 4:
            m = cv2.dilate((micro.astype(np.uint8) * 255), np.ones((3, 3), np.uint8))
            # Do not dilate onto the line.
            m[skeleton] = 0
            try:
                cleaned = cv2.inpaint(out, m, 2, cv2.INPAINT_NS)
                if cleaned.shape == out.shape:
                    out = cleaned
            except Exception:
                pass

    # Verify continuity: the line must not shrink / shift / gain a notch.
    after = v13_gates.detect_thin_flex_continuity_v20(image, out, (bx, by, bw, bh))
    if bool(after.get("hard_fail", False)):
        return None
    drop = float(after.get("continuity_drop", 0.0))
    if drop > (1.0 - 0.97):   # FLEX_LINE_CONTINUITY_RATIO = 0.97
        return None
    shift = float(after.get("edge_position_shift_px", 0.0))
    if shift > 2.0:
        return None
    return out
