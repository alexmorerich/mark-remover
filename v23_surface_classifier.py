#!/usr/bin/env python3
"""V23 — stronger low-texture product-surface classifier (patch plan §Patch 6).

Several product surfaces are visually smooth and were historically mistaken for
background by the early pipeline:

    * dark smooth battery backs,
    * near-white trays / plastic surfaces,
    * translucent stacked film,
    * metallic / glass plates,
    * thin black frames and flex cables,
    * small detached product components near the watermark.

V22 already classifies most of these (:func:`v22_patch.classify_surface_v22`).
V23 wraps that classifier and adds the extra signals the V23 patch plan asks for
without ever *loosening* a decision: every V23 signal can only mark MORE of the
box as product-like, which can only BAN a destructive fill, never allow one.

This module is additive and side-effect free. It imports only the frozen
detectors (``v13_gates``) and the V22 classifier; it is never imported by
``v17_final_audit`` / ``v22_patch`` so the dependency graph stays acyclic.
"""
from __future__ import annotations

import cv2
import numpy as np

import v13_gates

try:
    import v22_patch
except Exception:  # pragma: no cover
    v22_patch = None


# Thresholds (reuse V22's where they exist; keep V23-specific ones explicit).
NEAR_WHITE_LUMA = getattr(v22_patch, "NEAR_WHITE_LUMA", 235.0)
DARK_SMOOTH_LUMA = getattr(v22_patch, "DARK_SMOOTH_LUMA", 80.0)
SURFACE_SATURATION_MIN = getattr(v22_patch, "SURFACE_SATURATION_MIN", 18.0)
LONG_THIN_ASPECT = getattr(v22_patch, "LONG_THIN_ASPECT", 4.0)

# A small non-white component (label chip / connector) sitting inside the
# footprint must block a broad fill even when the surrounding box reads white.
SMALL_COMPONENT_MIN_AREA_FRAC = 0.004    # >= 0.4% of box area to count as a part
SMALL_COMPONENT_MAX_AREA_FRAC = 0.45     # <= 45% (above this it is the surface)


def _as_tuple(bbox):
    if isinstance(bbox, dict):
        return (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
    return tuple(int(v) for v in bbox)


def classify_surface_v23(image, bbox, product_mask=None, watermark_mask=None,
                         roi_class=""):
    """Return a superset of :func:`v22_patch.classify_surface_v22`'s dict with
    the additional V23 signals::

        {... all v22 keys ...,
         small_detached_components,        # bool: non-white parts inside footprint
         small_detached_component_count,   # int
         long_thin_component_crosses_bbox, # bool (re-affirmed)
         near_white_attached_product,      # bool: looks white but is a product face
         smooth_dark_product,              # bool
         translucent_stack,                # bool
         v23_product_like_overlap,         # float (>= v22's, never lower)
         v23_surface_class,                # resolved single label
         destructive_fill_unsafe}          # bool: any product-like signal present

    Every V23 signal is *conservative*: it only ever marks more of the box as
    product-like, so it can only BAN a destructive fill, never enable one.
    """
    bx, by, bw, bh = _as_tuple(bbox)
    # Base on the V22 classification (already validated + frozen-detector backed).
    if v22_patch is not None:
        try:
            base = dict(v22_patch.classify_surface_v22(
                image, bbox, product_mask, watermark_mask, roi_class))
        except Exception:
            base = {}
    else:
        base = {}
    out = {
        "near_white_product_surface": bool(base.get("near_white_product_surface")),
        "translucent_stack_surface": bool(base.get("translucent_stack_surface")),
        "dark_smooth_product_surface": bool(base.get("dark_smooth_product_surface")),
        "long_thin_component_crosses_bbox":
            bool(base.get("long_thin_component_crosses_bbox")),
        "connected_component_crosses_bbox":
            bool(base.get("connected_component_crosses_bbox")),
        "v22_product_like_overlap": float(base.get("v22_product_like_overlap", 0.0)),
        "surface_class": base.get("surface_class", roi_class or ""),
        # V23 additions:
        "small_detached_components": False,
        "small_detached_component_count": 0,
        "near_white_attached_product": bool(base.get("near_white_product_surface")),
        "smooth_dark_product": bool(base.get("dark_smooth_product_surface")),
        "translucent_stack": bool(base.get("translucent_stack_surface")),
    }

    H, W = image.shape[:2]
    if bw >= 4 and bh >= 4:
        inner = image[by:by + bh, bx:bx + bw]
        if inner.size:
            g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            # Exclude watermark strokes so the mark cannot read as a "part".
            not_wm = np.ones((bh, bw), bool)
            if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
                wm = watermark_mask[by:by + bh, bx:bx + bw] > 0
                wm = cv2.dilate(wm.astype(np.uint8), np.ones((3, 3), np.uint8),
                                iterations=2) > 0
                not_wm = ~wm
            # Small detached product components: non-white / coloured / dark
            # blobs inside the box that are NOT the dominant surface.
            nonwhite = (g.astype(np.float32) < NEAR_WHITE_LUMA)
            coloured = (sat.astype(np.float32) > SURFACE_SATURATION_MIN)
            dark = (g.astype(np.float32) < DARK_SMOOTH_LUMA)
            part = ((nonwhite | coloured | dark) & not_wm).astype(np.uint8)
            box_area = float(max(1, bw * bh))
            try:
                n, _lab, stats, _c = cv2.connectedComponentsWithStats(part, 8)
                cnt = 0
                for i in range(1, n):
                    a = float(stats[i, cv2.CC_STAT_AREA])
                    frac = a / box_area
                    if SMALL_COMPONENT_MIN_AREA_FRAC <= frac <= \
                            SMALL_COMPONENT_MAX_AREA_FRAC:
                        cnt += 1
                if cnt:
                    out["small_detached_components"] = True
                    out["small_detached_component_count"] = int(cnt)
            except Exception:
                pass

    # Re-affirm long-thin component (flex/frame) via the frozen overlap detector;
    # this can only ADD the signal, never remove V22's.
    try:
        ov = v13_gates.estimate_product_overlap_v13(image, (bx, by, bw, bh),
                                                    product_mask)
        long_line = float(ov.get("long_line_score", 0.0))
        if long_line > v13_gates.LONG_LINE_OVERRIDE:
            out["long_thin_component_crosses_bbox"] = True
    except Exception:
        pass

    # v23_product_like_overlap is never lower than v22's.
    out["v23_product_like_overlap"] = float(out["v22_product_like_overlap"])

    # Resolved single label (most specific wins); keep V22's when it is specific.
    sc = out.get("surface_class") or ""
    if not sc or sc in ("plain_white", "low_texture_background",
                        "simple_product_surface", "product_surface", ""):
        if out["translucent_stack"]:
            sc = "translucent_stack_surface"
        elif out["smooth_dark_product"]:
            sc = "dark_smooth_product_surface"
        elif out["near_white_attached_product"]:
            sc = "near_white_product_surface"
        elif out["long_thin_component_crosses_bbox"]:
            sc = "thin_flex_cable"
        elif out["small_detached_components"]:
            sc = "detached_components_surface"
        else:
            sc = sc or "plain_white"
    out["v23_surface_class"] = sc

    out["destructive_fill_unsafe"] = bool(
        out["near_white_attached_product"]
        or out["smooth_dark_product"]
        or out["translucent_stack"]
        or out["long_thin_component_crosses_bbox"]
        or out["connected_component_crosses_bbox"]
        or out["small_detached_components"])
    return out
