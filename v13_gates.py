#!/usr/bin/env python3
"""V13 — Final Visual Fidelity gates.

Additive, post-candidate visual-truthfulness checks applied to BOTH
``clean_repaired`` and ``clean_covered`` outputs before they are published.

The V13 principle (see the integrated patch plan):

    Do not relax QA to increase clean_repaired. Improve repair quality so more
    candidates pass the existing strict gate honestly. Also apply the final
    visual gate to BOTH repaired and covered outputs.

So nothing in this module weakens a V12 gate. It *adds* shape / silhouette /
dot-chain detectors and one composite gate — ``final_visual_publish_gate_v13``
— that runs on the chosen output (repaired or covered) and refuses to publish a
visually-bad result as a success. A failing ``clean_repaired`` is demoted to an
honest ``clean_covered``; a failing ``clean_covered`` is reported truthfully via
the honesty counters (all of which must be 0 for an acceptable release).

The detectors are intentionally tuned to fire only on genuinely visible
artifacts (sharp-edged rectangles / wedges / polygons, bright blobs on dark
product surfaces, broken product contours, aligned dot-chain residue). A clean
inpaint on a plain or noisy surface — no border edge, no luma offset — must
pass, exactly as in V12, so good repairs are never demoted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Reuse V12 thresholds / detectors where they exist; fall back to safe defaults
# so this module is importable and unit-testable on its own.
try:  # pragma: no cover - exercised indirectly
    from progressive_repair import (
        RESIDUAL_DOT_CHAIN_MAX as _PR_DOT_CHAIN_MAX,
        BAND_VISIBLE_REPAIRED_MAX as _PR_BAND_REP_MAX,
        BAND_VISIBLE_COVERED_MAX as _PR_BAND_COV_MAX,
        BAND_LUMA_DELTA_REPAIRED_MAX as _PR_BAND_LUMA_REP_MAX,
        BAND_LUMA_DELTA_COVERED_MAX as _PR_BAND_LUMA_COV_MAX,
        detect_rectangular_band_visibility as _pr_band,
    )
    _HAVE_PR = True
except Exception:  # pragma: no cover
    _PR_DOT_CHAIN_MAX = 0.35
    _PR_BAND_REP_MAX = 0.18
    _PR_BAND_COV_MAX = 0.28
    _PR_BAND_LUMA_REP_MAX = 6.0
    _PR_BAND_LUMA_COV_MAX = 10.0
    _pr_band = None
    _HAVE_PR = False


# ---------------------------------------------------------------------------
# V13 thresholds. Stricter for clean_repaired, looser (but still real) for an
# honest clean_covered. Derived from the patch plan's "Reject examples".
# ---------------------------------------------------------------------------
DOT_CHAIN_V2_MAX = _PR_DOT_CHAIN_MAX
DOT_CHAIN_V2_COVERED_MAX = _PR_DOT_CHAIN_MAX + 0.10

VISIBLE_PATCH_REPAIRED_MAX = 0.22
VISIBLE_PATCH_COVERED_MAX = 0.34

STRAIGHT_EDGE_MAX = 0.55
POLYGONALITY_MAX = 0.50
RECTANGULARITY_PATCH_MAX = 0.20
HARD_BOUNDARY_MAX = 0.14

SILHOUETTE_CONTOUR_IOU_MIN = 0.985
CONTOUR_BREAK_MAX = 0.12
EDGE_RETENTION_MIN = 0.90
NEW_BRIGHT_BLOB_MAX = 0.08
DARK_SURFACE_RATIO_FOR_BLOB = 0.30

# estimate_product_overlap_v13 routing-override triggers.
OVERLAP_OVERRIDE = 0.08
DARK_RATIO_OVERRIDE = 0.25
EDGE_DENSITY_OVERRIDE = 0.08
LONG_LINE_OVERRIDE = 0.35


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _ring_mask(shape, bbox, frac=0.4):
    """Boolean mask of an annulus around ``bbox`` (the surrounding surface)."""
    bx, by, bw, bh = bbox
    H, W = shape[:2]
    pad = max(6, int(round(max(bw, bh) * frac)))
    y1, y2 = max(0, by - pad), min(H, by + bh + pad)
    x1, x2 = max(0, bx - pad), min(W, bx + bw + pad)
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    m[by:by + bh, bx:bx + bw] = False
    return m


def _changed_region(original, cleaned, bbox, pad_frac=0.5, thr=6.0):
    """Binary mask (uint8) of pixels that actually changed, within a padded
    window around the bbox. Returns (mask_full_image, (wy1, wx1))."""
    bx, by, bw, bh = bbox
    H, W = original.shape[:2]
    pad = max(8, int(round(max(bw, bh) * pad_frac)))
    wy1, wy2 = max(0, by - pad), min(H, by + bh + pad)
    wx1, wx2 = max(0, bx - pad), min(W, bx + bw + pad)
    og = cv2.cvtColor(original[wy1:wy2, wx1:wx2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    cg = cv2.cvtColor(cleaned[wy1:wy2, wx1:wx2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    changed = (np.abs(og - cg) > thr).astype(np.uint8) * 255
    full = np.zeros((H, W), dtype=np.uint8)
    full[wy1:wy2, wx1:wx2] = changed
    return full, (wy1, wx1)


# ---------------------------------------------------------------------------
# Section 9 — product overlap estimation that does not trust only the ring.
# ---------------------------------------------------------------------------
def estimate_product_overlap_v13(image, bbox, existing_product_mask=None):
    """Estimate how much the watermark footprint overlaps product pixels using
    bbox-interior signals (non-white / dark / edge density / long lines), not
    only the surrounding ring. Returns a dict including a routing override
    class when the bbox is clearly on product rather than plain background.
    """
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    out = {
        "product_overlap": 0.0,
        "non_white_ratio": 0.0,
        "dark_ratio_in_bbox": 0.0,
        "edge_density_in_bbox": 0.0,
        "long_line_score": 0.0,
        "override_class": None,
    }
    if bw <= 1 or bh <= 1:
        return out
    inner = image[by:by + bh, bx:bx + bw]
    if inner.size == 0:
        return out
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    out["non_white_ratio"] = float((g < 235).mean())
    out["dark_ratio_in_bbox"] = float((g < 80).mean())
    edges = cv2.Canny(g, 50, 150)
    out["edge_density_in_bbox"] = float((edges > 0).mean())

    # Long straight lines inside the bbox (cables / frames / connectors).
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(10, bw // 6),
                            minLineLength=max(10, bw // 3), maxLineGap=4)
    if lines is not None and len(lines) > 0:
        longest = 0.0
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            longest = max(longest, float(np.hypot(x2 - x1, y2 - y1)))
        out["long_line_score"] = _clamp01(longest / max(1.0, float(max(bw, bh))))

    if existing_product_mask is not None and \
            existing_product_mask.shape[:2] == image.shape[:2]:
        pm = existing_product_mask[by:by + bh, bx:bx + bw] > 0
        out["product_overlap"] = float(pm.mean())
    else:
        out["product_overlap"] = out["non_white_ratio"]

    if (out["product_overlap"] > OVERLAP_OVERRIDE or
            out["dark_ratio_in_bbox"] > DARK_RATIO_OVERRIDE or
            out["edge_density_in_bbox"] > EDGE_DENSITY_OVERRIDE or
            out["long_line_score"] > LONG_LINE_OVERRIDE):
        if out["long_line_score"] > LONG_LINE_OVERRIDE and \
                out["dark_ratio_in_bbox"] > DARK_RATIO_OVERRIDE:
            out["override_class"] = "thin_flex_cable"
        elif out["dark_ratio_in_bbox"] > DARK_RATIO_OVERRIDE:
            out["override_class"] = "dark_product_surface"
        elif out["edge_density_in_bbox"] > EDGE_DENSITY_OVERRIDE:
            out["override_class"] = "complex_product_detail"
        else:
            out["override_class"] = "text_or_label_area"
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# Section 7 — residual dot-chain detection v2 (expanded search window).
# ---------------------------------------------------------------------------
def detect_residual_dot_chain_v2(original, cleaned, bbox, mask=None):
    """Detect a readable line of aligned residual fragments (broken glyphs /
    dot-chain) left after cleaning, searching a horizontally-expanded window so
    trailing ``.com`` glyphs outside the detected bbox are still caught.
    """
    bx, by, bw, bh = bbox
    H, W = original.shape[:2]
    out = {
        "dot_chain_score": 0.0,
        "component_count": 0,
        "component_area_ratio": 0.0,
        "horizontal_span_ratio": 0.0,
        "alignment_score": 0.0,
        "mean_component_luma_delta": 0.0,
        "passed": True,
    }
    if bw <= 1 or bh <= 1:
        return out
    # Expanded residual search window (x 20-30%, y 35-50%, min 8px).
    ex = max(8, int(round(bw * 0.28)))
    ey = max(8, int(round(bh * 0.45)))
    y1, y2 = max(0, by - ey), min(H, by + bh + ey)
    x1, x2 = max(0, bx - ex), min(W, bx + bw + ex)
    og = cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    cg = cv2.cvtColor(cleaned[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if og.size == 0:
        return out

    # Surface baseline from the ring around the window.
    ring = _ring_mask(original.shape, bbox, 0.5)
    rg = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ring_vals = rg[ring]
    base = float(np.median(ring_vals)) if ring_vals.size else float(np.median(cg))
    noise = float(ring_vals.std()) if ring_vals.size else float(cg.std())
    noise = max(noise, 1.0)

    # Residual strokes/fragments stand above the surrounding surface either as
    # a high-pass edge response (thin glyphs) OR as a solid luma deviation from
    # the baseline (filled dot/glyph fragments). Use both so a solid fragment
    # registers as ONE component, not just its border ring.
    hp = np.abs(cg - cv2.GaussianBlur(cg, (0, 0), sigmaX=2.0))
    base_hp = float(np.median(np.abs(rg[ring] -
                    cv2.GaussianBlur(rg, (0, 0), 2.0)[ring]))) if ring_vals.size else 0.0
    thr = max(8.0, base_hp + 2.5 * noise)
    dev = np.abs(cg - base)
    dev_thr = max(20.0, 3.0 * noise)
    # Count DISCRETE fragments from the solid luma-deviation mask. The
    # high-pass response is used for energy, NOT for counting: a Gaussian blur
    # leaks each fragment's dark energy into the white gaps, so an |hp|>thr mask
    # bridges a readable dot-chain into a single blob (the V13.0 count=1 bug).
    comp_mask = (dev > dev_thr).astype(np.uint8)
    comp_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8))
    if int(comp_mask.sum()) == 0:
        # Faint residue with no solid deviation — fall back to the high-pass.
        comp_mask = (hp > thr).astype(np.uint8)
        comp_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN,
                                     np.ones((2, 2), np.uint8))
    if int(comp_mask.sum()) == 0:
        return out

    nlab, _, stats, cents = cv2.connectedComponentsWithStats(comp_mask)
    win_area = float(comp_mask.size)
    xs, areas, lumas = [], 0, []
    cxs = []
    count = 0
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 2:
            continue
        count += 1
        areas += a
        cx = float(cents[i, 0])
        cxs.append(cx)
        x0 = int(stats[i, cv2.CC_STAT_LEFT]); w0 = int(stats[i, cv2.CC_STAT_WIDTH])
        xs.extend([x0, x0 + w0])
        comp_region = cg[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] +
                         stats[i, cv2.CC_STAT_HEIGHT],
                         x0:x0 + w0]
        lumas.append(abs(float(comp_region.mean()) - base))

    if count == 0:
        return out
    span = (max(xs) - min(xs)) / max(1.0, float(cg.shape[1]))
    area_ratio = areas / win_area
    # Alignment: low vertical spread of component centroids ⇒ a line.
    cys = [float(cents[i, 1]) for i in range(1, nlab)
           if int(stats[i, cv2.CC_STAT_AREA]) >= 2]
    if len(cys) >= 2:
        y_spread = float(np.std(cys)) / max(1.0, float(cg.shape[0]))
        alignment = _clamp01(1.0 - y_spread * 4.0)
    else:
        alignment = 0.0
    mean_luma_delta = float(np.mean(lumas)) if lumas else 0.0

    # Composite dot-chain score.
    score = _clamp01(0.4 * min(1.0, count / 5.0) +
                     0.3 * min(1.0, span / 0.4) +
                     0.3 * alignment) * min(1.0, mean_luma_delta / max(noise, 3.0))

    out.update({
        "dot_chain_score": round(score, 4),
        "component_count": count,
        "component_area_ratio": round(area_ratio, 4),
        "horizontal_span_ratio": round(span, 4),
        "alignment_score": round(alignment, 4),
        "mean_component_luma_delta": round(mean_luma_delta, 3),
    })

    # Ring-relative guard — discount the surface's OWN texture. On a metallic /
    # textured / busy surface the surrounding ring is just as fragmented as the
    # bbox window, so the "fragments" are product texture, not watermark
    # residue. Only an EXCESS of structure over the surround is real residue
    # (the same excess-over-surface principle the residual/silhouette gates use).
    ring_sel = ring
    if int(ring_sel.sum()) > 0:
        ring_frag = float(((np.abs(rg - base) > dev_thr) & ring_sel).sum()) / \
            float(int(ring_sel.sum()))
    else:
        ring_frag = 0.0
    win_frag = float(comp_mask.mean())
    surface_excess = win_frag - ring_frag
    out["surface_excess"] = round(surface_excess, 4)
    has_real_excess = surface_excess > 0.012

    # Suggested reject rules from the patch plan, gated on real surface excess.
    reject = has_real_excess and (
        (count >= 3 and span > 0.15 and alignment > 0.60 and
         mean_luma_delta > noise * 1.4) or
        (count >= 4 and area_ratio > 0.012 and span > 0.12)
    )
    if not reject:
        out["dot_chain_score"] = round(min(out["dot_chain_score"],
                                           DOT_CHAIN_V2_MAX), 4)
    out["passed"] = not reject
    return out


# ---------------------------------------------------------------------------
# Section 8 — visible patch shape detector (rectangles / wedges / polygons /
# low-texture islands / hard luma boundaries).
# ---------------------------------------------------------------------------
def detect_visible_patch_shape_v13(original, cleaned, bbox):
    bx, by, bw, bh = bbox
    out = {
        "changed_area_ratio": 0.0,
        "straight_edge_score": 0.0,
        "polygonality": 0.0,
        "rectangularity": 0.0,
        "edge_box_score": 0.0,
        "luma_delta_mean": 0.0,
        "luma_delta_max": 0.0,
        "texture_drop": 0.0,
        "hard_boundary_score": 0.0,
        "visible_patch_score": 0.0,
        "passed": True,
    }
    if bw <= 1 or bh <= 1:
        return out
    changed, _ = _changed_region(original, cleaned, bbox)
    wm_area = float(max(1, bw * bh))
    n_changed = float((changed > 0).sum())
    out["changed_area_ratio"] = round(n_changed / wm_area, 4)
    if n_changed < 8:
        return out

    gray_u8 = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    gray_c = gray_u8.astype(np.float32)
    inner = gray_c[by:by + bh, bx:bx + bw]
    ring = gray_c[_ring_mask(cleaned.shape, bbox, 0.4)]
    if inner.size and ring.size:
        out["luma_delta_mean"] = round(abs(float(np.median(inner)) -
                                           float(np.median(ring))), 3)
        out["luma_delta_max"] = round(abs(float(inner.mean()) -
                                          float(ring.mean())), 3)
        # Laplacian needs an integer/uint8 source for a CV_64F destination.
        tex_inner = float(cv2.Laplacian(
            gray_u8[by:by + bh, bx:bx + bw], cv2.CV_64F).var())
        pad = max(12, max(bw, bh) // 3)
        H, W = gray_c.shape
        rr = gray_u8[max(0, by - pad):min(H, by + bh + pad),
                     max(0, bx - pad):min(W, bx + bw + pad)]
        tex_ring = float(cv2.Laplacian(rr, cv2.CV_64F).var())
        out["texture_drop"] = round(0.0 if tex_ring < 25.0
                                    else max(0.0, 1.0 - tex_inner /
                                             max(tex_ring, 1e-6)), 4)

    # Shape of the changed region's largest contour.
    cnts, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(big))
        peri = float(cv2.arcLength(big, True))
        x, y, w, h = cv2.boundingRect(big)
        rect_area = float(max(1, w * h))
        out["rectangularity"] = round(_clamp01(area / rect_area), 4)
        approx = cv2.approxPolyDP(big, 0.02 * peri, True)
        n_v = len(approx)
        # Few vertices + high fill ⇒ a clean geometric (man-made) shape.
        if n_v >= 3:
            out["polygonality"] = round(_clamp01((area / rect_area) *
                                        (1.0 - min(1.0, (n_v - 4) / 8.0))
                                        if n_v >= 4 else 0.6), 4)
        # Straightness: fraction of contour explained by few straight segments.
        out["straight_edge_score"] = round(
            _clamp01(1.0 - (n_v / max(6.0, peri / 8.0))), 4)

    # Hard luma boundary on the bbox perimeter (sharp man-made step).
    g = gray_c
    H, W = g.shape
    deltas = []
    if by > 0 and by + bh < H:
        deltas.append(np.abs(g[by, bx:bx + bw] - g[by - 1, bx:bx + bw]))
        deltas.append(np.abs(g[by + bh - 1, bx:bx + bw] - g[by + bh, bx:bx + bw]))
    if bx > 0 and bx + bw < W:
        deltas.append(np.abs(g[by:by + bh, bx] - g[by:by + bh, bx - 1]))
        deltas.append(np.abs(g[by:by + bh, bx + bw - 1] - g[by:by + bh, bx + bw]))
    if deltas:
        alld = np.concatenate(deltas)
        out["edge_box_score"] = round(float((alld > 10).mean()), 4)
        out["hard_boundary_score"] = round(float((alld > 18).mean()), 4)

    # A patch is VISIBLE only when it has a real boundary (edge or luma step):
    # an over-large but seamless change does not read as a patch.
    boundary = max(min(1.0, out["edge_box_score"] / 0.10),
                   min(1.0, out["luma_delta_mean"] / 6.0))
    visible = _clamp01(
        0.40 * out["straight_edge_score"] * boundary +
        0.25 * out["edge_box_score"] +
        0.20 * min(1.0, out["luma_delta_mean"] / 12.0) +
        0.15 * out["texture_drop"] * boundary)
    out["visible_patch_score"] = round(visible, 4)
    return out


def visible_patch_gate(metrics, *, repaired=True):
    limit = VISIBLE_PATCH_REPAIRED_MAX if repaired else VISIBLE_PATCH_COVERED_MAX
    if metrics["visible_patch_score"] > limit:
        return False, "visible_patch"
    # Hard geometric rejects (apply to both).
    if (metrics["straight_edge_score"] > STRAIGHT_EDGE_MAX and
            metrics["changed_area_ratio"] > 1.8 and
            metrics["luma_delta_mean"] > 5):
        return False, "straight_edged_patch"
    if (metrics["rectangularity"] > RECTANGULARITY_PATCH_MAX and
            metrics["edge_box_score"] > 0.14):
        return False, "rectangular_patch"
    if metrics["hard_boundary_score"] > HARD_BOUNDARY_MAX:
        return False, "hard_luma_boundary"
    return True, ""


# ---------------------------------------------------------------------------
# Section 10 — product silhouette / contour preservation gate.
# ---------------------------------------------------------------------------
def detect_product_silhouette_damage_v13(original, cleaned, product_mask, bbox,
                                         watermark_mask=None):
    bx, by, bw, bh = bbox
    out = {
        "product_overlap": 0.0,
        "product_contour_iou": 1.0,
        "product_edge_retention": 1.0,
        "contour_break_score": 0.0,
        "new_bright_blob_score": 0.0,
        "dark_surface_ratio": 0.0,
        "passed": True,
    }
    if bw <= 1 or bh <= 1:
        return out
    o = original[by:by + bh, bx:bx + bw]
    r = cleaned[by:by + bh, bx:bx + bw]
    if o.shape != r.shape or o.size == 0:
        return out
    og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    out["dark_surface_ratio"] = round(float((og < 80).mean()), 4)

    if product_mask is not None and product_mask.shape[:2] == original.shape[:2]:
        prod = product_mask[by:by + bh, bx:bx + bw] > 0
    else:
        prod = np.zeros((bh, bw), dtype=bool)
    out["product_overlap"] = round(float(prod.mean()), 4)

    # Exclude the watermark footprint — its OWN strokes are SUPPOSED to vanish,
    # so the edges that disappear there are a successful removal, not silhouette
    # damage. Without this exclusion every clean repair looks like a contour
    # break (this was the V13.0 over-rejection bug).
    if watermark_mask is not None and \
            watermark_mask.shape[:2] == original.shape[:2]:
        wmc = (watermark_mask[by:by + bh, bx:bx + bw] > 0).astype(np.uint8)
        wm = cv2.dilate(wmc, np.ones((3, 3), np.uint8), iterations=2) > 0
    else:
        wm = np.zeros((bh, bw), dtype=bool)
    prod_eff = prod & (~wm)

    ov = out["product_overlap"]
    # Nothing to protect on a (near-)plain background.
    if ov < 0.15 or int(prod_eff.sum()) < 20:
        out["passed"] = True
        return out

    eo = (cv2.Canny(og, 50, 150) > 0) & prod_eff
    er = (cv2.Canny(rg, 50, 150) > 0) & prod_eff
    eo_p = int(eo.sum())
    if eo_p >= 20:
        retained = int((eo & er).sum())
        out["product_edge_retention"] = round(retained / eo_p, 4)
        out["contour_break_score"] = round(1.0 - retained / eo_p, 4)
        union = int((eo | er).sum())
        out["product_contour_iou"] = round(retained / max(1, union), 4)

    # New bright/dark blob on the product surface (worst on dark surfaces).
    signed = rg.astype(np.float32) - og.astype(np.float32)
    blob = (np.abs(signed) > 25) & prod_eff
    if int(blob.sum()) > 0:
        bu = blob.astype(np.uint8) * 255
        nlab, _, stats, _ = cv2.connectedComponentsWithStats(bu)
        biggest = max((int(stats[i, cv2.CC_STAT_AREA])
                       for i in range(1, nlab)), default=0)
        out["new_bright_blob_score"] = round(
            biggest / max(1, int(prod_eff.sum())), 4)

    # Reject only on GROSS silhouette destruction or a clearly-visible blob on
    # a dark product surface — never on the minor texture shift a clean inpaint
    # leaves on the surrounding product.
    reject = (
        (out["new_bright_blob_score"] > NEW_BRIGHT_BLOB_MAX and
         out["dark_surface_ratio"] > DARK_SURFACE_RATIO_FOR_BLOB) or
        (out["product_edge_retention"] < 0.70) or
        (out["product_contour_iou"] < 0.75)
    )
    out["passed"] = not reject
    return out


# ---------------------------------------------------------------------------
# Section 11 — protected product text / label preservation. We flag the case
# where a cover would have erased real printed product text inside the bbox.
# ---------------------------------------------------------------------------
def detect_protected_product_text_v13(original, cleaned, bbox, watermark_mask=None):
    bx, by, bw, bh = bbox
    out = {"protected_components": 0, "protected_text_erased": False,
           "passed": True}
    if bw <= 1 or bh <= 1:
        return out
    o = original[by:by + bh, bx:bx + bw]
    if o.size == 0:
        return out
    og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    # High-contrast components on the product surface that are NOT the
    # watermark (the watermark mask, if given, is excluded).
    surface = float(np.median(og))
    high = (np.abs(og.astype(np.float32) - surface) > 45).astype(np.uint8)
    if watermark_mask is not None and watermark_mask.shape[:2] == original.shape[:2]:
        wm = (watermark_mask[by:by + bh, bx:bx + bw] > 0).astype(np.uint8)
        high = cv2.bitwise_and(high, 1 - wm)
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(high)
    comps = [i for i in range(1, nlab)
             if 6 <= int(stats[i, cv2.CC_STAT_AREA]) <= bw * bh * 0.2]
    out["protected_components"] = len(comps)
    if comps:
        # Were those high-contrast strokes flattened away by the repair?
        rg = cv2.cvtColor(cleaned[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2GRAY)
        high_after = (np.abs(rg.astype(np.float32) - surface) > 45)
        before = int(high.sum())
        after = int(high_after.sum())
        if before > 0 and after / before < 0.4:
            out["protected_text_erased"] = True
            out["passed"] = False
    return out


# ---------------------------------------------------------------------------
# FinalVisualVerdict + the composite gate (Section 2).
# ---------------------------------------------------------------------------
@dataclass
class FinalVisualVerdict:
    publish_ok: bool
    status: str                       # clean_repaired | clean_covered
    reject_reasons: list = field(default_factory=list)
    metrics_valid: bool = True
    residual_pass: bool = True
    dot_chain_pass: bool = True
    visible_patch_pass: bool = True
    rectangular_band_pass: bool = True
    polygon_patch_pass: bool = True
    product_damage_pass: bool = True
    silhouette_pass: bool = True
    protected_text_pass: bool = True
    scores: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "v13_publish_ok": self.publish_ok,
            "v13_publish_status": self.status,
            "v13_reject_reasons": self.reject_reasons,
            "v13_metrics_valid": self.metrics_valid,
            "v13_residual_pass": self.residual_pass,
            "v13_dot_chain_pass": self.dot_chain_pass,
            "v13_visible_patch_pass": self.visible_patch_pass,
            "v13_rectangular_band_pass": self.rectangular_band_pass,
            "v13_polygon_patch_pass": self.polygon_patch_pass,
            "v13_product_damage_pass": self.product_damage_pass,
            "v13_silhouette_pass": self.silhouette_pass,
            "v13_protected_text_pass": self.protected_text_pass,
            "v13_scores": self.scores,
        }


def _band_metrics(cleaned, bbox):
    if _pr_band is not None:
        return _pr_band(cleaned, bbox)
    # Minimal fallback band detector.
    m = detect_visible_patch_shape_v13(cleaned, cleaned, bbox)
    return {"visible_band_score": m["visible_patch_score"],
            "band_luma_delta": m["luma_delta_mean"],
            "band_edge_box_score": m["edge_box_score"],
            "band_rectangularity": m["rectangularity"]}


def final_visual_publish_gate_v13(original, cleaned, bbox, product_mask,
                                  qa_info=None, *, repaired=True,
                                  watermark_mask=None):
    """The single V13 visual gate. Applies to BOTH clean_repaired and
    clean_covered. ``repaired`` selects the stricter (repaired) thresholds.

    Returns a :class:`FinalVisualVerdict`. ``publish_ok`` is True only when
    every sub-gate passes simultaneously.
    """
    qa_info = qa_info or {}
    reasons = []

    metrics_valid = bool(qa_info.get("metrics_valid", True))
    if not metrics_valid:
        reasons.append("metrics_invalid")

    residual_pass = bool(qa_info.get("residual_pass", True))
    if not residual_pass:
        reasons.append("residual")

    dc = detect_residual_dot_chain_v2(original, cleaned, bbox, watermark_mask)
    dc_limit = DOT_CHAIN_V2_MAX if repaired else DOT_CHAIN_V2_COVERED_MAX
    dot_chain_pass = dc["passed"] and dc["dot_chain_score"] <= dc_limit
    if not dot_chain_pass:
        reasons.append("dot_chain")

    band = _band_metrics(cleaned, bbox)
    band_limit = _PR_BAND_REP_MAX if repaired else _PR_BAND_COV_MAX
    luma_limit = _PR_BAND_LUMA_REP_MAX if repaired else _PR_BAND_LUMA_COV_MAX
    rectangular_band_pass = (band.get("visible_band_score", 0.0) <= band_limit and
                             band.get("band_luma_delta", 0.0) <= luma_limit)
    if not rectangular_band_pass:
        reasons.append("visible_band")

    patch = detect_visible_patch_shape_v13(original, cleaned, bbox)
    visible_patch_pass, patch_reason = visible_patch_gate(patch, repaired=repaired)
    polygon_patch_pass = not (patch["polygonality"] > POLYGONALITY_MAX and
                              patch["changed_area_ratio"] > 0.5)
    if not visible_patch_pass:
        reasons.append(patch_reason or "visible_patch")
    if not polygon_patch_pass:
        reasons.append("polygon_patch")

    product_damage_pass = bool(qa_info.get("product_gate_pass", True))
    if not product_damage_pass:
        reasons.append("product_damage")

    sil = detect_product_silhouette_damage_v13(original, cleaned,
                                               product_mask, bbox,
                                               watermark_mask=watermark_mask)
    silhouette_pass = sil["passed"]
    if not silhouette_pass:
        reasons.append("silhouette_damage")

    ptext = detect_protected_product_text_v13(original, cleaned, bbox,
                                              watermark_mask)
    protected_text_pass = ptext["passed"]
    if not protected_text_pass:
        reasons.append("protected_text_erased")

    publish_ok = len(reasons) == 0
    return FinalVisualVerdict(
        publish_ok=publish_ok,
        status="clean_repaired" if (publish_ok and repaired) else "clean_covered",
        reject_reasons=reasons,
        metrics_valid=metrics_valid,
        residual_pass=residual_pass,
        dot_chain_pass=dot_chain_pass,
        visible_patch_pass=visible_patch_pass,
        rectangular_band_pass=rectangular_band_pass,
        polygon_patch_pass=polygon_patch_pass,
        product_damage_pass=product_damage_pass,
        silhouette_pass=silhouette_pass,
        protected_text_pass=protected_text_pass,
        scores={
            "dot_chain": dc["dot_chain_score"],
            "visible_band": band.get("visible_band_score", 0.0),
            "visible_patch_shape": patch["visible_patch_score"],
            "straight_edge": patch["straight_edge_score"],
            "polygonality": patch["polygonality"],
            "rectangularity": patch["rectangularity"],
            "hard_boundary": patch["hard_boundary_score"],
            "contour_break": sil["contour_break_score"],
            "silhouette_iou": sil["product_contour_iou"],
            "new_bright_blob": sil["new_bright_blob_score"],
            "product_overlap": sil["product_overlap"],
            "changed_area_ratio": patch["changed_area_ratio"],
        },
    )


# Honesty-counter keys (all must be 0 in an acceptable release).
HONESTY_COUNTERS = [
    "clean_repaired_with_dot_chain",
    "clean_repaired_with_visible_patch",
    "clean_repaired_with_product_damage",
    "clean_repaired_with_silhouette_damage",
    "clean_covered_with_dot_chain",
    "clean_covered_with_visible_patch",
    "clean_covered_with_product_damage",
    "clean_covered_with_silhouette_damage",
    "final_publish_failures",
]


def update_honesty_counters(counters: dict, status: str,
                            verdict: FinalVisualVerdict) -> None:
    """Tally any bad artifact that survived into a published output."""
    prefix = "clean_repaired" if status == "clean_repaired" else "clean_covered"
    if not verdict.dot_chain_pass:
        counters[f"{prefix}_with_dot_chain"] = \
            counters.get(f"{prefix}_with_dot_chain", 0) + 1
    if not (verdict.visible_patch_pass and verdict.rectangular_band_pass and
            verdict.polygon_patch_pass):
        counters[f"{prefix}_with_visible_patch"] = \
            counters.get(f"{prefix}_with_visible_patch", 0) + 1
    if not verdict.product_damage_pass:
        counters[f"{prefix}_with_product_damage"] = \
            counters.get(f"{prefix}_with_product_damage", 0) + 1
    if not verdict.silhouette_pass:
        counters[f"{prefix}_with_silhouette_damage"] = \
            counters.get(f"{prefix}_with_silhouette_damage", 0) + 1
    if not verdict.publish_ok:
        counters["final_publish_failures"] = \
            counters.get("final_publish_failures", 0) + 1
