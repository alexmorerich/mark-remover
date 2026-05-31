#!/usr/bin/env python3
"""V14 — Short Patch: better candidates, same V13 honesty.

The V14 principle (see the V14 Short Patch Plan):

    Keep V13 strict. Make the candidates better.

V13 added the strict ``final_visual_publish_gate_v13`` that refuses to publish a
visually-bad result as ``clean_repaired``. It is honest but the candidate /
cover generators are not strong enough, so too many near-miss repairs are
demoted into ``clean_covered`` (especially through the blunt
``final_adaptive_cover``).

V14 does **not** weaken that gate. It adds, all *before* the unchanged V13 gate
re-runs on the result:

  1. adaptive soft-metric context (Section 3.2) — context-aware soft limits used
     only to decide whether a failed repair is a rescuable *near-miss*;
  2. ``near_miss_rescue`` (Section 3.3) — component cleanup, 1px feather, gamma /
     Lab colour match and local seam smoothing on a soft-fail repaired candidate,
     then a re-run of the V13 *repaired* gate. Accepted only if it fully passes;
  3. ``segmented_micro_cover`` (Section 3.4) — a fragment-based beam-search cover
     that replaces the full-bbox cover and is *banned* from a full-bbox fill on
     product / flex-cable / long-line / glass / metallic / protected-text zones;
  4. cover-side honesty counters (Section 7) — all must be 0 in a release.

Hard V13 metrics (residual, dot-chain, protected text, silhouette, product
damage, geometric patches) are never relaxed: a repair that trips any of them is
a *hard* fail and is sent straight to cover, never rescued. Only the *soft*
visual-fidelity reasons (a faintly-visible patch / band that a gentle colour and
seam correction can remove) are eligible for rescue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

import v13_gates

# Reuse production helpers where available; fall back to local equivalents so
# this module is importable and unit-testable on its own.
try:  # pragma: no cover - exercised indirectly
    from progressive_repair import (
        cleanup_residual_components_with_ring_fill as _pr_cleanup_components,
        detect_residual_watermark as _pr_detect_residual,
    )
    _HAVE_PR = True
except Exception:  # pragma: no cover
    _pr_cleanup_components = None
    _pr_detect_residual = None
    _HAVE_PR = False


# ---------------------------------------------------------------------------
# Failure classification (Section 3.3).
#
# A repaired candidate that fails the V13 gate is classified by *why* it failed.
# Only soft visual-fidelity reasons are rescuable; any hard reason (readable
# residue, dot-chain, protected text, silhouette / product damage, a clean
# geometric patch) goes straight to cover.
# ---------------------------------------------------------------------------
HARD_FAIL_REASONS = {
    "metrics_invalid",
    "residual",
    "dot_chain",
    "product_damage",
    "silhouette_damage",
    "protected_text_erased",
    "straight_edged_patch",
    "rectangular_patch",
    "hard_luma_boundary",
    "polygon_patch",
}
SOFT_FAIL_REASONS = {
    "visible_patch",
    "visible_band",
}


def classify_failure(verdict) -> str:
    """Return ``"soft_fail"`` (rescue eligible), ``"hard_fail"`` (cover only) or
    ``"clean"`` (the verdict already passed)."""
    reasons = list(getattr(verdict, "reject_reasons", []) or [])
    if not reasons:
        return "clean"
    if any(r in HARD_FAIL_REASONS for r in reasons):
        return "hard_fail"
    if all(r in SOFT_FAIL_REASONS for r in reasons):
        return "soft_fail"
    # Unknown reason ⇒ be conservative and treat as hard (no rescue).
    return "hard_fail"


# ---------------------------------------------------------------------------
# Section 3.2 — adaptive soft-metric context.
# ---------------------------------------------------------------------------
@dataclass
class SoftContext:
    plain_white: bool = False
    near_white: bool = False
    saturated_flat_product: bool = False
    thin_flex_cable: bool = False
    complex_product_detail: bool = False
    dark_surface: bool = False
    glass_or_gradient: bool = False
    metallic_or_reflective: bool = False
    product_overlap: float = 0.0
    edge_density: float = 0.0
    long_line_score: float = 0.0
    # Adaptive soft limits (only used to gate near-miss rescue, never to relax
    # a hard V13 metric).
    ssim_weight: float = 0.7
    color_delta_limit: float = 12.0
    seam_mode: str = "normal"
    boundary_mode: str = "normal"
    edge_retention_limit: float = 0.90


def soft_qa_context(image, bbox, product_mask=None, base_color_limit=12.0,
                    roi_class: str = "") -> SoftContext:
    """Build a context-aware soft-metric profile for the ROI (Section 3.2).

    The classification is multi-signal (a single ROI may be mixed) and only
    drives whether a *soft* fail is rescuable — the hard V13 metrics are
    unaffected.
    """
    bx, by, bw, bh = bbox
    ctx = SoftContext(product_overlap=0.0)
    if bw <= 1 or bh <= 1:
        return ctx
    inner = image[by:by + bh, bx:bx + bw]
    if inner.size == 0:
        return ctx
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    white_ratio = float((g >= 235).mean())
    dark_ratio = float((g < 80).mean())
    sat = float(hsv[..., 1].mean())
    edges = cv2.Canny(g, 50, 150)
    edge_density = float((edges > 0).mean())
    grad = cv2.Sobel(g, cv2.CV_32F, 1, 0).std() + cv2.Sobel(g, cv2.CV_32F, 0, 1).std()

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(10, bw // 6),
                            minLineLength=max(10, bw // 3), maxLineGap=4)
    long_line = 0.0
    if lines is not None and len(lines) > 0:
        longest = max(float(np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]))
                      for l in lines)
        long_line = float(min(1.0, longest / max(1.0, float(max(bw, bh)))))

    overlap = 0.0
    if product_mask is not None and product_mask.shape[:2] == image.shape[:2]:
        pr = product_mask[by:by + bh, bx:bx + bw]
        if pr.size:
            overlap = float((pr > 0).mean())
    else:
        overlap = float((g < 235).mean())

    ctx.product_overlap = round(overlap, 4)
    ctx.edge_density = round(edge_density, 4)
    ctx.long_line_score = round(long_line, 4)

    ctx.plain_white = white_ratio > 0.92 and edge_density < 0.02
    ctx.near_white = (not ctx.plain_white) and white_ratio > 0.70 \
        and edge_density < 0.05
    ctx.dark_surface = dark_ratio > 0.30
    ctx.thin_flex_cable = long_line > 0.35 and dark_ratio > 0.20
    ctx.complex_product_detail = edge_density > 0.08
    ctx.glass_or_gradient = (not ctx.plain_white) and grad > 30.0 \
        and edge_density < 0.06
    ctx.metallic_or_reflective = sat < 60 and float(g.std()) > 35 \
        and edge_density < 0.08
    ctx.saturated_flat_product = sat > 90 and edge_density < 0.04

    # Suggested adaptive logic from the plan (Section 3.2).
    if ctx.plain_white or ctx.near_white:
        ctx.ssim_weight = 0.2
        ctx.color_delta_limit = 8.0
        ctx.seam_mode = "strict"
        ctx.boundary_mode = "strict"
    elif ctx.saturated_flat_product:
        ctx.color_delta_limit = min(16.5, base_color_limit * 1.35)
        ctx.ssim_weight = 0.7
    elif ctx.thin_flex_cable or ctx.complex_product_detail:
        ctx.color_delta_limit = 10.0
        ctx.edge_retention_limit = 0.90
        ctx.boundary_mode = "strict"
    return ctx


# ---------------------------------------------------------------------------
# Low-level primitives (self-contained; no RepairContext dependency).
# ---------------------------------------------------------------------------
def _ring_median_color(image, bbox, pad_frac=0.5):
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    pad = max(8, int(round(max(bw, bh) * pad_frac)))
    y1, y2 = max(0, by - pad), min(H, by + bh + pad)
    x1, x2 = max(0, bx - pad), min(W, bx + bw + pad)
    win = image[y1:y2, x1:x2]
    if win.size == 0:
        return np.array([200, 200, 200], dtype=np.float32)
    m = np.ones(win.shape[:2], dtype=bool)
    iy1, iy2 = by - y1, by - y1 + bh
    ix1, ix2 = bx - x1, bx - x1 + bw
    m[max(0, iy1):max(0, iy2), max(0, ix1):max(0, ix2)] = False
    ring = win[m]
    if ring.size == 0:
        ring = win.reshape(-1, win.shape[-1])
    return np.median(ring.reshape(-1, image.shape[-1]), axis=0).astype(np.float32)


def _changed_mask(original, cleaned, bbox, pad_frac=0.5, thr=6.0):
    bx, by, bw, bh = bbox
    H, W = original.shape[:2]
    pad = max(8, int(round(max(bw, bh) * pad_frac)))
    y1, y2 = max(0, by - pad), min(H, by + bh + pad)
    x1, x2 = max(0, bx - pad), min(W, bx + bw + pad)
    og = cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    cg = cv2.cvtColor(cleaned[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = np.abs(og - cg) > thr
    return m


def boundary_jump_estimate(original, cleaned, bbox) -> float:
    """Max absolute luma step (0-255) across the bbox perimeter of ``cleaned``.

    Drives the ``clean_covered_with_boundary_jump_gt_50`` honesty counter and
    the seam term of the micro-cover beam score.
    """
    bx, by, bw, bh = bbox
    g = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    if bw <= 1 or bh <= 1:
        return 0.0
    deltas = []
    if by > 0 and by + bh < H:
        deltas.append(np.abs(g[by, bx:bx + bw] - g[by - 1, bx:bx + bw]))
        deltas.append(np.abs(g[by + bh - 1, bx:bx + bw] - g[by + bh, bx:bx + bw]))
    if bx > 0 and bx + bw < W:
        deltas.append(np.abs(g[by:by + bh, bx] - g[by:by + bh, bx - 1]))
        deltas.append(np.abs(g[by:by + bh, bx + bw - 1] - g[by:by + bh, bx + bw]))
    if not deltas:
        return 0.0
    alld = np.concatenate(deltas)
    return float(np.percentile(alld, 95))


# ---------------------------------------------------------------------------
# Section 3.3 — near-miss rescue.
# ---------------------------------------------------------------------------
def _component_residual_cleanup(original, cleaned, bbox, watermark_mask):
    """Paint just the isolated residual fragments inside the watermark
    footprint with the ring background colour (NOT a full-bbox re-inpaint)."""
    bx, by, bw, bh = bbox
    H, W = cleaned.shape[:2]
    src = watermark_mask
    if src is None or src.shape[:2] != (H, W):
        return cleaned
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
    footprint = cv2.dilate((src > 0).astype(np.uint8), kd, iterations=1) > 0
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0))
    rpad = max(12, max(bw, bh) // 3)
    rc = gray[max(0, by - rpad):min(H, by + bh + rpad),
              max(0, bx - rpad):min(W, bx + bw + rpad)]
    thr = max(8.0, float(rc.mean()) + 2.0 * float(rc.std())) \
        if rc.size else 8.0
    comp = ((hp > thr) & footprint).astype(np.uint8) * 255
    if int(comp.sum()) == 0:
        return cleaned
    ring = _ring_median_color(cleaned, bbox).astype(np.uint8)
    if _pr_cleanup_components is not None:
        return _pr_cleanup_components(cleaned, bbox, comp, ring)
    out = cleaned.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sel = cv2.dilate(comp, k, iterations=1) > 0
    out[sel] = ring
    blurred = cv2.GaussianBlur(out, (3, 3), 0)
    edge = (cv2.dilate(sel.astype(np.uint8), k, iterations=1) > 0) & (~sel)
    out[edge] = blurred[edge]
    return out


def _gamma_lab_match(original, cleaned, bbox, context, max_shift=6.0,
                     damp=0.6):
    """Nudge the *changed* pixels' colour toward the surrounding ring in Lab,
    damped and hard-clipped to a few units so we correct a mild colour / luma
    drift (what makes a clean inpaint read as a faint patch) without repainting
    real product detail."""
    changed = _changed_mask(original, cleaned, bbox)
    if not changed.any():
        return cleaned
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring_bgr = _ring_median_color(cleaned, bbox).reshape(1, 1, 3).astype(np.uint8)
    ring_lab = cv2.cvtColor(ring_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    cur = lab[changed].mean(axis=0)
    shift = ring_lab - cur
    shift = np.clip(shift * damp, -max_shift, max_shift)
    # On saturated flat product the chroma channels may drift a touch more.
    if context is not None and context.saturated_flat_product:
        shift[1:] = np.clip(shift[1:] * 1.0, -max_shift * 1.4, max_shift * 1.4)
    out = lab.copy()
    alpha = cv2.GaussianBlur(changed.astype(np.float32), (0, 0), sigmaX=2.0)
    alpha = alpha[..., None]
    out = out + shift.reshape(1, 1, 3) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _seam_smooth(original, cleaned, bbox, band_px=3):
    """Feather a thin band along the bbox perimeter so any residual hard luma
    step is broken up (kills the visible-band / hard-boundary near-miss)."""
    bx, by, bw, bh = bbox
    H, W = cleaned.shape[:2]
    perim = np.zeros((H, W), np.uint8)
    cv2.rectangle(perim, (bx, by), (bx + bw - 1, by + bh - 1), 255, band_px)
    if int(perim.sum()) == 0:
        return cleaned
    alpha = cv2.GaussianBlur(perim.astype(np.float32) / 255.0, (0, 0),
                             sigmaX=max(1.0, band_px / 2.0))[..., None]
    blurred = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=1.2)
    out = (blurred.astype(np.float32) * alpha +
           cleaned.astype(np.float32) * (1.0 - alpha))
    return np.clip(out, 0, 255).astype(np.uint8)


def near_miss_rescue(original, cleaned, bbox, product_mask=None,
                     watermark_mask=None, context: Optional[SoftContext] = None):
    """Apply the soft-fail rescue chain (Section 3.3) and return the rescued
    image. Operations are gentle and bounded; they never repaint product
    structure. The caller must re-run the V13 repaired gate and only accept the
    result if it fully passes.
    """
    out = cleaned
    out = _component_residual_cleanup(original, out, bbox, watermark_mask)
    out = _gamma_lab_match(original, out, bbox, context)
    out = _seam_smooth(original, out, bbox)
    return out


# ---------------------------------------------------------------------------
# Section 3.4 — segmented micro-cover beam search.
# ---------------------------------------------------------------------------
def full_bbox_cover_banned(image, bbox, product_mask=None, watermark_mask=None,
                           context: Optional[SoftContext] = None) -> tuple:
    """A full-bbox opaque cover is banned on structured / product-overlap
    regions (Section 3.4). Returns ``(banned: bool, reason: str)``."""
    ov = v13_gates.estimate_product_overlap_v13(image, bbox, product_mask)
    if ov["product_overlap"] > 0.10:
        return True, "product_overlap"
    if ov["edge_density_in_bbox"] > 0.04:
        return True, "edge_density"
    if ov["long_line_score"] > 0.25:
        return True, "long_lines"
    if context is not None:
        if context.thin_flex_cable:
            return True, "thin_flex_cable"
        if context.metallic_or_reflective:
            return True, "metallic"
        if context.glass_or_gradient:
            return True, "glass_gradient"
    ptext = v13_gates.detect_protected_product_text_v13(
        image, image, bbox, watermark_mask)
    if ptext.get("protected_components", 0) >= 2:
        return True, "protected_text_nearby"
    return False, ""


def _full_bbox_fill(image, bbox, watermark_mask=None):
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    fp = np.zeros((H, W), np.uint8)
    pad_y = 3
    pad_x = max(8, int(bw * 0.12))
    fy1, fy2 = max(0, by - pad_y), min(H, by + bh + pad_y)
    fx1, fx2 = max(0, bx - pad_x), min(W, bx + bw + pad_x)
    fp[fy1:fy2, fx1:fx2] = 255
    result = cv2.inpaint(image, fp, 4, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(fp.astype(np.float32) / 255.0, (0, 0), sigmaX=1.5)
    a3 = np.stack([alpha] * 3, axis=-1)
    return (result.astype(np.float32) * a3 +
            image.astype(np.float32) * (1 - a3)).astype(np.uint8)


def _stroke_band_fill(image, bbox, watermark_mask, radius=3):
    """Inpaint only the watermark stroke band (its full horizontal span, tight
    height) — a micro-fragment cover that preserves product structure."""
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    if watermark_mask is None or watermark_mask.shape[:2] != (H, W) \
            or not np.any(watermark_mask > 0):
        return None
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
    wide = cv2.dilate((watermark_mask > 0).astype(np.uint8) * 255, kd, iterations=1)
    band = np.zeros((H, W), np.uint8)
    ys, xs = np.where(wide > 0)
    if ys.size:
        band[int(ys.min()):int(ys.max()) + 1,
             int(xs.min()):int(xs.max()) + 1] = 255
    else:
        band = wide
    return cv2.inpaint(image, band, radius, cv2.INPAINT_TELEA)


def _stroke_only_fill(image, bbox, watermark_mask, dilate_px=2, radius=2):
    """Inpaint ONLY the watermark stroke pixels (lightly dilated) — the
    lightest-touch micro cover, which preserves the most product structure /
    printed text. Best on flex cables, glass and busy product surfaces where a
    band or segmented fill would flatten real detail."""
    H, W = image.shape[:2]
    if watermark_mask is None or watermark_mask.shape[:2] != (H, W) \
            or not np.any(watermark_mask > 0):
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (dilate_px * 2 + 1, dilate_px * 2 + 1))
    m = cv2.dilate((watermark_mask > 0).astype(np.uint8) * 255, k, iterations=1)
    return cv2.inpaint(image, m, radius, cv2.INPAINT_TELEA)


def _segmented_fragment_fill(image, bbox, watermark_mask, product_mask):
    """Background fragments → ring-median fill; product fragments → stroke
    inpaint only. Never paints a solid block over product pixels."""
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    out = image.copy()
    if product_mask is None or product_mask.shape[:2] != (H, W):
        return _stroke_band_fill(image, bbox, watermark_mask)
    prod_roi = product_mask[by:by + bh, bx:bx + bw] > 0
    bg_mask = ~prod_roi
    if bg_mask.any():
        ring = _ring_median_color(image, bbox)
        fill = np.full((bh, bw, 3), ring, dtype=np.float32)
        noise = np.random.normal(0, 1.0, fill.shape)
        fill = np.clip(fill + noise, 0, 255)
        a = cv2.GaussianBlur(bg_mask.astype(np.float32), (0, 0), sigmaX=1.5)
        a3 = np.stack([a] * 3, axis=-1)
        roi = out[by:by + bh, bx:bx + bw].astype(np.float32)
        roi = fill * a3 + roi * (1 - a3)
        out[by:by + bh, bx:bx + bw] = np.clip(roi, 0, 255).astype(np.uint8)
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        fg_full = np.zeros((H, W), np.uint8)
        fg_full[by:by + bh, bx:bx + bw] = (prod_roi.astype(np.uint8) * 255)
        combined = cv2.bitwise_and((watermark_mask > 0).astype(np.uint8) * 255,
                                   fg_full)
        if np.any(combined > 0):
            out = cv2.inpaint(out, combined, 2, cv2.INPAINT_TELEA)
    return out


def _micro_cover_score(original, cover, bbox, product_mask, watermark_mask):
    """Integrated beam score (Section 3.4). Lower is better.

    The boundary-jump term is intentionally *uncapped* so a catastrophic cover
    (a hard luma block, jump ≫ 50) is never preferred over a soft one, and a
    protected-text-loss term is added so the beam ranks in agreement with the
    V13 gate (which rejects on protected-text erasure / silhouette damage)."""
    patch = v13_gates.detect_visible_patch_shape_v13(original, cover, bbox)
    sil = v13_gates.detect_product_silhouette_damage_v13(
        original, cover, product_mask, bbox, watermark_mask=watermark_mask)
    dc = v13_gates.detect_residual_dot_chain_v2(original, cover, bbox,
                                                watermark_mask)
    ptext = v13_gates.detect_protected_product_text_v13(
        original, cover, bbox, watermark_mask)
    bj = boundary_jump_estimate(original, cover, bbox)
    color = patch["luma_delta_mean"]
    return (4.0 * dc["dot_chain_score"] +
            4.0 * patch["visible_patch_score"] +
            3.0 * (bj / 50.0) +
            3.0 * sil["contour_break_score"] +
            2.5 * (1.0 - sil["product_edge_retention"]) +
            2.0 * min(1.0, color / 20.0) +
            2.0 * patch["texture_drop"] +
            1.5 * patch["rectangularity"] +
            5.0 * (1.0 if ptext.get("protected_text_erased") else 0.0))


def cover_hides_watermark(original, cover, bbox, watermark_mask) -> bool:
    """Mask-aware check that the watermark is no longer readable in ``cover``
    (mirrors the production cover gate). Used to feed a *fresh* residual verdict
    to the V13 gate for each micro-cover candidate, instead of trusting the
    stale qa_info of the candidate the cover replaced."""
    bx, by, bw, bh = bbox
    H, W = original.shape[:2]
    o = original[by:by + bh, bx:bx + bw]
    c = cover[by:by + bh, bx:bx + bw]
    if o.size == 0 or c.shape != o.shape:
        return True
    rpad = max(12, max(bw, bh) // 3)
    ring = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)[
        max(0, by - rpad):min(H, by + bh + rpad),
        max(0, bx - rpad):min(W, bx + bw + rpad)]
    mask_roi = None
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
        mask_roi = cv2.dilate((watermark_mask > 0).astype(np.uint8) * 255, kd,
                              iterations=1)[by:by + bh, bx:bx + bw]
    if _pr_detect_residual is not None:
        try:
            return bool(_pr_detect_residual(o, c, ring_gray=ring,
                                            mask_roi=mask_roi).passed)
        except Exception:
            return True
    # Fallback: no readable dot-chain residue ⇒ hidden.
    dc = v13_gates.detect_residual_dot_chain_v2(original, cover, bbox,
                                                watermark_mask)
    return dc["passed"]


@dataclass
class MicroCoverResult:
    image: np.ndarray
    method: str
    used_full_bbox: bool
    score: float
    candidates: list = field(default_factory=list)
    ranked: list = field(default_factory=list)  # [(name, image, used_full_bbox, score)]


def segmented_micro_cover(image, bbox, watermark_mask=None, product_mask=None,
                          context: Optional[SoftContext] = None,
                          base_cover=None) -> MicroCoverResult:
    """Fragment-based beam-search cover (Section 3.4). Generates several micro
    covers, bans the full-bbox fill on structured / product zones, scores the
    integrated result and returns the best. ``base_cover`` (the legacy
    final_adaptive_cover output, if any) competes too so V14 never loses to V13.
    """
    banned, ban_reason = full_bbox_cover_banned(image, bbox, product_mask,
                                                 watermark_mask, context)
    cands = []  # (name, image, used_full_bbox)

    seg = _segmented_fragment_fill(image, bbox, watermark_mask, product_mask)
    if seg is not None:
        cands.append(("segmented_micro_cover", seg, False))
    so = _stroke_only_fill(image, bbox, watermark_mask)
    if so is not None:
        cands.append(("stroke_only_micro_cover", so, False))
    for r in (2, 3):
        sb = _stroke_band_fill(image, bbox, watermark_mask, radius=r)
        if sb is not None:
            cands.append((f"stroke_band_micro_cover_r{r}", sb, False))
    if not banned:
        cands.append(("full_bbox_micro_cover",
                      _full_bbox_fill(image, bbox, watermark_mask), True))
    if base_cover is not None:
        # The legacy cover may itself be a full-bbox fill; mark it so the
        # honesty counter can detect a full-bbox-on-product publish.
        cands.append(("final_adaptive_cover", base_cover,
                      banned is False))
    if not cands:
        return MicroCoverResult(image=image, method="passthrough",
                                used_full_bbox=False, score=float("inf"))

    scored = []
    for name, img, full in cands:
        try:
            s = _micro_cover_score(image, img, bbox, product_mask,
                                   watermark_mask)
        except Exception:
            s = float("inf")
        scored.append((s, name, img, full))
    scored.sort(key=lambda c: c[0])
    best_s, best_name, best_img, best_full = scored[0]
    return MicroCoverResult(
        image=best_img, method=best_name, used_full_bbox=best_full,
        score=round(best_s, 4),
        candidates=[{"method": n, "score": round(s, 4), "full_bbox": f}
                    for s, n, _, f in scored],
        ranked=[(n, im, f, round(s, 4)) for s, n, im, f in scored])


# ---------------------------------------------------------------------------
# Section 7 — cover-side honesty counters. All must be 0 in a release.
# ---------------------------------------------------------------------------
COVER_HONESTY_COUNTERS = [
    "clean_covered_with_visible_patch",
    "clean_covered_with_product_damage",
    "clean_covered_with_silhouette_damage",
    "clean_covered_with_boundary_jump_gt_50",
    "clean_covered_with_full_bbox_on_product",
    "clean_covered_with_protected_text_loss",
]


def update_cover_honesty_counters(counters: dict, verdict, *,
                                   used_full_bbox_on_product: bool = False,
                                   boundary_jump: float = 0.0) -> None:
    if not (verdict.visible_patch_pass and verdict.rectangular_band_pass and
            verdict.polygon_patch_pass):
        counters["clean_covered_with_visible_patch"] = \
            counters.get("clean_covered_with_visible_patch", 0) + 1
    if not verdict.product_damage_pass:
        counters["clean_covered_with_product_damage"] = \
            counters.get("clean_covered_with_product_damage", 0) + 1
    if not verdict.silhouette_pass:
        counters["clean_covered_with_silhouette_damage"] = \
            counters.get("clean_covered_with_silhouette_damage", 0) + 1
    if boundary_jump > 50.0:
        counters["clean_covered_with_boundary_jump_gt_50"] = \
            counters.get("clean_covered_with_boundary_jump_gt_50", 0) + 1
    if used_full_bbox_on_product:
        counters["clean_covered_with_full_bbox_on_product"] = \
            counters.get("clean_covered_with_full_bbox_on_product", 0) + 1
    if not verdict.protected_text_pass:
        counters["clean_covered_with_protected_text_loss"] = \
            counters.get("clean_covered_with_protected_text_loss", 0) + 1


# ---------------------------------------------------------------------------
# Orchestrator — wired into mark_remover at the post-candidate decision point.
# ---------------------------------------------------------------------------
@dataclass
class V14Outcome:
    image: np.ndarray
    status: str                       # clean_repaired | clean_covered
    verdict: object                   # FinalVisualVerdict (re-evaluated)
    telemetry: dict = field(default_factory=dict)


def v14_finalize(original, candidate_image, bbox, product_mask, watermark_mask,
                 gate_fn: Callable, *, is_cover: bool, v13_verdict,
                 roi_class: str = "", base_cover_image=None,
                 pre_cover_repaired_image=None) -> V14Outcome:
    """Apply the V14 chain after the V13 gate has run once.

    ``gate_fn(image, repaired: bool) -> FinalVisualVerdict`` re-runs the
    *unchanged* V13 visual gate on a new image. V14 only ever asks the gate to
    bless a result — it never bypasses it.

    Resolution order:
      * a passing ``clean_repaired`` is returned untouched;
      * a soft-fail repaired candidate is rescued; if the rescue passes the V13
        repaired gate it stays ``clean_repaired``;
      * otherwise a segmented micro-cover is produced, re-checked under the V13
        covered gate, and published as an honest ``clean_covered``.
    """
    ST_REP, ST_COV = "clean_repaired", "clean_covered"
    tele = {
        "v14_failure_class": None,
        "v14_near_miss_rescued": False,
        "v14_cover_method": None,
        "v14_used_full_bbox_on_product": False,
        "v14_boundary_jump": 0.0,
        "v14_micro_cover_candidates": [],
        "v14_full_bbox_banned": None,
    }
    context = soft_qa_context(original, bbox, product_mask, roi_class=roi_class)

    # 1) Already a clean repaired publish — nothing to do.
    if (not is_cover) and v13_verdict.publish_ok:
        return V14Outcome(candidate_image, ST_REP, v13_verdict, tele)

    # 2) Soft-fail repaired candidate → attempt near-miss rescue.
    rescue_source = None
    rescue_verdict = None
    if (not is_cover) and (not v13_verdict.publish_ok):
        rescue_source = candidate_image
        rescue_verdict = v13_verdict
    elif is_cover and pre_cover_repaired_image is not None:
        # The repair loop already fell to cover, but it preserved its best
        # failed repaired candidate — that, not the cover, is the rescue target.
        rescue_source = pre_cover_repaired_image
        rescue_verdict = gate_fn(pre_cover_repaired_image, True)

    if rescue_source is not None and rescue_verdict is not None:
        fc = classify_failure(rescue_verdict)
        tele["v14_failure_class"] = fc
        if fc == "soft_fail":
            rescued = near_miss_rescue(original, rescue_source, bbox,
                                       product_mask, watermark_mask, context)
            rv = gate_fn(rescued, True)
            if rv.publish_ok:
                tele["v14_near_miss_rescued"] = True
                return V14Outcome(rescued, ST_REP, rv, tele)

    # 3) Cover needed → segmented micro-cover beam search. Evaluate candidates
    # in beam-score order against the V13 *covered* gate (with a fresh,
    # per-candidate watermark-hiding verdict) and publish the first that passes;
    # only if none passes do we fall back to the lowest-score cover. This is how
    # V14 turns "honest but failing" covers into honest *passing* covers.
    banned, ban_reason = full_bbox_cover_banned(original, bbox, product_mask,
                                                 watermark_mask, context)
    tele["v14_full_bbox_banned"] = ban_reason if banned else ""
    mc = segmented_micro_cover(original, bbox, watermark_mask, product_mask,
                               context, base_cover=base_cover_image)
    ov = v13_gates.estimate_product_overlap_v13(original, bbox, product_mask)

    ranked = mc.ranked or [(mc.method, mc.image, mc.used_full_bbox, mc.score)]
    evaluated = []  # (name, img, full, score, verdict, hides)
    chosen = None
    for name, img, full, score in ranked:
        hides = cover_hides_watermark(original, img, bbox, watermark_mask)
        verdict = gate_fn(img, False, hides)
        evaluated.append((name, img, full, score, verdict, hides))
        if verdict.publish_ok:
            chosen = (name, img, full, score, verdict)
            break
    if chosen is None:
        # Tiered fallback: hiding the watermark is a cover's primary job, so a
        # candidate that hides it always beats one that does not, even if it
        # still trips a soft visual gate. Among equals, lowest beam score wins
        # (ranked is already sorted, so the first match is the best).
        hiding = [e for e in evaluated if e[5]]
        pick = (hiding or evaluated)[0]
        chosen = pick[:5]
    name, img, full, score, cover_verdict = chosen

    bj = boundary_jump_estimate(original, img, bbox)
    used_full_on_product = bool(full and ov["product_overlap"] > 0.10)
    tele.update({
        "v14_cover_method": name,
        "v14_used_full_bbox_on_product": used_full_on_product,
        "v14_boundary_jump": round(bj, 2),
        "v14_micro_cover_candidates": mc.candidates,
    })
    return V14Outcome(img, ST_COV, cover_verdict, tele)
