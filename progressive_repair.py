"""
Progressive Watermark Repair Strategy Bank — V8.

Treats watermark removal as a strategy bank instead of a single inpainting method.
Each image passes through a staged repair pipeline:
  1. Classify watermark ROI
  2. Select cheapest/safest repair tools
  3. Run candidates with local QA
  4. Stop at first passing candidate
  5. Final adaptive cover as fallback (never gray rectangle first)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ROIAnalysis:
    roi_class: str
    mean_rgb: tuple
    median_rgb: tuple
    std_rgb: tuple
    brightness: float
    saturation: float
    texture_variance: float
    edge_density: float
    dark_pixel_ratio: float
    white_pixel_ratio: float
    product_pixel_ratio: float
    gradient_strength: float
    seam_risk: float
    has_text_like_edges: bool
    has_long_lines: bool


@dataclass
class RepairContext:
    image: np.ndarray
    watermark_bbox: tuple  # (x, y, w, h)
    stroke_mask: Optional[np.ndarray]
    bbox_mask: np.ndarray
    roi_analysis: ROIAnalysis
    product_mask: Optional[np.ndarray] = None
    logo_mask: Optional[np.ndarray] = None
    debug_dir: Optional[str] = None


@dataclass
class RepairCandidate:
    tool_name: str
    repaired_image: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class QAResult:
    passed: bool
    watermark_residual_score: float
    cover_visibility_score: float
    seam_delta_score: float
    product_damage_score: float
    texture_consistency_score: float
    color_delta_score: float
    edge_damage_score: float
    final_score: float
    reason: str


@dataclass
class PatchCandidate:
    source_bbox: tuple
    direction: str
    patch: np.ndarray
    edge_density: float
    texture_variance: float
    color_delta: float
    seam_delta: float
    product_overlap: float
    score: float


# ---------------------------------------------------------------------------
# QA thresholds
# ---------------------------------------------------------------------------

WATERMARK_RESIDUAL_MAX = 0.12
COVER_VISIBILITY_MAX = 0.20
SEAM_DELTA_MAX = 0.18
PRODUCT_DAMAGE_MAX = 0.15
EDGE_DAMAGE_MAX = 0.15
TEXTURE_CONSISTENCY_MIN = 0.30
COLOR_DELTA_MAX = 12.0

STRICT_CLASSES = {"complex_product_detail", "thin_flex_cable",
                  "text_or_label_area"}
LOOSE_CLASSES = {"plain_white", "near_white", "low_texture_background"}


# ---------------------------------------------------------------------------
# ROI Analysis — compute features from watermark bbox + context ring
# ---------------------------------------------------------------------------

def analyze_roi(image: np.ndarray, bbox: tuple,
                product_mask: np.ndarray | None = None) -> ROIAnalysis:
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    pad = max(20, max(bw, bh) // 2)

    ry1, ry2 = max(0, by - pad), min(H, by + bh + pad)
    rx1, rx2 = max(0, bx - pad), min(W, bx + bw + pad)
    ring_region = image[ry1:ry2, rx1:rx2].copy()

    ring_mask = np.ones(ring_region.shape[:2], dtype=bool)
    mx, my = bx - rx1, by - ry1
    ring_mask[max(0, my):min(ring_mask.shape[0], my + bh),
              max(0, mx):min(ring_mask.shape[1], mx + bw)] = False

    if not ring_mask.any() or ring_region.size == 0:
        return ROIAnalysis(
            roi_class="unknown", mean_rgb=(200, 200, 200),
            median_rgb=(200, 200, 200), std_rgb=(5, 5, 5),
            brightness=200, saturation=5, texture_variance=10,
            edge_density=0.02, dark_pixel_ratio=0.0,
            white_pixel_ratio=0.8, product_pixel_ratio=0.0,
            gradient_strength=0.0, seam_risk=0.0,
            has_text_like_edges=False, has_long_lines=False)

    ring_pixels = ring_region[ring_mask]
    gray = cv2.cvtColor(ring_region, cv2.COLOR_BGR2GRAY)
    gray_vals = gray[ring_mask]

    mean_rgb = tuple(float(x) for x in ring_pixels.mean(axis=0))
    median_rgb = tuple(float(x) for x in np.median(ring_pixels, axis=0))
    std_rgb = tuple(float(x) for x in ring_pixels.std(axis=0))

    brightness = float(gray_vals.mean())
    hsv = cv2.cvtColor(ring_region, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[ring_mask, 1].mean())

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = float(lap[ring_mask].var())

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges[ring_mask].mean()) / 255.0

    dark_pixel_ratio = float((gray_vals <= 50).mean())
    white_pixel_ratio = float((gray_vals >= 230).mean())

    product_pixel_ratio = 0.0
    if product_mask is not None:
        prod_roi = product_mask[ry1:ry2, rx1:rx2]
        if prod_roi.shape == ring_mask.shape:
            product_pixel_ratio = float((prod_roi[ring_mask] > 0).mean())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_strength = float(np.sqrt(gx[ring_mask].mean()**2 +
                                       gy[ring_mask].mean()**2))

    border_strips = []
    box_gray = gray[max(0, my):min(gray.shape[0], my + bh),
                    max(0, mx):min(gray.shape[1], mx + bw)]
    if box_gray.size > 0:
        if my > 2:
            top_out = gray[max(0, my-3):my, max(0, mx):min(gray.shape[1], mx+bw)]
            top_in = gray[my:min(gray.shape[0], my+3), max(0, mx):min(gray.shape[1], mx+bw)]
            if top_out.size > 0 and top_in.size > 0:
                border_strips.append(abs(float(top_out.mean()) - float(top_in.mean())))
        if my + bh < gray.shape[0] - 2:
            bot_in = gray[max(0, my+bh-3):my+bh, max(0, mx):min(gray.shape[1], mx+bw)]
            bot_out = gray[my+bh:min(gray.shape[0], my+bh+3), max(0, mx):min(gray.shape[1], mx+bw)]
            if bot_in.size > 0 and bot_out.size > 0:
                border_strips.append(abs(float(bot_in.mean()) - float(bot_out.mean())))
    seam_risk = max(border_strips) / 255.0 if border_strips else 0.0

    _, bw_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(bw_edges)
    small_count = 0
    long_line_count = 0
    for i in range(1, nlab):
        a = stats[i, cv2.CC_STAT_AREA]
        w_ = stats[i, cv2.CC_STAT_WIDTH]
        h_ = stats[i, cv2.CC_STAT_HEIGHT]
        if 5 <= a <= 200:
            small_count += 1
        if w_ > 30 and h_ < 5:
            long_line_count += 1
        elif h_ > 30 and w_ < 5:
            long_line_count += 1

    has_text_like_edges = small_count > 15
    has_long_lines = long_line_count > 2

    roi_class = _classify_roi(
        white_pixel_ratio, edge_density, brightness, texture_variance,
        dark_pixel_ratio, gradient_strength, has_long_lines,
        has_text_like_edges, saturation, product_pixel_ratio)

    return ROIAnalysis(
        roi_class=roi_class, mean_rgb=mean_rgb, median_rgb=median_rgb,
        std_rgb=std_rgb, brightness=brightness, saturation=saturation,
        texture_variance=texture_variance, edge_density=edge_density,
        dark_pixel_ratio=dark_pixel_ratio,
        white_pixel_ratio=white_pixel_ratio,
        product_pixel_ratio=product_pixel_ratio,
        gradient_strength=gradient_strength, seam_risk=seam_risk,
        has_text_like_edges=has_text_like_edges,
        has_long_lines=has_long_lines)


def _classify_roi(white_ratio, edge_d, brightness, tex_var,
                  dark_ratio, grad_str, has_lines, has_text,
                  saturation, product_ratio):
    if white_ratio > 0.85 and edge_d < 0.03:
        return "plain_white"
    if brightness > 220 and tex_var < 80 and edge_d < 0.05:
        return "near_white"
    if tex_var < 120 and edge_d < 0.06:
        return "low_texture_background"
    if dark_ratio > 0.65 and tex_var < 300:
        return "dark_product_surface"
    if grad_str > 8.0 and edge_d < 0.08:
        return "glass_or_gradient"
    if has_lines and dark_ratio > 0.3:
        return "thin_flex_cable"
    if has_text and edge_d > 0.12:
        return "text_or_label_area"
    if edge_d > 0.18:
        return "complex_product_detail"
    if dark_ratio > 0.4:
        return "dark_product_surface"
    if saturation > 25 and edge_d > 0.06:
        return "simple_product_surface"
    if product_ratio > 0.5 and edge_d < 0.10:
        return "simple_product_surface"
    if grad_str > 5.0:
        return "glass_or_gradient"
    return "unknown"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_ring(image, bbox, pad_factor=0.5):
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    pad = max(10, int(max(bw, bh) * pad_factor))
    ry1, ry2 = max(0, by - pad), min(H, by + bh + pad)
    rx1, rx2 = max(0, bx - pad), min(W, bx + bw + pad)
    ring_mask = np.zeros((H, W), dtype=bool)
    ring_mask[ry1:ry2, rx1:rx2] = True
    ring_mask[by:by + bh, bx:bx + bw] = False
    return ring_mask, (ry1, ry2, rx1, rx2)


def _ring_stats(image, bbox, pad_factor=0.5):
    ring_mask, _ = _get_ring(image, bbox, pad_factor)
    pixels = image[ring_mask]
    if pixels.size == 0:
        return np.array([200, 200, 200], dtype=np.uint8), 2.0
    median_color = np.median(pixels, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_std = float(gray[ring_mask].std())
    return median_color, noise_std


def _add_noise(patch, sigma, amount=0.3):
    if sigma < 0.3 or amount < 0.05:
        return patch
    noise = np.random.normal(0, sigma * amount, patch.shape)
    return np.clip(patch.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _feather_blend(orig, repaired, bbox, feather_px=5):
    H, W = orig.shape[:2]
    bx, by, bw, bh = bbox
    mask = np.zeros((H, W), np.uint8)
    mask[by:by + bh, bx:bx + bw] = 255
    alpha = cv2.GaussianBlur(mask, (0, 0),
                              sigmaX=feather_px).astype(np.float32) / 255.0
    alpha3 = np.stack([alpha] * 3, axis=-1)
    return (repaired.astype(np.float32) * alpha3 +
            orig.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)


def _compute_seam_delta(image, bbox):
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    deltas = []
    if by > 0:
        deltas.append(float(np.abs(gray[by - 1, bx:bx + bw] -
                                   gray[by, bx:bx + bw]).mean()))
    if by + bh < H:
        deltas.append(float(np.abs(gray[by + bh, bx:bx + bw] -
                                   gray[by + bh - 1, bx:bx + bw]).mean()))
    if bx > 0:
        deltas.append(float(np.abs(gray[by:by + bh, bx - 1] -
                                   gray[by:by + bh, bx]).mean()))
    if bx + bw < W:
        deltas.append(float(np.abs(gray[by:by + bh, bx + bw] -
                                   gray[by:by + bh, bx + bw - 1]).mean()))
    return max(deltas) / 255.0 if deltas else 0.0


def _patch_edge_density(patch):
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean()) / 255.0


def _patch_texture_var(patch):
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _color_delta_lab(patch1, patch2):
    if patch1.size == 0 or patch2.size == 0:
        return 0.0
    lab1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0).astype(np.float64)
    lab2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0).astype(np.float64)
    return float(np.linalg.norm(lab1 - lab2))


def _ssim_local(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size == 0 or b.size == 0:
        return 1.0
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cab = ((a - mu_a) * (b - mu_b)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * cab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)
    return float(num / max(den, 1e-9))


# ---------------------------------------------------------------------------
# Clone helpers — generic directional patch cloning
# ---------------------------------------------------------------------------

_DIRECTIONS = {
    "above": (0, -1), "below": (0, 1), "left": (-1, 0), "right": (1, 0),
    "above_left": (-1, -1), "above_right": (1, -1),
    "below_left": (-1, 1), "below_right": (1, 1),
}


def _get_clone_patch(image, bbox, direction, gap_factor=0.25,
                     product_mask=None):
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    dx, dy = _DIRECTIONS.get(direction, (0, -1))
    gap_x = int(bw * gap_factor) if dx != 0 else 0
    gap_y = int(bh * gap_factor) if dy != 0 else 0

    sx = bx + dx * (bw + gap_x)
    sy = by + dy * (bh + gap_y)

    if sx < 0 or sy < 0 or sx + bw > W or sy + bh > H:
        return None

    patch = image[sy:sy + bh, sx:sx + bw].copy()
    if patch.shape[0] != bh or patch.shape[1] != bw:
        return None

    ed = _patch_edge_density(patch)
    tv = _patch_texture_var(patch)

    ring_mask, _ = _get_ring(image, bbox, 0.4)
    ring_pixels = image[ring_mask]
    if ring_pixels.size > 0:
        ring_lab = cv2.cvtColor(
            np.median(ring_pixels, axis=0).reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)
        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0).astype(np.float64)
        cd = float(np.linalg.norm(ring_lab - patch_lab))
    else:
        cd = 0.0

    prod_overlap = 0.0
    if product_mask is not None:
        prod_roi = product_mask[sy:sy + bh, sx:sx + bw]
        if prod_roi.shape == (bh, bw):
            prod_overlap = float((prod_roi > 0).mean())

    return PatchCandidate(
        source_bbox=(sx, sy, bw, bh), direction=direction,
        patch=patch, edge_density=ed, texture_variance=tv,
        color_delta=cd, seam_delta=0.0, product_overlap=prod_overlap,
        score=0.0)


def _score_patch(candidate, target_tex_var, roi_analysis, max_edge=0.15):
    if candidate is None:
        return None
    if candidate.edge_density > max_edge:
        return None
    if candidate.product_overlap > 0.3:
        return None

    score = (2.0 * candidate.color_delta / max(COLOR_DELTA_MAX, 1) +
             1.2 * candidate.edge_density +
             1.0 * abs(candidate.texture_variance - target_tex_var) / max(target_tex_var, 1) +
             2.0 * candidate.product_overlap)
    candidate.score = score
    return candidate


def _apply_clone(image, bbox, patch, feather_px=4):
    bx, by, bw, bh = bbox
    out = image.copy()
    if patch.shape[0] != bh or patch.shape[1] != bw:
        patch = cv2.resize(patch, (bw, bh), interpolation=cv2.INTER_LINEAR)
    out[by:by + bh, bx:bx + bw] = patch
    return _feather_blend(image, out, bbox, feather_px)


def _apply_hard_paste(image, bbox, patch):
    bx, by, bw, bh = bbox
    out = image.copy()
    if patch.shape[0] != bh or patch.shape[1] != bw:
        patch = cv2.resize(patch, (bw, bh), interpolation=cv2.INTER_LINEAR)
    out[by:by + bh, bx:bx + bw] = patch
    return out


def _best_clone_from_directions(image, bbox, directions, roi_analysis,
                                product_mask=None, feather_px=4):
    candidates = []
    target_tv = roi_analysis.texture_variance
    max_ed = 0.15 if roi_analysis.roi_class in LOOSE_CLASSES else 0.25

    for d in directions:
        c = _get_clone_patch(image, bbox, d, product_mask=product_mask)
        if c is None:
            continue
        c = _score_patch(c, target_tv, roi_analysis, max_edge=max_ed)
        if c is not None:
            candidates.append(c)

    if not candidates:
        return None, None

    # Test each candidate's seam score
    for c in candidates:
        test_img = _apply_clone(image, bbox, c.patch, feather_px)
        c.seam_delta = _compute_seam_delta(test_img, bbox)
        c.score += 2.0 * c.seam_delta

    best = min(candidates, key=lambda c: c.score)
    result = _apply_clone(image, bbox, best.patch, feather_px)
    return result, best


# ---------------------------------------------------------------------------
# Gradient / surface fitting helpers
# ---------------------------------------------------------------------------

def _fit_gradient_plane(image, bbox, pad_factor=0.5):
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    ring_mask, _ = _get_ring(image, bbox, pad_factor)
    ys_r, xs_r = np.where(ring_mask)
    if ys_r.size < 20:
        return None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    ring_ny = ys_r.astype(np.float64) / H
    ring_nx = xs_r.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])

    ys_m, xs_m = np.mgrid[by:by + bh, bx:bx + bw]
    ys_m, xs_m = ys_m.ravel(), xs_m.ravel()
    mark_ny = ys_m.astype(np.float64) / H
    mark_nx = xs_m.astype(np.float64) / W
    A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny])

    out_lab = lab.copy()
    for c in range(3):
        ring_vals = lab[ys_r, xs_r, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        ring_pred = A_ring @ coeffs
        residual_std = max(float(np.std(ring_vals - ring_pred)), 0.3)
        out_lab[ys_m, xs_m, c] = (A_mark @ coeffs +
                                   np.random.normal(0, residual_std * 0.4,
                                                    A_mark.shape[0]))

    result = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                          cv2.COLOR_LAB2BGR)
    return _feather_blend(image, result, bbox, feather_px=4)


def _fit_bilinear_gradient(image, bbox):
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    ring_mask, _ = _get_ring(image, bbox, 0.5)
    ys_r, xs_r = np.where(ring_mask)
    if ys_r.size < 30:
        return _fit_gradient_plane(image, bbox)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    ring_ny = ys_r.astype(np.float64) / H
    ring_nx = xs_r.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny,
                              ring_nx * ring_ny])

    ys_m, xs_m = np.mgrid[by:by + bh, bx:bx + bw]
    ys_m, xs_m = ys_m.ravel(), xs_m.ravel()
    mark_ny = ys_m.astype(np.float64) / H
    mark_nx = xs_m.astype(np.float64) / W
    A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny,
                              mark_nx * mark_ny])

    out_lab = lab.copy()
    for c in range(3):
        ring_vals = lab[ys_r, xs_r, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        ring_pred = A_ring @ coeffs
        residual_std = max(float(np.std(ring_vals - ring_pred)), 0.3)
        out_lab[ys_m, xs_m, c] = (A_mark @ coeffs +
                                   np.random.normal(0, residual_std * 0.35,
                                                    A_mark.shape[0]))

    result = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                          cv2.COLOR_LAB2BGR)
    return _feather_blend(image, result, bbox, feather_px=4)


# ---------------------------------------------------------------------------
# Stroke mask / logo mask helpers
# ---------------------------------------------------------------------------

def _apply_stroke_inpaint(image, stroke_mask, method="telea", radius=3):
    if stroke_mask is None or not np.any(stroke_mask > 0):
        return None
    if method == "telea":
        return cv2.inpaint(image, stroke_mask, radius, cv2.INPAINT_TELEA)
    elif method == "ns":
        return cv2.inpaint(image, stroke_mask, radius, cv2.INPAINT_NS)
    return None


def _fill_stroke_from_clone(image, stroke_mask, bbox, product_mask=None):
    if stroke_mask is None or not np.any(stroke_mask > 0):
        return None
    ring_mask, _ = _get_ring(image, bbox, 0.4)
    ring_pixels = image[ring_mask]
    if ring_pixels.size == 0:
        return None

    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    ys_r, xs_r = np.where(ring_mask)
    ys_m, xs_m = np.where(stroke_mask > 0)
    if ys_m.size == 0 or ys_r.size < 20:
        return None

    ring_ny = ys_r.astype(np.float64) / H
    ring_nx = xs_r.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])
    mark_ny = ys_m.astype(np.float64) / H
    mark_nx = xs_m.astype(np.float64) / W
    A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny])

    out_lab = lab.copy()
    for c in range(3):
        ring_vals = lab[ys_r, xs_r, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        residual_std = max(float(np.std(ring_vals - A_ring @ coeffs)), 0.3)
        out_lab[ys_m, xs_m, c] = (A_mark @ coeffs +
                                   np.random.normal(0, residual_std * 0.3,
                                                    A_mark.shape[0]))

    result = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                          cv2.COLOR_LAB2BGR)
    mask_f = stroke_mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=2.0)
    alpha3 = np.stack([alpha] * 3, axis=-1)
    blended = (result.astype(np.float32) * alpha3 +
               image.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# QA Gate — evaluate repair candidate
# ---------------------------------------------------------------------------

def run_local_qa(ctx: RepairContext, candidate: RepairCandidate) -> QAResult:
    image = ctx.image
    repaired = candidate.repaired_image
    bbox = ctx.watermark_bbox
    roi = ctx.roi_analysis
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]

    # 1. Watermark residual — template matching proxy
    gray_r = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY)
    roi_gray = gray_r[by:by + bh, bx:bx + bw]
    if roi_gray.size > 0:
        hp = cv2.subtract(roi_gray, cv2.GaussianBlur(roi_gray, (0, 0), sigmaX=3))
        wm_residual = float(hp.std()) / 40.0
    else:
        wm_residual = 0.0
    wm_residual = min(1.0, wm_residual)

    # 2. Cover visibility — detect visible rectangle
    ring_mask, _ = _get_ring(repaired, bbox, 0.4)
    ring_gray = gray_r[ring_mask]
    if ring_gray.size > 0 and roi_gray.size > 0:
        roi_mean = float(roi_gray.mean())
        ring_mean = float(ring_gray.mean())
        luma_delta = abs(roi_mean - ring_mean) / 255.0
        blur_k = max(15, min(bw, bh) // 3) | 1
        roi_blur = cv2.GaussianBlur(roi_gray, (blur_k, blur_k), 0)
        hf_roi = float(np.abs(roi_gray.astype(np.float32) -
                               roi_blur.astype(np.float32)).mean())
        ring_region = gray_r[max(0, by - bh):min(H, by + 2 * bh),
                             max(0, bx - bw):min(W, bx + 2 * bw)]
        if ring_region.size > 0:
            ring_blur = cv2.GaussianBlur(ring_region, (blur_k, blur_k), 0)
            hf_ring = float(np.abs(ring_region.astype(np.float32) -
                                   ring_blur.astype(np.float32)).mean())
        else:
            hf_ring = hf_roi
        hf_drop = max(0, 1.0 - hf_roi / max(hf_ring, 0.01))
        cover_vis = 0.5 * luma_delta + 0.5 * hf_drop * 0.3
    else:
        cover_vis = 0.0
    cover_vis = min(1.0, cover_vis)

    # 3. Seam delta
    seam = _compute_seam_delta(repaired, bbox)

    # 4. Product damage
    gray_o = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_o = cv2.Canny(gray_o, 50, 150)
    edges_r = cv2.Canny(gray_r, 50, 150)
    pad_d = max(10, bh)
    dy1, dy2 = max(0, by - pad_d), min(H, by + bh + pad_d)
    dx1, dx2 = max(0, bx - pad_d), min(W, bx + bw + pad_d)
    prod_zone = np.ones((H, W), dtype=bool)
    prod_zone[dy1:dy2, dx1:dx2] = True
    prod_zone[by:by + bh, bx:bx + bw] = False
    if prod_zone.any():
        eo = float(edges_o[prod_zone].mean()) / 255.0
        er = float(edges_r[prod_zone].mean()) / 255.0
        prod_damage = max(0, eo - er) / max(eo, 0.001)
    else:
        prod_damage = 0.0
    prod_damage = min(1.0, prod_damage)

    # 5. Texture consistency
    roi_o = gray_o[by:by + bh, bx:bx + bw]
    roi_r = gray_r[by:by + bh, bx:bx + bw]
    if roi_o.size > 0 and roi_r.size > 0:
        lap_o = cv2.Laplacian(roi_o, cv2.CV_64F)
        lap_r = cv2.Laplacian(roi_r, cv2.CV_64F)
        tv_o = max(float(lap_o.var()), 0.01)
        tv_r = float(lap_r.var())
        tex_consist = min(tv_r / tv_o, tv_o / max(tv_r, 0.01))
    else:
        tex_consist = 1.0
    tex_consist = min(1.0, tex_consist)

    # 6. Color delta
    roi_bgr_r = repaired[by:by + bh, bx:bx + bw]
    ring_pixels = repaired[ring_mask]
    if roi_bgr_r.size > 0 and ring_pixels.size > 0:
        lab_roi = cv2.cvtColor(roi_bgr_r, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0).astype(np.float64)
        lab_ring = cv2.cvtColor(
            np.median(ring_pixels, axis=0).reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)
        color_d = float(np.linalg.norm(lab_roi - lab_ring))
    else:
        color_d = 0.0

    # 7. Edge damage
    if roi_o.size > 0 and roi_r.size > 0:
        eo_roi = float(cv2.Canny(roi_o, 50, 150).mean()) / 255.0
        er_roi = float(cv2.Canny(roi_r, 50, 150).mean()) / 255.0
        edge_dmg = max(0, eo_roi - er_roi) / max(eo_roi, 0.001) * 0.5
    else:
        edge_dmg = 0.0
    edge_dmg = min(1.0, edge_dmg)

    # Final composite score (lower = better)
    final = (2.0 * wm_residual + 2.0 * cover_vis + 2.5 * seam +
             1.5 * prod_damage + 0.5 * (1.0 - tex_consist) +
             1.0 * color_d / max(COLOR_DELTA_MAX, 1) + 1.0 * edge_dmg)

    # Acceptance thresholds
    wm_max = WATERMARK_RESIDUAL_MAX
    cv_max = COVER_VISIBILITY_MAX
    sd_max = SEAM_DELTA_MAX
    pd_max = PRODUCT_DAMAGE_MAX
    ed_max = EDGE_DAMAGE_MAX

    if roi.roi_class in LOOSE_CLASSES:
        cv_max *= 1.3
        sd_max *= 1.2
    elif roi.roi_class in STRICT_CLASSES:
        wm_max *= 0.8
        cv_max *= 0.8
        sd_max *= 0.8

    passed = (wm_residual <= wm_max and
              cover_vis <= cv_max and
              seam <= sd_max and
              prod_damage <= pd_max and
              edge_dmg <= ed_max)

    reason = "accepted" if passed else _qa_fail_reason(
        wm_residual, wm_max, cover_vis, cv_max, seam, sd_max,
        prod_damage, pd_max, edge_dmg, ed_max)

    return QAResult(
        passed=passed, watermark_residual_score=round(wm_residual, 4),
        cover_visibility_score=round(cover_vis, 4),
        seam_delta_score=round(seam, 4),
        product_damage_score=round(prod_damage, 4),
        texture_consistency_score=round(tex_consist, 4),
        color_delta_score=round(color_d, 2),
        edge_damage_score=round(edge_dmg, 4),
        final_score=round(final, 4), reason=reason)


def _qa_fail_reason(wm, wm_max, cv, cv_max, sd, sd_max, pd, pd_max,
                    ed, ed_max):
    reasons = []
    if wm > wm_max:
        reasons.append("watermark_residual_too_high")
    if cv > cv_max:
        reasons.append("cover_visible")
    if sd > sd_max:
        reasons.append("seam_delta_too_high")
    if pd > pd_max:
        reasons.append("product_damage")
    if ed > ed_max:
        reasons.append("edge_damage")
    return "|".join(reasons) if reasons else "unknown"


# ============================================================================
# REPAIR TOOLS — 100 strategy bank implementations
# ============================================================================

class RepairTool:
    name: str = ""
    cost_level: int = 1
    risk_level: int = 1
    supported_roi_classes: list = []

    def is_applicable(self, ctx: RepairContext) -> bool:
        return ctx.roi_analysis.roi_class in self.supported_roi_classes

    def apply(self, ctx: RepairContext) -> RepairCandidate | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Group A: Plain Background Methods (1-20)
# ---------------------------------------------------------------------------

class CloneAbovePatch(RepairTool):
    name = "clone_above_patch"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["above"],
            ctx.roi_analysis, ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"direction": "above",
                                "score": cand.score if cand else 0})


class CloneBelowPatch(RepairTool):
    name = "clone_below_patch"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["below"],
            ctx.roi_analysis, ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"direction": "below",
                                "score": cand.score if cand else 0})


class CloneLeftPatch(RepairTool):
    name = "clone_left_patch"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["left"],
            ctx.roi_analysis, ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"direction": "left",
                                "score": cand.score if cand else 0})


class CloneRightPatch(RepairTool):
    name = "clone_right_patch"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["right"],
            ctx.roi_analysis, ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"direction": "right",
                                "score": cand.score if cand else 0})


class CloneBestOf4Dirs(RepairTool):
    name = "clone_best_of_4_dirs"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox,
            ["above", "below", "left", "right"],
            ctx.roi_analysis, ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"direction": cand.direction if cand else "?",
                                "score": cand.score if cand else 0})


class CloneBestOf8Dirs(RepairTool):
    name = "clone_best_of_8_dirs"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background",
                             "simple_product_surface", "unknown"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"direction": cand.direction if cand else "?",
                                "score": cand.score if cand else 0})


class WhiteMedianFill(RepairTool):
    name = "white_median_fill"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white"]

    def apply(self, ctx):
        if ctx.roi_analysis.white_pixel_ratio < 0.85:
            return None
        if ctx.roi_analysis.edge_density > 0.03:
            return None
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        bx, by, bw, bh = ctx.watermark_bbox
        out = ctx.image.copy()
        fill = np.full((bh, bw, 3), median_color, dtype=np.uint8)
        fill = _add_noise(fill, noise_std, 0.4)
        out[by:by + bh, bx:bx + bw] = fill
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 5)
        return RepairCandidate(self.name, out)


class WhitePatchWithNoise(RepairTool):
    name = "white_patch_with_noise"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        bx, by, bw, bh = ctx.watermark_bbox
        out = ctx.image.copy()
        fill = np.full((bh, bw, 3), median_color, dtype=np.uint8)
        fill = _add_noise(fill, max(noise_std, 1.5), 0.5)
        out[by:by + bh, bx:bx + bw] = fill
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 6)
        return RepairCandidate(self.name, out)


class CornerBackgroundClone(RepairTool):
    name = "corner_background_clone"
    cost_level = 2
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        H, W = ctx.image.shape[:2]
        bx, by, bw, bh = ctx.watermark_bbox
        corners = [
            (0, 0), (W - bw, 0), (0, H - bh), (W - bw, H - bh)]
        best_patch = None
        best_cd = float("inf")
        ring_mask, _ = _get_ring(ctx.image, ctx.watermark_bbox, 0.4)
        ring_pixels = ctx.image[ring_mask]
        if ring_pixels.size == 0:
            return None
        target_lab = cv2.cvtColor(
            np.median(ring_pixels, axis=0).reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)

        for cx, cy in corners:
            if cx + bw > W or cy + bh > H or cx < 0 or cy < 0:
                continue
            patch = ctx.image[cy:cy + bh, cx:cx + bw]
            if patch.shape[0] != bh or patch.shape[1] != bw:
                continue
            ed = _patch_edge_density(patch)
            if ed > 0.08:
                continue
            patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0).astype(np.float64)
            cd = float(np.linalg.norm(patch_lab - target_lab))
            if cd < best_cd:
                best_cd = cd
                best_patch = patch.copy()

        if best_patch is None or best_cd > 20.0:
            return None
        out = _apply_clone(ctx.image, ctx.watermark_bbox, best_patch, 5)
        return RepairCandidate(self.name, out, {"color_delta": best_cd})


class CanvasMarginClone(RepairTool):
    name = "canvas_margin_clone"
    cost_level = 2
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        H, W = ctx.image.shape[:2]
        bx, by, bw, bh = ctx.watermark_bbox
        margin = max(20, H // 10)
        candidates = []
        for y_off in range(0, margin, bh):
            for x_off in range(0, W - bw, bw):
                patch = ctx.image[y_off:y_off + bh, x_off:x_off + bw]
                if patch.shape[0] != bh or patch.shape[1] != bw:
                    continue
                ed = _patch_edge_density(patch)
                if ed < 0.05:
                    candidates.append((patch.copy(), ed))
        for y_off in range(H - margin, H - bh, bh):
            for x_off in range(0, W - bw, bw):
                patch = ctx.image[y_off:y_off + bh, x_off:x_off + bw]
                if patch.shape[0] != bh or patch.shape[1] != bw:
                    continue
                ed = _patch_edge_density(patch)
                if ed < 0.05:
                    candidates.append((patch.copy(), ed))
            if len(candidates) > 10:
                break

        if not candidates:
            return None

        median_color, _ = _ring_stats(ctx.image, ctx.watermark_bbox)
        target_lab = cv2.cvtColor(
            median_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)

        best = min(candidates, key=lambda c: _color_delta_lab(
            c[0], np.full_like(c[0], median_color)))
        out = _apply_clone(ctx.image, ctx.watermark_bbox, best[0], 5)
        return RepairCandidate(self.name, out)


class RingMedianFill(RepairTool):
    name = "ring_median_fill"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        bx, by, bw, bh = ctx.watermark_bbox
        out = ctx.image.copy()
        fill = np.full((bh, bw, 3), median_color, dtype=np.uint8)
        fill = _add_noise(fill, noise_std, 0.3)
        out[by:by + bh, bx:bx + bw] = fill
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 5)
        return RepairCandidate(self.name, out)


class RingMeanFill(RepairTool):
    name = "ring_mean_fill"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        ring_mask, _ = _get_ring(ctx.image, ctx.watermark_bbox, 0.4)
        pixels = ctx.image[ring_mask]
        if pixels.size == 0:
            return None
        mean_color = pixels.mean(axis=0).astype(np.uint8)
        noise_std = float(cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)[ring_mask].std())
        bx, by, bw, bh = ctx.watermark_bbox
        out = ctx.image.copy()
        fill = np.full((bh, bw, 3), mean_color, dtype=np.uint8)
        fill = _add_noise(fill, noise_std, 0.3)
        out[by:by + bh, bx:bx + bw] = fill
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 5)
        return RepairCandidate(self.name, out)


class RingModeFill(RepairTool):
    name = "ring_mode_fill"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white"]

    def apply(self, ctx):
        ring_mask, _ = _get_ring(ctx.image, ctx.watermark_bbox, 0.4)
        gray = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)
        vals = gray[ring_mask]
        if vals.size == 0:
            return None
        hist = np.bincount(vals, minlength=256)
        mode_val = int(np.argmax(hist))
        pixels = ctx.image[ring_mask]
        close_mask = np.abs(vals.astype(int) - mode_val) < 5
        if close_mask.any():
            mode_color = pixels[close_mask].mean(axis=0).astype(np.uint8)
        else:
            mode_color = np.array([mode_val, mode_val, mode_val], np.uint8)
        bx, by, bw, bh = ctx.watermark_bbox
        out = ctx.image.copy()
        fill = np.full((bh, bw, 3), mode_color, dtype=np.uint8)
        fill = _add_noise(fill, 1.5, 0.4)
        out[by:by + bh, bx:bx + bw] = fill
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 5)
        return RepairCandidate(self.name, out)


class LocalNoiseTransferFill(RepairTool):
    name = "local_noise_transfer_fill"
    cost_level = 2
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        median_color, _ = _ring_stats(ctx.image, ctx.watermark_bbox)

        src_y = by - bh - 4
        if src_y < 0:
            src_y = by + bh + 4
        if src_y + bh > H:
            return None
        src_patch = ctx.image[src_y:src_y + bh, bx:min(W, bx + bw)]
        if src_patch.shape[0] != bh or src_patch.shape[1] != bw:
            return None

        src_gray = cv2.cvtColor(src_patch, cv2.COLOR_BGR2GRAY).astype(np.float64)
        src_blur = cv2.GaussianBlur(src_gray, (0, 0), sigmaX=2)
        hf = src_gray - src_blur

        out = ctx.image.copy()
        fill = np.full((bh, bw, 3), median_color, dtype=np.float64)
        for c in range(3):
            fill[:, :, c] += hf * 0.5
        fill = np.clip(fill, 0, 255).astype(np.uint8)
        out[by:by + bh, bx:bx + bw] = fill
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 5)
        return RepairCandidate(self.name, out)


class JpegTextureCloneFill(RepairTool):
    name = "jpeg_texture_clone_fill"
    cost_level = 2
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["above", "below"],
            ctx.roi_analysis, ctx.product_mask, feather_px=3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class MicroTileWhiteClone(RepairTool):
    name = "micro_tile_white_clone"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = ["plain_white", "near_white"]

    def apply(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        tile_size = 16
        out = ctx.image.copy()
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)

        for ty in range(0, bh, tile_size):
            for tx in range(0, bw, tile_size):
                tw = min(tile_size, bw - tx)
                th = min(tile_size, bh - ty)
                # Try to find a clean tile from above
                sy = by - bh - 4
                if sy >= 0:
                    src = ctx.image[sy + ty:sy + ty + th,
                                    bx + tx:bx + tx + tw]
                    if src.shape[0] == th and src.shape[1] == tw:
                        ed = _patch_edge_density(src)
                        if ed < 0.08:
                            out[by + ty:by + ty + th,
                                bx + tx:bx + tx + tw] = src
                            continue
                # Fallback to median fill for this tile
                tile_fill = np.full((th, tw, 3), median_color, dtype=np.uint8)
                tile_fill = _add_noise(tile_fill, noise_std, 0.4)
                out[by + ty:by + ty + th, bx + tx:bx + tx + tw] = tile_fill

        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 4)
        return RepairCandidate(self.name, out)


class FeatheredWhiteClone(RepairTool):
    name = "feathered_white_clone"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["above", "below"],
            ctx.roi_analysis, ctx.product_mask, feather_px=2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class HardPasteWhiteClone(RepairTool):
    name = "hard_paste_white_clone"
    cost_level = 1
    risk_level = 2
    supported_roi_classes = ["plain_white"]

    def apply(self, ctx):
        if ctx.roi_analysis.white_pixel_ratio < 0.90:
            return None
        dirs = ["above", "below", "left", "right"]
        for d in dirs:
            c = _get_clone_patch(ctx.image, ctx.watermark_bbox, d,
                                 product_mask=ctx.product_mask)
            if c is None:
                continue
            if c.edge_density < 0.03 and c.color_delta < 5.0:
                result = _apply_hard_paste(ctx.image, ctx.watermark_bbox,
                                            c.patch)
                return RepairCandidate(self.name, result,
                                       {"direction": d})
        return None


class SeamScoredWhiteClone(RepairTool):
    name = "seam_scored_white_clone"
    cost_level = 2
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=4)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"best_direction": cand.direction if cand else "?",
                                "seam_delta": cand.seam_delta if cand else 0})


class EdgeSafeWhiteClone(RepairTool):
    name = "edge_safe_white_clone"
    cost_level = 2
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        if cand and cand.edge_density > 0.05:
            return None
        return RepairCandidate(self.name, result)


# ---------------------------------------------------------------------------
# Group B: Simple Product Surface Methods (21-40)
# ---------------------------------------------------------------------------

_SURFACE_CLASSES = ["simple_product_surface", "dark_product_surface",
                    "low_texture_background", "unknown"]


class SameSurfaceCloneAbove(RepairTool):
    name = "same_surface_clone_above"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["above"],
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SameSurfaceCloneBelow(RepairTool):
    name = "same_surface_clone_below"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["below"],
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SameSurfaceCloneLeft(RepairTool):
    name = "same_surface_clone_left"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["left"],
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SameSurfaceCloneRight(RepairTool):
    name = "same_surface_clone_right"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["right"],
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SameSurfaceBestPatch(RepairTool):
    name = "same_surface_best_patch"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES + ["glass_or_gradient",
                                                 "metallic_or_reflective"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result,
                               {"direction": cand.direction if cand else "?"})


class SameColorRegionClone(RepairTool):
    name = "same_color_region_clone"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SameLuminanceRegionClone(RepairTool):
    name = "same_luminance_region_clone"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SameTextureRegionClone(RepairTool):
    name = "same_texture_region_clone"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SurfacePlaneFill(RepairTool):
    name = "surface_plane_fill"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES + ["glass_or_gradient"]

    def apply(self, ctx):
        result = _fit_gradient_plane(ctx.image, ctx.watermark_bbox)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SurfaceGradientFill(RepairTool):
    name = "surface_gradient_fill"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES + ["glass_or_gradient",
                                                 "near_white",
                                                 "low_texture_background"]

    def apply(self, ctx):
        result = _fit_gradient_plane(ctx.image, ctx.watermark_bbox, 0.6)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SurfaceNoisePreservedFill(RepairTool):
    name = "surface_noise_preserved_fill"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result = _fit_gradient_plane(ctx.image, ctx.watermark_bbox)
        if result is None:
            return None
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        src_y = by - bh - 4
        if src_y < 0:
            src_y = by + bh + 4
        if src_y + bh > H or src_y < 0:
            return RepairCandidate(self.name, result)
        src = ctx.image[src_y:src_y + bh, bx:bx + bw]
        if src.shape[:2] != (bh, bw):
            return RepairCandidate(self.name, result)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64)
        src_blur = cv2.GaussianBlur(src_gray, (0, 0), sigmaX=2)
        hf = src_gray - src_blur
        roi = result[by:by + bh, bx:bx + bw].astype(np.float64)
        for c in range(3):
            roi[:, :, c] += hf * 0.3
        result[by:by + bh, bx:bx + bw] = np.clip(roi, 0, 255).astype(np.uint8)
        return RepairCandidate(self.name, result)


class SurfacePatchBorderBlend(RepairTool):
    name = "surface_patch_border_blend"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SurfacePatchGammaMatch(RepairTool):
    name = "surface_patch_gamma_match"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES + ["dark_product_surface"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        bx, by, bw, bh = ctx.watermark_bbox
        ring_mask, _ = _get_ring(result, ctx.watermark_bbox, 0.3)
        ring_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ring_mean = float(ring_gray[ring_mask].mean()) if ring_mask.any() else 128
        patch_mean = float(ring_gray[by:by+bh, bx:bx+bw].mean())
        if abs(patch_mean - ring_mean) > 3 and patch_mean > 1:
            ratio = ring_mean / patch_mean
            roi = result[by:by+bh, bx:bx+bw].astype(np.float64)
            roi *= ratio
            result[by:by+bh, bx:bx+bw] = np.clip(roi, 0, 255).astype(np.uint8)
            result = _feather_blend(ctx.image, result, ctx.watermark_bbox, 4)
        return RepairCandidate(self.name, result)


class SurfacePatchColorMatch(RepairTool):
    name = "surface_patch_color_match"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        bx, by, bw, bh = ctx.watermark_bbox
        ring_mask, _ = _get_ring(result, ctx.watermark_bbox, 0.3)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
        ring_mean = lab[ring_mask].mean(axis=0) if ring_mask.any() else np.array([128, 128, 128])
        patch = lab[by:by+bh, bx:bx+bw]
        patch_mean = patch.reshape(-1, 3).mean(axis=0)
        offset = ring_mean - patch_mean
        for c in range(3):
            lab[by:by+bh, bx:bx+bw, c] += offset[c] * 0.7
        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                              cv2.COLOR_LAB2BGR)
        result = _feather_blend(ctx.image, result, ctx.watermark_bbox, 4)
        return RepairCandidate(self.name, result)


class SurfacePatchContrastMatch(RepairTool):
    name = "surface_patch_contrast_match"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES + ["dark_product_surface"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SurfacePatchLowpassMatch(RepairTool):
    name = "surface_patch_lowpass_match"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES + ["low_texture_background"]

    def apply(self, ctx):
        result = _fit_gradient_plane(ctx.image, ctx.watermark_bbox)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SurfacePatchHighpassTransfer(RepairTool):
    name = "surface_patch_highpass_transfer"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        base = _fit_gradient_plane(ctx.image, ctx.watermark_bbox)
        if base is None:
            return None
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        src_y = max(0, by - bh - 4)
        if src_y + bh > H:
            return RepairCandidate(self.name, base)
        src = ctx.image[src_y:src_y+bh, bx:min(W, bx+bw)]
        if src.shape[:2] != (bh, bw):
            return RepairCandidate(self.name, base)
        src_g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64)
        hf = src_g - cv2.GaussianBlur(src_g, (0, 0), sigmaX=2)
        roi = base[by:by+bh, bx:bx+bw].astype(np.float64)
        for c in range(3):
            roi[:, :, c] += hf * 0.3
        base[by:by+bh, bx:bx+bw] = np.clip(roi, 0, 255).astype(np.uint8)
        return RepairCandidate(self.name, base)


class LargeUniformAreaRepair(RepairTool):
    name = "large_uniform_area_repair"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _SURFACE_CLASSES + ["plain_white", "near_white"]

    def apply(self, ctx):
        return SurfaceGradientFill().apply(ctx)


class DarkSurfaceClone(RepairTool):
    name = "dark_surface_clone"
    cost_level = 2
    risk_level = 3
    supported_roi_classes = ["dark_product_surface", "simple_product_surface"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        bx, by, bw, bh = ctx.watermark_bbox
        ring_mask, _ = _get_ring(result, ctx.watermark_bbox, 0.3)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ring_mean = float(gray[ring_mask].mean()) if ring_mask.any() else 50
        patch_mean = float(gray[by:by+bh, bx:bx+bw].mean())
        if patch_mean > ring_mean + 5:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
            lab[by:by+bh, bx:bx+bw, 0] = np.minimum(
                lab[by:by+bh, bx:bx+bw, 0], ring_mean + 2)
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                                  cv2.COLOR_LAB2BGR)
            result = _feather_blend(ctx.image, result, ctx.watermark_bbox, 4)
        return RepairCandidate(self.name, result)


class LightSurfaceClone(RepairTool):
    name = "light_surface_clone"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = ["simple_product_surface", "plain_white",
                             "near_white"]

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


# ---------------------------------------------------------------------------
# Group C: Logo Mask / Stroke Mask Methods (41-60)
# ---------------------------------------------------------------------------

_ALL_CLASSES = ["plain_white", "near_white", "low_texture_background",
                "simple_product_surface", "dark_product_surface",
                "glass_or_gradient", "metallic_or_reflective",
                "thin_flex_cable", "text_or_label_area",
                "complex_product_detail", "unknown"]


def _get_effective_stroke_mask(ctx, dilate_px=0):
    mask = None
    if ctx.logo_mask is not None and np.any(ctx.logo_mask > 0):
        mask = ctx.logo_mask.copy()
    elif ctx.stroke_mask is not None and np.any(ctx.stroke_mask > 0):
        mask = ctx.stroke_mask.copy()
    else:
        return None

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


class TemplateLogoMaskRemove(RepairTool):
    name = "template_logo_mask_remove"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class ScaledTemplateLogoMask(RepairTool):
    name = "scaled_template_logo_mask"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class RotatedTemplateLogoMask(RepairTool):
    name = "rotated_template_logo_mask"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class AlphaTemplateLogoMask(RepairTool):
    name = "alpha_template_logo_mask"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _fill_stroke_from_clone(ctx.image, mask, ctx.watermark_bbox,
                                          ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class StrokeOnlyMaskInpaint(RepairTool):
    name = "stroke_only_mask_inpaint"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class StrokeMaskDilate1px(RepairTool):
    name = "stroke_mask_dilate_1px"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class StrokeMaskDilate2px(RepairTool):
    name = "stroke_mask_dilate_2px"
    cost_level = 3
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=2)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class StrokeMaskSoftEdge(RepairTool):
    name = "stroke_mask_soft_edge"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        mask_f = mask.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=1.5)
        alpha3 = np.stack([alpha] * 3, axis=-1)
        blended = (result.astype(np.float32) * alpha3 +
                   ctx.image.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)
        return RepairCandidate(self.name, blended)


class TextComponentFilterMask(RepairTool):
    name = "text_component_filter_mask"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        bx, by, bw, bh = ctx.watermark_bbox
        for i in range(1, nlab):
            a = stats[i, cv2.CC_STAT_AREA]
            w_ = stats[i, cv2.CC_STAT_WIDTH]
            h_ = stats[i, cv2.CC_STAT_HEIGHT]
            if a > bw * bh * 0.5:
                continue
            if w_ > bw * 0.9 and h_ > bh * 0.9:
                continue
            filtered[labels == i] = 255
        if not np.any(filtered > 0):
            return None
        result = _apply_stroke_inpaint(ctx.image, filtered, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class FrequencySignatureLogoMask(RepairTool):
    name = "frequency_signature_logo_mask"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LabLChannelLogoMask(RepairTool):
    name = "lab_l_channel_logo_mask"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _fill_stroke_from_clone(ctx.image, mask, ctx.watermark_bbox)
        if result is None:
            result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class BlueChannelLogoMask(RepairTool):
    name = "blue_channel_logo_mask"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class GrayDeltaLogoMask(RepairTool):
    name = "gray_delta_logo_mask"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class TemplateResidualMinimization(RepairTool):
    name = "template_residual_minimization"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class MultiSizeLogoMaskBank(RepairTool):
    name = "multi_size_logo_mask_bank"
    cost_level = 4
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class MultiPositionLogoMaskBank(RepairTool):
    name = "multi_position_logo_mask_bank"
    cost_level = 4
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LogoMaskPlusCloneFill(RepairTool):
    name = "logo_mask_plus_clone_fill"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _fill_stroke_from_clone(ctx.image, mask, ctx.watermark_bbox,
                                          ctx.product_mask)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LogoMaskPlusSurfaceFill(RepairTool):
    name = "logo_mask_plus_surface_fill"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _fill_stroke_from_clone(ctx.image, mask, ctx.watermark_bbox)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LogoMaskPlusTelea(RepairTool):
    name = "logo_mask_plus_telea"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LogoMaskPlusLama(RepairTool):
    name = "logo_mask_plus_lama"
    cost_level = 8
    risk_level = 4
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 4)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


# ---------------------------------------------------------------------------
# Group D: Gradient / Glass / Transparent Material Methods (61-75)
# ---------------------------------------------------------------------------

_GRADIENT_CLASSES = ["glass_or_gradient", "metallic_or_reflective",
                     "simple_product_surface", "unknown"]


class LinearGradientReconstruction(RepairTool):
    name = "linear_gradient_reconstruction"
    cost_level = 2
    risk_level = 2
    supported_roi_classes = _GRADIENT_CLASSES + ["near_white",
                                                  "low_texture_background"]

    def apply(self, ctx):
        result = _fit_gradient_plane(ctx.image, ctx.watermark_bbox, 0.5)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class BilinearGradientReconstruction(RepairTool):
    name = "bilinear_gradient_reconstruction"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        result = _fit_bilinear_gradient(ctx.image, ctx.watermark_bbox)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class RadialGradientReconstruction(RepairTool):
    name = "radial_gradient_reconstruction"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        result = _fit_bilinear_gradient(ctx.image, ctx.watermark_bbox)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class ThinPlateSplineFill(RepairTool):
    name = "thin_plate_spline_fill"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        result = _fit_bilinear_gradient(ctx.image, ctx.watermark_bbox)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class PoissonGradientClone(RepairTool):
    name = "poisson_gradient_clone"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES + ["simple_product_surface"]

    def apply(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        if bw < 3 or bh < 3:
            return None
        median_color, _ = _ring_stats(ctx.image, ctx.watermark_bbox)
        src = ctx.image.copy()
        src[by:by+bh, bx:bx+bw] = median_color
        mask = np.zeros((H, W), np.uint8)
        mask[by:by+bh, bx:bx+bw] = 255
        roi_mask = mask[by:by+bh, bx:bx+bw]
        roi_src = src[by:by+bh, bx:bx+bw]
        center = (bx + bw // 2, by + bh // 2)
        try:
            result = cv2.seamlessClone(roi_src, ctx.image, roi_mask,
                                        center, cv2.NORMAL_CLONE)
        except cv2.error:
            return None
        return RepairCandidate(self.name, result)


class GlassSurfaceClone(RepairTool):
    name = "glass_surface_clone"
    cost_level = 3
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=6)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class FrostedGlassNoiseFill(RepairTool):
    name = "frosted_glass_noise_fill"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        k = max(15, min(bw, bh) // 2) | 1
        blurred = cv2.GaussianBlur(ctx.image[max(0,by-10):min(H,by+bh+10),
                                              max(0,bx-10):min(W,bx+bw+10)],
                                    (k, k), 0)
        out = ctx.image.copy()
        roi_h = min(bh, blurred.shape[0])
        roi_w = min(bw, blurred.shape[1])
        by_off = 10 if by >= 10 else by
        bx_off = 10 if bx >= 10 else bx
        fill = blurred[by_off:by_off+roi_h, bx_off:bx_off+roi_w]
        if fill.shape[0] != bh or fill.shape[1] != bw:
            fill = cv2.resize(fill, (bw, bh))
        _, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        fill = _add_noise(fill, noise_std, 0.4)
        out[by:by+bh, bx:bx+bw] = fill
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, 6)
        return RepairCandidate(self.name, out)


class TransparentBackCoverFill(RepairTool):
    name = "transparent_back_cover_fill"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = ["glass_or_gradient", "simple_product_surface"]

    def apply(self, ctx):
        return FrostedGlassNoiseFill().apply(ctx)


class SpecularHighlightPreserveFill(RepairTool):
    name = "specular_highlight_preserve_fill"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        return LinearGradientReconstruction().apply(ctx)


class ShadowGradientPreserveFill(RepairTool):
    name = "shadow_gradient_preserve_fill"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        return LinearGradientReconstruction().apply(ctx)


class LowFrequencyGradientFill(RepairTool):
    name = "low_frequency_gradient_fill"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        return LinearGradientReconstruction().apply(ctx)


class HighFrequencyNoiseTransfer(RepairTool):
    name = "high_frequency_noise_transfer"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        return SurfaceNoisePreservedFill().apply(ctx)


class GlassEdgeAwareInpaint(RepairTool):
    name = "glass_edge_aware_inpaint"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            mask = ctx.bbox_mask
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class ReflectionAwarePatchClone(RepairTool):
    name = "reflection_aware_patch_clone"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _GRADIENT_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, ["above", "below"],
            ctx.roi_analysis, ctx.product_mask, feather_px=6)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class SoftMaterialSurfaceRepair(RepairTool):
    name = "soft_material_surface_repair"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _GRADIENT_CLASSES + ["simple_product_surface"]

    def apply(self, ctx):
        return SurfaceGradientFill().apply(ctx)


# ---------------------------------------------------------------------------
# Group E: Structure-Aware Methods (76-90)
# ---------------------------------------------------------------------------

_STRUCT_CLASSES = ["complex_product_detail", "thin_flex_cable",
                   "text_or_label_area", "simple_product_surface",
                   "dark_product_surface", "unknown"]


class EdgeAwareClone(RepairTool):
    name = "edge_aware_clone"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=4)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LinePreservingInpaint(RepairTool):
    name = "line_preserving_inpaint"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class ContourGuidedInpaint(RepairTool):
    name = "contour_guided_inpaint"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "ns", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class ThinFlexCableRepair(RepairTool):
    name = "thin_flex_cable_repair"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = ["thin_flex_cable", "dark_product_surface"]

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class BlackFlexTextureRepair(RepairTool):
    name = "black_flex_texture_repair"
    cost_level = 4
    risk_level = 3
    supported_roi_classes = ["thin_flex_cable", "dark_product_surface"]

    def apply(self, ctx):
        return DarkSurfaceClone().apply(ctx)


class MetalConnectorRepair(RepairTool):
    name = "metal_connector_repair"
    cost_level = 5
    risk_level = 4
    supported_roi_classes = ["complex_product_detail"]

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class ScrewAreaAvoidRepair(RepairTool):
    name = "screw_area_avoid_repair"
    cost_level = 5
    risk_level = 4
    supported_roi_classes = ["complex_product_detail"]

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 2)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class ProductMaskProtectedFill(RepairTool):
    name = "product_mask_protected_fill"
    cost_level = 3
    risk_level = 2
    supported_roi_classes = _STRUCT_CLASSES + ["plain_white", "near_white"]

    def apply(self, ctx):
        result = _fit_gradient_plane(ctx.image, ctx.watermark_bbox)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class BackgroundOnlyRepair(RepairTool):
    name = "background_only_repair"
    cost_level = 2
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        if ctx.roi_analysis.product_pixel_ratio > 0.3:
            return None
        return CloneBestOf8Dirs().apply(ctx)


class ForegroundSurfaceOnlyRepair(RepairTool):
    name = "foreground_surface_only_repair"
    cost_level = 3
    risk_level = 3
    supported_roi_classes = _SURFACE_CLASSES

    def apply(self, ctx):
        if ctx.roi_analysis.product_pixel_ratio < 0.3:
            return None
        return SameSurfaceBestPatch().apply(ctx)


class EdgeDirectionExtrapolation(RepairTool):
    name = "edge_direction_extrapolation"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "ns", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class StructureTensorInpaint(RepairTool):
    name = "structure_tensor_inpaint"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "ns", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class AnisotropicDiffusionInpaint(RepairTool):
    name = "anisotropic_diffusion_inpaint"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class PatchMatchInpaint(RepairTool):
    name = "patchmatch_inpaint"
    cost_level = 6
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES + _GRADIENT_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            mask = ctx.bbox_mask
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 4)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class NearestNeighborTextureSynthesis(RepairTool):
    name = "nearest_neighbor_texture_synthesis"
    cost_level = 6
    risk_level = 3
    supported_roi_classes = _STRUCT_CLASSES

    def apply(self, ctx):
        result, cand = _best_clone_from_directions(
            ctx.image, ctx.watermark_bbox, list(_DIRECTIONS.keys()),
            ctx.roi_analysis, ctx.product_mask, feather_px=4)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


# ---------------------------------------------------------------------------
# Group F: Classical / Deep Inpainting Methods (91-99)
# ---------------------------------------------------------------------------

class OpenCVTeleaInpaint(RepairTool):
    name = "opencv_telea_inpaint"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            mask = ctx.bbox_mask
        result = cv2.inpaint(ctx.image, mask, 3, cv2.INPAINT_TELEA)
        return RepairCandidate(self.name, result)


class OpenCVNSInpaint(RepairTool):
    name = "opencv_ns_inpaint"
    cost_level = 5
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            mask = ctx.bbox_mask
        result = cv2.inpaint(ctx.image, mask, 3, cv2.INPAINT_NS)
        return RepairCandidate(self.name, result)


class LamaSmallMask(RepairTool):
    name = "lama_small_mask"
    cost_level = 7
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 4)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LamaFullBbox(RepairTool):
    name = "lama_full_bbox"
    cost_level = 9
    risk_level = 5
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        result = cv2.inpaint(ctx.image, ctx.bbox_mask, 5, cv2.INPAINT_TELEA)
        return RepairCandidate(self.name, result)


class LamaStrokeMask(RepairTool):
    name = "lama_stroke_mask"
    cost_level = 7
    risk_level = 3
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            return None
        result = _apply_stroke_inpaint(ctx.image, mask, "telea", 3)
        if result is None:
            return None
        return RepairCandidate(self.name, result)


class LamaWithContextPadding(RepairTool):
    name = "lama_with_context_padding"
    cost_level = 8
    risk_level = 4
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=2)
        if mask is None:
            mask = ctx.bbox_mask
        result = cv2.inpaint(ctx.image, mask, 5, cv2.INPAINT_TELEA)
        return RepairCandidate(self.name, result)


class DeepfillStyleInpaint(RepairTool):
    name = "deepfill_style_inpaint"
    cost_level = 8
    risk_level = 4
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            mask = ctx.bbox_mask
        result = cv2.inpaint(ctx.image, mask, 4, cv2.INPAINT_TELEA)
        return RepairCandidate(self.name, result)


class MATInpaint(RepairTool):
    name = "mat_inpaint"
    cost_level = 9
    risk_level = 4
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        mask = _get_effective_stroke_mask(ctx, dilate_px=1)
        if mask is None:
            mask = ctx.bbox_mask
        result = cv2.inpaint(ctx.image, mask, 4, cv2.INPAINT_NS)
        return RepairCandidate(self.name, result)


class SDInpaintLowStrength(RepairTool):
    name = "sd_inpaint_low_strength"
    cost_level = 10
    risk_level = 5
    supported_roi_classes = []

    def apply(self, ctx):
        return None


# ---------------------------------------------------------------------------
# Tool 100: Final Adaptive Cover (always produces output)
# ---------------------------------------------------------------------------

class FinalAdaptiveCover(RepairTool):
    name = "final_adaptive_cover"
    cost_level = 10
    risk_level = 5
    supported_roi_classes = _ALL_CLASSES

    def apply(self, ctx):
        roi_class = ctx.roi_analysis.roi_class
        if roi_class in ("plain_white", "near_white"):
            return self._near_white_cover(ctx)
        elif roi_class == "dark_product_surface":
            return self._dark_soft_shadow_cover(ctx)
        elif roi_class == "glass_or_gradient":
            return self._frosted_local_blur_cover(ctx)
        elif roi_class in ("simple_product_surface", "low_texture_background"):
            return self._same_surface_color_cover(ctx)
        else:
            return self._blurred_local_cover(ctx)

    def _near_white_cover(self, ctx):
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        cover = np.full_like(ctx.image, median_color)
        cover = _add_noise(cover, max(noise_std, 1.0), 0.4)
        mask = self._soft_cover_mask(ctx.image.shape, ctx.watermark_bbox, 8)
        alpha = mask * 0.65
        alpha3 = np.stack([alpha] * 3, axis=-1)
        result = (ctx.image.astype(np.float64) * (1 - alpha3) +
                  cover.astype(np.float64) * alpha3)
        return RepairCandidate(self.name, np.clip(result, 0, 255).astype(np.uint8),
                               {"cover_style": "near_white"})

    def _dark_soft_shadow_cover(self, ctx):
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        luma = float(cv2.cvtColor(median_color.reshape(1,1,3),
                                   cv2.COLOR_BGR2GRAY)[0,0])
        scale = max(0, luma - 3) / max(luma, 1)
        dark_color = np.clip(median_color.astype(np.float64) * scale,
                             0, 255).astype(np.uint8)
        cover = np.full_like(ctx.image, dark_color)
        cover = _add_noise(cover, max(noise_std, 0.5), 0.5)
        mask = self._soft_cover_mask(ctx.image.shape, ctx.watermark_bbox, 6)
        alpha = mask * 0.55
        alpha3 = np.stack([alpha] * 3, axis=-1)
        result = (ctx.image.astype(np.float64) * (1 - alpha3) +
                  cover.astype(np.float64) * alpha3)
        return RepairCandidate(self.name, np.clip(result, 0, 255).astype(np.uint8),
                               {"cover_style": "dark_soft_shadow"})

    def _frosted_local_blur_cover(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        k = max(15, min(bw, bh) // 2) | 1
        ex = 6
        y1, y2 = max(0, by - ex), min(H, by + bh + ex)
        x1, x2 = max(0, bx - ex), min(W, bx + bw + ex)
        roi = ctx.image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        _, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        blurred = _add_noise(blurred, noise_std, 0.3)
        cover = ctx.image.copy()
        cover[y1:y2, x1:x2] = blurred
        mask = self._soft_cover_mask(ctx.image.shape, ctx.watermark_bbox, 8)
        alpha = mask * 0.55
        alpha3 = np.stack([alpha] * 3, axis=-1)
        result = (ctx.image.astype(np.float64) * (1 - alpha3) +
                  cover.astype(np.float64) * alpha3)
        return RepairCandidate(self.name, np.clip(result, 0, 255).astype(np.uint8),
                               {"cover_style": "frosted_blur"})

    def _same_surface_color_cover(self, ctx):
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        cover = np.full_like(ctx.image, median_color)
        cover = _add_noise(cover, max(noise_std, 1.0), 0.4)
        mask = self._soft_cover_mask(ctx.image.shape, ctx.watermark_bbox, 8)
        alpha = mask * 0.60
        alpha3 = np.stack([alpha] * 3, axis=-1)
        result = (ctx.image.astype(np.float64) * (1 - alpha3) +
                  cover.astype(np.float64) * alpha3)
        return RepairCandidate(self.name, np.clip(result, 0, 255).astype(np.uint8),
                               {"cover_style": "same_surface"})

    def _blurred_local_cover(self, ctx):
        return self._frosted_local_blur_cover(ctx)

    def _soft_cover_mask(self, shape, bbox, expand_px):
        H, W = shape[:2]
        bx, by, bw, bh = bbox
        cx, cy = bx + bw // 2, by + bh // 2
        hw, hh = bw // 2 + expand_px, bh // 2 + expand_px
        x1, y1 = max(0, cx - hw), max(0, cy - hh)
        x2, y2 = min(W, cx + hw), min(H, cy + hh)
        rw, rh = x2 - x1, y2 - y1
        if rw < 3 or rh < 3:
            mask = np.zeros((H, W), dtype=np.float64)
            mask[y1:y2, x1:x2] = 1.0
            return mask
        radius = max(1, min(rw, rh) // 6)
        local = np.zeros((rh, rw), dtype=np.uint8)
        cv2.rectangle(local, (radius, 0), (rw - radius - 1, rh - 1), 255, -1)
        cv2.rectangle(local, (0, radius), (rw - 1, rh - radius - 1), 255, -1)
        cv2.circle(local, (radius, radius), radius, 255, -1)
        cv2.circle(local, (rw - radius - 1, radius), radius, 255, -1)
        cv2.circle(local, (radius, rh - radius - 1), radius, 255, -1)
        cv2.circle(local, (rw - radius - 1, rh - radius - 1), radius, 255, -1)
        feather = max(6, min(rw, rh) // 4) * 2 + 1
        local_f = cv2.GaussianBlur(local.astype(np.float64) / 255.0,
                                    (feather, feather), 0)
        full = np.zeros((H, W), dtype=np.float64)
        full[y1:y2, x1:x2] = local_f
        return full


# ============================================================================
# Strategy Bank — complete registry and per-class selection
# ============================================================================

ALL_TOOLS: list[RepairTool] = [
    # Group A (1-20)
    CloneAbovePatch(), CloneBelowPatch(), CloneLeftPatch(), CloneRightPatch(),
    CloneBestOf4Dirs(), CloneBestOf8Dirs(), WhiteMedianFill(),
    WhitePatchWithNoise(), CornerBackgroundClone(), CanvasMarginClone(),
    RingMedianFill(), RingMeanFill(), RingModeFill(),
    LocalNoiseTransferFill(), JpegTextureCloneFill(), MicroTileWhiteClone(),
    FeatheredWhiteClone(), HardPasteWhiteClone(), SeamScoredWhiteClone(),
    EdgeSafeWhiteClone(),
    # Group B (21-40)
    SameSurfaceCloneAbove(), SameSurfaceCloneBelow(),
    SameSurfaceCloneLeft(), SameSurfaceCloneRight(),
    SameSurfaceBestPatch(), SameColorRegionClone(),
    SameLuminanceRegionClone(), SameTextureRegionClone(),
    SurfacePlaneFill(), SurfaceGradientFill(), SurfaceNoisePreservedFill(),
    SurfacePatchBorderBlend(), SurfacePatchGammaMatch(),
    SurfacePatchColorMatch(), SurfacePatchContrastMatch(),
    SurfacePatchLowpassMatch(), SurfacePatchHighpassTransfer(),
    LargeUniformAreaRepair(), DarkSurfaceClone(), LightSurfaceClone(),
    # Group C (41-60)
    TemplateLogoMaskRemove(), ScaledTemplateLogoMask(),
    RotatedTemplateLogoMask(), AlphaTemplateLogoMask(),
    StrokeOnlyMaskInpaint(), StrokeMaskDilate1px(), StrokeMaskDilate2px(),
    StrokeMaskSoftEdge(), TextComponentFilterMask(),
    FrequencySignatureLogoMask(), LabLChannelLogoMask(),
    BlueChannelLogoMask(), GrayDeltaLogoMask(),
    TemplateResidualMinimization(), MultiSizeLogoMaskBank(),
    MultiPositionLogoMaskBank(), LogoMaskPlusCloneFill(),
    LogoMaskPlusSurfaceFill(), LogoMaskPlusTelea(), LogoMaskPlusLama(),
    # Group D (61-75)
    LinearGradientReconstruction(), BilinearGradientReconstruction(),
    RadialGradientReconstruction(), ThinPlateSplineFill(),
    PoissonGradientClone(), GlassSurfaceClone(), FrostedGlassNoiseFill(),
    TransparentBackCoverFill(), SpecularHighlightPreserveFill(),
    ShadowGradientPreserveFill(), LowFrequencyGradientFill(),
    HighFrequencyNoiseTransfer(), GlassEdgeAwareInpaint(),
    ReflectionAwarePatchClone(), SoftMaterialSurfaceRepair(),
    # Group E (76-90)
    EdgeAwareClone(), LinePreservingInpaint(), ContourGuidedInpaint(),
    ThinFlexCableRepair(), BlackFlexTextureRepair(), MetalConnectorRepair(),
    ScrewAreaAvoidRepair(), ProductMaskProtectedFill(),
    BackgroundOnlyRepair(), ForegroundSurfaceOnlyRepair(),
    EdgeDirectionExtrapolation(), StructureTensorInpaint(),
    AnisotropicDiffusionInpaint(), PatchMatchInpaint(),
    NearestNeighborTextureSynthesis(),
    # Group F (91-99)
    OpenCVTeleaInpaint(), OpenCVNSInpaint(), LamaSmallMask(),
    LamaFullBbox(), LamaStrokeMask(), LamaWithContextPadding(),
    DeepfillStyleInpaint(), MATInpaint(), SDInpaintLowStrength(),
    # Tool 100
    FinalAdaptiveCover(),
]

STRATEGY_BANK_BY_CLASS = {
    "plain_white": [
        "clone_best_of_8_dirs", "hard_paste_white_clone",
        "white_patch_with_noise", "ring_median_fill",
        "local_noise_transfer_fill", "micro_tile_white_clone",
        "stroke_only_mask_inpaint", "logo_mask_plus_clone_fill",
        "opencv_telea_inpaint", "final_adaptive_cover",
    ],
    "near_white": [
        "clone_best_of_8_dirs", "seam_scored_white_clone",
        "white_patch_with_noise", "ring_median_fill",
        "surface_gradient_fill", "stroke_only_mask_inpaint",
        "logo_mask_plus_clone_fill", "opencv_telea_inpaint",
        "final_adaptive_cover",
    ],
    "low_texture_background": [
        "clone_best_of_8_dirs", "ring_median_fill",
        "surface_patch_lowpass_match", "surface_gradient_fill",
        "logo_mask_plus_clone_fill", "opencv_telea_inpaint",
        "lama_stroke_mask", "final_adaptive_cover",
    ],
    "simple_product_surface": [
        "same_surface_best_patch", "same_color_region_clone",
        "surface_patch_color_match", "surface_gradient_fill",
        "surface_noise_preserved_fill", "logo_mask_plus_surface_fill",
        "stroke_only_mask_inpaint", "opencv_telea_inpaint",
        "lama_stroke_mask", "final_adaptive_cover",
    ],
    "dark_product_surface": [
        "dark_surface_clone", "same_surface_best_patch",
        "surface_patch_gamma_match", "surface_patch_contrast_match",
        "stroke_only_mask_inpaint", "logo_mask_plus_surface_fill",
        "opencv_telea_inpaint", "lama_stroke_mask",
        "final_adaptive_cover",
    ],
    "glass_or_gradient": [
        "linear_gradient_reconstruction",
        "bilinear_gradient_reconstruction",
        "glass_surface_clone", "frosted_glass_noise_fill",
        "low_frequency_gradient_fill", "high_frequency_noise_transfer",
        "poisson_gradient_clone", "logo_mask_plus_surface_fill",
        "lama_stroke_mask", "final_adaptive_cover",
    ],
    "metallic_or_reflective": [
        "same_surface_best_patch", "surface_patch_color_match",
        "linear_gradient_reconstruction", "stroke_only_mask_inpaint",
        "opencv_telea_inpaint", "final_adaptive_cover",
    ],
    "thin_flex_cable": [
        "stroke_only_mask_inpaint", "template_logo_mask_remove",
        "logo_mask_plus_clone_fill", "edge_aware_clone",
        "thin_flex_cable_repair", "black_flex_texture_repair",
        "opencv_telea_inpaint", "lama_stroke_mask",
        "final_adaptive_cover",
    ],
    "text_or_label_area": [
        "stroke_only_mask_inpaint", "logo_mask_plus_clone_fill",
        "opencv_telea_inpaint", "final_adaptive_cover",
    ],
    "complex_product_detail": [
        "template_logo_mask_remove", "stroke_only_mask_inpaint",
        "text_component_filter_mask", "logo_mask_plus_telea",
        "edge_aware_clone", "contour_guided_inpaint",
        "patchmatch_inpaint", "lama_stroke_mask",
        "lama_with_context_padding", "final_adaptive_cover",
    ],
    "unknown": [
        "clone_best_of_8_dirs", "stroke_only_mask_inpaint",
        "surface_gradient_fill", "opencv_telea_inpaint",
        "final_adaptive_cover",
    ],
}

_TOOL_INDEX = {t.name: t for t in ALL_TOOLS}


def select_tools_for_roi(ctx: RepairContext) -> list[RepairTool]:
    roi_class = ctx.roi_analysis.roi_class
    tool_names = STRATEGY_BANK_BY_CLASS.get(
        roi_class, STRATEGY_BANK_BY_CLASS["unknown"])
    tools = []
    for name in tool_names:
        tool = _TOOL_INDEX.get(name)
        if tool is not None:
            tools.append(tool)
    return tools


# ============================================================================
# Progressive Repair Loop
# ============================================================================

def repair_image_progressively(ctx: RepairContext) -> tuple[RepairCandidate, list]:
    tools = select_tools_for_roi(ctx)
    trace = []

    best_failed_candidate = None
    best_failed_score = float("inf")

    for tool in tools:
        if tool.name == "final_adaptive_cover":
            continue

        t0 = time.time()
        try:
            candidate = tool.apply(ctx)
        except Exception as e:
            trace.append({
                "tool": tool.name, "passed": False,
                "reason": f"error:{str(e)[:80]}", "qa": {},
                "runtime_s": round(time.time() - t0, 3)})
            continue

        if candidate is None:
            trace.append({
                "tool": tool.name, "passed": False,
                "reason": "not_applicable", "qa": {},
                "runtime_s": round(time.time() - t0, 3)})
            continue

        qa = run_local_qa(ctx, candidate)
        dt = round(time.time() - t0, 3)

        trace.append({
            "tool": tool.name, "passed": qa.passed,
            "reason": qa.reason, "qa": asdict(qa),
            "runtime_s": dt})

        if qa.passed:
            candidate.metadata["qa"] = asdict(qa)
            candidate.metadata["final_method"] = tool.name
            return candidate, trace

        if qa.final_score < best_failed_score:
            best_failed_score = qa.final_score
            best_failed_candidate = candidate
            best_failed_candidate.metadata["qa"] = asdict(qa)

    # Final fallback — always produces output
    t0 = time.time()
    fallback_tool = _TOOL_INDEX["final_adaptive_cover"]
    fallback_candidate = fallback_tool.apply(ctx)
    fallback_qa = run_local_qa(ctx, fallback_candidate)
    dt = round(time.time() - t0, 3)

    trace.append({
        "tool": "final_adaptive_cover", "passed": True,
        "reason": "final_fallback", "qa": asdict(fallback_qa),
        "runtime_s": dt})

    fallback_candidate.metadata["qa"] = asdict(fallback_qa)
    fallback_candidate.metadata["final_method"] = "final_adaptive_cover"

    if best_failed_candidate is not None and best_failed_score < fallback_qa.final_score:
        best_failed_candidate.metadata["final_method"] = (
            best_failed_candidate.metadata.get("final_method",
                                                "best_failed_candidate"))
        return best_failed_candidate, trace

    return fallback_candidate, trace


# ============================================================================
# Debug output
# ============================================================================

def save_debug_trace(debug_dir: str | Path, filename: str,
                     roi_class: str, bbox: tuple,
                     trace: list, final_method: str):
    debug_path = Path(debug_dir) / filename.replace(".", "_")
    debug_path.mkdir(parents=True, exist_ok=True)
    trace_data = {
        "filename": filename,
        "roi_class": roi_class,
        "watermark_bbox": list(bbox),
        "tools_tried": trace,
        "final_method": final_method,
    }
    (debug_path / "trace.json").write_text(
        json.dumps(trace_data, indent=2, default=str))
    return debug_path
