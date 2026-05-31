"""
Progressive Watermark Repair Strategy Bank — V9 Quality Patch.

V9 addresses three root issues from V8 benchmarking:
  - final_adaptive_cover dominated (37/50) — now gated by product-overlap guard
  - Status semantics were misleading — method_family now drives status mapping
  - QA missed visible rectangle artifacts — rectangular_patch_visibility gate added

Pipeline:
  1. Classify watermark ROI (expanded to 13 classes)
  2. Select cheapest/safest repair tools from strategy bank
  3. Validate mask confidence before use
  4. Run candidates with local QA (8 metrics including rect visibility)
  5. Product-overlap guard blocks full-bbox cover on product pixels
  6. Final adaptive cover as last resort (class-specific, smallest mask)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Method Family — drives truthful status mapping (P0)
# ---------------------------------------------------------------------------

class MethodFamily(Enum):
    REAL_PIXEL_CLONE = "real_pixel_clone"
    STATISTICAL_FILL = "statistical_fill"
    GRADIENT_RECONSTRUCTION = "gradient_reconstruction"
    STROKE_LOGO_REPAIR = "stroke_logo_repair"
    CLASSICAL_INPAINT = "classical_inpaint"
    DEEP_INPAINT = "deep_inpaint"
    ADAPTIVE_COVER = "adaptive_cover"


_TOOL_FAMILY_MAP = {
    "clone_above_patch": MethodFamily.REAL_PIXEL_CLONE,
    "clone_below_patch": MethodFamily.REAL_PIXEL_CLONE,
    "clone_left_patch": MethodFamily.REAL_PIXEL_CLONE,
    "clone_right_patch": MethodFamily.REAL_PIXEL_CLONE,
    "clone_best_of_4_dirs": MethodFamily.REAL_PIXEL_CLONE,
    "clone_best_of_8_dirs": MethodFamily.REAL_PIXEL_CLONE,
    "white_median_fill": MethodFamily.STATISTICAL_FILL,
    "plain_white_fast_path": MethodFamily.STATISTICAL_FILL,
    "white_patch_with_noise": MethodFamily.STATISTICAL_FILL,
    "corner_background_clone": MethodFamily.REAL_PIXEL_CLONE,
    "canvas_margin_clone": MethodFamily.REAL_PIXEL_CLONE,
    "ring_median_fill": MethodFamily.STATISTICAL_FILL,
    "ring_mean_fill": MethodFamily.STATISTICAL_FILL,
    "ring_mode_fill": MethodFamily.STATISTICAL_FILL,
    "local_noise_transfer_fill": MethodFamily.STATISTICAL_FILL,
    "jpeg_texture_clone_fill": MethodFamily.REAL_PIXEL_CLONE,
    "micro_tile_white_clone": MethodFamily.REAL_PIXEL_CLONE,
    "feathered_white_clone": MethodFamily.REAL_PIXEL_CLONE,
    "hard_paste_white_clone": MethodFamily.REAL_PIXEL_CLONE,
    "seam_scored_white_clone": MethodFamily.REAL_PIXEL_CLONE,
    "edge_safe_white_clone": MethodFamily.REAL_PIXEL_CLONE,
    "plain_white_direct_neighbor_clone_v2": MethodFamily.REAL_PIXEL_CLONE,
    "same_surface_clone_above": MethodFamily.REAL_PIXEL_CLONE,
    "same_surface_clone_below": MethodFamily.REAL_PIXEL_CLONE,
    "same_surface_clone_left": MethodFamily.REAL_PIXEL_CLONE,
    "same_surface_clone_right": MethodFamily.REAL_PIXEL_CLONE,
    "same_surface_best_patch": MethodFamily.REAL_PIXEL_CLONE,
    "same_color_region_clone": MethodFamily.REAL_PIXEL_CLONE,
    "same_luminance_region_clone": MethodFamily.REAL_PIXEL_CLONE,
    "same_texture_region_clone": MethodFamily.REAL_PIXEL_CLONE,
    "surface_plane_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "surface_gradient_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "surface_noise_preserved_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "surface_patch_border_blend": MethodFamily.REAL_PIXEL_CLONE,
    "surface_patch_gamma_match": MethodFamily.REAL_PIXEL_CLONE,
    "surface_patch_color_match": MethodFamily.REAL_PIXEL_CLONE,
    "surface_patch_contrast_match": MethodFamily.REAL_PIXEL_CLONE,
    "surface_patch_lowpass_match": MethodFamily.GRADIENT_RECONSTRUCTION,
    "surface_patch_highpass_transfer": MethodFamily.GRADIENT_RECONSTRUCTION,
    "large_uniform_area_repair": MethodFamily.GRADIENT_RECONSTRUCTION,
    "dark_surface_clone": MethodFamily.REAL_PIXEL_CLONE,
    "light_surface_clone": MethodFamily.REAL_PIXEL_CLONE,
    "template_logo_mask_remove": MethodFamily.STROKE_LOGO_REPAIR,
    "scaled_template_logo_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "rotated_template_logo_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "alpha_template_logo_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "stroke_only_mask_inpaint": MethodFamily.STROKE_LOGO_REPAIR,
    "stroke_mask_dilate_1px": MethodFamily.STROKE_LOGO_REPAIR,
    "stroke_mask_dilate_2px": MethodFamily.STROKE_LOGO_REPAIR,
    "stroke_mask_soft_edge": MethodFamily.STROKE_LOGO_REPAIR,
    "text_component_filter_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "frequency_signature_logo_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "lab_l_channel_logo_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "blue_channel_logo_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "gray_delta_logo_mask": MethodFamily.STROKE_LOGO_REPAIR,
    "template_residual_minimization": MethodFamily.STROKE_LOGO_REPAIR,
    "multi_size_logo_mask_bank": MethodFamily.STROKE_LOGO_REPAIR,
    "multi_position_logo_mask_bank": MethodFamily.STROKE_LOGO_REPAIR,
    "logo_mask_plus_clone_fill": MethodFamily.STROKE_LOGO_REPAIR,
    "logo_mask_plus_surface_fill": MethodFamily.STROKE_LOGO_REPAIR,
    "logo_mask_plus_telea": MethodFamily.STROKE_LOGO_REPAIR,
    "logo_mask_plus_lama": MethodFamily.DEEP_INPAINT,
    "linear_gradient_reconstruction": MethodFamily.GRADIENT_RECONSTRUCTION,
    "bilinear_gradient_reconstruction": MethodFamily.GRADIENT_RECONSTRUCTION,
    "radial_gradient_reconstruction": MethodFamily.GRADIENT_RECONSTRUCTION,
    "thin_plate_spline_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "poisson_gradient_clone": MethodFamily.GRADIENT_RECONSTRUCTION,
    "glass_surface_clone": MethodFamily.REAL_PIXEL_CLONE,
    "frosted_glass_noise_fill": MethodFamily.STATISTICAL_FILL,
    "transparent_back_cover_fill": MethodFamily.STATISTICAL_FILL,
    "specular_highlight_preserve_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "shadow_gradient_preserve_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "low_frequency_gradient_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "high_frequency_noise_transfer": MethodFamily.GRADIENT_RECONSTRUCTION,
    "glass_edge_aware_inpaint": MethodFamily.CLASSICAL_INPAINT,
    "reflection_aware_patch_clone": MethodFamily.REAL_PIXEL_CLONE,
    "soft_material_surface_repair": MethodFamily.GRADIENT_RECONSTRUCTION,
    "edge_aware_clone": MethodFamily.REAL_PIXEL_CLONE,
    "line_preserving_inpaint": MethodFamily.STROKE_LOGO_REPAIR,
    "contour_guided_inpaint": MethodFamily.CLASSICAL_INPAINT,
    "thin_flex_cable_repair": MethodFamily.STROKE_LOGO_REPAIR,
    "black_flex_texture_repair": MethodFamily.REAL_PIXEL_CLONE,
    "metal_connector_repair": MethodFamily.STROKE_LOGO_REPAIR,
    "screw_area_avoid_repair": MethodFamily.STROKE_LOGO_REPAIR,
    "product_mask_protected_fill": MethodFamily.GRADIENT_RECONSTRUCTION,
    "background_only_repair": MethodFamily.REAL_PIXEL_CLONE,
    "foreground_surface_only_repair": MethodFamily.REAL_PIXEL_CLONE,
    "edge_direction_extrapolation": MethodFamily.CLASSICAL_INPAINT,
    "structure_tensor_inpaint": MethodFamily.CLASSICAL_INPAINT,
    "anisotropic_diffusion_inpaint": MethodFamily.CLASSICAL_INPAINT,
    "patchmatch_inpaint": MethodFamily.CLASSICAL_INPAINT,
    "nearest_neighbor_texture_synthesis": MethodFamily.REAL_PIXEL_CLONE,
    "segmented_background_product_repair": MethodFamily.STROKE_LOGO_REPAIR,
    "opencv_telea_inpaint": MethodFamily.CLASSICAL_INPAINT,
    "opencv_ns_inpaint": MethodFamily.CLASSICAL_INPAINT,
    "lama_small_mask": MethodFamily.DEEP_INPAINT,
    "lama_full_bbox": MethodFamily.DEEP_INPAINT,
    "lama_stroke_mask": MethodFamily.DEEP_INPAINT,
    "lama_with_context_padding": MethodFamily.DEEP_INPAINT,
    "deepfill_style_inpaint": MethodFamily.DEEP_INPAINT,
    "mat_inpaint": MethodFamily.DEEP_INPAINT,
    "sd_inpaint_low_strength": MethodFamily.DEEP_INPAINT,
    "final_adaptive_cover": MethodFamily.ADAPTIVE_COVER,
}


def get_method_family(tool_name: str) -> MethodFamily:
    return _TOOL_FAMILY_MAP.get(tool_name, MethodFamily.CLASSICAL_INPAINT)


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
    rectangular_patch_visibility: float
    final_score: float
    reason: str
    # V10 additions (defaults keep backward compatibility)
    metrics_valid: bool = True
    residual_pass: bool = True
    residual_template_corr: float = 0.0
    residual_text_component: float = 0.0
    residual_improvement: float = 1.0
    cover_rectangularity: float = 0.0
    cover_luma_delta: float = 0.0
    # V11 additions
    ssim_score: float = 1.0
    boundary_jump: float = 0.0
    metrics_invalid_reasons: list = field(default_factory=list)
    residual_dot_chain_score: float = 0.0
    residual_component_area_ratio: float = 0.0
    residual_component_horizontal_span: float = 0.0
    residual_component_count: int = 0
    product_overlap: float = 0.0
    product_color_delta_lab: float = 0.0
    product_new_bright_blob_score: float = 0.0
    product_edge_retention: float = 1.0
    product_gate_pass: bool = True
    product_gate_reason: str = ""
    changed_area_ratio: float = 0.0


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
                  "text_or_label_area", "mixed_background_product"}
LOOSE_CLASSES = {"plain_white", "near_white", "low_texture_background"}

PLAIN_BG_CLASSES = {"plain_white", "near_white", "low_texture_background"}


# ---------------------------------------------------------------------------
# V11 Phase 1 — mandatory QA-metric validation (fail closed)
# ---------------------------------------------------------------------------

REQUIRED_QA_METRICS = [
    "ssim",
    "color_delta",
    "boundary_jump",
    "seam_delta",
    "cover_visibility",
    "product_damage",
    "residual_template_corr",
    "residual_text_component",
    "residual_improvement",
]

# The subset whose simultaneous all-zero state is the tell-tale sign that the
# QA stage never actually executed (a degenerate / empty ROI).
_QA_CORE_METRICS = [
    "ssim", "color_delta", "boundary_jump", "seam_delta",
    "cover_visibility", "product_damage",
]


def validate_qa_metrics(qa: dict) -> tuple[bool, list]:
    """V11 Phase 1. A candidate may only pass when every required metric was
    actually computed. Missing / null / NaN metrics fail closed, and an
    all-zero core set is treated as 'metrics were never computed'."""
    reasons = []

    for key in REQUIRED_QA_METRICS:
        if key not in qa:
            reasons.append(f"missing_required_qa_metric:{key}")
        elif qa[key] is None:
            reasons.append(f"null_required_qa_metric:{key}")
        elif isinstance(qa[key], float) and math.isnan(qa[key]):
            reasons.append(f"nan_required_qa_metric:{key}")

    core_values = [qa.get(k) for k in _QA_CORE_METRICS]
    numeric_core = [v for v in core_values
                    if isinstance(v, (int, float)) and not
                    (isinstance(v, float) and math.isnan(v))]

    if len(numeric_core) == 0:
        reasons.append("qa_metrics_not_computed")

    if len(numeric_core) > 0 and all(abs(v) < 1e-9 for v in numeric_core):
        reasons.append("qa_metrics_probably_not_computed")

    return len(reasons) == 0, reasons


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
    bg_ratio = 1.0 - product_ratio

    if white_ratio > 0.85 and edge_d < 0.03:
        return "plain_white"
    if brightness > 220 and tex_var < 80 and edge_d < 0.05:
        return "near_white"

    if product_ratio > 0.15 and bg_ratio > 0.25 and white_ratio > 0.30:
        return "mixed_background_product"

    if tex_var < 120 and edge_d < 0.06 and product_ratio < 0.10:
        return "low_texture_background"

    if has_lines and dark_ratio > 0.25:
        return "thin_flex_cable"

    if dark_ratio > 0.45:
        return "dark_product_surface"

    if edge_d > 0.18:
        return "complex_product_detail"

    if grad_str > 8.0 and edge_d < 0.08:
        return "glass_or_gradient"

    if grad_str > 4.0 and saturation > 10 and edge_d < 0.12:
        return "transparent_or_glossy"

    if has_text and edge_d > 0.10:
        return "text_or_label_area"

    if dark_ratio > 0.3:
        return "dark_product_surface"

    if product_ratio > 0.15 and edge_d > 0.04:
        return "metallic_or_reflective"

    if saturation > 15 and edge_d > 0.04:
        return "simple_product_surface"

    if product_ratio > 0.3:
        return "simple_product_surface"

    if tex_var < 200 and edge_d < 0.08:
        return "low_texture_background"

    if grad_str > 3.0:
        return "glass_or_gradient"

    return "metallic_or_reflective"


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


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


# ===========================================================================
# V10 — Truthful QA, residual watermark gate, cover rectangularity,
#       alpha-halo masks. See V10 patch plan.
# ===========================================================================

# Patch 1 — fail-closed on missing/uncomputable metrics.
class MissingQAMetric(Exception):
    pass


def _safe_metric(value, name):
    if value is None:
        raise MissingQAMetric(name)
    if isinstance(value, float) and math.isnan(value):
        raise MissingQAMetric(name)
    return value


# Patch 2 — residual watermark detection thresholds.
RESIDUAL_TEMPLATE_CORR_MAX = 0.18
RESIDUAL_OCR_CONF_MAX = 0.12
RESIDUAL_TEXT_COMPONENT_MAX = 0.16
RESIDUAL_IMPROVEMENT_MIN = 0.75

# Patch 4 — cover rectangularity gate thresholds (fail if exceeded).
COVER_RECTANGULARITY_MAX = 0.25
COVER_LUMA_DELTA_MAX = 8.0
COVER_TEXTURE_DROP_MAX = 0.55
COVER_EDGE_BOX_MAX = 0.20


@dataclass
class ResidualResult:
    template_corr: float
    ocr_conf: float
    text_component_score: float
    gray_delta_stroke_score: float
    improvement_ratio: float
    passed: bool
    # V11 Phase 2 — broken-glyph / dotted residual detection
    dot_chain_score: float = 0.0
    component_count: int = 0
    component_area_ratio: float = 0.0
    component_horizontal_span: float = 0.0
    failure_reason: str = ""


# V11 Phase 2 — residual component thresholds (fail if exceeded).
RESIDUAL_DOT_CHAIN_MAX = 0.35
RESIDUAL_COMPONENT_AREA_MAX = 0.08
RESIDUAL_COMPONENT_SPAN_MAX = 0.25


def detect_residual_components(clean_gray, sel_mask, base_mean, base_std):
    """V11 Phase 2 — detect leftover broken-glyph / dotted residuals inside
    the (horizontally expanded) watermark footprint. Catches the case where
    the full word shape is gone but a row of gray dots / glyph fragments
    remains, which template correlation alone misses.

    Returns (dot_chain_score, component_count, area_ratio, horizontal_span).
    """
    if clean_gray is None or sel_mask is None or clean_gray.size == 0:
        return 0.0, 0, 0.0, 0.0
    sel = sel_mask > 0
    n_sel = int(sel.sum())
    if n_sel < 12:
        return 0.0, 0, 0.0, 0.0

    cg = clean_gray.astype(np.float32)
    hp = np.abs(cg - cv2.GaussianBlur(cg, (0, 0), sigmaX=2.0))
    # Only pixels meaningfully above the surrounding surface baseline.
    thr = max(8.0, base_mean + 2.0 * base_std)
    fg = ((hp > thr) & sel).astype(np.uint8) * 255
    if int(fg.sum()) == 0:
        return 0.0, 0, 0.0, 0.0

    nlab, _, stats, cents = cv2.connectedComponentsWithStats(fg)
    h_roi, w_roi = clean_gray.shape[:2]
    glyph_like = []
    total_area = 0
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w_ = int(stats[i, cv2.CC_STAT_WIDTH])
        h_ = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < 2:
            continue
        # glyph fragment: small, not a full-ROI blob
        if a <= 0.20 * w_roi * h_roi and h_ <= max(6, h_roi):
            glyph_like.append((float(cents[i][0]), float(cents[i][1]),
                               a, w_))
            total_area += a

    comp_count = len(glyph_like)
    area_ratio = float(total_area) / max(1, n_sel)
    if comp_count == 0:
        return 0.0, 0, area_ratio, 0.0

    xs = sorted(c[0] for c in glyph_like)
    ys = [c[1] for c in glyph_like]
    h_span = (xs[-1] - xs[0]) / max(1.0, w_roi)

    # Horizontal chain: several similar-y fragments spread across the band.
    dot_chain = 0.0
    if comp_count >= 3:
        y_med = float(np.median(ys))
        y_tol = max(2.0, 0.30 * h_roi)
        aligned = sum(1 for y in ys if abs(y - y_med) <= y_tol)
        align_frac = aligned / comp_count
        count_term = min(1.0, comp_count / 6.0)
        span_term = min(1.0, h_span / 0.5)
        dot_chain = _clamp01(0.45 * align_frac + 0.30 * span_term +
                             0.25 * count_term)
    return float(dot_chain), comp_count, float(area_ratio), float(h_span)


def cleanup_residual_components_with_ring_fill(cleaned, bbox, comp_mask,
                                               ring_bgr_median):
    """V11 Phase 4 — precise second-pass cleanup of residual dot/glyph
    fragments on plain backgrounds: paint just the residual components with
    the ring background colour + a light feather. NOT a full-bbox re-inpaint.
    """
    if comp_mask is None or not np.any(comp_mask > 0):
        return cleaned
    out = cleaned.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.dilate((comp_mask > 0).astype(np.uint8), k, iterations=1)
    sel = m > 0
    out[sel] = ring_bgr_median
    blurred = cv2.GaussianBlur(out, (3, 3), 0)
    edge = cv2.dilate(m, k, iterations=1) - m
    esel = edge > 0
    out[esel] = blurred[esel]
    return out


_CANONICAL_TEMPLATE = None
_CANONICAL_TEMPLATE_LOADED = False


def _get_canonical_template():
    """Canonical sunsky-online.com watermark text template (grayscale)."""
    global _CANONICAL_TEMPLATE, _CANONICAL_TEMPLATE_LOADED
    if not _CANONICAL_TEMPLATE_LOADED:
        _CANONICAL_TEMPLATE_LOADED = True
        try:
            p = Path(__file__).resolve().parent / "watermark-template.png"
            _CANONICAL_TEMPLATE = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        except Exception:
            _CANONICAL_TEMPLATE = None
    return _CANONICAL_TEMPLATE


def _hp_norm(gray):
    """High-pass + zero-mean normalize for structure-only matching."""
    g = gray.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), sigmaX=2.0)
    s = float(hp.std())
    if s < 1e-6:
        return np.zeros_like(hp)
    return (hp - float(hp.mean())) / s


def _highpass_energy(gray):
    if gray is None or gray.size == 0:
        return 0.0
    g = gray.astype(np.float32)
    return float(np.abs(g - cv2.GaussianBlur(g, (0, 0), sigmaX=2.0)).mean())


def _residual_template_corr(roi_gray):
    """Best normalized correlation of the canonical text template against
    the ROI's high-pass structure, scanned across plausible scales."""
    tmpl = _get_canonical_template()
    if tmpl is None or roi_gray is None or roi_gray.size == 0:
        return 0.0
    h_roi, w_roi = roi_gray.shape[:2]
    if h_roi < 8 or w_roi < 16:
        return 0.0
    roi_hp = _hp_norm(roi_gray)
    best = 0.0
    th0, tw0 = tmpl.shape[:2]
    for frac in (1.0, 0.7, 0.5):
        tw = int(w_roi * frac)
        if tw < 16:
            continue
        th = max(6, int(th0 * tw / tw0))
        if tw > w_roi or th > h_roi:
            continue
        t_hp = _hp_norm(cv2.resize(tmpl, (tw, th)))
        try:
            res = cv2.matchTemplate(roi_hp, t_hp, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        best = max(best, float(res.max()))
    return _clamp01(best)


def _text_component_score(roi_gray):
    """Fraction-normalized count of text-like high-pass components."""
    if roi_gray is None or roi_gray.size == 0:
        return 0.0
    g = roi_gray.astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), sigmaX=2.0))
    if float(hp.max()) < 1e-6:
        return 0.0
    thr = max(6.0, float(hp.mean()) + 1.5 * float(hp.std()))
    bw = (hp >= thr).astype(np.uint8) * 255
    if int(bw.sum()) == 0:
        return 0.0
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    h_roi, w_roi = roi_gray.shape[:2]
    text_like = 0
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w_ = int(stats[i, cv2.CC_STAT_WIDTH])
        h_ = int(stats[i, cv2.CC_STAT_HEIGHT])
        if (3 <= a <= 0.25 * w_roi * h_roi and
                2 <= h_ <= max(6, h_roi) and 1 <= w_ <= w_roi):
            text_like += 1
    return _clamp01(text_like / 18.0)


def detect_residual_watermark(original_roi, cleaned_roi, ring_gray=None,
                              mask_roi=None):
    """Patch 2 — verify the watermark is no longer recognizable in the
    cleaned ROI. A smooth-but-readable watermark is still a failure.

    The decisive measurement is taken at the KNOWN watermark locations
    (mask_roi = the stroke/logo mask cropped to the bbox and dilated so it
    also covers the semi-transparent alpha halo). Measuring there instead of
    over the whole bbox avoids the dilution that makes a global high-pass
    ratio useless on textured/metallic surfaces (where the watermark is a
    small fraction of total ROI energy). After a genuine removal the former
    watermark pixels relax to the SURROUNDING surface's natural texture
    level (ring_gray baseline); leftover broken-but-readable glyphs — even
    an untouched halo — still stand above that baseline and are rejected.
    """
    if (original_roi is None or cleaned_roi is None or
            original_roi.size == 0 or cleaned_roi.size == 0):
        return ResidualResult(1.0, 1.0, 1.0, 1.0, 0.0, False)

    orig_gray = (cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)
                 if original_roi.ndim == 3 else original_roi)
    clean_gray = (cv2.cvtColor(cleaned_roi, cv2.COLOR_BGR2GRAY)
                  if cleaned_roi.ndim == 3 else cleaned_roi)

    og = orig_gray.astype(np.float32)
    cg = clean_gray.astype(np.float32)

    clean_corr = _residual_template_corr(clean_gray)
    orig_corr = _residual_template_corr(orig_gray)

    orig_hp = np.abs(og - cv2.GaussianBlur(og, (0, 0), sigmaX=2.0))
    clean_hp = np.abs(cg - cv2.GaussianBlur(cg, (0, 0), sigmaX=2.0))

    # Surface texture baseline from the surrounding ring (watermark-free).
    if ring_gray is not None and ring_gray.ndim == 2 and ring_gray.size > 9:
        rg = ring_gray.astype(np.float32)
        ring_hp = np.abs(rg - cv2.GaussianBlur(rg, (0, 0), sigmaX=2.0))
        base_mean = float(ring_hp.mean())
        base_std = float(ring_hp.std())
    else:
        base_mean = float(np.median(clean_hp))
        base_std = float(clean_hp.std())

    # Decide where to measure: at the watermark mask if we have one, else the
    # whole ROI.
    use_mask = (mask_roi is not None and mask_roi.shape == og.shape and
                int((mask_roi > 0).sum()) >= 8)
    if use_mask:
        sel = mask_roi > 0
        oe = float(orig_hp[sel].mean())
        ce = float(clean_hp[sel].mean())
        stroke_thr = max(8.0, base_mean + 2.0 * base_std)
        residual_stroke = (clean_hp > stroke_thr) & sel
        text_component = _clamp01(
            float(residual_stroke.sum()) / max(1, int(sel.sum())) / 0.20)
    else:
        oe = float(orig_hp.mean())
        ce = float(clean_hp.mean())
        stroke_thr = max(8.0, base_mean + 2.5 * base_std)
        residual_stroke = clean_hp > stroke_thr
        text_component = _clamp01(float(residual_stroke.mean()) / 0.06)

    improvement = 1.0 if oe < 1e-3 else _clamp01(1.0 - ce / oe)
    gray_delta_stroke = min(1.0, ce / 8.0)
    ocr_conf = 0.0  # OCR not available in this environment; treated as pass.

    # V11 Phase 2 — broken-glyph / dotted residual detection inside the
    # measured footprint, relative to the surrounding surface baseline.
    if use_mask:
        comp_sel = (mask_roi > 0).astype(np.uint8) * 255
    else:
        comp_sel = np.ones_like(clean_gray, np.uint8) * 255
    dot_chain, comp_count, comp_area, comp_span = detect_residual_components(
        clean_gray, comp_sel, base_mean, base_std)

    # Waive the improvement ratio only when the original carried essentially
    # NO watermark structure at the measured locations (nothing to remove).
    nothing_to_remove = (orig_corr <= 0.10 and oe < 4.0)
    improvement_ok = (improvement >= RESIDUAL_IMPROVEMENT_MIN or
                      nothing_to_remove)

    failure_reason = ""
    if clean_corr > RESIDUAL_TEMPLATE_CORR_MAX:
        failure_reason = "residual_template_corr_too_high"
    elif text_component > RESIDUAL_TEXT_COMPONENT_MAX:
        failure_reason = "residual_text_component_too_high"
    elif dot_chain > RESIDUAL_DOT_CHAIN_MAX:
        failure_reason = "residual_dot_chain_visible"
    elif (comp_area > RESIDUAL_COMPONENT_AREA_MAX and
          comp_span > RESIDUAL_COMPONENT_SPAN_MAX):
        failure_reason = "broken_glyph_residual_visible"
    elif not improvement_ok:
        failure_reason = "insufficient_watermark_improvement"

    passed = (failure_reason == "" and ocr_conf <= RESIDUAL_OCR_CONF_MAX)

    return ResidualResult(
        template_corr=round(clean_corr, 4), ocr_conf=round(ocr_conf, 4),
        text_component_score=round(text_component, 4),
        gray_delta_stroke_score=round(gray_delta_stroke, 4),
        improvement_ratio=round(improvement, 4), passed=passed,
        dot_chain_score=round(dot_chain, 4), component_count=comp_count,
        component_area_ratio=round(comp_area, 4),
        component_horizontal_span=round(comp_span, 4),
        failure_reason=failure_reason)


# V11 Phase 3 — product-overlap safety thresholds.
PRODUCT_OVERLAP_GATE_MIN = 0.15
PRODUCT_COLOR_DELTA_MAX = 10.0
PRODUCT_NEW_BRIGHT_BLOB_MAX = 0.12
PRODUCT_EDGE_RETENTION_MIN = 0.85
CHANGED_AREA_RATIO_MAX = 2.5


def compute_product_damage_metrics(original, repaired, bbox, product_mask):
    """V11 Phase 3/8 — when the watermark footprint overlaps the product, the
    repair must not damage the product surface (no bright/dark blob on a dark
    part, no large colour/texture/edge change). Measured inside
    product_mask ∩ changed_region.

    Returns a dict of metrics + product_overlap + changed_area_ratio.
    """
    bx, by, bw, bh = bbox
    H, W = original.shape[:2]
    out = {
        "product_overlap": 0.0,
        "product_color_delta_lab": 0.0,
        "product_new_bright_blob_score": 0.0,
        "product_edge_retention": 1.0,
        "changed_area_ratio": 0.0,
    }
    if bw <= 1 or bh <= 1:
        return out

    o = original[by:by + bh, bx:bx + bw]
    r = repaired[by:by + bh, bx:bx + bw]
    if o.shape != r.shape or o.size == 0:
        return out

    og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rg = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(og - rg)
    changed = diff > 6.0
    wm_px = max(1, bw * bh)

    # changed_area_ratio is reported relative to the watermark footprint, but
    # measured over a PADDED window around the bbox so a repair that spills
    # well past the watermark (the symptom this gate guards against) actually
    # registers a ratio > 1. The footprint (bbox) is a safe proxy for the
    # stroke-mask pixel count, which is a subset of the bbox.
    pad = max(8, max(bw, bh) // 2)
    wy1, wy2 = max(0, by - pad), min(H, by + bh + pad)
    wx1, wx2 = max(0, bx - pad), min(W, bx + bw + pad)
    ow = cv2.cvtColor(original[wy1:wy2, wx1:wx2],
                      cv2.COLOR_BGR2GRAY).astype(np.float32)
    rw = cv2.cvtColor(repaired[wy1:wy2, wx1:wx2],
                      cv2.COLOR_BGR2GRAY).astype(np.float32)
    n_changed = int((np.abs(ow - rw) > 6.0).sum())

    if product_mask is not None and product_mask.shape[:2] == original.shape[:2]:
        prod = product_mask[by:by + bh, bx:bx + bw] > 0
        out["product_overlap"] = float(prod.mean())
        region = prod & changed
        n_region = int(region.sum())
        if n_region >= 8:
            o_lab = cv2.cvtColor(o, cv2.COLOR_BGR2LAB).astype(np.float32)
            r_lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB).astype(np.float32)
            d_lab = np.linalg.norm(o_lab - r_lab, axis=2)
            out["product_color_delta_lab"] = float(d_lab[region].mean())

            # new bright/dark blob: large signed luma change forming a blob
            signed = rg - og
            blob = (np.abs(signed) > 25) & prod
            blob_u8 = blob.astype(np.uint8) * 255
            if int(blob_u8.sum()) > 0:
                nlab, _, stats, _ = cv2.connectedComponentsWithStats(blob_u8)
                biggest = 0
                for i in range(1, nlab):
                    biggest = max(biggest, int(stats[i, cv2.CC_STAT_AREA]))
                out["product_new_bright_blob_score"] = float(
                    biggest) / max(1, int(prod.sum()))

            eo = cv2.Canny(o, 50, 150) > 0
            er = cv2.Canny(r, 50, 150) > 0
            eo_p = int((eo & prod).sum())
            if eo_p > 0:
                retained = int((eo & er & prod).sum())
                out["product_edge_retention"] = float(retained) / eo_p

    out["changed_area_ratio"] = float(n_changed) / wm_px
    return out


def product_gate(metrics):
    """V11 Phase 3 — hard rejection rules for product-overlap cases."""
    ov = metrics.get("product_overlap", 0.0)
    reason = ""
    if ov > PRODUCT_OVERLAP_GATE_MIN:
        if metrics.get("product_color_delta_lab", 0.0) > PRODUCT_COLOR_DELTA_MAX:
            reason = "product_color_delta_too_high"
        elif metrics.get("product_new_bright_blob_score",
                         0.0) > PRODUCT_NEW_BRIGHT_BLOB_MAX:
            reason = "new_bright_blob_on_product"
        elif metrics.get("product_edge_retention",
                         1.0) < PRODUCT_EDGE_RETENTION_MIN:
            reason = "product_edge_loss"
    if (not reason and ov > 0.10 and
            metrics.get("changed_area_ratio", 0.0) > CHANGED_AREA_RATIO_MAX):
        reason = "repair_area_too_large_for_product_overlap"
    return reason == "", reason


def _border_edge_box_score(gray, bbox):
    """Fraction of bbox-perimeter pixels that sit on a strong straight luma
    step — the signature of a visible rectangular patch border."""
    bx, by, bw, bh = bbox
    H, W = gray.shape[:2]
    g = gray.astype(np.float32)
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
    return float((alld > 10).mean())


def compute_cover_metrics(repaired, bbox):
    """Patch 4 — detect a human-visible rectangular cover band."""
    bx, by, bw, bh = bbox
    H, W = repaired.shape[:2]
    gray = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY)
    inner = gray[by:by + bh, bx:bx + bw]
    ring_mask, _ = _get_ring(repaired, bbox, 0.4)
    ring = gray[ring_mask]
    zero = {"cover_rectangularity": 0.0, "cover_luma_delta": 0.0,
            "cover_texture_drop": 0.0, "cover_edge_box_score": 0.0,
            "cover_seam_score": 0.0}
    if inner.size == 0 or ring.size == 0:
        return zero

    luma_delta = abs(float(inner.mean()) - float(ring.mean()))
    tex_inner = float(cv2.Laplacian(inner, cv2.CV_64F).var())
    pad = max(12, max(bw, bh) // 3)
    rr = gray[max(0, by - pad):min(H, by + bh + pad),
              max(0, bx - pad):min(W, bx + bw + pad)]
    tex_ring = float(cv2.Laplacian(rr, cv2.CV_64F).var())
    if tex_ring < 25.0:
        texture_drop = 0.0
    else:
        texture_drop = max(0.0, 1.0 - tex_inner / max(tex_ring, 1e-6))
    edge_box = _border_edge_box_score(gray, bbox)
    seam = _compute_seam_delta(repaired, bbox)
    rectangularity = _clamp01(0.4 * edge_box +
                              0.3 * min(1.0, luma_delta / 12.0) +
                              0.3 * texture_drop)
    return {"cover_rectangularity": round(rectangularity, 4),
            "cover_luma_delta": round(luma_delta, 3),
            "cover_texture_drop": round(texture_drop, 4),
            "cover_edge_box_score": round(edge_box, 4),
            "cover_seam_score": round(seam, 4)}


def cover_passes_rect_gate(metrics):
    return (metrics["cover_rectangularity"] <= COVER_RECTANGULARITY_MAX and
            metrics["cover_luma_delta"] <= COVER_LUMA_DELTA_MAX and
            metrics["cover_texture_drop"] <= COVER_TEXTURE_DROP_MAX and
            metrics["cover_edge_box_score"] <= COVER_EDGE_BOX_MAX)


def _detect_alpha_halo(image, core_mask, bbox, delta_min=4, delta_max=35):
    """Patch 3 — semi-transparent watermark pixels around the core stroke."""
    if core_mask is None or not np.any(core_mask > 0):
        return None
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    ring_mask, _ = _get_ring(image, bbox, 0.5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ring_pixels = gray[ring_mask]
    if ring_pixels.size == 0:
        return None
    bg = float(np.median(ring_pixels))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(core_mask, k, iterations=2)
    band = cv2.bitwise_and(dil, cv2.bitwise_not(core_mask))
    delta = np.abs(gray - bg)
    sel = (band > 0) & (delta >= delta_min) & (delta <= delta_max)
    halo = np.zeros((H, W), np.uint8)
    halo[sel] = 255

    region = np.zeros((H, W), np.uint8)
    region[max(0, by - bh // 2):min(H, by + bh + bh // 2),
           max(0, bx - bw // 4):min(W, bx + bw + bw // 4)] = 255
    halo = cv2.bitwise_and(halo, region)

    core_px = max(1, int((core_mask > 0).sum()))
    if int((halo > 0).sum()) > 3 * core_px:  # runaway guard
        return None
    return halo


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

    # 8. Rectangular patch visibility (V9)
    rpv = _compute_rectangular_patch_visibility(repaired, bbox, ring_mask)

    # 9. Residual watermark gate (V10 Patch 2) — must be unreadable.
    orig_roi_bgr = image[by:by + bh, bx:bx + bw]
    clean_roi_bgr = repaired[by:by + bh, bx:bx + bw]
    # 2-D padded crop around the bbox for the surface-texture baseline.
    rpad = max(12, max(bw, bh) // 3)
    ring_crop = gray_r[max(0, by - rpad):min(H, by + bh + rpad),
                       max(0, bx - rpad):min(W, bx + bw + rpad)]
    # Known watermark locations, dilated to cover the alpha halo, cropped to
    # the bbox. Lets the residual gate measure exactly where the watermark
    # was rather than diluting over the whole (possibly textured) ROI.
    wm_mask = None
    src = ctx.logo_mask if (ctx.logo_mask is not None and
                            np.any(ctx.logo_mask > 0)) else ctx.stroke_mask
    if src is not None and src.shape[:2] == image.shape[:2]:
        # Horizontal-biased dilation so the gate also inspects inter-letter
        # gaps and trailing glyphs the stroke detector under-covers — those
        # are exactly where a tight inpaint leaves broken-but-readable text.
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
        wm_mask = cv2.dilate(src, kd, iterations=1)[by:by + bh, bx:bx + bw]
    residual = detect_residual_watermark(orig_roi_bgr, clean_roi_bgr,
                                         ring_gray=ring_crop, mask_roi=wm_mask)

    # 10. Cover rectangularity (V10 Patch 4) — never a visible band.
    cover_m = compute_cover_metrics(repaired, bbox)

    # 11. SSIM + raw boundary jump (V11 — real, named QA metrics).
    if roi_o.size > 0 and roi_r.size > 0:
        ssim_score = _ssim_local(roi_o, roi_r)
    else:
        ssim_score = 1.0
    boundary_jump = round(seam * 255.0, 2)

    # 12. Product-overlap safety metrics (V11 Phase 3/8).
    prod_m = compute_product_damage_metrics(image, repaired, bbox,
                                            ctx.product_mask)
    product_gate_pass, product_gate_reason = product_gate(prod_m)

    # Final composite score (lower = better)
    final = (2.0 * wm_residual + 2.0 * cover_vis + 2.5 * seam +
             1.5 * prod_damage + 0.5 * (1.0 - tex_consist) +
             1.0 * color_d / max(COLOR_DELTA_MAX, 1) + 1.0 * edge_dmg +
             1.5 * rpv + 2.0 * residual.template_corr +
             1.5 * residual.text_component_score +
             1.0 * (1.0 - residual.improvement_ratio))

    # Acceptance thresholds
    wm_max = WATERMARK_RESIDUAL_MAX
    cv_max = COVER_VISIBILITY_MAX
    sd_max = SEAM_DELTA_MAX
    pd_max = PRODUCT_DAMAGE_MAX
    ed_max = EDGE_DAMAGE_MAX
    rpv_max = _RECT_VIS_THRESHOLDS.get(roi.roi_class, 0.12)

    if roi.roi_class in LOOSE_CLASSES:
        cv_max *= 1.3
        sd_max *= 1.2
        rpv_max *= 1.3
    elif roi.roi_class in STRICT_CLASSES:
        wm_max *= 0.8
        cv_max *= 0.8
        sd_max *= 0.8
        rpv_max *= 0.8

    # V11 Phase 1 — mandatory QA-metric validation. Every required metric
    # must be actually computed; all-zero core set fails closed.
    qa_metrics = {
        "ssim": ssim_score,
        "color_delta": color_d,
        "boundary_jump": boundary_jump,
        "seam_delta": seam,
        "cover_visibility": cover_vis,
        "product_damage": prod_damage,
        "residual_template_corr": residual.template_corr,
        "residual_text_component": residual.text_component_score,
        "residual_improvement": residual.improvement_ratio,
    }
    metrics_valid, metrics_invalid_reasons = validate_qa_metrics(qa_metrics)

    # In-ROI edge loss IS the watermark being removed on flat/simple
    # backgrounds — only treat it as damage where the ROI may hold real
    # product structure (strict/detail classes). Product edges outside the
    # ROI are always guarded via prod_damage.
    edge_gate = edge_dmg if roi.roi_class in STRICT_CLASSES else 0.0

    # Patch 7 status semantics. The authoritative "is the watermark gone"
    # check is the V10 residual gate (residual.passed); the authoritative
    # "is there a visible band" checks are cover_vis and the cover
    # rectangularity gate. The legacy wm_residual and rpv metrics are noisy
    # on textured/dark surfaces (they fire on successful inpainting), so
    # they drive scoring but only hard-gate on strict/detail classes where
    # in-ROI structure is genuinely at risk.
    rpv_gate = rpv if roi.roi_class in STRICT_CLASSES else 0.0
    wm_gate = wm_residual if roi.roi_class in STRICT_CLASSES else wm_max

    cover_visibility_pass = cover_vis <= cv_max
    seam_pass = seam <= sd_max
    color_pass = color_d <= COLOR_DELTA_MAX * (1.6 if roi.roi_class in
                                               LOOSE_CLASSES else 1.0)
    product_damage_pass = prod_damage <= pd_max
    # The cover-rectangularity gate guards the adaptive-cover path (enforced
    # inside FinalAdaptiveCover.apply). It is NOT applied to inpaint repairs,
    # because an inpaint over a noisy surround is inherently smoother and
    # would be falsely flagged — cover_vis + seam catch true visible bands.
    cover_rect_pass = cover_m["cover_rectangularity"] <= COVER_RECTANGULARITY_MAX

    geometry_pass = (wm_gate <= wm_max and
                     cover_visibility_pass and
                     seam_pass and
                     product_damage_pass and
                     color_pass and
                     edge_gate <= ed_max and
                     rpv_gate <= rpv_max)

    passed = (metrics_valid and residual.passed and product_gate_pass and
              geometry_pass)

    if not metrics_valid:
        reason = (metrics_invalid_reasons[0] if metrics_invalid_reasons
                  else "qa_metrics_probably_not_computed")
    elif not residual.passed:
        reason = residual.failure_reason or "readable_residual_watermark"
    elif not product_gate_pass:
        reason = product_gate_reason or "product_damage"
    elif geometry_pass:
        reason = "accepted"
    else:
        reason = _qa_fail_reason(
            wm_gate, wm_max, cover_vis, cv_max, seam, sd_max,
            prod_damage, pd_max, edge_gate, ed_max, rpv_gate, rpv_max)
        extra = []
        if not color_pass:
            extra.append("color_delta_too_high")
        if extra:
            reason = "|".join([reason] + extra) if reason != "unknown" \
                else "|".join(extra)

    return QAResult(
        passed=passed, watermark_residual_score=round(wm_residual, 4),
        cover_visibility_score=round(cover_vis, 4),
        seam_delta_score=round(seam, 4),
        product_damage_score=round(prod_damage, 4),
        texture_consistency_score=round(tex_consist, 4),
        color_delta_score=round(color_d, 2),
        edge_damage_score=round(edge_dmg, 4),
        rectangular_patch_visibility=round(rpv, 4),
        final_score=round(final, 4), reason=reason,
        metrics_valid=metrics_valid, residual_pass=residual.passed,
        residual_template_corr=residual.template_corr,
        residual_text_component=residual.text_component_score,
        residual_improvement=residual.improvement_ratio,
        cover_rectangularity=cover_m["cover_rectangularity"],
        cover_luma_delta=cover_m["cover_luma_delta"],
        ssim_score=round(ssim_score, 4), boundary_jump=boundary_jump,
        metrics_invalid_reasons=metrics_invalid_reasons,
        residual_dot_chain_score=residual.dot_chain_score,
        residual_component_area_ratio=residual.component_area_ratio,
        residual_component_horizontal_span=residual.component_horizontal_span,
        residual_component_count=residual.component_count,
        product_overlap=round(prod_m["product_overlap"], 4),
        product_color_delta_lab=round(prod_m["product_color_delta_lab"], 3),
        product_new_bright_blob_score=round(
            prod_m["product_new_bright_blob_score"], 4),
        product_edge_retention=round(prod_m["product_edge_retention"], 4),
        product_gate_pass=product_gate_pass,
        product_gate_reason=product_gate_reason,
        changed_area_ratio=round(prod_m["changed_area_ratio"], 4))


def _qa_fail_reason(wm, wm_max, cv, cv_max, sd, sd_max, pd, pd_max,
                    ed, ed_max, rpv=0.0, rpv_max=1.0):
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
    if rpv > rpv_max:
        reasons.append("visible_rectangle_artifact")
    return "|".join(reasons) if reasons else "unknown"


# ---------------------------------------------------------------------------
# Rectangular Patch Visibility — detects human-visible rectangular patches (P0)
# ---------------------------------------------------------------------------

_RECT_VIS_THRESHOLDS = {
    "plain_white": 0.18,
    "near_white": 0.18,
    "low_texture_background": 0.22,
    "simple_product_surface": 0.20,
    "dark_product_surface": 0.20,
    "complex_product_detail": 0.14,
    "thin_flex_cable": 0.14,
    "text_or_label_area": 0.14,
    "glass_or_gradient": 0.22,
    "metallic_or_reflective": 0.22,
    "mixed_background_product": 0.16,
    "transparent_or_glossy": 0.22,
    "unknown": 0.20,
}


def _compute_rectangular_patch_visibility(repaired, bbox, ring_mask_bool):
    bx, by, bw, bh = bbox
    H, W = repaired.shape[:2]

    gray_r = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY)
    inner = gray_r[by:by + bh, bx:bx + bw]
    ring_gray = gray_r[ring_mask_bool]

    if inner.size == 0 or ring_gray.size == 0:
        return 0.0

    luma_delta = abs(float(inner.mean()) - float(ring_gray.mean())) / 255.0

    lab_r = cv2.cvtColor(repaired, cv2.COLOR_BGR2LAB)
    inner_lab = lab_r[by:by + bh, bx:bx + bw].reshape(-1, 3).mean(axis=0).astype(np.float64)
    ring_lab = lab_r[ring_mask_bool].reshape(-1, 3).mean(axis=0).astype(np.float64)
    color_delta = float(np.linalg.norm(inner_lab - ring_lab)) / 20.0

    tex_inner = float(cv2.Laplacian(inner, cv2.CV_64F).var())
    pad = max(12, max(bw, bh) // 3)
    ry1 = max(0, by - pad)
    ry2 = min(H, by + bh + pad)
    rx1 = max(0, bx - pad)
    rx2 = min(W, bx + bw + pad)
    ring_region = gray_r[ry1:ry2, rx1:rx2]
    tex_ring = float(cv2.Laplacian(ring_region, cv2.CV_64F).var())
    # On a flat surround there is no texture to "drop" — a smooth repaired
    # patch over smooth surroundings is correct, not a visible rectangle.
    if tex_ring < 25.0:
        hf_drop = 0.0
    else:
        hf_drop = max(0.0, 1.0 - tex_inner / max(tex_ring, 1e-6))

    border_deltas = []
    if by > 0:
        border_deltas.append(float(np.abs(
            gray_r[by - 1, bx:bx + bw].astype(np.float32) -
            gray_r[by, bx:bx + bw].astype(np.float32)).mean()) / 255.0)
    if by + bh < H:
        border_deltas.append(float(np.abs(
            gray_r[by + bh, bx:bx + bw].astype(np.float32) -
            gray_r[by + bh - 1, bx:bx + bw].astype(np.float32)).mean()) / 255.0)
    if bx > 0:
        border_deltas.append(float(np.abs(
            gray_r[by:by + bh, bx - 1].astype(np.float32) -
            gray_r[by:by + bh, bx].astype(np.float32)).mean()) / 255.0)
    if bx + bw < W:
        border_deltas.append(float(np.abs(
            gray_r[by:by + bh, bx + bw].astype(np.float32) -
            gray_r[by:by + bh, bx + bw - 1].astype(np.float32)).mean()) / 255.0)
    edge_score = max(border_deltas) if border_deltas else 0.0

    return min(1.0, 0.25 * luma_delta + 0.25 * color_delta +
               0.30 * hf_drop + 0.20 * edge_score)


# ---------------------------------------------------------------------------
# Mask Confidence Validation (P1)
# ---------------------------------------------------------------------------

def _validate_mask_confidence(mask, bbox):
    if mask is None or not np.any(mask > 0):
        return {"valid": False, "reason": "empty_mask",
                "occupancy": 0.0, "rectangularity": 0.0}
    bx, by, bw, bh = bbox
    bbox_area = max(1, bw * bh)
    mask_roi = mask[by:by + bh, bx:bx + bw]
    occupancy = float((mask_roi > 0).sum()) / bbox_area

    ys, xs = np.where(mask_roi > 0)
    if ys.size == 0:
        return {"valid": False, "reason": "empty_mask",
                "occupancy": 0.0, "rectangularity": 0.0}
    bbox_of_mask = max(1, (int(xs.max()) - int(xs.min()) + 1) *
                       (int(ys.max()) - int(ys.min()) + 1))
    rectangularity = ys.size / bbox_of_mask

    if occupancy > 0.35:
        return {"valid": False, "reason": "mask_too_large",
                "occupancy": occupancy, "rectangularity": rectangularity}
    if rectangularity > 0.75 and occupancy > 0.20:
        return {"valid": False, "reason": "bbox_like_mask",
                "occupancy": occupancy, "rectangularity": rectangularity}

    return {"valid": True, "occupancy": occupancy,
            "rectangularity": rectangularity}


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


class PlainWhiteFastPath(RepairTool):
    """V11 Phase 4 — on pure/near-white, low-texture, non-product backgrounds
    the safest repair is an exact background fill (ring colour + matched
    noise, 1px feather) rather than Telea, which leaves dotted gray residue.
    A precise residual-component second pass removes any leftover fragments.
    """
    name = "plain_white_fast_path"
    cost = 1
    risk = 1

    def is_applicable(self, ctx):
        roi = ctx.roi_analysis
        if roi.roi_class not in PLAIN_BG_CLASSES:
            return False
        return (roi.white_pixel_ratio > 0.80 and roi.edge_density < 0.05 and
                roi.product_pixel_ratio < 0.10)

    def apply(self, ctx):
        if not self.is_applicable(ctx):
            return None
        bx, by, bw, bh = ctx.watermark_bbox
        ring_med, sigma = _ring_stats(ctx.image, ctx.watermark_bbox)
        result = ctx.image.copy()
        patch = np.full((bh, bw, 3), ring_med, dtype=np.uint8)
        patch = _add_noise(patch, max(1.0, sigma))
        result[by:by + bh, bx:bx + bw] = patch
        result = _feather_blend(ctx.image, result, ctx.watermark_bbox,
                                feather_px=1)

        # Second pass — clear any residual dot/glyph fragments precisely.
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)[by:by + bh, bx:bx + bw]
        ring_mask, _ = _get_ring(ctx.image, ctx.watermark_bbox, 0.5)
        rg = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)[ring_mask]
        if rg.size > 0:
            rhp = np.abs(rg.astype(np.float32) - float(rg.mean()))
            base_mean, base_std = float(rhp.mean()), float(rhp.std())
            full = np.ones_like(gray, np.uint8) * 255
            dot, _c, _a, _s = detect_residual_components(
                gray, full, base_mean, base_std)
            if dot > 0.0:
                cg = gray.astype(np.float32)
                hp = np.abs(cg - cv2.GaussianBlur(cg, (0, 0), sigmaX=2.0))
                thr = max(8.0, base_mean + 2.0 * base_std)
                comp = (hp > thr).astype(np.uint8) * 255
                sub = result[by:by + bh, bx:bx + bw]
                sub = cleanup_residual_components_with_ring_fill(
                    sub, (0, 0, bw, bh), comp, ring_med)
                result[by:by + bh, bx:bx + bw] = sub
        return RepairCandidate(self.name, result)


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


class PlainWhiteDirectNeighborCloneV2(RepairTool):
    name = "plain_white_direct_neighbor_clone_v2"
    cost_level = 1
    risk_level = 1
    supported_roi_classes = ["plain_white", "near_white",
                             "low_texture_background"]

    def apply(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        expand = max(2, min(4, bh // 4))

        dirs = [
            (0, -(bh + expand)),
            (0, bh + expand),
            (-(bw + expand), 0),
            (bw + expand, 0),
            (-(bw + expand), -(bh + expand)),
            (bw + expand, -(bh + expand)),
            (-(bw + expand), bh + expand),
            (bw + expand, bh + expand),
        ]
        ring_mask, _ = _get_ring(ctx.image, ctx.watermark_bbox, 0.4)
        ring_pixels = ctx.image[ring_mask]
        if ring_pixels.size == 0:
            return None
        target_lab = cv2.cvtColor(
            np.median(ring_pixels, axis=0).reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)

        valid = []
        for dx, dy in dirs:
            sx, sy = bx + dx, by + dy
            if sx < 0 or sy < 0 or sx + bw > W or sy + bh > H:
                continue
            patch = ctx.image[sy:sy + bh, sx:sx + bw]
            if patch.shape[0] != bh or patch.shape[1] != bw:
                continue
            if ctx.product_mask is not None:
                prod_roi = ctx.product_mask[sy:sy + bh, sx:sx + bw]
                if prod_roi.shape == (bh, bw) and float((prod_roi > 0).mean()) > 0.02:
                    continue
            ed = _patch_edge_density(patch)
            if ed > 0.06:
                continue
            patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0).astype(np.float64)
            cd = float(np.linalg.norm(patch_lab - target_lab))
            if cd > 15.0:
                continue
            valid.append((patch.copy(), cd, ed, dx, dy))

        if not valid:
            return None

        best = None
        best_seam = float("inf")
        for patch, cd, ed, dx, dy in valid:
            test = ctx.image.copy()
            test[by:by + bh, bx:bx + bw] = patch
            seam = _compute_seam_delta(test, ctx.watermark_bbox)
            total = seam + 0.3 * cd / 15.0
            if total < best_seam:
                best_seam = total
                best = patch

        if best is None or best_seam > 0.15:
            return None

        out = ctx.image.copy()
        out[by:by + bh, bx:bx + bw] = best
        feather = max(1, min(2, bh // 8))
        out = _feather_blend(ctx.image, out, ctx.watermark_bbox, feather)
        return RepairCandidate(self.name, out,
                               {"seam_score": round(best_seam, 4)})


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
                "complex_product_detail", "mixed_background_product",
                "transparent_or_glossy", "unknown"]


def _get_effective_stroke_mask(ctx, dilate_px=0, add_halo=True):
    """Patch 3 — 3-layer mask: core stroke + alpha halo + soft safety.

    Core stroke comes from the logo/stroke mask. The alpha halo recovers
    the semi-transparent watermark pixels that ring the core (which, if
    left behind, become broken-but-readable residual text). The soft
    safety layer is the light dilation applied by callers.
    """
    core = None
    if ctx.logo_mask is not None and np.any(ctx.logo_mask > 0):
        core = ctx.logo_mask.copy()
    elif ctx.stroke_mask is not None and np.any(ctx.stroke_mask > 0):
        core = ctx.stroke_mask.copy()
    else:
        return None

    mask = core
    if add_halo:
        halo = _detect_alpha_halo(ctx.image, core, ctx.watermark_bbox)
        if halo is not None and np.any(halo > 0):
            combined = cv2.bitwise_or(core, halo)
            # Keep the combined mask text-shaped, not a full bbox rectangle.
            conf = _validate_mask_confidence(combined, ctx.watermark_bbox)
            if conf["valid"] or conf.get("occupancy", 1.0) <= 0.35:
                mask = combined

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


class SegmentedBackgroundProductRepair(RepairTool):
    name = "segmented_background_product_repair"
    cost_level = 4
    risk_level = 2
    supported_roi_classes = ["mixed_background_product", "thin_flex_cable",
                             "complex_product_detail", "dark_product_surface",
                             "simple_product_surface", "text_or_label_area"]

    def apply(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]

        if ctx.product_mask is None:
            return None
        prod_roi = ctx.product_mask[by:by + bh, bx:bx + bw]
        if prod_roi.shape != (bh, bw):
            return None
        product_ratio = float((prod_roi > 0).mean())
        if product_ratio < 0.05 or product_ratio > 0.95:
            return None

        bg_mask_roi = (prod_roi == 0).astype(np.uint8) * 255
        fg_mask_roi = (prod_roi > 0).astype(np.uint8) * 255

        out = ctx.image.copy()

        if np.any(bg_mask_roi > 0):
            bg_full = np.zeros((H, W), np.uint8)
            bg_full[by:by + bh, bx:bx + bw] = bg_mask_roi
            median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
            fill = np.full((H, W, 3), median_color, dtype=np.uint8)
            fill = _add_noise(fill, max(noise_std, 0.5), 0.3)
            bg_alpha = bg_full.astype(np.float32) / 255.0
            bg_alpha = cv2.GaussianBlur(bg_alpha, (0, 0), sigmaX=1.5)
            bg_alpha3 = np.stack([bg_alpha] * 3, axis=-1)
            out = (fill.astype(np.float32) * bg_alpha3 +
                   out.astype(np.float32) * (1 - bg_alpha3)).astype(np.uint8)

        if np.any(fg_mask_roi > 0):
            stroke = _get_effective_stroke_mask(ctx, dilate_px=1)
            if stroke is not None:
                fg_full = np.zeros((H, W), np.uint8)
                fg_full[by:by + bh, bx:bx + bw] = fg_mask_roi
                combined = cv2.bitwise_and(stroke, fg_full)
                if np.any(combined > 0):
                    out = cv2.inpaint(out, combined, 2, cv2.INPAINT_TELEA)

        return RepairCandidate(self.name, out,
                               {"product_ratio": round(product_ratio, 3)})


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
        """V10 Patch 4/7 — produce a surface-aware cover, then verify it is
        not a visible rectangle. If the primary cover trips the
        rectangularity gate, auto-retry across alternative cover styles and
        keep the one with the lowest rectangularity (auto_cover_retry).
        Never emits a single obvious gray/white band across mixed surfaces.
        """
        roi_class = ctx.roi_analysis.roi_class

        stroke_mask = _get_effective_stroke_mask(ctx, dilate_px=2)
        has_stroke = stroke_mask is not None and np.any(stroke_mask > 0)

        # Decide by ACTUAL in-bbox content, not the (sometimes wrong) ROI
        # class: if the watermark sits on low-structure surface, a full-bbox
        # Telea fill is safe and completely removes the text (incl. trailing
        # glyphs). Only when the bbox holds real product structure do we use
        # lighter covers to avoid smearing it.
        bxg, byg, bwg, bhg = ctx.watermark_bbox
        Hh, Ww = ctx.image.shape[:2]
        roi_g = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)[
            byg:byg + bhg, bxg:bxg + bwg]
        in_edge = (float(cv2.Canny(roi_g, 50, 150).mean()) / 255.0
                   if roi_g.size else 0.0)
        in_prod = 0.0
        if ctx.product_mask is not None:
            pr = ctx.product_mask[byg:byg + bhg, bxg:bxg + bwg]
            if pr.size:
                in_prod = float((pr > 0).mean())
        low_structure = in_edge < 0.10 and in_prod < 0.45

        builders = []
        if low_structure:
            # Uniform surround: the full-bbox Telea inpaint covers the entire
            # watermark line (no trailing glyph survives) and blends into the
            # surrounding surface. It is both the most complete hide and the
            # most natural result, so try it first.
            builders.append(("opaque_footprint",
                             lambda: self._opaque_footprint_cover(ctx,
                                                                  stroke_mask)))
            if has_stroke:
                builders.append(("stroke_level",
                                 lambda: self._stroke_level_cover(ctx,
                                                                  stroke_mask)))
        else:
            # Detailed / mixed / dark / metallic surround: a full-bbox fill
            # would smear product structure. Prefer the lightest-touch covers
            # (stroke / segmented) first, full footprint only as last resort.
            if has_stroke:
                builders.append(("stroke_level",
                                 lambda: self._stroke_level_cover(ctx,
                                                                  stroke_mask)))
            if ctx.product_mask is not None:
                builders.append(("segmented",
                                 lambda: self._segmented_cover(ctx)))
            builders.append(("opaque_footprint",
                             lambda: self._opaque_footprint_cover(ctx,
                                                                  stroke_mask)))

        # A cover must (a) actually HIDE the watermark and (b) not look like a
        # visible rectangle. Evaluate every builder on both axes and pick the
        # best: a hidden + non-rectangular cover wins outright; otherwise
        # prefer hiding, then minimal rectangularity (auto_cover_retry).
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        rpad = max(12, max(bw, bh) // 3)
        ring_crop = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)[
            max(0, by - rpad):min(H, by + bh + rpad),
            max(0, bx - rpad):min(W, bx + bw + rpad)]
        orig_roi = ctx.image[by:by + bh, bx:bx + bw]
        # Mask-aware hiding check (same as the production gate) so a clean
        # full-footprint inpaint is correctly credited as hiding the
        # watermark instead of being penalised by surface texture.
        cover_wm_mask = None
        if has_stroke:
            kd2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
            cover_wm_mask = cv2.dilate(stroke_mask, kd2,
                                       iterations=1)[by:by + bh, bx:bx + bw]

        def hides(cand):
            cl = cand.repaired_image[by:by + bh, bx:bx + bw]
            return detect_residual_watermark(orig_roi, cl,
                                             ring_gray=ring_crop,
                                             mask_roi=cover_wm_mask).passed

        best = None
        best_key = None
        best_style = None
        for style, build in builders:
            try:
                cand = build()
            except Exception:
                cand = None
            if cand is None:
                continue
            metrics = compute_cover_metrics(cand.repaired_image,
                                            ctx.watermark_bbox)
            cand.metadata["cover_metrics"] = metrics
            cand.metadata["cover_style"] = style
            hidden = hides(cand)
            rect_ok = cover_passes_rect_gate(metrics)
            if hidden and rect_ok:
                cand.metadata["cover_rect_gate"] = "pass"
                return cand
            # rank: hidden first (True>False), then low rectangularity
            key = (0 if hidden else 1, metrics["cover_rectangularity"])
            if best_key is None or key < best_key:
                best_key = key
                best = cand
                best_style = style

        if best is not None:
            best.metadata["cover_rect_gate"] = "best_effort"
            best.metadata["cover_style"] = best_style
            return best
        return self._frosted_local_blur_cover(ctx)

    def _opaque_footprint_cover(self, ctx, stroke_mask):
        """Fully remove the watermark by inpainting its entire footprint
        (strokes + alpha halo, generously dilated) from the surrounding
        surface. Telea propagates the real neighbouring pixels inward, so no
        colour is guessed and no noise band is introduced — the surest hide
        on a fairly uniform surface. The rectangularity gate validates it."""
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]

        # Cover the full bbox, extended horizontally beyond it: the detector
        # sometimes clips the leading/trailing glyph of ".com", so pad the
        # x-extent generously while keeping the y-extent tight, then Telea-
        # reconstruct from the surrounding surface.
        fp = np.zeros((H, W), np.uint8)
        pad_y = 3
        pad_x = max(8, int(bw * 0.12))
        fy1, fy2 = max(0, by - pad_y), min(H, by + bh + pad_y)
        fx1, fx2 = max(0, bx - pad_x), min(W, bx + bw + pad_x)
        fp[fy1:fy2, fx1:fx2] = 255

        result = cv2.inpaint(ctx.image, fp, 4, cv2.INPAINT_TELEA)
        # Light feathered blend back so the inpaint boundary is seamless.
        alpha = cv2.GaussianBlur(fp.astype(np.float32) / 255.0,
                                 (0, 0), sigmaX=1.5)
        alpha3 = np.stack([alpha] * 3, axis=-1)
        blended = (result.astype(np.float32) * alpha3 +
                   ctx.image.astype(np.float32) * (1 - alpha3)).astype(np.uint8)
        return RepairCandidate(self.name, blended,
                               {"cover_style": "opaque_footprint"})

    def _stroke_level_cover(self, ctx, stroke_mask):
        # Inpaint the text band from the surrounding surface — no synthetic
        # fill, no RGB speckle. Covers the full horizontal span of the stroke
        # row (so trailing glyphs/halo are included) but keeps the band's
        # height tight, so it disturbs less product detail than a full-bbox
        # fill. A lighter-touch alternative for structured surrounds.
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
        wide = cv2.dilate(stroke_mask, kd, iterations=1)
        band = np.zeros((H, W), np.uint8)
        ys, xs = np.where(wide > 0)
        if ys.size:
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            band[y1:y2, x1:x2] = 255
        else:
            band = wide
        result = cv2.inpaint(ctx.image, band, 3, cv2.INPAINT_TELEA)
        return RepairCandidate(self.name, result,
                               {"cover_style": "stroke_level"})

    def _segmented_cover(self, ctx):
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        out = ctx.image.copy()

        if ctx.product_mask is not None:
            prod_roi = ctx.product_mask[by:by + bh, bx:bx + bw]
            if prod_roi.shape == (bh, bw):
                bg_mask = (prod_roi == 0)
                if bg_mask.any():
                    median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
                    fill = np.full((bh, bw, 3), median_color, dtype=np.uint8)
                    fill = _add_noise(fill, max(noise_std, 0.5), 0.3)
                    bg_alpha = bg_mask.astype(np.float32)
                    bg_alpha = cv2.GaussianBlur(bg_alpha, (0, 0), sigmaX=1.5)
                    bg_alpha3 = np.stack([bg_alpha] * 3, axis=-1)
                    roi = out[by:by + bh, bx:bx + bw].astype(np.float32)
                    roi = fill.astype(np.float32) * bg_alpha3 + roi * (1 - bg_alpha3)
                    out[by:by + bh, bx:bx + bw] = roi.astype(np.uint8)

                stroke = _get_effective_stroke_mask(ctx, dilate_px=2)
                if stroke is not None:
                    fg_mask = (prod_roi > 0).astype(np.uint8) * 255
                    fg_full = np.zeros((H, W), np.uint8)
                    fg_full[by:by + bh, bx:bx + bw] = fg_mask
                    combined = cv2.bitwise_and(stroke, fg_full)
                    if np.any(combined > 0):
                        out = cv2.inpaint(out, combined, 2, cv2.INPAINT_TELEA)

                return RepairCandidate(self.name, out,
                                       {"cover_style": "segmented"})

        return self._blurred_local_cover(ctx)

    def _near_white_cover(self, ctx):
        median_color, noise_std = _ring_stats(ctx.image, ctx.watermark_bbox)
        bx, by, bw, bh = ctx.watermark_bbox
        H, W = ctx.image.shape[:2]
        cover = np.full_like(ctx.image, median_color)
        cover = _add_noise(cover, max(noise_std, 1.0), 0.4)
        mask = self._soft_cover_mask(ctx.image.shape, ctx.watermark_bbox, 8)
        alpha = mask * 0.95   # V10: opaque enough to hide, not just dim
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
        alpha = mask * 0.92   # V10: opaque enough to hide, not just dim
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
        alpha = mask * 0.92   # V10: opaque enough to hide, not just dim
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
        alpha = mask * 0.95   # V10: opaque enough to hide, not just dim
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
    # Group A (1-21)
    CloneAbovePatch(), CloneBelowPatch(), CloneLeftPatch(), CloneRightPatch(),
    CloneBestOf4Dirs(), CloneBestOf8Dirs(), WhiteMedianFill(),
    PlainWhiteFastPath(),
    WhitePatchWithNoise(), CornerBackgroundClone(), CanvasMarginClone(),
    RingMedianFill(), RingMeanFill(), RingModeFill(),
    LocalNoiseTransferFill(), JpegTextureCloneFill(), MicroTileWhiteClone(),
    FeatheredWhiteClone(), HardPasteWhiteClone(), SeamScoredWhiteClone(),
    EdgeSafeWhiteClone(), PlainWhiteDirectNeighborCloneV2(),
    # Group B (22-41)
    SameSurfaceCloneAbove(), SameSurfaceCloneBelow(),
    SameSurfaceCloneLeft(), SameSurfaceCloneRight(),
    SameSurfaceBestPatch(), SameColorRegionClone(),
    SameLuminanceRegionClone(), SameTextureRegionClone(),
    SurfacePlaneFill(), SurfaceGradientFill(), SurfaceNoisePreservedFill(),
    SurfacePatchBorderBlend(), SurfacePatchGammaMatch(),
    SurfacePatchColorMatch(), SurfacePatchContrastMatch(),
    SurfacePatchLowpassMatch(), SurfacePatchHighpassTransfer(),
    LargeUniformAreaRepair(), DarkSurfaceClone(), LightSurfaceClone(),
    # Group C (42-61)
    TemplateLogoMaskRemove(), ScaledTemplateLogoMask(),
    RotatedTemplateLogoMask(), AlphaTemplateLogoMask(),
    StrokeOnlyMaskInpaint(), StrokeMaskDilate1px(), StrokeMaskDilate2px(),
    StrokeMaskSoftEdge(), TextComponentFilterMask(),
    FrequencySignatureLogoMask(), LabLChannelLogoMask(),
    BlueChannelLogoMask(), GrayDeltaLogoMask(),
    TemplateResidualMinimization(), MultiSizeLogoMaskBank(),
    MultiPositionLogoMaskBank(), LogoMaskPlusCloneFill(),
    LogoMaskPlusSurfaceFill(), LogoMaskPlusTelea(), LogoMaskPlusLama(),
    # Group D (62-76)
    LinearGradientReconstruction(), BilinearGradientReconstruction(),
    RadialGradientReconstruction(), ThinPlateSplineFill(),
    PoissonGradientClone(), GlassSurfaceClone(), FrostedGlassNoiseFill(),
    TransparentBackCoverFill(), SpecularHighlightPreserveFill(),
    ShadowGradientPreserveFill(), LowFrequencyGradientFill(),
    HighFrequencyNoiseTransfer(), GlassEdgeAwareInpaint(),
    ReflectionAwarePatchClone(), SoftMaterialSurfaceRepair(),
    # Group E (77-92)
    EdgeAwareClone(), LinePreservingInpaint(), ContourGuidedInpaint(),
    ThinFlexCableRepair(), BlackFlexTextureRepair(), MetalConnectorRepair(),
    ScrewAreaAvoidRepair(), ProductMaskProtectedFill(),
    BackgroundOnlyRepair(), ForegroundSurfaceOnlyRepair(),
    EdgeDirectionExtrapolation(), StructureTensorInpaint(),
    AnisotropicDiffusionInpaint(), PatchMatchInpaint(),
    NearestNeighborTextureSynthesis(), SegmentedBackgroundProductRepair(),
    # Group F (93-101)
    OpenCVTeleaInpaint(), OpenCVNSInpaint(), LamaSmallMask(),
    LamaFullBbox(), LamaStrokeMask(), LamaWithContextPadding(),
    DeepfillStyleInpaint(), MATInpaint(), SDInpaintLowStrength(),
    # Tool 102
    FinalAdaptiveCover(),
]

STRATEGY_BANK_BY_CLASS = {
    # V10 Patch 5 — real pixels (clone) and statistical/gradient fills run
    # BEFORE stroke/logo repair on low-risk backgrounds.
    "plain_white": [
        "plain_white_fast_path",
        "ring_median_fill", "white_median_fill", "clone_best_of_8_dirs",
        "hard_paste_white_clone", "seam_scored_white_clone",
        "plain_white_direct_neighbor_clone_v2", "surface_gradient_fill",
        "alpha_template_logo_mask", "logo_mask_plus_clone_fill",
        "stroke_only_mask_inpaint", "opencv_telea_inpaint",
        "final_adaptive_cover",
    ],
    "near_white": [
        "plain_white_fast_path",
        "ring_median_fill", "white_median_fill", "clone_best_of_8_dirs",
        "seam_scored_white_clone", "surface_gradient_fill",
        "alpha_template_logo_mask", "logo_mask_plus_clone_fill",
        "stroke_only_mask_inpaint", "opencv_telea_inpaint",
        "final_adaptive_cover",
    ],
    "low_texture_background": [
        "plain_white_fast_path",
        "ring_median_fill", "clone_best_of_8_dirs",
        "surface_plane_fill", "surface_gradient_fill",
        "logo_mask_plus_surface_fill", "stroke_only_mask_inpaint",
        "opencv_telea_inpaint", "lama_stroke_mask",
        "final_adaptive_cover",
    ],
    "simple_product_surface": [
        "same_surface_best_patch", "same_color_region_clone",
        "surface_patch_color_match", "surface_gradient_fill",
        "surface_noise_preserved_fill",
        "segmented_background_product_repair",
        "stroke_only_mask_inpaint", "logo_mask_plus_surface_fill",
        "opencv_telea_inpaint", "lama_stroke_mask",
        "final_adaptive_cover",
    ],
    "dark_product_surface": [
        "stroke_only_mask_inpaint", "dark_surface_clone",
        "same_surface_best_patch", "surface_patch_gamma_match",
        "segmented_background_product_repair",
        "surface_patch_contrast_match",
        "logo_mask_plus_surface_fill",
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
        "linear_gradient_reconstruction",
        "bilinear_gradient_reconstruction", "surface_plane_fill",
        "same_luminance_region_clone", "high_frequency_noise_transfer",
        "logo_mask_plus_surface_fill", "stroke_only_mask_inpaint",
        "opencv_telea_inpaint", "lama_stroke_mask",
        "final_adaptive_cover",
    ],
    "thin_flex_cable": [
        "stroke_only_mask_inpaint", "thin_flex_cable_repair",
        "black_flex_texture_repair", "edge_aware_clone",
        "logo_mask_plus_clone_fill", "segmented_background_product_repair",
        "line_preserving_inpaint", "opencv_telea_inpaint",
        "lama_stroke_mask", "final_adaptive_cover",
    ],
    "text_or_label_area": [
        "stroke_only_mask_inpaint", "template_logo_mask_remove",
        "logo_mask_plus_clone_fill",
        "segmented_background_product_repair",
        "text_component_filter_mask",
        "opencv_telea_inpaint", "final_adaptive_cover",
    ],
    "complex_product_detail": [
        "template_logo_mask_remove", "stroke_only_mask_inpaint",
        "text_component_filter_mask",
        "segmented_background_product_repair",
        "logo_mask_plus_telea", "edge_aware_clone",
        "contour_guided_inpaint", "patchmatch_inpaint",
        "lama_stroke_mask", "lama_with_context_padding",
        "final_adaptive_cover",
    ],
    "mixed_background_product": [
        "segmented_background_product_repair",
        "template_logo_mask_remove", "stroke_only_mask_inpaint",
        "logo_mask_plus_clone_fill", "stroke_mask_dilate_1px",
        "edge_aware_clone", "opencv_telea_inpaint",
        "lama_stroke_mask", "final_adaptive_cover",
    ],
    "transparent_or_glossy": [
        "linear_gradient_reconstruction",
        "bilinear_gradient_reconstruction",
        "glass_surface_clone", "frosted_glass_noise_fill",
        "surface_gradient_fill", "high_frequency_noise_transfer",
        "logo_mask_plus_surface_fill",
        "poisson_gradient_clone", "lama_stroke_mask",
        "final_adaptive_cover",
    ],
    "unknown": [
        "stroke_only_mask_inpaint", "template_logo_mask_remove",
        "clone_best_of_8_dirs", "logo_mask_plus_clone_fill",
        "same_surface_best_patch", "surface_gradient_fill",
        "stroke_mask_soft_edge", "logo_mask_plus_surface_fill",
        "opencv_telea_inpaint", "lama_stroke_mask",
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

# V11 Phase 7 — beam-search budget per ROI class (number of tools to try
# before selecting the best passing candidate).
_MAX_CANDIDATES_BY_CLASS = {
    "plain_white": 6,
    "near_white": 6,
    "low_texture_background": 8,
    "metallic_or_reflective": 10,
    "transparent_or_glossy": 10,
    "glass_or_gradient": 10,
    "simple_product_surface": 10,
    "dark_product_surface": 10,
    "complex_product_detail": 12,
    "thin_flex_cable": 12,
    "mixed_background_product": 12,
    "text_or_label_area": 10,
    "unknown": 10,
}
# A passing candidate this good (and product-safe) is accepted immediately.
_EXCELLENT_SCORE = 0.9


def _check_product_overlap_guard(ctx):
    bx, by, bw, bh = ctx.watermark_bbox
    if ctx.product_mask is None:
        return False, 0.0, 0.0, 0.0
    prod_roi = ctx.product_mask[by:by + bh, bx:bx + bw]
    if prod_roi.shape != (bh, bw):
        return False, 0.0, 0.0, 0.0
    product_overlap = float((prod_roi > 0).mean())

    gray = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[by:by + bh, bx:bx + bw]
    edges = cv2.Canny(roi_gray, 50, 150)
    edge_density = float(edges.mean()) / 255.0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=5)
    long_line_score = 0.0
    if lines is not None:
        long_line_score = min(1.0, len(lines) / 10.0)

    block = ((product_overlap > 0.35 and edge_density > 0.10) or
             edge_density > 0.18 or
             long_line_score > 0.25)

    # V10 Patch 5 — on genuinely low-risk backgrounds (white / near-white /
    # low-texture), real-pixel clones and statistical fills are safe even
    # when a large flat product fills the frame. The guard exists to protect
    # textured/detailed product surfaces, not flat ones. Only keep the block
    # if the ROI actually carries in-ROI structure worth protecting.
    if block and ctx.roi_analysis.roi_class in LOOSE_CLASSES and \
            edge_density < 0.10:
        block = False
    return block, product_overlap, edge_density, long_line_score


def repair_image_progressively(ctx: RepairContext) -> tuple[RepairCandidate, list]:
    tools = select_tools_for_roi(ctx)
    trace = []

    block_full_bbox, prod_ov, prod_ed, prod_lls = _check_product_overlap_guard(ctx)

    _STROKE_ONLY_TOOLS = {
        "template_logo_mask_remove", "stroke_only_mask_inpaint",
        "stroke_mask_dilate_1px", "stroke_mask_soft_edge",
        "logo_mask_plus_clone_fill", "logo_mask_plus_telea",
        "logo_mask_plus_surface_fill", "text_component_filter_mask",
        "segmented_background_product_repair",
        "thin_flex_cable_repair", "line_preserving_inpaint",
        "contour_guided_inpaint", "edge_direction_extrapolation",
    }

    best_failed_candidate = None
    best_failed_score = float("inf")
    best_failed_tool = None

    # Patch 6 — strategy execution telemetry.
    strategy_list = [t.name for t in tools]
    families_attempted = set()
    tools_attempted = 0
    qa_reject_reasons = {}

    # V11 Phase 7 — beam selection. Do NOT stop at the first passing
    # candidate on risky classes; try a bounded set of tools, keep every
    # candidate that clears the hard gates, and pick the best by composite
    # score. Only first-pass early-exit on an "excellent", product-safe
    # candidate.
    passed_candidates = []  # (final_score, idx, tool_name, family, candidate, qa)
    tools_reachable = len(tools)
    max_try = _MAX_CANDIDATES_BY_CLASS.get(
        ctx.roi_analysis.roi_class, 10)

    for tool in tools:
        if tool.name == "final_adaptive_cover":
            continue

        if block_full_bbox and tool.name not in _STROKE_ONLY_TOOLS:
            family = get_method_family(tool.name)
            if family in (MethodFamily.REAL_PIXEL_CLONE,
                          MethodFamily.STATISTICAL_FILL,
                          MethodFamily.GRADIENT_RECONSTRUCTION):
                trace.append({
                    "tool": tool.name, "passed": False,
                    "reason": "blocked_by_product_overlap_guard",
                    "qa": {}, "runtime_s": 0.0})
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

        family = get_method_family(tool.name)
        tools_attempted += 1
        families_attempted.add(family.value)
        trace.append({
            "tool": tool.name, "passed": qa.passed,
            "reason": qa.reason, "qa": asdict(qa),
            "method_family": family.value,
            "runtime_s": dt})

        if qa.passed:
            candidate.metadata["qa"] = asdict(qa)
            passed_candidates.append(
                (qa.final_score, len(passed_candidates), tool.name,
                 family, candidate, qa))
            # Excellent + product-safe → accept immediately (early exit).
            excellent = (qa.final_score <= _EXCELLENT_SCORE and
                         qa.residual_pass and qa.product_gate_pass and
                         qa.product_overlap <= 0.05)
            if excellent:
                break
        else:
            qa_reject_reasons[tool.name] = qa.reason
            if qa.final_score < best_failed_score:
                best_failed_score = qa.final_score
                best_failed_candidate = candidate
                best_failed_candidate.metadata["qa"] = asdict(qa)
                best_failed_tool = tool.name

        # Beam budget: stop after exploring max_try real attempts.
        if tools_attempted >= max_try:
            break

    # V11 Phase 7 — pick the best passing candidate (lowest composite score).
    if passed_candidates:
        passed_candidates.sort(key=lambda c: c[0])
        _, _, win_name, win_family, win_cand, win_qa = passed_candidates[0]
        win_cand.metadata["qa"] = asdict(win_qa)
        win_cand.metadata["final_method"] = win_name
        win_cand.metadata["method_family"] = win_family.value
        win_cand.metadata["is_real_repair"] = (
            win_family != MethodFamily.ADAPTIVE_COVER)
        win_cand.metadata["telemetry"] = {
            "roi_class": ctx.roi_analysis.roi_class,
            "strategy_list": strategy_list,
            "tools_reachable": tools_reachable,
            "tools_tried": tools_attempted,
            "tools_rejected": len(qa_reject_reasons),
            "candidates_passed": len(passed_candidates),
            "tools_attempted": tools_attempted,
            "families_attempted": sorted(families_attempted),
            "final_method": win_name,
            "qa_reject_reasons": qa_reject_reasons,
        }
        return win_cand, trace

    # No repair passed the truthful gate. Fall back to adaptive cover, which
    # already self-checks the rectangularity gate (Patch 4).
    t0 = time.time()
    fallback_tool = _TOOL_INDEX["final_adaptive_cover"]
    fallback_candidate = fallback_tool.apply(ctx)
    fallback_qa = run_local_qa(ctx, fallback_candidate)

    # Patch 7 — auto_cover_retry: if the watermark is still readable through
    # the cover, escalate to a stronger stroke/segmented cover that fully
    # replaces the watermark pixels. No manual review.
    if not fallback_qa.residual_pass:
        retry = None
        stroke_mask = _get_effective_stroke_mask(ctx, dilate_px=2)
        if stroke_mask is not None and np.any(stroke_mask > 0):
            retry = fallback_tool._stroke_level_cover(ctx, stroke_mask)
        elif ctx.product_mask is not None:
            retry = fallback_tool._segmented_cover(ctx)
        if retry is not None:
            rq = run_local_qa(ctx, retry)
            if rq.residual_pass or rq.final_score <= fallback_qa.final_score:
                retry.metadata.setdefault("cover_style", "auto_cover_retry")
                fallback_candidate, fallback_qa = retry, rq
    dt = round(time.time() - t0, 3)

    trace.append({
        "tool": "final_adaptive_cover", "passed": True,
        "reason": "final_fallback", "qa": asdict(fallback_qa),
        "method_family": MethodFamily.ADAPTIVE_COVER.value,
        "runtime_s": dt})

    cover_meta = dict(fallback_candidate.metadata)
    cover_meta["qa"] = asdict(fallback_qa)
    cover_meta["final_method"] = "final_adaptive_cover"
    cover_meta["method_family"] = MethodFamily.ADAPTIVE_COVER.value
    cover_meta["is_real_repair"] = False
    cover_meta["telemetry"] = {
        "roi_class": ctx.roi_analysis.roi_class,
        "strategy_list": strategy_list,
        "tools_attempted": tools_attempted,
        "families_attempted": sorted(families_attempted),
        "final_method": "final_adaptive_cover",
        "qa_reject_reasons": qa_reject_reasons,
    }
    fallback_candidate.metadata = cover_meta
    return fallback_candidate, trace


# ============================================================================
# Debug output
# ============================================================================

def save_debug_trace(debug_dir: str | Path, filename: str,
                     roi_class: str, bbox: tuple,
                     trace: list, final_method: str,
                     method_family: str = "",
                     roi_features: dict | None = None,
                     telemetry: dict | None = None,
                     qa: dict | None = None):
    debug_path = Path(debug_dir) / filename.replace(".", "_")
    debug_path.mkdir(parents=True, exist_ok=True)

    non_cover_tried = [t for t in trace
                       if t.get("tool") != "final_adaptive_cover"
                       and t.get("reason") not in ("not_applicable",
                                                    "blocked_by_product_overlap_guard")]
    non_cover_passed = [t for t in non_cover_tried if t.get("passed")]
    family = method_family or get_method_family(final_method).value

    trace_data = {
        "filename": filename,
        "roi_class": roi_class,
        "watermark_bbox": list(bbox),
        "final_method": final_method,
        "method_family": family,
        "is_real_repair": family != MethodFamily.ADAPTIVE_COVER.value,
        "is_cover": family == MethodFamily.ADAPTIVE_COVER.value,
        "tools_tried_count": len(trace),
        "non_cover_tools_tried_count": len(non_cover_tried),
        "non_cover_tools_passed_count": len(non_cover_passed),
        "tools_tried": trace,
    }
    if roi_features:
        trace_data["roi_features"] = roi_features
    if telemetry:
        trace_data["telemetry"] = telemetry
    if qa:
        # Patch 9 — surface the truthful gate verdicts for diagnosis.
        trace_data["qa_metrics_valid"] = qa.get("metrics_valid")
        trace_data["residual_pass"] = qa.get("residual_pass")
        trace_data["residual_template_corr"] = qa.get("residual_template_corr")
        trace_data["residual_text_component"] = qa.get("residual_text_component")
        trace_data["cover_visibility_pass"] = (
            qa.get("cover_visibility_score", 1.0) <= COVER_VISIBILITY_MAX)
        trace_data["cover_rectangularity"] = qa.get("cover_rectangularity")
        trace_data["final_qa"] = qa

    (debug_path / "trace.json").write_text(
        json.dumps(trace_data, indent=2, default=str))
    return debug_path
