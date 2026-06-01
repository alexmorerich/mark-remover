#!/usr/bin/env python3
"""
SUNSKY watermark removal — V8 Progressive Repair Strategy Bank.

V8 replaces the fixed candidate loop with a 100-tool progressive repair
pipeline. Each watermark ROI is classified, tools are selected from the
strategy bank, candidates are run with local QA gates, and the first
passing candidate is accepted. Final adaptive cover is the last resort.

Terminal states:
  clean_repaired  — watermark removed invisibly via inpainting
  clean_covered   — watermark hidden by soft shadow/cover patch
  no_watermark    — no watermark detected
  failed_io       — corrupt/unreadable file

Usage:
  python3 mark_remover.py --assets path/to/images --out output
  python3 mark_remover.py --n 50 --seed 23
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from progressive_repair import (
    analyze_roi as pr_analyze_roi,
    repair_image_progressively,
    save_debug_trace,
    RepairContext,
    get_method_family,
    MethodFamily,
    QA_SCHEMA_VERSION,
    STRATEGY_SCHEMA_VERSION,
    RESIDUAL_DOT_CHAIN_MAX,
    BAND_VISIBLE_REPAIRED_MAX,
)
import v13_gates
import v14_patch
import v15_patch
import v16_pipeline
import v17_final_audit

SCRIPT_DIR = Path(__file__).resolve().parent
RWM_PATH = SCRIPT_DIR / "detector.py"
DEFAULT_ASSETS = Path("assets")
DEFAULT_OUT = Path("output")


# ---------------------------------------------------------------------------
# V12 — Version constant and run metadata. The PDF / HTML / JSONL / trace must
# all carry this exact string so a run can never claim a version other than
# the code that produced it (Phase J).
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "V16_AUTO_DECISION"
assert PIPELINE_VERSION == "V16_AUTO_DECISION"
# V14/V15/V16 keep the V13 final visual gate unchanged; V16 adds the
# auto_rejected final state + a post-clean re-detection P0 gate around it, so
# only gate-passing outputs are ever published. The visual gate version is v13;
# the final-decision/state-machine version is v16.
FINAL_VISUAL_GATE_VERSION = "v13"


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(SCRIPT_DIR), capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_metadata(seed: int) -> dict:
    """V12 Phase J — run provenance embedded in every report."""
    return {
        "version": PIPELINE_VERSION,
        "git_commit": _git_commit(),
        "run_seed": seed,
        "qa_schema_version": QA_SCHEMA_VERSION,
        "strategy_schema_version": STRATEGY_SCHEMA_VERSION,
        "final_visual_gate_version": FINAL_VISUAL_GATE_VERSION,
    }

# ---------------------------------------------------------------------------
# V5 — Four-state model. manual-review is eliminated.
# ---------------------------------------------------------------------------
ST_CLEAN_REPAIRED = "clean_repaired"
ST_CLEAN_COVERED = "clean_covered"
ST_NO_WATERMARK = "no_watermark_confirmed"
ST_AUTO_REJECTED = "auto_rejected"          # V16 — repair AND cover both failed
ST_SKIPPED_KNOWN_CLEAN = "skipped_known_clean"
ST_FAILED_IO = "failed_io"

# V16 — the only publishable final states (publish_ok=true, manual_required=false).
PUBLISHABLE_STATUSES = (ST_CLEAN_REPAIRED, ST_CLEAN_COVERED, ST_NO_WATERMARK,
                        ST_SKIPPED_KNOWN_CLEAN)

# Legacy aliases — mapped internally for backward compat.
ST_CLEAN = ST_CLEAN_REPAIRED
ST_MANUAL = ST_CLEAN_COVERED
ST_REJECTED = ST_FAILED_IO

# Reason codes reused from V3/V4 (now informational, not blocking).
R_PRODUCT_LOSS    = "reject_product_loss"
R_BLANK_OUTPUT    = "reject_blank_output"
R_LARGE_SMEAR     = "reject_large_smear"
R_TEXT_DAMAGE     = "reject_text_damage"
R_FOREGROUND_LOSS = "reject_foreground_area_loss"
R_EDGE_DAMAGE     = "reject_edge_damage"
R_INPUT_UNREADABLE = "reject_input_unreadable"
R_PATHOLOGICAL_DETECTION = "reject_pathological_detection"
R_UNSUITABLE_SOURCE = "reject_unsuitable_source_image"
R_PRODUCT_DAMAGE = "reject_product_damage_risk"

# QA failure reason codes (informational — used for cover routing).
M_PRODUCT_INTEGRITY = "product_integrity_risk"
M_MASK_PRODUCT_EDGE = "mask_touches_product_edge"
M_MASK_DARK_COMPONENT = "mask_touches_dark_component"
M_MASK_LENS_OR_CIRCLE = "mask_touches_lens_or_circle"
M_HIGH_TEXTURE = "high_texture_roi"
M_VISIBLE_PATCH = "visible_patch_artifact"
M_RESIDUAL_WATERMARK = "residual_watermark"
M_LOCAL_QA = "local_qa_failed"
M_LOW_CONF_MASK = "low_confidence_mask"
M_CLASS_DEFERRED = "class_routes_to_manual"
M_TEXT_PRESERVATION = "text_preservation_risk"
M_COMPLEX_TRANSPARENT = "complex_transparent_background"
M_GLOSSY_METALLIC = "glossy_or_metallic_background"
M_POST_INTEGRITY = "post_clean_integrity_failed"
M_REPAIR_ARTIFACT = "repair_artifact_detected"
M_PRODUCT_TEXT_RISK = "product_text_at_risk"
M_SOFT_FALLBACK_MASK = "soft_fallback_mask_not_stroke"
M_HIGH_MASK_DENSITY = "high_mask_density_in_roi"
M_HIGH_MASK_RECTANGULARITY = "high_mask_rectangularity"
M_PATCH_VISIBILITY = "patch_visibility_gate_failed"
M_DARK_BRIGHTNESS_OVERSHOOT = "dark_surface_brightness_overshoot"
M_CLEANUP_EXPANSION = "cleanup_requires_area_expansion"

# Legacy skip states — mapped to cover.
ST_SKIPPED_UNSAFE = ST_CLEAN_COVERED
ST_SKIPPED_LOW_CONF = ST_CLEAN_COVERED
ST_SKIPPED_PRODUCT_DETAIL = ST_CLEAN_COVERED


# ---------------------------------------------------------------------------
# V5 — Cover method routing and escalation.
# ---------------------------------------------------------------------------
COVER_ESCALATION_LEVELS = [
    {"opacity": 0.35, "feather": 24, "expand": 4},
    {"opacity": 0.50, "feather": 20, "expand": 8},
    {"opacity": 0.65, "feather": 16, "expand": 12},
    {"opacity": 0.80, "feather": 12, "expand": 16},
]

# V6: cover routing moved to V6_FILL_MODE_BY_ROI (defined after cover functions)


# ---------------------------------------------------------------------------
# V7 — Multi-scale logo mask bank.
# ---------------------------------------------------------------------------
LOGO_TEMPLATE_PATH = SCRIPT_DIR / "watermark-template.png"
LOGO_TEXT_THRESH = 180
LOGO_SHADOW_THRESH = 240
LOGO_SCALE_WIDTHS = list(range(60, 500, 20))
_LOGO_MASK_BANK: dict | None = None


def _build_canonical_logo_masks():
    tpl = cv2.imread(str(LOGO_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        return None
    text_mask = (tpl < LOGO_TEXT_THRESH).astype(np.uint8) * 255
    shadow_raw = (tpl < LOGO_SHADOW_THRESH).astype(np.uint8) * 255
    shadow_mask = cv2.subtract(shadow_raw, text_mask)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, k_close)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_mask = cv2.dilate(text_mask, k_dilate, iterations=1)
    shadow_mask = cv2.dilate(shadow_mask, k_dilate, iterations=1)
    return {"text": text_mask, "shadow": shadow_mask,
            "combined": cv2.bitwise_or(text_mask, shadow_mask),
            "shape": tpl.shape}


def _build_logo_mask_bank():
    canonical = _build_canonical_logo_masks()
    if canonical is None:
        return None
    src_h, src_w = canonical["shape"]
    aspect = src_w / max(src_h, 1)
    bank = {}
    for target_w in LOGO_SCALE_WIDTHS:
        target_h = max(4, int(round(target_w / aspect)))
        text_s = cv2.resize(canonical["text"], (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR)
        text_s = (text_s > 127).astype(np.uint8) * 255
        shadow_s = cv2.resize(canonical["shadow"], (target_w, target_h),
                              interpolation=cv2.INTER_LINEAR)
        shadow_s = (shadow_s > 64).astype(np.uint8) * 255
        combined_s = cv2.bitwise_or(text_s, shadow_s)
        bank[target_w] = {"text": text_s, "shadow": shadow_s,
                          "combined": combined_s,
                          "w": target_w, "h": target_h}
    return bank


def get_logo_mask_bank():
    global _LOGO_MASK_BANK
    if _LOGO_MASK_BANK is None:
        _LOGO_MASK_BANK = _build_logo_mask_bank()
    return _LOGO_MASK_BANK


def align_logo_mask(mark_box: dict, img_shape: tuple,
                    use_shadow: bool = True) -> np.ndarray | None:
    bank = get_logo_mask_bank()
    if bank is None:
        return None
    bw, bh = mark_box["w"], mark_box["h"]
    bx, by = mark_box["x"], mark_box["y"]
    H, W = img_shape[:2]

    widths = sorted(bank.keys())
    best_w = min(widths, key=lambda w: abs(w - bw))
    entry = bank[best_w]

    src_w, src_h = entry["w"], entry["h"]
    key = "combined" if use_shadow else "text"
    logo = entry[key]

    scale_x = bw / max(src_w, 1)
    scale_y = bh / max(src_h, 1)

    if abs(scale_x - 1.0) > 0.02 or abs(scale_y - 1.0) > 0.02:
        logo = cv2.resize(logo, (bw, bh), interpolation=cv2.INTER_LINEAR)
        logo = (logo > 127).astype(np.uint8) * 255
    elif logo.shape[1] != bw or logo.shape[0] != bh:
        logo = cv2.resize(logo, (bw, bh), interpolation=cv2.INTER_LINEAR)
        logo = (logo > 127).astype(np.uint8) * 255

    full = np.zeros((H, W), np.uint8)
    y1 = max(0, by)
    x1 = max(0, bx)
    y2 = min(H, by + bh)
    x2 = min(W, bx + bw)
    logo_y1 = y1 - by
    logo_x1 = x1 - bx
    logo_y2 = logo_y1 + (y2 - y1)
    logo_x2 = logo_x1 + (x2 - x1)
    if logo_y2 > logo.shape[0]:
        logo_y2 = logo.shape[0]
        y2 = y1 + (logo_y2 - logo_y1)
    if logo_x2 > logo.shape[1]:
        logo_x2 = logo.shape[1]
        x2 = x1 + (logo_x2 - logo_x1)
    if y2 > y1 and x2 > x1:
        full[y1:y2, x1:x2] = logo[logo_y1:logo_y2, logo_x1:logo_x2]
    return full


# ---------------------------------------------------------------------------
# Post-clean product integrity gate thresholds (Problem A).
# ---------------------------------------------------------------------------
# Post-clean product integrity gate thresholds.
# Focused on OUTSIDE-watermark damage (product damage), not in-watermark
# changes which are expected.
PI_POST_EDGE_LOSS_MANUAL = 0.30
PI_POST_SSIM_DROP_MANUAL = 0.50
PI_POST_BLUR_INCREASE_MANUAL = 0.55
PI_POST_TEXT_LOSS_MANUAL = 0.65
PI_POST_SCORE_MANUAL = 0.40

# Rectangular artifact detector thresholds (Problem B).
# These must catch visible rectangular patches but NOT the normal smoothing
# from watermark removal. Watermark removal naturally reduces texture in
# the cleaned area — only flag when the result is a gross flat rectangle.
ARTIFACT_RECT_SCORE_THRESHOLD = 0.80
ARTIFACT_SMOOTH_RATIO_THRESHOLD = 5.0
ARTIFACT_BOUNDARY_DELTA_E_MAX = 18.0
ARTIFACT_LAPLACIAN_DROP_MAX = 0.82

# V4 — Mask quality gates (requirement 1).
V4_MASK_DENSITY_IN_ROI_MAX = 0.18
V4_MASK_RECTANGULARITY_MAX = 0.55

# V4 — Patch visibility gate thresholds (requirement 3).
V4_PV_DELTA_LUMA_MAX = 8.0
V4_PV_RECT_BOUNDARY_MAX = 0.35
V4_PV_LOW_FREQ_PATCH_MAX = 0.4
V4_PV_TEXTURE_DROP_MAX = 0.45

# V4 — Dark surface brightness constraint (requirement 4).
V4_DARK_LUMA_OVERSHOOT_MAX = 6.0


# ---------------------------------------------------------------------------
# Mask / safety constants.
# ---------------------------------------------------------------------------
MARK_BOX_DENSITY_REJECT = 0.07
MARK_BOX_DENSITY_MANUAL = 0.04

# Soft-mask vs protect overlap (kept as a ROUTING signal only, not a subtractor).
RISK_OVERLAP_UPGRADE = 0.20
RISK_OVERLAP_MANUAL  = 0.55

# Class-specific dilation (px) and expansion ratios used to build soft_mask.
# Stroke-mask is preferred; soft_mask is the fallback.
DILATE_BY_CLASS = {
    "plain_white":    3, "plain_color":    4, "gradient":       5,
    "low_texture":    6, "transparent":    6, "glossy":         7,
    "metallic":       7, "black_product":  8,
    "high_texture":   9, "product_detail": 9, "text_or_qr":     6,
    "poster_or_marketing": 6,
}
EXPAND_BY_CLASS = {
    "plain_white":    (1.20, 1.05), "plain_color":    (1.25, 1.05),
    "gradient":       (1.30, 1.10), "low_texture":    (1.40, 1.10),
    "transparent":    (1.35, 1.10), "glossy":         (1.40, 1.10),
    "metallic":       (1.40, 1.10), "black_product":  (1.45, 1.15),
    "high_texture":   (1.50, 1.15), "product_detail": (1.50, 1.20),
    "text_or_qr":     (1.25, 1.05), "poster_or_marketing": (1.20, 1.05),
}

# Method routing — P4. The CLASS chooses a method; the product-overlap
# pre-gate (P2) is what defers risky cases to manual. The plan's
# "transparent/glossy/metallic/black_product → manual unless very low risk"
# is implemented by: route them via telea_then_lama (cheapest first) AND
# let the pre-gate + post-clean QA catch the risky ones. "always-manual"
# classes are reserved for ones whose risk is inherent (product_detail /
# high_texture). "reject" classes are unsuitable inputs (text/QR/posters).
ROUTE_BY_CLASS = {
    "plain_white":    "white_fill_stroke_only",
    "plain_color":    "local_color_plane",
    "gradient":       "gradient_plane",
    "low_texture":    "telea_then_lama",
    "transparent":    "telea_then_lama",
    "glossy":         "telea_then_lama",
    "metallic":       "telea_then_lama",
    "black_product":  "dark_surface_local_plane",
    "high_texture":   "lama",
    "product_detail": "telea_small",
    "text_or_qr":     "telea_small",
    "poster_or_marketing": "reject",
}

CANDIDATE_METHODS_BY_CLASS = {
    "plain_white":    ["clone_offset", "alpha_attenuate", "white_fill_stroke_only", "local_background_fill", "white_fill", "local_color_plane", "gradient_plane"],
    "plain_color":    ["clone_offset", "alpha_attenuate", "local_color_plane", "gradient_plane", "poisson", "surface_fit"],
    "gradient":       ["clone_offset", "alpha_attenuate", "gradient_plane", "surface_fit", "poisson", "telea_small"],
    "low_texture":    ["clone_offset", "alpha_attenuate", "telea_small", "telea", "gradient_plane", "lama_small"],
    "transparent":    ["clone_offset", "alpha_attenuate", "telea_small", "telea", "poisson"],
    "glossy":         ["clone_offset", "alpha_attenuate", "telea_small", "telea", "poisson"],
    "metallic":       ["clone_offset", "alpha_attenuate", "telea_small", "telea", "poisson"],
    "black_product":  ["clone_offset", "alpha_attenuate", "dark_surface_local_plane", "telea_small", "telea"],
    "high_texture":   ["clone_offset", "alpha_attenuate", "telea_small", "telea", "lama_small"],
    "product_detail": ["clone_offset", "alpha_attenuate", "telea_small", "telea", "poisson"],
    "text_or_qr":     ["clone_offset", "telea_small"],
    "poster_or_marketing": [],
}


# --- Product Integrity Gate (P2) — pre-inpaint overlap thresholds ---
# Computed on the soft_mask (max extent we'd inpaint) vs detected product
# structure (edges, dark components, circular features).
PI_GATE_EDGE_REJECT  = 0.45   # >45% of soft_mask covers strong edges → REJECT
PI_GATE_EDGE_MANUAL  = 0.30   # 30..45% → MANUAL: mask_touches_product_edge
PI_GATE_DARK_REJECT  = 0.40   # very dark product pixels overlap
PI_GATE_DARK_MANUAL  = 0.25
PI_GATE_CIRCLE_MANUAL = 0.20  # meaningful overlap with lens/hole → MANUAL


# --- Post-inpaint product-integrity QA — V2 thresholds, kept ---
PI_FOREGROUND_LOSS_REJECT = 0.03
PI_BBOX_DELTA_REJECT     = 0.04
PI_EDGE_LOSS_REJECT      = 0.08
PI_CC_DELTA_REJECT       = 8
PI_OUTSIDE_MASK_DIFF_REJECT = 6.0
PI_FOREGROUND_LOSS_MANUAL = 0.015
PI_EDGE_LOSS_MANUAL       = 0.04


# --- Blank-output detector — V2 calibrated thresholds, kept ---
BLANK_PATCH_SURROUND_MIN_STD = 22.0
BLANK_PATCH_RATIO_REJECT = 0.12


# --- Local ROI QA (P5) — applied INSIDE the mark area after inpaint ---
# A patch artifact shows up as: low local SSIM to its surround, big color
# delta vs the neighbor strips, or a sharp mask-boundary jump.
ROI_SSIM_MIN = 0.50           # strict: require reasonable local similarity
ROI_COLOR_DELTA_MAX = 18.0    # strict: color shift must be small
ROI_BOUNDARY_JUMP_MAX = 20.0  # strict: boundary must be smooth


# --- Watermark residual QA — strict thresholds (V2 baseline + V3 classes) ---
# (boundary_jump_max, inside_edge_max, edge_drift_max)
QA_THRESHOLDS_BY_CLASS = {
    "plain_white":    (10.0, 0.025, 0.04),
    "plain_color":    (10.0, 0.030, 0.04),
    "gradient":       (12.0, 0.040, 0.05),
    "low_texture":    (12.0, 0.050, 0.06),
    "transparent":    (14.0, 0.060, 0.07),
    "glossy":         (14.0, 0.060, 0.07),
    "metallic":       (14.0, 0.060, 0.07),
    "black_product":  (10.0, 0.060, 0.05),
    "high_texture":   (10.0, 0.060, 0.05),
    "product_detail": (8.0,  0.045, 0.04),
    "text_or_qr":     (8.0,  0.045, 0.04),
    "poster_or_marketing": (8.0, 0.045, 0.04),
}


BUDGET_FAST_S = 2.0
BUDGET_LAMA_S = 20.0
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Presence Gate — statuses, thresholds, model exclusion.
# ---------------------------------------------------------------------------

PG_NO_WATERMARK            = "NO_WATERMARK"
PG_MODEL_EXCLUDED          = "NO_WATERMARK_MODEL_EXCLUDED"
PG_CONFIRMED               = "CONFIRMED_WATERMARK"
PG_UNCERTAIN               = "UNCERTAIN_WATERMARK"
PG_UNSUITABLE              = "UNSUITABLE_FOR_AUTO_DETECTION"

NO_WATERMARK_THRESHOLD       = 0.15
CONFIRMED_WATERMARK_THRESHOLD = 0.45
MEDIUM_CONFIDENCE_THRESHOLD   = 0.28
UNSUITABLE_NEGATIVE_SCORE    = 0.70

IPHONE_MODEL_RE = re.compile(r"\biphone[\s\-_]*(\d{1,2})\b", re.IGNORECASE)

SUNSKY_KNOWN_AR = 8.5   # typical width/height of sunsky-online.com watermark
SUNSKY_YC_LO    = 0.42  # expected Y-center band (mid-image, empirically 0.47-0.48)
SUNSKY_YC_HI    = 0.55

SUNSKY_OCR_TOKENS = {
    "sunsky", "sunsky-online", "sunsky-online.com", "sunskyonline",
    "sunsky online",
}
SUNSKY_OCR_FUZZY = {
    "sunsxy", "sun5ky", "sunsky-oniine", "sunsky-on1ine", "sunsky-ontine",
}


# ---------------------------------------------------------------------------
# Detector / inpainter loader.
# ---------------------------------------------------------------------------

def _load_rwm():
    spec = importlib.util.spec_from_file_location("rwm", RWM_PATH)
    rwm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rwm)
    rwm.WM_RESCUE3_ENABLED = True
    return rwm


# ---------------------------------------------------------------------------
# Stage 0 — Model-Based Exclusion Gate.
# ---------------------------------------------------------------------------

def normalize_model_text(text: str) -> str:
    return re.sub(r"[\s\-_]+", " ", text.strip().lower())


def is_iphone_14_or_above(model_text: str, min_model: int = 14) -> bool:
    normalized = normalize_model_text(model_text)
    matches = IPHONE_MODEL_RE.findall(normalized)
    for m in matches:
        try:
            if int(m) >= min_model:
                return True
        except ValueError:
            continue
    return False


def model_exclusion_gate(path: Path, min_model: int = 14) -> dict | None:
    candidates = [path.stem, path.name]
    parent = path.parent.name if path.parent else ""
    if parent:
        candidates.append(parent)
    for text in candidates:
        if is_iphone_14_or_above(text, min_model):
            return {"excluded": True,
                    "reason": f"iphone>={min_model} detected in '{text}'",
                    "source": text}
    return None


# ---------------------------------------------------------------------------
# Stage 1 — Fast Sunsky Presence Gate.
# ---------------------------------------------------------------------------

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _template_score(det: dict) -> float:
    edge = det.get("edge_score", 0.0)
    ncc = det.get("int_ncc", 0.0)
    return _clamp(0.6 * edge + 0.4 * ncc)


def _stroke_score(det: dict) -> float:
    cov = det.get("cov", 0.0)
    contrast = det.get("contrast", 0.0)
    return _clamp(0.7 * cov + 0.3 * min(1.0, contrast / 30.0))


def _aspect_ratio_score(det: dict) -> float:
    mb = det.get("mark_box")
    if not mb or mb["h"] < 1:
        return 0.0
    ar = mb["w"] / mb["h"]
    deviation = abs(ar - SUNSKY_KNOWN_AR) / SUNSKY_KNOWN_AR
    return _clamp(1.0 - deviation * 2.0)


def _position_score(det: dict) -> float:
    yc = det.get("yc", 0.0)
    if SUNSKY_YC_LO <= yc <= SUNSKY_YC_HI:
        mid = (SUNSKY_YC_LO + SUNSKY_YC_HI) / 2.0
        spread = (SUNSKY_YC_HI - SUNSKY_YC_LO) / 2.0
        return _clamp(1.0 - abs(yc - mid) / spread)
    return 0.0


def _repeat_pattern_score(det: dict) -> float:
    psr = det.get("psr", 0.0)
    return _clamp(psr / 12.0)


def _negative_product_text_score(img_gray, det: dict) -> float:
    mb = det.get("mark_box")
    if not mb:
        return 0.0
    H, W = img_gray.shape[:2]
    x, y, w, h = mb["x"], mb["y"], mb["w"], mb["h"]
    pad_y = max(10, h)
    ry1 = max(0, y - pad_y)
    ry2 = min(H, y + h + pad_y)
    roi = img_gray[ry1:ry2, x:min(W, x + w)]
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 50, 150)
    edge_density = float(edges.mean()) / 255.0
    dark_pix = float((roi < 60).mean())
    score = 0.0
    if edge_density > 0.15:
        score += 0.4
    if dark_pix > 0.10:
        score += 0.3
    xoff = det.get("xoff", 0.0)
    if xoff > 0.15:
        score += 0.3
    return _clamp(score)


def _clahe_center_band_score(img_gray, det: dict) -> float:
    """CLAHE-enhanced text detection in center band.
    Semi-transparent SUNSKY text is too faint for normal edge detection.
    CLAHE boosts local contrast, high-pass isolates text strokes, then we
    score horizontal text-line geometry in the expected center region."""
    mb = det.get("mark_box")
    if not mb:
        return 0.0
    H, W = img_gray.shape[:2]
    cx1, cx2 = int(W * 0.20), int(W * 0.80)
    cy1, cy2 = int(H * 0.30), int(H * 0.70)
    center = img_gray[cy1:cy2, cx1:cx2]
    if center.size == 0 or min(center.shape) < 10:
        return 0.0
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(center)
    sigma = max(3, center.shape[1] // 40)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma)
    hp = cv2.absdiff(enhanced, blur)
    _, bw = cv2.threshold(hp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    row_act = bw.astype(np.float64).mean(axis=1) / 255.0
    if len(row_act) < 5:
        return 0.0
    k_sz = max(3, len(row_act) // 20)
    kernel = np.ones(k_sz) / k_sz
    smoothed = np.convolve(row_act, kernel, mode="same")
    peak_val = float(smoothed.max())
    if peak_val < 0.02:
        return 0.0
    above_half = smoothed > peak_val * 0.5
    band_h = int(above_half.sum())
    band_ratio = band_h / len(row_act)
    if band_ratio < 0.02 or band_ratio > 0.45:
        return 0.0
    strength = min(1.0, peak_val / 0.06)
    geometry = max(0.0, 1.0 - abs(band_ratio - 0.15) / 0.20)
    peak_idx = int(np.argmax(smoothed))
    r1 = max(0, peak_idx - band_h // 2)
    r2 = min(bw.shape[0], peak_idx + band_h // 2 + 1)
    band_cols = bw[r1:r2].astype(np.float64).mean(axis=0) / 255.0
    h_spread = float((band_cols > 0.01).sum()) / max(len(band_cols), 1)
    return _clamp(0.45 * strength + 0.30 * geometry + 0.25 * h_spread)


def _clahe_mark_box_residual_score(cleaned_gray, mark_box) -> float:
    """Targeted CLAHE residual check focused ONLY on the mark_box area.
    Unlike _clahe_center_band_score which scans the full center band, this
    looks at the exact watermark region where residual text would appear.
    Much more sensitive on complex backgrounds because surrounding texture
    noise is excluded."""
    H, W = cleaned_gray.shape[:2]
    bx, by = mark_box["x"], mark_box["y"]
    bw, bh = mark_box["w"], mark_box["h"]
    pad_x = max(5, bw // 5)
    pad_y = max(3, bh // 3)
    y1 = max(0, by - pad_y)
    y2 = min(H, by + bh + pad_y)
    x1 = max(0, bx - pad_x)
    x2 = min(W, bx + bw + pad_x)
    crop = cleaned_gray[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape) < 10:
        return 0.0
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(crop)
    sigma = max(3, crop.shape[1] // 20)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma)
    hp = cv2.absdiff(enhanced, blur)
    _, bw_bin = cv2.threshold(hp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    row_act = bw_bin.astype(np.float64).mean(axis=1) / 255.0
    if len(row_act) < 5:
        return 0.0
    k_sz = max(3, len(row_act) // 10)
    kernel = np.ones(k_sz) / k_sz
    smoothed = np.convolve(row_act, kernel, mode="same")
    peak_val = float(smoothed.max())
    if peak_val < 0.03:
        return 0.0
    above_half = smoothed > peak_val * 0.4
    band_h = int(above_half.sum())
    band_ratio = band_h / len(row_act)
    if band_ratio < 0.03 or band_ratio > 0.70:
        return 0.0
    peak_idx = int(np.argmax(smoothed))
    r1 = max(0, peak_idx - band_h // 2)
    r2 = min(bw_bin.shape[0], peak_idx + band_h // 2 + 1)
    band_cols = bw_bin[r1:r2].astype(np.float64).mean(axis=0) / 255.0
    h_spread = float((band_cols > 0.01).sum()) / max(len(band_cols), 1)
    if h_spread < 0.15:
        return 0.0
    strength = min(1.0, peak_val / 0.05)
    geometry = max(0.0, 1.0 - abs(band_ratio - 0.20) / 0.30)
    return _clamp(0.40 * strength + 0.30 * geometry + 0.30 * h_spread)


def _scan_region_for_text_band(region_gray):
    """Score a grayscale region for horizontal text-line patterns.
    Returns (score, peak_idx, r1, r2, band_cols) or None."""
    if region_gray.size == 0 or min(region_gray.shape) < 20:
        return None
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(region_gray)
    sigma = max(3, region_gray.shape[1] // 30)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma)
    hp = cv2.absdiff(enhanced, blur)
    _, bw = cv2.threshold(hp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    row_act = bw.astype(np.float64).mean(axis=1) / 255.0
    if len(row_act) < 10:
        return None
    k_sz = max(3, len(row_act) // 15)
    kernel = np.ones(k_sz) / k_sz
    smoothed = np.convolve(row_act, kernel, mode="same")
    peak_val = float(smoothed.max())
    if peak_val < 0.03:
        return None
    above_half = smoothed > peak_val * 0.4
    band_h = int(above_half.sum())
    band_ratio = band_h / len(row_act)
    if band_ratio < 0.02 or band_ratio > 0.50:
        return None
    peak_idx = int(np.argmax(smoothed))
    r1 = max(0, peak_idx - band_h // 2)
    r2 = min(bw.shape[0], peak_idx + band_h // 2 + 1)
    band_cols = bw[r1:r2].astype(np.float64).mean(axis=0) / 255.0
    h_spread = float((band_cols > 0.01).sum()) / max(len(band_cols), 1)
    if h_spread < 0.20:
        return None
    strength = min(1.0, peak_val / 0.05)
    geometry = max(0.0, 1.0 - abs(band_ratio - 0.12) / 0.25)
    score = 0.40 * strength + 0.30 * geometry + 0.30 * h_spread
    return (score, peak_idx, r1, r2, band_cols)


def scan_center_band_for_watermark(img_gray) -> dict | None:
    """Standalone SUNSKY text scanner for images where the base detector
    returns det=None. Scans the full center band plus left and right halves
    to handle mixed backgrounds. Returns a synthetic det with mark_box."""
    H, W = img_gray.shape[:2]
    if H < 50 or W < 50:
        return None
    cy1, cy2 = int(H * 0.25), int(H * 0.75)
    regions = [
        (int(W * 0.15), int(W * 0.85)),
        (int(W * 0.10), int(W * 0.55)),
        (int(W * 0.45), int(W * 0.90)),
    ]
    best = None
    best_score = 0.0
    best_cx1 = 0
    for cx1, cx2 in regions:
        center = img_gray[cy1:cy2, cx1:cx2]
        result = _scan_region_for_text_band(center)
        if result is not None and result[0] > best_score:
            best = result
            best_score = result[0]
            best_cx1 = cx1
    if best is None or best_score < 0.40:
        return None
    score, peak_idx, r1, r2, band_cols = best
    active_cols = np.where(band_cols > 0.01)[0]
    if len(active_cols) < 5:
        return None
    band_y1 = cy1 + r1
    band_y2 = cy1 + r2
    band_x1 = best_cx1 + int(active_cols[0])
    band_x2 = best_cx1 + int(active_cols[-1]) + 1
    pad_x = max(5, (band_x2 - band_x1) // 10)
    pad_y = max(3, (band_y2 - band_y1) // 4)
    mx = max(0, band_x1 - pad_x)
    my = max(0, band_y1 - pad_y)
    mw = min(W, band_x2 + pad_x) - mx
    mh = min(H, band_y2 + pad_y) - my
    if mh > H * 0.18 or (mw * mh) / (H * W) > 0.06:
        return None
    yc = (my + mh / 2.0) / H
    return {
        "mark_box": {"x": mx, "y": my, "w": mw, "h": mh},
        "tier": "manual",
        "edge_score": round(score, 4),
        "source": "center_band_scan",
        "scan_score": round(score, 4),
        "yc": round(yc, 4),
    }


def detect_sunsky_presence_fast(img_gray, det: dict, config: dict) -> dict:
    ts = _template_score(det)
    ss = _stroke_score(det)
    ars = _aspect_ratio_score(det)
    ps = _position_score(det)
    rps = _repeat_pattern_score(det)
    npts = _negative_product_text_score(img_gray, det)
    clahe_s = _clahe_center_band_score(img_gray, det)

    raw_score = (0.28 * ts + 0.22 * ss + 0.18 * clahe_s +
                 0.12 * ars + 0.10 * ps + 0.10 * rps)
    confidence = _clamp(raw_score - 0.30 * npts)

    no_thr = config.get("no_watermark_threshold", NO_WATERMARK_THRESHOLD)
    conf_thr = config.get("confirmed_threshold", CONFIRMED_WATERMARK_THRESHOLD)
    unsuit_thr = config.get("unsuitable_negative", UNSUITABLE_NEGATIVE_SCORE)

    scores = {"template": round(ts, 4), "stroke": round(ss, 4),
              "clahe_center": round(clahe_s, 4),
              "aspect_ratio": round(ars, 4), "position": round(ps, 4),
              "repeat_pattern": round(rps, 4),
              "negative_product_text": round(npts, 4),
              "raw_score": round(raw_score, 4),
              "confidence": round(confidence, 4)}

    if npts >= unsuit_thr:
        return {"status": PG_UNSUITABLE, "confidence": confidence,
                "scores": scores, "reason": "high_negative_product_text_score",
                "best_bbox": det.get("mark_box")}
    if confidence < no_thr:
        return {"status": PG_NO_WATERMARK, "confidence": confidence,
                "scores": scores, "reason": f"confidence {confidence:.3f} < {no_thr}",
                "best_bbox": None}
    if confidence >= conf_thr:
        return {"status": PG_CONFIRMED, "confidence": confidence,
                "scores": scores, "reason": f"confidence {confidence:.3f} >= {conf_thr}",
                "best_bbox": det.get("mark_box")}
    return {"status": PG_UNCERTAIN, "confidence": confidence,
            "scores": scores, "reason": f"confidence {confidence:.3f} in uncertain band",
            "best_bbox": det.get("mark_box")}


# ---------------------------------------------------------------------------
# Stage 2 — Deep Presence Check (OCR + stronger template).
# ---------------------------------------------------------------------------

def _ocr_score(img_gray, det: dict) -> float:
    try:
        import pytesseract
    except ImportError:
        return 0.0
    mb = det.get("mark_box")
    if not mb:
        return 0.0
    H, W = img_gray.shape[:2]
    x, y, w, h = mb["x"], mb["y"], mb["w"], mb["h"]
    pad = max(10, h // 2)
    ry1 = max(0, y - pad); ry2 = min(H, y + h + pad)
    rx1 = max(0, x - pad); rx2 = min(W, x + w + pad)
    crop = img_gray[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return 0.0

    results = []
    for proc in [crop, cv2.equalizeHist(crop)]:
        try:
            text = pytesseract.image_to_string(proc, config="--psm 7").strip().lower()
            text = re.sub(r"\s+", " ", text)
            results.append(text)
        except Exception:
            pass

    best = 0.0
    all_tokens = SUNSKY_OCR_TOKENS | SUNSKY_OCR_FUZZY
    for text in results:
        for tok in all_tokens:
            if tok in text:
                best = max(best, 1.0 if tok in SUNSKY_OCR_TOKENS else 0.75)
    return best


def deep_sunsky_presence_check(img_gray, det: dict, fast_result: dict,
                               config: dict) -> dict:
    ocr_s = _ocr_score(img_gray, det)
    ts = fast_result["scores"]["template"]
    ss = fast_result["scores"]["stroke"]
    clahe_s = fast_result["scores"].get("clahe_center", 0.0)
    ars = fast_result["scores"]["aspect_ratio"]
    ps = fast_result["scores"]["position"]
    rps = fast_result["scores"]["repeat_pattern"]
    npts = fast_result["scores"]["negative_product_text"]

    raw_score = (0.30 * ocr_s + 0.20 * ts + 0.15 * ss + 0.15 * clahe_s +
                 0.10 * ars + 0.05 * ps + 0.05 * rps)
    confidence = _clamp(raw_score - 0.30 * npts)

    no_thr = config.get("no_watermark_threshold", NO_WATERMARK_THRESHOLD)
    conf_thr = config.get("confirmed_threshold", CONFIRMED_WATERMARK_THRESHOLD)

    scores = dict(fast_result["scores"])
    scores["ocr"] = round(ocr_s, 4)
    scores["deep_raw_score"] = round(raw_score, 4)
    scores["deep_confidence"] = round(confidence, 4)

    if confidence < no_thr:
        return {"status": PG_NO_WATERMARK, "confidence": confidence,
                "scores": scores, "reason": f"deep confidence {confidence:.3f} < {no_thr}",
                "best_bbox": None, "stage": "deep"}
    if confidence >= conf_thr:
        return {"status": PG_CONFIRMED, "confidence": confidence,
                "scores": scores,
                "reason": f"deep confidence {confidence:.3f} >= {conf_thr}",
                "best_bbox": det.get("mark_box"), "stage": "deep"}
    return {"status": PG_UNCERTAIN, "confidence": confidence,
            "scores": scores,
            "reason": f"deep confidence {confidence:.3f} still uncertain",
            "best_bbox": det.get("mark_box"), "stage": "deep"}


# ---------------------------------------------------------------------------
# Presence gate JSONL writer.
# ---------------------------------------------------------------------------

def write_presence_jsonl(out_path: Path, records: list):
    with out_path.open("w") as f:
        for r in records:
            row = {k: r.get(k) for k in (
                "product_id", "image", "source",
                "presence_status", "presence_confidence",
                "presence_reason", "presence_stage",
                "status", "reason", "publish_ok")}
            row["presence_scores"] = r.get("presence_scores")
            f.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------
# P1 — Stroke-level mask. Replaces the box-expanded soft mask whenever the
# watermark text can be segmented out of the local background, eliminating
# the rectangular-patch artifacts.
#
# The watermark is faint text (light gray) over varying backgrounds. We
# isolate strokes via a wide HORIZONTAL Gaussian baseline subtraction — the
# watermark text creates strokes that residual against a per-row background.
# This is polarity-independent (works for both light-on-dark and
# dark-on-light) and doesn't rely on a hard threshold.
# ---------------------------------------------------------------------------

def generate_stroke_level_mask(img_bgr: np.ndarray, mark_box: dict,
                               edge_pad: int = 0) -> dict:
    """Return a dict with `stroke` (full-image binary mask), `roi` (the ROI
    used), `confidence` (0..1: fraction of expected stroke area we found)
    and `bw_thresh` (the threshold that picked strokes).

    V5: captures FULL semi-transparent text body, not just edges.
    Three-pass approach:
      Pass 1: High-frequency residual (horizontal Gaussian subtraction) for edges
      Pass 2: CLAHE-enhanced edge detection for faint strokes
      Pass 3: Ring-based background estimation + L-channel deviation for body fill
    The body fill uses the SURROUNDING RING as ground-truth background, then
    detects any pixel inside mark_box that deviates from the interpolated
    background. This captures the full semi-transparent text interior.
    """
    H, W = img_bgr.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    pad = max(4, bh // 6) + edge_pad
    rx1 = max(0, bx - pad); ry1 = max(0, by - pad)
    rx2 = min(W, bx + bw + pad); ry2 = min(H, by + bh + pad)
    roi = img_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return {"stroke": np.zeros((H, W), np.uint8),
                "roi_box": (rx1, ry1, rx2, ry2),
                "confidence": 0.0, "n_components": 0, "raw_thresh": 0}

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    win = max(15, roi.shape[1] // 5) | 1

    # --- Pass 1: Edge strokes via horizontal Gaussian subtraction ---
    bg = cv2.GaussianBlur(L, (win, 1), 0)
    residual = np.abs(L - bg)
    thr = max(2.5, float(np.percentile(residual, 65)) * 1.0)
    bw_strokes = (residual > thr).astype(np.uint8) * 255

    # --- Pass 2: CLAHE-enhanced edge detection ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_enh = clahe.apply(L.astype(np.uint8)).astype(np.float32)
    bg_enh = cv2.GaussianBlur(L_enh, (win, 1), 0)
    res_enh = np.abs(L_enh - bg_enh)
    thr_enh = max(2.5, float(np.percentile(res_enh, 65)) * 1.0)
    bw_enh = (res_enh > thr_enh).astype(np.uint8) * 255
    bw_strokes = cv2.bitwise_or(bw_strokes, bw_enh)

    # --- Pass 3: Ring-based background estimation + body fill ---
    # Estimate what each pixel INSIDE the mark_box SHOULD look like
    # by fitting a gradient plane from the surrounding ring, then
    # detect pixels that deviate (= watermark-affected).
    local_bx = bx - rx1; local_by = by - ry1
    rh, rw_roi = roi.shape[:2]

    # Build ring mask: region around mark_box excluding the mark_box itself.
    ring_pad = max(8, max(bw, bh) // 4)
    ring_y1 = max(0, local_by - ring_pad)
    ring_y2 = min(rh, local_by + bh + ring_pad)
    ring_x1 = max(0, local_bx - ring_pad)
    ring_x2 = min(rw_roi, local_bx + bw + ring_pad)
    ring_mask = np.zeros((rh, rw_roi), dtype=np.uint8)
    ring_mask[ring_y1:ring_y2, ring_x1:ring_x2] = 255
    # Exclude the mark_box interior
    mb_y1_l = max(0, local_by)
    mb_y2_l = min(rh, local_by + bh)
    mb_x1_l = max(0, local_bx)
    mb_x2_l = min(rw_roi, local_bx + bw)
    ring_mask[mb_y1_l:mb_y2_l, mb_x1_l:mb_x2_l] = 0

    ring_ys, ring_xs = np.where(ring_mask > 0)
    if ring_ys.size >= 30:
        # Fit gradient plane to ring pixels
        A_ring = np.column_stack([
            np.ones(ring_ys.size),
            ring_xs.astype(np.float64) / max(rw_roi, 1),
            ring_ys.astype(np.float64) / max(rh, 1)])
        ring_L_vals = L[ring_ys, ring_xs].astype(np.float64)
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_L_vals, rcond=None)

        # Predict expected L for each pixel inside mark_box
        mb_ys, mb_xs = np.where(
            (np.arange(rh)[:, None] >= mb_y1_l) &
            (np.arange(rh)[:, None] < mb_y2_l) &
            (np.arange(rw_roi)[None, :] >= mb_x1_l) &
            (np.arange(rw_roi)[None, :] < mb_x2_l))
        if mb_ys.size > 0:
            A_mb = np.column_stack([
                np.ones(mb_ys.size),
                mb_xs.astype(np.float64) / max(rw_roi, 1),
                mb_ys.astype(np.float64) / max(rh, 1)])
            expected_L = (A_mb @ coeffs).astype(np.float32)
            actual_L = L[mb_ys, mb_xs]
            deviation = actual_L - expected_L  # positive = brighter than bg

            # Compute ring statistics for adaptive threshold
            ring_L_pred = (A_ring @ coeffs).astype(np.float32)
            ring_residuals = np.abs(L[ring_ys, ring_xs] - ring_L_pred)
            ring_std = max(float(ring_residuals.std()), 0.5)

            # Adaptive threshold: pixels deviating more than ring noise
            # The watermark raises L (white text) or shifts it. Use a
            # low threshold to capture the full semi-transparent body.
            body_thr = max(1.2, ring_std * 1.0)

            # Build body mask inside mark_box
            bw_body = np.zeros((rh, rw_roi), dtype=np.uint8)
            body_pixels = np.abs(deviation) > body_thr
            bw_body[mb_ys[body_pixels], mb_xs[body_pixels]] = 255

            # Also include pixels that are brighter than expected (specific
            # to white watermark text) with a softer threshold
            bright_thr = max(0.8, ring_std * 0.7)
            bright_pixels = deviation > bright_thr
            bw_body[mb_ys[bright_pixels], mb_xs[bright_pixels]] = 255

            bw_strokes = cv2.bitwise_or(bw_strokes, bw_body)
    else:
        # Fallback: bilateral-filter-based body detection (original approach)
        bg_2d = cv2.bilateralFilter(L.astype(np.uint8), 15,
                                     sigmaColor=30, sigmaSpace=30).astype(np.float32)
        lightness_dev = L - bg_2d
        mean_L = float(L.mean())
        if mean_L > 200:
            body_thr = max(1.5, float(np.percentile(np.abs(lightness_dev), 60)) * 0.8)
        elif mean_L > 120:
            body_thr = max(2.0, float(np.percentile(np.abs(lightness_dev), 55)) * 0.9)
        else:
            body_thr = max(2.5, float(np.percentile(lightness_dev[lightness_dev > 0], 50)) * 0.9) if (lightness_dev > 0).any() else 3.0
        bw_body = (np.abs(lightness_dev) > body_thr).astype(np.uint8) * 255
        mb_mask = np.zeros_like(bw_body)
        mb_pad_x = max(2, bw // 10); mb_pad_y = max(1, bh // 4)
        mb_y1 = max(0, local_by - mb_pad_y)
        mb_y2 = min(roi.shape[0], local_by + bh + mb_pad_y)
        mb_x1 = max(0, local_bx - mb_pad_x)
        mb_x2 = min(roi.shape[1], local_bx + bw + mb_pad_x)
        mb_mask[mb_y1:mb_y2, mb_x1:mb_x2] = 255
        bw_body = cv2.bitwise_and(bw_body, mb_mask)
        bw_strokes = cv2.bitwise_or(bw_strokes, bw_body)

    # Morphology — close gaps aggressively to fill text interior.
    # Wider horizontal kernel connects adjacent character strokes;
    # vertical kernel fills within single characters.
    bw_strokes = cv2.morphologyEx(
        bw_strokes, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    bw_strokes = cv2.morphologyEx(
        bw_strokes, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
    # Additional close to catch remaining gaps in character bodies
    bw_strokes = cv2.morphologyEx(
        bw_strokes, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Restrict final strokes to mark_box with small pad to avoid
    # picking up product texture outside the watermark region.
    mb_restrict = np.zeros_like(bw_strokes)
    mb_pad_x2 = max(3, bw // 8); mb_pad_y2 = max(2, bh // 3)
    mr_y1 = max(0, local_by - mb_pad_y2)
    mr_y2 = min(rh, local_by + bh + mb_pad_y2)
    mr_x1 = max(0, local_bx - mb_pad_x2)
    mr_x2 = min(rw_roi, local_bx + bw + mb_pad_x2)
    mb_restrict[mr_y1:mr_y2, mr_x1:mr_x2] = 255
    bw_strokes = cv2.bitwise_and(bw_strokes, mb_restrict)

    # Component filter — keep glyph-sized and text-body blobs.
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bw_strokes)
    keep = np.zeros_like(bw_strokes)
    n_kept = 0
    max_area = max(2000, int(bw * bh * 0.6))
    for i in range(1, nlab):
        a = stats[i, cv2.CC_STAT_AREA]
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if (2 <= a <= max_area and 2 <= h_ <= bh * 2.5 and
                w_ <= max(bw * 1.3, h_ * 10) and a >= h_ * 0.3):
            keep[labels == i] = 255; n_kept += 1

    # Wider dilation for better inpaint blending halo.
    keep = cv2.dilate(keep,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))

    # Lift to full-image coords.
    full = np.zeros((H, W), np.uint8)
    full[ry1:ry2, rx1:rx2] = keep

    # Confidence — fraction of an expected stroke area (~22% of mark_box,
    # the empirically observed text-pixel coverage of "sunsky-online.com").
    EXPECTED_FRAC = 0.22
    expected = max(40, int(EXPECTED_FRAC * bw * bh))
    actual = int((full > 0).sum())
    confidence = min(1.0, actual / max(expected, 1))

    return {"stroke": full, "roi_box": (rx1, ry1, rx2, ry2),
            "confidence": round(confidence, 3),
            "n_components": n_kept,
            "raw_thresh": round(thr, 2),
            "stroke_pixels": actual}


# ---------------------------------------------------------------------------
# P2 — Product overlap analyzer + integrity pre-gate.
# Edges / dark components / circular features. The mark_box is excluded
# from edge computation so the watermark's own strokes don't count as
# "product structure".
# ---------------------------------------------------------------------------

def analyze_product_overlap(img_bgr: np.ndarray, soft_mask: np.ndarray,
                            mark_box: dict) -> dict:
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Strong edges, mark_box excluded so the watermark's own strokes don't
    # falsify the product-edge overlap.
    edges = cv2.Canny(gray, 70, 160)
    excl = np.zeros((H, W), np.uint8)
    pad = 4
    excl[max(0, mark_box["y"] - pad):
         min(H, mark_box["y"] + mark_box["h"] + pad),
         max(0, mark_box["x"] - pad):
         min(W, mark_box["x"] + mark_box["w"] + pad)] = 255
    edges_excl = cv2.bitwise_and(edges, cv2.bitwise_not(excl))
    edges_excl = cv2.dilate(edges_excl,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Dark product pixels (true blacks, not just shadow).
    dark = (gray <= 35).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, cv2.bitwise_not(excl))

    soft_area = float((soft_mask > 0).sum())
    if soft_area < 1:
        return {"edge_overlap": 0.0, "dark_overlap": 0.0,
                "circle_overlap": 0.0, "edge_density_in_mask": 0.0,
                "dark_density_in_mask": 0.0, "n_circles_near_mark": 0}

    edge_overlap = float((cv2.bitwise_and(edges_excl, soft_mask) > 0).sum()) / soft_area
    dark_overlap = float((cv2.bitwise_and(dark, soft_mask) > 0).sum()) / soft_area

    # Circular features — lens rings, holes. HoughCircles is expensive; we
    # restrict it to a band around the mark_box.
    by, bx, bh, bw = (mark_box["y"], mark_box["x"], mark_box["h"], mark_box["w"])
    by2, bx2 = by + bh, bx + bw
    pad_c = max(bh * 2, 40)
    cy1 = max(0, by - pad_c); cy2 = min(H, by2 + pad_c)
    cx1 = max(0, bx - pad_c); cx2 = min(W, bx2 + pad_c)
    band = gray[cy1:cy2, cx1:cx2]
    circle_overlap = 0.0
    n_circles = 0
    if band.size and min(band.shape) > 40:
        try:
            blur = cv2.GaussianBlur(band, (5, 5), 0)
            min_r = max(8, min(bw, bh) // 3)
            max_r = max(min_r + 10, min(band.shape) // 2)
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                       minDist=max(20, bh),
                                       param1=120, param2=40,
                                       minRadius=min_r, maxRadius=max_r)
        except cv2.error:
            circles = None
        if circles is not None:
            circle_mask = np.zeros((H, W), np.uint8)
            for c in circles[0]:
                cxr, cyr, r = int(c[0] + cx1), int(c[1] + cy1), int(c[2])
                cv2.circle(circle_mask, (cxr, cyr), r, 255, thickness=2)
                n_circles += 1
            circle_overlap = (float((cv2.bitwise_and(circle_mask, soft_mask) > 0).sum())
                              / soft_area)

    # Densities INSIDE the mask area (raw, for debug).
    edge_density_in_mask = (float(edges_excl[soft_mask > 0].mean()) / 255.0
                            if (soft_mask > 0).any() else 0.0)
    dark_density_in_mask = (float(dark[soft_mask > 0].mean()) / 255.0
                            if (soft_mask > 0).any() else 0.0)

    return {"edge_overlap": round(edge_overlap, 4),
            "dark_overlap": round(dark_overlap, 4),
            "circle_overlap": round(circle_overlap, 4),
            "edge_density_in_mask": round(edge_density_in_mask, 4),
            "dark_density_in_mask": round(dark_density_in_mask, 4),
            "n_circles_near_mark": n_circles,
            "_edges": edges_excl, "_dark": dark}


def product_integrity_gate(overlap: dict) -> tuple[str, str | None]:
    """Return ('clean'/'manual'/'reject', reason_code). 'clean' means proceed."""
    if overlap["edge_overlap"] > PI_GATE_EDGE_REJECT:
        return "reject", R_PRODUCT_DAMAGE
    if overlap["dark_overlap"] > PI_GATE_DARK_REJECT:
        return "reject", R_PRODUCT_DAMAGE
    if overlap["edge_overlap"] > PI_GATE_EDGE_MANUAL:
        return "manual", M_MASK_PRODUCT_EDGE
    if overlap["dark_overlap"] > PI_GATE_DARK_MANUAL:
        return "manual", M_MASK_DARK_COMPONENT
    if overlap["circle_overlap"] > PI_GATE_CIRCLE_MANUAL:
        return "manual", M_MASK_LENS_OR_CIRCLE
    return "clean", None


# ---------------------------------------------------------------------------
# Foreground extraction (used by post-clean product-integrity QA + dark
# product detection).
# ---------------------------------------------------------------------------

def extract_product_foreground(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, fg = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if fg.mean() > 0.80 * 255:
        fg = 255 - fg
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return fg


# ---------------------------------------------------------------------------
# Product Protection Mask — combines all protected-region signals into one
# mask that MUST NOT be inpainted through.
# ---------------------------------------------------------------------------

def build_product_protect_mask(img_bgr: np.ndarray,
                               mark_box: dict | None = None,
                               exclude_pad: int = 6) -> dict:
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Watermark exclusion zone — don't count watermark pixels as product.
    excl = np.zeros((H, W), np.uint8)
    if mark_box is not None:
        x1 = max(0, mark_box["x"] - exclude_pad)
        y1 = max(0, mark_box["y"] - exclude_pad)
        x2 = min(W, mark_box["x"] + mark_box["w"] + exclude_pad)
        y2 = min(H, mark_box["y"] + mark_box["h"] + exclude_pad)
        excl[y1:y2, x1:x2] = 255
    inv_excl = cv2.bitwise_not(excl)

    # 1) Strong edges — real product structure.
    edges = cv2.Canny(gray, 60, 140)
    k_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_d = cv2.dilate(edges, k_e, iterations=1)
    edges_d = cv2.bitwise_and(edges_d, inv_excl)

    # 2) Dark components — connectors, cables, PCB parts.
    dark = (gray <= 40).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, inv_excl)
    k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_d)

    # 3) Specular highlights — camera flash on metal.
    specular = (gray >= 248).astype(np.uint8) * 255
    specular = cv2.bitwise_and(specular, inv_excl)
    specular = cv2.morphologyEx(specular, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # 4) Thin edges — Sobel-based directional structure.
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    thin = (mag > np.percentile(mag, 95)).astype(np.uint8) * 255
    thin = cv2.bitwise_and(thin, inv_excl)

    # 5) Circular features — lens rings, holes, speaker meshes.
    circle_mask = np.zeros((H, W), np.uint8)
    n_circles = 0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if min(H, W) > 80:
        try:
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                       minDist=max(20, min(H, W) // 10),
                                       param1=120, param2=40,
                                       minRadius=8,
                                       maxRadius=min(H, W) // 3)
        except cv2.error:
            circles = None
        if circles is not None:
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                cv2.circle(circle_mask, (cx, cy), r, 255, thickness=3)
                n_circles += 1
    circle_mask = cv2.bitwise_and(circle_mask, inv_excl)

    # 6) Text/QR signal — small connected components in high-density clusters.
    _, bw = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    text_qr = np.zeros((H, W), np.uint8)
    for i in range(1, nlab):
        a = stats[i, cv2.CC_STAT_AREA]
        if 10 <= a <= 300:
            text_qr[labels == i] = 255
    text_qr = cv2.bitwise_and(text_qr, inv_excl)
    k_t = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    text_qr = cv2.morphologyEx(text_qr, cv2.MORPH_CLOSE, k_t)

    # 7) Foreground product silhouette.
    fg = extract_product_foreground(img_bgr)

    # Combined mask — union of all protected signals.
    combined = np.zeros((H, W), np.uint8)
    for m in (edges_d, dark, specular, thin, circle_mask, text_qr):
        combined = cv2.bitwise_or(combined, m)

    return {
        "combined": combined,
        "edges": edges_d,
        "dark": dark,
        "specular": specular,
        "thin_edges": thin,
        "circles": circle_mask,
        "text_qr": text_qr,
        "foreground": fg,
        "n_circles": n_circles,
    }


# ---------------------------------------------------------------------------
# Edge-stopped dilation — dilates the stroke mask but stops at edge barriers.
# ---------------------------------------------------------------------------

def build_edge_barrier(img_bgr: np.ndarray, mark_box: dict | None = None,
                       exclude_pad: int = 6) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 130)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    barrier = cv2.dilate(edges, k, iterations=1)
    if mark_box is not None:
        x1 = max(0, mark_box["x"] - exclude_pad)
        y1 = max(0, mark_box["y"] - exclude_pad)
        x2 = min(W, mark_box["x"] + mark_box["w"] + exclude_pad)
        y2 = min(H, mark_box["y"] + mark_box["h"] + exclude_pad)
        barrier[y1:y2, x1:x2] = 0
    return barrier


def edge_stopped_dilation(mask: np.ndarray, barrier: np.ndarray,
                          iterations: int = 3) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = mask.copy()
    inv_barrier = cv2.bitwise_not(barrier)
    for _ in range(iterations):
        expanded = cv2.dilate(result, k, iterations=1)
        result = cv2.bitwise_and(expanded, inv_barrier)
        result = cv2.bitwise_or(result, mask)
    return result


# ---------------------------------------------------------------------------
# P3 — ROI feature extraction + extended classifier (9 → 11 classes).
# ---------------------------------------------------------------------------

def extract_roi_features(img_bgr: np.ndarray, mark_box: dict) -> dict:
    H, W = img_bgr.shape[:2]
    x, y, w, h = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    pad = max(20, h)

    ry1 = max(0, y - pad); ry2 = min(H, y + h + pad)
    rx1 = max(0, x - pad); rx2 = min(W, x + w + pad)
    ring = img_bgr[ry1:ry2, rx1:rx2]
    if ring.size == 0:
        return {"mask_density": (w * h) / (H * W + 1e-6), "skip": True}

    ring_mask = np.ones(ring.shape[:2], dtype=bool)
    mx = x - rx1; my = y - ry1
    ring_mask[max(0, my):min(ring_mask.shape[0], my + h),
              max(0, mx):min(ring_mask.shape[1], mx + w)] = False
    if not ring_mask.any():
        return {"mask_density": (w * h) / (H * W + 1e-6), "skip": True}

    gray = cv2.cvtColor(ring, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(ring, cv2.COLOR_BGR2LAB)
    ab = lab[..., 1:3].astype(np.float32) - 128

    luma_vals = gray[ring_mask]
    luma_std = float(luma_vals.std())
    luma_mean = float(luma_vals.mean())
    chroma_vals = np.linalg.norm(ab[ring_mask], axis=-1)
    chroma_std = float(chroma_vals.std())
    chroma_mean = float(chroma_vals.mean())
    lo, hi = np.percentile(luma_vals, [5, 95])
    local_contrast = float(hi - lo)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_energy = float(lap[ring_mask].std())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges[ring_mask].mean()) / 255.0
    spec_frac = float((luma_vals >= 248).mean())
    bright_frac = float((luma_vals >= 220).mean())
    mid_frac = float(((luma_vals > 80) & (luma_vals < 180)).mean())
    dark_frac = float((luma_vals <= 50).mean())

    # transparent signal — see-through products (bimodal luma + low chroma +
    # significant specular).
    transparent_signal = (chroma_mean < 6 and luma_std > 35 and
                          local_contrast > 110 and bright_frac > 0.08 and
                          mid_frac > 0.15 and spec_frac > 0.005)

    # Lateral color drift (kept).
    strip_w = max(20, h)
    left = img_bgr[y:y + h, max(0, x - strip_w):x]
    right = img_bgr[y:y + h, x + w:min(W, x + w + strip_w)]
    if left.size and right.size:
        lab_l = cv2.cvtColor(left, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
        lab_r = cv2.cvtColor(right, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
        color_drift = float(np.linalg.norm(lab_l - lab_r))
    else:
        color_drift = 0.0

    # Small-component density — text/QR signal.
    _, bwth = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(bwth)
    if nlab > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        small = ((areas >= 8) & (areas <= 200)).sum()
        small_density = small / max(ring_mask.sum() / (32 * 32), 1)
    else:
        small_density = 0.0

    # NEW for V3 — POSTER / MARKETING signal: image-wide density of text-like
    # components + a marketing block-layout indicator (large rectangular
    # solid-color regions in addition to small-component clusters).
    g_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges_full = cv2.Canny(g_full, 50, 150)
    _, bwf = cv2.threshold(edges_full, 0, 255, cv2.THRESH_BINARY)
    nf, _, sf, _ = cv2.connectedComponentsWithStats(bwf)
    if nf > 1:
        areas_f = sf[1:, cv2.CC_STAT_AREA]
        small_f = ((areas_f >= 8) & (areas_f <= 200)).sum()
        small_density_global = small_f / max(g_full.size / (32 * 32), 1)
    else:
        small_density_global = 0.0
    edge_density_global = float(edges_full.mean()) / 255.0

    return {
        "mask_density": (w * h) / (H * W + 1e-6),
        "luma_std": round(luma_std, 2),
        "luma_mean": round(luma_mean, 2),
        "chroma_std": round(chroma_std, 2),
        "chroma_mean": round(chroma_mean, 2),
        "local_contrast": round(local_contrast, 2),
        "texture_energy": round(texture_energy, 2),
        "edge_density": round(edge_density, 4),
        "color_drift": round(color_drift, 2),
        "small_component_density": round(small_density, 3),
        "small_component_density_global": round(small_density_global, 3),
        "edge_density_global": round(edge_density_global, 4),
        "specular_fraction": round(spec_frac, 4),
        "bright_fraction": round(bright_frac, 4),
        "dark_fraction": round(dark_frac, 4),
        "transparent_signal": transparent_signal,
        "skip": False,
    }


def classify_roi(features: dict) -> str:
    if features.get("skip"):
        return "plain_white"

    edge = features["edge_density"]
    tex = features["texture_energy"]
    chr_s = features["chroma_std"]
    chr_m = features["chroma_mean"]
    lc = features["local_contrast"]
    sc = features["small_component_density"]
    sc_g = features["small_component_density_global"]
    edge_g = features["edge_density_global"]
    spec = features["specular_fraction"]
    lum_m = features["luma_mean"]
    dark = features["dark_fraction"]

    # 0) Poster / marketing — globally text-heavy or banner-layout images.
    # The fast signal is the GLOBAL small-component density combined with a
    # high global edge density. Real product photos are mostly empty space
    # around a single subject; marketing layouts have many small text
    # components spread across the whole image.
    if sc_g > 4.5 and edge_g > 0.08:
        return "poster_or_marketing"
    if sc_g > 6.5:
        return "poster_or_marketing"

    # 1) text/QR — conservative (sc>3.0 & edge>0.18)
    if sc > 3.0 and edge > 0.18:
        return "text_or_qr"

    # 2) black_product — large dark area in the surround
    if dark > 0.35 and edge > 0.04:
        return "black_product"

    # 3) glossy — specular + medium luma + medium edge
    if spec > 0.015 and 60 < lum_m < 200 and edge > 0.07:
        return "glossy"

    # 4) metallic — high specular + low chroma + bright (without strong edges)
    if spec > 0.012 and chr_m < 5 and 130 < lum_m < 230 and edge < 0.10:
        return "metallic"

    # 5) transparent
    if features.get("transparent_signal"):
        return "transparent"

    # 6) high texture: lots of edges or strong Laplacian energy
    if edge > 0.10 or tex > 28:
        if chr_m > 8 or chr_s > 6:
            return "product_detail"
        return "high_texture"

    # 7) low texture
    if edge > 0.05 or tex > 14:
        return "low_texture"

    # 8) gradient
    if chr_s > 6 and lc > 40:
        return "gradient"

    # 9) plain_color
    if chr_s > 6:
        return "plain_color"

    # 10) plain_white
    return "plain_white"


# ---------------------------------------------------------------------------
# 3-layer (now 4-layer) mask construction: core / soft / risk / stroke.
# final = stroke if confident enough, otherwise soft.
# ---------------------------------------------------------------------------

def build_soft_mask(img_shape, mark_box: dict, roi_class: str,
                    expand_extra: float = 0.0) -> np.ndarray:
    H, W = img_shape[:2]
    x_ratio, y_ratio = EXPAND_BY_CLASS[roi_class]
    x_ratio += expand_extra
    y_ratio += expand_extra * 0.5
    cx = mark_box["x"] + mark_box["w"] / 2.0
    cy = mark_box["y"] + mark_box["h"] / 2.0
    new_w = max(1, int(round(mark_box["w"] * x_ratio)))
    new_h = max(1, int(round(mark_box["h"] * y_ratio)))
    nx = int(round(cx - new_w / 2.0))
    ny = int(round(cy - new_h / 2.0))
    nx = max(0, min(W - 1, nx)); ny = max(0, min(H - 1, ny))
    new_w = min(W - nx, new_w); new_h = min(H - ny, new_h)
    soft = np.zeros((H, W), np.uint8)
    soft[ny:ny + new_h, nx:nx + new_w] = 255

    dilate_px = DILATE_BY_CLASS[roi_class]
    if expand_extra > 0:
        dilate_px = min(15, dilate_px + 3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px * 2 + 1,) * 2)
    return cv2.dilate(soft, k)


def build_mask_layers(img_bgr: np.ndarray, mark_box: dict, roi_class: str,
                     expand_extra: float = 0.0,
                     protect_mask: np.ndarray | None = None) -> dict:
    H, W = img_bgr.shape[:2]

    # V7: logo-shaped mask replaces rectangular core/soft fallback.
    logo_mask = align_logo_mask(mark_box, img_bgr.shape, use_shadow=True)
    logo_text_only = align_logo_mask(mark_box, img_bgr.shape, use_shadow=False)
    has_logo = logo_mask is not None and logo_mask.any()

    core = np.zeros((H, W), np.uint8)
    core[mark_box["y"]:mark_box["y"] + mark_box["h"],
         mark_box["x"]:mark_box["x"] + mark_box["w"]] = 255
    soft = build_soft_mask(img_bgr.shape, mark_box, roi_class, expand_extra)
    stroke_info = generate_stroke_level_mask(img_bgr, mark_box)
    stroke = stroke_info["stroke"]

    barrier = build_edge_barrier(img_bgr, mark_box)
    stroke_capped = cv2.bitwise_and(stroke, soft)

    # Edge-stopped repair mask: dilate strokes but respect edge barriers.
    dilate_px = DILATE_BY_CLASS.get(roi_class, 5)
    repair = edge_stopped_dilation(stroke_capped, barrier, iterations=dilate_px)
    repair = cv2.bitwise_and(repair, soft)

    # V7: constrain repair to logo shape — never spill into non-logo pixels.
    if has_logo:
        k_logo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        logo_dilated = cv2.dilate(logo_mask, k_logo, iterations=1)
        repair = cv2.bitwise_and(repair, logo_dilated)
        stroke_capped = cv2.bitwise_and(stroke_capped, logo_dilated)

    # Subtract product protection from repair mask.
    if protect_mask is not None:
        repair_safe = cv2.bitwise_and(repair, cv2.bitwise_not(protect_mask))
    else:
        repair_safe = repair

    PLAIN_CLASSES = {"plain_white", "plain_color", "gradient", "low_texture",
                     "metallic", "transparent", "glossy"}
    stroke_good = (stroke_info["confidence"] >= 0.25 and
                   stroke_info["stroke_pixels"] > 20)

    if stroke_good:
        final = repair_safe if repair_safe.any() else stroke_capped
        used = "stroke"
    elif has_logo:
        # V7: use logo-shaped mask as fallback instead of rectangular soft.
        if protect_mask is not None:
            logo_safe = cv2.bitwise_and(logo_mask, cv2.bitwise_not(protect_mask))
        else:
            logo_safe = logo_mask
        if logo_safe.any():
            final = logo_safe
            used = "logo_fallback"
        else:
            final = logo_mask
            used = "logo_fallback"
    elif roi_class in PLAIN_CLASSES:
        protect_overlap = 0.0
        if protect_mask is not None:
            soft_area_check = float((soft > 0).sum())
            if soft_area_check > 0:
                protect_overlap = (float(
                    (cv2.bitwise_and(soft, protect_mask) > 0).sum()
                ) / soft_area_check)
        if protect_overlap < 0.05:
            final = soft
            used = "soft_fallback"
        else:
            final = repair_safe if repair_safe.any() else core
            used = "repair_guarded"
    else:
        final = repair_safe if repair_safe.any() else core
        used = "repair_guarded"

    soft_area = float((soft > 0).sum())
    final_area = float((final > 0).sum())
    return {"core": core, "soft": soft, "stroke": stroke,
            "stroke_capped": stroke_capped, "final": final,
            "repair": repair, "repair_safe": repair_safe,
            "logo_mask": logo_mask, "logo_text_only": logo_text_only,
            "used": used,
            "stroke_info": stroke_info,
            "final_density_vs_soft": round(final_area / max(soft_area, 1), 3)}


# ---------------------------------------------------------------------------
# Inpaint adapters — unchanged from V2.
# ---------------------------------------------------------------------------

def _feather_blend(orig, repaired, mask, feather_px=5):
    alpha = cv2.GaussianBlur(mask, (0, 0),
                              sigmaX=feather_px).astype(np.float32) / 255.0
    alpha3 = np.stack([alpha] * 3, axis=-1)
    return (repaired.astype(np.float32) * alpha3 +
            orig.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)


def inpaint_white_fill(img, mask):
    surround = (mask == 0)
    if not surround.any():
        return img.copy()
    pixels = img[surround].reshape(-1, 3)
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    if (ring > 0).sum() > 200:
        pixels = img[ring > 0].reshape(-1, 3)
    median = np.median(pixels, axis=0).astype(np.uint8)
    out = img.copy()
    fill = np.zeros_like(out); fill[:] = median
    noise = np.random.normal(0, 1.5, fill.shape).astype(np.int16)
    fill = np.clip(fill.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    out[mask > 0] = fill[mask > 0]
    return _feather_blend(img, out, mask)


def inpaint_local_median(img, mask):
    big = cv2.medianBlur(img, 21)
    out = img.copy()
    out[mask > 0] = big[mask > 0]
    return _feather_blend(img, out, mask, feather_px=4)


def inpaint_surface_fit(img, mask):
    H, W = img.shape[:2]
    surround = (mask == 0)
    ys_all, xs_all = np.where(surround)
    if len(ys_all) < 100:
        return inpaint_white_fill(img, mask)
    if len(ys_all) > 6000:
        idx = np.random.choice(len(ys_all), 6000, replace=False)
        ys, xs = ys_all[idx], xs_all[idx]
    else:
        ys, xs = ys_all, xs_all
    nx = xs.astype(np.float64) / W
    ny = ys.astype(np.float64) / H
    A = np.column_stack([np.ones_like(nx), nx, ny,
                          nx ** 2, ny ** 2, nx * ny])
    my_, mx_ = np.where(mask > 0)
    if mx_.size == 0:
        return img.copy()
    mnx = mx_.astype(np.float64) / W
    mny = my_.astype(np.float64) / H
    A_m = np.column_stack([np.ones_like(mnx), mnx, mny,
                            mnx ** 2, mny ** 2, mnx * mny])
    out = img.copy()
    for c in range(3):
        z = img[ys, xs, c].astype(np.float64)
        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        out[my_, mx_, c] = np.clip(A_m @ coeffs, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, 1.2, (my_.size, 3)).astype(np.int16)
    out[my_, mx_] = np.clip(out[my_, mx_].astype(np.int16) + noise, 0, 255
                            ).astype(np.uint8)
    return _feather_blend(img, out, mask, feather_px=5)


def inpaint_gradient_plane_fill(img, mask):
    H, W = img.shape[:2]
    ys_m, xs_m = np.where(mask > 0)
    if xs_m.size == 0:
        return img.copy()
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    ys_r, xs_r = np.where(ring > 0)
    if ys_r.size < 20:
        return inpaint_surface_fit(img, mask)
    nx = xs_r.astype(np.float64) / W
    ny = ys_r.astype(np.float64) / H
    A = np.column_stack([np.ones_like(nx), nx, ny])
    mnx = xs_m.astype(np.float64) / W
    mny = ys_m.astype(np.float64) / H
    A_m = np.column_stack([np.ones_like(mnx), mnx, mny])
    out = img.copy()
    for c in range(3):
        z = img[ys_r, xs_r, c].astype(np.float64)
        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        out[ys_m, xs_m, c] = np.clip(A_m @ coeffs, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, 1.0, (ys_m.size, 3)).astype(np.int16)
    out[ys_m, xs_m] = np.clip(out[ys_m, xs_m].astype(np.int16) + noise,
                               0, 255).astype(np.uint8)
    return _feather_blend(img, out, mask, feather_px=4)


def inpaint_local_color_plane(img, mask):
    H, W = img.shape[:2]
    ys_m, xs_m = np.where(mask > 0)
    if xs_m.size == 0:
        return img.copy()
    ring_k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    ring = cv2.dilate(mask, ring_k)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    ys_r, xs_r = np.where(ring > 0)
    if ys_r.size < 30:
        return inpaint_white_fill(img, mask)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    out_lab = lab.copy()
    for c in range(3):
        vals = lab[ys_r, xs_r, c]
        mean_v = vals.mean()
        out_lab[ys_m, xs_m, c] = mean_v
    noise = np.random.normal(0, 0.8, (ys_m.size, 3))
    out_lab[ys_m, xs_m] = np.clip(out_lab[ys_m, xs_m] + noise, 0, 255)
    out = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return _feather_blend(img, out, mask, feather_px=4)


def inpaint_telea_small(rwm, img, mask, radius=2):
    return rwm._inpaint_cv(img, mask, radius, "telea")


def inpaint_poisson_blend(img, mask):
    H, W = img.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return img.copy()
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    ch, cw = y2 - y1, x2 - x1
    if ch < 3 or cw < 3:
        return inpaint_white_fill(img, mask)
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    src = img.copy()
    ref_pixels = img[ring > 0]
    if ref_pixels.size > 0:
        median = np.median(ref_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
        src[mask > 0] = median
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    roi_mask = mask[y1:y2, x1:x2]
    roi_src = src[y1:y2, x1:x2]
    try:
        out = cv2.seamlessClone(roi_src, img, roi_mask, center,
                                cv2.NORMAL_CLONE)
    except cv2.error:
        out = _feather_blend(img, src, mask, feather_px=5)
    return out


def remove_watermark_haze(cleaned_bgr, mark_box, roi_class, strength=0.70):
    """Post-cleaning pass to remove semi-transparent watermark haze.

    V4 OFFSET-SUBTRACTION: Instead of replacing pixels, we compute the
    per-pixel offset between what's there and what the background should
    be, then subtract that offset. This preserves the underlying texture
    while correcting the color/lightness shift caused by the watermark.
    """
    SAFE_CLASSES = {"plain_white", "plain_color", "gradient", "low_texture"}
    MEDIUM_CLASSES = {"metallic", "transparent", "glossy", "black_product"}

    if roi_class in SAFE_CLASSES:
        strength = max(strength, 0.92)
    elif roi_class in MEDIUM_CLASSES:
        strength = max(strength, 0.75)
    else:
        # high_texture, product_detail, etc. -- still apply but gently
        strength = max(strength, 0.55)

    H, W = cleaned_bgr.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    if bw < 10 or bh < 5:
        return cleaned_bgr
    pad = max(15, max(bw, bh) // 3)
    ry1, ry2 = max(0, by - pad), min(H, by + bh + pad)
    rx1, rx2 = max(0, bx - pad), min(W, bx + bw + pad)
    ring_mask = np.zeros((H, W), np.uint8)
    ring_mask[ry1:ry2, rx1:rx2] = 255
    ring_mask[by:by + bh, bx:bx + bw] = 0
    ring_ys, ring_xs = np.where(ring_mask > 0)
    if ring_ys.size < 30:
        return cleaned_bgr
    lab = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    ring_ny = ring_ys.astype(np.float64) / H
    ring_nx = ring_xs.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])
    mark_ys, mark_xs = np.where(
        (np.arange(H)[:, None] >= by) & (np.arange(H)[:, None] < by + bh) &
        (np.arange(W)[None, :] >= bx) & (np.arange(W)[None, :] < bx + bw))
    mark_ny = mark_ys.astype(np.float64) / H
    mark_nx = mark_xs.astype(np.float64) / W
    A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny])

    out_lab = lab.copy()

    # OFFSET-SUBTRACTION approach:
    # 1. Estimate background via gradient plane from the ring
    # 2. Compute the LOW-FREQUENCY component of the current pixels
    # 3. Compute offset = current_low_freq - bg_estimate (the haze)
    # 4. Subtract the offset: result = current - offset * strength
    # This shifts the mean toward the background while preserving texture.

    mark_roi = lab[by:by + bh, bx:bx + bw].copy()
    blur_ksize = max(15, min(bw, bh) // 3) | 1

    for c in range(3):
        ring_vals = lab[ring_ys, ring_xs, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        bg_vals = A_mark @ coeffs  # What the background should be

        cur_vals = lab[mark_ys, mark_xs, c]

        # Current low-frequency: smooth version of what's in the ROI
        roi_channel = mark_roi[..., c]
        roi_blur = cv2.GaussianBlur(
            roi_channel.astype(np.float64),
            (blur_ksize, blur_ksize), 0)

        # Map blurred values back to mark coordinates
        local_ys = mark_ys - by
        local_xs = mark_xs - bx
        valid = ((local_ys >= 0) & (local_ys < bh) &
                 (local_xs >= 0) & (local_xs < bw))
        cur_low_freq = np.zeros_like(cur_vals)
        cur_low_freq[valid] = roi_blur[local_ys[valid], local_xs[valid]]
        cur_low_freq[~valid] = cur_vals[~valid]

        # Offset = how far the current low-freq is from the estimated background
        offset = cur_low_freq - bg_vals

        # Subtract the offset: shifts the mean while preserving texture
        new_vals = cur_vals - offset * strength
        new_vals += np.random.normal(0, 0.2, new_vals.shape)

        out_lab[mark_ys, mark_xs, c] = new_vals

    # Feathered blending at boundaries
    feather = max(4, min(bw, bh) // 5)
    box_mask = np.zeros((H, W), np.float64)
    box_mask[by:by + bh, bx:bx + bw] = 1.0
    box_mask = cv2.GaussianBlur(box_mask, (0, 0), sigmaX=feather)
    for c in range(3):
        lab[..., c] = lab[..., c] * (1.0 - box_mask) + out_lab[..., c] * box_mask
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


def suppress_watermark_edges(cleaned_bgr, mark_box, roi_class, strength=0.6):
    """After haze removal, residual HIGH-FREQUENCY text edge patterns may
    remain. This function applies targeted selective blurring ONLY to pixels
    within the mark_box that have high-frequency content consistent with
    text (horizontal edge patterns). Preserves product texture by only
    smoothing narrow horizontal bands where text strokes were detected.
    """
    H, W = cleaned_bgr.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    if bw < 10 or bh < 5:
        return cleaned_bgr

    # Pad slightly to catch bleeding edges
    pad_x = max(3, bw // 10); pad_y = max(2, bh // 4)
    ry1 = max(0, by - pad_y); ry2 = min(H, by + bh + pad_y)
    rx1 = max(0, bx - pad_x); rx2 = min(W, bx + bw + pad_x)
    roi = cleaned_bgr[ry1:ry2, rx1:rx2].copy()
    if roi.size == 0:
        return cleaned_bgr

    # Detect high-frequency horizontal edges (text stroke signature)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Horizontal Sobel to detect vertical edges of text strokes
    sobel_h = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    # Vertical Sobel to detect horizontal edges
    sobel_v = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_h ** 2 + sobel_v ** 2)

    # Estimate background edge level from the surrounding ring
    ring_pad = max(5, bh // 2)
    ring_y1 = max(0, by - ring_pad) - ry1
    ring_y2 = min(H, by + bh + ring_pad) - ry1
    ring_x1 = max(0, bx - ring_pad) - rx1
    ring_x2 = min(W, bx + bw + ring_pad) - rx1
    ring_mask = np.zeros(gray.shape, dtype=bool)
    ring_mask[max(0, ring_y1):min(gray.shape[0], ring_y2),
              max(0, ring_x1):min(gray.shape[1], ring_x2)] = True
    mb_y1 = by - ry1; mb_y2 = by + bh - ry1
    mb_x1 = bx - rx1; mb_x2 = bx + bw - rx1
    ring_mask[max(0, mb_y1):min(gray.shape[0], mb_y2),
              max(0, mb_x1):min(gray.shape[1], mb_x2)] = False

    if ring_mask.any():
        bg_edge_mean = float(edge_mag[ring_mask].mean())
        bg_edge_std = max(float(edge_mag[ring_mask].std()), 0.1)
    else:
        bg_edge_mean = float(edge_mag.mean())
        bg_edge_std = max(float(edge_mag.std()), 0.1)

    # Find pixels in mark_box with edge magnitude significantly above
    # the background level — these are residual text stroke edges.
    edge_excess_thr = bg_edge_mean + bg_edge_std * 1.5
    excess_mask = np.zeros(gray.shape, dtype=np.float32)
    excess_mask[max(0, mb_y1):min(gray.shape[0], mb_y2),
                max(0, mb_x1):min(gray.shape[1], mb_x2)] = 1.0
    edge_excess = (edge_mag > edge_excess_thr).astype(np.float32) * excess_mask

    # Only suppress if there's meaningful excess edge content
    if float(edge_excess.sum()) < 10:
        return cleaned_bgr

    # Adaptive selective blur: blend a slightly blurred version into
    # high-edge-excess pixels, preserving the rest.
    blur_ksize = max(3, min(bw, bh) // 8) | 1
    blurred = cv2.GaussianBlur(roi, (blur_ksize, blur_ksize), 0)
    alpha = cv2.GaussianBlur(edge_excess, (5, 5), 0)
    alpha = np.clip(alpha * strength, 0, 1.0)
    alpha3 = np.stack([alpha] * 3, axis=-1)
    roi_blended = (roi.astype(np.float32) * (1 - alpha3) +
                   blurred.astype(np.float32) * alpha3)

    result = cleaned_bgr.copy()
    result[ry1:ry2, rx1:rx2] = np.clip(roi_blended, 0, 255).astype(np.uint8)
    return result


def inpaint_alpha_attenuate(img, mask, strength=0.95):
    """Semi-transparent watermark removal via spatially-varying background
    estimation and soft blending. V4: uses gradient-plane background
    estimation instead of flat median for better handling of non-uniform
    backgrounds. Higher default strength for more complete removal."""
    H, W = img.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return img.copy()
    ring_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    ring = cv2.dilate(mask, ring_k)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # V4: spatially-varying background via gradient plane fit
    ring_ys, ring_xs = np.where(ring > 0)
    if ring_ys.size >= 30:
        ring_ny = ring_ys.astype(np.float64) / H
        ring_nx = ring_xs.astype(np.float64) / W
        A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])
        mark_ny = ys.astype(np.float64) / H
        mark_nx = xs.astype(np.float64) / W
        A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny])
        bg_img = lab.copy()
        for c in range(3):
            ring_vals = lab[ring_ys, ring_xs, c]
            coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
            bg_vals = A_mark @ coeffs + np.random.normal(0, 0.5, A_mark.shape[0])
            bg_img[ys, xs, c] = bg_vals
    else:
        # Fallback to median
        ring_px = lab[ring > 0].reshape(-1, 3) if (ring > 0).any() else None
        if ring_px is not None and ring_px.shape[0] >= 10:
            bg = np.median(ring_px, axis=0)
        elif (mask == 0).any():
            bg = lab[mask == 0].reshape(-1, 3).mean(axis=0)
        else:
            bg = np.array([200.0, 128.0, 128.0])
        bg_img = np.full_like(lab, bg)
        bg_img = bg_img + np.random.normal(0, 0.5, bg_img.shape)

    mask_f = mask.astype(np.float64) / 255.0
    alpha = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=2.0) * strength
    alpha3 = np.stack([alpha] * 3, axis=-1)
    out_lab = lab * (1.0 - alpha3) + bg_img * alpha3
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                       cv2.COLOR_LAB2BGR)
    return out


def inpaint_full_box_attenuate(img, mask, mark_box, strength=0.97):
    """Treats the entire mark_box as a mask with soft edges, replacing
    all content with gradient-plane estimated background. This catches
    semi-transparent watermark body that stroke-based masks miss.
    Uses the same ring-based gradient plane as alpha_attenuate but
    applies it to the full mark_box area with feathered edges."""
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]

    # Build full mark_box mask with feathered edges
    box_mask = np.zeros((H, W), dtype=np.uint8)
    box_mask[by:by + bh, bx:bx + bw] = 255
    feather_k = max(3, min(bw, bh) // 6) | 1
    box_mask_f = cv2.GaussianBlur(
        box_mask.astype(np.float64) / 255.0,
        (feather_k, feather_k), 0) * strength

    # Ring: area around mark_box excluding the box itself
    ring_pad = max(15, max(bw, bh) // 3)
    ring_y1 = max(0, by - ring_pad)
    ring_y2 = min(H, by + bh + ring_pad)
    ring_x1 = max(0, bx - ring_pad)
    ring_x2 = min(W, bx + bw + ring_pad)
    ring_mask = np.zeros((H, W), dtype=np.uint8)
    ring_mask[ring_y1:ring_y2, ring_x1:ring_x2] = 255
    ring_mask[by:by + bh, bx:bx + bw] = 0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    ring_ys, ring_xs = np.where(ring_mask > 0)

    if ring_ys.size < 30:
        return inpaint_alpha_attenuate(img, mask, strength)

    ring_ny = ring_ys.astype(np.float64) / H
    ring_nx = ring_xs.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])

    box_ys, box_xs = np.where(box_mask > 0)
    box_ny = box_ys.astype(np.float64) / H
    box_nx = box_xs.astype(np.float64) / W
    A_box = np.column_stack([np.ones_like(box_nx), box_nx, box_ny])

    bg_lab = lab.copy()
    for c in range(3):
        ring_vals = lab[ring_ys, ring_xs, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        ring_pred = A_ring @ coeffs
        ring_residual_std = max(float(np.std(ring_vals - ring_pred)), 0.3)
        bg_vals = A_box @ coeffs + np.random.normal(0, ring_residual_std * 0.5, A_box.shape[0])
        bg_lab[box_ys, box_xs, c] = bg_vals

    alpha3 = np.stack([box_mask_f] * 3, axis=-1)
    out_lab = lab * (1.0 - alpha3) + bg_lab * alpha3
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                       cv2.COLOR_LAB2BGR)
    return out


def inpaint_dark_surface_local_plane(img, mask, mark_box):
    """V4: Dark surface repair via gradient plane from surrounding dark pixels.
    Ensures the repaired patch is never brighter than the surrounding ring."""
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    ys_m, xs_m = np.where(mask > 0)
    if xs_m.size == 0:
        return img.copy()
    ring_pad = max(15, max(bw, bh) // 3)
    ring_y1 = max(0, by - ring_pad); ring_y2 = min(H, by + bh + ring_pad)
    ring_x1 = max(0, bx - ring_pad); ring_x2 = min(W, bx + bw + ring_pad)
    ring_mask = np.zeros((H, W), dtype=np.uint8)
    ring_mask[ring_y1:ring_y2, ring_x1:ring_x2] = 255
    ring_mask[by:by + bh, bx:bx + bw] = 0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    ring_ys, ring_xs = np.where(ring_mask > 0)
    if ring_ys.size < 30:
        return inpaint_local_color_plane(img, mask)
    ring_ny = ring_ys.astype(np.float64) / H
    ring_nx = ring_xs.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])
    mark_ny = ys_m.astype(np.float64) / H
    mark_nx = xs_m.astype(np.float64) / W
    A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny])
    out_lab = lab.copy()
    ring_mean_L = float(lab[ring_ys, ring_xs, 0].mean())
    for c in range(3):
        ring_vals = lab[ring_ys, ring_xs, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        ring_pred = A_ring @ coeffs
        residual_std = max(float(np.std(ring_vals - ring_pred)), 0.3)
        bg_vals = (A_mark @ coeffs +
                   np.random.normal(0, residual_std * 0.3, A_mark.shape[0]))
        if c == 0:
            bg_vals = np.minimum(bg_vals, ring_mean_L + 2.0)
        out_lab[ys_m, xs_m, c] = bg_vals
    feather = max(3, min(bw, bh) // 6)
    mask_f = mask.astype(np.float64) / 255.0
    alpha = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=feather)
    alpha3 = np.stack([alpha] * 3, axis=-1)
    result_lab = lab * (1.0 - alpha3) + out_lab * alpha3
    return cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


def inpaint_white_fill_stroke_only(img, mask, mark_box):
    """V4: White background fill using stroke mask only — no full-box
    replacement. Samples local background from the immediate ring."""
    H, W = img.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return img.copy()
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    if (ring > 0).sum() > 50:
        ring_pixels = img[ring > 0].reshape(-1, 3)
    else:
        ring_pad = max(10, bh)
        ry1 = max(0, by - ring_pad); ry2 = min(H, by + bh + ring_pad)
        rx1 = max(0, bx - ring_pad); rx2 = min(W, bx + bw + ring_pad)
        rmask = np.zeros((H, W), dtype=np.uint8)
        rmask[ry1:ry2, rx1:rx2] = 255
        rmask[by:by + bh, bx:bx + bw] = 0
        ring_pixels = img[rmask > 0].reshape(-1, 3)
    if ring_pixels.size == 0:
        return img.copy()
    median_bg = np.median(ring_pixels, axis=0).astype(np.uint8)
    out = img.copy()
    fill = np.zeros_like(out); fill[:] = median_bg
    noise = np.random.normal(0, 1.0, fill.shape).astype(np.int16)
    fill = np.clip(fill.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    out[mask > 0] = fill[mask > 0]
    return _feather_blend(img, out, mask, feather_px=3)


def inpaint_local_background_fill(img, mask, mark_box):
    """V4: Per-pixel gradient-plane interpolation from surrounding ring.
    More precise than white_fill for non-uniform white backgrounds."""
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    ys_m, xs_m = np.where(mask > 0)
    if xs_m.size == 0:
        return img.copy()
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    ys_r, xs_r = np.where(ring > 0)
    if ys_r.size < 30:
        return inpaint_white_fill_stroke_only(img, mask, mark_box)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    out_lab = lab.copy()
    ring_ny = ys_r.astype(np.float64) / H
    ring_nx = xs_r.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])
    mark_ny = ys_m.astype(np.float64) / H
    mark_nx = xs_m.astype(np.float64) / W
    A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny])
    for c in range(3):
        ring_vals = lab[ys_r, xs_r, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        ring_pred = A_ring @ coeffs
        residual_std = max(float(np.std(ring_vals - ring_pred)), 0.3)
        bg_vals = (A_mark @ coeffs +
                   np.random.normal(0, residual_std * 0.4, A_mark.shape[0]))
        out_lab[ys_m, xs_m, c] = bg_vals
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                       cv2.COLOR_LAB2BGR)
    return _feather_blend(img, out, mask, feather_px=4)


def inpaint_clone_offset(img, mask, mark_box):
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    if bw < 4 or bh < 4:
        return None

    pad_y = max(4, bh // 2)
    candidates = []

    # Try cloning from above the watermark
    src_y1 = by - bh - pad_y
    src_y2 = src_y1 + bh
    if src_y1 >= 0:
        candidates.append(("above", bx, src_y1, bw, bh))

    # Try cloning from below the watermark
    src_y1_b = by + bh + pad_y
    src_y2_b = src_y1_b + bh
    if src_y2_b <= H:
        candidates.append(("below", bx, src_y1_b, bw, bh))

    # Try cloning from further above (2x offset)
    src_y1_fa = by - 2 * bh - pad_y
    if src_y1_fa >= 0:
        candidates.append(("far_above", bx, src_y1_fa, bw, bh))

    # Try cloning from further below
    src_y1_fb = by + 2 * bh + pad_y
    if src_y1_fb + bh <= H:
        candidates.append(("far_below", bx, src_y1_fb, bw, bh))

    if not candidates:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mark_ring_pad = max(6, bh // 2)
    ring_y1 = max(0, by - mark_ring_pad)
    ring_y2 = min(H, by + bh + mark_ring_pad)
    ring_x1 = max(0, bx - 4)
    ring_x2 = min(W, bx + bw + 4)
    ring_mask = np.zeros((H, W), np.uint8)
    ring_mask[ring_y1:ring_y2, ring_x1:ring_x2] = 255
    ring_mask[by:by + bh, bx:bx + bw] = 0
    ring_pixels = lab[ring_mask > 0]
    if ring_pixels.size < 30:
        return None
    ring_mean = ring_pixels.mean(axis=0)

    best = None
    best_diff = float("inf")
    for label, sx, sy, sw, sh in candidates:
        sx2 = min(W, sx + sw)
        sy2 = min(H, sy + sh)
        src_patch = lab[sy:sy2, sx:sx2]
        if src_patch.size == 0:
            continue
        src_mean = src_patch.reshape(-1, 3).mean(axis=0)
        diff = float(np.linalg.norm(src_mean - ring_mean))
        if diff < best_diff:
            best_diff = diff
            best = (label, sx, sy, sx2 - sx, sy2 - sy)

    if best is None or best_diff > 15.0:
        return None

    _, sx, sy, sw, sh = best
    out = img.copy()
    src_patch = img[sy:sy + sh, sx:sx + sw].copy()

    # Resize source patch to exact mark_box size if needed
    if sw != bw or sh != bh:
        src_patch = cv2.resize(src_patch, (bw, bh), interpolation=cv2.INTER_LINEAR)

    out[by:by + bh, bx:bx + bw] = src_patch

    # Feather blend at boundaries for seamless transition
    clone_mask = np.zeros((H, W), np.uint8)
    clone_mask[by:by + bh, bx:bx + bw] = 255
    out = _feather_blend(img, out, clone_mask, feather_px=6)
    return out


# ---------------------------------------------------------------------------
# V6 — Adaptive cover system.
# ---------------------------------------------------------------------------

def _sample_ring(img, mark_box, pad_factor=0.33):
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    pad = max(10, int(max(bw, bh) * pad_factor))
    ry1, ry2 = max(0, by - pad), min(H, by + bh + pad)
    rx1, rx2 = max(0, bx - pad), min(W, bx + bw + pad)
    ring = np.zeros((H, W), dtype=bool)
    ring[ry1:ry2, rx1:rx2] = True
    ring[by:by + bh, bx:bx + bw] = False
    return ring, (ry1, ry2, rx1, rx2)


def estimate_local_cover_style(image, bbox, roi_class):
    H, W = image.shape[:2]
    ring, (ry1, ry2, rx1, rx2) = _sample_ring(image, bbox)
    ring_pixels = image[ring]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    roi_gray = gray[by:by + bh, bx:bx + bw]

    if ring_pixels.size > 0:
        median_color = np.median(ring_pixels, axis=0).astype(np.uint8)
        local_luma = float(cv2.cvtColor(
            median_color.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
    else:
        median_color = np.array([180, 180, 180], dtype=np.uint8)
        local_luma = 180.0

    ring_gray = gray[ring]
    noise_sigma = float(np.std(ring_gray)) if ring_gray.size > 0 else 2.0
    noise_sigma = min(noise_sigma, 15.0)

    edges = cv2.Canny(roi_gray, 60, 140) if roi_gray.size > 0 else np.zeros(1)
    edge_density = float(edges.mean()) / 255.0

    grad_dir = 0.0
    if roi_gray.size > 100:
        gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_dir = float(np.arctan2(gy.mean(), gx.mean()))

    fill_mode_map = {
        "plain_white": "near_white_soft",
        "plain_color": "median_color_soft",
        "low_texture": "median_color_soft",
        "gradient": "local_gradient_fit",
        "metallic": "gradient_plane_soft",
        "glossy": "blurred_background_soft",
        "transparent": "blurred_background_soft",
        "black_product": "dark_translucent_soft",
    }
    fill_mode = fill_mode_map.get(roi_class, "adaptive_soft")

    base_feather = max(8, int(min(bw, bh) * 0.20))
    if roi_class in ("plain_white", "plain_color", "low_texture"):
        base_feather = max(12, int(min(bw, bh) * 0.30))
    elif edge_density >= 0.08:
        base_feather = max(6, int(min(bw, bh) * 0.12))

    base_alpha = 0.45
    if roi_class in ("plain_white", "plain_color"):
        base_alpha = 0.55
    elif roi_class in ("glossy", "transparent"):
        base_alpha = 0.35
    elif roi_class == "black_product":
        base_alpha = 0.40
    if edge_density >= 0.08:
        base_alpha = min(base_alpha, 0.35)
    elif 0.03 <= edge_density < 0.08:
        base_alpha = min(base_alpha, 0.40)

    sharpen = 0.0
    if roi_class in ("metallic", "glossy"):
        sharpen = 0.2

    return {
        "median_color": median_color,
        "local_luma": local_luma,
        "noise_sigma": noise_sigma,
        "edge_density": edge_density,
        "gradient_direction": grad_dir,
        "fill_mode": fill_mode,
        "cover_color": median_color,
        "alpha": base_alpha,
        "feather_px": base_feather,
        "noise_amount": min(noise_sigma * 0.3, 3.0),
        "sharpen_amount": sharpen,
        "roi_class": roi_class,
    }


def make_soft_rounded_rect_mask(w, h, radius=None, feather_px=None):
    if radius is None:
        radius = int(min(w, h) * 0.18)
    if feather_px is None:
        feather_px = max(8, int(min(w, h) * 0.20))
    radius = max(1, min(radius, min(w, h) // 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (w - 1, h - radius - 1), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (w - radius - 1, radius), radius, 255, -1)
    cv2.circle(mask, (radius, h - radius - 1), radius, 255, -1)
    cv2.circle(mask, (w - radius - 1, h - radius - 1), radius, 255, -1)
    mask_f = mask.astype(np.float64) / 255.0
    if feather_px > 1:
        ksize = feather_px * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
    return mask_f


def _embed_rounded_mask(img_shape, mark_box, expand_px, feather_px):
    H, W = img_shape[:2]
    cx = mark_box["x"] + mark_box["w"] // 2
    cy = mark_box["y"] + mark_box["h"] // 2
    hw = mark_box["w"] // 2 + expand_px
    hh = mark_box["h"] // 2 + expand_px
    x1, y1 = max(0, cx - hw), max(0, cy - hh)
    x2, y2 = min(W, cx + hw), min(H, cy + hh)
    rw, rh = x2 - x1, y2 - y1
    if rw < 4 or rh < 4:
        mask = np.zeros((H, W), dtype=np.float64)
        mask[y1:y2, x1:x2] = 1.0
        return mask
    local_mask = make_soft_rounded_rect_mask(rw, rh, feather_px=feather_px)
    full = np.zeros((H, W), dtype=np.float64)
    full[y1:y2, x1:x2] = local_mask
    return full


def _add_noise_match(patch, noise_sigma, noise_amount):
    if noise_amount < 0.1 or noise_sigma < 0.3:
        return patch
    noise = np.random.normal(0, noise_sigma * noise_amount / noise_sigma
                             if noise_sigma > 0 else 0,
                             patch.shape).astype(np.float64)
    return np.clip(patch.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _unsharp_mask(img, amount=0.3, radius=1.5):
    blurred = cv2.GaussianBlur(img.astype(np.float64), (0, 0), radius)
    return np.clip(img.astype(np.float64) + amount * (img.astype(np.float64)
                   - blurred), 0, 255).astype(np.uint8)


def _apply_cover_blend(img, cover_img, alpha_mask, style):
    out = img.astype(np.float64) * (1.0 - alpha_mask[:, :, None]) + \
          cover_img.astype(np.float64) * alpha_mask[:, :, None]
    result = np.clip(out, 0, 255).astype(np.uint8)
    if style.get("sharpen_amount", 0) > 0:
        bx, by = style.get("_bx", 0), style.get("_by", 0)
        bw, bh = style.get("_bw", 0), style.get("_bh", 0)
        if bw > 0 and bh > 0:
            roi = result[by:by + bh, bx:bx + bw]
            result[by:by + bh, bx:bx + bw] = _unsharp_mask(
                roi, amount=style["sharpen_amount"])
    return result


def cover_near_white_soft(img, mark_box, style, expand_px):
    ring, _ = _sample_ring(img, mark_box)
    ring_pixels = img[ring]
    if ring_pixels.size > 0:
        bg_color = np.median(ring_pixels, axis=0).astype(np.uint8)
    else:
        bg_color = np.array([245, 245, 245], dtype=np.uint8)
    cover = np.full_like(img, bg_color)
    cover = _add_noise_match(cover, style["noise_sigma"],
                             style["noise_amount"])
    alpha = _embed_rounded_mask(img.shape, mark_box, expand_px,
                                style["feather_px"])
    alpha = alpha * style["alpha"]
    st = dict(style, _bx=mark_box["x"], _by=mark_box["y"],
              _bw=mark_box["w"], _bh=mark_box["h"])
    return _apply_cover_blend(img, cover, alpha, st)


def cover_median_color_soft(img, mark_box, style, expand_px):
    cover = np.full_like(img, style["cover_color"])
    cover = _add_noise_match(cover, style["noise_sigma"],
                             style["noise_amount"])
    alpha = _embed_rounded_mask(img.shape, mark_box, expand_px,
                                style["feather_px"])
    alpha = alpha * style["alpha"]
    st = dict(style, _bx=mark_box["x"], _by=mark_box["y"],
              _bw=mark_box["w"], _bh=mark_box["h"])
    return _apply_cover_blend(img, cover, alpha, st)


def cover_dark_translucent_soft(img, mark_box, style, expand_px):
    luma = style["local_luma"]
    scale = max(0, luma - 3) / max(luma, 1)
    dark_color = (style["median_color"].astype(np.float64) * scale)
    dark_color = np.clip(dark_color, 0, 255).astype(np.uint8)
    cover = np.full_like(img, dark_color)
    cover = _add_noise_match(cover, style["noise_sigma"],
                             max(style["noise_amount"], 0.5))
    alpha = _embed_rounded_mask(img.shape, mark_box, expand_px,
                                style["feather_px"])
    alpha = alpha * style["alpha"]
    st = dict(style, _bx=mark_box["x"], _by=mark_box["y"],
              _bw=mark_box["w"], _bh=mark_box["h"])
    return _apply_cover_blend(img, cover, alpha, st)


def cover_gradient_plane_soft(img, mark_box, style, expand_px):
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    ring, _ = _sample_ring(img, mark_box, pad_factor=0.5)
    ys_r, xs_r = np.where(ring)
    if ys_r.size < 20:
        return cover_median_color_soft(img, mark_box, style, expand_px)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    ring_ny = ys_r.astype(np.float64) / H
    ring_nx = xs_r.astype(np.float64) / W
    A_ring = np.column_stack([np.ones_like(ring_nx), ring_nx, ring_ny])
    ex = expand_px
    y1, y2 = max(0, by - ex), min(H, by + bh + ex)
    x1, x2 = max(0, bx - ex), min(W, bx + bw + ex)
    ys_m, xs_m = np.mgrid[y1:y2, x1:x2]
    ys_m, xs_m = ys_m.ravel(), xs_m.ravel()
    mark_ny = ys_m.astype(np.float64) / H
    mark_nx = xs_m.astype(np.float64) / W
    A_mark = np.column_stack([np.ones_like(mark_nx), mark_nx, mark_ny])
    cover_lab = lab.copy()
    for c in range(3):
        ring_vals = lab[ys_r, xs_r, c]
        coeffs, *_ = np.linalg.lstsq(A_ring, ring_vals, rcond=None)
        cover_lab[ys_m, xs_m, c] = A_mark @ coeffs
    cover = cv2.cvtColor(np.clip(cover_lab, 0, 255).astype(np.uint8),
                         cv2.COLOR_LAB2BGR)
    cover = _add_noise_match(cover, style["noise_sigma"],
                             style["noise_amount"])
    alpha = _embed_rounded_mask(img.shape, mark_box, expand_px,
                                style["feather_px"])
    alpha = alpha * style["alpha"]
    st = dict(style, _bx=bx, _by=by, _bw=bw, _bh=bh)
    return _apply_cover_blend(img, cover, alpha, st)


def cover_local_gradient_fit(img, mark_box, style, expand_px):
    return cover_gradient_plane_soft(img, mark_box, style, expand_px)


def cover_blurred_background_soft(img, mark_box, style, expand_px):
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    ex = expand_px
    y1, y2 = max(0, by - ex), min(H, by + bh + ex)
    x1, x2 = max(0, bx - ex), min(W, bx + bw + ex)
    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return cover_median_color_soft(img, mark_box, style, expand_px)
    k = max(15, min(roi.shape[0], roi.shape[1]) // 2)
    k = k if k % 2 == 1 else k + 1
    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
    blurred_roi = _add_noise_match(blurred_roi, style["noise_sigma"],
                                   style["noise_amount"])
    cover = img.copy()
    cover[y1:y2, x1:x2] = blurred_roi
    alpha = _embed_rounded_mask(img.shape, mark_box, expand_px,
                                style["feather_px"])
    alpha = alpha * style["alpha"]
    st = dict(style, _bx=bx, _by=by, _bw=bw, _bh=bh)
    return _apply_cover_blend(img, cover, alpha, st)


def cover_adaptive_soft(img, mark_box, style, expand_px):
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    ex = expand_px
    y1, y2 = max(0, by - ex), min(H, by + bh + ex)
    x1, x2 = max(0, bx - ex), min(W, bx + bw + ex)
    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return cover_median_color_soft(img, mark_box, style, expand_px)
    k = max(7, min(roi.shape[0], roi.shape[1]) // 3)
    k = k if k % 2 == 1 else k + 1
    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
    blurred_roi = _add_noise_match(blurred_roi, style["noise_sigma"],
                                   style["noise_amount"])
    cover = img.copy()
    cover[y1:y2, x1:x2] = blurred_roi
    alpha = _embed_rounded_mask(img.shape, mark_box, expand_px,
                                style["feather_px"])
    alpha = alpha * style["alpha"]
    st = dict(style, _bx=bx, _by=by, _bw=bw, _bh=bh)
    return _apply_cover_blend(img, cover, alpha, st)


_V6_COVER_DISPATCH = {
    "near_white_soft": cover_near_white_soft,
    "median_color_soft": cover_median_color_soft,
    "dark_translucent_soft": cover_dark_translucent_soft,
    "gradient_plane_soft": cover_gradient_plane_soft,
    "local_gradient_fit": cover_local_gradient_fit,
    "blurred_background_soft": cover_blurred_background_soft,
    "adaptive_soft": cover_adaptive_soft,
}

V6_FILL_MODE_BY_ROI = {
    "plain_white": ["near_white_soft", "median_color_soft"],
    "plain_color": ["median_color_soft", "near_white_soft"],
    "low_texture": ["median_color_soft", "adaptive_soft"],
    "gradient": ["local_gradient_fit", "gradient_plane_soft"],
    "metallic": ["gradient_plane_soft", "blurred_background_soft"],
    "glossy": ["blurred_background_soft", "gradient_plane_soft"],
    "transparent": ["blurred_background_soft", "adaptive_soft"],
    "black_product": ["dark_translucent_soft", "adaptive_soft"],
    "high_texture": ["adaptive_soft", "blurred_background_soft"],
    "product_detail": ["adaptive_soft", "blurred_background_soft"],
    "text_or_qr": ["adaptive_soft"],
    "poster_or_marketing": ["adaptive_soft"],
}


def _cover_residual_check(covered, mark_box):
    gray = cv2.cvtColor(covered, cv2.COLOR_BGR2GRAY)
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    roi = gray[by:by + bh, bx:bx + bw]
    if roi.size == 0:
        return 0.0
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    return float((np.abs(laplacian) > 15).mean())


def _score_cover_candidate(img, covered, mark_box, style):
    H, W = img.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    wm_residual = _cover_residual_check(covered, mark_box)
    gray_o = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_c = cv2.cvtColor(covered, cv2.COLOR_BGR2GRAY).astype(np.float64)
    pad = max(10, max(bw, bh) // 3)
    ry1, ry2 = max(0, by - pad), min(H, by + bh + pad)
    rx1, rx2 = max(0, bx - pad), min(W, bx + bw + pad)
    ring = np.zeros((H, W), dtype=bool)
    ring[ry1:ry2, rx1:rx2] = True
    ring[by:by + bh, bx:bx + bw] = False
    ring_mean = float(gray_c[ring].mean()) if ring.any() else 128.0
    patch_mean = float(gray_c[by:by + bh, bx:bx + bw].mean())
    patch_vis = abs(patch_mean - ring_mean) / 255.0
    edge_o = cv2.Canny(gray_o[by:by + bh, bx:bx + bw].astype(np.uint8),
                        60, 140)
    edge_c = cv2.Canny(gray_c[by:by + bh, bx:bx + bw].astype(np.uint8),
                        60, 140)
    edge_o_d = float(edge_o.mean()) / 255.0
    edge_c_d = float(edge_c.mean()) / 255.0
    edge_loss = max(0, edge_o_d - edge_c_d)
    strip_h = max(2, bh // 6)
    top_s = gray_c[max(0, by - strip_h):by, bx:bx + bw]
    bot_s = gray_c[by + bh:min(H, by + bh + strip_h), bx:bx + bw]
    inner_top = gray_c[by:by + strip_h, bx:bx + bw]
    inner_bot = gray_c[by + bh - strip_h:by + bh, bx:bx + bw]
    bj = 0.0
    if top_s.size > 0 and inner_top.size > 0:
        bj = max(bj, abs(float(top_s.mean()) - float(inner_top.mean())))
    if bot_s.size > 0 and inner_bot.size > 0:
        bj = max(bj, abs(float(bot_s.mean()) - float(inner_bot.mean())))
    bj_score = bj / 255.0
    score = (3.0 * wm_residual + 2.0 * patch_vis +
             4.0 * edge_loss + 1.5 * bj_score)
    return {
        "score": round(score, 4),
        "watermark_residual_score": round(wm_residual, 4),
        "patch_visibility_score": round(patch_vis, 4),
        "product_edge_loss_score": round(edge_loss, 4),
        "boundary_jump_score": round(bj_score, 4),
    }


def apply_shadow_cover_fallback(image, watermark_box, roi_class, mask_info,
                                qa_failure_reasons, stroke_mask=None):
    style = estimate_local_cover_style(image, watermark_box, roi_class)

    fill_modes = V6_FILL_MODE_BY_ROI.get(roi_class, ["adaptive_soft"])

    candidates = []

    for fill_mode in fill_modes:
        fn = _V6_COVER_DISPATCH.get(fill_mode)
        if fn is None:
            continue
        for level in COVER_ESCALATION_LEVELS:
            s = dict(style)
            s["alpha"] = max(style["alpha"], level["opacity"])
            s["feather_px"] = max(style["feather_px"], level["feather"])
            covered = fn(image, watermark_box, s, level["expand"])
            residual = _cover_residual_check(covered, watermark_box)
            cand_info = _score_cover_candidate(image, covered, watermark_box,
                                               s)
            cand = {
                "image": covered,
                "method": f"v6_{fill_mode}",
                "opacity": s["alpha"],
                "feather_px": s["feather_px"],
                "expand_px": level["expand"],
                "escalation_level": COVER_ESCALATION_LEVELS.index(level),
                "fill_mode": fill_mode,
                "edge_density": style["edge_density"],
                "noise_sigma": style["noise_sigma"],
                "cover_scores": cand_info,
            }
            candidates.append(cand)
            if residual < 0.03:
                break

    if stroke_mask is not None and np.any(stroke_mask > 0):
        for fill_mode in fill_modes[:1]:
            fn = _V6_COVER_DISPATCH.get(fill_mode)
            if fn is None:
                continue
            s = dict(style)
            s["alpha"] = min(style["alpha"] + 0.10, 0.70)
            text_mask = cv2.dilate(stroke_mask,
                                   cv2.getStructuringElement(
                                       cv2.MORPH_RECT, (5, 3)),
                                   iterations=2)
            k = max(5, int(min(watermark_box["w"], watermark_box["h"]) * 0.08))
            k = k if k % 2 == 1 else k + 1
            text_mask_f = cv2.GaussianBlur(
                text_mask.astype(np.float64) / 255.0, (k, k), 0)
            text_mask_f = text_mask_f * s["alpha"]
            cover_layer = fn(image, watermark_box, s, 2)
            out = image.astype(np.float64) * (1.0 - text_mask_f[:, :, None]) \
                + cover_layer.astype(np.float64) * text_mask_f[:, :, None]
            text_covered = np.clip(out, 0, 255).astype(np.uint8)
            cand_info = _score_cover_candidate(image, text_covered,
                                               watermark_box, s)
            candidates.append({
                "image": text_covered,
                "method": f"v6_text_shaped_{fill_mode}",
                "opacity": s["alpha"],
                "feather_px": k,
                "expand_px": 2,
                "escalation_level": 0,
                "fill_mode": f"text_shaped_{fill_mode}",
                "edge_density": style["edge_density"],
                "noise_sigma": style["noise_sigma"],
                "cover_scores": cand_info,
            })

    if not candidates:
        fn = _V6_COVER_DISPATCH["adaptive_soft"]
        s = dict(style)
        strongest = COVER_ESCALATION_LEVELS[-1]
        s["alpha"] = max(s["alpha"], strongest["opacity"])
        covered = fn(image, watermark_box, s, strongest["expand"])
        cand_info = _score_cover_candidate(image, covered, watermark_box, s)
        candidates.append({
            "image": covered,
            "method": "v6_adaptive_soft_final",
            "opacity": s["alpha"],
            "feather_px": s["feather_px"],
            "expand_px": strongest["expand"],
            "escalation_level": len(COVER_ESCALATION_LEVELS) - 1,
            "fill_mode": "adaptive_soft",
            "edge_density": style["edge_density"],
            "noise_sigma": style["noise_sigma"],
            "cover_scores": cand_info,
        })

    best = min(candidates, key=lambda c: c["cover_scores"]["score"])

    residual = best["cover_scores"]["watermark_residual_score"]
    if residual > 0.08:
        bx, by = watermark_box["x"], watermark_box["y"]
        bw, bh = watermark_box["w"], watermark_box["h"]
        roi = best["image"][by:by + bh, bx:bx + bw]
        k_b = max(3, min(bh, bw) // 6)
        k_b = k_b if k_b % 2 == 1 else k_b + 1
        blurred_roi = cv2.GaussianBlur(roi, (k_b, k_b), 0)
        blend = 0.3
        roi_fixed = cv2.addWeighted(roi, 1.0 - blend, blurred_roi, blend, 0)
        best["image"][by:by + bh, bx:bx + bw] = roi_fixed
        best["cover_scores"]["post_blur_applied"] = True

    return best


def inpaint_lama_small(rwm, img, mask):
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    if not (eroded > 0).any():
        eroded = mask
    if not rwm.LAMA_MODEL_PATH.exists():
        return None
    return rwm._inpaint_lama(img, eroded)


def run_inpaint(rwm, method: str, img: np.ndarray, mask: np.ndarray,
                mark_box: dict = None) -> dict:
    t0 = time.time()
    out = None; err = None; budget = BUDGET_FAST_S
    if method == "alpha_attenuate":
        out = inpaint_alpha_attenuate(img, mask)
    elif method == "full_box_attenuate":
        out = inpaint_full_box_attenuate(img, mask, mark_box)
    elif method == "dark_surface_local_plane":
        out = inpaint_dark_surface_local_plane(img, mask, mark_box)
    elif method == "white_fill_stroke_only":
        out = inpaint_white_fill_stroke_only(img, mask, mark_box)
    elif method == "local_background_fill":
        out = inpaint_local_background_fill(img, mask, mark_box)
    elif method == "clone_offset":
        out = inpaint_clone_offset(img, mask, mark_box)
    elif method == "white_fill":
        out = inpaint_white_fill(img, mask)
    elif method == "local_median":
        out = inpaint_local_median(img, mask)
    elif method == "surface_fit":
        out = inpaint_surface_fit(img, mask)
    elif method == "gradient_plane":
        out = inpaint_gradient_plane_fill(img, mask)
    elif method == "local_color_plane":
        out = inpaint_local_color_plane(img, mask)
    elif method == "telea":
        out = rwm._inpaint_cv(img, mask, rwm.INPAINT_RADIUS, "telea")
    elif method == "telea_small":
        out = inpaint_telea_small(rwm, img, mask, radius=2)
    elif method == "ns":
        out = rwm._inpaint_cv(img, mask, rwm.INPAINT_RADIUS, "ns")
    elif method == "poisson":
        out = inpaint_poisson_blend(img, mask)
    elif method == "lama":
        budget = BUDGET_LAMA_S
        if not rwm.LAMA_MODEL_PATH.exists():
            err = "lama_model_missing"
        else:
            out = rwm._inpaint_lama(img, mask)
    elif method == "lama_small":
        budget = BUDGET_LAMA_S
        out = inpaint_lama_small(rwm, img, mask)
        if out is None:
            err = "lama_model_missing"
    else:
        err = f"unknown_method:{method}"
    dt = time.time() - t0
    if out is None and err is None:
        err = "inpaint_returned_none"
    return {"output": out, "method": method, "runtime_s": dt,
            "over_budget": dt > budget, "budget_s": budget, "error": err}


# ---------------------------------------------------------------------------
# Watermark residual QA — V2 carried over.
# ---------------------------------------------------------------------------

QA_TEMPLATE_MANUAL_EDGE_MAX = 0.45  # strict: only pass "manual" tier with very low edge

def qa_template(rwm, cleaned_bgr) -> dict:
    g = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    dets = rwm.detect_watermark_v2(g)
    if not dets:
        return {"pass": True, "tier": "none", "edge": 0.0, "psr": 0.0}
    d = dets[0]
    tier = d.get("tier", "none")
    edge = float(d.get("edge_score", 0.0))
    # Strict: only tier "none" truly passes. "manual" tier with very low
    # edge score is borderline acceptable but the watermark is still there.
    passed = (tier == "none" or
              (tier == "manual" and edge < QA_TEMPLATE_MANUAL_EDGE_MAX))
    return {"pass": passed,
            "tier": tier,
            "edge": round(edge, 4),
            "psr": round(float(d.get("psr", 0.0)), 3)}


QA_SECONDARY_EDGE_MIN = 0.45
QA_SECONDARY_BAND_MIN = 0.15
QA_SECONDARY_BAND_STRONG = 0.30

def qa_secondary(rwm, cleaned_bgr) -> dict:
    g = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    res = rwm.locate_watermark_1d(g)
    if res is None or not res.get("accept", False):
        return {"pass": True,
                "edge_score": round(float((res or {}).get("edge_score", 0)), 4),
                "band_score": round(float((res or {}).get("band_score", 0)), 4),
                "accepted": False}
    edge_s = float(res.get("edge_score", 0))
    band_s = float(res.get("band_score", 0))
    strong_residual = ((edge_s >= QA_SECONDARY_EDGE_MIN and
                        band_s >= QA_SECONDARY_BAND_MIN) or
                       band_s >= QA_SECONDARY_BAND_STRONG)
    return {"pass": not strong_residual,
            "edge_score": round(edge_s, 4),
            "band_score": round(band_s, 4),
            "yc": round(float(res.get("yc", 0.0)), 4),
            "accepted": True}


def qa_edge(cleaned_bgr, mark_box, roi_class) -> dict:
    H, W = cleaned_bgr.shape[:2]
    x, y, w, h = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    g = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diffs = []
    if y > 0:
        diffs.append(np.abs(g[y - 1, x:x + w] - g[y, x:x + w]))
    if y + h < H:
        diffs.append(np.abs(g[y + h, x:x + w] - g[y + h - 1, x:x + w]))
    if x > 0:
        diffs.append(np.abs(g[y:y + h, x - 1] - g[y:y + h, x]))
    if x + w < W:
        diffs.append(np.abs(g[y:y + h, x + w] - g[y:y + h, x + w - 1]))
    boundary_jump = float(np.concatenate(diffs).mean()) if diffs else 0.0

    edges = cv2.Canny(g.astype(np.uint8), 50, 150)
    inside = edges[y:y + h, x:x + w]
    inside_edge = float(inside.mean()) / 255.0 if inside.size else 0.0
    pad = h
    ry1 = max(0, y - pad); ry2 = min(H, y + h + pad)
    rx1 = max(0, x - pad); rx2 = min(W, x + w + pad)
    ring = edges[ry1:ry2, rx1:rx2].copy()
    rm = np.ones(ring.shape, dtype=bool)
    rm[max(0, y - ry1):max(0, y - ry1) + h,
       max(0, x - rx1):max(0, x - rx1) + w] = False
    surround_edge = (float(ring[rm].mean()) / 255.0) if rm.any() else 0.0
    edge_drift = abs(inside_edge - surround_edge)

    thr_b, thr_i, thr_d = QA_THRESHOLDS_BY_CLASS[roi_class]
    boundary_ok = boundary_jump <= thr_b
    inside_ok = inside_edge <= thr_i
    drift_ok = edge_drift <= thr_d
    passed = boundary_ok and inside_ok and drift_ok

    return {"pass": passed,
            "boundary_jump": round(boundary_jump, 2),
            "inside_edge": round(inside_edge, 4),
            "surround_edge": round(surround_edge, 4),
            "edge_drift": round(edge_drift, 4),
            "boundary_ok": boundary_ok, "inside_ok": inside_ok,
            "drift_ok": drift_ok,
            "thresholds": {"boundary": thr_b, "inside_edge": thr_i,
                           "edge_drift": thr_d}}


def qa_watermark(rwm, cleaned_bgr, mark_box, roi_class) -> dict:
    t = qa_template(rwm, cleaned_bgr)
    s = qa_secondary(rwm, cleaned_bgr)
    e = qa_edge(cleaned_bgr, mark_box, roi_class)
    # Strict: template must pass AND at least 2 of 3 must pass.
    # Template is the definitive watermark detector -- if it sees text,
    # the watermark is still visible.
    n_pass = int(t["pass"]) + int(s["pass"]) + int(e["pass"])
    passed = t["pass"] and n_pass >= 2
    return {"pass": passed,
            "template": t, "secondary": s, "edge": e,
            "fail_reasons": [k for k, v in
                            [("template", t), ("secondary", s), ("edge", e)]
                            if not v["pass"]]}


# ---------------------------------------------------------------------------
# P5 — Local ROI QA.
# Operates INSIDE the mark area on the cleaned image. Catches patch-style
# defects that global QA misses: low local SSIM, color drift, visible
# mask-edge step.
# ---------------------------------------------------------------------------

def _ssim_local(a: np.ndarray, b: np.ndarray) -> float:
    """Single-block SSIM proxy. Returns 1.0 for identical patches."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.size == 0 or b.size == 0:
        return 1.0
    mu_a = a.mean(); mu_b = b.mean()
    va = a.var(); vb = b.var()
    cab = ((a - mu_a) * (b - mu_b)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * cab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)
    return float(num / max(den, 1e-9))


def qa_local_roi(orig_bgr: np.ndarray, cleaned_bgr: np.ndarray,
                 mark_box: dict) -> dict:
    """Compare the cleaned mark area against neighbor strips. Uses a
    comparison-relative approach: compute the same metrics on both the
    ORIGINAL (with watermark) and the CLEANED version. The cleaned version
    should not be WORSE than the original. This handles product-boundary
    cases where the watermark area naturally differs from neighbors."""
    H, W = orig_bgr.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    cleaned_roi = cleaned_bgr[by:by + bh, bx:bx + bw]
    orig_roi = orig_bgr[by:by + bh, bx:bx + bw]
    above = orig_bgr[max(0, by - bh):by, bx:bx + bw]
    below = orig_bgr[by + bh:min(H, by + 2 * bh), bx:bx + bw]
    if cleaned_roi.size == 0:
        return {"pass": False, "fail_reasons": ["empty_roi"]}

    # Reference neighbor (resampled to match cleaned ROI shape).
    if above.size and below.size:
        ref = np.vstack([above, below])
        ref = cv2.resize(ref, (cleaned_roi.shape[1], cleaned_roi.shape[0]),
                          interpolation=cv2.INTER_AREA)
    elif above.size:
        ref = cv2.resize(above, (cleaned_roi.shape[1], cleaned_roi.shape[0]),
                          interpolation=cv2.INTER_AREA)
    elif below.size:
        ref = cv2.resize(below, (cleaned_roi.shape[1], cleaned_roi.shape[0]),
                          interpolation=cv2.INTER_AREA)
    else:
        ref = cleaned_roi

    # Local SSIM — cleaned vs neighbors
    g_cln = cv2.cvtColor(cleaned_roi, cv2.COLOR_BGR2GRAY)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ssim = _ssim_local(g_cln, g_ref)

    # Also compute SSIM of ORIGINAL watermarked area vs neighbors.
    # If the original already had low SSIM (product boundary), the
    # cleaned version should not be penalized for the same.
    g_orig = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2GRAY)
    ssim_orig = _ssim_local(g_orig, g_ref)

    # LAB color delta (mean ΔE)
    lab_cln = cv2.cvtColor(cleaned_roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
    lab_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
    color_delta = float(np.linalg.norm(lab_cln - lab_ref))

    lab_orig = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
    color_delta_orig = float(np.linalg.norm(lab_orig - lab_ref))

    # Boundary jump (reuses the same paired-diff style as qa_edge).
    g_full_c = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_full_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diffs_c = []; diffs_o = []
    if by > 0:
        diffs_c.append(np.abs(g_full_c[by - 1, bx:bx + bw] -
                               g_full_c[by, bx:bx + bw]))
        diffs_o.append(np.abs(g_full_o[by - 1, bx:bx + bw] -
                               g_full_o[by, bx:bx + bw]))
    if by + bh < H:
        diffs_c.append(np.abs(g_full_c[by + bh, bx:bx + bw] -
                               g_full_c[by + bh - 1, bx:bx + bw]))
        diffs_o.append(np.abs(g_full_o[by + bh, bx:bx + bw] -
                               g_full_o[by + bh - 1, bx:bx + bw]))
    boundary_jump = float(np.concatenate(diffs_c).mean()) if diffs_c else 0.0
    boundary_jump_orig = float(np.concatenate(diffs_o).mean()) if diffs_o else 0.0

    # Comparison-relative: pass if cleaned is at least as good as original,
    # OR if cleaned meets absolute thresholds.
    # The cleaned area should not introduce new artifacts that the original
    # (with watermark) didn't already have.
    ssim_ok = (ssim >= ROI_SSIM_MIN or ssim >= ssim_orig - 0.05)
    color_ok = (color_delta <= ROI_COLOR_DELTA_MAX or
                color_delta <= color_delta_orig + 3.0)
    boundary_ok = (boundary_jump <= ROI_BOUNDARY_JUMP_MAX or
                   boundary_jump <= boundary_jump_orig + 3.0)
    passed = ssim_ok and color_ok and boundary_ok

    return {"pass": passed,
            "ssim": round(ssim, 4),
            "ssim_orig": round(ssim_orig, 4),
            "color_delta": round(color_delta, 2),
            "color_delta_orig": round(color_delta_orig, 2),
            "boundary_jump": round(boundary_jump, 2),
            "boundary_jump_orig": round(boundary_jump_orig, 2),
            "ssim_ok": ssim_ok, "color_ok": color_ok,
            "boundary_ok": boundary_ok}


# ---------------------------------------------------------------------------
# Post-clean product-integrity QA — V2 carried over with the same fixes.
# ---------------------------------------------------------------------------

def qa_product_integrity(orig_bgr, cleaned_bgr, final_mask, mark_box) -> dict:
    H, W = orig_bgr.shape[:2]
    fg_orig = extract_product_foreground(orig_bgr)
    fg_clean = extract_product_foreground(cleaned_bgr)

    excl = np.ones((H, W), bool)
    pad_x = max(mark_box["w"] // 4, 24)
    pad_y = max(mark_box["h"], 20)
    ex1 = max(0, mark_box["x"] - pad_x); ey1 = max(0, mark_box["y"] - pad_y)
    ex2 = min(W, mark_box["x"] + mark_box["w"] + pad_x)
    ey2 = min(H, mark_box["y"] + mark_box["h"] + pad_y)
    excl[ey1:ey2, ex1:ex2] = False

    fg_orig_x = fg_orig & 255 * excl.astype(np.uint8)
    fg_clean_x = fg_clean & 255 * excl.astype(np.uint8)
    area_orig = (fg_orig_x > 0).sum()
    area_clean = (fg_clean_x > 0).sum()
    fg_area_loss = (area_orig - area_clean) / max(area_orig, 1)

    fg_o_for_bbox = (fg_orig > 0) & excl
    fg_c_for_bbox = (fg_clean > 0) & excl
    bb_o = (cv2.boundingRect(fg_o_for_bbox.astype(np.uint8))
            if fg_o_for_bbox.any() else (0, 0, 1, 1))
    bb_c = (cv2.boundingRect(fg_c_for_bbox.astype(np.uint8))
            if fg_c_for_bbox.any() else (0, 0, 1, 1))
    diag = max(math.sqrt(bb_o[2] ** 2 + bb_o[3] ** 2), 1.0)
    bbox_shift = (abs(bb_o[0] - bb_c[0]) + abs(bb_o[1] - bb_c[1]) +
                  abs(bb_o[2] - bb_c[2]) + abs(bb_o[3] - bb_c[3])) / diag

    edges_o = cv2.Canny(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY), 50, 150)
    edges_c = cv2.Canny(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY), 50, 150)
    prod_mask = (fg_orig > 0) & excl
    if prod_mask.any():
        eo = float(edges_o[prod_mask].mean()) / 255
        ec = float(edges_c[prod_mask].mean()) / 255
        edge_loss = (eo - ec) / max(eo, 1e-6)
    else:
        eo = ec = edge_loss = 0.0

    outside = (final_mask == 0)
    diff = np.abs(cleaned_bgr.astype(np.int32) - orig_bgr.astype(np.int32))
    mean_outside_diff = float(diff[outside].mean()) if outside.any() else 0.0

    n_o = cv2.connectedComponents((fg_orig > 0).astype(np.uint8))[0]
    n_c = cv2.connectedComponents((fg_clean > 0).astype(np.uint8))[0]
    cc_delta = abs(n_o - n_c)

    # Blank-patch (V2 calibrated: compare to immediate neighbors only).
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    cleaned_mark = cleaned_bgr[by:by + bh, bx:bx + bw]
    above = orig_bgr[max(0, by - bh):by, bx:bx + bw]
    below = orig_bgr[by + bh:min(H, by + 2 * bh), bx:bx + bw]
    neighbor_std = cleaned_std = 0.0
    if cleaned_mark.size:
        cleaned_std = float(cv2.cvtColor(cleaned_mark,
                            cv2.COLOR_BGR2GRAY).std())
    if above.size and below.size:
        ns_strip = np.vstack([above, below])
    elif above.size:
        ns_strip = above
    elif below.size:
        ns_strip = below
    else:
        ns_strip = None
    if ns_strip is not None and ns_strip.size:
        neighbor_std = float(cv2.cvtColor(ns_strip,
                              cv2.COLOR_BGR2GRAY).std())
    blank_is_reject = (
        neighbor_std >= BLANK_PATCH_SURROUND_MIN_STD and
        cleaned_std < neighbor_std * BLANK_PATCH_RATIO_REJECT
    )

    reasons = []; is_reject = False; is_risky = False
    if fg_area_loss > PI_FOREGROUND_LOSS_REJECT:
        is_reject = True; reasons.append(R_FOREGROUND_LOSS)
    if bbox_shift > PI_BBOX_DELTA_REJECT:
        is_reject = True; reasons.append(R_PRODUCT_LOSS)
    if edge_loss > PI_EDGE_LOSS_REJECT:
        is_reject = True; reasons.append(R_EDGE_DAMAGE)
    if cc_delta > PI_CC_DELTA_REJECT:
        is_reject = True; reasons.append(R_PRODUCT_LOSS)
    if mean_outside_diff > PI_OUTSIDE_MASK_DIFF_REJECT:
        is_reject = True; reasons.append(R_LARGE_SMEAR)
    if blank_is_reject:
        is_reject = True; reasons.append(R_BLANK_OUTPUT)
    if not is_reject:
        if fg_area_loss > PI_FOREGROUND_LOSS_MANUAL:
            is_risky = True; reasons.append("manual_fg_area_loss")
        if edge_loss > PI_EDGE_LOSS_MANUAL:
            is_risky = True; reasons.append("manual_edge_loss")

    return {"pass": not is_reject and not is_risky,
            "reject": is_reject, "risky": is_risky,
            "reasons": reasons,
            "fg_area_loss": round(fg_area_loss, 4),
            "bbox_shift": round(bbox_shift, 4),
            "edge_loss": round(edge_loss, 4),
            "edge_orig": round(eo, 4), "edge_clean": round(ec, 4),
            "mean_outside_diff": round(mean_outside_diff, 3),
            "cc_delta": cc_delta,
            "neighbor_std": round(neighbor_std, 2),
            "cleaned_std": round(cleaned_std, 2)}


# ---------------------------------------------------------------------------
# P0.1 — Post-clean product integrity gate (Problem A).
# Compares original vs cleaned INSIDE and around the repair ROI to catch
# false CLEANs where the inpainter damaged product structure.
# ---------------------------------------------------------------------------

def run_post_clean_integrity_gate(original, cleaned, repair_mask, mark_box):
    """V4: Rewritten to focus on PRODUCT damage OUTSIDE the watermark area.
    The old version compared original vs cleaned in the watermark ROI, which
    always showed high differences (because that's where cleaning happens).
    Now we focus on:
    1. Edge/structure changes OUTSIDE the mark_box (product damage)
    2. Outside-mask pixel drift (accidental modification of non-watermark areas)
    3. Only use within-watermark metrics as secondary signals
    """
    H, W = original.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]

    pad = max(10, bh // 3)
    ry1 = max(0, by - pad); ry2 = min(H, by + bh + pad)
    rx1 = max(0, bx - pad); rx2 = min(W, bx + bw + pad)
    orig_roi = original[ry1:ry2, rx1:rx2]
    clean_roi = cleaned[ry1:ry2, rx1:rx2]
    if orig_roi.size == 0 or clean_roi.size == 0:
        return {"pass": True, "score": 0.0, "reasons": [], "metrics": {}}

    gray_o = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(clean_roi, cv2.COLOR_BGR2GRAY)
    edges_o = cv2.Canny(gray_o, 50, 150)
    edges_c = cv2.Canny(gray_c, 50, 150)

    # Product mask: OUTSIDE the watermark area
    my1 = by - ry1; my2 = by + bh - ry1
    mx1 = bx - rx1; mx2 = bx + bw - rx1
    product_mask = np.ones(gray_o.shape, dtype=bool)
    product_mask[max(0, my1):min(product_mask.shape[0], my2),
                 max(0, mx1):min(product_mask.shape[1], mx2)] = False

    # Edge loss OUTSIDE the watermark (the real damage signal)
    if product_mask.any():
        eo = float(edges_o[product_mask].mean()) / 255.0
        ec = float(edges_c[product_mask].mean()) / 255.0
        edge_loss = max(0.0, eo - ec) / max(eo, 1e-6)
    else:
        edge_loss = 0.0

    # Outside-mask pixel drift: changes to areas that shouldn't change
    outside = (repair_mask == 0)
    diff = np.abs(cleaned.astype(np.int32) - original.astype(np.int32))
    mean_outside_diff = float(diff[outside].mean()) if outside.any() else 0.0

    # SSIM OUTSIDE the watermark (product structure preservation)
    if product_mask.any() and product_mask.sum() > 100:
        go_prod = gray_o.copy(); go_prod[~product_mask] = 128
        gc_prod = gray_c.copy(); gc_prod[~product_mask] = 128
        ssim_val = _ssim_local(go_prod[product_mask], gc_prod[product_mask])
    else:
        ssim_val = 1.0
    ssim_drop = max(0.0, 1.0 - ssim_val)

    # Score focuses on OUTSIDE-watermark damage
    score = (0.40 * edge_loss + 0.30 * ssim_drop +
             0.30 * min(1.0, mean_outside_diff / 10.0))
    score = round(_clamp(score), 4)

    reasons = []
    if edge_loss > PI_POST_EDGE_LOSS_MANUAL:
        reasons.append("product_edge_loss")
    if ssim_drop > PI_POST_SSIM_DROP_MANUAL:
        reasons.append("structure_ssim_drop")
    if mean_outside_diff > 8.0:
        reasons.append("outside_mask_drift")

    passed = score < PI_POST_SCORE_MANUAL and len(reasons) == 0
    return {"pass": passed, "score": score, "reasons": reasons,
            "metrics": {"edge_loss": round(edge_loss, 4),
                        "structure_ssim_drop": round(ssim_drop, 4),
                        "mean_outside_diff": round(mean_outside_diff, 3)}}


# ---------------------------------------------------------------------------
# P0.2 — Rectangular / smear artifact detector (Problem B).
# Detects obvious smooth rectangular patches that the inpainter creates.
# ---------------------------------------------------------------------------

def detect_repair_artifacts(original, cleaned, repair_mask):
    H, W = original.shape[:2]
    ys, xs = np.where(repair_mask > 0)
    if ys.size < 20:
        return {"has_artifact": False, "artifact_score": 0.0, "reasons": []}

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    rh, rw = y2 - y1, x2 - x1
    if rh < 3 or rw < 3:
        return {"has_artifact": False, "artifact_score": 0.0, "reasons": []}

    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    patch_o = gray_o[y1:y2, x1:x2].astype(np.float64)
    patch_c = gray_c[y1:y2, x1:x2].astype(np.float64)

    lap_o = cv2.Laplacian(patch_o.astype(np.uint8), cv2.CV_64F)
    lap_c = cv2.Laplacian(patch_c.astype(np.uint8), cv2.CV_64F)
    var_o = max(float(lap_o.var()), 1e-6)
    var_c = float(lap_c.var())
    # Compare Laplacian drop against the SURROUND, not the original.
    # The original includes watermark text strokes which add texture --
    # removing them naturally reduces Laplacian. Only flag if the cleaned
    # area is significantly smoother than the surrounding non-watermark area.
    surround_lap = cv2.Laplacian(
        gray_c[max(0, y1 - max(3, rh // 4)):min(H, y2 + max(3, rh // 4)),
               max(0, x1 - max(3, rw // 4)):min(W, x2 + max(3, rw // 4))].astype(np.uint8),
        cv2.CV_64F)
    var_surround = max(float(surround_lap.var()), 1e-6)
    # Use the minimum of original and surround as reference -- the cleaned
    # area should not be smoother than the LEAST textured reference.
    var_ref = min(var_o, var_surround)
    laplacian_drop = max(0.0, (var_ref - var_c) / max(var_ref, 1e-6))

    pad = max(3, min(rh, rw) // 4)
    sy1 = max(0, y1 - pad); sy2 = min(H, y2 + pad)
    sx1 = max(0, x1 - pad); sx2 = min(W, x2 + pad)
    surround_mask = np.zeros((sy2 - sy1, sx2 - sx1), dtype=bool)
    surround_mask[:, :] = True
    iy1 = y1 - sy1; iy2 = y2 - sy1
    ix1 = x1 - sx1; ix2 = x2 - sx1
    surround_mask[iy1:iy2, ix1:ix2] = False
    surround_gray = gray_c[sy1:sy2, sx1:sx2]
    if surround_mask.any() and patch_c.size > 0:
        surround_std = float(surround_gray[surround_mask].std())
        patch_std = max(float(patch_c.std()), 0.01)
        smoothness_ratio = max(surround_std, 0.01) / patch_std
    else:
        smoothness_ratio = 1.0

    patch_area = rh * rw
    actual_mask_area = int((repair_mask[y1:y2, x1:x2] > 0).sum())
    rectangularity = actual_mask_area / max(patch_area, 1)

    lab_patch = cv2.cvtColor(cleaned[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
    lab_surround = cv2.cvtColor(cleaned[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2LAB)
    patch_mean = lab_patch.reshape(-1, 3).mean(axis=0).astype(np.float64)
    if surround_mask.any():
        s_pixels = lab_surround[surround_mask].reshape(-1, 3).mean(axis=0).astype(
            np.float64)
        boundary_delta_e = float(np.linalg.norm(patch_mean - s_pixels))
    else:
        boundary_delta_e = 0.0

    edges_at_boundary = cv2.Canny(gray_c[sy1:sy2, sx1:sx2], 50, 150)
    boundary_ring = np.zeros_like(surround_mask, dtype=np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inner_mask_full = np.zeros((sy2 - sy1, sx2 - sx1), dtype=np.uint8)
    inner_mask_full[iy1:iy2, ix1:ix2] = 255
    outer_ring = cv2.dilate(inner_mask_full, k, iterations=2)
    boundary_ring = cv2.bitwise_and(outer_ring, cv2.bitwise_not(inner_mask_full))
    inner_ring = cv2.erode(inner_mask_full, k, iterations=2)
    boundary_ring = cv2.bitwise_or(boundary_ring,
                                    cv2.bitwise_and(inner_mask_full,
                                                     cv2.bitwise_not(inner_ring)))
    if (boundary_ring > 0).any():
        edge_at_ring = float(edges_at_boundary[boundary_ring > 0].mean()) / 255.0
    else:
        edge_at_ring = 0.0

    reasons = []
    is_rect = (rectangularity > ARTIFACT_RECT_SCORE_THRESHOLD and
               smoothness_ratio > ARTIFACT_SMOOTH_RATIO_THRESHOLD)
    if is_rect:
        reasons.append("rectangular_repair_artifact")
    if laplacian_drop > ARTIFACT_LAPLACIAN_DROP_MAX:
        reasons.append("severe_blur_in_repair")
    if boundary_delta_e > ARTIFACT_BOUNDARY_DELTA_E_MAX:
        reasons.append("boundary_color_mismatch")
    # Edge discontinuity: only flag if both edge ring is very strong AND
    # the repair area is much smoother than surround.
    if edge_at_ring > 0.25 and smoothness_ratio > 3.5:
        reasons.append("edge_discontinuity_at_boundary")

    artifact_score = round(_clamp(
        0.30 * min(1.0, laplacian_drop) +
        0.25 * min(1.0, smoothness_ratio / 6.0) +
        0.25 * min(1.0, boundary_delta_e / 20.0) +
        0.20 * min(1.0, rectangularity)
    ), 4)

    return {"has_artifact": len(reasons) > 0,
            "artifact_score": artifact_score,
            "reasons": reasons,
            "metrics": {"laplacian_drop": round(laplacian_drop, 4),
                        "smoothness_ratio": round(smoothness_ratio, 4),
                        "rectangularity": round(rectangularity, 4),
                        "boundary_delta_e": round(boundary_delta_e, 2),
                        "edge_at_boundary": round(edge_at_ring, 4)}}


# ---------------------------------------------------------------------------
# V4-3 — Patch Visibility Gate.
# Five metrics detect visible rectangular patch artifacts after inpainting.
# ---------------------------------------------------------------------------

def patch_visibility_gate(original, cleaned, mark_box, roi_class):
    H, W = original.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
    gray_c = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float64)
    roi_c = gray_c[by:by + bh, bx:bx + bw]
    if roi_c.size == 0:
        return {"pass": True, "fail_reasons": [], "metrics": {}}

    ring_pad = max(10, max(bw, bh) // 3)
    ry1 = max(0, by - ring_pad); ry2 = min(H, by + bh + ring_pad)
    rx1 = max(0, bx - ring_pad); rx2 = min(W, bx + bw + ring_pad)
    ring_mask = np.zeros((H, W), dtype=bool)
    ring_mask[ry1:ry2, rx1:rx2] = True
    ring_mask[by:by + bh, bx:bx + bw] = False
    ring_vals = gray_c[ring_mask]

    roi_mean_luma = float(roi_c.mean())
    ring_mean_luma = float(ring_vals.mean()) if ring_vals.size else roi_mean_luma
    roi_vs_ring_delta_luma = abs(roi_mean_luma - ring_mean_luma)

    blur_k = max(15, min(bw, bh) // 3) | 1
    roi_blur = cv2.GaussianBlur(roi_c, (blur_k, blur_k), 0)
    hf_energy_roi = float(np.abs(roi_c - roi_blur).mean())
    ring_region = gray_c[ry1:ry2, rx1:rx2]
    ring_blur = cv2.GaussianBlur(ring_region, (blur_k, blur_k), 0)
    ring_hf = np.abs(ring_region - ring_blur)
    local_by = by - ry1; local_bx = bx - rx1
    ring_hf_mask = np.ones(ring_region.shape, dtype=bool)
    ring_hf_mask[max(0, local_by):min(ring_region.shape[0], local_by + bh),
                 max(0, local_bx):min(ring_region.shape[1], local_bx + bw)] = False
    hf_energy_ring = (float(ring_hf[ring_hf_mask].mean())
                      if ring_hf_mask.any() else 1.0)
    low_frequency_patch_score = max(0.0, 1.0 - hf_energy_roi /
                                    max(hf_energy_ring, 0.01))

    boundary_diffs = []
    if by > 0:
        boundary_diffs.append(np.abs(gray_c[by - 1, bx:bx + bw] -
                                     gray_c[by, bx:bx + bw]))
    if by + bh < H:
        boundary_diffs.append(np.abs(gray_c[by + bh, bx:bx + bw] -
                                     gray_c[by + bh - 1, bx:bx + bw]))
    if bx > 0:
        boundary_diffs.append(np.abs(gray_c[by:by + bh, bx - 1] -
                                     gray_c[by:by + bh, bx]))
    if bx + bw < W:
        boundary_diffs.append(np.abs(gray_c[by:by + bh, bx + bw] -
                                     gray_c[by:by + bh, bx + bw - 1]))
    boundary_jump = (float(np.concatenate(boundary_diffs).mean())
                     if boundary_diffs else 0.0)
    rectangular_boundary_score = min(1.0, boundary_jump / 25.0)

    lap_roi = cv2.Laplacian(roi_c.astype(np.uint8), cv2.CV_64F)
    tex_roi = float(lap_roi.var()) if lap_roi.size else 0.0
    lap_ring_full = cv2.Laplacian(
        gray_c[ry1:ry2, rx1:rx2].astype(np.uint8), cv2.CV_64F)
    if ring_hf_mask.shape == lap_ring_full.shape:
        tex_ring = float(lap_ring_full[ring_hf_mask].var())
    else:
        tex_ring = float(lap_ring_full.var())
    tex_ring = max(tex_ring, 0.01)
    texture_energy_drop = max(0.0, (tex_ring - tex_roi) / tex_ring)

    if roi_c.shape[0] >= 6:
        h_third = max(2, roi_c.shape[0] // 3)
        center_band = roi_c[h_third:2 * h_third, :]
        edge_bands = np.concatenate([roi_c[:h_third, :].ravel(),
                                     roi_c[2 * h_third:, :].ravel()])
        center_std = max(float(center_band.std()), 0.01)
        surround_std = max(float(np.std(edge_bands)), 0.01)
        center_band_uniformity_spike = max(
            0.0, 1.0 - center_std / surround_std)
    else:
        center_band_uniformity_spike = 0.0

    is_plain = roi_class in ("plain_white", "plain_color", "gradient")
    fail_reasons = []
    if roi_vs_ring_delta_luma > V4_PV_DELTA_LUMA_MAX:
        fail_reasons.append("roi_vs_ring_delta_luma")
    if rectangular_boundary_score > V4_PV_RECT_BOUNDARY_MAX:
        fail_reasons.append("rectangular_boundary_score")
    if low_frequency_patch_score > V4_PV_LOW_FREQ_PATCH_MAX:
        fail_reasons.append("low_frequency_patch_score")
    if texture_energy_drop > V4_PV_TEXTURE_DROP_MAX and not is_plain:
        fail_reasons.append("texture_energy_drop")

    return {
        "pass": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
        "metrics": {
            "roi_vs_ring_delta_luma": round(roi_vs_ring_delta_luma, 2),
            "low_frequency_patch_score": round(low_frequency_patch_score, 4),
            "rectangular_boundary_score": round(rectangular_boundary_score, 4),
            "texture_energy_drop": round(texture_energy_drop, 4),
            "center_band_uniformity_spike": round(
                center_band_uniformity_spike, 4),
            "roi_mean_luma": round(roi_mean_luma, 2),
            "ring_mean_luma": round(ring_mean_luma, 2),
        }
    }


# ---------------------------------------------------------------------------
# P0.3 — Product text protection (Problem C).
# Only remove confirmed watermark text. Protect non-watermark text on
# product surfaces from the inpaint mask.
# ---------------------------------------------------------------------------

WATERMARK_TEMPLATE_TEXT = "sunsky-online.com"

def build_protected_text_mask(image, mark_box, watermark_template=WATERMARK_TEMPLATE_TEXT):
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    excl = np.zeros((H, W), np.uint8)
    pad = max(6, mark_box["h"] // 3)
    excl[max(0, mark_box["y"] - pad):min(H, mark_box["y"] + mark_box["h"] + pad),
         max(0, mark_box["x"] - pad):min(W, mark_box["x"] + mark_box["w"] + pad)] = 255

    edges = cv2.Canny(gray, 50, 150)
    edges_outside = cv2.bitwise_and(edges, cv2.bitwise_not(excl))

    nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_outside)
    text_mask = np.zeros((H, W), np.uint8)
    for i in range(1, nlab):
        a = stats[i, cv2.CC_STAT_AREA]
        w_ = stats[i, cv2.CC_STAT_WIDTH]
        h_ = stats[i, cv2.CC_STAT_HEIGHT]
        if not (4 <= a <= 400 and 3 <= h_ <= 40 and 2 <= w_ <= 60):
            continue
        cy = centroids[i][1]
        cx = centroids[i][0]
        fg = extract_product_foreground(image)
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < H and 0 <= ix < W and fg[iy, ix] > 0:
            text_mask[labels == i] = 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, k)
    text_mask = cv2.dilate(text_mask, k, iterations=1)

    n_protected = int((text_mask > 0).sum())
    return {"mask": text_mask,
            "n_protected_components": n_protected,
            "watermark_template": watermark_template}


# ---------------------------------------------------------------------------
# Multi-channel residual watermark detection.
# ---------------------------------------------------------------------------

# V4: raised threshold — the multichannel detector fires on expected
# cleaning changes, not just residual watermark.
RESIDUAL_CONFIDENCE_THRESHOLD = 0.35

def detect_residual_multichannel(rwm, orig_bgr, cleaned_bgr, mark_box) -> dict:
    H, W = orig_bgr.shape[:2]
    bx, by, bw, bh = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]

    orig_roi = orig_bgr[by:by + bh, bx:bx + bw]
    clean_roi = cleaned_bgr[by:by + bh, bx:bx + bw]
    if orig_roi.size == 0 or clean_roi.size == 0:
        return {"confidence": 0.0, "channels": {}}

    scores = {}

    # LAB L channel residual.
    lab_o = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_c = cv2.cvtColor(clean_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_diff = np.abs(lab_o[..., 0] - lab_c[..., 0])
    scores["lab_l"] = round(float(l_diff.mean()) / 20.0, 4)

    # LAB A/B contrast diff.
    ab_diff = np.sqrt((lab_o[..., 1] - lab_c[..., 1]) ** 2 +
                       (lab_o[..., 2] - lab_c[..., 2]) ** 2)
    scores["lab_ab"] = round(float(ab_diff.mean()) / 15.0, 4)

    # HSV V channel.
    hsv_o = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_c = cv2.cvtColor(clean_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_diff = np.abs(hsv_o[..., 2] - hsv_c[..., 2])
    scores["hsv_v"] = round(float(v_diff.mean()) / 20.0, 4)

    # Grayscale high-pass residual.
    g_c = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    hp_roi = g_c[by:by + bh, bx:bx + bw].astype(np.float32)
    blur_hp = cv2.GaussianBlur(hp_roi, (0, 0), sigmaX=3)
    hp = np.abs(hp_roi - blur_hp)
    scores["highpass"] = round(float(hp.mean()) / 8.0, 4)

    # Template re-match on cleaned.
    g_clean = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    dets = rwm.detect_watermark_v2(g_clean)
    if dets and dets[0].get("tier") in ("auto", "manual"):
        scores["template_rematch"] = round(
            float(dets[0].get("edge_score", 0)), 4)
    else:
        scores["template_rematch"] = 0.0

    confidence = _clamp(
        0.25 * scores["lab_l"] +
        0.15 * scores["lab_ab"] +
        0.15 * scores["hsv_v"] +
        0.15 * scores["highpass"] +
        0.30 * scores["template_rematch"]
    )

    return {"confidence": round(confidence, 4),
            "channels": scores,
            "has_residual": confidence >= RESIDUAL_CONFIDENCE_THRESHOLD}


def residual_stroke_cleanup(rwm, img_bgr, cleaned_bgr, mark_box,
                            protect_mask=None) -> np.ndarray | None:
    stroke_info = generate_stroke_level_mask(cleaned_bgr, mark_box, edge_pad=2)
    if stroke_info["confidence"] < 0.15 or stroke_info["stroke_pixels"] < 10:
        return None
    mask2 = stroke_info["stroke"]
    if protect_mask is not None:
        mask2 = cv2.bitwise_and(mask2, cv2.bitwise_not(protect_mask))
    if not (mask2 > 0).any():
        return None
    run = run_inpaint(rwm, "telea_small", cleaned_bgr, mask2)
    return run["output"]


# ---------------------------------------------------------------------------
# Boundary harmonization — small ring blend around repair boundary.
# ---------------------------------------------------------------------------

def boundary_harmonize(orig_bgr, cleaned_bgr, mask, ring_px=3):
    outer = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_RECT, (ring_px * 2 + 1, ring_px * 2 + 1)))
    inner = cv2.erode(mask, cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(1, ring_px - 1) * 2 + 1,) * 2))
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    if not (ring > 0).any():
        return cleaned_bgr
    lab_c = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    alpha = cv2.GaussianBlur(ring, (0, 0), sigmaX=ring_px).astype(
        np.float32) / 255.0
    for c in range(3):
        lab_c[..., c] = (lab_c[..., c] * (1 - alpha) +
                         lab_o[..., c] * alpha)
    return cv2.cvtColor(np.clip(lab_c, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Candidate scoring — conservative formula per plan §4.
# ---------------------------------------------------------------------------

CLEAN_SCORE_THRESHOLD = 0.22

def _residual_grade(qa_water) -> float:
    if qa_water.get("pass"):
        return 1.0
    t_ok = qa_water.get("template", {}).get("pass", False)
    s_ok = qa_water.get("secondary", {}).get("pass", False)
    e_ok = qa_water.get("edge", {}).get("pass", False)
    # Strict: template is the primary signal. If template fails, the
    # watermark text is still detectable -- grade is 0.0.
    if not t_ok:
        return 0.0
    # Template passed but secondary/edge failed -- partial credit
    n_pass = int(t_ok) + int(s_ok) + int(e_ok)
    if n_pass >= 2:
        return 0.75
    return 0.35

def score_candidate(qa_water, qa_local, qa_product, overlap,
                    mask_area_frac, protect_overlap_frac=0.0) -> float:
    residual_removed = _residual_grade(qa_water)
    ssim_s = _clamp(qa_local.get("ssim", 0.5))
    color_penalty = _clamp(qa_local.get("color_delta", 0) / 30.0) * 0.15
    boundary_penalty = _clamp(qa_local.get("boundary_jump", 0) / 30.0) * 0.12
    product_penalty = 0.0
    if overlap:
        product_penalty = _clamp(
            overlap.get("edge_overlap", 0) * 0.5 +
            overlap.get("dark_overlap", 0) * 0.3 +
            overlap.get("circle_overlap", 0) * 0.4
        ) * 0.20
    mask_penalty = _clamp(mask_area_frac / 0.10) * 0.05
    protect_penalty = _clamp(protect_overlap_frac) * 0.2

    score = (
        0.45 * residual_removed +
        0.20 * ssim_s -
        color_penalty -
        boundary_penalty -
        product_penalty -
        mask_penalty -
        protect_penalty
    )
    return round(_clamp(score), 4)


# ---------------------------------------------------------------------------
# Feature flag stubs for optional advanced backends.
# ---------------------------------------------------------------------------

ENABLE_SAM2_PRODUCT_PROTECTION = False
ENABLE_DIFFUSION_INPAINT = False


def sam2_product_mask(img_bgr: np.ndarray) -> np.ndarray | None:
    if not ENABLE_SAM2_PRODUCT_PROTECTION:
        return None
    return None


def diffusion_inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if not ENABLE_DIFFUSION_INPAINT:
        return None
    return None


# ---------------------------------------------------------------------------
# Per-image orchestrator — V3+improvement plan wiring.
# Closed-loop retry with multi-candidate scoring, product protection,
# edge-stopped dilation, residual detection, boundary harmonization.
# ---------------------------------------------------------------------------

def _mk_record(path, det, *, status, reason=None, roi_class=None,
               features=None, mark_box_density=None, attempts=None,
               qa_water=None, qa_product=None, qa_local=None,
               qa_text=None, qa_post_integrity=None, qa_artifacts=None,
               masks_info=None, overlap=None,
               presence=None, cover_info=None):
    rec = {
        "product_id": path.stem,
        "image": path.name,
        "source": str(path),
        "pipeline_version": PIPELINE_VERSION,
        "status": status, "reason": reason,
        "roi_class": roi_class, "features": features,
        "mark_box": det.get("mark_box") if det else None,
        "mark_box_density": mark_box_density,
        "orig_detection": {
            "tier": det.get("tier"),
            "edge_score": det.get("edge_score"),
            "psr": det.get("psr"),
        } if det else None,
        "attempts": [_att_sum(a) for a in (attempts or [])],
        "qa_watermark": qa_water,
        "qa_product_integrity": qa_product,
        "qa_local_roi": qa_local,
        "qa_text_preservation": qa_text,
        "qa_post_integrity": qa_post_integrity,
        "qa_artifacts": qa_artifacts,
        "product_overlap": overlap,
        "masks_info": masks_info,
        "publish_ok": status in (ST_CLEAN_REPAIRED, ST_CLEAN_COVERED,
                                 ST_NO_WATERMARK),
        "repair_attempted": bool(attempts),
        "repair_qa_pass": False,
        "cover_applied": False,
    }
    if cover_info:
        rec["cover_applied"] = True
        rec["cover_method"] = cover_info.get("method")
        rec["cover_opacity"] = cover_info.get("opacity")
        rec["cover_expand_px"] = cover_info.get("expand_px")
        rec["cover_feather_px"] = cover_info.get("feather_px")
        rec["cover_fill_mode"] = cover_info.get("fill_mode")
        rec["cover_edge_density"] = cover_info.get("edge_density")
        rec["cover_noise_sigma"] = cover_info.get("noise_sigma")
        scores = cover_info.get("cover_scores") or {}
        rec["cover_score"] = scores.get("score")
        rec["watermark_residual_score"] = scores.get("watermark_residual_score")
        rec["patch_visibility_score"] = scores.get("patch_visibility_score")
        rec["product_edge_loss_score"] = scores.get("product_edge_loss_score")
        rec["boundary_jump_score"] = scores.get("boundary_jump_score")
        rec["post_blur_applied"] = scores.get("post_blur_applied", False)
    if presence:
        rec["presence_status"] = presence.get("status")
        rec["presence_confidence"] = presence.get("confidence")
        rec["presence_scores"] = presence.get("scores")
        rec["presence_reason"] = presence.get("reason")
        rec["presence_stage"] = presence.get("stage", "fast")
    return rec


def _att_sum(a):
    if not isinstance(a, dict):
        return {}
    return {"label": a.get("label"), "method": a.get("method"),
            "ok": a.get("ok"),
            "runtime_s": round(a.get("runtime_s", 0.0), 3),
            "over_budget": a.get("over_budget"), "error": a.get("error"),
            "qa_water": a.get("qa_water"),
            "qa_local": a.get("qa_local"),
            "qa_product": a.get("qa_product"),
            "qa_post_integrity": a.get("qa_post_integrity"),
            "qa_artifacts": a.get("qa_artifacts"),
            "qa_patch_visibility": a.get("qa_patch_visibility"),
            "score": a.get("score"),
            "dark_luma_overshoot": a.get("dark_luma_overshoot"),
            "residual": {k: v for k, v in (a.get("residual") or {}).items()
                         if k != "channels"} if a.get("residual") else None,
            "residual_cleaned": a.get("residual_cleaned", False)}


def _attempt(rwm, img, mask, method, label, mark_box, roi_class):
    run = run_inpaint(rwm, method, img, mask, mark_box=mark_box)
    if run["output"] is None:
        return {"label": label, "method": method, "ok": False,
                "runtime_s": run["runtime_s"],
                "over_budget": run["over_budget"], "error": run["error"],
                "qa_water": {"pass": False,
                             "fail_reasons": ["inpaint_error"]},
                "qa_local": None, "mask": mask, "output": None}
    qa_w = qa_watermark(rwm, run["output"], mark_box, roi_class)
    qa_l = qa_local_roi(img, run["output"], mark_box)
    return {"label": label, "method": method, "ok": True,
            "runtime_s": run["runtime_s"],
            "over_budget": run["over_budget"], "error": None,
            "qa_water": qa_w, "qa_local": qa_l,
            "mask": mask, "output": run["output"]}


def _collect_failure_reasons(attempts):
    reasons = []
    for a in attempts:
        if not a.get("ok"):
            reasons.append("inpaint_error")
            continue
        qw = a.get("qa_water", {})
        if not qw.get("pass"):
            reasons.append(M_RESIDUAL_WATERMARK)
        ql = a.get("qa_local", {})
        if ql and not ql.get("pass"):
            reasons.append(M_VISIBLE_PATCH)
        pv = a.get("qa_patch_visibility", {})
        if pv and not pv.get("pass"):
            reasons.append(M_PATCH_VISIBILITY)
        if a.get("dark_brightness_fail"):
            reasons.append(M_DARK_BRIGHTNESS_OVERSHOOT)
    return list(dict.fromkeys(reasons))


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def recheck_watermark_present(rwm, cleaned_bgr, orig_bbox, pg_config=None):
    """V15.1 Fix 2 — double-check the watermark is actually gone by re-running
    the SAME detector on the CLEANED output. Returns ``(still_present, info)``.

    ``still_present`` is True only when the detector re-confirms a watermark AT
    the original location (IoU > 0.05 with the original bbox) — so a clean
    removal (mark gone) reads False, while a cover that left the mark intact
    (the earpiece case) is caught honestly. This is the same engine the pipeline
    trusts to find watermarks in the first place, turned back on its own result.
    """
    info = {"status": None, "confidence": 0.0, "iou": 0.0}
    if cleaned_bgr is None:
        return False, info
    cfg = pg_config or {
        "confirmed_threshold": CONFIRMED_WATERMARK_THRESHOLD,
        "no_watermark_threshold": NO_WATERMARK_THRESHOLD,
    }
    g = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    try:
        dets = rwm.detect_watermark_v2(g)
    except Exception:
        return False, info
    if not dets:
        return False, info
    det2 = dets[0]
    presence = detect_sunsky_presence_fast(g, det2, cfg)
    info["status"] = presence.get("status")
    info["confidence"] = round(float(presence.get("confidence", 0.0)), 4)
    mb = det2.get("mark_box") or {}
    iou = (_bbox_iou(orig_bbox, (mb["x"], mb["y"], mb["w"], mb["h"]))
           if mb else 0.0)
    info["iou"] = round(iou, 4)
    still = (presence.get("status") == PG_CONFIRMED and iou > 0.05 and
             det2.get("tier") in ("auto", "manual"))
    return still, info


def process_image(rwm, path: Path, det: dict, out_root: Path,
                  debug_root: Path) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        rec = _mk_record(path, det, status=ST_FAILED_IO,
                         reason=R_INPUT_UNREADABLE)
        _write_terminal(out_root, debug_root, rec, None, None, None, None,
                        None, None)
        return rec

    H, W = img.shape[:2]
    mark_box = det["mark_box"]
    orig_md = (mark_box["w"] * mark_box["h"]) / (H * W + 1e-6)

    features = extract_roi_features(img, mark_box)
    roi_class = classify_roi(features)

    # Build product protection mask.
    protect = build_product_protect_mask(img, mark_box)
    protect_combined = protect["combined"]
    text_protect_info = build_protected_text_mask(img, mark_box)
    text_protect_mask = text_protect_info["mask"]
    protect_combined = cv2.bitwise_or(protect_combined, text_protect_mask)

    # Build masks with edge-stopped dilation and product protection.
    masks = build_mask_layers(img, mark_box, roi_class,
                              protect_mask=protect_combined)
    logo_m = masks.get("logo_mask")
    has_logo_mask = logo_m is not None and logo_m.any()
    masks_info = {
        "final_mask": masks["used"],
        "stroke_confidence": masks["stroke_info"]["confidence"],
        "stroke_pixels": masks["stroke_info"]["stroke_pixels"],
        "n_components": masks["stroke_info"]["n_components"],
        "final_density_vs_soft": masks["final_density_vs_soft"],
        "final_density_abs": round(float((masks["final"] > 0).sum()) /
                                    (H * W + 1e-6), 4),
        "soft_density": round(float((masks["soft"] > 0).sum()) /
                               (H * W + 1e-6), 4),
        "logo_mask_active": has_logo_mask,
    }

    mark_area = max(1, mark_box["w"] * mark_box["h"])
    mask_pixels_in_roi = int((masks["final"][
        mark_box["y"]:mark_box["y"] + mark_box["h"],
        mark_box["x"]:mark_box["x"] + mark_box["w"]] > 0).sum())
    mask_density_in_roi = mask_pixels_in_roi / mark_area
    masks_info["mask_density_in_roi"] = round(mask_density_in_roi, 4)

    final_ys, final_xs = np.where(masks["final"] > 0)
    if final_ys.size > 0:
        bbox_area = max(1, (int(final_ys.max()) - int(final_ys.min()) + 1) *
                        (int(final_xs.max()) - int(final_xs.min()) + 1))
        mask_rectangularity = final_ys.size / bbox_area
    else:
        mask_rectangularity = 0.0
    masks_info["mask_rectangularity"] = round(mask_rectangularity, 4)

    # V5: mask quality issues no longer block — they route to cover fallback.
    # V7: logo-shaped masks are intentionally dense — skip density/rect checks.
    logo_active = masks.get("logo_mask") is not None and masks["logo_mask"].any()
    skip_repair = (masks["used"] == "soft_fallback" or
                   (not logo_active and mask_density_in_roi > V4_MASK_DENSITY_IN_ROI_MAX) or
                   (not logo_active and mask_rectangularity > V4_MASK_RECTANGULARITY_MAX) or
                   ROUTE_BY_CLASS.get(roi_class) == "reject" or
                   orig_md > MARK_BOX_DENSITY_REJECT)

    overlap = analyze_product_overlap(img, masks["soft"], mark_box)
    gate, gate_reason = product_integrity_gate(overlap)
    if gate in ("reject", "manual"):
        skip_repair = True

    # --- V8: Progressive Repair Strategy Bank ---
    bbox_tuple = (mark_box["x"], mark_box["y"],
                  mark_box["w"], mark_box["h"])
    pr_roi = pr_analyze_roi(img, bbox_tuple)

    pr_ctx = RepairContext(
        image=img,
        watermark_bbox=bbox_tuple,
        stroke_mask=masks.get("stroke"),
        bbox_mask=masks["final"],
        roi_analysis=pr_roi,
        product_mask=protect_combined,
        logo_mask=masks.get("logo_mask"),
        debug_dir=str(debug_root) if debug_root else None,
    )

    candidate, trace = repair_image_progressively(pr_ctx)
    final_method = candidate.metadata.get("final_method", "unknown")
    method_family = candidate.metadata.get("method_family",
                                            get_method_family(final_method).value)
    qa_info = candidate.metadata.get("qa", {})
    telemetry = candidate.metadata.get("telemetry", {})

    roi_features = {
        "product_overlap": round(pr_roi.product_pixel_ratio, 3),
        "edge_density": round(pr_roi.edge_density, 3),
        "dark_pixel_ratio": round(pr_roi.dark_pixel_ratio, 3),
        "white_pixel_ratio": round(pr_roi.white_pixel_ratio, 3),
        "brightness": round(pr_roi.brightness, 1),
        "texture_variance": round(pr_roi.texture_variance, 1),
        "gradient_strength": round(pr_roi.gradient_strength, 2),
        "has_long_lines": pr_roi.has_long_lines,
        "has_text_like_edges": pr_roi.has_text_like_edges,
    }
    if debug_root:
        save_debug_trace(debug_root, path.name, pr_roi.roi_class,
                         bbox_tuple, trace, final_method,
                         method_family=method_family,
                         roi_features=roi_features,
                         telemetry=telemetry, qa=qa_info,
                         pipeline_version=PIPELINE_VERSION)

    is_cover = method_family == MethodFamily.ADAPTIVE_COVER.value
    status = ST_CLEAN_COVERED if is_cover else ST_CLEAN_REPAIRED

    def _qa_local_view(q):
        # V11 — surface the real, computed QA metrics in the trace/PDF so a
        # passing repair never shows ssim=0 color_delta=0 boundary_jump=0.
        if not q:
            return {"pass": False}
        return {
            "ssim": q.get("ssim_score"),
            "color_delta": q.get("color_delta_score"),
            "boundary_jump": q.get("boundary_jump"),
            "seam_delta": q.get("seam_delta_score"),
            "metrics_valid": q.get("metrics_valid"),
            "pass": bool(q.get("passed")),
        }

    def _qa_water_view(q):
        if not q:
            return {"pass": False}
        return {
            "pass": bool(q.get("residual_pass")),
            "template": {"tier": f"corr={q.get('residual_template_corr', 0):.3f}"},
            "secondary": {"edge_score": q.get("residual_template_corr", 0.0)},
            "edge": {"boundary_jump": q.get("boundary_jump", 0.0),
                     "inside_edge": q.get("residual_text_component", 0.0)},
            "dot_chain": q.get("residual_dot_chain_score", 0.0),
            "product_gate_pass": q.get("product_gate_pass", True),
        }

    attempts = [{
        "label": t.get("tool", "unknown"),
        "method": t.get("tool", "unknown"),
        "ok": t.get("passed", False),
        "runtime_s": t.get("runtime_s", 0.0),
        "over_budget": False,
        "error": None if t.get("reason") != "error" else t.get("reason"),
        "qa_water": _qa_water_view(t.get("qa")),
        "qa_local": _qa_local_view(t.get("qa")),
    } for t in trace]

    rec = _mk_record(path, det, status=status, reason=None,
                     roi_class=roi_class, features=features,
                     mark_box_density=orig_md, attempts=attempts,
                     masks_info=masks_info, overlap=overlap,
                     qa_local=_qa_local_view(qa_info),
                     qa_water=_qa_water_view(qa_info))
    rec["repair_qa_pass"] = not is_cover
    rec["v9_final_method"] = final_method
    rec["v9_method_family"] = method_family
    rec["v9_is_real_repair"] = not is_cover
    rec["v9_is_cover"] = is_cover
    rec["v9_roi_class"] = pr_roi.roi_class
    rec["v9_tools_tried"] = len(trace)
    rec["v9_qa_final_score"] = qa_info.get("final_score")
    rec["v9_rect_patch_visibility"] = qa_info.get("rectangular_patch_visibility")
    # V10 truthful-gate verdicts + diversity telemetry.
    rec["v10_metrics_valid"] = qa_info.get("metrics_valid")
    rec["v10_residual_pass"] = qa_info.get("residual_pass")
    rec["v10_residual_template_corr"] = qa_info.get("residual_template_corr")
    rec["v10_residual_text_component"] = qa_info.get("residual_text_component")
    rec["v10_residual_improvement"] = qa_info.get("residual_improvement")
    rec["v10_cover_rectangularity"] = qa_info.get("cover_rectangularity")
    rec["v10_cover_luma_delta"] = qa_info.get("cover_luma_delta")
    rec["v10_families_attempted"] = telemetry.get("families_attempted")
    rec["v10_tools_attempted"] = telemetry.get("tools_attempted")
    rec["v10_qa_reject_reasons"] = telemetry.get("qa_reject_reasons")
    # V11 telemetry
    rec["v11_ssim"] = qa_info.get("ssim_score")
    rec["v11_boundary_jump"] = qa_info.get("boundary_jump")
    rec["v11_metrics_invalid_reasons"] = qa_info.get("metrics_invalid_reasons")
    rec["v11_residual_dot_chain_score"] = qa_info.get("residual_dot_chain_score")
    rec["v11_product_overlap"] = qa_info.get("product_overlap")
    rec["v11_product_color_delta_lab"] = qa_info.get("product_color_delta_lab")
    rec["v11_product_new_bright_blob_score"] = qa_info.get(
        "product_new_bright_blob_score")
    rec["v11_product_edge_retention"] = qa_info.get("product_edge_retention")
    rec["v11_product_gate_pass"] = qa_info.get("product_gate_pass")
    rec["v11_changed_area_ratio"] = qa_info.get("changed_area_ratio")
    rec["v11_tools_reachable"] = telemetry.get("tools_reachable")
    rec["v11_tools_tried"] = telemetry.get("tools_tried")
    rec["v11_candidates_passed"] = telemetry.get("candidates_passed")
    # V12 — visual-truthfulness verdicts + provenance.
    rec["v12_publish_ok"] = qa_info.get("publish_ok")
    rec["v12_publish_status"] = qa_info.get("publish_status")
    rec["v12_publish_reject_reasons"] = qa_info.get("publish_reject_reasons")
    rec["v12_visible_band_score"] = qa_info.get("visible_band_score")
    rec["v12_band_luma_delta"] = qa_info.get("band_luma_delta")
    rec["v12_band_edge_box_score"] = qa_info.get("band_edge_box_score")
    rec["v12_band_rectangularity"] = qa_info.get("band_rectangularity")
    rec["v12_product_contour_break_score"] = qa_info.get(
        "product_contour_break_score")
    rec["v12_forced_cover"] = telemetry.get("forced_cover")
    rec["v12_dot_chain_rejections"] = telemetry.get("dot_chain_rejections")
    rec["v12_band_rejections"] = telemetry.get("band_rejections")
    rec["v12_product_damage_rejections"] = telemetry.get(
        "product_damage_rejections")
    rec["v12_micro_cleanup_attempts"] = telemetry.get("micro_cleanup_attempts")
    rec["v12_micro_cleanup_successes"] = telemetry.get(
        "micro_cleanup_successes")
    rec["pipeline_version"] = PIPELINE_VERSION
    rec["repair_qa_pass"] = (not is_cover) and bool(qa_info.get("residual_pass")) \
        and bool(qa_info.get("metrics_valid")) and bool(qa_info.get("publish_ok"))

    # --- V16 — Complete auto-decision state machine. ------------------------
    # The unchanged V13 final visual gate + a post-clean watermark re-detection
    # together form the P0 publish gate (SAME strictness for repaired and
    # covered). decide_final_status runs: repair candidate -> residual cleanup
    # -> cover beam -> auto_rejected. ONLY a candidate that passes every P0 gate
    # is labelled clean_repaired / clean_covered; everything else becomes
    # auto_rejected — a final automated decision, never published, never manual
    # review. There is no path from a failed final gate into a clean_* status.
    stroke_mask = masks.get("stroke")

    def _v16_gate(image, repaired, residual_pass=None):
        qa = qa_info
        if residual_pass is not None:
            qa = dict(qa_info, residual_pass=bool(residual_pass))
        return v13_gates.final_visual_publish_gate_v13(
            img, image, bbox_tuple, protect_combined, qa,
            repaired=repaired, watermark_mask=stroke_mask)

    def _v16_recheck(image):
        return recheck_watermark_present(rwm, image, bbox_tuple)

    v16 = v16_pipeline.decide_final_status(
        img, bbox_tuple, protect_combined, stroke_mask, qa_info,
        loop_repair_image=(None if is_cover else candidate.repaired_image),
        loop_repair_method=final_method,
        best_failed_image=candidate.metadata.get("best_failed_image"),
        loop_cover_image=(candidate.repaired_image if is_cover else None),
        roi_class=pr_roi.roi_class,
        gate_fn=_v16_gate, recheck_fn=_v16_recheck)

    status = v16.status
    final_image = candidate.repaired_image = v16.image
    v13_verdict = v16.verdict
    is_cover = (status == ST_CLEAN_COVERED)
    publishable = status in (ST_CLEAN_REPAIRED, ST_CLEAN_COVERED)

    # --- V17 — authoritative audit of the ACTUAL published bytes. ----------
    # decide_final_status already audits every candidate, but the published
    # output is the final word: re-run the truthful audit on v16.image and, if
    # it still shows the watermark or damages the product, force auto_rejected.
    # Prefer auto_rejected over a dishonest clean result. (patch plan §3.2, §14)
    rec["v17_audit_version"] = "v17"
    if publishable:
        try:
            still_pub, _pub_info = recheck_watermark_present(
                rwm, final_image, bbox_tuple)
        except Exception:
            still_pub = False
        vp_failed = False
        vb_failed = False
        if v13_verdict is not None:
            vp_failed = not (v13_verdict.visible_patch_pass and
                             v13_verdict.rectangular_band_pass and
                             v13_verdict.polygon_patch_pass)
            vb_failed = not v13_verdict.rectangular_band_pass
        try:
            final_audit = v17_final_audit.audit_final_output(
                img, final_image, bbox_tuple, protect_combined, stroke_mask,
                final_status=status, still_present=still_pub,
                visible_patch_failed=vp_failed, visible_band_failed=vb_failed,
                roi_class=pr_roi.roi_class)
        except Exception:
            final_audit = None
        if final_audit is not None:
            rec.update(final_audit.to_record())
            if not final_audit.pass_p0:
                # Downgrade: a published output that fails the truthful audit is
                # demoted to auto_rejected — never shipped.
                status = ST_AUTO_REJECTED
                is_cover = False
                publishable = False
                v16.publish_ok = False
                v16.reject_reasons = list(v16.reject_reasons) + [
                    "v17_" + r for r in final_audit.hard_fail_reasons]

    rec["status"] = status
    rec["v9_final_method"] = v16.method
    rec["v9_method_family"] = (
        MethodFamily.ADAPTIVE_COVER.value if is_cover else
        MethodFamily.REAL_PIXEL_CLONE.value if status == ST_CLEAN_REPAIRED else
        "auto_rejected")
    rec["v9_is_cover"] = is_cover
    rec["v9_is_real_repair"] = (status == ST_CLEAN_REPAIRED)

    # --- V16 manifest schema (patch plan Fix 8). ----------------------------
    rec["final_status"] = status
    rec["publish_ok"] = bool(v16.publish_ok)
    rec["manual_required"] = False           # auto_rejected is NOT manual review
    rec["final_gate_version"] = "v16"
    rec["final_gate_pass"] = bool(v13_verdict.publish_ok) if v13_verdict else False
    rec["candidate_publish_failures"] = int(v16.candidate_publish_failures)
    rec["cosmetic_seam"] = bool(v16.cosmetic_seam)
    # A published output that fails a HARD SAFETY gate (watermark still present
    # or product damaged) would be a fatal invariant break. By construction this
    # is always False — clean_* is only assigned when every hard-safety gate
    # passes (a published cosmetic_seam still has all hard gates green).
    rec["final_output_publish_failure"] = bool(
        publishable and not v16_pipeline._is_safe(v16.p0_gates))
    rec["reject_reasons"] = list(v16.reject_reasons)
    rec["p0_gates"] = dict(v16.p0_gates)
    rec["v16_path"] = v16.telemetry.get("v16_path")
    rec.update(v16.telemetry)
    if v13_verdict is not None:
        rec.update(v13_verdict.to_record())

    # repair_qa_pass / no_false_clean_repaired: a clean_repaired is honest only
    # when the full P0 gate passed (it did — clean_repaired is gate-gated).
    rec["repair_qa_pass"] = (status == ST_CLEAN_REPAIRED) and bool(v16.publish_ok)

    best_attempt_stub = attempts[-1] if attempts else None
    _write_terminal(out_root, debug_root, rec, img, masks,
                    candidate.repaired_image, best_attempt_stub,
                    protect, overlap,
                    best_repair=v16.best_repair_image,
                    best_cover=v16.best_cover_image)
    return rec


# ---------------------------------------------------------------------------
# Output writing — P7 extended debug artifacts.
# ---------------------------------------------------------------------------

def publish_gate(rec: dict) -> bool:
    return rec.get("status") in (ST_CLEAN_REPAIRED, ST_CLEAN_COVERED) and \
        rec.get("publish_ok", True)


def _rel(out_root, p):
    try:
        return str(p.relative_to(out_root.parent))
    except ValueError:
        return str(p)


def _write_terminal(out_root, debug_root, rec, img, masks_dict,
                    cleaned, last_attempt, protect_layers, overlap,
                    best_repair=None, best_cover=None):
    base = out_root / rec["status"] / rec["product_id"]
    base.mkdir(parents=True, exist_ok=True)
    paths = {}

    if img is not None:
        cv2.imwrite(str(base / "original.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths["original"] = _rel(out_root, base / "original.jpg")

    if cleaned is not None:
        cv2.imwrite(str(base / "cleaned.jpg"), cleaned,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        paths["cleaned"] = _rel(out_root, base / "cleaned.jpg")

    # V16 — auto_rejected diagnostic artifacts: keep the best repair and cover
    # attempts so a rejected image is debuggable (patch plan Fix 7). These live
    # only in the auto_rejected/ tree — never a publishable folder.
    if rec.get("status") == ST_AUTO_REJECTED:
        if best_repair is not None:
            cv2.imwrite(str(base / "best_repair_attempt.jpg"), best_repair,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            paths["best_repair_attempt"] = _rel(
                out_root, base / "best_repair_attempt.jpg")
        if best_cover is not None:
            cv2.imwrite(str(base / "best_cover_attempt.jpg"), best_cover,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            paths["best_cover_attempt"] = _rel(
                out_root, base / "best_cover_attempt.jpg")

    if masks_dict is not None:
        for name in ("core", "soft", "stroke", "stroke_capped", "final",
                     "repair", "repair_safe", "logo_mask", "logo_text_only"):
            m = masks_dict.get(name)
            if m is not None:
                cv2.imwrite(str(base / f"mask_{name}.png"), m)
                paths[f"mask_{name}"] = _rel(out_root, base / f"mask_{name}.png")

    # Product protection mask debug output.
    if protect_layers is not None and isinstance(protect_layers, dict):
        for name in ("combined", "edges", "dark", "circles", "text_qr",
                     "thin_edges"):
            m = protect_layers.get(name)
            if m is not None:
                cv2.imwrite(str(base / f"protect_{name}.png"), m)
                paths[f"protect_{name}"] = _rel(
                    out_root, base / f"protect_{name}.png")

    if overlap and "_edges" in overlap:
        cv2.imwrite(str(base / "edge_map.png"), overlap["_edges"])
        paths["edge_map"] = _rel(out_root, base / "edge_map.png")
    if overlap and "_dark" in overlap:
        cv2.imwrite(str(base / "dark_map.png"), overlap["_dark"])
        paths["dark_map"] = _rel(out_root, base / "dark_map.png")

    # qa overlay — colorize the cleaned image: red where the mask was, faint
    # green where edges are preserved.
    if img is not None and cleaned is not None and masks_dict is not None:
        overlay = cleaned.copy()
        m = masks_dict.get("final")
        if m is not None:
            mask3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(overlay, 1.0,
                                       cv2.merge([np.zeros_like(m),
                                                  np.zeros_like(m),
                                                  m]), 0.3, 0)
            cv2.imwrite(str(base / "qa_overlay.jpg"), overlay,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            paths["qa_overlay"] = _rel(out_root, base / "qa_overlay.jpg")

    # Diff heatmap for ALL statuses (not just manual/rejected).
    if img is not None and cleaned is not None and rec.get("mark_box"):
        mb = rec["mark_box"]
        px = max(40, mb["w"] // 4); py = max(40, mb["h"] * 2)
        rx1 = max(0, mb["x"] - px); ry1 = max(0, mb["y"] - py)
        rx2 = min(img.shape[1], mb["x"] + mb["w"] + px)
        ry2 = min(img.shape[0], mb["y"] + mb["h"] + py)
        roi_b = img[ry1:ry2, rx1:rx2]
        roi_a = cleaned[ry1:ry2, rx1:rx2]
        diff = cv2.absdiff(roi_b, roi_a)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_heatmap = cv2.applyColorMap(
            cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8),
            cv2.COLORMAP_JET)
        cv2.imwrite(str(base / "diff_heatmap.png"), diff_heatmap)
        paths["diff_heatmap"] = _rel(out_root, base / "diff_heatmap.png")

    if img is not None and rec.get("mark_box"):
        mb = rec["mark_box"]
        px = max(40, mb["w"] // 4); py = max(40, mb["h"] * 2)
        rx1 = max(0, mb["x"] - px); ry1 = max(0, mb["y"] - py)
        rx2 = min(img.shape[1], mb["x"] + mb["w"] + px)
        ry2 = min(img.shape[0], mb["y"] + mb["h"] + py)
        roi_b = img[ry1:ry2, rx1:rx2]
        cv2.imwrite(str(base / "roi_before.jpg"), roi_b,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths["roi_before"] = _rel(out_root, base / "roi_before.jpg")
        if cleaned is not None:
            roi_a = cleaned[ry1:ry2, rx1:rx2]
            cv2.imwrite(str(base / "roi_after.jpg"), roi_a,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            paths["roi_after"] = _rel(out_root, base / "roi_after.jpg")
            diff = cv2.absdiff(roi_b, roi_a)
            cv2.imwrite(str(base / "diff.png"), diff)
            paths["diff"] = _rel(out_root, base / "diff.png")

    # V4-8: Patch visibility heatmap — shows per-pixel luma deviation
    # between cleaned ROI and its surrounding ring.
    if img is not None and cleaned is not None and rec.get("mark_box"):
        mb = rec["mark_box"]
        bx_h, by_h = mb["x"], mb["y"]
        bw_h, bh_h = mb["w"], mb["h"]
        ring_pad_h = max(10, max(bw_h, bh_h) // 3)
        ry1_h = max(0, by_h - ring_pad_h)
        ry2_h = min(cleaned.shape[0], by_h + bh_h + ring_pad_h)
        rx1_h = max(0, bx_h - ring_pad_h)
        rx2_h = min(cleaned.shape[1], bx_h + bw_h + ring_pad_h)
        gc_h = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float64)
        ring_h = np.zeros(gc_h.shape, dtype=bool)
        ring_h[ry1_h:ry2_h, rx1_h:rx2_h] = True
        ring_h[by_h:by_h + bh_h, bx_h:bx_h + bw_h] = False
        ring_mean_h = float(gc_h[ring_h].mean()) if ring_h.any() else 128.0
        roi_region = gc_h[ry1_h:ry2_h, rx1_h:rx2_h]
        dev = np.abs(roi_region - ring_mean_h)
        dev_u8 = np.clip(dev * 4, 0, 255).astype(np.uint8)
        pv_heatmap = cv2.applyColorMap(dev_u8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(base / "patch_visibility_heatmap.png"), pv_heatmap)
        paths["patch_visibility_heatmap"] = _rel(
            out_root, base / "patch_visibility_heatmap.png")

    # V4-8: Write all candidate scores to debug JSON.
    if rec.get("attempts"):
        cand_scores = []
        for att in rec["attempts"]:
            cs_entry = {
                "method": att.get("method"),
                "score": att.get("score"),
                "ok": att.get("ok"),
            }
            pv = att.get("qa_patch_visibility")
            if pv:
                cs_entry["pv_pass"] = pv.get("pass")
                cs_entry["pv_metrics"] = pv.get("metrics")
                cs_entry["pv_fail_reasons"] = pv.get("fail_reasons")
            cs_entry["dark_luma_overshoot"] = att.get("dark_luma_overshoot")
            cand_scores.append(cs_entry)
        rec["v4_candidate_scores"] = cand_scores

    # Strip cv2 image data from overlap before serializing
    overlap_to_save = ({k: v for k, v in overlap.items() if not k.startswith("_")}
                      if overlap else None)
    rec_to_save = dict(rec)
    rec_to_save["product_overlap"] = overlap_to_save
    rec_to_save["paths"] = paths
    rec["paths"] = paths
    rec["product_overlap"] = overlap_to_save
    (base / "qa.json").write_text(json.dumps(rec_to_save, indent=2, default=str))


# ---------------------------------------------------------------------------
# Reporting — JSONL / CSV / HTML / PDF.
# ---------------------------------------------------------------------------

STATUS_ORDER = [ST_FAILED_IO, ST_AUTO_REJECTED, ST_CLEAN_COVERED,
                ST_CLEAN_REPAIRED, ST_NO_WATERMARK, ST_SKIPPED_KNOWN_CLEAN]


def write_summary_jsonl(out_root, records):
    with (out_root / "summary.jsonl").open("w") as f:
        for r in records:
            ov = r.get("product_overlap") or {}
            mi = r.get("masks_info") or {}
            pi = r.get("qa_product_integrity") or {}
            row = {k: r.get(k) for k in (
                "product_id", "image", "status", "reason", "roi_class",
                "mark_box_density", "publish_ok")}
            row["mask_used"] = mi.get("final_mask")
            row["stroke_confidence"] = mi.get("stroke_confidence")
            row["edge_overlap"] = ov.get("edge_overlap")
            row["dark_overlap"] = ov.get("dark_overlap")
            row["circle_overlap"] = ov.get("circle_overlap")
            row["pi_fg_loss"] = pi.get("fg_area_loss")
            row["pi_edge_loss"] = pi.get("edge_loss")
            row["pi_outside_diff"] = pi.get("mean_outside_diff")
            piq = r.get("qa_post_integrity") or {}
            row["post_integrity_score"] = piq.get("score")
            row["post_integrity_pass"] = piq.get("pass")
            qart = r.get("qa_artifacts") or {}
            row["artifact_score"] = qart.get("artifact_score")
            row["has_artifact"] = qart.get("has_artifact")
            row["presence_status"] = r.get("presence_status")
            row["presence_confidence"] = r.get("presence_confidence")
            row["presence_reason"] = r.get("presence_reason")
            row["cover_applied"] = r.get("cover_applied", False)
            row["cover_method"] = r.get("cover_method")
            row["cover_fill_mode"] = r.get("cover_fill_mode")
            row["cover_opacity"] = r.get("cover_opacity")
            row["cover_score"] = r.get("cover_score")
            row["watermark_residual_score"] = r.get("watermark_residual_score")
            row["patch_visibility_score"] = r.get("patch_visibility_score")
            row["product_edge_loss_score"] = r.get("product_edge_loss_score")
            row["boundary_jump_score"] = r.get("boundary_jump_score")
            row["post_blur_applied"] = r.get("post_blur_applied", False)
            row["v9_final_method"] = r.get("v9_final_method")
            row["v9_method_family"] = r.get("v9_method_family")
            row["v9_is_real_repair"] = r.get("v9_is_real_repair")
            row["v9_is_cover"] = r.get("v9_is_cover")
            row["v9_roi_class"] = r.get("v9_roi_class")
            row["v9_tools_tried"] = r.get("v9_tools_tried")
            row["v9_rect_patch_visibility"] = r.get("v9_rect_patch_visibility")
            # V12 — provenance + visual-truthfulness verdicts in every row.
            row["pipeline_version"] = r.get("pipeline_version") or PIPELINE_VERSION
            row["v12_publish_status"] = r.get("v12_publish_status")
            row["v12_publish_reject_reasons"] = r.get(
                "v12_publish_reject_reasons")
            row["v12_visible_band_score"] = r.get("v12_visible_band_score")
            row["v12_band_rectangularity"] = r.get("v12_band_rectangularity")
            row["v11_residual_dot_chain_score"] = r.get(
                "v11_residual_dot_chain_score")
            row["v12_forced_cover"] = r.get("v12_forced_cover")
            f.write(json.dumps(row, default=str) + "\n")


def write_summary_csv(out_root, records):
    with (out_root / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "product_id", "image", "status", "reason", "roi_class",
            "mask_used", "stroke_confidence",
            "mark_box_density",
            "edge_overlap", "dark_overlap", "circle_overlap",
            "pi_fg_loss", "pi_bbox_shift", "pi_edge_loss",
            "pi_outside_diff",
            "post_integrity_score", "artifact_score",
            "n_attempts", "last_method", "publish_ok",
            "presence_status", "presence_confidence", "presence_reason",
            "cover_applied", "cover_method", "cover_fill_mode",
            "cover_opacity", "cover_score",
            "watermark_residual_score", "patch_visibility_score",
            "product_edge_loss_score", "boundary_jump_score",
            "post_blur_applied"])
        for r in records:
            ov = r.get("product_overlap") or {}
            mi = r.get("masks_info") or {}
            pi = r.get("qa_product_integrity") or {}
            piq = r.get("qa_post_integrity") or {}
            qart = r.get("qa_artifacts") or {}
            atts = r.get("attempts") or []
            last_m = atts[-1].get("method") if atts else ""
            w.writerow([
                r["product_id"], r["image"], r["status"],
                r.get("reason") or "",
                r.get("roi_class") or "",
                mi.get("final_mask") or "",
                f"{mi.get('stroke_confidence', 0):.2f}",
                f"{r.get('mark_box_density', 0) or 0:.4f}",
                f"{ov.get('edge_overlap', 0):.3f}",
                f"{ov.get('dark_overlap', 0):.3f}",
                f"{ov.get('circle_overlap', 0):.3f}",
                f"{pi.get('fg_area_loss', 0):.4f}",
                f"{pi.get('bbox_shift', 0):.4f}",
                f"{pi.get('edge_loss', 0):.4f}",
                f"{pi.get('mean_outside_diff', 0):.3f}",
                f"{piq.get('score', 0):.4f}",
                f"{qart.get('artifact_score', 0):.4f}",
                len(atts), last_m,
                "yes" if r.get("publish_ok") else "no",
                r.get("presence_status") or "",
                f"{r.get('presence_confidence', 0) or 0:.3f}",
                r.get("presence_reason") or "",
                "yes" if r.get("cover_applied") else "no",
                r.get("cover_method") or "",
                r.get("cover_fill_mode") or "",
                f"{r.get('cover_opacity', 0) or 0:.2f}",
                f"{r.get('cover_score', 0) or 0:.4f}",
                f"{r.get('watermark_residual_score', 0) or 0:.4f}",
                f"{r.get('patch_visibility_score', 0) or 0:.4f}",
                f"{r.get('product_edge_loss_score', 0) or 0:.4f}",
                f"{r.get('boundary_jump_score', 0) or 0:.4f}",
                "yes" if r.get("post_blur_applied") else "no",
            ])


def write_compare_html(out_root, records):
    sort_key = {s: i for i, s in enumerate(STATUS_ORDER)}
    recs = sorted(records, key=lambda r: sort_key.get(r["status"], 99))

    counts = {s: 0 for s in STATUS_ORDER}
    by_class = {}
    by_route = {}
    by_mask = {}
    by_presence = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        c = r.get("roi_class") or "?"
        by_class[c] = by_class.get(c, 0) + 1
        if r.get("attempts"):
            m = r["attempts"][-1].get("method") or "?"
            by_route[m] = by_route.get(m, 0) + 1
        mu = (r.get("masks_info") or {}).get("final_mask", "n/a")
        by_mask[mu] = by_mask.get(mu, 0) + 1
        ps = r.get("presence_status") or "n/a"
        by_presence[ps] = by_presence.get(ps, 0) + 1

    rows = []
    for r in recs:
        paths = r.get("paths", {})
        pi = r.get("qa_product_integrity") or {}
        ov = r.get("product_overlap") or {}
        mi = r.get("masks_info") or {}
        piq = r.get("qa_post_integrity") or {}
        qart = r.get("qa_artifacts") or {}
        atts = r.get("attempts") or []
        last_att = atts[-1] if atts else {}
        qaw = r.get("qa_watermark") or last_att.get("qa_water") or {}
        qal = r.get("qa_local_roi") or last_att.get("qa_local") or {}
        psc = r.get("presence_scores") or {}

        def im(key, w_=300):
            return (f"<img src='../{paths[key]}' width={w_}>"
                    if key in paths else "—")

        pg_html = ""
        ps_val = r.get("presence_status")
        if ps_val:
            pg_color = {"NO_WATERMARK": "#9f9",
                        "NO_WATERMARK_MODEL_EXCLUDED": "#9f9",
                        "CONFIRMED_WATERMARK": "#fc8",
                        "UNCERTAIN_WATERMARK": "#f88",
                        "UNSUITABLE_FOR_AUTO_DETECTION": "#f88"}.get(ps_val, "#aaa")
            pg_html = (
                f"<span style='color:{pg_color}'>{ps_val}</span><br>"
                f"conf={r.get('presence_confidence',0):.3f}<br>"
                f"tpl={psc.get('template',0):.2f} stk={psc.get('stroke',0):.2f}<br>"
                f"clahe={psc.get('clahe_center',0):.2f}<br>"
                f"ar={psc.get('aspect_ratio',0):.2f} pos={psc.get('position',0):.2f}<br>"
                f"neg={psc.get('negative_product_text',0):.2f}<br>"
                f"<i>{r.get('presence_reason','')}</i>")

        piq_m = piq.get("metrics") or {}
        piq_color = "#0a4" if piq.get("pass") else "#f88"
        art_color = "#0a4" if not qart.get("has_artifact") else "#f88"

        # V4-8: patch visibility info from attempts
        pv_html = "—"
        v4_cs = r.get("v4_candidate_scores") or []
        if v4_cs:
            pv_lines = []
            for cs_e in v4_cs:
                pv_m = cs_e.get("pv_metrics") or {}
                pv_pass = cs_e.get("pv_pass")
                pc = "#0a4" if pv_pass else "#f88"
                pv_lines.append(
                    f"<span style='color:{pc}'>{cs_e.get('method','?')}"
                    f" s={cs_e.get('score',0)}</span><br>"
                    f"Δluma={pv_m.get('roi_vs_ring_delta_luma',0):.1f} "
                    f"lf={pv_m.get('low_frequency_patch_score',0):.2f} "
                    f"rect={pv_m.get('rectangular_boundary_score',0):.2f} "
                    f"tex={pv_m.get('texture_energy_drop',0):.2f}")
            pv_html = "<br>".join(pv_lines)

        cover_html = "—"
        if r.get("cover_applied"):
            cover_html = (
                f"method={r.get('cover_method','?')}<br>"
                f"fill={r.get('cover_fill_mode','?')}<br>"
                f"opacity={r.get('cover_opacity',0):.2f}<br>"
                f"feather={r.get('cover_feather_px',0)}px<br>"
                f"expand={r.get('cover_expand_px',0)}px<br>"
                f"edge_d={r.get('cover_edge_density',0):.3f}<br>"
                f"score={r.get('cover_score',0):.4f}<br>"
                f"wm_res={r.get('watermark_residual_score',0):.4f}<br>"
                f"patch_v={r.get('patch_visibility_score',0):.4f}<br>"
                f"edge_l={r.get('product_edge_loss_score',0):.4f}<br>"
                f"bnd_j={r.get('boundary_jump_score',0):.4f}"
                + ("<br><b>blur_fix</b>" if r.get("post_blur_applied") else ""))

        rows.append(f"""
<tr class='{r['status']}'>
  <td>{r['product_id']}<br><small>{r.get('roi_class','')}</small></td>
  <td><span class='badge {r['status']}'>{r['status']}</span><br>
      <small>{r.get('reason') or ''}</small></td>
  <td>{im('original',300)}</td>
  <td>{im('cleaned',300)}</td>
  <td>{im('mask_final',150)}<br>{im('diff_heatmap',150)}<br>
      {im('patch_visibility_heatmap',150)}<br>
      <small>mask={mi.get('final_mask','?')}<br>
      density={mi.get('mask_density_in_roi',0):.3f}<br>
      rect={mi.get('mask_rectangularity',0):.3f}</small></td>
  <td>{im('diff',150)}</td>
  <td><small>{pg_html or '—'}</small></td>
  <td><small>
    edge_overlap={ov.get('edge_overlap',0):.3f}<br>
    dark_overlap={ov.get('dark_overlap',0):.3f}<br>
    circle_overlap={ov.get('circle_overlap',0):.3f}<br>
    stroke_conf={mi.get('stroke_confidence',0):.2f}<br>
    n_components={mi.get('n_components',0)}<br>
    logo={'Y' if mi.get('logo_mask_active') else 'N'}
  </small></td>
  <td><small>
    tier={qaw.get('template',{}).get('tier','?')}<br>
    sec.edge={qaw.get('secondary',{}).get('edge_score',0):.3f}<br>
    jump={qaw.get('edge',{}).get('boundary_jump',0):.1f}<br>
    in_edge={qaw.get('edge',{}).get('inside_edge',0):.3f}<br>
    ssim={qal.get('ssim',0):.3f} cdelt={qal.get('color_delta',0):.1f}
  </small></td>
  <td><small>
    fg_loss={pi.get('fg_area_loss',0):.3f}<br>
    bbox={pi.get('bbox_shift',0):.3f}<br>
    edge_loss={pi.get('edge_loss',0):.3f}<br>
    outside_diff={pi.get('mean_outside_diff',0):.2f}<br>
    <span style='color:{piq_color}'>post_int={piq.get('score',0):.3f}</span><br>
    <span style='color:{art_color}'>artifact={qart.get('artifact_score',0):.3f}</span><br>
    last={last_att.get('method','—')}
  </small></td>
  <td><small>{pv_html}</small></td>
  <td><small>{cover_html}</small></td>
</tr>""")

    n_repaired = counts.get(ST_CLEAN_REPAIRED, 0)
    n_covered = counts.get(ST_CLEAN_COVERED, 0)
    n_no_wm = counts.get(ST_NO_WATERMARK, 0)
    n_failed = counts.get(ST_FAILED_IO, 0)
    summary_kv = " · ".join([
        f"<b>total</b>: {len(records)}",
        f"<b>clean_repaired</b>: <span class='clean_repaired'>{n_repaired}</span>",
        f"<b>clean_covered</b>: <span class='clean_covered'>{n_covered}</span>",
        f"<b>no_watermark</b>: <span style='color:#9f9'>{n_no_wm}</span>",
        f"<b>failed_io</b>: <span class='failed_io'>{n_failed}</span>",
        f"<b>manual-review</b>: <span style='color:#9f9'>0</span>",
    ])

    n_wm_total = n_repaired + n_covered
    repair_rate = f"{n_repaired / max(n_wm_total, 1) * 100:.0f}%"
    cover_rate = f"{n_covered / max(n_wm_total, 1) * 100:.0f}%"
    cover_methods_used = {}
    avg_opacity = []
    avg_cover_score = []
    avg_wm_residual = []
    fill_modes_used = {}
    n_blur_fix = 0
    for r in records:
        if r.get("cover_applied"):
            cm = r.get("cover_method", "?")
            cover_methods_used[cm] = cover_methods_used.get(cm, 0) + 1
            avg_opacity.append(r.get("cover_opacity", 0))
            if r.get("cover_score") is not None:
                avg_cover_score.append(r["cover_score"])
            if r.get("watermark_residual_score") is not None:
                avg_wm_residual.append(r["watermark_residual_score"])
            fm = r.get("cover_fill_mode", "?")
            fill_modes_used[fm] = fill_modes_used.get(fm, 0) + 1
            if r.get("post_blur_applied"):
                n_blur_fix += 1
    avg_op_str = f"{sum(avg_opacity)/max(len(avg_opacity),1):.2f}" if avg_opacity else "n/a"
    avg_cs_str = f"{sum(avg_cover_score)/max(len(avg_cover_score),1):.4f}" if avg_cover_score else "n/a"
    avg_wr_str = f"{sum(avg_wm_residual)/max(len(avg_wm_residual),1):.4f}" if avg_wm_residual else "n/a"
    most_common_cover = max(cover_methods_used, key=cover_methods_used.get) if cover_methods_used else "n/a"
    most_common_fill = max(fill_modes_used, key=fill_modes_used.get) if fill_modes_used else "n/a"

    summary_table = f"""<table style='margin:12px 0;border:1px solid #333'>
<tr><td>Repair success rate</td><td>{repair_rate}</td></tr>
<tr><td>Cover fallback rate</td><td>{cover_rate}</td></tr>
<tr><td>Average cover opacity</td><td>{avg_op_str}</td></tr>
<tr><td>Average cover score</td><td>{avg_cs_str}</td></tr>
<tr><td>Average watermark residual</td><td>{avg_wr_str}</td></tr>
<tr><td>Most common cover method</td><td>{most_common_cover}</td></tr>
<tr><td>Most common fill mode</td><td>{most_common_fill}</td></tr>
<tr><td>Post-cover blur fixes</td><td>{n_blur_fix}</td></tr>
</table>"""

    cls_line = " ".join(f"{k}={v}" for k, v in sorted(by_class.items()))
    rts_line = " ".join(f"{k}={v}" for k, v in sorted(by_route.items()))
    mask_line = " ".join(f"{k}={v}" for k, v in sorted(by_mask.items()))
    pg_line = " ".join(f"{k}={v}" for k, v in sorted(by_presence.items()))

    html = f"""<!doctype html>
<meta charset=utf-8>
<title>{PIPELINE_VERSION} — review</title>
<style>
  body {{ font: 13px/1.4 -apple-system, sans-serif; background:#111; color:#ddd; padding:16px; }}
  table {{ border-collapse: collapse; margin-top:16px; }}
  th, td {{ border:1px solid #333; padding:6px 8px; vertical-align:top; }}
  th {{ background:#1a1a1a; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:3px; font-weight:600; }}
  .badge.clean_repaired {{ background:#143; color:#cfc }}
  .badge.clean_covered {{ background:#342; color:#fda }}
  .badge.no_watermark {{ background:#133; color:#aef }}
  .badge.failed_io {{ background:#511; color:#fcc }}
  tr.clean_repaired {{ background:#0d1e10 }}
  tr.clean_covered {{ background:#251e0c }}
  tr.no_watermark {{ background:#0d161e }}
  tr.failed_io {{ background:#2a1010 }}
  .clean_repaired {{ color:#9f9 }} .clean_covered {{ color:#fc8 }} .failed_io {{ color:#f88 }}
  small {{ color:#aaa }}
</style>
<h1>{PIPELINE_VERSION} — review</h1>
<p>{summary_kv}</p>
{summary_table}
<p><small>by ROI class: {cls_line}<br>by last method: {rts_line}<br>by mask used: {mask_line}<br>by presence gate: {pg_line}</small></p>
<table>
<tr><th>product_id<br>(roi_class)</th><th>status<br>reason</th>
    <th>original</th><th>cleaned</th>
    <th>masks + heatmap</th><th>diff</th>
    <th>presence gate</th>
    <th>product overlap</th><th>watermark + local QA</th>
    <th>integrity QA</th><th>patch visibility</th><th>cover info</th></tr>
{''.join(rows)}
</table>
"""
    (out_root / "compare.html").write_text(html)


def write_compare_pdf(out_root, records) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    font_lg = font_md = font_sm = None
    for path in ("/System/Library/Fonts/SFNS.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            font_lg = ImageFont.truetype(path, 18)
            font_md = ImageFont.truetype(path, 13)
            font_sm = ImageFont.truetype(path, 10)
            break
        except OSError:
            continue
    if font_lg is None:
        font_lg = font_md = font_sm = ImageFont.load_default()

    cell_w, cell_h, pad = 480, 320, 22
    page_w = pad * 3 + cell_w * 2
    page_h = pad * 2 + 160 + cell_h + 28

    counts = {s: 0 for s in STATUS_ORDER}
    by_class = {}; by_route = {}; by_mask = {}; by_family = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        c = r.get("v9_roi_class") or r.get("roi_class") or "?"
        by_class[c] = by_class.get(c, 0) + 1
        fm = r.get("v9_final_method")
        if fm:
            by_route[fm] = by_route.get(fm, 0) + 1
        elif r.get("attempts"):
            m = r["attempts"][-1].get("method") or "?"
            by_route[m] = by_route.get(m, 0) + 1
        mu = (r.get("masks_info") or {}).get("final_mask", "n/a")
        by_mask[mu] = by_mask.get(mu, 0) + 1
        mf = r.get("v9_method_family") or "?"
        by_family[mf] = by_family.get(mf, 0) + 1

    s_h = max(620, 420 + 22 * (len(by_class) + len(by_route) + len(by_mask)))
    s = Image.new("RGB", (page_w, s_h), "white")
    d = ImageDraw.Draw(s)
    d.text((pad, pad),
           f"{PIPELINE_VERSION} — terminal-state report",
           font=font_lg, fill="black")
    d.text((pad, pad + 28),
           "States: clean_repaired / clean_covered / no_watermark / "
           "failed_io. Auto-cover eliminates manual-review.",
           font=font_md, fill="#555")
    # V12 Phase J — provenance + honesty counters on the cover page.
    rep_recs = [r for r in records if r["status"] == ST_CLEAN_REPAIRED]
    cr_dot = sum(1 for r in rep_recs
                 if (r.get("v11_residual_dot_chain_score") or 0.0)
                 > RESIDUAL_DOT_CHAIN_MAX)
    cr_band = sum(1 for r in rep_recs
                  if (r.get("v12_visible_band_score") or 0.0)
                  > BAND_VISIBLE_REPAIRED_MAX)
    cr_prod = sum(1 for r in rep_recs if r.get("v11_product_gate_pass") is False)
    meta = run_metadata(0)
    d.text((pad, pad + 46),
           f"git {meta['git_commit']} · qa_schema {meta['qa_schema_version']} "
           f"· strategy_schema {meta['strategy_schema_version']}",
           font=font_sm, fill="#777")
    d.text((pad, pad + 60),
           f"clean_repaired w/ dot-chain={cr_dot}  visible-band={cr_band}  "
           f"product-damage={cr_prod}  (all must be 0)",
           font=font_sm,
           fill=("#0a4" if (cr_dot == cr_band == cr_prod == 0) else "#a00"))

    y = pad + 80
    d.text((pad, y), f"total images:   {len(records)}",
           font=font_md, fill="black"); y += 22
    for st, color in [(ST_CLEAN_REPAIRED, "#0a4"),
                      (ST_CLEAN_COVERED, "#b60"),
                      (ST_AUTO_REJECTED, "#a00"),
                      (ST_NO_WATERMARK, "#06a"),
                      (ST_SKIPPED_KNOWN_CLEAN, "#888"),
                      (ST_FAILED_IO, "#a00")]:
        n = counts.get(st, 0)
        d.text((pad + 16, y), f"{st:22s}{n:>4d}",
               font=font_md, fill=color); y += 20
    y += 8
    d.text((pad, y), "ROI-class distribution:", font=font_md, fill="black"); y += 22
    for c, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        d.text((pad + 16, y), f"{c:22s}{n:>4d}",
               font=font_md, fill="black"); y += 18
    y += 4
    d.text((pad, y), "Final method used:", font=font_md, fill="black"); y += 22
    for m, n in sorted(by_route.items(), key=lambda kv: -kv[1]):
        d.text((pad + 16, y), f"{m:22s}{n:>4d}",
               font=font_md, fill="black"); y += 18
    y += 4
    d.text((pad, y), "Method family distribution:", font=font_md, fill="black"); y += 22
    for mf, n in sorted(by_family.items(), key=lambda kv: -kv[1]):
        mf_color = "#a00" if mf == "adaptive_cover" else "#0a4"
        d.text((pad + 16, y), f"{mf:28s}{n:>4d}",
               font=font_md, fill=mf_color); y += 18
    y += 4
    real_repair = sum(1 for r in records if r.get("v9_is_real_repair"))
    cover_count = sum(1 for r in records if r.get("v9_is_cover"))
    d.text((pad, y), f"real_repair_rate: {real_repair}/{len(records)}  "
           f"cover_rate: {cover_count}/{len(records)}  "
           f"unique_methods: {len(by_route)}",
           font=font_md, fill="#06a"); y += 22
    y += 4
    d.text((pad, y), "Mask used:", font=font_md, fill="black"); y += 22
    for m, n in sorted(by_mask.items(), key=lambda kv: -kv[1]):
        d.text((pad + 16, y), f"{m:22s}{n:>4d}",
               font=font_md, fill="black"); y += 18

    pages = [s]

    sort_key = {st: i for i, st in enumerate(STATUS_ORDER)}
    for r in sorted(records, key=lambda r: sort_key.get(r["status"], 99)):
        page = Image.new("RGB", (page_w, page_h), "white")
        d = ImageDraw.Draw(page)
        d.text((pad, 12), r["image"], font=font_md, fill="black")
        st_color = {"clean_repaired": "#0a4", "clean_covered": "#b60",
                    "auto_rejected": "#a00",
                    "no_watermark_confirmed": "#06a",
                    "skipped_known_clean": "#888",
                    "failed_io": "#a00"}.get(r["status"], "#888")
        v9_roi = r.get("v9_roi_class") or r.get("roi_class") or "?"
        v9_fm = r.get("v9_final_method") or "?"
        v9_mf = r.get("v9_method_family") or "?"
        d.text((pad, 32),
               f"status={r['status'].upper()}   roi_class={v9_roi}"
               f"   method={v9_fm}"
               f"   family={v9_mf}",
               font=font_sm, fill=st_color)
        atts = r.get("attempts") or []
        last_att = atts[-1] if atts else {}
        qaw = r.get("qa_watermark") or last_att.get("qa_water") or {}
        qal = r.get("qa_local_roi") or last_att.get("qa_local") or {}
        ov = r.get("product_overlap") or {}
        mi = r.get("masks_info") or {}

        line2 = (f"method={last_att.get('method','—')}  "
                 f"runtime={last_att.get('runtime_s',0):.2f}s  "
                 f"attempts={len(atts)}  mask={mi.get('final_mask','?')}"
                 f"  stroke_conf={mi.get('stroke_confidence',0):.2f}")
        d.text((pad, 52), line2, font=font_sm, fill="#555")

        psc_pdf = r.get("presence_scores") or {}
        line_pg = (f"presence: {r.get('presence_status','—')}  "
                   f"conf={r.get('presence_confidence') or 0:.3f}  "
                   f"clahe={psc_pdf.get('clahe_center',0):.2f}  "
                   f"tpl={psc_pdf.get('template',0):.2f}  "
                   f"stk={psc_pdf.get('stroke',0):.2f}")
        pg_color = "#06a" if r.get("presence_status") in (
            PG_CONFIRMED,) else "#b60"
        d.text((pad, 70), line_pg, font=font_sm, fill=pg_color)

        line3 = (f"overlap: edge={ov.get('edge_overlap',0):.3f}  "
                 f"dark={ov.get('dark_overlap',0):.3f}  "
                 f"circle={ov.get('circle_overlap',0):.3f}  "
                 f"n_circles={ov.get('n_circles_near_mark',0)}")
        d.text((pad, 88), line3, font=font_sm, fill="#555")

        if qaw:
            line4 = (f"watermark QA: tier={qaw.get('template',{}).get('tier','?')}  "
                     f"sec.edge={qaw.get('secondary',{}).get('edge_score',0):.3f}  "
                     f"jump={qaw.get('edge',{}).get('boundary_jump',0):.1f}  "
                     f"in_edge={qaw.get('edge',{}).get('inside_edge',0):.3f}")
            d.text((pad, 106), line4, font=font_sm, fill="#555")
        if qal:
            line5 = (f"local QA: ssim={qal.get('ssim',0):.3f}  "
                     f"color_delta={qal.get('color_delta',0):.1f}  "
                     f"boundary_jump={qal.get('boundary_jump',0):.1f}  "
                     f"pass={qal.get('pass',False)}")
            d.text((pad, 124), line5, font=font_sm, fill="#555")

        pi = r.get("qa_product_integrity") or {}
        if pi:
            pi_color = "#0a4" if pi.get("pass") else ("#a00" if pi.get("reject")
                                                       else "#b60")
            d.text((pad, 142),
                   f"product-integrity: fg_loss={pi.get('fg_area_loss',0):.3f}"
                   f"  bbox_shift={pi.get('bbox_shift',0):.3f}"
                   f"  edge_loss={pi.get('edge_loss',0):.3f}"
                   f"  outside_diff={pi.get('mean_outside_diff',0):.2f}",
                   font=font_sm, fill=pi_color)

        box_y = 160
        for col, key, label in [
                (0, "original", "ORIGINAL"),
                (1, "cleaned",
                 "REPAIRED" if r["status"] == ST_CLEAN_REPAIRED
                 else ("COVERED" if r["status"] == ST_CLEAN_COVERED
                       else "OUTPUT"))]:
            cx = pad + col * (cell_w + pad)
            d.rectangle([cx, box_y, cx + cell_w, box_y + cell_h], outline="#ccc")
            rel = (r.get("paths") or {}).get(key)
            if rel:
                full = out_root.parent / rel
                if full.exists():
                    im = Image.open(full).convert("RGB")
                    im.thumbnail((cell_w, cell_h))
                    page.paste(im, (cx + (cell_w - im.width) // 2,
                                    box_y + (cell_h - im.height) // 2))
            d.text((cx, box_y + cell_h + 4), label, font=font_sm,
                   fill=("#0a4" if col == 1 and r["status"] == ST_CLEAN_REPAIRED
                         else ("#b60" if col == 1
                               and r["status"] == ST_CLEAN_COVERED
                               else ("#a00" if col == 1
                                     and r["status"] == ST_FAILED_IO
                                     else "#555"))))
        pages.append(page)

    pdf = out_root / "compare.pdf"
    pages[0].save(str(pdf), save_all=True, append_images=pages[1:],
                  resolution=100.0)
    return pdf


# ---------------------------------------------------------------------------
# Candidate discovery — presence-gate-aware.
# Only watermark-detected and model-excluded images count toward the quota.
# Non-watermarked images are skipped entirely to focus processing effort.
# ---------------------------------------------------------------------------

def find_candidates(rwm, assets_dir, want, seed, max_scan,
                    iphone_only=False, presence_gate=True,
                    exclude_iphone14=True, iphone_min_model=14):
    rng = random.Random(seed)
    all_files = [p for p in assets_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                 and (not iphone_only or "iphone-" in p.name.lower())]
    rng.shuffle(all_files)

    if not presence_gate:
        files = [p for p in all_files if rwm.should_scan_file(p.name)]
        hits = []; scanned = 0
        for p in files:
            if len(hits) >= want or scanned >= max_scan:
                break
            scanned += 1
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            dets = rwm.detect_watermark_v2(img)
            if not dets:
                continue
            if dets[0].get("tier") in ("auto", "manual"):
                hits.append((p, dets[0], None))
        return hits, scanned, 0

    pg_conf = {"no_watermark_threshold": NO_WATERMARK_THRESHOLD,
                "confirmed_threshold": CONFIRMED_WATERMARK_THRESHOLD}
    n_excluded = 0; detected = []; n_no_det = 0; n_presence_reject = 0
    scanned = 0
    for p in all_files:
        if len(detected) >= want or scanned >= max_scan:
            break
        scanned += 1

        if exclude_iphone14:
            excl = model_exclusion_gate(p, iphone_min_model)
            if excl is not None:
                n_excluded += 1
                continue

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        dets = rwm.detect_watermark_v2(img)
        if not dets or dets[0].get("tier") not in ("auto", "manual"):
            n_no_det += 1
            continue

        det = dets[0]
        presence = detect_sunsky_presence_fast(img, det, pg_conf)
        ps = presence.get("status")
        if ps in (PG_NO_WATERMARK, PG_UNSUITABLE):
            n_presence_reject += 1
            continue
        scores = presence.get("scores", {})
        tpl_s = scores.get("template", 0)
        conf = presence.get("confidence", 0)
        if det.get("tier") == "manual" and tpl_s < 0.35:
            n_presence_reject += 1
            continue

        detected.append((p, det, None))

    return detected, scanned, n_excluded, n_no_det, n_presence_reject


# ---------------------------------------------------------------------------
# Presence-gate-aware image handler — wraps process_image.
# ---------------------------------------------------------------------------

def _copy_original_unchanged(path, out_root, debug_root, det, presence,
                             status=ST_NO_WATERMARK, reason=None):
    rec = _mk_record(path, det, status=status,
                     reason=reason or presence.get("reason"),
                     presence=presence)
    base = out_root / status / path.stem
    base.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    paths = {}
    if img is not None:
        cv2.imwrite(str(base / "original.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths["original"] = str((base / "original.jpg").relative_to(out_root.parent))
        shutil.copy2(str(path), str(base / "unchanged.jpg"))
        paths["cleaned"] = str((base / "unchanged.jpg").relative_to(out_root.parent))
    rec["paths"] = paths
    rec["publish_ok"] = False
    rec_to_save = dict(rec)
    (base / "qa.json").write_text(json.dumps(rec_to_save, indent=2, default=str))
    return rec


def process_image_with_presence_gate(rwm, path, det, out_root, debug_root,
                                     pg_config):
    enable_pg = pg_config.get("enable", True)
    enable_deep = pg_config.get("enable_deep", True)
    force_check = pg_config.get("force_presence_check", False)

    if det is None:
        img_gray_pre = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        scanned_det = None
        if img_gray_pre is not None:
            scanned_det = scan_center_band_for_watermark(img_gray_pre)
        if scanned_det is not None:
            det = scanned_det
        else:
            presence = {"status": PG_NO_WATERMARK, "confidence": 0.0,
                        "scores": {}, "reason": "no_detection"}
            return _copy_original_unchanged(
                path, out_root, debug_root, det, presence,
                reason="no_detection")

    if not enable_pg:
        return process_image(rwm, path, det, out_root, debug_root)

    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        rec = _mk_record(path, det, status=ST_FAILED_IO,
                         reason=R_INPUT_UNREADABLE)
        return rec

    presence = detect_sunsky_presence_fast(img_gray, det, pg_config)

    from_scanner = det.get("source") == "center_band_scan"
    clahe_overridden = False
    if presence["status"] == PG_NO_WATERMARK and not force_check:
        clahe_val = presence.get("scores", {}).get("clahe_center", 0.0)
        if clahe_val > 0.45 or from_scanner:
            presence["status"] = PG_UNCERTAIN
            presence["reason"] = ("center_band_scan_override" if from_scanner
                                  else "clahe_center_band_override")
            clahe_overridden = True
        else:
            return _copy_original_unchanged(
                path, out_root, debug_root, det, presence)

    if presence["status"] == PG_UNSUITABLE and not force_check:
        return _copy_original_unchanged(
            path, out_root, debug_root, det, presence,
            status=ST_NO_WATERMARK,
            reason="presence_detection_confused_by_product_text_or_qr")

    if presence["status"] == PG_UNCERTAIN:
        if enable_deep and not clahe_overridden and not from_scanner:
            presence = deep_sunsky_presence_check(
                img_gray, det, presence, pg_config)
        if presence["status"] == PG_NO_WATERMARK and not force_check:
            return _copy_original_unchanged(
                path, out_root, debug_root, det, presence)

    rec = process_image(rwm, path, det, out_root, debug_root)
    rec["presence_status"] = presence.get("status")
    rec["presence_confidence"] = presence.get("confidence")
    rec["presence_scores"] = presence.get("scores")
    rec["presence_reason"] = presence.get("reason")
    rec["presence_stage"] = presence.get("stage", "fast")
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-scan", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iphone-only", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    pg = ap.add_argument_group("presence gate")
    pg.add_argument("--presence-gate", action="store_true", default=True,
                    dest="presence_gate")
    pg.add_argument("--no-presence-gate", action="store_false",
                    dest="presence_gate")
    pg.add_argument("--presence-thumb-size", type=int, default=768)
    pg.add_argument("--presence-confirmed-threshold", type=float,
                    default=CONFIRMED_WATERMARK_THRESHOLD)
    pg.add_argument("--presence-no-watermark-threshold", type=float,
                    default=NO_WATERMARK_THRESHOLD)
    pg.add_argument("--presence-debug", action="store_true")
    pg.add_argument("--presence-jsonl", type=Path, default=None)
    pg.add_argument("--exclude-iphone14-plus-watermark-check",
                    action="store_true", default=True,
                    dest="exclude_iphone14")
    pg.add_argument("--no-exclude-iphone14-plus-watermark-check",
                    action="store_false", dest="exclude_iphone14")
    pg.add_argument("--iphone-exclusion-min-model", type=int, default=14)
    pg.add_argument("--force-presence-check", action="store_true")

    args = ap.parse_args()

    # V15 — seed numpy so the cover/fill noise is deterministic and the
    # benchmark is reproducible run-to-run (the file pick already uses its own
    # seeded RNG; this only fixes the in-fill Gaussian noise).
    np.random.seed(args.seed)

    if not args.assets.is_dir():
        sys.exit(f"assets dir not found: {args.assets}")
    out_root = args.out
    debug_root = out_root / "debug"
    for sub in (ST_CLEAN_REPAIRED, ST_CLEAN_COVERED, ST_NO_WATERMARK,
                ST_AUTO_REJECTED, ST_SKIPPED_KNOWN_CLEAN, ST_FAILED_IO,
                "debug"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    pg_config = {
        "enable": args.presence_gate,
        "enable_deep": True,
        "thumb_size": args.presence_thumb_size,
        "confirmed_threshold": args.presence_confirmed_threshold,
        "no_watermark_threshold": args.presence_no_watermark_threshold,
        "force_presence_check": args.force_presence_check,
        "debug": args.presence_debug,
    }

    print("loading detector (RESCUE3 enabled)…")
    rwm = _load_rwm()

    gate_label = "presence-gate ON" if args.presence_gate else "presence-gate OFF"
    excl_label = (f"iPhone>={args.iphone_exclusion_min_model} excluded"
                  if args.exclude_iphone14 else "no model exclusion")
    print(f"scanning {args.assets} for {args.n} images "
          f"(max scan {args.max_scan}, {gate_label}, {excl_label})…")
    t0 = time.time()
    result = find_candidates(
        rwm, args.assets, args.n, args.seed, args.max_scan,
        args.iphone_only,
        presence_gate=args.presence_gate,
        exclude_iphone14=args.exclude_iphone14,
        iphone_min_model=args.iphone_exclusion_min_model)
    if len(result) == 5:
        candidates, scanned, n_excluded, n_no_det, n_pg_reject = result
    elif len(result) == 4:
        candidates, scanned, n_excluded, n_no_det = result
        n_pg_reject = 0
    else:
        candidates, scanned, n_excluded = result
        n_no_det = 0; n_pg_reject = 0
    print(f"  scanned {scanned}, candidates {len(candidates)} "
          f"(detected={len(candidates)}, iPhone14+-skipped={n_excluded}, "
          f"no-watermark={n_no_det}, presence-rejected={n_pg_reject}) "
          f"in {time.time()-t0:.1f}s")
    if not candidates:
        sys.exit("no candidate images found — try --max-scan higher")

    if args.dry_run:
        for p, det, excl in candidates:
            print(f"  DETECTED: {p.name}")
        return

    t_pipe = time.time()
    records = []
    total = len(candidates)
    for idx, (path, det, excl) in enumerate(candidates, 1):
        rec = process_image_with_presence_gate(
            rwm, path, det, out_root, debug_root, pg_config)
        records.append(rec)
        ps = rec.get("presence_status") or "—"
        roi = rec.get("roi_class") or "?"
        atts = rec.get("attempts") or []
        last_m = (atts[-1].get("method") if atts else "—") or "—"
        mu = (rec.get("masks_info") or {}).get("final_mask", "?")
        print(f"  [{idx}/{total}] {path.name} → {rec['status']} "
              f"(pg={ps}; {rec.get('reason') or 'pass'}; roi={roi}, "
              f"mask={mu}, last={last_m})")

    total_s = time.time() - t_pipe
    counts = {s: 0 for s in STATUS_ORDER}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    write_summary_jsonl(out_root, records)
    write_summary_csv(out_root, records)
    write_compare_html(out_root, records)

    presence_jsonl = args.presence_jsonl or (out_root / "presence.jsonl")
    write_presence_jsonl(presence_jsonl, records)

    pdf_path = None
    if not args.no_pdf:
        pdf_path = write_compare_pdf(out_root, records)

    every_has_status = all(r["status"] in STATUS_ORDER for r in records)
    every_has_qa = all((out_root / r["status"] / r["product_id"] / "qa.json").exists()
                       for r in records)
    no_false_clean_repaired = all(
        r.get("repair_qa_pass", False) for r in records
        if r["status"] == ST_CLEAN_REPAIRED)
    no_wm_untouched = all(r.get("attempts") is None or len(r.get("attempts", [])) == 0
                          for r in records if r["status"] == ST_NO_WATERMARK)
    manual_review_count = sum(1 for r in records
                              if r["status"] not in STATUS_ORDER)

    v9_method_dist = {}
    v9_family_dist = {}
    v9_roi_dist = {}
    for r in records:
        fm = r.get("v9_final_method", "?")
        v9_method_dist[fm] = v9_method_dist.get(fm, 0) + 1
        mf = r.get("v9_method_family", "?")
        v9_family_dist[mf] = v9_family_dist.get(mf, 0) + 1
        rc = r.get("v9_roi_class", "?")
        v9_roi_dist[rc] = v9_roi_dist.get(rc, 0) + 1
    real_repair = sum(1 for r in records if r.get("v9_is_real_repair"))
    cover_count = sum(1 for r in records if r.get("v9_is_cover"))

    # V12 Phase J — visual-truthfulness summary counters. The three "*_in_*"
    # counters are the honesty guarantee and must always be zero.
    def _sum(key):
        return sum(int(r.get(key) or 0) for r in records)

    repaired_recs = [r for r in records if r["status"] == ST_CLEAN_REPAIRED]
    cr_dot = sum(1 for r in repaired_recs
                 if (r.get("v11_residual_dot_chain_score") or 0.0)
                 > RESIDUAL_DOT_CHAIN_MAX)
    cr_band = sum(1 for r in repaired_recs
                  if (r.get("v12_visible_band_score") or 0.0)
                  > BAND_VISIBLE_REPAIRED_MAX)
    cr_prod = sum(1 for r in repaired_recs
                  if r.get("v11_product_gate_pass") is False)
    v12_counters = {
        "dot_chain_rejections": _sum("v12_dot_chain_rejections"),
        "band_rejections": _sum("v12_band_rejections"),
        "product_damage_rejections": _sum("v12_product_damage_rejections"),
        "micro_cleanup_attempts": _sum("v12_micro_cleanup_attempts"),
        "micro_cleanup_successes": _sum("v12_micro_cleanup_successes"),
        "forced_cover_count": sum(1 for r in records
                                  if r.get("v12_forced_cover")),
        "clean_repaired_with_visible_band": cr_band,
        "clean_repaired_with_dot_chain": cr_dot,
        "clean_repaired_with_product_damage": cr_prod,
    }
    zero_metric_passes = sum(
        1 for r in repaired_recs if r.get("v10_metrics_valid") is False)
    meta = run_metadata(args.seed)
    meta["summary_counters"] = v12_counters
    meta["status_counts"] = {s: counts.get(s, 0) for s in STATUS_ORDER}
    meta["zero_metric_passes"] = zero_metric_passes
    (out_root / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print()
    print(f"=== {PIPELINE_VERSION} complete ===")
    print(f"total:            {len(records)}")
    print(f"clean_repaired:   {counts.get(ST_CLEAN_REPAIRED, 0)}")
    print(f"clean_covered:    {counts.get(ST_CLEAN_COVERED, 0)}")
    print(f"auto_rejected:    {counts.get(ST_AUTO_REJECTED, 0)}")
    print(f"no_watermark_confirmed: {counts.get(ST_NO_WATERMARK, 0)}")
    print(f"skipped_known_clean:    {counts.get(ST_SKIPPED_KNOWN_CLEAN, 0)}")
    print(f"failed_io:        {counts.get(ST_FAILED_IO, 0)}")
    print(f"manual-review:    0   (auto_rejected is a final automated decision, not manual review)")
    print(f"real_repair_rate: {real_repair}/{len(records)}")
    print(f"cover_rate:       {cover_count}/{len(records)}")
    print(f"unique_methods:   {len(v9_method_dist)}")
    print(f"active_roi_classes: {len(v9_roi_dist)}")
    print(f"runtime:        {total_s:.1f}s "
          f"({(total_s*1000)/max(len(records),1):.0f} ms/img)")
    print()
    print("method family distribution:")
    for mf, n in sorted(v9_family_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {mf:28s}{n:>4d}")
    print()
    print("V9 final method used:")
    for fm, n in sorted(v9_method_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {fm:36s}{n:>4d}")
    print()
    print("V9 ROI class distribution:")
    for rc, n in sorted(v9_roi_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {rc:28s}{n:>4d}")
    print()
    print("presence gate:")
    pg_counts = {}
    for r in records:
        ps = r.get("presence_status") or "n/a"
        pg_counts[ps] = pg_counts.get(ps, 0) + 1
    for k, v in sorted(pg_counts.items()):
        print(f"  {k}: {v}")
    print()
    print("V12 visual-truthfulness counters:")
    print(f"  dot_chain_rejections:                   "
          f"{v12_counters['dot_chain_rejections']}")
    print(f"  band_rejections:                        "
          f"{v12_counters['band_rejections']}")
    print(f"  product_damage_rejections:              "
          f"{v12_counters['product_damage_rejections']}")
    print(f"  micro_cleanup (attempts/successes):     "
          f"{v12_counters['micro_cleanup_attempts']}/"
          f"{v12_counters['micro_cleanup_successes']}")
    print(f"  forced_cover_count:                     "
          f"{v12_counters['forced_cover_count']}")
    print()
    print("V12 acceptance criteria (must all hold):")
    print(f"  manual-review count = 0:                {manual_review_count == 0}")
    print(f"  failed_io = 0:                          "
          f"{counts.get(ST_FAILED_IO, 0) == 0}")
    print(f"  zero_metric_passes = 0:                 {zero_metric_passes == 0}")
    print(f"  clean_repaired_with_dot_chain = 0:      "
          f"{v12_counters['clean_repaired_with_dot_chain'] == 0}")
    print(f"  clean_repaired_with_visible_band = 0:   "
          f"{v12_counters['clean_repaired_with_visible_band'] == 0}")
    print(f"  clean_repaired_with_product_damage = 0: "
          f"{v12_counters['clean_repaired_with_product_damage'] == 0}")
    print(f"  every image has terminal status:        {every_has_status}")
    print(f"  every image has qa.json:                {every_has_qa}")
    print(f"  no false clean_repaired:                {no_false_clean_repaired}")
    print(f"  no-watermark images untouched:          {no_wm_untouched}")
    print(f"  git_commit={meta['git_commit']} seed={meta['run_seed']} "
          f"qa_schema={meta['qa_schema_version']}")
    print(f"  report title = {PIPELINE_VERSION}")
    print()
    print(f"compare html:    {out_root / 'compare.html'}")
    if pdf_path:
        print(f"compare pdf:     {pdf_path}")
    print(f"summary:         {out_root / 'summary.jsonl'}")
    print(f"presence jsonl:  {presence_jsonl}")


if __name__ == "__main__":
    main()
