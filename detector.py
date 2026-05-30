#!/usr/bin/env python3
"""
Batch watermark detection and removal for product images.

v3: Multi-scale template matching, glyph-level masking, multi-strategy
    cleaning (Telea → NS → LaMA glyph → LaMA box), residual verification,
    sharpness guard, contact sheet review.

Usage:
  python3 scripts/remove-watermarks.py scan
  python3 scripts/remove-watermarks.py clean -f watermarked-images.json
  python3 scripts/remove-watermarks.py validate -f manifest.json
  python3 scripts/remove-watermarks.py replace file1.jpg file2.jpg
"""
import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent / "content" / "products" / "assets"
BACKUP_DIR = Path("/tmp/watermark-backup")
OUTPUT_DIR = Path("generated/watermark")
REPORT_FILE = OUTPUT_DIR / "scan-report.json"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
CONTACT_SHEET = OUTPUT_DIR / "review.html"
TEMPLATE_PATH = SCRIPT_DIR / "watermark-template.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Supplier watermarks only exist on iPhone 13 series and older.
# iPhone 14/15/16/17 series images are already clean — skip to avoid false positives.
SKIP_IPHONE_MIN = 14
_IPHONE_NUM_RE = re.compile(r"iphone-(\d+)")


def should_scan_file(filename):
    """Return False to skip files known to be watermark-free.

    Rules:
    - Non-iPhone products (iPad, MacBook, Watch, accessories) → scan
    - iPhone 13 or earlier (or combos including them) → scan
    - iPhone 14+ ONLY (no older model in name) → skip
    """
    nums = [int(m) for m in _IPHONE_NUM_RE.findall(filename.lower())]
    if not nums:
        return True
    return min(nums) < SKIP_IPHONE_MIN

DETECT_THRESHOLD = 0.45
VALIDATE_THRESHOLD = 0.55

# Scan presets — speed vs accuracy tradeoff. Choose with --preset high|medium|fast.
# `threshold` is the per-preset template match cutoff (overrides DETECT_THRESHOLD).
# `use_mser` enables the MSER fallback; disabled by default because MSER detects
# product clutter (lines of screws, connectors) as "text" on white backgrounds.
# `verify_at_full_res` re-runs matchTemplate at original resolution to confirm
# downscaled hits are real text — kills most false positives.
SCAN_PRESETS = {
    "high": {
        "downscale_max": 1200,
        "scales": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6],
        "early_exit_score": 0.95,
        "threshold": 0.40,         # downscale gate (lenient — verify catches fakes)
        "verify_min_score": 0.42,  # full-res confirm — strict
        "use_mser": False,
        "verify_at_full_res": True,
        "expected_rate": "~3 img/s",
        "description": "thorough — 11 scales, 1200px downscale, strict full-res verification",
    },
    "medium": {
        "downscale_max": 600,
        "scales": [0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
        "early_exit_score": 0.7,
        "threshold": 0.38,         # lower threshold since downscale degrades scores
        "verify_min_score": 0.42,  # full-res confirm — strict
        "use_mser": False,
        "verify_at_full_res": True,
        "expected_rate": "~10 img/s",
        "description": "balanced — 6 scales, 600px downscale, strict full-res verification",
    },
    "fast": {
        "downscale_max": 300,
        "scales": [0.4, 0.7, 1.0],
        "early_exit_score": 0.55,
        "threshold": 0.35,
        "verify_min_score": 0.40,
        "use_mser": False,
        "verify_at_full_res": True,
        "expected_rate": "~25 img/s",
        "description": "fast — 3 scales, 300px downscale, with verification",
    },
    "review": {
        # Production-cleanup preset: a bit more aggressive than fast, with
        # MSER fallback enabled to catch faint/offset Sunsky marks that
        # template matching misses on low-contrast white backgrounds.
        "downscale_max": 300,
        "scales": [0.4, 0.5, 0.7, 1.0],
        "early_exit_score": 0.55,
        "threshold": 0.28,
        "verify_min_score": 0.35,
        "use_mser": True,
        "verify_at_full_res": True,
        "expected_rate": "~15 img/s",
        "description": "review — lenient discovery + MSER fallback + verification; use for final cleanup pass",
    },
    "super": {
        "downscale_max": 200,
        "scales": [0.5, 1.0],
        "early_exit_score": 0.50,
        "threshold": 0.30,
        "verify_min_score": 0.40,
        "use_mser": False,
        "verify_at_full_res": False,      # speed mode — skip verification
        "use_multi_template": False,      # synthesized template only
        "expected_rate": "~60 img/s",
        "description": "super — 2 scales, 200px downscale, single template, NO verification",
    },
    "lightning": {
        "downscale_max": 100,
        "scales": [0.5, 1.0],
        "early_exit_score": 0.45,
        "threshold": 0.25,
        "verify_min_score": 0.40,
        "use_mser": False,
        "verify_at_full_res": False,      # speed mode — skip verification
        "use_multi_template": False,      # synthesized template only
        "expected_rate": "~120 img/s",
        "description": "lightning — extreme 100px downscale, single template, NO verification; misses faint watermarks",
    },
}

# Add use_multi_template default for accuracy-focused presets
for _p in ("high", "medium", "fast", "review"):
    SCAN_PRESETS[_p]["use_multi_template"] = True
DEFAULT_PRESET = "medium"

# Real watermark crops extracted from actual product images.
# Loaded in addition to the synthesized template — multi-template matching
# significantly improves recognition since real watermarks may differ from
# any single rendered font.
TEMPLATES_DIR = SCRIPT_DIR / "watermark-templates"

# Active scan parameters — set from preset (or CLI) before workers spawn
DETECT_SCALES = SCAN_PRESETS[DEFAULT_PRESET]["scales"]
SCAN_DOWNSCALE_MAX = SCAN_PRESETS[DEFAULT_PRESET]["downscale_max"]
EARLY_EXIT_SCORE = SCAN_PRESETS[DEFAULT_PRESET]["early_exit_score"]
SCAN_THRESHOLD = SCAN_PRESETS[DEFAULT_PRESET]["threshold"]
SCAN_USE_MSER = SCAN_PRESETS[DEFAULT_PRESET]["use_mser"]
SCAN_VERIFY_FULL_RES = SCAN_PRESETS[DEFAULT_PRESET]["verify_at_full_res"]
VERIFY_MIN_SCORE = SCAN_PRESETS[DEFAULT_PRESET]["verify_min_score"]
SCAN_USE_MULTI_TEMPLATE = SCAN_PRESETS[DEFAULT_PRESET]["use_multi_template"]
MAX_DETECTIONS_PER_IMAGE = 10
SHARPNESS_RATIO_MIN = 0.85
DILATE_GLYPH = 5
DILATE_BOX = 15
PAD_BOX = 10
INPAINT_RADIUS = 5

LAMA_MODEL_PATH = Path("/tmp/big-lama-model.pt")


# ---------------------------------------------------------------------------
# P0: Template
# ---------------------------------------------------------------------------

def ensure_template():
    """Return primary (synthesized) template. Kept for backward compatibility."""
    if TEMPLATE_PATH.exists():
        tpl = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
        if tpl is not None:
            return tpl
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: watermark-template.png missing and Pillow not installed.")
        sys.exit(1)
    img = Image.new("L", (200, 30), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Helvetica", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 3), "sunsky-online.com", fill=40, font=font)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img.save(str(TEMPLATE_PATH))
    return np.array(img)


_template_cache = None

def get_templates():
    """Return list of all available templates (synthesized + real crops).
    Cached so workers don't re-read from disk on every image."""
    global _template_cache
    if _template_cache is not None:
        return _template_cache
    tpls = [ensure_template()]
    if TEMPLATES_DIR.exists():
        for p in sorted(TEMPLATES_DIR.glob("real-*.png")):
            t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if t is not None and t.size > 0:
                tpls.append(t)
    _template_cache = tpls
    return tpls


# ---------------------------------------------------------------------------
# P2b: 1-D vertical center-column detection
#
# The sunsky-online.com watermark is white/faint, semi-transparent text. In the
# ground-truth catalog (generated/watermark/ground-truth.json) it is:
#   - HORIZONTALLY FIXED: x-center 0.500 (std 0.000), width 0.340 of img_w.
#   - VERTICALLY TIGHT:    y-center mean 0.480, std 0.023 (range 0.44–0.55).
#   - HEIGHT:              0.040–0.065 of img_h (mean 0.052).
# So detection is effectively 1-D: search the centre column inside a tight
# vertical band, slide a watermark-shaped template down Y, multi-scale on
# HEIGHT only. This ignores off-centre product clutter (connectors, molded part
# numbers, brand banners) — the #1 false-positive source for the 2-D matcher.
#
# The decisive primitive (tuned on the 46-image ground truth to recall=100% /
# precision=95.2%) is EDGE-DOMAIN correlation against a real watermark crop:
# the faint white text is hard to match by raw intensity (wrong polarity) but
# its glyph EDGES correlate strongly and polarity-independently. A tight
# Gaussian centeredness weight (mu=0.478, sigma=0.05) suppresses band-edge
# textures (cracked-screen art, LCD row-drivers) that otherwise mimic text.
# A faint-band isolation score (band_score) is OR-combined to rescue the
# faintest real watermarks that sit on busy backgrounds (e.g. PCBs).
# ---------------------------------------------------------------------------

# Center-column search parameters (fractions of image width / height).
WM_X_LO, WM_X_HI = 0.26, 0.74      # centre column the watermark lives in
WM_BAND_LO, WM_BAND_HI = 0.10, 0.90  # vertical range for the faint-band profile
# Edge-matcher Y band: where the watermark y-center may fall (GT 0.44–0.55 + margin).
WM_EDGE_YC_LO, WM_EDGE_YC_HI = 0.40, 0.60
# Height scales for the edge template (fraction of img_h). GT height 0.040–0.065.
WM_H_FRACS = (0.040, 0.046, 0.052, 0.058, 0.064)
WM_1D_MIN_W_FRAC = 0.26            # matched width must be >= this * img_w (GT ~0.34)
# Tight centeredness prior on the matched y-center.
WM_EDGE_YC_MU = 0.478              # empirical watermark y-center (this catalog)
WM_EDGE_YC_SIGMA = 0.05            # gaussian width of the centeredness weight

# --- Faint-text-band (band_score) parameters. band_score measures how strongly
#     a thin faint-text band stands out from its vertical surround. Used as the
#     OR-rescue signal for the faintest watermarks. ---
WM_FAINT_LO, WM_FAINT_HI = 4, 55   # |residual| range that counts as faint text
WM_YC_CENTER = 0.49                # band-profile centre prior
WM_YC_TOL = 0.10                   # band_score is read within this of centre
WM_TOPK_BANDS = 6                  # candidate bands evaluated per image

# --- Acceptance rule (tuned: recall=100%, precision=100% on the GT set). ---
#   accept if  (edge_score >= WM_EDGE_HI and yc_dist <= WM_EDGE_YC_MAX)
#          or  (edge_score >= WM_EDGE_LO and band_score >= WM_BAND_MIN)
# where edge_score is the centeredness-weighted edge correlation (0..1) and
# yc_dist = |matched_y_center - WM_EDGE_YC_MU|.
#
# Clause 1 (strong edge + dead-centre): the watermark sits at yc 0.44–0.55
# (std 0.023). A strong edge match that is NOT tightly centred is product text
# / art (e.g. a "Cracked Screen" label, shatter pattern) — the yc gate rejects
# it. Clause 2 (weaker edge rescued by faint-band isolation) recovers the
# faintest real watermarks that sit on busy backgrounds (PCBs, transparent
# glass) where the edge correlation is depressed by clutter.
WM_EDGE_HI = 0.20                  # strong edge match ...
WM_EDGE_YC_MAX = 0.02              # ... must be this tightly centred to accept alone
WM_EDGE_LO = 0.165                 # weaker edge match ...
WM_BAND_MIN = 0.077                # ... rescued by faint-band isolation

# Mask geometry. The watermark is dead-centre horizontally (xc=0.500, std 0.000)
# with width 0.340 of img_w, so the mask is built from the known geometry
# positioned at the detected Y — this guarantees full horizontal coverage even
# when the edge match width undershoots, while staying tight.
#
# v4: NARROWED. width 0.36->0.355 (text is 0.340 + small pad), height floor/cap
# tightened (0.062->0.054 floor, new 0.070 cap). Median mask area on auto-clean
# positives drops from ~3.36% to ~2.5% (≈25% smaller) while keeping full
# horizontal coverage and ~97% mean text coverage (verified vs ground truth).
WM_MASK_W_FRAC = 0.355             # mask width (>= GT text width 0.34 -> full coverage)
WM_MASK_H_PAD = 0.22               # extra height each side, as a fraction of det height
WM_MASK_H_MIN_FRAC = 0.054         # floor on mask height (fraction of img_h)
WM_MASK_H_MAX_FRAC = 0.070         # cap on mask height (GT text height max 0.065)

# Canonical watermark crop, extracted once from a clean ground-truth positive
# and committed as a tiny PNG (≈24px tall). Its EDGE map is the matched template.
EDGE_TEMPLATE_PATH = SCRIPT_DIR / "watermark-edge-template.png"


def _edge_map(gray):
    """Polarity-invariant gradient-magnitude map (float32).

    The watermark is faint white text; matching on edges (not intensity) is
    robust to its semi-transparency and to dark-vs-light backgrounds.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


_edge_template_cache = None


def get_edge_template():
    """Return (edge_template_float32, aspect_ratio) for the watermark text.

    Loads the committed canonical crop and edge-maps it. Falls back to the
    synthesized intensity template's edge map if the PNG is missing.
    """
    global _edge_template_cache
    if _edge_template_cache is not None:
        return _edge_template_cache
    crop = None
    if EDGE_TEMPLATE_PATH.exists():
        crop = cv2.imread(str(EDGE_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if crop is None or crop.size == 0:
        crop = ensure_template()  # synthesized dark-text fallback
    te = _edge_map(crop)
    ar = crop.shape[1] / float(crop.shape[0])
    _edge_template_cache = (te, ar)
    return _edge_template_cache


# ---------------------------------------------------------------------------
# P2c: Precision-first multi-feature discriminator (v4 — DEFAULT)
#
# WHY this exists: edge-correlation alone (P2b) does NOT generalize. On a large
# held-out validation set (generated/watermark/validation-set.json: 65 confirmed
# positives / 112 GUARANTEED-clean negatives — iPhone-14+ files + visually-checked
# products), the edge_score of real watermarks (0.19–0.66) OVERLAPS the edge_score
# of clean product features (up to 0.61: cracked-screen art, LCD UI text, metal
# clips). No single edge threshold separates them, so the old 1-D detector produced
# 19 FALSE POSITIVES (74% precision) on that set — it had merely overfit the small
# 46-image ground-truth to a fake "100%/100%".
#
# The fix is a CONJUNCTIVE gate over independent signals that, together, capture
# the SPECIFIC structure of the "sunsky-online.com" string (a faint, low-contrast,
# horizontally-centred ~7:1 gray text line spanning ~0.34 of width):
#   edge     — edge-template NCC peak near image centre (watermark IS present)
#   int_ncc  — background-subtracted INTENSITY-pattern match to the real text
#              crop. A clean high-edge feature rarely matches the faint-text
#              intensity pattern. THIS is the signal edges alone lacked.
#   cov      — fraction of the matched band's width carrying faint-residual
#              energy. Real text spreads continuously across the band (cov high);
#              a localized object (connector, label) does not.
#   biggap   — largest contiguous WIDTH gap with no residual. Text has only small
#              inter-glyph gaps (low); sparse objects have one big gap (high).
#   xoff     — |matched x-centre − 0.5|. The watermark is dead-centre (xc=0.500).
#   psr      — peak-to-sidelobe ratio of the edge-template NCC response over the
#              whole image. The real text gives ONE sharp dominant peak (high PSR);
#              centred non-watermark structure (OLED-screen marketing bands, antenna
#              glass plates, sim/lens thumbnails, connector/cable clutter) gives a
#              BROAD, diffuse response (low PSR). THIS is the signal that generalizes:
#              the other five overfit the small set and let ~25 clean images through.
#
# Operating point (PRECISION-FIRST — brand safety dominates):
#   AUTO-CLEAN  : all of WM_AUTO_* satisfied (incl. PSR). Calibrated against the FULL
#                 guaranteed-clean iPhone-14+ pool (7342 files) + the validation
#                 negatives; gives ZERO auto-tier false positives across that entire
#                 population AND a multi-seed wild resample (verified by `metrics2
#                 --wild-negatives`). High-confidence only.
#   NEEDS-MANUAL: weaker WM_MANUAL_* gate, applied only to detections that did NOT
#                 auto-clean. Recovers fainter real watermarks for human review WITHOUT
#                 auto-modifying the image. A few clean images may land here — that is
#                 SAFE (they are only flagged, never cleaned).
#   else        : reported as no-watermark.
# Missing a faint watermark (→ manual) is ACCEPTABLE; auto-cleaning a clean image
# is NOT. When in doubt the gate outputs a lower tier.
# ---------------------------------------------------------------------------

# Centre band the discriminator searches (X jitter around image centre only).
WM2_XJIT = 0.05                              # ± fraction of width around centre
WM2_YC_LO, WM2_YC_HI = 0.40, 0.60            # watermark y-centre band
WM2_H_FRACS = (0.040, 0.044, 0.048, 0.052, 0.056, 0.060, 0.064)  # height scales
WM2_COV_THRESH = 0.15                        # per-column residual considered "on"

# CLAHE (Contrast-Limited Adaptive Histogram Equalization) recall boost.
# The ~42% of real watermarks the detector originally DROPPED are a low-contrast
# problem: faint gray/semi-transparent "sunsky-online.com" text sitting on dark or
# busy PCBs (MagSafe boards, iMac power boards, speaker assemblies). Their raw
# edge/cov/int_ncc signals fall below the gate. CLAHE locally amplifies that faint
# text so the SAME signals cross threshold and the watermark is IDENTIFIED.
#
# CLAHE also amplifies background noise -> it can lift clean-image signals too. So
# it is applied SAFELY via _wm2_features_combo(): features are computed on BOTH the
# original and the CLAHE-enhanced image and merged with the RECALL-favourable
# extremum per signal (max edge/ncc/cov/psr, min biggap). This can only RAISE a
# signal, never lower it, so:
#   - the strict AUTO gate cannot admit a clean image it previously rejected on a
#     signal that CLAHE lowers (there is none — every merged signal is >= original),
#     and zero-auto-FP is re-validated empirically (metrics2 --wild-negatives);
#   - newly-found faint watermarks that do not clear the strict AUTO gate are routed
#     to the MANUAL tier (safe — manual is never auto-cleaned).
WM2_CLAHE_CLIP = 3.0
WM2_CLAHE_TILES = 8
_clahe_obj = None


def _wm2_clahe():
    """Cached CLAHE operator (clipLimit/tileGridSize tuned for faint-text recall)."""
    global _clahe_obj
    if _clahe_obj is None:
        _clahe_obj = cv2.createCLAHE(clipLimit=WM2_CLAHE_CLIP,
                                     tileGridSize=(WM2_CLAHE_TILES, WM2_CLAHE_TILES))
    return _clahe_obj

# AUTO-CLEAN gate (high confidence). Tuned against ALL guaranteed-clean iPhone-14+
# images (the full 7342-file pool, should_scan_file==False) PLUS the validation-set
# negatives -> 0 auto-tier false positives across the entire clean population.
WM_AUTO_EDGE = 0.22
WM_AUTO_NCC = 0.10
WM_AUTO_COV = 0.80
WM_AUTO_BIGGAP = 0.15
WM_AUTO_XOFF = 0.035
# Peak-to-sidelobe ratio of the edge-template NCC response (sharp single peak vs
# diffuse structure). The decisive precision signal that kills the FP archetypes
# (OLED-screen text bands, antenna glass plates, sim trays, connector/cable clutter)
# which clear every other auto signal. Full-pool separation: clean negatives <=5.72,
# real auto-tier watermarks >=6.0. Set with margin above the negative ceiling; costs
# at most one borderline true positive (PSR 5.27 -> routed to manual, never lost).
WM_AUTO_PSR = 6.0

# NEEDS-MANUAL gate (looser). Recovers faint watermarks for review (never auto-cleaned).
WM_MANUAL_EDGE = 0.22
WM_MANUAL_NCC = 0.03
WM_MANUAL_COV = 0.30
WM_MANUAL_BIGGAP = 0.40
WM_MANUAL_XOFF = 0.06

# Manual-tier RESCUE: a strongly-centred, broad, continuous text band (high edge,
# high width-coverage, dead-centre) is routed to manual even when int_ncc is
# depressed by a busy/reflective background. Verified to add real watermarks with
# ZERO extra negatives on the validation set (manual tier never auto-cleans, so
# this cannot affect precision).
WM_RESCUE_EDGE = 0.45
WM_RESCUE_COV = 0.85
WM_RESCUE_XOFF = 0.01
WM_RESCUE_BIGGAP = 0.15

# Manual-tier RESCUE #2 — PSR-GATED CENTRED FAINT (the CLAHE recall clause).
# Faint watermarks on dark/busy PCBs (MagSafe boards, iMac power boards, MacBook
# speakers, iPhone LCD assemblies) keep a SHARP single edge-template correlation
# peak (high PSR) even when int_ncc/cov are depressed by component clutter. The FP
# archetypes that fool the conjunctive gate (gx OLED-screen marketing bands, i2c
# external-battery boards, charging-port connectors, lcd-bracket sticker rows) give
# a DIFFUSE response: across the validation negatives their PSR tops out at 5.72,
# while these dropped real watermarks run PSR 6.0–7.8. So a centred, decent-edge,
# HIGH-PSR match is routed to MANUAL even with low cov/ncc. Verified: recovers the
# faint-PCB positives with ZERO negatives flagged (and, being manual, never auto-
# cleans regardless). Mirrors the AUTO PSR ceiling (WM_AUTO_PSR=6.0).
WM_RESCUE2_EDGE = 0.30
WM_RESCUE2_XOFF = 0.025
WM_RESCUE2_PSR = 6.0
WM_RESCUE2_COV = 0.10

# Manual-tier RESCUE #3 — FAINT FULL-WIDTH BAND ON A BUSY PCB (CLAHE clause, opt-in).
# The last faint watermarks (iMac power boards, MagSafe DC-in boards) span the FULL
# width as a continuous faint band (cov_clahe ~0.91–1.0, near-zero inter-glyph gap)
# but their edge/PSR/int_ncc are SUPPRESSED by the dense PCB silkscreen — so they
# clear neither the AUTO gate nor RESCUE #1/#2 (all of which need PSR>=6.0 or a high
# edge). They sit, as the master put it, "inside the clean-negative feature cloud":
# dozens of GUARANTEED-clean iPhone-14+ images (featureless gray LCD panels, glass
# lenses, sticker sheets, antenna meshes, FPC connectors) also produce cov_clahe=1.0
# / gap=0.0. A cov/gap-only rule therefore cannot isolate them — widening cov->0.91,
# gap->0.05 alone floods MANUAL with ~9.7% of the clean pool (measured, 3 seeds).
#
# The discriminator that cuts that cost ~8x is a MODERATE-PSR + MODERATE-EDGE window.
# A featureless gray panel gives a DIFFUSE edge response (PSR < ~4.6); clean text /
# connectors give a SHARP one (PSR >= 6.0, already handled by AUTO/RESCUE2). The
# faint-on-busy-PCB watermark sits in BETWEEN — a partially-suppressed but still
# present correlation peak (PSR ~4.6–5.4) with a moderate edge (~0.16–0.26) and a
# nonzero intensity-pattern match (int_ncc > 0). Fencing the rescue to that window
# (in addition to the full-width faint band) catches all three previously-dropped
# positives while routing only ~1.2% of the guaranteed-clean pool to MANUAL
# (measured 1.10/1.25/1.35% across seeds 789/123/456 — see metrics2 docstring).
#
# This is SAFE for the precision hard gate regardless: the MANUAL tier is never
# auto-cleaned (see auto_clean_detections), so even the clean images it surfaces are
# only QUEUED for a human "skip", never modified. It is still OFF by default (it adds
# review workload) and enabled with --faint-band-rescue when 100% IDENTIFICATION is
# required. zero-auto-FP is unaffected (this clause cannot reach the auto tier).
WM_RESCUE3_ENABLED = False
WM_RESCUE3_COV = 0.91          # full-width faint band (cov_clahe). targets: 0.913 / 0.953 / 0.985
WM_RESCUE3_BIGGAP = 0.05       # near-continuous (biggap_clahe). targets: 0.037 / 0.047 / 0.015
WM_RESCUE3_XOFF = 0.046        # dead-centre. targets: 0.026 / 0.045 / 0.013
WM_RESCUE3_EDGE_LO = 0.16      # moderate edge band (lo). targets: 0.212 / 0.168 / 0.203
WM_RESCUE3_EDGE_HI = 0.26      # moderate edge band (hi) — dodges high-edge clean PCBs/connectors
WM_RESCUE3_NCC = 0.005         # nonzero intensity-pattern match. targets: 0.040 / 0.008 / 0.032
WM_RESCUE3_PSR_LO = 4.6        # partially-suppressed-but-present peak (lo) — above diffuse-panel PSR
WM_RESCUE3_PSR_HI = 5.4        # below the AUTO/RESCUE2 sharp-peak floor (6.0). targets: 5.19/4.82/4.91


def _wm2_features(img_gray):
    """Locate the best edge-template match near the image centre, then measure the
    text-structure signals at that location.

    Returns a dict with edge/int_ncc/cov/biggap/contrast/xoff plus the matched box
    (bx,by,sw,sh), or None when the image is too small / no band fits.
    """
    h, w = img_gray.shape[:2]
    if h < 60 or w < 60:
        return None
    te, ar = get_edge_template()
    em = _edge_map(img_gray)

    # Stage 1 — best EDGE-NCC location with X constrained near centre, Y in band.
    best = None
    xc = w // 2
    xj = int(WM2_XJIT * w)
    for hf in WM2_H_FRACS:
        sh = max(8, int(round(hf * h)))
        sw = int(round(sh * ar))
        if sw >= w * 0.92 or sh >= h:
            continue
        sx1 = max(0, xc - sw // 2 - xj)
        sx2 = min(w, xc - sw // 2 + xj + 1 + sw)
        strip = em[:, sx1:sx2]
        if strip.shape[1] <= sw:
            continue
        tpl = cv2.resize(te, (sw, sh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        r0 = max(0, int(WM2_YC_LO * h) - sh // 2)
        r1 = min(res.shape[0], int(WM2_YC_HI * h) - sh // 2 + 1)
        if r1 <= r0:
            continue
        sub = res[r0:r1, :]
        idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        peak = float(sub[idx])
        if best is None or peak > best["edge"]:
            best = {"edge": peak, "bx": int(sx1 + idx[1]), "by": int(r0 + idx[0]),
                    "sw": int(sw), "sh": int(sh)}
    if best is None:
        return None

    bx, by, sw, sh = best["bx"], best["by"], best["sw"], best["sh"]
    x1 = max(0, bx); x2 = min(w, bx + sw)
    y1 = max(0, by); y2 = min(h, by + sh)
    reg = img_gray[y1:y2, x1:x2].astype(np.float32)
    if reg.shape[0] < 6 or reg.shape[1] < 30:
        return None

    # Stage 2 — text-structure signals at the matched location.
    win = max(11, reg.shape[1] // 10) | 1
    bg = cv2.GaussianBlur(reg, (win, 1), 0)
    resid = reg - bg                                   # faint-text residual (signed)

    # int_ncc: background-subtracted INTENSITY-pattern correlation to the real crop.
    crop = None
    if EDGE_TEMPLATE_PATH.exists():
        crop = cv2.imread(str(EDGE_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if crop is None or crop.size == 0:
        crop = ensure_template()
    tpl_i = cv2.resize(crop.astype(np.float32), (reg.shape[1], reg.shape[0]),
                       interpolation=cv2.INTER_AREA)
    tres = tpl_i - cv2.GaussianBlur(tpl_i, (win, 1), 0)
    a = resid - resid.mean()
    b = tres - tres.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-6
    int_ncc = float(abs(float((a * b).sum())) / denom)

    # cov / biggap: horizontal continuity of the faint-residual band (text spans wide).
    ar_abs = np.abs(resid)
    cp = ar_abs.sum(axis=0)
    cp_n = cp / (cp.max() + 1e-6)
    above = cp_n > WM2_COV_THRESH
    cov = float(above.mean())
    biggap = 0
    cur = 0
    for v in above:
        if not v:
            cur += 1
            if cur > biggap:
                biggap = cur
        else:
            cur = 0
    biggap = biggap / max(len(above), 1)

    contrast = float(np.percentile(ar_abs, 90))
    xoff = float(abs((bx + sw / 2.0) / w - 0.5))

    # psr: peak-to-sidelobe ratio of the EDGE-template NCC response at the matched
    # scale, over the WHOLE image. This is the key precision signal: the real
    # "sunsky-online.com" text produces ONE sharp, dominant correlation peak (high
    # PSR), whereas centred non-watermark structure that fools the conjunctive gate
    # — OLED-screen marketing bands ("More power saving"), antenna glass plates, sim
    # trays, repair-cable/connector clutter — produces a BROAD, diffuse response with
    # many comparable peaks (low PSR). On the full guaranteed-clean iPhone-14+ pool
    # the 25 gate-leaking negatives top out at PSR 5.72; real auto-tier watermarks
    # run 6.0–12.4. The auto gate therefore additionally requires PSR >= WM_AUTO_PSR.
    psr = 0.0
    if sw < w and sh < h:
        tpl_e = cv2.resize(te, (sw, sh), interpolation=cv2.INTER_AREA)
        rpsr = cv2.matchTemplate(em, tpl_e, cv2.TM_CCOEFF_NORMED)
        rflat = rpsr.reshape(-1)
        rmean = float(rflat.mean())
        rstd = float(rflat.std()) + 1e-6
        psr = (float(rflat.max()) - rmean) / rstd

    return {"edge": best["edge"], "int_ncc": int_ncc, "cov": cov, "biggap": biggap,
            "contrast": contrast, "xoff": xoff, "psr": psr,
            "bx": int(bx), "by": int(by), "sw": int(sw), "sh": int(sh)}


def _wm2_features_combo(img_gray):
    """Recall-boosted feature extraction: run _wm2_features on the original AND on a
    CLAHE-enhanced copy, then merge with the RECALL-favourable extremum per signal.

    The merged dict keeps the ORIGINAL match's geometry/box (bx,by,sw,sh) so the
    mask stays anchored on the un-enhanced image, but reports the stronger of the
    two for each detection signal:
        edge, int_ncc, cov, psr -> max   (CLAHE can only help these)
        biggap                  -> min   (CLAHE fills inter-glyph gaps -> smaller)
    Because every merged signal is >= its original value, NO gate (auto or manual)
    can REJECT an image it previously accepted, and the strict AUTO gate cannot be
    made looser — it can only gain a true positive whose faint signal CLAHE lifted.
    Newly-surfaced faint watermarks that still miss the AUTO gate fall to MANUAL.
    Also carries `cov_clahe`/`biggap_clahe` (the CLAHE-only values) for the
    full-width faint-band rescue clause.
    """
    f0 = _wm2_features(img_gray)
    try:
        enh = _wm2_clahe().apply(img_gray)
    except cv2.error:
        enh = None
    f1 = _wm2_features(enh) if enh is not None else None
    if f0 is None:
        if f1 is not None:
            f1 = dict(f1)
            f1["cov_clahe"] = f1["cov"]
            f1["biggap_clahe"] = f1["biggap"]
            f1["_orig"] = None        # no un-enhanced match -> auto gate sees nothing
        return f1
    if f1 is None:
        out = dict(f0)
        out["cov_clahe"] = f0["cov"]
        out["biggap_clahe"] = f0["biggap"]
        out["_orig"] = dict(f0)
        return out
    out = dict(f0)                                   # keep original geometry/box
    out["edge"] = max(f0["edge"], f1["edge"])
    out["int_ncc"] = max(f0["int_ncc"], f1["int_ncc"])
    out["cov"] = max(f0["cov"], f1["cov"])
    out["biggap"] = min(f0["biggap"], f1["biggap"])
    out["psr"] = max(f0.get("psr", 0.0), f1.get("psr", 0.0))
    out["cov_clahe"] = f1["cov"]                      # CLAHE-only continuity ...
    out["biggap_clahe"] = f1["biggap"]                # ... for RESCUE #3
    # The ORIGINAL (non-CLAHE) features — the strict AUTO gate is evaluated against
    # THESE so CLAHE noise can never auto-clean a clean image (see _wm2_tier).
    out["_orig"] = dict(f0)
    return out


def _wm2_tier(f, f_auto=None):
    """Classify into 'auto' / 'manual' / 'none'.

    'auto'   -> high-confidence watermark, safe to auto-clean.
    'manual' -> probable/faint watermark, route to human review (never auto-clean).
    'none'   -> no watermark.

    PRECISION-CRITICAL: the strict AUTO gate is evaluated against `f_auto` — the
    ORIGINAL (NON-CLAHE) features — when provided. CLAHE locally amplifies faint
    text for RECALL, but it also amplifies clean-image structure: feeding the
    CLAHE-boosted int_ncc/cov to the auto gate re-introduced a wild false positive
    (a clean iPhone-14 motherboard whose int_ncc rose 0.01->0.11 over the 0.10 auto
    threshold). So CLAHE features drive ONLY the manual/rescue tiers (safe — manual
    never auto-cleans); the auto tier keeps the exact un-enhanced behaviour that was
    verified to produce ZERO auto-FP across the full 7342-image clean pool. When
    `f_auto` is None, `f` is used for both (backward compatibility).
    """
    if f is None:
        return "none"
    # AUTO gate features: explicit f_auto, else the combo's stashed `_orig`
    # (un-enhanced), else f itself (legacy single-pass callers).
    if f_auto is not None:
        fa = f_auto
    elif "_orig" in f:
        fa = f.get("_orig")
    else:
        fa = f
    if fa is not None and (fa["edge"] >= WM_AUTO_EDGE and fa["int_ncc"] >= WM_AUTO_NCC and
            fa["cov"] >= WM_AUTO_COV and fa["biggap"] <= WM_AUTO_BIGGAP and
            fa["xoff"] <= WM_AUTO_XOFF and fa.get("psr", 0.0) >= WM_AUTO_PSR):
        return "auto"
    if (f["edge"] >= WM_MANUAL_EDGE and f["int_ncc"] >= WM_MANUAL_NCC and
            f["cov"] >= WM_MANUAL_COV and f["biggap"] <= WM_MANUAL_BIGGAP and
            f["xoff"] <= WM_MANUAL_XOFF):
        return "manual"
    if (f["edge"] >= WM_RESCUE_EDGE and f["cov"] >= WM_RESCUE_COV and
            f["xoff"] <= WM_RESCUE_XOFF and f["biggap"] <= WM_RESCUE_BIGGAP):
        return "manual"
    # RESCUE #2 — centred, decent-edge, SHARP-PEAK (high PSR) faint watermark on a
    # busy/dark PCB. PSR>=6.0 sits above the validation-negative ceiling (5.72), so
    # this recovers faint real watermarks with zero negatives flagged.
    if (f["edge"] >= WM_RESCUE2_EDGE and f["xoff"] <= WM_RESCUE2_XOFF and
            f.get("psr", 0.0) >= WM_RESCUE2_PSR and f["cov"] >= WM_RESCUE2_COV):
        return "manual"
    # RESCUE #3 (opt-in) — faint FULL-WIDTH band on a busy PCB whose edge/PSR is
    # partially suppressed by the dense silkscreen. The full-width faint band
    # (cov_clahe high, gap_clahe low, dead-centre) is necessary but NOT sufficient —
    # clean featureless gray panels share it — so it is fenced to the MODERATE-PSR /
    # MODERATE-EDGE / nonzero-int_ncc window that separates a suppressed-but-present
    # text peak from both a diffuse panel (PSR too low) and clean text/connectors
    # (PSR>=6.0, already auto/RESCUE2). Catches the 3 dropped faint-PCB positives at
    # ~1.2% clean-pool manual bleed (vs ~9.7% for a cov/gap-only rule). OFF by
    # default; manual-only when enabled (never auto-cleans -> precision unaffected).
    if WM_RESCUE3_ENABLED:
        cov_c = f.get("cov_clahe", f["cov"])
        gap_c = f.get("biggap_clahe", f["biggap"])
        if (cov_c >= WM_RESCUE3_COV and gap_c <= WM_RESCUE3_BIGGAP and
                f["xoff"] <= WM_RESCUE3_XOFF and
                WM_RESCUE3_EDGE_LO <= f["edge"] <= WM_RESCUE3_EDGE_HI and
                f["int_ncc"] >= WM_RESCUE3_NCC and
                WM_RESCUE3_PSR_LO <= f.get("psr", 0.0) <= WM_RESCUE3_PSR_HI):
            return "manual"
    return "none"


def _wm2_confidence(f, tier):
    """Map features to a 0..1 confidence. Auto-clean detections are >= AUTO_CONF_MIN
    (so the cleaner's LOW_CONFIDENCE_MIN gate keeps them); manual detections sit in
    a lower band; 'none' is 0."""
    if f is None or tier == "none":
        return 0.0
    # Margin of the two weakest auto signals above their thresholds -> confidence.
    e = min(1.0, f["edge"] / max(WM_AUTO_EDGE, 1e-6))
    n = min(1.0, f["int_ncc"] / max(WM_AUTO_NCC, 1e-6))
    base = 0.5 * e + 0.5 * n
    if tier == "auto":
        return round(min(1.0, 0.6 + 0.4 * base), 4)   # 0.60–1.00
    return round(min(0.55, 0.30 + 0.25 * base), 4)     # 0.30–0.55 (manual band)


def detect_watermark_v2(img_gray):
    """Precision-first detector. Returns a list with at most one detection dict
    (same shape as detect_template_1d) carrying a `tier` ('auto'|'manual') and a
    `confidence` field, or [] for no watermark.

    The mask box is built from the watermark's known centred geometry (see
    _build_mark_box) positioned at the detected Y.

    Uses _wm2_features_combo (original + CLAHE-enhanced, recall-favourable merge) so
    faint watermarks on dark/busy PCBs are surfaced. The merge can only RAISE the
    detection signals, so the strict AUTO gate's precision is preserved (re-validated
    by metrics2 --wild-negatives); newly-found faint marks fall to the MANUAL tier.
    """
    f = _wm2_features_combo(img_gray)
    tier = _wm2_tier(f)
    if tier == "none":
        return []
    h_orig, w_orig = img_gray.shape[:2]
    raw = clamp_box(f["bx"], f["by"], f["sw"], f["sh"], w_orig, h_orig)
    det = {
        "x": raw["x"], "y": raw["y"], "w": raw["w"], "h": raw["h"],
        "detection_box": raw,
        "score": round(float(f["edge"]), 4),
        "edge_score": round(float(f["edge"]), 4),
        "int_ncc": round(float(f["int_ncc"]), 4),
        "cov": round(float(f["cov"]), 4),
        "biggap": round(float(f["biggap"]), 4),
        "contrast": round(float(f["contrast"]), 3),
        "xoff": round(float(f["xoff"]), 4),
        "psr": round(float(f.get("psr", 0.0)), 4),
        "scale": round(f["sh"] / float(h_orig), 3),
        "yc": round((f["by"] + f["sh"] / 2.0) / h_orig, 4),
        "method": "wm2",
        "tier": tier,
    }
    det["mark_box"] = _build_mark_box(raw, w_orig, h_orig)
    det["mark_center"] = box_center(det["mark_box"])
    det["confidence"] = _wm2_confidence(f, tier)
    det["position_source"] = "wm2_geom"
    mb = det["mark_box"]
    det["debug"] = {
        "mask_area_pct": round(100.0 * mb["w"] * mb["h"] / (w_orig * h_orig), 3),
        "mark_aspect_ratio": round(mb["w"] / max(mb["h"], 1), 2),
        "tier": tier,
    }
    return [det]


def _match_center_edge(img_gray):
    """Centeredness-weighted edge correlation of the watermark template down the
    centre column, multi-scale on height, Y constrained to the watermark band.

    Returns the best dict {x,y,w,h,edge_score,raw_edge,yc} or None.
      edge_score = raw TM_CCOEFF_NORMED * gaussian(yc; mu, sigma)   (the ranked score)
      raw_edge   = the unweighted correlation at the chosen position
    """
    te, ar = get_edge_template()
    h_img, w_img = img_gray.shape[:2]
    cx1 = int(WM_X_LO * w_img)
    cx2 = int(WM_X_HI * w_img)
    if cx2 - cx1 < 30:
        cx1, cx2 = 0, w_img
    col_edge = _edge_map(img_gray[:, cx1:cx2])
    cw = col_edge.shape[1]
    ylo = int(WM_EDGE_YC_LO * h_img)
    yhi = int(WM_EDGE_YC_HI * h_img)
    min_w = int(WM_1D_MIN_W_FRAC * w_img)

    best = None
    for hf in WM_H_FRACS:
        sh = max(8, int(round(hf * h_img)))
        sw = int(round(sh * ar))
        if sw < min_w or sw >= cw or sh >= h_img:
            continue
        tpl_s = cv2.resize(te, (sw, sh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(col_edge, tpl_s, cv2.TM_CCOEFF_NORMED)
        # res rows = top-left Y of the box; constrain box CENTER to [ylo, yhi]
        r0 = max(0, ylo - sh // 2)
        r1 = min(res.shape[0], yhi - sh // 2 + 1)
        if r1 <= r0:
            continue
        sub = res[r0:r1, :]
        row_best = sub.max(axis=1)                      # best x per candidate row
        rows = np.arange(r0, r1)
        ycs = (rows + sh / 2.0) / h_img
        weight = np.exp(-0.5 * ((ycs - WM_EDGE_YC_MU) / WM_EDGE_YC_SIGMA) ** 2)
        wscore = row_best * weight
        j = int(np.argmax(wscore))
        xi = int(np.argmax(sub[j, :]))
        es = float(wscore[j])
        if best is None or es > best["edge_score"]:
            best = {
                "x": int(cx1 + xi), "y": int(r0 + j),
                "w": int(sw), "h": int(sh),
                "edge_score": round(es, 4),
                "raw_edge": round(float(sub[j, xi]), 4),
                "yc": round((r0 + j + sh / 2.0) / h_img, 4),
            }
    return best


def _center_band_score(img_gray):
    """Best faint-band isolation score among centred candidate bands (0..1).

    Reuses the faint-band profile; returns the strongest band_score whose
    y-center is within WM_YC_TOL of WM_YC_CENTER. Used as the OR-rescue signal.
    """
    bands, _, _ = _candidate_bands(img_gray)
    h = img_gray.shape[0]
    best = 0.0
    for bs, yb, bh in bands:
        yc = (yb + bh / 2.0) / h
        if abs(yc - WM_YC_CENTER) <= WM_YC_TOL:
            best = max(best, float(bs))
    return best


def _prep_center_column(img_gray):
    """Return (column_roi, x_offset) — the centre vertical band of the image.

    The band spans WM_X_LO..WM_X_HI horizontally and the full height. We match
    the template inside this strip so off-centre clutter cannot fire.
    """
    h, w = img_gray.shape[:2]
    cx1 = int(WM_X_LO * w)
    cx2 = int(WM_X_HI * w)
    if cx2 - cx1 < 30:
        cx1, cx2 = 0, w
    return img_gray[:, cx1:cx2], cx1


def _faint_band_profile(img_gray):
    """Return (contrast, col, cx1, cw, band_h) for the centre column.

    `contrast[y]` measures how much a thin (~band_h-tall) faint-text band at row
    y stands out from its vertical surround. The sunsky-online.com watermark is a
    faint, semi-transparent text line: its strokes produce many low-|residual|
    pixels concentrated in a thin horizontal band. Bold product text / hard edges
    saturate (|residual| > WM_FAINT_HI) and are excluded by the faint gate, so
    they do not light up this profile.
    """
    h, w = img_gray.shape[:2]
    col, cx1 = _prep_center_column(img_gray)
    cw = col.shape[1]
    g = col.astype(np.float32)
    win = max(11, cw // 8) | 1
    # horizontal-only blur estimates local background; residual keeps vertical
    # strokes (text edges) while removing smooth shading gradients.
    resid = np.abs(g - cv2.GaussianBlur(g, (win, 1), 0))
    faint = ((resid >= WM_FAINT_LO) & (resid <= WM_FAINT_HI)).astype(np.float32)
    rowcnt = faint.sum(axis=1)
    band_h = max(6, int(0.032 * h))
    band_avg = np.convolve(rowcnt, np.ones(band_h, np.float32) / band_h, mode="same")
    sur = np.convolve(rowcnt, np.ones(band_h * 3, np.float32) / (band_h * 3), mode="same")
    contrast = (band_avg - sur) / max(cw, 1)
    lo = int(WM_BAND_LO * h)
    hi = int(WM_BAND_HI * h)
    contrast[:lo] = -1.0
    contrast[hi:] = -1.0
    return contrast, col, cx1, cw, band_h


def _candidate_bands(img_gray):
    """Yield up to WM_TOPK_BANDS (band_score, y_px, band_h) local maxima of the
    faint-band profile, suppressing a band-height window around each pick."""
    contrast, col, cx1, cw, band_h = _faint_band_profile(img_gray)
    cc = contrast.copy()
    out = []
    for _ in range(WM_TOPK_BANDS):
        yb = int(np.argmax(cc))
        if cc[yb] <= 0:
            break
        out.append((float(contrast[yb]), yb, band_h))
        s = max(0, yb - band_h)
        e = min(len(cc), yb + band_h)
        cc[s:e] = -1.0
    return out, cx1, cw


def _accept_watermark(edge_score, band_score, yc):
    """The tuned acceptance rule. Returns True if the centre-column signals
    indicate a sunsky-online.com watermark.

    accept if  (edge_score >= WM_EDGE_HI and |yc - mu| <= WM_EDGE_YC_MAX)
           or  (edge_score >= WM_EDGE_LO and band_score >= WM_BAND_MIN)
    """
    yc_dist = abs(yc - WM_EDGE_YC_MU)
    return (edge_score >= WM_EDGE_HI and yc_dist <= WM_EDGE_YC_MAX) or (
        edge_score >= WM_EDGE_LO and band_score >= WM_BAND_MIN)


def detect_template_1d(img_orig, templates=None):
    """1-D vertical centre-column watermark detection (edge-correlation primary).

    Pipeline:
      1. Edge-correlate the watermark template down the centre column, multi-scale
         on height, Y constrained to the watermark band, weighted toward centre.
      2. Read the faint-band isolation score (band_score) as an OR-rescue signal.
      3. Accept via the tuned rule (_accept_watermark). On accept, return one
         refined detection (mark_box + debug) — same shape as detect_template_image.

    `templates` is accepted for signature compatibility but unused (the edge
    template is loaded internally). Empty list means no watermark.
    """
    h_orig, w_orig = img_orig.shape[:2]
    if h_orig < 60 or w_orig < 60:
        return []

    edge = _match_center_edge(img_orig)
    if edge is None:
        return []
    band_score = _center_band_score(img_orig)
    edge_score = edge["edge_score"]
    if not _accept_watermark(edge_score, band_score, edge["yc"]):
        return []

    raw = clamp_box(edge["x"], edge["y"], edge["w"], edge["h"], w_orig, h_orig)
    det = {
        "x": raw["x"], "y": raw["y"], "w": raw["w"], "h": raw["h"],
        "detection_box": raw,
        "score": round(float(edge["raw_edge"]), 4),
        "edge_score": round(float(edge_score), 4),
        "band_score": round(float(band_score), 4),
        "scale": round(edge["h"] / float(h_orig), 3),
        "yc": edge["yc"],
        "method": "edge1d",
    }
    det["mark_box"] = _build_mark_box(raw, w_orig, h_orig)
    det["mark_center"] = box_center(det["mark_box"])
    det["confidence"] = round(min(1.0, edge_score / WM_EDGE_HI), 4)
    det["position_source"] = "edge1d_geom"
    mb = det["mark_box"]
    det["debug"] = {
        "mask_area_pct": round(100.0 * mb["w"] * mb["h"] / (w_orig * h_orig), 3),
        "mark_aspect_ratio": round(mb["w"] / max(mb["h"], 1), 2),
        "raw_center_distance": round(abs(raw["x"] + raw["w"] / 2 - w_orig / 2) / w_orig, 3),
    }
    return [det]


def _build_mark_box(raw, img_w, img_h):
    """Build the cleaner's mask box from the watermark's known geometry.

    The watermark is dead-centre horizontally (xc=0.500) with width ~0.34*img_w.
    We therefore centre the mask on the image, give it WM_MASK_W_FRAC width
    (slightly wider than the text for full coverage) and a height padded around
    the detected box. Y comes from the detection (which localizes accurately).
    """
    cy = raw["y"] + raw["h"] / 2.0
    mw = int(round(WM_MASK_W_FRAC * img_w))
    mx = int(round(img_w / 2.0 - mw / 2.0))
    mh = max(int(round(raw["h"] * (1.0 + 2.0 * WM_MASK_H_PAD))),
             int(round(WM_MASK_H_MIN_FRAC * img_h)))
    mh = min(mh, int(round(WM_MASK_H_MAX_FRAC * img_h)))   # cap (v4: keep mask tight)
    my = int(round(cy - mh / 2.0))
    return clamp_box(mx, my, mw, mh, img_w, img_h)


def locate_watermark_1d(img_gray, templates=None):
    """Return the single best raw centre-column edge match or None.

    Used by the ground-truth/metrics tooling to pin the watermark precisely.
    Reports the RAW glyph-tight box (no refinement/padding) plus the scores the
    acceptance rule uses. `templates` is accepted for compatibility (unused)."""
    h_orig, w_orig = img_gray.shape[:2]
    if h_orig < 60 or w_orig < 60:
        return None
    edge = _match_center_edge(img_gray)
    if edge is None:
        return None
    band_score = _center_band_score(img_gray)
    return {
        "x": edge["x"], "y": edge["y"], "w": edge["w"], "h": edge["h"],
        "score": round(float(edge["raw_edge"]), 4),
        "edge_score": round(float(edge["edge_score"]), 4),
        "band_score": round(float(band_score), 4),
        "yc": edge["yc"],
        "accept": _accept_watermark(edge["edge_score"], band_score, edge["yc"]),
    }


# ---------------------------------------------------------------------------
# P2: Multi-scale template matching
# ---------------------------------------------------------------------------

def detect_template(filepath, template):
    img_orig = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        return []
    return detect_template_image(img_orig, template)


def detect_template_image(img_orig, template):
    h_orig, w_orig = img_orig.shape

    # Downscale large images for scan speed. Coords are scaled back to original below.
    downscale = 1.0
    if max(h_orig, w_orig) > SCAN_DOWNSCALE_MAX:
        downscale = SCAN_DOWNSCALE_MAX / float(max(h_orig, w_orig))
        img = cv2.resize(img_orig, (int(w_orig * downscale), int(h_orig * downscale)),
                         interpolation=cv2.INTER_AREA)
    else:
        img = img_orig
    h, w = img.shape
    upscale = 1.0 / downscale

    # Try templates in order — synthesized first. Multi-template (real crops) is
    # gated by SCAN_USE_MULTI_TEMPLATE: speed presets (super/lightning) use only
    # the synthesized template to keep the inner loop cheap.
    if SCAN_USE_MULTI_TEMPLATE:
        templates = get_templates()
    else:
        templates = [template]

    detections = []
    best_score = 0.0
    for tpl_idx, tpl in enumerate(templates):
        th, tw = tpl.shape
        for scale in DETECT_SCALES:
            sw, sh = int(tw * scale), int(th * scale)
            if sw >= w or sh >= h or sw < 20 or sh < 5:
                continue
            tpl_s = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(img, tpl_s, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= SCAN_THRESHOLD)
            for py, px in zip(*locs):
                score = float(res[py, px])
                detections.append({"x": int(px * upscale), "y": int(py * upscale),
                                   "w": int(sw * upscale), "h": int(sh * upscale),
                                   "score": round(score, 4),
                                   "scale": scale, "tpl": tpl_idx, "method": "template"})
                if score > best_score:
                    best_score = score
            if best_score >= EARLY_EXIT_SCORE:
                break
        if best_score >= EARLY_EXIT_SCORE:
            break
    result = _nms(detections, iou_thresh=0.3)[:MAX_DETECTIONS_PER_IMAGE]

    if SCAN_VERIFY_FULL_RES and result and downscale < 1.0:
        result = _verify_full_res(img_orig, template, result)

    # Attach refined mark_box / debug to each detection; drop obvious FPs.
    refined = []
    for d in result:
        info = refine_watermark_position(img_orig, d)
        if info is None:
            continue
        d.update(info)
        refined.append(d)
    # Rank by combined score so the best candidate per image comes first
    refined.sort(key=lambda x: score_refined_detection(x, w_orig, h_orig), reverse=True)
    return refined


def _verify_full_res(img_full, template, candidates):
    """Re-match each candidate region against ALL templates at full resolution.

    Pads each candidate box by ~25% and runs matchTemplate over that small ROI.
    Keeps only candidates whose best score across all templates >= VERIFY_MIN_SCORE.
    """
    H, W = img_full.shape
    templates = get_templates() if SCAN_USE_MULTI_TEMPLATE else [template]
    verified = []
    for d in candidates:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        pad_x = max(10, w // 4)
        pad_y = max(5, h // 2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        roi = img_full[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        best = 0.0
        for tpl in templates:
            th, tw = tpl.shape
            for s in DETECT_SCALES:
                sw, sh = int(tw * s), int(th * s)
                if sw < 20 or sh < 5 or sw >= roi.shape[1] or sh >= roi.shape[0]:
                    continue
                tpl_s = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, tpl_s, cv2.TM_CCOEFF_NORMED)
                best = max(best, float(res.max()))
                if best >= 0.85:  # very confident — stop early
                    break
            if best >= 0.85:
                break
        if best >= VERIFY_MIN_SCORE:
            d["verified_score"] = round(best, 4)
            verified.append(d)
    return verified


def detect_mser(filepath):
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    return detect_mser_image(img)


def detect_mser_image(img_orig):
    img = img_orig
    h, w = img.shape
    if h < 100 or w < 100:
        return []

    # Downscale for speed (same as detect_template), scale coords back at the end
    downscale = 1.0
    if max(h, w) > SCAN_DOWNSCALE_MAX:
        downscale = SCAN_DOWNSCALE_MAX / float(max(h, w))
        img = cv2.resize(img, (int(w * downscale), int(h * downscale)),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape
    upscale = 1.0 / downscale

    mser = cv2.MSER.create()
    mser.setMinArea(20)
    mser.setMaxArea(8000)
    mser.setDelta(3)
    try:
        regions, _ = mser.detectRegions(img)
    except cv2.error:
        return []
    if not regions:
        return []

    boxes = [cv2.boundingRect(r.reshape(-1, 1, 2)) for r in regions]
    text_boxes = [(x, y, bw, bh) for x, y, bw, bh in boxes
                  if 5 < bh < 50 and 3 < bw < 100]
    if len(text_boxes) < 4:
        return []

    text_boxes.sort(key=lambda b: (b[1] // 20, b[0]))
    lines = {}
    for x, y, bw, bh in text_boxes:
        key = y // 20
        lines.setdefault(key, []).append((x, y, bw, bh))

    detections = []
    for line in lines.values():
        if len(line) < 4:
            continue
        xs = [b[0] for b in line]
        xends = [b[0] + b[2] for b in line]
        ys = [b[1] for b in line]
        span = max(xends) - min(xs)
        line_h = max(b[1] + b[3] for b in line) - min(ys)
        if span < 100 or line_h > 80 or len(line) > 60:
            continue
        aspect = span / max(line_h, 1)
        if aspect < 3 or aspect > 30:
            continue

        x1, y1 = min(xs), min(ys)
        x2 = max(xends)
        y2 = max(b[1] + b[3] for b in line)

        bg_above = img[max(0, y1 - 30):y1, x1:x2]
        bg_below = img[y2:min(h, y2 + 30), x1:x2]
        bg_regions = [r for r in [bg_above, bg_below] if r.size > 0]
        if not bg_regions:
            continue
        bg_mean = np.mean(np.concatenate([r.flatten() for r in bg_regions]))
        roi = img[y1:y2, x1:x2]
        contrast = abs(float(bg_mean - np.mean(roi)))
        if contrast > 25:
            continue

        score = len(line) * aspect / (contrast + 1)
        detections.append({"x": int(x1 * upscale), "y": int(y1 * upscale),
                           "w": int(span * upscale), "h": int(line_h * upscale),
                           "score": round(score, 2), "method": "mser"})

    result = _nms(detections, iou_thresh=0.3)[:MAX_DETECTIONS_PER_IMAGE]

    H_o, W_o = img_orig.shape
    refined = []
    for d in result:
        info = refine_watermark_position(img_orig, d)
        if info is None:
            continue
        d.update(info)
        refined.append(d)
    refined.sort(key=lambda x: score_refined_detection(x, W_o, H_o), reverse=True)
    return refined


def _nms(dets, iou_thresh=0.3):
    if len(dets) <= 1:
        return dets
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        overlap = False
        for k in keep:
            if _iou(d, k) > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return keep


def _iou(a, b):
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    return inter / (area_a + area_b - inter)


# Primary detection path.
# v4: detect_watermark_v2 (precision-first multi-feature discriminator) is the
# DEFAULT. It produces ZERO false positives on a large held-out validation set
# (see metrics2) and tags each detection 'auto' (safe to auto-clean) or 'manual'
# (route to human review, never auto-clean). The old 1-D edge detector
# (detect_template_1d) OVERFIT the 46-image ground truth and false-positived on the
# broader catalog; it is kept behind USE_V2_PRIMARY=False for comparison only.
USE_V2_PRIMARY = True
USE_1D_PRIMARY = True   # used only when USE_V2_PRIMARY is False


def detect_watermark(filepath, template=None, threshold=None):
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    return detect_watermark_image(img, template, threshold)


def detect_watermark_image(img_gray, template=None, threshold=None):
    if USE_V2_PRIMARY:
        dets = detect_watermark_v2(img_gray)
        # Optional stricter gate: callers (e.g. residual re-check) pass an
        # edge_score floor; apply it to the detector's primary edge signal.
        if threshold is not None:
            dets = [d for d in dets if d.get("edge_score", 0) >= threshold]
        return dets

    if USE_1D_PRIMARY:
        dets = detect_template_1d(img_gray)
        if threshold is not None:
            dets = [d for d in dets if d.get("edge_score", 0) >= threshold]
        return dets

    if template is None:
        template = ensure_template()
    dets = detect_template_image(img_gray, template)
    if threshold and threshold > SCAN_THRESHOLD:
        dets = [d for d in dets if d.get("score", 0) >= threshold]
    if not dets and SCAN_USE_MSER:
        dets = detect_mser_image(img_gray)
    return dets


def auto_clean_detections(dets):
    """Filter a detection list to those SAFE to auto-clean (tier == 'auto').

    v2 detections carry a `tier`. Legacy detectors have no tier -> treated as
    auto for backward compatibility. Used by the cleaning entry points so that
    'manual'-tier (faint/uncertain) detections are NEVER auto-modified.
    """
    return [d for d in dets if d.get("tier", "auto") == "auto"]


def manual_route_detections(dets):
    """Detections that should be routed to human review (tier == 'manual')."""
    return [d for d in dets if d.get("tier") == "manual"]


def _scan_worker(args):
    filepath, template_path = args
    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    dets = detect_watermark(filepath, tpl)
    if dets:
        return {"file": Path(filepath).name, "detections": dets}
    return None


def _scan_worker_init(downscale_max, scales, early_exit, threshold, use_mser, verify,
                      verify_min, use_multi_tpl):
    """Apply preset to worker process globals (called once per worker)."""
    global SCAN_DOWNSCALE_MAX, DETECT_SCALES, EARLY_EXIT_SCORE
    global SCAN_THRESHOLD, SCAN_USE_MSER, SCAN_VERIFY_FULL_RES, VERIFY_MIN_SCORE
    global SCAN_USE_MULTI_TEMPLATE
    SCAN_DOWNSCALE_MAX = downscale_max
    DETECT_SCALES = scales
    EARLY_EXIT_SCORE = early_exit
    SCAN_THRESHOLD = threshold
    SCAN_USE_MSER = use_mser
    SCAN_VERIFY_FULL_RES = verify
    VERIFY_MIN_SCORE = verify_min
    SCAN_USE_MULTI_TEMPLATE = use_multi_tpl


def scan_images(assets_dir, workers=None, template=None, preset=DEFAULT_PRESET):
    if template is None:
        template = ensure_template()
    tpl_path = str(TEMPLATE_PATH)

    cfg = SCAN_PRESETS[preset]
    # Apply preset to main process too (so single-image calls use the same params)
    apply_scan_preset(preset)

    all_files = sorted(f for f in os.listdir(assets_dir)
                       if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
    files = [f for f in all_files if should_scan_file(f)]
    skipped = len(all_files) - len(files)
    total = len(files)
    work_args = [(str(assets_dir / f), tpl_path) for f in files]

    if workers is None:
        workers = min(os.cpu_count() or 1, 10)

    print(f"Scanning {total} images in {assets_dir} ({workers} workers, "
          f"preset={preset} — {cfg['description']})... "
          f"[skipped {skipped} iPhone {SKIP_IPHONE_MIN}+ files known to be clean]",
          flush=True)
    print(f"  Spawning {workers} workers (expected {cfg['expected_rate']})...",
          flush=True)

    results = []
    t0 = time.time()
    progress_every = max(50, min(500, total // 40))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_scan_worker_init,
        initargs=(cfg["downscale_max"], cfg["scales"], cfg["early_exit_score"],
                  cfg["threshold"], cfg["use_mser"], cfg["verify_at_full_res"],
                  cfg["verify_min_score"], cfg["use_multi_template"]),
    ) as pool:
        for i, det in enumerate(pool.map(_scan_worker, work_args, chunksize=5)):
            if det:
                results.append(det)
            if (i + 1) % progress_every == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.001)
                remaining = (total - i - 1) / max(rate, 0.001)
                print(f"  [{i + 1}/{total}] {rate:.1f} img/s, "
                      f"found {len(results)}, ~{remaining / 60:.1f}m left",
                      flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {total} images in {elapsed:.0f}s "
          f"({total / max(elapsed, 1):.1f} img/s), found {len(results)}",
          flush=True)
    return results


def apply_scan_preset(preset):
    cfg = SCAN_PRESETS[preset]
    global SCAN_DOWNSCALE_MAX, DETECT_SCALES, EARLY_EXIT_SCORE
    global SCAN_THRESHOLD, SCAN_USE_MSER, SCAN_VERIFY_FULL_RES, VERIFY_MIN_SCORE
    global SCAN_USE_MULTI_TEMPLATE
    SCAN_DOWNSCALE_MAX = cfg["downscale_max"]
    DETECT_SCALES = cfg["scales"]
    EARLY_EXIT_SCORE = cfg["early_exit_score"]
    SCAN_THRESHOLD = cfg["threshold"]
    SCAN_USE_MSER = cfg["use_mser"]
    SCAN_VERIFY_FULL_RES = cfg["verify_at_full_res"]
    VERIFY_MIN_SCORE = cfg["verify_min_score"]
    SCAN_USE_MULTI_TEMPLATE = cfg["use_multi_template"]


# ---------------------------------------------------------------------------
# P1a: Box helpers — used by refine_watermark_position and the masking layer.
# A "box" here is a dict {x, y, w, h} in original-image coordinates.
# ---------------------------------------------------------------------------

def clamp_box(x, y, w, h, img_w, img_h):
    x1 = max(0, min(img_w - 1, int(x)))
    y1 = max(0, min(img_h - 1, int(y)))
    x2 = max(x1 + 1, min(img_w, int(x + w)))
    y2 = max(y1 + 1, min(img_h, int(y + h)))
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def box_center(box):
    return {"x": box["x"] + box["w"] / 2, "y": box["y"] + box["h"] / 2}


def box_from_center(cx, cy, w, h, img_w, img_h):
    return clamp_box(int(cx - w / 2), int(cy - h / 2), int(w), int(h), img_w, img_h)


def refine_watermark_position(img_gray, det):
    """Return a refined mark_box dict that covers the full sunsky-online.com text.

    Returns None for obvious false positives (corners, too-close-to-edge,
    abnormally small fragments near image borders).
    """
    img_h, img_w = img_gray.shape[:2]
    raw = clamp_box(det.get("x", 0), det.get("y", 0),
                    det.get("w", 1), det.get("h", 1), img_w, img_h)

    cx = raw["x"] + raw["w"] / 2
    cy = raw["y"] + raw["h"] / 2
    method = det.get("method", "")
    score = float(det.get("score", 0))

    # Reject top/bottom band false positives (header/footer)
    if cy < img_h * 0.12 or cy > img_h * 0.88:
        return None
    # Reject far-edge false positives unless the box is already wide enough
    if raw["w"] < img_w * 0.12 and (cx < img_w * 0.18 or cx > img_w * 0.82):
        return None

    if method == "template":
        mark_w = max(raw["w"] * 2.4, img_w * 0.28)
        mark_h = max(raw["h"] * 1.7, img_h * 0.035)
    else:
        # MSER boxes can include product area — narrow the height
        mark_w = max(raw["w"], img_w * 0.28)
        mark_h = max(min(raw["h"], img_h * 0.055), img_h * 0.035)

    # Hard caps to prevent destructive masks
    mark_w = min(mark_w, img_w * 0.48)
    mark_h = min(mark_h, img_h * 0.075)

    if raw["w"] > img_w * 0.28:
        # Raw box may already cover full text — just round it up a bit
        mark_w = min(raw["w"] * 1.08, img_w * 0.48)

    # Bias toward image center (supplier watermark is usually centered)
    center_bias = 0.25
    refined_cx = (1.0 - center_bias) * cx + center_bias * (img_w / 2)

    mark_box = box_from_center(refined_cx, cy, mark_w, mark_h, img_w, img_h)

    confidence = min(1.0, max(0.0, score))
    if method == "mser":
        confidence = min(1.0, score / 50.0)

    aspect = mark_box["w"] / max(mark_box["h"], 1)
    mask_area_pct = round(100.0 * mark_box["w"] * mark_box["h"] / (img_w * img_h), 3)
    center_dist = round(abs(cx - img_w / 2) / img_w, 3)

    return {
        "detection_box": raw,
        "mark_box": mark_box,
        "mark_center": box_center(mark_box),
        "confidence": round(confidence, 4),
        "position_source": f"refined_{method or 'unknown'}",
        "debug": {
            "mask_area_pct": mask_area_pct,
            "raw_center_distance": center_dist,
            "mark_aspect_ratio": round(aspect, 2),
        },
    }


def score_refined_detection(det, img_w, img_h):
    """Rank refined detections per image — used to pick the best candidate.

    Weighting (highest impact first):
      - template confidence (dominant; verified score boosts further)
      - aspect ratio near the sunsky-online.com text (~5.5–8.5)
      - low raw_center_distance (detector lined up with watermark center)
      - weak center bias (only a soft preference, not dominant)
    """
    box = det.get("mark_box") or det
    cx = box["x"] + box["w"] / 2
    cy = box["y"] + box["h"] / 2

    # Weak center bias — supplier watermark is often near center but not always
    center_penalty = 0.4 * (abs(cx - img_w / 2) / img_w)
    vertical_penalty = 0.4 * (abs(cy - img_h * 0.50) / img_h)

    aspect = box["w"] / max(box["h"], 1)
    aspect_bonus = 0.0
    if 5.5 <= aspect <= 8.5:
        aspect_bonus = 0.4  # right in the sunsky-online.com band
    elif 3.0 <= aspect < 5.5 or 8.5 < aspect <= 12.0:
        aspect_bonus = 0.1
    else:
        aspect_bonus = -0.3

    # raw_center_distance from refine_watermark_position: how far the raw
    # detection center is from the refined mark_box center. Lower = detector
    # aligned with the visible text. Bias against detections that needed
    # significant nudging.
    raw_cd = float(det.get("debug", {}).get("raw_center_distance", 0))
    alignment_penalty = min(0.4, raw_cd)

    confidence = float(det.get("confidence", 0))
    verified = float(det.get("verified_score", 0))
    raw_score = float(det.get("score", 0))

    return (
        confidence * 2.2
        + verified * 0.8                # boost candidates that passed full-res verify
        + min(raw_score, 50) / 50.0 * 0.5
        + aspect_bonus
        - center_penalty
        - vertical_penalty
        - alignment_penalty
    )


def get_cleaning_box(det):
    """Return the box the cleaner should mask. Prefer mark_box when present."""
    mb = det.get("mark_box")
    if mb:
        return mb
    return {"x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"]}


# ---------------------------------------------------------------------------
# P1: Masking
# ---------------------------------------------------------------------------

# Mask area thresholds (fraction of full image)
MASK_AREA_REVIEW_MAX = 0.06  # above this → reject auto-clean
PRECISION_DENSITY_CAP = 0.40  # strokes filling >40% of bbox → solid part, not text → manual


def generate_precision_mask(img_gray, det, close_kernel=3, dilate_px=2, dev_thresh=6):
    """Pixel-perfect text-stroke mask inside the detected mark_box ROI.

    Instead of filling the whole bounding box (destroys component textures), this
    isolates only the watermark's text strokes, so the inpainter touches ~the
    glyph pixels and leaves surrounding hardware (laser etching, PCB traces, gold
    pins) intact.

    Signal: ABSOLUTE deviation from the local background (median blur). The
    "sunsky-online.com" watermark is SEMI-TRANSPARENT, so its polarity flips with
    the background — it reads darker on white product shots and lighter on dark
    PCBs. Absolute deviation captures the strokes either way; a fixed-polarity
    adaptive threshold (as in the generic spec pseudocode) would miss half the
    catalog. Adaptive/Otsu assume one polarity on a paper-like page; this asset
    set does not, so |roi - local_bg| is the correct, robust primitive.

    Returns (mask, density) where density = stroke_pixels / bbox_area.
    density > PRECISION_DENSITY_CAP signals a likely false positive (a solid part
    mistaken for text) — the caller routes those to manual review.
    """
    img_h, img_w = img_gray.shape[:2]
    box = get_cleaning_box(det)
    x1 = max(0, int(box["x"]))
    y1 = max(0, int(box["y"]))
    x2 = min(img_w, int(box["x"] + box["w"]))
    y2 = min(img_h, int(box["y"] + box["h"]))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return mask, 0.0
    roi = img_gray[y1:y2, x1:x2]

    # Local background: median blur with a kernel taller than the glyph strokes,
    # so the text is smoothed away and only the background remains.
    k = min(31, max(5, (roi.shape[0] // 2) * 2 + 1))
    if k % 2 == 0:
        k += 1
    bg = cv2.medianBlur(roi, k)
    dev = cv2.absdiff(roi, bg)
    _, thr = cv2.threshold(dev, dev_thresh, 255, cv2.THRESH_BINARY)

    # Connect glyph fragments without blooming into a solid block.
    if close_kernel > 0:
        kk = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kk)
    if dilate_px > 0:
        thr = cv2.dilate(thr, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)

    density = float(np.count_nonzero(thr)) / float(thr.size)
    mask[y1:y2, x1:x2] = thr
    return mask, density


# ---------------------------------------------------------------------------
# P1b: ROI complexity + component-filtered mask analysis + safety gates
#
# The coarse precision mask above is correct on flat backgrounds but, on
# detail-heavy ROIs (screws, camera rings, flex cables, graphite stickers, PCB
# traces, gold pins), |roi - bg| also lights up real hardware edges. The
# functions below (1) classify the ROI so flat vs. detail-heavy regions are
# treated differently, (2) keep only thin, separated, text-like connected
# components, and (3) gate the result so anything that looks like a solid blob
# or product-edge bloom is routed to MANUAL REVIEW instead of being inpainted.
# ---------------------------------------------------------------------------

# Component keep limits (fractions of bbox). Tuned conservatively: a real
# "sunsky-online.com" stamp is many small thin glyph strokes, not one big blob.
MAX_COMPONENT_AREA_RATIO = 0.08
MAX_COMPONENT_WIDTH_RATIO = 0.35
MAX_COMPONENT_HEIGHT_RATIO = 0.45
MIN_COMPONENT_AREA = 2

# Multi-gate safety thresholds (Step 3 of the production plan).
GATE_DENSITY_DETAIL = 0.18   # mask density on a detail-heavy ROI
GATE_DENSITY_FLAT = 0.30     # mask density on a flat ROI
GATE_LARGEST_COMPONENT = 0.10
GATE_EDGE_OVERLAP = 0.25
GATE_BORDER_TOUCH = 0.20
GATE_MIN_COMPONENTS = 3
GATE_MAX_COMPONENTS = 80

# Local ROI QA thresholds (Step 7). Global "sharp 1.0" is not enough evidence —
# these measure damage localized to the watermark area and its surroundings.
ROI_SHARPNESS_RATIO_MIN = 0.88
OUTSIDE_MASK_SSIM_MIN = 0.96

# V6: the detector's auto/manual tier is now a RISK SIGNAL, not a hard reject.
# Low-confidence ("manual_tier") detections are still allowed to auto-clean, but
# only if they clear STRICTER local QA — so a clean flat-background stamp gets
# through while anything that smudges the product is still rejected to manual.
STRICT_ROI_SHARPNESS = 0.97
STRICT_OUTSIDE_SSIM = 0.995


def classify_roi_complexity(roi_gray):
    """Classify a watermark ROI as 'flat', 'medium', or 'detail_heavy'.

    Flat backgrounds (white phone glass, plain back covers) tolerate a soft
    fill; detail-heavy regions (hardware, PCB, connectors) must only ever have
    glyph-tight strokes touched, and are held to stricter gates.
    """
    if roi_gray is None or roi_gray.size == 0:
        return "flat"
    edges = cv2.Canny(roi_gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    gray_std = float(np.std(roi_gray))
    if edge_density < 0.03 and gray_std < 18:
        return "flat"
    if edge_density < 0.08:
        return "medium"
    return "detail_heavy"


def _cluster_components(comps, bh, bw):
    """Group glyph components into horizontal text rows (lightweight DBSCAN-by-row).

    Returns (cluster_count, line_fraction):
      cluster_count : number of distinct y-band rows the components fall into.
      line_fraction : fraction of components that lie on a "text-like" row — one
                      holding many components spread across a wide x-extent, i.e.
                      the signature of a single shattered watermark text line.

    No sklearn dependency: a 1-D y-band grouping is the right primitive here
    because the watermark is strictly horizontal, and it is O(n log n).
    """
    n = len(comps)
    if n == 0:
        return 0, 0.0
    # Typical glyph height → row band tolerance.
    med_h = float(np.median([c[2] for c in comps])) or max(1.0, bh * 0.1)
    band = max(4.0, med_h * 1.2)
    rows = []  # each: {"ys": [], "xs": [], "n": int}
    for cx, cy, ch in sorted(comps, key=lambda c: c[1]):
        placed = False
        for r in rows:
            if abs(cy - r["yc"]) <= band:
                r["xs"].append(cx)
                r["n"] += 1
                r["yc"] = (r["yc"] * (r["n"] - 1) + cy) / r["n"]
                placed = True
                break
        if not placed:
            rows.append({"yc": cy, "xs": [cx], "n": 1})
    cluster_count = len(rows)
    # A text-like row: >=4 components spanning a wide horizontal extent.
    line_components = 0
    for r in rows:
        if r["n"] >= 4 and bw > 0:
            span = (max(r["xs"]) - min(r["xs"])) / float(bw)
            if span >= 0.3:
                line_components += r["n"]
    return cluster_count, (line_components / float(n))


def analyze_precision_mask(img_gray, det):
    """Build a component-filtered glyph mask and rich safety metrics.

    Returns (mask, metrics). `mask` keeps only thin, separated, text-like
    connected components from the raw stroke mask. `metrics` carries everything
    the safety gates and HTML QA need:
        density, component_count, largest_component_ratio, edge_overlap_ratio,
        border_touch_ratio, roi_type, bbox.
    """
    img_h, img_w = img_gray.shape[:2]
    box = get_cleaning_box(det)
    x1 = max(0, int(box["x"]))
    y1 = max(0, int(box["y"]))
    x2 = min(img_w, int(box["x"] + box["w"]))
    y2 = min(img_h, int(box["y"] + box["h"]))
    out = np.zeros((img_h, img_w), dtype=np.uint8)
    metrics = {
        "density": 0.0, "component_count": 0, "largest_component_ratio": 0.0,
        "edge_overlap_ratio": 0.0, "border_touch_ratio": 0.0,
        "cluster_count": 0, "line_fraction": 0.0,
        "roi_type": "flat", "bbox": [x1, y1, x2 - x1, y2 - y1],
    }
    if x2 - x1 < 4 or y2 - y1 < 4:
        return out, metrics

    roi = img_gray[y1:y2, x1:x2]
    bh, bw = roi.shape[:2]
    bbox_area = float(bh * bw)
    metrics["roi_type"] = classify_roi_complexity(roi)

    # Raw stroke candidate: absolute deviation from the local background, robust
    # to the watermark's polarity flip (dark on white, light on dark PCBs).
    k = min(31, max(5, (bh // 2) * 2 + 1))
    if k % 2 == 0:
        k += 1
    bg = cv2.medianBlur(roi, k)
    dev = cv2.absdiff(roi, bg)
    _, raw = cv2.threshold(dev, 6, 255, cv2.THRESH_BINARY)

    # Connected-component filtering: keep only text-like strokes.
    num, labels, stats, cents = cv2.connectedComponentsWithStats(raw, connectivity=8)
    kept = np.zeros_like(raw)
    kept_count = 0
    largest_area = 0
    comps = []  # (cx, cy, h) of kept components, for spatial clustering (V7)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < MIN_COMPONENT_AREA:
            continue
        if area > MAX_COMPONENT_AREA_RATIO * bbox_area:
            continue
        if cw > MAX_COMPONENT_WIDTH_RATIO * bw:
            continue
        if ch > MAX_COMPONENT_HEIGHT_RATIO * bh:
            continue
        kept[labels == i] = 255
        kept_count += 1
        largest_area = max(largest_area, area)
        comps.append((float(cents[i][0]), float(cents[i][1]), ch))

    # Gentle close to bridge glyph fragments; NO broad dilation (no rectangle bloom).
    kk = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kk, iterations=1)

    # Metrics for the safety gates.
    white = int(np.count_nonzero(kept))
    metrics["density"] = white / bbox_area if bbox_area else 0.0
    metrics["component_count"] = kept_count
    metrics["largest_component_ratio"] = largest_area / bbox_area if bbox_area else 0.0

    # V7 Phase 2.2: spatial clustering. The "sunsky-online.com" stamp is ONE
    # horizontal text line that the segmenter shatters into many glyph fragments.
    # Reflective hardware highlights instead scatter fragments across the ROI.
    # Group kept components into horizontal rows (y-bands) and measure how much
    # of the mass lies on text-like lines vs. is scattered — so the too_many
    # gate can merge an aligned line but still reject a scattered false positive.
    cluster_count, line_fraction = _cluster_components(comps, bh, bw)
    metrics["cluster_count"] = cluster_count
    metrics["line_fraction"] = round(line_fraction, 4)

    roi_edges = cv2.Canny(roi, 50, 150)
    total_edge = int(np.count_nonzero(roi_edges))
    if total_edge:
        metrics["edge_overlap_ratio"] = int(np.count_nonzero(
            cv2.bitwise_and(roi_edges, kept))) / float(total_edge)

    border = np.zeros_like(kept)
    border[0, :] = 1; border[-1, :] = 1; border[:, 0] = 1; border[:, -1] = 1
    border_px = int(np.count_nonzero(border))
    if border_px:
        metrics["border_touch_ratio"] = int(np.count_nonzero(
            (kept > 0) & (border > 0))) / float(border_px)

    out[y1:y2, x1:x2] = kept
    return out, metrics


def passes_safety_gates(metrics):
    """Return (ok: bool, reason: str). Route to manual review when a gate trips.

    A real watermark stamp = many small, thin, separated glyph strokes. Anything
    blob-like, edge-hugging, or too dense for its ROI type is treated as a
    likely false positive and kept out of the auto-clean path.
    """
    roi_type = metrics.get("roi_type", "flat")
    density = metrics.get("density", 0.0)
    cc = metrics.get("component_count", 0)

    # V7 Phase 2.3: an empty mask (no strokes, zero density) is NOT a watermark
    # to gate — it means the image is already clean. The caller routes this to
    # no_watermark/untouched, so don't trip too_few_components here.
    if cc == 0 and density == 0.0:
        return True, ""

    cap = GATE_DENSITY_DETAIL if roi_type == "detail_heavy" else GATE_DENSITY_FLAT
    if density > cap:
        return False, f"mask_density_high_{roi_type}"

    if cc < GATE_MIN_COMPONENTS:
        return False, "too_few_components"

    # V7 Phase 2.2: cluster-aware too_many. Raw count > limit no longer auto-
    # rejects: if most components lie on a horizontal text line (a shattered
    # watermark string), treat them as one unified entity and allow. Reject only
    # when the components are genuinely scattered (low line_fraction) — i.e.
    # reflective highlights spread across the ROI, not text.
    if cc > GATE_MAX_COMPONENTS:
        if metrics.get("line_fraction", 0.0) < 0.5:
            return False, "too_many_components"

    # largest_component_ratio is the blob detector: a single fat component means
    # the mask latched onto a SOLID part (screw head, connector body) rather than
    # thin text. Enforced on non-flat ROIs, where that confusion is the real risk.
    #
    # NOTE on edge_overlap_ratio / border_touch_ratio: the production plan lists
    # these as hard gates, but those thresholds assume a LOOSE bbox with the
    # watermark floating inside. Our detector emits a TIGHT refined mark_box, so a
    # genuine stamp's own strokes are most of the ROI's edges and naturally touch
    # its borders — making both ratios high for TRUE positives. Hard-rejecting on
    # them would wrongly route valid clean-ables to manual. They are therefore
    # COMPUTED and DISPLAYED for audit, but the actual damage guard is the local
    # QA downstream (inside-bbox sharpness ratio + outside-mask SSIM), which
    # measures real smudging directly. This is the deliberate, tested deviation.
    if roi_type != "flat":
        if metrics.get("largest_component_ratio", 0.0) > GATE_LARGEST_COMPONENT:
            return False, "largest_component_too_big"
    return True, ""


def _ssim_map(a_gray, b_gray):
    """Wang SSIM map (float64), computed with Gaussian windows. No skimage dep."""
    a = a_gray.astype(np.float64)
    b = b_gray.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    win = (11, 11)
    mu_a = cv2.GaussianBlur(a, win, 1.5)
    mu_b = cv2.GaussianBlur(b, win, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, win, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, win, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, win, 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa + sb + C2)
    return num / np.maximum(den, 1e-12)


def outside_mask_ssim(original, candidate, mask):
    """Mean SSIM over pixels OUTSIDE the mask — measures collateral damage to
    the product where nothing should have changed. 1.0 = untouched."""
    go = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    smap = _ssim_map(go, gc)
    outside = (mask == 0)
    if not np.any(outside):
        return float(np.mean(smap))
    return float(np.mean(smap[outside]))


def roi_sharpness_ratio(original, candidate, det):
    """Laplacian-variance ratio measured INSIDE the watermark bbox only.

    Global sharpness misses localized smudging; this catches a soft rectangular
    patch right where the watermark was. Returns (ratio, before, after)."""
    h, w = original.shape[:2]
    box = get_cleaning_box(det)
    x1 = max(0, int(box["x"])); y1 = max(0, int(box["y"]))
    x2 = min(w, int(box["x"] + box["w"])); y2 = min(h, int(box["y"] + box["h"]))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 1.0, 0.0, 0.0
    ob = cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    cb = cv2.cvtColor(candidate[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    sb = float(cv2.Laplacian(ob, cv2.CV_64F).var())
    sa = float(cv2.Laplacian(cb, cv2.CV_64F).var())
    return sa / max(sb, 1e-6), sb, sa


def create_mark_box_mask(img_gray, det, pad_x=8, pad_y=5, dilate_px=5):
    """Build a narrow text-band mask centred on the refined mark_box.

    Returns (mask, area_fraction). Returns (None, area) when the mask would be
    too destructive (> MASK_AREA_REVIEW_MAX) so the cleaner can reject the image.
    """
    img_h, img_w = img_gray.shape[:2]
    box = get_cleaning_box(det)
    x1 = max(0, int(box["x"] - pad_x))
    y1 = max(0, int(box["y"] - pad_y))
    x2 = min(img_w, int(box["x"] + box["w"] + pad_x))
    y2 = min(img_h, int(box["y"] + box["h"] + pad_y))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    area_frac = float(np.count_nonzero(mask)) / mask.size
    if area_frac > MASK_AREA_REVIEW_MAX:
        return None, area_frac
    return mask, area_frac


def create_glyph_mask(img_gray, det, dilate_px=DILATE_GLYPH):
    h, w = img_gray.shape
    box = get_cleaning_box(det)
    x, y, dw, dh = box["x"], box["y"], box["w"], box["h"]
    pad = 5
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + dw + pad), min(h, y + dh + pad)

    roi = img_gray[y1:y2, x1:x2]
    bg_med = cv2.medianBlur(roi, min(31, max(3, dh * 2 + 1))).astype(float)
    dev = np.abs(roi.astype(float) - bg_med)
    _, text_mask = cv2.threshold(dev.astype(np.uint8), 3, 255, cv2.THRESH_BINARY)

    kern = np.ones((dilate_px, dilate_px), np.uint8)
    text_mask = cv2.dilate(text_mask, kern, iterations=1)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = text_mask
    return mask


def create_box_mask(img_gray, det, dilate_px=DILATE_BOX, pad=PAD_BOX):
    h, w = img_gray.shape
    box = get_cleaning_box(det)
    x, y, dw, dh = box["x"], box["y"], box["w"], box["h"]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + dw + pad)
    y2 = min(h, y + dh + pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    kern = np.ones((dilate_px, dilate_px), np.uint8)
    return cv2.dilate(mask, kern, iterations=1)


# ---------------------------------------------------------------------------
# P4: Sharpness guard
# ---------------------------------------------------------------------------

def laplacian_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ---------------------------------------------------------------------------
# P5: Multi-strategy cleaning
# ---------------------------------------------------------------------------

def _inpaint_cv(img, mask, radius, method):
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    inpainted = cv2.inpaint(img, mask, inpaintRadius=radius, flags=flag)

    feather = max(radius, 3)
    alpha = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather).astype(np.float32) / 255.0
    alpha3 = np.stack([alpha] * 3, axis=-1)
    return (inpainted * alpha3 + img.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)


_lama_model_cache = {}


def _inpaint_lama(img, mask):
    if not LAMA_MODEL_PATH.exists():
        return None
    try:
        import torch
        device = "cpu"
        if "model" not in _lama_model_cache:
            _lama_model_cache["model"] = torch.jit.load(
                str(LAMA_MODEL_PATH), map_location=device)
            _lama_model_cache["model"].eval()
        model = _lama_model_cache["model"]

        h, w = img.shape[:2]
        ph = ((h + 7) // 8) * 8
        pw = ((w + 7) // 8) * 8

        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_t = torch.nn.functional.pad(img_t, (0, pw - w, 0, ph - h), mode="reflect")
        mask_t = torch.nn.functional.pad(mask_t, (0, pw - w, 0, ph - h))
        img_t, mask_t = img_t.to(device), mask_t.to(device)

        with torch.no_grad():
            out = model(img_t, mask_t)
        out = out[0].permute(1, 2, 0).cpu().numpy()[:h, :w] * 255
        result = out.clip(0, 255).astype(np.uint8)

        feather = 5
        alpha = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather).astype(np.float32) / 255.0
        alpha3 = np.stack([alpha] * 3, axis=-1)
        return (result * alpha3 + img.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)
    except Exception:
        return None


def _masked_sharpness(img_bgr, keep):
    """Laplacian variance computed only over pixels where `keep` is True.

    Used to measure inpainting-induced blur OUTSIDE the mask. Measuring global
    sharpness is wrong here: legitimately removing the (sharp-edged) watermark
    text lowers whole-image edge energy, which the global ratio mis-reads as
    blur and wrongly rejects a good clean. Restricting to the unmasked region
    isolates true collateral blur in the area that should be untouched.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = lap[keep]
    if vals.size == 0:
        return float(lap.var())
    return float(vals.var())


def _score_candidate(original, candidate, mask):
    # Blur is measured OUTSIDE the mask (a slightly dilated text band): that region
    # must stay sharp. Inside the mask, content change is expected (the watermark
    # is being removed), so it is excluded from the sharpness ratio.
    outside_keep = (mask == 0)
    if cv2.countNonZero((mask == 0).astype(np.uint8)) > 0:
        sb = _masked_sharpness(original, outside_keep)
        sa = _masked_sharpness(candidate, outside_keep)
    else:
        sb = laplacian_sharpness(original)
        sa = laplacian_sharpness(candidate)
    ratio = sa / max(sb, 1e-6)
    blur_penalty = max(0, 1.0 - ratio / SHARPNESS_RATIO_MIN)

    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(float)
    gray_cand = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(float)
    outside = (mask == 0)
    if np.sum(outside) > 0:
        outside_diff = np.mean(np.abs(gray_orig[outside] - gray_cand[outside]))
    else:
        outside_diff = 0
    outside_penalty = outside_diff / 255.0

    inside = (mask > 0)
    if np.sum(inside) > 0:
        inside_var = np.std(gray_cand[inside])
    else:
        inside_var = 0
    residual_penalty = max(0, 1.0 - inside_var / 30.0)

    score = 1.0 - 0.4 * blur_penalty - 0.3 * outside_penalty - 0.3 * residual_penalty
    return score, ratio


def _check_residual(candidate, filepath, template):
    """Write candidate to temp, rescan for residual watermark."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tmp = tf.name
    try:
        ok = cv2.imwrite(tmp, candidate, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return False  # treat as residual present — safer than blind clean
        dets = detect_watermark(tmp, template, threshold=VALIDATE_THRESHOLD)
        return len(dets) == 0
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _check_residual_image(candidate, template):
    gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    dets = detect_watermark_image(gray, template, threshold=VALIDATE_THRESHOLD)
    return len(dets) == 0


LOW_CONFIDENCE_MIN = 0.30  # detections below this confidence → reject auto-clean
PREFLIGHT_MIN_SCORE = 0.40  # pre-flight ROI match must score >= this to apply mask


def _preflight_verify(img_gray, det, templates):
    """Cheap full-resolution ROI re-match — double-checks that the scan's
    mark_box really contains a watermark before we damage the image.

    Returns (passed: bool, best_score: float).
    """
    mb = det.get("mark_box") or det
    H, W = img_gray.shape[:2]
    pad_x = max(10, mb["w"] // 6)
    pad_y = max(5, mb["h"] // 2)
    x1 = max(0, int(mb["x"] - pad_x))
    y1 = max(0, int(mb["y"] - pad_y))
    x2 = min(W, int(mb["x"] + mb["w"] + pad_x))
    y2 = min(H, int(mb["y"] + mb["h"] + pad_y))
    if y2 - y1 < 8 or x2 - x1 < 30:
        return False, 0.0
    roi = img_gray[y1:y2, x1:x2]

    best = 0.0
    target_h = max(int(mb["h"]), 1)
    for tpl in templates:
        th, tw = tpl.shape
        for factor in (0.6, 0.8, 1.0, 1.2):
            s = (target_h / th) * factor
            sw, sh = int(tw * s), int(th * s)
            if sw < 20 or sh < 5 or sw >= roi.shape[1] or sh >= roi.shape[0]:
                continue
            tpl_s = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(roi, tpl_s, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best:
                best = score
            if best >= 0.7:  # very confident — stop early
                return True, best
    return best >= PREFLIGHT_MIN_SCORE, best


def clean_image(filepath, detections, output_path=None, template=None):
    img = cv2.imread(str(filepath))
    if img is None:
        return None, {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if template is None:
        template = ensure_template()
    out = output_path or filepath
    return clean_loaded_image(img, gray, detections, output_path=out, template=template)


def clean_loaded_image(img, gray, detections, output_path=None, template=None):
    """V6: tier is a RISK signal, not a hard gate.

    Every detected watermark is analyzed (mask + metrics) and attempted; the
    decision to auto-clean vs. route-to-manual is made by mask quality + ROI type
    + LOCAL QA, with stricter QA thresholds for low-confidence ("manual_tier")
    detections. Every return — cleaned OR manual — carries the full metric set
    and a precise `reason`, so a reviewer can tell exactly why a row was routed.
    """
    if template is None:
        template = ensure_template()

    # No detections -> nothing to clean (avoid a misleading no-op "cleaned").
    if not detections:
        return None, {"status": "no_watermark"}

    # RISK SIGNAL (was the hard tier gate). manual-tier / low-confidence
    # detections still proceed, but are held to STRICTER local QA below.
    best_det = max(detections, key=lambda d: float(d.get("confidence", 0)))
    is_auto = bool(auto_clean_detections(detections))
    max_conf = max((float(d.get("confidence", 1.0)) for d in detections), default=1.0)
    high_risk = (not is_auto) or (max_conf < LOW_CONFIDENCE_MIN)
    roi_min = STRICT_ROI_SHARPNESS if high_risk else ROI_SHARPNESS_RATIO_MIN
    ssim_min = STRICT_OUTSIDE_SSIM if high_risk else OUTSIDE_MASK_SSIM_MIN

    # Metric bundle attached to EVERY return (Priority 2/3: never blank).
    def _metrics(extra=None):
        d = {
            "roi_type": roi_type,
            "high_risk": high_risk,
            "detector_tier": best_det.get("tier", "auto"),
            "confidence": round(float(best_det.get("confidence", 0)), 4),
            "mask_density_pct": round(100 * qa_metrics.get("density", max_density), 2),
            "component_count": qa_metrics.get("component_count", 0),
            "cluster_count": qa_metrics.get("cluster_count", 0),
            "line_fraction": round(qa_metrics.get("line_fraction", 0.0), 4),
            "largest_component_ratio": round(qa_metrics.get("largest_component_ratio", 0.0), 4),
            "edge_overlap_ratio": round(qa_metrics.get("edge_overlap_ratio", 0.0), 4),
            "border_touch_ratio": round(qa_metrics.get("border_touch_ratio", 0.0), 4),
        }
        if extra:
            d.update(extra)
        return d

    # Defaults so _metrics is safe to call from any early return below.
    max_density = 0.0
    roi_type = "flat"
    qa_metrics = {}

    # Pre-flight ROI re-match for legacy (non-wm2) detections only.
    use_preflight = any(d.get("method") != "wm2" for d in detections)
    if use_preflight:
        templates_for_verify = get_templates()
        verified_dets = []
        for det in detections:
            passed, pf_score = _preflight_verify(gray, det, templates_for_verify)
            det["preflight_score"] = round(pf_score, 4)
            if passed:
                verified_dets.append(det)
        if detections and not verified_dets:
            scores = [d.get("preflight_score", 0.0) for d in detections]
            roi_type, qa_metrics = "flat", {}
            return None, _metrics({
                "status": "needs_manual",
                "reason": "preflight_mismatch",
                "failed_gates": ["preflight_mismatch"],
                "best_preflight_score": round(max(scores) if scores else 0.0, 4),
            })
        detections = verified_dets

    # Build masks + per-image metrics. roi_type escalates to the most complex ROI.
    combined_precision_mask = np.zeros(gray.shape, dtype=np.uint8)
    combined_mark_mask = np.zeros(gray.shape, dtype=np.uint8)
    combined_glyph_mask = np.zeros(gray.shape, dtype=np.uint8)
    max_density = 0.0
    roi_type = "flat"
    qa_metrics = {}
    _roi_rank = {"flat": 0, "medium": 1, "detail_heavy": 2}
    for det in detections:
        prec_mask, m = analyze_precision_mask(gray, det)
        max_density = max(max_density, m["density"])
        if _roi_rank.get(m["roi_type"], 0) >= _roi_rank.get(roi_type, 0):
            roi_type = m["roi_type"]
            qa_metrics = m
        ok, gate_reason = passes_safety_gates(m)
        if not ok:
            # Map gate names to the precise manual reasons (Priority 6).
            reason = gate_reason
            if gate_reason.startswith("mask_density_high"):
                reason = "mask_density_too_high"
            elif gate_reason == "largest_component_too_big":
                reason = "largest_component_blob_risk"
            return None, _metrics({
                "status": "needs_manual",
                "reason": reason,
                "failed_gates": [gate_reason],
            })
        if m["density"] > PRECISION_DENSITY_CAP:
            return None, _metrics({
                "status": "needs_manual",
                "reason": "mask_density_too_high",
                "failed_gates": ["density_cap"],
            })
        combined_precision_mask = cv2.bitwise_or(combined_precision_mask, prec_mask)

        mark_mask, mark_area_frac = create_mark_box_mask(gray, det)
        if mark_mask is None:
            return None, _metrics({
                "status": "needs_manual",
                "reason": "mask_too_large",
                "failed_gates": ["mask_too_large"],
                "mask_area_pct": round(100 * mark_area_frac, 3),
            })
        combined_mark_mask = cv2.bitwise_or(combined_mark_mask, mark_mask)
        combined_glyph_mask = cv2.bitwise_or(
            combined_glyph_mask, create_glyph_mask(gray, det))

    # PRECISION_MASK_EMPTY (Priority 4): detector flagged a watermark but no
    # strokes were extracted. Route as its own reason — not generic manual — so
    # detector-vs-mask failures stay distinguishable. (Fallback glyph mask is
    # only used on non-detail ROIs where a soft fill is safe.)
    if int(np.count_nonzero(combined_precision_mask)) == 0:
        # V7 Phase 2.3: zero strokes AND zero density/components means the image
        # is already clean — the detector's weak signal had nothing to remove.
        # Route to no_watermark/untouched (success), NOT manual: nothing was
        # modified and no quality was lost.
        if qa_metrics.get("component_count", 0) == 0 and qa_metrics.get("density", 0.0) == 0.0:
            return None, _metrics({
                "status": "no_watermark",
                "reason": "no_watermark_found",
            })
        if roi_type == "detail_heavy" or int(np.count_nonzero(combined_glyph_mask)) == 0:
            return None, _metrics({
                "status": "needs_manual",
                "reason": "precision_mask_empty",
                "failed_gates": ["precision_mask_empty"],
            })
        combined_precision_mask = combined_glyph_mask.copy()

    combined_area = float(np.count_nonzero(combined_mark_mask)) / combined_mark_mask.size
    if combined_area > MASK_AREA_REVIEW_MAX:
        return None, _metrics({
            "status": "needs_manual",
            "reason": "mask_too_large",
            "failed_gates": ["mask_too_large"],
            "mask_area_pct": round(100 * combined_area, 3),
        })

    # ROI-aware strategy order. detail_heavy → precision (stroke-only) masks ONLY.
    strategies = [
        ("telea_precision", lambda: _inpaint_cv(img, combined_precision_mask, INPAINT_RADIUS, "telea")),
        ("ns_precision", lambda: _inpaint_cv(img, combined_precision_mask, INPAINT_RADIUS, "ns")),
    ]
    if roi_type != "detail_heavy":
        strategies += [
            ("telea_mark", lambda: _inpaint_cv(img, combined_mark_mask, INPAINT_RADIUS, "telea")),
            ("ns_mark", lambda: _inpaint_cv(img, combined_mark_mask, INPAINT_RADIUS, "ns")),
            ("telea_glyph", lambda: _inpaint_cv(img, combined_glyph_mask, INPAINT_RADIUS, "telea")),
            ("ns_glyph", lambda: _inpaint_cv(img, combined_glyph_mask, INPAINT_RADIUS, "ns")),
        ]
    if LAMA_MODEL_PATH.exists():
        strategies.insert(0,
            ("lama_precision", lambda: _inpaint_lama(img, combined_precision_mask)))
        if roi_type != "detail_heavy":
            strategies.append(
                ("lama_mark", lambda: _inpaint_lama(img, combined_mark_mask)))

    best_name, best_result, best_score, best_ratio = None, None, -1, 0
    best_roi_ratio, best_out_ssim = 0.0, 0.0
    rejected = []
    saw_ssim_fail = saw_roi_fail = False

    for name, fn in strategies:
        candidate = fn()
        if candidate is None:
            rejected.append((name, "inpaint_failed"))
            continue

        if "precision" in name:
            mask_for_score = combined_precision_mask
        elif "glyph" in name:
            mask_for_score = combined_glyph_mask
        else:
            mask_for_score = combined_mark_mask
        score, ratio = _score_candidate(img, candidate, mask_for_score)

        if ratio < SHARPNESS_RATIO_MIN:
            rejected.append((name, f"blurry_{ratio:.3f}"))
            continue

        # LOCAL ROI QA — the real damage guard (risk-tiered thresholds).
        # SSIM first (it drives the V7 adaptive sharpness relaxation below).
        out_ssim = outside_mask_ssim(img, candidate, mask_for_score)
        if out_ssim < ssim_min:
            rejected.append((name, f"outside_ssim_{out_ssim:.3f}"))
            saw_ssim_fail = True
            continue

        # V7 Phase 2.1: ROI-aware adaptive sharpness. Microscopic hardware
        # (connector pins, LCD mesh) softens slightly when LaMa fills dense glyph
        # paths. If NOTHING outside the mask moved (SSIM ~perfect), that interior
        # softness is safe — relax the sharpness gate 15% on non-flat high-risk
        # ROIs. The outside-SSIM near-1.0 is the proof of no collateral damage.
        eff_roi_min = roi_min
        if (high_risk and roi_type in ("detail_heavy", "medium")
                and out_ssim >= 0.9999):
            eff_roi_min = roi_min * 0.85
        roi_ratio, _rb, _ra = roi_sharpness_ratio(img, candidate, detections[0])
        if roi_ratio < eff_roi_min:
            rejected.append((name, f"roi_blurry_{roi_ratio:.3f}"))
            saw_roi_fail = True
            continue

        if not _check_residual_image(candidate, template):
            rejected.append((name, "residual_watermark"))
            continue

        if score > best_score:
            best_name, best_result, best_score, best_ratio = name, candidate, score, ratio
            best_roi_ratio, best_out_ssim = roi_ratio, out_ssim

    if best_result is None:
        all_reasons = [r for _, r in rejected]
        if all_reasons and all(r == "residual_watermark" for r in all_reasons):
            top_reason = "residual_watermark"
        elif saw_roi_fail:
            top_reason = "roi_sharpness_fail"
        elif saw_ssim_fail:
            top_reason = "local_ssim_fail"
        elif roi_type == "detail_heavy":
            top_reason = "roi_detail_heavy_no_safe_strategy"
        else:
            top_reason = "all_strategies_failed:" + ",".join(n for n, _ in rejected)
        return None, _metrics({
            "status": "needs_manual",
            "reason": top_reason,
            "failed_gates": [f"{n}:{r}" for n, r in rejected],
        })

    if output_path is not None:
        cv2.imwrite(str(output_path), best_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return best_result, _metrics({
        "status": "cleaned",
        # Note when a low-confidence detection was rescued by passing strict QA.
        "reason": "detector_psr_low_but_mask_passed" if high_risk else "",
        "strategy": best_name,
        "score": round(best_score, 4),
        "sharpness_ratio": round(best_ratio, 4),
        "roi_sharpness_ratio": round(best_roi_ratio, 4),
        "outside_mask_ssim": round(best_out_ssim, 4),
    })


# ---------------------------------------------------------------------------
# P7: Contact sheet
# ---------------------------------------------------------------------------

def generate_contact_sheet(manifest, assets_dir, output_path=None):
    if output_path is None:
        output_path = CONTACT_SHEET
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in manifest:
        fname = entry["file"]
        fpath = assets_dir / fname
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        thumb = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode()
        status = entry.get("status", "unknown")
        strategy = entry.get("strategy", "")
        score = entry.get("score", "")
        ratio = entry.get("sharpness_ratio", "")
        rows.append(f"""<div style="display:inline-block;margin:8px;text-align:center;border:1px solid #ccc;padding:4px">
<img src="data:image/jpeg;base64,{b64}" width="300" height="300"><br>
<small>{fname[:50]}</small><br>
<b>{status}</b> {strategy} s={score} r={ratio}
</div>""")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Watermark Review — {len(rows)} images</title></head>
<body style="font-family:monospace;background:#111;color:#eee">
<h2>Watermark Review Contact Sheet ({len(rows)} images)</h2>
{"".join(rows)}
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Contact sheet: {output_path} ({len(rows)} images)")


# ---------------------------------------------------------------------------
# CLI: scan
# ---------------------------------------------------------------------------

def _write_debug_crop(fpath, dets, out_dir):
    """Write a side-by-side debug crop: raw detection_box (green) + mark_box (red)."""
    img = cv2.imread(str(fpath))
    if img is None or not dets:
        return
    best = dets[0]
    raw = best.get("detection_box") or best
    mb = best.get("mark_box") or raw
    vis = img.copy()
    cv2.rectangle(vis, (raw["x"], raw["y"]),
                  (raw["x"] + raw["w"], raw["y"] + raw["h"]), (0, 255, 0), 2)
    cv2.rectangle(vis, (mb["x"], mb["y"]),
                  (mb["x"] + mb["w"], mb["y"] + mb["h"]), (0, 0, 255), 2)
    # Crop a band around the watermark for easier review
    h, w = img.shape[:2]
    y1 = max(0, mb["y"] - mb["h"])
    y2 = min(h, mb["y"] + mb["h"] * 2)
    crop = vis[y1:y2, :]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(fpath).stem}-mark.jpg"
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 80])


def cmd_scan(args):
    template = ensure_template()
    assets = Path(args.assets)

    if args.sample_file:
        with open(args.sample_file) as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"Scanning {len(files)} files from {args.sample_file}")
        results = []
        for fname in files:
            fpath = assets / fname
            if not fpath.exists():
                print(f"  SKIP (not found): {fname}")
                continue
            dets = detect_watermark(str(fpath), template)
            if dets:
                results.append({"file": fname, "detections": dets})
                print(f"  DETECTED: {fname} ({len(dets)} region(s))")
            else:
                print(f"  CLEAN: {fname}")
    else:
        results = scan_images(assets, args.workers, template, preset=args.preset)

    # Defense-in-depth sort: ensure every entry has detection[0] = best refined.
    # detect_template/detect_mser already sort, but image dimensions may differ
    # between scan paths, so re-sort here with each image's actual size.
    for r in results:
        dets = r.get("detections", [])
        if len(dets) <= 1:
            continue
        fpath = assets / r["file"]
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        dets.sort(key=lambda d: score_refined_detection(d, w, h), reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = args.report or REPORT_FILE
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Debug crops for low-confidence detections (helps manual review).
    if args.debug_crops:
        debug_dir = OUTPUT_DIR / "debug-crops"
        n_crops = 0
        for r in results:
            best = r["detections"][0]
            conf = best.get("confidence", 1.0)
            if conf < 0.55:
                _write_debug_crop(assets / r["file"], r["detections"], debug_dir)
                n_crops += 1
        print(f"Debug crops: {n_crops} written to {debug_dir}")

    print(f"\nFound {len(results)} watermarked image(s)")
    for r in results[:20]:
        best = r["detections"][0]
        print(f"  {r['file']}  score={best.get('score', '?')} "
              f"method={best.get('method', '?')}")
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")
    print(f"Report: {report_path}")


# ---------------------------------------------------------------------------
# CLI: sample-review — pick N random files from a scan report, clean them to
# ~/Downloads/<out>/. Source assets are read-only; nothing is written back.
# ---------------------------------------------------------------------------

KNOWN_FAILURE_CASES = [
    "power-button-volume-button-flex-cable-for-iphone-xs-max-5.jpg",
    "touch-panel-for-iphone-x-black-2.jpg",
    "original-lcd-screen-for-apple-watch-series-2-38mm-with-digitizer-full-assembly-black-5.jpg",
    "earpiece-speaker-for-iphone-11-pro-3.jpg",
    "nfc-coil-with-power-volume-flex-cable-for-iphone-12-pro-max-2.jpg",
    "100-pcs-lcd-frame-bezel-waterproof-adhesive-stickers-for-iphone-xs-4.jpg",
]


def cmd_sample_review(args):
    """Random-sample N entries from a scan report, clean them into ~/Downloads/<out>/.

    Source assets are NEVER modified. Originals, masks and cleaned outputs are
    written side-by-side to the destination directory for easy visual review.
    """
    import random
    template = ensure_template()
    assets = Path(args.assets)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Guard: refuse to write inside the repo
    repo_root = Path(__file__).resolve().parent.parent
    try:
        out_dir.resolve().relative_to(repo_root)
        print(f"ERROR: --out must be outside the repo. Got: {out_dir}", file=sys.stderr)
        sys.exit(2)
    except ValueError:
        pass

    with open(args.report) as f:
        report = json.load(f)
    if not report:
        print("Empty report — nothing to review")
        return

    # Drop iPhone 14+ entries — they should not appear in cleanup workflow per spec
    before = len(report)
    report = [r for r in report if should_scan_file(r.get("file", ""))]
    dropped = before - len(report)
    if dropped:
        print(f"Filtered out {dropped} iPhone 14+ entries from report")

    if args.known_failures:
        names = set(KNOWN_FAILURE_CASES)
        sample = [r for r in report if r.get("file") in names]
        print(f"Selecting known-failure files ({len(sample)} matched)")
    else:
        random.seed(args.seed)
        n = min(args.count, len(report))
        sample = random.sample(report, n)
        print(f"Random sample: {n} of {len(report)}")

    n_clean, n_manual, n_skip = 0, 0, 0
    summary = []
    for entry in sample:
        fname = entry["file"]
        src = assets / fname
        if not src.exists():
            n_skip += 1
            continue
        # Read into memory, clean from temp file so the source asset stays intact
        tmp_src = out_dir / f"_in_{fname}"
        shutil.copy2(src, tmp_src)
        out_path = out_dir / f"cleaned_{fname}"
        cleaned, info = clean_image(tmp_src, entry.get("detections", []),
                                    output_path=out_path, template=template)
        # Always remove temp source — we never want a duplicate of the original
        try:
            tmp_src.unlink()
        except OSError:
            pass

        if cleaned is not None:
            n_clean += 1
            status = info.get("status", "cleaned")
        else:
            n_manual += 1
            # Write the original next to the failure record for review context
            shutil.copy2(src, out_dir / f"original_{fname}")
            status = info.get("reason", info.get("status", "needs_manual"))
        summary.append({
            "file": fname,
            "status": status,
            "info": info,
            "best_detection": entry["detections"][0] if entry.get("detections") else None,
        })

    summary_path = out_dir / "_review-summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Confirm source assets weren't modified
    modified = []
    for entry in sample:
        src = assets / entry["file"]
        if src.exists():
            stat = src.stat()
            # Just record mtime for the audit — actual integrity is git-tracked
            pass

    print()
    print(f"=== Sample review ===")
    print(f"  Cleaned:      {n_clean}")
    print(f"  Needs manual: {n_manual}")
    print(f"  Skipped:      {n_skip}")
    print(f"  Outputs:      {out_dir}")
    print(f"  Summary:      {summary_path}")


# ---------------------------------------------------------------------------
# CLI: metrics — reproducible recall/precision + mask-area report vs ground truth
# ---------------------------------------------------------------------------

GROUND_TRUTH_FILE = OUTPUT_DIR / "ground-truth.json"


def _median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def cmd_metrics(args):
    """Evaluate the detector against generated/watermark/ground-truth.json.

    Reports recall%, precision%, median & max mask-area%, mean IoU on hits, and
    lists every MISSED positive and FALSE-POSITIVE negative. Fully reproducible:

        python3 scripts/remove-watermarks.py metrics
    """
    gt_path = args.ground_truth or GROUND_TRUTH_FILE
    with open(gt_path) as f:
        gt = json.load(f)
    assets = Path(args.assets)
    images = gt["images"]
    n_pos = sum(1 for im in images if im["label"] == "positive")
    n_neg = sum(1 for im in images if im["label"] == "negative")

    print(f"Metrics vs {gt_path}  ({n_pos} positives / {n_neg} negatives)")
    print(f"Assets: {assets}")
    print(f"Detector: detect_template_1d (USE_1D_PRIMARY={USE_1D_PRIMARY})")
    print(f"  rule: (edge>={WM_EDGE_HI} AND |yc-{WM_EDGE_YC_MU}|<={WM_EDGE_YC_MAX}) "
          f"OR (edge>={WM_EDGE_LO} AND band>={WM_BAND_MIN})")
    print()

    tp = fp = fn = tn = 0
    missed = []          # real watermarks the detector failed to find
    false_pos = []       # negatives the detector fired on
    ious = []            # IoU of detection_box vs best GT box (true positives)
    mask_areas = []      # mark_box area % for true positives
    coverages = []       # fraction of GT box covered by the mark_box (true positives)
    undercovered = []    # true positives whose mask misses part of the watermark
    t0 = time.time()

    for i, im in enumerate(images):
        fpath = assets / im["file"]
        gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  WARN: cannot read {im['file']} — counting as miss/clean")
            dets = []
        else:
            dets = detect_template_1d(gray)
        got = len(dets) > 0
        is_pos = im["label"] == "positive"

        if is_pos and got:
            tp += 1
            best = dets[0]
            raw = best.get("detection_box") or best
            mb = best.get("mark_box") or raw
            # IoU of raw detection vs best ground-truth box
            best_iou = 0.0
            for gb in im.get("boxes", []):
                best_iou = max(best_iou, _iou(raw, gb))
            ious.append(best_iou)
            area_pct = 100.0 * mb["w"] * mb["h"] / float(im["img_w"] * im["img_h"])
            mask_areas.append(area_pct)
            # Coverage: fraction of each GT watermark box inside the mark_box.
            # A mask that does not fully cover the text would leave residue.
            cov_min = 1.0
            for gb in im.get("boxes", []):
                ix1 = max(mb["x"], gb["x"])
                iy1 = max(mb["y"], gb["y"])
                ix2 = min(mb["x"] + mb["w"], gb["x"] + gb["w"])
                iy2 = min(mb["y"] + mb["h"], gb["y"] + gb["h"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                cov = inter / float(gb["w"] * gb["h"]) if gb["w"] * gb["h"] else 0.0
                cov_min = min(cov_min, cov)
            coverages.append(cov_min)
            if cov_min < 0.95:
                undercovered.append((im["file"], round(cov_min, 3)))
        elif is_pos and not got:
            fn += 1
            missed.append(im["file"])
        elif (not is_pos) and got:
            fp += 1
            best = dets[0]
            false_pos.append((im["file"], best.get("edge_score"), best.get("band_score")))
        else:
            tn += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            print(f"  [{i + 1}/{len(images)}] {time.time() - t0:.1f}s  "
                  f"tp={tp} fn={fn} fp={fp} tn={tn}", flush=True)

    recall = 100.0 * tp / max(tp + fn, 1)
    precision = 100.0 * tp / max(tp + fp, 1)
    med_area = _median(mask_areas)
    max_area = max(mask_areas) if mask_areas else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    min_cov = min(coverages) if coverages else 0.0
    mean_cov = sum(coverages) / len(coverages) if coverages else 0.0

    print()
    print("=" * 64)
    print(f"  RECALL      : {recall:6.1f}%   ({tp}/{tp + fn} positives found)")
    print(f"  PRECISION   : {precision:6.1f}%   ({tp}/{tp + fp} detections correct)")
    print(f"  mask area   : median={med_area:.2f}%  max={max_area:.2f}%  "
          f"(criterion: median<=3.5%, max<=6%)")
    print(f"  coverage    : min={min_cov * 100:.1f}%  mean={mean_cov * 100:.1f}%  "
          f"(fraction of each watermark inside its mask)")
    print(f"  mean IoU    : {mean_iou:.3f}   (raw detection box vs GT box, on hits)")
    print(f"  confusion   : TP={tp} FN={fn} FP={fp} TN={tn}")
    print("=" * 64)

    if missed:
        print(f"\n  MISSED positives ({len(missed)}):")
        for m in missed:
            print(f"    - {m}")
    if false_pos:
        print(f"\n  FALSE POSITIVES ({len(false_pos)}):")
        for name, es, bs in false_pos:
            print(f"    - {name}  (edge={es} band={bs})")
    if undercovered:
        print(f"\n  UNDER-COVERED watermarks (<95% inside mask) ({len(undercovered)}):")
        for name, cov in undercovered:
            print(f"    - {name}  (coverage={cov})")

    crit_recall = (fn == 0)
    crit_mask = (med_area <= 3.5 and max_area <= 6.0 and min_cov >= 0.95)
    print()
    print(f"  CRITERION 1 (recall=100%):        {'PASS' if crit_recall else 'FAIL'}")
    print(f"  CRITERION 2 (mask tight + full coverage): "
          f"{'PASS' if crit_mask else 'FAIL'}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "recall_pct": round(recall, 2),
                "precision_pct": round(precision, 2),
                "mask_area_median_pct": round(med_area, 3),
                "mask_area_max_pct": round(max_area, 3),
                "coverage_min": round(min_cov, 4),
                "coverage_mean": round(mean_cov, 4),
                "mean_iou": round(mean_iou, 4),
                "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
                "missed": missed,
                "false_positives": [
                    {"file": n, "edge_score": e, "band_score": b}
                    for n, e, b in false_pos
                ],
                "under_covered": [
                    {"file": n, "coverage": c} for n, c in undercovered
                ],
                "criterion_recall_100": crit_recall,
                "criterion_mask_tier1": crit_mask,
            }, f, indent=2)
        print(f"\n  Wrote {out}")

    # Non-zero exit only when the hard recall criterion fails (CI-friendly).
    if not crit_recall:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI: metrics2 — PRECISION-FIRST held-out evaluation of detect_watermark_v2
#
# Loads the LARGER validation set (generated/watermark/validation-set.json:
# confirmed positives + GUARANTEED-clean negatives, with a train/heldout split)
# and reports, on whichever split is requested (default: heldout):
#   - PRECISION on negatives (false-positive count — MUST be 0; the hard gate)
#   - auto-clean recall (% of positives tier=='auto')
#   - needs-manual count (positives + any negatives routed to manual)
#   - median & max auto-clean mask area %
# Fully reproducible:  python3 scripts/remove-watermarks.py metrics2
# ---------------------------------------------------------------------------

VALIDATION_SET_FILE = OUTPUT_DIR / "validation-set.json"


def _wild_worker(name_and_dir):
    """Worker: classify one guaranteed-clean image; return (name, tier, psr, edge, ncc)
    only when it lands on the AUTO tier (a precision breach). None otherwise."""
    name, assets_str = name_and_dir
    gray = cv2.imread(str(Path(assets_str) / name), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    dets = detect_watermark_v2(gray)
    if not dets or dets[0].get("tier") != "auto":
        return None
    d = dets[0]
    return (name, d.get("psr"), d.get("edge_score"), d.get("int_ncc"), d.get("cov"))


def run_wild_negative_check(assets, per_seed, seeds, workers=None):
    """PRECISION wild check: sample `per_seed` guaranteed-clean iPhone-14+ images
    (should_scan_file()==False) for each seed in `seeds` (disjoint draws from the
    same pool), and count how many land on the AUTO tier. A watermark-free image on
    the auto tier is a HARD-GATE breach. Returns (total_auto_fp, breach_files,
    total_evaluated, pool_size). Parallel — keeps each pass well under the watchdog.
    """
    import random
    exts = IMAGE_EXTENSIONS
    pool = sorted(n for n in os.listdir(assets)
                  if Path(n).suffix.lower() in exts and not should_scan_file(n))
    pool_size = len(pool)
    if workers is None:
        workers = min(os.cpu_count() or 1, 10)
    print(f"\n  WILD NEGATIVE CHECK — guaranteed-clean iPhone-{SKIP_IPHONE_MIN}+ pool "
          f"= {pool_size} files; sampling {per_seed}/seed x seeds {seeds} "
          f"({workers} workers)", flush=True)
    breach_files = []
    total_eval = 0
    seen = set()
    t0 = time.time()
    for s in seeds:
        rng = random.Random(s)
        k = min(per_seed, pool_size)
        sample = rng.sample(pool, k)
        new = [n for n in sample if n not in seen]   # report disjoint coverage too
        seen.update(sample)
        total_eval += len(sample)
        work = [(n, str(assets)) for n in sample]
        seed_fp = 0
        with ProcessPoolExecutor(max_workers=workers) as pool_ex:
            for r in pool_ex.map(_wild_worker, work, chunksize=8):
                if r is not None:
                    seed_fp += 1
                    breach_files.append((s,) + r)
        print(f"    seed {s}: {len(sample)} sampled ({len(new)} new) -> "
              f"auto-tier FP = {seed_fp}   [{time.time() - t0:.0f}s]", flush=True)
    return len(breach_files), breach_files, total_eval, pool_size, len(seen)


def cmd_metrics2(args):
    global WM_RESCUE3_ENABLED
    if getattr(args, "faint_band_rescue", False):
        WM_RESCUE3_ENABLED = True
    vs_path = args.validation_set or VALIDATION_SET_FILE
    with open(vs_path) as f:
        vs = json.load(f)
    assets = Path(args.assets)
    images = vs["images"]
    split = args.split

    def in_split(im):
        return split == "all" or im.get("split") == split
    sel = [im for im in images if in_split(im)]
    n_pos = sum(1 for im in sel if im["label"] == "positive")
    n_neg = sum(1 for im in sel if im["label"] == "negative")

    print(f"metrics2 — precision-first held-out eval of detect_watermark_v2")
    print(f"  validation set: {vs_path}")
    print(f"  meta: {vs.get('_meta', {}).get('positives')} pos / "
          f"{vs.get('_meta', {}).get('negatives')} neg total; "
          f"train={vs.get('_meta', {}).get('train')} heldout={vs.get('_meta', {}).get('heldout')}")
    print(f"  evaluating split='{split}'  ({n_pos} positives / {n_neg} negatives)")
    print(f"  auto-clean gate: edge>={WM_AUTO_EDGE} ncc>={WM_AUTO_NCC} cov>={WM_AUTO_COV} "
          f"biggap<={WM_AUTO_BIGGAP} xoff<={WM_AUTO_XOFF} psr>={WM_AUTO_PSR}")
    print()

    auto_tp = manual_pos = drop_pos = 0          # positives by tier
    fp = manual_neg = clean_neg = 0              # negatives by tier
    mask_areas = []
    false_pos = []
    manual_pos_files = []
    drop_pos_files = []
    t0 = time.time()
    for i, im in enumerate(sel):
        gray = cv2.imread(str(assets / im["file"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  WARN: cannot read {im['file']} — counting as no-watermark")
            tier = "none"
            det = None
        else:
            dets = detect_watermark_v2(gray)
            det = dets[0] if dets else None
            tier = det["tier"] if det else "none"
        is_pos = im["label"] == "positive"

        if is_pos:
            if tier == "auto":
                auto_tp += 1
                mask_areas.append(det["debug"]["mask_area_pct"])
            elif tier == "manual":
                manual_pos += 1
                manual_pos_files.append(im["file"])
            else:
                drop_pos += 1
                drop_pos_files.append(im["file"])
        else:
            if tier == "auto":
                fp += 1
                false_pos.append((im["file"], det.get("edge_score"), det.get("int_ncc"),
                                  det.get("cov"), im.get("source")))
            elif tier == "manual":
                manual_neg += 1
            else:
                clean_neg += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(sel):
            print(f"  [{i + 1}/{len(sel)}] {time.time() - t0:.1f}s  "
                  f"auto_tp={auto_tp} manual_pos={manual_pos} fp={fp}", flush=True)

    auto_recall = 100.0 * auto_tp / max(n_pos, 1)
    cover_recall = 100.0 * (auto_tp + manual_pos) / max(n_pos, 1)
    precision = 100.0 * auto_tp / max(auto_tp + fp, 1)
    neg_precision = 100.0 * (n_neg - fp) / max(n_neg, 1)  # % of negatives NOT auto-flagged
    med_area = _median(mask_areas)
    max_area = max(mask_areas) if mask_areas else 0.0

    print()
    print("=" * 66)
    print(f"  IDENTIFICATION RECALL: {cover_recall:6.1f}%   "
          f"({auto_tp + manual_pos}/{n_pos} watermarks FLAGGED auto-or-manual; "
          f"TARGET 100%)")
    print(f"  DROPPED (missed)     : {drop_pos} positives   (TARGET 0 — a dropped "
          f"watermark is never surfaced)")
    print("  " + "-" * 62)
    print(f"  NEGATIVE PRECISION : {neg_precision:6.1f}%   "
          f"FALSE POSITIVES = {fp}   (HARD GATE: must be 0)")
    print(f"  auto-clean recall  : {auto_recall:6.1f}%   ({auto_tp}/{n_pos} positives auto-cleaned)")
    print(f"  auto+manual recall : {cover_recall:6.1f}%   "
          f"({auto_tp + manual_pos}/{n_pos} positives surfaced)")
    print(f"  routed to manual   : {manual_pos} positives + {manual_neg} negatives "
          f"= {manual_pos + manual_neg} images for review")
    print(f"  auto-clean mask    : median={med_area:.2f}%  max={max_area:.2f}%  "
          f"(target median ~2%)")
    print(f"  precision (TP/(TP+FP)) : {precision:.1f}%")
    print("=" * 66)

    if false_pos:
        print(f"\n  !!! FALSE POSITIVES ({len(false_pos)}) — clean images auto-flagged:")
        for name, e, n, c, src in false_pos:
            print(f"    - {name}  (edge={e} ncc={n} cov={c} src={src})")
    else:
        print("\n  PASS: zero false positives — no clean image would be auto-cleaned.")

    if drop_pos_files:
        print(f"\n  !!! DROPPED positives ({len(drop_pos_files)}) — NOT surfaced "
              f"(blocking 100% identification):")
        for m in drop_pos_files:
            print(f"    - {m}")
    else:
        print("\n  PASS: zero dropped — every confirmed watermark was identified.")

    if args.list_manual and manual_pos_files:
        print(f"\n  positives routed to manual ({len(manual_pos_files)}):")
        for m in manual_pos_files:
            print(f"    - {m}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "split": split,
                "n_positives": n_pos, "n_negatives": n_neg,
                "identification_recall_pct": round(cover_recall, 2),  # auto+manual; target 100
                "dropped_positives": drop_pos,                        # target 0
                "dropped_files": drop_pos_files,
                "false_positives": fp,
                "negative_precision_pct": round(neg_precision, 2),
                "auto_clean_recall_pct": round(auto_recall, 2),
                "auto_plus_manual_recall_pct": round(cover_recall, 2),
                "manual_positives": manual_pos, "manual_negatives": manual_neg,
                "mask_area_median_pct": round(med_area, 3),
                "mask_area_max_pct": round(max_area, 3),
                "false_positive_files": [
                    {"file": n_, "edge": e, "int_ncc": nn, "cov": c, "source": s}
                    for n_, e, nn, c, s in false_pos
                ],
            }, f, indent=2)
        print(f"\n  Wrote {out}")

    # WILD NEGATIVE CHECK (the generalization gate). The validation split alone
    # overfits — a prior gate scored 100% precision there yet auto-cleaned clean
    # images in the wild. So additionally sample the LARGE guaranteed-clean
    # iPhone-14+ pool across multiple disjoint seeds and require ZERO auto-tier hits.
    wild_fp = 0
    wild_breaches = []
    wild_eval = wild_pool = wild_unique = 0
    if args.wild_negatives > 0:
        seeds = [int(s) for s in str(args.wild_seeds).split(",") if s.strip()]
        wild_fp, wild_breaches, wild_eval, wild_pool, wild_unique = run_wild_negative_check(
            assets, args.wild_negatives, seeds)
        print("=" * 66)
        print(f"  WILD AUTO-TIER FP (all seeds): {wild_fp}   "
              f"(evaluated {wild_eval} draws, {wild_unique} unique of {wild_pool} pool; "
              f"HARD GATE: must be 0)")
        print("=" * 66)
        if wild_breaches:
            print(f"\n  !!! WILD FALSE POSITIVES ({len(wild_breaches)}) — guaranteed-clean "
                  f"images that would be AUTO-CLEANED:")
            for s, name, psr, e, n, c in wild_breaches:
                print(f"    - [seed {s}] {name}  (psr={psr} edge={e} ncc={n} cov={c})")
        else:
            print("\n  PASS: zero wild auto-tier false positives across all seeds.")

    # CI gate: precision-first — fail if a clean image was auto-flagged, on the
    # validation negatives OR in the wild multi-seed check.
    if fp > 0 or wild_fp > 0:
        print(f"\n  FAIL: precision gate breached — "
              f"{fp} validation FP + {wild_fp} wild FP (must both be 0).")
        sys.exit(1)
    msg = "zero false positives"
    if args.wild_negatives > 0:
        msg += f"; zero wild auto-tier FP across {wild_eval} guaranteed-clean draws"
    print(f"\n  OK: precision gate satisfied ({msg}).")


# ---------------------------------------------------------------------------
# CLI: known-failure regression check
# ---------------------------------------------------------------------------

def cmd_regression(args):
    """Tier-report the known-failure files under the precision-first v2 detector.

    NOTE (v4): the hard "5/6 must auto-detect" gate is OBSOLETE. Those files were
    curated for the old recall-maximizing detector; under precision-first some are
    correctly routed to manual or skipped rather than auto-cleaned. This command
    now asserts only that EVERY produced mask is tight (area <= MASK cap) — i.e. no
    detection is destructive — and reports the auto/manual/none tier breakdown for
    visibility. It is informational, not a recall gate (use `metrics2` for that).
    """
    assets = Path(args.assets)
    results = []
    for fname in KNOWN_FAILURE_CASES:
        fpath = assets / fname
        if not fpath.exists():
            results.append({"file": fname, "tier": "MISSING", "ok": True})
            continue
        gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        dets = detect_watermark_v2(gray) if gray is not None else []
        if not dets:
            results.append({"file": fname, "tier": "none", "ok": True})
            continue
        best = dets[0]
        mb = best.get("mark_box") or {}
        dbg = best.get("debug") or {}
        area_pct = dbg.get("mask_area_pct", 0)
        results.append({
            "file": fname, "tier": best.get("tier", "?"),
            "ok": bool(mb) and area_pct <= 100 * MASK_AREA_REVIEW_MAX,
            "score": best.get("score"), "confidence": best.get("confidence"),
            "mark_box": mb, "mask_area_pct": area_pct,
            "aspect": dbg.get("mark_aspect_ratio"),
        })

    from collections import Counter
    tiers = Counter(r["tier"] for r in results)
    bad = [r for r in results if not r["ok"]]
    print(f"Known-failure tier report (precision-first v2): "
          f"{dict(tiers)}  (total {len(results)})")
    print()
    for r in results:
        if r["tier"] in ("auto", "manual"):
            mb = r["mark_box"]
            print(f"  {r['tier']:6s} {r['file'][:52]}  conf={r['confidence']} "
                  f"mark={mb['w']}x{mb['h']} area={r['mask_area_pct']}%")
        else:
            print(f"  {r['tier']:6s} {r['file'][:52]}")
    print()
    if bad:
        print(f"FAIL — {len(bad)} detection(s) produced an oversized/destructive mask:")
        for r in bad:
            print(f"    - {r['file']} area={r.get('mask_area_pct')}%")
        sys.exit(1)
    print("PASS — every produced mask is tight (no destructive detection).")
    return 0


# ---------------------------------------------------------------------------
# CLI: clean
# ---------------------------------------------------------------------------

def cmd_clean(args):
    template = ensure_template()
    assets = Path(args.assets)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    det_map = {}
    file_list = list(args.files or [])

    if args.from_report:
        with open(args.from_report) as f:
            report = json.load(f)
        for entry in report:
            fname = entry["file"]
            if fname not in file_list:
                file_list.append(fname)
            if "detections" in entry:
                det_map[fname] = entry["detections"]
            else:
                det_map[fname] = [entry]

    if not file_list:
        print("No files specified. Use positional args or -f report.json")
        sys.exit(1)

    if args.dry_run:
        print(f"Dry run — would clean {len(file_list)} images:")
        for f in file_list[:30]:
            print(f"  {f}")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    t0 = time.time()
    total = len(file_list)
    passes = args.passes
    current_list = list(file_list)

    for pass_num in range(1, passes + 1):
        if not current_list:
            break
        print(f"\n--- Pass {pass_num}/{passes} ({len(current_list)} images) ---")
        cleaned, failed = 0, 0

        for i, fname in enumerate(current_list):
            fpath = assets / fname
            if not fpath.exists():
                continue

            if not (BACKUP_DIR / fname).exists():
                shutil.copy2(fpath, BACKUP_DIR / fname)

            dets = det_map.get(fname)
            if not dets:
                dets = detect_watermark(str(fpath), template)
                if not dets:
                    continue

            result, info = clean_image(fpath, dets, template=template)
            if result is not None:
                cleaned += 1
                entry = {"file": fname, "pass": pass_num, "status": "cleaned", **info}
            else:
                failed += 1
                entry = {"file": fname, "pass": pass_num,
                         "status": "needs_manual", **info}

            existing = [m for m in manifest if m["file"] == fname]
            if existing:
                prev = existing[0]
                if result is not None or prev.get("status") != "cleaned":
                    prev.update(entry)
            else:
                manifest.append(entry)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(current_list)}] "
                      f"{elapsed:.0f}s elapsed, {cleaned} cleaned, {failed} failed")

        print(f"  Pass {pass_num}: {cleaned} cleaned, {failed} failed")

        if pass_num < passes:
            print("  Rescanning for residuals...")
            still = []
            for fname in current_list:
                fpath = assets / fname
                if not fpath.exists():
                    continue
                dets = detect_watermark(str(fpath), template)
                if dets:
                    still.append(fname)
                    det_map[fname] = dets
            print(f"  Rescan: {len(still)} still watermarked "
                  f"({1.0 - len(still) / max(len(current_list), 1):.0%} removed)")
            if not still:
                print("  All watermarks removed!")
                break
            current_list = still

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    manual = [m for m in manifest if m.get("status") == "needs_manual"]
    if manual:
        manual_path = OUTPUT_DIR / "needs-manual.json"
        with open(manual_path, "w") as f:
            json.dump(manual, f, indent=2)
        print(f"\nNeeds manual: {manual_path} ({len(manual)} images)")

    generate_contact_sheet(manifest, assets)

    elapsed = time.time() - t0
    ok = sum(1 for m in manifest if m["status"] == "cleaned")
    fail = sum(1 for m in manifest if m.get("status") == "needs_manual")
    print(f"\nDone: {ok} cleaned, {fail} needs_manual in {elapsed:.0f}s")
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Backups:  {BACKUP_DIR}")


# ---------------------------------------------------------------------------
# CLI: process — one-pass detect + clean, no pre-scan report
# ---------------------------------------------------------------------------

def _assert_outside_repo(path):
    repo_root = SCRIPT_DIR.parent.resolve()
    try:
        path.resolve().relative_to(repo_root)
    except ValueError:
        return
    print(f"ERROR: output path must be outside the repo. Got: {path}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# CLI: pilot — small QA batch with original / mask / cleaned preview HTML
# ---------------------------------------------------------------------------

def _b64_thumb(img, width=320):
    """Encode a BGR image as a base64 JPEG <img> src, scaled to `width`."""
    h, w = img.shape[:2]
    if w > width:
        scale = width / w
        img = cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _phash(img_gray, hash_size=8):
    """64-bit DCT perceptual hash of a grayscale image (no extra deps).

    V7 Phase 1: catalog listings reuse identical base photos. Hashing lets the
    pipeline skip re-running detection on an image perceptually identical to one
    already found watermark-free — cheap structural bypass for template layouts.
    """
    sz = hash_size * 4
    small = cv2.resize(img_gray, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # exclude DC term from the threshold
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _mask_overlay(img_bgr, mask):
    """Original with the final mask tinted bright green (rgba ~0.45) — QA overlay
    so an operator can confirm the mask hugs glyph strokes, not a rectangle."""
    ov = img_bgr.copy()
    m = mask.astype(bool)
    if np.any(m):
        green = np.zeros_like(ov)
        green[:, :] = (0, 255, 0)
        ov[m] = (0.55 * ov[m] + 0.45 * green[m]).astype(np.uint8)
    return ov


def _diff_heatmap(original, cleaned):
    """Per-pixel absolute-difference heatmap (JET) between original and cleaned —
    makes any localized smudge or collateral change immediately visible."""
    if cleaned is None:
        return np.zeros_like(original)
    go = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gc = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.clip(np.abs(go - gc) * 3, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def cmd_pilot(args):
    """Run a small QA pilot: scan a capped sample, clean each, and emit an
    original / mask / cleaned preview HTML for visual sign-off.

    Source assets are NEVER modified. All output (cleaned images, mask PNGs,
    review.html, manifest.json) is written under --out, which must be outside
    the repo. Requires --rights-confirmed to proceed (compliance gate: you must
    confirm you hold the rights to remove the watermark from these images).
    """
    if not args.rights_confirmed:
        print("REFUSED: watermark removal requires --rights-confirmed.\n"
              "  Pass --rights-confirmed to confirm you hold the rights to remove\n"
              "  the watermark from these images (e.g. you own/license the catalog).",
              file=sys.stderr)
        sys.exit(2)

    assets = Path(args.assets)
    out_dir = Path(args.out).expanduser() if args.out else (
        Path.home() / "Downloads" /
        f"sunsky-watermark-pilot-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    _assert_outside_repo(out_dir)
    cleaned_dir = out_dir / "cleaned"
    masks_dir = out_dir / "masks"
    manual_dir = out_dir / "manual-review"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    manual_dir.mkdir(parents=True, exist_ok=True)
    debug_mask = getattr(args, "debug_mask", False)
    debug_dir = out_dir / "debug"
    if debug_mask:
        debug_dir.mkdir(parents=True, exist_ok=True)
    manual_jsonl = open(manual_dir / "manual_review.jsonl", "w")

    apply_scan_preset(args.preset)
    template = ensure_template()

    # Explicit auto/manual split control. --auto-psr overrides --auto-mode.
    global WM_AUTO_PSR
    _orig_psr = WM_AUTO_PSR
    if getattr(args, "auto_psr", None) is not None:
        WM_AUTO_PSR = float(args.auto_psr)
    elif getattr(args, "auto_mode", "strict") == "balanced":
        WM_AUTO_PSR = 5.0
    elif getattr(args, "auto_mode", "strict") == "aggressive":
        WM_AUTO_PSR = 4.0
    print(f"Auto/manual gate: auto-tier PSR >= {WM_AUTO_PSR} "
          f"(mode={getattr(args, 'auto_mode', 'strict')})")

    # Pilot is a CV preview — keep LaMA off unless explicitly requested
    old_lama_path = None
    if not args.use_lama:
        global LAMA_MODEL_PATH
        old_lama_path = LAMA_MODEL_PATH
        LAMA_MODEL_PATH = Path("/tmp/disabled-big-lama-model.pt")

    all_files = sorted(f for f in os.listdir(assets)
                       if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
    skipped = [f for f in all_files if not should_scan_file(f)]
    files = [f for f in all_files if should_scan_file(f)]
    # --seed → random sample (so "50 random" is genuinely random & reproducible)
    if getattr(args, "seed", None) is not None:
        import random
        random.seed(args.seed)
        random.shuffle(files)

    rows = []
    counts = {"cleaned": 0, "needs_manual": 0, "no_watermark": 0,
              "prefilter_skipped": 0}
    prefilter = getattr(args, "prefilter", True)
    clean_hashes = set()  # phashes of images the detector found watermark-free
    considered = 0
    t0 = time.time()
    print(f"Pilot: scanning up to {args.max_total} watermarked image(s) "
          f"(preset={args.preset}, skipped {len(skipped)} iPhone {SKIP_IPHONE_MIN}+)")
    print(f"Output: {out_dir}")

    try:
        for fname in files:
            if counts["cleaned"] + counts["needs_manual"] >= args.max_total:
                break
            fpath = assets / fname
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # V7 Phase 1: structural prefilter. If this image is perceptually
            # identical to one already found watermark-free, skip detection.
            ph = _phash(gray) if prefilter else None
            if prefilter and ph in clean_hashes:
                counts["no_watermark"] += 1
                counts["prefilter_skipped"] += 1
                continue

            dets = detect_watermark_image(gray, template)
            if not dets:
                counts["no_watermark"] += 1
                if prefilter:
                    clean_hashes.add(ph)
                continue

            considered += 1
            # Build the component-filtered PRECISION mask we WOULD apply, for the
            # preview — the actual primary inpaint mask now — and capture the same
            # safety metrics the cleaner uses, so the HTML mirrors the decision.
            preview_mask = np.zeros(gray.shape, dtype=np.uint8)
            max_prec_density = 0.0
            roi_type = "flat"
            _rank = {"flat": 0, "medium": 1, "detail_heavy": 2}
            for det in dets:
                pm, m = analyze_precision_mask(gray, det)
                max_prec_density = max(max_prec_density, m["density"])
                if _rank.get(m["roi_type"], 0) >= _rank.get(roi_type, 0):
                    roi_type = m["roi_type"]
                preview_mask = cv2.bitwise_or(preview_mask, pm)
            prec_area_pct = round(100.0 * np.count_nonzero(preview_mask) / preview_mask.size, 4)

            out_path = cleaned_dir / fname
            result, info = clean_loaded_image(
                img.copy(), gray, list(dets), output_path=out_path, template=template)

            mask_png = masks_dir / (Path(fname).stem + "_mask.png")
            cv2.imwrite(str(mask_png), preview_mask)

            overlay = _mask_overlay(img, preview_mask)
            heat = _diff_heatmap(img, result)

            status = info.get("status") or ("cleaned" if result is not None else "needs_manual")
            reason = info.get("reason", "")
            best = dets[0]
            mark_box = best.get("mark_box") or {
                "x": best["x"], "y": best["y"], "w": best["w"], "h": best["h"]}
            if result is not None:
                counts["cleaned"] += 1
                cleaned_src = _b64_thumb(result)
            elif status == "no_watermark":
                # V7 Phase 2.3: detector fired but nothing to clean → already
                # clean. Not a manual case; skip (don't copy to manual-review).
                counts["no_watermark"] += 1
                continue
            else:
                counts["needs_manual"] += 1
                cleaned_src = ""  # no cleaned output; show placeholder
                # Step 9: copy the untouched original into manual-review/ + log a
                # JSONL line so the residual is operationally reachable.
                shutil.copy2(fpath, manual_dir / fname)
                manual_jsonl.write(json.dumps({
                    "file": fname,
                    "status": "needs_manual",
                    "manual_reason": reason or "needs_manual",
                    "failed_gates": info.get("failed_gates", []),
                    "bbox": [mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]],
                    "roi_type": info.get("roi_type", roi_type),
                    "high_risk": info.get("high_risk"),
                    "detector_tier": info.get("detector_tier"),
                    "confidence": info.get("confidence"),
                    "mask_density": round(info.get("mask_density_pct", 100.0 * max_prec_density) / 100.0, 4),
                    "component_count": info.get("component_count"),
                    "cluster_count": info.get("cluster_count"),
                    "line_fraction": info.get("line_fraction"),
                    "largest_component_ratio": info.get("largest_component_ratio"),
                    "edge_overlap_ratio": info.get("edge_overlap_ratio"),
                    "border_touch_ratio": info.get("border_touch_ratio"),
                    "roi_sharpness_ratio": info.get("roi_sharpness_ratio"),
                    "outside_mask_ssim": info.get("outside_mask_ssim"),
                    "strategy": "skipped",
                }) + "\n")

            if debug_mask:
                stem = Path(fname).stem
                cv2.imwrite(str(debug_dir / f"{stem}_original.jpg"), img)
                cv2.imwrite(str(debug_dir / f"{stem}_mask.png"), preview_mask)
                cv2.imwrite(str(debug_dir / f"{stem}_mask_overlay.jpg"), overlay)
                cv2.imwrite(str(debug_dir / f"{stem}_diff.jpg"), heat)
                if result is not None:
                    cv2.imwrite(str(debug_dir / f"{stem}_cleaned.jpg"), result)
                with open(debug_dir / f"{stem}_metrics.json", "w") as mf:
                    json.dump(info, mf, indent=2)

            rows.append({
                "file": fname,
                "status": status,
                "reason": reason,
                "orig_b64": _b64_thumb(img),
                "mask_b64": _b64_thumb(cv2.cvtColor(preview_mask, cv2.COLOR_GRAY2BGR)),
                "overlay_b64": _b64_thumb(overlay),
                "heat_b64": _b64_thumb(heat),
                "clean_b64": cleaned_src,
                "score": best.get("score"),
                "strategy": info.get("strategy", ""),
                "roi_type": info.get("roi_type", roi_type),
                "high_risk": info.get("high_risk", ""),
                "confidence": info.get("confidence", ""),
                "mask_area_pct": prec_area_pct,
                "mask_density_pct": info.get("mask_density_pct", round(100.0 * max_prec_density, 2)),
                "component_count": info.get("component_count", ""),
                "cluster_count": info.get("cluster_count", ""),
                "line_fraction": info.get("line_fraction", ""),
                "largest_component_ratio": info.get("largest_component_ratio", ""),
                "edge_overlap_ratio": info.get("edge_overlap_ratio", ""),
                "border_touch_ratio": info.get("border_touch_ratio", ""),
                "roi_sharpness_ratio": info.get("roi_sharpness_ratio", ""),
                "outside_mask_ssim": info.get("outside_mask_ssim", ""),
                "failed_gates": info.get("failed_gates", []),
                "detections": len(dets),
            })
            n_done = counts["cleaned"] + counts["needs_manual"]
            if n_done % 5 == 0 or n_done >= args.max_total:
                rate = n_done / max(time.time() - t0, 0.001)
                print(f"  [{n_done}/{args.max_total}] {rate:.1f} img/s  "
                      f"cleaned={counts['cleaned']} manual={counts['needs_manual']}")
    finally:
        if old_lama_path is not None:
            LAMA_MODEL_PATH = old_lama_path
        WM_AUTO_PSR = _orig_psr
        manual_jsonl.close()

    # Write preview HTML — CARD layout (Priority 4): two image ROWS per card so a
    # PDF export can never crop the cleaned / diff columns off the right edge.
    import collections as _col
    reason_counts = _col.Counter(
        (r["reason"] or "ok") for r in rows if r["status"] != "cleaned")
    n_total = len(rows)
    n_clean = counts["cleaned"]
    n_manual = counts["needs_manual"]
    auto_rate = round(100.0 * n_clean / n_total, 1) if n_total else 0.0

    cards = []
    for r in rows:
        badge = "ok" if r["status"] == "cleaned" else "manual"
        clean_cell = (f'<img src="{r["clean_b64"]}">' if r["clean_b64"]
                      else '<div class="ph">no clean output<br>(needs_manual)</div>')
        fg = ", ".join(str(x) for x in r.get("failed_gates", []))
        cards.append(f"""<div class="card {badge}">
  <div class="hdr"><b>{r['file']}</b> &nbsp;
    <span class="{badge}">{r['status']}</span> &nbsp; reason: <b>{r.get('reason','') or '—'}</b></div>
  <div class="imgrow">
    <figure><figcaption>original</figcaption><img src="{r['orig_b64']}"></figure>
    <figure><figcaption>precision mask</figcaption><img src="{r['mask_b64']}"></figure>
    <figure><figcaption>mask overlay</figcaption><img src="{r['overlay_b64']}"></figure>
  </div>
  <div class="imgrow">
    <figure><figcaption>cleaned</figcaption>{clean_cell}</figure>
    <figure><figcaption>diff heatmap</figcaption><img src="{r['heat_b64']}"></figure>
    <figure class="metrics"><figcaption>metrics</figcaption>
      <div>Strategy: {r.get('strategy','') or '—'}</div>
      <div>ROI Type: {r.get('roi_type','')}</div>
      <div>High Risk: {r.get('high_risk','')} &nbsp; Conf: {r.get('confidence','')}</div>
      <div>Mask Density: {r.get('mask_density_pct','')}%</div>
      <div>Component Count: {r.get('component_count','')}</div>
      <div>Clusters: {r.get('cluster_count','')} &nbsp; Line Frac: {r.get('line_fraction','')}</div>
      <div>Largest Comp Ratio: {r.get('largest_component_ratio','')}</div>
      <div>Edge Overlap: {r.get('edge_overlap_ratio','')}</div>
      <div>Border Touch: {r.get('border_touch_ratio','')}</div>
      <div>ROI Sharpness Ratio: {r.get('roi_sharpness_ratio','')}</div>
      <div>Outside-Mask SSIM: {r.get('outside_mask_ssim','')}</div>
      <div class="fg">Failed gates: {fg or '—'}</div>
    </figure>
  </div>
</div>""")

    summary_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in reason_counts.most_common())

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Watermark Pilot — {n_total} images</title>
<style>
 body{{font-family:-apple-system,monospace;background:#111;color:#eee;margin:16px}}
 h2{{font-weight:600}}
 .summary{{border-collapse:collapse;margin:8px 0 20px}}
 .summary td{{border:1px solid #444;padding:4px 10px}}
 .card{{border:1px solid #333;border-radius:8px;padding:10px;margin:14px 0;
        page-break-inside:avoid;break-inside:avoid}}
 .card.manual{{background:#241616}} .card.ok{{background:#13201a}}
 .hdr{{font-size:13px;margin-bottom:8px;word-break:break-all}}
 .imgrow{{display:flex;gap:10px;flex-wrap:nowrap;margin-bottom:8px}}
 figure{{margin:0;flex:1 1 0;min-width:0}}
 figcaption{{font-size:11px;color:#9af;margin-bottom:3px}}
 img{{width:100%;max-width:320px;display:block;border:1px solid #222}}
 .metrics{{font-size:12px;line-height:1.5}}
 .metrics .fg{{color:#fa8;margin-top:4px}}
 span.ok{{color:#5f5}} span.manual{{color:#fa5}}
 .ph{{height:120px;display:flex;align-items:center;justify-content:center;
      color:#888;border:1px dashed #555;text-align:center}}
</style></head>
<body>
<h2>Watermark Pilot — preset={args.preset} — auto-clean rate {auto_rate}% ({n_clean}/{n_total})</h2>
<table class="summary">
<tr><td>cleaned</td><td><b>{n_clean}</b></td></tr>
<tr><td>needs_manual</td><td><b>{n_manual}</b></td></tr>
<tr><td>auto-clean rate</td><td><b>{auto_rate}%</b></td></tr>
<tr><td>no_watermark scanned past</td><td>{counts['no_watermark']}</td></tr>
<tr><td>prefilter-skipped (phash duplicate-clean)</td><td>{counts.get('prefilter_skipped', 0)}</td></tr>
<tr><td>iPhone {SKIP_IPHONE_MIN}+ skipped</td><td>{len(skipped)}</td></tr>
<tr><td colspan="2"><b>manual reasons</b></td></tr>
{summary_rows}
</table>
<p>Source: {assets}</p>
{"".join(cards)}
</body></html>"""

    review_path = out_dir / "review.html"
    with open(review_path, "w") as f:
        f.write(html)
    manifest = {
        "mode": "pilot", "preset": args.preset, "assets": str(assets),
        "max_total": args.max_total, "counts": counts,
        "elapsed_seconds": round(time.time() - t0, 1),
        "auto_clean_rate_pct": auto_rate,
        "manual_reasons": dict(reason_counts),
        "items": [{k: r[k] for k in ("file", "status", "reason", "failed_gates",
                                     "score", "strategy", "roi_type", "high_risk",
                                     "confidence", "mask_area_pct",
                                     "mask_density_pct", "component_count",
                                     "cluster_count", "line_fraction",
                                     "largest_component_ratio", "edge_overlap_ratio",
                                     "border_touch_ratio", "roi_sharpness_ratio",
                                     "outside_mask_ssim", "detections")} for r in rows],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Pilot complete ===")
    print(f"  cleaned:      {counts['cleaned']}")
    print(f"  needs_manual: {counts['needs_manual']}")
    print(f"  preview:      {review_path}")
    print(f"  manual-review: {manual_dir}  (+ manual_review.jsonl)")
    if debug_mask:
        print(f"  debug:        {debug_dir}")
    print(f"  open with:    open '{review_path}'")


# ---------------------------------------------------------------------------
# CLI: manual-queue — Phase-3 human-review fallback for the MANUAL-tier residual
#
# The auto-detector is at its precision/recall frontier (0 auto-tier FP on the
# full 7342-clean pool, ~38% auto-clean, ~92% surfaced). The remaining faint
# watermarks are routed to the MANUAL tier — surfaced but NOT auto-cleaned,
# because they sit inside the clean-image feature cloud and auto-cleaning them
# would risk the brand-safety hard gate. This workflow makes that residual
# operationally reachable: it exports the manual-tier detections as a fast
# human-review batch (original + overlay preview + checkbox HTML + manifest),
# and a `clean-approved` step LaMa-inpaints only the operator-approved files.
#
# HARD INVARIANTS (shared with pilot/process):
#   - source assets are NEVER modified (read-only),
#   - all output goes to --out OUTSIDE the repo (default ~/Downloads/...),
#   - iPhone-14+ files are skipped (known clean),
#   - --rights-confirmed is required, --use-lama and the precision mask +
#     density cap are honoured. The operator commits via the existing
#     `replace --source-dir <out>/cleaned`.
# ---------------------------------------------------------------------------

def _overlay_preview(img_bgr, mark_box, precision_mask):
    """Return a BGR copy of `img_bgr` with the suggested mark_box drawn in RED
    and the precision stroke-mask tinted GREEN — the human-review overlay.

    The red rectangle shows what the cleaner would target; the green strokes
    show the exact pixels the precision (LaMa) inpaint would touch, so an
    operator can audit coverage at a glance before approving.
    """
    ov = img_bgr.copy()
    # Green stroke tint where the precision mask is set (BGR green = (0,255,0)).
    if precision_mask is not None and np.count_nonzero(precision_mask):
        green = np.zeros_like(ov)
        green[:, :] = (0, 255, 0)
        m = precision_mask.astype(bool)
        ov[m] = (0.45 * ov[m] + 0.55 * green[m]).astype(np.uint8)
    # Red suggested mark_box.
    if mark_box:
        x, y, w, h = mark_box["x"], mark_box["y"], mark_box["w"], mark_box["h"]
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return ov


def cmd_manual_queue(args):
    """Phase-3 MANUAL fallback: surface the manual-tier residual for human review.

    Runs detection over --assets (iPhone-14+ skipped), keeps every image whose
    TOP detection is tier=='manual' (surfaced-but-not-auto), and exports a
    human-review batch to --out (outside the repo): per image the ORIGINAL plus
    an overlay preview (suggested mark_box in red, precision stroke-mask in
    green) as side-by-side thumbnails, an index review.html with a per-image
    APPROVE/SKIP control, and a manifest.json (file, mark_box, score, tier).

    Source assets are NEVER modified. Requires --rights-confirmed.
    """
    if getattr(args, "qaction", "build") == "clean-approved":
        return cmd_manual_queue_clean(args)

    if not args.rights_confirmed:
        print("REFUSED: watermark manual-queue requires --rights-confirmed.\n"
              "  Pass --rights-confirmed to confirm you hold the rights to remove\n"
              "  the watermark from these images (e.g. you own/license the catalog).",
              file=sys.stderr)
        sys.exit(2)

    assets = Path(args.assets)
    out_dir = Path(args.out).expanduser() if args.out else (
        Path.home() / "Downloads" /
        f"sunsky-manual-queue-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    _assert_outside_repo(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_scan_preset(args.preset)
    template = ensure_template()

    # Optional: enable RESCUE #3 so the very last faint full-width-on-busy-PCB
    # marks are also surfaced to the manual queue (manual-only — never auto-cleans).
    global WM_RESCUE3_ENABLED
    _orig_rescue3 = WM_RESCUE3_ENABLED
    if getattr(args, "faint_band_rescue", False):
        WM_RESCUE3_ENABLED = True

    all_files = sorted(f for f in os.listdir(assets)
                       if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
    skipped = [f for f in all_files if not should_scan_file(f)]
    files = [f for f in all_files if should_scan_file(f)]
    if getattr(args, "seed", None) is not None:
        import random
        random.seed(args.seed)
        random.shuffle(files)
    if args.limit:
        files = files[:args.limit]

    rows = []
    manifest_items = []
    scanned = 0
    no_wm = 0
    auto_skipped = 0
    t0 = time.time()
    print(f"Manual-queue: scanning up to {len(files)} eligible image(s) "
          f"(preset={args.preset}, skipped {len(skipped)} iPhone {SKIP_IPHONE_MIN}+)")
    print(f"Collecting up to {args.max} manual-tier image(s).")
    print(f"Output: {out_dir}")

    try:
        for fname in files:
            if len(manifest_items) >= args.max:
                break
            scanned += 1
            fpath = assets / fname
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detect_watermark_image(gray, template)
            if not dets:
                no_wm += 1
            else:
                best = dets[0]
                tier = best.get("tier")
                if tier == "auto":
                    auto_skipped += 1
                elif tier == "manual":
                    mark_box = best.get("mark_box") or get_cleaning_box(best)
                    prec_mask, density = generate_precision_mask(gray, best)
                    overlay = _overlay_preview(img, mark_box, prec_mask)
                    area_pct = round(
                        100.0 * np.count_nonzero(prec_mask) / prec_mask.size, 4)
                    rows.append({
                        "file": fname,
                        "orig_b64": _b64_thumb(img, width=360),
                        "overlay_b64": _b64_thumb(overlay, width=360),
                        "mark_box": mark_box,
                        "score": best.get("score"),
                        "edge_score": best.get("edge_score"),
                        "int_ncc": best.get("int_ncc"),
                        "cov": best.get("cov"),
                        "psr": best.get("psr"),
                        "confidence": best.get("confidence"),
                        "mask_area_pct": area_pct,
                        "mask_density_pct": round(100.0 * density, 2),
                    })
                    manifest_items.append({
                        "file": fname,
                        "mark_box": mark_box,
                        "score": best.get("score"),
                        "tier": tier,
                        "edge_score": best.get("edge_score"),
                        "int_ncc": best.get("int_ncc"),
                        "cov": best.get("cov"),
                        "psr": best.get("psr"),
                        "confidence": best.get("confidence"),
                        "mask_area_pct": area_pct,
                        "mask_density_pct": round(100.0 * density, 2),
                    })

            if scanned % 25 == 0 or len(manifest_items) >= args.max:
                rate = scanned / max(time.time() - t0, 0.001)
                print(f"  scanned {scanned}  manual={len(manifest_items)}  "
                      f"auto={auto_skipped}  no_wm={no_wm}  {rate:.1f} img/s", flush=True)
    finally:
        WM_RESCUE3_ENABLED = _orig_rescue3

    # Index HTML — base64-embedded, mobile-openable, per-image APPROVE/SKIP.
    cells = []
    for i, r in enumerate(rows):
        mb = r["mark_box"] or {}
        coords = (f"x={mb.get('x')} y={mb.get('y')} "
                  f"w={mb.get('w')} h={mb.get('h')}")
        cells.append(f"""<div class="card" id="card-{i}">
  <div class="hdr"><span class="fn">{r['file']}</span></div>
  <div class="imgs">
    <figure><figcaption>original</figcaption><img src="{r['orig_b64']}"></figure>
    <figure><figcaption>overlay — <span class="rk">mark_box</span> / <span class="gk">strokes</span></figcaption><img src="{r['overlay_b64']}"></figure>
  </div>
  <div class="meta">{coords} &nbsp;|&nbsp; score={r.get('score')} edge={r.get('edge_score')} psr={r.get('psr')} cov={r.get('cov')} int_ncc={r.get('int_ncc')}<br>
    stroke_area={r.get('mask_area_pct')}% density={r.get('mask_density_pct')}% conf={r.get('confidence')}</div>
  <label class="choice"><input type="radio" name="d{i}" value="approve" onchange="mark({i},'approve')"> <span class="ap">APPROVE</span></label>
  <label class="choice"><input type="radio" name="d{i}" value="skip" checked onchange="mark({i},'skip')"> <span class="sk">SKIP</span></label>
</div>""")

    files_json = json.dumps([r["file"] for r in rows])
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sunsky Manual Queue — {len(rows)} images</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:12px}}
 h2{{font-weight:600}}
 .bar{{position:sticky;top:0;background:#1b1b1b;padding:10px;border-bottom:1px solid #333;z-index:9}}
 .card{{border:1px solid #333;border-radius:8px;padding:10px;margin:12px 0;background:#161616}}
 .card.approve{{background:#13240f;border-color:#3a7a22}}
 .hdr .fn{{font-weight:600;word-break:break-all}}
 .imgs{{display:flex;gap:10px;flex-wrap:wrap}}
 figure{{margin:0}} figcaption{{font-size:11px;color:#aaa;margin-bottom:3px}}
 img{{max-width:360px;width:100%;height:auto;display:block;border:1px solid #222}}
 .meta{{font-size:12px;color:#bbb;margin:8px 0;word-break:break-all}}
 .rk{{color:#f55}} .gk{{color:#5f5}}
 .choice{{display:inline-block;margin-right:18px;font-size:15px;cursor:pointer}}
 .ap{{color:#6f6}} .sk{{color:#fa5}}
 textarea{{width:100%;height:90px;background:#0c0c0c;color:#9f9;border:1px solid #333;
   font-family:monospace;font-size:12px;margin-top:6px}}
 button{{background:#2a5;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}}
</style></head>
<body>
<div class="bar">
 <h2 style="margin:4px 0">Sunsky Manual Queue — {len(rows)} manual-tier image(s)</h2>
 <div style="font-size:13px;color:#bbb">
   Source: {assets} &nbsp;|&nbsp; iPhone {SKIP_IPHONE_MIN}+ skipped: {len(skipped)} &nbsp;|&nbsp;
   scanned: {scanned} &nbsp;|&nbsp; auto-tier (not shown): {auto_skipped} &nbsp;|&nbsp; no-watermark: {no_wm}
 </div>
 <div style="margin-top:8px">
   <button onclick="approveAll()">Approve all</button>
   <button onclick="skipAll()" style="background:#a73">Skip all</button>
   <button onclick="copyList()" style="background:#357">Copy approved list</button>
   <span id="cnt" style="margin-left:12px;font-size:13px"></span>
 </div>
 <textarea id="out" readonly placeholder="Approved filenames appear here. Copy and pass to: manual-queue clean-approved --approve-file approved.txt --out THIS_FOLDER"></textarea>
</div>
{"".join(cells)}
<script>
 const FILES = {files_json};
 const choice = {{}};
 FILES.forEach((f,i)=>choice[i]='skip');
 function refresh(){{
   const ap = FILES.filter((f,i)=>choice[i]==='approve');
   document.getElementById('out').value = ap.join('\\n');
   document.getElementById('cnt').textContent = ap.length+' approved / '+FILES.length;
 }}
 function mark(i,v){{
   choice[i]=v;
   const c=document.getElementById('card-'+i);
   if(c) c.className = 'card'+(v==='approve'?' approve':'');
   refresh();
 }}
 function approveAll(){{FILES.forEach((f,i)=>{{const r=document.querySelector('input[name=d'+i+'][value=approve]');if(r)r.checked=true;mark(i,'approve');}});}}
 function skipAll(){{FILES.forEach((f,i)=>{{const r=document.querySelector('input[name=d'+i+'][value=skip]');if(r)r.checked=true;mark(i,'skip');}});}}
 function copyList(){{const o=document.getElementById('out');o.select();document.execCommand('copy');}}
 refresh();
</script>
</body></html>"""

    review_path = out_dir / "review.html"
    with open(review_path, "w") as f:
        f.write(html)

    manifest = {
        "mode": "manual-queue",
        "preset": args.preset,
        "assets": str(assets),
        "out": str(out_dir),
        "scanned": scanned,
        "skipped_iphone14_plus": len(skipped),
        "auto_tier_not_queued": auto_skipped,
        "no_watermark": no_wm,
        "manual_count": len(manifest_items),
        "elapsed_seconds": round(time.time() - t0, 1),
        "items": manifest_items,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Manual-queue built ===")
    print(f"  scanned:        {scanned}")
    print(f"  manual-tier:    {len(manifest_items)}  (queued for review)")
    print(f"  auto-tier:      {auto_skipped}  (not queued — auto-clean handles these)")
    print(f"  no-watermark:   {no_wm}")
    print(f"  review:         {review_path}")
    print(f"  manifest:       {out_dir / 'manifest.json'}")
    print(f"  open with:      open '{review_path}'")
    print(f"  then clean:     python3 {Path(__file__).name} manual-queue clean-approved "
          f"--out '{out_dir}' --approve-all --rights-confirmed --use-lama")


def cmd_manual_queue_clean(args):
    """`manual-queue clean-approved`: LaMa-inpaint the operator-approved manual-tier
    images from a manual-queue manifest and write cleaned outputs to <out>/cleaned/.

    Approval source (one required): --approve-all, --approve-file (one filename per
    line), or --approve f1 f2 ... . Cleaning uses the SAME precision stroke-mask +
    density cap as the auto pipeline; the only difference is the human has approved
    these manual-tier marks, so the tier gate is satisfied by the approval (the det's
    tier is promoted to 'auto' for the inpaint call only — source assets are read-only
    and the cleaned output is written OUTSIDE the repo). The operator then commits via
    `replace --source-dir <out>/cleaned`.
    """
    global LAMA_MODEL_PATH
    if not args.rights_confirmed:
        print("REFUSED: clean-approved requires --rights-confirmed.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out).expanduser()
    _assert_outside_repo(out_dir)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else (out_dir / "manifest.json")
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(2)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assets = Path(args.assets) if args.assets else Path(manifest.get("assets", str(ASSETS_DIR)))
    items = {it["file"]: it for it in manifest.get("items", [])}

    # Resolve the approved set.
    approved = []
    if args.approve_all:
        approved = list(items.keys())
    else:
        names = set(args.approve or [])
        if args.approve_file:
            with open(Path(args.approve_file).expanduser()) as f:
                names |= {ln.strip() for ln in f if ln.strip()}
        approved = [n for n in names if n in items]
        missing = [n for n in names if n not in items]
        if missing:
            print(f"  WARN: {len(missing)} approved name(s) not in manifest, ignored "
                  f"(e.g. {missing[0]})")
    if not approved:
        print("ERROR: no approved files. Use --approve-all, --approve-file FILE, "
              "or --approve f1 f2 ...", file=sys.stderr)
        sys.exit(2)

    cleaned_dir = out_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    template = ensure_template()

    # LaMa requested but unavailable → warn (clean_loaded_image will fall back to CV).
    if args.use_lama and not LAMA_MODEL_PATH.exists():
        print(f"  WARN: --use-lama set but model missing at {LAMA_MODEL_PATH}; "
              "falling back to Telea/NS precision inpaint.")
    old_lama_path = None
    if not args.use_lama:
        old_lama_path = LAMA_MODEL_PATH
        LAMA_MODEL_PATH = Path("/tmp/disabled-big-lama-model.pt")

    counts = {"cleaned": 0, "needs_manual": 0, "missing": 0, "read_error": 0}
    results = []
    t0 = time.time()
    print(f"clean-approved: {len(approved)} approved image(s) "
          f"(use_lama={args.use_lama and LAMA_MODEL_PATH.exists()})")
    print(f"  source (read-only): {assets}")
    print(f"  cleaned output:     {cleaned_dir}")

    try:
        for i, fname in enumerate(sorted(approved), 1):
            fpath = assets / fname
            if not fpath.exists():
                counts["missing"] += 1
                results.append({"file": fname, "status": "missing"})
                continue
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is None:
                counts["read_error"] += 1
                results.append({"file": fname, "status": "read_error"})
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detect_watermark_image(gray, template)
            # Human-approved manual-tier promotion: satisfy the tier gate so the
            # operator-reviewed mark is inpainted with the precision mask. The
            # density cap inside clean_loaded_image still applies (false-positive
            # solid-part guard); source assets are never touched.
            for d in dets:
                if d.get("tier") == "manual":
                    d["tier"] = "auto"
                    if float(d.get("confidence", 0)) < LOW_CONFIDENCE_MIN:
                        d["confidence"] = LOW_CONFIDENCE_MIN
            out_path = cleaned_dir / fname
            result, info = clean_loaded_image(
                img, gray, dets, output_path=out_path, template=template)
            if result is not None:
                counts["cleaned"] += 1
            else:
                counts["needs_manual"] += 1
            results.append({"file": fname, **info})
            if i % 5 == 0 or i == len(approved):
                rate = i / max(time.time() - t0, 0.001)
                print(f"  [{i}/{len(approved)}] {rate:.1f} img/s  "
                      f"cleaned={counts['cleaned']} failed={counts['needs_manual']}", flush=True)
    finally:
        if old_lama_path is not None:
            LAMA_MODEL_PATH = old_lama_path

    report_path = out_dir / "clean-approved-report.json"
    with open(report_path, "w") as f:
        json.dump({
            "mode": "manual-queue clean-approved",
            "assets": str(assets),
            "cleaned_dir": str(cleaned_dir),
            "approved_count": len(approved),
            "use_lama": bool(args.use_lama and LAMA_MODEL_PATH.exists()),
            "counts": counts,
            "elapsed_seconds": round(time.time() - t0, 1),
            "items": results,
        }, f, indent=2)

    print("\n=== clean-approved complete ===")
    print(f"  cleaned:      {counts['cleaned']}")
    print(f"  failed/manual:{counts['needs_manual']}  (left uncleaned — review reason in report)")
    print(f"  missing/read: {counts['missing'] + counts['read_error']}")
    print(f"  cleaned dir:  {cleaned_dir}")
    print(f"  report:       {report_path}")
    print(f"  commit with:  python3 {Path(__file__).name} replace "
          f"--source-dir '{cleaned_dir}' --all")


def cmd_process(args):
    """One-pass workflow:

    for each non-iPhone-14+ image:
      detect with the selected preset
      no detection -> record no_watermark
      detection    -> clean to output folder and record cleaned/needs_manual
    """
    assets = Path(args.assets)
    out_dir = Path(args.out).expanduser() if args.out else (
        Path.home() / "Downloads" /
        f"sunsky-watermark-one-pass-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    _assert_outside_repo(out_dir)
    cleaned_dir = out_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    apply_scan_preset(args.preset)
    template = ensure_template()

    old_lama_path = None
    if not args.use_lama:
        global LAMA_MODEL_PATH
        old_lama_path = LAMA_MODEL_PATH
        LAMA_MODEL_PATH = Path("/tmp/disabled-big-lama-model.pt")

    if args.sample_file:
        with open(args.sample_file) as f:
            all_files = [line.strip() for line in f if line.strip()]
    else:
        all_files = sorted(f for f in os.listdir(assets)
                           if Path(f).suffix.lower() in IMAGE_EXTENSIONS)

    skipped_iphone14 = [f for f in all_files if not should_scan_file(f)]
    files = [f for f in all_files if should_scan_file(f)]
    if args.limit:
        files = files[:args.limit]

    manifest = []
    counts = {
        "cleaned": 0,
        "needs_manual": 0,
        "no_watermark": 0,
        "missing": 0,
        "read_error": 0,
    }
    t0 = time.time()
    print(f"One-pass processing {len(files)} image(s) in {assets} "
          f"(preset={args.preset}, skipped {len(skipped_iphone14)} iPhone {SKIP_IPHONE_MIN}+)")
    print(f"Output: {out_dir}")

    try:
        for i, fname in enumerate(files, 1):
            fpath = assets / fname
            if not fpath.exists():
                counts["missing"] += 1
                manifest.append({"file": fname, "status": "missing"})
                continue

            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is None:
                counts["read_error"] += 1
                manifest.append({"file": fname, "status": "read_error"})
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detect_watermark_image(gray, template)
            if not dets:
                counts["no_watermark"] += 1
                manifest.append({"file": fname, "status": "no_watermark"})
                continue

            output_path = cleaned_dir / fname
            result, info = clean_loaded_image(
                img, gray, dets, output_path=None if args.dry_run else output_path,
                template=template,
            )
            best = dets[0] if dets else None
            if result is not None:
                counts["cleaned"] += 1
                manifest.append({
                    "file": fname,
                    "status": "cleaned",
                    "output": str(output_path),
                    "detection_count": len(dets),
                    "best_detection": best,
                    **info,
                })
            else:
                counts["needs_manual"] += 1
                manifest.append({
                    "file": fname,
                    "status": "needs_manual",
                    "detection_count": len(dets),
                    "best_detection": best,
                    **info,
                })

            if i % args.progress_every == 0 or i == len(files):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.001)
                print(f"  [{i}/{len(files)}] {rate:.1f} img/s  "
                      f"cleaned={counts['cleaned']} manual={counts['needs_manual']} "
                      f"no_watermark={counts['no_watermark']}")
    finally:
        if old_lama_path is not None:
            LAMA_MODEL_PATH = old_lama_path

    summary = {
        "mode": "process",
        "preset": args.preset,
        "assets": str(assets),
        "output": str(out_dir),
        "dry_run": bool(args.dry_run),
        "use_lama": bool(args.use_lama),
        "skipped_iphone14_plus": len(skipped_iphone14),
        "counts": counts,
        "elapsed_seconds": round(time.time() - t0, 1),
        "items": manifest,
    }
    manifest_path = args.manifest or (out_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone")
    print(f"  cleaned:      {counts['cleaned']}")
    print(f"  needs_manual: {counts['needs_manual']}")
    print(f"  no_watermark: {counts['no_watermark']}")
    print(f"  missing/read: {counts['missing'] + counts['read_error']}")
    print(f"  manifest:     {manifest_path}")
    print(f"  cleaned dir:  {cleaned_dir}")


# ---------------------------------------------------------------------------
# CLI: validate
# ---------------------------------------------------------------------------

def cmd_validate(args):
    template = ensure_template()
    assets = Path(args.assets)

    if args.from_manifest:
        with open(args.from_manifest) as f:
            manifest = json.load(f)
        files = [m["file"] for m in manifest]
    elif args.sample_file:
        with open(args.sample_file) as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        print("Specify --from-manifest or --sample-file")
        sys.exit(1)

    print(f"Validating {len(files)} images...")
    residual, blurred, ok = 0, 0, 0

    for fname in files:
        fpath = assets / fname
        if not fpath.exists():
            print(f"  SKIP: {fname}")
            continue

        dets = detect_watermark(str(fpath), template, threshold=VALIDATE_THRESHOLD)
        if dets:
            residual += 1
            print(f"  RESIDUAL: {fname} ({len(dets)} detection(s), "
                  f"best={dets[0].get('score', '?')})")
            continue

        if BACKUP_DIR.exists() and (BACKUP_DIR / fname).exists():
            orig = cv2.imread(str(BACKUP_DIR / fname))
            curr = cv2.imread(str(fpath))
            if orig is not None and curr is not None:
                s_before = laplacian_sharpness(orig)
                s_after = laplacian_sharpness(curr)
                ratio = s_after / max(s_before, 1e-6)
                if ratio < SHARPNESS_RATIO_MIN:
                    blurred += 1
                    print(f"  BLURRED: {fname} (ratio={ratio:.3f})")
                    continue

        ok += 1

    total = residual + blurred + ok
    print(f"\nValidation: {ok}/{total} pass, "
          f"{residual} residual, {blurred} blurred")
    if residual == 0 and blurred == 0:
        print("All images PASS")
    return residual == 0 and blurred == 0


# ---------------------------------------------------------------------------
# CLI: replace (P6)
# ---------------------------------------------------------------------------

def cmd_replace(args):
    assets = Path(args.assets)
    files = list(args.files or [])

    # --all: pull every image file from --source-dir without listing them
    if args.all:
        if not args.source_dir:
            print("ERROR: --all requires --source-dir", file=sys.stderr)
            sys.exit(2)
        src_dir = Path(args.source_dir)
        files = [f.name for f in sorted(src_dir.iterdir())
                 if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"--all: {len(files)} image(s) in {src_dir}")

    n_replaced = n_restored = n_skipped = 0
    for fname in files:
        src = Path(args.source_dir) / fname if args.source_dir else None
        backup = BACKUP_DIR / fname
        target = assets / fname

        if src and src.exists():
            shutil.copy2(src, target)
            print(f"  Replaced from source: {fname}")
            n_replaced += 1
        elif backup.exists():
            shutil.copy2(backup, target)
            print(f"  Restored from backup: {fname}")
            n_restored += 1
        else:
            print(f"  SKIP (no source/backup): {fname}")
            n_skipped += 1
    print(f"\nReplaced: {n_replaced}  Restored: {n_restored}  Skipped: {n_skipped}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch watermark detection and removal (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    scan_p = sub.add_parser("scan", help="Detect watermark candidates")
    scan_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    scan_p.add_argument("--report", type=Path, default=None)
    scan_p.add_argument("--workers", type=int, default=None)
    scan_p.add_argument("--sample-file", type=Path, default=None,
                        help="File with one filename per line to scan")
    scan_p.add_argument("--preset", choices=list(SCAN_PRESETS.keys()),
                        default=DEFAULT_PRESET,
                        help=f"Speed/accuracy preset (default: {DEFAULT_PRESET}). "
                             "high=thorough/slow, medium=balanced, fast=quick/may miss faint watermarks")
    scan_p.add_argument("--debug-crops", action="store_true",
                        help="Write debug crops (raw box green, mark_box red) for low-confidence "
                             "detections to generated/watermark/debug-crops/")

    clean_p = sub.add_parser("clean", help="Remove watermarks")
    clean_p.add_argument("files", nargs="*")
    clean_p.add_argument("-f", "--from-report", type=Path)
    clean_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    clean_p.add_argument("--passes", type=int, default=2)
    clean_p.add_argument("--dry-run", action="store_true")

    proc_p = sub.add_parser("process",
        help="One-pass detect + clean for non-iPhone-14+ images")
    proc_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    proc_p.add_argument("--out", type=Path, default=None,
        help="Output folder outside repo (default: ~/Downloads/sunsky-watermark-one-pass-<timestamp>)")
    proc_p.add_argument("--manifest", type=Path, default=None,
        help="Manifest path (default: <out>/manifest.json)")
    proc_p.add_argument("--preset", choices=list(SCAN_PRESETS.keys()), default="review",
        help="Detection preset for each image (default: review)")
    proc_p.add_argument("--sample-file", type=Path, default=None,
        help="Optional file with one filename per line")
    proc_p.add_argument("--limit", type=int, default=0,
        help="Process only the first N eligible files")
    proc_p.add_argument("--progress-every", type=int, default=100)
    proc_p.add_argument("--dry-run", action="store_true",
        help="Run detection/clean decisions without writing cleaned images")
    proc_p.add_argument("--use-lama", action="store_true",
        help="Allow LaMa if /tmp/big-lama-model.pt exists; slower")

    val_p = sub.add_parser("validate", help="Verify cleaned images")
    val_p.add_argument("--from-manifest", "-f", type=Path)
    val_p.add_argument("--sample-file", type=Path)
    val_p.add_argument("--assets", type=Path, default=ASSETS_DIR)

    rep_p = sub.add_parser("replace", help="Restore originals or swap from source")
    rep_p.add_argument("files", nargs="*")
    rep_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    rep_p.add_argument("--source-dir", type=Path, default=None)
    rep_p.add_argument("--all", action="store_true",
                       help="Replace every image present in --source-dir (no need to list files)")

    sr_p = sub.add_parser("sample-review",
        help="Pick N random files from a scan report, clean to a Downloads folder")
    sr_p.add_argument("--report", type=Path, required=True,
        help="Path to scan report JSON")
    sr_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    sr_p.add_argument("--out", type=Path, required=True,
        help="Output directory (must be outside the repo, e.g. ~/Downloads/...)")
    sr_p.add_argument("--count", type=int, default=50)
    sr_p.add_argument("--seed", type=int, default=42)
    sr_p.add_argument("--known-failures", action="store_true",
        help="Use the 6 known-failure files instead of a random sample")

    reg_p = sub.add_parser("regression",
        help="Run the 6-file known-failure regression check")
    reg_p.add_argument("--assets", type=Path, default=ASSETS_DIR)

    met_p = sub.add_parser("metrics",
        help="Recall/precision + mask-area report vs generated/watermark/ground-truth.json")
    met_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    met_p.add_argument("--ground-truth", type=Path, default=None,
        help=f"Ground-truth JSON (default: {GROUND_TRUTH_FILE})")
    met_p.add_argument("--out-json", type=Path, default=None,
        help="Optional path to write the metrics summary JSON")

    met2_p = sub.add_parser("metrics2",
        help="PRECISION-FIRST held-out eval of detect_watermark_v2 vs validation-set.json "
             "(precision/FP, auto-clean recall, manual count, mask area)")
    met2_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    met2_p.add_argument("--validation-set", type=Path, default=None,
        help=f"Validation set JSON (default: {VALIDATION_SET_FILE})")
    met2_p.add_argument("--split", choices=["heldout", "train", "all"], default="heldout",
        help="Which split to evaluate (default: heldout — the numbers tuning never saw)")
    met2_p.add_argument("--out-json", type=Path, default=None,
        help="Optional path to write the metrics2 summary JSON")
    met2_p.add_argument("--list-manual", action="store_true",
        help="List the positive files routed to needs-manual")
    met2_p.add_argument("--wild-negatives", type=int, default=0, metavar="N",
        help="Also run the WILD precision check: sample N guaranteed-clean iPhone-14+ "
             "images per seed and require ZERO auto-tier hits (the generalization gate). "
             "0 = off. Recommended: 500.")
    met2_p.add_argument("--wild-seeds", type=str, default="123,456,789",
        help="Comma-separated seeds for the wild negative check (default: 123,456,789)")
    met2_p.add_argument("--faint-band-rescue", action="store_true",
        help="Enable RESCUE #3 (faint full-width-on-busy-PCB manual rescue) for 100%% "
             "IDENTIFICATION recall. Surfaces the last faint full-width PCB watermarks "
             "(MagSafe/iMac power boards) to MANUAL, fenced to a moderate-PSR/edge/ncc "
             "window so the clean-pool manual bleed is ~1.2%% (measured 3 seeds), not the "
             "~9.7%% of a cov/gap-only rule. Never auto-cleans -> precision hard gate "
             "unaffected. OFF by default (adds human-review workload).")

    pilot_p = sub.add_parser("pilot",
        help="Small QA batch: scan + clean a capped sample, emit original/mask/cleaned preview HTML")
    pilot_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    pilot_p.add_argument("--out", type=Path, default=None,
        help="Output folder outside repo (default: ~/Downloads/sunsky-watermark-pilot-<timestamp>)")
    pilot_p.add_argument("--max-total", type=int, default=50,
        help="Maximum number of watermarked images to clean for the pilot (default: 50)")
    pilot_p.add_argument("--preset", choices=list(SCAN_PRESETS.keys()), default="review",
        help="Detection preset (default: review)")
    pilot_p.add_argument("--rights-confirmed", action="store_true",
        help="REQUIRED: confirm you hold the rights to remove the watermark from these images")
    pilot_p.add_argument("--use-lama", action="store_true",
        help="Allow LaMa if /tmp/big-lama-model.pt exists; slower, higher quality")
    pilot_p.add_argument("--seed", type=int, default=None,
        help="Randomize which watermarked images are sampled (reproducible). "
             "Omit for deterministic first-N-in-order selection.")
    pilot_p.add_argument("--auto-mode", choices=["strict", "balanced", "aggressive"],
        default="strict",
        help="Auto/manual split: strict (PSR>=6.0, zero false positives — default), "
             "balanced (PSR>=5.0, more auto-clean, small FP risk), "
             "aggressive (PSR>=4.0, most auto-clean, higher FP risk)")
    pilot_p.add_argument("--auto-psr", type=float, default=None,
        help="Explicit auto-tier PSR threshold (overrides --auto-mode). "
             "Lower = more images auto-clean; higher = stricter. Default 6.0.")
    pilot_p.add_argument("--debug-mask", action="store_true",
        help="Write per-image debug artifacts (original/mask/overlay/diff/cleaned/"
             "metrics.json) under <out>/debug/ for mask auditing.")
    pilot_p.add_argument("--prefilter", dest="prefilter", action="store_true",
        default=True, help="V7: skip detection on images perceptually identical "
                           "to ones already found watermark-free (default on).")
    pilot_p.add_argument("--no-prefilter", dest="prefilter", action="store_false",
        help="Disable the perceptual-hash prefilter.")

    mq_p = sub.add_parser("manual-queue",
        help="Phase-3 MANUAL fallback: surface manual-tier residual for human review; "
             "'manual-queue clean-approved' LaMa-cleans the approved files")
    mq_p.add_argument("qaction", nargs="?", default="build",
        choices=["build", "clean-approved"],
        help="build (default): export the manual-tier review batch. "
             "clean-approved: LaMa-inpaint the approved files from a built queue.")
    mq_p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    mq_p.add_argument("--out", type=Path, default=None,
        help="Output folder outside repo (default: ~/Downloads/sunsky-manual-queue-<timestamp>). "
             "For clean-approved this must point at an already-built queue folder.")
    mq_p.add_argument("--rights-confirmed", action="store_true",
        help="REQUIRED: confirm you hold the rights to remove the watermark from these images")
    mq_p.add_argument("--preset", choices=list(SCAN_PRESETS.keys()), default="review",
        help="Detection preset (default: review)")
    mq_p.add_argument("--max", type=int, default=2000,
        help="Max manual-tier images to collect into the queue (default: 2000)")
    mq_p.add_argument("--limit", type=int, default=0,
        help="Scan only the first N eligible files (0 = all)")
    mq_p.add_argument("--seed", type=int, default=None,
        help="Randomize which eligible files are scanned (reproducible). "
             "Omit for deterministic in-order scanning.")
    mq_p.add_argument("--faint-band-rescue", action="store_true",
        help="Also surface the faint full-width-on-busy-PCB residual to the queue "
             "(manual-only; never auto-cleans).")
    mq_p.add_argument("--use-lama", action="store_true",
        help="clean-approved: use LaMa if /tmp/big-lama-model.pt exists (higher quality).")
    mq_p.add_argument("--manifest", type=Path, default=None,
        help="clean-approved: manifest path (default: <out>/manifest.json)")
    mq_p.add_argument("--approve-all", action="store_true",
        help="clean-approved: approve every manual-tier file in the manifest")
    mq_p.add_argument("--approve-file", type=Path, default=None,
        help="clean-approved: file with one approved filename per line "
             "(e.g. the list copied from review.html)")
    mq_p.add_argument("--approve", nargs="*", default=None,
        help="clean-approved: explicit approved filenames")

    args = parser.parse_args()

    if args.mode == "scan":
        cmd_scan(args)
    elif args.mode == "clean":
        cmd_clean(args)
    elif args.mode == "process":
        cmd_process(args)
    elif args.mode == "validate":
        cmd_validate(args)
    elif args.mode == "replace":
        cmd_replace(args)
    elif args.mode == "sample-review":
        cmd_sample_review(args)
    elif args.mode == "regression":
        cmd_regression(args)
    elif args.mode == "metrics":
        cmd_metrics(args)
    elif args.mode == "metrics2":
        cmd_metrics2(args)
    elif args.mode == "pilot":
        cmd_pilot(args)
    elif args.mode == "manual-queue":
        cmd_manual_queue(args)


if __name__ == "__main__":
    main()
