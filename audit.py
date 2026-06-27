"""
audit.py — the Final Audit Agent for the sunsky watermark pipeline.

This is the LAST quality gate before an image is published. It does NOT detect,
mask, repair, or improve anything — the Owner Agent (``v28_clean.py``) is
responsible for making images clean. This agent's only job is to *independently*
decide whether a finished (original → final) pair is safe to publish, and to
REJECT anything that is not.

Design stance (from the spec):

  * Independent. The verdict is re-derived from PIXELS, not trusted from the
    owner's metadata. Owner fields (bbox/mask/roi_type/status) are used only as
    hints for *where to look*; the decision comes from a fresh look at the bytes
    that would actually be written.
  * Multi-signal. No single detector decides. Residual watermark is checked with
    OCR *and* template correlation *and* a dot-chain rhythm detector *and* a
    ghost-text silhouette detector; artifacts, product damage and false-positives
    each combine several measurements.
  * Conservative. When evidence is ambiguous the gate REJECTS. In high-risk ROIs
    (thin flex cable, printed label, metallic/glass/dark surface) a mere warning
    is escalated to a reject. Product safety outranks watermark removal; visual
    quality outranks pass rate.

Output is the Audit Logo-Finder contract: a single ``audit_decision`` class
(PASS / PASS_WITH_MINOR_BACKGROUND_ARTIFACT / REJECT_RESIDUAL_WATERMARK /
REJECT_VISIBLE_PATCH / REJECT_PRODUCT_DAMAGE / REJECT_PROTECTED_TEXT_DAMAGE /
REJECT_UNNATURAL_IMAGE / REJECT_UNCERTAIN), a ``publish_allowed`` boolean, nine
named 0..1 ``scores``, an ``evidence`` array (type/bbox/severity/reason) and a
``recommended_next_action`` (publish / retry_repair / try_cover / auto_reject).

CLI:
  # one pair
  python3 audit.py --original in.jpg --final out.jpg [--meta rec.json] [--mask m.png]
  # batch: a manifest, or v28's own _log.json (records carry path/out)
  python3 audit.py --manifest items.json --out-dir audit_out/ [--html]
  python3 audit.py --v28-log cleaned/_log.json --out-dir audit_out/ [--html]
  # dependency-light proof that the reject paths fire (pure CV, no OCR/LaMa)
  python3 audit.py --selftest
"""
import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_clean as v28e   # cheap import: torch/easyocr stay lazy inside it

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds. Deliberately conservative; every one is a "publish only if clearly
# safe" cut. Tunable from one place.
# ─────────────────────────────────────────────────────────────────────────────
DIFF_THR          = 22      # max-channel abs-diff level that counts as a real edit
MIN_EDIT_FRAC     = 2e-5    # edits smaller than this fraction of the frame are noise
LARGE_EDIT_FRAC   = 0.06    # an edit footprint bigger than ~6% of frame is suspicious

# A. residual watermark
RESID_TPL_HARD    = v28e.CLEAN     # 0.30 template corr on final band => mark structure readable
RESID_TPL_WARN    = v28e.ACCEPT    # 0.24 => possible faint ghost
RESID_SENSITIVE_CONF = 0.10        # leak-fix: min finder-OCR conf for a SUNSKY fragment on textured
                                   # ROI to count as residual (catches 'bnline am'@0.23-class marks)

# B. repair artifact
SEAM_HARD         = 26.0    # mean gradient along the patch perimeter (sharp rectangular seam)
SEAM_WARN         = 18.0
COLOR_DELTA_HARD  = 16.0    # ΔE-ish between inner ring and outer ring across the seam
COLOR_DELTA_WARN  = 10.0
FLAT_RATIO_HARD   = 0.12    # patch interior detail collapsed to <12% of its surroundings => flat block
FLAT_RATIO_WARN   = 0.22
TEX_RATIO_HARD    = 6.0     # |log| texture-energy ratio inside vs ring this large => texture mismatch
TEX_RATIO_WARN    = 3.2

# C. product damage
EDGE_KEEP_HARD    = 0.25    # final kept <25% of the original's structured-edge energy => destroyed
EDGE_KEEP_WARN    = 0.45
HOUGH_MIN_LINES   = 4       # this many long straight lines in the original footprint => real geometry
PTEXT_OVERLAP     = 0.10    # product-text coverage of the footprint that must be preserved

# risk policy
WHITE_BG_LUMA     = 232     # median luma at/above this (and low texture) => low-risk white tray
WHITE_BG_STD      = 14

HIGH_RISK_ROI = {
    "thin_flex_cable", "text_or_label_area", "metallic_or_reflective",
    "glass_or_gradient", "transparent_or_glossy", "dark_product_surface",
    "complex_product_detail",
}

# Surfaces where the NON-OCR residual detectors (template correlation, dot-chain rhythm, ghost
# silhouette) and the visible-patch artifact check are RELIABLE: flat, structure-less trays.
# Everywhere else product structure (flex traces, mesh, connector pins, glossy/metal highlights)
# correlates with these matched filters and yields false REJECT_RESIDUAL / REJECT_VISIBLE_PATCH —
# the over-flag documented in audit-engine-misjudgment-request.md §2. So on a STRUCTURED surface
# OCR (verify_clean composite + per-channel) is the SOLE residual authority and the watermark-
# footprint inpaint is not treated as a visible patch. The OCR residual path, product-damage,
# protected-text and false-positive/off-target checks stay fully active on every surface — this
# narrows the unreliable template-only path, it does not weaken recall.
FLAT_BG_ROI = {"white_background", "near_white_background"}


def _flat_bg(roi):
    return roi in FLAT_BG_ROI


PASS, WARN, FAIL = "pass", "warning", "fail"
_ORD = {PASS: 0, WARN: 1, FAIL: 2}


def _worst(*states):
    return max(states, key=lambda s: _ORD[s])


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def _imread(path):
    try:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    return img


def _gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _detail(gray):
    """Local high-frequency energy = |gray - blur|. Used for flat/texture checks."""
    g = gray.astype(np.float32)
    return cv2.absdiff(g, cv2.GaussianBlur(g, (0, 0), 1.5))


def _light_stroke_frac(bgr, box):
    """Fraction of pixels in ``box`` that look like the watermark's own glyphs: a thin
    stroke that is LIGHTER than its local background by the semi-transparent margin
    (~6..80 levels). This is what makes the template residual *watermark-specific*: a
    flat gray fill or a hard rectangular seam has ~0 such light strokes, so it won't be
    mistaken for a surviving mark, while a real light-gray ghost lights it up."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    g = _gray(bgr[y1:y2, x1:x2]).astype(np.float32)
    h = max(3, (y2 - y1))
    bg = cv2.medianBlur(_gray(bgr[y1:y2, x1:x2]), min(15, max(3, int(0.6 * h) | 1))).astype(np.float32)
    hf = cv2.absdiff(g, cv2.GaussianBlur(g, (0, 0), max(1.0, h * 0.12)))
    light = ((g - bg) > 6) & ((g - bg) < 80) & (hf > 4)
    return float(light.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Edit footprint: what *actually* changed between original and final.
# This is the audit's ground truth, independent of the owner's declared mask.
# ─────────────────────────────────────────────────────────────────────────────
def edit_footprint(orig, final, owner_mask=None):
    """Return (foot_box, foot_mask, diff_box, edit_frac) or (None, ...) if nothing
    meaningful changed. ``diff_box`` is the pure pixel-diff region (truth of what
    changed); ``foot_box`` additionally unions in the owner's declared mask so the
    artifact checks also cover pixels the owner claims to have touched."""
    H, W = orig.shape[:2]
    d = cv2.absdiff(orig, final).max(axis=2)
    raw = (d > DIFF_THR).astype(np.uint8) * 255
    # CLOSE first so thin-but-real changes (1px flex traces / glyph strokes) bridge into
    # blobs; then filter on connected-component PIXEL COUNT (contourArea reads ~0 for a
    # 1px line and would wrongly drop genuine geometry damage).
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    min_area = max(6.0, MIN_EDIT_FRAC * H * W)
    comps = [(stats[i, cv2.CC_STAT_AREA],
              (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
               stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
               stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]))
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    diff_box = None
    if comps:
        # union only components that matter relative to the dominant edit — ignores
        # scattered JPEG-recompression speckle far from the real change.
        big = max(a for a, _ in comps)
        keep = [b for a, b in comps if a >= max(min_area, 0.08 * big)]
        x1 = min(b[0] for b in keep); y1 = min(b[1] for b in keep)
        x2 = max(b[2] for b in keep); y2 = max(b[3] for b in keep)
        diff_box = [int(x1), int(y1), int(x2), int(y2)]

    union = diff_box
    if owner_mask is not None and owner_mask.any():
        ys, xs = np.where(owner_mask > 0)
        mb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        union = mb if union is None else [min(union[0], mb[0]), min(union[1], mb[1]),
                                          max(union[2], mb[2]), max(union[3], mb[3])]
    if union is None:
        return None, None, None, 0.0
    fm = np.zeros((H, W), np.uint8)
    fm[union[1]:union[3], union[0]:union[2]] = 255
    edit_frac = float((d > DIFF_THR).sum()) / (H * W)
    return union, fm, diff_box, edit_frac


def _rings(foot_box, shape, w=None):
    """Inner band (just inside the seam) and outer band (just outside) of the patch.
    Both are where a LaMa seam / colour mismatch / halo shows up."""
    H, W = shape[:2]
    x1, y1, x2, y2 = foot_box
    bw, bh = x2 - x1, y2 - y1
    if w is None:
        w = max(3, int(round(0.10 * min(bw, bh))))
    w = max(2, min(w, 18))
    full = np.zeros((H, W), np.uint8)
    full[y1:y2, x1:x2] = 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * w + 1, 2 * w + 1))
    inner = cv2.subtract(full, cv2.erode(full, k))
    outer = cv2.subtract(cv2.dilate(full, k), full)
    # keep the outer ring inside the image
    return inner, outer, w


# ─────────────────────────────────────────────────────────────────────────────
# ROI classification — risk tier of the surface under the watermark.
# Inferred from the ORIGINAL patch; owner-supplied roi_type wins if present.
# ─────────────────────────────────────────────────────────────────────────────
def classify_roi(orig, foot_box, meta_roi=None):
    if meta_roi:
        return str(meta_roi), True
    x1, y1, x2, y2 = foot_box
    patch = orig[y1:y2, x1:x2]
    if patch.size == 0:
        return "unknown", False
    g = _gray(patch)
    med = float(np.median(g))
    std = float(g.std())
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1].mean())
    edge_frac = float((cv2.Canny(g, 60, 160) > 0).mean())
    # bright + flat => the white/near-white tray the bulk of the catalog uses
    if med >= WHITE_BG_LUMA and std <= WHITE_BG_STD:
        return "white_background", False
    if med >= 205 and std <= 26:
        return "near_white_background", False
    if med <= 70:
        return "dark_product_surface", False
    if edge_frac >= 0.16:
        return "complex_product_detail", False
    if std >= 60 and sat <= 40:
        return "metallic_or_reflective", False
    return "simple_background", False


def is_high_risk(roi):
    return roi in HIGH_RISK_ROI


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


def _resid_band_crop(img, band):
    """Crop a window ~2× text-height around the watermark row, clamped to the image."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in band]
    h = max(8, y2 - y1)
    cy = (y1 + y2) // 2
    by1, by2 = max(0, cy - h), min(H, cy + h)
    bx1, bx2 = max(0, x1 - h), min(W, x2 + h)
    return img[by1:by2, bx1:bx2]


def _faint_mask(crop):
    """Ghost-level deviation strokes with speckle removed. The morphological OPEN is what
    makes this robust to uniform sensor/JPEG noise (single-pixel specks vanish; connected
    glyph strokes survive) — without it, flat noise reads as a dense 'ghost' everywhere."""
    g = _gray(crop).astype(np.float32)
    bg = cv2.medianBlur(_gray(crop), min(15, max(3, (crop.shape[0] // 2) | 1))).astype(np.float32)
    dev = np.abs(g - bg)
    faint = ((dev > 4) & (dev < 30)).astype(np.uint8)
    faint = cv2.morphologyEx(faint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return faint.astype(np.float32)


def dot_chain_score(final, band):
    """Residual 'dot-chain' / repeated-glyph rhythm: a removed watermark often leaves a
    PERIODIC horizontal train of faint marks at the glyph pitch — a periodic peak in the
    autocorrelation of the band's per-column faint-stroke energy. Gated on sparse,
    speckle-free strokes so flat noise and dense product texture don't trip it."""
    crop = _resid_band_crop(final, band)
    if crop.size == 0 or min(crop.shape[:2]) < 8 or crop.shape[1] < 30:
        return 0.0
    faint = _faint_mask(crop)
    cov = float(faint.mean())
    if cov < 0.006 or cov > 0.35:        # nothing, or too dense to be sparse glyph dots
        return 0.0
    prof = faint.sum(axis=0).astype(np.float32)
    if prof.std() < 1e-3:
        return 0.0
    prof = prof - prof.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-6)
    lo = max(3, int(0.4 * max(1, band[3] - band[1])))
    hi = min(len(ac) - 1, crop.shape[1] // 3)
    if hi <= lo:
        return 0.0
    peak = float(np.max(ac[lo:hi]))      # periodicity strength, excluding lag 0
    return round(_clamp01((peak - 0.2) / 0.6), 3)


def ghost_text_score(final, band):
    """Faint transparent-gray text silhouette: speckle-free faint strokes CONCENTRATED on
    one text row (a band that stands out from the rest) and spanning much of the width.
    Concentration (peak row-band vs the band's mean) — not absolute density — is what
    separates a ghost from isotropic noise, which has no standout row."""
    crop = _resid_band_crop(final, band)
    if crop.size == 0 or min(crop.shape[:2]) < 8 or crop.shape[1] < 30:
        return 0.0
    faint = _faint_mask(crop)
    cov = float(faint.mean())
    if cov < 0.006 or cov > 0.5:
        return 0.0
    th = max(3, (band[3] - band[1]) // 2)
    rowden = faint.mean(axis=1)
    if rowden.size < th:
        return 0.0
    win = np.convolve(rowden, np.ones(th) / th, mode="valid")
    if win.size == 0:
        return 0.0
    concentration = float(win.max()) / (float(rowden.mean()) + 1e-6)   # ~1 = flat/noise
    r0 = int(np.argmax(win))
    sub = faint[r0:r0 + th]
    hextent = float((sub.sum(axis=0) > 0).mean()) if sub.size else 0.0
    return round(_clamp01((concentration - 1.4) / 1.6) * _clamp01(hextent / 0.5), 3)


# Residual escalation thresholds for the two shape-based ghost detectors. Set high so a
# faint product texture doesn't masquerade as a residual mark (the canonical-FP lesson).
DOT_HARD, DOT_WARN = 0.55, 0.40
GHOST_HARD, GHOST_WARN = 0.55, 0.38

# ─────────────────────────────────────────────────────────────────────────────
# A. RESIDUAL WATERMARK  — is the mark still visible on the FINAL image?
# Primary gate: fresh multi-modal OCR (the reliable, readable-residue detector).
# Corroborator: template correlation, but only when the matched band actually
# carries watermark-like light strokes — raw correlation over-flags sharp edges
# and clean white (v28 README §3.6), so it never rejects on its own when OCR is
# present and clean. When we are BLIND (no OCR) or on a high-risk surface, a
# tint-corroborated template hit is escalated, per "OCR is not sufficient".
# ─────────────────────────────────────────────────────────────────────────────
def _canonical_band(shape):
    """The sunsky stamp is a FIXED centre mark (cy/H~0.478, ~0.34*W wide). Cropping the sensitive
    residual OCR to this tight centre band — NOT the detected resid_band, which for a leak case is
    derived from the very detection that under-read the mark — is what lets the finder's upscale
    surface a texture-garbled mark: on the full frame it stays sub-0.1, on this band it reads 0.13."""
    H, W = shape[:2]
    return [int(0.33 * W), int(0.42 * H), int(0.67 * W), int(0.56 * H)]


def _sensitive_resid_ocr(final, reader):
    """Detection-grade residual OCR (the finder's multi-pass CLAHE + per-channel) on the canonical
    centre band of the FINAL. Returns (max_sunsky_conf, hit) over SUNSKY-SIGNATURE fragments only.
    On textured/dark surfaces a real mark can garble verify_clean's default pass to nothing while
    this still reads a SUNSKY fragment (e.g. 'unskv Snlinecam'@0.13 on an earpiece mesh). 0.0 when
    nothing sunsky-like is read.

    SIGNATURE GATE (mirrors check_residual_watermark path (1)): detect_smart returns ANY legible band
    text, INCLUDING product captions that happen to sit in the centre band ('through hole design' ->
    'hole desicn'@0.98, 'gold sinking process'). Returning the bare max-conf hit graded EVERY such
    caption as a residual -> a false residual_watermark on clean textured SKUs -> the orchestrator
    emits a retry_hint -> the owner re-inpaints (method=repair_hint) a clean image and DAMAGES it
    (the i2C-spec-sheet false positive). Grade each hit with the finder's STRONG/MED grader and keep
    only signature-consistent reads (strength >= 0.55); this is what makes the "0.0 when nothing
    sunsky-like" promise above actually true. Recall-safe: a genuine mark OCRs to a sunsky/online/.com
    fragment (grade 1.0) and still scores its full conf, so the leak-fix this path exists for holds."""
    crop = _resid_band_crop(final, _canonical_band(final.shape))
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return 0.0, None
    try:
        hits = v28e.detect_smart(reader, crop)
    except Exception:
        return 0.0, None
    import logo_finder as _lf
    sig = [h for h in hits if _lf._text_grade(h[4] or "")[0] >= 0.55]
    if not sig:
        return 0.0, None
    best = max(sig, key=lambda h: h[5])
    return float(best[5]), best


def check_residual_watermark(final, anchor_box, reader, high_risk=False, roi=None):
    reasons, notes = [], []
    state = PASS
    sev = 0.0
    resid_box = None        # set when a residual location is recovered (drives the retry re-inpaint)

    # (1) fresh multi-modal OCR over a region wider than the mask (composite + per-channel)
    ocr_hit = False
    if reader is not None:
        try:
            hits = v28e.verify_clean(reader, final, anchor_box)
        except Exception as e:
            hits = []
            notes.append(f"ocr_check_err:{type(e).__name__}")
        # SIGNATURE GATE: verify_clean returns ANY text found in the band, but a residual SUNSKY mark
        # is a sunsky/online/.com fragment — not a product caption ("gold sinking process") or a UI
        # label ("Stocks"). Grade each hit with the finder's STRONG/MED grader and keep only
        # signature-consistent hits (strength >= 0.55). Without this, non-watermark band text fires a
        # false residual_watermark -> the orchestrator emits a retry hint -> the owner re-inpaints a
        # clean image (method=repair_hint) and DAMAGES it. Recall-safe: a genuine faint mark OCRs to a
        # sunsky fragment (grade >= 0.55) and still FAILs; the (1b) sensitive-OCR fallback below still
        # runs on textured ROIs when nothing signature-like is found.
        import logo_finder as _lf
        sig_hits = [h for h in hits if _lf._text_grade(h[4] or "")[0] >= 0.55]
        if sig_hits:
            ocr_hit = True
            state = FAIL
            reasons.append("residual_watermark")
            sev = max(sev, 1.0 + 0.1 * len(sig_hits))
            txt = ",".join((h[4] or "")[:14] for h in sig_hits[:3])
            notes.append(f"ocr_resid={len(sig_hits)}:{txt}")
        elif hits:
            notes.append(f"ocr_band_text_nonsig({len(hits)}):{(hits[0][4] or '')[:14]}")
    else:
        notes.append("ocr_unavailable")

    # (1b) LEAK-FIX (audit-engine-leak-fix-request.md §3): on a structured surface the template path
    # is suppressed (misjudgment patch) and trusts OCR — but verify_clean is LESS sensitive than the
    # finder's detection OCR and can read nothing on a texture-garbled mark, leaking a visible
    # watermark to PASS. So when verify_clean is empty on textured/dark ROI, re-check the centre band
    # with the finder's sensitive OCR; any SUNSKY fragment at conf >= RESID_SENSITIVE_CONF is a
    # weak-but-present mark => REJECT. Genuinely-clean texture reads nothing sunsky-like and stays PASS.
    if reader is not None and not ocr_hit and not _flat_bg(roi):
        sconf, shit = _sensitive_resid_ocr(final, reader)
        if sconf >= RESID_SENSITIVE_CONF:
            ocr_hit = True
            state = FAIL
            if "residual_watermark" not in reasons:
                reasons.append("residual_watermark")
            sev = max(sev, 1.0 + sconf)
            notes.append(f"sensitive_ocr_resid:{(shit[4] or '')[:14]}@{sconf:.2f}")
        else:
            notes.append(f"sensitive_ocr_clean(max={sconf:.2f})")

    # (2) template correlation, gated on watermark-like light strokes in the match
    tpl_score, tpl_box = 0.0, None
    try:
        tpl_score, tpl_box = v28e.verify_residue(final, anchor_box)
    except Exception as e:
        notes.append(f"template_check_err:{type(e).__name__}")
    light_frac = _light_stroke_frac(final, tpl_box) if tpl_box is not None else 0.0
    watermark_like = 0.01 <= light_frac <= 0.6        # real glyph tint, not a flat/edge artifact
    notes.append(f"tpl={tpl_score:.3f} light={light_frac:.3f}")

    # Template correlation may REJECT on its own only on a flat tray, where it is reliable. On a
    # structured surface the matched filter scores the SAME on clean and watermarked content
    # (request §2), so without OCR corroboration (this block runs only when OCR read nothing) it
    # must NOT reject there — OCR is the authority. The score is still reported for transparency.
    if not ocr_hit and watermark_like and not _flat_bg(roi):
        # Template SEES structure here but is unreliable on a structured surface — it scores the same
        # on clean product texture (request §2), so it is suppressed and cannot reject. BUT a faint
        # PARTIAL residual (e.g. a surviving '.com' tail) that OCR is blind to lives EXACTLY here and
        # would leak to PASS — the Phase-4 metallic leak. When the suppressed signal is non-trivial
        # (tpl>=HARD + watermark-like tint), ARBITRATE with the finder: the pipeline's detection
        # ensemble, run on the SAVED final. It separates a real residual from product texture where
        # template/dot/ghost cannot (0 FP across 181 cleaned finals; the leak read LIKELY@0.43,
        # deterministic) — and auditing the saved bytes closes the gap where the owner's in-memory
        # self-check passes but JPEG re-encode tips detection over threshold. A hit -> retry (a
        # retry_hint drives a region re-inpaint, proven to clear it), NEVER a hard reject on this path.
        if tpl_score >= RESID_TPL_HARD:
            try:
                import logo_finder as _lf
                fr = _lf.find(final, reader)
                if fr.get("presence") in ("CONFIRMED_WATERMARK", "LIKELY_WATERMARK") and fr.get("candidates"):
                    state = FAIL
                    if "residual_watermark" not in reasons:
                        reasons.append("residual_watermark")
                    sev = max(sev, 1.5)
                    resid_box = [int(v) for v in fr["candidates"][0]["_box"]]
                    notes.append(f"finder_resid:{fr['presence']}@{resid_box}")
                else:
                    notes.append(f"finder_arbitrated_clean(tpl={tpl_score:.3f})")
            except Exception as e:
                notes.append(f"finder_check_err:{type(e).__name__}")
        else:
            notes.append(f"template_only_suppressed(roi={roi},tpl={tpl_score:.3f}); OCR-clean => not residual")
    elif not ocr_hit and watermark_like:
        if tpl_score >= 0.42:                          # strong structure + tint => surviving mark
            state = FAIL
            if "residual_watermark" not in reasons:
                reasons.append("residual_watermark")
            sev = max(sev, 1.0 + (tpl_score - 0.42))
            notes.append("template_strong+tint")
        elif tpl_score >= RESID_TPL_HARD:              # ambiguous: warn, escalate only if blind/risky
            esc = (reader is None) or high_risk
            state = _worst(state, FAIL if esc else WARN)
            if esc and "residual_watermark" not in reasons:
                reasons.append("residual_watermark")
            sev = max(sev, 1.0 if esc else 0.6)
            notes.append("template_mid+tint" + ("+escalated" if esc else ""))

    return {"state": state, "severity": sev, "reasons": reasons,
            "template_resid": round(float(tpl_score), 3),
            "light_frac": round(float(light_frac), 3), "notes": notes, "resid_box": resid_box}


# ─────────────────────────────────────────────────────────────────────────────
# B. REPAIR ARTIFACT — seam / flat block / colour mismatch / texture mismatch / halo
# All measured on the patch boundary and interior of the actual edit footprint.
# ─────────────────────────────────────────────────────────────────────────────
def check_repair_artifact(orig, final, foot_box, high_risk, roi=None):
    reasons, notes = [], []
    state = PASS
    sev = 0.0
    H, W = final.shape[:2]
    x1, y1, x2, y2 = foot_box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return {"state": PASS, "severity": 0.0, "reasons": [], "notes": ["footprint_too_small"]}
    if not _flat_bg(roi):
        # The only edit is the watermark-footprint inpaint; on a structured surface its seam /
        # texture / colour discontinuity against the surrounding cable/mesh/metal is the EXPECTED
        # result of removing the mark, not a defect (request §3.2). A genuine smear that destroys
        # product structure is still caught by check_product_damage (active on every surface), so
        # suppress only the visible-patch verdict here — don't blanket-disable artifact checks.
        return {"state": PASS, "severity": 0.0, "reasons": [],
                "notes": [f"patch_check_suppressed_on_structured_roi({roi})"],
                "patch_visibility_score": 0.0, "color_delta_score": 0.0,
                "texture_mismatch_score": 0.0, "bbox": [int(x1), int(y1), int(bw), int(bh)]}

    inner, outer, _ = _rings(foot_box, final.shape)
    fg = _gray(final).astype(np.float32)
    og = _gray(orig).astype(np.float32)

    # -- rectangular seam: gradient energy along the patch perimeter on the FINAL --
    grad = cv2.magnitude(cv2.Sobel(fg, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(fg, cv2.CV_32F, 0, 1, ksize=3))
    seam = float(grad[inner > 0].mean()) if (inner > 0).any() else 0.0
    # ... only meaningful if the ORIGINAL boundary wasn't itself a hard edge there
    og_grad = cv2.magnitude(cv2.Sobel(og, cv2.CV_32F, 1, 0, ksize=3),
                            cv2.Sobel(og, cv2.CV_32F, 0, 1, ksize=3))
    seam_o = float(og_grad[inner > 0].mean()) if (inner > 0).any() else 0.0
    seam_excess = seam - seam_o
    if seam_excess >= SEAM_HARD:
        state = FAIL; reasons.append("rectangular_band")
        sev = max(sev, 1.0 + (seam_excess - SEAM_HARD) / SEAM_HARD)
        notes.append(f"seam_excess={seam_excess:.1f}>=hard")
    elif seam_excess >= SEAM_WARN:
        state = _worst(state, WARN)
        sev = max(sev, 0.5 + (seam_excess - SEAM_WARN) / (SEAM_HARD - SEAM_WARN) * 0.5)
        notes.append(f"seam_excess={seam_excess:.1f}>=warn")

    # -- flat block: did the patch interior detail collapse vs its surroundings? --
    det = _detail(_gray(final))
    cx1, cy1 = x1 + bw // 4, y1 + bh // 4
    cx2, cy2 = x2 - bw // 4, y2 - bh // 4
    interior = det[cy1:cy2, cx1:cx2]
    ring_det = det[outer > 0]
    in_d = float(interior.mean()) if interior.size else 0.0
    ring_d = float(ring_det.mean()) if ring_det.size else 0.0
    flat_ratio = in_d / (ring_d + 1e-6)
    dE = 0.0
    tex_ratio = 1.0
    if ring_d > 1.2:   # only judge flatness against a textured neighbourhood
        if flat_ratio <= FLAT_RATIO_HARD:
            state = FAIL; reasons.append("visible_patch")
            sev = max(sev, 1.0 + (FLAT_RATIO_HARD - flat_ratio))
            notes.append(f"flat_ratio={flat_ratio:.3f}<=hard")
        elif flat_ratio <= FLAT_RATIO_WARN:
            state = _worst(state, WARN)
            sev = max(sev, 0.5 + (FLAT_RATIO_WARN - flat_ratio) / (FLAT_RATIO_WARN - FLAT_RATIO_HARD) * 0.5)
            notes.append(f"flat_ratio={flat_ratio:.3f}<=warn")

    # -- colour mismatch across the seam (inner ring vs outer ring), LAB ΔE --
    lab = cv2.cvtColor(final, cv2.COLOR_BGR2LAB).astype(np.float32)
    if (inner > 0).any() and (outer > 0).any():
        ci = lab[inner > 0].mean(axis=0)
        co = lab[outer > 0].mean(axis=0)
        dE = float(np.linalg.norm(ci - co))
        if dE >= COLOR_DELTA_HARD:
            state = FAIL
            if "visible_patch" not in reasons:
                reasons.append("color_delta_too_high")
            sev = max(sev, 1.0 + (dE - COLOR_DELTA_HARD) / COLOR_DELTA_HARD)
            notes.append(f"color_dE={dE:.1f}>=hard")
        elif dE >= COLOR_DELTA_WARN:
            state = _worst(state, WARN)
            sev = max(sev, 0.5 + (dE - COLOR_DELTA_WARN) / (COLOR_DELTA_HARD - COLOR_DELTA_WARN) * 0.5)
            notes.append(f"color_dE={dE:.1f}>=warn")

    # -- texture mismatch: interior high-freq energy vs the ring (both directions) --
    if ring_d > 1.0 and in_d > 0:
        tex_ratio = max(in_d / (ring_d + 1e-6), ring_d / (in_d + 1e-6))
        if tex_ratio >= TEX_RATIO_HARD:
            # a strong collapse is already caught as flat; this catches the inverse
            # (interior far busier than surroundings => cloned/repeated texture)
            if in_d > ring_d:
                state = FAIL
                if "texture_inconsistent" not in reasons:
                    reasons.append("texture_inconsistent")
                sev = max(sev, 1.0)
                notes.append(f"tex_ratio={tex_ratio:.1f}>=hard")
        elif tex_ratio >= TEX_RATIO_WARN and in_d > ring_d:
            state = _worst(state, WARN)
            notes.append(f"tex_ratio={tex_ratio:.1f}>=warn")

    # high-risk surfaces: a warning here is not safe to publish
    if high_risk and state == WARN:
        state = FAIL
        if not reasons:
            reasons.append("visible_patch")
        sev = max(sev, 1.0)
        notes.append("high_risk_escalate")

    # 0..1 sub-scores for the Logo-Finder contract
    seam_score = _clamp01(seam_excess / SEAM_HARD)
    flat_score = _clamp01((FLAT_RATIO_WARN - flat_ratio) / max(1e-6, FLAT_RATIO_WARN)) if ring_d > 1.2 else 0.0
    return {"state": state, "severity": sev, "reasons": reasons, "notes": notes,
            "patch_visibility_score": round(max(seam_score, flat_score), 3),
            "color_delta_score": round(_clamp01(dE / COLOR_DELTA_HARD), 3),
            "texture_mismatch_score": round(_clamp01(tex_ratio / TEX_RATIO_HARD), 3),
            "bbox": [int(x1), int(y1), int(bw), int(bh)]}


# ─────────────────────────────────────────────────────────────────────────────
# C. PRODUCT DAMAGE — did the repair destroy product structure, geometry or text?
#
# The footprint IS the watermark by construction, so "structure disappeared inside
# the footprint" is the EXPECTED result of a good clean — counting it as damage was
# the v1 false-positive bug. The discriminator: the watermark is a thin line of SHORT
# glyph strokes, whereas product geometry (flex traces, connector pins, frame edges)
# shows up as LONG straight lines. So damage = long lines present in the original
# footprint that vanish in the final; the watermark's own removal contributes none.
# ─────────────────────────────────────────────────────────────────────────────
def _side_details(det, foot_box, w):
    """Mean high-freq detail of the four strips just OUTSIDE the footprint (top, bottom,
    left, right), each separately. Per-side (not pooled) so one adjacent product feature
    on a single edge cannot masquerade as 'the patch is embedded in texture'."""
    H, W = det.shape[:2]
    x1, y1, x2, y2 = foot_box
    strips = {
        "left":  det[y1:y2, max(0, x1 - w):x1],
        "right": det[y1:y2, x2:min(W, x2 + w)],
        "top":   det[max(0, y1 - w):y1, x1:x2],
        "bot":   det[y2:min(H, y2 + w), x1:x2],
    }
    return {k: float(v.mean()) for k, v in strips.items() if v.size}


def check_product_damage(orig, final, foot_box, roi, high_risk):
    """Measured on the FINAL only (the watermark is already gone, so its glyphs can't be
    mistaken for product structure). Damage = the cleaned patch is a SMOOTH HOLE punched
    into a product that is textured ON MOST SIDES (i.e. the patch is *embedded* in
    structure, not merely next to one feature).
      * watermark on flat/smooth bg (white tray, solid back, mirror, cream backlight) ->
        not surrounded by texture -> PASS (the overwhelming common case).
      * watermark over a flex cable / connector that LaMa smeared into a blur -> textured
        on most sides, interior collapsed -> edge_damage / product_damage.
    Requiring texture on >=3 of 4 sides is what stops a single adjacent connector from
    false-flagging a clean fill on an otherwise uniform surface."""
    reasons, notes = [], []
    sev = 0.0
    x1, y1, x2, y2 = foot_box
    bw, bh = x2 - x1, y2 - y1
    w = max(4, int(0.20 * min(bw, bh)))
    det = _detail(_gray(final))
    ix1, iy1 = x1 + bw // 5, y1 + bh // 5
    ix2, iy2 = x2 - bw // 5, y2 - bh // 5
    interior = det[iy1:iy2, ix1:ix2]
    in_d = float(interior.mean()) if interior.size else 0.0
    sides = _side_details(det, foot_box, w)
    TEX = 4.0
    n_textured = sum(1 for v in sides.values() if v >= TEX)
    # reference = median of the textured sides (robust to one smooth edge)
    tex_vals = sorted(v for v in sides.values() if v >= TEX)
    ref = tex_vals[len(tex_vals) // 2] if tex_vals else 0.0
    ratio = in_d / (ref + 1e-6) if ref else 1.0

    edge_state = PASS
    pd_state = PASS
    embedded = n_textured >= 3                 # textured on most sides => patch is *inside* product
    if embedded:
        if ratio <= 0.25:
            edge_state = FAIL
            reasons.append("thin_flex_damage" if roi == "thin_flex_cable" else "edge_damage")
            sev = max(sev, 1.0 + (0.25 - ratio))
            notes.append(f"smooth hole in textured product: in={in_d:.1f} ref={ref:.1f} "
                         f"ratio={ratio:.2f} sides_tex={n_textured}/4")
            if ref >= 8.0:                     # strongly structured product => geometry destroyed
                pd_state = FAIL
                reasons.append("product_damage")
                notes.append("structured_geometry_destroyed")
        elif ratio <= 0.45:
            edge_state = WARN
            notes.append(f"interior smoother than product surround: ratio={ratio:.2f} sides_tex={n_textured}/4")
    else:
        notes.append(f"not embedded in texture (sides_tex={n_textured}/4 "
                     f"{ {k: round(v, 1) for k, v in sides.items()} }); clean fill ok")

    if high_risk and edge_state == WARN:
        edge_state = FAIL
        if not any(r in reasons for r in ("edge_damage", "thin_flex_damage")):
            reasons.append("edge_damage")
        sev = max(sev, 1.0)
        notes.append("high_risk_escalate")

    return {"state": _worst(edge_state, pd_state), "edge_state": edge_state,
            "pd_state": pd_state, "severity": sev, "reasons": reasons,
            "ref_detail": round(ref, 2), "in_detail": round(in_d, 2),
            "sides_textured": n_textured, "notes": notes}


# Additive-damage threshold. Calibrated on the job_50 labelled set: damaged finals (solid dark
# band / re-darken splatter / white wipe) score >= 0.106; clean finals <= 0.051. 0.07 separates
# with ~2x margin. This is the gap check_product_damage (smooth-hole model) and check_repair_artifact
# (flat-bg only) both leave open.
DARK_FILL_FRAC = 0.07


def check_band_darkening(orig, final):
    """The sunsky mark is a LIGHT semi-transparent overlay, so a faithful removal can only lighten
    or leave the band. Substantial DARKENING of a mid/light surface (a solid dark fill or re-darken
    splatter) or a WHITE WIPE of a mid-light surface is ADDED damage — the additive class that the
    smooth-hole product-damage model misses and the flat-bg-only repair-artifact check skips.
    Surface-agnostic; measured over the canonical band so it needs no edit footprint.
    A genuine residual watermark (light grey) produces ~0 here, so this never relaxes the no-leak
    path — it only adds a new FAIL on destructive lightness changes."""
    H = orig.shape[0]
    go = _gray(orig[int(0.40 * H):int(0.58 * H)]).astype(np.int16)
    gf = _gray(final[int(0.40 * H):int(0.58 * H)]).astype(np.int16)
    if go.size == 0:
        return {"state": PASS, "severity": 0.0, "reasons": [], "notes": []}
    d = go - gf
    dark_fill = float(((d >= 25) & (go >= 70) & (gf < 100)).mean())      # mid/light pushed dark
    white_wipe = float(((gf >= 248) & (go > 140) & (go < 240)).mean())   # mid-light wiped to white
    score = max(dark_fill, white_wipe)
    if score >= DARK_FILL_FRAC:
        kind = "dark_fill" if dark_fill >= white_wipe else "white_wipe"
        return {"state": FAIL, "severity": 1.0 + min(1.0, score), "reasons": [f"destructive_{kind}"],
                "notes": [f"additive band damage: dark_fill={dark_fill:.3f} white_wipe={white_wipe:.3f}"]}
    return {"state": PASS, "severity": 0.0, "reasons": [],
            "notes": [f"band_darkening ok (dark={dark_fill:.3f} wipe={white_wipe:.3f})"]}


def check_protected_text(orig, final, foot_box, reader, high_risk):
    """Product text (SKU, "Original", pin numbers, colour names) that the repair
    flattened. Two exclusions keep this from firing on the watermark itself:
      * SUNSKY_TOKENS reads are dropped (that IS the watermark), and
      * text lying mostly INSIDE the edit footprint is dropped — the footprint is the
        masked watermark by construction, so removing text there is the goal. Real
        collateral damage is product text BESIDE the mark (mostly outside the
        footprint) that nonetheless vanished, i.e. an over-wide mask / inpaint bleed.
    """
    reasons, notes = [], []
    state = PASS
    sev = 0.0
    H, W = orig.shape[:2]
    x1, y1, x2, y2 = foot_box
    pad_x = int(0.5 * (y2 - y1)) + 8
    pad_y = int(0.5 * (y2 - y1)) + 8
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)

    def frac_inside_footprint(b):
        ix1, iy1 = max(b[0], x1), max(b[1], y1)
        ix2, iy2 = min(b[2], x2), min(b[3], y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / area

    if reader is None:
        return {"state": PASS, "severity": 0.0, "reasons": [],
                "notes": ["ocr_unavailable: protected-text check skipped"]}

    # read all text (not just watermark) in original vs final, in IMAGE coordinates
    def all_text(img):
        sub = img[cy1:cy2, cx1:cx2]
        if sub.size == 0:
            return []
        prep = v28e.v27.clahe_upscale(sub, 2.0)
        try:
            raw = reader.readtext(cv2.cvtColor(prep, cv2.COLOR_BGR2RGB), detail=1, paragraph=False)
        except Exception as e:
            notes.append(f"ptext_ocr_err:{type(e).__name__}")
            return []
        out = []
        for box, text, conf in raw:
            if conf < 0.25:
                continue
            if v28e.v27.SUNSKY_TOKENS.search(text):   # that's the watermark, not product text
                continue
            pts = np.array(box) / 2.0
            bx1, by1 = pts.min(0); bx2, by2 = pts.max(0)
            out.append((bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1, text.strip(), conf))
        return out

    # only product text BESIDE the mark (mostly outside the footprint) is protectable
    o_text = [b for b in all_text(orig) if frac_inside_footprint(b) < 0.6]
    if not o_text:
        return {"state": PASS, "severity": 0.0, "reasons": [], "notes": notes + ["no_adjacent_product_text"]}
    f_text = all_text(final)

    def matches(b, pool):
        for c in pool:
            ix1, iy1 = max(b[0], c[0]), max(b[1], c[1])
            ix2, iy2 = min(b[2], c[2]), min(b[3], c[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter
            if ua > 0 and inter / ua > 0.2:
                return True
        return False

    lost = [b for b in o_text if not matches(b, f_text)]
    if lost:
        state = FAIL
        reasons.append("protected_text_damage")
        sev = 1.0 + 0.1 * len(lost)
        notes.append("lost_text:" + "|".join(b[4][:12] for b in lost[:3]))
    elif high_risk:
        notes.append("product_text_preserved")
    return {"state": state, "severity": sev, "reasons": reasons, "notes": notes}


# ─────────────────────────────────────────────────────────────────────────────
# D. FALSE-POSITIVE PROCESSING — was the image modified where it shouldn't have been?
# ─────────────────────────────────────────────────────────────────────────────
def check_false_positive(orig, final, foot_box, diff_box, edit_frac, orig_marks, meta):
    """orig_marks: watermark hits independently re-detected on the ORIGINAL (or None if
    detection was unavailable/errored — in which case we cannot assert a false-positive
    from the watermark's absence)."""
    reasons, notes = [], []
    state = PASS
    sev = 0.0
    H, W = orig.shape[:2]

    owner_status = (meta or {}).get("status")
    if owner_status == "no_mark" and diff_box is not None:
        # owner declared the image untouched, yet pixels changed
        state = FAIL
        reasons.append("false_positive_processing")
        sev = 1.0
        notes.append("status=no_mark but pixels changed")

    if diff_box is not None and orig_marks is not None:
        anchor = v28e.v27.union_bbox(orig_marks) if orig_marks else None
        if anchor is None:
            meta_anchor = (meta or {}).get("full_box") or (meta or {}).get("anchor")
            if meta_anchor:
                anchor = [int(v) for v in meta_anchor[:4]]
                notes.append("orig re-detect empty; using owner anchor")
        dcx = (diff_box[0] + diff_box[2]) / 2
        dcy = (diff_box[1] + diff_box[3]) / 2
        if anchor is None:
            # neither we nor the owner can point at a watermark, yet pixels changed.
            # A *large* edit is clearly off-target; a watermark-sized one is only a
            # warning (our detector may simply have missed a faint mark).
            if edit_frac >= LARGE_EDIT_FRAC:
                state = _worst(state, FAIL)
                if "false_positive_processing" not in reasons:
                    reasons.append("false_positive_processing")
                sev = max(sev, 1.0)
                notes.append("no watermark anywhere in original but large edit present")
            elif edit_frac >= 5 * MIN_EDIT_FRAC:
                state = _worst(state, WARN)
                notes.append("edit present but no watermark re-detected in original")
        else:
            h = max(10, anchor[3] - anchor[1])
            off_x = max(anchor[0] - dcx, dcx - anchor[2], 0)
            off_y = max(anchor[1] - dcy, dcy - anchor[3], 0)
            if off_x > 2.0 * h * v28e.ASPECT or off_y > 3.0 * h:
                state = _worst(state, FAIL)
                reasons.append("off_target_repair")
                sev = max(sev, 1.0)
                notes.append(f"edit center far from watermark (off_x={off_x:.0f},off_y={off_y:.0f})")

    # an edit far larger than a watermark footprint (~1%) is suspicious on its own
    if edit_frac >= LARGE_EDIT_FRAC:
        st = FAIL if edit_frac >= 2 * LARGE_EDIT_FRAC else WARN
        state = _worst(state, st)
        if st == FAIL and "unnecessary_modification" not in reasons:
            reasons.append("unnecessary_modification")
        sev = max(sev, 1.0 if st == FAIL else 0.6)
        notes.append(f"edit_frac={edit_frac:.3f} large")

    return {"state": state, "severity": sev, "reasons": reasons, "notes": notes}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration: one (original, final[, meta, mask]) → spec audit record.
# ─────────────────────────────────────────────────────────────────────────────
_SCORE_KEYS = ("residual_logo_score", "dot_chain_score", "ghost_text_score",
               "patch_visibility_score", "product_damage_score", "texture_mismatch_score",
               "color_delta_score", "edge_damage_score", "protected_text_damage_score")


def _empty_scores():
    return {k: 0.0 for k in _SCORE_KEYS}


def _reject(reason, decision="REJECT_UNCERTAIN", action="auto_reject", notes=""):
    """Mandatory reject for missing/corrupt/invalid inputs — safety cannot be verified,
    so it lands as REJECT_UNCERTAIN with publish_allowed False."""
    return {"audit_decision": decision, "publish_allowed": False,
            "scores": _empty_scores(), "evidence": [],
            "recommended_next_action": action, "reject_reason": reason,
            "notes": notes or reason}


def locate_original_watermark(orig, meta, foot_box, reader):
    """Where is/was the watermark? Returns (residual_band, orig_marks). The residual
    search must cover the WHOLE mark — including any part the owner LEFT untouched (a
    half-clean leaves the '.com' tail where final==original, so the diff footprint does
    NOT cover it). So we localize from the owner's declared extent UNION an independent
    re-detection on the ORIGINAL, falling back to the edit footprint, then the frame."""
    cands = []
    ma = meta.get("full_box") or meta.get("anchor")
    if ma:
        cands.append([int(v) for v in ma[:4]])
    orig_marks = None
    if reader is not None:
        try:
            orig_marks = v28e.detect_smart(reader, orig) or []
        except Exception:
            orig_marks = None
    if orig_marks:
        cands.append([int(v) for v in v28e.v27.union_bbox(orig_marks)])
    if not cands and foot_box is not None:
        cands.append(list(foot_box))
    if not cands:
        band = [0, 0, orig.shape[1], orig.shape[0]]
    else:
        band = [min(c[0] for c in cands), min(c[1] for c in cands),
                max(c[2] for c in cands), max(c[3] for c in cands)]
    return band, orig_marks


def audit_pair(original_path, final_path, meta=None, mask_path=None, reader=None):
    meta = meta or {}
    # ---- mandatory input gate ----
    if not final_path or not os.path.exists(str(final_path)):
        return _reject("missing_final_image", notes="final image absent")
    final = _imread(final_path)
    if final is None:
        return _reject("corrupted_image", notes="final image unreadable")
    if not original_path or not os.path.exists(str(original_path)):
        return _reject("missing_metadata", notes="original image absent — cannot verify independently")
    orig = _imread(original_path)
    if orig is None:
        return _reject("corrupted_image", notes="original image unreadable")
    if orig.shape[:2] != final.shape[:2]:
        if abs(orig.shape[1] / orig.shape[0] - final.shape[1] / final.shape[0]) > 0.02:
            return _reject("invalid_mask", notes="original/final aspect mismatch")
        final = cv2.resize(final, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)

    owner_mask = None
    if mask_path and os.path.exists(str(mask_path)):
        owner_mask = cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE)
        if owner_mask is not None and owner_mask.shape[:2] != orig.shape[:2]:
            owner_mask = cv2.resize(owner_mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

    foot_box, foot_mask, diff_box, edit_frac = edit_footprint(orig, final, owner_mask)

    # Localize the watermark band from the ORIGINAL + owner meta (covers parts the owner
    # left untouched, which the diff footprint misses). Shared by residual + false-positive.
    resid_band, orig_marks = locate_original_watermark(orig, meta, foot_box, reader)

    dot = dot_chain_score(final, resid_band)
    ghost = ghost_text_score(final, resid_band)

    # ---- nothing was changed ----
    if foot_box is None:
        # No edit. Safe to publish IFF the original is itself watermark-free in the band.
        roi0, _ = classify_roi(orig, resid_band, meta.get("roi_type"))
        rw = check_residual_watermark(final, resid_band, reader, high_risk=is_high_risk(roi0), roi=roi0)
        rec = _decide({"residual_watermark": rw}, roi=roi0, high_risk=is_high_risk(roi0),
                      dot=dot, ghost=ghost, reader_available=reader is not None,
                      foot_box=resid_band, edited=False)
        rec["edit_frac"] = 0.0
        rec["foot_box"] = None
        return rec

    roi, roi_from_meta = classify_roi(orig, foot_box, meta.get("roi_type"))
    high_risk = is_high_risk(roi)

    results = {
        "residual_watermark": check_residual_watermark(final, resid_band, reader, high_risk, roi=roi),
        "repair_artifact":    check_repair_artifact(orig, final, foot_box, high_risk, roi=roi),
        "product_damage":     check_product_damage(orig, final, foot_box, roi, high_risk),
        "band_darkening":     check_band_darkening(orig, final),
        "protected_text":     check_protected_text(orig, final, foot_box, reader, high_risk),
        "false_positive":     check_false_positive(orig, final, foot_box, diff_box, edit_frac, orig_marks, meta),
    }
    rec = _decide(results, roi=roi, high_risk=high_risk, dot=dot, ghost=ghost,
                  reader_available=reader is not None, foot_box=foot_box)
    rec["roi_type"] = roi
    rec["roi_inferred"] = not roi_from_meta
    rec["edit_frac"] = round(edit_frac, 5)
    rec["foot_box"] = foot_box
    return rec


def _decide(results, roi, high_risk, dot=0.0, ghost=0.0, reader_available=True, foot_box=None, edited=True):
    """Fuse the detectors into the Audit Logo-Finder contract: 9 named scores, one
    decision class, structured evidence, and a recommended next action. Decision priority
    is safety-first — residual watermark → product damage → protected text → visible patch
    → unnatural → uncertain → (minor bg artifact) → pass."""
    rw = results.get("residual_watermark", {})
    ra = results.get("repair_artifact", {})
    pd = results.get("product_damage", {})
    bd = results.get("band_darkening", {})
    pt = results.get("protected_text", {})
    fp = results.get("false_positive", {})
    edge_state, pd_state = pd.get("edge_state", PASS), pd.get("pd_state", PASS)

    scores = {
        "residual_logo_score":        round(_clamp01(rw.get("severity", 0.0)), 3),
        "dot_chain_score":            round(float(dot), 3),
        "ghost_text_score":           round(float(ghost), 3),
        "patch_visibility_score":     float(ra.get("patch_visibility_score", 0.0)),
        "product_damage_score":       round(max(_clamp01(pd.get("severity", 0.0)) if pd_state == FAIL else 0.0,
                                                 _clamp01(bd.get("severity", 0.0)) if bd.get("state") == FAIL else 0.0), 3),
        "texture_mismatch_score":     float(ra.get("texture_mismatch_score", 0.0)),
        "color_delta_score":          float(ra.get("color_delta_score", 0.0)),
        "edge_damage_score":          round(1.0 if edge_state == FAIL else (0.6 if edge_state == WARN else 0.0), 3),
        "protected_text_damage_score": round(_clamp01(pt.get("severity", 0.0)), 3),
    }

    # dot-chain and ghost are matched-filter shape detectors: like template correlation they fire
    # on periodic product structure (mesh, connector pins, flex traces) and are trustworthy only on
    # a flat tray. Off a flat background they are diagnostic-only (still reported in `scores`) and
    # cannot drive a REJECT or even a minor-artifact downgrade — OCR is the residual authority there
    # (request §2/§3). On a flat tray they keep full force.
    flat = _flat_bg(roi)
    dot_eff = dot if flat else 0.0
    ghost_eff = ghost if flat else 0.0
    residual_fail = (rw.get("state") == FAIL) or dot_eff >= DOT_HARD or ghost_eff >= GHOST_HARD
    product_fail = (pd_state == FAIL) or (edge_state == FAIL) or (bd.get("state") == FAIL)
    patch_fail = ra.get("state") == FAIL
    ptext_fail = pt.get("state") == FAIL
    unnatural_fail = fp.get("state") == FAIL
    # uncertain = we can't confidently verify a REPAIR on a risky surface (OCR off). It is
    # about the edit, so it does not apply when nothing was modified.
    uncertain = edited and (not reader_available) and high_risk and not residual_fail

    def ev(typ, bbox, score, reason):
        sev = "high" if score >= 0.85 else ("medium" if score >= 0.5 else "low")
        return {"type": typ, "bbox": [int(v) for v in (bbox if bbox else [0, 0, 0, 0])],
                "severity": sev, "reason": reason}

    evidence = []
    if residual_fail:
        evidence.append(ev("residual_watermark", foot_box, max(scores["residual_logo_score"], dot_eff, ghost_eff),
                           "watermark trace on final: " + (";".join(rw.get("notes", [])[:2])
                                                           or f"dot_chain={dot:.2f} ghost={ghost:.2f}")))
    if product_fail:
        evidence.append(ev("product_damage", foot_box, max(scores["product_damage_score"], scores["edge_damage_score"]),
                           "product structure/edges destroyed in repair patch"))
    if ptext_fail:
        evidence.append(ev("protected_text_damage", foot_box, scores["protected_text_damage_score"],
                           "product text lost beside the mark: " + ";".join(pt.get("notes", [])[:1])))
    if patch_fail:
        evidence.append(ev("visible_patch", ra.get("bbox", foot_box), scores["patch_visibility_score"],
                           "visible repair patch / seam / colour mismatch"))
    if scores["texture_mismatch_score"] >= 0.85:
        evidence.append(ev("texture_mismatch", ra.get("bbox", foot_box), scores["texture_mismatch_score"],
                           "patch interior texture inconsistent with surroundings"))
    if unnatural_fail:
        evidence.append(ev("visible_patch", foot_box, 0.9, "off-target / unnecessary modification: "
                           + (";".join(fp.get("notes", [])[:2]) or "edit not justified by a watermark")))

    if residual_fail:
        decision, action = "REJECT_RESIDUAL_WATERMARK", "retry_repair"
    elif product_fail:
        decision, action = "REJECT_PRODUCT_DAMAGE", "auto_reject"
    elif ptext_fail:
        decision, action = "REJECT_PROTECTED_TEXT_DAMAGE", "auto_reject"
    elif patch_fail:
        decision, action = "REJECT_VISIBLE_PATCH", ("auto_reject" if high_risk else "try_cover")
    elif unnatural_fail:
        decision, action = "REJECT_UNNATURAL_IMAGE", "auto_reject"
    elif uncertain:
        decision, action = "REJECT_UNCERTAIN", "auto_reject"
    elif (ra.get("state") == WARN or scores["patch_visibility_score"] >= 0.5
          or dot_eff >= DOT_WARN or ghost_eff >= GHOST_WARN or scores["color_delta_score"] >= 0.5):
        decision, action = "PASS_WITH_MINOR_BACKGROUND_ARTIFACT", "publish"
    else:
        decision, action = "PASS", "publish"

    note_bits = []
    for nm, blk in (("resid", rw), ("artifact", ra), ("product", pd), ("banddark", bd), ("ptext", pt), ("fp", fp)):
        if blk.get("notes"):
            note_bits.append(f"{nm}[" + ";".join(blk["notes"][:2]) + "]")
    notes = f"roi={roi} high_risk={high_risk} dot={dot:.2f} ghost={ghost:.2f}. " + " ".join(note_bits)

    out = {"audit_decision": decision, "publish_allowed": decision in ("PASS", "PASS_WITH_MINOR_BACKGROUND_ARTIFACT"),
           "scores": scores, "evidence": evidence, "recommended_next_action": action, "notes": notes}
    # Tell the Owner WHERE the mark still is so the retry does a region-targeted re-inpaint instead of
    # re-detecting (the first pass leaked precisely because detection under-read it). Prefer the finder's
    # recovered box (set on the structured arbitration path); fall back to the edit footprint.
    if decision == "REJECT_RESIDUAL_WATERMARK" and action == "retry_repair":
        hint_box = rw.get("resid_box") or foot_box
        if hint_box:
            out["retry_hint"] = {"box": [int(v) for v in hint_box]}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def write_reports(records, out_dir, html=False):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "audit_results.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    passed = sum(1 for r in records if r.get("publish_allowed"))
    rejected = len(records) - passed
    class_counts = {}
    for r in records:
        d = r["audit_decision"]
        class_counts[d] = class_counts.get(d, 0) + 1
    summary = {
        "total": len(records),
        "passed": passed,
        "rejected": rejected,
        "reject_rate": round(rejected / len(records), 4) if records else 0.0,
        "decision_classes": dict(sorted(class_counts.items(), key=lambda kv: -kv[1])),
    }
    with open(os.path.join(out_dir, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "audit_failures.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "decision", "next_action", "evidence_types", "roi_type", "notes"])
        for r in records:
            if not r.get("publish_allowed"):
                ev_types = ";".join(e["type"] for e in r.get("evidence", []))
                w.writerow([r.get("id", ""), r["audit_decision"], r.get("recommended_next_action", ""),
                            ev_types, r.get("roi_type", ""),
                            r.get("notes", "").replace("\n", " ")[:300]])

    if html:
        _write_html(records, out_dir)
    return summary


def _thumb_b64(path, width=300):
    img = _imread(path) if path else None
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf).decode() if ok else ""


def _write_html(records, out_dir):
    rows = []
    for r in records:
        color = "#1b7f37" if r.get("publish_allowed") else "#c0392b"
        o = _thumb_b64(r.get("original_path"))
        fi = _thumb_b64(r.get("final_path"))
        oimg = f'<img src="data:image/jpeg;base64,{o}">' if o else "<i>orig n/a</i>"
        fimg = f'<img src="data:image/jpeg;base64,{fi}">' if fi else "<i>final n/a</i>"
        ev = "<br>".join(f'{e["type"]} ({e["severity"]})' for e in r.get("evidence", [])) or "—"
        rows.append(
            f'<tr><td>{r.get("id","")}</td>'
            f'<td style="color:{color};font-weight:700">{r["audit_decision"]}'
            f'<br><small>{r.get("recommended_next_action","")}</small></td>'
            f'<td>{oimg}</td><td>{fimg}</td>'
            f'<td><b>{ev}</b><br><small>roi={r.get("roi_type","?")}</small><br>'
            f'<small>{r.get("notes","")[:260]}</small></td></tr>')
    html = (
        "<!doctype html><meta charset=utf-8><title>Audit compare</title>"
        "<style>body{font:13px system-ui;margin:18px}table{border-collapse:collapse}"
        "td{border:1px solid #ddd;padding:6px;vertical-align:top}img{display:block;max-width:300px}"
        "th{background:#222;color:#fff;padding:6px}</style>"
        "<h2>Final Audit — before / after</h2><table>"
        "<tr><th>id</th><th>verdict</th><th>original</th><th>final</th><th>reasons / notes</th></tr>"
        + "".join(rows) + "</table>")
    with open(os.path.join(out_dir, "compare.html"), "w") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# Batch drivers
# ─────────────────────────────────────────────────────────────────────────────
def _lazy_reader(no_ocr):
    if no_ocr:
        return None
    try:
        return v28e.make_reader()
    except Exception as e:
        print(f"[audit] OCR unavailable ({type(e).__name__}: {str(e)[:80]}); "
              f"running template+CV signals only.", file=sys.stderr)
        return None


def run_manifest(items, out_dir, html=False, no_ocr=False):
    """items: list of dicts with at least {original, final}; optional meta/mask/id."""
    reader = _lazy_reader(no_ocr)
    records = []
    for i, it in enumerate(items):
        rid = it.get("id") or it.get("slug") or Path(str(it.get("final", i))).stem
        meta = it.get("meta")
        if meta is None and it.get("meta_path") and os.path.exists(it["meta_path"]):
            meta = json.load(open(it["meta_path"]))
        t0 = time.time()
        try:
            rec = audit_pair(it.get("original"), it.get("final"), meta=meta,
                             mask_path=it.get("mask"), reader=reader)
        except Exception as e:
            # never let one bad image stall the orchestrated batch — fail safe to a
            # publishable=False auto-reject so the run completes and the orchestrator
            # can move on (it halts only if an item is MISSING from audit_results).
            rec = {"audit_decision": "REJECT_UNCERTAIN", "publish_allowed": False,
                   "scores": _empty_scores(), "evidence": [],
                   "recommended_next_action": "auto_reject",
                   "notes": f"audit_error: {type(e).__name__}: {str(e)[:200]}"}
        rec["id"] = rid
        rec["original_path"] = it.get("original")
        rec["final_path"] = it.get("final")
        rec["elapsed_s"] = round(time.time() - t0, 2)
        records.append(rec)
        ev = ",".join(e["type"] for e in rec.get("evidence", [])) or "clean"
        print(f"[{i+1}/{len(items)}] {rid} -> {rec['audit_decision']} "
              f"[{rec.get('recommended_next_action')}] {ev} ({rec['elapsed_s']}s)", flush=True)
    summary = write_reports(records, out_dir, html=html)
    print(f"\nAUDIT DONE: {summary['passed']}/{summary['total']} pass, "
          f"{summary['rejected']} reject (rate {summary['reject_rate']}). "
          f"reports in {out_dir}", flush=True)
    return records, summary


def items_from_v28_log(log_path):
    """Adapt v28's own --log JSON (list of clean_one records) into audit items.
    Each record has path (original), out (final), and is itself the meta."""
    recs = json.load(open(log_path))
    items = []
    for r in recs:
        if not r.get("ok") or not r.get("out"):
            continue
        items.append({"id": r.get("slug") or Path(r["path"]).stem,
                      "original": r["path"], "final": r["out"], "meta": r})
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: prove the reject paths fire with pure synthetic CV (no OCR/LaMa).
# ─────────────────────────────────────────────────────────────────────────────
def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="audit_selftest_")
    fails = []

    def save(name, img):
        p = os.path.join(tmp, name)
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 96])[1].tofile(p)
        return p

    def expect(label, rec, want, reason_substr=None):
        # want is "PASS" / "REJECT" (publish gate) or a specific decision class
        if want == "REJECT":
            ok = not rec["publish_allowed"]
        elif want == "PASS":
            ok = rec["publish_allowed"]
        else:
            ok = rec["audit_decision"] == want
        if reason_substr:
            blob = (rec["audit_decision"] + " " + rec.get("reject_reason", "") + " "
                    + " ".join(e["type"] for e in rec.get("evidence", [])))
            ok = ok and reason_substr in blob
        flag = "ok " if ok else "FAIL"
        if not ok:
            fails.append(label)
        ev = ",".join(e["type"] for e in rec.get("evidence", [])) or "clean"
        print(f"  [{flag}] {label:34} -> {rec['audit_decision']:34} {ev}")

    rng = np.random.default_rng(7)

    # densely textured product surface (cross-hatched traces on a dark slab) on a white tray.
    # Cross-hatch (both axes) so a central edit is embedded in structure on all four sides — the
    # condition under which check_product_damage (the structural backstop that stays active on
    # textured ROI) can recognise a smooth hole punched into the product.
    def base_scene():
        img = np.full((400, 600, 3), 245, np.uint8)
        img[150:250, 180:430] = (90, 95, 100)               # a dark product slab
        for x in range(188, 426, 10):                       # connector-trace verticals
            cv2.line(img, (x, 154), (x, 246), (180, 185, 190), 1)
        for y in range(156, 246, 10):                       # cross traces -> textured on all sides
            cv2.line(img, (184, y), (428, y), (168, 174, 182), 1)
        img = (img.astype(np.int16) + rng.integers(-4, 5, img.shape)).clip(0, 255).astype(np.uint8)
        return img

    print("audit.py self-test (synthetic, no OCR):")

    # 1) clean white-tray edit, faithfully filled -> PASS
    orig = np.full((400, 600, 3), 246, np.uint8)
    orig = (orig.astype(np.int16) + rng.integers(-3, 4, orig.shape)).clip(0, 255).astype(np.uint8)
    final = orig.copy()
    final[60:90, 220:380] = (245, 246, 246)                 # subtle, blends with tray
    final = (final.astype(np.int16) + rng.integers(-3, 4, final.shape)).clip(0, 255).astype(np.uint8)
    expect("clean white-tray fill", audit_pair(save("o1.jpg", orig), save("f1.jpg", final), reader=None), "PASS")

    # 2) gray rectangular patch on a textured surface -> REJECT visible_patch/seam
    orig = base_scene()
    final = orig.copy()
    final[170:230, 250:360] = (120, 122, 124)               # flat gray block
    expect("gray rectangular patch", audit_pair(save("o2.jpg", orig), save("f2.jpg", final), reader=None),
           "REJECT")

    # 3) product structure destroyed: the repair smeared the cross-hatched traces inside the
    #    patch into a smooth hole while the surround stays textured. On a textured ROI the
    #    visible-patch/seam signal is intentionally suppressed (misjudgment fix), so this MUST be
    #    caught by check_product_damage (the structural backstop) -> REJECT_PRODUCT_DAMAGE.
    orig = base_scene()
    final = orig.copy()
    final[186:228, 250:372] = cv2.GaussianBlur(final[186:228, 250:372], (0, 0), 9)
    expect("product structure destroyed", audit_pair(save("o3.jpg", orig), save("f3.jpg", final), reader=None),
           "REJECT_PRODUCT_DAMAGE")

    # 4) off-target / unnecessary: a big edit on a plain image with no watermark
    orig = np.full((400, 600, 3), 246, np.uint8)
    final = orig.copy()
    final[40:240, 60:360] = (200, 180, 150)                 # huge unrelated change
    expect("large unrelated modification", audit_pair(save("o4.jpg", orig), save("f4.jpg", final), reader=None),
           "REJECT")

    # 5) no change at all -> PASS (nothing to publish-block; residual check is no-op w/o OCR)
    orig = base_scene()
    expect("no modification", audit_pair(save("o5.jpg", orig), save("f5.jpg", orig.copy()), reader=None), "PASS")

    # 6) missing final -> REJECT (mandatory gate)
    expect("missing final image", audit_pair(save("o6.jpg", base_scene()), os.path.join(tmp, "nope.jpg"), reader=None),
           "REJECT", "missing_final_image")

    # 7) corrupted final -> REJECT
    bad = os.path.join(tmp, "bad.jpg"); open(bad, "wb").write(b"not an image")
    expect("corrupted final image", audit_pair(save("o7.jpg", base_scene()), bad, reader=None),
           "REJECT", "corrupted_image")

    print(f"\nself-test: {7 - len(fails)}/7 expectations met" +
          (f"  FAILURES: {fails}" if fails else "  ✓ all reject/pass paths fire"))
    return len(fails) == 0


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Final Audit Agent — independent publish gate")
    ap.add_argument("--original"); ap.add_argument("--final")
    ap.add_argument("--meta"); ap.add_argument("--mask")
    ap.add_argument("--manifest", help="JSON list of {original,final[,meta,mask,id]}")
    ap.add_argument("--v28-log", help="v28 --log JSON; audits every cleaned record")
    ap.add_argument("--out-dir", default="audit_out")
    ap.add_argument("--html", action="store_true", help="also write compare.html")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR signals (template+CV only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.manifest or args.v28_log:
        items = (items_from_v28_log(args.v28_log) if args.v28_log
                 else json.load(open(args.manifest)))
        run_manifest(items, args.out_dir, html=args.html, no_ocr=args.no_ocr)
        return

    if args.original and args.final:
        meta = json.load(open(args.meta)) if args.meta and os.path.exists(args.meta) else None
        reader = _lazy_reader(args.no_ocr)
        rec = audit_pair(args.original, args.final, meta=meta, mask_path=args.mask, reader=reader)
        rec["original_path"] = args.original
        rec["final_path"] = args.final
        print(json.dumps(rec, indent=2))
        return

    ap.error("need --selftest, --manifest, --v28-log, or --original/--final")


if __name__ == "__main__":
    main()
