#!/usr/bin/env python3
"""V17 — Truthful final-output audit + product-aware generator gating.

V16's invariant was right ("publish only when the watermark is gone and the
product is undamaged, else auto_rejected") but the *implementation* let two
classes of bad output escape:

1. A candidate could pass the per-candidate gate, then be mutated by cleanup /
   cover polish / status conversion, and the *actual published bytes* were never
   re-audited. V17 audits the real final output.
2. ``visible_patch`` / ``visible_band`` were treated as purely cosmetic. A faint
   seam on a white background is cosmetic; the same seam **on product pixels** is
   product damage. V17 makes the cosmetic verdict context-aware: cosmetic only
   when the changed region is confirmed pure background.

This module is intentionally additive and side-effect free. It reuses the
existing V13 detectors (single source of truth for the visual signals) and
returns a :class:`FinalAuditResult`. The pipeline treats a failed audit as
authoritative: a clean_* status whose audit fails P0 is forced to
``auto_rejected``.

Design principle (patch plan §14): do NOT weaken the gate. Make the final
published output subject to truthful visual auditing, and refuse to paint
rectangular bands over product pixels. Prefer auto_rejected over a dishonest
clean result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import v13_gates


# ---------------------------------------------------------------------------
# Thresholds (patch plan §1.2, §4.1, §5). Kept here so the audit's product-aware
# rules are explicit and tunable independently of the candidate-stage gate.
# ---------------------------------------------------------------------------
PRODUCT_OVERLAP_HARD = 0.03          # > this on a cosmetic fail => product damage
CHANGED_EDGE_DENSITY_MAX = 0.03      # structure inside the changed region
PROTECTED_TEXT_OVERLAP_MAX = 0.10    # watermark over real product text
SILHOUETTE_PRODUCT_OVERLAP = 0.03    # min product to call a silhouette break real
DOT_CHAIN_READABLE = 0.45            # residual dot-chain score that reads as text
NEW_DARK_BLOB_DELTA = 28.0           # luma jump that reads as a blob on dark stock
METALLIC_BOUNDARY_JUMP = 22.0        # hard luma step across a metallic surface


# Hard P0 reasons V17 introduces (patch plan §4.2). Any of these on a published
# output is a release blocker — CI must exit non-zero.
P0_RESIDUAL_REASONS = (
    "published_residual_watermark",
    "published_dot_chain",
)
P0_PRODUCT_REASONS = (
    "visible_patch_on_product",
    "visible_band_on_product",
    "changed_product_silhouette",
    "changed_thin_flex_structure",
    "changed_protected_text",
    "changed_dark_surface_blob",
    "changed_metallic_surface_block",
)

# V18 — tiny low-contrast glyph residue (patch plan §8). A published output may
# still show faint dot/dash residue aligned along the original watermark
# baseline; it is too low-contrast for the residual-OCR detector but reads to a
# human. This is a HARD fail, reported separately so CI can track it.
P0_RESIDUE_REASONS = (
    "published_low_contrast_glyph_residue",
)

# Low-contrast glyph-residue thresholds (patch plan §8).
GLYPH_RESIDUE_LUMA_MIN = 4.0         # min |luma delta| vs surface to count a dot
GLYPH_RESIDUE_LUMA_MAX = 60.0        # above this it is real product detail, not residue
GLYPH_RESIDUE_MIN_COMPONENTS = 4     # a chain needs several aligned dots
GLYPH_RESIDUE_ALIGN_TOL = 0.18       # vertical spread / band height to read as a line
GLYPH_RESIDUE_BAND_EXCESS = 1.8      # in-band component density / ring density


@dataclass
class FinalAuditResult:
    """Verdict on the ACTUAL final output bytes (not a candidate)."""
    pass_p0: bool = True
    hard_fail_reasons: list = field(default_factory=list)
    residual_score: float = 0.0
    dot_chain_score: float = 0.0
    template_residual_score: float = 0.0
    product_damage_score: float = 0.0
    silhouette_delta: float = 0.0
    visible_patch_score: float = 0.0
    visible_band_score: float = 0.0
    changed_product_ratio: float = 0.0
    pure_background_change: bool = True
    has_readable_residual: bool = False
    glyph_residue_score: float = 0.0           # V18 (patch plan §8)

    def to_record(self) -> dict:
        return {
            "v17_pass_p0": self.pass_p0,
            "v17_hard_fail_reasons": list(self.hard_fail_reasons),
            "v18_glyph_residue_score": round(self.glyph_residue_score, 4),
            "v17_residual_score": round(self.residual_score, 4),
            "v17_dot_chain_score": round(self.dot_chain_score, 4),
            "v17_template_residual_score": round(self.template_residual_score, 4),
            "v17_product_damage_score": round(self.product_damage_score, 4),
            "v17_silhouette_delta": round(self.silhouette_delta, 4),
            "v17_visible_patch_score": round(self.visible_patch_score, 4),
            "v17_visible_band_score": round(self.visible_band_score, 4),
            "v17_changed_product_ratio": round(self.changed_product_ratio, 4),
            "v17_pure_background_change": self.pure_background_change,
            "v17_has_readable_residual": self.has_readable_residual,
        }


# ---------------------------------------------------------------------------
# Geometry helpers — the CHANGED region and its product overlap.
# ---------------------------------------------------------------------------
def _changed_mask(original, output, bbox, thr=6.0):
    """uint8 (0/255) mask of pixels that actually changed, inside a padded
    window around bbox. Re-uses the V13 changed-region helper for parity."""
    full, _ = v13_gates._changed_region(original, output, bbox, thr=thr)
    return full


def _wm_exclusion(shape, watermark_mask):
    """Boolean mask of pixels NOT belonging to the (dilated) watermark, so the
    watermark's OWN strokes are never mistaken for product structure."""
    H, W = shape[:2]
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        wm = (watermark_mask > 0).astype(np.uint8)
        wm = cv2.dilate(wm, np.ones((3, 3), np.uint8), iterations=2) > 0
        return ~wm
    return np.ones((H, W), dtype=bool)


def changed_region_product_overlap(original, output, bbox, product_mask,
                                   watermark_mask=None):
    """Fraction of the CHANGED pixels that land on product, and the edge density
    of the original *inside* that changed region. The watermark's own strokes
    are excluded — they are supposed to change. These two signals decide whether
    a cosmetic seam is on background (safe) or on product (damage)."""
    changed = _changed_mask(original, output, bbox)
    ch = (changed > 0)
    n_changed = int(ch.sum())
    if n_changed < 8:
        return 0.0, 0.0
    not_wm = _wm_exclusion(original.shape, watermark_mask)
    if product_mask is not None and product_mask.shape[:2] == original.shape[:2]:
        prod_ratio = float((ch & (product_mask > 0) & not_wm).sum()) / \
            float(n_changed)
    else:
        prod_ratio = 0.0
    # Structure of the ORIGINAL within the changed footprint, EXCLUDING the
    # watermark strokes — a real edge there means the change sits on product
    # detail, not on empty background.
    og = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(og, 50, 150) > 0
    edge_density = float((edges & ch & not_wm).sum()) / float(n_changed)
    return prod_ratio, edge_density


def _protected_text_overlap(original, bbox, watermark_mask):
    """Approximate fraction of the bbox occupied by real (non-watermark) printed
    text — high-contrast strokes that are NOT inside the watermark mask."""
    bx, by, bw, bh = bbox
    if bw <= 1 or bh <= 1:
        return 0.0
    o = original[by:by + bh, bx:bx + bw]
    if o.size == 0:
        return 0.0
    og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32)
    surface = float(np.median(og))
    high = (np.abs(og - surface) > 45).astype(np.uint8)
    if watermark_mask is not None and \
            watermark_mask.shape[:2] == original.shape[:2]:
        wm = (watermark_mask[by:by + bh, bx:bx + bw] > 0).astype(np.uint8)
        high = cv2.bitwise_and(high, 1 - wm)
    return float(high.mean())


# ---------------------------------------------------------------------------
# THE FINAL AUDIT — runs on the actual published bytes (patch plan §3).
# ---------------------------------------------------------------------------
def audit_final_output(original, output, mark_box, product_mask, watermark_mask,
                       *, final_status="clean_repaired", still_present=False,
                       visible_patch_failed=False, visible_band_failed=False,
                       roi_class=""):
    """Audit the ACTUAL final output. Returns a :class:`FinalAuditResult` whose
    ``pass_p0`` is False when the published output would either still show the
    watermark or damage the product.

    Parameters
    ----------
    still_present : bool
        Authoritative post-clean watermark re-detection (detector re-run).
    visible_patch_failed / visible_band_failed : bool
        Whether the candidate-stage cosmetic gates flagged a seam. V17 decides
        here whether that seam is on background (cosmetic, allowed) or on product
        (hard fail).
    """
    bx, by, bw, bh = mark_box
    bbox = (bx, by, bw, bh)
    res = FinalAuditResult()
    reasons: list = []

    if bw <= 1 or bh <= 1:
        return res

    # ----- 1. Residual watermark on the published output (P0). -------------
    try:
        dc = v13_gates.detect_residual_dot_chain_v2(
            original, output, bbox, watermark_mask)
    except Exception:
        dc = {"dot_chain_score": 0.0, "passed": True}
    res.dot_chain_score = float(dc.get("dot_chain_score", 0.0))
    dot_chain_readable = (not dc.get("passed", True)) or \
        res.dot_chain_score > DOT_CHAIN_READABLE

    # The detector re-detect is the AUTHORITATIVE residual signal (template
    # correlation false-passes on bright metal); still_present comes from it.
    res.has_readable_residual = bool(still_present) or dot_chain_readable
    res.residual_score = 1.0 if still_present else res.dot_chain_score
    if still_present:
        reasons.append("published_residual_watermark")
    elif dot_chain_readable:
        reasons.append("published_dot_chain")

    # V18 — faint low-contrast glyph residue aligned on the watermark baseline
    # (patch plan §8). A HARD fail: too faint for residual-OCR but human-visible.
    try:
        gr = detect_low_contrast_glyph_residue_v18(
            original, output, bbox, watermark_mask)
    except Exception:
        gr = {"residue_score": 0.0, "passed": True}
    res.glyph_residue_score = float(gr.get("residue_score", 0.0))
    if not gr.get("passed", True):
        reasons.append("published_low_contrast_glyph_residue")

    # ----- 2. Where did the change land? (background vs product) -----------
    prod_ratio, changed_edge_density = changed_region_product_overlap(
        original, output, bbox, product_mask, watermark_mask)
    res.changed_product_ratio = prod_ratio
    ptext_overlap = _protected_text_overlap(original, bbox, watermark_mask)

    # ----- 3. Silhouette / product-structure damage (P0). ------------------
    try:
        sil = v13_gates.detect_product_silhouette_damage_v13(
            original, output, product_mask, bbox, watermark_mask=watermark_mask)
    except Exception:
        sil = {"passed": True, "product_overlap": 0.0,
               "contour_break_score": 0.0, "new_bright_blob_score": 0.0,
               "dark_surface_ratio": 0.0}
    res.silhouette_delta = float(sil.get("contour_break_score", 0.0))
    res.product_damage_score = float(sil.get("new_bright_blob_score", 0.0))
    sil_product_overlap = float(sil.get("product_overlap", 0.0))
    touches_silhouette = (not sil.get("passed", True)) and \
        sil_product_overlap > SILHOUETTE_PRODUCT_OVERLAP

    # ----- 4. Pure-background test (patch plan §4.1). ----------------------
    overlaps_protected_text = ptext_overlap > PROTECTED_TEXT_OVERLAP_MAX
    res.pure_background_change = (
        prod_ratio < PRODUCT_OVERLAP_HARD
        and changed_edge_density < CHANGED_EDGE_DENSITY_MAX
        and not touches_silhouette
        and not overlaps_protected_text
    )

    # ----- 5. Cosmetic seams become HARD failures on product (patch §4). ---
    try:
        patch = v13_gates.detect_visible_patch_shape_v13(original, output, bbox)
        res.visible_patch_score = float(patch.get("visible_patch_score", 0.0))
        band = v13_gates._band_metrics(output, bbox)
        res.visible_band_score = float(band.get("visible_band_score", 0.0))
    except Exception:
        pass

    if visible_patch_failed and not res.pure_background_change:
        reasons.append("visible_patch_on_product")
    if visible_band_failed and not res.pure_background_change:
        reasons.append("visible_band_on_product")

    # ----- 6. Direct structural-damage detectors (do not rely on cosmetic
    #          flags alone — a destructive fill can damage product without
    #          tripping the patch/band shape gates). --------------------------
    if touches_silhouette:
        reasons.append("changed_product_silhouette")

    if overlaps_protected_text and _protected_text_erased(
            original, output, bbox, watermark_mask):
        reasons.append("changed_protected_text")

    # Thin flex cable: a continuous dark line broken by the fill.
    if roi_class == "thin_flex_cable" or _looks_like_flex(original, bbox):
        if _flex_line_broken(original, output, bbox, product_mask):
            reasons.append("changed_thin_flex_structure")

    # Dark product surface: a new bright/dark blob on dark stock.
    if (sil.get("dark_surface_ratio", 0.0) > v13_gates.DARK_SURFACE_RATIO_FOR_BLOB
            and prod_ratio > PRODUCT_OVERLAP_HARD
            and _dark_surface_blob(original, output, bbox) > NEW_DARK_BLOB_DELTA):
        reasons.append("changed_dark_surface_blob")

    # Metallic / reflective: a flat block where a gradient used to be.
    if roi_class == "metallic_or_reflective" and prod_ratio > PRODUCT_OVERLAP_HARD:
        if _metallic_block(original, output, bbox):
            reasons.append("changed_metallic_surface_block")

    # De-dupe while preserving order.
    seen = set()
    res.hard_fail_reasons = [r for r in reasons
                             if not (r in seen or seen.add(r))]
    res.pass_p0 = len(res.hard_fail_reasons) == 0
    return res


# ---------------------------------------------------------------------------
# V18 — low-contrast glyph residue (patch plan §8). Catches faint dot/dash
# chains the residual-OCR detector misses: small low-contrast components aligned
# along the original watermark baseline, spaced at glyph cadence, denser inside
# the watermark band than in the surrounding ring. Only fires when the pattern
# is aligned with the original mark — natural product texture is not punished.
# ---------------------------------------------------------------------------
def detect_low_contrast_glyph_residue_v18(original, output, bbox,
                                          watermark_mask=None):
    bx, by, bw, bh = bbox
    if bw <= 4 or bh <= 4:
        return {"residue_score": 0.0, "component_count": 0,
                "aligned": False, "band_excess": 0.0, "passed": True}
    out = output[by:by + bh, bx:bx + bw]
    if out.size == 0:
        return {"residue_score": 0.0, "component_count": 0,
                "aligned": False, "band_excess": 0.0, "passed": True}
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Local surface estimate via a large median blur; residue = small deviations.
    ksize = max(3, (min(bw, bh) // 2) * 2 + 1)
    ksize = min(ksize, 31)
    surface = cv2.medianBlur(g.astype(np.uint8), ksize).astype(np.float32)
    delta = np.abs(g - surface)
    glyph = ((delta > GLYPH_RESIDUE_LUMA_MIN) &
             (delta < GLYPH_RESIDUE_LUMA_MAX)).astype(np.uint8)
    # Exclude the watermark's own (dilated) strokes — they are expected to differ
    # if the fill is imperfect, but we want the RESIDUE that survived, so we keep
    # them; instead exclude large blobs that are clearly product detail.
    n, labels, stats, cents = cv2.connectedComponentsWithStats(glyph, connectivity=8)
    comps = []
    area_box = float(bw * bh)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w_i = int(stats[i, cv2.CC_STAT_WIDTH])
        h_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        # dot/dash sized: small area, not a long product edge.
        if a < 2 or a > area_box * 0.02:
            continue
        if w_i > bw * 0.5 or h_i > bh * 0.6:
            continue
        comps.append((cents[i][0], cents[i][1], a))
    cc = len(comps)
    if cc < GLYPH_RESIDUE_MIN_COMPONENTS:
        return {"residue_score": 0.0, "component_count": cc,
                "aligned": False, "band_excess": 0.0, "passed": True}
    ys = np.array([c[1] for c in comps], dtype=np.float32)
    # Aligned along a baseline => low vertical spread relative to box height.
    vspread = float(ys.std()) / float(max(1.0, bh))
    aligned = vspread < GLYPH_RESIDUE_ALIGN_TOL
    # Density inside the central watermark band vs the top/bottom ring.
    band_lo, band_hi = bh * 0.30, bh * 0.70
    in_band = float(np.mean((ys >= band_lo) & (ys <= band_hi)))
    ring_frac = 1.0 - in_band
    band_excess = in_band / max(0.05, ring_frac)
    score = 0.0
    if aligned:
        score = float(np.clip(cc / 12.0, 0.0, 1.0)) * \
            float(np.clip(band_excess / GLYPH_RESIDUE_BAND_EXCESS, 0.0, 1.0))
    passed = not (aligned and band_excess >= GLYPH_RESIDUE_BAND_EXCESS and
                  cc >= GLYPH_RESIDUE_MIN_COMPONENTS)
    return {"residue_score": round(score, 4), "component_count": cc,
            "aligned": bool(aligned), "band_excess": round(band_excess, 4),
            "passed": bool(passed)}


# ---------------------------------------------------------------------------
# Small structural probes used by the audit.
# ---------------------------------------------------------------------------
def _protected_text_erased(original, output, bbox, watermark_mask):
    try:
        pt = v13_gates.detect_protected_product_text_v13(
            original, output, bbox, watermark_mask)
        return bool(pt.get("protected_text_erased", False))
    except Exception:
        return False


def _looks_like_flex(image, bbox):
    """Heuristic: a long thin dark line crosses the bbox (cable / connector)."""
    try:
        ov = v13_gates.estimate_product_overlap_v13(image, bbox)
        return (ov.get("long_line_score", 0.0) > v13_gates.LONG_LINE_OVERRIDE and
                ov.get("dark_ratio_in_bbox", 0.0) > 0.15)
    except Exception:
        return False


def _flex_line_continuity(image, bbox):
    """Length of the longest straight dark line through the bbox (px)."""
    bx, by, bw, bh = bbox
    inner = image[by:by + bh, bx:bx + bw]
    if inner.size == 0:
        return 0.0
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(10, bw // 6),
                            minLineLength=max(10, bw // 3), maxLineGap=4)
    longest = 0.0
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            longest = max(longest, float(np.hypot(x2 - x1, y2 - y1)))
    return longest


def _flex_line_broken(original, output, bbox, product_mask, ratio=0.95):
    before = _flex_line_continuity(original, bbox)
    if before < max(8.0, (bbox[2]) / 3.0):
        return False
    after = _flex_line_continuity(output, bbox)
    return after < before * ratio


def _dark_surface_blob(original, output, bbox):
    """Max luma delta introduced on dark (<80) pixels of the original."""
    bx, by, bw, bh = bbox
    o = original[by:by + bh, bx:bx + bw]
    r = output[by:by + bh, bx:bx + bw]
    if o.size == 0 or o.shape != r.shape:
        return 0.0
    og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rg = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = og < 80
    if int(dark.sum()) < 20:
        return 0.0
    return float(np.abs(rg[dark] - og[dark]).mean())


def _metallic_block(original, output, bbox):
    """A metallic surface reads as damaged when the texture variance drops
    sharply AND a hard luma step appears on the bbox boundary."""
    try:
        patch = v13_gates.detect_visible_patch_shape_v13(original, output, bbox)
    except Exception:
        return False
    texture_drop = float(patch.get("texture_drop", 0.0))
    hard_boundary = float(patch.get("hard_boundary_score", 0.0))
    luma_max = float(patch.get("luma_delta_max", 0.0))
    return (texture_drop > 0.5 and
            (hard_boundary > v13_gates.HARD_BOUNDARY_MAX or
             luma_max > METALLIC_BOUNDARY_JUMP / 255.0 * 100.0))


# ---------------------------------------------------------------------------
# Generator gating — restrict uniform_background_fill to STRICT pure background
# only (patch plan §1.3). On any product signal, this fill must not run.
# ---------------------------------------------------------------------------
def allow_uniform_background_fill(image, bbox, product_mask, watermark_mask,
                                  *, ring_uniform=None):
    """True only when a full-footprint uniform background fill is provably safe:
    no product overlap, no interior structure, no long lines, no protected text,
    a uniform surrounding ring, and the box not touching the product silhouette.
    Any failure routes to segmented / stroke-only repair or auto-reject instead.
    """
    bx, by, bw, bh = bbox
    if bw <= 1 or bh <= 1:
        return False
    try:
        ov = v13_gates.estimate_product_overlap_v13(image, bbox, product_mask)
    except Exception:
        return False

    long_line = float(ov.get("long_line_score", 1.0))
    protected_text = _protected_text_overlap(image, bbox, watermark_mask)

    # Product overlap and interior edge density measured EXCLUDING the watermark
    # strokes (the mark's own pixels must not masquerade as product). With a real
    # product_mask, this is exact; without one we fall back to non-white pixels
    # outside the watermark footprint.
    not_wm = _wm_exclusion(image.shape, watermark_mask)
    inner = np.zeros(image.shape[:2], bool)
    inner[by:by + bh, bx:bx + bw] = True
    inner &= not_wm
    n_inner = int(inner.sum())
    if n_inner < 16:
        # The watermark fills the whole box — nothing but mark to judge; rely on
        # the product_mask overlap and the uniform-ring precondition.
        if product_mask is not None and product_mask.shape[:2] == image.shape[:2]:
            product_overlap = float((product_mask[by:by + bh, bx:bx + bw] > 0)
                                    .mean())
        else:
            product_overlap = 0.0
        edge_density = 0.0
    else:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g, 50, 150) > 0
        edge_density = float((edges & inner).sum()) / float(n_inner)
        if product_mask is not None and product_mask.shape[:2] == image.shape[:2]:
            product_overlap = float(((product_mask > 0) & inner).sum()) / \
                float(n_inner)
        else:
            product_overlap = float((g[inner] < 235).mean())

    if ring_uniform is None:
        try:
            import v15_patch
            ring_uniform = v15_patch.is_uniform_local_background(
                image, bbox, product_mask)
        except Exception:
            ring_uniform = False

    # Does the box touch the product silhouette?
    try:
        sil = v13_gates.detect_product_silhouette_damage_v13(
            image, image, product_mask, bbox, watermark_mask=watermark_mask)
        touches_silhouette = float(sil.get("product_overlap", 0.0)) > 0.15
    except Exception:
        touches_silhouette = product_overlap > 0.15

    # long_line is the normalised (0..1) longest-line score from
    # estimate_product_overlap_v13. Any appreciable straight line in the box
    # (cable / frame / connector) vetoes a full-footprint fill.
    return bool(
        product_overlap < PRODUCT_OVERLAP_HARD
        and edge_density < 0.04
        and long_line < 0.15
        and protected_text <= 1e-6
        and ring_uniform
        and not touches_silhouette
    )
