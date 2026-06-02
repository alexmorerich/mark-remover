#!/usr/bin/env python3
"""V22 — Safety-preserving recovery after V21.

V21 moved the result set in the right direction (30 clean_repaired / 3
clean_covered / 17 auto_rejected, every hard-safety counter zero), but the V21
compare PDF still showed three gaps:

1. A published ``clean_repaired`` output could still carry a visible *band* on a
   non-white / product-like surface (the polarising-film / stacked-grey case). On
   a non-white product surface a visible band IS product damage — it must block
   publishing, not be waved through as a cosmetic seam.
2. Several visually "clean" repairs still left faint *partial-glyph* residue
   (``s`` / ``m`` / ``.com`` fragments, paired pits, short dashes) inside the old
   ``sunsky-online.com`` footprint — too weak for OCR / the dot-chain check, but
   still human-visible on smooth cardboard / white / grey / metallic surfaces.
3. The candidate bank kept misclassifying low-texture *product* surfaces (dark
   smooth backs, near-white trays, translucent stacked films, thin flex) as
   background, so destructive cover / fill attempts were generated where they can
   only fail.

V22 does NOT relax any gate. It is purely additive and side-effect free, in the
same spirit as V17–V21:

* :func:`detect_visible_band_on_nonwhite_surface_v22` — a HARD P0 audit for a
  visible band on a non-white / product-like surface (patch plan §Patch 1).
* :func:`detect_alpha_footprint_residue_v22` — an alpha-footprint partial-glyph
  residue detector keyed to the *solved* Sunsky glyph alpha (patch plan §Patch 2).
* :func:`classify_surface_v22` — extra low-texture product-surface signals
  (near-white tray / translucent stack / dark smooth / long thin flex) that the
  plain product mask misses (patch plan §Patch 4).
* :func:`pre_audit_candidate_v22` — a cheap pre-audit used only to reject
  obviously-unsafe candidates early; the V17 final audit stays authoritative
  (patch plan §Patch 8).

Every detector returns a plain dict and fails *closed* (``hard_fail=False`` on any
exception) so it can only ever ADD a publish blocker, never ship a worse output.
This module imports only the frozen :mod:`v13_gates` detectors and the
:mod:`sunsky_reverse_alpha` alpha asset; it must NOT import :mod:`v17_final_audit`
(which imports it) to keep the dependency graph acyclic.
"""
from __future__ import annotations

import cv2
import numpy as np

import v13_gates

try:                                    # the solved glyph alpha footprint asset.
    import sunsky_reverse_alpha as _sra
except Exception:                       # pragma: no cover - always importable
    _sra = None


# ---------------------------------------------------------------------------
# Thresholds (patch plan §Patch 1, §Patch 2, §Patch 4). Kept explicit so the V22
# rules are tunable independently of the frozen V13 / V17 gates.
# ---------------------------------------------------------------------------
# Visible-band-on-non-white-surface (§Patch 1).
BAND_VISIBLE_MIN = 0.50              # v13 band score above => the change is band-like
BAND_RECTANGULARITY_MIN = 0.55      # rectangular / straight-edged changed region
BAND_TONAL_JUMP_MIN = 6.0           # mean luma jump vs the local ring (8-bit)
SURFACE_WHITE_LUMA = 248.0          # median luma below => NOT pure white background
SURFACE_SATURATION_MIN = 18.0       # median saturation above => coloured surface
NONWHITE_ROI = frozenset({
    "metallic_or_reflective", "glass_or_gradient", "transparent_or_glossy",
    "dark_product_surface", "simple_product_surface", "low_texture_background",
    "thin_flex_cable", "translucent_stack_surface", "gradient",
})

# Alpha-footprint partial-glyph residue (§Patch 2).
AF_LUMA_MIN = 3.0                   # min |luma delta| vs local surface to count
AF_LUMA_MAX = 58.0                  # above this it is real product detail, skip
AF_GLYPH_CORE = 0.10                # alpha >= this => glyph core
AF_HALO_LO = 0.02                   # alpha in [lo, core) => halo
AF_MIN_COMPONENTS = 5               # a surviving fragment chain needs several dots
AF_DENSITY_EXCESS = 1.6             # in-footprint density / surrounding ring density
AF_SCORE_HARDFAIL = 0.50           # residue score above => published-residue P0 fail
AF_RESIDUE_AREA_MAX = 0.22          # residue area / footprint above => too much

# Low-texture product-surface classification (§Patch 4).
NEAR_WHITE_LUMA = 235.0             # luma above => looks like (near) white surface
DARK_SMOOTH_LUMA = 80.0             # luma below => dark surface
DARK_SMOOTH_EDGE_MAX = 0.04         # low interior edge density => smooth
TRANSLUCENT_SAT_MAX = 70.0          # low/medium saturation grey-green film
LONG_THIN_ASPECT = 4.0             # bbox-crossing component length/width ratio


# ---------------------------------------------------------------------------
# Small shared helpers.
# ---------------------------------------------------------------------------
def _as_tuple(mark_box):
    if isinstance(mark_box, dict):
        return (int(mark_box["x"]), int(mark_box["y"]),
                int(mark_box["w"]), int(mark_box["h"]))
    return tuple(int(v) for v in mark_box)


def _changed_mask(original, output, bbox, thr=6.0):
    """uint8 (0/255) mask of pixels that actually changed inside a window around
    ``bbox`` — reuses the frozen V13 changed-region helper for parity."""
    try:
        full, _ = v13_gates._changed_region(original, output, bbox, thr=thr)
        return full
    except Exception:
        bx, by, bw, bh = bbox
        H, W = original.shape[:2]
        m = np.zeros((H, W), np.uint8)
        d = cv2.absdiff(original, output).max(axis=2)
        m[d > thr] = 255
        return m


def _ring_luma(image, bbox, pad_frac=0.6):
    """Median luma of a ring just OUTSIDE the bbox (the local surface tone)."""
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x0, y0 = max(0, bx - px), max(0, by - py)
    x1, y1 = min(W, bx + bw + px), min(H, by + bh + py)
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ring = np.ones((y1 - y0, x1 - x0), bool)
    iy0, ix0 = by - y0, bx - x0
    ring[max(0, iy0):iy0 + bh, max(0, ix0):ix0 + bw] = False
    vals = g[y0:y1, x0:x1][ring]
    return float(np.median(vals)) if vals.size else float(np.median(g))


# ---------------------------------------------------------------------------
# Patch 4 — low-texture product-surface classification. The plain product mask
# misses smooth product surfaces that look like background; these signals are
# self-contained estimates the sparse mask cannot fool.
# ---------------------------------------------------------------------------
def classify_surface_v22(image, bbox, product_mask=None, watermark_mask=None,
                         roi_class=""):
    """Classify the surface the watermark sits on. Returns a dict::

        {near_white_product_surface, translucent_stack_surface,
         dark_smooth_product_surface, long_thin_component_crosses_bbox,
         connected_component_crosses_bbox, v22_product_like_overlap,
         surface_class}

    ``v22_product_like_overlap`` is the fraction of the bbox that reads as
    product-like material (non-white / coloured / dark / structured). It is used
    to ban destructive covers on surfaces the plain mask misses (patch plan
    §Patch 5, §Patch 7)."""
    bx, by, bw, bh = _as_tuple(bbox)
    out = {
        "near_white_product_surface": False,
        "translucent_stack_surface": False,
        "dark_smooth_product_surface": False,
        "long_thin_component_crosses_bbox": False,
        "connected_component_crosses_bbox": False,
        "v22_product_like_overlap": 0.0,
        "surface_class": roi_class or "",
    }
    H, W = image.shape[:2]
    if bw < 4 or bh < 4:
        return out
    inner = image[by:by + bh, bx:bx + bw]
    if inner.size == 0:
        return out
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    med_luma = float(np.median(g))
    med_sat = float(np.median(sat))

    # Watermark strokes are SUPPOSED to differ from the surface; exclude them so
    # the mark cannot masquerade as product structure.
    not_wm = np.ones((bh, bw), bool)
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        wm = (watermark_mask[by:by + bh, bx:bx + bw] > 0)
        wm = cv2.dilate(wm.astype(np.uint8), np.ones((3, 3), np.uint8),
                        iterations=2) > 0
        not_wm = ~wm

    edges = cv2.Canny(inner, 50, 150) > 0
    edge_density = float((edges & not_wm).sum()) / float(max(1, not_wm.sum()))

    # product-like overlap: non-white OR coloured OR dark OR structured pixels.
    nonwhite = (g < NEAR_WHITE_LUMA)
    coloured = (sat > SURFACE_SATURATION_MIN)
    dark = (g < DARK_SMOOTH_LUMA)
    product_like = (nonwhite | coloured | dark) & not_wm
    edge_dil = cv2.dilate((edges & not_wm).astype(np.uint8),
                          np.ones((3, 3), np.uint8)) > 0
    product_like = product_like | edge_dil
    if product_mask is not None and product_mask.shape[:2] == (H, W):
        pm = (product_mask[by:by + bh, bx:bx + bw] > 0)
        product_like = product_like | (pm & not_wm)
    overlap = float(product_like.sum()) / float(max(1, not_wm.sum()))
    out["v22_product_like_overlap"] = round(overlap, 4)

    # Does a connected product component cross / touch the box? (silhouette
    # continuity around the mark).
    try:
        sil = v13_gates.detect_product_silhouette_damage_v13(
            image, image, product_mask, (bx, by, bw, bh),
            watermark_mask=watermark_mask)
        out["connected_component_crosses_bbox"] = \
            float(sil.get("product_overlap", 0.0)) > 0.10
    except Exception:
        pass

    # Long thin component crossing the box (flex cable / black frame edge).
    try:
        ov = v13_gates.estimate_product_overlap_v13(image, (bx, by, bw, bh),
                                                    product_mask)
        long_line = float(ov.get("long_line_score", 0.0))
        dark_box = float(ov.get("dark_ratio_in_bbox", 0.0))
        out["long_thin_component_crosses_bbox"] = (
            long_line > v13_gates.LONG_LINE_OVERRIDE and dark_box > 0.10)
    except Exception:
        pass

    # Near-white product surface: looks white, but it is attached to a product
    # body / silhouette continuity crosses the box, so it is NOT pure background.
    if med_luma > NEAR_WHITE_LUMA and (
            out["connected_component_crosses_bbox"] or edge_density > 0.02):
        out["near_white_product_surface"] = True

    # Dark smooth product surface: dark, large connected dark area, low edges.
    dark_ratio = float((dark & not_wm).sum()) / float(max(1, not_wm.sum()))
    if med_luma < DARK_SMOOTH_LUMA and dark_ratio > 0.45 and \
            edge_density < DARK_SMOOTH_EDGE_MAX * 2:
        out["dark_smooth_product_surface"] = True

    # Translucent / stacked film: low-saturation grey/green, parallel straight
    # sheet edges, a large non-white connected area, low-but-nonzero texture.
    grey_green = (sat <= TRANSLUCENT_SAT_MAX) & (g > DARK_SMOOTH_LUMA) & \
        (g < NEAR_WHITE_LUMA)
    grey_frac = float((grey_green & not_wm).sum()) / float(max(1, not_wm.sum()))
    if grey_frac > 0.45 and 0.005 < edge_density < 0.10 and med_sat < TRANSLUCENT_SAT_MAX:
        out["translucent_stack_surface"] = True

    # Resolve a single surface_class label (most specific wins) when ROI is vague.
    if not roi_class or roi_class in ("plain_white", "low_texture_background",
                                      "simple_product_surface", ""):
        if out["translucent_stack_surface"]:
            out["surface_class"] = "translucent_stack_surface"
        elif out["dark_smooth_product_surface"]:
            out["surface_class"] = "dark_smooth_product_surface"
        elif out["near_white_product_surface"]:
            out["surface_class"] = "near_white_product_surface"
        elif out["long_thin_component_crosses_bbox"]:
            out["surface_class"] = "thin_flex_cable"
        else:
            out["surface_class"] = roi_class or (
                "plain_white" if med_luma > NEAR_WHITE_LUMA else "product_surface")
    return out


def surface_is_nonwhite_or_product(image, bbox, product_mask=None,
                                   watermark_mask=None, roi_class=""):
    """True when the surface under the bbox is NOT pure white background — a band
    here is product damage, not a cosmetic seam (patch plan §Patch 1)."""
    bx, by, bw, bh = _as_tuple(bbox)
    inner = image[by:by + bh, bx:bx + bw] if bw > 0 and bh > 0 else image
    if inner.size == 0:
        return False
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    med_luma = float(np.median(g))
    med_sat = float(np.median(hsv[:, :, 1]))
    if med_luma < SURFACE_WHITE_LUMA:
        return True
    if med_sat > SURFACE_SATURATION_MIN:
        return True
    if (roi_class or "") in NONWHITE_ROI:
        return True
    surf = classify_surface_v22(image, bbox, product_mask, watermark_mask,
                                roi_class)
    return bool(surf["near_white_product_surface"] or
                surf["translucent_stack_surface"] or
                surf["dark_smooth_product_surface"] or
                surf["connected_component_crosses_bbox"] or
                surf["v22_product_like_overlap"] > 0.20)


# ---------------------------------------------------------------------------
# Patch 1 — published visible-band hard fail on non-white / product surfaces.
# ---------------------------------------------------------------------------
def detect_visible_band_on_nonwhite_surface_v22(original, output, bbox,
                                                product_mask=None,
                                                watermark_mask=None,
                                                roi_class=""):
    """Return a dict::

        {score, surface_is_nonwhite, surface_is_product_like, band_like,
         rectangularity, tonal_jump, hard_fail, reason}

    ``hard_fail`` is True when the changed region reads as a rectangular tonal
    BAND (high band score + rectangular compactness + a tonal jump vs the local
    ring) AND the original surface is NOT pure white background. A faint band on
    pure white stays cosmetic (allowed); the same band on grey film / black
    frame / metallic / near-white product is product damage and blocks publish.

    Reverse-alpha repairs subtract the glyph and leave a *glyph-shaped*
    (low-rectangularity) change, so they do not trip this; only a slab / band
    fill or cover on a non-white surface does."""
    bx, by, bw, bh = _as_tuple(bbox)
    out = {"score": 0.0, "surface_is_nonwhite": False,
           "surface_is_product_like": False, "band_like": False,
           "rectangularity": 0.0, "tonal_jump": 0.0, "hard_fail": False,
           "reason": "published_visible_band_on_nonwhite_surface"}
    if bw < 6 or bh < 4:
        return out
    H, W = original.shape[:2]

    # 1) Is the changed region band-like?
    try:
        band = v13_gates._band_metrics(output, (bx, by, bw, bh))
        band_score = float(band.get("visible_band_score", 0.0))
        rectangularity = float(band.get("band_rectangularity", 0.0))
    except Exception:
        band_score, rectangularity = 0.0, 0.0
    out["rectangularity"] = round(rectangularity, 4)

    changed = _changed_mask(original, output, (bx, by, bw, bh))
    ch = (changed > 0)
    n_changed = int(ch.sum())
    if n_changed < 12:
        out["score"] = round(band_score, 4)
        return out

    # Compactness of the changed region: a band fills a wide short rectangle.
    ys, xs = np.where(ch)
    rw = int(xs.max() - xs.min() + 1)
    rh = int(ys.max() - ys.min() + 1)
    fill = n_changed / float(max(1, rw * rh))
    compact = max(fill, rectangularity)

    # Tonal jump: mean luma of the changed region vs the local ring.
    og = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
    changed_luma = float(og[ch].mean())
    ring_luma = _ring_luma(output, (bx, by, bw, bh))
    tonal_jump = abs(changed_luma - ring_luma)
    out["tonal_jump"] = round(tonal_jump, 4)

    band_like = (
        (band_score > BAND_VISIBLE_MIN and rectangularity > BAND_RECTANGULARITY_MIN)
        or (compact > 0.72 and tonal_jump > BAND_TONAL_JUMP_MIN and rw > bw * 0.45)
    )
    out["band_like"] = bool(band_like)
    out["score"] = round(max(band_score, compact if band_like else 0.0), 4)

    # 2) Is the surface non-white / product-like?
    nonwhite = surface_is_nonwhite_or_product(
        original, (bx, by, bw, bh), product_mask, watermark_mask, roi_class)
    out["surface_is_nonwhite"] = bool(nonwhite)
    # product-like = the change actually lands on product pixels, or the mask-safe
    # surface estimate says so.
    surf = classify_surface_v22(original, (bx, by, bw, bh), product_mask,
                                watermark_mask, roi_class)
    out["surface_is_product_like"] = bool(
        surf["v22_product_like_overlap"] > 0.20 or
        surf["near_white_product_surface"] or
        surf["dark_smooth_product_surface"] or
        surf["translucent_stack_surface"])

    out["hard_fail"] = bool(band_like and nonwhite)
    return out


# ---------------------------------------------------------------------------
# Patch 2 — alpha-footprint partial-glyph residue detector. Uses the SOLVED glyph
# alpha as the canonical footprint, so it catches faint surviving ``s`` / ``m`` /
# ``.com`` / paired-pit fragments that are too weak for OCR and not shaped like a
# dot chain, but still sit on the original Sunsky baseline.
# ---------------------------------------------------------------------------
def _footprint(original, bbox, watermark_mask):
    if _sra is not None:
        try:
            f = _sra.footprint_mask_for_box(original, bbox, watermark_mask,
                                            dilate_px=1)
            if f is not None:
                return f
        except Exception:
            pass
    # Fallback: the watermark mask crop itself.
    bx, by, bw, bh = _as_tuple(bbox)
    H, W = original.shape[:2]
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        f = np.zeros((H, W), np.uint8)
        f[by:by + bh, bx:bx + bw] = watermark_mask[by:by + bh, bx:bx + bw]
        if int((f > 0).sum()) >= 4:
            return f
    return None


def detect_alpha_footprint_residue_v22(original, output, bbox,
                                       watermark_mask=None, product_mask=None):
    """Return a dict::

        {score, component_count, residue_area_frac, footprint_excess,
         hard_fail, reason}

    Logic (patch plan §Patch 2):
      1. Take the solved glyph footprint (alpha >= core) for ``bbox``.
      2. In the OUTPUT, compute a local high-pass delta vs the local median
         surface and keep only LOW-contrast deviations (|delta| in [min, max]) —
         this excludes real product detail.
      3. Keep small components inside the footprint; score by their density in
         the footprint vs a surrounding ring, their count and their alignment on
         the Sunsky baseline.
      4. ``hard_fail`` when the residue score is above threshold and the residue
         area is small enough to BE residue (not a large product feature)."""
    bx, by, bw, bh = _as_tuple(bbox)
    out = {"score": 0.0, "component_count": 0, "residue_area_frac": 0.0,
           "footprint_excess": 0.0, "hard_fail": False,
           "reason": "published_partial_glyph_residue"}
    if bw < 6 or bh < 4:
        return out
    H, W = original.shape[:2]
    foot = _footprint(original, (bx, by, bw, bh), watermark_mask)
    if foot is None:
        return out
    foot_bool = foot[by:by + bh, bx:bx + bw] > 0
    foot_area = int(foot_bool.sum())
    if foot_area < 8:
        return out

    region = output[by:by + bh, bx:bx + bw]
    if region.size == 0:
        return out
    g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    ksize = max(3, (min(bw, bh) // 2) * 2 + 1)
    ksize = min(ksize, 31)
    surface = cv2.medianBlur(g, ksize)
    hp = cv2.absdiff(g, surface).astype(np.float32)
    resid = ((hp >= AF_LUMA_MIN) & (hp <= AF_LUMA_MAX)).astype(np.uint8)

    # Components inside the footprint that are small (dots / dashes / pits).
    box_area = float(bw * bh)
    comp_max = max(2, int(0.02 * box_area))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(resid, connectivity=8)
    in_foot = []
    keep = np.zeros((bh, bw), np.uint8)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w_i = int(stats[i, cv2.CC_STAT_WIDTH])
        h_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < 2 or a > comp_max:
            continue
        if w_i > bw * 0.5 or h_i > bh * 0.6:
            continue
        cx, cy = cents[i]
        yi, xi = int(round(cy)), int(round(cx))
        if not (0 <= yi < bh and 0 <= xi < bw):
            continue
        if not foot_bool[yi, xi]:
            continue
        in_foot.append((cx, cy, a))
        keep[lab == i] = 255

    cc = len(in_foot)
    out["component_count"] = cc
    edited = int((keep > 0).sum())
    out["residue_area_frac"] = round(edited / float(max(1, foot_area)), 4)
    if cc < AF_MIN_COMPONENTS:
        return out

    # Density of residue inside the footprint vs a surrounding ring (dilate the
    # footprint and subtract). Genuine surviving glyph residue is denser inside.
    foot_full = (foot > 0)
    ring = cv2.dilate(foot_full.astype(np.uint8),
                      np.ones((5, 5), np.uint8), iterations=2) > 0
    ring = ring & (~foot_full)
    resid_full = np.zeros((H, W), np.uint8)
    resid_full[by:by + bh, bx:bx + bw] = resid
    in_density = float((resid_full[foot_full] > 0).mean()) if foot_full.any() else 0.0
    ring_density = float((resid_full[ring] > 0).mean()) if ring.any() else 1e-6
    excess = in_density / max(1e-6, ring_density)
    out["footprint_excess"] = round(excess, 4)

    # Baseline alignment: low vertical spread => sits on the Sunsky baseline.
    ys = np.array([c[1] for c in in_foot], dtype=np.float32)
    vspread = float(ys.std()) / float(max(1.0, bh))
    aligned = vspread < 0.22

    score = 0.0
    if excess >= AF_DENSITY_EXCESS and aligned:
        score = float(np.clip(cc / 12.0, 0.0, 1.0)) * \
            float(np.clip(excess / (AF_DENSITY_EXCESS * 1.5), 0.0, 1.0))
    out["score"] = round(score, 4)

    too_big = out["residue_area_frac"] > AF_RESIDUE_AREA_MAX
    out["hard_fail"] = bool(score >= AF_SCORE_HARDFAIL and not too_big and aligned)
    return out


# ---------------------------------------------------------------------------
# Patch 8 — cheap pre-audit for early candidate rejection. The V17 final audit
# stays authoritative; this only prunes obviously-unsafe candidates before the
# expensive audit so the beam is not wasted ranking them.
# ---------------------------------------------------------------------------
def pre_audit_candidate_v22(original, candidate, bbox, product_mask=None,
                            watermark_mask=None, roi_class="", ctx=None):
    """Return a dict of cheap pre-audit signals + ``reject`` boolean. ``reject``
    is True only for clearly-unsafe candidates (a visible band on a non-white
    surface, or strong alpha-footprint residue); everything else passes through
    to the authoritative V17 audit."""
    res = {"alpha_footprint_residue_score": 0.0,
           "visible_band_on_nonwhite_surface": 0.0,
           "reject": False, "reject_reason": ""}
    if candidate is None or getattr(candidate, "shape", None) != original.shape:
        return res
    try:
        vb = detect_visible_band_on_nonwhite_surface_v22(
            original, candidate, bbox, product_mask, watermark_mask, roi_class)
        res["visible_band_on_nonwhite_surface"] = vb["score"]
        if vb["hard_fail"]:
            res["reject"] = True
            res["reject_reason"] = "visible_band_on_nonwhite_surface"
    except Exception:
        pass
    try:
        af = detect_alpha_footprint_residue_v22(
            original, candidate, bbox, watermark_mask, product_mask)
        res["alpha_footprint_residue_score"] = af["score"]
        if af["hard_fail"] and not res["reject"]:
            res["reject"] = True
            res["reject_reason"] = "alpha_footprint_residue"
    except Exception:
        pass
    return res
