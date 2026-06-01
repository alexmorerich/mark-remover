#!/usr/bin/env python3
"""V18 — Reduce auto-rejects WITHOUT weakening safety.

The V17 final audit is correct: it never ships a residual watermark or a damaged
product. The reason the clean rate is capped is upstream — candidate *generation*
still produces destructive full-band / full-bbox fills on product-overlap
regions, the audit rejects them, and because no safer candidate exists the image
becomes ``auto_rejected``.

V18 does NOT touch the gate. It adds **safer, product-preserving candidates**
before the final audit runs, and it **bans destructive generators** on
product-overlap regions so the beam is not wasted on candidates that can only be
rejected. Every candidate it produces still passes through the unchanged V16/V17
P0 gate; nothing here can ship an output the audit would reject.

Design contract (patch plan §3):
    Do:   add product-aware segmented repair, stroke-mask reliability, line /
          dark / metallic / colored specialized repair, product-safety-first
          ranking, low-contrast glyph-residue auditing.
    Don't: lower residual thresholds, allow full-bbox cover on product, treat
           high removal score as compensation for product damage.

This module is additive and side-effect free: every generator is wrapped so a
failure returns ``None`` (the caller simply skips it), and it imports only the
frozen detectors (``v13_gates``) and existing repair primitives
(``progressive_repair``, ``v14_patch``, ``v15_patch``). It must NOT be imported
by ``v17_final_audit`` (which it imports) to keep the dependency graph acyclic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import v13_gates
import v14_patch
import v15_patch
import v17_final_audit

try:
    import progressive_repair as pr
except Exception:  # pragma: no cover - pr is always importable in practice
    pr = None

# V19 — reverse-alpha recovery + product-text protection (patch plan §1.2, §1.7).
try:
    import sunsky_reverse_alpha as _sra
except Exception:  # pragma: no cover
    _sra = None
try:
    import product_text_detector as _ptd
except Exception:  # pragma: no cover
    _ptd = None
try:
    import lama_onnx_backend as _lama
except Exception:  # pragma: no cover
    _lama = None


# ---------------------------------------------------------------------------
# Routing thresholds (patch plan §2). Kept explicit so product-overlap routing
# is tunable independently of the candidate-stage and final-audit gates.
# ---------------------------------------------------------------------------
PRODUCT_OVERLAP_LOW = 0.03           # <= pure-background regime
PRODUCT_OVERLAP_MID = 0.10           # <= light product, segmented-only regime
PURE_BACKGROUND_SCORE_MIN = 0.85     # ring-uniform + low-edge => background fill ok
FLEX_LINE_CONTINUITY_RATIO = 0.97    # line length must survive a flex repair
DARK_SURFACE_BLOB_MAX = 22.0         # luma jump on dark stock that reads as a blob

# Generator identities (patch plan §2 ban table).
DESTRUCTIVE_GENERATORS = frozenset({
    "uniform_background_fill", "full_bbox_cover", "forced_removal_fill",
    "white_fill", "ring_median_full_band", "v16_uniform_background_fill",
    "v16_forced_removal",
})


# ---------------------------------------------------------------------------
# ProductContext — computed ONCE per image, before any candidate is generated
# (patch plan §2). All signals reuse the frozen V13 detectors.
# ---------------------------------------------------------------------------
@dataclass
class ProductContext:
    product_overlap: float = 0.0
    touches_silhouette: bool = False
    edge_density_inside_box: float = 0.0
    long_line_score: float = 0.0
    protected_text_score: float = 0.0
    dark_surface_ratio: float = 0.0
    metallic_gradient_score: float = 0.0
    flex_line_score: float = 0.0
    pure_background_score: float = 0.0
    roi_class: str = ""

    def to_record(self) -> dict:
        return {
            "v18_product_overlap": round(self.product_overlap, 4),
            "v18_touches_silhouette": bool(self.touches_silhouette),
            "v18_edge_density_inside_box": round(self.edge_density_inside_box, 4),
            "v18_long_line_score": round(self.long_line_score, 4),
            "v18_protected_text_score": round(self.protected_text_score, 4),
            "v18_dark_surface_ratio": round(self.dark_surface_ratio, 4),
            "v18_metallic_gradient_score": round(self.metallic_gradient_score, 4),
            "v18_flex_line_score": round(self.flex_line_score, 4),
            "v18_pure_background_score": round(self.pure_background_score, 4),
            "v18_roi_class": self.roi_class,
        }


def _metallic_gradient_score(image, bbox) -> float:
    """Strength of a smooth luma gradient across the bbox interior (0..1). A high
    value means a reflective / metallic surface whose directionality a flat fill
    would destroy."""
    bx, by, bw, bh = bbox
    inner = image[by:by + bh, bx:bx + bw]
    if inner.size == 0 or bw < 4 or bh < 4:
        return 0.0
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = float(np.mean(np.abs(np.diff(g, axis=1)))) if g.shape[1] > 1 else 0.0
    gy = float(np.mean(np.abs(np.diff(g, axis=0)))) if g.shape[0] > 1 else 0.0
    # Smooth, low-amplitude directional change reads as a gradient; sharp noisy
    # change reads as texture/edges. Normalise to a soft 0..1.
    span = float(g.max() - g.min())
    if span < 12.0:
        return 0.0
    smooth = max(gx, gy)
    return float(np.clip(span / 160.0, 0.0, 1.0)) * \
        float(np.clip(1.0 - smooth / 24.0, 0.0, 1.0))


def compute_product_context(image, bbox, product_mask=None, watermark_mask=None,
                            roi_class="") -> ProductContext:
    """Pre-generation safety routing signals (patch plan §2)."""
    ctx = ProductContext(roi_class=roi_class or "")
    bx, by, bw, bh = bbox
    if bw <= 1 or bh <= 1:
        return ctx
    try:
        ov = v13_gates.estimate_product_overlap_v13(image, bbox, product_mask)
    except Exception:
        ov = {}
    ctx.product_overlap = float(ov.get("product_overlap", 0.0))
    ctx.edge_density_inside_box = float(ov.get("edge_density_in_bbox", 0.0))
    ctx.long_line_score = float(ov.get("long_line_score", 0.0))
    dark_ratio_box = float(ov.get("dark_ratio_in_bbox", 0.0))

    try:
        sil = v13_gates.detect_product_silhouette_damage_v13(
            image, image, product_mask, bbox, watermark_mask=watermark_mask)
    except Exception:
        sil = {}
    ctx.dark_surface_ratio = float(sil.get("dark_surface_ratio", 0.0))
    sil_overlap = float(sil.get("product_overlap", 0.0))
    ctx.touches_silhouette = sil_overlap > 0.15

    try:
        ctx.protected_text_score = float(
            v17_final_audit._protected_text_overlap(image, bbox, watermark_mask))
    except Exception:
        ctx.protected_text_score = 0.0

    # A flex cable reads as a long dark line through the box.
    ctx.flex_line_score = ctx.long_line_score if dark_ratio_box > 0.12 else \
        ctx.long_line_score * 0.4
    ctx.metallic_gradient_score = _metallic_gradient_score(image, bbox)

    # pure_background_score: high only when there is no product signal at all.
    try:
        ring_uniform = bool(v15_patch.is_uniform_local_background(
            image, bbox, product_mask))
    except Exception:
        ring_uniform = False
    bg = 1.0
    bg -= min(1.0, ctx.product_overlap / PRODUCT_OVERLAP_LOW) * 0.5
    bg -= min(1.0, ctx.edge_density_inside_box / 0.06) * 0.25
    bg -= min(1.0, ctx.long_line_score / 0.20) * 0.15
    bg -= min(1.0, ctx.protected_text_score / 0.02) * 0.10
    if not ring_uniform:
        bg -= 0.30
    if ctx.touches_silhouette:
        bg -= 0.30
    ctx.pure_background_score = float(np.clip(bg, 0.0, 1.0))
    return ctx


# ---------------------------------------------------------------------------
# Generator gating (patch plan §2). On product overlap, destructive full-band /
# full-bbox / forced-removal fills are BANNED; only stroke-level or segmented
# product-safe repair may run.
# ---------------------------------------------------------------------------
def allowed_generators(ctx: ProductContext) -> set:
    if ctx.product_overlap < PRODUCT_OVERLAP_LOW and \
            ctx.pure_background_score >= PURE_BACKGROUND_SCORE_MIN:
        return {"uniform_background_fill", "bbox_clone", "ring_median_fill",
                "segmented", "stroke_only", "specialized", "forced_removal_fill"}
    if ctx.product_overlap < PRODUCT_OVERLAP_MID and not ctx.touches_silhouette:
        return {"segmented_micro_cover", "stroke_only_inpaint",
                "ring_clone_background_fragments", "segmented", "stroke_only",
                "specialized"}
    # Heavy product overlap or silhouette contact: product-safe repair only.
    return {"segmented", "stroke_only", "specialized"}


def should_ban_destructive(ctx: ProductContext) -> bool:
    """True when full-band / full-bbox / forced-removal fills must NOT run
    (patch plan §1.3, §2): any meaningful product overlap or silhouette contact.
    """
    return (ctx.product_overlap > PRODUCT_OVERLAP_LOW or ctx.touches_silhouette
            or ctx.protected_text_score > 1e-6)


# ---------------------------------------------------------------------------
# Stroke-mask reliability (patch plan §4). Build a stroke-level mask to inpaint;
# fall back conservatively and report whether a destructive fill is permissible.
# ---------------------------------------------------------------------------
def _stroke_mask(image, bbox, watermark_mask, dilate_px=2) -> Optional[np.ndarray]:
    """Full-image uint8 (0/255) mask of watermark stroke pixels to repair, or
    ``None`` when no reliable stroke mask exists."""
    H, W = image.shape[:2]
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        m = (watermark_mask > 0).astype(np.uint8)
        bx, by, bw, bh = bbox
        in_box = int(m[by:by + bh, bx:bx + bw].sum())
        if in_box >= 12:
            if dilate_px > 0:
                m = cv2.dilate(m, np.ones((dilate_px * 2 + 1,) * 2, np.uint8))
            return (m * 255).astype(np.uint8)
    return None


def stroke_mask_confidence(image, bbox, watermark_mask, product_mask=None) -> dict:
    """Mask-quality signals (patch plan §4). ``logo_fallback`` masks on product
    must never drive a destructive fill — only stroke-only conservative inpaint.
    """
    sm = _stroke_mask(image, bbox, watermark_mask, dilate_px=0)
    bx, by, bw, bh = bbox
    box_area = max(1, bw * bh)
    if sm is None:
        stroke_px = 0
        coverage = 0.0
        mask_type = "logo_fallback"
    else:
        stroke_px = int((sm[by:by + bh, bx:bx + bw] > 0).sum())
        coverage = stroke_px / float(box_area)
        # A thin glyph stroke covers a small fraction of the box; a near-full box
        # is a widened/logo fallback, not a precise stroke mask.
        mask_type = "stroke" if coverage < 0.45 else "widened_text"
    try:
        ptext = float(v17_final_audit._protected_text_overlap(
            image, bbox, watermark_mask))
    except Exception:
        ptext = 0.0
    try:
        ov = v13_gates.estimate_product_overlap_v13(image, bbox, product_mask)
        product_overlap = float(ov.get("product_overlap", 0.0))
    except Exception:
        product_overlap = 0.0
    return {
        "mask_type": mask_type,
        "stroke_precision": round(float(np.clip(1.0 - coverage / 0.45, 0.0, 1.0)),
                                  4),
        "stroke_pixels": stroke_px,
        "coverage": round(coverage, 4),
        "product_overlap": round(product_overlap, 4),
        "protected_text_overlap": round(ptext, 4),
        "destructive_fill_allowed": bool(
            mask_type == "stroke" and product_overlap <= PRODUCT_OVERLAP_LOW),
    }


# ---------------------------------------------------------------------------
# Compositing helper — replace ONLY masked pixels of ``base`` with ``donor``,
# feathered, so a repair never paints outside the watermark strokes.
# ---------------------------------------------------------------------------
def _paste_at_mask(base, donor, mask, feather_px=3):
    if donor is None or donor.shape != base.shape:
        return base
    m = (mask > 0).astype(np.float32)
    if feather_px > 0:
        k = feather_px * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    m3 = cv2.merge([m, m, m])
    out = base.astype(np.float32) * (1.0 - m3) + donor.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Phase 3 — segmented product/background repair. Split the ROI into background
# and product fragments and repair each with the right tool; NEVER one global
# fill across all fragments. Delegates to the existing v14 fragment fill, which
# already does bg->ring-median and product->stroke inpaint.
# ---------------------------------------------------------------------------
def segmented_product_background_repair(image, bbox, watermark_mask,
                                        product_mask=None):
    try:
        out = v14_patch._segmented_fragment_fill(
            image, bbox, watermark_mask, product_mask)
        if out is not None and out.shape == image.shape:
            return out
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Phase 5 — specialized repair tools for high-reject ROI classes. Each restricts
# the change to the watermark strokes and verifies it did not damage the product
# surface it sits on; on any doubt it returns None (the caller skips it).
# ---------------------------------------------------------------------------
def repair_thin_flex_line_preserving(image, bbox, watermark_mask,
                                     product_mask=None):
    """Inpaint only the watermark strokes with a line-friendly NS fill and reject
    if the cable's straight-line continuity drops (patch plan §5.1)."""
    sm = _stroke_mask(image, bbox, watermark_mask, dilate_px=1)
    if sm is None:
        return None
    before = v17_final_audit._flex_line_continuity(image, bbox)
    if pr is not None:
        out = pr._apply_stroke_inpaint(image, sm, method="ns", radius=2)
    else:
        out = None
    if out is None:
        return None
    after = v17_final_audit._flex_line_continuity(out, bbox)
    if before >= max(8.0, bbox[2] / 3.0) and after < before * FLEX_LINE_CONTINUITY_RATIO:
        return None
    return out


def repair_dark_surface_stroke_clone(image, bbox, watermark_mask,
                                     product_mask=None):
    """Clone/plane-fill the strokes from local dark ring pixels; never a
    white/gray median fill; reject a bright blob on dark stock (patch plan §5.2).
    """
    sm = _stroke_mask(image, bbox, watermark_mask, dilate_px=2)
    if sm is None:
        return None
    out = None
    if pr is not None:
        try:
            out = pr._fill_stroke_from_clone(image, sm, bbox, product_mask)
        except Exception:
            out = None
        if out is None:
            out = pr._apply_stroke_inpaint(image, sm, method="ns", radius=2)
    if out is None:
        return None
    if v17_final_audit._dark_surface_blob(image, out, bbox) > DARK_SURFACE_BLOB_MAX:
        return None
    return out


def repair_metallic_gradient_plane(image, bbox, watermark_mask,
                                   product_mask=None):
    """Fit a local gradient plane from the ring and paint it over the strokes
    only, preserving directional reflection; reject a flat rectangular block
    (patch plan §5.3)."""
    if pr is None:
        return None
    sm = _stroke_mask(image, bbox, watermark_mask, dilate_px=2)
    if sm is None:
        return None
    try:
        plane = pr._fit_gradient_plane(image, bbox)
    except Exception:
        plane = None
    if plane is None:
        return None
    out = _paste_at_mask(image, plane, sm, feather_px=3)
    # Reject a flat block: a metallic surface flattened into a rectangle.
    try:
        if v17_final_audit._metallic_block(image, out, bbox):
            return None
    except Exception:
        pass
    return out


def repair_colored_surface_hue_matched(image, bbox, watermark_mask,
                                       product_mask=None):
    """Hue/saturation-matched local fill of the strokes (Lab plane fit handles
    colour); never a gray/white fill (patch plan §5.4)."""
    if pr is None:
        return None
    sm = _stroke_mask(image, bbox, watermark_mask, dilate_px=2)
    if sm is None:
        return None
    out = None
    try:
        out = pr._fill_stroke_from_clone(image, sm, bbox, product_mask)
    except Exception:
        out = None
    if out is None:
        return None
    return out


def stroke_only_inpaint(image, bbox, watermark_mask, product_mask=None):
    """Conservative Telea inpaint of the watermark strokes only — the universal
    product-safe fallback (patch plan §2, §4)."""
    try:
        out = v14_patch._stroke_only_fill(image, bbox, watermark_mask)
        if out is not None and out.shape == image.shape:
            return out
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Candidate assembly (patch plan §2, §3, §5). Returns product-safe candidates in
# product-safety priority order. Every one still goes through the unchanged P0
# gate in the caller, so this can only ADD safe options, never ship a bad one.
# ---------------------------------------------------------------------------
def build_safe_candidates(image, bbox, watermark_mask, product_mask,
                          ctx: ProductContext) -> List[Tuple[str, np.ndarray]]:
    allowed = allowed_generators(ctx)
    cands: List[Tuple[str, np.ndarray]] = []

    def _add(name, img):
        if img is not None and getattr(img, "shape", None) == image.shape:
            cands.append((name, img))

    roi = ctx.roi_class or ""
    # ROI-specific specialized repair first (most surface-appropriate).
    if "specialized" in allowed:
        if roi in ("thin_flex_cable",) or ctx.flex_line_score > v13_gates.LONG_LINE_OVERRIDE:
            _add("v18_flex_line_preserving",
                 repair_thin_flex_line_preserving(image, bbox, watermark_mask,
                                                  product_mask))
        if roi in ("dark_product_surface",) or \
                ctx.dark_surface_ratio > v13_gates.DARK_SURFACE_RATIO_FOR_BLOB:
            _add("v18_dark_surface_stroke_clone",
                 repair_dark_surface_stroke_clone(image, bbox, watermark_mask,
                                                  product_mask))
        if roi in ("metallic_or_reflective", "glass_or_gradient",
                   "transparent_or_glossy") or ctx.metallic_gradient_score > 0.3:
            _add("v18_metallic_gradient_plane",
                 repair_metallic_gradient_plane(image, bbox, watermark_mask,
                                                product_mask))
        # Colored / adhesive surfaces — always worth a hue-matched attempt.
        _add("v18_colored_hue_matched",
             repair_colored_surface_hue_matched(image, bbox, watermark_mask,
                                                product_mask))

    # Segmented product/background repair (surgical, fragment-aware).
    if "segmented" in allowed:
        _add("v18_segmented_product_background",
             segmented_product_background_repair(image, bbox, watermark_mask,
                                                 product_mask))

    # Universal stroke-only fallback.
    if "stroke_only" in allowed:
        _add("v18_stroke_only_inpaint",
             stroke_only_inpaint(image, bbox, watermark_mask, product_mask))

    # De-dup by name, keep first (priority) occurrence.
    seen = set()
    uniq: List[Tuple[str, np.ndarray]] = []
    for name, img in cands:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, img))
    return uniq


# ---------------------------------------------------------------------------
# Phase 6 — product-safety-first ranking penalties. A candidate that removes the
# mark well but damages product pixels must NOT dominate the beam. This computes
# a cheap penalty (higher = worse) used to order candidates before the gate, so
# the safest options are tried first.
# ---------------------------------------------------------------------------
def candidate_product_penalty(original, candidate_img, bbox, product_mask,
                              watermark_mask, ctx: ProductContext) -> float:
    """Lower is safer. Pure-background regions are ranked by removal/smoothness;
    product regions by how little product they disturb (patch plan §6)."""
    if candidate_img is None:
        return 1e9
    try:
        prod_ratio, edge_density = \
            v17_final_audit.changed_region_product_overlap(
                original, candidate_img, bbox, product_mask, watermark_mask)
    except Exception:
        prod_ratio, edge_density = 0.0, 0.0
    penalty = 0.0
    if ctx.product_overlap > PRODUCT_OVERLAP_LOW or ctx.touches_silhouette:
        # Product region: every product pixel touched is expensive.
        penalty += 1000.0 * prod_ratio
        penalty += 300.0 * edge_density
        # Hard pre-rank penalties (patch plan §6) for visible damage.
        try:
            patch = v13_gates.detect_visible_patch_shape_v13(
                original, candidate_img, bbox)
            if prod_ratio > PRODUCT_OVERLAP_LOW:
                penalty += 1000.0 * float(patch.get("visible_patch_score", 0.0))
                penalty += 500.0 * float(patch.get("texture_drop", 0.0))
        except Exception:
            pass
        if ctx.dark_surface_ratio > v13_gates.DARK_SURFACE_RATIO_FOR_BLOB:
            penalty += 500.0 * (v17_final_audit._dark_surface_blob(
                original, candidate_img, bbox) / 255.0)
    else:
        # Pure background: rank by residual removal + boundary smoothness.
        try:
            band = v13_gates._band_metrics(candidate_img, bbox)
            penalty += 50.0 * float(band.get("visible_band_score", 0.0))
        except Exception:
            pass
    return float(penalty)


def rank_candidates(original, candidates, bbox, product_mask, watermark_mask,
                    ctx: ProductContext):
    """Return ``candidates`` reordered safest-first. ``candidates`` is a list of
    ``(name, image)`` tuples (patch plan §6)."""
    scored = []
    for name, img in candidates:
        p = candidate_product_penalty(original, img, bbox, product_mask,
                                      watermark_mask, ctx)
        scored.append((p, name, img))
    scored.sort(key=lambda t: t[0])
    return [(name, img) for _p, name, img in scored]


# ---------------------------------------------------------------------------
# V19 — reverse-alpha recovery candidate (patch plan §1.2, §4.3). This is a
# NON-DESTRUCTIVE real-pixel repair: it subtracts the semi-transparent overlay
# and keeps the product pixels beneath, so it is allowed even on product overlap
# and over product text (where covers / inpaint are banned). It is tried FIRST
# in the candidate bank; it still passes through the unchanged P0 / V17 audit.
# ---------------------------------------------------------------------------
def reverse_alpha_candidate(image, bbox, watermark_mask, product_mask,
                            ctx: ProductContext):
    """Return ``(name, image, record)``. ``image`` is ``None`` when reverse-alpha
    is unavailable or its local safety check fails (the caller skips it). The
    record carries the per-image ``reverse_alpha_*`` manifest fields."""
    if _sra is None or not _sra.alpha_asset_available():
        return None, None, {"reverse_alpha_attempted": False,
                             "reverse_alpha_passed": False,
                             "reverse_alpha_source": "none",
                             "reverse_alpha_reject_reason": "alpha_asset_missing"}
    ptmask = None
    ptext_overlap = 0.0
    if _ptd is not None:
        try:
            ptm = _ptd.detect_product_text(image, bbox, watermark_mask)
            ptmask = ptm.mask
            ptext_overlap = ptm.overlap_in_box(bbox)
        except Exception:
            ptmask = None
    try:
        res = _sra.repair_sunsky_reverse_alpha(
            image, bbox, watermark_mask=watermark_mask, product_mask=product_mask,
            protected_text_mask=ptmask, ctx=ctx, allow_thin_cleanup=True)
    except Exception:
        return None, None, {"reverse_alpha_attempted": True,
                            "reverse_alpha_passed": False,
                            "reverse_alpha_reject_reason": "exception"}
    rec = res.to_record()
    rec["reverse_alpha_over_text"] = bool(ptext_overlap > 0.02)
    if res.pass_local_safety and res.image is not None:
        return "v19_sunsky_reverse_alpha", res.image, rec
    return None, None, rec


def lama_stroke_candidate(image, bbox, watermark_mask, product_mask,
                          ctx: ProductContext):
    """Optional CPU LaMA-ONNX stroke-only candidate (patch plan §1.6). Returns
    ``None`` unless the model is present and every precondition holds."""
    if _lama is None or not _lama.available():
        return None
    sm = _stroke_mask(image, bbox, watermark_mask, dilate_px=1)
    if sm is None:
        return None
    conf = stroke_mask_confidence(image, bbox, watermark_mask, product_mask)
    try:
        ptext = 0.0
        if _ptd is not None:
            ptext = _ptd.detect_product_text(
                image, bbox, watermark_mask).overlap_in_box(bbox)
        return _lama.run_lama_crop_paste(
            image, sm, mask_type=conf.get("mask_type", "logo_fallback"),
            product_overlap=float(conf.get("product_overlap", 0.0)),
            protected_text_overlap=ptext)
    except Exception:
        return None


def manifest_routing_fields(image, bbox, watermark_mask, product_mask,
                            ctx: ProductContext, ban_destructive: bool) -> dict:
    """V19 mask-type routing manifest (patch plan §4.5)."""
    conf = stroke_mask_confidence(image, bbox, watermark_mask, product_mask)
    alpha_conf = 0.0
    if _sra is not None and _sra.alpha_asset_available():
        a = _sra.load_alpha_asset()
        alpha_conf = float(a["alpha"].max()) if a else 0.0
    return {
        "mask_type": conf.get("mask_type", "logo_fallback"),
        "stroke_mask_confidence": conf.get("stroke_precision", 0.0),
        "alpha_asset_confidence": round(alpha_conf, 4),
        "allowed_destructive_generators": bool(not ban_destructive),
    }
