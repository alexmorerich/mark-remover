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

# V22 — low-texture product-surface classification + band/glyph detectors.
try:
    import v22_patch
except Exception:  # pragma: no cover
    v22_patch = None


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
    # V20 — conservative union safety overlap (patch plan §Patch 8). Fraction of
    # the bbox covered by the union product-safe mask (non-white / edges / dark /
    # colored / metallic / glass / flex / text / dilated silhouette). Destructive
    # generators are banned whenever ANY of these signals are present.
    product_mask_safe_overlap: float = 0.0
    # V22 — low-texture product-surface signals the plain mask misses (patch plan
    # §Patch 4). Populated from v22_patch.classify_surface_v22.
    connected_component_crosses_bbox: bool = False
    near_white_product_surface: bool = False
    translucent_stack_surface: bool = False
    dark_smooth_product_surface: bool = False
    long_thin_component_crosses_bbox: bool = False
    v22_product_like_overlap: float = 0.0
    v22_surface_class: str = ""

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
            "v20_product_mask_safe_overlap": round(self.product_mask_safe_overlap, 4),
            "v18_roi_class": self.roi_class,
            # V22 surface classification (patch plan §Patch 4).
            "v22_connected_component_crosses_bbox":
                bool(self.connected_component_crosses_bbox),
            "v22_near_white_product_surface":
                bool(self.near_white_product_surface),
            "v22_translucent_stack_surface": bool(self.translucent_stack_surface),
            "v22_dark_smooth_product_surface":
                bool(self.dark_smooth_product_surface),
            "v22_long_thin_component_crosses_bbox":
                bool(self.long_thin_component_crosses_bbox),
            "v22_product_like_overlap": round(self.v22_product_like_overlap, 4),
            "v22_surface_class": self.v22_surface_class,
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


def build_product_mask_safe(image, bbox, product_mask=None, watermark_mask=None):
    """V20 — conservative UNION product mask for safety decisions (patch plan
    §Patch 8). Unions every product-like signal so a destructive fill never runs
    over dark / colored / metallic / glass / flex / text / silhouette pixels the
    plain product mask might miss. The watermark's own strokes are excluded so
    the mark cannot masquerade as product. Returns a full-image uint8 (0/255)."""
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    union = np.zeros((H, W), np.uint8)
    if bw <= 1 or bh <= 1:
        return union
    bx2, by2 = min(W, bx + bw), min(H, by + bh)
    bx, by = max(0, bx), max(0, by)
    if bx2 <= bx or by2 <= by:
        return union
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sub = union[by:by2, bx:bx2]
    gg = g[by:by2, bx:bx2]
    ss = sat[by:by2, bx:bx2]
    # non-white connected component, dark surface, colored (saturated) surface.
    sub[gg < 235] = 255                              # non-white
    sub[gg < 80] = 255                               # dark surface
    sub[ss > 60] = 255                               # colored / saturated
    # edge-density (structure) + metallic gradient + glass boundary all read as
    # local edges; Canny over the box captures them.
    edges = cv2.Canny(gg, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    sub[edges > 0] = 255
    union[by:by2, bx:bx2] = sub
    # existing product mask + dilated silhouette.
    if product_mask is not None and product_mask.shape[:2] == (H, W):
        union[(product_mask > 0)] = 255
    # long-line / flex mask: a HoughLines pass adds straight cable lines.
    lines = cv2.HoughLinesP(cv2.Canny(gg, 50, 150), 1, np.pi / 180,
                            threshold=max(10, bw // 6),
                            minLineLength=max(10, bw // 3), maxLineGap=4)
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv2.line(union, (bx + x1, by + y1), (bx + x2, by + y2), 255, 3)
    # protected product text.
    try:
        if v17_final_audit._protected_text_overlap(image, bbox, watermark_mask) > 0.01:
            og = g[by:by2, bx:bx2].astype(np.float32)
            surface = float(np.median(og))
            text = (np.abs(og - surface) > 45).astype(np.uint8) * 255
            union[by:by2, bx:bx2] = cv2.bitwise_or(union[by:by2, bx:bx2], text)
    except Exception:
        pass
    # Exclude the watermark's own dilated strokes (they are SUPPOSED to change).
    if watermark_mask is not None and watermark_mask.shape[:2] == (H, W):
        wm = cv2.dilate((watermark_mask > 0).astype(np.uint8),
                        np.ones((3, 3), np.uint8), iterations=2)
        union[wm > 0] = 0
    return union


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

    # V20 — conservative union product-safe overlap (patch plan §Patch 8). Used
    # to ban destructive fills on any product-like signal, not just product_mask.
    try:
        safe = build_product_mask_safe(image, bbox, product_mask, watermark_mask)
        bx, by, bw, bh = bbox
        sub = safe[by:by + bh, bx:bx + bw]
        ctx.product_mask_safe_overlap = float((sub > 0).mean()) if sub.size else 0.0
    except Exception:
        ctx.product_mask_safe_overlap = ctx.product_overlap

    # V22 — low-texture product-surface classification (patch plan §Patch 4). The
    # plain product mask misses dark-smooth / near-white / translucent-stack /
    # thin-flex surfaces; these self-contained signals let candidate generation
    # know earlier that the region is product-like, so destructive covers are
    # banned on them (patch plan §Patch 5, §Patch 7).
    if v22_patch is not None:
        try:
            surf = v22_patch.classify_surface_v22(
                image, bbox, product_mask, watermark_mask, roi_class)
            ctx.connected_component_crosses_bbox = bool(
                surf["connected_component_crosses_bbox"])
            ctx.near_white_product_surface = bool(
                surf["near_white_product_surface"])
            ctx.translucent_stack_surface = bool(surf["translucent_stack_surface"])
            ctx.dark_smooth_product_surface = bool(
                surf["dark_smooth_product_surface"])
            ctx.long_thin_component_crosses_bbox = bool(
                surf["long_thin_component_crosses_bbox"])
            ctx.v22_product_like_overlap = float(surf["v22_product_like_overlap"])
            ctx.v22_surface_class = surf["surface_class"]
        except Exception:
            pass
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


# V20 — union safe-mask overlap above which destructive fills are banned (patch
# plan §Patch 8). Conservative: any appreciable product-like structure vetoes a
# destructive fill, even when the plain product_mask reads low.
PRODUCT_MASK_SAFE_BAN = 0.12


def should_ban_destructive(ctx: ProductContext) -> bool:
    """True when full-band / full-bbox / forced-removal fills must NOT run
    (patch plan §1.3, §2, §Patch 8, §Patch 5/7): any meaningful product overlap,
    silhouette contact, protected text, union product-safe-mask coverage, OR a
    V22 low-texture product surface (dark-smooth / translucent-stack / near-white
    product / thin-flex) the plain mask would miss.
    """
    v22_product_surface = (
        ctx.dark_smooth_product_surface or ctx.translucent_stack_surface
        or (ctx.near_white_product_surface and ctx.connected_component_crosses_bbox)
        or ctx.long_thin_component_crosses_bbox)
    return (ctx.product_overlap > PRODUCT_OVERLAP_LOW or ctx.touches_silhouette
            or ctx.protected_text_score > 1e-6
            or ctx.product_mask_safe_overlap > PRODUCT_MASK_SAFE_BAN
            or v22_product_surface)


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
# V20 — mixed product/background segmented repair (patch plan §Patch 5). When the
# watermark crosses BOTH product and white background, a single method is unsafe:
# background fill is safe on white but dangerous on product; stroke repair is too
# weak for background residue. This splits the footprint and uses reverse-alpha
# on product pixels and a clone-offset fill on pure-background pixels.
# ---------------------------------------------------------------------------
def _clone_offset_fill(image, bbox, target_mask, product_mask):
    """Copy REAL pixels from a clean band above/below the watermark into the
    background portion of the footprint (patch plan §Patch 5 background clone).
    Returns the filled image, or ``None`` if no safe source offset exists."""
    H, W = image.shape[:2]
    bx, by, bw, bh = bbox
    tgt = (target_mask > 0)
    if int(tgt.sum()) < 4:
        return None
    # Candidate vertical offsets (above/below) and small horizontal shifts.
    dys = [-3 * bh, -2 * bh, -1 * bh, bh, 2 * bh, 3 * bh]
    dxs = [-bw // 2, 0, bw // 2]
    best = None
    best_cost = 1e18
    ys, xs = np.where(tgt)
    for dy in dys:
        for dx in dxs:
            sy, sx = ys + dy, xs + dx
            ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
            if float(ok.mean()) < 0.98:
                continue
            sy, sx = sy[ok], sx[ok]
            ty, tx = ys[ok], xs[ok]
            # Source must be pure background: not on product.
            if product_mask is not None and product_mask.shape[:2] == (H, W):
                if float((product_mask[sy, sx] > 0).mean()) > 0.02:
                    continue
            # Cost: brightness mismatch between source and the ring (prefer a
            # tone-matched, low-variance source band).
            src_vals = image[sy, sx].astype(np.float32)
            cost = float(src_vals.std())
            if cost < best_cost:
                best_cost = cost
                best = (sy, sx, ty, tx)
    if best is None:
        return None
    sy, sx, ty, tx = best
    out = image.copy()
    out[ty, tx] = image[sy, sx]
    return out


def segmented_reverse_alpha_background_clone(image, bbox, watermark_mask,
                                             product_mask=None):
    """Mixed product/background repair (patch plan §Patch 5). Reverse-alpha on
    product pixels, clone-offset fill on pure-background pixels, blended only
    along the watermark footprint. Rejects a rectangular boundary or a clone that
    lands on product. Returns ``None`` on any doubt (the caller skips it)."""
    if _sra is None or not _sra.alpha_asset_available():
        return None
    sm = _stroke_mask(image, bbox, watermark_mask, dilate_px=1)
    if sm is None:
        return None
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    foot = (sm > 0)

    # Product vs background split inside the footprint.
    if product_mask is not None and product_mask.shape[:2] == (H, W):
        prod = (product_mask > 0)
    else:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prod = (g < 235)
    prod_foot = foot & prod
    bg_foot = foot & (~prod)

    # 1) reverse-alpha recovers the whole footprint (non-destructive on product).
    ptmask = None
    if _ptd is not None:
        try:
            ptmask = _ptd.detect_product_text(image, bbox, watermark_mask).mask
        except Exception:
            ptmask = None
    try:
        res = _sra.repair_sunsky_reverse_alpha(
            image, bbox, watermark_mask=watermark_mask, product_mask=product_mask,
            protected_text_mask=ptmask, allow_thin_cleanup=True)
    except Exception:
        res = None
    out = res.image.copy() if (res is not None and res.image is not None) \
        else image.copy()

    # 2) clone-offset fill ONLY the pure-background portion of the footprint.
    if int(bg_foot.sum()) >= 4:
        bg_mask = np.zeros((H, W), np.uint8)
        bg_mask[bg_foot] = 255
        cloned = _clone_offset_fill(image, bbox, bg_mask, product_mask)
        if cloned is not None:
            out[bg_foot] = cloned[bg_foot]

    if out.shape != image.shape:
        return None
    # 3) reject a rectangular artificial boundary on product.
    try:
        if v17_final_audit._metallic_block(image, out, bbox):
            return None
        patch = v13_gates.detect_visible_patch_shape_v13(image, out, bbox)
        if float(patch.get("hard_boundary_score", 0.0)) > \
                v13_gates.HARD_BOUNDARY_MAX and int(prod_foot.sum()) > 8:
            return None
    except Exception:
        pass
    return out


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


def thin_flex_reverse_alpha_line_preserve(image, bbox, watermark_mask,
                                          product_mask=None):
    """V20 — line-preserving reverse-alpha for thin flex cables (patch plan
    §Patch 6). Reverse-alpha is non-destructive, so it recovers the overlay
    footprint without cutting the cable; we still verify continuity and reject if
    the longest cable line shrinks or its endpoints shift. Returns ``None`` on
    any doubt (the caller skips it)."""
    if _sra is None or not _sra.alpha_asset_available():
        return None
    ptmask = None
    if _ptd is not None:
        try:
            ptmask = _ptd.detect_product_text(image, bbox, watermark_mask).mask
        except Exception:
            ptmask = None
    try:
        res = _sra.repair_sunsky_reverse_alpha(
            image, bbox, watermark_mask=watermark_mask, product_mask=product_mask,
            protected_text_mask=ptmask, allow_thin_cleanup=True)
    except Exception:
        return None
    if res is None or res.image is None:
        return None
    out = res.image
    try:
        cont = v13_gates.detect_thin_flex_continuity_v20(image, out, bbox)
        if cont.get("hard_fail"):
            return None
    except Exception:
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
    # V23 — build the precise alpha/stroke mask set once (patch plan §Patch 2);
    # the component-aware mixed repair and thin-flex line restore consume it.
    v23_mask_set = None
    try:
        import v23_masks
        v23_mask_set = v23_masks.build_v23_mask_set(
            image, bbox, watermark_mask, product_mask, roi)
    except Exception:
        v23_mask_set = None
    # ROI-specific specialized repair first (most surface-appropriate).
    if "specialized" in allowed:
        if roi in ("thin_flex_cable",) or ctx.flex_line_score > v13_gates.LONG_LINE_OVERRIDE:
            # V23 — thin-flex / black-frame line-preserving restore (non-
            # destructive reverse-alpha + line-continuity verification).
            try:
                import v23_flex_repair
                _add("v23_thin_flex_line_restore",
                     v23_flex_repair.v23_thin_flex_line_restore(
                         image, bbox, watermark_mask, product_mask, v23_mask_set))
            except Exception:
                pass
            # V20 — reverse-alpha line-preserving repair first (non-destructive).
            _add("v20_thin_flex_reverse_alpha_line_preserve",
                 thin_flex_reverse_alpha_line_preserve(image, bbox,
                                                       watermark_mask, product_mask))
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

    # V23 — component-aware mixed product/background repair (patch plan §Patch 3):
    # per-fragment reverse-alpha on product + verified clone on background + a
    # capped micro residue cleanup. Tried before the V20 segmented clone.
    if "segmented" in allowed:
        try:
            import v23_mixed_repair
            _add("v23_component_aware_mixed_repair",
                 v23_mixed_repair.v23_component_aware_mixed_repair(
                     image, bbox, watermark_mask, product_mask, v23_mask_set,
                     roi, ctx))
        except Exception:
            pass
    # V20 — mixed product/background segmented repair (reverse-alpha on product,
    # clone-offset fill on pure background). Tried before the v14 fragment fill.
    if "segmented" in allowed:
        _add("v20_segmented_reverse_alpha_background_clone",
             segmented_reverse_alpha_background_clone(
                 image, bbox, watermark_mask, product_mask))

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


# ---------------------------------------------------------------------------
# V20 — reverse-alpha variant beam (patch plan §Patch 3). Returns an ordered
# list of ``(name, image)`` reverse-alpha variants (safest/cleanest first) plus
# an aggregate record. Each variant is a non-destructive real-pixel recovery
# screened by the same local pre-screen; all still pass through the unchanged P0
# audit in the pipeline. Tried BEFORE destructive generators.
# ---------------------------------------------------------------------------
def reverse_alpha_variant_beam(image, bbox, watermark_mask, product_mask,
                               ctx: ProductContext):
    """Return ``(candidates, record)`` where ``candidates`` is a list of
    ``(name, image)`` and ``record`` carries the V20 beam telemetry."""
    rec = {"reverse_alpha_variants_attempted": 0,
           "reverse_alpha_variants_passed": 0,
           "reverse_alpha_variant_names": []}
    if _sra is None or not _sra.alpha_asset_available():
        return [], rec
    ptmask = None
    over_text = False
    if _ptd is not None:
        try:
            ptm = _ptd.detect_product_text(image, bbox, watermark_mask)
            ptmask = ptm.mask
            over_text = bool(ptm.overlap_in_box(bbox) > 0.02)
        except Exception:
            ptmask = None
    try:
        beam = _sra.build_variant_beam(
            image, bbox, watermark_mask=watermark_mask, product_mask=product_mask,
            protected_text_mask=ptmask, ctx=ctx, allow_thin_cleanup=True)
    except Exception:
        return [], rec
    rec["reverse_alpha_variants_attempted"] = len(_sra.VARIANT_NAMES)
    rec["reverse_alpha_variants_passed"] = len(beam)
    rec["reverse_alpha_variant_names"] = [n for n, _ in beam]
    rec["reverse_alpha_over_text"] = over_text
    cands = [(name, res.image) for name, res in beam
             if res.image is not None]
    return cands, rec


def residue_micro_cleanup_beam(image, bbox, watermark_mask, product_mask,
                               ctx: ProductContext, base_candidates):
    """V21 (patch plan §Patch 2). Given the already-built reverse-alpha repair
    candidates (``base_candidates`` = list of ``(name, image)``), generate
    residue micro-cleanup variants that touch ONLY small aligned residue
    components inside the glyph footprint. Returns ``(candidates, record)``.

    The micro-clean operates on the *reverse-alpha output*, not the original, so
    it removes faint paired-dot / ghost residue the overlay subtraction left
    behind. Every variant still passes the unchanged P0 / V17 audit downstream —
    a micro-clean that creates a blob or damages product is rejected there."""
    rec = {"v21_residue_micro_attempted": 0,
           "v21_residue_micro_candidates": 0,
           "v21_residue_micro_candidate_names": []}
    if _sra is None or not _sra.alpha_asset_available():
        return [], rec
    roi = (ctx.roi_class if ctx is not None else "") or ""
    # Only the top reverse-alpha bases are worth micro-cleaning (the cleanest
    # placements); cap to 2 to keep the beam small and deterministic.
    bases = [(n, im) for (n, im) in (base_candidates or [])
             if im is not None and str(n).startswith(
                 ("v20_reverse_alpha", "v19_sunsky_reverse_alpha",
                  "v21_reverse_alpha"))][:2]
    out, seen = [], set()
    for bname, bimg in bases:
        rec["v21_residue_micro_attempted"] += 1
        try:
            variants = _sra.build_residue_micro_cleanup_beam(
                image, bimg, bbox, watermark_mask=watermark_mask,
                product_mask=product_mask, roi_class=roi)
        except Exception:
            variants = []
        for vname, vimg in variants:
            if vname.endswith("noop_control"):
                continue   # the base itself is already in the beam
            key = vname + "<" + bname
            if key in seen or vimg is None or vimg.shape != image.shape:
                continue
            seen.add(key)
            out.append((vname, vimg))
    rec["v21_residue_micro_candidates"] = len(out)
    rec["v21_residue_micro_candidate_names"] = [n for n, _ in out]
    return out, rec


def alpha_footprint_residue_cleanup_beam_v22(image, bbox, watermark_mask,
                                             product_mask, ctx: ProductContext,
                                             base_candidates):
    """V22 (patch plan §Patch 3). Broader-but-footprint-capped partial-glyph
    cleanup. Runs on the reverse-alpha / v21 bases and strips faint surviving
    s / m / .com fragments inside the solved glyph footprint. Returns
    ``(candidates, record)``. Every variant still passes the unchanged P0 / V17 /
    V22 audit downstream — a cleanup that creates a blob or band is rejected
    there, and an over-large residue is refused (image stays auto_rejected)."""
    rec = {"v22_residue_cleanup_attempted": 0,
           "v22_residue_cleanup_candidates": 0,
           "v22_residue_cleanup_candidate_names": []}
    if _sra is None or not _sra.alpha_asset_available():
        return [], rec
    roi = (ctx.roi_class if ctx is not None else "") or ""
    bases = [(n, im) for (n, im) in (base_candidates or [])
             if im is not None and str(n).startswith(
                 ("v20_reverse_alpha", "v19_sunsky_reverse_alpha",
                  "v21_reverse_alpha", "v21_residue_micro"))][:3]
    out, seen = [], set()
    for bname, bimg in bases:
        rec["v22_residue_cleanup_attempted"] += 1
        try:
            variants = _sra.build_alpha_footprint_residue_cleanup_beam_v22(
                image, bimg, bbox, watermark_mask=watermark_mask,
                product_mask=product_mask, roi_class=roi)
        except Exception:
            variants = []
        for vname, vimg in variants:
            key = vname + "<" + bname
            if key in seen or vimg is None or vimg.shape != image.shape:
                continue
            seen.add(key)
            out.append((vname, vimg))
    rec["v22_residue_cleanup_candidates"] = len(out)
    rec["v22_residue_cleanup_candidate_names"] = [n for n, _ in out]
    return out, rec


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


_V21_DAMAGE_TOKENS = ("product_damage", "silhouette", "dark_surface",
                      "metallic", "protected_text")
_V21_COVER_TOKENS = ("visible_patch", "visible_band", "wedge", "slab", "cover")
_V21_SMOOTH_ROI = ("metallic_or_reflective", "glass_or_gradient",
                   "simple_product_surface", "dark_product_surface",
                   "low_texture_background", "transparent_or_glossy")


def build_v21_records(rec: dict, status: str) -> dict:
    """V21 (patch plan §1, §6, §8). Derive the additive failure-taxonomy,
    mask-quality and residual-explain records from signals already present on
    ``rec``. Pure post-processing — changes no decision, only diagnostics, so it
    can never alter what is published. Returns a dict of new qa.json keys."""
    rexp = rec.get("residual_explain") or {}
    hard = [str(r) for r in (rec.get("v17_hard_fail_reasons") or [])]
    reasons = " ".join(str(r) for r in (rec.get("reject_reasons") or []))
    hjoined = " ".join(hard)
    roi = (rec.get("v18_roi_class") or rec.get("v9_roi_class") or "")
    overlap = float(rec.get("v18_product_overlap", 0.0) or 0.0)
    pure_bg = float(rec.get("v18_pure_background_score", 0.0) or 0.0)
    touches = bool(rec.get("v18_touches_silhouette"))
    ocr_pos = bool(rexp.get("ocr_positive"))
    dot_pos = bool(rexp.get("dot_chain_positive"))
    glyph_pos = bool(rexp.get("low_contrast_glyph_positive"))
    template_pos = bool(rexp.get("template_positive"))
    fp_suspect = bool(rexp.get("possible_product_structure_false_positive"))

    # residual_kind
    if ocr_pos:
        residual_kind = "readable_text"
    elif dot_pos:
        residual_kind = "dot_chain"
    elif glyph_pos:
        residual_kind = "low_contrast_ghost"
    elif template_pos and fp_suspect:
        residual_kind = "template_only"
    else:
        residual_kind = "none"

    # changed_region_kind
    if touches:
        changed_region_kind = "silhouette_touching"
    elif overlap > PRODUCT_OVERLAP_MID:
        changed_region_kind = "product"
    elif pure_bg >= PURE_BACKGROUND_SCORE_MIN and overlap < PRODUCT_OVERLAP_LOW:
        changed_region_kind = "pure_background"
    elif overlap >= PRODUCT_OVERLAP_LOW:
        changed_region_kind = "mixed"
    else:
        changed_region_kind = "unknown"

    # primary_reject_class (only meaningful when auto_rejected)
    if residual_kind in ("readable_text", "dot_chain", "low_contrast_ghost"):
        primary = "true_residual"
    elif "thin_flex" in hjoined or "flex" in reasons:
        primary = "thin_flex_break"
    elif "protected_text" in hjoined:
        primary = "protected_text_risk"
    elif any(t in hjoined or t in reasons for t in _V21_COVER_TOKENS):
        primary = "cover_artifact"
    elif any(t in hjoined or t in reasons for t in _V21_DAMAGE_TOKENS):
        primary = "product_damage"
    elif fp_suspect:
        primary = "detector_false_positive_suspect"
    elif int(rec.get("v18_n_safe_candidates", 0) or 0) == 0:
        primary = "no_safe_candidate"
    else:
        primary = "mask_uncertain"

    # recommended_next_candidate
    if residual_kind in ("dot_chain", "low_contrast_ghost"):
        nxt = "residue_micro_clean"
    elif changed_region_kind == "mixed":
        nxt = "mixed_roi_split_v2"
    elif roi == "thin_flex_cable":
        nxt = "thin_flex_line_preserve_v2"
    elif roi in _V21_SMOOTH_ROI:
        nxt = "surface_reverse_alpha_refine"
    else:
        nxt = "none"

    # mask source mapping
    mt = str(rec.get("mask_type", "") or "")
    if "logo_fallback" in mt or mt == "logo":
        mask_source = "logo_fallback"
    elif "alpha_stroke" in mt:
        mask_source = "alpha_stroke_intersection"
    elif "alpha" in mt:
        mask_source = "alpha_ncc"
    elif "stroke" in mt:
        mask_source = "stroke"
    else:
        mask_source = mt or "unknown"

    method = str(rec.get("v9_final_method", "") or "")
    micro_attempted = int(rec.get("v21_residue_micro_attempted", 0) or 0) > 0
    if method.startswith("v21_residue_micro") and \
            status in ("clean_repaired", "clean_covered"):
        micro_result = "passed"
    elif micro_attempted:
        micro_result = "failed"
    else:
        micro_result = "not_applicable"

    return {
        "v21_failure_taxonomy": {
            "primary_reject_class": primary if status == "auto_rejected"
            else "published",
            "residual_kind": residual_kind,
            "changed_region_kind": changed_region_kind,
            "recommended_next_candidate": nxt,
        },
        "v21_mask_quality": {
            "mask_source": mask_source,
            "alpha_ncc_score": round(float(
                rec.get("reverse_alpha_ncc", 0.0) or 0.0), 4),
            "stroke_coverage": round(float(
                rec.get("stroke_mask_confidence", 0.0) or 0.0), 4),
            "fallback_reason": "product_overlap" if (
                mask_source == "logo_fallback" and overlap > PRODUCT_OVERLAP_LOW)
            else "",
            "mask_product_overlap": round(overlap, 4),
        },
        "v21_residual_explain": {
            "authoritative_residual": (
                "ocr" if ocr_pos else "dot_chain" if dot_pos
                else "ghost_dot" if glyph_pos
                else "template_only" if template_pos else "none"),
            "template_only_suspect": fp_suspect,
            "on_product": bool(overlap > PRODUCT_OVERLAP_LOW),
            "on_background": bool(pure_bg >= PURE_BACKGROUND_SCORE_MIN),
            "micro_cleanup_attempted": micro_attempted,
            "micro_cleanup_result": micro_result,
        },
    }


# V23 reject buckets (patch plan §Patch 1). EXACT names — the report and CI key
# off these, so they must not drift.
V23_REJECT_BUCKETS = (
    "residual_after_reverse_alpha",
    "partial_glyph_residue",
    "visible_band_on_nonwhite_surface",
    "cover_or_fill_on_product",
    "mixed_product_background_starved",
    "thin_flex_continuity_risk",
    "protected_text_overlap",
    "dark_surface_blob_risk",
    "metallic_gradient_flattening",
    "no_precise_stroke_mask",
    "false_positive_residual_suspect",
    "unknown_safe_reject",
)


def classify_v23_reject_bucket(rec: dict) -> str:
    """Map an auto-rejected record to exactly one V23 bucket (patch plan
    §Patch 1). Pure function of signals already on ``rec`` — no decision change."""
    hard = " ".join(str(r) for r in (rec.get("v17_hard_fail_reasons") or []))
    reasons = " ".join(str(r) for r in (rec.get("reject_reasons") or []))
    blob = hard + " " + reasons
    rexp = rec.get("residual_explain") or {}
    v21 = rec.get("v21_failure_taxonomy") or {}
    residual_kind = v21.get("residual_kind", "")
    surface = str(rec.get("v22_surface_class") or rec.get("v9_roi_class") or "")
    mask_type = str(rec.get("mask_type", "") or "")

    if "partial_glyph_residue" in blob or "alpha_footprint_residue" in blob:
        return "partial_glyph_residue"
    if "visible_band_on_nonwhite" in blob:
        return "visible_band_on_nonwhite_surface"
    if "protected_text" in blob:
        return "protected_text_overlap"
    if any(t in blob for t in ("cover_on_product", "visible_patch_on_product",
                               "wedge", "slab")):
        return "cover_or_fill_on_product"
    if "thin_flex" in blob or "flex" in reasons or surface == "thin_flex_cable":
        return "thin_flex_continuity_risk"
    if "dark_surface" in blob or surface == "dark_smooth_product_surface":
        return "dark_surface_blob_risk"
    if "metallic" in blob or surface in ("metallic_or_reflective",
                                         "glass_or_gradient"):
        return "metallic_gradient_flattening"
    if residual_kind in ("readable_text", "dot_chain", "low_contrast_ghost"):
        return "residual_after_reverse_alpha"
    if bool(rexp.get("possible_product_structure_false_positive")):
        return "false_positive_residual_suspect"
    if "logo_fallback" in mask_type or mask_type in ("", "logo"):
        # A starved mixed region (product+background) with no precise stroke mask.
        cr = (rec.get("v21_failure_taxonomy") or {}).get("changed_region_kind")
        if cr == "mixed":
            return "mixed_product_background_starved"
        return "no_precise_stroke_mask"
    return "unknown_safe_reject"


def build_v23_records(rec: dict, status: str) -> dict:
    """V23 (patch plan §Patch 1). Derive the sharper reject taxonomy from signals
    already present on ``rec``. Pure post-processing — changes no decision."""
    bucket = classify_v23_reject_bucket(rec) if status == "auto_rejected" \
        else "published"
    reasons = [str(r) for r in (rec.get("reject_reasons") or [])]
    hard = [str(r) for r in (rec.get("v17_hard_fail_reasons") or [])]
    top_gate = hard[0] if hard else (reasons[0] if reasons else "none")
    v21 = rec.get("v21_failure_taxonomy") or {}
    return {
        "v23_reject_taxonomy": {
            "bucket": bucket,
            "primary_reason": (hard[0] if hard else
                               v21.get("primary_reject_class", "none")),
            "top_failed_method": str(rec.get("v9_final_method", "") or "none"),
            "top_failed_gate": top_gate,
            "roi_class": str(rec.get("v9_roi_class", "") or ""),
            "mask_type": str(rec.get("mask_type", "") or
                             rec.get("v23_mask_type", "") or "unknown"),
            "product_like_overlap": round(float(
                rec.get("v22_product_like_overlap", 0.0) or 0.0), 4),
            "candidate_count": int(rec.get("v18_n_safe_candidates", 0) or 0)
            + int(rec.get("reverse_alpha_variants_passed", 0) or 0),
            "safe_candidate_count": int(rec.get("v18_n_safe_candidates", 0) or 0),
        }
    }


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
