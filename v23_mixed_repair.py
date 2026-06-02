#!/usr/bin/env python3
"""V23 — component-aware mixed product/background repair (patch plan §Patch 3).

When a watermark crosses BOTH product and background, no single method is safe:

    * a background fragment needs a clone / median fill,
    * a product fragment needs reverse-alpha (+ micro residue cleanup only),
    * a full-box repair plants a visible patch.

V23 splits the solved-alpha footprint into connected fragments, repairs each with
the right tool, merges with feathering ONLY at fragment boundaries, runs a cheap
pre-audit, and returns ``None`` on any doubt. It builds on V20's
``segmented_reverse_alpha_background_clone`` (reverse-alpha on product pixels,
validated clone-offset fill on pure-background pixels) and adds a capped micro
residue cleanup over the product fragment.

Every output still passes through the unchanged V16/V17/V20/V22 P0 audit in the
pipeline — this can only add a safer candidate, never ship one the audit rejects.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import v23_masks

try:
    import sunsky_reverse_alpha as _sra
except Exception:  # pragma: no cover
    _sra = None
try:
    import v17_final_audit as _audit
except Exception:  # pragma: no cover
    _audit = None


def _as_tuple(bbox):
    if isinstance(bbox, dict):
        return (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
    return tuple(int(v) for v in bbox)


def _rectangular_boundary(original, out, bbox) -> bool:
    """Reject a repair that left a straight/rectangular boundary on product
    (patch plan §Patch 3 acceptance). Uses the frozen V13 patch-shape detector."""
    try:
        import v13_gates
        patch = v13_gates.detect_visible_patch_shape_v13(original, out, bbox)
        return bool(patch.get("hard_boundary_score", 0.0) >
                    getattr(v13_gates, "HARD_BOUNDARY_MAX", 0.62)
                    and patch.get("rectangularity", 0.0) > 0.7)
    except Exception:
        return False


def v23_component_aware_mixed_repair(image, bbox, watermark_mask=None,
                                     product_mask=None, mask_set=None,
                                     roi_class="", ctx=None) -> Optional[np.ndarray]:
    """Return a repaired full image, or ``None`` when no safe mixed repair exists.

    The repair NEVER paints a solid block over product pixels: product fragments
    are recovered with reverse-alpha and (optionally) a capped micro cleanup,
    background fragments are filled by clone-offset from a verified-clean source.
    """
    if _sra is None or not _sra.alpha_asset_available():
        return None
    bx, by, bw, bh = _as_tuple(bbox)
    if bw < 6 or bh < 4:
        return None
    if mask_set is None:
        try:
            mask_set = v23_masks.build_v23_mask_set(
                image, bbox, watermark_mask, product_mask, roi_class)
        except Exception:
            return None

    prod_frag = (mask_set.product_fragment_mask > 0)
    bg_frag = (mask_set.background_fragment_mask > 0)
    if not prod_frag.any() and not bg_frag.any():
        return None
    # A pure-product or pure-background footprint is not "mixed" — let the
    # dedicated single-surface candidates handle it.
    if not prod_frag.any() or not bg_frag.any():
        return None

    # Base: V20 reverse-alpha-on-product + clone-offset-on-background. This is the
    # proven mixed repair; V23 adds a capped micro cleanup on top.
    try:
        import v18_patch
        base = v18_patch.segmented_reverse_alpha_background_clone(
            image, bbox, watermark_mask, product_mask)
    except Exception:
        base = None
    if base is None:
        # Fall back to a direct reverse-alpha recovery (product-safe) so the
        # product fragment is at least recovered; background may stay if no clean
        # clone source exists — the audit will reject if residue survives.
        try:
            res = _sra.repair_sunsky_reverse_alpha(
                image, bbox, watermark_mask=watermark_mask,
                product_mask=product_mask, allow_thin_cleanup=True)
            base = res.image if (res is not None and res.image is not None) else None
        except Exception:
            base = None
    if base is None or base.shape != image.shape:
        return None

    out = base
    # Capped micro residue cleanup over the safe micro mask, restricted to the
    # PRODUCT fragment (background already clone-filled). Never a box fill.
    micro = (mask_set.safe_micro_mask > 0) & prod_frag
    if int(micro.sum()) >= 4:
        m = (micro.astype(np.uint8) * 255)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        try:
            cleaned = cv2.inpaint(out, m, 2, cv2.INPAINT_NS)
            if cleaned.shape == out.shape:
                out = cleaned
        except Exception:
            pass

    # Pre-audit: reject a rectangular boundary or a flattened metallic block.
    if _rectangular_boundary(image, out, (bx, by, bw, bh)):
        return None
    if _audit is not None:
        try:
            if _audit._metallic_block(image, out, (bx, by, bw, bh)):
                return None
        except Exception:
            pass
    return out
