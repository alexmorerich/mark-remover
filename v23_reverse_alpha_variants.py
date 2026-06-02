#!/usr/bin/env python3
"""V23 — surface-specific reverse-alpha refinement beam (patch plan §Patch 5).

Reverse-alpha is the safest watermark remover (it subtracts the overlay and
keeps the real product pixels beneath), but the remaining auto-rejects need
surface-specific variants that suppress faint residue WITHOUT flattening the
product surface or planting a bright blob.

These variants all operate on top of a solved reverse-alpha placement, so each
one is a real-pixel recovery — never a box fill. They are appended to the
reverse-alpha beam in :func:`sunsky_reverse_alpha.build_variant_beam` BEFORE any
destructive method, screened by the same cheap local pre-screen
(``_screen_variant``), ranked by ``(residual, ghost_dot, changed_product)``, and
finally audited authoritatively by the V17/V20/V22 final audit. Nothing here can
ship an output the audit would reject.

To avoid an import cycle (``sunsky_reverse_alpha`` calls into this module) every
sunsky helper is imported lazily at call time — the module is already loaded by
the time these run.
"""
from __future__ import annotations

import cv2
import numpy as np

# Variant identifiers (appended to sunsky_reverse_alpha.VARIANT_NAMES).
V23_VARIANT_NAMES = (
    "v23_ra_local_alpha_plane",
    "v23_ra_dark_surface_bias",
    "v23_ra_metallic_gradient_locked",
    "v23_ra_cardboard_texture_reinject",
    "v23_ra_near_white_surface_cleanup",
    "v23_ra_baseline_component_cleanup",
)

GAIN_LO, GAIN_HI = 0.75, 1.25     # alpha-gain plane range (patch plan §Patch 5)


def _foot(sra, image, placement):
    """Footprint mask (bool) of the solved alpha for this placement."""
    buf = sra._full_alpha_buffer(image.shape, placement)
    return buf, (buf >= 0.02)


def _ring_stats(image, bbox):
    bx, by, bw, bh = bbox
    H, W = image.shape[:2]
    pad = max(6, max(bw, bh) // 3)
    y1, y2 = max(0, by - pad), min(H, by + bh + pad)
    x1, x2 = max(0, bx - pad), min(W, bx + bw + pad)
    reg = image[y1:y2, x1:x2]
    if reg.size == 0:
        return 200.0, 6.0
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.median(g)), float(np.std(g))


def _local_alpha_plane(sra, image, placement):
    """v23_ra_local_alpha_plane — fit alpha_gain(x,y) = a + b*x + c*y over the
    footprint (gain clamped to [0.75, 1.25], gradient limited) and pick the plane
    that minimises high-pass residual. A spatially-varying gain corrects an
    overlay that is slightly stronger on one side of the glyph."""
    buf = sra._full_alpha_buffer(image.shape, placement)
    foot = buf >= 0.02
    if int(foot.sum()) < 8:
        return None
    ys, xs = np.where(foot)
    y0, x0 = ys.mean(), xs.mean()
    yn = (ys - y0) / (ys.ptp() + 1e-6)
    xn = (xs - x0) / (xs.ptp() + 1e-6)
    best_img, best_resid = None, 2.0
    # Deterministic small grid: constant level + mild x/y slopes.
    for a in np.linspace(0.85, 1.15, 4):
        for b in (-0.12, 0.0, 0.12):
            for c in (-0.12, 0.0, 0.12):
                gain = np.ones_like(buf)
                g = np.clip(a + b * xn + c * yn, GAIN_LO, GAIN_HI)
                gain[ys, xs] = g
                cand = sra.apply_reverse_alpha(image, buf * gain,
                                               placement.logo_bgr)
                rc = sra.residual_confidence(cand, placement.bbox,
                                             placement.alpha_map)
                if rc < best_resid:
                    best_resid, best_img = rc, cand
    return best_img


def _dark_surface_bias(sra, image, placement):
    """v23_ra_dark_surface_bias — on dark smooth surfaces, avoid gray/white
    inpaint: recover with reverse-alpha then pull footprint pixels toward the
    local dark median, keeping luma movement small. A bright blob is rejected by
    the screen (``_dark_blob``)."""
    med, _std = _ring_stats(image, placement.bbox)
    if med >= 110.0:
        return None     # not a dark surface
    buf = sra._full_alpha_buffer(image.shape, placement)
    foot = buf >= 0.02
    if int(foot.sum()) < 6:
        return None
    base = sra.apply_placement(image, placement)
    out = base.astype(np.float32)
    # Clamp recovered footprint luma toward the dark ring median (no brightening).
    target = med
    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bright = foot & (g > target + 10.0)
    if bright.any():
        scale = (target + 10.0) / np.maximum(g[bright], 1.0)
        out[bright] = out[bright] * scale[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _metallic_gradient_locked(sra, image, placement):
    """v23_ra_metallic_gradient_locked — on metallic / glass surfaces allow only
    a low-frequency correction ALONG the local gradient direction (preserving the
    reflection), never a flattening fill. The audit rejects flattening
    (``_metallic_block``)."""
    buf = sra._full_alpha_buffer(image.shape, placement)
    foot = buf >= 0.02
    if int(foot.sum()) < 8:
        return None
    base = sra.apply_placement(image, placement)
    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))))
    gy = float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))))
    # Blur ALONG the dominant gradient axis to smooth residue while keeping the
    # perpendicular reflection edge energy.
    if gx >= gy:
        k = (9, 1)
    else:
        k = (1, 9)
    smooth = cv2.blur(base, k)
    foot_u8 = (foot.astype(np.uint8) * 255)
    return sra._blend_at_mask(base, smooth, foot_u8, feather_px=2)


def _cardboard_texture_reinject(sra, image, placement):
    """v23_ra_cardboard_texture_reinject — on matte / cardboard backs, reverse-
    alpha leaves a faint smooth patch. Reinject low-amplitude texture sampled
    from the surrounding ring so the footprint matches the grain; reject if it
    becomes a visible rectangle (handled downstream)."""
    med, std = _ring_stats(image, placement.bbox)
    if std < 4.0 or std > 40.0:
        return None     # not a textured matte surface
    buf = sra._full_alpha_buffer(image.shape, placement)
    foot = buf >= 0.02
    if int(foot.sum()) < 8:
        return None
    base = sra.apply_placement(image, placement)
    out = base.astype(np.float32)
    # Deterministic low-amplitude grain from a fixed Laplacian-of-ring pattern.
    bx, by, bw, bh = placement.bbox
    H, W = image.shape[:2]
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grain = (gray - cv2.GaussianBlur(gray, (0, 0), 1.5))
    amp = float(np.clip(std * 0.25, 0.0, 6.0))
    grain = np.clip(grain, -amp, amp)
    out[foot] = out[foot] + grain[foot][:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _near_white_surface_cleanup(sra, image, placement):
    """v23_ra_near_white_surface_cleanup — on near-white trays / plastic that are
    NOT pure background, do a footprint micro-cleanup (median) and local luma
    match instead of a full white fill, so an attached product silhouette is
    preserved."""
    med, _std = _ring_stats(image, placement.bbox)
    if med < 200.0:
        return None     # not near-white
    buf = sra._full_alpha_buffer(image.shape, placement)
    foot = buf >= 0.02
    if int(foot.sum()) < 6:
        return None
    base = sra.apply_placement(image, placement)
    smooth = cv2.medianBlur(base, 5)
    foot_u8 = (foot.astype(np.uint8) * 255)
    return sra._blend_at_mask(base, smooth, foot_u8, feather_px=2)


def _baseline_component_cleanup(sra, image, placement):
    """v23_ra_baseline_component_cleanup — strip the small ``s`` / ``m`` /
    ``.com`` / paired-dot fragments that survive on the Sunsky baseline. Only
    small components aligned along the lower band of the footprint are touched,
    and the aggregate area is capped."""
    buf = sra._full_alpha_buffer(image.shape, placement)
    foot = (buf >= 0.02)
    if int(foot.sum()) < 8:
        return None
    base = sra.apply_placement(image, placement)
    bx, by, bw, bh = placement.bbox
    box_area = float(max(1, bw * bh))
    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.0))
    cand = ((hp >= 3.0) & (hp <= 58.0) & foot).astype(np.uint8)
    try:
        n, lab, stats, _c = cv2.connectedComponentsWithStats(cand, 8)
    except Exception:
        return None
    keep = np.zeros_like(cand)
    total = 0.0
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        if a / box_area > 0.015:
            continue
        keep[lab == i] = 255
        total += a
    if total <= 0 or total / box_area > 0.06:
        return None
    dil = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    return cv2.inpaint(base, dil, 2, cv2.INPAINT_NS)


_BUILDERS = (
    ("v23_ra_local_alpha_plane", _local_alpha_plane),
    ("v23_ra_dark_surface_bias", _dark_surface_bias),
    ("v23_ra_metallic_gradient_locked", _metallic_gradient_locked),
    ("v23_ra_cardboard_texture_reinject", _cardboard_texture_reinject),
    ("v23_ra_near_white_surface_cleanup", _near_white_surface_cleanup),
    ("v23_ra_baseline_component_cleanup", _baseline_component_cleanup),
)


def build_v23_variants(image, placement, *, watermark_mask=None,
                       product_mask=None, protected_text_mask=None,
                       roi_class=""):
    """Return a list of ``(name, candidate_image, placement)`` V23 reverse-alpha
    refinement variants. ``placement`` is the chosen (NCC/fixed) AlphaPlacement
    from the caller. Each builder returns ``None`` when it does not apply to the
    surface; failures are swallowed so the beam never breaks."""
    try:
        import sunsky_reverse_alpha as sra
    except Exception:
        return []
    out = []
    for name, fn in _BUILDERS:
        try:
            cand = fn(sra, image, placement)
        except Exception:
            cand = None
        if cand is not None and cand.shape == image.shape:
            out.append((name, cand, placement))
    return out
