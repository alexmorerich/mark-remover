#!/usr/bin/env python3
"""product_preserve_clean.py — structure-preserving watermark removal for catalog images.

WHY THIS EXISTS
---------------
The production reclean's ``canonical_band`` method (and any LaMa/inpaint pass) removes the
watermark by INPAINTING the watermark band. Inpaint *fills/hallucinates*, so wherever a product
part sits under the mark it gets destroyed:
  * thin flex cables broken into pieces (battery-flex S-curve, earpiece ribbon),
  * gold flex contacts erased to black (home-key flex),
  * screws / components smeared,
  * a pointer arrowhead erased (back-camera-lens).

The ``sunsky-online.com`` mark is a SEMI-TRANSPARENT light-gray overlay (logo_bgr~180, peak
alpha~0.77), so the product is still present *under* it. The right removal therefore RECOVERS the
surface instead of inpainting it:

  * over WHITE background  -> the mark only darkens white slightly; set those pixels back to white.
  * over a DARK structure  -> the mark only lightens the dark surface; set those pixels back to the
                              structure's true (dark) colour. Continuity is preserved because we
                              never cut the structure — we only re-tone the mark's footprint.
  * solid product shapes (screws, arrow, phone, lenses) are PROTECTED from the white-pass, so they
    are never touched.

This validated 9/9 of the catalog-damage sample (job_fix9) with zero structure loss; it is the
approach to graft into the cleaning pipeline for white-background SKUs (the bulk of the catalog).

Alpha-unblend (product = (obs - a*logo)/(1-a) using assets/sunsky_alpha.png) is the in-principle
ideal recovery and is included as ``unblend``; it needs sub-pixel template alignment to fully cancel
the mark, so the re-tone methods above are the robust default.

USAGE
  python3 product_preserve_clean.py IN.jpg OUT.jpg            # auto: pick method from surface
  python3 product_preserve_clean.py --mode dark_cable IN OUT  # force a method
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_META = os.path.join(HERE, "assets", "sunsky_alpha_meta.json")
_ALPHA = os.path.join(HERE, "assets", "sunsky_alpha.png")

# canonical watermark row (fixed centre band, cy/H ~ 0.478)
BAND_Y0, BAND_Y1 = 0.40, 0.58
WHITE = 252           # in the band, anything below this on white bg is mark (true bg is ~255)
DARK = 160            # below this gray = dark product structure (mark over black tops out ~139)


def _band_mask(shape):
    H, W = shape[:2]
    m = np.zeros((H, W), bool)
    m[int(BAND_Y0 * H):int(BAND_Y1 * H)] = True
    return m


def _solid_mask(gray, band):
    """Solid product shapes in the band (screws / arrow / phone / lenses) to PROTECT — eroding the
    non-white mask drops thin mark strokes but keeps filled shapes; dilate adds a safety margin."""
    nonwhite = (gray < 235).astype(np.uint8)
    core = cv2.erode(nonwhite, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    solid = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    return solid & band


def whiten_on_white(bgr):
    """Mark sits on a WHITE background (screws/components/back-of-phone shots, pointer arrows).
    Whiten the mark; protect every solid product shape so it is never touched."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    band = _band_mask(bgr.shape)
    solid = _solid_mask(g, band)
    out = bgr.copy()
    out[band & (~solid) & (g < WHITE)] = (255, 255, 255)
    return out


def clean_dark_cable_on_white(bgr, dark_thr=DARK):
    """Mark CROSSES a dark thin structure (black flex cable) on white. Whiten the mark on white,
    and re-darken the mark where it lightened the cable back to the cable's true colour — so the
    cable stays continuous (inpaint would break it)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    band = _band_mask(bgr.shape)
    dark = g < dark_thr
    cable = cv2.morphologyEx((dark & band).astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # bridge mark gaps
    true_cable = bgr[(g < 90) & band]                                                # below mark-on-cable (~139)
    cc = np.median(true_cable, axis=0) if len(true_cable) > 20 else np.array([40, 40, 40])
    out = bgr.copy()
    out[band & (~dark) & (g < WHITE)] = (255, 255, 255)                              # mark over white
    out[band & (cable > 0) & (g > int(cc.mean()) + 22)] = cc                         # mark over cable
    return out


def band_residual_cleanup(bgr):
    """Universal final residual sweep over the FIXED watermark band, for marks that the primary
    clean (LaMa or structure-preserve) under-removed and the finder is blind to (faint gray on
    white / metal — the residual class). Whitens any surviving mark on the white background (solid
    product shapes protected) and re-darkens it on a uniform near-black surface. Idempotent on
    already-clean images; mid-tone / coloured / metallic surfaces are left untouched (no damage)."""
    H, W = bgr.shape[:2]
    y1, y2 = int(BAND_Y0 * H), int(BAND_Y1 * H)
    out = bgr.copy()
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    band = np.zeros((H, W), bool)
    band[y1:y2] = True
    bvals = g[band]
    if bvals.size == 0:
        return out
    # GATE: only sweep a GENUINELY PLAIN band — a flat white field, or a uniform achromatic near-black
    # surface. On textured / coloured / metallic / mixed bands this returns the input UNCHANGED, because
    # glyph_clean already removed the mark there precisely and a band-wide whiten or re-darken would
    # re-damage the product (the exact dark-band / white-wipe / splatter the gates now reject).
    if float((bvals > 238).mean()) >= 0.55:                            # flat white field
        solid = _solid_mask(g, band)
        out[band & (~solid) & (g < 248)] = (255, 255, 255)             # whiten any surviving faint mark
    blk = bgr[(g < 70) & band]
    if len(blk) > 40 and float(bvals.std()) < 55:                      # uniform (low-variance) dark band
        cc = np.median(blk, axis=0)
        if float(np.std(cc)) < 12:                                     # achromatic black, not a dark colour
            darku = cv2.morphologyEx(((g < 160) & band).astype(np.uint8), cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            out[band & (darku > 0) & (g > int(cc.mean()) + 22) & (g < 170)] = cc   # mark lightening black
    return out


_CANON_MATTE = None


def _canon_matte():
    """Tight (padding-trimmed) 'sunsky-online.com' glyph matte — the KNOWN mark shape."""
    global _CANON_MATTE
    if _CANON_MATTE is None:
        m = cv2.imread(os.path.join(HERE, "assets", "watermark-canonical.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            _CANON_MATTE = False
        else:
            r = np.where((m > 40).any(axis=1))[0]
            c = np.where((m > 40).any(axis=0))[0]
            _CANON_MATTE = m[r.min():r.max() + 1, c.min():c.max() + 1].astype(np.float32)
    return _CANON_MATTE


def _toward180(roi):
    """Per-pixel response that is HIGH on mark strokes regardless of mark direction: the mark pulls
    every covered pixel toward logo grey (~180), so the signed deviation-from-local-background in
    the direction of 180 isolates it (high-contrast strokes only; weak where bg≈180)."""
    bg = cv2.medianBlur(roi, 9)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bgg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.clip((g - bgg) * np.sign(180.0 - bgg), 0, None)


def glyph_clean(bgr, box, pad=0.18, thr=7, mk=9, use_template=False):
    """Structure-preserving removal by GLYPH-TIGHT inpaint — the texture-safe primary path.

    The sunsky mark is thin, semi-transparent text that pulls covered pixels TOWARD logo grey (~180).
    Default = DETECTION ONLY: mask the pixels deviating toward 180 from a local-median background and
    inpaint just those thin strokes. Safe on ANY surface (the mask is the glyphs, never the box). The
    detection ROI is widened to at least the canonical mark width so a too-narrow localisation still
    covers the whole mark (fixes partial-clean ghosts).

    ``use_template=True`` ALSO unions in the known 'sunsky-online.com' matte (scaled + correlation-
    aligned) — this is a GUARDED ESCALATION for LOW-contrast surfaces (lavender / grey metallic ≈ 180)
    where detection sees nothing. It is risky: the matte can misalign onto a product feature (lens,
    hole) and inpaint there, so the caller must only use it when detection leaves a residual AND must
    reject the result if it introduces damage (see owner_agent escalation)."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return bgr.copy()
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    mask = np.zeros((H, W), np.uint8)

    # detection strokes over a ROI widened to >= canonical mark width (fixes narrow localisation)
    want_w = max(int(bw * (1 + 2 * pad)), 280)
    X1, X2 = max(0, cx - want_w // 2), min(W, cx + want_w // 2)
    Y1, Y2 = max(0, y1 - int(bh * pad)), min(H, y2 + int(bh * pad))
    roi = bgr[Y1:Y2, X1:X2]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg = cv2.cvtColor(cv2.medianBlur(roi, mk), cv2.COLOR_BGR2GRAY).astype(np.int16)
    L = 180
    dev = g - bg
    toward = np.sign(dev) == np.sign(L - bg)
    stroke = ((np.abs(dev) > thr) & toward
              & (g > np.minimum(bg, L) - 10) & (g < np.maximum(bg, L) + 10))
    mask[Y1:Y2, X1:X2] = stroke.astype(np.uint8) * 255

    if use_template:                                          # guarded escalation (low-contrast only)
        matte = _canon_matte()
        if matte is not False:
            mh, mw = matte.shape
            gw = max(int(bw * 1.05), 260)
            gh = max(8, int(gw * mh / mw))
            sw, sh = 50, 22                                   # tight alignment search (avoid drift onto product)
            rX1, rY1 = max(0, cx - gw // 2 - sw), max(0, cy - gh // 2 - sh)
            rX2, rY2 = min(W, cx + gw // 2 + sw), min(H, cy + gh // 2 + sh)
            resp = _toward180(bgr[rY1:rY2, rX1:rX2])
            templ = (cv2.resize(matte, (gw, gh)) / 255.0).astype(np.float32)
            if resp.shape[0] >= gh and resp.shape[1] >= gw:
                gx, gy = cv2.minMaxLoc(cv2.matchTemplate(resp.astype(np.float32), templ, cv2.TM_CCORR))[3]
                gx, gy = rX1 + gx, rY1 + gy
            else:
                gx, gy = cx - gw // 2, cy - gh // 2
            gm = (cv2.resize(matte, (gw, gh)) > 45).astype(np.uint8) * 255
            ya, yb, xa, xb = max(0, gy), min(H, gy + gh), max(0, gx), min(W, gx + gw)
            if yb > ya and xb > xa:
                mask[ya:yb, xa:xb] = np.maximum(mask[ya:yb, xa:xb], gm[ya - gy:yb - gy, xa - gx:xb - gx])

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3), k3)
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def _alpha_meta():
    meta = json.load(open(_META))
    a0 = cv2.imread(_ALPHA, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    a0 /= a0.max()
    return a0, float(np.mean(meta["logo_bgr"])), float(meta["peak_alpha"])


def unblend(bgr, box, feather=1.0):
    """Alpha-unblend the mark inside ``box`` (recovers the surface under a semi-transparent mark).
    Ideal recovery but alignment-sensitive — align the alpha template before relying on this."""
    a0, logo, peak = _alpha_meta()
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    a = cv2.resize(a0, (max(1, x2 - x1), max(1, y2 - y1))) * peak
    amap = np.zeros((H, W), np.float32)
    amap[y1:y2, x1:x2] = a
    if feather:
        amap = cv2.GaussianBlur(amap, (0, 0), feather)
    amap = np.clip(amap, 0, 0.93)[..., None]
    return np.clip((bgr.astype(np.float32) - amap * logo) / (1.0 - amap), 0, 255).astype(np.uint8)


def _auto_mode(bgr):
    """Heuristic: a dark thin structure through the band -> dark_cable; else white."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    band = _band_mask(bgr.shape)
    thin_dark = ((g < 90) & band).mean()        # true-black pixels in the band
    return "dark_cable" if 0.002 < thin_dark < 0.08 else "white"


def clean(bgr, mode="auto"):
    if mode == "auto":
        mode = _auto_mode(bgr)
    if mode == "dark_cable":
        return clean_dark_cable_on_white(bgr)
    return whiten_on_white(bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", default="auto", choices=["auto", "white", "dark_cable"])
    a = ap.parse_args()
    bgr = cv2.imdecode(np.fromfile(a.src, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"cannot read {a.src}")
    out = clean(bgr, a.mode)
    cv2.imwrite(a.dst, out)
    print(f"wrote {a.dst} (mode={a.mode if a.mode!='auto' else _auto_mode(bgr)})")


if __name__ == "__main__":
    main()
