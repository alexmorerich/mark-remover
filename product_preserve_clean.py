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
