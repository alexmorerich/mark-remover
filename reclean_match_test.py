#!/usr/bin/env python3
"""Localize the sunsky watermark by template matching (robust to OCR's partial hits).

The watermark is a fixed-appearance, horizontally-centered, semi-transparent overlay.
We match b2bweb's canonical alpha matte (white 'sunsky-online.com' on black) against a
CLAHE-boosted grayscale of the image, multi-scale, in the central band, trying BOTH
polarities (the mark is lighter on dark backs, ~invisible on light ones). Returns the
full bbox + score. This stage does NOT inpaint — it draws the found box for review.
"""
import os, sys, glob
import numpy as np, cv2

TEMPLATE = "/Users/alexkou/Documents/github/b2bweb/scripts/watermark-canonical-alpha.png"
DBG = "/Users/alexkou/Documents/github/mark-remover/reclean_debug"


def load_template():
    t = cv2.imread(TEMPLATE, cv2.IMREAD_GRAYSCALE)  # white text on black
    # tight-crop to the glyph ink so the aspect ratio is the real text aspect
    ys, xs = np.where(t > 40)
    t = t[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return t


def find_watermark(bgr, t):
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    y0, y1 = int(0.28 * H), int(0.72 * H)
    band = gray[y0:y1]
    th0, tw0 = t.shape[:2]
    best = (-2, 0, 0, 0, 0, "")
    # watermark width is ~0.30-0.62 of image width; sweep scales + both polarities
    for frac in [0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60]:
        tw = int(frac * W)
        th = max(1, int(tw * th0 / tw0))
        if th >= band.shape[0] or tw >= W:
            continue
        for pol, tt0 in (("pos", t), ("neg", 255 - t)):
            tt = cv2.resize(tt0, (tw, th))
            res = cv2.matchTemplate(band, tt, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mloc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (mx, mloc[0], mloc[1] + y0, tw, th, pol)
    score, x, y, tw, th, pol = best
    return dict(score=round(float(score), 3), bbox=[x, y, x + tw, y + th], pol=pol)


def main():
    t = load_template()
    print("template (tight-cropped):", t.shape[::-1], "aspect %.1f:1" % (t.shape[1] / t.shape[0]))
    samples = sorted(glob.glob(os.path.join(DBG, "orig_*.jpg")))
    for p in samples:
        bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        r = find_watermark(bgr, t)
        x1, y1, x2, y2 = r["bbox"]
        H, W = bgr.shape[:2]
        vis = bgr.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        out = p.replace("orig_", "match_")
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(out)
        print(f"{os.path.basename(p)[:34]:34s} {W}x{H} score={r['score']} pol={r['pol']} bbox={r['bbox']}")


if __name__ == "__main__":
    main()
