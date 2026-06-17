#!/usr/bin/env python3
"""Build a before/after comparison PDF for 50 random HIGH+MED suspects.
BEFORE = current (buggy) cleaned file in place; AFTER = canonical-band re-clean from backup.
Each page: full before|after + a zoomed crop of the central watermark band."""
import os, sys, csv, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/Users/alexkou/Documents/github/mark-remover")
import reclean

CSV = "/Users/alexkou/Documents/github/mark-remover/reclean_pairs.csv"
OUT = "/Users/alexkou/Documents/github/mark-remover/reclean_sample_50.pdf"
N = 50
CELL = 500
PAD = 20


def to_pil(bgr, maxw):
    h, w = bgr.shape[:2]
    s = maxw / w
    im = cv2.resize(bgr, (int(w * s), max(1, int(h * s))))
    return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def band(bgr):
    H = bgr.shape[0]
    return bgr[int(0.40 * H):int(0.56 * H), :]


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    reclean.rb._init_worker("cpu")
    rows = [r for r in csv.DictReader(open(CSV)) if r["risk"] in ("HIGH", "MED")]
    random.seed()
    samp = random.sample(rows, min(N, len(rows)))
    Ft, Fs = font(20), font(15)
    pages = []
    for i, r in enumerate(samp, 1):
        cp, bp = r["cleaned_path"], r["backup_path"]
        before = reclean.rb._imread(cp)
        after, _ = reclean.reclean_one(bp, verify=False)
        if before is None or after is None:
            print("skip", cp); continue
        pb, pa = to_pil(before, CELL), to_pil(after, CELL)
        bb, ba = to_pil(band(before), CELL), to_pil(band(after), CELL)
        pw = PAD * 3 + CELL * 2
        ph = 10 + 26 + 22 + max(pb.height, pa.height) + 26 + max(bb.height, ba.height) + PAD
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        d.text((PAD, 8), f"[{i}/{len(samp)}]  {r['risk']}  {os.path.basename(cp)[:74]}", fill="black", font=Fs)
        y = 10 + 26
        d.text((PAD, y), "BEFORE  (current — residual watermark)", fill=(190, 0, 0), font=Fs)
        d.text((PAD * 2 + CELL, y), "AFTER  (re-cleaned from backup)", fill=(0, 130, 0), font=Fs)
        y += 22
        page.paste(pb, (PAD, y)); page.paste(pa, (PAD * 2 + CELL, y))
        y += max(pb.height, pa.height) + 6
        d.text((PAD, y), "zoom — watermark band:", fill=(90, 90, 90), font=Fs)
        y += 20
        page.paste(bb, (PAD, y)); page.paste(ba, (PAD * 2 + CELL, y))
        pages.append(page)
        if i % 10 == 0:
            print(f"built {i}/{len(samp)}", flush=True)
    pages[0].save(OUT, "PDF", save_all=True, append_images=pages[1:], resolution=100.0)
    print("PDF:", OUT, "| pages:", len(pages), "| size:", os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
