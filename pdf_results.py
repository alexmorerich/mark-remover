#!/usr/bin/env python3
"""Before/after results PDF for the completed re-clean.

reclean rows : BEFORE = pristine backup (watermark present) | AFTER = current file on
               disk (re-cleaned, watermark gone) + a zoomed watermark-band strip.
restore rows : the current file (= pristine backup copied back) — a false positive that
               was needlessly inpainted and has now been restored. Shown to prove it is
               a clean product with no watermark and no inpaint damage.
"""
import os, sys, csv, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

MR = "/Users/alexkou/Documents/github/mark-remover"
CSV = os.path.join(MR, "reclean_routing.csv")
OUT = os.path.join(MR, "reclean_results_50.pdf")
N_RECLEAN, N_RESTORE = 40, 10
CELL, PAD = 500, 20


def to_pil(bgr, maxw):
    h, w = bgr.shape[:2]
    s = maxw / w
    return Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (int(w * s), max(1, int(h * s)))), cv2.COLOR_BGR2RGB))


def band(bgr):
    H = bgr.shape[0]
    return bgr[int(0.40 * H):int(0.56 * H), :]


def imread(p):
    try:
        return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    rows = list(csv.DictReader(open(CSV)))
    rc = [r for r in rows if r["action"] == "reclean"]
    rs = [r for r in rows if r["action"] == "restore"]
    random.seed()
    rc = random.sample(rc, min(N_RECLEAN, len(rc)))
    rs = random.sample(rs, min(N_RESTORE, len(rs)))
    Fs = font(15)
    pages = []

    # --- re-cleaned reals: backup (watermark) -> current (clean) ---
    for i, r in enumerate(rc, 1):
        before = imread(r["backup_path"])      # watermark present
        after = imread(r["cleaned_path"])      # current re-cleaned file
        if before is None or after is None:
            continue
        pb, pa = to_pil(before, CELL), to_pil(after, CELL)
        bb, ba = to_pil(band(before), CELL), to_pil(band(after), CELL)
        pw = PAD * 3 + CELL * 2
        ph = 10 + 26 + 22 + max(pb.height, pa.height) + 26 + max(bb.height, ba.height) + PAD
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        d.text((PAD, 8), f"[{i}/{len(rc)}]  REAL WATERMARK — re-cleaned   {os.path.basename(r['cleaned_path'])[:66]}", fill="black", font=Fs)
        y = 10 + 26
        d.text((PAD, y), "BEFORE  (original backup — watermark)", fill=(190, 0, 0), font=Fs)
        d.text((PAD * 2 + CELL, y), "AFTER  (re-cleaned, in place)", fill=(0, 130, 0), font=Fs)
        y += 22
        page.paste(pb, (PAD, y)); page.paste(pa, (PAD * 2 + CELL, y))
        y += max(pb.height, pa.height) + 6
        d.text((PAD, y), "zoom — watermark band:", fill=(90, 90, 90), font=Fs)
        y += 20
        page.paste(bb, (PAD, y)); page.paste(ba, (PAD * 2 + CELL, y))
        pages.append(page)

    # --- restored false positives: current file = pristine, no watermark ---
    for j, r in enumerate(rs, 1):
        img = imread(r["cleaned_path"])
        if img is None:
            continue
        pim = to_pil(img, CELL)
        pw = PAD * 2 + CELL
        ph = 10 + 26 + 22 + pim.height + PAD
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        d.text((PAD, 8), f"[FP {j}/{len(rs)}]  FALSE POSITIVE — restored   {os.path.basename(r['cleaned_path'])[:60]}", fill="black", font=Fs)
        d.text((PAD, 10 + 26), f"no watermark (OCR mis-fired on product text: '{r.get('ocr_match','')[:30]}') — pristine original restored, no inpaint damage", fill=(120, 80, 0), font=Fs)
        page.paste(pim, (PAD, 10 + 26 + 22))
        pages.append(page)

    pages[0].save(OUT, "PDF", save_all=True, append_images=pages[1:], resolution=100.0)
    print("PDF:", OUT, "| pages:", len(pages), "| size:", os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
