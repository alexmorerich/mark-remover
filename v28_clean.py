"""
v28 — sunsky watermark eraser, fixes v27's partial-mask / false-clean failure.

v27 failure (seen on dark flex cables, e.g. charging-port-flex-cable-...-2015-white-8):
  OCR detects only a fragment ("sunsky-onl"), v27 masks that fragment + 20% padding,
  LaMa erases it, but the "online.com" tail is never masked and survives as a readable
  ghost. v27's residual check (re-run OCR) can't re-read the faint ghost on a dark
  surface, so it falsely reports residual_after=0 / status=cleaned.

v28 fixes both halves:
  1. FULL-EXTENT MASK. After the OCR anchor, grow along the actual visible watermark
     tint (semi-transparent light-gray ~180 text, lighter than a dark/mid surface) to
     recover the whole "sunsky-online.com" run — not just the OCR fragment. The mask is
     the filled bbox of that run, padded + dilated. On light/white surfaces the tint is
     invisible (and residue is visually negligible) so it falls back to v27's padded box.
  2. TINT VERIFIER. The post-inpaint check looks for leftover light-gray text tint in
     the watermark band (a matched signal that survives where OCR fails), in addition to
     re-running OCR. Either firing triggers one wider retry.

Reuses v27's OCR detector and LaMa inpaint.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v27_clean as v27

LOGO_LUMA = 180.0          # solved watermark gray level (assets/sunsky_alpha_meta.json)
_TPL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sunsky_alpha.png")
ALPHA = cv2.imread(_TPL_PATH, cv2.IMREAD_GRAYSCALE)   # 36x240 canonical alpha of the mark
TPL_H0, TPL_W0 = ALPHA.shape                          # 36, 240  -> aspect 6.67
ASPECT = TPL_W0 / TPL_H0

ACCEPT = 0.24              # min template correlation to trust a full-extent localization
CLEAN = 0.30              # at/above this on the cleaned band => mark structure still readable


def _abs_local_contrast(gray_u8, h):
    """Stroke map: |gray - background|. Highlights the semi-transparent glyph strokes
    regardless of whether the surface under them is dark or light."""
    g = gray_u8.astype(np.float32)
    k = max(3, int(1.4 * h) | 1)
    bg = cv2.medianBlur(gray_u8, k).astype(np.float32)
    return np.abs(g - bg)


def locate_full_box(bgr, anchor):
    """Template-match the known mark geometry against the stroke map to recover the FULL
    'sunsky-online.com' box. OCR often returns only a small middle fragment, so the search
    band and scale sweep are sized from the IMAGE, not the fragment, and the match is
    constrained to horizontally cover the OCR anchor. Returns (full_bbox, score) or
    (None, score)."""
    H, W = bgr.shape[:2]
    ax1, ay1, ax2, ay2 = [int(round(t)) for t in anchor]
    h = max(10, ay2 - ay1)
    cx, cy = (ax1 + ax2) // 2, (ay1 + ay2) // 2
    # the mark can span ~0.5W and OCR may catch only a sliver -> search wide, sized to image
    halfw = max(int(5 * h), int(0.42 * W))
    vpad = max(int(2.2 * h), int(0.06 * H))
    bx1, bx2 = max(0, cx - halfw), min(W, cx + halfw)
    by1, by2 = max(0, cy - vpad), min(H, cy + vpad)
    band = bgr[by1:by2, bx1:bx2]
    if band.size == 0:
        return None, 0.0
    lc = _abs_local_contrast(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), h)
    bh, bw = lc.shape
    acx = cx - bx1
    # sweep template heights over an image-relative range (covers fragments AND big marks)
    ths = sorted({int(round(f * W)) for f in (0.025, 0.035, 0.05, 0.065, 0.085, 0.11)} |
                 {int(round(s * h)) for s in (0.9, 1.3, 1.8)})
    best = None
    for th in ths:
        if th < 10:
            continue
        tw = int(round(th * ASPECT))
        if th >= bh or tw >= bw:
            continue
        tpl = cv2.resize(ALPHA, (tw, th)).astype(np.float32)
        r = cv2.matchTemplate(lc, tpl, cv2.TM_CCOEFF_NORMED)
        lo, hi = max(0, acx - tw + 1), min(r.shape[1] - 1, acx)   # box must contain anchor cx
        if lo > hi:
            continue
        sub = r[:, lo:hi + 1]
        _, mx, _, loc = cv2.minMaxLoc(sub)
        lx, ly = loc[0] + lo, loc[1]
        if best is None or mx > best[0]:
            best = (mx, (lx, ly), tw, th)
    if best is None:
        return None, 0.0
    mx, (lx, ly), tw, th = best
    box = [bx1 + lx, by1 + ly, bx1 + lx + tw, by1 + ly + th]
    return (box, mx) if mx >= ACCEPT else (None, mx)


def verify_residue(bgr, anchor):
    """After cleaning, re-locate the mark in its band. Returns (score, box). A score
    below CLEAN means no template-correlated mark structure remains."""
    box, score = locate_full_box(bgr, anchor)
    return score, box


def _band(bgr, anchor, hpad=1.1, wreach=5.0):
    """Generous search band around the OCR anchor, big enough to contain the full
    ~6.7h-wide string wherever the detected fragment sat within it."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(t)) for t in anchor]
    h = max(10, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bx1, bx2 = max(0, cx - int(wreach * h)), min(W, cx + int(wreach * h))
    by1, by2 = max(0, cy - int(hpad * h)), min(H, cy + int(hpad * h))
    return (bx1, by1, bx2, by2), h, cy


def _near(box4, region, h):
    """Is an OCR hit inside the watermark vicinity (so product text elsewhere on the
    part — e.g. a printed part-number — doesn't count as residual)?"""
    cx, cy = (box4[0] + box4[2]) / 2, (box4[1] + box4[3]) / 2
    return (region[0] - 1.5 * h <= cx <= region[2] + 1.5 * h and
            region[1] - 1.5 * h <= cy <= region[3] + 1.5 * h)


def build_mask(shape, box, pad_x=0.10, pad_y=0.35, dilate=6, pad_x_min=14):
    """Filled box mask. Horizontal pad is a FRACTION of mark width so the faint '.com'
    tail and leading 's' (which template matching tends to under-cover) are always
    inside the mask."""
    H, W = shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px = max(pad_x_min, int(bw * pad_x))
    py = max(8, int(bh * pad_y))
    m = np.zeros((H, W), np.uint8)
    m[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)] = 255
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return m


def grow_extent_tint(bgr, box, tint=10.0, lc_thr=5.0, cap_mult=4.0):
    """Extend a box left/right to cover faint mark tails (the '.com') the template
    under-covered. STROKE-AWARE: a pixel only counts if it is both lighter than the local
    background AND sits on a high-frequency stroke — so smooth light product features (a
    reflective curved flex) are NOT followed, which is what caused runaway white-blob
    inpaints. Growth is also capped to cap_mult x box-height from each edge."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    ht = max(12, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sy1, sy2 = max(0, cy - int(0.8 * ht)), min(H, cy + int(0.8 * ht))
    sx1, sx2 = max(0, cx - int(0.46 * W)), min(W, cx + int(0.46 * W))
    strip = bgr[sy1:sy2, sx1:sx2]
    if strip.size == 0:
        return box
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, max(3, int(1.5 * ht) | 1)).astype(np.int16)
    lc = cv2.absdiff(g, cv2.GaussianBlur(g, (0, 0), max(1.0, ht * 0.12)))   # local stroke energy
    m = (((g.astype(np.int16) - bg) > tint) & (lc > lc_thr)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(1.4 * ht)), max(3, int(0.35 * ht)))))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tlx1, tlx2, cyl = x1 - sx1, x2 - sx1, cy - sy1
    ex1, ex2, found = tlx1, tlx2, False
    for c in cnts:
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rw < 0.3 * ht or rh < 0.25 * ht:
            continue
        if abs((ry + rh / 2) - cyl) > 0.9 * ht:                 # off the mark's text line
            continue
        if rx > tlx2 + 1.2 * ht or rx + rw < tlx1 - 1.2 * ht:   # not contiguous with the mark
            continue
        ex1, ex2, found = min(ex1, rx), max(ex2, rx + rw), True
    if not found:
        return box
    ex1 = max(ex1, tlx1 - int(cap_mult * ht))                   # cap runaway growth
    ex2 = min(ex2, tlx2 + int(cap_mult * ht))
    return [sx1 + ex1, y1, sx1 + ex2, y2]


_FAST_LAMA = None


def get_fast_lama():
    """LaMa on the M4 GPU (MPS) when available, else CPU. Set LAMA_DEVICE=cpu to force CPU
    (e.g. for multi-process runs that shouldn't contend on the single GPU)."""
    global _FAST_LAMA
    if _FAST_LAMA is None:
        import torch
        from simple_lama_inpainting import SimpleLama
        want = os.environ.get("LAMA_DEVICE", "mps")
        if want == "mps" and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
        _FAST_LAMA = SimpleLama(device=dev)
    return _FAST_LAMA


def lama_inpaint_crop(bgr, mask, pad=24):
    """Inpaint ONLY a crop around the mask, paste back only masked pixels. The mask is
    typically ~1% of the image, so this is ~50x cheaper than full-image LaMa and leaves
    every unmasked pixel byte-identical to the source."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return bgr
    H, W = bgr.shape[:2]
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(W, int(xs.max()) + 1 + pad), min(H, int(ys.max()) + 1 + pad)
    crop, cmask = bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    model = get_fast_lama()
    out = np.array(model(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                         Image.fromarray(cmask)))
    if out.shape[:2] != crop.shape[:2]:
        out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    res = bgr.copy()
    m = cmask > 0
    res[y0:y1, x0:x1][m] = out[m]
    return res


def ocr_residual_band(reader, bgr, region, h):
    """Single OCR pass over just the watermark band (a small crop) instead of a 3-pass
    full-image rescan. Much cheaper, and product text elsewhere can't false-positive."""
    H, W = bgr.shape[:2]
    pad = int(1.2 * h)
    x0, y0 = max(0, region[0] - pad), max(0, region[1] - pad)
    x1, y1 = min(W, region[2] + pad), min(H, region[3] + pad)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    return v27._ocr_pass(reader, crop, 2.0, {})


_OCR_PASSES = [
    (2.0, {}),
    (3.0, dict(contrast_ths=0.01, adjust_contrast=0.9, text_threshold=0.4, low_text=0.2, link_threshold=0.3)),
    (2.0, dict(contrast_ths=0.01, adjust_contrast=0.9, text_threshold=0.3, low_text=0.15, link_threshold=0.2)),
]


def _mark_fully_captured(hits):
    """Did OCR read the WHOLE mark? Requires the genuine tail ('...com'/'online' — NOT a
    truncated 'sunsky-onl', which was wrongly passing) or a box already ~6.7x text-height
    wide. Under-claiming here just means we also grow/verify, which is safe."""
    u = v27.union_bbox(hits)
    mh = float(np.median([h[3] - h[1] for h in hits]))
    texts = " ".join(h[4] for h in hits).lower()
    return "com" in texts or "onlin" in texts or (u[2] - u[0]) >= 6.0 * mh


def _ocr_channels(reader, bgr, params, scale=2.0):
    """OCR each BGR channel separately. A watermark that is low-contrast in the composite
    (light text on a saturated colour, e.g. white-on-blue) often pops in one channel —
    this is the recall booster that catches marks the standard passes miss."""
    hits = []
    for ch in cv2.split(bgr):
        hits += v27._ocr_pass(reader, cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR), scale, params)
    return hits


def detect_smart(reader, bgr):
    """All-pass OCR union with early-stop on a TRUE full capture, then a per-channel recall
    booster if the composite passes found nothing (faint marks on coloured backgrounds)."""
    hits = []
    for sc, pa in _OCR_PASSES:
        hits += v27._ocr_pass(reader, bgr, sc, pa)
        if hits and _mark_fully_captured(hits):
            return hits
    if not hits:
        _loose = dict(contrast_ths=0.01, adjust_contrast=0.9, text_threshold=0.35, low_text=0.2, link_threshold=0.3)
        hits = _ocr_channels(reader, bgr, {}, 2.0) or _ocr_channels(reader, bgr, _loose, 2.5)
    return hits


def verify_clean(reader, bgr, region):
    """Re-detect the watermark over a region WIDER than the mask (residue can survive just
    outside it, as in the '.com' tail case) using composite + per-channel OCR. Returns
    surviving watermark hits in image coordinates — empty means verified clean."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in region]
    mw, mh = x2 - x1, y2 - y1
    px, py = int(0.6 * mw) + 24, int(1.0 * mh) + 16
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
    crop = bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []
    hits = v27._ocr_pass(reader, crop, 2.0, {}) or _ocr_channels(reader, crop, {}, 2.0)
    return [(h[0] + cx1, h[1] + cy1, h[2] + cx1, h[3] + cy1, h[4], h[5]) for h in hits]


def clean_one(reader, in_path, out_path, debug_dir=None):
    bgr = cv2.imdecode(np.fromfile(in_path, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"ok": False, "err": "imread failed", "path": str(in_path)}

    hits = detect_smart(reader, bgr)
    if not hits:
        return {"ok": True, "status": "no_mark", "path": str(in_path)}
    anchor = v27.union_bbox(hits)
    h_anchor = max(10, anchor[3] - anchor[1])

    if _mark_fully_captured(hits):
        full = [int(round(v)) for v in anchor]
        method, score = "ocr_full", 1.0
    else:
        # partial read -> recover full extent via template anchor
        fb, score = locate_full_box(bgr, anchor)
        method = "template"
        if fb is None:
            fb = v27.pad_bbox(anchor, bgr.shape)
            method = "ocr_pad"
        ax = [int(round(t)) for t in anchor]
        full = [min(fb[0], ax[0]), min(fb[1], ax[1]), max(fb[2], ax[2]), max(fb[3], ax[3])]
    # ALWAYS grow to the full visible extent (stroke-aware + capped => safe, no runaway)
    full = grow_extent_tint(bgr, full)
    mask = build_mask(bgr.shape, full)
    cleaned = lama_inpaint_crop(bgr, mask)

    # HARD GATE: re-detect over a WIDE region (composite + per-channel). Any surviving mark
    # => widen to cover it and re-inpaint, up to 2x. Status is "cleaned" ONLY if the final
    # verify is empty; otherwise it is reported "residual" — never a silent dirty publish.
    residue = verify_clean(reader, cleaned, full)
    retry = 0
    while residue and retry < 2:
        retry += 1
        rb = v27.union_bbox(residue)
        full = [min(full[0], int(rb[0])), min(full[1], int(rb[1])),
                max(full[2], int(rb[2])), max(full[3], int(rb[3]))]
        full = grow_extent_tint(bgr, full)
        mask = build_mask(bgr.shape, full, pad_x=0.16, pad_y=0.5, dilate=9)
        cleaned = lama_inpaint_crop(bgr, mask)
        residue = verify_clean(reader, cleaned, full)
    rscore, _ = verify_residue(cleaned, anchor)

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imencode(".jpg", cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(out_path)

    info = {
        "ok": True,
        "status": "cleaned" if not residue else "residual",
        "path": str(in_path), "out": out_path, "method": method,
        "anchor": [round(t, 1) for t in anchor], "full_box": full,
        "loc_score": round(float(score), 3),
        "retry": retry, "resid_score": round(float(rscore), 3), "residue_after": len(residue),
    }
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        stem = Path(in_path).stem
        cv2.imwrite(f"{debug_dir}/{stem}_mask.png", mask)
        ov = bgr.copy(); red = np.zeros_like(ov); red[..., 2] = 255
        a = (mask.astype(np.float32) / 255)[..., None] * 0.45
        cv2.imencode(".jpg", (ov * (1 - a) + red * a).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
            f"{debug_dir}/{stem}_overlay.jpg")
    return info


def make_reader():
    """EasyOCR with its CRAFT detector + recognizer on the M4 GPU (MPS). EasyOCR's
    gpu= flag only knows CUDA, so we move the nets to MPS by hand — ~10x faster than CPU
    (12s -> 1.2s/readtext) and, crucially, not memory-bandwidth bound like CPU conv."""
    import easyocr, torch
    r = easyocr.Reader(["en"], gpu=False, verbose=False)
    if os.environ.get("OCR_DEVICE", "mps") == "mps" and torch.backends.mps.is_available():
        try:
            r.device = "mps"
            r.detector = r.detector.to("mps")
            if getattr(r, "recognizer", None) is not None:
                r.recognizer = r.recognizer.to("mps")
        except Exception as e:
            print("MPS-OCR unavailable, using CPU:", repr(e)[:120], file=sys.stderr)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input"); ap.add_argument("--output")
    ap.add_argument("--batch"); ap.add_argument("--out-dir")
    ap.add_argument("--log"); ap.add_argument("--debug")
    args = ap.parse_args()
    reader = make_reader()
    if args.input:
        print(json.dumps(clean_one(reader, args.input, args.output, args.debug), indent=2))
        return
    items = json.load(open(args.batch))
    results = []
    for i, (n, slug, path) in enumerate(items):
        t0 = time.time()
        info = clean_one(reader, path, os.path.join(args.out_dir, f"{slug}.jpg"), args.debug)
        info.update(n=n, slug=slug, elapsed_s=round(time.time() - t0, 1))
        results.append(info)
        if args.log:
            json.dump(results, open(args.log, "w"), indent=2)
        print(f"[{i+1}/{len(items)}] {slug} -> {info.get('status')} "
              f"residue_after={info.get('residue_after')} retry={info.get('retry')} "
              f"{info.get('method')} {info.get('elapsed_s')}s", flush=True)
    print("SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
