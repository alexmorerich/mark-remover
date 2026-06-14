#!/usr/bin/env python3
"""
logo_finder.py — Owner Logo Finder (the DETECT stage).

Implements docs/OWNER_LOGO_FINDER.md: an AGGRESSIVE, high-recall finder for SUNSKY-style
watermarks that emits a graded presence verdict + scored candidates + repair-routing masks.
It is the single owned detector — b2bweb's canonical matte and mark-remover's per-channel OCR
are folded in here, not run as separate agents.

Design contract (detect → repair → audit; this is detect):
  * Ensemble of complementary signals so no single blind spot hides a mark:
      - OCR text fragments  (v27 `_ocr_pass`, sensitive, multi-pass)         legibility
      - per-channel OCR     (colour-hidden marks the luminance pass misses)  colour
      - canonical matte NCC (b2bweb shape filter, background-subtracted)      shape
      - center prior        (cy/H≈0.475 measured on 700 real marks)          location
      - alpha-gray energy   (faint semi-transparent horizontal stroke band)  transparency
      - edge / dot-chain    (horizontal logo rhythm of "…online.com")        rhythm
  * GRADED scoring → presence class. The precision guard is text grading: a STRONG fragment
    (sunsky / online / .com) is real; a bare NOISE token (cn/rcn/c8n/sky alone) is downgraded
    to UNCERTAIN, never CONFIRMED — this is what stops the noise-token false positives.
  * False positives are acceptable: UNCERTAIN routes to conservative cover/review, and every
    downstream repair is backed up + audited (final safety is the Audit stage, not here).

CLI:
  python3 logo_finder.py --input a.jpg [--json out.json] [--masks maskdir] [--device mps]
  python3 logo_finder.py --batch paths.txt|_wm_scan.jsonl --out finds.jsonl [--device mps]
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v27_clean as v27
import run_bulk as rb

# ── geometry: the mark is a fixed centre stamp (cy/H p5–p95 = [0.469,0.487] on 700 marks) ──
CENTER_X, CENTER_Y = 0.50, 0.475
BAND_V0, BAND_V1 = 0.22, 0.78          # vertical search band (96.5% of marks fall inside)
_SENS = dict(contrast_ths=0.01, adjust_contrast=0.9, text_threshold=0.4,
             low_text=0.2, link_threshold=0.3)

# ── text grading (the precision guard) ─────────────────────────────────────────────────
# STRONG: unmistakable sunsky-online.com fragments.  MED: plausible online/.com pieces.
# NOISE: bare tokens v27's permissive regex also matches but which are usually product text.
_STRONG = re.compile(r"s[uo0]ns[kh]|nline|onlin|sky-?o|onl[il]ne|\.com|online\.?com", re.I)
_MED = re.compile(r"\bonl|nlin|line|sky[-\s]|\.c[o0]|[o0]nne|hline", re.I)
_NOISE = re.compile(r"^[\W_]*(c[0-9]?n|rcn|c8n|cemn|stock\w*|sky|sansky\s+\w+)[\W_]*$", re.I)

MATTE_PATH = os.environ.get("CANON_MATTE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "watermark-canonical.png"))
_MATTE = None

PRESENCE = ("CONFIRMED_WATERMARK", "LIKELY_WATERMARK", "UNCERTAIN_WATERMARK",
            "NO_WATERMARK", "UNSUITABLE_IMAGE")


# ───────────────────────────── 1. preprocessing views ─────────────────────────────
def _views(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    k = max(5, (min(gray.shape) // 20) | 1)
    resid = cv2.absdiff(gray, cv2.medianBlur(gray, k))      # faint-stroke / alpha residual
    edge = cv2.Canny(clahe, 60, 160)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return dict(gray=gray, clahe=clahe, resid=resid, edge=edge, hsv=hsv)


def _band_box(shape):
    H, W = shape[:2]
    return [0, int(BAND_V0 * H), W, int(BAND_V1 * H)]


# ───────────────────────────── 2. OCR signals (text) ──────────────────────────────
def _ocr_band(reader, bgr):
    """OCR composite + (saturation-gated) per-channel booster over the centre band.
    Returns boxes in FULL-image coords: (x1,y1,x2,y2,text,conf)."""
    if reader is None:
        return []
    H = bgr.shape[0]
    y0, y1 = int(BAND_V0 * H), int(BAND_V1 * H)
    crop = bgr[y0:y1]
    hits = v27._ocr_pass(reader, crop, 2.0, _SENS)
    if not hits:                                            # per-channel rescue (colour-hidden)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if float(np.percentile(hsv[..., 1], 92)) > 55:
            for ch in cv2.split(crop):
                c3 = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)
                hits += v27._ocr_pass(reader, c3, 2.0, _SENS)
    return [(x1, yy0 + y0, x2, yy1 + y0, t, c) for (x1, yy0, x2, yy1, t, c) in hits]


def _text_grade(text):
    """(strength in [0,1], canonical label or None). STRONG→1.0, MED→0.55, NOISE→0.1."""
    t = (text or "").strip()
    if _STRONG.search(t):
        return 1.0, "sunsky-online.com"
    if _NOISE.match(t):
        return 0.10, None
    if _MED.search(t):
        return 0.55, "online/.com"
    return 0.25, None


# ───────────────────────────── 3. canonical matte (shape) ─────────────────────────
def _load_matte():
    global _MATTE
    if _MATTE is None:
        m = cv2.imread(MATTE_PATH, cv2.IMREAD_GRAYSCALE) if os.path.exists(MATTE_PATH) else None
        _MATTE = m.astype(np.float32) if m is not None else False
    return _MATTE


def _canon_match(views):
    """(best NCC, [x1,y1,x2,y2]) of the matte over the centre band, or (0.0, None). CPU."""
    matte = _load_matte()
    if matte is False:
        return 0.0, None
    gray = views["gray"]
    h, w = gray.shape[:2]
    if h < 60 or w < 60:
        return 0.0, None
    dev = views["resid"].astype(np.float32)
    xc, xj = w // 2, int(0.15 * w)
    best = None
    for hf in (0.040, 0.046, 0.052, 0.058, 0.064):
        sw, sh = max(20, int(0.34 * w)), max(8, int(hf * h))
        if sw >= w or sh >= h:
            continue
        tpl = cv2.resize(matte, (sw, sh), interpolation=cv2.INTER_AREA)
        sx1, sx2 = max(0, xc - sw // 2 - xj), min(w, xc - sw // 2 + xj + 1 + sw)
        strip = dev[:, sx1:sx2]
        if strip.shape[1] <= sw:
            continue
        try:
            res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        r0 = max(0, int(0.38 * h) - sh // 2)
        r1 = min(res.shape[0], int(0.62 * h) - sh // 2 + 1)
        if r1 <= r0:
            continue
        sub = res[r0:r1, :]
        idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        peak = float(sub[idx])
        if best is None or peak > best[0]:
            best = (peak, [int(sx1 + idx[1]), int(r0 + idx[0]),
                           int(sx1 + idx[1] + sw), int(r0 + idx[0] + sh)])
    return best if best else (0.0, None)


# ───────────────────────── 4. geometric / texture component scores ────────────────
def _center_prior(box, shape):
    H, W = shape[:2]
    cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
    return math.exp(-(((cx - CENTER_X) / 0.22) ** 2 + ((cy - CENTER_Y) / 0.055) ** 2))


def _alpha_gray(views, box):
    """Faint, low-saturation horizontal stroke energy inside the box (the watermark is a
    light semi-transparent grey line). High when faint grey strokes sit on a calm surface."""
    x1, y1, x2, y2 = [int(v) for v in box]
    resid = views["resid"][y1:y2, x1:x2]
    sat = views["hsv"][y1:y2, x1:x2, 1]
    if resid.size == 0:
        return 0.0
    energy = float(np.mean(resid > 12))                      # fraction of faint-edge pixels
    rows = resid.mean(axis=1)
    concentration = float(rows.max() / (rows.mean() + 1e-6)) / 6.0   # thin horizontal band
    grayness = 1.0 - min(1.0, float(np.percentile(sat, 75)) / 80.0)  # low saturation = grey
    return max(0.0, min(1.0, 0.5 * min(1, energy * 4) + 0.3 * min(1, concentration) + 0.2 * grayness))


def _dot_chain(views, box):
    """Horizontal logo rhythm: '…online.com' has a regular glyph pitch. Autocorrelation of
    the column projection peaks at that pitch for text, stays flat for random texture."""
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = views["resid"][y1:y2, x1:x2].astype(np.float32)
    if crop.shape[1] < 40:
        return 0.0
    col = crop.mean(axis=0)
    col -= col.mean()
    if np.allclose(col, 0):
        return 0.0
    ac = np.correlate(col, col, "full")[len(col) - 1:]
    ac /= (ac[0] + 1e-6)
    lo, hi = 4, min(len(ac) - 1, crop.shape[1] // 3)
    return float(np.clip(ac[lo:hi].max(), 0, 1)) if hi > lo else 0.0


def _edge_score(views, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    e = views["edge"][y1:y2, x1:x2]
    return float(np.clip(np.mean(e > 0) * 6, 0, 1)) if e.size else 0.0


# ───────────────────────────── 5. ROI classification ──────────────────────────────
def _roi_class(bgr, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "low_texture_background"
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright, sat = float(g.mean()), float(hsv[..., 1].mean())
    var = float(g.var())
    edged = float(np.mean(cv2.Canny(g, 60, 160) > 0))
    ar = (x2 - x1) / max(1, (y2 - y1))
    if sat > 70 and ar > 6 and (y2 - y1) < 0.25 * bgr.shape[0]:
        return "thin_flex_cable"
    if bright > 235 and sat < 18 and var < 120:
        return "plain_white"
    if bright > 205 and sat < 28:
        return "near_white"
    if edged > 0.16:
        return "complex_product_detail"
    if bright < 60:
        return "dark_product_surface"
    if sat > 60:
        return "mixed_background_product"
    if 90 < bright < 200 and var < 400:
        return "metallic_or_reflective"
    if var < 150:
        return "low_texture_background"
    return "simple_product_surface"


_REPAIR_FRIENDLY = {"plain_white", "near_white", "low_texture_background",
                    "simple_product_surface", "metallic_or_reflective"}


# ───────────────────────────── 6. masks ───────────────────────────────────────────
def _masks(shape, box, roi):
    H, W = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1

    def rect(pad_x, pad_y):
        m = np.zeros((H, W), np.uint8)
        m[max(0, y1 - pad_y):min(H, y2 + pad_y), max(0, x1 - pad_x):min(W, x2 + pad_x)] = 255
        return m

    stroke = rect(max(2, bw // 40), max(2, bh // 8))
    soft = cv2.dilate(stroke, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    line = rect(int(0.06 * bw) + 6, max(4, bh // 4))                 # full text line
    # cover fallback = generous canonical band (used when stroke repair is unsafe)
    cover = np.zeros((H, W), np.uint8)
    cover[int(0.43 * H):int(0.52 * H), int(0.23 * W):int(0.77 * W)] = 255
    return dict(stroke_mask=stroke, soft_mask=soft, line_mask=line, cover_mask=cover)


# ───────────────────────────── 7. main find() ─────────────────────────────────────
def find(bgr, reader=None, category=None):
    """Return the OWNER_LOGO_FINDER output contract dict for one image."""
    if bgr is None or min(bgr.shape[:2]) < 60:
        return {"presence": "UNSUITABLE_IMAGE", "candidates": []}
    views = _views(bgr)

    # gather raw candidate boxes from text + shape
    ocr = _ocr_band(reader, bgr)
    cscore, cbox = _canon_match(views)

    cands = []
    # (a) text-led candidates
    for (x1, y1, x2, y2, txt, conf) in ocr:
        box = v27.pad_bbox([x1, y1, x2, y2], bgr.shape, 0.12, 0.6, 6)
        strength, label = _text_grade(txt)
        ev = {
            "template_score": round(cscore, 3),
            "edge_score": round(_edge_score(views, box), 3),
            "dot_chain_score": round(_dot_chain(views, box), 3),
            "text_fragment_score": round(strength * float(conf) ** 0.3, 3),
            "center_prior_score": round(_center_prior(box, bgr.shape), 3),
            "alpha_gray_score": round(_alpha_gray(views, box), 3),
        }
        cands.append(dict(box=box, ev=ev, txt=txt, label=label or "sunsky-fragment",
                          strength=strength, conf=float(conf), src="ocr"))

    # (b) shape-led candidate (matte) when OCR read nothing legible there
    if cbox is not None and cscore >= 0.40 and not ocr:
        box = v27.pad_bbox(cbox, bgr.shape, 0.05, 0.4, 6)
        ev = {
            "template_score": round(cscore, 3),
            "edge_score": round(_edge_score(views, box), 3),
            "dot_chain_score": round(_dot_chain(views, box), 3),
            "text_fragment_score": 0.0,
            "center_prior_score": round(_center_prior(box, bgr.shape), 3),
            "alpha_gray_score": round(_alpha_gray(views, box), 3),
        }
        cands.append(dict(box=box, ev=ev, txt="", label="canon-matte",
                          strength=0.0, conf=cscore, src="canon"))

    # score + classify each candidate
    out_cands, best_presence = [], "NO_WATERMARK"
    for c in cands:
        ev = c["ev"]
        score = (0.40 * ev["text_fragment_score"] + 0.22 * ev["template_score"]
                 + 0.15 * ev["center_prior_score"] + 0.10 * ev["alpha_gray_score"]
                 + 0.07 * ev["dot_chain_score"] + 0.06 * ev["edge_score"])
        # penalties: product text (high-conf NON-sunsky) and busy random texture w/o rhythm
        if c["src"] == "ocr" and c["strength"] <= 0.10:
            score -= 0.18                                   # bare noise token
        if ev["edge_score"] > 0.6 and ev["dot_chain_score"] < 0.2 and ev["text_fragment_score"] < 0.3:
            score -= 0.10                                   # random product texture
        score = max(0.0, min(1.0, score))

        # presence (text grading is the precision guard)
        legible = c["strength"] >= 1.0 and c["conf"] >= 0.30
        corroborated = ev["template_score"] >= 0.45 and ev["center_prior_score"] >= 0.4
        if legible or (score >= 0.62):
            presence = "CONFIRMED_WATERMARK"
        elif score >= 0.42 or (c["strength"] >= 0.55 and ev["center_prior_score"] >= 0.4) or corroborated:
            presence = "LIKELY_WATERMARK"
        elif score >= 0.20 or ev["center_prior_score"] >= 0.5:
            presence = "UNCERTAIN_WATERMARK"
        else:
            presence = "NO_WATERMARK"

        roi = _roi_class(bgr, c["box"])
        risk = ("high" if roi in ("thin_flex_cable", "complex_product_detail", "text_or_label_area",
                                  "glass_or_gradient", "transparent_or_glossy")
                else "low" if roi in ("plain_white", "near_white", "low_texture_background") else "medium")
        if presence in ("CONFIRMED_WATERMARK", "LIKELY_WATERMARK"):
            action = "repair" if roi in _REPAIR_FRIENDLY else "cover"
        elif presence == "UNCERTAIN_WATERMARK":
            action = "cover"
        else:
            action = "reject"

        x1, y1, x2, y2 = c["box"]
        out_cands.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(score, 3),
            "scale": 1.0, "matched_template": c["label"], "evidence": ev,
            "roi_class": roi, "risk_level": risk, "recommended_action": action,
            "text": c["txt"], "_box": c["box"],
        })
        if PRESENCE.index(presence) < PRESENCE.index(best_presence):
            best_presence = presence

    out_cands.sort(key=lambda d: d["score"], reverse=True)
    return {"presence": best_presence, "candidates": out_cands}


# ───────────────────────────── 8. CLI ─────────────────────────────────────────────
def _save_masks(bgr, cand, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    masks = _masks(bgr.shape, cand["_box"], cand["roi_class"])
    paths = {}
    for name, m in masks.items():
        p = os.path.join(out_dir, f"{stem}.{name}.png")
        cv2.imwrite(p, m)
        paths[name] = p
    return paths


def _iter_paths(src):
    if src.endswith(".jsonl"):
        for l in open(src):
            if l.strip():
                yield json.loads(l)["path"]
    else:
        for l in open(src):
            if l.strip():
                yield l.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--batch")
    ap.add_argument("--out", default="logo_finds.jsonl")
    ap.add_argument("--json")
    ap.add_argument("--masks")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rb._init_worker(args.device)
    reader = rb._get_reader()

    if args.input:
        bgr = rb._imread(args.input)
        res = find(bgr, reader)
        if args.masks and res["candidates"]:
            res["candidates"][0]["masks"] = _save_masks(bgr, res["candidates"][0], args.masks,
                                                         os.path.splitext(os.path.basename(args.input))[0])
        for c in res["candidates"]:
            c.pop("_box", None)
        out = json.dumps(res, indent=2)
        print(out)
        if args.json:
            open(args.json, "w").write(out)
        return

    if not args.batch:
        ap.error("need --input or --batch")
    paths = list(_iter_paths(args.batch))
    if args.limit:
        paths = paths[:args.limit]
    f = open(args.out, "a", buffering=1)
    tally = {p: 0 for p in PRESENCE}
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        bgr = rb._imread(p)
        res = find(bgr, reader)
        tally[res["presence"]] = tally.get(res["presence"], 0) + 1
        for c in res["candidates"]:
            c.pop("_box", None)
        f.write(json.dumps({"path": p, **res}) + "\n")
        if i % 25 == 0 or i == len(paths):
            r = i / (time.time() - t0)
            print(f"[{i}/{len(paths)}] {r:.2f}/s  " +
                  " ".join(f"{k.split('_')[0]}={v}" for k, v in tally.items() if v), flush=True)
    print("DONE", tally)


if __name__ == "__main__":
    main()
