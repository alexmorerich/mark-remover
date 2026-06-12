"""
v27 — sunsky watermark eraser.

Failure mode of v26 (visible on every cleaned image in cleaned_50_v26_before_after.pdf):
  Mask covered only the dark text strokes, leaving the anti-aliased glyph halo as a
  readable ghost in the cleaned output.

Fix:
  1. Locate watermark text with EasyOCR (CLAHE + 2x upscale recall boost for faint marks).
  2. Fuzzy-match recognized tokens against {sunsky, online, com, sunsky-online.com}.
  3. Union matched boxes → single bbox. Pad generously (15% of text-h on every side).
  4. Build a FILLED rectangle mask (not stroke pixels). Dilate again by ~6 px.
  5. LaMa inpaint at native resolution. LaMa is excellent on flat / smooth-gradient
     backgrounds where the watermark sits in 95% of catalog images.
  6. Re-run OCR on cleaned. If any of the tokens still detected → expand bbox 1.5x
     and retry once. Record residual_clear=True/False.

CLI:
  python3 v27_clean.py --input <jpg> --output <jpg> [--debug <dir>]
  python3 v27_clean.py --batch <jsonl> --out-dir <dir>
"""
import argparse, json, os, sys, re, time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

SUNSKY_TOKENS = re.compile(
    # Match anything that smells like sunsky-online.com after CRAFT/OCR garbling.
    # Patterns observed: "sunsky-Snline:=", "sunsky-online cemn|", "sunshy-onite:com",
    # "hline com", "onlme com", "Snline c8n", etc.
    r"(s[a-z0-9]{2,5}k[a-z0-9]?|snl[a-z]ne|onl[a-z]{1,2}e|onh?n[a-z]|onhe|nline|hline|"
    r"\.c[a-z0-9]{1,2}m|c[0-9]{0,2}n[a-z]?$|c8n|cemn|ine\s+com|nl[a-z]ne|sky)",
    re.IGNORECASE,
)


def clahe_upscale(bgr, scale=2.0):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab2 = cv2.merge((L, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return out


def _ocr_pass(reader, bgr, scale, params):
    prep = clahe_upscale(bgr, scale=scale)
    rgb = cv2.cvtColor(prep, cv2.COLOR_BGR2RGB)
    raw = reader.readtext(rgb, detail=1, paragraph=False, **params)
    hits = []
    for box, text, conf in raw:
        if conf < 0.03:
            continue
        if not SUNSKY_TOKENS.search(text):
            continue
        pts = np.array(box) / scale
        x1, y1 = pts.min(axis=0); x2, y2 = pts.max(axis=0)
        hits.append((float(x1), float(y1), float(x2), float(y2), text, float(conf)))
    return hits


def detect_watermark_boxes(reader, bgr):
    """Multi-pass OCR for robustness. Pass 1 uses default thresholds (best on most images),
    pass 2 cranks up CLAHE + 3x upscale + looser CRAFT, pass 3 is a relaxed broad search."""
    passes = [
        (2.0, {}),  # easyocr defaults at 2x
        (3.0, dict(contrast_ths=0.01, adjust_contrast=0.9,
                   text_threshold=0.4, low_text=0.2, link_threshold=0.3)),
        (2.0, dict(contrast_ths=0.01, adjust_contrast=0.9,
                   text_threshold=0.3, low_text=0.15, link_threshold=0.2)),
    ]
    for scale, params in passes:
        hits = _ocr_pass(reader, bgr, scale, params)
        if hits:
            return hits
    return []


def union_bbox(boxes):
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def pad_bbox(bbox, img_shape, pad_frac_x=0.20, pad_frac_y=0.35, pad_px_min=8):
    """Pad bbox: 20% of width on each side horizontally, 35% of height vertically,
    minimum 8 px. Watermark glyph anti-alias halo extends ~1 glyph height around text."""
    H, W = img_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = max(pad_px_min, int(bw * pad_frac_x))
    py = max(pad_px_min, int(bh * pad_frac_y))
    return [
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(W, int(x2 + px)),
        min(H, int(y2 + py)),
    ]


def build_mask(img_shape, bbox, dilate_px=6):
    H, W = img_shape[:2]
    m = np.zeros((H, W), dtype=np.uint8)
    if bbox is None:
        return m
    x1, y1, x2, y2 = bbox
    m[y1:y2, x1:x2] = 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k, iterations=1)
    return m


_LAMA = None


def get_lama():
    global _LAMA
    if _LAMA is None:
        import torch
        from simple_lama_inpainting import SimpleLama
        # MPS sometimes breaks on conv ops with arbitrary sizes; CPU is safer + still tractable
        dev = torch.device("cpu")
        _LAMA = SimpleLama(device=dev)
    return _LAMA


def lama_inpaint(bgr, mask):
    lama = get_lama()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    mask_pil = Image.fromarray(mask)
    out_pil = lama(img_pil, mask_pil)
    out_rgb = np.array(out_pil)
    if out_rgb.shape[:2] != bgr.shape[:2]:
        out_rgb = cv2.resize(out_rgb, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def clean_one(reader, in_path, out_path, debug_dir=None):
    bgr = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"ok": False, "err": "imread failed", "path": str(in_path)}

    boxes = detect_watermark_boxes(reader, bgr)
    if not boxes:
        return {"ok": True, "status": "no_mark", "path": str(in_path)}

    bbox = union_bbox(boxes)
    bbox_p = pad_bbox(bbox, bgr.shape)
    mask = build_mask(bgr.shape, bbox_p, dilate_px=6)

    cleaned = lama_inpaint(bgr, mask)

    # residual check
    residual_boxes = detect_watermark_boxes(reader, cleaned)
    retry = False
    if residual_boxes:
        retry = True
        bbox_r = union_bbox([(b[0], b[1], b[2], b[3]) for b in residual_boxes] + [tuple(bbox)])
        bbox_r = pad_bbox(bbox_r, bgr.shape, pad_frac_x=0.30, pad_frac_y=0.50, pad_px_min=12)
        mask2 = build_mask(bgr.shape, bbox_r, dilate_px=10)
        cleaned = lama_inpaint(bgr, mask2)
        residual_boxes = detect_watermark_boxes(reader, cleaned)
        mask = mask2

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imencode(".jpg", cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(out_path)

    info = {
        "ok": True,
        "status": "cleaned" if not residual_boxes else "residual",
        "path": str(in_path),
        "out": out_path,
        "matches": [{"text": b[4], "conf": b[5], "box": b[:4]} for b in boxes],
        "bbox_padded": bbox_p,
        "retry": retry,
        "residual_after": len(residual_boxes),
    }
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        stem = Path(in_path).stem
        cv2.imwrite(f"{debug_dir}/{stem}_mask.png", mask)
        # overlay
        overlay = bgr.copy()
        red = np.zeros_like(overlay); red[..., 2] = 255
        a = (mask.astype(np.float32) / 255.0)[..., None] * 0.45
        overlay = (overlay * (1 - a) + red * a).astype(np.uint8)
        cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
            f"{debug_dir}/{stem}_overlay.jpg")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="single image path")
    ap.add_argument("--output", help="single output path")
    ap.add_argument("--batch", help="JSONL of {n,slug,path} → cleans all")
    ap.add_argument("--out-dir", help="output dir for batch")
    ap.add_argument("--debug", help="debug dir (mask + overlay)")
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    if args.input:
        info = clean_one(reader, args.input, args.output, debug_dir=args.debug)
        print(json.dumps(info, indent=2))
        return

    if args.batch:
        items = json.load(open(args.batch))
        results = []
        for idx, item in enumerate(items):
            n, slug, path = item
            out_path = os.path.join(args.out_dir, f"{slug}.jpg")
            t0 = time.time()
            info = clean_one(reader, path, out_path, debug_dir=args.debug)
            info["n"] = n; info["slug"] = slug; info["elapsed_s"] = round(time.time()-t0, 1)
            print(f"[{idx+1}/{len(items)}] {n}/47 {slug} -> {info.get('status')} {info.get('elapsed_s')}s residual_after={info.get('residual_after')}")
            results.append(info)
            with open(os.path.join(args.out_dir, "_v27_log.json"), "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
