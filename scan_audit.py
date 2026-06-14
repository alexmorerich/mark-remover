"""
scan_audit.py — Final-Audit pass over the Owner scanner's "non-watermarked" verdict.

The Owner's scanner (b2bweb detector.py, multi-scale TEMPLATE matching) classified each
image as watermarked (`flagged`) or not. This tool independently re-checks the images it
called clean — and the images it never scanned — with a COMPLEMENTARY detector
(v28 `detect_smart`: multi-pass EasyOCR/CRAFT + per-channel recall booster + fuzzy
sunsky regex). Re-using the owner's own method would reproduce its blind spots; an OCR
detector catches the faint / garbled / coloured-background marks template matching misses.

A hit on an image the owner called clean ⇒ a SUSPECTED scan false-negative: a watermark
that would be published. Per the audit charter we are aggressive on the find here.

It is a long job, so all state is on disk and the run is resumable (re-launch with the
same --out-dir and it skips everything already in done.txt):

  out-dir/
    progress.json   total / done / hits / speed / eta            (machine-readable status)
    heartbeat.txt   ISO timestamp, rewritten every few images     (liveness)
    run.log         human progress lines
    misses.jsonl    one record per suspected missed watermark
    misses/<stem>.jpg   crop with the detected box drawn, for eyeballing
    done.txt        every processed path (resume ledger)
    summary.json    final tally

Usage:
  python3 scan_audit.py --set A --scan <_wm_scan.jsonl> --out-dir auditA [--limit N] [--shard i/n]
  python3 scan_audit.py --set B --scan <_wm_scan.jsonl> --assets <assets_dir> --out-dir auditB
  python3 scan_audit.py --paths-file list.txt --out-dir auditX
"""
import argparse, glob, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_clean as v28e


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_set(which, scan_path, assets_dir):
    """Set A = scanned & status != flagged. Set B = images in assets_dir the scan never
    covered (the skipped population). Returns a sorted list of absolute paths."""
    scan = [json.loads(l) for l in open(scan_path)]
    scanned = {r["path"] for r in scan}
    if which == "A":
        return sorted(r["path"] for r in scan if r.get("status") != "flagged")
    if which == "B":
        if not assets_dir:
            assets_dir = os.path.dirname(scan_path)
        allimg = set()
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            allimg |= set(glob.glob(os.path.join(assets_dir, ext)))
        return sorted(allimg - scanned)
    raise ValueError(which)


# The sunsky mark is a fixed CENTRE stamp: across 700 real detections cy/H sat in
# [0.469, 0.487] (p5–p95) and 96.5% of marks fall in a generous centre band. Auditing
# that band (not the whole frame) is faster AND more precise — product SKU/label text
# at the top/bottom edges can't false-positive.
CANON_V0, CANON_V1 = 0.22, 0.78
_SENS = dict(contrast_ths=0.01, adjust_contrast=0.9, text_threshold=0.4,
             low_text=0.2, link_threshold=0.3)


def _band(bgr):
    H, W = bgr.shape[:2]
    y0, y1 = int(CANON_V0 * H), int(CANON_V1 * H)
    return bgr[y0:y1, :], y0


def _saturated(crop):
    """Does the band carry a saturated-colour region? That is the ONLY place a
    light/white-on-colour mark can hide from a composite OCR pass, so it gates the
    (expensive) per-channel booster — neutral white/grey trays skip it entirely."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.percentile(hsv[..., 1], 92)) > 55


# ── b2bweb canonical-matte detector (MERGED from remove-watermarks.py) ──────────
# A SHAPE / appearance signal, complementary to OCR: a background-subtracted matched
# filter against the median watermark "matte" (b2bweb's watermark-canonical.png). It
# rescues faint / illegible marks where the glyph SHAPE still correlates but OCR reads
# nothing — a DIFFERENT blind spot from per-channel OCR (which rescues colour-hidden
# marks). b2bweb tuned the standalone threshold to 95.8% recall on their validation
# split. It can false-fire on centred grid patterns (screw sets, sticker sheets), so
# canon-only hits are tagged 'review', not auto-confirmed.
_REPO_MATTE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "watermark-canonical.png")
_CANON_MATTE_PATH = os.environ.get(
    "CANON_MATTE", _REPO_MATTE if os.path.exists(_REPO_MATTE)
    else "/Users/alexkou/Documents/github/b2bweb/scripts/watermark-canonical.png")
_CW_FRAC = 0.34
_CH_FRACS = (0.040, 0.046, 0.052, 0.058, 0.064)
_CX_BAND, _CY_LO, _CY_HI, _CBG = 0.15, 0.38, 0.62, 0.05
# b2bweb standalone min was 0.35; 0.40 trims the grid-pattern FPs seen in the bench.
CANON_STANDALONE = float(os.environ.get("CANON_STANDALONE", "0.40"))
_CANON_MATTE = None


def _load_matte():
    global _CANON_MATTE
    if _CANON_MATTE is None:
        m = (cv2.imread(_CANON_MATTE_PATH, cv2.IMREAD_GRAYSCALE)
             if os.path.exists(_CANON_MATTE_PATH) else None)
        _CANON_MATTE = m.astype(np.float32) if m is not None else False   # False = unavailable
    return _CANON_MATTE


def _canon_absdiff(gray):
    h, w = gray.shape[:2]
    k = int(round(_CBG * min(h, w)))
    k = max(5, k);  k += (k % 2 == 0)
    if k > min(h, w) - 1:
        k = max(5, (min(h, w) // 2) * 2 + 1)
    return cv2.absdiff(gray, cv2.medianBlur(gray, k)).astype(np.float32)


def canon_detect(bgr):
    """(best NCC score, [x1,y1,x2,y2]) of the canonical matte over the centre band,
    or (0.0, None). CPU-only — no GPU, so it can run as a cheap rescue pass."""
    matte = _load_matte()
    if matte is False:
        return 0.0, None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]
    if h_img < 60 or w_img < 60:
        return 0.0, None
    dev = _canon_absdiff(gray)
    xc, xj = w_img // 2, int(round(_CX_BAND * w_img))
    best = None
    for hf in _CH_FRACS:
        sw, sh = max(20, int(round(_CW_FRAC * w_img))), max(8, int(round(hf * h_img)))
        if sw >= w_img or sh >= h_img:
            continue
        tpl = cv2.resize(matte, (sw, sh), interpolation=cv2.INTER_AREA)
        sx1, sx2 = max(0, xc - sw // 2 - xj), min(w_img, xc - sw // 2 + xj + 1 + sw)
        strip = dev[:, sx1:sx2]
        if strip.shape[1] <= sw:
            continue
        try:
            res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        r0 = max(0, int(_CY_LO * h_img) - sh // 2)
        r1 = min(res.shape[0], int(_CY_HI * h_img) - sh // 2 + 1)
        if r1 <= r0:
            continue
        sub = res[r0:r1, :]
        idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        peak = float(sub[idx])
        if best is None or peak > best[0]:
            best = (peak, [int(sx1 + idx[1]), int(r0 + idx[0]),
                           int(sx1 + idx[1] + sw), int(r0 + idx[0] + sh)])
    return best if best is not None else (0.0, None)


def detect(reader, bgr, fast=False, use_canon=True, canon_standalone=False):
    """ENSEMBLE of BOTH projects' watermark finders, on the canonical centre band:
       1. OCR composite (sensitive)                          -> method 'ocr'      [primary]
       2. per-channel OCR booster (saturation-gated)         -> method 'perchan'  [mark-remover]
       3. canonical-matte NCC (shape, not legibility)        -> method 'canon'    [b2bweb]

    OCR + per-channel are the recall workhorses (per-channel rescues the colour-hidden
    marks — proven on production). The b2bweb canonical matte is folded in as a
    CORROBORATION signal by default: its NCC is recorded on every hit (agreement raises
    triage confidence) but it does NOT fire standalone — on this catalog a standalone
    canon rescue is ~12% false-positive (flex traces / product text / grid patterns) with
    ~no real rescue, so standalone is opt-in (`canon_standalone`, used by the cheap CPU
    `--canon-only` pass when you want max recall and will review the noise).

    Returns (hits_full_image_coords, methods_that_fired, canon_score)."""
    crop, y0 = _band(bgr)
    methods = []
    hits = v28e.v27._ocr_pass(reader, crop, 2.0, _SENS) if reader is not None else []
    if hits:
        methods.append("ocr")
    if not hits and not fast and reader is not None and _saturated(crop):
        hits = v28e._ocr_channels(reader, crop, {}, 2.0)
        if hits:
            methods.append("perchan")
    hits_full = [(x1, y1 + y0, x2, y2 + y0, t, c) for (x1, y1, x2, y2, t, c) in hits]

    cscore = 0.0
    if use_canon:
        cscore, cbox = canon_detect(bgr)
        if cscore >= CANON_STANDALONE:
            if hits_full:
                methods.append("canon")                      # corroborates a real OCR hit
            elif canon_standalone and cbox is not None:      # opt-in standalone rescue
                methods.append("canon")
                hits_full = [(cbox[0], cbox[1], cbox[2], cbox[3], f"[canon {cscore:.2f}]", cscore)]
    return hits_full, methods, round(float(cscore), 3)


def save_miss_crop(bgr, box, out_path):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    px = int(0.25 * (x2 - x1)) + 20
    py = int(1.2 * (y2 - y1)) + 20
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
    crop = bgr[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
    cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(out_path)


def run(paths, out_dir, label, fast=False, use_canon=True, canon_only=False,
        canon_standalone=False, heartbeat_every=20):
    # the cheap CPU rescue pass IS a standalone-canon run by definition
    canon_standalone = canon_standalone or canon_only
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "misses"), exist_ok=True)
    done_path = os.path.join(out_dir, "done.txt")
    done = set()
    if os.path.exists(done_path):
        done = set(l.strip() for l in open(done_path) if l.strip())
    prior_hits = confirmed = review = 0
    miss_path = os.path.join(out_dir, "misses.jsonl")
    if os.path.exists(miss_path):
        for l in open(miss_path):
            prior_hits += 1
            try:
                tier = json.loads(l).get("tier", "confirmed")
            except Exception:
                tier = "confirmed"
            review += (tier == "review")
            confirmed += (tier != "review")

    todo = [p for p in paths if p not in done]
    total = len(paths)
    log = open(os.path.join(out_dir, "run.log"), "a")
    done_f = open(done_path, "a")
    miss_f = open(miss_path, "a")

    def logline(s):
        line = f"{_now()} {s}"
        log.write(line + "\n"); log.flush()
        print(line, flush=True)

    mode = "canon_only" if canon_only else ("ensemble" if use_canon else "ocr_only")
    logline(f"START set={label} total={total} already_done={len(done)} todo={len(todo)} "
            f"mode={mode} canon_thr={CANON_STANDALONE} device={os.environ.get('OCR_DEVICE','mps')}")
    reader = None if canon_only else v28e.make_reader()

    t0 = time.time()
    hits = prior_hits
    n = 0
    for p in todo:
        n += 1
        try:
            bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                logline(f"WARN unreadable {p}")
                done_f.write(p + "\n"); done_f.flush()
                continue
            h, methods, cscore = detect(reader, bgr, fast=fast, use_canon=use_canon,
                                        canon_standalone=canon_standalone)
            if h:
                hits += 1
                # tier: a legible-text hit (ocr/perchan) is a confirmed miss; a
                # canon-only (shape) hit is FP-prone on grid patterns -> 'review'.
                tier = "confirmed" if any(m in methods for m in ("ocr", "perchan")) else "review"
                if tier == "confirmed":
                    confirmed += 1
                else:
                    review += 1
                anchor = v28e.v27.union_bbox(h)
                texts = " | ".join((x[4] or "") for x in h)
                conf = max((x[5] for x in h), default=0.0)
                stem = Path(p).stem
                crop_rel = f"misses/{stem}.jpg"
                try:
                    save_miss_crop(bgr, anchor, os.path.join(out_dir, crop_rel))
                except Exception as e:
                    crop_rel = f"(crop_err:{type(e).__name__})"
                rec = {"path": p, "texts": texts, "anchor": [round(float(v), 1) for v in anchor],
                       "conf": round(float(conf), 3), "n_boxes": len(h), "crop": crop_rel,
                       "methods": methods, "canon_score": cscore, "tier": tier}
                miss_f.write(json.dumps(rec) + "\n"); miss_f.flush()
                logline(f"MISS #{hits} [{tier}] {'+'.join(methods)} conf={conf:.2f} "
                        f"canon={cscore:.2f} '{texts[:36]}'  {Path(p).name}")
            done_f.write(p + "\n"); done_f.flush()
        except Exception as e:
            logline(f"ERROR {type(e).__name__}: {str(e)[:100]}  {p}")
            done_f.write(p + "\n"); done_f.flush()

        if n % heartbeat_every == 0 or n == len(todo):
            el = time.time() - t0
            spd = n / el if el else 0
            remaining = len(todo) - n
            eta_h = (remaining / spd / 3600) if spd else 0
            done_total = len(done) + n
            prog = {"set": label, "total": total, "done": done_total,
                    "processed_this_run": n, "hits": hits,
                    "confirmed": confirmed, "review": review,
                    "hit_rate": round(hits / max(1, done_total), 4),
                    "speed_img_s": round(spd, 2), "eta_h_remaining": round(eta_h, 2),
                    "elapsed_s": round(el), "updated": _now(),
                    "pct": round(100 * done_total / total, 1)}
            json.dump(prog, open(os.path.join(out_dir, "progress.json"), "w"), indent=2)
            open(os.path.join(out_dir, "heartbeat.txt"), "w").write(_now() + "\n")
            if n % (heartbeat_every * 10) == 0:
                logline(f"... {done_total}/{total} ({prog['pct']}%) hits={hits} "
                        f"{spd:.2f} img/s eta {eta_h:.1f}h")

    summary = {"set": label, "total": total, "processed_this_run": n,
               "done_total": len(done) + n, "suspected_misses": hits,
               "confirmed": confirmed, "review": review,
               "miss_rate": round(hits / max(1, len(done) + n), 4),
               "mode": mode, "finished": _now(), "elapsed_s": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    logline(f"DONE set={label} processed={n} suspected_misses={hits} "
            f"(confirmed={confirmed} review={review})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["A", "B"], help="A=scanned&clean  B=never-scanned")
    ap.add_argument("--scan", help="path to _wm_scan.jsonl")
    ap.add_argument("--assets", help="assets dir (for set B); defaults to scan's dir")
    ap.add_argument("--paths-file", help="explicit newline-separated image paths")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap N (after shard) for a smaller run")
    ap.add_argument("--shard", help="i/n — process only paths where index%%n==i")
    ap.add_argument("--fast", action="store_true", help="single OCR pass (faster, lower recall)")
    ap.add_argument("--no-canon", action="store_true", help="disable the b2bweb canonical-matte signal")
    ap.add_argument("--canon-only", action="store_true",
                    help="CPU-only canonical-matte rescue pass (no OCR/GPU) — cheap second sweep")
    ap.add_argument("--canon-standalone", action="store_true",
                    help="let canonical fire standalone in the OCR ensemble too (noisier; review-tiered)")
    args = ap.parse_args()

    if args.paths_file:
        paths = [l.strip() for l in open(args.paths_file) if l.strip()]
        label = Path(args.paths_file).stem
    else:
        if not args.set or not args.scan:
            ap.error("need --set and --scan (or --paths-file)")
        paths = build_set(args.set, args.scan, args.assets)
        label = args.set

    if args.shard:
        i, nsh = (int(x) for x in args.shard.split("/"))
        paths = [p for k, p in enumerate(paths) if k % nsh == i]
        label = f"{label}_shard{i}of{nsh}"
    if args.limit:
        paths = paths[:args.limit]

    run(paths, args.out_dir, label, fast=args.fast, use_canon=not args.no_canon,
        canon_only=args.canon_only, canon_standalone=args.canon_standalone)


if __name__ == "__main__":
    main()
