#!/usr/bin/env python3
"""
run_bulk.py — scale the v27 watermark eraser over large image libraries.

Built on v27_clean.py's detector + LaMa, this adds the machinery a 20k-image run
needs that the single-image tool lacks: a brand-safe two-phase workflow, MPS GPU
inpainting, crop-inpaint (LaMa runs on a small padded crop, not the whole image),
parallel workers, in-place editing with mandatory backup, and resumable logs.

Workflow (brand-safe: nothing is altered until you've reviewed the suspects):

  Phase 1   scan      detect-only over EVERY image -> review manifest + overlays
            (you inspect _wm_review/ and, optionally, list false positives in an
             --exclude file before processing)
  Phase 2   process   inpaint ONLY flagged images, IN PLACE, originals backed up
            validate  re-detect processed images, confirm zero residual
            restore   copy originals back from backup (undo a bad batch)

The detector is the gate: images with no watermark come back "clean" and are
never written or backed up. You point this at all 20,000 — the clean ~half is
skipped automatically.

Examples:
  python3 run_bulk.py scan    --images-dir /path/to/library --workers 4
  # ...review /path/to/library/_wm_review/ , optionally build exclude.txt...
  python3 run_bulk.py process --images-dir /path/to/library --device mps --workers 2
  python3 run_bulk.py validate --images-dir /path/to/library
  python3 run_bulk.py restore  --images-dir /path/to/library     # if needed
"""
import argparse, json, os, re, sys, time, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import cv2
from PIL import Image

# Reuse the proven v27 detector + mask geometry. v27_clean has only light
# top-level imports, so importing it here is cheap and side-effect free.
from v27_clean import detect_watermark_boxes, union_bbox, pad_bbox, build_mask, _ocr_pass

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SCAN_REPORT = "_wm_scan.jsonl"
REVIEW_DIR = "_wm_review"
BACKUP_DIR = "_wm_backup"
PROCESS_LOG = "_wm_process.jsonl"
# Directory names this tool creates — never treat them as input images.
SELF_DIRS = {REVIEW_DIR, BACKUP_DIR}

# Crop-inpaint context: how much background to give LaMa around the masked box.
CTX_MARGIN_PX = 96

# --------------------------------------------------------------- model filter
# Supplier (sunsky) watermarks only ride on iPhone-13-series-and-older and the
# other older products (MacBook, Apple Watch, generic parts). iPhone 14+ photos
# were never watermarked / are already clean, and make up more than half the
# library — so we drop them BEFORE any OCR. That removes the single most
# expensive detector path: a clean image runs all 3 OCR passes (none hit), so
# skipping the already-clean half is the biggest speed win. Ported from b2bweb's
# proven gate (SKIP_IPHONE_MIN). A file is skipped ONLY when it names an iPhone
# and EVERY iPhone it names is >= 14 ("iphone-14-pro"); anything that also names
# an older iPhone ("...13-and-14..."), or names no iPhone at all (MacBook, Watch,
# generic parts), is KEPT — those still carry the watermark.
SKIP_IPHONE_MIN = 14
_IPHONE_MODEL_RE = re.compile(r"iphone[\s_\-]*([0-9]{1,2})", re.IGNORECASE)


def is_skippable_model(name):
    nums = [int(n) for n in _IPHONE_MODEL_RE.findall(name)]
    if not nums:
        return False
    return all(n >= SKIP_IPHONE_MIN for n in nums)


# ----------------------------------------------------------------------------- IO
def _imread(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite_atomic(path, bgr):
    """Write to a temp file then os.replace — never leave a half-written image
    if interrupted mid-write. Format is preserved by the original extension."""
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise RuntimeError(f"imencode failed for {path}")
    tmp = path + ".wmtmp"
    buf.tofile(tmp)
    os.replace(tmp, path)


def discover_images(images_dir):
    images_dir = os.path.abspath(images_dir)
    out = []
    for root, dirs, files in os.walk(images_dir):
        # don't descend into our own scratch dirs
        dirs[:] = [d for d in dirs if d not in SELF_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, f))
    out.sort()
    return out


def _load_done(jsonl_path):
    """Return the set of 'path' values already recorded — the resume key."""
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["path"])
                except Exception:
                    pass
    return done


# ----------------------------------------------------------------- engine (worker)
_READER = None
_LAMA = None
_LAMA_DEVICE = "cpu"


def _get_reader():
    global _READER
    if _READER is None:
        import easyocr
        # OCR device follows the worker device (set by _init_worker). MPS gives
        # ~6x per pass (0.7s vs 4.3s) BUT MPS + spawn-multiprocessing is unstable
        # on a 16GB Apple box — a 2-worker pool gets killed by signal mid-run. So
        # the PARALLEL path runs OCR on CPU across many workers (stable, and 8 CPU
        # workers beat 1 MPS worker on throughput); single-worker runs that ask
        # for mps/cuda still get the GPU.
        use_gpu = _LAMA_DEVICE in ("mps", "cuda")
        try:
            _READER = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception as e:
            sys.stderr.write(f"[ocr] gpu={use_gpu} init failed ({e}); using cpu\n")
            _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def _get_lama():
    global _LAMA, _LAMA_DEVICE
    if _LAMA is None:
        import torch
        from simple_lama_inpainting import SimpleLama
        want = _LAMA_DEVICE
        if want == "mps" and not torch.backends.mps.is_available():
            want = "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            want = "cpu"
        try:
            _LAMA = SimpleLama(device=torch.device(want))
            _LAMA_DEVICE = want
        except Exception as e:
            sys.stderr.write(f"[lama] {want} init failed ({e}); falling back to cpu\n")
            _LAMA = SimpleLama(device=torch.device("cpu"))
            _LAMA_DEVICE = "cpu"
    return _LAMA


def _maybe_empty_cache():
    """Release cached GPU memory. INTENTIONALLY NOT wired into the per-image path
    anymore: calling empty_cache every image was counterproductive (its cost grows
    with MPS memory and it did NOT bound the creep). Memory is bounded instead by
    run_overnight.sh CHUNKING — a fresh process every 500 imgs frees everything on
    exit. Kept for optional manual use; do not re-add to _scan_one/_process_one."""
    try:
        import torch
        if _LAMA_DEVICE == "mps":
            torch.mps.empty_cache()
        elif _LAMA_DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _round8(v):
    return int(v) - (int(v) % 8)


def _crop_window(bbox_p, shape, margin=CTX_MARGIN_PX):
    """A padded crop around the masked box, snapped to multiples of 8 (keeps
    LaMa's conv stack happy, especially on MPS where arbitrary sizes can break)."""
    H, W = shape[:2]
    x1, y1, x2, y2 = bbox_p
    x0 = max(0, _round8(x1 - margin))
    y0 = max(0, _round8(y1 - margin))
    x3 = min(W, _round8(x2 + margin) + 8)
    y3 = min(H, _round8(y2 + margin) + 8)
    return x0, y0, x3, y3


def _lama_crop_inpaint(bgr, mask):
    """Inpaint ONLY the bounding region of the mask, then hard-paste the masked
    pixels back. Everything outside the mask stays byte-for-byte the original —
    so a stray detection can only ever touch the box it found, never the rest of
    the photo. Returns (clean_bgr, did_inpaint)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return bgr, False
    bbox_p = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    x0, y0, x3, y3 = _crop_window(bbox_p, bgr.shape)
    crop = bgr[y0:y3, x0:x3]
    mcrop = mask[y0:y3, x0:x3]

    lama = _get_lama()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    out_pil = lama(Image.fromarray(rgb), Image.fromarray(mcrop))
    out_rgb = np.array(out_pil)
    if out_rgb.shape[:2] != crop.shape[:2]:
        out_rgb = cv2.resize(out_rgb, (crop.shape[1], crop.shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    m3 = (mcrop > 0)[..., None]
    composited = np.where(m3, out_bgr, crop)
    result = bgr.copy()
    result[y0:y3, x0:x3] = composited
    return result, True


def _residual_hits(bgr):
    """Single fast OCR pass for the post-inpaint recheck. The full 3-pass recall
    sweep is for *finding* faint marks; re-verifying a just-cleaned region only
    needs the reliable default pass, and skipping passes 2-3 here saves ~2 passes
    on every cleaned image (the dominant cost once OCR is on the GPU)."""
    return _ocr_pass(_get_reader(), bgr, 2.0, {})


def clean_image(bgr, bbox_p=None):
    """Detect -> crop-inpaint -> residual recheck -> one wider retry. Returns
    (clean_bgr_or_None, info). clean_bgr is None when nothing was detected.

    If bbox_p (the padded bbox the SCAN phase already found and stored in the
    manifest) is passed, the initial detection is SKIPPED and that box is reused
    for the mask. This removes the redundant SECOND detect the two-phase flow
    otherwise runs on every flagged image (scan detects, then process re-detected
    the same image). The post-inpaint residual recheck still runs, so removal is
    still verified — only the duplicated work is dropped."""
    boxes = None
    if bbox_p is None:
        boxes = detect_watermark_boxes(_get_reader(), bgr)
        if not boxes:
            return None, {"status": "no_mark"}
        bbox = union_bbox(boxes)
        bbox_p = pad_bbox(bbox, bgr.shape)
    else:
        bbox = bbox_p  # reuse: the scan's padded box is the retry base
    mask = build_mask(bgr.shape, bbox_p, dilate_px=6)
    cleaned, _ = _lama_crop_inpaint(bgr, mask)

    residual = _residual_hits(cleaned)
    retry = False
    if residual:
        retry = True
        bbox_r = union_bbox([(b[0], b[1], b[2], b[3]) for b in residual] + [tuple(bbox)])
        bbox_r = pad_bbox(bbox_r, bgr.shape, pad_frac_x=0.30, pad_frac_y=0.50, pad_px_min=12)
        mask = build_mask(bgr.shape, bbox_r, dilate_px=10)
        cleaned, _ = _lama_crop_inpaint(bgr, mask)
        residual = _residual_hits(cleaned)

    info = {
        "status": "cleaned" if not residual else "residual",
        "matches": [{"text": b[4], "conf": round(b[5], 3)} for b in boxes] if boxes else [],
        "bbox_padded": bbox_p,
        "retry": retry,
        "residual_after": len(residual),
        "reused_scan_bbox": boxes is None,
    }
    return cleaned, info


# ----------------------------------------------------------------- worker entries
def _init_worker(device):
    global _LAMA_DEVICE
    _LAMA_DEVICE = device


def _scan_one(path, review_dir):
    try:
        bgr = _imread(path)
        if bgr is None:
            return {"path": path, "status": "error", "err": "imread failed"}
        boxes = detect_watermark_boxes(_get_reader(), bgr)
        if not boxes:
            return {"path": path, "status": "clean"}
        bbox = union_bbox(boxes)
        bbox_p = pad_bbox(bbox, bgr.shape)
        rec = {
            "path": path,
            "status": "flagged",
            "bbox_padded": bbox_p,
            "conf_max": round(max(b[5] for b in boxes), 3),
            "matches": [b[4] for b in boxes][:6],
        }
        if review_dir:
            _write_overlay(bgr, bbox_p, review_dir, path)
        return rec
    except Exception as e:
        return {"path": path, "status": "error", "err": f"{type(e).__name__}: {e}"}


def _write_overlay(bgr, bbox_p, review_dir, path, max_w=700):
    """A downscaled thumbnail with the masked box drawn in red, so a human can
    skim _wm_review/ and catch a false positive before any pixels change."""
    x1, y1, x2, y2 = bbox_p
    vis = bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    h, w = vis.shape[:2]
    if w > max_w:
        s = max_w / w
        vis = cv2.resize(vis, (int(w * s), int(h * s)))
    name = Path(path).stem + "_flag.jpg"
    out = os.path.join(review_dir, name)
    cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tofile(out)


def _process_one(path, images_dir, backup_dir, bbox_p=None):
    try:
        bgr = _imread(path)
        if bgr is None:
            return {"path": path, "status": "error", "err": "imread failed"}
        cleaned, info = clean_image(bgr, bbox_p=bbox_p)
        if cleaned is None:
            # detector now disagrees with the scan — leave the file untouched
            return {"path": path, "status": "no_mark_skip"}
        # backup BEFORE overwriting
        rel = os.path.relpath(path, images_dir)
        bpath = os.path.join(backup_dir, rel)
        os.makedirs(os.path.dirname(bpath), exist_ok=True)
        if not os.path.exists(bpath):
            shutil.copy2(path, bpath)
        _imwrite_atomic(path, cleaned)
        info["path"] = path
        info["backup"] = bpath
        return info
    except Exception as e:
        return {"path": path, "status": "error", "err": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------- subcommands
def _run_pool(func, items, workers, device, log_path, every=25):
    """Map func over items across a spawn-based process pool (safe for torch/MPS),
    appending each result to log_path as it lands. Returns the list of results."""
    results = []
    counts = {}
    ctx = mp.get_context("spawn")
    t0 = time.time()
    logf = open(log_path, "a", buffering=1)
    try:
        if workers <= 1:
            _init_worker(device)
            for i, it in enumerate(items, 1):
                r = func(*it)
                results.append(r)
                counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
                logf.write(json.dumps(r) + "\n")
                if i % every == 0 or i == len(items):
                    _progress(i, len(items), counts, t0)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_init_worker,
                                     initargs=(device,)) as ex:
                futs = {ex.submit(func, *it): it for it in items}
                for i, fut in enumerate(as_completed(futs), 1):
                    r = fut.result()
                    results.append(r)
                    counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
                    logf.write(json.dumps(r) + "\n")
                    if i % every == 0 or i == len(items):
                        _progress(i, len(items), counts, t0)
    finally:
        logf.close()
    return results


def _progress(i, n, counts, t0):
    dt = time.time() - t0
    rate = i / dt if dt > 0 else 0
    eta = (n - i) / rate if rate > 0 else 0
    summ = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"[{i}/{n}] {rate:.2f} img/s  eta {eta/60:.1f}m  | {summ}", flush=True)


def cmd_scan(args):
    images_dir = os.path.abspath(args.images_dir)
    report = args.report or os.path.join(images_dir, SCAN_REPORT)
    review = None if args.no_overlays else (args.review_dir or os.path.join(images_dir, REVIEW_DIR))
    if review:
        os.makedirs(review, exist_ok=True)

    all_imgs = discover_images(images_dir)
    if not args.include_iphone14plus:
        kept = [p for p in all_imgs if not is_skippable_model(os.path.basename(p))]
        print(f"model filter: {len(all_imgs)} images found, "
              f"skipped {len(all_imgs) - len(kept)} iphone-{SKIP_IPHONE_MIN}+ (already clean), "
              f"{len(kept)} to scan")
        all_imgs = kept
    done = _load_done(report)
    todo = [p for p in all_imgs if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"scan: {len(all_imgs)} images, {len(done)} already scanned, {len(todo)} to do")
    if not todo:
        return _scan_summary(report)

    # GPU (mps/cuda) is only stable single-worker on this box; a multi-worker
    # pool runs OCR on MPS gets signal-killed, so force CPU when workers>1.
    dev = args.device if args.workers <= 1 else "cpu"
    items = [(p, review) for p in todo]
    _run_pool(_scan_one, items, args.workers, dev, report)
    _scan_summary(report)
    print(f"\nReview overlays: {review}\nManifest: {report}")
    print("Next: inspect overlays, then  python3 run_bulk.py process "
          f"--images-dir {images_dir} --device {args.device}")


def _scan_summary(report):
    recs = [json.loads(l) for l in open(report) if l.strip()] if os.path.exists(report) else []
    flagged = sum(1 for r in recs if r.get("status") == "flagged")
    clean = sum(1 for r in recs if r.get("status") == "clean")
    err = sum(1 for r in recs if r.get("status") == "error")
    print(f"\nSCAN: {len(recs)} total | flagged={flagged} clean={clean} error={err}")
    return recs


def cmd_process(args):
    images_dir = os.path.abspath(args.images_dir)
    report = args.report or os.path.join(images_dir, SCAN_REPORT)
    backup = args.backup_dir or os.path.join(images_dir, BACKUP_DIR)
    plog = args.process_log or os.path.join(images_dir, PROCESS_LOG)
    if not os.path.exists(report):
        sys.exit(f"no scan manifest at {report} — run `scan` first")
    os.makedirs(backup, exist_ok=True)

    recs = [json.loads(l) for l in open(report) if l.strip()]
    flagged = [r["path"] for r in recs if r.get("status") == "flagged"]
    # reuse the bbox the scan already found -> process skips the redundant re-detect
    bbox_by_path = {r["path"]: r.get("bbox_padded") for r in recs if r.get("status") == "flagged"}
    excludes = set()
    if args.exclude and os.path.exists(args.exclude):
        excludes = {l.strip() for l in open(args.exclude) if l.strip()}
        # accept either full paths or basenames in the exclude file
        flagged = [p for p in flagged if p not in excludes and os.path.basename(p) not in excludes]
    done = _load_done(plog)
    todo = [p for p in flagged if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"process: {len(flagged)} flagged (excluded {len(excludes)}), "
          f"{len(done)} already done, {len(todo)} to do | in-place, backup -> {backup}")
    if not todo:
        return
    if not args.yes:
        resp = input(f"Edit {len(todo)} images IN PLACE on '{images_dir}'? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit("aborted")

    # Same stability rule as scan: GPU only for single-worker; multi-worker -> CPU.
    dev = args.device if args.workers <= 1 else "cpu"
    if dev != args.device:
        print(f"note: {args.workers} workers -> cpu (mps/cuda is single-worker-only here)")
    items = [(p, images_dir, backup, bbox_by_path.get(p)) for p in todo]
    res = _run_pool(_process_one, items, args.workers, dev, plog)
    resid = [r for r in res if r.get("status") == "residual"]
    errs = [r for r in res if r.get("status") == "error"]
    print(f"\nPROCESS done. residual={len(resid)} errors={len(errs)}  log={plog}")
    if resid:
        print("  residual (left written, watermark may still be faintly readable):")
        for r in resid[:20]:
            print("   ", r["path"])
    if errs:
        for r in errs[:20]:
            print("   ERR", r["path"], r.get("err"))


def cmd_validate(args):
    images_dir = os.path.abspath(args.images_dir)
    plog = args.process_log or os.path.join(images_dir, PROCESS_LOG)
    if not os.path.exists(plog):
        sys.exit(f"no process log at {plog}")
    recs = [json.loads(l) for l in open(plog) if l.strip()]
    targets = [r["path"] for r in recs if r.get("status") in ("cleaned", "residual")]
    print(f"validate: re-detecting watermark on {len(targets)} processed images")
    _init_worker("cpu")
    bad = []
    for i, p in enumerate(targets, 1):
        bgr = _imread(p)
        if bgr is None:
            continue
        if detect_watermark_boxes(_get_reader(), bgr):
            bad.append(p)
        if i % 50 == 0 or i == len(targets):
            print(f"[{i}/{len(targets)}] still-detectable={len(bad)}", flush=True)
    print(f"\nVALIDATE: {len(targets)} checked, {len(bad)} still show a watermark")
    for p in bad[:30]:
        print("   ", p)


def cmd_restore(args):
    images_dir = os.path.abspath(args.images_dir)
    backup = args.backup_dir or os.path.join(images_dir, BACKUP_DIR)
    if not os.path.isdir(backup):
        sys.exit(f"no backup dir at {backup}")
    n = 0
    for root, _, files in os.walk(backup):
        for f in files:
            bpath = os.path.join(root, f)
            rel = os.path.relpath(bpath, backup)
            dest = os.path.join(images_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(bpath, dest)
            n += 1
    print(f"RESTORE: copied {n} originals back from {backup} -> {images_dir}")


def main():
    ap = argparse.ArgumentParser(description="Bulk watermark removal (v27 engine).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--images-dir", required=True)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
        p.add_argument("--limit", type=int, default=0, help="cap items (testing)")

    ps = sub.add_parser("scan", help="detect-only over all images -> manifest + overlays")
    common(ps)
    ps.add_argument("--report"); ps.add_argument("--review-dir")
    ps.add_argument("--no-overlays", action="store_true")
    ps.add_argument("--include-iphone14plus", action="store_true",
                    help="disable the iphone-14+ skip filter (scan the whole library)")
    ps.set_defaults(func=cmd_scan)

    pp = sub.add_parser("process", help="inpaint flagged images in place (with backup)")
    common(pp)
    pp.add_argument("--report"); pp.add_argument("--backup-dir")
    pp.add_argument("--process-log"); pp.add_argument("--exclude",
                    help="file of paths/basenames to skip (false positives)")
    pp.add_argument("--yes", action="store_true", help="skip the in-place confirmation")
    pp.set_defaults(func=cmd_process)

    pv = sub.add_parser("validate", help="re-detect on processed images")
    pv.add_argument("--images-dir", required=True)
    pv.add_argument("--process-log")
    pv.set_defaults(func=cmd_validate)

    pr = sub.add_parser("restore", help="copy originals back from backup")
    pr.add_argument("--images-dir", required=True)
    pr.add_argument("--backup-dir")
    pr.set_defaults(func=cmd_restore)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
