#!/usr/bin/env python3
"""Re-clean / restore the sunsky watermark by its CANONICAL fixed geometry.

Single-owner pipeline with three separate modules behind stable interfaces
(governance: detect | repair | audit — keep them apart):

  DETECT  reclean_routing.csv  (filename, cleaned_path, backup_path, action, ocr_match)
          Derived once from the scan manifest (_wm_scan.jsonl). Consumed read-only
          here. Routing rule (validated on 20/20 real + the noise-token set):
            match has 'sunsk'/'sunsh' OR an online/com fragment  -> action=reclean
            only a noise token (no watermark; false positive)     -> action=restore

  REPAIR  repair_one(backup, action)
            reclean : LaMa-inpaint a generous centered band from the PRISTINE backup
                      (mark is a fixed center stamp: cx/W .50 cy/H .475 w/W ~.45 h/H ~.06,
                      measured across 700 detections — coverage no longer needs detection)
            restore : return the pristine backup unchanged (undo a false-positive inpaint)

  AUDIT   audit_band(cleaned)
            OCR the canonical band of the repaired image; a surviving sunsky fragment
            means the inpaint missed -> status 'residual' -> auto-rejected to the review
            queue (never silently accepted). Restores need no watermark audit.

Records written to the re-clean manifest use the SAME core schema as _wm_process.jsonl
(status, matches, bbox_padded, retry, residual_after, path, backup) so validate/restore
tooling keeps working; re-clean adds action/method/reclean fields.

Modes:
  python3 reclean.py --samples
  python3 reclean.py --run --csv reclean_routing.csv --action reclean --device mps --limit N --apply
  python3 reclean.py --run --csv reclean_routing.csv --action restore --apply
"""
import os, sys, json, glob, argparse, time
import numpy as np, cv2
sys.path.insert(0, "/Users/alexkou/Documents/github/mark-remover")
import run_bulk as rb

# canonical band as fractions of (W,H): centered x, vertical center .475
CANON_FX = (0.23, 0.77)
CANON_FY = (0.43, 0.52)
DBG = "/Users/alexkou/Documents/github/mark-remover/reclean_debug"


def canonical_mask(shape, fx=CANON_FX, fy=CANON_FY, dilate=4):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    m[int(fy[0] * H):int(fy[1] * H), int(fx[0] * W):int(fx[1] * W)] = 255
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
        m = cv2.dilate(m, k)
    return m


def _canon_bbox(shape):
    H, W = shape[:2]
    return [int(CANON_FX[0] * W), int(CANON_FY[0] * H), int(CANON_FX[1] * W), int(CANON_FY[1] * H)]


# ----------------------------------------------------------------- REPAIR module
def repair_one(backup_path, action):
    """'reclean' -> canonical-band LaMa inpaint from pristine backup;
    'restore' -> pristine backup unchanged. Returns (image_or_None, rec)."""
    bgr = rb._imread(backup_path)
    if bgr is None:
        return None, {"method": action, "err": "imread_failed", "bbox_padded": None}
    if action == "restore":
        return bgr, {"method": "restore", "bbox_padded": None}
    mask = canonical_mask(bgr.shape)
    cleaned, did = rb._lama_crop_inpaint(bgr, mask)
    return cleaned, {"method": "canonical_band", "bbox_padded": _canon_bbox(bgr.shape), "did_inpaint": did}


# ------------------------------------------------------------------ AUDIT module
def audit_band(cleaned_bgr):
    """Re-OCR the canonical band of the repaired image. Returns (hit_count, matches)."""
    H, W = cleaned_bgr.shape[:2]
    crop = cleaned_bgr[int(CANON_FY[0] * H):int(CANON_FY[1] * H),
                       int(CANON_FX[0] * W):int(CANON_FX[1] * W)]
    hits = rb._residual_hits(crop)
    return len(hits), [{"text": h[4], "conf": round(float(h[5]), 3)} for h in hits]


# kept for build_sample_pdf.py (before/after preview): repair + optional audit, no manifest I/O
def reclean_one(backup_path, verify=True):
    img, rec = repair_one(backup_path, "reclean")
    if img is None:
        return None, {"status": "error", "err": rec.get("err")}
    info = {"status": "recleaned", "bbox_padded": rec.get("bbox_padded")}
    if verify:
        resid, _ = audit_band(img)
        info["residual_after"] = resid
        info["status"] = "recleaned" if resid == 0 else "residual"
    return img, info


# --------------------------------------------------------------- resumable runner
def _load_done(logpath):
    """Resume checkpoint: paths already written to the re-clean manifest are skipped."""
    done = set()
    if os.path.exists(logpath):
        with open(logpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line).get("path")
                    if p:
                        done.add(p)
                except Exception:
                    pass
    return done


def cmd_samples(args):
    rb._init_worker("cpu")
    for p in sorted(glob.glob(os.path.join(DBG, "orig_*.jpg"))):
        t = time.time()
        cleaned, info = reclean_one(p)
        if cleaned is None:
            print(p, info); continue
        out = p.replace("orig_", "recleaned_")
        cv2.imencode(".jpg", cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(out)
        print(f"{os.path.basename(p)[:34]:34s} {info}  {time.time()-t:.1f}s -> {os.path.basename(out)}")


def cmd_run(args):
    """DETECT(routing) -> REPAIR -> AUDIT, resumable (skip-done) and chunkable (--limit)."""
    import csv
    rb._init_worker(args.device)
    rows = [r for r in csv.DictReader(open(args.csv)) if r.get("action") == args.action]
    done = _load_done(args.log)
    todo = [r for r in rows if r["cleaned_path"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    log = open(args.log, "a", buffering=1)
    review = open(args.review, "a", buffering=1)
    n_ok = n_resid = n_err = 0
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        cp, bp = r["cleaned_path"], r["backup_path"]
        det_match = r.get("ocr_match", "") or ""
        img, rec = repair_one(bp, args.action)                       # REPAIR
        if img is None:
            n_err += 1
            log.write(json.dumps({"status": "error", "path": cp, "backup": bp,
                                  "action": args.action, "err": rec.get("err"),
                                  "reclean": True}) + "\n")
            continue
        if args.action == "reclean":                                 # AUDIT
            resid, amatches = audit_band(img)
            status = "cleaned" if resid == 0 else "residual"
        else:
            resid, status = 0, "restored"
        if args.apply:
            rb._imwrite_atomic(cp, img)                              # overwrite in place; backup preserved
        rec_out = {
            "status": status,
            "matches": [{"text": t, "conf": None} for t in det_match.split("|") if t],
            "bbox_padded": rec.get("bbox_padded"),
            "retry": False,
            "residual_after": resid,
            "path": cp,
            "backup": bp,
            "action": args.action,
            "method": rec.get("method"),
            "reclean": True,
            "applied": bool(args.apply),
        }
        log.write(json.dumps(rec_out) + "\n")
        if status == "residual":                                     # auto_rejected -> review queue
            n_resid += 1
            review.write(cp + "\n")
        else:
            n_ok += 1
        if i % 25 == 0 or i == len(todo):
            dt = time.time() - t0
            rate = i / dt if dt else 0
            print(f"[{i}/{len(todo)}] {rate:.2f}/s ok={n_ok} residual={n_resid} err={n_err}", flush=True)
    log.close(); review.close()
    print(f"DONE run action={args.action}: ok={n_ok} residual={n_resid} err={n_err} (this chunk of {len(todo)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--run", action="store_true", help="resumable routed re-clean/restore")
    ap.add_argument("--csv")
    ap.add_argument("--action", default="reclean", choices=["reclean", "restore"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true", help="overwrite images in place")
    ap.add_argument("--log", default="/Users/alexkou/Documents/github/mark-remover/_reclean.jsonl")
    ap.add_argument("--review", default="/Users/alexkou/Documents/github/mark-remover/_reclean_review.txt")
    args = ap.parse_args()
    if args.samples:
        cmd_samples(args)
    elif args.run:
        if not args.csv:
            ap.error("--run needs --csv")
        cmd_run(args)
    else:
        ap.error("need --samples or --run")


if __name__ == "__main__":
    main()
