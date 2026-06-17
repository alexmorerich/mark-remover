#!/usr/bin/env python3
"""Calibrate the fast-detector config against the existing scan manifest.

For a sample of manifest-labeled images, run the (optionally base-capped) detector
pass-by-pass with early stop, recording which pass first hits and per-pass time.
From one run we get, for every maxpass option:
  - RECALL on flagged (watermarked) images = fraction still detected within k passes
  - per-image time (clean images are where passes differ; they run every pass)

Pass reduction can only lose recall, never add false positives, so we only need to
measure recall on the flagged set + speed. cap is set via WM_OCR_BASE_CAP env.
"""
import os, json, time, random, sys
sys.path.insert(0, "/Users/alexkou/Documents/github/mark-remover")
import numpy as np, cv2
import v27_clean as v

ASSETS = "/Users/alexkou/Documents/github/b2bweb/content/products/assets"
PASSES = [
    (2.0, {}),
    (3.0, dict(contrast_ths=0.01, adjust_contrast=0.9, text_threshold=0.4, low_text=0.2, link_threshold=0.3)),
    (2.0, dict(contrast_ths=0.01, adjust_contrast=0.9, text_threshold=0.3, low_text=0.15, link_threshold=0.2)),
]
N_FLAG = int(os.environ.get("BENCH_FLAG", "150"))
N_CLEAN = int(os.environ.get("BENCH_CLEAN", "90"))


def imread(p):
    try:
        d = np.fromfile(p, np.uint8)
        return cv2.imdecode(d, cv2.IMREAD_COLOR) if d.size else None
    except Exception:
        return None


def main():
    print(f"cap={v.OCR_BASE_CAP}  (WM_OCR_BASE_CAP)")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)  # CPU: stable, recall is device-independent
    recs = [json.loads(l) for l in open(os.path.join(ASSETS, "_wm_scan.jsonl")) if l.strip()]
    flagged = [r["path"] for r in recs if r.get("status") == "flagged"]
    clean = [r["path"] for r in recs if r.get("status") == "clean"]
    random.seed(11)
    fs = random.sample(flagged, min(N_FLAG, len(flagged)))
    cs = random.sample(clean, min(N_CLEAN, len(clean)))

    for paths, label in [(fs, "FLAGGED"), (cs, "CLEAN")]:
        ptime = [0.0, 0.0, 0.0]
        hitpass = {0: 0, 1: 0, 2: 0, 3: 0}
        n = 0
        t0 = time.time()
        for p in paths:
            bgr = imread(p)
            if bgr is None:
                continue
            n += 1
            hp = 0
            for i, (sc, pa) in enumerate(PASSES):
                t = time.time()
                hits = v._ocr_pass(reader, bgr, sc, pa)
                ptime[i] += time.time() - t
                if hits:
                    hp = i + 1
                    break
            hitpass[hp] += 1
        if not n:
            continue
        T1 = ptime[0]
        T2 = ptime[0] + ptime[1]
        T3 = sum(ptime)
        print(f"\n[{label}] n={n} wall={time.time()-t0:.0f}s")
        print(f"  first-hit-pass dist: {hitpass}")
        print(f"  time/img:  maxpass1={T1/n:.2f}s  maxpass2={T2/n:.2f}s  maxpass3={T3/n:.2f}s")
        if label == "FLAGGED":
            r1 = hitpass[1] / n
            r2 = (hitpass[1] + hitpass[2]) / n
            r3 = (hitpass[1] + hitpass[2] + hitpass[3]) / n
            print(f"  RECALL@cap: maxpass1={r1:.3f}  maxpass2={r2:.3f}  maxpass3={r3:.3f}  miss_all_3={hitpass[0]/n:.3f}")


if __name__ == "__main__":
    main()
