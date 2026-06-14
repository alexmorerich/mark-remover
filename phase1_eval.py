#!/usr/bin/env python3
"""Phase 1 — Logo Finder validation. Run the finder ONLY (no cleaning) over the 500-image
benchmark, emit FOUND / NOT_FOUND per image, and compute recall/precision vs the labels.

Exit criteria: recall > 98%, precision > 90% (recall dominates — a miss = guaranteed publish
failure). Reported at two thresholds: strict (CONFIRMED/LIKELY) and loose (+UNCERTAIN).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bulk as rb
import logo_finder as lf

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
OUT = os.path.join(BENCH, "phase1_results.jsonl")


def main():
    rb._init_worker("mps")
    reader = rb._get_reader()
    rows = [json.loads(l) for l in open(os.path.join(BENCH, "benchmark_labels.jsonl"))]
    f = open(OUT, "w", buffering=1)
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        bgr = rb._imread(os.path.join(BENCH, r["file"]))
        p = lf.find(bgr, reader)["presence"] if bgr is not None else "UNSUITABLE_IMAGE"
        strict = p in ("CONFIRMED_WATERMARK", "LIKELY_WATERMARK")
        loose = strict or p == "UNCERTAIN_WATERMARK"
        f.write(json.dumps({"id": r["id"], "label": r["label"], "category": r["category"],
                            "presence": p, "found_strict": strict, "found_loose": loose}) + "\n")
        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] {i/(time.time()-t0):.2f}/s", flush=True)
    f.close()

    res = [json.loads(l) for l in open(OUT)]
    print("\n=== PHASE 1 — finder recall/precision (labels = b2bweb scan, UNVERIFIED) ===")
    for mode in ("found_strict", "found_loose"):
        TP = sum(1 for r in res if r["label"] == "watermark" and r[mode])
        FN = sum(1 for r in res if r["label"] == "watermark" and not r[mode])
        FP = sum(1 for r in res if r["label"] == "clean" and r[mode])
        TN = sum(1 for r in res if r["label"] == "clean" and not r[mode])
        rec = TP / (TP + FN) if TP + FN else 0
        prec = TP / (TP + FP) if TP + FP else 0
        gate = "PASS" if (rec > 0.98 and prec > 0.90) else "FAIL"
        print(f"{mode:13s}: recall={rec:.3f} precision={prec:.3f}  "
              f"TP={TP} FN={FN} FP={FP} TN={TN}  → {gate}")
    # recall by category (where do we miss?)
    print("\nmisses by category (watermark labelled, NOT found strict):")
    cats = {}
    for r in res:
        if r["label"] == "watermark":
            cats.setdefault(r["category"], [0, 0])
            cats[r["category"]][0] += 1
            cats[r["category"]][1] += 0 if r["found_strict"] else 1
    for c, (tot, miss) in sorted(cats.items()):
        print(f"  {c:14s}: {miss}/{tot} missed")


if __name__ == "__main__":
    main()
