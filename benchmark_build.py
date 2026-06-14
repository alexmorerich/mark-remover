#!/usr/bin/env python3
"""Phase 0 — assemble the permanent 500-image benchmark (250 watermark + 250 clean),
stratified across surface categories.

Labels come from the INDEPENDENT b2bweb scan (flagged+strong-sunsky = watermark; clean =
clean) — deliberately NOT from logo_finder, so Phase 1 (finder recall/precision) is an
honest test, not circular. Category is a surface bucket from logo_finder._roi_class over
the canonical band (bucketing only, not labelling). Output is a candidate set with
verified=false; the Audit Agent verifies once and freezes it. Never rebuilt after that.
"""
import json, os, re, random, shutil, sys, math
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logo_finder as lf
import run_bulk as rb

A = "/Users/alexkou/Documents/github/b2bweb/content/products/assets"
SCAN = os.path.join(A, "_wm_scan.jsonl")
BK = os.path.join(A, "_wm_backup")
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
IMG = os.path.join(BENCH, "images")
N_EACH, PER_CAT, BUDGET = 250, 30, 1600

# product-TYPE keywords decide the hard categories (a center-band surface ROI can't tell a
# camera module from a flex cable); fall back to the surface ROI bucket otherwise.
KW = [
    ("flex", re.compile(r"flex|ribbon|\bcable\b|charging-port", re.I)),
    ("camera", re.compile(r"camera|\blens\b", re.I)),
    ("glass", re.compile(r"glass|tempered|digitizer|back-cover", re.I)),
    ("label", re.compile(r"sticker|adhesive|\blabel\b|\btape\b", re.I)),
    ("metallic", re.compile(r"metal|alloy|steel|bracket|\bplate\b|shield", re.I)),
]
ROI2CAT = {"plain_white": "white", "near_white": "near_white", "low_texture_background": "near_white",
           "dark_product_surface": "dark", "metallic_or_reflective": "metallic"}
STRONG = re.compile(r"sunsk|sunsh|nline|onlin|\.com|online", re.I)


def cat_of(path, bgr):
    name = os.path.basename(path).lower()
    for cat, rx in KW:
        if rx.search(name):
            return cat
    H, W = bgr.shape[:2]
    box = [int(.23 * W), int(.43 * H), int(.77 * W), int(.52 * H)]
    return ROI2CAT.get(lf._roi_class(bgr, box), "mixed")


def classify(pool, use_backup):
    out = []
    for r in pool:
        if len(out) >= BUDGET:
            break
        b = os.path.basename(r["path"])
        path = os.path.join(BK, b) if use_backup else r["path"]
        if not os.path.exists(path):
            continue
        bgr = rb._imread(path)
        if bgr is None:
            continue
        out.append((path, cat_of(path, bgr), b))
    return out


def select(classified):
    chosen, bycat, seen = [], {}, set()
    for path, c, b in classified:                       # pass 1 — capped for diversity
        if len(chosen) >= N_EACH:
            break
        if bycat.get(c, 0) >= PER_CAT:
            continue
        seen.add(path); bycat[c] = bycat.get(c, 0) + 1; chosen.append((path, c, b))
    for path, c, b in classified:                       # pass 2 — fill to N
        if len(chosen) >= N_EACH:
            break
        if path in seen:
            continue
        seen.add(path); bycat[c] = bycat.get(c, 0) + 1; chosen.append((path, c, b))
    return chosen, bycat


def main():
    if os.path.isdir(IMG):
        shutil.rmtree(IMG)                              # rebuild clean (not frozen until Audit verifies)
    os.makedirs(IMG, exist_ok=True)
    rb._init_worker("cpu")
    recs = [json.loads(l) for l in open(SCAN) if l.strip()]
    flagged = [r for r in recs if r.get("status") == "flagged"
               and any(STRONG.search(str(m)) for m in (r.get("matches") or []))]
    clean = [r for r in recs if r.get("status") == "clean"]
    random.seed(42); random.shuffle(flagged); random.shuffle(clean)

    wm, wmcat = select(classify(flagged, True))
    cl, clcat = select(classify(clean, False))

    mf = open(os.path.join(BENCH, "benchmark_labels.jsonl"), "w")
    n = 0
    for label, items in (("watermark", wm), ("clean", cl)):
        for path, cat, b in items:
            n += 1
            bid = f"bench{n:04d}"
            fname = f"{bid}_{label}_{cat}.jpg"
            shutil.copy2(path, os.path.join(IMG, fname))
            mf.write(json.dumps({
                "id": bid, "file": f"images/{fname}", "source": path,
                "label": label, "category": cat,
                "label_source": "b2bweb_scan", "verified": False,
            }) + "\n")
    mf.close()
    print(f"benchmark: {n} images  ({len(wm)} watermark + {len(cl)} clean)")
    print("watermark by category:", dict(sorted(wmcat.items())))
    print("clean by category:    ", dict(sorted(clcat.items())))


if __name__ == "__main__":
    main()
