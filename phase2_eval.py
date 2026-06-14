#!/usr/bin/env python3
"""Phase 2 — mask quality. Finder + mask generator on 100 benchmark watermark images.
Gate: ≥95% masks usable (covers the logo, stays in the centre band, doesn't blanket product).
Protected-text rejects and finder misses are excluded from the denominator (not mask failures).
"""
import json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bulk as rb, logo_finder as lf, owner_agent as oa

BENCH = os.path.dirname(os.path.abspath(__file__)) + "/benchmark"


def main():
    rb._init_worker("mps")
    reader = rb._get_reader()
    wm = [json.loads(l) for l in open(BENCH + "/benchmark_labels.jsonl") if json.loads(l)["label"] == "watermark"][:100]
    results, overlays = [], []
    for r in wm:
        bgr = rb._imread(os.path.join(BENCH, r["file"]))
        res = lf.find(bgr, reader)
        if not res["candidates"] or res["presence"] in ("NO_WATERMARK", "UNSUITABLE_IMAGE"):
            results.append({"id": r["id"], "status": "finder_miss"}); continue
        c = res["candidates"][0]; roi = c["roi_class"]; box = c["_box"]
        method, mk = oa.repair_strategy(roi)
        if method == "reject":
            results.append({"id": r["id"], "status": "reject_protected", "roi": roi}); continue
        mask = lf._masks(bgr.shape, box, roi)[mk]
        H, W = bgr.shape[:2]
        area = float((mask > 0).mean())
        x1, y1, x2, y2 = box
        bb = np.zeros((H, W), np.uint8); bb[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1
        covers = float(((mask > 0) & (bb > 0)).sum() / max(1, bb.sum()))      # mask covers the logo box
        ys = np.where(mask > 0)[0]
        within = bool(len(ys) and ys.min() >= 0.30 * H and ys.max() <= 0.63 * H)   # stays in centre band
        usable = bool(covers >= 0.9 and area < 0.16 and within)
        results.append({"id": r["id"], "status": "mask", "mask": mk, "roi": roi,
                        "area": round(area, 3), "covers": round(covers, 2), "within": within, "usable": usable})
        if len(overlays) < 12:
            ov = bgr.copy(); m = mask > 0
            ov[m] = (0.45 * ov[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            overlays.append((r["id"], roi, mk, usable, ov))

    masks_made = [x for x in results if x["status"] == "mask"]
    usable = sum(1 for x in masks_made if x["usable"])
    rej = sum(1 for x in results if x["status"] == "reject_protected")
    miss = sum(1 for x in results if x["status"] == "finder_miss")
    rate = usable / max(1, len(masks_made))
    json.dump(results, open(BENCH + "/phase2_results.json", "w"), indent=1)
    # overlay contact sheet
    from PIL import Image, ImageDraw, ImageFont
    fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15); CW = 380; strips = []
    for iid, roi, mk, ok, ov in overlays:
        s = CW / ov.shape[1]; im = cv2.resize(ov, (CW, int(ov.shape[0] * s)))
        lab = Image.new("RGB", (CW, 18), "white")
        ImageDraw.Draw(lab).text((2, 1), f"{iid} [{roi}] {mk} {'OK' if ok else 'CHECK'}", fill="black", font=fnt)
        strips += [lab, Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))]
    cols = 3
    rowsimg = [strips[i:i + 2] for i in range(0, len(strips), 2)]
    cellh = max(s.height for s in strips)
    sheet = Image.new("RGB", (CW * cols, ((len(rowsimg) + cols - 1) // cols) * (cellh)), "white")
    for i, (lab, im) in enumerate(rowsimg):
        cx = (i % cols) * CW; cy = (i // cols) * cellh
        combo = Image.new("RGB", (CW, cellh), "white"); combo.paste(lab, (0, 0)); combo.paste(im, (0, 18))
        sheet.paste(combo, (cx, cy))
    sheet.save(BENCH + "/_phase2_masks.png")
    print(f"PHASE 2: {len(wm)} wm | masks generated {len(masks_made)} | USABLE {usable} ({100*rate:.1f}%) "
          f"→ {'PASS' if rate >= 0.95 else 'FAIL'} (gate 95%)")
    print(f"protected-reject {rej} | finder_miss {miss}")
    import collections
    print("mask types:", dict(collections.Counter(x["mask"] for x in masks_made)))
    print("non-usable:", [(x["id"], x["roi"], x["mask"], x["area"], x["covers"], x["within"]) for x in masks_made if not x["usable"]][:10])


if __name__ == "__main__":
    main()
