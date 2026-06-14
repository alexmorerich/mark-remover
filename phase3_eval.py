#!/usr/bin/env python3
"""Phase 3 — cleaner. Finder + mask + cleaner (one inpaint pass, no retry/cover-escalation,
to isolate the cleaner's quality) on 200 benchmark watermark images.
Gates: residual watermark < 5% · product damage < 2%.
Residual = re-detect on the cleaned output (CONFIRMED/LIKELY). Damage = audit.py verdict
REJECT_PRODUCT_DAMAGE on the (original, cleaned) pair.
"""
import json, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bulk as rb, logo_finder as lf, owner_agent as oa

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = HERE + "/benchmark"
FIN = BENCH + "/phase3_finals"


def main():
    os.makedirs(FIN, exist_ok=True)
    rb._init_worker("mps")
    reader = rb._get_reader()
    wm = [json.loads(l) for l in open(BENCH + "/benchmark_labels.jsonl") if json.loads(l)["label"] == "watermark"][:200]
    cleaned, residual, miss, prot, items = 0, 0, 0, 0, []
    for i, r in enumerate(wm, 1):
        src = os.path.join(BENCH, r["file"])
        bgr = rb._imread(src)
        res = lf.find(bgr, reader)
        if not res["candidates"] or res["presence"] in ("NO_WATERMARK", "UNSUITABLE_IMAGE"):
            miss += 1; continue
        c = res["candidates"][0]; roi = c["roi_class"]; box = c["_box"]
        method, mk = oa.repair_strategy(roi)
        if method == "reject":
            prot += 1; continue
        out = rb._lama_crop_inpaint(bgr, lf._masks(bgr.shape, box, roi)[mk])[0]
        fp = os.path.join(FIN, r["id"] + ".jpg")
        rb._imwrite_atomic(fp, out)
        cleaned += 1
        rd = lf.find(out, reader)["presence"] in ("CONFIRMED_WATERMARK", "LIKELY_WATERMARK")
        residual += int(rd)
        items.append({"id": r["id"], "original": src, "final": fp, "roi": roi, "residual": rd})
        if i % 40 == 0:
            print(f"[{i}/{len(wm)}] cleaned={cleaned} residual={residual}", flush=True)

    # damage: audit the pairs
    ip = BENCH + "/phase3_items.json"; json.dump(items, open(ip, "w"))
    subprocess.run([sys.executable, HERE + "/audit.py", "--manifest", ip, "--out-dir", BENCH + "/phase3_audit"],
                   stdout=open(BENCH + "/phase3_audit.log", "w"), stderr=subprocess.STDOUT, check=False)
    ar = {}
    arp = BENCH + "/phase3_audit/audit_results.jsonl"
    if os.path.exists(arp):
        for l in open(arp):
            if l.strip():
                d = json.loads(l); ar[d.get("id")] = d.get("audit_decision", "")
    damage = sum(1 for it in items if ar.get(it["id"]) == "REJECT_PRODUCT_DAMAGE")
    text_dmg = sum(1 for it in items if ar.get(it["id"]) == "REJECT_PROTECTED_TEXT_DAMAGE")

    json.dump({"items": items, "audit": ar}, open(BENCH + "/phase3_results.json", "w"))
    rr = residual / max(1, cleaned); dr = damage / max(1, cleaned)
    print(f"\n=== PHASE 3 — cleaner ({cleaned} cleaned, {miss} finder-miss, {prot} protected-reject) ===")
    print(f"residual watermark: {residual}/{cleaned} = {100*rr:.1f}%   → {'PASS' if rr < 0.05 else 'FAIL'} (gate <5%)")
    print(f"product damage:     {damage}/{cleaned} = {100*dr:.1f}%   → {'PASS' if dr < 0.02 else 'FAIL'} (gate <2%)")
    print(f"(protected-text-damage flagged by audit: {text_dmg})")


if __name__ == "__main__":
    main()
