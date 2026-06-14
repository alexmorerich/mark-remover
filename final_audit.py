#!/usr/bin/env python3
"""AUDIT module — independent final verification of the re-clean output.

Stronger than the inline per-image band check: takes a random sample of the
re-cleaned files ON DISK, runs the FULL 3-pass detector, and counts a REAL
residual only when a surviving match is a sunsky anchor / online-com fragment
(noise-token-only hits are product text, not watermark — same rule the routing
used). Reports the residual rate so the re-clean has a measured quality number.
"""
import os, sys, json, random, re, argparse
import numpy as np, cv2
sys.path.insert(0, "/Users/alexkou/Documents/github/mark-remover")
import run_bulk as rb

LOG = "/Users/alexkou/Documents/github/mark-remover/_reclean.jsonl"
FRAG = re.compile(r"sunsk|sunsh|onl|nlin|line|\.c|cemn|c8n|sky-o", re.IGNORECASE)


def is_real_wm(text):
    return bool(FRAG.search(text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="/Users/alexkou/Documents/github/mark-remover/_final_audit.json")
    args = ap.parse_args()
    rb._init_worker(args.device)

    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    cleaned = [r["path"] for r in recs if r.get("action") == "reclean" and r.get("status") == "cleaned"]
    random.seed(7)
    samp = random.sample(cleaned, min(args.n, len(cleaned)))

    real_resid, noise_only, errs, details = 0, 0, 0, []
    for i, p in enumerate(samp, 1):
        bgr = rb._imread(p)
        if bgr is None:
            errs += 1; continue
        boxes = rb.detect_watermark_boxes(rb._get_reader(), bgr)
        texts = [b[4] for b in boxes]
        real = [t for t in texts if is_real_wm(t)]
        if real:
            real_resid += 1
            details.append({"path": os.path.basename(p), "matches": texts})
        elif texts:
            noise_only += 1
        if i % 25 == 0:
            print(f"[{i}/{len(samp)}] real_residual={real_resid} noise_only={noise_only} err={errs}", flush=True)

    n = len(samp)
    rate = real_resid / n if n else 0
    out = {
        "sampled": n, "real_residual": real_resid, "noise_only_hits": noise_only,
        "errors": errs, "real_residual_rate": round(rate, 4),
        "ci95_upper_pct": round(100 * (3.0 / n if real_resid == 0 else (real_resid + 1.96 * (rate * (1 - rate) / n) ** 0.5)), 2),
        "residual_examples": details[:20],
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== FINAL AUDIT ===")
    print(f"sampled {n} re-cleaned files | REAL residual watermark: {real_resid} ({100*rate:.2f}%)")
    print(f"noise-only OCR hits (product text, not watermark): {noise_only} | read errors: {errs}")
    print(f"verdict: {'PASS — clean' if real_resid == 0 else 'REVIEW — residuals found'}")
    print("written", args.out)


if __name__ == "__main__":
    main()
