#!/usr/bin/env python3
"""Owner's INDEPENDENT no-leak safety sweep for the Audit misjudgment patch.

The residual *precision* fix has one catastrophic failure mode: over-correcting so a real
watermark PASSes. This is the Owner's black-box guard (no audit.py code read): feed
KNOWN-watermarked images (full sunsky mark) to audit.py as unedited finals
({original == final == backup}); EVERY one must be REJECTED (publish_allowed == False).
Any PASS = a real watermark would leak → the patch must revert, regardless of the Audit
Agent's own regression. Weighted toward textured / flex / dark / metallic surfaces — exactly
where the fix changes behavior.

  python3 noleak_sweep.py --build      # freeze the set (run now to pre-stage)
  python3 noleak_sweep.py --check       # run audit + verify 0 leaks (run after they push)
"""
import argparse, json, os, re, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = "/Users/alexkou/Documents/github/b2bweb/content/products/assets"
BK = os.path.join(A, "_wm_backup")
SCAN = os.path.join(A, "_wm_scan.jsonl")
ITEMS = os.path.join(HERE, "benchmark", "noleak_items.json")
OUTDIR = os.path.join(HERE, "benchmark", "noleak_audit")
STRONG = re.compile(r"sunsk|sunsh|nline|onlin|\.com|online", re.I)
KW = [("flex", re.compile(r"flex|ribbon|\bcable\b|charging-port", re.I)),
      ("camera", re.compile(r"camera|\blens\b", re.I)),
      ("glass", re.compile(r"glass|tempered|digitizer|back-cover", re.I)),
      ("metallic", re.compile(r"metal|alloy|steel|bracket|\bplate\b|shield", re.I)),
      ("label", re.compile(r"sticker|adhesive|\blabel\b|\btape\b", re.I))]
QUOTA = {"flex": 18, "camera": 14, "glass": 14, "metallic": 14, "label": 8, "other": 32}  # textured-weighted


def cat(name):
    for c, rx in KW:
        if rx.search(name):
            return c
    return "other"


def build():
    recs = [json.loads(l) for l in open(SCAN) if l.strip()]
    wm = [r for r in recs if r.get("status") == "flagged"
          and any(STRONG.search(str(m)) for m in (r.get("matches") or []))]
    random.seed(11); random.shuffle(wm)
    got = {k: 0 for k in QUOTA}; items = []; seen = set()
    for r in wm:
        b = os.path.basename(r["path"]); p = os.path.join(BK, b)
        if not os.path.exists(p):
            continue
        iid = os.path.splitext(b)[0]
        if iid in seen:                                    # unique ids only (the [:60] truncation collided → 4 dups)
            continue
        c = cat(b.lower())
        if got[c] >= QUOTA[c]:
            continue
        seen.add(iid); got[c] += 1
        items.append({"id": iid, "original": p, "final": p})
        if sum(got.values()) >= sum(QUOTA.values()):
            break
    os.makedirs(os.path.dirname(ITEMS), exist_ok=True)
    json.dump(items, open(ITEMS, "w"))
    print(f"no-leak set frozen: {len(items)} known-watermarked images | by category: {got}")
    print(f"  saved {ITEMS}")


def check():
    if not os.path.exists(ITEMS):
        print("run --build first"); return
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    subprocess.run([sys.executable, os.path.join(HERE, "audit.py"),
                    "--manifest", ITEMS, "--out-dir", OUTDIR], env=env, check=False)
    rp = os.path.join(OUTDIR, "audit_results.jsonl")
    res = [json.loads(l) for l in open(rp) if l.strip()] if os.path.exists(rp) else []
    leaks = [r for r in res if r.get("publish_allowed")]
    n = len(res)
    print(f"\n=== NO-LEAK SWEEP: {n} known-watermarked images ===")
    print(f"rejected (caught): {n - len(leaks)}/{n}")
    print(f"LEAKS (watermarked but PASS): {len(leaks)}  -> {'SAFE ✓' if not leaks else 'FAIL — patch must REVERT'}")
    for r in leaks[:15]:
        print("  LEAK:", r.get("id"), r.get("audit_decision"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.check:
        check()
    if not (a.build or a.check):
        ap.error("--build or --check")


if __name__ == "__main__":
    main()
