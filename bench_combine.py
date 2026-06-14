"""
bench_combine.py — quantify whether COMBINING the two projects' watermark-ID methods
beats either alone, on b2bweb's labeled golden + validation sets.

Methods under test:
  owner   : v27.detect_watermark_boxes  (3 composite OCR passes) — the production SCAN
  chan    : owner composite + per-channel OCR on the canonical band (mark-remover v28 delta)
  canon   : b2bweb canonical-matte NCC slide (appearance, not legibility)
  chan|canon : the union (the proposed combined audit detector)

Reports recall over labeled positives and false-positive count over labeled negatives.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import v28_clean as v28e
import scan_audit  # reuse the band per-channel detector

# ── ported b2bweb canonical-matte detector (assets + tuned thresholds) ───────────
_CDIR = "/Users/alexkou/Documents/github/b2bweb/scripts"
_MATTE = cv2.imread(f"{_CDIR}/watermark-canonical.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
CW_FRAC = 0.34
CH_FRACS = (0.040, 0.046, 0.052, 0.058, 0.064)
CX_BAND, CY_LO, CY_HI = 0.15, 0.38, 0.62
CS, BG_MED = 0.35, 0.05    # CANONICAL_STANDALONE_MIN, CANONICAL_BG_MED_FRAC


def _absdiff(gray):
    h, w = gray.shape[:2]
    k = int(round(BG_MED * min(h, w)))
    k = max(5, k);  k += (k % 2 == 0)
    if k > min(h, w) - 1:
        k = max(5, (min(h, w) // 2) * 2 + 1)
    return cv2.absdiff(gray, cv2.medianBlur(gray, k)).astype(np.float32)


def canon_score(gray):
    h_img, w_img = gray.shape[:2]
    if h_img < 60 or w_img < 60:
        return 0.0
    dev = _absdiff(gray)
    xc, xj = w_img // 2, int(round(CX_BAND * w_img))
    best = 0.0
    for hf in CH_FRACS:
        sw, sh = max(20, int(round(CW_FRAC * w_img))), max(8, int(round(hf * h_img)))
        if sw >= w_img or sh >= h_img:
            continue
        tpl = cv2.resize(_MATTE, (sw, sh), interpolation=cv2.INTER_AREA)
        sx1, sx2 = max(0, xc - sw // 2 - xj), min(w_img, xc - sw // 2 + xj + 1 + sw)
        strip = dev[:, sx1:sx2]
        if strip.shape[1] <= sw:
            continue
        try:
            res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        r0 = max(0, int(CY_LO * h_img) - sh // 2)
        r1 = min(res.shape[0], int(CY_HI * h_img) - sh // 2 + 1)
        if r1 <= r0:
            continue
        best = max(best, float(res[r0:r1, :].max()))
    return best


# ── resolve a labeled file name to a PRISTINE image (backup if it was cleaned) ──
A = "/Users/alexkou/Documents/github/b2bweb/content/products/assets"


def resolve(fname):
    for cand in (f"{A}/_wm_backup/{fname}", f"{A}/{fname}"):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(f"{A}/**/{fname}", recursive=True)
    return hits[0] if hits else None


def load_labeled():
    items = {}
    for jf in ("generated/watermark/ground-truth.json",
               "generated/watermark/validation-set.json"):
        p = f"/Users/alexkou/Documents/github/b2bweb/{jf}"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for im in d.get("images", []):
            f = im["file"]
            lab = im.get("label", "")
            if lab not in ("positive", "negative"):
                continue
            items.setdefault(f, lab)   # dedupe across the two sets
    return items


def main():
    reader = v28e.make_reader()
    labeled = load_labeled()
    print(f"labeled images: {len(labeled)} "
          f"({sum(v=='positive' for v in labeled.values())}+ / "
          f"{sum(v=='negative' for v in labeled.values())}-)", flush=True)

    rows = []
    for f, lab in sorted(labeled.items()):
        p = resolve(f)
        if not p:
            print(f"  MISSING {f}", flush=True)
            continue
        bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        owner = bool(v28e.v27.detect_watermark_boxes(reader, bgr))   # owner's scan method
        chan_hits, _, _ = scan_audit.detect(reader, bgr, use_canon=False)  # +per-channel band (no canon)
        chan = bool(chan_hits)
        cscore = canon_score(gray)
        canon = cscore >= CS
        rows.append((f, lab, owner, chan, canon, cscore))
        tag = "+" if lab == "positive" else "-"
        print(f"  {tag} owner={int(owner)} chan={int(chan)} canon={int(canon)}"
              f"({cscore:.2f})  {f[:52]}", flush=True)

    pos = [r for r in rows if r[1] == "positive"]
    neg = [r for r in rows if r[1] == "negative"]

    def rec(sel):  # recall over positives
        return sum(1 for r in pos if sel(r)) / max(1, len(pos))

    def fp(sel):   # false positives over negatives
        return sum(1 for r in neg if sel(r))

    variants = {
        "owner (v27 scan)":        lambda r: r[2],
        "chan (audit now)":        lambda r: r[3],
        "canon (b2bweb matte)":    lambda r: r[4],
        "chan | canon (COMBINED)": lambda r: r[3] or r[4],
        "owner | canon":           lambda r: r[2] or r[4],
    }
    print(f"\n=== RESULTS  (positives={len(pos)}  negatives={len(neg)}) ===", flush=True)
    print(f"{'variant':28} {'recall':>8} {'misses':>7} {'FP':>4}", flush=True)
    for name, sel in variants.items():
        print(f"{name:28} {rec(sel)*100:7.1f}% {sum(1 for r in pos if not sel(r)):7} {fp(sel):4}", flush=True)

    # what does canon RESCUE that the per-channel audit still misses?
    rescued = [r[0] for r in pos if (not r[3]) and r[4]]
    print(f"\ncanon rescues {len(rescued)} positives that chan-audit misses:", flush=True)
    for f in rescued:
        print(f"   + {f}", flush=True)
    json.dump([{"file": r[0], "label": r[1], "owner": r[2], "chan": r[3],
                "canon": r[4], "cscore": round(r[5], 3)} for r in rows],
              open("/tmp/audit_test/bench_combine.json", "w"), indent=2)
    print("\nwrote /tmp/audit_test/bench_combine.json", flush=True)


if __name__ == "__main__":
    main()
