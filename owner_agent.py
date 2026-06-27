#!/usr/bin/env python3
"""owner_agent.py — Owner Agent: original → detect → repair/cover → final + manifest.

Production pipeline per docs/OWNER_AGENT_WORKFLOW.md. The Owner produces publish-SAFE
candidate finals and an owner_manifest.jsonl for the Audit Agent — it never publishes.
Simple by design: the goal is clean final images Audit can PASS, not a benchmark.

  original → detect → repair / cover / copy / reject → save → manifest → (Audit) → retry

Job layout (originals/ is never modified):
  job_xxx/{originals,finals,masks,rejected,logs}/  owner_manifest.jsonl  audit_feedback.jsonl

CLI:
  python3 owner_agent.py --job job_xxx --device mps              # Step 1–6: process originals/
  python3 owner_agent.py --job job_xxx --feedback --device mps   # Step 8–9: retry per audit feedback
"""
import argparse, json, os, shutil, sys, time
from collections import Counter
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logo_finder as lf
import run_bulk as rb
import product_preserve_clean as ppc

MAX_RETRY = 2
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

# logo_finder presence → manifest watermark_status
WM_STATUS = {
    "CONFIRMED_WATERMARK": "watermark_found", "LIKELY_WATERMARK": "watermark_found",
    "UNCERTAIN_WATERMARK": "uncertain_watermark", "NO_WATERMARK": "no_watermark",
    "UNSUITABLE_IMAGE": "unsuitable",
}
# Audit FAIL reasons we know how to act on (Step 8)
FAIL_REASONS = ("FAIL_RESIDUAL_WATERMARK", "FAIL_VISIBLE_PATCH", "FAIL_PRODUCT_DAMAGE",
                "FAIL_PROTECTED_TEXT_DAMAGE", "FAIL_UNCERTAIN")


def job_dirs(job):
    d = {k: os.path.join(job, k) for k in ("originals", "finals", "masks", "rejected", "logs")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    d["manifest"] = os.path.join(job, "owner_manifest.jsonl")
    d["feedback"] = os.path.join(job, "audit_feedback.jsonl")
    return d


# ── Step 3: pick the safest repair for where the mark sits ──────────────────────
def repair_strategy(roi):
    """ROI → (method, mask_key). 'reject' protects product text. Stroke-level on product
    surfaces / flex / complex detail (no big rectangle); soft inpaint on plain backgrounds."""
    if roi == "text_or_label_area":
        return "reject", None                                  # E — never remove product text
    if roi in ("plain_white", "near_white"):
        return "repair", "soft_mask"                           # A — soft inpaint / clean cover
    if roi in ("low_texture_background", "simple_product_surface"):
        return "repair", "soft_mask"                           # B — texture-aware inpaint
    if roi in ("thin_flex_cable", "complex_product_detail"):
        return "repair", "stroke_mask"                         # D — very conservative
    return "repair", "stroke_mask"                             # C — product surface, stroke-level


def _mark_present(bgr, reader):
    """Light internal quality gate — is a watermark still findable on this image?"""
    return lf.find(bgr, reader)["presence"] in ("CONFIRMED_WATERMARK", "LIKELY_WATERMARK")


def _mark_mostly_on_white(bgr):
    """The watermark band is predominantly white background — i.e. the mark sits mostly on white
    with thin/discrete product structure in it (flex cables, screws, components, pointer arrows).
    These are exactly the cases LaMa inpaint DESTROYS, so route them to structure-preserving
    recovery. Large product surfaces filling the band (screens/plates) stay on the inpaint path."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H = g.shape[0]
    band = g[int(0.42 * H):int(0.56 * H)]
    return band.size > 0 and float((band > 238).mean()) >= 0.55


# ── Destructive-clean guard ──────────────────────────────────────────────────────
# The sunsky mark is a LIGHT semi-transparent overlay, so a faithful removal only
# lightens-or-equals the band. A clean that DARKENS a mid/light surface (solid dark
# fill / re-darken splatter) or WIPES a mid-light surface to near-white has damaged
# the product, not removed a mark. Calibrated on job_50: damaged finals score >= 0.106,
# clean finals <= 0.051 — threshold 0.07 separates with ~2x margin (both colour-robust).
DESTRUCTIVE_CLEAN_THR = 0.07


def _destructive_score(orig, final):
    """max(dark-fill frac, white-wipe frac) over the watermark band (0.40–0.58 H)."""
    if orig is None or final is None or orig.shape != final.shape:
        return 0.0
    H = orig.shape[0]
    go = cv2.cvtColor(orig[int(0.40 * H):int(0.58 * H)], cv2.COLOR_BGR2GRAY).astype(np.int16)
    gf = cv2.cvtColor(final[int(0.40 * H):int(0.58 * H)], cv2.COLOR_BGR2GRAY).astype(np.int16)
    if go.size == 0:
        return 0.0
    d = go - gf
    dark_fill = float(((d >= 25) & (go >= 70) & (gf < 100)).mean())     # mid/light pushed dark
    white_wipe = float(((gf >= 248) & (go > 140) & (go < 240)).mean())  # mid-light wiped to white
    return max(dark_fill, white_wipe)


def _added_localized_damage(orig, tpl):
    """The template escalation produced HIGH-CONTRAST changes (>70 grey levels, e.g. black<->white)
    = it inpainted a product EDGE (camera lens rim, screw hole), not the faint low-contrast mark.
    The mark is a light overlay, so its removal is always a SMALL change — a big change is damage."""
    if orig is None or tpl is None or orig.shape != tpl.shape:
        return True
    go = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gt = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return int((np.abs(go - gt) > 70).sum()) > 400


# ── Steps 1–5 for one image. Returns (record, image_or_'COPY'_or_None, dest, mask) ──
def process_one(img_path, job, reader, attempt=0, fail_reason=None, retry_box=None):
    iid = os.path.splitext(os.path.basename(img_path))[0]
    rec = {"id": iid, "original": os.path.relpath(img_path, job), "final": None,
           "owner_status": None, "method": None, "watermark_status": None,
           "notes": "", "attempt": attempt}

    # Step 1: load + validate
    bgr = rb._imread(img_path)
    if bgr is None or min(bgr.shape[:2]) < 8:
        rec.update(owner_status="auto_rejected", method="reject",
                   watermark_status="unsuitable", notes="unreadable_image")
        return rec, None, "reject", None

    # Step 2: detect (recall-oriented ensemble)
    res = lf.find(bgr, reader)
    ws = WM_STATUS.get(res["presence"], "no_watermark")
    rec["watermark_status"] = ws

    # ── Retry with audit hint: region-targeted residual re-inpaint ──────────────
    # On a FAIL_RESIDUAL_WATERMARK retry the Audit reports WHERE it still sees the
    # mark (retry_box, full-image px == original dims). The first pass leaked here
    # precisely because the detector under-read this mark, so the retry must NOT
    # depend on re-detection — note we re-inpaint even when lf.find now reports
    # no_watermark (it would otherwise copy the dirty original straight through).
    # Union the hint with any fresh detection, pad wide, and re-inpaint through
    # clean_image's reuse-box path (skips re-detect, runs its aggressive 2nd pass on
    # residual). Audit re-audits the result — a surviving mark still REJECTs and
    # MAX_RETRY still caps. Retry-path only: retry_box is None on the first pass, so
    # first-pass finals are byte-identical.
    if retry_box is not None and fail_reason == "FAIL_RESIDUAL_WATERMARK":
        det = res["candidates"][0]["_box"] if res.get("candidates") else None
        base = rb.union_bbox([tuple(det), tuple(retry_box)]) if det else list(retry_box)
        bbox_p = rb.pad_bbox(base, bgr.shape, pad_frac_x=0.30, pad_frac_y=0.50, pad_px_min=12)
        cleaned, cinfo = rb.clean_image(bgr, bbox_p=bbox_p)    # bbox_p passed → no re-detect
        if cleaned is None:                                    # box was passed, so this is defensive
            cleaned = bgr
        rec.update(owner_status="cleaned", method="repair_hint",
                   notes=f"residual retry @ audit hint {[int(v) for v in retry_box]} "
                         f"-> bbox_p={bbox_p} residual_after={cinfo.get('residual_after')}")
        rec["retry_hint_box"] = [int(v) for v in retry_box]
        rec["bbox_padded"] = list(bbox_p)
        return rec, cleaned, "final", None

    if ws == "no_watermark":
        rec.update(owner_status="copied_no_watermark", method="copy",
                   notes="no visible watermark detected")
        return rec, "COPY", "final", None                      # copy original bytes through
    if ws == "unsuitable" or not res["candidates"]:
        rec.update(owner_status="auto_rejected", method="reject", notes="unsuitable image")
        return rec, None, "reject", None

    cand = res["candidates"][0]
    roi, box = cand["roi_class"], cand["_box"]

    # Step 8 feedback overrides (only on retry)
    if fail_reason == "FAIL_PROTECTED_TEXT_DAMAGE":
        rec.update(owner_status="auto_rejected", method="reject",
                   notes="protected product text — not retried aggressively")
        return rec, None, "reject", None

    method, mask_key = repair_strategy(roi)
    if method == "reject":                                      # protected text area
        rec.update(owner_status="auto_rejected", method="reject",
                   notes=f"watermark overlaps protected area ({roi})")
        return rec, None, "reject", None

    if fail_reason == "FAIL_RESIDUAL_WATERMARK":
        method, mask_key = "cover", "cover_mask"                # stronger
    elif fail_reason == "FAIL_VISIBLE_PATCH":
        mask_key = "stroke_mask"                                # softer / smaller footprint
    elif fail_reason == "FAIL_PRODUCT_DAMAGE":
        if roi in ("thin_flex_cable", "complex_product_detail"):
            rec.update(owner_status="auto_rejected", method="reject",
                       notes="product damage risk — not safely repairable")
            return rec, None, "reject", None
        mask_key = "stroke_mask"

    # Step 4: generate the final.
    masks = lf._masks(bgr.shape, box, roi)
    # White-background SKUs: RECOVER the surface under the semi-transparent mark instead of
    # inpainting it. LaMa inpaint fills/hallucinates and destroys thin product geometry sitting in
    # the mark band — flex cables broken, gold contacts erased, screws/components smeared, pointer
    # arrowheads removed (the canonical_band damage class). product_preserve_clean whitens the mark
    # on white (solid shapes protected) and re-darkens it where it crosses a dark cable — no inpaint,
    # structure preserved. Fall back to inpaint only if a residual survives; non-white / large-surface
    # SKUs and retries keep the inpaint path unchanged.
    # Primary removal is GLYPH-TIGHT inpaint (ppc.glyph_clean): it removes only the thin mark
    # strokes, so product structure is preserved on ALL surfaces (white, dark flex, metallic,
    # screens, colour bezels). The old LaMa box-inpaint filled a rectangle under the band and
    # DESTROYED the product there (dark-band / splatter damage class) — it is removed from the
    # publish path. White-background SKUs keep the fast structure-preserve whiten, with glyph as
    # the residual fallback.
    if not fail_reason and _mark_mostly_on_white(bgr):
        final = ppc.clean(bgr)
        method = "structure_preserve"
        # Fall back to glyph if structure_preserve left a mark OR damaged the product: whiten_on_white
        # wipes a light-coloured product (e.g. a yellow cover), and the dark_cable re-tone splatters a
        # flex cable. glyph-tight inpaint is safe on those, so prefer it whenever structure_preserve
        # isn't both clean AND non-destructive.
        if _mark_present(final, reader) or _destructive_score(bgr, final) >= DESTRUCTIVE_CLEAN_THR:
            final = ppc.glyph_clean(bgr, box)
            method = "glyph_inpaint"
    else:
        final = ppc.glyph_clean(bgr, box)
        method = "glyph_inpaint"

    # Residual escalation — detector-first, then a GUARDED template pass:
    #  (a) a wider/looser DETECTION pass catches high-contrast residual safely;
    #  (b) only if a mark STILL remains (low-contrast surface, bg≈mark grey) do we try the known-
    #      matte TEMPLATE pass, accepted ONLY if it adds no band damage AND no high-contrast edge
    #      damage — the matte can misalign onto a product feature (lens/hole) and inpaint it. If it
    #      would damage, we keep the safe detection result. We never box-fill to chase a mark.
    if _mark_present(final, reader):
        wide = ppc.glyph_clean(bgr, box, pad=0.30, thr=5)
        if not _mark_present(wide, reader):
            final, method = wide, "glyph_inpaint"
        else:
            tpl = ppc.glyph_clean(bgr, box, pad=0.30, thr=5, use_template=True)
            if (_destructive_score(bgr, tpl) < DESTRUCTIVE_CLEAN_THR
                    and not _added_localized_damage(bgr, tpl)):
                final, method = tpl, "glyph_template"
            else:
                final, method = wide, "glyph_inpaint"
        if _mark_present(final, reader):
            rec.update(owner_status="auto_rejected", method="reject",
                       notes="residual watermark survived glyph clean — held")
            return rec, None, "reject", None

    rec.update(owner_status="cleaned", method=method)
    return rec, final, "final", masks[mask_key]


def _save(rec, img, dest, mask, dirs, src_path, job):
    """Step 5: write final / rejected + optional mask; fill rec['final']."""
    fname = os.path.basename(src_path)
    if dest == "final":
        outp = os.path.join(dirs["finals"], fname)
        if isinstance(img, str) and img == "COPY":
            shutil.copy2(src_path, outp)                       # no re-encode for clean images
        else:
            orig = rb._imread(src_path)
            swept = ppc.band_residual_cleanup(img)            # flat-band faint-mark sweep (kill faint)
            # The sweep can re-damage a textured/coloured surface that glyph_clean already cleaned;
            # if it introduces destructive change the pre-sweep image did NOT have, keep the safe
            # pre-sweep (glyph) image instead of the swept one.
            if _destructive_score(orig, swept) >= DESTRUCTIVE_CLEAN_THR > _destructive_score(orig, img):
                swept = img
            img = swept
            if _destructive_score(orig, img) >= DESTRUCTIVE_CLEAN_THR:   # DAMAGE GUARD on published bytes:
                # the clean DARKENED / WIPED the product band (solid fill / splatter / white wipe).
                # Hold the image rather than publish damage — conservative publish; a held SKU is
                # recoverable by a texture-safe re-clean, a black-blobbed product is not.
                shutil.copy2(src_path, os.path.join(dirs["rejected"], fname))
                rec.update(owner_status="auto_rejected", method="reject", final=None,
                           notes=(rec.get("notes", "") +
                                  " | destructive clean suppressed — held for texture-safe re-clean").lstrip(" |"))
                return
            rb._imwrite_atomic(outp, img)                     # mark the finder is blind to before publish
        rec["final"] = os.path.relpath(outp, job)
        if mask is not None:
            cv2.imwrite(os.path.join(dirs["masks"], rec["id"] + ".png"), mask)
    else:                                                      # reject → keep a copy of the original
        shutil.copy2(src_path, os.path.join(dirs["rejected"], fname))
        rec["final"] = None


def _manifest_latest(path):
    """id → latest record (resume / retry state)."""
    latest = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    latest[r["id"]] = r
                except Exception:
                    pass
    return latest


def run_job(job, device, limit=0):
    dirs = job_dirs(job)
    rb._init_worker(device)
    reader = rb._get_reader()
    done = set(_manifest_latest(dirs["manifest"]).keys())
    files = [f for f in sorted(os.listdir(dirs["originals"])) if f.lower().endswith(IMG_EXT)]
    if limit:
        files = files[:limit]
    mf = open(dirs["manifest"], "a", buffering=1)
    tally = Counter()
    t0 = time.time()
    for i, f in enumerate(files, 1):
        iid = os.path.splitext(f)[0]
        if iid in done:
            continue
        src = os.path.join(dirs["originals"], f)
        rec, img, dest, mask = process_one(src, job, reader)
        _save(rec, img, dest, mask, dirs, src, job)
        mf.write(json.dumps(rec) + "\n")
        tally[rec["owner_status"]] += 1
        if i % 20 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {i/(time.time()-t0):.2f}/s {dict(tally)}", flush=True)
    print("OWNER DONE:", dict(tally))
    return tally


def handle_feedback(job, device):
    """Step 8–9: read audit_feedback.jsonl ({id, verdict}); retry FAILs (≤2), else auto_reject."""
    dirs = job_dirs(job)
    rb._init_worker(device)
    reader = rb._get_reader()
    latest = _manifest_latest(dirs["manifest"])
    if not os.path.exists(dirs["feedback"]):
        print("no audit_feedback.jsonl"); return
    mf = open(dirs["manifest"], "a", buffering=1)
    tally = Counter()
    for line in open(dirs["feedback"]):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        iid, verdict = item["id"], item.get("verdict", "")
        if verdict == "PASS" or verdict not in FAIL_REASONS:
            continue
        retry_box = item.get("retry_box")                      # audit hint (residual retries only); None otherwise
        prev = latest.get(iid, {})
        attempt = int(prev.get("attempt", 0)) + 1
        src = os.path.join(job, prev.get("original", os.path.join("originals", iid + ".jpg")))
        if attempt > MAX_RETRY:                                # Step 9
            rec = {"id": iid, "original": prev.get("original"), "final": None,
                   "owner_status": "auto_rejected", "method": "reject",
                   "watermark_status": prev.get("watermark_status"),
                   "notes": f"max retries ({MAX_RETRY}) reached after {verdict}", "attempt": attempt}
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dirs["rejected"], os.path.basename(src)))
            mf.write(json.dumps(rec) + "\n"); tally["auto_rejected"] += 1
            continue
        rec, img, dest, mask = process_one(src, job, reader, attempt=attempt,
                                           fail_reason=verdict, retry_box=retry_box)
        _save(rec, img, dest, mask, dirs, src, job)
        mf.write(json.dumps(rec) + "\n")
        tally[rec["owner_status"]] += 1
    print("FEEDBACK HANDLED:", dict(tally))
    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="job dir (contains originals/)")
    ap.add_argument("--feedback", action="store_true", help="retry per audit_feedback.jsonl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.feedback:
        handle_feedback(args.job, args.device)
    else:
        run_job(args.job, args.device, args.limit)


if __name__ == "__main__":
    main()
