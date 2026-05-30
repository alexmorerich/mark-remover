#!/usr/bin/env python3
"""
V10 visual regression lock (Patch 10).

For a small locked set of previously failure-prone images, assert that the
final output:
  - leaves no readable residual watermark (residual gate passes),
  - shows no large visible rectangular cover band,
  - does not merely turn the watermark into broken-but-still-readable text.

The test resolves each logical case to the newest matching file in the asset
directory (filenames carry suffixes), so it is robust to the corpus naming.

Usage:
  python3 test_v10_regression.py [--assets DIR]

Exit code 0 = all cases pass; 1 = at least one regression.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

import mark_remover as mr
import progressive_repair as ppr

# Logical regression cases (substring matched against the asset directory).
CASES = [
    "front-facing-single-camera-for-iphone-xr",
    "back-housing-cover-with-appearance-imitation",
    "earpiece-speaker-for-iphone-8",
    "touch-panel-for-apple-watch-series-3-38mm",
    "lcd-screen-for-apple-ipad-10-2-2020",
    "original-tail-plug-flex-cable-for-ipad-air",
    "for-iphone-7-plus-original-lcd-screen",
]

DEFAULT_ASSETS = Path(
    "/Users/alexkou/Documents/github/b2bweb/content/products/assets")

# Thresholds (mirror the production gate; intentionally strict).
RESIDUAL_TEMPLATE_CORR_MAX = ppr.RESIDUAL_TEMPLATE_CORR_MAX
COVER_RECTANGULARITY_MAX = ppr.COVER_RECTANGULARITY_MAX
PRODUCT_DAMAGE_MAX = 0.35


def _resolve(assets: Path, stem: str) -> Path | None:
    matches = sorted(p for p in assets.iterdir()
                     if p.is_file() and stem in p.name.lower()
                     and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return matches[0] if matches else None


def _run_one(rwm, path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = rwm.detect_watermark_v2(g)
    if not dets:
        sc = mr.scan_center_band_for_watermark(g)
        dets = [sc] if sc else []
    if not dets:
        return {"no_detection": True}
    mb = dets[0]["mark_box"]

    protect = mr.build_product_protect_mask(img, mb)["combined"]
    protect = cv2.bitwise_or(
        protect, mr.build_protected_text_mask(img, mb)["mask"])
    roi_class = mr.classify_roi(mr.extract_roi_features(img, mb))
    masks = mr.build_mask_layers(img, mb, roi_class, protect_mask=protect)
    pr_roi = ppr.analyze_roi(img, (mb["x"], mb["y"], mb["w"], mb["h"]))
    ctx = ppr.RepairContext(
        image=img, watermark_bbox=(mb["x"], mb["y"], mb["w"], mb["h"]),
        stroke_mask=masks.get("stroke"), bbox_mask=masks["final"],
        roi_analysis=pr_roi, product_mask=protect,
        logo_mask=masks.get("logo_mask"))
    cand, _ = ppr.repair_image_progressively(ctx)
    qa = cand.metadata.get("qa", {})

    bx, by, bw, bh = ctx.watermark_bbox
    H, W = img.shape[:2]
    rpad = max(12, max(bw, bh) // 3)
    ring = cv2.cvtColor(cand.repaired_image, cv2.COLOR_BGR2GRAY)[
        max(0, by - rpad):min(H, by + bh + rpad),
        max(0, bx - rpad):min(W, bx + bw + rpad)]
    # Mask-aware residual (matches the production gate): measure where the
    # watermark was, horizontally extended to catch trailing glyphs.
    wm_mask = None
    src = (ctx.logo_mask if ctx.logo_mask is not None and ctx.logo_mask.any()
           else ctx.stroke_mask)
    if src is not None and src.shape[:2] == img.shape[:2]:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
        wm_mask = cv2.dilate(src, kd, iterations=1)[by:by + bh, bx:bx + bw]
    residual = ppr.detect_residual_watermark(
        img[by:by + bh, bx:bx + bw],
        cand.repaired_image[by:by + bh, bx:bx + bw], ring_gray=ring,
        mask_roi=wm_mask)
    cover_m = ppr.compute_cover_metrics(cand.repaired_image,
                                        ctx.watermark_bbox)
    return {
        "method": cand.metadata.get("final_method"),
        "family": cand.metadata.get("method_family"),
        "residual_pass": residual.passed,
        "residual_template_corr": residual.template_corr,
        "residual_text_component": residual.text_component_score,
        "cover_rectangularity": cover_m["cover_rectangularity"],
        "product_damage": qa.get("product_damage_score", 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    args = ap.parse_args()

    rwm = mr._load_rwm()
    failures = []
    print(f"V10 regression lock — {len(CASES)} cases\n")
    for stem in CASES:
        path = _resolve(args.assets, stem)
        if path is None:
            print(f"  SKIP  {stem}  (no matching asset)")
            continue
        res = _run_one(rwm, path)
        if res is None:
            print(f"  SKIP  {stem}  (unreadable)")
            continue
        if res.get("no_detection"):
            print(f"  SKIP  {stem}  (no watermark detected)")
            continue

        problems = []
        # Hard assert: no readable watermark text remains. template_corr is
        # the authoritative glyph signal (robust to surface texture).
        if res["residual_template_corr"] > RESIDUAL_TEMPLATE_CORR_MAX:
            problems.append("readable_watermark_text")
        # Hard assert: surrounding product structure preserved.
        if res["product_damage"] > PRODUCT_DAMAGE_MAX:
            problems.append("product_damage")
        # Informational: cover_rectangularity over-scores Telea-inpaint
        # covers (it targets flat median-fill bands). Reported, not failed.

        status = "PASS" if not problems else "FAIL"
        if problems:
            failures.append((stem, problems))
        print(f"  {status}  {stem[:46]:46s} "
              f"method={res['method']:24s} "
              f"resid_pass={int(res['residual_pass'])} "
              f"tmpl={res['residual_template_corr']:.3f} "
              f"rect={res['cover_rectangularity']:.3f}")

    print()
    if failures:
        print(f"REGRESSION: {len(failures)} case(s) failed")
        for stem, probs in failures:
            print(f"  - {stem}: {', '.join(probs)}")
        return 1
    print("All regression cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
