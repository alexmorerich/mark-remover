#!/usr/bin/env python3
"""V19 — Solve the Sunsky watermark alpha asset (patch plan §1.3).

Produces a reproducible alpha asset used by ``sunsky_reverse_alpha``:

    assets/sunsky_alpha.png        canonical per-pixel alpha (uint8, 0..255 = 0..1)
    assets/sunsky_alpha_meta.json  logo colour, peak alpha, provenance
    assets/sunsky_logo_color.json  solved overlay colour (BGR)

Two modes (patch plan §1.3):

  Mode A (--mode A)  controlled-capture solve. Given the watermark rendered over
                     known flat backgrounds (black / gray / white), solve
                         alpha = (I - B) / (L - B)
                     per pixel. Use when controlled captures are available.

  Mode B (--mode B)  empirical catalog solve (DEFAULT). Use real product images
                     whose watermark falls on a plain near-white background:
                       1. detect the watermark box,
                       2. estimate the local clean background B (grayscale max
                          filter — the text is darker than white paper),
                       3. invert the blend with a fixed light-gray logo prior to
                          get a per-pixel alpha,
                       4. align every crop to a canonical glyph grid and
                          median-combine,
                       5. keep the full halo down to alpha >= 0.02 and drop tiny
                          specks with connected-component area filtering,
                       6. solve the logo colour from the high-alpha pixels.

The alpha map keeps the FULL ``sunsky-online.com`` line including the trailing
``.com`` glyphs and the low-alpha halo, exactly as the reverse-alpha engine
expects.

Run:
    python3 scripts/sunsky_alpha_solve.py --mode B --assets bench_assets --max 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ASSET_DIR = REPO_DIR / "assets"

sys.path.insert(0, str(REPO_DIR))
import detector  # noqa: E402

CANON_H, CANON_W = 36, 240          # canonical glyph grid (aspect of the line)
LOGO_PRIOR = 190.0                  # light-gray overlay prior (luma)
DENOM_MIN = 18.0                    # B - L floor for a stable inversion
MAX_ALPHA = 0.80
HALO_MIN_ALPHA = 0.02               # keep halo down to this
SPECK_MIN_AREA = 3                  # connected-component speck filter (px)


def _template():
    tpls = detector.get_templates()
    return tpls[0] if tpls else cv2.imread(str(detector.TEMPLATE_PATH),
                                           cv2.IMREAD_GRAYSCALE)


def _detect_box(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        dets = detector.detect_template_image(gray, _template())
    except Exception:
        return None
    if not dets:
        return None
    mb = dets[0].get("mark_box")
    if not mb:
        return None
    return (mb["x"], mb["y"], mb["w"], mb["h"])


def _alpha_from_white_bg(img_bgr, box):
    """Per-pixel alpha from a watermark on a near-white background. Returns a
    canonical-size float alpha crop, or ``None`` if the band is not white bg."""
    bx, by, bw, bh = box
    H, W = img_bgr.shape[:2]
    bx, by = max(0, bx), max(0, by)
    bw, bh = min(bw, W - bx), min(bh, H - by)
    if bw < 12 or bh < 6:
        return None
    band = img_bgr[by:by + bh, bx:bx + bw]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Require a bright, fairly uniform background (plain product backdrop).
    if float(np.median(gray)) < 205 or float(gray.std()) > 42:
        return None
    # Background estimate: the text is darker than the paper, so a grayscale
    # max-filter (dilation) reconstructs the clean background under the glyphs.
    ksize = max(3, (bh // 2) | 1)
    B = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
    B = cv2.GaussianBlur(B, (0, 0), 1.5)
    denom = np.maximum(B - LOGO_PRIOR, DENOM_MIN)
    alpha = np.clip((B - gray) / denom, 0.0, MAX_ALPHA)
    return cv2.resize(alpha, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA), \
        band, B


def _logo_color_from(band, B, alpha_canon):
    """Solve overlay colour L from I = a*L + (1-a)*B at high-alpha pixels."""
    bh, bw = band.shape[:2]
    a = cv2.resize(alpha_canon, (bw, bh), interpolation=cv2.INTER_AREA)
    hi = a > 0.30
    if int(hi.sum()) < 8:
        return None
    Bc = cv2.cvtColor(
        cv2.merge([B, B, B]).astype(np.uint8), cv2.COLOR_BGR2BGR
    ) if False else None
    # Per-channel: L = (I - (1-a)*B_gray) / a, using gray B as background proxy.
    a3 = a[:, :, None]
    Bg = np.repeat(B[:, :, None], 3, axis=2)
    I = band.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = (I - (1.0 - a3) * Bg) / np.maximum(a3, 1e-3)
    L = L[hi]
    L = L[np.isfinite(L).all(axis=1)]
    if L.size == 0:
        return None
    return [float(np.clip(np.median(L[:, 0]), 0, 255)),
            float(np.clip(np.median(L[:, 1]), 0, 255)),
            float(np.clip(np.median(L[:, 2]), 0, 255))]


def _template_fallback_alpha():
    """Derive a canonical alpha shape from the synthesized glyph template when
    no empirical white-bg crops are available."""
    tpl = _template().astype(np.float32)
    tpl = cv2.resize(tpl, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
    # The template draws dark glyphs on a light field; coverage = darkness.
    cov = (tpl.max() - tpl) / (tpl.max() - tpl.min() + 1e-6)
    return np.clip(cov, 0.0, 1.0) * 0.42


def _finalize_alpha(alpha):
    """Keep halo down to HALO_MIN_ALPHA, drop tiny specks."""
    a = alpha.copy()
    a[a < HALO_MIN_ALPHA] = 0.0
    mask = (a > 0.05).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < SPECK_MIN_AREA:
            a[labels == i] = 0.0
    return a


def solve_mode_b(assets_dir: Path, max_images: int):
    files = sorted([p for p in assets_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    alphas = []
    logos = []
    used = 0
    for p in files:
        if used >= max_images:
            break
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        box = _detect_box(img)
        if box is None:
            continue
        out = _alpha_from_white_bg(img, box)
        if out is None:
            continue
        alpha_canon, band, B = out
        if float(alpha_canon.max()) < 0.05:
            continue
        alphas.append(alpha_canon)
        lc = _logo_color_from(band, B, alpha_canon)
        if lc is not None:
            logos.append(lc)
        used += 1
        print(f"  + {p.name}: box={box} peak_alpha={alpha_canon.max():.3f}")

    if len(alphas) >= 3:
        stack = np.stack(alphas, axis=0)
        alpha = np.median(stack, axis=0).astype(np.float32)
        source = f"empirical_median_of_{len(alphas)}"
    else:
        print(f"  only {len(alphas)} white-bg crops — using template fallback")
        alpha = _template_fallback_alpha()
        source = "template_fallback"

    alpha = _finalize_alpha(alpha)
    if logos:
        logo_bgr = [float(np.median([l[c] for l in logos])) for c in range(3)]
    else:
        logo_bgr = [LOGO_PRIOR, LOGO_PRIOR, LOGO_PRIOR]
    return alpha, logo_bgr, source, used


def solve_mode_a(captures_dir: Path):
    """Controlled-capture solve: alpha = (I - B) / (L - B). Expects files named
    *_black.*, *_gray.*, *_white.* of the SAME watermark over flat fields."""
    def _load(suffix):
        for p in captures_dir.iterdir():
            if suffix in p.stem.lower():
                return cv2.imread(str(p), cv2.IMREAD_COLOR)
        return None
    white = _load("white")
    black = _load("black")
    if white is None or black is None:
        raise SystemExit("Mode A needs *_white.* and *_black.* captures")
    Iw = white.astype(np.float32)
    Ib = black.astype(np.float32)
    # On black B=0 -> Ib = a*L ; on white B=255 -> Iw = a*L + (1-a)*255.
    # Subtract: Iw - Ib = (1-a)*255 -> a = 1 - (Iw - Ib)/255.
    a = 1.0 - np.clip((Iw - Ib) / 255.0, 0.0, 1.0)
    alpha = np.clip(a.mean(axis=2), 0.0, MAX_ALPHA)
    alpha = cv2.resize(alpha, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
    alpha = _finalize_alpha(alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = Ib / np.maximum(a, 1e-3)
    L = L[np.isfinite(L).all(axis=2)]
    logo_bgr = [float(np.clip(np.median(L[:, c]), 0, 255)) for c in range(3)] \
        if L.size else [LOGO_PRIOR] * 3
    return alpha, logo_bgr, "controlled_capture", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="B")
    ap.add_argument("--assets", type=Path, default=REPO_DIR / "bench_assets")
    ap.add_argument("--captures", type=Path, default=REPO_DIR / "captures")
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ASSET_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"sunsky alpha solve — mode {args.mode}")
    if args.mode == "A":
        alpha, logo_bgr, source, used = solve_mode_a(args.captures)
    else:
        alpha, logo_bgr, source, used = solve_mode_b(args.assets, args.max)

    png = (np.clip(alpha, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(args.out / "sunsky_alpha.png"), png)
    meta = {
        "source": source,
        "mode": args.mode,
        "n_images_used": used,
        "canonical_size": [CANON_H, CANON_W],
        "logo_bgr": logo_bgr,
        "peak_alpha": float(alpha.max()),
        "halo_min_alpha": HALO_MIN_ALPHA,
        "nonzero_px": int((alpha > 0).sum()),
        "logo_prior_luma": LOGO_PRIOR,
    }
    (args.out / "sunsky_alpha_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "sunsky_logo_color.json").write_text(
        json.dumps({"logo_bgr": logo_bgr}, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out/'sunsky_alpha.png'}")


if __name__ == "__main__":
    main()
