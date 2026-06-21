#!/usr/bin/env python3
"""integration/e2e.py — end-to-end check the integrator runs AFTER merging agent branches.

Builds the assembled pipeline and runs it over a folder of images, tallying terminal states
(pass / skip / manual_review). This is the cross-agent smoke that contract tests (per-agent
boundaries) do not cover — proof the six merged pieces actually compose on real inputs.

    python3 -m integration.e2e IMAGES_DIR [--n 20] [--device cpu] [--heuristic]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration import build_default_pipeline
from shared.contract import Status


def run(images_dir: str, n: int = 20, device: str = "cpu", use_audit: bool = True) -> dict:
    import cv2
    paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                   glob.glob(os.path.join(images_dir, "*.png")))[:n]
    reader = None
    if use_audit or True:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=(device != "cpu"))
        except Exception:
            pass
    pipe = build_default_pipeline(device=device, reader=reader, use_audit=use_audit)

    tally, rows = Counter(), []
    for p in paths:
        bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            tally["unreadable"] += 1
            continue
        out = pipe.process(bgr)
        tally[out.status.value] += 1
        rows.append({"image": os.path.basename(p), "status": out.status.value,
                     "attempts": out.attempts})
    return {"total": len(paths), "tally": dict(tally), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="integration end-to-end smoke over a folder")
    ap.add_argument("images_dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--heuristic", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(args.images_dir, args.n, args.device, use_audit=not args.heuristic), indent=2))


if __name__ == "__main__":
    main()
