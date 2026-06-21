#!/usr/bin/env python3
"""cli.py — run the six-agent in-process pipeline on a single image.

    python3 cli.py IMG.jpg --device mps [--heuristic] [--out clean.jpg]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import build_default_pipeline
from shared.contract import Status


def main():
    ap = argparse.ArgumentParser(description="Watermark-removal pipeline (single image)")
    ap.add_argument("image", help="path to one image")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", help="write the cleaned image here (PASS only)")
    ap.add_argument("--heuristic", action="store_true",
                    help="use the dependency-light heuristic Validator instead of audit.py")
    args = ap.parse_args()

    import cv2
    bgr = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"cannot read {args.image}")

    reader = None
    try:                                          # optional: full OCR recall if easyocr is present
        import easyocr
        reader = easyocr.Reader(["en"], gpu=(args.device != "cpu"))
    except Exception:
        pass

    pipe = build_default_pipeline(device=args.device, reader=reader, use_audit=not args.heuristic)
    outcome = pipe.process(bgr)
    if outcome.status is Status.PASS and args.out:
        cv2.imwrite(args.out, outcome.image)
    print(json.dumps(outcome.as_dict(), indent=2))


if __name__ == "__main__":
    main()
