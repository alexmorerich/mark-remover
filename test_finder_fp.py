#!/usr/bin/env python3
"""Regression test for the watermark-finder OCR false-positive fix (logo_finder.py).

Locks in two guarantees:
  1. _text_grade no longer mis-grades common English product captions as watermark fragments
     (the `[o0]nne` -> "C-onne-ct" bug). Real sunsky/online fragments still grade STRONG.
  2. _is_product_caption DROPs a confidently-read, off-centre, uncorroborated non-STRONG OCR
     box (a product caption) while letting real marks through three ways: STRONG text, OR a
     near-centre box, OR matte+dot-chain corroboration.

Pure-function tests — no OCR / torch, deterministic.
  python3 test_finder_fp.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logo_finder as lf

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ── 1. text grading ────────────────────────────────────────────────────────────
# captions that USED to mis-grade (and other common product words) must stay WEAK (<=0.25)
for cap in ["Connect The Cable", "Dust-proof Earspeaker Mesh", "Check The Frame",
            "3D Touch Function Test", "Touch Test", "Front Camera Ring", "Exclusive Glue",
            "install the 2 connectors.", "Tear off the waterproof sticker"]:
    s, _ = lf._text_grade(cap)
    check(s <= 0.25, f"caption stays weak ({s:.2f}): {cap!r}")

# real watermark fragments must still grade STRONG (recall preserved)
for frag in ["sunsky-online.com", "online.com", "sunsky-online", "www.sunsky-online.com", ".com"]:
    s, _ = lf._text_grade(frag)
    check(s >= 1.0, f"mark fragment stays STRONG ({s:.2f}): {frag!r}")

# the specific regression: "Connect" must not reach the MED pattern (was 0.55 via [o0]nne)
s_connect, _ = lf._text_grade("Connect The Cable")
check(s_connect < 0.55, f"'Connect The Cable' below MED: {s_connect:.2f}")


# ── 2. product-caption guard ─────────────────────────────────────────────────────
def ev(center, tpl=0.30, dot=0.30):
    return {"center_prior_score": center, "template_score": tpl, "dot_chain_score": dot}


# off-centre, confident, non-STRONG  -> DROPPED (the false positive)
check(lf._is_product_caption("ocr", 0.25, 0.85, ev(0.00)), "off-centre WEAK caption dropped")
check(lf._is_product_caption("ocr", 0.55, 0.80, ev(0.05)), "off-centre MED caption dropped")
# real marks bypass three ways — must NOT be dropped:
check(not lf._is_product_caption("ocr", 1.0, 0.90, ev(0.00)), "STRONG text kept")
check(not lf._is_product_caption("ocr", 0.25, 0.90, ev(0.60)), "near-centre box kept")
check(not lf._is_product_caption("ocr", 0.25, 0.90, ev(0.00, tpl=0.50, dot=0.40)), "matte+rhythm corroborated kept")
# a low-confidence read is not a 'clear caption' — keep it (conservative for recall)
check(not lf._is_product_caption("ocr", 0.25, 0.30, ev(0.00)), "low-confidence read kept")
# shape-led (canon-matte) candidates are never text captions
check(not lf._is_product_caption("canon", 0.0, 0.50, ev(0.00)), "canon-matte candidate kept")

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED")
    sys.exit(1)
print("ALL PASSED")
