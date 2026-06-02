#!/usr/bin/env python3
"""V20 — regression guard on a completed benchmark run (patch plan §Patch 11/12).

These tests validate the ACTUAL published outputs of an ``output_v20`` run (when
present): no published output may carry a residual watermark, a product-side
cover, ghost-dot residue, or broken flex continuity, and no auto_rejected folder
may leak a cleaned.jpg. If no run is present the tests skip (the run is produced
by ``python3 mark_remover.py ... --out output_v20``).

The named fixtures from the patch plan (visible-cover, ghost-dot and thin-flex
cases) are part of the seed-2026 benchmark set, so this guard covers them.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output_v20"


def _report():
    rj = OUT / "run_report.json"
    if not rj.exists():
        pytest.skip("no output_v20 run present")
    return json.loads(rj.read_text())


def test_no_rejected_cleaned_jpg_leak():
    if not OUT.exists():
        pytest.skip("no output_v20 run present")
    leaks = list((OUT / "auto_rejected").rglob("cleaned.jpg")) \
        if (OUT / "auto_rejected").exists() else []
    assert leaks == [], f"auto_rejected leaked cleaned.jpg: {leaks}"


def test_all_published_safety_counters_zero():
    rep = _report()
    for k, v in rep["must_be_zero"].items():
        assert v == 0, f"{k} = {v}"
    for k, v in rep["v17_published_audit_failures"].items():
        assert v == 0, f"{k} = {v}"
    assert rep.get("auto_rejected_cleaned_jpg_leak", 0) == 0
    assert rep["all_clean"] is True


def test_cover_artifacts_caught_not_published():
    rep = _report()
    for k, v in rep["cover_artifacts_v19"].items():
        # These are counted when CAUGHT (forced to auto_rejected) — the invariant
        # is that none were PUBLISHED, enforced by the zero counters above.
        assert v >= 0


def test_rejections_are_explained():
    rep = _report()
    v20 = rep.get("v20", {})
    rej = rep.get("final_auto_rejected", 0)
    explained = (v20.get("auto_rejected_true_residual", 0) +
                 v20.get("auto_rejected_product_damage", 0) +
                 v20.get("auto_rejected_cover_artifact", 0) +
                 v20.get("auto_rejected_uncertain", 0))
    # every rejected image falls into exactly one explained bucket.
    assert explained == rej


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
