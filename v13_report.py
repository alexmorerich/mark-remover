#!/usr/bin/env python3
"""V13 — honesty / fidelity report.

Scans an output tree's per-image ``qa.json`` records (written by
``_write_terminal``) and tallies the V13 honesty counters. All bad-output
counters must be 0 for an acceptable release (patch plan section 14 / 19).

Usage:
    python3 v13_report.py output
Exit code is non-zero when any bad-output counter is > 0, so this also serves
as a CI gate.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import v13_gates

BAD_COUNTERS = list(v13_gates.HONESTY_COUNTERS)


def _iter_records(out_root: Path):
    for qa in out_root.rglob("qa.json"):
        try:
            yield json.loads(qa.read_text())
        except Exception:
            continue


def build_report(out_root: Path) -> dict:
    counters = Counter()
    status_counts = Counter()
    methods = set()
    n = 0
    demoted = 0
    for rec in _iter_records(out_root):
        status = rec.get("status")
        if status not in ("clean_repaired", "clean_covered"):
            status_counts[status] += 1
            continue
        n += 1
        status_counts[status] += 1
        methods.add(rec.get("v9_final_method", "unknown"))
        if rec.get("v13_demoted_from_repaired"):
            demoted += 1
        prefix = status
        if not rec.get("v13_dot_chain_pass", True):
            counters[f"{prefix}_with_dot_chain"] += 1
        if not (rec.get("v13_visible_patch_pass", True) and
                rec.get("v13_rectangular_band_pass", True) and
                rec.get("v13_polygon_patch_pass", True)):
            counters[f"{prefix}_with_visible_patch"] += 1
        if not rec.get("v13_product_damage_pass", True):
            counters[f"{prefix}_with_product_damage"] += 1
        if not rec.get("v13_silhouette_pass", True):
            counters[f"{prefix}_with_silhouette_damage"] += 1
        if not rec.get("v13_publish_ok", True):
            counters["final_publish_failures"] += 1

    report = {
        "version": "V13_FINAL_VISUAL_FIDELITY",
        "qa_schema_version": "v13",
        "final_visual_gate_version": "v13",
        "n_published": n,
        "clean_repaired": status_counts.get("clean_repaired", 0),
        "clean_covered": status_counts.get("clean_covered", 0),
        "no_watermark": status_counts.get("no_watermark", 0),
        "failed_io": status_counts.get("failed_io", 0),
        "unique_final_methods": len(methods),
        "v13_demoted_from_repaired": demoted,
        "honesty_counters": {k: int(counters.get(k, 0)) for k in BAD_COUNTERS},
    }
    report["all_clean"] = all(v == 0 for v in report["honesty_counters"].values())
    return report


def main(argv):
    out_root = Path(argv[1]) if len(argv) > 1 else Path("output")
    report = build_report(out_root)
    (out_root / "v13_honesty.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["all_clean"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
