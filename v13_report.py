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
import v14_patch

BAD_COUNTERS = list(v13_gates.HONESTY_COUNTERS)
# V14 — cover-side honesty counters (Section 7). All must be 0 in a release.
V14_COVER_COUNTERS = list(v14_patch.COVER_HONESTY_COUNTERS)

# V14 — release target metrics (Section 7).
V14_TARGETS = {
    "clean_repaired_min": 29,
    "clean_covered_max": 21,
    "final_adaptive_cover_max": 12,
    "opencv_telea_final_max": 10,
    "real_pixel_clone_min": 12,
    "segmented_micro_cover_min": 8,
    "near_miss_rescued_min": 5,
    "unique_final_methods_min": 12,
}


def _iter_records(out_root: Path):
    for qa in out_root.rglob("qa.json"):
        try:
            yield json.loads(qa.read_text())
        except Exception:
            continue


def build_report(out_root: Path) -> dict:
    counters = Counter()
    cover_counters = Counter()
    status_counts = Counter()
    method_use = Counter()
    methods = set()
    n = 0
    demoted = 0
    near_miss_rescued = 0
    segmented_micro_cover = 0
    for rec in _iter_records(out_root):
        status = rec.get("status")
        if status not in ("clean_repaired", "clean_covered"):
            status_counts[status] += 1
            continue
        n += 1
        status_counts[status] += 1
        fm = rec.get("v9_final_method", "unknown")
        methods.add(fm)
        method_use[fm] += 1
        if rec.get("v13_demoted_from_repaired"):
            demoted += 1
        if rec.get("v14_near_miss_rescued"):
            near_miss_rescued += 1
        cover_method = rec.get("v14_cover_method") or ""
        if cover_method.startswith("segmented_micro_cover") or \
                cover_method.startswith("stroke_band_micro_cover"):
            segmented_micro_cover += 1
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

        # V14 — cover-side honesty counters.
        if status == "clean_covered":
            if not (rec.get("v13_visible_patch_pass", True) and
                    rec.get("v13_rectangular_band_pass", True) and
                    rec.get("v13_polygon_patch_pass", True)):
                cover_counters["clean_covered_with_visible_patch"] += 1
            if not rec.get("v13_product_damage_pass", True):
                cover_counters["clean_covered_with_product_damage"] += 1
            if not rec.get("v13_silhouette_pass", True):
                cover_counters["clean_covered_with_silhouette_damage"] += 1
            if (rec.get("v14_boundary_jump") or 0.0) > 50.0:
                cover_counters["clean_covered_with_boundary_jump_gt_50"] += 1
            if rec.get("v14_used_full_bbox_on_product"):
                cover_counters["clean_covered_with_full_bbox_on_product"] += 1
            if not rec.get("v13_protected_text_pass", True):
                cover_counters["clean_covered_with_protected_text_loss"] += 1

    n_repaired = status_counts.get("clean_repaired", 0)
    n_covered = status_counts.get("clean_covered", 0)
    final_adaptive_cover = method_use.get("final_adaptive_cover", 0)
    opencv_telea = method_use.get("opencv_telea_inpaint", 0)
    report = {
        "version": "V14_BETTER_CANDIDATES",
        "qa_schema_version": "v13",
        "final_visual_gate_version": "v13",
        "n_published": n,
        "clean_repaired": n_repaired,
        "clean_covered": n_covered,
        "no_watermark": status_counts.get("no_watermark", 0),
        "failed_io": status_counts.get("failed_io", 0),
        "unique_final_methods": len(methods),
        "v13_demoted_from_repaired": demoted,
        "honesty_counters": {k: int(counters.get(k, 0)) for k in BAD_COUNTERS},
        "v14_cover_honesty_counters": {
            k: int(cover_counters.get(k, 0)) for k in V14_COVER_COUNTERS},
        "v14_near_miss_rescued": near_miss_rescued,
        "v14_segmented_micro_cover": segmented_micro_cover,
        "v14_final_adaptive_cover": final_adaptive_cover,
        "v14_opencv_telea_final": opencv_telea,
        "v14_targets": dict(V14_TARGETS),
    }
    report["all_clean"] = (
        all(v == 0 for v in report["honesty_counters"].values()) and
        all(v == 0 for v in report["v14_cover_honesty_counters"].values()))
    return report


def main(argv):
    out_root = Path(argv[1]) if len(argv) > 1 else Path("output")
    report = build_report(out_root)
    (out_root / "v13_honesty.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["all_clean"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
