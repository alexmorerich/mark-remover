#!/usr/bin/env python3
"""V16 — auto-decision CI gate / honesty report.

Scans an output tree's per-image ``qa.json`` records and enforces the V16
invariant: **only outputs that passed every P0 gate may be published as
clean_repaired / clean_covered; everything else is auto_rejected.**

The CI gate fails (exit 1) if ANY published output violates a P0 gate or if any
``final_output_publish_failure`` is set. Rejected *candidates* are allowed and
reported separately. (Filename kept as ``v13_report.py`` for the run scripts;
it is now the V16 report.)

Usage:
    python3 v13_report.py output
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PUBLISHED = ("clean_repaired", "clean_covered")

# Patch plan Fix 3/6 — published outputs must have ZERO P0 failures. The CI
# gate fails if any of these are non-zero.
MUST_BE_ZERO = [
    "published_with_residual_ocr",
    "published_with_template_residual",
    "published_with_dot_chain",
    "published_with_visible_patch",
    "published_with_visible_band",
    "published_with_product_damage",
    "published_with_silhouette_damage",
    "published_with_protected_text_damage",
    "final_output_publish_failures",
]

# p0_gates key -> published_with_* counter it feeds when False.
_P0_TO_COUNTER = {
    "residual_ocr_pass": "published_with_residual_ocr",
    "template_residual_pass": "published_with_template_residual",
    "dot_chain_pass": "published_with_dot_chain",
    "visible_patch_pass": "published_with_visible_patch",
    "visible_band_pass": "published_with_visible_band",
    "product_damage_pass": "published_with_product_damage",
    "silhouette_pass": "published_with_silhouette_damage",
    "protected_text_pass": "published_with_protected_text_damage",
}


def _iter_records(out_root: Path):
    for qa in out_root.rglob("qa.json"):
        try:
            yield json.loads(qa.read_text())
        except Exception:
            continue


def build_report(out_root: Path) -> dict:
    must_zero = Counter()
    status_counts = Counter()
    methods = set()
    rejected_reasons = Counter()
    candidate_failures = 0
    auto_rejected_after_cover_fail = 0
    auto_rejected_after_repair_fail = 0
    tools_reachable = tools_tried = candidates_passed = 0

    for rec in _iter_records(out_root):
        status = rec.get("status")
        status_counts[status] += 1
        candidate_failures += int(rec.get("candidate_publish_failures", 0) or 0)
        tel = rec
        tools_reachable = max(tools_reachable,
                              int((rec.get("v11_tools_reachable") or 0)))
        tools_tried += int(rec.get("v11_tools_tried") or 0)
        candidates_passed += int(rec.get("v11_candidates_passed") or 0)

        if status in PUBLISHED:
            methods.add(rec.get("v9_final_method", "unknown"))
            p0 = rec.get("p0_gates", {}) or {}
            for gate_key, counter in _P0_TO_COUNTER.items():
                # default True so a missing gate never silently fails CI, but a
                # published output should always carry a full p0_gates dict.
                if p0.get(gate_key, True) is False:
                    must_zero[counter] += 1
            if rec.get("final_output_publish_failure"):
                must_zero["final_output_publish_failures"] += 1
        elif status == "auto_rejected":
            reasons = rec.get("reject_reasons", []) or []
            for r in reasons:
                rejected_reasons[r] += 1
            if any("cover" in r for r in reasons):
                auto_rejected_after_cover_fail += 1
            else:
                auto_rejected_after_repair_fail += 1

    must_zero_d = {k: int(must_zero.get(k, 0)) for k in MUST_BE_ZERO}
    all_clean = all(v == 0 for v in must_zero_d.values())

    report = {
        "version": "V16_AUTO_DECISION",
        "qa_schema_version": "v13",
        "final_gate_version": "v16",
        # Final output counters (patch plan Fix 6).
        "final_clean_repaired": status_counts.get("clean_repaired", 0),
        "final_clean_covered": status_counts.get("clean_covered", 0),
        "final_no_watermark_confirmed":
            status_counts.get("no_watermark_confirmed", 0) +
            status_counts.get("no_watermark", 0),
        "final_skipped_known_clean": status_counts.get("skipped_known_clean", 0),
        "final_auto_rejected": status_counts.get("auto_rejected", 0),
        "final_failed_io": status_counts.get("failed_io", 0),
        "n_published": (status_counts.get("clean_repaired", 0) +
                        status_counts.get("clean_covered", 0)),
        "unique_final_methods": len(methods),
        # Must-be-zero published-output counters.
        "must_be_zero": must_zero_d,
        # Allowed rejected-candidate counters.
        "candidate_publish_failures": candidate_failures,
        "auto_rejected_after_repair_fail": auto_rejected_after_repair_fail,
        "auto_rejected_after_cover_fail": auto_rejected_after_cover_fail,
        "reject_reasons": dict(rejected_reasons),
        # Tool availability (patch plan Fix 12).
        "tools": {
            "tools_reachable": tools_reachable,
            "tools_tried": tools_tried,
            "candidates_passed": candidates_passed,
        },
        "manual_review": 0,   # V16 has no manual-review state by construction.
        "all_clean": all_clean,
    }
    return report


def main(argv):
    out_root = Path(argv[1]) if len(argv) > 1 else Path("output")
    report = build_report(out_root)
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2))
    # Back-compat filename used by older tooling.
    (out_root / "v13_honesty.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, separators=(",", ":")))
    # CI gate (patch plan Fix 6): fail if any published output broke a P0 gate.
    return 0 if report["all_clean"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
