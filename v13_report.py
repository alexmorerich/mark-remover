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

# V16.1 — HARD SAFETY published-output counters that MUST be zero. A published
# output must have the watermark verifiably gone AND the product undamaged.
# Visible band / patch are COSMETIC (tracked below, allowed): a faint seam on
# the background is acceptable, leaving the mark or damaging the product is not.
MUST_BE_ZERO = [
    "published_with_residual_ocr",
    "published_with_template_residual",
    "published_with_dot_chain",
    "published_with_product_damage",
    "published_with_silhouette_damage",
    "published_with_protected_text_damage",
    "final_output_publish_failures",
]

# Cosmetic counters — reported, allowed to be > 0 (a soft seam on a verifiably
# mark-removed, product-safe output).
COSMETIC_COUNTERS = [
    "published_with_visible_patch",
    "published_with_visible_band",
]

# V17 — published visual-audit counters (patch plan §11.1). EVERY one of these
# must be zero: they count published outputs whose ACTUAL bytes failed the
# truthful final audit (residual watermark or product damage). By construction
# they are zero (a failed audit forces auto_rejected), and CI enforces it.
V17_MUST_BE_ZERO = [
    "published_residual_watermark",
    "published_dot_chain",
    "published_product_damage",
    "published_visible_patch_on_product",
    "published_visible_band_on_product",
    "published_silhouette_damage",
    "published_protected_text_damage",
    # V18 (patch plan §8, §11) — faint glyph residue on a published output.
    "published_low_contrast_glyph_residue",
]

# V17/V18 audit hard-fail reason -> published_* counter.
_V17_REASON_TO_COUNTER = {
    "published_residual_watermark": "published_residual_watermark",
    "published_dot_chain": "published_dot_chain",
    "published_low_contrast_glyph_residue":
        "published_low_contrast_glyph_residue",
    "visible_patch_on_product": "published_visible_patch_on_product",
    "visible_band_on_product": "published_visible_band_on_product",
    "changed_product_silhouette": "published_silhouette_damage",
    "changed_protected_text": "published_protected_text_damage",
    "changed_thin_flex_structure": "published_product_damage",
    "changed_dark_surface_blob": "published_product_damage",
    "changed_metallic_surface_block": "published_product_damage",
    # V19 — cover-shape artifact on product (patch plan §4.4).
    "wedge_or_slab_shape": "published_visible_patch_on_product",
}

# V19 — cover-shape artifact reasons (patch plan §4.4, §5) aggregated for the
# diagnostics block; these are caught before publish (forced to auto_rejected).
_V19_COVER_ARTIFACT_REASONS = {
    "visible_patch_on_product": "visible_cover_on_product",
    "wedge_or_slab_shape": "wedge_cover_on_product",
    "visible_band_on_product": "band_cover_on_product",
    "changed_dark_surface_blob": "dark_blob_cover",
}


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
    v17_zero = Counter()
    status_counts = Counter()
    methods = set()
    rejected_reasons = Counter()
    candidate_failures = 0
    auto_rejected_after_cover_fail = 0
    auto_rejected_after_repair_fail = 0
    cosmetic_seam_published = 0
    tools_reachable = tools_tried = candidates_passed = 0
    # V18 — reject taxonomy aggregation (patch plan §1).
    rejected_by_roi_class = Counter()
    rejected_by_method = Counter()
    rejected_by_mask_type = Counter()
    candidate_failures_by_reason = Counter()
    published_seam_by_roi_class = Counter()
    v18_ban_destructive_count = 0
    v18_published_via_safe_candidate = 0
    # V19 — reverse-alpha + cover-artifact + text-protection aggregation.
    ra_attempted = ra_passed = 0
    ra_failed_residual = ra_failed_damage = 0
    ra_failed_align = ra_failed_missing = 0
    ra_over_text_attempted = ra_over_text_passed = 0
    ra_published = 0
    cover_artifacts = Counter()
    protected_text_cases = 0
    ppocr_enabled = False
    # V20 — diagnostics (patch plan §Patch 10).
    v20_variants_attempted = v20_variants_passed = 0
    v20_ghost_dot_failures = 0
    v20_cover_on_product_hard_fails = 0
    v20_segmented_mixed_published = 0
    v20_fp_suspects = 0
    v20_rej_true_residual = v20_rej_product_damage = 0
    v20_rej_cover_artifact = v20_rej_uncertain = 0
    _DAMAGE_TOKENS = ("product_damage", "silhouette", "flex", "dark_surface",
                      "metallic", "protected_text", "thin_flex")
    _COVER_TOKENS = ("visible_patch", "visible_band", "wedge", "slab", "cover")

    for rec in _iter_records(out_root):
        status = rec.get("status")
        status_counts[status] += 1
        # V20 — reverse-alpha variant beam + audit sub-records.
        v20_variants_attempted += int(
            rec.get("reverse_alpha_variants_attempted", 0) or 0)
        v20_variants_passed += int(
            rec.get("reverse_alpha_variants_passed", 0) or 0)
        if (rec.get("reverse_alpha_ghost_dots") or {}).get("hard_fail"):
            v20_ghost_dot_failures += 1
        if (rec.get("cover_artifact_v20") or {}).get("hard_fail_reason"):
            v20_cover_on_product_hard_fails += 1
        if (rec.get("residual_explain") or {}).get(
                "possible_product_structure_false_positive"):
            v20_fp_suspects += 1
        if status in PUBLISHED and str(rec.get("v9_final_method", "")).startswith(
                "v20_segmented_reverse_alpha_background_clone"):
            v20_segmented_mixed_published += 1
        if status == "auto_rejected":
            rexp = rec.get("residual_explain") or {}
            reasons_l = [str(r) for r in (rec.get("reject_reasons") or [])]
            joined = " ".join(reasons_l)
            if rexp.get("ocr_positive") or rexp.get("dot_chain_positive") or \
                    rexp.get("low_contrast_glyph_positive"):
                v20_rej_true_residual += 1
            elif any(t in joined for t in _COVER_TOKENS):
                v20_rej_cover_artifact += 1
            elif any(t in joined for t in _DAMAGE_TOKENS):
                v20_rej_product_damage += 1
            else:
                v20_rej_uncertain += 1
        # V19 — reverse-alpha telemetry (present on every record once wired).
        if rec.get("reverse_alpha_attempted"):
            ra_attempted += 1
            if rec.get("reverse_alpha_passed"):
                ra_passed += 1
                if str(rec.get("v9_final_method", "")).startswith(
                        "v19_sunsky_reverse_alpha") and status in PUBLISHED:
                    ra_published += 1
            else:
                reason = str(rec.get("reverse_alpha_reject_reason") or "")
                if "residual" in reason:
                    ra_failed_residual += 1
                elif "product" in reason or "blob" in reason or "text" in reason:
                    ra_failed_damage += 1
                elif "placement" in reason or "align" in reason:
                    ra_failed_align += 1
                elif "missing" in reason:
                    ra_failed_missing += 1
            if rec.get("reverse_alpha_over_text"):
                ra_over_text_attempted += 1
                if rec.get("reverse_alpha_passed"):
                    ra_over_text_passed += 1
        if (rec.get("v18_protected_text_score") or 0.0) > 0.01:
            protected_text_cases += 1
        # V19 — cover-shape artifacts caught (in any record's audit reasons).
        for reason in (rec.get("v17_hard_fail_reasons") or []):
            ck = _V19_COVER_ARTIFACT_REASONS.get(reason)
            if ck:
                cover_artifacts[ck] += 1
        candidate_failures += int(rec.get("candidate_publish_failures", 0) or 0)
        if status in PUBLISHED and rec.get("cosmetic_seam"):
            cosmetic_seam_published += 1
        if rec.get("v18_ban_destructive"):
            v18_ban_destructive_count += 1
        # A published output whose chosen method is a V18 safe candidate.
        if status in PUBLISHED and \
                str(rec.get("v9_final_method", "")).startswith("v18_"):
            v18_published_via_safe_candidate += 1
            if rec.get("cosmetic_seam"):
                published_seam_by_roi_class[
                    rec.get("v18_roi_class") or rec.get("v9_roi_class") or
                    "unknown"] += 1
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
            # V17 — published-output truthful-audit failures (must be zero).
            for reason in (rec.get("v17_hard_fail_reasons") or []):
                counter = _V17_REASON_TO_COUNTER.get(reason)
                if counter:
                    v17_zero[counter] += 1
        elif status == "auto_rejected":
            reasons = rec.get("reject_reasons", []) or []
            for r in reasons:
                rejected_reasons[r] += 1
            if any("cover" in r for r in reasons):
                auto_rejected_after_cover_fail += 1
            else:
                auto_rejected_after_repair_fail += 1
            # V18 — taxonomy aggregation (patch plan §1): which ROI class / best
            # method / mask type the rejected images cluster around.
            tax = rec.get("v18_reject_taxonomy", {}) or {}
            rejected_by_roi_class[
                tax.get("roi_class") or rec.get("v18_roi_class") or
                rec.get("v9_roi_class") or "unknown"] += 1
            rejected_by_method[
                tax.get("best_repair_method") or
                rec.get("v9_final_method") or "unknown"] += 1
            rejected_by_mask_type[tax.get("mask_used") or "unknown"] += 1
            for r in (tax.get("candidate_fail_reasons") or []):
                candidate_failures_by_reason[r] += 1

    must_zero_d = {k: int(must_zero.get(k, 0)) for k in MUST_BE_ZERO}
    cosmetic_d = {k: int(must_zero.get(k, 0)) for k in COSMETIC_COUNTERS}
    v17_zero_d = {k: int(v17_zero.get(k, 0)) for k in V17_MUST_BE_ZERO}
    all_clean = (all(v == 0 for v in must_zero_d.values()) and
                 all(v == 0 for v in v17_zero_d.values()))

    try:
        import product_text_detector as _ptd
        ppocr_enabled = bool(_ptd.ppocr_available())
    except Exception:
        ppocr_enabled = False

    report = {
        "version": "V20_PATCH",
        # V20 (patch plan §Patch 10) — explicit per-layer versions so a regression
        # comparison can never confuse the state machine, audit and gate.
        "state_machine_version": "v16",
        "final_audit_version": "v17",
        "patch_version": "v20_safer_reverse_alpha",
        "gate_version": "v13_frozen",
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
        # Must-be-zero published-output HARD-SAFETY counters.
        "must_be_zero": must_zero_d,
        # V17 — published truthful-audit failures (must all be zero).
        "v17_published_audit_failures": v17_zero_d,
        # Cosmetic counters (allowed > 0): a soft seam on a mark-removed,
        # product-safe published output.
        "cosmetic": cosmetic_d,
        "published_with_cosmetic_seam": cosmetic_seam_published,
        # Allowed rejected-candidate counters.
        "candidate_publish_failures": candidate_failures,
        "auto_rejected_after_repair_fail": auto_rejected_after_repair_fail,
        "auto_rejected_after_cover_fail": auto_rejected_after_cover_fail,
        "reject_reasons": dict(rejected_reasons),
        # V18 — reject taxonomy aggregation (patch plan §1).
        "auto_rejected_by_reason": dict(rejected_reasons),
        "auto_rejected_by_roi_class": dict(rejected_by_roi_class),
        "auto_rejected_by_method": dict(rejected_by_method),
        "auto_rejected_by_mask_type": dict(rejected_by_mask_type),
        "candidate_failures_by_reason": dict(candidate_failures_by_reason),
        "published_cosmetic_seams_by_roi_class": dict(published_seam_by_roi_class),
        "v18_ban_destructive_count": v18_ban_destructive_count,
        "v18_published_via_safe_candidate": v18_published_via_safe_candidate,
        # V19 — reverse-alpha recovery diagnostics (patch plan §5).
        "reverse_alpha": {
            "attempted": ra_attempted,
            "passed": ra_passed,
            "published": ra_published,
            "failed_residual": ra_failed_residual,
            "failed_product_damage": ra_failed_damage,
            "failed_alignment": ra_failed_align,
            "failed_alpha_asset_missing": ra_failed_missing,
        },
        # V19 — cover-shape artifacts caught before publish (patch plan §4.4).
        "cover_artifacts_v19": {
            "visible_cover_on_product":
                int(cover_artifacts.get("visible_cover_on_product", 0)),
            "wedge_cover_on_product":
                int(cover_artifacts.get("wedge_cover_on_product", 0)),
            "band_cover_on_product":
                int(cover_artifacts.get("band_cover_on_product", 0)),
            "dark_blob_cover":
                int(cover_artifacts.get("dark_blob_cover", 0)),
        },
        # V20 — diagnostics (patch plan §Patch 10).
        "v20": {
            "reverse_alpha_variants_attempted": v20_variants_attempted,
            "reverse_alpha_variants_passed": v20_variants_passed,
            "reverse_alpha_ghost_dot_failures": v20_ghost_dot_failures,
            "cover_on_product_hard_fails": v20_cover_on_product_hard_fails,
            "segmented_mixed_repairs_published": v20_segmented_mixed_published,
            "detector_false_positive_suspects": v20_fp_suspects,
            "auto_rejected_true_residual": v20_rej_true_residual,
            "auto_rejected_product_damage": v20_rej_product_damage,
            "auto_rejected_cover_artifact": v20_rej_cover_artifact,
            "auto_rejected_uncertain": v20_rej_uncertain,
        },
        # V19 — product-text protection (patch plan §1.7).
        "text_protection": {
            "ppocr_enabled": ppocr_enabled,
            "protected_text_overlap_cases": protected_text_cases,
            "reverse_alpha_over_text_attempted": ra_over_text_attempted,
            "reverse_alpha_over_text_passed": ra_over_text_passed,
        },
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


def write_v20_reasons_csv(out_root: Path) -> Path:
    """V20 — per-image one-line reason CSV (patch plan §Patch 10)::

        id,status,final_method,roi_class,product_overlap,residual_reason,
        damage_reason,published
    """
    import csv
    rows = []
    for rec in _iter_records(out_root):
        status = rec.get("status", "")
        rexp = rec.get("residual_explain") or {}
        residual_reason = ""
        if rexp.get("ocr_positive"):
            residual_reason = "ocr"
        elif rexp.get("dot_chain_positive"):
            residual_reason = "dot_chain"
        elif rexp.get("low_contrast_glyph_positive"):
            residual_reason = "low_contrast_glyph"
        elif rexp.get("possible_product_structure_false_positive"):
            residual_reason = "template_only_fp_suspect"
        damage_reason = ""
        for r in (rec.get("reject_reasons") or []):
            rs = str(r)
            if any(t in rs for t in ("product_damage", "silhouette", "flex",
                                     "dark_surface", "metallic", "protected_text",
                                     "visible_patch", "visible_band", "wedge")):
                damage_reason = rs
                break
        rows.append({
            "id": rec.get("product_id", ""),
            "status": status,
            "final_method": rec.get("v9_final_method", ""),
            "roi_class": rec.get("v18_roi_class") or rec.get("v9_roi_class") or "",
            "product_overlap": rec.get("v18_product_overlap", ""),
            "residual_reason": residual_reason,
            "damage_reason": damage_reason,
            "published": status in PUBLISHED,
        })
    rows.sort(key=lambda r: (r["status"], r["id"]))
    csv_path = out_root / "v20_reasons.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "status", "final_method",
                                          "roi_class", "product_overlap",
                                          "residual_reason", "damage_reason",
                                          "published"])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def main(argv):
    out_root = Path(argv[1]) if len(argv) > 1 else Path("output")
    report = build_report(out_root)
    # V20 — only published outputs may contain cleaned.jpg (patch plan §Patch 2).
    rejected_cleaned = list((out_root / "auto_rejected").rglob("cleaned.jpg"))
    report["auto_rejected_cleaned_jpg_leak"] = len(rejected_cleaned)
    if rejected_cleaned:
        report["all_clean"] = False
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2))
    # Back-compat filename used by older tooling.
    (out_root / "v13_honesty.json").write_text(json.dumps(report, indent=2))
    try:
        write_v20_reasons_csv(out_root)
    except Exception:
        pass
    print(json.dumps(report, separators=(",", ":")))
    # CI gate: fail if any published output broke a P0 gate OR a rejected folder
    # leaked a cleaned.jpg (patch plan §Patch 2, §Patch 10).
    return 0 if report["all_clean"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
