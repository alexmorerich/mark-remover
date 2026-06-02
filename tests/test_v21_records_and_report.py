#!/usr/bin/env python3
"""V21 — failure-taxonomy / mask-quality / residual-explain records + report
metrics (patch plan §1, §6, §8, §10, §11)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v18_patch  # noqa: E402
import v13_report  # noqa: E402
import mark_remover  # noqa: E402


def test_version_is_v21():
    assert mark_remover.PIPELINE_VERSION == "V21_PATCH"
    assert mark_remover.PATCH_VERSION == "v21"
    # Frozen layers are NOT renamed.
    assert mark_remover.FINAL_VISUAL_GATE_VERSION == "v13"
    assert mark_remover.STATE_MACHINE_VERSION == "v16"
    assert mark_remover.FINAL_AUDIT_VERSION == "v17"


def test_taxonomy_true_residual_recommends_micro_clean():
    rec = {
        "status": "auto_rejected",
        "v18_roi_class": "simple_product_surface",
        "v18_product_overlap": 0.4,
        "residual_explain": {"low_contrast_glyph_positive": True},
        "reject_reasons": ["repair_failed_final_gate"],
    }
    out = v18_patch.build_v21_records(rec, "auto_rejected")
    tax = out["v21_failure_taxonomy"]
    assert tax["primary_reject_class"] == "true_residual"
    assert tax["residual_kind"] == "low_contrast_ghost"
    assert tax["recommended_next_candidate"] == "residue_micro_clean"


def test_taxonomy_product_damage_class():
    rec = {
        "status": "auto_rejected",
        "v18_roi_class": "dark_product_surface",
        "v18_product_overlap": 0.5,
        "v17_hard_fail_reasons": ["changed_dark_surface_blob"],
        "residual_explain": {},
    }
    out = v18_patch.build_v21_records(rec, "auto_rejected")
    assert out["v21_failure_taxonomy"]["primary_reject_class"] == "product_damage"


def test_published_taxonomy_is_published_class():
    rec = {"status": "clean_repaired", "v18_roi_class": "plain_white",
           "residual_explain": {}}
    out = v18_patch.build_v21_records(rec, "clean_repaired")
    assert out["v21_failure_taxonomy"]["primary_reject_class"] == "published"


def test_mask_quality_logo_fallback_flagged_on_product():
    rec = {"status": "auto_rejected", "mask_type": "logo_fallback",
           "v18_product_overlap": 0.3, "residual_explain": {}}
    out = v18_patch.build_v21_records(rec, "auto_rejected")
    mq = out["v21_mask_quality"]
    assert mq["mask_source"] == "logo_fallback"
    assert mq["fallback_reason"] == "product_overlap"


def test_residual_explain_authoritative_field():
    rec = {"status": "auto_rejected",
           "residual_explain": {"dot_chain_positive": True,
                                "template_positive": True},
           "v18_product_overlap": 0.2}
    out = v18_patch.build_v21_records(rec, "auto_rejected")
    assert out["v21_residual_explain"]["authoritative_residual"] == "dot_chain"
    assert out["v21_residual_explain"]["on_product"] is True


def test_report_emits_v21_block_and_stays_clean(tmp_path):
    # A minimal published record + a rejected record; the report must surface the
    # v21 block and keep the safety invariant green.
    root = tmp_path / "out"
    pub = root / "clean_repaired" / "p1"
    rej = root / "auto_rejected" / "p2"
    pub.mkdir(parents=True)
    rej.mkdir(parents=True)
    (pub / "qa.json").write_text(json.dumps({
        "status": "clean_repaired", "publish_ok": True,
        "v9_final_method": "v21_residue_micro_hue_matched_clone",
        "p0_gates": {}, "v17_hard_fail_reasons": [],
        "v21_mask_quality": {"mask_source": "alpha_ncc"},
        "v21_failure_taxonomy": {"primary_reject_class": "published"},
        "v21_residue_micro_attempted": 1,
    }))
    (rej / "qa.json").write_text(json.dumps({
        "status": "auto_rejected", "reject_reasons": ["repair_failed_final_gate"],
        "v21_mask_quality": {"mask_source": "logo_fallback"},
        "v21_failure_taxonomy": {"primary_reject_class": "no_safe_candidate",
                                 "recommended_next_candidate": "mixed_roi_split_v2"},
        "residual_explain": {},
    }))
    report = v13_report.build_report(root)
    assert report["version"] == "V21_PATCH"
    assert "v21" in report
    assert report["v21"]["residue_micro_published"] == 1
    assert report["v21"]["logo_fallback_count"] == 1
    assert report["v21"]["alpha_ncc_mask_count"] == 1
    assert report["v21"]["repairable_rejects_remaining"] == 1
    # Safety invariant unaffected by the V21 additions.
    assert report["all_clean"] is True
