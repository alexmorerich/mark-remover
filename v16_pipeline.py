#!/usr/bin/env python3
"""V16 — Complete auto-decision pipeline with zero dirty publish.

The single invariant V16 enforces:

    Only a final output that PASSES the final visual publish gate may be
    labelled ``clean_repaired`` or ``clean_covered``. Everything else is
    ``auto_rejected`` — a *final, automated* decision, never a manual review and
    never published.

This module replaces the V13/V15 "demote a failed repair straight to
clean_covered" semantics (which let a still-bad cover ship as clean_covered)
with an explicit state machine:

    repair candidate passes P0 gate          -> clean_repaired
    else residual micro-cleanup passes        -> clean_repaired
    else best cover candidate passes P0 gate  -> clean_covered
    else                                      -> auto_rejected

Key distinctions (patch plan Fix 2):
  * ``candidate_publish_failures`` — intermediate repair/cover candidates the
    gate rejected. May be > 0. Perfectly healthy.
  * ``final_output_publish_failure`` — a *published* output that fails the gate.
    Must be False for every clean_* output, always. By construction it is, since
    clean_* is only assigned when ``_p0`` returns publishable.

The P0 gate (must-be-zero, SAME strictness for repaired and covered, Fix 5) is
the unchanged ``final_visual_publish_gate_v13`` PLUS a post-clean re-detection of
the watermark on the actual output (residual-OCR / template-residual). Aesthetic
scoring may differ between repaired and covered; the *safety* gate may not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import v13_gates
import v14_patch
import v15_patch


# ---------------------------------------------------------------------------
# P0 gate dictionary built from a FinalVisualVerdict + a post-clean re-detect.
# ---------------------------------------------------------------------------
def _p0_gates_from(verdict, still_present: bool) -> dict:
    return {
        "residual_ocr_pass": (not still_present),
        "template_residual_pass": bool(verdict.residual_pass),
        "dot_chain_pass": bool(verdict.dot_chain_pass),
        "visible_patch_pass": bool(verdict.visible_patch_pass and
                                   verdict.rectangular_band_pass and
                                   verdict.polygon_patch_pass),
        "visible_band_pass": bool(verdict.rectangular_band_pass),
        "product_damage_pass": bool(verdict.product_damage_pass),
        "silhouette_pass": bool(verdict.silhouette_pass),
        "protected_text_pass": bool(verdict.protected_text_pass),
    }


@dataclass
class V16Result:
    status: str                       # clean_repaired | clean_covered | auto_rejected
    image: object                     # published image, or best attempt if rejected
    method: str
    verdict: object                   # FinalVisualVerdict of the chosen / best attempt
    p0_gates: dict = field(default_factory=dict)
    publish_ok: bool = False
    candidate_publish_failures: int = 0
    reject_reasons: list = field(default_factory=list)
    best_repair_image: object = None
    best_cover_image: object = None
    telemetry: dict = field(default_factory=dict)


def decide_final_status(img, bbox, product_mask, watermark_mask, qa_info,
                        *, loop_repair_image=None, loop_repair_method="repair",
                        best_failed_image=None, loop_cover_image=None,
                        roi_class="", gate_fn: Callable, recheck_fn: Callable):
    """Run the V16 state machine and return a :class:`V16Result`.

    ``gate_fn(image, repaired_bool, residual_pass=None) -> FinalVisualVerdict``
    re-runs the unchanged V13 visual gate.
    ``recheck_fn(image) -> (still_present: bool, info: dict)`` re-runs the
    watermark detector on the output (post-clean re-detection / residual-OCR).
    """
    context = v14_patch.soft_qa_context(img, bbox, product_mask,
                                        roi_class=roi_class)
    state = {"candidate_failures": 0}
    reject_reasons = []

    def _p0(image, repaired):
        # Fresh per-candidate watermark-hiding verdict for the template/dot
        # residual gate, then the V13 visual gate, then — only if the visual
        # gate passes (cheap-first) — the expensive post-clean detector
        # re-check. A candidate that already fails the visual gate is rejected
        # without paying for re-detection.
        hides = v14_patch.cover_hides_watermark(img, image, bbox, watermark_mask)
        verdict = gate_fn(image, repaired, hides)
        if not verdict.publish_ok:
            return False, verdict, _p0_gates_from(verdict, False), False
        still, _info = recheck_fn(image)
        return (not still), verdict, _p0_gates_from(verdict, still), still

    uniform = None
    if v15_patch.is_uniform_local_background(img, bbox, product_mask):
        try:
            uniform = v15_patch.uniform_background_fill(img, bbox, watermark_mask)
        except Exception:
            uniform = None

    # ---------------- REPAIR PATH ----------------
    # Repair candidates, strongest-first: a uniform-background neighbour fill,
    # the beam's chosen repair, then its best near-miss repair (if the loop fell
    # to cover). Each must pass the full P0 gate to publish as clean_repaired.
    repair_cands = []
    if uniform is not None:
        repair_cands.append(("v16_uniform_background_fill", uniform))
    if loop_repair_image is not None:
        repair_cands.append((loop_repair_method, loop_repair_image))
    if best_failed_image is not None:
        repair_cands.append(("best_failed_repair", best_failed_image))

    best_repair = repair_cands[0][1] if repair_cands else None
    for name, rimg in repair_cands:
        ok, verdict, gates, _still = _p0(rimg, True)
        if ok:
            return V16Result(
                status="clean_repaired", image=rimg, method=name,
                verdict=verdict, p0_gates=gates, publish_ok=True,
                candidate_publish_failures=state["candidate_failures"],
                best_repair_image=rimg,
                telemetry={"v16_path": "repair"})
        state["candidate_failures"] += 1
        reject_reasons.append("repair_failed_final_gate")
        # Residual micro-cleanup / near-miss rescue, only on a soft fail.
        if v14_patch.classify_failure(verdict) == "soft_fail":
            try:
                rescued = v14_patch.near_miss_rescue(
                    img, rimg, bbox, product_mask, watermark_mask, context)
            except Exception:
                rescued = None
            if rescued is not None:
                ok2, verdict2, gates2, _s2 = _p0(rescued, True)
                if ok2:
                    return V16Result(
                        status="clean_repaired", image=rescued,
                        method=name + "+v16_micro_cleanup", verdict=verdict2,
                        p0_gates=gates2, publish_ok=True,
                        candidate_publish_failures=state["candidate_failures"],
                        best_repair_image=rescued,
                        telemetry={"v16_path": "repair_rescued"})
                state["candidate_failures"] += 1

    # ---------------- COVER BEAM ----------------
    try:
        mc = v14_patch.segmented_micro_cover(
            img, bbox, watermark_mask, product_mask, context,
            base_cover=loop_cover_image)
        cover_cands = [(n, im) for (n, im, _full, _s) in mc.ranked]
    except Exception:
        cover_cands = []
    # A uniform-background fill is also a valid cover; a forced full-text-band
    # removal is the last-resort cover (Fix 4 cover beam).
    if uniform is not None:
        cover_cands.insert(0, ("v16_uniform_background_fill", uniform))
    try:
        cover_cands.append(("v16_forced_removal",
                            v15_patch.forced_removal_fill(img, bbox,
                                                          watermark_mask)))
    except Exception:
        pass

    best_cover = cover_cands[0][1] if cover_cands else None
    last_verdict = None
    last_gates = {}
    for name, cimg in cover_cands:
        ok, verdict, gates, _still = _p0(cimg, False)
        last_verdict, last_gates = verdict, gates
        if ok:
            return V16Result(
                status="clean_covered", image=cimg, method=name,
                verdict=verdict, p0_gates=gates, publish_ok=True,
                candidate_publish_failures=state["candidate_failures"],
                best_repair_image=best_repair, best_cover_image=cimg,
                telemetry={"v16_path": "cover"})
        state["candidate_failures"] += 1
        for r in verdict.reject_reasons:
            reject_reasons.append("cover_failed_" + r)
        if best_cover is None:
            best_cover = cimg

    # ---------------- AUTO REJECTED ----------------
    # Repair AND cover both failed the P0 gate. Final automated decision — not a
    # manual review, not published. Keep the best attempt for diagnostics.
    final_img = best_cover if best_cover is not None else best_repair
    if final_img is None:
        final_img = img
    verdict = last_verdict
    gates = last_gates
    if verdict is None:
        # No cover candidate ran at all — re-evaluate the best repair attempt so
        # the diagnostics record real gate verdicts.
        _ok, verdict, gates, _s = _p0(final_img, False)
    # de-dup reasons, preserve order
    seen = set()
    reasons = [r for r in reject_reasons if not (r in seen or seen.add(r))]
    return V16Result(
        status="auto_rejected", image=final_img, method="auto_rejected",
        verdict=verdict, p0_gates=gates, publish_ok=False,
        candidate_publish_failures=state["candidate_failures"],
        reject_reasons=reasons or ["repair_and_cover_failed_final_gate"],
        best_repair_image=best_repair, best_cover_image=best_cover,
        telemetry={"v16_path": "auto_rejected"})
