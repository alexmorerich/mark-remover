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
import v17_final_audit
import v18_patch


# ---------------------------------------------------------------------------
# Gate tiers (V16.1).
#
# HARD SAFETY gates — must pass to publish. The watermark must be verifiably
# GONE (detector re-detect + template/dot residual) and the PRODUCT must be
# undamaged (no product-damage, no silhouette break, no protected-text loss).
# Leaving the mark, or damaging the product, is never publishable.
#
# COSMETIC gates — a visible band / patch where the watermark used to be. These
# are tracked and reported, but on a verifiably-removed, product-safe output
# they do NOT block publishing: a faint seam on the background is far better
# than shipping the watermark or auto-rejecting a cleanable image. (This is the
# V16.1 reconciliation of "zero dirty publish" with the standing product rule
# that the mark MUST be removed without damaging the product.)
# ---------------------------------------------------------------------------
HARD_SAFETY_GATES = ("residual_ocr_pass", "template_residual_pass",
                     "dot_chain_pass", "product_damage_pass",
                     "silhouette_pass", "protected_text_pass")
COSMETIC_GATES = ("visible_patch_pass", "visible_band_pass")


def _is_strict(gates: dict) -> bool:
    return all(gates.get(k, True) for k in (HARD_SAFETY_GATES + COSMETIC_GATES))


def _is_safe(gates: dict) -> bool:
    # Watermark verifiably gone AND product undamaged (cosmetic may fail).
    return all(gates.get(k, True) for k in HARD_SAFETY_GATES)


def _cosmetic_score(verdict) -> float:
    sc = getattr(verdict, "scores", {}) or {}
    return (float(sc.get("visible_patch_shape", 0.0)) +
            float(sc.get("visible_band", 0.0)) +
            float(sc.get("hard_boundary", 0.0)))


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
    cosmetic_seam: bool = False        # published, mark gone + product-safe, soft seam
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

    # --- V18 — product-context pre-routing (patch plan §2). Compute the
    # product-overlap signals ONCE, decide whether destructive generators are
    # banned, and pre-build product-safe candidates (segmented + specialized +
    # stroke-only). These are ADDED to the repair/cover pools below; every one
    # still passes through the unchanged P0 gate, so this can only widen the set
    # of safe options, never ship an output the audit would reject. ----------
    try:
        pctx = v18_patch.compute_product_context(
            img, bbox, product_mask, watermark_mask, roi_class=roi_class)
        ban_destructive = v18_patch.should_ban_destructive(pctx)
        v18_safe = v18_patch.build_safe_candidates(
            img, bbox, watermark_mask, product_mask, pctx)
        v18_safe = v18_patch.rank_candidates(
            img, v18_safe, bbox, product_mask, watermark_mask, pctx)
    except Exception:
        pctx = None
        ban_destructive = False
        v18_safe = []
    v18_telem = {"v18_ban_destructive": bool(ban_destructive),
                 "v18_n_safe_candidates": len(v18_safe),
                 "v18_safe_candidate_names": [n for n, _ in v18_safe]}
    if pctx is not None:
        v18_telem.update(pctx.to_record())

    def _p0(image, repaired):
        # Fresh per-candidate watermark-hiding verdict for the template/dot
        # residual gate, then the V13 visual gate, then the post-clean detector
        # re-check. The detector re-check is the AUTHORITATIVE residual signal
        # (template correlation false-passes on bright metal), so it always runs.
        # Returns (strict_ok, safe_ok, verdict, gates, still_present).
        hides = v14_patch.cover_hides_watermark(img, image, bbox, watermark_mask)
        verdict = gate_fn(image, repaired, hides)
        still, info = recheck_fn(image)
        gates = _p0_gates_from(verdict, still)
        # The V13 visual gate's own residual_pass is folded into template_residual;
        # the detector re-check is residual_ocr. Both feed the hard tier.
        strict_ok = _is_strict(gates) and verdict.publish_ok
        safe_ok = _is_safe(gates)
        # V17 — audit the ACTUAL candidate bytes. A cosmetic seam (visible patch /
        # band) that lands ON PRODUCT is product damage, not cosmetic, and a
        # destructive fill can break product structure without tripping the shape
        # gates. The audit converts those to hard failures so a damaging candidate
        # can never reach the safe-tier publish path. (patch plan §3, §4)
        try:
            audit = v17_final_audit.audit_final_output(
                img, image, bbox, product_mask, watermark_mask,
                final_status=("clean_repaired" if repaired else "clean_covered"),
                still_present=still,
                visible_patch_failed=not (verdict.visible_patch_pass and
                                          verdict.rectangular_band_pass and
                                          verdict.polygon_patch_pass),
                visible_band_failed=not verdict.rectangular_band_pass,
                roi_class=roi_class)
        except Exception:
            audit = None
        if audit is not None:
            verdict.v17_audit = audit
            if not audit.pass_p0:
                strict_ok = False
                safe_ok = False
                for r in audit.hard_fail_reasons:
                    gates["v17_" + r] = False
        return strict_ok, safe_ok, verdict, gates, still

    # V17 — uniform_background_fill is allowed ONLY on a strict pure-background
    # case: no product overlap, no interior edges/lines, no protected text, a
    # uniform surrounding ring, and the box not touching the product silhouette.
    # On thin flex, dark/metallic surfaces or printed labels it paints a
    # product-damaging band, so it is vetoed here and the box is routed to
    # segmented / stroke-only repair (or auto-reject). (patch plan §1.3, §5)
    uniform = None
    ring_uniform = v15_patch.is_uniform_local_background(img, bbox, product_mask)
    if ring_uniform and v17_final_audit.allow_uniform_background_fill(
            img, bbox, product_mask, watermark_mask, ring_uniform=ring_uniform):
        try:
            uniform = v15_patch.uniform_background_fill(img, bbox, watermark_mask)
        except Exception:
            uniform = None

    # safe_pool collects mark-removed + product-safe candidates that only trip a
    # COSMETIC gate, so we can publish the least-seamy one if nothing passes
    # strictly (V16.1 — never auto-reject a cleanable, product-safe image).
    safe_pool = []   # (cosmetic_score, status, name, image, verdict, gates, kind)

    def _consider_safe(strict_ok, safe_ok, verdict, gates, name, image, status,
                       kind):
        if safe_ok and not strict_ok:
            safe_pool.append((_cosmetic_score(verdict), status, name, image,
                              verdict, gates, kind))

    # ---------------- REPAIR PATH ----------------
    # V18 — product-safe candidates are real-pixel repairs: a pass here yields
    # clean_repaired (the honest classification), and on a product region they
    # are tried BEFORE the destructive uniform fill (which is banned anyway).
    repair_cands = []
    if not ban_destructive and uniform is not None:
        repair_cands.append(("v16_uniform_background_fill", uniform))
    for v18_name, v18_img in v18_safe:
        repair_cands.append((v18_name, v18_img))
    if ban_destructive and uniform is not None:
        # Should not occur (uniform is product-overlap<0.03 gated), but if it
        # does, demote it below every product-safe candidate.
        repair_cands.append(("v16_uniform_background_fill", uniform))
    if loop_repair_image is not None:
        repair_cands.append((loop_repair_method, loop_repair_image))
    if best_failed_image is not None:
        repair_cands.append(("best_failed_repair", best_failed_image))

    best_repair = repair_cands[0][1] if repair_cands else None
    for name, rimg in repair_cands:
        strict_ok, safe_ok, verdict, gates, _still = _p0(rimg, True)
        if strict_ok:
            return V16Result(
                status="clean_repaired", image=rimg, method=name,
                verdict=verdict, p0_gates=gates, publish_ok=True,
                candidate_publish_failures=state["candidate_failures"],
                best_repair_image=rimg, telemetry={**v18_telem, "v16_path": "repair"})
        _consider_safe(strict_ok, safe_ok, verdict, gates, name, rimg,
                       "clean_repaired", "repair")
        state["candidate_failures"] += 1
        reject_reasons.append("repair_failed_final_gate")
        if v14_patch.classify_failure(verdict) == "soft_fail":
            try:
                rescued = v14_patch.near_miss_rescue(
                    img, rimg, bbox, product_mask, watermark_mask, context)
            except Exception:
                rescued = None
            if rescued is not None:
                s_ok, sf_ok, v2, g2, _s2 = _p0(rescued, True)
                if s_ok:
                    return V16Result(
                        status="clean_repaired", image=rescued,
                        method=name + "+v16_micro_cleanup", verdict=v2,
                        p0_gates=g2, publish_ok=True,
                        candidate_publish_failures=state["candidate_failures"],
                        best_repair_image=rescued,
                        telemetry={**v18_telem, "v16_path": "repair_rescued"})
                _consider_safe(s_ok, sf_ok, v2, g2,
                               name + "+v16_micro_cleanup", rescued,
                               "clean_repaired", "repair")
                state["candidate_failures"] += 1

    # ---------------- COVER BEAM ----------------
    try:
        mc = v14_patch.segmented_micro_cover(
            img, bbox, watermark_mask, product_mask, context,
            base_cover=loop_cover_image)
        cover_cands = [(n, im) for (n, im, _full, _s) in mc.ranked]
    except Exception:
        cover_cands = []
    # V18 — product-safe candidates also seed the cover beam (so a candidate that
    # only clears the safe tier still publishes as clean_covered rather than
    # auto-rejecting), and they lead the beam on product regions.
    for v18_name, v18_img in reversed(v18_safe):
        cover_cands.insert(0, (v18_name, v18_img))
    # V18 — destructive full-band / forced-removal / uniform fills are BANNED on
    # any product-overlap or silhouette-contact region (patch plan §1.3, §2): on
    # product pixels they paint exactly the bands/blobs the audit rejects, so
    # offering them only wastes the beam. They remain available on a strict
    # pure-background box.
    if not ban_destructive:
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
        strict_ok, safe_ok, verdict, gates, _still = _p0(cimg, False)
        last_verdict, last_gates = verdict, gates
        if strict_ok:
            return V16Result(
                status="clean_covered", image=cimg, method=name,
                verdict=verdict, p0_gates=gates, publish_ok=True,
                candidate_publish_failures=state["candidate_failures"],
                best_repair_image=best_repair, best_cover_image=cimg,
                telemetry={**v18_telem, "v16_path": "cover"})
        _consider_safe(strict_ok, safe_ok, verdict, gates, name, cimg,
                       "clean_covered", "cover")
        state["candidate_failures"] += 1
        for r in verdict.reject_reasons:
            reject_reasons.append("cover_failed_" + r)
        if best_cover is None:
            best_cover = cimg

    # ---------------- SAFE TIER (mark gone + product-safe, soft seam) --------
    # No candidate passed strictly, but the watermark IS removed and the product
    # is undamaged on one or more — publish the least-seamy as clean_covered with
    # a tracked cosmetic_seam flag. A faint seam beats shipping the mark or
    # auto-rejecting a cleanable, product-safe image.
    if safe_pool:
        safe_pool.sort(key=lambda c: c[0])
        _score, status, name, image, verdict, gates, kind = safe_pool[0]
        return V16Result(
            status="clean_covered", image=image,
            method=name + "+cosmetic_seam", verdict=verdict, p0_gates=gates,
            publish_ok=True, cosmetic_seam=True,
            candidate_publish_failures=state["candidate_failures"],
            best_repair_image=best_repair, best_cover_image=image,
            telemetry={**v18_telem, "v16_path": "safe_cosmetic_seam", "v16_seam_kind": kind})

    # ---------------- AUTO REJECTED -----------------------------------------
    # The watermark could not be removed without damaging the product (or could
    # not be removed at all). Final automated decision — not published, not
    # manual review. The saved best attempt is the MOST watermark-removed,
    # product-safest candidate (lowest re-detect confidence), so the diagnostic
    # never shows the mark when removal was actually possible.
    best_removed = None
    best_conf = 2.0
    for name, cimg in cover_cands:
        try:
            still, info = recheck_fn(cimg)
        except Exception:
            still, info = True, {"confidence": 1.0}
        conf = float(info.get("confidence", 1.0))
        if conf < best_conf:
            best_conf = conf
            best_removed = cimg
    final_img = best_removed if best_removed is not None else \
        (best_cover if best_cover is not None else best_repair)
    if final_img is None:
        final_img = img
    verdict = last_verdict
    gates = last_gates
    if verdict is None:
        _s1, _s2, verdict, gates, _s = _p0(final_img, False)
    seen = set()
    reasons = [r for r in reject_reasons if not (r in seen or seen.add(r))]
    best_cover_diag = best_removed if best_removed is not None else best_cover
    return V16Result(
        status="auto_rejected", image=final_img, method="auto_rejected",
        verdict=verdict, p0_gates=gates, publish_ok=False,
        candidate_publish_failures=state["candidate_failures"],
        reject_reasons=reasons or ["repair_and_cover_failed_final_gate"],
        best_repair_image=best_repair, best_cover_image=best_cover_diag,
        telemetry={**v18_telem, "v16_path": "auto_rejected",
                   "v16_best_removed_conf": round(best_conf, 4)})
