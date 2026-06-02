#!/usr/bin/env python3
"""V23 — failure-reason-driven retry ladder (patch plan §Patch 7).

The candidate bank should not blindly try a fixed list; it should REACT to why
the previous candidates failed. This module is the pure routing brain:
:func:`route_retry` maps the accumulated failure reasons to the next candidate
families to try and to a set of families that must be REMOVED from the pool
(never retried). It changes no pixels and makes no publish decision — the
pipeline still runs the unchanged P0 audit on whatever the ladder produces.

The hard safety rule it encodes (patch plan §Patch 7, §16.3): once a candidate
fails because a cover/fill landed on product, the entire cover/fill family is
removed from the pool for that image — a cover is NEVER retried on product.
"""
from __future__ import annotations

COVER_FILL_FAMILY = frozenset({
    "uniform_background_fill", "v16_uniform_background_fill",
    "forced_removal_fill", "v16_forced_removal", "full_bbox_cover",
    "segmented_micro_cover", "white_fill", "ring_median_full_band",
})

# Reverse-alpha / non-destructive families that are always safe to retry.
REVERSE_ALPHA_FAMILY = frozenset({
    "reverse_alpha", "reverse_alpha_variants", "component_aware_mixed_repair",
    "thin_flex_line_restore", "stroke_only_micro_cleanup", "baseline_cleanup",
})


def _blob(reasons, hard):
    return " ".join(str(r) for r in (list(reasons or []) + list(hard or [])))


def route_retry(reject_reasons, hard_fail_reasons=None) -> dict:
    """Return ``{"families": [...], "ban_families": set(), "ban_cover": bool}``.

    ``families`` are the candidate families to (re)try, ordered; ``ban_families``
    are families to drop from the pool; ``ban_cover`` is ``True`` when any cover/
    fill must never be retried for this image.
    """
    blob = _blob(reject_reasons, hard_fail_reasons)
    families: list = []
    ban: set = set()
    ban_cover = False

    def add(*fams):
        for f in fams:
            if f not in families:
                families.append(f)

    # cover_or_fill_on_product → remove ALL cover/fill candidates (never retry).
    if any(t in blob for t in ("cover_on_product", "visible_patch_on_product",
                               "wedge", "slab", "visible_band_on_product")):
        ban |= set(COVER_FILL_FAMILY)
        ban_cover = True

    # residual / partial-glyph residue → micro cleanup ladder.
    if any(t in blob for t in ("partial_glyph_residue", "alpha_footprint_residue",
                               "residual_watermark", "dot_chain",
                               "low_contrast_glyph")):
        add("baseline_cleanup", "residue_micro_component_inpaint",
            "residue_micro_surface_blur", "cardboard_texture_reinject")

    # visible band on non-white surface → reverse-alpha / mixed / stroke only.
    if "visible_band_on_nonwhite" in blob:
        add("reverse_alpha_variants", "component_aware_mixed_repair",
            "stroke_only_micro_cleanup")
        ban |= set(COVER_FILL_FAMILY)
        ban_cover = True

    # product damage / silhouette / thin flex → line restore + reverse-alpha.
    if any(t in blob for t in ("product_damage", "silhouette", "thin_flex",
                               "flex")):
        add("thin_flex_line_restore", "reverse_alpha", "reverse_alpha_variants")
        ban |= set(COVER_FILL_FAMILY)
        ban_cover = True

    # protected text overlap → reverse-alpha ONLY (no inpaint/clone/cleanup).
    if "protected_text" in blob:
        families = ["reverse_alpha"]
        ban |= set(COVER_FILL_FAMILY)
        ban |= {"stroke_only_micro_cleanup", "residue_micro_component_inpaint",
                "baseline_cleanup", "cardboard_texture_reinject"}
        ban_cover = True

    return {"families": families, "ban_families": ban, "ban_cover": ban_cover}


def filter_cover_pool(cover_cands, ban_families):
    """Drop any candidate whose name matches a banned family (patch plan
    §Patch 7). ``cover_cands`` is a list of ``(name, image)``."""
    if not ban_families:
        return cover_cands
    out = []
    for name, img in cover_cands:
        n = str(name)
        if any(fam in n for fam in ban_families):
            continue
        out.append((name, img))
    return out
