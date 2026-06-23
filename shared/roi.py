#!/usr/bin/env python3
"""shared/roi.py — the ONE canonical ROI-class vocabulary the agents share.

`roi_type` (logo_finder calls it `roi_class`) names the surface UNDER the mark. The detector
produces it; the orchestrator routes on it (white/simple → tier 1, textured/complex → start at
tier 2); the classic cleaner picks its engine by it. Before this module those three kept private,
drifting copies — config even listed classes the finder never emits. Centralising the vocabulary
here (one import for everyone) stops the drift, and a contract test asserts the detector only ever
emits a class named here.

Sets are frozensets of lower-case strings; callers compare ``(roi_type or "").lower()`` against them.
"""
from __future__ import annotations

# White / near-white / very-low-texture surfaces: classic Tier-1 structure-preserving recovery is
# safe and best here. Superset of the classes logo_finder emits plus forward-compatible synonyms.
WHITE_ROIS: frozenset = frozenset({
    "plain_white", "near_white", "low_texture_background",
    "white_bg", "white_background", "near_white_background", "pure_background",
})

# Textured / structured / specular / dark surfaces. Plain Telea inpainting SCARS these (the
# historical product-damage path), so the orchestrator starts them at a stronger tier and the
# classic cleaner REFUSES them unless a texture-safe (glyph-tight) engine is available.
TEXTURED_ROIS: frozenset = frozenset({
    "dark_product_surface", "complex_product_detail", "thin_flex_cable",
    "mixed_background_product", "metallic_or_reflective",
    # forward-compatible synonyms (not currently emitted by logo_finder._roi_class)
    "metal", "glass", "glass_or_gradient", "transparent_or_glossy", "flex_cable", "screen_lcd",
})

# Non-white but low-risk surfaces classic may still inpaint with plain Telea.
TELEA_SAFE_ROIS: frozenset = frozenset({"simple_product_surface"})

# Orchestrator routing: ROIs that should START at a stronger tier than 1.
COMPLEX_ROIS: frozenset = TEXTURED_ROIS

# Every roi_class logo_finder._roi_class can return. The contract test asserts the detector never
# emits a class outside this set (so routing / cleaning never silently mis-handle an unknown one).
FINDER_EMITTED_ROIS: frozenset = frozenset({
    "plain_white", "near_white", "low_texture_background", "simple_product_surface",
    "complex_product_detail", "dark_product_surface", "mixed_background_product",
    "thin_flex_cable", "metallic_or_reflective",
})
