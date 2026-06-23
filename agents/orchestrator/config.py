#!/usr/bin/env python3
"""orchestrator config — all tunables, not literals (charter: agents/orchestrator/AGENT.md).

max_retries, the QA threshold, the escalation bands, and per-tier threshold overrides live here so
the routing policy in select_tier stays pure. Owned by the orchestrator agent (routing policy); no
other agent imports it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ROIs where the surface under the mark is textured / structured / specular — start at a stronger
# tier rather than burning a tier-1 attempt unlikely to pass.
COMPLEX_ROIS = (
    "metal", "metallic_or_reflective", "glass", "glass_or_gradient", "transparent_or_glossy",
    "thin_flex_cable", "flex_cable", "screen_lcd", "complex_product_detail",
    "dark_product_surface", "mixed_background_product",
)
SIMPLE_ROIS = (
    "white_bg", "white_background", "plain_white", "near_white", "near_white_background",
    "pure_background", "low_texture_background", "simple_product_surface",
)


@dataclass
class PipelineConfig:
    max_retries: int = 3                  # tier-escalation budget: how many rungs of the ladder may be tried
    max_intra_tier_retries: int = 2       # Phase-1 local re-inpaint budget at the CURRENT tier (per tier),
    #                                       spent before escalating when the validator returns a retry_box
    qa_threshold: float = 0.70            # global QA gate; passed = qa_score >= this
    device: str = "mps"

    # select_tier policy
    start_tier_simple: int = 1            # high score + simple type → tier 1
    start_tier_complex: int = 2           # complex / textured / large → start at tier 2
    simple_score_floor: float = 0.55      # below this detector confidence → treat as harder
    large_area_frac: float = 0.18         # mask larger than this frac of the image → harder
    qa_band_jump: float = 0.30            # very low qa_score on a retry → jump to strongest tier

    tier_qa_threshold: dict = field(default_factory=dict)   # per-tier overrides; can only TIGHTEN

    complex_rois: tuple = COMPLEX_ROIS
    simple_rois: tuple = SIMPLE_ROIS

    def qa_threshold_for(self, tier: int) -> float:
        return self.tier_qa_threshold.get(tier, self.qa_threshold)
