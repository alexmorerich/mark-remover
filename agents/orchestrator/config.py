#!/usr/bin/env python3
"""orchestrator config — all tunables, not literals (charter: agents/orchestrator/AGENT.md).

max_retries, the QA threshold, the escalation bands, and per-tier threshold overrides live here so
the routing policy in select_tier stays pure. Owned by the orchestrator agent (routing policy); no
other agent imports it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ROIs where the surface under the mark is textured / structured / specular → start at a stronger
# tier rather than burning a tier-1 attempt unlikely to pass. Single-sourced in shared/roi.py so the
# detector (producer), this router, and the classic cleaner can never drift apart.
from shared.roi import COMPLEX_ROIS


@dataclass
class PipelineConfig:
    max_retries: int | None = None        # tier-escalation budget (rungs of the ladder). None ⇒ the FULL
    #                                       ladder (len(tiers)): registering a tier extends reach with no
    #                                       config edit (open/closed). Set an int to cap below the ladder.
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

    complex_rois: frozenset = COMPLEX_ROIS
