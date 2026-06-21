#!/usr/bin/env python3
"""integration/pipeline.py — assemble the six agents into one runnable pipeline.

This is the ONE module that imports every concrete agent (each agent itself imports only
`shared.contract`). It lives in the neutral `integration/` area, NOT inside any single agent,
so no agent directory depends on another (agents/README.md §0). The heavy backends
(torch/LaMa/easyocr) load lazily on first real use, so building the pipeline is cheap.
"""
from __future__ import annotations

from agents.classic_cleaner import ClassicCleaner
from agents.detector import Detector
from agents.diffusion_cleaner import DiffusionCleaner
from agents.neural_cleaner import NeuralCleaner
from agents.orchestrator import Orchestrator, PipelineConfig
from agents.validator import AuditValidator, Validator


def build_default_pipeline(device: str = "mps", reader=None, use_audit: bool = True,
                           config: PipelineConfig | None = None) -> Orchestrator:
    """Wire the production Orchestrator: logo_finder Detector, the three cleaner tiers, and (by
    default) the audit.py-backed Validator. Constructing this is cheap; backends load on first use."""
    cfg = config or PipelineConfig(device=device)
    detector = Detector(reader=reader)
    cleaners = {1: ClassicCleaner(), 2: NeuralCleaner(device=device), 3: DiffusionCleaner()}
    validator = AuditValidator(reader=reader, qa_threshold=cfg.qa_threshold) if use_audit \
        else Validator(cfg.qa_threshold)
    return Orchestrator(detector, cleaners, validator, cfg)
