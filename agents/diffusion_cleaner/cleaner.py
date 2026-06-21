#!/usr/bin/env python3
"""diffusion-cleaner agent — tier 3, generative inpaint (charter: agents/diffusion_cleaner/AGENT.md).

STUB: no diffusion backend is installed. It satisfies the Cleaner interface so the escalation
ladder and the open/closed registry are real and unit-tested, but performs a last-resort no-op:
it returns the image unchanged, the validator fails it, and the orchestrator routes the image to
MANUAL_REVIEW. Drop a real Stable-Diffusion / Imagen inpaint into `clean()` to activate the tier —
no orchestrator change required (that is exactly this agent's scope under the parallel-dev plan).

Boundaries (contract tests): returns a result and NEVER mutates the input; no detect/route/QA.
"""
from __future__ import annotations

from shared.contract import CleanRequest, CleanResult, Cleaner


class DiffusionCleaner(Cleaner):
    tier = 3
    name = "diffusion-cleaner"

    def clean(self, req: CleanRequest) -> CleanResult:
        return CleanResult(req.image, self.tier, status="noop",
                           meta={"agent": self.name, "method": "noop_stub", "engine": None,
                                 "note": "diffusion backend not installed — escalates to MANUAL_REVIEW"})
