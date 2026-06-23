#!/usr/bin/env python3
"""diffusion-cleaner agent — tier 3, generative inpaint (charter: agents/diffusion_cleaner/AGENT.md).

Tier 3 is the last resort: generative (Stable-Diffusion / Imagen) inpainting for hard cases where
classic (tier 1) and neural (tier 2) leave a visible scar. It is product-preserving BY CONSTRUCTION —
only the masked region is regenerated and composited back; every unmasked pixel stays byte-for-byte
the original, so the prompt cannot drift the product's shape, ports, or labels.

No diffusion backend is bundled. ``clean()`` lazily probes for one (``diffusers`` + a CUDA/MPS device);
if none is available it returns an EXPLICIT refusal — status="refused", failure_reason=
"infrastructure_missing" — rather than the old silent no-op that returned the image unchanged. An
honest refusal lets the orchestrator route to MANUAL_REVIEW on a real signal instead of mistaking an
untouched (still-watermarked) image for a clean. Install a backend and the functional path activates
with no orchestrator change (open/closed — exactly this agent's scope under the parallel-dev plan).

Boundaries (contract tests): returns a result and NEVER mutates the input; no detect/route/QA.
"""
from __future__ import annotations

import os
import time

import numpy as np

from shared.contract import CleanRequest, CleanResult, Cleaner

# Operational refusal code: this tier cannot run at all (no backend). It is deliberately NOT one of
# the product-risk FailureReason members and NOT in shared.contract.NON_RETRYABLE_FAILURE_REASONS, so
# the orchestrator treats it as escalatable (skip this dead tier); at the top of the ladder it lands
# on MANUAL_REVIEW. Matches the task contract: reason == "infrastructure_missing".
REASON_INFRA_MISSING = "infrastructure_missing"

# Verbatim from agents/diffusion_cleaner/AGENT.md § Product-Preserve Prompt Template.
PRODUCT_PRESERVE_PROMPT = (
    "Inpaint only the masked region. Preserve the product exactly as in original: same shape, "
    "color, ports, labels, and geometry. Reconstruct only background texture consistent with "
    "surroundings. NO new objects, NO new text, NO new shadows. Maintain identical resolution "
    "and framing."
)
NEGATIVE_PROMPT = ("new text, watermark, logo, letters, extra object, distorted product, blur, "
                   "artifacts, new shadow, new highlight, changed shape, changed color")

_SEED = 12345          # deterministic seed — same input+mask reproduces the same fill (charter Safety)
_MODEL_ID = os.environ.get("DIFFUSION_MODEL_ID", "runwayml/stable-diffusion-inpainting")


def _focus_xyxy(mask: np.ndarray, retry_box):
    """Focus rectangle as (x1,y1,x2,y2): the retry_box ``[x,y,w,h]`` if present (region-targeted
    retry), else the mask's own bounding box. None when the mask is empty."""
    H, W = mask.shape[:2]
    if retry_box is not None:
        x, y, w, h = (int(round(float(v))) for v in retry_box)
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class DiffusionCleaner(Cleaner):
    tier = 3
    name = "diffusion-cleaner"
    supports_retry_box = True            # focuses the generative inpaint on the validator's residual region

    def __init__(self):
        self._probed = False
        self._pipe = None
        self._device = None

    def _backend(self):
        """Lazily probe for a usable diffusion backend; cache the (possibly-None) result. Kept out of
        __init__ so building the pipeline stays cheap (contract test). Returns the pipeline or None."""
        if self._probed:
            return self._pipe
        self._probed = True
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
        except Exception:
            return None                       # diffusers / torch not installed
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device, dtype = "mps", torch.float32
        else:
            return None                       # no GPU/accelerator → treat the tier as unavailable
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(_MODEL_ID, torch_dtype=dtype).to(device)
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            return None                       # weights missing / load failed → unavailable
        self._pipe, self._device = pipe, device
        return self._pipe

    def clean(self, req: CleanRequest) -> CleanResult:
        t0 = time.time()
        mask = np.asarray(req.mask)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        pipe = self._backend()
        if pipe is None:                      # ── explicit refusal, never a silent unchanged image ──
            return CleanResult(req.image, self.tier, status="refused",
                               meta={"agent": self.name, "method": "diffusion_inpaint", "engine": None,
                                     "failure_reason": REASON_INFRA_MISSING,
                                     "note": "no diffusion backend (diffusers + CUDA/MPS) available "
                                             "— escalates to MANUAL_REVIEW"})

        # ── functional path: active only when a backend is installed ──
        import cv2
        import torch
        from PIL import Image as PILImage

        box = _focus_xyxy(mask, req.retry_box)
        if box is None:                       # nothing to inpaint — no mask pixels
            return CleanResult(req.image, self.tier, status="noop",
                               meta={"agent": self.name, "method": "diffusion_inpaint",
                                     "engine": "diffusers", "note": "empty mask"})
        H, W = mask.shape[:2]
        x1, y1, x2, y2 = box
        # pad for generative context so the model has surroundings to match (charter: consistent texture)
        px, py = max(8, (x2 - x1) // 4), max(8, (y2 - y1) // 4)
        cx1, cy1, cx2, cy2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
        crop, mcrop = req.image[cy1:cy2, cx1:cx2], mask[cy1:cy2, cx1:cx2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        gen = torch.Generator(device=self._device).manual_seed(_SEED)
        out_pil = pipe(prompt=PRODUCT_PRESERVE_PROMPT, negative_prompt=NEGATIVE_PROMPT,
                       image=PILImage.fromarray(rgb), mask_image=PILImage.fromarray(mcrop),
                       generator=gen).images[0]
        out_rgb = np.array(out_pil)
        if out_rgb.shape[:2] != crop.shape[:2]:           # keep identical resolution / framing
            out_rgb = cv2.resize(out_rgb, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # product-preserve composite: ONLY masked pixels change; the rest is the untouched original
        result = req.image.copy()
        m3 = (mcrop > 0)[..., None]
        result[cy1:cy2, cx1:cx2] = np.where(m3, out_bgr, crop)
        return CleanResult(result, self.tier, status="cleaned",
                           meta={"agent": self.name, "method": "diffusion_inpaint", "engine": "diffusers",
                                 "model": _MODEL_ID, "seed": _SEED, "device": self._device,
                                 "changed_pct": round(float((mask > 0).mean()) * 100.0, 3),
                                 "retry_box": list(req.retry_box) if req.retry_box is not None else None,
                                 "notes": ["product_preserve_prompt"],
                                 "ms": round((time.time() - t0) * 1000, 1)})
