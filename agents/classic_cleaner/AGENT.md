# Classic Cleaner Agent

## Mission

Tier 1 cleaner. Remove straightforward watermarks using deterministic OpenCV-style methods:
mask cleanup, inpainting, cloning, edge-aware blending, and local texture/noise matching.

## Inputs

- Original image path.
- Detector `mask_path`, `bbox`, `watermark_type`, and `score`.
- Orchestrator cleaner configuration for this attempt.

## Outputs

Append a `cleaner` block:

```json
{
  "agent": "classic-cleaner",
  "status": "cleaned",
  "output_path": "work/outputs/image001.classic.jpg",
  "mask_used_path": "work/masks/image001.classic.png",
  "method": "opencv_telea_inpaint",
  "changed_pct": 0.42,
  "notes": []
}
```

If the case is outside this agent's safe range:

```json
{
  "agent": "classic-cleaner",
  "status": "refused",
  "reason": "mask_over_product_detail"
}
```

## Best fit

- White or plain backgrounds.
- Low-risk masks.
- Thin semi-transparent text over simple gradients.
- Small marks where local surrounding texture is reliable.

## Suggested methods

- Morphological mask cleanup and dilation for halos.
- `cv2.inpaint` with Telea/Navier-Stokes.
- Patch clone from neighboring clean regions.
- Local noise re-injection to avoid flat rectangles.
- Edge-aware feathering after repair.

## Safety rules

- Keep changes local to the mask plus minimal feather.
- Refuse if the mask overlaps fragile product text, PCB traces, flex cables, or detailed edges.
- Preserve dimensions, metadata-critical orientation, and color profile when possible.
- Never overwrite the original input.

## Boundaries

- Do not call neural or diffusion models directly.
- Do not decide pass/fail; the Validator owns QA.
- Do not retry yourself; return one result for the Orchestrator to route.

## Failure Codes

Set `CleanResult.meta["failure_reason"]` to one of the following `FailureReason` enum members
(imported from `shared.contract`) when refusing a request. The Orchestrator reads this value to
decide whether to escalate or terminate.

```python
from shared.contract import FailureReason

# mask intersects a labeled product-detail region
FailureReason.MASK_OVER_PRODUCT_DETAIL   # "mask_over_product_detail"

# background is textured/metallic — Telea/NS fill would fail QA fidelity
FailureReason.BACKGROUND_TOO_COMPLEX     # "background_too_complex"

# mask covers too large a proportion of the image area; context window unreliable
FailureReason.MASK_TOO_LARGE             # "mask_too_large"  (shared with Neural Tier 2)

# general-purpose bail-out for any other unsafe condition
FailureReason.UNSAFE_FOR_CLASSIC         # "unsafe_for_classic"
```

Refused `CleanResult` shape:
```json
{
  "agent": "classic-cleaner",
  "status": "refused",
  "failure_reason": "mask_over_product_detail"
}
```

## Definition of done

- Output image exists when `status=cleaned`.
- `changed_pct` is measured against the original.
- The mask used by the cleaner is saved for Validator and manual-review diagnostics.


# Classic Cleaner Agent

## Mission
Tier 1 cleaner. Remove straightforward watermarks using deterministic OpenCV-style methods.

## Tier Contract
- **Tier:** 1 (Fast-path)
- **Scope:** Simple backgrounds, low-risk masks, thin semi-transparent text.
- **Constraints:** Cannot handle complex textures, glass, metal, or high-density product details.

## Inputs
- `original_image`: path
- `mask_path`, `bbox`, `watermark_type`, `score`: from Detector
- `retry_context`: (if any)

## Outputs (Status)
Append a `cleaner` block:
```json
{
  "agent": "classic-cleaner",
  "status": "cleaned",
  "output_path": "...",
  "method": "opencv_telea_inpaint",
  "changed_pct": 0.42
}
