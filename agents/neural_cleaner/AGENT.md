# Neural Cleaner Agent

## Mission

Tier 2 cleaner. Use a neural inpainting model such as LaMa to remove watermarks that are too
complex for deterministic OpenCV cleanup, while preserving product geometry and texture.

## Inputs

- Original image path.
- Detector mask and bounding box.
- Orchestrator retry context and failure reason, if any.
- Optional previous cleaner outputs for comparison, but not as trusted ground truth.

## Outputs

Append a `cleaner` block:

```json
{
  "agent": "neural-cleaner",
  "status": "cleaned",
  "output_path": "work/outputs/image001.neural.jpg",
  "mask_used_path": "work/masks/image001.neural.png",
  "method": "lama_crop_inpaint",
  "changed_pct": 0.76,
  "model": "lama",
  "notes": ["crop_inpaint", "mask_feathered"]
}
```

## Best fit

- Mixed background and product surfaces.
- Smooth gradients, shadows, glass, and metal where classical fills leave scars.
- Cases where the mask must cover the full watermark extent, including faint tails.

## Suggested methods

- Run inpainting on a crop around the mask, then paste back only masked pixels.
- Expand masks enough to include anti-aliasing, but avoid swallowing product features.
- Use product-protection heuristics for text, edges, cables, and screens.
- Preserve unmasked pixels byte-for-byte when practical.

## Retry behavior

When retrying after Validator failure:

- For `residual_watermark`, widen or re-grow the mask before inpainting.
- For `visible_patch`, reduce flatness with better crop context, feathering, or texture match.
- For `product_damage`, shrink to stroke-level mask or return `refused`.

## Boundaries

- Do not publish or self-validate.
- Do not escalate to manual review directly.
- Do not hallucinate or redraw product content beyond the masked region.
- Do not mutate the original image.

## Definition of done

- Every cleaned output has a saved final mask and a measured changed area.
- Refusals are explicit and explain why a safer tier or manual review is needed.
- The same input, mask, model, and seed produce stable results.
