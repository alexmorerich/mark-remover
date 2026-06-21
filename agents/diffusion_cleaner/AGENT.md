# Diffusion Cleaner Agent

## Mission

Tier 3 cleaner. Use diffusion or other generative inpainting only for difficult cases where
classic and neural repair cannot meet quality requirements, and where semantic reconstruction
is safer than leaving a visible patch.

## Inputs

- Original image path.
- Detector mask, bounding box, watermark type, and score.
- Orchestrator retry context, including previous failure reasons.
- Optional prompt/config templates approved for product-preserving inpainting.

## Outputs

Append a `cleaner` block:

```json
{
  "agent": "diffusion-cleaner",
  "status": "cleaned",
  "output_path": "work/outputs/image001.diffusion.jpg",
  "mask_used_path": "work/masks/image001.diffusion.png",
  "method": "diffusion_inpaint",
  "changed_pct": 1.2,
  "model": "sd_or_imagen",
  "seed": 12345,
  "notes": ["product_preserve_prompt"]
}
```

If generation would alter product identity:

```json
{
  "agent": "diffusion-cleaner",
  "status": "refused",
  "reason": "semantic_product_risk"
}
```

## Best fit

- Complex backgrounds where LaMa leaves visible scars.
- Large or opaque marks that require plausible reconstruction.
- Non-critical background areas where generative variation is acceptable.

## Prompting constraints

- Preserve the original product, shape, labels, ports, edges, color, and texture.
- Inpaint only the masked watermark area.
- Avoid adding new logos, text, shadows, highlights, labels, or product details.
- Keep output resolution and framing identical to the input.

## Safety rules

- Use deterministic seeds and save model/config metadata.
- Compare generated output against the original outside the mask.
- Refuse high-risk product-detail masks instead of inventing missing product structure.
- Send every output to Validator; never self-approve.

## Boundaries

- Do not run as the default cleaner.
- Do not change routing or retry count.
- Do not replace the original input.
- Do not remove real product markings or labels.

## Definition of done

- Output includes seed, model, prompt/config reference, and mask path.
- Unmasked regions remain visually and structurally consistent with the original.
- Refusal is preferred over product hallucination.
