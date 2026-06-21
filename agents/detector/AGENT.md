# Detector Agent

## Mission

Find whether an image contains a watermark. If it does, emit a usable mask, location, type, and
confidence score for downstream routing. If it does not, mark the image as `skip`.

## Inputs

- Original image path.
- Optional detection configuration: model thresholds, known watermark priors, canonical templates.
- Optional previous retry context from the Orchestrator, used only to widen or refine detection.

## Outputs

Append the `detector` block to the shared handoff record:

```json
{
  "has_watermark": true,
  "mask_path": "work/masks/image001.png",
  "bbox": [120, 80, 460, 130],
  "watermark_type": "semi_transparent_text",
  "score": 0.93,
  "evidence": ["ocr:sunsky", "center_prior", "template_match"],
  "risk_hint": "medium"
}
```

When no watermark is found:

```json
{
  "has_watermark": false,
  "score": 0.04,
  "evidence": [],
  "risk_hint": "none"
}
```

## Responsibilities

- Maximize recall while keeping evidence explainable.
- Produce a mask that covers the watermark glyphs and anti-aliased halo.
- Classify the watermark type, for example `semi_transparent_text`, `opaque_logo`,
  `corner_stamp`, or `unknown_mark`.
- Produce a normalized score from `0.0` to `1.0`.
- Mark no-watermark images clearly so the Orchestrator can route them to `skip`.

## Suggested implementation

- Combine OCR, template matching, positional priors, local contrast maps, and stroke masks.
- Merge overlapping detections into one dominant candidate unless multiple independent marks exist.
- Save debug overlays for low-score or high-impact cases.
- Keep mask coordinates in the original image pixel space.

## Boundaries

- Do not modify image pixels.
- Do not select a cleaner.
- Do not retry or escalate to manual review.
- Do not pass or fail the final output.

## Definition of done

- Detection emits deterministic records for the same input and config.
- Every positive detection has `mask_path`, `bbox`, `watermark_type`, and `score`.
- Every no-watermark decision sets `has_watermark=false` and leaves downstream agents enough
  evidence to audit why it skipped.
