# Validator Agent

## Mission

Independently check whether a cleaner output is good enough to pass. The Validator is the QA gate:
it does not repair images and does not trust a cleaner's self-assessment.

## Inputs

- Original image path.
- Cleaner output path.
- Detector mask and bounding box.
- Cleaner metadata: agent, method, changed area, mask used.
- Orchestrator retry context.

## Outputs

Append a `validator` block:

```json
{
  "verdict": "pass",
  "qa_score": 0.98,
  "failed_checks": [],
  "residual_score": 0.01,
  "artifact_score": 0.02,
  "damage_score": 0.00
}
```

Failure example:

```json
{
  "verdict": "fail",
  "qa_score": 0.62,
  "failed_checks": ["residual_watermark", "visible_patch"],
  "residual_score": 0.47,
  "artifact_score": 0.31,
  "damage_score": 0.04,
  "recommended_retry": "neural-cleaner"
}
```

## Checks

| Check | Failure key | Purpose |
|---|---|---|
| Residual detector | `residual_watermark` | Re-detect watermark on cleaner output. |
| Patch scar detector | `visible_patch` | Catch flat fills, gray bands, or unnatural texture. |
| Edge preservation | `edge_damage` | Compare product contours against the original. |
| Text preservation | `text_damage` | Catch damaged product labels or protected text. |
| Color consistency | `color_shift` | Detect changes outside the repair region. |
| Structure consistency | `structure_damage` | Protect cables, screens, ports, and fine product details. |

## Scoring rule

- `verdict=pass` only when residual, artifact, and damage scores are all below thresholds.
- Thresholds should be strictest for high-risk masks and product-detail overlaps.
- Any protected text or product-structure damage must fail, even if the watermark is gone.

## Boundaries

- Do not modify output pixels.
- Do not call cleaners directly.
- Do not increment retry counters.
- Do not approve a result just because it is the best available attempt.

## Definition of done

- Every validation emits `verdict`, `qa_score`, and `failed_checks`.
- Failures explain whether the suspected issue is residual watermark, visible artifact, or product
  damage.
- The same original/output pair receives the same verdict under the same config.
