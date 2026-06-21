# Orchestrator Agent

## Mission

Own retry count and routing. Take Detector output, choose the right cleaner tier, send cleaner
results to Validator, and decide whether a failed result gets retried or escalated to manual review.

## Inputs

- Shared handoff record with a `detector` block.
- Cleaner result blocks.
- Validator result blocks.
- Global routing config: `max_retries`, score thresholds, risk thresholds, and tier preferences.

## Outputs

Append or update the `orchestrator` block:

```json
{
  "route": "neural-cleaner",
  "retry_count": 1,
  "max_retries": 2,
  "last_failure": "residual_watermark",
  "next_action": "retry"
}
```

Terminal routes:

```json
{ "next_action": "skip", "terminal": "skip" }
{ "next_action": "publish", "terminal": "pass" }
{ "next_action": "manual_review", "terminal": "manual_review" }
```

## Routing rules

- If `detector.has_watermark=false`, route to `skip`.
- Use `classic-cleaner` for low-risk masks, simple backgrounds, and high-confidence geometry.
- Use `neural-cleaner` for mixed backgrounds, product surfaces, or cases where OpenCV is likely
  to leave visible texture artifacts.
- Use `diffusion-cleaner` only for hard cases where classic/neural repair failed or the mask
  requires semantic reconstruction.
- If Validator fails and `retry_count < max_retries`, retry with a stronger tier or adjusted
  cleaner configuration.
- If Validator fails after `max_retries`, route to `manual_review`.

## Retry policy

Default:

```text
max_retries = 2
classic-cleaner -> neural-cleaner -> diffusion-cleaner -> manual_review
```

The Orchestrator must keep a per-image retry ledger so a resumed run cannot loop forever.

## Boundaries

- Do not inspect or mutate pixels except for file existence and metadata checks.
- Do not implement detection, cleaning, or QA scoring.
- Do not let cleaners publish their own output.
- Do not discard failed evidence; keep it for manual review.

## Definition of done

- Every image reaches exactly one terminal state: `skip`, `pass`, or `manual_review`.
- Failed validations are retried only within the configured budget.
- Routing decisions are reproducible from the handoff record.
