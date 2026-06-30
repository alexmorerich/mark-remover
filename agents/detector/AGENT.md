# Detector Agent (Finder)

> **Version:** 2.0
> **Status:** Authoritative — long-term architectural contract for the Finder/Detector agent.
> **Last updated:** 2026-06-30
>
> This document is the authoritative source of the Finder's behavior across future development.
> "Detector" and "Finder" name the same agent. Project-specific detection rules elaborated in
> [`AGENT_1_detect.md`](../AGENT_1_detect.md) and [`docs/OWNER_LOGO_FINDER.md`](../../docs/OWNER_LOGO_FINDER.md)
> remain in force except where they conflict with the architectural contracts below.

## Role — Evidence Collector, not final classifier

The Finder is an **Evidence Collector**. Its job is to gather and forward every credible signal
that an image *might* carry a watermark — not to render the final verdict. The decision of whether
an image is truly clean, and whether to publish, belongs downstream. The Finder is deliberately
biased toward recall: it would rather forward a doubtful candidate than silently drop a real mark.

## Responsibility separation

Each stage owns exactly one concern. The Finder must never absorb another stage's concern.

| Agent | Owns |
|---|---|
| **Finder (this agent)** | **Recall** — collect all credible evidence; never miss a watermark |
| **Validator** | **Precision** — confirm or reject Finder candidates; suppress false positives |
| **Repair Engine** | **Watermark removal** — execute the cleaning strategy |
| **QA** | **Publication quality** — pass/fail the final published output |

## Inputs

- Original image path.
- Optional detection configuration: model thresholds, known watermark priors, canonical templates.
- Optional previous retry context from the Orchestrator, used only to widen or refine detection.

## Non-negotiable behavioral contracts

These are invariants. They hold regardless of thresholds, scores, or future detectors.

1. **Never discard credible evidence.** Any signal that survives basic sanity is carried forward.
2. **Weak evidence is still evidence.** Low-confidence does not mean "ignore."
3. **Multiple weak signals must accumulate.** Several independent weak cues compound into a
   reportable candidate; they are not each dismissed in isolation.
4. **Score is advisory only.** The numeric score informs routing but must **never** override or
   erase evidence. A low score with non-empty evidence is still a candidate.
5. **Missing OCR is not evidence of absence.** Faint, semi-transparent, or styled marks routinely
   defeat OCR. OCR silence proves nothing.
6. **Missing template correlation is not evidence of absence.** A mark may be cropped, scaled, or
   novel. No template hit proves nothing.
7. **When uncertain, preserve the candidate and forward it downstream.** Uncertainty routes to the
   Validator/forensic path — never to silent suppression.

## Evidence Model

Evidence is the Finder's primary product. Each piece of evidence is an explainable record:

```json
{ "source": "ocr", "detail": "sunsky", "strength": 0.31, "bbox": [120, 80, 460, 130] }
```

- `source` — which detector produced it (must be a registered detector; see Evidence Registry).
- `detail` — human-readable explanation (token read, template id, prior name, etc.).
- `strength` — normalized `0.0`–`1.0` confidence from that single detector. Weak strengths are
  retained, not dropped.
- `bbox` — region the evidence localizes, when available.

### Evidence Registry (extensible)

Detectors are registered, not hard-coded, so new detection methods can be added without rewriting
routing logic. Every registered detector contributes evidence to the same model, and routing
reasons over the union.

- Current registered detectors: OCR (multi-pass / CLAHE), template / canonical correlation,
  positional / center prior, structural (V2) detector, local-contrast & stroke masks.
- Future detectors (e.g. learned classifiers, frequency-domain probes) register the same way and
  participate automatically. **No registered detector may be ignored when deciding NO_WATERMARK.**

## Outputs

Append the `detector` block to the shared handoff record. Always include the routing decision and
the full evidence list.

```json
{
  "has_watermark": true,
  "routing": "AUTO_CLEAN",
  "mask_path": "work/masks/image001.png",
  "bbox": [120, 80, 460, 130],
  "watermark_type": "semi_transparent_text",
  "score": 0.93,
  "evidence": [
    { "source": "ocr", "detail": "sunsky", "strength": 0.88, "bbox": [120, 80, 460, 130] },
    { "source": "center_prior", "detail": "canonical band", "strength": 0.62 },
    { "source": "template_match", "detail": "sunsky_v3", "strength": 0.71 }
  ],
  "risk_hint": "medium"
}
```

## Routing rules

The Finder emits one of three routes. Routing reasons over **evidence**, not the score alone.

- **AUTO_CLEAN** — strong, well-localized, low-risk evidence (e.g. confident OCR/template hit over a
  clear ROI). Hand a usable mask to the Repair Engine.
- **FORENSIC_REQUIRED** — any credible-but-uncertain state: weak/partial evidence, accumulated weak
  signals, conflicting detectors, faint marks over busy product, or a localized region with no OCR.
  This is the **default when uncertain**. Routes to the Validator/forensic path for precision.
- **NO_WATERMARK** — reserved and intentionally difficult to reach.

### NO_WATERMARK is intentionally hard to reach

`NO_WATERMARK` is valid **only when every registered detector reports no evidence** — an empty
evidence union across the entire Evidence Registry. If *any* registered detector contributes even
weak evidence, the route is FORENSIC_REQUIRED, not NO_WATERMARK. Missing OCR alone, or a missing
template hit alone, can never justify NO_WATERMARK.

### Required output metadata for FORENSIC_REQUIRED

When routing FORENSIC_REQUIRED, the record must additionally carry enough context for the Validator
to do precision work without re-deriving detection:

- `evidence` — the full, non-empty list of contributing signals (never truncated).
- `bbox` / `mask_path` — best available localization, even if coarse or low-confidence.
- `watermark_type` — best guess, or `unknown_mark`.
- `score` — advisory only; must not be used to suppress the candidate.
- `risk_hint` — `low` / `medium` / `high`, derived from ROI + product overlap.
- `uncertainty_reason` — why this is uncertain (e.g. `faint_over_product`, `ocr_silent_prior_hit`,
  `conflicting_detectors`, `accumulated_weak_signals`).

## Responsibilities

- Maximize recall while keeping evidence explainable.
- Produce a mask that covers the watermark glyphs and anti-aliased halo.
- Classify the watermark type, for example `semi_transparent_text`, `opaque_logo`,
  `corner_stamp`, or `unknown_mark`.
- Produce a normalized advisory score from `0.0` to `1.0`.
- Route per the rules above; reserve NO_WATERMARK for the all-detectors-silent case.

## Forbidden behaviors

These must never occur:

- **Never** discard, truncate, or down-weight evidence to make a cleaner verdict.
- **Never** let the score override or erase evidence.
- **Never** treat missing OCR or missing template correlation as proof of a clean image.
- **Never** emit NO_WATERMARK while any registered detector holds evidence.
- **Never** suppress an uncertain candidate instead of forwarding it (no silent skips).
- **Never** take over a downstream concern: do not modify pixels, do not select or run a cleaner,
  do not retry/escalate to manual review, do not pass or fail the final published output.

## Design philosophy

The Finder **maximizes recall, not precision.** This is safe only because the pipeline is layered:
every candidate is validated downstream, every repair is backed up and reversible, and QA gates
publication. Over-collecting evidence is cheap and recoverable; a missed watermark is not. When the
Finder is wrong, it should be wrong by forwarding too much — never by staying silent. Precision is
the Validator's job; the Finder's contribution to the whole system is to **guarantee nothing real
is lost before validation begins.**

## Suggested implementation

- Combine OCR, template matching, positional priors, local contrast maps, and stroke masks; register
  each as an Evidence Registry source.
- Merge overlapping detections into one dominant candidate unless multiple independent marks exist.
- Save debug overlays for low-score or high-impact cases.
- Keep mask coordinates in the original image pixel space.

## Definition of done

- Detection emits deterministic records for the same input and config.
- Every positive detection has `mask_path`, `bbox`, `watermark_type`, `score`, and non-empty `evidence`.
- Every FORENSIC_REQUIRED record carries the full required metadata above.
- `NO_WATERMARK` is emitted only with an empty evidence union across all registered detectors, and
  leaves downstream agents enough context to audit why it was reached.
