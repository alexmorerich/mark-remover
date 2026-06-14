# Owner Logo Finder — DETECT module contract

_Authoritative spec for the production-side watermark finder (the DETECT stage of the
detect → repair → audit pipeline). Provided by Alex 2026-06-14. This is the contract the
single owned detector conforms to; it elaborates `CONTRACT_v1`'s `detect` block._

**Status / reconciliation (single owner):**
- Implemented as ONE owned module in mark-remover behind `CONTRACT_v1`. The Paseo
  `scan_audit.py` ensemble is folded in, not run as a parallel agent.
- Philosophy = "aggressive to find, conservative to publish." This is SAFE **only**
  because every repair is backed up and reversible and the Audit stage can revert any
  over-repair. That invariant is the linchpin — aggressive recall loads the audit gate.
- The graded presence classes below (CONFIRMED / LIKELY / UNCERTAIN / NO / UNSUITABLE)
  are what fix the FP-contamination found in the pre-merge audit: low-confidence OCR
  noise (e.g. `'Rcn'` conf 0.06) must grade to UNCERTAIN/NO, never CONFIRMED.
- Mask generation + `recommended_action` are PROPOSALS from detect; the Repair module
  executes the strategy. Detect never inpaints.

---

## Role

Production-side Logo Finder: find all likely SUNSKY-style watermarks before repair or
cover is attempted. Priority: (1) high recall, (2) stable mask generation, (3) good
routing for repair/cover, (4) avoid missing watermarks. False positives are acceptable
if later stages can reject or downgrade them.

## Detection philosophy

Aggressive. Detect weak, partial, low-alpha, cropped, blurred, multi-scale candidates —
not only perfect logo matches. Probable-structure signals: repeated dot-chain patterns;
SUNSKY / sunsky-online text fragments; light-gray semi-transparent strokes; center-prior
placement; repeated horizontal logo rhythm; known watermark scale ranges; known
alpha/transparency behavior; low-contrast text over product or background.

## Input
Original image; dimensions; optional product category; optional template library;
optional existing detection metadata.

## Main steps
1. **Preprocessing** — multiple normalized views: RGB, grayscale, CLAHE gray, edge map,
   high-pass, local-contrast map, alpha-like residual for faint gray text, downscaled
   fast-scan. Don't rely on one representation.
2. **Candidate region search** — likely areas first (center, lower-middle, repeated
   horizontal bands, low-contrast gray text, product/background boundary), expand to full
   image if confidence weak.
3. **Template / shape matching** — shared template library (full `sunsky-online.com`,
   `sunsky`, `online`, `.com`, dot-chain fragments, partial letters, low-alpha, scaled,
   blurred). Score by NCC, edge similarity, stroke continuity, dot-chain rhythm, baseline
   consistency, expected aspect ratio, expected alpha/grayness.
4. **Text / OCR-assisted** — OCR is optional/secondary; use when template confidence is
   borderline, image is complex, or text fragments suspected. Look for sunsky /
   sunsky-online / sunsky-online.com and partials (sky, online, .com). Not the only truth.
5. **Multi-scale** — 0.5/0.75/1.0/1.25/1.5x (+2.0x if large); merge overlapping candidates.
6. **Candidate scoring** —
   `score = template + edge + dot_chain + text_fragment + center_prior + alpha_gray +
   baseline_consistency − product_text_conflict − random_texture`.
   Classes: CONFIRMED_WATERMARK, LIKELY_WATERMARK, UNCERTAIN_WATERMARK, NO_WATERMARK,
   UNSUITABLE_IMAGE. Owner treats CONFIRMED + LIKELY as actionable; UNCERTAIN routes to
   conservative repair/cover candidate generation, not ignored.
7. **Mask generation** — per confirmed/likely: tight stroke mask, soft expanded mask,
   full text-line mask, conservative repair mask, cover fallback mask. Prefer stroke-level
   on product surface; expanded on plain background; avoid protected product text; avoid
   crossing sharp product boundaries unless necessary; preserve cables, labels, screws,
   edges, holes, ports, silhouettes.
8. **ROI classification** — plain_white, near_white, low_texture_background,
   mixed_background_product, simple_product_surface, metallic_or_reflective,
   glass_or_gradient, transparent_or_glossy, dark_product_surface, thin_flex_cable,
   complex_product_detail, text_or_label_area. Route repair by ROI class.
9. **Output contract** — structured JSON:

```json
{
  "presence": "CONFIRMED_WATERMARK | LIKELY_WATERMARK | UNCERTAIN_WATERMARK | NO_WATERMARK | UNSUITABLE_IMAGE",
  "candidates": [
    {
      "bbox": [x, y, w, h],
      "score": 0.0,
      "scale": 1.0,
      "matched_template": "sunsky-online.com",
      "evidence": {
        "template_score": 0.0, "edge_score": 0.0, "dot_chain_score": 0.0,
        "text_fragment_score": 0.0, "center_prior_score": 0.0, "alpha_gray_score": 0.0
      },
      "roi_class": "mixed_background_product",
      "risk_level": "low | medium | high",
      "recommended_action": "repair | cover | reject",
      "masks": { "stroke_mask": "...", "soft_mask": "...", "line_mask": "...", "cover_mask": "..." }
    }
  ]
}
```

## Decision policy
Strong evidence → CONFIRMED. Multiple weak signals aligned → LIKELY. Weak but visually
suspicious → UNCERTAIN. NO_WATERMARK only when no meaningful template/dot-chain/text/
alpha/center-prior evidence. Owner bias: higher recall, wider discovery, multiple repair
attempts, more candidate masks, conservative product protection. Detection ≠ final safety;
final safety belongs to the Audit Agent.
