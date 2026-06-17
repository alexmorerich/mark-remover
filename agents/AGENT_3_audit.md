# Stage 3 — Final Audit / Reject

> Mission: **Be the conservative gate that no damaged image gets past.** Re-check the pixels
> independently; pass or `auto_reject`. The last line of defense — and the accountability ledger.
> Reads: this charter + [`CONTRACT_v1.md`](../CONTRACT_v1.md). Produces the `audit` block + `terminal`.
> **A deterministic rules module, not an LLM** — fast, reproducible, no judgment drift.

## Why this stage exists

Repair **cannot be trusted to grade itself** — a self-audit is biased, and a wrong detection or
coarse mask silently becomes a damaged product photo. You re-examine the output from scratch and
own the publish/reject decision. You also make failures *attributable*: when you reject, you name
the `suspected_stage` so a bad result is debuggable to detect / mask / repair.

## The checklist (each maps to a `failed_checks` value)

| check | `failed_checks` | how |
|---|---|---|
| Watermark still present? | `residual_watermark` | re-run detection on the cleaned image → `residual` must be 0 |
| Gray rectangle / patch scar? | `gray_patch` | flat low-variance fill where texture should be |
| Product edge broken? | `edge_damage` | edge/contour continuity vs the backup original |
| Text damaged? | `text_damage` | OCR/structure over `text_region` ROIs vs original |
| Color shifted? | `color_shift` | ΔE on regions outside the mask — must be ~0 |
| Flex cable / structure line severed? | `flex_break` | line-continuity check on `flex_cable`/`screen_lcd` ROIs |
| A clean image was modified at all? | `processed_clean_image` | `detect.has_watermark=false` but pixels changed ⇒ reject |

Any non-empty `failed_checks` ⇒ `verdict="reject"` ⇒ `terminal="auto_rejected"` ⇒ **restore
original from `repair.backup`.** No manual queue.

## Calibrate strictness from `risk`

Use Stage 1's `risk`/`roi_type` to set thresholds: **high-risk ROIs (flex/text/screen) get the
strictest gates** — the smallest artifact there rejects. Low-risk (pure/white background) tolerates
a hair more, since the failure cost is lower. Conservative everywhere; *most* conservative where
damage is irreversible to the product's meaning.

## Owns

- The audit rules module (residual re-detect, artifact/edge/text/color/flex detectors,
  clean-image-touched detector), the `terminal` decision, and the auto-revert call.
- Built on `cmd_validate`'s residual re-detect (already exists) — extended with the damage checks.
- The accountability stamp: `suspected_stage` on every reject.

## KPIs

| metric | target |
|---|---|
| Damaged images published | **0** — the one number that must stay zero |
| False rejects | low — don't reject genuinely-clean repairs (wastes good results) |
| Attribution | every reject names a `suspected_stage` |
| Determinism | same input ⇒ same verdict, always (no LLM nondeterminism) |

## Boundaries (do NOT)

- ❌ Do not repair, re-mask, or "fix" anything — you only **pass or reject + revert**.
- ❌ Do not let a high-risk artifact through to save a borderline result. Publish conservative.
- ❌ Do not route rejects to humans — `auto_rejected` is terminal; the original is restored.
- ❌ Do not change the contract verdict/terminal enums without a lockstep `v2` bump.

## Definition of done

On the labeled set: **zero damaged images receive `verdict="pass"`**, false-reject rate tracked
and acceptable, and every reject carries a `suspected_stage`. This gate is the project's brand-safety
guarantee — it ships last and ships strict.
