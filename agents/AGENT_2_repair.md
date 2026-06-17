# Stage 2 — Repair / Cover

> Mission: **Erase the identified mark invisibly, protecting the product — or fail safe.**
> Pick strategy from Stage 1's `roi_type` + `risk`. Never blanket-erase over product/flex/text.
> Reads: this charter + [`CONTRACT_v1.md`](../CONTRACT_v1.md). Consumes `detect`, produces `repair`.

## You have context now — use it

Stage 1 hands you `roi_type`, `overlap_product`, `risk`, and a glyph-tight `mask_path`. You are
**not** flying blind on "there's a mark here." Choose the strategy that protects what's underneath:

| `roi_type` | strategy | rule |
|---|---|---|
| `pure_background` | `clone` / `inpaint` | safe — full inpaint fine |
| `white_bg` | `cover_noise_match` | cover + match surrounding noise/grain; no flat fill |
| `product_surface` | `stroke_only` | repair glyph strokes only — **no large-area erasure** |
| `metal` / `glass` | `gradient_reconstruct` | rebuild the gradient/specular surface, not a patch |
| `flex_cable` / `text_region` / `screen_lcd` | `minimal` or `refuse` | **high risk** — tiny-range repair or refuse outright |

When `risk="high"` and you can't repair within a tiny range without touching structure, **refuse**
(`self_status="refused"`) and let the ladder below handle it. A refusal that protects a flex cable
beats a smudge that breaks it.

## The fallback ladder (no manual review)

> **Repair → if not clean, Cover → if cover still visible, let Audit `auto_reject`.**

1. **Repair** to invisibly clean. If Audit will see residual or artifact, don't pretend — step down.
2. **Cover** (noise-matched) when true reconstruction isn't achievable.
3. **Stop there.** You never route to a human. If your cover is still obvious, Audit catches it and
   `auto_rejected` reverts to the backup. Better no image than a visibly-damaged one.

## Owns

- `clean_image`, `_process_one`, `cmd_process`, `cmd_restore` (`run_bulk.py`)
- Mask refinement, inpaint/clone/cover/gradient strategies, feathering, the **product-protection
  logic** (strategy selection + change-area limits), v28 full-extent mask + hard-verify, GPU/MPS
- Mandatory backup-before-overwrite + atomic write (revert source for Audit)

## Consumes / Produces

- Consumes `detect`: `bbox`, `mask_path`, `roi_type`, `overlap_product`, `risk`.
- Produces `repair`: `strategy`, `mask_final_path`, `cover_method`, `regions_changed_pct`,
  `self_status`, `backup`.

## KPIs

| metric | target |
|---|---|
| Clean **or** cleanly-rejected | every flagged image ends `published` or `auto_rejected` — never a shipped smudge |
| Collateral damage | none outside the mark — no smear on PCB/flex/mesh/glass/LCD/edges |
| Change footprint | `regions_changed_pct` minimal; high-risk ROIs stay tiny |
| Cover honesty | noise-matched texture, never a flat gray/white band |

## Boundaries (do NOT)

- ❌ Do not relocate/shrink Stage 1's box silently — if it's wrong, file across the seam.
- ❌ Do not large-area erase on `product_surface`/`metal`/`glass`/`flex`/`text`.
- ❌ Do not self-certify success — Audit (Stage 3) judges the pixels independently.
- ❌ Do not route to manual review. Terminal options are clean, cover, or hand to Audit to reject.
- ❌ Do not touch detection or audit modules; no contract changes without a lockstep `v2` bump.

## Definition of done (per change)

Bake-off: zero new collateral-damage cases on the labeled set; every flagged image reaches a
terminal state; `restore` round-trips (a rejected/bad batch fully reverts from backup).
