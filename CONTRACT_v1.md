# Contract v1 — Detect → Repair → Audit

> Status: **ACTIVE** · The shared surface across the three stages. Owned **jointly**,
> versioned, lockstep. Grounded in `run_bulk.py` + `v27_clean.py`/`v28_clean.py`.

## The pipeline

```
Stage 1: Detect / Locate      Stage 2: Repair / Cover      Stage 3: Final Audit / Reject
  has_watermark? where?  ──►   strategy by ROI + risk  ──►   residual? artifact? damage?
  roi_type, risk, mask          repair → cover → reject       PASS or AUTO_REJECT (revert)
        (aggressive)                 (protective)                   (conservative)
```

Three responsibilities, one rule between them: **a later stage relies on an earlier stage only
through the record below — never by reaching into its code.** Repair does **not** re-judge
*whether* there's a mark; Audit does **not** trust that Repair succeeded — it re-checks the pixels.

## Safety model (the important change)

The zero-damage guarantee lives at the **back**, not the front:

- **Detect is aggressive** (recall-max). Over-flagging a clean image is *recoverable*, not fatal.
- **Every repair backs up the original first** (`_process_one` already does this) → fully reversible.
- **Audit is the conservative gate.** A wrongly-modified clean image, a gray patch, a damaged
  edge/flex/text, or a color shift **fails audit → `auto_rejected` → original restored from backup.**

> **Invariant (never violated): no damaged image is ever published.** Enforced at Audit,
> guaranteed reversible by backup. *Aggressive to find, conservative to publish.*

There is **no manual-review queue.** If Repair can't make it clean, it covers; if the cover is
still visible, Audit rejects it automatically. Humans are never in the per-image production loop.

---

## The record (one JSONL line per image; each stage appends its block)

```json
{
  "path": "/abs/img.jpg",
  "detect": { "has_watermark": true, "confidence": 0.92, "bbox": [120,80,460,130],
              "mask_path": "masks/img.png", "roi_type": "mixed_background_product",
              "overlap_product": 0.18, "risk": "medium", "matches": ["sunsky-online com"] },
  "repair": { "strategy": "stroke_only", "mask_final_path": "masks/img_final.png",
              "cover_method": null, "regions_changed_pct": 0.7,
              "self_status": "cleaned", "backup": "/backup/img.jpg" },
  "audit":  { "verdict": "pass", "residual": 0, "failed_checks": [], "suspected_stage": null },
  "terminal": "published"
}
```

Stages **append, never overwrite** — the full record is the audit trail that makes any failure
attributable to detect / mask / repair / audit.

### Stage 1 — `detect` block (Stage 1 → Stage 2)

| field | type | meaning |
|---|---|---|
| `has_watermark` | bool | gate; `false` ⇒ pipeline ends, `terminal="clean"` |
| `confidence` | float 0–1 | detector confidence |
| `bbox` | `[x1,y1,x2,y2]` px | coarse location (original px, top-left origin, x2/y2 exclusive) |
| `mask_path` | string | **pixel mask PNG** (glyph-tight, not a filled box) — drives a clean cut |
| `roi_type` | enum | what's under the mark (table below) |
| `overlap_product` | float 0–1 | fraction of the mark sitting on the product (vs background) |
| `risk` | enum | `low` · `medium` · `high` — derived from `roi_type` + `overlap_product` |
| `matches` | string[] ≤6 | OCR fragments that fired (audit trail) |

**`roi_type` enum:** `pure_background` · `white_bg` · `product_surface` · `metal` · `glass` ·
`flex_cable` · `text_region` · `screen_lcd` · `mixed_background_product`.

**`risk` derivation (Stage 1 owns it):** `flex_cable`/`text_region`/`screen_lcd` ⇒ **high**;
`metal`/`glass`/`product_surface` with `overlap_product` > 0.1 ⇒ **medium+**;
`pure_background`/`white_bg` with low overlap ⇒ **low**.

### Stage 2 — `repair` block (Stage 2 → Stage 3)

| field | type | meaning |
|---|---|---|
| `strategy` | enum | `clone` · `inpaint` · `cover_noise_match` · `stroke_only` · `gradient_reconstruct` · `minimal` · `refuse` |
| `mask_final_path` | string | the mask Repair actually applied (may differ from detect's) |
| `cover_method` | string\|null | set only if it fell back to covering |
| `regions_changed_pct` | float | % of image pixels altered (audit watches this) |
| `self_status` | enum | `cleaned` · `covered` · `no_mark_skip` (Repair's own detector disagreed) · `refused` |
| `backup` | string | path to the pre-edit original (revert source) |

### Stage 3 — `audit` block (terminal verdict)

| field | type | meaning |
|---|---|---|
| `verdict` | enum | `pass` · `reject` |
| `residual` | int | watermark detections still present after repair (must be 0) |
| `failed_checks` | string[] | any of: `residual_watermark` · `gray_patch` · `edge_damage` · `text_damage` · `color_shift` · `flex_break` · `processed_clean_image` |
| `suspected_stage` | enum\|null | `detect` · `mask` · `repair` — for accountability when it rejects |

### Terminal states (closed set)

| `terminal` | meaning | on-disk effect |
|---|---|---|
| `clean` | no watermark (Stage 1) | untouched |
| `published` | repaired + passed audit | cleaned image kept |
| `auto_rejected` | failed audit | **original restored from backup**; excluded from publish |

---

## Change policy

1. **Versioned `v1`.** Schema / enum / risk-rule / terminal-state changes bump to `v2`, announced.
2. **Lockstep.** All three stages adopt a new version together; the file is jointly owned.
3. **Additive-first.** New optional fields are back-compatible; new required fields / enum values are major bumps.
4. **The record is the only boundary.** No stage reaches into another's modules.
