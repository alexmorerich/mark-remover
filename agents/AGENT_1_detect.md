# Stage 1 — Detect / Locate

> Mission: **Find every watermark and describe its context richly enough that Repair and Audit
> can protect the product.** Aggressive on recall — the Audit gate + backup make over-flagging
> recoverable.
> Reads: this charter + [`CONTRACT_v1.md`](../CONTRACT_v1.md). Produces the `detect` block.

## What you decide (and must output — structured)

Not just "is there a mark" — the full context that downstream needs:

- `has_watermark`, `confidence`
- `bbox` **and** a glyph-tight `mask_path` (a pixel mask PNG, **not** a filled rectangle — a coarse
  box is what turns into gray blocks at Stage 2)
- `roi_type` — what's under the mark: `pure_background` · `white_bg` · `product_surface` · `metal`
  · `glass` · `flex_cable` · `text_region` · `screen_lcd` · `mixed_background_product`
- `overlap_product` — fraction of the mark sitting on the product vs background
- `risk` — `low/medium/high`, derived from `roi_type` + `overlap_product`
- `matches` — OCR fragments, as audit trail

**The risk/ROI fields are the heart of this stage.** Repair is blind without them; Audit calibrates
its strictness from them. A correct `bbox` with a wrong `roi_type` is a failure.

## Be aggressive — recall is the job

Detection over white/clean backgrounds is easy; the misses are **faint/semi-transparent marks over
busy product** where easyOCR's CRAFT never localizes them. Push recall: low thresholds, multi-pass
OCR, V2 structural detector, **positional prior** (the Sunsky mark sits in a consistent place — check
that ROI even when OCR can't read it), and ensemble union. Over-flagging a clean image is **not
fatal** — Audit catches `processed_clean_image` and reverts. So optimize recall, stay honest on `risk`.

> Recall is a *pipeline* property, not an OCR one. The lever for the faint tail is the positional
> prior, not louder OCR.

## Owns

- `_scan_one`, `cmd_scan`, overlays (`run_bulk.py`)
- OCR: `detect_watermark_boxes`, `_ocr_pass`, `SUNSKY_TOKENS`, CLAHE passes (`v27_clean.py`)
- V2 structural detector; **new:** ROI classifier, `overlap_product` estimator, risk scorer,
  pixel-mask emitter, positional-prior / canonical-region module, ensemble union
- The measurement harness + labeled set (`validation-set.json`, `ground-truth.json`) — **port from
  b2bweb first**; this makes recall a number

## KPIs

| metric | target |
|---|---|
| Recall | ↑ toward complete — *measured* vs ground truth, with CI |
| ROI/risk correctness | `roi_type` + `risk` right on the labeled set (drives downstream safety) |
| Mask tightness | glyph-hugging; covers anti-alias halo, no needless product pixels |
| False positives | over-flagging OK **only** if Audit reliably catches it; don't flag absurdly |

## Boundaries (do NOT)

- ❌ Do not inpaint, cover, or touch Repair/Audit modules.
- ❌ Do not emit a filled-box mask and call it a mask — Stage 2 needs glyph precision.
- ❌ Do not ship a recall or risk claim without a ground-truth number behind it.
- ❌ Do not change the contract schema/enums/risk-rule without a lockstep `v2` bump.

## First task (current priority)

Port b2bweb's labeled set + write the bake-off scorer **into mark-remover**, then report current
OCR + V2 recall vs ground truth. Until this lands, recall is unmeasured and nothing downstream is
judgeable.
