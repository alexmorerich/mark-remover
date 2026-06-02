# Mark Remover

A safety-first pipeline that removes the **`sunsky-online.com`** semi-transparent
watermark from B2B product photos **without leaving residue and without damaging
the product**. It is a fully self-deciding system: every image ends in exactly one
terminal state — `clean_repaired`, `clean_covered`, `no_watermark_confirmed`, or
`auto_rejected` — and **no output that still shows the mark or damages the product
is ever published.** When the watermark cannot be removed safely, the image is
*rejected* rather than shipped dirty. There is no manual-review state.

> **Prime directive — safety beats clean rate.** The pipeline rejects a
> cleanable-looking image before it publishes a residual watermark, a visible band
> or patch on product, a ghost-dot residue, or a damaged product surface. The
> clean rate may only rise by adding *safer candidates* — never by loosening a
> gate or accepting a worse output.

The audit stack is layered and each layer is independently versioned so a run can
never claim a capability the code did not produce:

| Layer | Version | Status |
|-------|---------|--------|
| Pipeline / orchestrator | `V22_PATCH` | current |
| Patch (safer candidates + publish blockers) | `v22` | current |
| State machine | `v16` | frozen |
| Final audit | `v17` (+ `v22` detectors) | frozen core |
| Visual / product gate | `v13` | **frozen** |

The **state machine, the final-audit core, and the V13 visual/product gate are
frozen**. Every release adds only *safer candidates* and *stricter publish
blockers*; it never edits a gate. This document describes the current (V22) design.

---

## Quick start

```bash
# Clean N random watermarked images and write a side-by-side comparison PDF.
python3 mark_remover.py --assets bench_assets --n 50 --seed 2026 --out output_v22

# Aggregate the per-image qa.json records into a CI honesty report.
python3 v13_report.py output_v22       # exits non-zero if any safety gate broke
                                       # or if an auto_rejected folder leaked a cleaned.jpg

# (Re)solve the reverse-alpha alpha asset from real watermarked images.
python3 scripts/sunsky_alpha_solve.py --mode B --assets bench_assets --max 50

# Run the test suite (unit + regression).
python3 -m pytest -q
```

Published outputs land under `output_v22/<status>/<product_id>/` with
`original.jpg`, **`cleaned.jpg`** (published states only) and a full `qa.json`
manifest. An `auto_rejected/<product_id>/` folder instead holds
**`best_attempt.jpg` + `reject_reason.txt`** (never `cleaned.jpg`), plus a
`debug/` folder of residual-replay heatmaps. A `compare.pdf`, `compare.html`,
`summary.jsonl`, `run_report.json` and `v20_reasons.csv` are written at the root.

**Dependencies:** Python 3, OpenCV (`cv2`), NumPy, Pillow. `onnxruntime` is
*optional* — it enables the PP-OCRv3 product-text detector and the LaMA-ONNX
backend; without it the pipeline uses robust heuristic fallbacks and runs
unchanged.

---

## 1. How the watermark position is identified

The Sunsky watermark is a **fixed, faint, semi-transparent `sunsky-online.com`
text line**, almost always in the **horizontal centre band** of the image.
Detection produces a `mark_box = {x, y, w, h}` and proceeds in cheap-to-expensive
layers, so easy cases exit early.

### 1.1 Multi-scale template correlation
`detector.py` correlates a synthesized `sunsky-online.com` glyph template
(`watermark-template.png`) — plus optional real-crop templates — against the
grayscale image at several scales using `cv2.matchTemplate` / `TM_CCOEFF_NORMED`.
Large images are downscaled for the scan and the coordinates scaled back.
Non-maximum suppression (`IoU 0.3`) collapses overlapping hits, and an optional
full-resolution pass re-scores the survivors.

### 1.2 Position refinement
Each surviving detection is passed to `refine_watermark_position`, which snaps the
box onto the **full text line** — including the trailing `.com` glyphs and the
low-contrast halo — and returns the canonical `mark_box`. Candidates are ranked by
a score rewarding a faint low-contrast correlation peak, an **aspect ratio in the
`sunsky-online.com` range (≈ 5.5–8.5 w/h)**, and a central-band position. Because
the overlay is semi-transparent its polarity flips with the background (darker than
white paper, lighter than dark stock); the detector is built around this faint,
polarity-ambiguous signal, not a hard edge.

### 1.3 Presence gate (never touch a clean image)
A fast presence gate classifies each image `CONFIRMED_WATERMARK` / `UNCERTAIN` /
`NO_WATERMARK` from a thumbnail-scale template + stroke + CLAHE-contrast check,
escalating to a deeper OCR + template pass only when uncertain. Two model rules
protect known-clean stock:
- **iPhone 14 and newer** product photos carry no Sunsky watermark and are
  excluded from cleaning entirely.
- Images with no confirmed detection are passed through untouched
  (`no_watermark_confirmed`).

### 1.4 Reverse-alpha NCC alignment
For the recovery engine, fixed geometry alone is not trusted.
`sunsky_reverse_alpha` builds **two placements** of the solved glyph-alpha asset
over the band — a *fixed* one (asset resized to `mark_box`) and an *NCC-aligned*
one that searches scale `0.88–1.12` and a small x/y window for the placement whose
glyph shape best correlates with the high-pass structure in the band — and keeps
whichever leaves the **lower residual**. The chosen alignment NCC and the
mask-source (alpha-NCC / stroke / logo-fallback) are recorded per image (see §3.7).

### 1.5 ROI classification
The box interior is classified — `plain_white`, `near_white`,
`low_texture_background`, `thin_flex_cable`, `dark_product_surface`,
`metallic_or_reflective`, `glass_or_gradient`, `simple_product_surface`,
`mixed_background_product`, … — which routes the cleaning strategy and the audit
thresholds used downstream.

### 1.6 Conservative union product mask
Before any repair, `v18_patch.build_product_mask_safe` unions **every** product-
like signal inside the box — non-white, dark, **saturated/colored**, edge density,
metallic gradient, glass boundary, long-line/flex, protected text, dilated
silhouette — and **excludes the watermark's own strokes**. This is the signal that
stops a destructive fill (or a ghost-residue mis-classification) from treating a
**smooth red or silver product surface** as background — a gap a sparse edge/dark
mask alone would miss. Its coverage fraction gates which generators are allowed.

### 1.7 Glyph footprint
For every micro-edit, `sunsky_reverse_alpha.footprint_mask_for_box` derives the
**exact glyph footprint** — the solved alpha shape thresholded at `α > 0.06`
(falling back to the stroke mask when the asset is unavailable). All residue
cleanup and surface refinement is *restricted to this footprint*; nothing outside
the old overlay shape is ever touched.

### 1.8 Low-texture product-surface classification
The plain product mask misses product surfaces that *look* like background. A
dedicated classifier (`v22_patch.classify_surface_v22`) adds self-contained
signals that the sparse mask cannot fool:

- **`near_white_product_surface`** — a near-white tray / plastic part that is
  attached to a product body (a connected component or silhouette crosses the box),
  so it is **not** pure background;
- **`translucent_stack_surface`** — low-saturation grey/green stacked films with
  parallel sheet edges and a large non-white connected area;
- **`dark_smooth_product_surface`** — a dark, low-edge back cover or housing;
- **`long_thin_component_crosses_bbox`** — a flex cable or black frame edge running
  through the box;
- **`v22_product_like_overlap`** — the fraction of the box that reads as product
  material (non-white / coloured / dark / structured).

These let candidate generation know *earlier* that a region is product-like, so
destructive covers are banned where they could only fail (§2.5).

---

## 2. The mark-cleaning strategies

Cleaning is a **candidate-bank state machine** (`v16_pipeline.decide_final_status`).
Many candidate images are generated, each is run through the **same P0 safety
gate**, and the *first* candidate that passes strictly is published. Strategies are
ordered safest / most-faithful first. **Destructive generators are banned on any
product overlap, silhouette contact, protected text, union-product-mask coverage,
or any low-texture product surface** (`v18_patch.should_ban_destructive`), so the
beam is never wasted on fills that could only be rejected.

A **`ProductContext`** is computed once per image (product overlap, silhouette
contact, interior edge density, long-line/flex score, protected-text score,
dark-surface ratio, metallic-gradient score, pure-background score, union
product-mask overlap, the §1.8 surface classes, ROI class) and gates which
generators are allowed.

**Candidate priority order** (`v16_pipeline`):

```
 1. reverse-alpha variant beam              (non-destructive, leads the beam)
 2. residue micro-cleanup beam              (strips faint paired-dot ghosts off #1)
 3. alpha-footprint partial-glyph cleanup   (strips faint s / m / .com fragments)
 4. smooth-surface reverse-alpha refine
 5. thin-flex reverse-alpha line-preserve
 6. segmented reverse-alpha + background clone   (mixed product/background)
 7. metallic / dark / colored stroke-only specialized repair
 8. stroke-only inpaint
 9. background clone-offset fill            ┐
10. uniform background fill                 │ background-only, banned on any product
11. cover                                   ┘
12. auto_rejected                           (safety could not be met)
```

Every candidate, wherever it sits in this order, still passes through the unchanged
P0 / V17 / V22 audit before it can be published.

### 2.1 Reverse-alpha recovery — first, non-destructive
The watermark is an alpha blend `watermarked = α·logo + (1−α)·original`. Given the
solved per-pixel `α` map and the fixed `logo` colour, the real pixel is
**recovered** by inverting the blend:

```
original = (watermarked − α·logo) / (1 − α)
```

This does **not** paint, clone, or hallucinate — it *subtracts the overlay and
keeps the product pixels underneath*. That is why it may run even over product
detail, **product text**, flex cables, dark surfaces and metallic gradients, where
covers and inpainting are banned.

- **Alpha asset** (`assets/sunsky_alpha.png` + `sunsky_alpha_meta.json`) is solved
  reproducibly by `scripts/sunsky_alpha_solve.py` (Mode A controlled captures /
  Mode B empirical catalog solve, default).
- **Safe inversion** clamps `α ∈ [0, 0.85]`, floors `(1−α) ≥ 0.25`, and **only
  rewrites pixels inside the alpha mask** — everything else is byte-identical.

### 2.2 Reverse-alpha variant beam
Rather than a single recovery, a small **deterministic beam** of recovery variants
is generated and each is screened with the same local pre-screen, publishing the
first that clears the authoritative P0 audit
(`sunsky_reverse_alpha.build_variant_beam`):

- `reverse_alpha_fixed` / `reverse_alpha_ncc` — fixed and aligned placement.
- `reverse_alpha_ncc_local_gain` — fits a scalar `α`-gain in `[0.75, 1.25]`
  minimising high-pass residual in the glyph footprint.
- `reverse_alpha_per_channel_logo` — solves a per-channel logo-colour nudge (`±12`)
  from high-confidence alpha pixels, excluding protected text.
- `reverse_alpha_core_halo_split` — inverts the glyph **core** (`α ≥ 0.12`)
  aggressively but **softens the halo** (`0.02 ≤ α < 0.12`) toward a locally blurred
  surface; most ghosts come from the core/halo boundary.
- `reverse_alpha_low_alpha_halo_cleanup` / `…_no_cleanup` — halo cleanup over
  **only** the dilated footprint.
- `reverse_alpha_texture_preserve_blend` — a bilateral pass confined to the glyph
  halo that suppresses faint residue on cardboard / smooth-colored / brushed-metal
  backs **while preserving the surface's high-frequency texture** (it does not
  flatten the surface into a patch).

The beam is fully reproducible (no randomness) and **ranked safest/cleanest first**
by `(residual, ghost-dot score, changed-product fraction)`. A variant that leaves an
aligned ghost-dot chain *on product* is filtered out at the pre-screen.

### 2.3 Residue cleanup beams — footprint-capped, never a band
Reverse-alpha reliably removes the *readable* text but can leave faint
**low-contrast residue** aligned on the old watermark baseline — invisible to OCR,
visible to a human on smooth surfaces. Two beams strip it, both running **on the
reverse-alpha output, not the original**, and both **restricted to the glyph
footprint**:

- **Residue micro-cleanup beam**
  (`sunsky_reverse_alpha.build_residue_micro_cleanup_beam`) targets the **paired
  dots / pits** failure mode. It detects only small, aligned, low-contrast residue
  components inside the footprint (high-pass deviation in `[4, 60]` 8-bit;
  components larger than 1.5 % of the box are treated as real product detail and
  ignored), and **caps the edited area** to `≤ 20 %` of the footprint *and* `≤ 4 %`
  of the `mark_box`. Variants: surface blur, hue-matched clone, component inpaint,
  reverse-alpha gain.

- **Alpha-footprint partial-glyph cleanup beam**
  (`sunsky_reverse_alpha.build_alpha_footprint_residue_cleanup_beam_v22`) targets
  faint **partial-glyph fragments** — surviving `s` / `m` / `.com` pieces and short
  dashes — that are not shaped like a dot chain. It uses a slightly broader window
  (deviation in `[3, 58]`) but keeps every safety limit: edits land **only** inside
  the solved glyph footprint, **never** a full bbox or band, edited area `≤ 30 %` of
  the footprint *and* `≤ 6 %` of the `mark_box`, and each component must remain small
  enough to be residue. Variants: alpha-footprint median blend, directional surface
  clone, bilateral texture blend, component NS inpaint, component Telea inpaint.

If more area than the cap lights up, the residue is **not** isolated — the beam
returns nothing and the image stays `auto_rejected` rather than cleaning a
product feature. Every variant still passes the unchanged P0 / V17 / V22 audit
downstream: a cleanup that creates a blob, a band, or disturbs product texture is
rejected exactly like any other candidate.

### 2.4 Product-aware specialized & mixed repair
When a single reverse-alpha pass is not enough:
- **`thin_flex_reverse_alpha_line_preserve`** — reverse-alpha on a flex cable, then
  verify cable-line continuity (§3.6) and reject if it drops.
- **`segmented_reverse_alpha_background_clone`** — when the watermark crosses *both*
  product and background, split the footprint: reverse-alpha on the product pixels, a
  **clone-offset fill** copying *real* clean pixels from a band above/below into the
  pure-background pixels, blended only along the footprint. Rejects a clone that lands
  on product or creates a rectangular boundary.
- `dark_surface_stroke_clone`, `metallic_gradient_plane`,
  `colored_surface_hue_matched`, `segmented_product_background`, `stroke_only_inpaint`
  — ROI-appropriate real-pixel repairs, each restricted to the watermark strokes and
  self-rejecting on any sign of damage.

### 2.5 Background fills & cover — destructive, background-only
A `uniform_background_fill` / `forced_removal` / cover is permitted **only** on a
proven pure-background box: no product overlap, a uniform surrounding ring, no
interior structure, no protected text, not touching the silhouette, and **not** on
any §1.8 product surface (dark-smooth / translucent-stack / near-white product /
thin-flex). A visible cover that lands on product is a **hard failure** (§3.4 /
§3.5), not a cosmetic seam — `clean_covered` is a rare *background-only* terminal
state, never a product-repair state.

### 2.6 Optional LaMA-ONNX backend
When `onnxruntime` + a model are present, a CPU LaMA crop/paste candidate is offered
for hard stroke-only cases. It pastes back **only the masked pixels**, is a candidate
(never a publish shortcut), and is a no-op when the model is absent.

### What is deliberately *not* used
Global invisible-watermark diffusion / SynthID removal / generative inpainting are
**excluded** from the publish path on SKU-critical product pixels: diffusion can
hallucinate geometry, soften labels and change SKU-critical detail.

---

## 3. Cleaning-quality review: methods & criteria

Every candidate and every **final published byte stream** is audited. The audit is
*truthful*: it runs on the actual output, not on an intermediate candidate, and a
failure forces `auto_rejected`. The gate is the **same strictness for repaired and
covered** outputs. The audit runs twice — once per candidate inside the state
machine, and once more on the final published bytes; if the final bytes fail, the
image is demoted to `auto_rejected` even after it was chosen.

### 3.1 Two tiers of gate
- **Hard-safety gates (must pass to publish):** watermark verifiably **gone** and
  product **undamaged** — `residual_ocr_pass` (a fresh post-clean re-detection, the
  authoritative residual signal), `template_residual_pass`, `dot_chain_pass`,
  `product_damage_pass`, `silhouette_pass`, `protected_text_pass`.
- **Cosmetic gates (tracked, non-blocking on a clean+safe output):**
  `visible_patch_pass`, `visible_band_pass`. A faint seam **on pure white
  background** is acceptable; the same seam **on product** is promoted to a hard
  failure (§3.3).

### 3.2 Truthful final audit
`v17_final_audit.audit_final_output` re-checks the published bytes for residual
watermark / readable dot-chain / low-contrast glyph residue, **where the change
landed** (the fraction of changed pixels on product), and structural-damage probes
(broken thin-flex continuity, new bright/dark blob on dark stock, flattened metallic
gradient, erased protected text, broken silhouette).

### 3.3 Visible band on a non-white / product surface (hard fail)
`v22_patch.detect_visible_band_on_nonwhite_surface_v22` closes the case where a
repair leaves a rectangular **tonal band** across the old watermark footprint. The
band is only cosmetic on **pure white** background; on grey stacked film, a black
frame, a metallic plate, or a near-white product surface it is product damage. The
detector hard-fails (`published_visible_band_on_nonwhite_surface`) when:

1. the changed region is **band-like** — a high V13 band score with rectangular
   compactness, *or* a near-rectangular fill with a tonal jump against the local
   ring; **and**
2. the original surface is **not pure white** — median luma `< 248`, or saturation
   above a small threshold, or a non-white ROI class, or a §1.8 product surface.

A glyph-shaped reverse-alpha change (low rectangularity) never trips it; only a
slab/band fill or cover on a non-white surface does.

### 3.4 Alpha-footprint partial-glyph residue (hard fail)
`v22_patch.detect_alpha_footprint_residue_v22` catches faint surviving
**partial-glyph fragments** (`s`, `m`, `.com`, paired pits, short dashes) that are
too weak for OCR and the dot-chain check but still mark the old `sunsky-online.com`
position. It uses the **solved glyph alpha as the canonical footprint**, measures a
local high-pass deviation in the output (keeping only low-contrast components,
`|Δluma| ∈ [3, 58]`, so real product detail is excluded), and scores the residue by
its **density inside the footprint versus a surrounding ring**, its component count,
and its **alignment on the Sunsky baseline**. Above threshold (and with the residue
small enough to *be* residue, not a product feature) it hard-fails with
`published_partial_glyph_residue`.

### 3.5 Reverse-alpha ghost-dot detector
`detect_reverse_alpha_ghost_dots` catches the related failure mode where
reverse-alpha removes the glyph but leaves faint **paired dots / pits** aligned on
the original baseline. It scores baseline alignment + paired spacing and decides
whether the dots sit **on product** using a self-contained colored/dark surface
estimate (so a sparse product mask cannot hide a chain on a flat red or silver
back-cover). On product → hard fail; on pure background it is routed to the residue
cleanup beams (§2.3) before rejection. The same detector runs inside the variant-beam
pre-screen, so a ghosting variant never becomes the published one.

### 3.6 Product-side cover audit & thin-flex continuity
- **Cover audit** — for `clean_covered` outputs, `detect_cover_shape_artifact_v20`
  hard-fails whenever **more than 1 %** of the changed pixels land on product **and**
  the cover is visible as a shape (rectangular slab / wedge), a straight artificial
  boundary, a local tone mismatch, a dark blob on dark stock, or it crosses the
  product silhouette. The `cover_artifact_v20` record (changed-on-product fraction,
  rectangularity, straight-boundary score, local color delta, silhouette crossing) is
  attached to every covered output.
- **Thin-flex continuity** — `detect_thin_flex_continuity_v20` traces the longest
  dark cable line before/after a repair; if its length drops beyond a small tolerance
  or its endpoints shift, the cable was cut or notched and the output hard-fails.

### 3.7 Explainable diagnostics
Several **additive** records are attached to every `qa.json` (post-processing only —
they change no decision and never alter what is published), so a reviewer can see
*why* an image was rejected and *what* could still fix it:

- **`v21_failure_taxonomy`** — `primary_reject_class` (`true_residual` /
  `product_damage` / `cover_artifact` / `thin_flex_break` / `protected_text_risk` /
  `mask_uncertain` / `detector_false_positive_suspect` / `no_safe_candidate`),
  `residual_kind`, `changed_region_kind`, and a `recommended_next_candidate`.
- **`v21_mask_quality`** — `mask_source` (`alpha_ncc` / `stroke` /
  `alpha_stroke_intersection` / `logo_fallback`), the alpha-NCC score, stroke
  coverage, fallback reason, and the mask's product overlap.
- **`v21_residual_explain`** — separates *true residual watermark* from a *detector
  false-positive on product structure*, and records whether the residue is on product
  vs background and what the cleanup returned.
- **`v22_surface` / `v22_visible_band` / `v22_alpha_footprint`** — the §1.8 surface
  class and product-like overlap, and the band / partial-glyph residue scores on the
  final bytes.

These build on the per-candidate residual-replay heatmaps written to `debug/`
(`original_residual_heatmap.png`, `candidate_residual_heatmap.png`,
`residual_delta_heatmap.png`, `changed_mask.png`, `product_mask.png`,
`alpha_footprint.png`) — making rejections explainable **without loosening the
detector**.

### 3.8 Published vs rejected artifacts are physically separated
Only `clean_repaired` / `clean_covered` / `no_watermark_confirmed` folders may
contain `cleaned.jpg`. An `auto_rejected` folder holds `best_attempt.jpg` +
`reject_reason.txt` and is labelled **“BEST ATTEMPT — NOT PUBLISHED”** in
`compare.pdf` / `compare.html`. `v13_report.py` **fails CI** if any
`auto_rejected/**/cleaned.jpg` exists, so a rejected attempt can never be mistaken
for a delivered asset.

### 3.9 Pass / fail criteria (CI must-be-zero)
`v13_report.py` aggregates every `qa.json` and **exits non-zero** if any
published-output counter is greater than zero:

```
# hard-safety
published_residual_watermark                    = 0
published_dot_chain                             = 0
published_low_contrast_glyph_residue            = 0
published_product_damage                        = 0
published_visible_patch_on_product              = 0
published_visible_band_on_product               = 0
published_silhouette_damage                     = 0
published_protected_text_damage                 = 0
final_output_publish_failures                   = 0
auto_rejected_cleaned_jpg_leak                  = 0
# V22 publish blockers
published_visible_band_on_nonwhite_surface      = 0
published_partial_glyph_residue                 = 0
published_alpha_footprint_residue               = 0
published_cover_on_product                      = 0
published_product_silhouette_change             = 0
published_protected_text_change                 = 0
clean_repaired_with_any_p0_fail                 = 0
clean_covered_with_any_p0_fail                  = 0
```

The report also emits `v20`, `v21` and **`v22`** diagnostic blocks (variant-beam
attempted/passed, ghost-dot failures, cover-on-product hard fails, residue-cleanup
attempted/published, surface-class breakdown, reject-class / recommended-next
breakdowns) plus a per-image `v20_reasons.csv`.

### 3.10 Human eyeball
The run writes a side-by-side `compare.pdf` / `compare.html` so a human can confirm,
per image, that the mark is gone and the product is intact — the final acceptance
check before delivery. Marginal cases are flagged on the page with a
**`V22 WARN`** header line (`alpha_residue` / `band` / `product_like` scores), and
the cover page prints the two V22 leak counters (both must be `0`).

---

## 4. Benchmark (50 images, seed 2026)

| Metric                                       | Value |
|----------------------------------------------|------:|
| total images                                 |    50 |
| `clean_repaired`                             |    30 |
| `clean_covered`                              |     3 |
| `auto_rejected`                              |    17 |
| `failed_io`                                  |     0 |
| `manual_review`                              |     0 |
| **published total**                          |    33 |
| hard-safety failures                         |     0 |
| published band on non-white surface          |     0 |
| published partial-glyph residue              |     0 |
| cover artifacts published on product         |     0 |
| `auto_rejected` cleaned.jpg leak             |     0 |

`v13_report.py output_v22` reports `all_clean = True`. Every published output passes
the truthful final audit; all 17 remaining rejects are honest — the watermark could
not be removed without risking the product, or no safe candidate existed.
`clean_covered` is restricted to proven pure background.

**Acceptance criteria** (all hold): `failed_io == 0`, `manual_review == 0`, every
V22 publish-blocker counter `== 0`, `clean_repaired ≥ 30/50`, `clean_covered ≤ 3/50`,
`auto_rejected ≤ 17/50`. The clean rate is never forced by weakening a gate: if a new
blocker rejects an image, that is the correct outcome.

**Known limitation (honest):** on heavily *textured* product surfaces (e.g. kraft
cardboard) and some bright-metallic surfaces, a reverse-alpha recovery can leave a
**noise-floor faint** trace that no cheap detector separates reliably from natural
texture without false-rejecting genuinely clean images. These traces are far fainter
than a readable watermark, are surfaced in `compare.pdf` for human review, and the
clearly-readable cases are always caught. Per the prime directive, the pipeline never
publishes product damage or a clearly-readable mark, and prefers an honest rejection
to a worse output.

---

## 5. Architecture / file map

| Layer | File | Role |
|-------|------|------|
| Orchestrator | `mark_remover.py` | CLI, scan, presence gate, per-image flow, qa.json, residual heatmaps, PDF/HTML, published/rejected split |
| Detector | `detector.py` | multi-scale template detection + `mark_box` refinement |
| Known-mark registry | `sunsky_registry.py` | binds detect + recovery for `sunsky_online` |
| Reverse-alpha engine | `sunsky_reverse_alpha.py` | alpha inversion, placement, variant beam, residue micro-cleanup beam, alpha-footprint cleanup beam, footprint mask, thin cleanup |
| Alpha solver | `scripts/sunsky_alpha_solve.py` | reproducible alpha asset (Mode A/B) |
| Product-text protection | `product_text_detector.py` | PP-OCRv3 ONNX + heuristic fallback |
| Optional inpaint backend | `lama_onnx_backend.py` | CPU LaMA crop/paste candidate |
| Repair primitives | `progressive_repair.py` | stroke inpaint, clone, gradient-plane fits, ROI classification |
| Safe candidates | `v18_patch.py` | `ProductContext`, union product mask, reverse-alpha + segmented/flex/specialized candidates, residue/footprint beam wrappers, ranking, diagnostic records |
| V22 detectors | `v22_patch.py` | visible-band-on-non-white, alpha-footprint partial-glyph residue, surface classification, candidate pre-audit |
| State machine | `v16_pipeline.py` | candidate bank → P0 gate → terminal status |
| Final audit | `v17_final_audit.py` | truthful audit + ghost-dot, cover, thin-flex, V22 detectors |
| Visual gates | `v13_gates.py` | **frozen** V13 visual + product-integrity + thin-flex continuity detectors |
| Report / CI | `v13_report.py` | aggregate qa.json, must-be-zero gate (incl. V22), diagnostics, rejected-leak check |
| Tests | `tests/`, `test_v1*_*.py` | unit + regression suites; V22 fixtures in `tests/fixtures/` |

### Design principle
> Improve the clean rate by adding **safer recovery candidates and stricter
> audits**, never by accepting worse outputs or loosening the frozen V13/V17 gates.
> Order: **reverse-alpha variant beam → residue micro-cleanup → alpha-footprint
> partial-glyph cleanup → smooth-surface refine → line-preserving / segmented
> reverse-alpha → stroke-only → cover only on pure background → auto_rejected when
> safety is uncertain.** Never trade product fidelity for a higher clean rate.
