# Mark Remover — Design

Automated detection and removal of the `sunsky-online.com` watermark from
sunsky product images, built as a **complete automated decision pipeline**:
every image is classified into one truthful final state with **zero manual
review**, and an output is only published as clean when it passes a strict
visual-safety gate. Anything that cannot be cleaned safely is **`auto_rejected`**
— never shipped with a residual mark or product damage.

The pipeline has three responsibilities, documented in detail below:

1. **[Identify the watermark position](#1-watermark-position-identification)** — where, and how confidently, the mark is present.
2. **[Clean the mark](#2-mark-cleaning-strategies)** — a tiered bank of repair and cover strategies chosen by ROI context.
3. **[Review cleaning quality](#3-cleaning-quality-review)** — a hard visual-safety gate that decides what is publishable, with a re-detection double-check and a CI gate.

```
                 ┌─────────────────────────────────────────────────────────┐
   input image → │  ① PRESENCE GATE  →  ② ROI CLASS  →  ③ STRATEGY BANK     │
                 │        │                                   │             │
                 │   confirmed?                          repair / cover     │
                 │        │                                   │             │
                 │        ▼                                   ▼             │
                 │   ④ FINAL VISUAL PUBLISH GATE (P0 hard-safety + cosmetic) │
                 │        │                                                  │
                 │   ┌────┴───────────────┬───────────────────┐             │
                 │   ▼                    ▼                   ▼             │
                 │ clean_repaired   clean_covered        auto_rejected      │
                 └─────────────────────────────────────────────────────────┘
```

**Core invariant:** only an output that passes every **hard-safety** gate
(watermark verifiably gone **and** product undamaged) may be labelled
`clean_repaired` or `clean_covered`. There is no path from a failed gate into a
clean status. Pipeline version `V16_AUTO_DECISION`; the final visual gate is
versioned `v13` (it is intentionally frozen — improvements go into the candidate
generators, not the gate).

---

## 1. Watermark Position Identification

Localization is **precision-first** and runs in two independent stages: a
geometric edge-template detector that finds the bounding box, and a presence
gate that confirms a watermark is really there before any pixels are touched.

### 1.1 Stage 1 — geometric edge-template detection (`detector.py`)

The sunsky watermark is a faint, semi-transparent `sunsky-online.com` text line
with a **known, centred geometry**: it sits horizontally near the image centre,
at a stable vertical band, with a stable width/height ratio. The detector
exploits that prior instead of searching blindly.

`detect_watermark_v2(img_gray)` →

- **Edge map, not raw pixels.** The watermark is low-contrast, so matching is
  done on an edge-response map (`_edge_map`) of a canonical watermark **edge
  template** (`get_edge_template`). This survives the faint alpha blend that
  defeats intensity matching.
- **Centre-column, multi-scale search.** The template is correlated
  (`TM_CCOEFF_NORMED`) only within the horizontal centre window
  (`WM_X_LO … WM_X_HI`) and the vertical watermark band, across a set of height
  scales (`WM_H_FRACS = 4.0 %–6.4 %` of image height). A Gaussian
  **centeredness prior** (`WM_EDGE_YC_MU ≈ 0.478`, `σ = 0.05`) weights matches
  toward the empirical watermark Y-centre, so product edges elsewhere score low.
- **Feature vector per candidate.** Each candidate position yields:
  `edge` (weighted correlation), `int_ncc` (intensity NCC), `cov` (template
  coverage), `biggap` (largest gap in the matched run — text is continuous),
  `xoff` (horizontal offset from centre), `psr` (peak-to-sidelobe ratio — a
  sharp, isolated peak), and `contrast`. A CLAHE-enhanced pass is merged in
  (`_wm2_features_combo`) so faint marks on dark/busy PCBs are still surfaced;
  the merge can only **raise** signals, so it never makes the strict gate
  accept a clean image.
- **Two-tier decision (`_wm2_tier`), precision over recall:**
  - **AUTO** — all strict thresholds met (`edge ≥ 0.22`, `cov ≥ 0.80`,
    `biggap ≤ 0.15`, `xoff ≤ 0.035`, `psr ≥ 6.0`, …). High-confidence presence.
  - **MANUAL** — a looser gate that recovers faint marks. Routed for review,
    **never auto-cleaned** by itself.
  - **none** — no watermark; the image is returned untouched.
- **Mask box from known geometry (`_build_mark_box`).** Rather than trust the
  raw match extent, the publishable mark box is reconstructed from the
  catalogued text geometry positioned at the detected Y: width
  `WM_MASK_W_FRAC = 0.355` of image width (≥ the true text width so trailing
  `.com` glyphs are covered), height padded (`WM_MASK_H_PAD`) and clamped to the
  `5.4 %–7.0 %` band. This yields a stable, fully-covering box even when the raw
  correlation clips a glyph.

### 1.2 Stage 2 — presence confirmation gate (`mark_remover.py`)

A detection box is necessary but not sufficient. `detect_sunsky_presence_fast`
re-scores the candidate with a weighted ensemble and assigns a final presence
status, so a chance edge match never triggers cleaning:

```
confidence = 0.28·template + 0.22·stroke + 0.18·clahe_center
           + 0.12·aspect_ratio + 0.10·position + 0.10·repeat_pattern
           − 0.30·negative_product_text
```

- **template / stroke** — canonical-template correlation and stroke-shape score
  at the box.
- **clahe_center** — contrast-enhanced centre-band response (faint marks).
- **aspect_ratio / position** — match the catalogued text shape and location.
- **repeat_pattern** — the regular glyph cadence of the text line.
- **negative_product_text** — *subtracts* confidence when the region looks like
  real printed product text/labels, the main false-positive source.

Status (thresholds configurable): **`CONFIRMED_WATERMARK`** (≥ confirmed
threshold) → clean; **`UNCERTAIN_WATERMARK`** → clean only under strong
secondary evidence; **`NO_WATERMARK`** → published untouched as
`no_watermark_confirmed`; **`UNSUITABLE`** (high product-text score) → skipped.
A deep OCR pass (`pytesseract`, optional) can further confirm faint cases.

> **Empirical note.** Only a minority of sunsky CDN images actually carry the
> watermark (newer iPhone-14+/recent products are shipped clean). The presence
> gate is what lets the pipeline run over a mixed feed and touch only the
> genuinely-marked images.

### 1.3 The repair mask — three layers + full-text widening

Position is not just a box; cleaning needs a pixel mask. `build_mask_layers`
produces a layered mask so each strategy can pick the tightest safe footprint:

- **core stroke mask** — the watermark glyph strokes (from the logo/stroke
  template), edge-stopped and product-protected.
- **alpha halo mask** — the semi-transparent pixels ringing each stroke, which
  would otherwise read as broken-but-readable residue.
- **uncertain / soft mask** — a feathered safety layer for blending.
- **product-protect mask** — dark/edge/circle/text regions of the *product*
  that must never be painted over (`build_product_protect_mask` +
  `build_protected_text_mask`).
- **full-text widening (`v15_patch.widen_text_mask`)** — the detected mask is
  template-located across a horizontally-expanded band and grown to cover the
  **whole** `sunsky-online.com` line (clipped trailing glyphs included), with
  three guards: skip busy product bands, drop components overlapping product
  pixels, and refuse runaway expansion. This kills faint trailing-glyph ghosts.

---

## 2. Mark Cleaning Strategies

Cleaning is **context-driven** and **beam-searched**: the ROI around the mark is
classified, an ordered bank of strategies is tried, candidates are scored, and
the best is put forward to the quality gate. Strategies fall into two families —
**repair** (reconstruct the original surface invisibly) and **cover** (hide the
mark with a surface-matched fill). Repair is always preferred; cover is the
fallback; `auto_rejected` is the floor.

### 2.1 ROI classification (`progressive_repair._classify_roi`)

The mark's local surroundings are classified into one of twelve ROI classes from
white-ratio, edge-density, brightness, texture variance, dark-ratio, gradient
strength, line/text presence, saturation and product-pixel ratio:

`plain_white` · `near_white` · `low_texture_background` ·
`mixed_background_product` · `simple_product_surface` · `metallic_or_reflective`
· `glass_or_gradient` · `transparent_or_glossy` · `dark_product_surface` ·
`thin_flex_cable` · `complex_product_detail` · `text_or_label_area`.

The class selects and orders the candidate tools (e.g. white-fill tools are
**banned** on dark/flex surfaces where they would leave a bright block).

### 2.2 Repair strategy bank

An ordered bank of repair tools, grouped by mechanism. A bounded **beam** tries
the cheap-first tools, keeps every candidate that clears the local QA hard
gates, and selects the best by composite visual score (a single excellent,
low-risk candidate may short-circuit). Reachable tool count is reported honestly
(`tools_reachable`) — deep-learning tools are only counted when their backend is
actually installed.

- **Real-pixel cloning** — copy genuine neighbouring pixels (8-direction clone,
  ring/border clone, same-surface/same-colour/same-texture clone). No invented
  colour.
- **Statistical / surface fill** — white-median, ring-median, surface-plane and
  surface-gradient fills with matched noise, for uniform/low-texture surrounds.
- **Logo & stroke-mask inpaint** — inpaint only the watermark strokes (Telea/NS)
  using scaled/rotated/alpha template-logo masks; the lightest touch, best at
  preserving product detail.
- **Gradient & glass reconstruction** — linear/bilinear/radial gradient fits,
  Poisson/thin-plate-spline blends and glass-surface clones for smooth
  reflective/gradient regions.
- **Structure-aware repair** — line-preserving, edge-aware and
  contour-guided inpainting for flex cables, connectors and printed edges.
- **Classical & deep inpainting** — OpenCV Telea/NS as a baseline; LaMA / MAT /
  DeepFill on **stroke-level masks only** when available (never full-bbox
  neural inpaint on product structure).

**Pure-background fast path (`v15_patch`).** When the local surround is uniform
(judged on the *immediate* perimeter, so distant product never vetoes a
pure-white case), the mark is simply replaced by an imitation of the
surrounding background (`uniform_background_fill`, a feathered Telea from the
ring). On a uniform surface this is exact and invisible, and it avoids the
garbled-glyph artefacts that template-subtraction can leave on plain white.

**Near-miss rescue (`v14_patch.near_miss_rescue`).** A repair that is clean
except for a faint seam is not discarded: a bounded chain of
component-level residual cleanup → gamma/Lab colour match (pull the changed
pixels toward the surrounding ring) → 1px seam smoothing is applied, then the
result is re-checked against the gate. It is kept as a repair only if it now
fully passes.

### 2.3 Cover strategy beam (fallback)

When no repair passes, a **segmented micro-cover beam** generates several
fragment-based covers, each scored and gate-checked independently:

- `segmented_micro_cover` — fill background fragments from the ring, inpaint
  only the product∩stroke fragments (never a solid block over product).
- `stroke_only` / `stroke_band` — inpaint just the (widened) stroke mask /
  stroke band; lightest touch, best detail preservation.
- `dark_stroke_cover` — on dark / flex surfaces, Telea from neighbouring **dark**
  pixels (never a light/median fill that would leave a bright block).
- `forced_removal_fill` — last-resort Telea over the full text band; removes the
  mark even on hard surfaces.
- **full-bbox cover is banned** whenever `product_overlap > 0.10`, edge density
  is high, long lines are present, or a flex-cable / metallic / glass /
  protected-text zone is detected — a full opaque fill there would damage the
  product.

Every cover candidate is **polished** before scoring (`polish_cover`: a clamped
Lab match toward the local ring + a wider feathered seam, with an *uncapped*
boundary-jump penalty) so a bright/hard patch is never selected over a soft one.
The beam publishes the first cover that passes the gate; otherwise the image
falls to `auto_rejected` with its **most mark-removed, product-safest** attempt
saved for diagnostics.

---

## 3. Cleaning Quality Review

Quality review is the heart of the system: it is what makes every published
output trustworthy. It is a single **final visual publish gate** (`v13_gates.py`)
plus a post-clean **re-detection** double-check, wired into an explicit
auto-decision state machine (`v16_pipeline.decide_final_status`).

### 3.1 The state machine

```
repair candidate passes the P0 gate        → clean_repaired
else residual micro-cleanup passes          → clean_repaired
else best cover candidate passes the P0 gate → clean_covered
else                                         → auto_rejected
```

`auto_rejected` is a **final, automated** decision — `publish_ok = false`,
`manual_required = false`. It is not manual review and it is never published;
diagnostics (`original.jpg`, `best_repair_attempt.jpg`, `best_cover_attempt.jpg`,
masks, `qa.json`) are written to `auto_rejected/`.

### 3.2 Review methods — the detectors

Each detector measures a specific failure mode in the changed region, **relative
to the surrounding surface** (so a clean inpaint on a noisy/textured surface is
not penalised for the surface's own texture):

| Detector | What it measures | Failure signal |
|----------|------------------|----------------|
| **Residual (re-detection)** | Re-run the watermark detector on the *output* at the original box (`recheck_watermark_present`) | The mark is still detectable — the authoritative residual check |
| **Template residual** | Canonical-template correlation over the full text region | A faint readable ghost remains |
| **Dot-chain v2** (`detect_residual_dot_chain_v2`) | Aligned discrete glyph fragments in an expanded window | A broken-but-readable dot/glyph chain (excess over the ring's own texture) |
| **Visible-patch shape** (`detect_visible_patch_shape_v13`) | Straightness, rectangularity, polygonality, hard luma boundary, texture drop of the changed contour | A man-made rectangle / wedge / pale band with a real boundary |
| **Rectangular band** | Pale-band visibility + luma delta | A visible band across the mark footprint |
| **Product silhouette** (`detect_product_silhouette_damage_v13`) | Contour IoU, edge retention, contour break, new bright/dark blobs on product pixels (**excluding** the watermark footprint) | Smeared cable / broken contour / bright blob on a dark surface |
| **Protected product text** (`detect_protected_product_text_v13`) | High-contrast non-watermark strokes flattened away | Real printed product text destroyed |
| **Product overlap routing** (`estimate_product_overlap_v13`) | Non-white / dark / edge / long-line signals inside the box | Routing override + a ban on full-bbox cover on product |

### 3.3 Review criteria — hard-safety vs cosmetic tiers

Gates are split into two tiers. **Safety is identical for repaired and covered
outputs; only aesthetic ranking may differ.**

- **P0 HARD-SAFETY gates — must pass to publish (CI MUST-BE-ZERO):**
  `residual_ocr` (re-detection), `template_residual`, `dot_chain`,
  `product_damage`, `silhouette`, `protected_text`. Leaving the mark, or
  damaging the product, is never publishable.
- **COSMETIC gates — tracked, not publish-blocking:** `visible_patch`,
  `visible_band`. A verifiably mark-removed, product-safe output with only a
  faint seam publishes as `clean_covered` flagged `cosmetic_seam` — a faint seam
  on the background is acceptable; shipping the mark or rejecting a cleanable,
  product-safe image is not.

So an image is `auto_rejected` only when the watermark cannot be removed **or**
removing it would damage the product — exactly the cases that should not ship.

### 3.4 Candidate vs final failures

Intermediate candidates are *expected* to fail the gate; only the final
published output matters:

- `candidate_publish_failures` — repair/cover candidates the gate rejected. **May
  be > 0.** Healthy.
- `final_output_publish_failure` — a *published* output that failed a hard-safety
  gate. **Always 0** by construction (clean status is only assigned when the
  hard gates pass).

### 3.5 Final statuses

| Status | `publish_ok` | `manual_required` | Meaning |
|--------|:-:|:-:|---------|
| `clean_repaired` | ✅ | ❌ | A repair passed every hard-safety gate; mark removed invisibly. |
| `clean_covered` | ✅ | ❌ | A cover passed every hard-safety gate (may carry a tracked `cosmetic_seam`). |
| `no_watermark_confirmed` | ✅ | ❌ | Strong negative evidence; image untouched. |
| `skipped_known_clean` | ✅ | ❌ | Known-clean category (e.g. iPhone 14+), optionally sample-audited. |
| `auto_rejected` | ❌ | ❌ | Repair **and** cover both failed; final automated decision, not published. |
| `failed_io` | ❌ | ❌ | Corrupt/unreadable file. |

### 3.6 CI gate & manifest (`v13_report.py`)

`v13_report.py` scans every per-image `qa.json` and **fails the run (exit 1)** if
any published output violates a hard-safety gate or carries a
`final_output_publish_failure`. It reports `auto_rejected`, rejected-candidate
counts, cosmetic-seam counts and honest tool availability separately. Each
manifest row carries:

```json
{
  "final_status": "clean_repaired | clean_covered | no_watermark_confirmed | skipped_known_clean | auto_rejected | failed_io",
  "publish_ok": true, "manual_required": false,
  "final_gate_pass": true, "final_gate_version": "v16",
  "candidate_publish_failures": 0, "final_output_publish_failure": false,
  "cosmetic_seam": false, "reject_reasons": [],
  "p0_gates": {
    "residual_ocr_pass": true, "template_residual_pass": true,
    "dot_chain_pass": true, "visible_patch_pass": true, "visible_band_pass": true,
    "product_damage_pass": true, "silhouette_pass": true, "protected_text_pass": true
  }
}
```

**Acceptance criteria for a release run:** `manual_review = 0`,
`final_output_publish_failures = 0`, and every `published_with_*` hard-safety
counter `= 0` (residual_ocr, template_residual, dot_chain, product_damage,
silhouette, protected_text). `auto_rejected > 0` and
`candidate_publish_failures > 0` are allowed.

---

## Architecture / File Structure

| File | Role |
|------|------|
| `detector.py` | **① Detection** — edge-template localization, AUTO/MANUAL tiering, mark-box geometry. |
| `mark_remover.py` | Orchestration: presence gate, ROI features, mask layers, per-image pipeline, post-clean re-detection, PDF/HTML/JSONL output. |
| `progressive_repair.py` | **② Strategies** — ROI classifier, repair strategy bank + beam, local QA, residual/band/product detectors, cover builders. |
| `v13_gates.py` | **③ Final visual publish gate** — dot-chain v2, visible-patch-shape, silhouette, product-overlap, protected-text detectors (frozen `v13`). |
| `v14_patch.py` | Adaptive soft-metric context, near-miss rescue, segmented micro-cover beam, cover polish, `cover_hides_watermark`. |
| `v15_patch.py` | Full-text mask widening, ghost-aware residual, dark-stroke cover, uniform-background fill, forced-removal fill. |
| `v16_pipeline.py` | **Auto-decision state machine** — hard-safety vs cosmetic tiers, repair → cleanup → cover → `auto_rejected`. |
| `v13_report.py` | **CI gate** — must-be-zero hard-safety counters, candidate/auto-reject reporting, manifest. |
| `watermark-template.png` | Canonical watermark text template for edge matching / logo masks. |
| `test_v1*_*.py` | Regression + unit locks for the detectors, gate, state machine and CI gate. |

## Usage

```bash
# Clean a directory of images (50 random watermarked candidates), with PDF compare
python3 mark_remover.py --assets path/to/images --n 50 --seed 7 --out output

# Honesty / CI gate (exit 1 if any published output is dirty)
python3 v13_report.py output
```

`--no-pdf` skips the comparison PDF; `--dry-run` lists candidates without
processing. The file pick and in-fill noise are seeded (`--seed`), so a run is
deterministic and reproducible.

## Output Structure

```
output/
├── compare.pdf / compare.html     before/after, per-image, with QA metrics
├── summary.jsonl                  one manifest row per image
├── run_report.json                CI report (hard-safety counters, statuses)
├── clean_repaired/   <id>/        original.jpg · cleaned.jpg · qa.json
├── clean_covered/    <id>/        (may include cosmetic_seam)
├── no_watermark_confirmed/ <id>/
├── skipped_known_clean/ <id>/
├── auto_rejected/    <id>/        original + best_repair_attempt + best_cover_attempt + diagnostics
├── failed_io/
└── debug/                         per-image masks, overlays, repair trace
```

## Dependencies

```
opencv-python>=4.5
numpy>=1.20
pillow>=9.0
pytesseract>=0.3     # optional — deep OCR presence confirmation
torch>=1.10          # optional — LaMA / MAT / DeepFill stroke-mask inpaint
```

## License

Proprietary. Internal use only.
