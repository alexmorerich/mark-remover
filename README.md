# Mark Remover — Design

Automated detection and removal of the `sunsky-online.com` watermark from
sunsky product images, built as a **complete automated decision pipeline**:
every image is classified into one truthful final state with **zero manual
review**, and an output is only published as clean when it passes a strict
visual-safety gate. Anything that cannot be cleaned safely is **`auto_rejected`**
— never shipped with a residual mark or product damage.

The pipeline has three responsibilities, documented in detail below:

1. **[Identify the watermark position](#1-watermark-position-identification)** — where, and how confidently, the mark is present.
2. **[Clean the mark](#2-mark-cleaning-strategies)** — a tiered bank of repair and cover strategies chosen by ROI context, with destructive generators fenced off from product pixels.
3. **[Review cleaning quality](#3-cleaning-quality-review)** — a hard visual-safety gate **plus a truthful audit of the actual published bytes**, with a re-detection double-check and a CI gate.

```
                 ┌──────────────────────────────────────────────────────────┐
   input image → │  ① PRESENCE GATE  →  ② ROI CLASS  →  ③ STRATEGY BANK      │
                 │        │                                   │              │
                 │   confirmed?                       product-aware routing  │
                 │        │                                   │              │
                 │        ▼                                   ▼              │
                 │   ④ FINAL VISUAL PUBLISH GATE  (P0 hard-safety + cosmetic) │
                 │        │                                                   │
                 │        ▼                                                   │
                 │   ⑤ FINAL-OUTPUT AUDIT (re-audit the actual published      │
                 │      bytes: residual? product damage? seam on product?)    │
                 │        │                                                   │
                 │   ┌────┴───────────────┬───────────────────┐              │
                 │   ▼                    ▼                   ▼              │
                 │ clean_repaired   clean_covered        auto_rejected       │
                 └──────────────────────────────────────────────────────────┘
```

**Core invariant:** only an output whose **actual published pixels** pass every
**hard-safety** check — watermark verifiably gone **and** product undamaged —
may be labelled `clean_repaired` or `clean_covered`. There is no path from a
failed check into a clean status, and a candidate that passes the per-candidate
gate but is then mutated by cleanup/cover polish is **re-audited on its final
bytes** before it can ship. Anything uncertain is `auto_rejected`.

The pipeline is built as **layered, independently-versioned stages** so a
regression can never confuse one for another:

| Layer | Version | Responsibility |
|-------|:-------:|----------------|
| **Visual gate** | `v13` (frozen) | The hard-safety + cosmetic detectors. **Never weakened.** |
| **State machine** | `v16` | repair → cleanup → cover → `auto_rejected` decision flow. |
| **Final audit** | `v17` | Truthful re-audit of the *actual published bytes*; authoritative. |
| **Candidate patch** | `v18` | Product-aware routing + safe candidate generation to **reduce rejects without weakening the gate**. |

The guiding principle is that **the clean rate is raised by giving the pipeline
more safe, product-preserving candidates — never by loosening a gate.** The
visual detectors (`v13_gates.py`) are intentionally frozen; V18's work is
entirely upstream of them, in candidate generation, plus one *stricter* audit
signal (low-contrast glyph residue). If a watermark cannot be removed without
damaging the product, the image is still `auto_rejected`.

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

**Pure-background fast path (`v15_patch`), strictly gated (V17).** When the local
surround is uniform, the mark is replaced by an imitation of the surrounding
background (`uniform_background_fill`, a feathered Telea from the ring) — exact
and invisible on a uniform surface, and free of the garbled-glyph artefacts that
template-subtraction can leave on plain white. Because a full-footprint fill is
*destructive on product pixels*, V17 only permits it when
`v17_final_audit.allow_uniform_background_fill` confirms a **strict** pure-
background case: product overlap `< 0.03` (watermark strokes excluded), interior
edge density `< 0.04`, no long lines, no protected text, a uniform surrounding
ring, **and** the box not touching the product silhouette. If any check fails the
box is routed to segmented / stroke-only repair (or `auto_rejected`) — never a
band painted across a cable, dark surface, label or silhouette.

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

### 2.4 Product-context routing & safe candidates (`v18_patch.py`)

Most `auto_rejected` images are **not** "uncleanable" — they are cases where the
old candidate bank could only offer a *destructive* fill (a full band, a forced
removal, a uniform block) that the final audit correctly rejected, leaving no
safe alternative. V18 closes that gap by **routing on product context before any
candidate is generated** and **adding product-preserving candidates** to the
repair/cover pools. Every V18 candidate still passes through the same `_p0`
chokepoint (gate + re-detect + final audit), so this can only widen the set of
*safe* options — it can never ship an output the audit would reject.

**Pre-generation routing — `ProductContext`.** Computed once per image from the
frozen V13 detectors: `product_overlap`, `touches_silhouette`,
`edge_density_inside_box`, `long_line_score`, `protected_text_score`,
`dark_surface_ratio`, `metallic_gradient_score`, `flex_line_score`, and a derived
`pure_background_score`. From these, `allowed_generators(ctx)` decides which
generators may run:

| Regime | Condition | Allowed generators |
|--------|-----------|--------------------|
| **Pure background** | `product_overlap < 0.03` **and** `pure_background_score ≥ 0.85` | uniform-fill, bbox-clone, ring-median, segmented, stroke-only, **and** forced-removal |
| **Light product** | `product_overlap < 0.10`, not touching silhouette | segmented micro-cover, stroke-only inpaint, ring-clone of background fragments |
| **Heavy product / silhouette** | otherwise | **stroke-level or segmented product-safe repair only** |

`should_ban_destructive(ctx)` returns true on **any** product overlap (`> 0.03`),
silhouette contact, or protected text — and the state machine then withholds the
uniform fill and forced-removal from the beam entirely, so the beam is never
wasted on candidates that can only be rejected.

**Segmented product/background repair** (`segmented_product_background_repair`)
splits the ROI into background and product fragments and repairs each with the
right tool (ring/plane fill on background, stroke-only inpaint on product) —
**never one global fill across all fragments**, which is what produces white
slabs, dark blobs and wedges.

**Specialized repair tools for the high-reject ROI classes** — each restricts the
change to the watermark strokes and self-verifies before being offered:

| Tool | ROI target | Method & safety check |
|------|-----------|------------------------|
| `repair_thin_flex_line_preserving` | flex cables, long-line surfaces | NS inpaint of strokes only; **rejected** if cable line-continuity drops below 97% |
| `repair_dark_surface_stroke_clone` | dark modules / backs / screens | Lab-plane clone from local **dark** ring pixels; **rejected** on a bright blob |
| `repair_metallic_gradient_plane` | metallic / glass / glossy | fit a local gradient plane, paint strokes only; **rejected** if it flattens into a block |
| `repair_colored_surface_hue_matched` | blue adhesive / coloured covers | hue/saturation-matched Lab fill; never a gray/white fill |
| `stroke_only_inpaint` | universal fallback | conservative Telea over the strokes only |

**Product-safety-first ranking** (`rank_candidates`) orders candidates by a cheap
product-damage penalty *before* the gate, so on a product region the safest
options are tried first and a high-removal-but-destructive candidate can never
dominate the beam: `score −= 1000·(changed product ratio) + 1000·(visible patch
on product) + 500·(texture drop) + 500·(dark-surface blob)`. On a pure-background
box, ranking falls back to residual-removal and boundary smoothness.

**Stroke-mask reliability** (`stroke_mask_confidence`) reports whether the mask is
a precise `stroke` mask or a wide `logo_fallback`/`widened_text` mask. The rule:
a `logo_fallback` mask on product (`product_overlap > 0.03`) may **only** drive a
stroke-only conservative inpaint — never a destructive fill.

---

## 3. Cleaning Quality Review

Quality review is the heart of the system: it is what makes every published
output trustworthy. It is a **final visual publish gate** (`v13_gates.py`) plus a
post-clean **re-detection** double-check, wired into an explicit auto-decision
state machine (`v16_pipeline.decide_final_status`), plus a **truthful audit of the
actual published bytes** (`v17_final_audit.py`) that is made *authoritative*: it
can demote any candidate the gate accepted.

### 3.1 The state machine

```
repair candidate passes P0 gate + final audit        → clean_repaired
else residual micro-cleanup passes gate + final audit → clean_repaired
else best cover candidate passes gate + final audit   → clean_covered
else                                                  → auto_rejected
```

Every candidate flows through one chokepoint (`_p0`) that runs the visual gate,
the detector re-check, **and** the V17 final audit on that candidate's bytes; a
candidate only counts as publishable if all three agree. The published output is
then audited **once more** on its final bytes (`mark_remover` call site) — if it
still shows the mark or damages the product, it is forced to `auto_rejected`.
This closes the V16 gap where a candidate could pass an early gate and then be
mutated by cleanup, cover polish or status conversion without re-auditing.

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
- **COSMETIC gates — context-aware (V17):** `visible_patch`, `visible_band`.
  A seam is cosmetic **only when the changed region is confirmed pure
  background** — product overlap `< 0.03` (watermark strokes excluded), changed-
  region edge density `< 0.03`, not touching the product silhouette, not over
  protected text. On pure background, a faint seam is acceptable and the output
  publishes as `clean_covered` flagged `cosmetic_seam`. **On product pixels the
  same seam is product damage and becomes a P0 hard failure** — the image is
  `auto_rejected` rather than shipped with a band painted across a cable, dark
  surface, label or silhouette.

So an image is `auto_rejected` only when the watermark cannot be removed **or**
removing it would damage the product — exactly the cases that should not ship.

### 3.4 The V17 final-output audit (`v17_final_audit.py`)

The audit re-runs the visual detectors on the **actual final output** and
returns a `FinalAuditResult` (`pass_p0`, `hard_fail_reasons`, plus residual /
silhouette / patch / band / changed-product-ratio scores). It is the truthful
last word, and it introduces these P0 hard-fail reasons:

| Reason | Trigger |
|--------|---------|
| `published_residual_watermark` | Detector re-detect still fires on the output (authoritative) |
| `published_dot_chain` | A readable broken-glyph / dot-chain remains |
| `visible_patch_on_product` | A visible patch whose changed region is **not** pure background |
| `visible_band_on_product` | A visible band whose changed region is **not** pure background |
| `changed_product_silhouette` | Contour break on product pixels (silhouette bitten / squared off) |
| `changed_thin_flex_structure` | A continuous flex-cable line is broken by the fill |
| `changed_dark_surface_blob` | A new bright/dark blob on a dark product surface |
| `changed_metallic_surface_block` | A flat block where a metallic gradient existed |
| `changed_protected_text` | Real printed product text flattened away |
| `published_low_contrast_glyph_residue` | **(V18)** A faint dot/dash chain aligned on the watermark baseline survived |

The watermark's own strokes are always excluded when measuring "product
structure", so a clean removal is never mistaken for damage. A failed audit sets
`publish_ok = false` and forces `auto_rejected`; a high removal score can never
compensate for product damage.

**V18 — low-contrast glyph residue (`detect_low_contrast_glyph_residue_v18`).**
Some outputs clear the residual-OCR and dot-chain detectors yet still show a faint
low-contrast dot/dash chain to a human on a simple surface. V18 adds a dedicated
detector that fires **only** when small low-contrast components are (a) aligned
along the original watermark baseline (low vertical spread), (b) denser inside the
watermark band than in the surrounding ring, and (c) numerous enough to read as
text. Natural product texture — scattered, unaligned — is never punished. This is
a new **stricter** P0 reason (`published_low_contrast_glyph_residue`); it never
loosens an existing one.

### 3.5 Candidate vs final failures

Intermediate candidates are *expected* to fail the gate; only the final
published output matters:

- `candidate_publish_failures` — repair/cover candidates the gate rejected. **May
  be > 0.** Healthy.
- `final_output_publish_failure` — a *published* output that failed a hard-safety
  gate. **Always 0** by construction (clean status is only assigned when the
  hard gates pass).

### 3.6 Final statuses

| Status | `publish_ok` | `manual_required` | Meaning |
|--------|:-:|:-:|---------|
| `clean_repaired` | ✅ | ❌ | A repair passed every hard-safety gate; mark removed invisibly. |
| `clean_covered` | ✅ | ❌ | A cover passed every hard-safety gate (may carry a tracked `cosmetic_seam`). |
| `no_watermark_confirmed` | ✅ | ❌ | Strong negative evidence; image untouched. |
| `skipped_known_clean` | ✅ | ❌ | Known-clean category (e.g. iPhone 14+), optionally sample-audited. |
| `auto_rejected` | ❌ | ❌ | Repair **and** cover both failed; final automated decision, not published. |
| `failed_io` | ❌ | ❌ | Corrupt/unreadable file. |

### 3.7 CI gate & manifest (`v13_report.py`)

`v13_report.py` scans every per-image `qa.json` and **fails the run (exit 1)** if
any published output violates a hard-safety gate, carries a
`final_output_publish_failure`, **or trips a V17 published-audit failure**. It
reports `auto_rejected`, rejected-candidate counts, cosmetic-seam counts and
honest tool availability separately. Each manifest row carries the gate verdict,
the `p0_gates` dict, and the V17 audit fields (`v17_pass_p0`,
`v17_hard_fail_reasons`, residual / silhouette / patch / band scores):

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
  },
  "v17_pass_p0": true, "v17_hard_fail_reasons": []
}
```

The report also emits a `v17_published_audit_failures` block — every one of
these must be zero:

```json
"v17_published_audit_failures": {
  "published_residual_watermark": 0, "published_dot_chain": 0,
  "published_product_damage": 0, "published_visible_patch_on_product": 0,
  "published_visible_band_on_product": 0, "published_silhouette_damage": 0,
  "published_protected_text_damage": 0
}
```

The V18 report adds a `published_low_contrast_glyph_residue` counter to the
must-be-zero block, plus a **reject taxonomy** so rejects can be diagnosed instead
of blindly tuned (patch plan §1): `auto_rejected_by_roi_class`,
`auto_rejected_by_method`, `auto_rejected_by_mask_type`,
`candidate_failures_by_reason`, and per-image `v18_reject_taxonomy` records
(ROI class · mask type · best method · hard-fail reasons · changed-product-ratio ·
residual / patch / band / glyph-residue scores). The report header carries the
explicit layered versions (`state_machine_version`, `final_audit_version`,
`patch_version`, `gate_version`) so a regression comparison can never confuse
them.

**Acceptance criteria for a release run (mandatory):** `manual_review = 0`,
`final_output_publish_failures = 0`, every `published_with_*` hard-safety counter
`= 0`, every `v17_published_audit_failures` counter `= 0` (including
`published_low_contrast_glyph_residue`). `auto_rejected > 0` and
`candidate_publish_failures > 0` are allowed and healthy. **The clean rate is
never improved by publishing a damaged or residual-watermark image** — when in
doubt, the pipeline rejects.

> **Latest 50-image run (V18, `bench_assets`, seed 2026):** clean_repaired 26,
> clean_covered 4, auto_rejected 20; all hard-safety and final-audit counters 0,
> `all_clean = true`; **12 of the 30 published outputs cleaned via a V18
> product-safe candidate**, and destructive generators were banned on 28 images.
> V18 raised real repairs (`clean_repaired` 20 → 26) without weakening any gate.
> The remaining 20 rejects are genuine safety stops — most are a watermark
> printed **over real product text** (removing it would destroy the text), or a
> seam the audit cannot confirm sits on pure background. `auto_rejected` is the
> correct floor for those; it is never traded away for a higher clean rate.

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
| `v16_pipeline.py` | **Auto-decision state machine** — hard-safety vs cosmetic tiers, repair → cleanup → cover → `auto_rejected`; runs the V17 audit at the per-candidate chokepoint, gates the uniform fill, and seeds the pools with V18 product-safe candidates. |
| `v17_final_audit.py` | **③ Truthful final-output audit** — re-audits the actual published bytes, context-aware cosmetic→hard conversion, product-aware `allow_uniform_background_fill`, V18 low-contrast glyph-residue detector. |
| `v18_patch.py` | **② Product-aware candidate generation (V18)** — `ProductContext` routing, destructive-generator bans, segmented product/background repair, flex / dark / metallic / coloured specialized repair tools, stroke-mask confidence, product-safety-first ranking. |
| `v13_report.py` | **CI gate** — must-be-zero hard-safety + final-audit counters, V18 reject taxonomy aggregation, layered version stamps, candidate/auto-reject reporting, manifest. |
| `watermark-template.png` | Canonical watermark text template for edge matching / logo masks. |
| `test_v1*_*.py` | Regression + unit locks for the detectors, gate, state machine, CI gate, the V17 final audit (`test_v17_patch.py`) and the V18 patch (`test_v18_patch.py`). |

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
