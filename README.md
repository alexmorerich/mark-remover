# Mark Remover

A safety-first pipeline that removes the **`sunsky-online.com`** semi-transparent
watermark from B2B product photos **without leaving residue and without damaging
the product**. It is an automated, fully self-deciding system: every image ends
in exactly one terminal state — `clean_repaired`, `clean_covered`,
`no_watermark_confirmed`, or `auto_rejected` — and **no output that still shows
the mark or damages the product is ever published.** When the watermark cannot
be removed safely, the image is rejected rather than shipped dirty. There is no
manual-review state.

> **Prime directive:** safety beats clean rate. The pipeline will reject a
> cleanable-looking image before it will publish a residual watermark, a visible
> patch on product, or a damaged product surface.

Current release: **V19** — adds a deterministic reverse-alpha recovery engine and
a stricter cover-on-product audit on top of the frozen V13/V16/V17 safety gates.

---

## Quick start

```bash
# Clean N random watermarked images and write a side-by-side comparison PDF.
python3 mark_remover.py --assets bench_assets --n 50 --seed 2026 --out output

# Aggregate the per-image qa.json records into a CI honesty report.
python3 v13_report.py output          # exits non-zero if any safety gate broke

# (Re)solve the reverse-alpha asset from real watermarked images.
python3 scripts/sunsky_alpha_solve.py --mode B --assets bench_assets --max 50

# Run the test suite.
python3 -m pytest tests/ -q
```

Outputs land under `output/<status>/<product_id>/` with `original.jpg`,
`cleaned.jpg`, and a full `qa.json` manifest. A `compare.pdf`, `compare.html`,
`summary.jsonl` and `run_report.json` are written at the root.

**Dependencies:** Python 3, OpenCV (`cv2`), NumPy, Pillow. `onnxruntime` is
*optional* — it enables the PP-OCRv3 text detector and the LaMA-ONNX backend;
without it the pipeline uses robust heuristic fallbacks and runs unchanged.

---

## 1. How the watermark position is identified

The Sunsky watermark is a **fixed, faint, semi-transparent `sunsky-online.com`
text line**, almost always rendered in the **horizontal centre band** of the
image. Detection produces a `mark_box` = `{x, y, w, h}` (top-left + size) and
proceeds in layers, cheap-to-expensive, so easy cases exit early.

### 1.1 Multi-scale template correlation
`detector.py` correlates a synthesized `sunsky-online.com` glyph template
(`watermark-template.png`) — plus optional real-crop templates — against the
grayscale image at several scales using `cv2.matchTemplate` with
`TM_CCOEFF_NORMED`. Large images are downscaled for the scan and the coordinates
are scaled back. Non-maximum suppression (`IoU 0.3`) collapses overlapping hits,
and an optional full-resolution verification pass re-scores the survivors.

### 1.2 Position refinement
Each surviving detection is passed to `refine_watermark_position`, which snaps
the box onto the **full text line** — including the trailing `.com` glyphs and
the low-contrast halo — and returns the canonical `mark_box`. Candidates are then
ranked by a combined score that rewards:
- a correlation peak consistent with a faint, low-contrast text strip,
- an **aspect ratio in the `sunsky-online.com` range (~5.5–8.5 width/height)**,
- a position inside the central band.

Because the watermark is semi-transparent, its polarity flips with the
background (darker than white paper, lighter than dark stock); the detector is
built around this faint, polarity-ambiguous signal rather than a hard edge.

### 1.3 Presence gate (avoid touching clean images)
Before any cleaning, a fast **presence gate** classifies each image as
`CONFIRMED_WATERMARK` / `UNCERTAIN` / `NO_WATERMARK` using a thumbnail-scale
template + stroke + CLAHE-contrast check, escalating to a deeper OCR + template
pass only when uncertain. Two model rules protect known-clean stock:
- **iPhone 14 and newer** product photos carry no Sunsky watermark and are
  excluded from cleaning entirely.
- Images with no confirmed detection are passed through untouched
  (`no_watermark_confirmed`) — a clean image is never modified.

### 1.4 Reverse-alpha NCC alignment (V19)
For the recovery engine, fixed geometry alone is not trusted. `sunsky_reverse_alpha`
generates **two placements** of the solved glyph-alpha asset over the detected
band — a *fixed* one (asset resized to `mark_box`) and an *NCC-aligned* one that
searches scale `0.88–1.12` and a small x/y window for the placement whose glyph
shape best correlates with the high-pass structure in the band — and keeps the
placement that leaves the **lower residual** (mirrors the Doubao/Jimeng
visible-watermark engines).

### 1.5 ROI classification
The box interior is classified (`extract_roi_features` → `classify_roi`) into an
ROI class — `plain_white`, `near_white`, `low_texture_background`,
`thin_flex_cable`, `dark_product_surface`, `metallic_or_reflective`,
`glass_or_gradient`, `simple_product_surface`, … — which routes the choice of
cleaning strategy and the audit thresholds used downstream.

---

## 2. The mark-cleaning strategies

Cleaning is a **candidate-bank state machine** (`v16_pipeline.decide_final_status`).
Many candidate images are generated, each is run through the **same P0 safety
gate**, and the *first* candidate that passes strictly is published. Strategies
are ordered safest / most-faithful first. Crucially, **destructive generators are
banned on any product overlap or silhouette contact** (`v18_patch.should_ban_destructive`),
so the beam is never wasted on fills that could only be rejected.

A **`ProductContext`** is computed once per image (product overlap, silhouette
contact, interior edge density, long-line/flex score, protected-text score,
dark-surface ratio, metallic-gradient score, pure-background score) and gates
which generators are allowed.

### 2.1 Reverse-alpha recovery — **first, non-destructive** (V19)
The watermark is an alpha blend: `watermarked = α·logo + (1−α)·original`. Given a
solved per-pixel `α` map and the fixed `logo` colour, the real pixel is
**recovered** by inverting the blend:

```
original = (watermarked − α·logo) / (1 − α)
```

This does **not** paint, clone, or hallucinate — it *subtracts the overlay and
keeps the product pixels underneath*. That is why it is allowed to run even over
product detail, **product text**, flex cables, dark surfaces and metallic
gradients, where covers and inpainting are banned.

- **Alpha asset** (`assets/sunsky_alpha.png` + `sunsky_alpha_meta.json`) is solved
  reproducibly by `scripts/sunsky_alpha_solve.py`:
  - *Mode A* — controlled captures over black/gray/white fields, `α = (I−B)/(L−B)`.
  - *Mode B (default)* — empirical catalog solve from real white-background
    watermark crops: estimate the clean background `B`, invert the blend with a
    light-gray logo prior, align every crop to a canonical glyph grid,
    median-combine, keep the full halo down to `α ≥ 0.02`, drop specks, and solve
    the overlay colour `L` from the high-alpha pixels.
- **Safe inversion** clamps `α ∈ [0, 0.85]`, floors `(1−α) ≥ 0.25`, and **only
  rewrites pixels inside the alpha mask** — everything else is byte-identical.
- **Thin residual cleanup** runs over *only* the dilated glyph footprint
  (`INPAINT_NS`), never a full bbox, and **skips protected product-text pixels**.
  On a *provably pure-background* box (zero product overlap, confirmed by
  product mask + changed-region check) the cleanup may widen slightly and use a
  texture-preserving `INPAINT_TELEA` to fully clear the footprint — safe because
  there is no product to damage.

### 2.2 Product-aware specialized repair (V18)
When reverse-alpha is not selected, ROI-appropriate **real-pixel** repairs are
tried, each of which restricts the change to the watermark strokes and self-rejects
on any sign of damage:
- `thin_flex_line_preserving` — line-friendly stroke inpaint that rejects if the
  cable's straight-line continuity drops.
- `dark_surface_stroke_clone` — clones strokes from local dark ring pixels;
  rejects a bright blob on dark stock.
- `metallic_gradient_plane` — fits a local gradient plane and paints it over the
  strokes only; rejects a flat rectangular block.
- `colored_surface_hue_matched` — hue/saturation-matched local fill (no gray/white).
- `segmented_product_background_repair` — splits the ROI into background and
  product fragments and repairs each with the right tool, never one global fill.
- `stroke_only_inpaint` — the universal conservative stroke-only fallback.

### 2.3 Background fills — **destructive, background-only**
On a strict pure-background box (no product overlap, uniform surrounding ring,
no interior structure, no protected text, not touching the silhouette) a
`uniform_background_fill` / `forced_removal` is permitted. These are **vetoed on
any product signal** (`v17_final_audit.allow_uniform_background_fill`).

### 2.4 Cover beam — last resort, audited as product damage
If no repair passes, a segmented micro-cover beam is tried. A visible cover that
lands **on product pixels** is treated as a **hard failure** (see §3.4), not a
cosmetic seam. A cover is only valid on pure background or when it is
shape-conformal and visually matched.

### 2.5 Optional LaMA-ONNX backend
When `onnxruntime` + a model are present, a CPU LaMA crop/paste candidate is
offered for hard stroke-only cases (stroke mask, low product overlap, zero
protected-text overlap). It crops around the mask, inpaints, and pastes back
**only the masked pixels**. It is a candidate generator, never a publish
shortcut, and is a no-op when the model is absent.

### What is deliberately *not* used
Global invisible-watermark diffusion / SynthID removal / face protection /
metadata stripping are **excluded** from the publish path. Diffusion can
hallucinate product geometry, soften labels and change SKU-critical detail; for a
visible B2B product watermark that is unacceptable.

---

## 3. Cleaning-quality review: methods & criteria

Every candidate and every **final published byte stream** is audited. The audit
is *truthful*: it runs on the actual output, not on an intermediate candidate,
and a failure forces `auto_rejected`. The gate is the **same strictness for
repaired and covered** outputs.

### 3.1 Two tiers of gate
- **Hard-safety gates (must pass to publish):** the watermark must be verifiably
  **gone** and the product **undamaged**.
  - `residual_ocr_pass` — a fresh post-clean re-detection of the watermark on the
    output (the authoritative residual signal; template correlation alone
    false-passes on bright metal).
  - `template_residual_pass`, `dot_chain_pass` — no template / dot-chain residue.
  - `product_damage_pass`, `silhouette_pass`, `protected_text_pass` — no product
    damage, no broken silhouette, no erased product text.
- **Cosmetic gates (tracked, not blocking on a clean+safe output):**
  `visible_patch_pass`, `visible_band_pass`. A faint seam **on background** is
  acceptable; the same seam **on product** is promoted to a hard failure.

### 3.2 Truthful final audit (V17)
`v17_final_audit.audit_final_output` re-checks the published bytes for:
- residual watermark / readable dot-chain / **low-contrast glyph residue**
  (faint dot-dash chains aligned on the original baseline, too faint for OCR but
  visible to a human),
- **where the change landed** — the fraction of *changed* pixels that fall on
  product and the edge density of the original inside that footprint, which
  decides whether a seam is "on background" (safe) or "on product" (damage),
- structural damage probes: broken thin-flex continuity, new bright/dark blob on
  dark stock, flattened metallic gradient, erased protected text, broken
  silhouette.

### 3.3 Reverse-alpha local safety pre-screen (V19)
Before a reverse-alpha candidate even reaches the gate, it must clear cheap local
checks: residual confidence must **drop** versus the untouched band, it must not
change essentially all product pixels, and it must not introduce a dark-surface
blob. The authoritative gate above still has the final say.

### 3.4 Stricter cover-on-product audit (V19)
For `clean_covered` outputs whose change lands on product
(`product_overlap > 0.03`), `detect_cover_shape_artifact_v19` hard-fails any
visible artificial-cover shape: a **rectangular slab**, a **wedge/notch**, a
**pale band**, a **dark blob on dark stock**, a **straight artificial boundary**,
or a cover that **crosses the product silhouette**. A clean_covered result is
valid only when the cover sits on pure background or is shape-conformal and
visually matched.

### 3.5 Pass / fail criteria (CI must-be-zero)
`v13_report.py` aggregates every `qa.json` and **exits non-zero** if any of these
published-output counters is greater than zero:

```
published_residual_watermark            = 0
published_dot_chain                     = 0
published_low_contrast_glyph_residue    = 0
published_product_damage                = 0
published_visible_patch_on_product      = 0
published_visible_band_on_product       = 0
published_silhouette_damage             = 0
published_protected_text_damage         = 0
final_output_publish_failures           = 0
```

The report also emits V19 diagnostics: `reverse_alpha` (attempted / passed /
published / failure breakdown), `cover_artifacts_v19`, and `text_protection`.

### 3.6 Human eyeball
The run writes a side-by-side `compare.pdf` / `compare.html` so a human can
confirm, per image, that the mark is gone and the product is intact — the final
acceptance check before delivery.

---

## 4. Benchmark (50 images, seed 2026)

| Metric                  | V18 | **V19** |
|-------------------------|----:|--------:|
| clean_repaired          | 26  | **28**  |
| clean_covered           | 4   | **6**   |
| auto_rejected           | 20  | **16**  |
| hard-safety failures    | 0   | **0**   |
| cover-artifacts on product | — | **0**   |
| reverse-alpha published | —   | **16** (12 over product text) |

All published outputs pass the truthful V17/V19 final audit. The remaining
rejects are honest: the frozen residual detector still saw the mark, so they were
**not** published. Clean rate improved only by adding a *safer* recovery
candidate — never by accepting a worse output.

---

## 5. Architecture / file map

| Layer | File | Role |
|-------|------|------|
| Orchestrator | `mark_remover.py` | CLI, scan, presence gate, per-image flow, qa.json, PDF/HTML |
| Detector | `detector.py` | multi-scale template detection + `mark_box` refinement |
| Known-mark registry | `sunsky_registry.py` | binds detect + recovery for `sunsky_online` (V19) |
| Reverse-alpha engine | `sunsky_reverse_alpha.py` | alpha inversion, placement, thin cleanup (V19) |
| Alpha solver | `scripts/sunsky_alpha_solve.py` | reproducible alpha asset (Mode A/B) (V19) |
| Product-text protection | `product_text_detector.py` | PP-OCRv3 ONNX + heuristic fallback (V19) |
| Optional inpaint backend | `lama_onnx_backend.py` | CPU LaMA crop/paste candidate (V19) |
| Repair primitives | `progressive_repair.py` | stroke inpaint, clone, gradient-plane fits |
| Safe candidates | `v18_patch.py` | `ProductContext`, product-safe + reverse-alpha candidates, ranking |
| State machine | `v16_pipeline.py` | candidate bank → P0 gate → terminal status |
| Final audit | `v17_final_audit.py` | truthful audit of published bytes + V19 cover-shape audit |
| Visual gates | `v13_gates.py` | frozen V13 visual + product-integrity detectors |
| Report / CI | `v13_report.py` | aggregate qa.json, must-be-zero safety gate, V19 diagnostics |
| Tests | `tests/`, `test_v1*_*.py` | V10–V19 unit + regression suites |

### V19 design principle
> Improve the clean rate by adding a **safer recovery candidate**, not by
> accepting worse outputs. Order: **reverse-alpha → stroke-only → segmented →
> cover only on pure background → auto_rejected when safety is uncertain.** Never
> trade product fidelity for a higher clean rate.
