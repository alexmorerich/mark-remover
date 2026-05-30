# Mark Remover

Automated watermark detection and removal pipeline for sunsky-online.com product images. V8 replaces the fixed candidate loop with a **100-tool progressive repair strategy bank** — each watermark ROI is classified, tools are selected from an ordered strategy bank, candidates run through a local QA gate, and the first passing candidate wins. Adaptive cover is the absolute last resort.

## V10 — Quality Patch (QA truthfulness, residual gate, honest covers)

V10 does **not** add more repair methods. It fixes the reason bad outputs were
being *accepted*: the QA gate was unreliable. The principle is *a smaller
number of truthful, well-tested methods beats a larger bank behind a weak gate.*

- **Truthful QA gate (P1).** QA metrics no longer default to `0.0`; missing
  metrics fail closed (`missing_required_qa_metric`) and an all-zero metric
  set is rejected (`qa_metrics_probably_not_computed`). No more `pass=True` on
  uncomputed scores.
- **Residual watermark gate (P2).** After each repair, a second detector
  measures leftover watermark *at the known glyph locations* (mask-aware,
  horizontally extended to catch trailing glyphs) relative to the surrounding
  surface's own texture baseline — canonical-template correlation + a
  stroke/text-component score + a watermark-region improvement ratio. A
  smooth-but-readable or broken-but-readable watermark is a **failure**.
- **3-layer mask (P3).** Core stroke + alpha halo (semi-transparent ring) +
  soft safety dilation, so tight inpaints stop leaving a readable halo.
- **Honest covers (P4).** `final_adaptive_cover` is now a **Telea inpaint over
  the full watermark footprint** (extended past the detected bbox to catch
  clipped glyphs) — never a translucent gray band or RGB-noise blob. A
  cover-rectangularity gate plus a hide-check select the cover, routed by
  *actual in-bbox structure* (uniform → full-footprint fill; structured →
  light-touch band / segmented).
- **Strategy reorder + diversity telemetry (P5/P6).** On plain / low-texture /
  metallic backgrounds, clone and statistical/gradient fills run *before*
  stroke-only repair. Each result logs `strategy_list`, `families_attempted`,
  and per-tool `qa_reject_reasons`.
- **Status semantics + single source of truth (P7/P8).** `clean_repaired`
  requires `residual_pass AND metrics_valid AND` the geometry gate; failures
  escalate automatically to a stronger cover (`auto_cover_retry`) — manual
  review stays 0. One `RepairCandidate` drives report / JSONL / PDF.
- **Debug + regression lock (P9/P10).** `trace.json` carries residual
  verdicts, cover metrics and diversity telemetry; `test_v10_regression.py`
  asserts no readable watermark / no product damage on failure-prone cases.

### V10 results (50-image benchmark, seed 2026)

| Metric | Result |
|--------|--------|
| readable watermark remaining | 0/50 (max template-corr 0.157 < 0.18 gate) |
| manual_review | 0 |
| failed_io | 0 |
| qa zero-metric passes | 0 |
| visible gray-rectangle covers | 0 (covers are inpaint-based) |
| clean_repaired / clean_covered | ~28–30 / ~20–22 |

Covers are no longer translucent dims — they reconstruct the surface, so a
"covered" result looks like a natural removal rather than a patch.

## V8 Performance (50-image benchmark)

| Metric | V7 | V8 |
|--------|-----|-----|
| clean_repaired | 72% | 78% |
| clean_covered | 28% | 22% |
| failed | 0 | 0 |
| manual_review | 0 | 0 |
| avg latency | 380ms | 312ms |
| tools available | 7 | 100 |
| ROI classes | 12 | 11 |

## Design Philosophy

**Try real pixels first, synthesize second, cover last.**

Every repair attempt follows a strict escalation:

1. **Clone real pixels** from surrounding background (cheapest, most natural)
2. **Statistical fill** — median/mean/mode from context ring (fast, safe on plain areas)
3. **Gradient reconstruction** — fit a color plane from the border ring (handles gradients)
4. **Stroke-level logo removal** — target only watermark text strokes (minimal collateral)
5. **Classical inpainting** — OpenCV Telea/Navier-Stokes (proven algorithms)
6. **Deep inpainting** — LaMA, DeepFill, MAT (neural methods for complex texture)
7. **Adaptive cover** — final fallback that hides the mark naturally (never a gray rectangle)

## Architecture

```
Image Pool
    |
    v
+-------------------------+
|  Candidate Selection    |  iPhone 14+ skip, template detection,
|                         |  2-stage presence gate (fast + deep OCR)
+-----------+-------------+
            |
            v  watermark confirmed
+-----------+-------------+
|  V8 ROI Classifier      |  11 background classes from context ring
|  (progressive_repair)   |  features: brightness, texture, edges,
|                         |  gradient, saturation, dark/white ratio
+-----------+-------------+
            |
            v
+-----------+-------------+
|  Strategy Bank          |  Per-class ordered tool list
|  Selection              |  Cheapest/safest tools first
+-----------+-------------+
            |
            v
+-----------+-------------+
|  Progressive Repair     |  For each tool in order:
|  Loop                   |    1. Apply repair
|                         |    2. Run local QA gate (7 metrics)
|                         |    3. Accept if passed -> DONE
|                         |    4. Track best-failed for fallback
+-----------+-------------+
            |
            v  no tool passed QA
+-----------+-------------+
|  Final Adaptive Cover   |  Median-color soft patch with noise +
|  (tool #100)            |  feather blending. Never gray rectangle.
+-----------+-------------+
            |
            v
    clean_repaired / clean_covered
```

## The 100 Repair Tools

Tools are organized in 6 groups by technique. Each tool has a `cost_level` (1=cheap, 5=expensive) and `risk_level` (1=safe, 5=risky). The strategy bank orders them cheapest/safest first.

### Group A: Plain Background Cloning (Tools 1-20)

Clone or fill from surrounding plain/white areas. Cost 1, Risk 1. Best for white, near-white, and low-texture backgrounds.

| # | Tool | Technique |
|---|------|-----------|
| 1 | `clone_above_patch` | Clone strip from directly above bbox |
| 2 | `clone_below_patch` | Clone strip from directly below bbox |
| 3 | `clone_left_patch` | Clone strip from left of bbox |
| 4 | `clone_right_patch` | Clone strip from right of bbox |
| 5 | `clone_best_of_4_dirs` | Try 4 cardinal directions, pick lowest seam |
| 6 | `clone_best_of_8_dirs` | Try 8 directions (incl. diagonals), pick lowest seam |
| 7 | `white_median_fill` | Median white fill from context ring |
| 8 | `white_patch_with_noise` | White fill + matched noise texture |
| 9 | `corner_background_clone` | Clone from image corners (assumed clean) |
| 10 | `canvas_margin_clone` | Clone from nearest canvas margin |
| 11 | `ring_median_fill` | Median color from ring + noise injection |
| 12 | `ring_mean_fill` | Mean color from ring + noise injection |
| 13 | `ring_mode_fill` | Mode (most frequent) color from ring |
| 14 | `local_noise_transfer_fill` | Ring median + noise power spectrum matching |
| 15 | `jpeg_texture_clone_fill` | Clone with JPEG artifact texture preservation |
| 16 | `micro_tile_white_clone` | Micro-tile 4x4 blocks from ring |
| 17 | `feathered_white_clone` | Heavy-feather Gaussian blend clone |
| 18 | `hard_paste_white_clone` | Hard paste with 1px feather (plain white only) |
| 19 | `seam_scored_white_clone` | Multi-source clone scored by seam quality |
| 20 | `edge_safe_white_clone` | Clone that avoids edge-dense source patches |

### Group B: Same-Surface Matching (Tools 21-40)

Match color, luminance, gamma, or texture from a similar surface region. Cost 2, Risk 2. Best for product surfaces, colored backgrounds, and dark areas.

| # | Tool | Technique |
|---|------|-----------|
| 21 | `same_surface_clone_above` | Clone from same-color surface above |
| 22 | `same_surface_clone_below` | Clone from same-color surface below |
| 23 | `same_surface_clone_left` | Clone from same-color surface left |
| 24 | `same_surface_clone_right` | Clone from same-color surface right |
| 25 | `same_surface_best_patch` | Best of 4 same-surface clones by seam score |
| 26 | `same_color_region_clone` | Find nearest region with matching mean color |
| 27 | `same_luminance_region_clone` | Match by luminance histogram |
| 28 | `same_texture_region_clone` | Match by Laplacian texture variance |
| 29 | `surface_plane_fill` | Least-squares color plane from ring pixels |
| 30 | `surface_gradient_fill` | Gradient plane + noise from ring stats |
| 31 | `surface_noise_preserved_fill` | Gradient plane + high-freq noise power transfer |
| 32 | `surface_patch_border_blend` | Blend multiple border strips into fill |
| 33 | `surface_patch_gamma_match` | Clone + gamma correction to match ring |
| 34 | `surface_patch_color_match` | Clone + Lab color space histogram matching |
| 35 | `surface_patch_contrast_match` | Clone + contrast/brightness normalization |
| 36 | `surface_patch_lowpass_match` | Low-pass filtered fill (blur to match) |
| 37 | `surface_patch_highpass_transfer` | Keep low-freq fill, transfer high-freq texture |
| 38 | `large_uniform_area_repair` | Optimized fill for large uniform backgrounds |
| 39 | `dark_surface_clone` | Dark-optimized clone with gamma correction |
| 40 | `light_surface_clone` | Light-optimized clone with exposure matching |

### Group C: Logo & Stroke Mask Removal (Tools 41-60)

Target only the watermark text strokes using template masks, stroke segmentation, or frequency-domain detection. Cost 2-3, Risk 2-3. Minimal collateral damage to surrounding image.

| # | Tool | Technique |
|---|------|-----------|
| 41 | `template_logo_mask_remove` | Match canonical watermark template, mask strokes only |
| 42 | `scaled_template_logo_mask` | Multi-scale template matching (0.8x-1.2x) |
| 43 | `rotated_template_logo_mask` | Rotation-tolerant template matching (+-5 deg) |
| 44 | `alpha_template_logo_mask` | Alpha-channel template for semi-transparent marks |
| 45 | `stroke_only_mask_inpaint` | Inpaint only detected stroke pixels |
| 46 | `stroke_mask_dilate_1px` | Stroke mask + 1px dilation for safety margin |
| 47 | `stroke_mask_dilate_2px` | Stroke mask + 2px dilation |
| 48 | `stroke_mask_soft_edge` | Stroke mask with Gaussian soft edges |
| 49 | `text_component_filter_mask` | Connected-component filter for text-like shapes |
| 50 | `frequency_signature_logo_mask` | FFT-based watermark frequency isolation |
| 51 | `lab_l_channel_logo_mask` | Logo detection in Lab L-channel (brightness) |
| 52 | `blue_channel_logo_mask` | Logo detection in blue channel (common for white text) |
| 53 | `gray_delta_logo_mask` | High-pass gray difference mask |
| 54 | `template_residual_minimization` | Iterative template subtraction to minimize residual |
| 55 | `multi_size_logo_mask_bank` | Bank of 7 pre-scaled logo masks |
| 56 | `multi_position_logo_mask_bank` | Bank of offset-shifted logo masks |
| 57 | `logo_mask_plus_clone_fill` | Logo stroke mask + directional clone fill |
| 58 | `logo_mask_plus_surface_fill` | Logo stroke mask + surface gradient fill |
| 59 | `logo_mask_plus_telea` | Logo stroke mask + OpenCV Telea inpaint |
| 60 | `logo_mask_plus_lama` | Logo stroke mask + LaMA neural inpaint |

### Group D: Gradient & Glass Reconstruction (Tools 61-75)

Reconstruct gradient fields, glass surfaces, specular highlights, and transparent materials. Cost 3, Risk 3. Best for glass_or_gradient, metallic, and transparent backgrounds.

| # | Tool | Technique |
|---|------|-----------|
| 61 | `linear_gradient_reconstruction` | Fit linear gradient plane from border ring |
| 62 | `bilinear_gradient_reconstruction` | Bilinear interpolation from 4 border strips |
| 63 | `radial_gradient_reconstruction` | Radial gradient fit for circular backgrounds |
| 64 | `thin_plate_spline_fill` | Thin-plate spline interpolation from border |
| 65 | `poisson_gradient_clone` | Poisson blending from adjacent patch |
| 66 | `glass_surface_clone` | Clone from adjacent glass/reflective region |
| 67 | `frosted_glass_noise_fill` | Gradient + frosted glass noise texture |
| 68 | `transparent_back_cover_fill` | Optimized for transparent back covers |
| 69 | `specular_highlight_preserve_fill` | Preserve specular highlights during fill |
| 70 | `shadow_gradient_preserve_fill` | Preserve shadow gradients during fill |
| 71 | `low_frequency_gradient_fill` | Low-pass gradient field reconstruction |
| 72 | `high_frequency_noise_transfer` | Transfer high-freq noise from ring to fill |
| 73 | `glass_edge_aware_inpaint` | Edge-aware inpaint for glass surfaces |
| 74 | `reflection_aware_patch_clone` | Clone that respects reflection patterns |
| 75 | `soft_material_surface_repair` | Optimized for soft/matte materials |

### Group E: Structure-Aware Repair (Tools 76-90)

Preserve edges, lines, contours, and structural features during repair. Cost 3-4, Risk 3-4. Best for complex product details, flex cables, metal connectors, and structured surfaces.

| # | Tool | Technique |
|---|------|-----------|
| 76 | `edge_aware_clone` | Clone along edge direction to preserve lines |
| 77 | `line_preserving_inpaint` | Detect and extend lines through repair zone |
| 78 | `contour_guided_inpaint` | Follow product contours during fill |
| 79 | `thin_flex_cable_repair` | Specialized for thin flex cable backgrounds |
| 80 | `black_flex_texture_repair` | Dark flex cable texture synthesis |
| 81 | `metal_connector_repair` | Metal connector surface matching |
| 82 | `screw_area_avoid_repair` | Avoid modifying screw/fastener regions |
| 83 | `product_mask_protected_fill` | Fill only non-product pixels |
| 84 | `background_only_repair` | Repair background while preserving product |
| 85 | `foreground_surface_only_repair` | Repair only product surface pixels |
| 86 | `edge_direction_extrapolation` | Extrapolate edge directions into repair zone |
| 87 | `structure_tensor_inpaint` | Structure tensor guided diffusion |
| 88 | `anisotropic_diffusion_inpaint` | Anisotropic diffusion fill |
| 89 | `patchmatch_inpaint` | PatchMatch-based texture synthesis |
| 90 | `nearest_neighbor_texture_synthesis` | NN texture synthesis from ring patches |

### Group F: Classical & Deep Inpainting (Tools 91-99)

Standard inpainting algorithms (OpenCV) and neural network methods. Cost 4-5, Risk 3-5. Tried after cheaper methods fail.

| # | Tool | Technique |
|---|------|-----------|
| 91 | `opencv_telea_inpaint` | OpenCV Telea algorithm (r=5) |
| 92 | `opencv_ns_inpaint` | OpenCV Navier-Stokes algorithm (r=5) |
| 93 | `lama_small_mask` | LaMA with minimal stroke mask |
| 94 | `lama_full_bbox` | LaMA with full bounding box mask |
| 95 | `lama_stroke_mask` | LaMA with stroke-segmented mask |
| 96 | `lama_with_context_padding` | LaMA with extra context padding |
| 97 | `deepfill_style_inpaint` | DeepFill v2 style contextual attention |
| 98 | `mat_inpaint` | MAT (Mask-Aware Transformer) inpaint |
| 99 | `sd_inpaint_low_strength` | Stable Diffusion inpaint at low denoising |

### Tool 100: Final Adaptive Cover

**`final_adaptive_cover`** — The absolute last resort. Draws a soft-blended patch using the median color from the context ring, adds matched noise texture, and feather-blends at the edges. Produces `clean_covered` status. Never a gray rectangle.

## ROI Classification (11 Classes)

The watermark region's background determines which tools are selected and in what order. Features are extracted from a context ring around the bbox (excluding the watermark itself).

| Class | Decision Rule | Primary Strategy |
|-------|---------------|-----------------|
| `plain_white` | white_ratio > 0.85, edge_density < 0.03 | Clone/white fill, cheapest tools first |
| `near_white` | brightness > 220, texture < 80, edge < 0.05 | White fill + gradient, seam-scored clones |
| `low_texture_background` | texture < 120, edge < 0.06 | Clone + ring median, surface gradient |
| `simple_product_surface` | saturation > 25, edge > 0.06 | Same-surface matching, color/texture match |
| `dark_product_surface` | dark_ratio > 0.65 or dark_ratio > 0.4 | Dark-optimized clone, gamma-matched surface |
| `glass_or_gradient` | gradient > 8.0, edge < 0.08 | Linear/bilinear gradient reconstruction |
| `metallic_or_reflective` | (via fallthrough rules) | Surface patch matching, gradient, stroke mask |
| `thin_flex_cable` | has_long_lines, dark_ratio > 0.3 | Stroke mask, edge-aware, cable-specific tools |
| `text_or_label_area` | has_text_edges, edge > 0.12 | Stroke-only mask, logo mask + clone |
| `complex_product_detail` | edge > 0.18 | Template logo mask, contour-guided, PatchMatch |
| `unknown` | No rule matched | Generic: clone + stroke + gradient + Telea |

### Feature Extraction

15 features computed from the context ring:

| Feature | Method |
|---------|--------|
| `mean_rgb` / `median_rgb` / `std_rgb` | Ring pixel statistics |
| `brightness` | Grayscale mean |
| `saturation` | HSV S-channel mean |
| `texture_variance` | Laplacian variance |
| `edge_density` | Canny edge ratio |
| `dark_pixel_ratio` | Pixels <= 50 luma |
| `white_pixel_ratio` | Pixels >= 230 luma |
| `product_pixel_ratio` | Overlap with product protection mask |
| `gradient_strength` | Sobel gradient magnitude |
| `seam_risk` | Border strip luma discontinuity |
| `has_text_like_edges` | Connected component analysis (small edge clusters) |
| `has_long_lines` | Elongated edge components (w>30, h<5 or inverse) |

## Strategy Bank

Each ROI class maps to an ordered list of tools. The progressive repair loop tries them in order and stops at the first QA-passing candidate.

```
plain_white:             clone_8dir -> hard_paste -> white_noise -> ring_median ->
                         noise_transfer -> micro_tile -> stroke_mask -> logo+clone ->
                         telea -> adaptive_cover

near_white:              clone_8dir -> seam_scored -> white_noise -> ring_median ->
                         surface_gradient -> stroke_mask -> logo+clone -> telea ->
                         adaptive_cover

low_texture_background:  clone_8dir -> ring_median -> lowpass_match ->
                         surface_gradient -> logo+clone -> telea -> lama_stroke ->
                         adaptive_cover

simple_product_surface:  same_surface -> same_color -> color_match ->
                         surface_gradient -> noise_preserved -> logo+surface ->
                         stroke_mask -> telea -> lama_stroke -> adaptive_cover

dark_product_surface:    dark_clone -> same_surface -> gamma_match ->
                         contrast_match -> stroke_mask -> logo+surface -> telea ->
                         lama_stroke -> adaptive_cover

glass_or_gradient:       linear_gradient -> bilinear_gradient -> glass_clone ->
                         frosted_noise -> low_freq_gradient -> high_freq_noise ->
                         poisson -> logo+surface -> lama_stroke -> adaptive_cover

metallic_or_reflective:  same_surface -> color_match -> linear_gradient ->
                         stroke_mask -> telea -> adaptive_cover

thin_flex_cable:         stroke_mask -> template_logo -> logo+clone ->
                         edge_aware -> cable_repair -> black_flex -> telea ->
                         lama_stroke -> adaptive_cover

text_or_label_area:      stroke_mask -> logo+clone -> telea -> adaptive_cover

complex_product_detail:  template_logo -> stroke_mask -> text_filter ->
                         logo+telea -> edge_aware -> contour_guided ->
                         patchmatch -> lama_stroke -> lama_context ->
                         adaptive_cover

unknown:                 clone_8dir -> stroke_mask -> surface_gradient ->
                         telea -> adaptive_cover
```

## Local QA Gate

Every repair candidate is evaluated by 7 metrics. All must pass for the candidate to be accepted.

| Metric | What It Measures | Threshold | Strict | Loose |
|--------|-----------------|-----------|--------|-------|
| `watermark_residual` | High-pass residual activity in repair zone | <= 0.12 | 0.096 | 0.12 |
| `cover_visibility` | Visible rectangle (luma delta + HF drop) | <= 0.20 | 0.16 | 0.26 |
| `seam_delta` | Boundary discontinuity at repair edges | <= 0.18 | 0.144 | 0.216 |
| `product_damage` | Edge retention loss outside repair zone | <= 0.15 | 0.15 | 0.15 |
| `texture_consistency` | Laplacian texture variance ratio | >= 0.30 | 0.30 | 0.30 |
| `color_delta` | CIE Lab color distance (repair vs ring) | <= 12.0 | 12.0 | 12.0 |
| `edge_damage` | Edge density loss inside repair zone | <= 0.15 | 0.15 | 0.15 |

**Threshold classes:**
- **Strict** (`complex_product_detail`, `thin_flex_cable`, `text_or_label_area`): Thresholds tightened by 20% — these ROIs are high-risk for visible artifacts
- **Loose** (`plain_white`, `near_white`, `low_texture_background`): Thresholds relaxed by 20-30% — these ROIs are forgiving

**Composite score** (lower = better):
```
final = 2.0 * watermark_residual + 2.0 * cover_visibility + 2.5 * seam_delta
      + 1.5 * product_damage + 0.5 * (1 - texture_consistency)
      + 1.0 * color_delta/12 + 1.0 * edge_damage
```

When no tool passes QA, the best-failed candidate (lowest composite score) competes against the final adaptive cover. The better one wins.

## Shared Helpers

Core functions reused across all 100 tools:

| Helper | Purpose |
|--------|---------|
| `_get_ring()` | Extract context ring mask around bbox |
| `_ring_stats()` | Median color + noise sigma from ring |
| `_add_noise()` | Gaussian noise injection (matched sigma) |
| `_feather_blend()` | Gaussian feather blend at bbox boundary |
| `_compute_seam_delta()` | 4-border seam discontinuity score |
| `_get_clone_patch()` | Extract clone source from specified direction |
| `_score_patch()` | Score a candidate patch (edge + texture + color + seam) |
| `_apply_clone()` | Paste clone patch with feather blend |
| `_best_clone_from_directions()` | Try 8 directions, pick best by score |
| `_fit_gradient_plane()` | Least-squares linear gradient from ring |
| `_fit_bilinear_gradient()` | Bilinear gradient from 4 border strips |
| `_apply_stroke_inpaint()` | Inpaint using stroke-level mask |
| `_fill_stroke_from_clone()` | Fill stroke pixels with directional clone |
| `_color_delta_lab()` | CIE Lab color distance |
| `_ssim_local()` | Local SSIM between original and repair |

## Pipeline Integration

`progressive_repair.py` is imported by `mark_remover.py` and replaces the old V7 multi-candidate loop in `process_image()`:

```python
from progressive_repair import (
    analyze_roi as pr_analyze_roi,
    repair_image_progressively,
    save_debug_trace,
    RepairContext,
)
```

The integration point:
1. Existing detection, presence gate, mask construction, and product protection remain unchanged
2. `pr_analyze_roi()` classifies the ROI using the 11-class system
3. A `RepairContext` is built from the image, bbox, masks, and ROI analysis
4. `repair_image_progressively()` runs the strategy bank loop
5. The returned `RepairCandidate` provides the cleaned image
6. Debug traces are saved as JSON for analysis

## Debug Output

Each processed image produces a `trace.json` in the debug directory:

```json
{
  "filename": "product-image.jpg",
  "roi_class": "plain_white",
  "watermark_bbox": [120, 180, 280, 24],
  "tools_tried": [
    {
      "tool": "clone_best_of_8_dirs",
      "passed": false,
      "reason": "seam_delta_too_high",
      "qa": { "seam_delta_score": 0.22, "..." : "..." },
      "runtime_s": 0.012
    },
    {
      "tool": "white_patch_with_noise",
      "passed": true,
      "reason": "accepted",
      "qa": { "final_score": 1.85, "..." : "..." },
      "runtime_s": 0.003
    }
  ],
  "final_method": "white_patch_with_noise"
}
```

## Output Statuses

| Status | `publish_ok` | Meaning |
|--------|-------------|---------|
| `clean_repaired` | `true` | A strategy bank tool passed QA. Watermark invisibly removed. |
| `clean_covered` | `true` | Final adaptive cover applied. Watermark hidden naturally. |
| `no_watermark` | `true` | Presence gate determined no watermark. Image untouched. |
| `failed_io` | `false` | Corrupt or unreadable file. |

## Usage

### Basic run (50 random images)

```bash
python3 mark_remover.py --assets /path/to/images --n 50 --seed 42
```

### Custom output directory

```bash
python3 mark_remover.py --assets /path/to/images --n 100 --out /path/to/output
```

### Dry run (list candidates without processing)

```bash
python3 mark_remover.py --assets /path/to/images --dry-run --n 50
```

### Full options

| Flag | Default | Description |
|------|---------|-------------|
| `--assets` | `assets` | Image directory to scan |
| `--n` | `50` | Number of images to process |
| `--max-scan` | `2000` | Maximum images to scan for candidates |
| `--seed` | `42` | Random seed for reproducible sampling |
| `--out` | `output` | Output directory |
| `--presence-gate` | on | Enable 2-stage presence verification |
| `--no-presence-gate` | -- | Disable presence gate |
| `--exclude-iphone14-plus-watermark-check` | on | Skip iPhone 14+ images |
| `--dry-run` | off | List candidates without processing |
| `--no-pdf` | off | Skip PDF generation |

## Output Structure

```
output/
├── compare.html              Side-by-side comparison viewer
├── compare.pdf               Printable comparison document
├── summary.jsonl             Machine-readable results
├── presence.jsonl            Presence gate scores
├── clean_repaired/           Watermark invisibly removed
│   └── {image-stem}/
│       ├── original.jpg
│       ├── cleaned.jpg
│       └── qa.json
├── clean_covered/            Watermark hidden by adaptive cover
│   └── {image-stem}/
│       ├── original.jpg
│       ├── cleaned.jpg
│       └── qa.json
├── no_watermark/             No watermark detected
└── debug/
    └── {image-stem}/
        └── trace.json        Progressive repair trace
```

## File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `mark_remover.py` | ~4,900 | Main pipeline: detection, masks, QA, orchestration |
| `progressive_repair.py` | ~2,930 | V8 strategy bank: 100 tools, ROI classifier, QA gate |
| `detector.py` | ~4,500 | Multi-stage watermark detection engine |
| `watermark-template.png` | -- | Canonical 24px watermark template for logo mask bank |

## Dependencies

```
opencv-python>=4.5
numpy>=1.20
pillow>=9.0
pytesseract>=0.3       # optional, for deep OCR presence check
```

Optional for neural inpainting (Group F tools):
```
torch>=1.10            # LaMA, DeepFill, MAT
```

## Version History

| Version | Architecture | Key Change |
|---------|-------------|------------|
| V1-V4 | Single-method | Fixed inpainting with manual review routing |
| V5 | No-manual auto-cover | Eliminated manual review; soft cover fallback |
| V6 | Adaptive cover quality | 7 cover methods, local recognizability scoring |
| V7 | Logo mask bank | Multi-scale logo template masks, stroke-level repair |
| **V8** | **Progressive repair strategy bank** | **100 tools, 11 ROI classes, per-class strategy ordering, local QA gate** |

## License

Proprietary. Internal use only.
