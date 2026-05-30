# Mark Remover

Precision-first watermark detection and removal pipeline for sunsky-online.com product images. Designed for batch processing with zero false positives on auto-clean and human-in-the-loop routing for borderline cases.

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Clean rate | ≥60% | 80% |
| Manual-review rate | ≤25% | 20% |
| False no-watermark | 0 | 0 |
| Visible watermark in clean | 0 | 0 |
| Product damage | 0 | 0 |

## Design Concept

Mark Remover follows a **precision-over-recall** philosophy. The system would rather send an image to manual review than risk publishing a visible watermark or damaged product photo. Every stage is designed to fail safely.

### Pipeline Architecture

```
Image Pool
    │
    ▼
┌──────────────────────┐
│  Candidate Selection │  Model exclusion (iPhone 14+ skip)
│                      │  Template detection (detect_watermark_v2)
│                      │  Presence pre-filter (fast presence gate)
└──────────┬───────────┘
           │ watermark-confirmed images only
           ▼
┌──────────────────────┐
│  Presence Gate       │  Fast: template + stroke + CLAHE + position + aspect ratio
│  (2-stage)           │  Deep: OCR re-confirmation for uncertain cases
└──────────┬───────────┘
           │ CONFIRMED / UNCERTAIN
           ▼
┌──────────────────────┐
│  ROI Classification  │  12 background classes determine method routing
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Product Protection  │  Edge overlap, dark components, circular features
│  Gate                │  Severe overlap → reject, moderate → manual
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Mask Construction   │  Stroke-level segmentation of watermark text
│                      │  Fallback: soft mask, repair-guarded mask
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Multi-Method        │  Up to 6 inpainting candidates per image
│  Inpainting          │  Method routing by ROI class
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Quality Assurance   │  Product integrity, artifact detection,
│  Gates               │  residual watermark check, local ROI QA,
│                      │  JPEG-decode verification
└──────────┬───────────┘
           │
           ▼
    CLEAN / MANUAL-REVIEW / REJECTED
```

## Core Components

### 1. Watermark Detection (`detect_watermark_v2`)

Multi-scale edge-correlation detector tuned for the semi-transparent "sunsky-online.com" text watermark.

**How it works:**
- Searches a center column (x ≈ 0.50 of image width) in the vertical band y ∈ [0.40, 0.60]
- Matches a polarity-invariant gradient template (Sobel edge map) at 7 height scales (0.040–0.064 of image height)
- Uses Gaussian-weighted centeredness scoring — detections near y=0.48 score higher
- Peak-to-sidelobe ratio (PSR) separates real text peaks (~6.0–12.4) from product clutter (≤5.72)
- CLAHE enhancement recovers faint watermarks on dark/busy PCBs (+42% recall, 0 added false positives)

**Detection tiers:**
| Tier | Meaning | Criteria |
|------|---------|----------|
| `auto` | High confidence, safe to auto-clean | Edge ≥ threshold, NCC ≥ threshold, PSR ≥ 6.0 |
| `manual` | Probable watermark, needs review | Looser thresholds + rescue clauses for faint marks |
| `none` | No watermark detected | Below all thresholds |

### 2. Presence Gate (2-Stage Verification)

Reduces false positives by scoring seven independent signals before committing to processing.

**Fast stage — weighted signal combination:**

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Template match | 28% | Edge correlation + NCC with watermark template |
| Stroke pattern | 22% | Coverage uniformity + contrast of detected strokes |
| CLAHE center band | 18% | High-pass residual activity in the watermark band |
| Aspect ratio | 12% | Deviation from expected 8.5:1 watermark proportions |
| Vertical position | 10% | Centering near expected y=0.48 |
| Repeat pattern (PSR) | 10% | Peak sharpness in correlation response |
| Negative product text | −30% penalty | Edge/dark density in surrounding area (penalizes busy regions) |

**Confidence bands:**
- < 0.15 → `NO_WATERMARK` (skip processing)
- 0.15–0.45 → `UNCERTAIN` (proceed to deep check)
- ≥ 0.45 → `CONFIRMED_WATERMARK` (proceed to cleaning)

**Deep stage** adds OCR (30% weight) — searches for "sunsky", "online", ".com" variants using Tesseract on the watermark region. Resolves uncertain cases.

### 3. Candidate Selection (`find_candidates`)

Pre-filters the image pool before any processing begins:

1. **Model exclusion** — iPhone 14+ images are skipped entirely (they never carry sunsky watermarks)
2. **Template detection** — `detect_watermark_v2` must return auto or manual tier
3. **Presence pre-filter** — Manual-tier detections with template score < 0.35 are rejected (catches product text/edge false positives)
4. **Watermark-only quota** — Only confirmed detections count toward the requested sample size

### 4. ROI Classification (12 Background Classes)

The watermark region's background determines which inpainting method will produce the best result. Features extracted: edge density, texture energy, chroma statistics, local contrast, specular fraction, dark fraction, luma mean, and small-component density.

| Class | Example | Primary Method |
|-------|---------|---------------|
| `plain_white` | White product backgrounds | `gradient_plane` / `white_fill` |
| `plain_color` | Solid colored backgrounds | `gradient_plane` / `local_color_plane` |
| `low_texture` | Subtle gradients, light textures | `gradient_plane` |
| `gradient` | Color gradients | `gradient_plane` / `poisson` |
| `transparent` | Glass, clear materials | `poisson` / `telea` |
| `glossy` | Reflective, shiny surfaces | `poisson` |
| `metallic` | Metal components, connectors | `poisson` |
| `black_product` | Dark PCBs, dark products | `lama` |
| `high_texture` | Dense textures, silkscreen | `lama` |
| `product_detail` | Fine details, components | `telea_small` |
| `text_or_qr` | Product text, QR codes | `telea_small` |
| `poster_or_marketing` | Marketing layouts | **REJECTED** (unsuitable) |

### 5. Inpainting Methods

Seven algorithms, each suited to different background types:

| Method | Technique | Best for |
|--------|-----------|----------|
| `white_fill` | Direct white replacement | Plain white backgrounds |
| `local_color_plane` | Local color sampling | Solid color backgrounds |
| `gradient_plane` | Least-squares gradient estimation from surrounding ring | Gradients, plain surfaces |
| `alpha_attenuate` | Alpha-channel watermark attenuation | Semi-transparent overlays |
| `full_box_attenuate` | Full bounding box replacement with gradient plane | Plain/low-texture only |
| `telea` / `telea_small` | OpenCV Telea inpainting (r=5 / r=2) | Medium texture, fine details |
| `poisson` | Poisson blending with surrounding texture | Complex gradients, metallic |
| `lama` | LaMA neural network inpainting | High texture, dense patterns |

Each image tries up to 6 candidate methods in sequence. The first candidate that passes all QA gates wins.

### 6. Mask Construction

Three mask strategies, selected by stroke detection confidence:

- **Stroke mask** (confidence ≥ 0.25, >20 pixels) — Precise segmentation of individual watermark text strokes. Minimal surrounding damage. Best results.
- **Repair-guarded mask** — Expanded stroke mask with Gaussian feathering. Used when stroke confidence is moderate.
- **Soft fallback mask** — Gentle box-region mask with heavy feathering. Last resort when strokes can't be isolated.

Dilation and expansion ratios are tuned per ROI class — plain backgrounds get minimal dilation (3–4px), complex backgrounds get larger margins (7–9px).

### 7. Quality Assurance Gates

Every cleaned image passes through multiple independent QA checks:

**Product integrity** (pre and post inpaint):
- Edge retention — Canny edges preserved outside watermark area
- SSIM — Structural similarity between original and cleaned
- Outside-mask drift — Accidental modifications beyond the repair zone
- Blank-patch detection — Overly smooth repairs

**Artifact detection:**
- Laplacian drop — Unnatural smoothness increase
- Boundary Delta-E — Color discontinuity at mask edges
- Rectangularity — Visible rectangular patches

**Residual watermark check:**
- Multi-channel LAB/HSV analysis
- High-pass grayscale residual detection
- Template re-match on the cleaned image
- Composite confidence score — ≥ 0.35 flags residual

**Local ROI QA:**
- SSIM inside repair zone vs surrounding strips
- Color delta at repair boundary
- Boundary smoothness / jump detection

**JPEG-decode verification:**
- Re-encodes to JPEG and re-checks — catches artifacts that only appear after compression

### 8. Progressive Cleanup

When no candidate passes QA on the first attempt, the system tries progressive cleanup:

1. Gradient-plane haze removal (strength 0.85)
2. Edge suppression in the watermark band
3. Residual stroke cleanup at increasing strengths
4. JPEG-decode re-verification at each step

This recovers borderline images that have faint residual traces after initial inpainting.

## Output Statuses

| Status | `publish_ok` | Meaning |
|--------|-------------|---------|
| `clean` | `true` | Passed all QA gates. Safe to publish. |
| `manual-review` | `false` | QA flagged an issue. Needs human review. |
| `rejected` | `false` | Unsuitable source (posters, pathological detection, input error). |
| `no-watermark` | N/A | Presence gate determined no watermark. Image untouched. |

## Usage

### Basic run (50 random images)

```bash
python3 mark_remover.py --n 50 --seed 42
```

### Custom output directory

```bash
python3 mark_remover.py --n 100 --out /path/to/output
```

### Dry run (list candidates without processing)

```bash
python3 mark_remover.py --dry-run --n 50
```

### Full options

```bash
python3 mark_remover.py \
  --assets /path/to/product/images \
  --n 50 \
  --max-scan 2000 \
  --seed 42 \
  --out /path/to/output \
  --presence-gate \
  --exclude-iphone14-plus-watermark-check \
  --no-pdf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--assets` | `content/products/assets` | Image directory to scan |
| `--n` | `50` | Number of images to process |
| `--max-scan` | `2000` | Maximum images to scan for candidates |
| `--seed` | `42` | Random seed for reproducible sampling |
| `--out` | `output/zero-dirty-v3` | Output directory |
| `--presence-gate` | on | Enable 2-stage presence verification |
| `--no-presence-gate` | — | Disable presence gate |
| `--exclude-iphone14-plus-watermark-check` | on | Skip iPhone 14+ images |
| `--dry-run` | off | List candidates without processing |
| `--no-pdf` | off | Skip PDF generation |

## Output Structure

```
output/
├── compare.html          # Side-by-side comparison viewer
├── compare.pdf           # Printable comparison document
├── summary.jsonl         # Machine-readable results
├── presence.jsonl        # Presence gate scores for all images
├── clean/                # Successfully cleaned images
│   └── {image-stem}/
│       ├── original.jpg
│       └── cleaned.jpg
├── manual-review/        # Images needing human review
│   └── {image-stem}/
│       ├── original.jpg
│       ├── cleaned.jpg   # Best attempt
│       └── qa.json       # QA metrics and failure reasons
├── rejected/             # Unsuitable images
└── no-watermark/         # Images with no watermark detected
```

## Dependencies

```
opencv-python>=4.5
numpy>=1.20
pillow>=9.0
pytesseract>=0.3       # optional, for deep OCR presence check
```

Optional for LaMA inpainting:
```
torch>=1.10
```

## Architecture Decisions

**Why precision-first?** A visible watermark or damaged product photo on a B2B e-commerce site erodes buyer trust. False positives (processing a clean image) waste compute but cause no harm. False negatives (missing a watermark) are caught by manual review. But auto-publishing a damaged image has real business cost.

**Why 7 inpainting methods?** No single algorithm handles all backgrounds well. White fill works on plain backgrounds but creates rectangles on gradients. LaMA handles textures but is slow and can hallucinate on simple surfaces. The ROI classifier routes each image to the method most likely to succeed, with fallback candidates if the first choice fails QA.

**Why 2-stage presence gate?** The initial template detector (detect_watermark_v2) is tuned for recall — it uses loose manual-tier thresholds to avoid missing faint watermarks. This creates false positives on images with product text or component edges. The presence gate adds precision: seven independent signals vote on whether the detected pattern is actually a sunsky watermark. The deep stage adds OCR for borderline cases.

**Why progressive cleanup?** JPEG compression can re-introduce faint artifacts that weren't visible in the raw cleaned image. The progressive cleanup loop re-checks after JPEG encoding and applies increasing cleanup strength until the residual is gone or the attempt limit is reached.

## License

Proprietary. Internal use only.
