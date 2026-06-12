# Mark Remover — v27

A standalone watermark eraser that removes the **`sunsky-online.com`**
semi-transparent text watermark from B2B product photos and leaves no readable
residue. The tool reads one or many images, locates the watermark with OCR,
inpaints the masked region with LaMa, and verifies the result by re-running the
detector on the cleaned output.

On a 47-image bench (`/Users/alexkou/Downloads/clearmark-v27-20260611/cleaned/`)
v27 produced **47 / 47 cleaned with `residual_after = 0`** and no readable ghost
trace — fixing the failure mode of the previous iteration, where the
anti-aliased glyph **halo** survived as a faintly readable shadow of
`sunsky-online.com` (most visible on flat coloured backs and white adhesive
frames).

The single entrypoint is `v27_clean.py`. There are no candidate banks, no
multi-version gates, no state machines. The pipeline is three calls:

```
detect_watermark_boxes(reader, bgr)   →   pad+dilate mask   →   LaMa inpaint
              ↑                                                       │
              └────────── re-run detector for residual check ─────────┘
```

---

## Quick start

```bash
# Single image
python3 v27_clean.py --input in.jpg --output out.jpg [--debug debug_dir]

# Batch — JSONL of [[n, slug, path], ...] entries
python3 v27_clean.py --batch items.json --out-dir cleaned/ [--debug debug_dir]
```

`--debug` writes a `<slug>_mask.png` and a red-overlay `<slug>_overlay.jpg` per
image so the mask placement can be inspected without re-running detection.

**Dependencies:** Python 3, OpenCV (`cv2`), NumPy, Pillow, `easyocr`,
`simple_lama_inpainting`, PyTorch (CPU is fine).
LaMa weights land at `~/.cache/torch/hub/checkpoints/big-lama.pt` on first
run.

---

## 1. How the watermark position is identified

The watermark is a fixed-string text overlay — `sunsky-online.com` — but the
overlay's transparency, the surface texture underneath, and OCR's CRAFT detector
all conspire to produce noisy, fragmented recognitions. v27 absorbs that noise
in three layers.

### 1.1 Multi-pass EasyOCR detection

`detect_watermark_boxes()` runs up to three OCR passes, each preceded by a
**CLAHE contrast boost on the L channel + cubic upscale**, and **stops at the
first pass that yields a match**:

| Pass | Scale | CRAFT params | Purpose |
|-----:|------:|-------------|---------|
| 1 | 2.0× | EasyOCR defaults | Catches the majority of faint marks on smooth backgrounds. |
| 2 | 3.0× | `contrast_ths=0.01`, `adjust_contrast=0.9`, `text_threshold=0.4`, `low_text=0.2`, `link_threshold=0.3` | Recovers extremely faint marks on near-white surfaces. |
| 3 | 2.0× | `contrast_ths=0.01`, `adjust_contrast=0.9`, `text_threshold=0.3`, `low_text=0.15`, `link_threshold=0.2` | Broad search on busy-background pages where pass 1 / 2 found nothing. |

**Why default-first matters.** Looser thresholds raise recall *and* fragment the
text into chunks too short for the fuzzy matcher (§1.2). Running them after the
default pass means tighter, more correct boxes win when they exist; the looser
passes only fire when nothing else worked.

A detection is kept if `conf ≥ 0.03` **and** the recognized text matches the
fuzzy regex (§1.2). Coordinates are scaled back to the original resolution
before being unioned.

### 1.2 Fuzzy regex for OCR-garbled `sunsky-online.com`

OCR rarely reads the watermark cleanly. Observed garblings include
`sunsky-Snline:=`, `sunsky-online cemn|`, `sunshy-onite:com`, `onlme com`,
`Snline c8n`, `hline com`. The matcher is a union of permissive patterns covering
each token:

```
s..k.?   snl.ne   onl..e   nline   hline
.c..m    c8n      cemn     ine com  nl.ne   sky
```

Any one fragment match qualifies the OCR box. Multiple matches across the line
are merged in §1.3.

### 1.3 Single bounding box

`union_bbox()` returns the axis-aligned union of every matched fragment box —
one rectangle per image, regardless of how many OCR pieces the watermark broke
into.

### 1.4 Generous padding

`pad_bbox()` expands the union by **20 % of bbox width on each horizontal side**
and **35 % of bbox height vertically**, with a floor of **8 px** on either axis.
The vertical pad is deliberately larger than the horizontal one: the
anti-aliased glyph **halo** extends roughly one glyph-height above and below the
text, and that halo is what made v26 leave readable ghosts.

### 1.5 No detection → no edit

When `detect_watermark_boxes()` returns nothing, the image is reported with
`status = "no_mark"` and not touched. v27 has no mark-presence prior beyond
OCR — if no OCR pass + fuzzy regex turned up the watermark, nothing is written.

---

## 2. The mark-cleaning strategy

v27 uses **one** strategy: a generously padded, morphologically dilated,
filled-rectangle mask, then a single LaMa inpaint at native resolution. There
are no per-surface variants, no reverse-alpha, no covers, no clones.

### 2.1 Build the mask

`build_mask()` rasterizes the padded bbox as a filled white rectangle on a black
canvas at the source resolution, then dilates it with a circular structuring
element of radius **6 px**:

```python
m[y1:y2, x1:x2] = 255
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
m = cv2.dilate(m, k, iterations=1)
```

**Filled, not stroke.** A stroke-only mask (the dark glyph pixels) is what made
v26 fail — LaMa erased the strokes but the surrounding anti-aliased pixels
(`α ∈ [0.02, 0.10]`) carried enough tint to remain readable. A filled rectangle
plus the 6 px dilation guarantees the halo is in the mask.

### 2.2 LaMa inpaint at native resolution

`lama_inpaint()` calls `simple_lama_inpainting.SimpleLama` with `device='cpu'`
(MPS occasionally breaks on convolutional ops with arbitrary sizes; CPU is
deterministic and still tractable for catalog images). The image and mask are
passed at full resolution; if the model returns a different size the result is
re-scaled with `cv2.INTER_LANCZOS4` to match.

LaMa is the right inpainter for this workload because **≈ 95 % of catalog
images carry the watermark on a flat or smooth-gradient background** —
near-white trays, white backdrops, soft blue/yellow/green coloured backs — which
is LaMa's strongest regime.

### 2.3 Single retry on residual

If §3.1's residual check finds anything that still matches the watermark
pattern, the cleaner runs **one** retry with a wider mask:

| | First pass | Retry |
|---|---|---|
| Horizontal pad | 20 % bbox width | 30 % |
| Vertical pad | 35 % bbox height | 50 % |
| Minimum pad | 8 px | 12 px |
| Dilation | 6 px | 10 px |

The retry's bbox is the union of the first pass's padded box *and* the residual
OCR hit, so the next mask physically covers what survived. There is no third
retry — if the retry also fails, the image is written and reported with
`status = "residual"` so the operator sees the truth instead of a forged pass.

### 2.4 What v27 deliberately does *not* do

- **No reverse-alpha.** Reverse-alpha needs the solved per-pixel `α` map for the
  watermark; v27 trades that complexity for a simpler filled-box + LaMa
  inpaint. On flat or smooth-gradient backgrounds this is a strict win.
- **No covers / clones / synthetic fills.** Inpaint only.
- **No diffusion-based generative inpainting.** LaMa was chosen for its
  faithfulness on flat backgrounds without hallucinating geometry or text.
- **No per-surface specialization.** A single recipe handles every sample in
  the bench; complexity is only justified when it removes a real failure.

---

## 3. Cleaning-quality review: methods & criteria

The quality review runs **on the bytes that would be written**, not on
intermediate state, and the resulting status is recorded verbatim.

### 3.1 Non-circular residual check

After inpainting, `detect_watermark_boxes()` is re-run on the **cleaned** image
using the same multi-pass OCR + fuzzy regex. "Non-circular" means the residual
check shares the detector but **does not reuse the detection** — it does a
fresh OCR pass on the new pixels. A residual detection therefore proves OCR can
still read the watermark on the output, not an artifact of cached state.

The retry described in §2.3 is the only response to a non-empty residual. After
the retry, the residual count is recorded in the report — there is no second
retry, and the image's `status` is set from the *final* residual list, not the
intermediate one.

### 3.2 Per-image status taxonomy

Every image lands in exactly one of these states, written verbatim into the
JSON record:

| `status` | Meaning |
|---------|---------|
| `no_mark` | No OCR pass found a fuzzy-regex match — image is left untouched. |
| `cleaned` | Mark detected, inpainted, **and** residual check came back empty (zero matches after the final pass). |
| `residual` | Mark detected and inpainted, but the post-inpaint OCR still flagged something. The image is still written to `--out-dir`, but the count of residual hits is recorded for review. |
| `ok = False` (`err = "imread failed"`) | `cv2.imdecode` could not read the input. |

### 3.3 Per-image record fields

Each clean writes a JSON record (printed on `--input` runs, accumulated to
`<out-dir>/_v27_log.json` on `--batch` runs) with:

| Field | Source | Used for |
|-------|--------|---------|
| `status` | §3.2 | Pass/fail classification |
| `matches[]` | Pass 1/2/3 OCR survivors | Sanity-check what was detected |
| `bbox_padded` | `pad_bbox()` output | Mask placement audit |
| `retry` | §2.3 trigger | Detects which images needed the wider mask |
| `residual_after` | Length of post-inpaint detection list | Hard quality gate |
| `out` | Output path | Provenance |
| `elapsed_s` | Per-image wall time (batch mode) | Throughput visibility |

### 3.4 Debug artifacts (`--debug`)

For every image processed, `--debug <dir>` writes:

- `<stem>_mask.png` — the final dilated mask actually fed to LaMa.
- `<stem>_overlay.jpg` — the source image with the mask blended at 45 %
  opacity in red, so a human reviewer can confirm the mask sits on the
  watermark and not on product geometry.

### 3.5 Run-level acceptance criteria

Used to qualify a batch (the way the 47-sample bench was qualified):

```
residual_after == 0     for every record  →  zero readable residue
status ∈ {cleaned, no_mark}                →  no forced "residual" publish
retry rate                                  →  tracked, not gated; high retry
                                              rate flags a regression in §1
mask overlay (human spot-check)             →  no mask landing on product text,
                                              labels, or silhouette edges
```

The bench at `/Users/alexkou/Downloads/clearmark-v27-20260611/cleaned/`
(47 images) reports **47 / 47 with `residual_after = 0`** and a single side-by-
side PDF at `/Users/alexkou/Downloads/cleaned_47_v27_before_after.pdf` for the
human eyeball check.

### 3.6 Scope & honest limits

- v27 is the **lab / standalone cleaner**. It is the right tool when the
  watermark sits on a flat or smooth-gradient background (≈ 95 % of the catalog
  on the bench), and was qualified on a 47-image set that matches that
  distribution.
- A **filled-rectangle mask on product surface** is a known risk: LaMa will
  happily inpaint over product text, labels, and intricate geometry. v27 has
  no product-mask, no product-text guard, and no silhouette check; the
  `--debug` overlay is the only check that the mask is not over product
  pixels. On images with a busy / product-overlapped watermark, expect the
  mask to be too aggressive.
- For the brand-safe production path (where erasing product detail is
  unacceptable), the integrated b2bweb pipeline at
  `https://github.com/alexmorerich/b2bweb` (`scripts/remove-watermarks.py`)
  carries the additional gates — rect-mask rejection, product-damage scoring,
  protected-text checks — that v27 deliberately omits. Treat v27 as the
  **algorithm-side proof** that a generous mask + LaMa clears the halo-ghost
  failure, and port it across with those gates when used on production
  product images.

---

## 4. File map

| File | Role |
|------|------|
| `v27_clean.py` | The entrypoint. Detection, masking, LaMa inpaint, residual check, CLI for single and batch runs. |
| `watermark-template.png` | Reference glyph of `sunsky-online.com`, retained for cross-checking; v27 uses OCR rather than template correlation. |

> **Legacy engine.** The pre-v27 stack (V10–V23 — `mark_remover.py`,
> `detector.py`, `progressive_repair.py`, the `vN_*.py` modules, and their
> tests) was removed in the v27 cleanup. None of it was used by `v27_clean.py`.
> It is preserved in git history under the **`legacy-v23-engine`** tag; restore
> any file with `git checkout legacy-v23-engine -- <path>`.

### Design principle

> One pass of OCR-driven generous masking + LaMa inpaint is enough for the
> 95 % flat-background majority of catalog images, and a single wider-mask
> retry handles the remainder. Anything that resists the retry is reported
> truthfully as `residual` rather than published clean.
