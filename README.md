# Mark Remover — v28

A standalone eraser for the **`sunsky-online.com`** semi-transparent text
watermark on B2B product photos. It locates the mark, inpaints it with LaMa,
and — critically — **re-detects the mark on its own output and only publishes an
image as `cleaned` when the watermark can no longer be found.** If anything
survives, it widens and re-inpaints, or reports the image honestly. There are no
silent dirty passes.

The single entrypoint is **`v28_clean.py`**. It reuses the OCR primitives and
the LaMa wrapper from `v27_clean.py`; everything new (full-extent localization,
per-channel detection, the verification gate, and GPU acceleration) lives in
`v28_clean.py`.

```
                        ┌─────────────────────────────────────────────┐
 detect_smart ──► anchor ──► full extent ──► crop+LaMa ──► verify_clean ──► status
 (OCR ×passes,           (template match +    (MPS GPU,    (re-detect WIDE,   cleaned /
  per-channel)            stroke-aware grow)   paste-back)  composite+channel) residual /
       ▲                                                          │           no_mark
       └──────────────── recover: widen mask, re-inpaint (≤2×) ◄──┘
```

On a 100-image bench the v28 pipeline reports **100 / 100 re-detection-verified
clean** (0 residual, 0 missed) at **~4.7 s/image** on an Apple M4 (single
process, GPU).

---

## Quick start

```bash
# Single image
python3 v28_clean.py --input in.jpg --output out.jpg [--debug debug_dir]

# Batch — JSON array of [n, slug, path] entries
python3 v28_clean.py --batch items.json --out-dir cleaned/ --log cleaned/_log.json [--debug debug_dir]
```

Recommended environment (Apple Silicon):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # let unsupported MPS ops fall back to CPU
export OCR_DEVICE=mps                   # run EasyOCR on the GPU (10x vs CPU)
export LAMA_DEVICE=mps                  # run LaMa on the GPU
```

**Dependencies:** Python 3, OpenCV (`cv2`), NumPy, Pillow, `easyocr`,
`simple_lama_inpainting`, PyTorch. LaMa weights land at
`~/.cache/torch/hub/checkpoints/big-lama.pt` on first run. `assets/sunsky_alpha.png`
(the solved watermark alpha template, 36×240) ships with the repo and is loaded
at import.

`--debug <dir>` writes `<stem>_mask.png` (the final mask) and `<stem>_overlay.jpg`
(the mask blended 45 % red over the source) per image, for human mask-placement
review.

---

## 1. How the watermark position is identified

The mark is the fixed string `sunsky-online.com`, but its **transparency**, the
**surface texture** underneath, and the **random position** conspire to defeat a
single detector. v28 localizes in four layers, each compensating for the
previous one's blind spot.

### 1.1 Multi-pass EasyOCR (CRAFT) detection — `detect_smart`

`detect_smart()` runs up to three EasyOCR passes, each preceded by a **CLAHE
contrast boost on the L channel + cubic upscale**, and **stops early at the
first pass that captures the whole mark** (§1.4):

| Pass | Scale | CRAFT params | Purpose |
|-----:|------:|-------------|---------|
| 1 | 2.0× | EasyOCR defaults | Catches the majority of faint marks on smooth backgrounds. |
| 2 | 3.0× | `text_threshold=0.4, low_text=0.2, link_threshold=0.3` | Recovers very faint marks on near-white surfaces. |
| 3 | 2.0× | `text_threshold=0.3, low_text=0.15, link_threshold=0.2` | Broad search on busy pages where 1–2 found nothing. |

Each recognition is kept only if `conf ≥ 0.03` **and** its text matches the
fuzzy regex below. Coordinates are scaled back to native resolution.

### 1.2 Fuzzy regex for OCR-garbled `sunsky-online.com`

OCR rarely reads the mark cleanly — observed garblings include `sunsky-Snline:=`,
`onlme com`, `Snline c8n`, `hline com`, `Gunsky-onjine con`. The matcher
(`SUNSKY_TOKENS` in `v27_clean.py`) is a union of permissive token patterns
(`s..k.?`, `onl..e`, `nline`, `.c..m`, `c8n`, `sky`, …); any one fragment match
qualifies a box. Matched boxes are unioned into one anchor (`union_bbox`).

### 1.3 Per-channel recall booster (coloured backgrounds)

A mark rendered light-grey **on a saturated colour** (e.g. white-on-blue on a
translucent phone) can be nearly invisible in the composite image and missed by
every pass above — the classic "complete miss". `detect_smart` therefore falls
back to **`_ocr_channels`**, which OCRs each **B, G, R channel separately**. The
mark's contrast against a coloured surface collapses in the composite but
survives in the complementary channel (the blue channel is what recovers the
white-on-blue case). This fires only when the composite passes find nothing, so
it costs nothing on the common case.

### 1.4 Full-extent recovery — `_mark_fully_captured`, `locate_full_box`, `grow_extent_tint`

OCR usually returns only a **fragment** of the mark, and masking just that
fragment leaves a readable tail (the "`…online.com` ghost"). v28 establishes the
**full** extent:

- **`_mark_fully_captured`** decides whether OCR already read the whole string:
  true only if the recognized text contains the genuine tail (`com` / `onlin`)
  **or** the union box is already ≥ 6.7× text-height wide (the mark's aspect).
  A *truncated* read like `sunsky-onl` is explicitly **not** treated as full —
  this was the bug that caused partial removal.
- If not fully captured, **`locate_full_box`** template-matches the known mark
  geometry (`assets/sunsky_alpha.png`, aspect **6.67**) against an
  **absolute-local-contrast stroke map** of a band around the anchor. The search
  band and scale sweep are sized from the **image**, not the OCR fragment (OCR
  may catch a tiny sliver of a wide mark), and the match is constrained to cover
  the anchor. Accepted at correlation ≥ `ACCEPT` (0.24).
- **`grow_extent_tint`** then extends the box left/right along the actually
  visible mark. It is **stroke-aware**: a pixel counts only if it is *both*
  lighter than the local background *and* sits on a high-frequency stroke — so
  the grow follows glyphs but **not** smooth reflective product features (which
  previously caused runaway white-blob inpaints). Growth is **capped** to
  4× box-height per side.

### 1.5 No detection → `no_mark`

If every layer above finds nothing, the image is reported `no_mark` and left
untouched. v28 has no mark-presence prior beyond detection.

---

## 2. The mark-cleaning strategies

### 2.1 Full-extent filled mask — `build_mask`

The padded full-extent box is rasterized as a **filled white rectangle** (not
stroke pixels — a stroke-only mask leaves the anti-aliased glyph halo readable),
then dilated. The horizontal pad is a **fraction of mark width** (10 %, min
14 px) so the faint `.com` tail and leading `s` that template matching tends to
under-cover are always inside the mask; vertical pad is 35 % of height.

### 2.2 Crop-region LaMa inpaint — `lama_inpaint_crop`

The mask is typically **~1 % of the image area**, so inpainting the whole frame
is ~98 % wasted work. `lama_inpaint_crop` crops a small region around the mask
(+24 px context), runs LaMa **only on that crop**, and **pastes back only the
masked pixels** — every unmasked pixel stays byte-identical to the source. This
is the single biggest speed lever (full-image 5.5 s → crop 0.1 s, ~55×) and also
improves fidelity (no global recomposition).

LaMa is the right inpainter here because ≈ 95 % of catalog marks sit on a flat
or smooth-gradient background (white trays, coloured backs), LaMa's strongest
regime, and it does not hallucinate geometry or text.

### 2.3 GPU (MPS) acceleration — `make_reader`, `get_fast_lama`

Both conv-heavy nets run on the Apple-Silicon GPU:

- **LaMa** via `SimpleLama(device="mps")` (`LAMA_DEVICE`).
- **EasyOCR** — its `gpu=` flag only knows CUDA, so `make_reader` moves the
  CRAFT detector and recognizer onto MPS by hand (`r.detector.to("mps")` …),
  ~10× faster than CPU (12 s → 1.2 s per `readtext`).

This matters because on CPU these nets are **memory-bandwidth bound**, so
multi-process CPU workers do *not* scale (they saturate the bus). Moving the work
to the otherwise-idle GPU is what removes the bottleneck. `LAMA_DEVICE=cpu` /
`OCR_DEVICE=cpu` force CPU when needed.

### 2.4 What v28 deliberately does *not* do

- **No reverse-alpha / per-pixel α solve**, no covers, clones, or synthetic
  fills — inpaint only.
- **No diffusion inpainting** — LaMa is chosen for faithfulness over generation.
- **No product-mask / protected-text gate.** A filled rectangle on product
  surface is a known risk; the `--debug` overlay is the human check. For the
  brand-safe production path with those gates, see the b2bweb pipeline.

---

## 3. Cleaning-quality review: methods & criteria

The review runs **on the bytes that would be written**, and the resulting status
is recorded verbatim. The guiding rule: **an image is `cleaned` only if the mark
can no longer be detected on the output.**

### 3.1 Hard verification gate — `verify_clean`

After inpainting, `verify_clean` re-detects the watermark on the **cleaned**
image over a region **wider than the mask** (+0.6× mark-width horizontally, +1×
height vertically — residue can survive *just outside* the mask, as in the
`.com`-tail case), using **both** the composite image **and** each colour channel
(the §1.3 recall booster, so faint residue is caught). It returns the surviving
watermark hits in image coordinates; an empty list means verified clean.

This is "non-circular": it does a **fresh** multi-modal detection on the new
pixels, not a re-read of cached state, so a hit proves the mark is still findable
on the output.

### 3.2 Recovery loop

If `verify_clean` returns residue, the cleaner **widens** the box to cover the
residue, re-grows, builds a harder mask (16 % horizontal pad, 50 % vertical,
larger dilation), re-inpaints, and re-verifies — **up to 2×**. Recovery resolves
the rare under-cover before any status is assigned.

### 3.3 Per-image status taxonomy

| `status` | Meaning |
|---------|---------|
| `no_mark` | No detector layer found the mark — image left untouched. |
| `cleaned` | Mark detected, inpainted, **and** the final `verify_clean` returned empty (zero re-detections). |
| `residual` | Mark detected and inpainted, but re-detection still flags it after recovery. Written to `--out-dir`, but flagged — **never presented as clean**. |
| `ok=False` | `cv2.imdecode` could not read the input. |

### 3.4 Per-image record fields

Each clean writes a JSON record (printed on `--input`, accumulated to `--log` on
`--batch`):

| Field | Meaning |
|-------|---------|
| `status` | §3.3 classification. |
| `method` | `ocr_full` (OCR read the whole mark), `template` (recovered via template match), or `ocr_pad`. |
| `anchor` / `full_box` | OCR union box / final masked extent — mask-placement audit. |
| `loc_score` | Template-match correlation for the localization. |
| `retry` | Number of recovery passes (§3.2). |
| `residue_after` | Re-detection count on the final output — **the hard gate** (0 ⇔ `cleaned`). |
| `resid_score` | Template residual correlation (informational; not the gate — it over-flags smudges and clean white backgrounds). |
| `elapsed_s` | Per-image wall time. |

### 3.5 Run-level acceptance criteria

```
residue_after == 0     for every record         →  no re-detectable mark
status ∈ {cleaned, no_mark}                      →  no published residual
retry rate                                        →  tracked, not gated; high rate
                                                     flags a §1 detection regression
mask overlay (human spot-check, --debug)          →  mask not over product text / edges
side-by-side before/after PDF                     →  final human eyeball
```

### 3.6 Scope & honest limits

- v28 is the **lab / standalone cleaner**, qualified on benches whose marks sit
  on flat or smooth-gradient backgrounds (≈ 95 % of the catalog).
- The verification gate is **OCR-readability based**: it reliably catches
  *readable* residue but a sub-readable faint ghost can pass. Combined with the
  full-extent mask this is rare in practice.
- A faint mark on a busy, low-contrast background may still defeat detection
  (`no_mark`); the per-channel booster (§1.3) is the mitigation.
- On dark textured flex cables LaMa can leave a faint cosmetic **smudge** in the
  masked footprint — no readable text, but visible on close inspection.
- A filled rectangle can erase product detail if the mark overlaps geometry;
  there is no product-protection gate (see §2.4).

---

## 4. Performance & scaling

**Measured (Apple M4, single process, GPU):** ~4.7 s/image average (min ~1.1 s,
max ~69 s on the hardest channel-search cases). Per-image cost is dominated by
**EasyOCR** (detection + verification); LaMa is negligible after cropping
(~0.05 s).

| Lever | Effect | Notes |
|---|---|---|
| Crop-region inpaint (§2.2) | LaMa 5.5 s → 0.1 s (~55×) | already in v28 |
| GPU/MPS for OCR + LaMa (§2.3) | OCR 12 s → 1.2 s (~10×); per-image 38 s → ~5 s | already in v28 |
| Horizontal scale-out | **linear** | embarrassingly parallel; shard images across N GPUs/machines |
| OCR batching | 2–4× | feed K images per CRAFT/CRNN forward pass to saturate the GPU |
| Thoroughness tuning | 1.5–2× | composite-only verify unless coloured surface; skip channel passes when not needed |
| Lighter watermark detector | 1.5–2× | replace general EasyOCR with template/matched-filter for the common case, OCR as fallback |
| Bigger GPU (M-Max/Ultra, A100/L4) | 2–5× | conv nets scale with GPU |

**Throughput targets for 10 000 images:**

| Setup | Time |
|---|---|
| One M4, v28 as-is | ~13 h |
| One M4 + batching + tuning | ~3–4 h |
| One cloud A100 + batching | ~1–2 h |
| Scale-out, 10–20 GPU instances | ~15–30 min |

CPU multi-processing is **not** a lever — the nets are memory-bandwidth bound on
CPU. The reliable speedups are GPU utilization (batching) and horizontal
scale-out.

---

## 5. File map

| File | Role |
|------|------|
| `v28_clean.py` | The entrypoint: detection, full-extent localization, crop+MPS LaMa inpaint, verification gate, recovery, CLI. |
| `v27_clean.py` | Reused primitives: EasyOCR passes (`_ocr_pass`), fuzzy regex (`SUNSKY_TOKENS`), `union_bbox`, `pad_bbox`, base LaMa wrapper. |
| `assets/sunsky_alpha.png` | Solved 36×240 alpha template of the mark (aspect 6.67), used by `locate_full_box`. |
| `assets/sunsky_alpha_meta.json` | Template provenance (canonical size, peak α, logo luma). |

### Design principle

> Localize the **whole** mark (OCR + per-channel recall + template + stroke-aware
> growth), inpaint **only** its footprint on the GPU, then **prove** it is gone
> by re-detecting on the output. Publish `cleaned` only when nothing is found;
> otherwise recover or report the truth.
