# Mark Remover

Removes the **`sunsky-online.com`** semi-transparent text watermark from B2B product
photos at catalog scale, leaving no readable residue and **no damage to clean images**.

The system is a single-owner, three-stage pipeline with stable typed interfaces:

```
   DETECT                         REPAIR                         AUDIT
 logo_finder.py        →   v28_clean.py / reclean.py     →   verify gate / final_audit.py
 (find every mark,         (full-extent or canonical-band     (re-detect on the OUTPUT;
  graded presence)          LaMa inpaint, or restore FP)       publish only if gone & safe)
        │                          │                                  │
        │  presence + bbox         │  backup every change             │  residual re-check
        │  + masks + ROI + risk    │  (fully reversible)              │  + auto-reject queue
        └──────────────────────────┴──────────────────────────────────┘
                       one append-only manifest per image
```

**Safety model — "aggressive to find, conservative to publish."** Detection chases recall
freely because the zero-damage guarantee lives at the **audit gate**: every repair is
backed up and reversible, and an image is published `cleaned` **only when the mark can no
longer be detected on its own output**. Final safety is never assumed at detect time.

**Governance.** One owner; `detect`, `repair`, and `audit` are separate modules behind a
stable contract. Before changing repair, inspect the detection output schema; before
changing detection, run the end-to-end benchmark. Every change preserves manifest
compatibility, the final audit, and the auto-reject policy. The detect contract is
[`docs/OWNER_LOGO_FINDER.md`](docs/OWNER_LOGO_FINDER.md).

| Module | Stage | Responsibility |
|---|---|---|
| `logo_finder.py` | DETECT | aggressive ensemble finder → graded presence + scored candidates + masks + routing |
| `v27_clean.py` | DETECT (primitives) | multi-pass EasyOCR passes + fuzzy `sunsky` token regex |
| `v28_clean.py` | REPAIR (integrated engine) | `detect_smart` → full-extent mask → crop+MPS LaMa → hard verify gate + recovery |
| `reclean.py` | REPAIR (catalog scale) | canonical-band LaMa re-clean / restore false positives, resumable |
| `run_bulk.py` | REPAIR (engine) | LaMa crop-inpaint, OCR reader, in-place write + backup |
| `final_audit.py` | AUDIT | independent re-detection over repaired output, FP-graded |
| `reconcile_manifest.py` | — | merge repair results into one authoritative manifest |

---

## 1. How the watermark position is identified

The watermark is a **fixed centre stamp**: across 700 reliable detections its vertical
centre `cy/H` sat in `[0.469, 0.487]` (p5–p95), width ≈ `0.45·W`, height ≈ `0.06·H`. That
tight geometric prior anchors everything. But transparency, the surface texture
underneath, and OCR garbling defeat any single detector, so detection is an **ensemble**
that is then **graded** rather than thresholded on one score.

### 1.1 Multi-representation preprocessing
Each image is expanded into several views so no representation is a single point of
failure: grayscale, CLAHE-enhanced gray, a **faint-stroke residual map**
(`|gray − medianBlur|`, which makes a semi-transparent mark pop), a Canny edge map, and an
HSV saturation map.

### 1.2 The ensemble signals (`logo_finder.py`)
| Signal | What it catches | Blind spot it covers |
|---|---|---|
| **OCR text** (`v27._ocr_pass`, sensitive, multi-pass + CLAHE + 2× upscale) | legible `sunsky-online.com` fragments | the primary, high-precision signal |
| **Per-channel OCR** (saturation-gated) | a light mark over a saturated colour that vanishes in the luminance composite | colour-hidden marks |
| **Canonical matte NCC** (b2bweb `watermark-canonical.png`, background-subtracted, multi-scale) | the glyph **shape** where OCR reads nothing | illegible / faint marks |
| **Center prior** (Gaussian on distance from `cy/H≈0.475`) | location plausibility | rejects edge-of-frame text |
| **Alpha-gray energy** | faint, low-saturation, thin horizontal stroke band | transparency signature |
| **Edge / dot-chain rhythm** (column-projection autocorrelation) | the regular glyph pitch of `…online.com` | distinguishes logo rhythm from random texture |

Detection runs **multi-scale** and merges overlapping candidates.

### 1.3 Full-extent recovery (`v28_clean.py`)
OCR usually returns only a **fragment**; masking just that leaves a readable `…online.com`
tail. The integrated engine establishes the **whole** extent:
* `_mark_fully_captured` — treats a read as complete only if it contains the genuine tail
  (`com`/`onlin`) **or** the union box is already ≥ 6.7× text-height wide (the mark aspect).
  A truncated `sunsky-onl` is explicitly *not* full — the bug that caused partial removal.
* `locate_full_box` — template-matches the known geometry (`assets/sunsky_alpha.png`, aspect
  6.67) against an absolute-local-contrast **stroke map** of a band around the anchor; the
  search is sized from the **image**, not the OCR fragment, and constrained to cover the anchor.
* `grow_extent_tint` — extends the box along the visible mark, **stroke-aware** (a pixel
  counts only if lighter than local background *and* on a high-frequency stroke), so it
  follows glyphs but not smooth reflective product features. Growth capped to 4× height/side.

### 1.4 Graded scoring → presence (the precision guard)
Each candidate gets a weighted score
(`0.40·text + 0.22·template + 0.15·center + 0.10·alpha + 0.07·dot_chain + 0.06·edge`, minus
product-text and random-texture penalties) and is graded:

`CONFIRMED_WATERMARK · LIKELY_WATERMARK · UNCERTAIN_WATERMARK · NO_WATERMARK · UNSUITABLE_IMAGE`

The key precision guard is **text grading**. `v27`'s sunsky regex is deliberately
over-permissive (it also matches `sky`, `cn`, `c8n` — noise common in product captions), so
the finder classifies each OCR hit:
* **STRONG** (`sunsky` / `online` / `.com`) → can be `CONFIRMED`
* **MED** (`onl` / `nline` / `.c0m`) → `LIKELY`
* **NOISE** (bare `cn` / `rcn` / `sky`) → downgraded, **never `CONFIRMED`**

This is what stopped the noise-token false positives that earlier damaged clean photos.
Validated on known sets: known-watermarked **6/6 CONFIRMED**, re-cleaned images **6/6
NO_WATERMARK**, restored false positives graded NO/UNCERTAIN — never CONFIRMED.

### 1.5 Output contract
Per candidate `logo_finder` emits `bbox`, `score`, `evidence` (the per-signal breakdown),
`roi_class`, `risk_level`, `recommended_action` (`repair`/`cover`/`reject`), and four repair
masks (below). `CONFIRMED` and `LIKELY` are actionable; `UNCERTAIN` routes to conservative
cover/review, never ignored. Full schema in `docs/OWNER_LOGO_FINDER.md`. If nothing fires,
the image is `NO_WATERMARK` and left untouched.

---

## 2. The mark-cleaning strategies

### 2.1 Full-extent filled mask (`v28_clean.build_mask`)
The padded full-extent box is rasterized as a **filled white rectangle** (a stroke-only
mask leaves the anti-aliased glyph halo readable) and dilated. Horizontal pad is a fraction
of mark width (10 %, min 14 px) so the faint `.com` tail and leading `s` are always covered;
vertical pad 35 %.

### 2.2 Crop-region LaMa inpaint (`lama_inpaint_crop` / `run_bulk._lama_crop_inpaint`)
The mask is ~1 % of image area, so a crop is taken around it (+24 px context), LaMa runs
**only on that crop**, and **only the masked pixels are pasted back** — every unmasked pixel
stays byte-identical to the source. This is the biggest speed lever (full-image 5.5 s → crop
0.1 s, ~55×) and improves fidelity. LaMa is the right inpainter because ≈ 95 % of catalog
marks sit on flat / smooth-gradient backgrounds, its strongest regime, and it does not
hallucinate geometry or text.

### 2.3 Canonical-band re-clean (`reclean.py`, catalog scale)
For bulk re-processing, because the mark is a fixed centre stamp, repair masks a **generous
centred band** (`x∈[0.23,0.77]·W`, `y∈[0.43,0.52]·H`, dilated) and inpaints from the
**pristine backup** — coverage no longer depends on perfect detection, eliminating the
partial-clean failure mode entirely, and re-cleaning is deterministic/repeatable.

### 2.4 ROI-routed masks (`logo_finder`)
The finder classifies the surface (`plain_white`, `near_white`, `metallic_or_reflective`,
`mixed_background_product`, `thin_flex_cable`, `complex_product_detail`, …) and emits four
masks so repair matches the surface:

| Mask | Use |
|---|---|
| `stroke_mask` | tight stroke-level erase on busy product surface (minimal disturbance) |
| `soft_mask` | dilated stroke for anti-alias halo on simple surfaces |
| `line_mask` | full text-line erase |
| `cover_mask` | generous canonical band — fallback when stroke repair is unsafe |

Repair-friendly ROIs → **repair (inpaint)**; risky ROIs (`flex_cable`/`complex_detail`/
`text_area`/`glass`) → **cover**, restoring plausible background without large-area erasure
that could damage cables, labels, screws, ports, or silhouettes.

### 2.5 Restore (false-positive handling)
A flagged image with **no real watermark** (noise-token match) is not inpainted — its
pristine backup is copied back, undoing any needless repair. First-class repair action,
routed by the same text grading used in detection.

### 2.6 GPU (MPS) acceleration
Both conv-heavy nets run on the Apple-Silicon GPU: LaMa via `SimpleLama(device="mps")`;
EasyOCR's CRAFT detector + recognizer moved onto MPS by hand (`make_reader`), ~10× over CPU.
On CPU these nets are **memory-bandwidth bound**, so multi-process CPU workers do not scale —
moving work to the idle GPU is what removes the bottleneck.

### 2.7 Deliberately *not* done
No reverse-alpha / per-pixel α solve, no diffusion inpainting (LaMa chosen for faithfulness
over generation). A filled rectangle on product surface is a known risk; the ROI router +
`cover` action + the `--debug` overlay + the audit gate are the mitigations.

---

## 3. Cleaning-quality review — methods & criteria

Quality is enforced at the **audit gate**, never assumed. Detection is allowed to be wrong;
the audit guarantees the published image is both **watermark-free** and **undamaged**. The
guiding rule: **an image is `cleaned` only if the mark can no longer be detected on the
output.**

### 3.1 Hard verification gate (`v28_clean.verify_clean`)
After inpainting, the mark is **re-detected on the cleaned output** over a region *wider*
than the mask (+0.6× width, +1× height — residue can survive just outside, as in the
`.com`-tail case), using **both** the composite **and** each colour channel. It is
non-circular: a fresh multi-modal detection on the new pixels, not a re-read of cached
state. An empty result means verified clean.

### 3.2 Inline residual audit (`reclean.audit_band`, catalog scale)
After each bulk repair, OCR re-runs on the **canonical band** of the result; a surviving
sunsky fragment ⇒ `status: residual`. Band-only makes it faster and more sensitive than a
full-frame pass.

### 3.3 Recovery loop
If verification returns residue, the cleaner widens the box to cover it, re-grows, builds a
harder mask (16 % horizontal / 50 % vertical pad, larger dilation), re-inpaints, and
re-verifies — **up to 2×** — before any status is assigned.

### 3.4 auto-reject policy
Terminal states are `cleaned` / `restored` / `residual` / `no_mark` — never silent publish.
Any `residual` is written to a **review queue** (`_reclean_review.txt`) and **auto-rejected**
from publication. Nothing visibly watermarked or uncertain ships.

### 3.5 Reversibility = zero-damage guarantee
Every repaired image has a pristine backup (`_wm_backup/`). Because every change is
reversible, aggressive detection is safe: a wrongly-repaired clean image can always be
restored. This is the structural reason the detect stage can chase recall.

### 3.6 Independent final audit (`final_audit.py`)
A random sample of the repaired files on disk is run through the **full 3-pass detector**,
independent of the inline check. A detection counts as a **real** residual only if it carries
a strong sunsky/online-com fragment (bare noise tokens are product text, not watermark), so
it measures true residual rate, not OCR noise. Reports the rate with a 95 % CI and a
PASS/REVIEW verdict.

### 3.7 Per-image record fields
`status`, `method` (`ocr_full`/`template`/`ocr_pad`), `anchor`/`full_box` (mask-placement
audit), `loc_score`, `retry` (recovery passes), `residue_after` (re-detection count — **the
hard gate**, 0 ⇔ `cleaned`), `residual_after`, `bbox_padded`, `backup`, `elapsed_s`. Stages
append, never overwrite, so any failure is attributable. `reconcile_manifest.py` merges
repair results into one authoritative `_wm_process.jsonl` (pre-reclean manifest backed up
first; rewrite atomic).

### 3.8 Run-level acceptance criteria
```
residue_after / residual_after == 0   for every record   →  no re-detectable mark
status ∈ {cleaned, restored, no_mark}                     →  no published residual
false positives ≤ baseline                                →  re-cleaned output reads NO_WATERMARK
independent final-audit real-residual rate ≈ 0 (95% CI)   →  precision confirmed
retry rate                                                →  tracked, not gated (high ⇒ §1 regression)
before/after PDF + --debug mask overlay                   →  final human eyeball
```

### 3.9 Honest limits
The verification gate is OCR-readability based: it reliably catches *readable* residue, but
a sub-readable faint ghost can pass (rare with the full-extent mask). A faint mark on a busy,
low-contrast background may defeat detection (`no_mark`); the per-channel booster is the
mitigation. On dark textured flex cables LaMa can leave a faint cosmetic smudge (no readable
text). These are why the ROI router prefers `cover` on risky surfaces and the audit is the
final arbiter.

---

## 4. Performance & scaling

**Measured (Apple M4, single process, GPU):** ~4.7 s/image (min ~1.1 s, max ~69 s on the
hardest channel-search cases), dominated by EasyOCR; LaMa is ~0.05 s after cropping. Catalog
re-clean (canonical band, no per-image detect) runs faster (~1 img/s sustained).

| Lever | Effect |
|---|---|
| Crop-region inpaint | LaMa 5.5 s → 0.1 s (~55×) |
| GPU/MPS for OCR + LaMa | per-image 38 s → ~5 s (~8×) |
| Horizontal scale-out | linear — shard across N GPUs |
| OCR batching | 2–4× (feed K images per CRAFT/CRNN pass) |

**10 000 images:** one M4 as-is ~13 h; + batching/tuning ~3–4 h; one A100 + batching ~1–2 h;
10–20 GPU scale-out ~15–30 min. CPU multi-processing is **not** a lever (bandwidth-bound).

---

## Usage

```bash
# DETECT — graded find on one image (prints the contract; --masks writes mask PNGs)
python3 logo_finder.py --input in.jpg --json out.json --masks maskdir --device mps
python3 logo_finder.py --batch paths.txt --out logo_finds.jsonl --device mps

# REPAIR (integrated per-image engine: detect → full-extent → inpaint → verify)
python3 v28_clean.py --input in.jpg --output out.jpg [--debug dbg]
python3 v28_clean.py --batch items.json --out-dir cleaned/ --log cleaned/_log.json

# REPAIR (catalog scale: routed re-clean / restore from backups, resumable)
python3 reclean.py --run --csv reclean_routing.csv --action reclean --device mps --apply
python3 reclean.py --run --csv reclean_routing.csv --action restore --apply

# AUDIT — independent residual check over the repaired output
python3 final_audit.py --n 200 --device mps
```

Recommended on Apple Silicon: `export PYTORCH_ENABLE_MPS_FALLBACK=1 OCR_DEVICE=mps LAMA_DEVICE=mps`.
**Dependencies:** Python 3, OpenCV, NumPy, Pillow, `easyocr`, `simple_lama_inpainting`,
PyTorch. LaMa weights land at `~/.cache/torch/hub/checkpoints/big-lama.pt` on first run.

## Operational notes
* **MPS stability:** long runs use a fresh process per N images (chunking) to bound Metal
  memory; the manifest is the resume checkpoint (re-launch skips done work).
* **One GPU job at a time:** two concurrent OCR/LaMa jobs split the GPU and reintroduce Metal
  instability — run detect/repair/audit serially under the single owner.
* **State on disk:** progress, heartbeat, run log, and per-image records are persisted so any
  long job is resumable and never depends on chat.

## Repository layout
```
logo_finder.py            DETECT — Owner Logo Finder (ensemble + graded presence + masks)
v27_clean.py              DETECT primitives — multi-pass OCR + fuzzy sunsky regex
v28_clean.py              REPAIR — integrated GPU cleaner (full-extent mask + hard verify gate)
reclean.py                REPAIR — canonical-band re-clean / restore (catalog scale, resumable)
run_bulk.py               REPAIR engine — LaMa crop-inpaint, reader, write+backup
final_audit.py            AUDIT — independent residual re-detection
reconcile_manifest.py     manifest merge into one authoritative source
assets/watermark-canonical.png   canonical matte (shape signal, logo_finder)
assets/sunsky_alpha.png          solved alpha template aspect 6.67 (v28 full-extent)
docs/OWNER_LOGO_FINDER.md        the DETECT contract
```
