# Audit Logo-Finder — comprehensive design (position id · cleaning · quality review)

> The audit stage's design doc — the independent publish gate. Parallels [`OWNER_LOGO_FINDER.md`](OWNER_LOGO_FINDER.md) (the detect contract). Covers watermark position identification (§1), cleaning strategies (§2), and the three-gate cleaning-quality review with methods + criteria (§3).

A standalone system that finds and erases the **`sunsky-online.com`** semi-transparent
text watermark on B2B product photos — and, as a separate, independent stage, **audits
the result so no watermarked, damaged, or badly-patched image is ever published.**

The guiding rule across the whole pipeline is **"aggressive find / conservative
publish":** detect the mark with every signal available, but only release an image when
it can be *proven* clean and undamaged. When in doubt, **reject rather than publish.**

```
            ┌────────────── DETECT ──────────────┐   ┌──── REPAIR ────┐   ┌─────────── AUDIT ───────────┐
 image ───► OCR ×passes → per-channel → template  ───► full-extent mask ───► (A) scan false-neg sweep      ───► publish
            → canonical matte → full-extent box        crop + LaMa (GPU)      (B) logo-finder pair audit         / reject
                  §1  position id                          §2  cleaning            §3  quality review
```

Three concerns, three stages, kept deliberately separate (a detector regression must
never silently weaken the audit, and vice-versa):

| Stage | Question | Scripts |
|------|----------|---------|
| **Detect** | *Where is the mark?* | `v27_clean.py`, `v28_clean.py` (`detect_smart`), `scan_audit.py` (`detect`) |
| **Repair** | *Erase it without touching the product* | `v28_clean.py` (`build_mask`, `lama_inpaint_crop`) |
| **Audit** | *Is the output actually safe to publish?* | `scan_audit.py` (false-negative sweep), `audit.py` (logo-finder) |

---

## File map

| File | Role |
|------|------|
| `v28_clean.py` | Entry point: detection, full-extent localization, crop+MPS LaMa inpaint, owner-side hard verify gate, recovery, CLI. |
| `v27_clean.py` | Reused primitives: EasyOCR passes (`_ocr_pass`), fuzzy regex (`SUNSKY_TOKENS`), `union_bbox`, `pad_bbox`, base LaMa wrapper. |
| `audit.py` | **Audit logo-finder** — independent per-image `(original, final)` gate. Re-detects residual marks, finds repair artifacts / product damage / protected-text loss, emits the structured publish/reject contract (§3.3). |
| `scan_audit.py` | **Scan false-negative sweep** — re-checks every image the owner's scan called *clean*, with an ensemble detector complementary to the scan's, to catch missed marks. Resumable, file-state tracked. |
| `bench_combine.py` | Benchmarks each detection method (OCR / per-channel / canonical) vs the union on a labelled set — the evidence behind the detector design. |
| `assets/sunsky_alpha.png` (+ `_meta.json`) | Solved 36×240 alpha template of the mark (aspect 6.67), used by `locate_full_box`. |

**Dependencies:** Python 3, OpenCV, NumPy, Pillow, `easyocr`, `simple_lama_inpainting`,
PyTorch. On Apple Silicon set `OCR_DEVICE=mps`, `LAMA_DEVICE=mps`,
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Quick start

```bash
# clean one / many
python3 v28_clean.py --input in.jpg --output out.jpg [--debug dbg/]
python3 v28_clean.py --batch items.json --out-dir cleaned/ --log cleaned/_log.json

# audit cleaned (original,final) pairs — drops straight onto v28's own log
python3 audit.py --v28-log cleaned/_log.json --out-dir audit_out/ --html
python3 audit.py --manifest pairs.json --out-dir audit_out/        # owner→audit handoff
python3 audit.py --selftest                                        # dependency-light proof

# sweep the scan's "clean" verdict for missed marks (resumable)
python3 scan_audit.py --set A --scan _wm_scan.jsonl --out-dir auditA/
```

---

# 1. How the watermark position is identified  *(design §2.1)*

The mark is the fixed string `sunsky-online.com`, but its **transparency**, the **surface
texture** under it, and its **placement** defeat any single detector. Position is
established by layering complementary signals, each covering the previous one's blind
spot. No one detector is trusted alone.

### 1.0 The mark is a fixed centre stamp

Measured across 700 real detections, the watermark sits **dead-centre**: `cx/W ≈ 0.50`,
`cy/H ∈ [0.469, 0.487]` for 90 % of marks (96.5 % fall in a generous centre band), width
≈ 0.34·W, height ≈ 0.04–0.07·H, aspect ≈ 6.7:1. Detectors exploit this — searching the
**canonical centre band** is faster *and* more precise (product SKU/label text at the top
and bottom edges cannot false-positive).

### 1.1 Multi-pass EasyOCR (CRAFT) — `detect_smart`

Up to three OCR passes, each preceded by a CLAHE contrast boost + cubic upscale, stopping
early at the first pass that captures the whole mark:

| Pass | Scale | CRAFT params | Purpose |
|-----:|------:|-------------|---------|
| 1 | 2.0× | defaults | majority of faint marks on smooth backgrounds |
| 2 | 3.0× | `text_threshold=0.4, low_text=0.2, link_threshold=0.3` | very faint marks on near-white |
| 3 | 2.0× | `text_threshold=0.3, low_text=0.15, link_threshold=0.2` | busy pages where 1–2 found nothing |

### 1.2 Fuzzy regex for OCR-garbled text

OCR rarely reads the mark cleanly (`sunsky-Snline:=`, `onlme com`, `Snline c8n`, …). The
matcher `SUNSKY_TOKENS` is a union of permissive token patterns; any one fragment
qualifies a box. Matched boxes are unioned into one anchor.

### 1.3 Per-channel recall booster — `_ocr_channels`  *(catches the "complete miss")*

A light/white mark on a **saturated colour** (white-on-blue on a translucent phone) is
nearly invisible in the composite and missed by every pass above. `detect_smart` falls
back to OCR-ing each **B, G, R channel separately** — the mark's contrast collapses in
the composite but survives in the complementary channel. This fires only when the
composite passes find nothing, so it costs nothing on the common case. **In production
this single addition recovered ~2 % of images the scan had called clean** (see §3.1).

### 1.4 Full-extent recovery — `locate_full_box`, `grow_extent_tint`

OCR usually returns only a **fragment**; masking just that leaves a readable tail (the
`…online.com` ghost). v28 establishes the **full** extent:

- `_mark_fully_captured` decides whether OCR read the whole string (genuine `com`/`onlin`
  tail, or a box already ≥ 6.7× text-height wide). A truncated `sunsky-onl` is **not**
  treated as full.
- `locate_full_box` template-matches the known mark geometry (`assets/sunsky_alpha.png`,
  aspect 6.67) against an absolute-local-contrast **stroke map**, sized from the image,
  constrained to cover the OCR anchor. Accepted at correlation ≥ 0.24.
- `grow_extent_tint` extends the box left/right along the actually-visible mark. It is
  **stroke-aware** (a pixel counts only if it is both lighter than the local background
  *and* on a high-frequency stroke) so it follows glyphs but not smooth reflective product
  features, and growth is **capped** to 4× box-height per side.

### 1.5 Canonical matte — complementary shape signal  *(merged from b2bweb)*

A second, OCR-independent appearance detector: a background-subtracted matched filter
against the median watermark **matte** (`watermark-canonical.png`). It correlates with the
glyph **shape** even when the mark is too faint to *read*. In `scan_audit.py` it is folded
in as a **corroboration** signal (its NCC is recorded on every hit to raise confidence)
but does **not** fire standalone by default — on this catalogue a standalone matte rescue
is ~12 % false-positive on centred grid patterns (screw sheets, flex traces, product
text) with little real rescue. Standalone is opt-in (`--canon-only`, a cheap CPU pass).
This calibration is backed by `bench_combine.py` on a labelled set.

### 1.6 Detector independence (the audit principle)

The scan and the audit deliberately use **different** detector mixes so the audit does not
inherit the scan's blind spots. The owner's scan = composite OCR only (`v27`); the
`scan_audit` re-check adds the **per-channel** booster (v27's exact blind spot) — same
method would reproduce the same misses.

---

# 2. The mark-cleaning strategies  *(design §2.2)*

### 2.1 Full-extent filled mask — `build_mask`

The padded full-extent box is rasterized as a **filled white rectangle** (a stroke-only
mask leaves the anti-aliased glyph halo readable), then dilated. Horizontal pad is a
fraction of mark width (10 %, min 14 px) so the faint `.com` tail and leading `s` are
always inside; vertical pad is 35 % of height.

### 2.2 Crop-region LaMa inpaint — `lama_inpaint_crop`

The mask is typically ~1 % of the frame, so inpainting the whole image is ~98 % wasted.
`lama_inpaint_crop` crops a small region around the mask (+24 px context), runs LaMa on
**only that crop**, and **pastes back only the masked pixels** — every unmasked pixel
stays byte-identical to the source. Biggest speed lever (full-image 5.5 s → crop 0.1 s,
~55×) and better fidelity (no global recomposition). LaMa is the right inpainter because
≈ 95 % of catalogue marks sit on flat / smooth-gradient backgrounds — its strongest regime
— and it does not hallucinate geometry or text.

### 2.3 GPU (MPS) acceleration — `make_reader`, `get_fast_lama`

Both conv-heavy nets run on the Apple-Silicon GPU. EasyOCR's `gpu=` flag only knows CUDA,
so `make_reader` moves the CRAFT detector + recognizer onto MPS by hand (~10× vs CPU).
On CPU these nets are memory-bandwidth bound, so multi-process CPU workers do *not* scale;
moving work to the idle GPU is what removes the bottleneck.

### 2.4 Recovery loop

After inpainting, if the owner-side verify gate (§3.1) finds residue, the cleaner widens
the box to cover it, re-grows, builds a harder mask (16 % horizontal pad, 50 % vertical,
larger dilation), re-inpaints and re-verifies — up to 2×.

### 2.5 What the cleaner deliberately does *not* do

- No reverse-alpha / per-pixel α solve, no covers, clones, or synthetic fills — inpaint
  only.
- No diffusion inpainting — LaMa is chosen for faithfulness over generation.
- No product-mask gate in the cleaner itself — **that protection lives in the audit
  (§3)**, which is independent and can reject a repair the cleaner was happy with.

---

# 3. Cleaning-quality review: methods & criteria  *(design §2.3)*

Review runs **on the bytes that would be written**, and the resulting status is recorded
verbatim. There are **three independent gates**, each stricter than the last, so a failure
of any one blocks publication:

```
 owner verify gate ──► AUDIT (A) scan false-neg sweep ──► AUDIT (B) logo-finder pair audit ──► publish
   (re-detect on        (did the scan MISS a mark on        (is THIS final image clean,
    own output)          an image it called clean?)          undamaged, un-patched?)
```

### 3.1 Gate 1 — owner-side hard verify — `verify_clean`

After inpainting, `verify_clean` re-detects the watermark on the **cleaned** image over a
region **wider than the mask** (+0.6× width, +1× height — residue can survive just outside,
as in the `.com`-tail case), using **both** the composite image **and** each colour
channel. It returns surviving hits in image coordinates; empty ⇒ verified clean. This is
non-circular: a **fresh** multi-modal detection on the new pixels.

**An image is `cleaned` only if this returns empty (`residue_after == 0`).** Otherwise it
is reported `residual` — written but flagged, **never presented as clean**.

### 3.2 Gate 2 — scan false-negative sweep — `scan_audit.py`

The owner's scan classified each image *flagged* (clean it) or *clean* (leave it). Gate 1
only protects the flagged images. **Gate 2 independently re-checks the images the scan
called clean** — because a missed mark there ships with the watermark on.

- **Independent detector.** Complementary to the scan (adds the per-channel booster,
  v27's blind spot), focused on the canonical centre band.
- **Aggressive find.** Low OCR confidence is *kept* — a hard-to-read mark is exactly the
  kind the scan misses; the saved crop is the verdict, not the score.
- **Resumable, file-state tracked** (per long-job rule): `progress.json`, `heartbeat.txt`,
  `run.log`, `misses.jsonl` (+ a cropped image per miss), `done.txt` ledger, `summary.json`.
- **Provenance + tiers.** Each miss records which methods fired (`ocr` / `perchan` /
  `canon`); legible-text hits are `confirmed`, shape-only (canon) hits are `review`.

> On a real 24 k-image catalogue this gate found a **systematic ~2 % miss rate** — `sunsky`
> marks on saturated colour backgrounds (adhesive/sticker multipacks) that the composite-OCR
> scan could not read but the per-channel booster recovered.

### 3.3 Gate 3 — audit logo-finder — `audit.py`

The final, per-image publish gate for `(original, final)` pairs. **It is stricter than the
owner and does not trust the owner's result** — it re-verifies the final image
independently and asks: *would a customer still notice a watermark, patch, blur, gray
block, broken cable, damaged text, or unnatural repair?* If yes → reject.

It first re-derives the **edit footprint** from the original↔final pixel diff (the ground
truth of what changed, independent of any declared mask), classifies the surface **ROI**,
then runs:

| Detector | What it catches |
|---|---|
| **Residual watermark** | fresh multi-modal OCR + template correlation on the final |
| **Dot-chain** (`dot_chain_score`) | periodic horizontal rhythm of faint repeated marks |
| **Ghost text** (`ghost_text_score`) | faint transparent-gray text silhouette (sub-OCR) |
| **Repair artifact** | rectangular seam, flat block, colour mismatch, texture mismatch |
| **Product damage** | structure/geometry destroyed — *ring-gated*: only flags a smooth hole **inside textured product**, so erasing the mark off a flat tray is not mistaken for damage |
| **Protected text** | product SKU/label text beside the mark that vanished |
| **False-positive** | edit off-target / unnecessary / larger than a mark footprint |

**Risk-based policy.** ROI drives strictness. Low-risk (white tray, near-white) tolerates
minor uncertainty; **high-risk** (thin flex cable, text/label, metallic, glass/gradient,
dark surface, complex detail) escalates any warning to a reject. Product safety outranks
watermark removal; visual quality outranks pass rate.

**Output contract** (`audit_results.jsonl`, one per image):

```json
{
  "audit_decision": "PASS | PASS_WITH_MINOR_BACKGROUND_ARTIFACT | REJECT_RESIDUAL_WATERMARK |
                     REJECT_VISIBLE_PATCH | REJECT_PRODUCT_DAMAGE | REJECT_PROTECTED_TEXT_DAMAGE |
                     REJECT_UNNATURAL_IMAGE | REJECT_UNCERTAIN",
  "publish_allowed": true,
  "scores": { "residual_logo_score": 0.0, "dot_chain_score": 0.0, "ghost_text_score": 0.0,
              "patch_visibility_score": 0.0, "product_damage_score": 0.0,
              "texture_mismatch_score": 0.0, "color_delta_score": 0.0,
              "edge_damage_score": 0.0, "protected_text_damage_score": 0.0 },
  "evidence": [ { "type": "residual_watermark | visible_patch | product_damage |
                          protected_text_damage | texture_mismatch",
                  "bbox": [x, y, w, h], "severity": "low|medium|high", "reason": "..." } ],
  "recommended_next_action": "publish | retry_repair | try_cover | auto_reject"
}
```

`recommended_next_action` routes the reject: residual → `retry_repair`, visible patch →
`try_cover`, product/text damage or uncertainty → `auto_reject` (no manual review queue).

**Mandatory rejects** (no exceptions): missing/corrupt/invalid inputs → `REJECT_UNCERTAIN`;
any visible residual watermark; any visible repair patch on product; any product-text or
structure damage; any off-target/unnecessary large edit.

### 3.4 Run-level acceptance criteria

```
Gate 1   residue_after == 0 for every cleaned record           no re-detectable mark
Gate 2   scan_audit misses == 0 (after re-clean of any found)  scan left nothing behind
Gate 3   publish_allowed == true for every published image     final gate satisfied
         every reject carries evidence + a next action         actionable, auto-routed
mask overlay (--debug) + before/after compare.html             human spot-check
```

### 3.5 Validation

- `audit.py --selftest` — dependency-light synthetic proof that every reject path fires
  (visible patch, product damage, missing/corrupt input) and clean images pass.
- Real `(original, final)` pairs from the cleaner: clean outputs are publish-allowed
  (0 false rejects); adversarially-corrupted finals (watermark left in, half-clean `.com`
  tail, gray cover, off-target slab) are rejected with the correct class and next action.
- `bench_combine.py` quantifies each detector's recall/precision on a labelled set.

### 3.6 Scope & honest limits

- The cleaner is qualified on marks over flat / smooth-gradient backgrounds (≈ 95 % of the
  catalogue). On dark textured flex cables LaMa can leave a faint cosmetic smudge — the
  audit's ring-gated product-damage detector is the catch for that.
- The residual gate is OCR/shape based: it reliably catches *readable* residue; the
  dot-chain and ghost-text detectors extend sensitivity to sub-readable ghosts, kept
  conservative (speckle-removed, concentration-scored) so flat noise and product texture
  do not false-reject.
- The canonical matte is a corroboration signal here, not a standalone detector (§1.5).

---

## 4. Performance & scaling

**Measured (Apple M4, single process, GPU):** clean ~4.7 s/image (LaMa negligible after
cropping); audit logo-finder ~2–3 s/image; scan sweep ~3 s/image (saturation-gated
per-channel, canonical band). All stages are embarrassingly parallel — shard across
GPUs/machines for linear scale-out. CPU multi-processing is **not** a lever (the nets are
memory-bandwidth bound on CPU).

## 5. Owner → audit handoff

The convenient, decoupled contract is a **manifest + folder**, not a synchronous wait. The
owner writes one JSONL line per finished image — `{original, final, bbox, mask, method,
status}` — and drops the finals in a folder; the audit consumes it (`audit.py --manifest`)
and emits `audit_results.jsonl` + a reject queue. The same mechanism serves both a small
**stratified acceptance sample** (validate the pipeline's accuracy before trusting it) and
the **full publish gate** (every image) — sample to validate, full to gate.

### Design principle

> Localise the **whole** mark with every complementary signal; inpaint **only** its
> footprint on the GPU; then **prove** — independently, stricter than the cleaner — that
> the output is free of the mark *and* of repair damage before it is allowed to publish.
> If it is not clearly safe, reject it.
