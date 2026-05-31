# Mark Remover

Automated watermark detection and removal pipeline for sunsky-online.com product images. V8 replaces the fixed candidate loop with a **100-tool progressive repair strategy bank** — each watermark ROI is classified, tools are selected from an ordered strategy bank, candidates run through a local QA gate, and the first passing candidate wins. Adaptive cover is the absolute last resort. V13 adds a final visual-fidelity gate that runs on every published image — repaired or covered — so no visible rectangle, wedge, pale band, dark-surface blob, broken product contour or dot-chain residue is ever shipped as a success. V14 keeps that gate untouched and makes the *candidates* better (near-miss rescue + segmented micro-cover beam), lifting true `clean_repaired` output 25 → 29 / 50. **V15 (current)** is a cover-quality patch: it polishes covers to remove bright/hard patches, widens the watermark mask to the full `sunsky-online.com` text line (no faint trailing ghosts), adds a ghost-aware full-region residual verdict, and inpaints dark-cable strokes from neighbouring pixels — all while preserving the V13 honesty guarantee (zero false `clean_repaired`).

## V15 — Cover Quality (polish, full-text mask, ghost-aware residual, dark-stroke inpaint)

V15 answers a simple visual complaint: *the repairs look clean, but some covered
images still show the mark.* The 29 `clean_repaired` were genuinely clean — the
problem was entirely on the cover side, in two forms: a **bright/hard patch**
(e.g. a white box over a dark flex cable) and **faint residual text** (a trailing
`sunsky-online.com` ghost the mask-relative residual gate missed). V15 fixes the
generators, never the gate.

- **Cover polish (`v14_patch.polish_cover`).** Every micro-cover candidate is
  passed through a bounded Lab match toward the local ring + a wider feathered
  seam *before* the beam scores it, and the beam's boundary-jump term is now
  *uncapped* — so a catastrophic patch can never be selected over a soft one.
  Worst-case boundary jumps roughly halved (touch-test flex **222 → 102**,
  ipad-mini panel **242 → 110**, front-camera flex **207 → 93**).
- **Full-text mask widening (`v15_patch.widen_text_mask`).** The detector often
  clips trailing glyphs; V15 template-locates the whole watermark text row in a
  horizontally-expanded band and grows the stroke mask to cover the clipped
  glyphs — with three guards (skip busy product bands, drop product-overlapping
  components, refuse runaway expansion) so it never eats product detail.
- **Ghost-aware residual (`v15_patch.full_region_residual_ok`).** Hiding is now
  verified across the *full* text region, not only inside the tight mask, using
  the canonical-watermark **template correlation** (deliberately *not* a generic
  text-component count, which false-positives on flex-cable / connector texture).
  A faint surviving ghost is caught and the cover is rejected / re-tried.
- **Dark-stroke inpaint (`v15_patch.dark_stroke_cover`).** On a dark / flex-cable
  surface the widened stroke mask is inpainted from neighbouring (dark) pixels
  with Telea — never a light/median fill that would leave a bright block.

### V15 results (50-image benchmark, seed 7)

| Metric | V13 | V14 | V15 | |
|--------|-----|-----|-----|---|
| clean_repaired | 25 | 29 | 28–29¹ | ✅ |
| clean_covered | 25 | 21 | 21–22¹ | ✅ |
| false `clean_repaired` | 0 | 0 | **0** | ✅ |
| product / silhouette damage on covers | 0 | 0 | **0** | ✅ |
| readable residue on covers | — | 0 | **0** | ✅ |
| worst cover boundary jump | — | 222 | **102** | ↓ |
| manual_review / failed_io | 0 | 0 | 0 | ✅ |

¹ ±1 is run-to-run noise from the non-seeded fill noise on borderline near-miss
candidates.

**What V15 fixed:** the bright-box covers are gone (soft blended wedges now), and
faint trailing-text ghosts are removed (e.g. the volume-button flex no longer
shows a readable `sunsky-online.com`). **What remains (honest):** six covers
where the watermark straddles a high-contrast cable / background **edge** still
report `boundary_jump > 50` — removing the mark there unavoidably disturbs the
edge. Those need **V16 structure-aware reconstruction** (edge/structure-tensor
or stroke-level neural inpaint); V15 ships the cover-quality generators + honest
labelling of that remaining gap. `test_v15_patch.py` locks the three V15
primitives; the full V10–V15 suite stays green (69 cases).

## V14 — Better Candidates (near-miss rescue + segmented micro-cover beam, V13 gate unchanged)

## V14 — Better Candidates (near-miss rescue + segmented micro-cover beam, V13 gate unchanged)

V14 is a *short patch* with one principle: **keep V13 strict, make the candidates
better.** The V13 final visual gate (`final_visual_publish_gate_v13`) is **not
weakened in any way** — the goal is not to label more dirty output as clean, it
is to generate candidates that *truly pass* the existing gate. All V14 logic
lives in the additive, self-contained `v14_patch.py` and is wired in at the
single post-candidate decision point in `process_image`; V14 only ever asks the
unchanged V13 gate to bless a result.

- **Adaptive soft-metric context (Section 3.2).** `soft_qa_context` builds a
  multi-signal surface profile (plain/near white, saturated flat product, thin
  flex cable, complex detail, dark, glass/gradient, metallic) and context-aware
  *soft* limits. These only decide whether a failed repair is a rescuable
  near-miss — the **hard** V13 metrics (residual, dot-chain, protected text,
  silhouette, product damage, geometric patches) are never relaxed.
- **Near-miss rescue before cover (Section 3.3).** A `clean_repaired` candidate
  that fails V13 only on *soft* visual-fidelity reasons (a faint patch / band) is
  classified `soft_fail` and run through `near_miss_rescue`: component-level
  residual cleanup → gamma/Lab colour match of just the changed pixels toward the
  surrounding ring → 1px seam smoothing. The rescued image is re-checked against
  the V13 **repaired** gate and kept as `clean_repaired` only if it now fully
  passes. Any *hard* reason ⇒ no rescue, straight to cover. The repair loop also
  surfaces its best failed candidate so a candidate that fell all the way to
  cover can still be rescued.
- **Segmented micro-cover beam search (Section 3.4).** `segmented_micro_cover`
  replaces the full-bbox `final_adaptive_cover`. It generates several
  fragment-based covers (segmented background/product fill, stroke-only,
  stroke-band r2/r3, and — only when allowed — a full-bbox fill), scores the
  integrated result (dot-chain + visible-patch + *uncapped* boundary jump +
  silhouette + edge loss + colour + texture + rectangularity + protected-text),
  then publishes the first candidate that passes the V13 **covered** gate (with a
  fresh, per-candidate watermark-hiding verdict). A **full-bbox cover is banned**
  whenever `product_overlap > 0.10`, `edge_density > 0.04`, long lines, or a
  flex-cable / metallic / glass / protected-text zone is detected.
- **Cover-side honesty counters (Section 7).** `v14_patch.COVER_HONESTY_COUNTERS`
  +`v13_report.py` now tally `clean_covered_with_{visible_patch, product_damage,
  silhouette_damage, boundary_jump_gt_50, full_bbox_on_product,
  protected_text_loss}`. These are surfaced honestly (the CI gate stays red while
  any are non-zero) — never hidden.
- **Version → `V14_BETTER_CANDIDATES`.** The V13 final visual gate version stays
  `v13` (it is unchanged); the pipeline/report version is bumped to V14.

### V14 results (50-image benchmark, seed 2026)

| Metric | V13 baseline | V14 | Target | |
|--------|--------------|-----|--------|---|
| clean_repaired | 25 | **29** | ≥ 29 | ✅ |
| clean_covered | 25 | **21** | ≤ 21 | ✅ |
| near-miss rescued | 0 | **5** | ≥ 5 | ✅ |
| segmented micro-cover used | 0 | **10** | ≥ 8 | ✅ |
| final_adaptive_cover used | — | 8 | ≤ 12 | ✅ |
| full-bbox cover on product overlap | — | **0** | 0 | ✅ |
| unique final methods | 8 | **14** | ≥ 12 | ✅ |
| `clean_repaired_with_*` (dirty repairs) | 0 | **0** | 0 | ✅ |
| product / silhouette damage on covers | 0 | **0** | 0 | ✅ |
| manual_review / failed_io | 0 / 0 | 0 / 0 | 0 | ✅ |

**The V13 honesty guarantee is fully preserved: every `clean_repaired_with_*`
counter is 0** — no dirty repair was ever upgraded to hit the target. The +4
genuine repairs came from real candidate improvement (5 near-miss rescues, minus
1 still-honest demotion), not from relaxing the gate.

**Cover side (honest, still the hard problem):** the report still flags
`clean_covered_with_visible_patch = 11`, `…_boundary_jump_gt_50 = 6`,
`…_protected_text_loss = 8`, `…_dot_chain = 2`. These are genuine and concentrate
on the hardest mixed surfaces (metallic / flex-cable / glass) where *hiding* the
watermark and passing *every* soft visual gate are in direct tension. V14 still
**hides the watermark and refuses a full-bbox-on-product fill** on every one of
them — it does not fake a pass. Driving these to 0 needs the V16 high-frequency
texture reconstruction the plan defers; V14 ships the better generators + honest
labelling of the remaining gap. `test_v14_regression.py` locks the unconditional
V14 guarantees on the eight Section 8 weak cases and the v14_patch primitives;
the full V10–V14 suite stays green (62 cases).

## V13 — Final Visual Fidelity (publish gate on covers too, shape/silhouette detectors)

V13 keeps the V12 honesty direction but fixes its two remaining gaps: too many
genuine repairs were demoted to `clean_covered`, and some published covers still
carried a visible rectangle, wedge, pale band, bright patch on a dark surface,
or a broken product contour. **V13 does not relax any V12 gate.** It adds a
single final visual gate that runs on the *chosen output* — repaired **or**
covered — plus new shape / silhouette / dot-chain detectors, so a visually-bad
result can never be published as a success.

- **`final_visual_publish_gate_v13()` — applies to BOTH repaired and covered
  (P0).** After the beam picks a candidate, `process_image` runs one composite
  verdict requiring *all* of: `metrics_valid`, `residual_pass`, dot-chain v2,
  rectangular-band, visible-patch-shape, product-damage, silhouette, and
  protected-text. A `clean_repaired` that fails any gate is **honestly demoted**
  to `clean_covered` (re-scored under the looser covered thresholds); a cover is
  never silently upgraded. Returns a `FinalVisualVerdict` with every sub-gate
  boolean. Lives in the new self-contained `v13_gates.py`.
- **Visible-patch-shape detector (Section 8).** `detect_visible_patch_shape_v13`
  flags rectangles, wedges, straight-edged polygons, hard luma boundaries and
  over-smoothed low-texture islands in the changed region — artifacts the band
  detector alone misses. A change is only "visible" when it carries a real
  boundary (edge or luma step), so a seamless inpaint still passes.
- **Product-silhouette / contour gate (Section 10).**
  `detect_product_silhouette_damage_v13` measures contour IoU, edge retention,
  contour break and new bright/dark blobs on the product surface. Crucially it
  **excludes the watermark footprint** (its strokes are *supposed* to vanish)
  and only fires on real product overlap — so it catches a smeared cable or a
  bright blob on a black frame without demoting a clean removal.
- **Dot-chain detector v2 (Section 7).** `detect_residual_dot_chain_v2` searches
  a horizontally-expanded window (trailing `.com` glyphs) and counts **discrete**
  fragments from the solid luma-deviation mask (the high-pass response bridges
  the gaps and would merge a readable chain into one blob), scoring component
  count, horizontal span, alignment and luma delta.
- **Product-overlap routing v13 (Section 9).** `estimate_product_overlap_v13`
  judges overlap from bbox-interior signals (non-white / dark / edge density /
  long lines), not only the surrounding ring, and emits a routing override class
  (`dark_product_surface`, `thin_flex_cable`, `complex_product_detail`,
  `text_or_label_area`).
- **Protected product text (Section 11).** `detect_protected_product_text_v13`
  refuses to publish an output that flattened away real high-contrast printed
  product text near the watermark.
- **Honesty counters that must all be 0 (Section 14).** `v13_report.py` scans the
  per-image `qa.json` records and tallies
  `clean_{repaired,covered}_with_{dot_chain,visible_patch,product_damage,silhouette_damage}`
  and `final_publish_failures`. It exits non-zero if any are non-zero, doubling
  as a CI gate. Every report carries `V13_FINAL_VISUAL_FIDELITY` with
  `qa_schema_version = strategy_schema_version = final_visual_gate_version = v13`.

### V13 results (50-image benchmark, seed 2026)

| Metric | Result | Target |
|--------|--------|--------|
| total / failed_io | 50 / 0 | 0 |
| clean_repaired / clean_covered | 24 / 26 | honesty first |
| manual_review | 0 | 0 |
| zero-metric passes | 0 | 0 |
| `clean_repaired_with_dot_chain` | 0 | 0 |
| `clean_repaired_with_visible_patch` | 0 | 0 |
| `clean_repaired_with_product_damage` | 0 | 0 |
| `clean_repaired_with_silhouette_damage` | 0 | 0 |
| `v13_demoted_from_repaired` (caught by the new gate) | 4 | — |
| unique final methods | 9 | ≥ 8 |

The headline V13 guarantee holds: **every `clean_repaired_with_*` counter is 0** —
no output is published as a genuine repair while carrying readable residue, a
visible patch/band, product damage, or a broken silhouette. The new final
visual gate honestly demoted 4 outputs that V12 would have shipped as
`clean_repaired`.

**Cover side (honest, not yet perfect):** the report still flags some
`clean_covered` outputs — `clean_covered_with_dot_chain = 2`,
`clean_covered_with_visible_patch = 9` (`final_publish_failures = 16`). These
are genuine: `final_adaptive_cover` still leaves a visibly patchy region on a
few hard metallic / low-texture surfaces. The detectors are working (they are
not hiding these); driving these counters to 0 is the next step and needs the
**cover-quality remediation** in the plan's §3–6 (segmented micro-cover beam,
dark/flex tool activation, adaptive feathering, HF texture matching, residual
micro-cleanup inside the candidate loop). V13 ships the *detection + honest
labelling* of that gap; it does not yet rebuild the cover generator.

`test_v13_final_visual_gate.py`, `test_v13_detectors.py` and
`test_v13_report_honesty.py` lock the new gates; the V10/V11/V12 regression
suites stay green (45 cases total).

## V12 — Visual Truthfulness (unified publish gate, band detector, residual cleanup)

V12 does **not** add a new tool bank. It makes visual acceptance match human
inspection: the V11 compare PDF still passed dot-chain residue, pale
rectangular bands, and bright patches on dark product surfaces as
`clean_repaired`, and the report still said `V10_QUALITY_PATCH`. V12 closes
those holes and makes the status describe **publish quality, not method
family**.

- **`final_publish_gate()` — the single source of truth (Phase A).** No
  candidate becomes `clean_repaired` from its method family alone. After every
  candidate, one unified verdict requires *all* of: `metrics_valid`,
  `residual_pass` (incl. dot-chain), `product_gate_pass`, **and** the band
  gate. Anything short is, at best, an honest `clean_covered`. Returns
  `{publish_ok, status, reject_reasons, residual_gate, product_gate, band_gate}`.
- **Stronger dot-chain / broken-glyph detector (Phase B).** The ceiling drops
  `0.35 → 0.28`, plus a count-based rule: reject when `component_count ≥ 4`
  **and** `horizontal_span > 0.22` **and** `component_area_ratio > 0.035` — a
  row of aligned fragments a human still reads as a line, even when no whole
  letter survives.
- **Residual micro-cleanup before cover escalation (Phase C).** When the only
  thing blocking a repair is leftover dots on a low-overlap surface,
  `cleanup_residual_components_with_ring_fill` paints **just those component
  pixels** (ring-median + feather), re-runs the full QA, and accepts if it now
  clears every gate — instead of escalating to a larger visible cover.
- **Rectangular-band gate on *every* candidate (Phase D).**
  `detect_rectangular_band_visibility` runs on repairs too, not only covers. A
  band is only flagged when it has a real signature — straight box edges or a
  luma offset — so a smooth inpaint on a textured surface is **not**
  mis-demoted. `clean_repaired` requires `visible_band_score ≤ 0.18`,
  `rectangularity ≤ 0.20`, `band_luma_delta ≤ 6`, `edge_box ≤ 0.16`; covers use
  looser ceilings but still reject obvious bands.
- **Product damage over the full changed region (Phase E).** The product gate
  measures `product_mask ∩ changed_region` inside a padded window, scoring
  colour delta, new bright/dark blob, edge retention, **contour break**
  (`1 − edge_retention`), and changed-area ratio. Bright fill on a black
  surface is essentially never allowed.
- **Dark-product / thin-flex routing (Phase F).** White / median / plain fills
  are **banned outright** on `dark_product_surface` and `thin_flex_cable`
  (`WHITE_FILL_BANNED_TOOLS`); those classes lead with stroke-mask,
  dark-surface clone, gamma/contrast match, and segmented repair.
- **Beam penalties for visual residue (Phase H).** The composite score adds
  `4·dot_chain + 3·visible_band + 3·bright_blob + 2.5·contour_break +
  2·rectangularity`, so a candidate with good seam/colour but bad visual
  residue can never out-rank a genuinely clean one.
- **Honest status semantics + provenance (Phase I/J).** A cover after ≥1
  rejected repair is reported as a forced best-effort cover (`forced_cover`).
  Every report — PDF title, HTML, `summary.jsonl`, `run_metadata.json`, and
  each `trace.json` — carries `V12_VISUAL_TRUTHFULNESS` with
  `qa_schema_version = strategy_schema_version = v12`, the git commit and run
  seed. The cover page prints the three honesty counters
  (`clean_repaired_with_{dot_chain,visible_band,product_damage}`), which must
  all be 0.

### V12 results (50-image benchmark, seed 2026)

| Metric | Result | Target |
|--------|--------|--------|
| total / failed_io | 50 / 0 | 0 |
| clean_repaired / clean_covered | 28 / 22 | 28–34 / ≤ 22 |
| manual_review | 0 | 0 |
| zero-metric passes | 0 | 0 |
| `clean_repaired_with_dot_chain` | 0 | 0 |
| `clean_repaired_with_visible_band` | 0 | 0 |
| `clean_repaired_with_product_damage` | 0 | 0 |
| band rejections (demoted to cover) | 21 | — |
| unique final methods | 10 | ≥ 10 |

The band gate alone demoted **21** would-be-false `clean_repaired` outputs to
honest covers — exactly the V12 goal. Per the plan, V12 is **not** optimised
for `clean_repaired` percentage: honest `clean_covered` outranks dishonest
`clean_repaired`. `test_v12_visual_truthfulness.py` locks every guarantee
above (16 cases).

## V11 — Honest QA (no zero-metric passes, dot-chain & product-damage gates)

V11 does **not** add repair methods. It makes the existing ones *honest*: the
V10 gate still accepted bad candidates as `clean_repaired`. The compare PDF
showed `localQA ssim=0.000 color_delta=0.0 boundary_jump=0.0 pass=True`,
dotted/broken-glyph residuals, and a black product surface damaged by a bright
clone. V11 closes each of those holes.

- **Truthful QA, enforced (Phase 1).** A single mandatory
  `validate_qa_metrics()` runs before any candidate can pass.
  `REQUIRED_QA_METRICS` (ssim, color_delta, boundary_jump, seam_delta,
  cover_visibility, product_damage, residual_template_corr,
  residual_text_component, residual_improvement) must all be present, non-null
  and non-NaN; an all-zero core set fails closed
  (`qa_metrics_probably_not_computed`). **`ssim` and `boundary_jump` are now
  actually computed** and surfaced — the PDF's `localQA` line shows real
  values, not zeros, and `pass` reflects the true publish decision.
- **Dot-chain / broken-glyph residual gate (Phase 2).** Beyond template
  correlation, a connected-component detector inside the (horizontally
  expanded) watermark footprint catches a *row of gray dots* or broken glyph
  fragments — the failure mode where the word shape is gone but readable
  pieces remain. Fails on `residual_dot_chain_score > 0.35` or
  (`component_area_ratio > 0.08` and `horizontal_span > 0.25`). On plain
  backgrounds a precise component-level second pass
  (`cleanup_residual_components_with_ring_fill`) clears them — never a
  full-bbox Telea re-run.
- **Product-overlap safety gate (Phase 3/8).** When the footprint overlaps the
  product, the candidate must pass a stricter test measured inside
  `product_mask ∩ changed_region`: `product_color_delta_lab ≤ 10`,
  `product_new_bright_blob_score ≤ 0.12`, `product_edge_retention ≥ 0.85`, and
  `changed_area_ratio ≤ 2.5` (measured over a padded window so a repair that
  spills past the watermark registers). This rejects a white/bright clone
  pasted onto a dark product surface (the Taptic-Engine-style failure).
- **Plain-white fast path + Telea suppression (Phase 4).** `plain_white_fast_path`
  (exact ring fill + matched noise + 1px feather + residual cleanup) runs
  **first** on white/near-white/low-texture backgrounds; Telea drops to
  next-to-last. Telea's dotted gray residue now fails the dot-chain gate
  instead of being accepted.
- **Beam selection, not first-pass (Phase 7).** The loop no longer stops at
  the first passing candidate on risky classes. It tries a bounded set
  (`_MAX_CANDIDATES_BY_CLASS`, 6–12), keeps every candidate that clears the
  hard gates, and picks the best by composite score — only short-circuiting on
  an *excellent*, product-safe candidate. This restores method diversity and
  stops weak early candidates from winning. Telemetry logs `tools_reachable`,
  `tools_tried`, `tools_rejected`, `candidates_passed`.
- **Honest publish rule.** `clean_repaired` now requires
  `metrics_valid AND residual_pass AND product_gate_pass AND geometry_gate`.
  Failures escalate to the adaptive cover — `manual_review` stays **0**.
- **Debug + regression lock (Phase 9/10).** Records carry `v11_*` fields
  (ssim, dot-chain, product overlap/blob/edge-retention, changed-area, beam
  counts); the run prints honesty metrics (`zero_metric_passes`,
  dot-chain/product rejections, `unique_final_methods`).
  `test_v11_regression.py` locks every guarantee above.

### V11 results (50-image benchmark, seed 2026)

| Metric | Result | Target |
|--------|--------|--------|
| total / failed_io | 50 / 0 | 0 |
| clean_repaired / clean_covered | 34 / 16 | covered ≤ 15 preferred |
| manual_review | 0 | 0 |
| zero-metric passes | 0 | 0 |
| product-gate failures among `clean_repaired` | 0 | 0 |
| readable residual among `clean_repaired` | 0 | 0 |
| repaired records with real (non-zero) `ssim` | 34 / 34 | all |
| unique final methods | 11 | ≥ 10 |

`clean_covered` (16) is right at the ≤15 preference; every cover is a genuine
surface reconstruction, not a band — per the plan, honest quality outranks
raising `clean_repaired`. The residual gate measures leftover watermark
**relative to the surrounding surface's own texture baseline**, so a
matched-noise / Telea fill is not mistaken for residual (the bug that, in an
earlier V11 build, forced 36/50 to cover).

Every `clean_repaired` result has `metrics_valid AND residual_pass AND
product_gate_pass`; the `localQA` line in the compare PDF now shows real
`ssim` / `color_delta` / `boundary_jump` values and the true `pass` decision.

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
|  Final Adaptive Cover   |  V10: full-footprint Telea inpaint, routed by
|  (tool #100)            |  in-bbox structure. Reconstructs surface, never
|                         |  a gray band. auto_cover_retry if still readable.
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

### Tool 100: Final Adaptive Cover (rewritten in V10)

**`final_adaptive_cover`** — the absolute last resort, producing `clean_covered`.
In V10 it no longer dims the watermark with a translucent gray patch (which
left it readable). It now **reconstructs the surface**:

- **Routing by actual in-bbox content**, not ROI class (the classifier is
  sometimes wrong). Low-structure surround (in-bbox edge density < 0.10 and
  product overlap < 0.45) → `opaque_footprint`; structured surround →
  light-touch `stroke_level` band / `segmented`, then `opaque_footprint`.
- **`opaque_footprint`** — Telea inpaint over the full watermark bbox,
  extended horizontally (~12%) past the detected box to catch glyphs the
  detector clipped (e.g. a trailing `.com`). Reconstructs the real
  surrounding surface; no synthetic fill, no noise.
- **`stroke_level`** — Telea over the text band (bounding rect of the
  horizontally-dilated strokes), a lighter touch that disturbs less product
  detail. No RGB-noise fill.
- **`segmented`** — splits the bbox into background vs. product surface using
  the product mask and repairs each separately.

Each candidate is checked for **hiding** (mask-aware residual) and
**rectangularity**; the first that both hides and passes the rect gate wins,
else the best-effort (hides first, then lowest rectangularity). Result: a
"covered" image looks like a natural removal, never a gray/white band.

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

**V10 reorder (P5):** on plain / near-white / low-texture / metallic classes,
real-pixel clones and statistical/gradient fills now run **before**
stroke-only logo repair — "try real pixels first, synthesize second, cover
last." The QA gate (not the order alone) is what stops a weak stroke output
from being accepted when a cleaner clone/fill is available.

```
plain_white:             clone_8dir -> white_median -> ring_median -> hard_paste ->
                         seam_scored -> direct_neighbor -> alpha_template ->
                         logo+clone -> stroke_mask -> telea -> adaptive_cover

near_white:              clone_8dir -> white_median -> ring_median -> seam_scored ->
                         surface_gradient -> alpha_template -> logo+clone ->
                         stroke_mask -> telea -> adaptive_cover

low_texture_background:  clone_8dir -> ring_median -> surface_plane ->
                         surface_gradient -> logo+surface -> stroke_mask -> telea ->
                         lama_stroke -> adaptive_cover

simple_product_surface:  same_surface -> same_color -> color_match ->
                         surface_gradient -> noise_preserved -> logo+surface ->
                         stroke_mask -> telea -> lama_stroke -> adaptive_cover

dark_product_surface:    dark_clone -> same_surface -> gamma_match ->
                         contrast_match -> stroke_mask -> logo+surface -> telea ->
                         lama_stroke -> adaptive_cover

glass_or_gradient:       linear_gradient -> bilinear_gradient -> glass_clone ->
                         frosted_noise -> low_freq_gradient -> high_freq_noise ->
                         poisson -> logo+surface -> lama_stroke -> adaptive_cover

metallic_or_reflective:  linear_gradient -> bilinear_gradient -> surface_plane ->
                         same_luminance -> high_freq_noise -> logo+surface ->
                         stroke_mask -> telea -> lama_stroke -> adaptive_cover

thin_flex_cable:         stroke_mask -> cable_repair -> black_flex -> edge_aware ->
                         logo+clone -> segmented -> line_preserving -> telea ->
                         lama_stroke -> adaptive_cover

text_or_label_area:      stroke_mask -> logo+clone -> telea -> adaptive_cover

complex_product_detail:  template_logo -> stroke_mask -> text_filter ->
                         logo+telea -> edge_aware -> contour_guided ->
                         patchmatch -> lama_stroke -> lama_context ->
                         adaptive_cover

unknown:                 clone_8dir -> stroke_mask -> surface_gradient ->
                         telea -> adaptive_cover
```

## Local QA Gate (V10 — truthful, residual-aware)

A repair candidate is accepted as `clean_repaired` only when **all** of the
following hold (P1/P2/P7):

1. **Metrics are valid** — `metrics_valid` is false if every core metric is
   exactly zero (degenerate/empty ROI), which fails closed as
   `qa_metrics_probably_not_computed`. Missing/`NaN` metrics raise
   `MissingQAMetric`. No more `pass=True` on uncomputed scores.
2. **Residual watermark gate passes** — the watermark must be unreadable
   (see below).
3. **Geometry gate passes** — the seam / cover-visibility / product-damage /
   color metrics below.

### Geometry metrics

| Metric | What It Measures | Threshold | Strict | Loose |
|--------|-----------------|-----------|--------|-------|
| `cover_visibility` | Visible rectangle (luma delta + HF drop) | <= 0.20 | 0.16 | 0.26 |
| `seam_delta` | Boundary discontinuity at repair edges | <= 0.18 | 0.144 | 0.216 |
| `product_damage` | Edge retention loss outside repair zone | <= 0.15 | 0.15 | 0.15 |
| `color_delta` | CIE Lab color distance (repair vs ring) | <= 12.0 | 12.0 | 19.2 |
| `watermark_residual`* | High-pass activity in repair zone | <= 0.12 | 0.096 | (gate off) |
| `rectangular_patch_visibility`* | Human-visible rectangular patch | per-class | tightened | (gate off) |

\* `watermark_residual` and `rpv` are **noisy on textured/dark surfaces** —
they fire on successful inpainting because removing the watermark *is* an
edge-density change. In V10 they hard-gate only on strict/detail classes
(where in-ROI structure is genuinely at risk) and otherwise drive scoring
only. The authoritative "is the watermark gone" check is the residual gate;
the authoritative "is there a band" checks are `cover_visibility` and the
cover-rectangularity gate. `texture_drop`/`rpv` are also suppressed on flat
surrounds (no texture to "drop").

### Residual watermark gate (P2)

Measured **at the known glyph locations** — the stroke/logo mask dilated with
a horizontal-biased kernel (21×9, so it spans inter-letter gaps and trailing
glyphs) — relative to the surrounding surface's own texture baseline. This is
robust on metallic/dark/textured surfaces (honest texture is in the baseline
too) yet still catches a leftover halo or a broken-but-readable `.com`.

| Field | Meaning | Pass when |
|-------|---------|-----------|
| `residual_template_corr` | best normalized correlation of the canonical `sunsky-online.com` template vs. cleaned high-pass | <= 0.18 |
| `residual_text_component` | fraction of mask pixels still carrying stroke-level high-pass above baseline | <= 0.16 |
| `residual_improvement` | stroke-energy reduction at the watermark region | >= 0.75 (waived only when the original had no watermark structure) |

### Cover gate (P4)

Adaptive covers additionally pass a rectangularity gate so they never read as
a band: `cover_rectangularity <= 0.25`, `cover_luma_delta <= 8`,
`cover_texture_drop <= 0.55`, `cover_edge_box_score <= 0.20`.

**Composite score** (lower = better), used to rank failed candidates:
```
final = 2.0*wm_residual + 2.0*cover_vis + 2.5*seam + 1.5*product_damage
      + 0.5*(1-texture_consistency) + 1.0*color_delta/12 + 1.0*edge_damage
      + 1.5*rpv + 2.0*residual_template_corr + 1.5*residual_text_component
      + 1.0*(1-residual_improvement)
```

When no tool passes, the pipeline falls through to `final_adaptive_cover`. If
even the cover still shows a readable watermark, `auto_cover_retry` escalates
to a stronger stroke/segmented cover — manual review stays at **0**.

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
| `mark_remover.py` | ~5,100 | Main pipeline: detection, masks, QA, orchestration, V12 publish gate + V13 final visual gate + **V14 finalize hook (near-miss rescue / micro-cover routing)** + provenance/counters |
| `progressive_repair.py` | ~4,900 | Strategy bank: 100 tools, ROI classifier, V12 truthful QA + residual/band/product gates + `final_publish_gate` + **V14 best-failed-candidate surfacing** |
| `v13_gates.py` | ~620 | **V13 final visual fidelity gates** (unchanged in V14): `final_visual_publish_gate_v13` + dot-chain v2, visible-patch-shape, product-silhouette, product-overlap, protected-text detectors + honesty counters |
| `v14_patch.py` | ~600 | **V14 better candidates + V15 cover polish**: `soft_qa_context`, `classify_failure`, `near_miss_rescue`, `segmented_micro_cover` beam search, `full_bbox_cover_banned`, `polish_cover`, cover honesty counters, `v14_finalize` orchestrator |
| `v15_patch.py` | ~230 | **V15 cover-quality patch**: `widen_text_mask`, `full_region_residual_ok` (ghost-aware), `dark_stroke_cover`, `is_dark_surface` |
| `v13_report.py` | ~150 | V13 honesty report / CI gate + **V14 cover-side counters & target metrics**: tallies must-be-zero counters from per-image `qa.json` |
| `test_v10_regression.py` | ~190 | V10 visual regression lock (no readable watermark / no product damage) |
| `test_v11_regression.py` | ~270 | V11 honesty lock (zero-metric/dot-chain/product-damage gates, plain-white routing) |
| `test_v12_visual_truthfulness.py` | ~260 | V12 lock: publish gate authority, dot-chain 0.28, band gate, dark-product ban, version/schema consistency |
| `test_v13_final_visual_gate.py` | ~70 | V13 lock: final visual gate accepts clean / rejects bad output, repaired **and** covered |
| `test_v13_detectors.py` | ~95 | V13 lock: dot-chain v2, visible-patch-shape, silhouette, product-overlap detectors |
| `test_v13_report_honesty.py` | ~55 | V13 lock: honesty counters increment on bad output, stay 0 on clean |
| `test_v14_regression.py` | ~230 | **V14 lock**: failure classification, full-bbox bans, near-miss rescue, micro-cover beam, cover counters + end-to-end honesty guarantees on the 8 Section 8 weak cases |
| `test_v15_patch.py` | ~120 | **V15 lock**: mask widening (bounded, skips busy product), ghost-aware full-region residual (catches faint ghost, ignores product texture), dark-stroke cover |
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
| V8 | Progressive repair strategy bank | 100 tools, 11 ROI classes, per-class strategy ordering, local QA gate |
| V9 | Method families + RPV gate | Truthful status via method family, product-overlap guard, rectangular-patch-visibility gate |
| V10 | Quality patch: truthful QA + honest covers | Fail-closed QA, mask-aware residual watermark gate, 3-layer mask, inpaint-based covers (no gray bands), strategy reorder, diversity telemetry, regression lock |
| V11 | Honest QA enforcement | Mandatory `validate_qa_metrics` (real ssim/boundary_jump, no zero-metric passes), dot-chain/broken-glyph residual gate, product-overlap damage gate (bright-blob/colour/edge/area), plain-white fast path, beam selection over first-pass, honesty telemetry + regression lock |
| **V12** | **Visual truthfulness** | **Unified `final_publish_gate` as the only `clean_repaired` authority, dot-chain ceiling 0.35→0.28 + count rule, residual micro-cleanup before cover, rectangular-band gate on every candidate, full-changed-region product gate + contour-break, dark-product/thin-flex white-fill ban, beam visual-residue penalties, forced-cover semantics, version/schema provenance in PDF/HTML/JSONL/trace + three must-be-zero counters** |
| **V13** | **Final visual fidelity** | **`final_visual_publish_gate_v13` applied to repaired AND covered outputs (honest demotion of weak repairs), new dot-chain-v2 / visible-patch-shape / product-silhouette / product-overlap / protected-text detectors in `v13_gates.py`, `v13_report.py` must-be-zero honesty counters, version → `V13_FINAL_VISUAL_FIDELITY` / schema `v13`** |
| **V14** | **Better candidates (gate unchanged)** | **Additive `v14_patch.py`: adaptive soft-metric context, near-miss rescue (component cleanup + gamma/Lab match + seam smoothing) of soft-fail repairs before cover, segmented micro-cover beam search replacing full-bbox cover with full-bbox-on-product/flex/glass/metallic/text bans, fresh per-candidate hiding verdict + V13-passing cover selection, cover-side honesty counters + target metrics in `v13_report.py`, best-failed-candidate surfacing from the repair loop, `test_v14_regression.py`. clean_repaired 25→29 / 50 with zero false repairs. Version → `V14_BETTER_CANDIDATES`, V13 gate version stays `v13`.** |
| **V15** | **Cover quality (gate unchanged)** | **Cover polish (`polish_cover`: clamped Lab match + wider seam feather, uncapped boundary term) → worst cover boundary jump 222→102; additive `v15_patch.py`: `widen_text_mask` (full `sunsky-online.com` line, 3 product-safety guards), `full_region_residual_ok` (ghost-aware, template-correlation based — ignores flex-cable texture), `dark_stroke_cover` (Telea from dark neighbours, no light fill); `test_v15_patch.py`. Bright-box covers + faint trailing ghosts removed, zero false repairs / product / silhouette damage. Remaining 6 cable-edge covers deferred to V16 structure reconstruction. Version → `V15_COVER_QUALITY`, V13 gate version stays `v13`.** |

## License

Proprietary. Internal use only.
