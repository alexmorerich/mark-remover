# Migration Plan: Consolidate watermark removal into `mark-remover`

> Status: PLAN (no code yet) · Owner of record after migration: `mark-remover`
> Source feature being absorbed: `b2bweb/scripts/remove-watermarks.py` (+ 3 helpers)

---

## ⟳ Status refresh — 2026-06-14

The plan below predates the current engine. Two updates:

**1. Project is now split into THREE stages** (Alex-approved 2026-06-14), not two — detection and
removal alone leave no backstop, so an independent audit gate is mandatory. The shared surface is
frozen as a versioned contract. Read these first:
- [`CONTRACT_v1.md`](./CONTRACT_v1.md) — the Detect → Repair → Audit hand-off (one accreting JSONL
  record per image). Rich `detect` block (`roi_type`, `overlap_product`, `risk`, pixel `mask_path`),
  `repair` block (strategy by ROI), `audit` verdict + `terminal` state.
- [`agents/AGENT_1_detect.md`](./agents/AGENT_1_detect.md) — Detect/Locate. **Aggressive recall**;
  emits ROI + risk; owns the measurement harness.
- [`agents/AGENT_2_repair.md`](./agents/AGENT_2_repair.md) — Repair/Cover. Strategy by `roi_type`;
  product-protection; **repair → cover → reject** ladder.
- [`agents/AGENT_3_audit.md`](./agents/AGENT_3_audit.md) — Final Audit/Reject. Independent
  deterministic gate; the brand-safety guarantee.

**Safety model:** *aggressive to find, conservative to publish.* The zero-damage guarantee moves to
the **audit gate** (backup makes every repair reversible), so detection can chase recall. **No
manual-review queue** — terminal states are `published` / `auto_rejected` / `clean`. This
**supersedes Phase 3's "keep manual queue as a safety valve"** below: if Repair can't clean it and a
cover is still visible, Audit `auto_rejected`s it and restores the original. Humans never sit in the
per-image production loop (sampling the clean pile to *measure* recall is a separate offline activity).

**2. Several items below are already DONE — the plan's "V10 / no in-place / no validate"
baseline is stale.** `run_bulk.py` + `v27_clean.py` + `v28_clean.py` now provide:
- `process` — inpaint **in-place + backup** ✅ (was Phase 3)
- `validate` — re-detect on cleaned images, residual gate ✅ (was Phase 3)
- `restore` — copy originals back from backup ✅ (was Phase 3)
- v28 full-extent mask + hard verify gate + GPU/MPS pipeline ✅

The architecture references below (`detector.py` / `mark_remover.py` / `progressive_repair.py`,
"~13,700 LOC, V10") describe an **earlier** mark-remover and no longer match the current
single-file v27/v28 engine. Treat the *strategy* below as current; treat the *file/LOC
specifics* as historical.

**Current priority — Phase 1 (still open, highest leverage):** port b2bweb's labeled set
(`validation-set.json` 1,811 lines + `ground-truth.json` 392 lines) and write the bake-off
scorer **into mark-remover**. Until this lands, recall is unmeasured and the Phase 2 detector
bake-off cannot run. This is Agent B's first task.

---

## Context / headline finding

`b2bweb/scripts/remove-watermarks.py` and `mark-remover` are **two implementations of the
same tool** — both remove the `sunsky-online.com` watermark from iPhone-part product images
(same canonical template, same "skip iPhone 14+" rule, same optional LaMA model at
`/tmp/big-lama-model.pt`). So this is a **consolidation of two generations of one tool**, not a
merge of two different features.

Each side leads on a different axis:

- **mark-remover has the better engine** — 100 repair tools / 6 families, 11-ROI strategy bank,
  truthful fail-closed QA, residual (RPV) gate, honest covers (no gray bands), cleaner modular
  architecture (`detector.py` / `mark_remover.py` / `progressive_repair.py`).
- **b2bweb's script has the better operational harness** — zero-false-positive presence gate,
  in-place replace + backup, manual-review queue, and a large labeled dataset + metrics harness.

**Direction:** keep mark-remover's engine; harvest b2bweb's ops features + data into it; then
delete b2bweb's copy and have b2bweb call mark-remover. Do **not** copy the script in as a second
pipeline (that recreates the duplication inside one repo).

### Repos
- b2bweb: `/Users/alexkou/Documents/github/b2bweb` (feature at `scripts/remove-watermarks.py`)
- mark-remover: `/Users/alexkou/Documents/github/mark-remover` (system of record after migration)

## Guiding principles
1. **One engine, not two pipelines.** mark-remover's engine is the core; b2bweb's script is
   harvested then deleted. Never run both.
2. **Data migrates first.** b2bweb's labeled set is the referee for every later decision.
3. **The bake-off is the go/no-go gate.** Nothing cuts over until the merged engine meets or
   beats today's precision/recall on the real library.
4. **Preserve zero false positives.** A false positive damages a clean product photo — the one
   regression you cannot ship.
5. **Prove correctness before changing transport.** Get parity via CLI first; turn it into an
   async service second.

---

## Phase 0 — Freeze & baseline (~0.5 day)
**Goal:** capture exactly what current production does, to prove no regression later.

- Tag current b2bweb watermark code as `watermark-baseline`.
- Run b2bweb's `metrics`/`metrics2` harness against `validation-set.json` + `ground-truth.json`;
  **record** precision, recall, false-positive count, manual-queue rate.
- Snapshot a fixed sample (~500 images spanning all ROI classes) from
  `content/products/assets/` to a read-only fixture dir. All later comparisons use this sample.

**Deliverable:** one-page baseline scorecard (the numbers the merged engine must beat).
**Gate:** numbers reproduced and written down. **Rollback:** n/a (read-only).

---

## Phase 1 — Migrate labeled data + build bake-off harness (~1 day)
**Goal:** let mark-remover be *measured* before changing its behavior.

Copy into mark-remover (no dependencies, lowest risk):
- `generated/watermark/validation-set.json` (1,811 lines)
- `generated/watermark/ground-truth.json` (392 lines)
- `scripts/watermark-golden-set/` (manifest + README)
- `scripts/watermark-regression-files.txt`
- canonical/template PNGs **only if** they differ from mark-remover's (compare first).

Build a **bake-off harness** in mark-remover that runs any detection+removal path against the
labeled set and emits the Phase 0 scorecard shape (precision, recall, FP count, manual rate, plus
residual-template-corr and cover metrics).

**Deliverable:** mark-remover can score itself against b2bweb's ground truth on demand.
**Gate:** harness reproduces Phase 0 baseline when pointed at b2bweb's logic (trust check).
**Rollback:** delete added files; engine untouched.

---

## Phase 2 — Detector reconciliation (the bake-off) (~2–3 days) · CRITICAL GATE
**Goal:** decide which detection path wins, using data not opinion.

Run three configs through the Phase 1 harness on the fixed sample:
- **A:** mark-remover `detector.py` (`detect_watermark_v2`) as-is.
- **B:** b2bweb's presence gate (fast + deep, zero-FP tuned).
- **C:** hybrid — b2bweb's zero-FP gate as the *guard*, mark-remover's detector for *localization*.

**Decision rule:** fewest false positives first, then highest recall. FP is dominant.
Likely (hypothesis, not assumption): b2bweb's gate wins on FP because it's tuned on the real
corpus → port it as the front gate, keep mark-remover for repair. Let the numbers decide.

**Deliverable:** chosen detection path with a scorecard meeting/beating baseline.
**GO/NO-GO:** merged detection ≥ baseline precision/recall **and** FP ≤ baseline. If it fails,
stop and tune here. **Rollback:** mark-remover keeps its own detector.

---

## Phase 3 — Port operational features (~3–5 days)
**Goal:** bring mark-remover up to b2bweb's production capability.

| Feature | Why | Source |
|---|---|---|
| In-place replace + backup | b2bweb edits the 4.3GB library in place, backs up to `/tmp/watermark-backup`; mark-remover only writes a new tree | `clean`/`replace` |
| `replace` / restore | Undo a bad batch, swap originals from source | b2bweb subcommand |
| `validate` | Verify a cleaned batch before commit | b2bweb subcommand |
| Manual-review queue | Human-in-loop for uncertain cases + preview HTML. **Decide:** keep, or rely on mark-remover auto-escalation. Recommend keep initially as a safety valve | b2bweb manual-queue |
| Metrics harness | Already done in Phase 1 | — |
| Drop `wm-alpha-clean.py` / `calibrate-watermarks.py` / `run_lama.py` | Superseded by the 100-tool engine; do not port | leave behind |

**Output-model decision:** the service must support **both** "write to new dir" (safe default)
and "in-place replace + backup" (b2bweb production mode), selectable by flag.

**Deliverable:** mark-remover does everything b2bweb's script did, on the better engine.
**Gate:** full scorecard after porting still ≥ baseline; manual rate not worse.
**Rollback:** feature-flag each ported capability; disable individually.

---

## Phase 4 — Define & freeze the contract (~0.5 day)
**Goal:** lock the one shared surface between b2bweb and mark-remover.

**Contract v1 (CLI-compatible, synchronous):**
- **Input:** assets dir (or file list); mode (`scan` | `process` | `validate` | `replace`);
  output target (`--out DIR` *or* `--in-place` + backup); flags (presence-gate threshold,
  exclude-iphone14+).
- **Output:** cleaned images (in place or out dir) + `scan-report.json`, `summary.jsonl`,
  manual-queue dir, metrics json — stable filenames/shapes.
- **Versioned:** label `contract v1`. Any input/output shape change is a deliberate, announced
  event touching both repos in lockstep.

**Deliverable:** written contract doc in both repos.
**Gate:** both sides agree it covers today's 4 npm scripts. **Rollback:** n/a.

---

## Phase 5 — Cut b2bweb over (~1 day)
**Goal:** b2bweb stops owning watermark logic; calls mark-remover.

- Repoint the 4 npm scripts (`watermark:scan/clean/validate/replace`) to invoke mark-remover
  (sibling repo / pinned dependency) instead of the local script.
- **Shadow mode first:** run both old script and new service on real batches, diff outputs.
  Zero unexpected diffs = green light.
- Flip npm scripts to mark-remover. Leave `scripts/remove-watermarks.py` in tree but unused.

**Deliverable:** b2bweb production workflow runs on mark-remover.
**Gate:** shadow diff clean; one real batch validated via the service.
**Rollback:** repoint npm scripts to the frozen local script (one-line revert).

---

## Phase 6 — Decommission & document (~0.5 day)
**Goal:** remove the duplication for good.

- Delete `scripts/remove-watermarks.py` + 3 helpers + templates/golden set/validation data from
  b2bweb (all now in mark-remover). Keep the `watermark-baseline` tag for history.
- b2bweb README: "watermark removal lives in mark-remover; see contract v1."
- mark-remover README: now system of record; document ops features + contract.

**Gate:** b2bweb has zero watermark logic except the contract call.
**Rollback:** restore from the baseline tag.

---

## Phase 7 (optional, later) — Async service for heavy infra
**Goal:** realize independent scaling (GPU/LaMA) per the infra decision.

Only after CLI parity is proven: wrap mark-remover behind a job queue + worker; b2bweb submits a
job and polls/gets a callback. **Contract v2** — same input/output semantics, async transport.
Correctness first, transport second.

---

## Two-agent governance
- **Web agent** owns b2bweb. **Watermark agent** owns mark-remover. Separate repos → no merge
  conflicts.
- **The only shared surface is contract v1.** Neither agent changes it unilaterally; contract
  changes are coordinated, versioned, lockstep across both repos.
- Until Phase 5 cutover, the watermark agent works entirely in mark-remover and b2bweb's copy
  stays frozen — so the two agents never touch the same code during the migration.

---

## Risk register
| Risk | Severity | Mitigation |
|---|---|---|
| Merged engine regresses on real library | High | Phase 2 bake-off is a hard GO/NO-GO gate vs labeled data |
| New false positives damage clean photos | High | FP is dominant criterion; shadow-mode diff before cutover |
| Lost zero-FP tuning from b2bweb's gate | Med | Port the gate as-is (Phase 2 B/C), don't reinvent |
| Manual-queue behavior changes silently | Med | Track manual rate in every scorecard; keep queue as safety valve |
| Contract drift between two agents | Med | Versioned frozen contract; lockstep changes only |
| In-place replace corrupts assets | Med | Mandatory backup (`/tmp/watermark-backup`); `replace`/restore command |

## Sequencing at a glance
**Data + harness → bake-off (GATE) → port ops → freeze contract → shadow → cutover → delete →
(later) async service.**

Estimate: ~8–13 working days for Phases 0–6 (Phase 2 is make-or-break). Phase 7 is separate.

## Key facts reference (from investigation)
- b2bweb watermark code: **zero coupling** to b2bweb internals (no DB/config/auth imports) —
  100% extractable. ~7,000 LOC main script + 3 helpers; integrated only via 4 npm scripts.
- mark-remover: ~13,700 LOC across `detector.py` / `mark_remover.py` / `progressive_repair.py`;
  V10; writes a new output tree (no in-place replace today); no manual queue; 7-case regression
  test only (vs b2bweb's 1,811-line labeled validation set).
- Shared deps: Python, opencv, numpy; optional torch/LaMA. Same watermark target, same template.
