# Orchestrator Architecture — ClearMark

_Production architecture for the watermark-removal system (Alex, 2026-06-14). Three
components; **no Manager Agent**. The Orchestrator is a deterministic Python state machine,
not an AI agent._

```
                 +------------------+
                 |   Orchestrator   |   deterministic state machine (orchestrator.py)
                 +--------+---------+
            +-------------+-------------+
            v                           v
   +----------------+          +----------------+
   |   Owner Agent  |          |   Audit Agent  |
   | (owner_agent)  |          |   (audit.py)   |
   | detect/repair  |          | verify final,  |
   | final+manifest |          | residual/patch/|
   |                |          | damage → PASS/  |
   |                |          | FAIL            |
   +----------------+          +----------------+
            \___________ files only ___________/
                          |
                  PASS → publish   FAIL → retry / reject
```

## Design principles
1. **Owner produces. Audit validates. Orchestrator decides.** Responsibilities never overlap.
2. **Agents never communicate directly** — only through files: `owner_manifest.jsonl`,
   `audit_results.jsonl`, `state.json`.
3. **Audit has veto authority.** If Audit rejects, the Owner cannot override.
4. **No AI-based workflow control.** Retries, publish, job state, queueing are the
   Orchestrator's deterministic logic — never an LLM.

## Directory structure (per batch)
```
job_xxx/
  originals/ finals/ publish/ rejected/ retry/ masks/ logs/
  owner_manifest.jsonl   audit_results.jsonl   audit_feedback.jsonl   state.json
```
`originals/` is never modified. `publish/` holds only Audit-PASSed images.

## Image state machine
```
NEW → OWNER_RUNNING → AUDIT_RUNNING → PASS | RETRY | REJECT      (no other transitions)
```
Per round the Orchestrator: marks pending images `OWNER_RUNNING`, runs the Owner, reads
`owner_manifest.jsonl` (a final ⇒ `AUDIT_RUNNING`; `auto_rejected`/`failed` ⇒ `REJECT`),
runs the Audit on the finals, reads `audit_results.jsonl`, and transitions each image to
`PASS` (→ publish), `RETRY` (→ Owner next round with the failure reason), or `REJECT`.

* **Retry:** `MAX_RETRY = 2`; `retry_count > 2 → REJECT`. Never an infinite loop.
* **Fail-safe:** an Audit *crash* (no result for an item) **halts** the run — it never
  mass-rejects recoverable images. Items stay `AUDIT_RUNNING` and re-audit on the next run.
* **Resumable:** `state.json` is written atomically; a re-run resets transient
  `OWNER_RUNNING`/`AUDIT_RUNNING` and continues. Same job → continues, never restarts.

## Owner manifest schema (one line/image)
```json
{"id":"image001","original":"originals/image001.jpg","final":"finals/image001.jpg",
 "owner_status":"cleaned","method":"repair","watermark_status":"watermark_found"}
```
`owner_status` ∈ `cleaned · copied_no_watermark · auto_rejected · failed`.

## Audit result schema (one line/image)
The Audit Agent emits a decision class + `publish_allowed` + `recommended_next_action`
(`publish`/`retry_repair`/`try_cover`/`auto_reject`). The Orchestrator maps decisions →
transitions: `publish_allowed` ⇒ PASS; a retryable action under the cap ⇒ RETRY; otherwise
REJECT. Decision classes: `PASS · PASS_WITH_MINOR_BACKGROUND_ARTIFACT ·
REJECT_RESIDUAL_WATERMARK · REJECT_VISIBLE_PATCH · REJECT_PRODUCT_DAMAGE ·
REJECT_PROTECTED_TEXT_DAMAGE · REJECT_UNNATURAL_IMAGE · REJECT_UNCERTAIN`.

## Publish logic
Publish **only** if Audit PASS. No exceptions. The Owner never publishes or self-approves.

## Development roadmap (gated)
P0 benchmark (500) · P1 finder (recall>98 / prec>90) · P2 masks (95% usable) ·
P3 cleaner (residual<5 / damage<2) · P4 pipeline 1k (PASS>95 / residual<1 / patch<1 /
damage<0.5) · P5 pilot 3k (stable, no category-wide failure) · P6 production 20k.

A **Research Agent** may be added later — **offline only**, never in production workflow control.

---
**Implementation:** `orchestrator.py` (`orchestrate()` loop + `state.json`) drives
`owner_agent.py` (Owner) and `audit.py` (Audit) as subprocesses, reading their file outputs.
Deterministic, resumable, fail-safe. Validated end-to-end on a seeded batch: 4 published, 1
rejected after 2 retries (Audit residual veto upheld).
