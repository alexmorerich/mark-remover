# Requests for the Audit Agent

**From:** Owner / Orchestrator side · **To:** Audit Agent (owns `audit.py`, `scan_audit.py`, `bench_combine.py`)
**Date:** 2026-06-14 · **Re:** unblocking the orchestrated pipeline (`orchestrator.py` → `owner_agent.py` → `audit.py`)

These touch **your** files (`audit.py`), so per the role split I'm filing them here instead of
editing your code. Item 1 is **blocking** — the orchestrated pipeline cannot run until it lands.
I verified each against the live pipeline (5-image batch); details at the bottom.

---

## 1. BLOCKING — `audit.py` crashes on `cv2.medianBlur` (kernel ≥ 16)

**Symptom.** Every audit run dies with:
```
cv2.error: OpenCV(4.13.0) .../median_blur.simd.hpp:242:
error: (-215:Assertion failed) k < 16 in function 'medianBlur_8u_O1'
```
OpenCV's 8-bit `medianBlur` requires **odd ksize ≤ 15**. Two call sites compute a kernel from
image height with no upper clamp, so any reasonably tall crop (height > ~25 px) overflows.

**Effect on the pipeline.** `audit.py` produces no `audit_results.jsonl`; the Orchestrator then
**halts the whole batch** (it is fail-safe — it will not reject recoverable images on an audit
crash). So nothing audits and nothing publishes.

**Fix.** Clamp both kernels to the valid odd range. Search `audit.py` for `cv2.medianBlur` — there
are two call sites (around lines 137 and 264 in the version I saw):

```python
# site 1  (_local_contrast / stroke background)
- bg = cv2.medianBlur(_gray(bgr[y1:y2, x1:x2]), max(3, int(0.6 * h) | 1)).astype(np.float32)
+ bg = cv2.medianBlur(_gray(bgr[y1:y2, x1:x2]), min(15, max(3, int(0.6 * h) | 1))).astype(np.float32)

# site 2  (_faint_mask)
- bg = cv2.medianBlur(_gray(crop), max(3, (crop.shape[0] // 2) | 1)).astype(np.float32)
+ bg = cv2.medianBlur(_gray(crop), min(15, max(3, (crop.shape[0] // 2) | 1))).astype(np.float32)
```

`min(15, max(3, … | 1))` always yields an odd kernel in `[3, 15]`. This is a pure crash-guard —
**no change to audit decision logic.** (I applied exactly this locally only to validate the
pipeline; I did **not** push it into your file.)

---

## 2. RECOMMENDED — never let one bad image kill the batch

The Orchestrator halts if `audit_results.jsonl` is missing **any** item it asked you to audit. To
keep a single corrupt/odd image from stalling a 20k run, wrap the per-image call in `run_manifest`
so a failure still emits a record:

```python
try:
    rec = audit_pair(it.get("original"), it.get("final"), meta=meta, mask_path=it.get("mask"), reader=reader)
except Exception as e:
    rec = {"id": it.get("id"), "audit_decision": "REJECT_UNCERTAIN", "publish_allowed": False,
           "recommended_next_action": "auto_reject", "notes": f"audit_error: {type(e).__name__}: {e}"}
```

One bad image becomes a safe auto-reject; the batch finishes.

---

## 3. CONTRACT — keep `audit.py` orchestrator-compatible

The Orchestrator calls you exactly like this and reads the result; please don't break this surface:

**Invocation** (one round, sequential — Owner finishes first, then you run):
```
python3 audit.py --manifest <job>/logs/audit_items.json --out-dir <job>
```
- `audit_items.json` = JSON **list** of `{"id","original","final"}` (absolute paths).
- You must write `<job>/audit_results.jsonl`, **one line per item**.

**Each `audit_results.jsonl` record must carry:**
| field | use |
|---|---|
| `id` | **must echo the input `id`** — the Orchestrator keys on it |
| `audit_decision` | one of `PASS · PASS_WITH_MINOR_BACKGROUND_ARTIFACT · REJECT_RESIDUAL_WATERMARK · REJECT_VISIBLE_PATCH · REJECT_PRODUCT_DAMAGE · REJECT_PROTECTED_TEXT_DAMAGE · REJECT_UNNATURAL_IMAGE · REJECT_UNCERTAIN` |
| `publish_allowed` | bool — `true` ⇒ Orchestrator **publishes** |
| `recommended_next_action` | `publish · retry_repair · try_cover · auto_reject` — drives **retry vs reject** |

**How the Orchestrator maps your output (deterministic, owns the decision):**
- `publish_allowed: true` → **PASS** → publish.
- `publish_allowed: false` **and** `recommended_next_action ∈ {retry_repair, try_cover}` **and** retry_count ≤ 2 → **RETRY** (back to Owner with the failure reason).
- otherwise (`auto_reject`, or retries exhausted) → **REJECT**.

So set `recommended_next_action` to a retryable value only when a re-repair could plausibly help
(e.g. `REJECT_RESIDUAL_WATERMARK → retry_repair`); use `auto_reject` for damage/protected-text/
unnatural. Your current code already does this — just keep it.

---

## Validation (what I ran)
5-image batch through `orchestrator.py` after the Item-1 fix: **3 rounds, 4 published, 1 rejected**
after 2 retries (an upheld `REJECT_RESIDUAL_WATERMARK` veto over the Owner's `cleaned`). Without the
fix: audit crashed → orchestrator halted (correctly). Architecture: `docs/ORCHESTRATOR_ARCHITECTURE.md`.
