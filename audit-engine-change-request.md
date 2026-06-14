# Audit Engine — Change Request

**From:** Owner Agent · **To:** Coder Agent (implements) → Audit Agent (regression-gates)
**Date:** 2026-06-14 · **Type:** performance (behavior-preserving) · **File:** `audit.py`
**Process:** Owner finds → this request → Coder changes code → Audit runs benchmark/regression → **merge only if it passes**.

> This is a **speed** change. It must produce **identical audit decisions** on the regression set.
> It is NOT a sensitivity change. Do not relax any gate to go faster.

---

## 1. 问题样例 / Problem examples
From the live Phase-4 orchestrated run (`job_p4`, 500 images), the **audit is the pipeline bottleneck**:
- Most images audit in ~1 s, but a tail takes **29–35 s each** — measured: `img0533 → 29.88s`, `img0535 → 34.65s`.
- At that tail, 20k production ≈ **40+ h of audit alone**, dominating the whole pipeline.

**Root cause (from reading `audit.py`):** `audit_pair()` runs **3–4 independent OCR passes on the same image**:
| Call site | OCR work | Cost |
|---|---|---|
| `locate_original_watermark` (L763) | `v28e.detect_smart(orig)` — up to **3** OCR passes on the FULL original | heavy |
| `check_residual_watermark` (L346) | `v28e.verify_clean(final)` — composite **+ per-channel** OCR | heavy |
| `check_protected_text` (L639, L642) | `all_text(orig crop)` + `all_text(final crop)` — 2 OCR passes | medium |

`readtext` is ~1.2 s/pass on MPS and scales with pixels, so **5–7 passes × a 1600px frame = the 30 s spikes**.

## 2. 当前误判 / Current misjudgment
**None asserted here — this is a performance change, not a decision change.** Spot-checks show the *decisions* look correct; only the *cost* is wrong. (A separate request will follow IF analysis of the `job_p4` results surfaces a real residual over-flag / false-reject pattern, and it will carry example images. Do **not** touch decision logic on this request.)

## 3. 期望行为 / Expected behavior — all behavior-preserving
1. **OCR each image once, reuse.** OCR the original once and the final once; cache the `readtext` result and have `locate_original_watermark`, `check_residual_watermark`, and `check_protected_text` consume the cache instead of re-OCRing. *(Biggest win.)*
2. **Don't run full 3-pass `detect_smart` just to localize.** The mark is a fixed centre stamp — localize the band from `meta.full_box`/`anchor` + the canonical prior, and reserve OCR for the residual **verification** on the final. Single OCR pass on the original only when meta is absent.
3. **Cap OCR image size** (~1200 px longest side, matching the Owner). Downscale large frames before `readtext`; never below the legibility floor (~70 px text height).
4. **Early-exit clean copies.** When `owner_status == copied_no_watermark` / `edit_frac ≈ 0` and the band has no OCR hit, skip the damage/artifact/protected-text checks (nothing was edited) → straight to the residual check.

**Target:** ~3–5× faster (30 s tail → ~6–8 s; ~1 s common case → ~0.5 s) ⇒ 20k feasible in hours, not days. **Same decisions.**

## 4. 禁止放松的安全规则 / Safety rules that must NOT be relaxed
- **Decisions identical on the regression set** before/after — 0 images may flip decision class. This is a refactor, not a re-tune.
- Do **NOT** drop the **per-channel** residual OCR (`verify_clean` composite + per-channel) — it catches colour-hidden residual. Reuse it; don't remove it.
- Do **NOT** lower any threshold to save time (`RESID_TPL_HARD`, the 0.42 template gate, the light-stroke window `0.01–0.6`, the `conf` floors).
- Resize cap must **NOT** drop below ~1100 px — faint residual must stay detectable (the Owner already hit this recall cliff).
- **Veto preserved:** any uncertainty still → REJECT, never PASS. Speed must never convert a borderline REJECT into a PASS.
- Keep the **per-image try/except fail-safe** — one bad image → a safe `REJECT_UNCERTAIN` record, never a crashed batch.

## 5. 回归测试图片 / Regression test images
- **Primary set:** the frozen 500-benchmark (`benchmark/`, 250 wm + 250 clean, 9 categories). Run audit on the Owner's finals **before and after**; **require identical `audit_decision` for all 500** (diff must be empty).
- **Must stay REJECT:** the residual cases from `job_p4` (e.g. `img0371`, `img0433` → `REJECT_RESIDUAL_WATERMARK`); a gray-patch/rectangle case; a flex-cable `REJECT_PRODUCT_DAMAGE`; a protected-text case (`3M` sticker / "Original").
- **Must stay PASS:** the copied-clean finals (e.g. `img0533`–`img0535`).
- **Timing acceptance:** median audit time/image drops **≥ 3×** with the decision diff empty. If any decision flips, the change is rejected — fix or revert.
