# Audit Engine — Leak-Fix Change Request (follow-up to the misjudgment patch)

**From:** Owner Agent · **To:** Coder Agent → Audit Agent (regression-gates)
**Date:** 2026-06-15 · **Type:** **SAFETY** (close a watermark leak) · **File:** `audit.py` · **Priority:** blocks production (Phase 6)
**Process:** Owner finds → this request → Coder fixes → Audit regression-gates → merge only if it passes.

> The misjudgment patch (`6f0c796`) is net-positive — **keep it**. But the Owner's independent
> no-leak sweep found it opened **one watermark leak** on extreme-textured surfaces. This closes
> that leak **without** re-introducing the false-positives the misjudgment patch fixed.

---

## 1. 问题样例 / Problem example
Owner's independent no-leak veto (100 known-watermarked images, textured-weighted, audited as
unedited finals): **99/100 REJECT, 1 LEAK.**
- **Leak:** `10-pcs-earpiece-receiver-mesh-covers-for-iphone-x-1.jpg` (750×750, iPhone earpiece
  **mesh** covers). The crop shows a clearly-**visible** `sunsky-online.com` over the mesh — yet the
  patched audit returns **PASS** (`publish_allowed=true`).
- Baseline (pre-patch) audit: this image **REJECTed**. So the patch *introduced* the leak.

## 2. 当前误判 / Current misjudgment (regression introduced by 6f0c796)
On textured/dark ROI the patch made the **template/dot/ghost** residual path diagnostic-only and
**trusts OCR**. But on this mesh the watermark garbles OCR:
- detection-grade composite OCR reads `'bnline am'` @ **conf 0.23** (a real `online.com` fragment — matches `SUNSKY_TOKENS`).
- the audit's `verify_clean` reads **0** (less sensitive than detection OCR on textured).
→ OCR signal lost **and** template suppressed ⇒ no residual evidence ⇒ **PASS**. A visible watermark publishes.

## 3. 期望行为 / Expected behavior
On textured/dark ROI, **do not blanket-PASS when OCR is merely weak** — distinguish "clean texture"
from "watermark garbled by texture":
1. Run a **detection-grade sensitive** residual OCR on the centre band (CLAHE-2× upscale + per-channel,
   the same the finder uses), not just `verify_clean`'s default.
2. If it returns **any `SUNSKY_TOKENS` fragment at conf ≥ ~0.10** → **REJECT_RESIDUAL** (a weak-but-present mark).
3. Only PASS textured when the sensitive OCR returns **nothing sunsky-like** (genuinely clean texture).
This catches `'bnline am'`-class marks while leaving truly-clean textured finals untouched.

## 4. 禁止放松的安全规则 / Safety rules that must NOT be relaxed
- **Close the leak:** the mesh-cover and the Owner's full 100-image no-leak set must ALL `REJECT` → **100/100, 0 leaks**.
- **Do NOT re-introduce the misjudgment over-flag:** the 86 clean cleaned-finals the misjudgment patch
  corrected (e.g. img0007/0031/0040/0051) must **stay PASS**. This is the hard two-sided constraint.
- Keep the cardinal rule: no legible/visible watermark may PASS. Weak OCR ≠ no watermark on textured.

## 5. 回归测试图片 / Regression test images (two-sided)
- **MUST become/stay REJECT:** the Owner's frozen `benchmark/noleak_items.json` (100 known-watermarked,
  textured-weighted) → re-audit must be **100/100 REJECT**. Specifically the mesh-cover above.
- **MUST stay PASS (don't re-over-flag):** the misjudgment patch's corrected finals
  `job_p4/finals/{img0007,img0031,img0040,img0051}.jpg` (audited vs their originals).
- **Acceptance:** no-leak sweep = 0 leaks **AND** the 4 corrected FPs still PASS **AND** REJECT_RESIDUAL
  rate on the job_p4 500 stays ≤ ~5%. If closing the leak pushes any of the 86 back to REJECT, the
  threshold/sensitivity needs tuning — iterate, don't ship either failure mode.

---
**Note:** the Owner is independently closing the *other* half of this leak — improving finder recall so
mesh/flex/dark watermarks are caught and cleaned (never copied through). Defense in depth: finder catches
it → cleaned; if ever missed → this audit fix still REJECTs it. Both should hold before Phase 6.
