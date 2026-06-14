# Audit Engine — Misjudgment Change Request

**From:** Owner Agent · **To:** Coder Agent (implements) → Audit Agent (regression-gates)
**Date:** 2026-06-14 · **Type:** decision **precision** fix (cut false REJECTs) · **File:** `audit.py`
**Process:** Owner finds → this request → Coder changes code → Audit runs benchmark/regression → **merge only if it passes**.

> This corrects a real **over-flag** (false REJECT). It is a **precision** fix — recall on real
> residual must NOT drop. A genuinely-leftover watermark must still REJECT.

---

## 1. 问题样例 / Problem examples
Phase-4 full-pipeline run (`job_p4`, 500 images, round-1, 296 decided):

| Audit decision | rate | vs Owner (Phase 3) |
|---|---|---|
| `REJECT_RESIDUAL_WATERMARK` | **32%** | Owner residual = **3.6%** → **~10× disagreement** |
| `REJECT_VISIBLE_PATCH` | **8%** | — |
| PASS (+minor) | 57% | gate needs >95% |

**Verified false positives** — the ORIGINAL clearly carries `sunsky-online.com`, the Owner's FINAL
has it **removed (clean)**, yet the Audit said `REJECT_RESIDUAL`:
`img0003, img0007, img0031, img0040, img0051` (compare `job_p4/originals/<id>.jpg` vs `job_p4/finals/<id>.jpg`).
All are **flex-cable / dark-part / camera-module / textured** surfaces. The `REJECT_VISIBLE_PATCH`
hits cluster on the same surfaces (ids in `job_p4/logs/audit.log`).

## 2. 当前误判 / Current misjudgment
- **`check_residual_watermark` template path (≈L360–383).** When OCR (`verify_clean`) reads **nothing**,
  the code still rejects on `verify_residue` template correlation + the light-stroke gate. On flex/cable/
  textured/dark backgrounds the matched filter correlates with **product structure**, and the light-stroke
  fraction passes → **false `REJECT_RESIDUAL`**. Root issue (known): template matching cannot separate clean
  from watermarked on this mark — clean and watermarked score alike — so **template-alone is unreliable on
  structured content**.
- **`check_repair_artifact`** similarly over-fires `REJECT_VISIBLE_PATCH` on textured surfaces: the inpaint
  over the (expected) watermark footprint reads as a "patch" against cable/texture.
- **Consequence:** ~43% of *clean* Owner output is rejected. Retries can't recover (re-cleaning a clean image
  → Audit still rejects → REJECT after 2 tries), so PASS can never reach 95%. The pipeline is blocked.

## 3. 期望行为 / Expected behavior
1. **Residual:** REJECT only on **legible OCR residual** (`verify_clean` reads a sunsky fragment) **OR** a
   template hit that is watermark-like **AND OCR-corroborated**. On `thin_flex_cable / dark_product_surface /
   complex_product_detail / metallic` ROI, **do not reject on template correlation alone** — require OCR
   corroboration there (template is unreliable on structure).
2. **Visible-patch:** don't fire when the candidate patch coincides with the **known watermark footprint** on a
   textured surface — that inpaint is expected.
3. **Calibration target:** on Owner-cleaned finals, the Audit's residual+patch reject-rate should drop toward the
   Owner's measured **~3.6%** (the two detectors should agree on clean output).

## 4. 禁止放松的安全规则 / Safety rules that must NOT be relaxed
- **Real residual must STILL REJECT.** A legible OR clearly-visible leftover watermark must be caught. Recall on
  real residual must not drop. This is a *false-positive* cut, not a sensitivity cut.
- Keep the **OCR residual path** (`verify_clean` composite + per-channel) fully intact — it is the *reliable*
  signal; the fix narrows the *template-only* path, it does not weaken OCR.
- Do **NOT** blanket-disable the residual or patch checks. Narrow the false positives on textured backgrounds;
  keep the true positives on clean/white/simple backgrounds where template is reliable.
- **Veto preserved:** a genuinely ambiguous *real* mark still → REJECT. No change may let a **legible** watermark PASS.

## 5. 回归测试图片 / Regression test images (two-sided — this is the crux)
- **MUST flip to PASS** (verified clean, currently wrong `REJECT_RESIDUAL`):
  `job_p4/finals/{img0003,img0007,img0031,img0040,img0051}.jpg` audited vs `job_p4/originals/<id>.jpg`.
- **MUST stay REJECT** (genuine residual — guards the fix from over-correcting): audit a watermarked image as
  its **own** final — build `[{"id":..,"original":<bk>,"final":<bk>}]` for ~10 `_wm_backup/` images (the "final"
  still carries the FULL watermark) → every one must `REJECT_RESIDUAL_WATERMARK`. Add any known half-clean
  (a surviving `.com` tail) → must REJECT.
- **MUST stay PASS** (already correct): the clean copies `img0533–img0535`.
- **Acceptance:** re-audit job_p4's 500 finals → `REJECT_RESIDUAL` ≤ ~5%, **every** genuine-residual regression
  case stays REJECT, **zero** legible-watermark case flips to PASS, and PASS climbs > 90%. If any genuine residual
  flips to PASS, the change is **rejected** — fix or revert.
