# Owner Agent Workflow — Watermark Removal Production Pipeline

_Authoritative spec for the production-side Owner Agent (Alex, 2026-06-14). Implemented by
`owner_agent.py`. The Owner **produces** publish-safe candidate finals; it never publishes —
final approval belongs to the Audit Agent._

## Goal
Process product images and produce publish-safe final images. The Owner is responsible for:
1. Detecting possible SUNSKY-style watermarks
2. Repairing or covering the watermark
3. Producing final images
4. Writing a simple manifest for the Audit Agent
5. Retrying failed images based on Audit feedback

**Core principle — do not over-engineer.** The goal is not to prove detection accuracy; it
is to generate clean final images that Audit can pass.

```
original image → detect → repair / cover → save final → write manifest → send to Audit
                                                                              ↓
                                              PASS → publish   FAIL → retry / auto_reject
```

## Folder structure (per job)
```
job_xxx/
  originals/   finals/   masks/   rejected/   logs/
  owner_manifest.jsonl   audit_feedback.jsonl
```
* Never modify `originals/`. Write finals to `finals/`, masks to `masks/`, unrecoverable to
  `rejected/`. One manifest line per image.

## Steps
1. **Load** — readable, RGB-convertible, reasonable resolution, not corrupt, not already
   processed. Unreadable ⇒ `auto_rejected` (`unreadable_image`).
2. **Detect** (recall-oriented): sunsky / sunsky-online / .com / dot-chain remnants /
   low-alpha gray text / repeated horizontal pattern / center placement. Classify:
   `watermark_found · uncertain_watermark · no_watermark · unsuitable`.
3. **Choose repair strategy** by where the mark sits:
   * **A. plain/near-white** → background clone / soft inpaint / clean cover (no gray block,
     no rectangle, no texture mismatch)
   * **B. simple light texture** → texture-aware inpaint (preserve texture, no smooth blob)
   * **C. product surface** → stroke-level repair (no large rectangle; don't destroy edges,
     labels, holes, camera rings, connectors, cables)
   * **D. thin flex cable / complex detail** → very conservative; if repair may damage →
     `auto_reject`
   * **E. product text / labels / QR / serial** → `auto_reject` (never remove product text)
4. **Generate candidate final** — reject internally if the result shows residual watermark,
   gray patch, rectangle cover, broken cable/edge, blurred label, damaged text, colour
   mismatch, AI smear, or a flattened metallic/glass surface.
5. **Save** — success → `finals/`; no watermark → copy original → `finals/`; failure →
   `rejected/`.
6. **Owner manifest** (`owner_manifest.jsonl`, one line/image):
   ```json
   {"id":"image001","original":"originals/image001.jpg","final":"finals/image001.jpg",
    "owner_status":"cleaned","method":"repair","watermark_status":"watermark_found","notes":""}
   ```
   * `owner_status` ∈ `cleaned · copied_no_watermark · auto_rejected · failed`
   * `method` ∈ `repair · cover · copy · reject`
   * `watermark_status` ∈ `watermark_found · uncertain_watermark · no_watermark · unsuitable`
7. **Send to Audit** — provide `originals/`, `finals/`, `owner_manifest.jsonl`. The Owner must
   not publish.
8. **Handle Audit feedback** (`audit_feedback.jsonl`, `{id, verdict}`):
   * `FAIL_RESIDUAL_WATERMARK` → stronger repair → cover → else `auto_reject`
   * `FAIL_VISIBLE_PATCH` → remove patch / softer repair / better texture-colour match → else `auto_reject`
   * `FAIL_PRODUCT_DAMAGE` → retry only if safely restorable, else `auto_reject`
   * `FAIL_PROTECTED_TEXT_DAMAGE` → `auto_reject` (don't retry aggressively)
   * `FAIL_UNCERTAIN` → one conservative retry, else `auto_reject`
9. **Retry limit = 2**, then `auto_reject`. Never loop forever.

## Final decision rules
Owner may output only `cleaned · copied_no_watermark · auto_rejected · failed`. Publishing is
allowed only when Audit says `PASS`. The Owner never marks an image publish-safe by itself.

## Non-negotiable safety — reject when uncertain
Reject if: the watermark overlaps important product text; repair breaks product structure;
repair creates a visible gray block or obvious rectangle; a cable/connector/label/QR/serial/
screw-hole/edge/camera-ring/outline is damaged; the result looks unnatural; or you are not
sure the image is safe. **When uncertain, reject instead of publishing.**

---
**Implementation:** `owner_agent.py` — `run_job()` (Steps 1–6) and `handle_feedback()`
(Steps 8–9), reusing `logo_finder.find()` for detection and `run_bulk._lama_crop_inpaint`
for repair/cover. Detection→`watermark_status`, `roi_class`→strategy, internal `find()`
re-check gates each final, retries are bounded at 2.
