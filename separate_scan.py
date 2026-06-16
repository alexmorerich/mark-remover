#!/usr/bin/env python3
"""
separate_scan.py — fan the asset library into mark-status folders for per-bucket QA.

Non-destructive (copy or symlink, never move), idempotent, fully traced. Does NOT
re-scan and never touches a GPU — it is a pure file router driven entirely off two
artifacts the scanner already produced:

  * the live assets dir            (the product images)
  * the scanner ledger `_wm_scan.jsonl`  ({"path":..., "status":...} per scanned image)

Buckets (precedence: 01 rule -> 03 flagged -> 02 scanned-clean -> else quarantine):

  scan_split/01_iphone14plus_nomark/  is_skippable_model(name)  -> no mark, 100% by rule
                                      (iPhone 14+; excluded from scan, so NOT in ledger)
  scan_split/02_other_nomark/         in ledger, status == "clean"  -> scanned, no mark
  scan_split/03_watermarked/          in ledger, status == "flagged" -> feeds the clean pipeline
  scan_split/_unscanned_review/       neither skippable nor scanned-clean -> QUARANTINE

The quarantine is the safety net: an image that was never scanned, or whose scan
errored, is NEVER assumed clean — it lands here and trips a warning rather than
being published as no-mark. In the documented pipeline (skippables filtered before
scan, every other image scanned to a clean/flagged verdict) the quarantine ends
EMPTY and 01+02+03 == total.

Every routing decision is recorded in scan_split/scan_routing.csv with a blank
`suspected_miss` column that the Audit pass (scan_audit.py over buckets 01/02) fills
in — turning "trust the skip rule / trust the scanner" into a verified split.

CLI:
  python3 separate_scan.py --assets <library dir> --scan <library>/_wm_scan.jsonl
  python3 separate_scan.py --assets <library dir> --scan <ledger> --symlink   # zero extra disk
"""
import argparse, csv, json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- single source of truth -------------------------------------------------
# Reuse the scanner's OWN classification + enumeration so this router can never
# drift from what was actually scanned. run_bulk's CLI is guarded under __main__
# and v27_clean has only light top-level imports, so the import is side-effect
# free. If the heavy CV stack (cv2/numpy/PIL) can't be imported in this env, fall
# back to faithful inlined mirrors — this router decodes no images, so it must
# still run without them. Any fallback is announced on stderr so it's never silent.
try:
    from run_bulk import (is_skippable_model, _IPHONE_MODEL_RE,
                          discover_images, IMG_EXTS, SELF_DIRS)
    _RULE_SRC = "run_bulk (single source of truth)"
except Exception as _imp_err:  # pragma: no cover - defensive, env without cv2/numpy
    import re
    _RULE_SRC = f"inlined mirror (run_bulk import failed: {type(_imp_err).__name__})"
    SKIP_IPHONE_MIN = 14
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    SELF_DIRS = {"_wm_review", "_wm_backup"}
    _IPHONE_MODEL_RE = re.compile(r"iphone[\s_\-]*([0-9]{1,2})", re.IGNORECASE)

    def is_skippable_model(name):                       # mirror of run_bulk.py:69
        nums = [int(n) for n in _IPHONE_MODEL_RE.findall(name)]
        return bool(nums) and all(n >= SKIP_IPHONE_MIN for n in nums)

    def discover_images(images_dir):                    # mirror of run_bulk.py:100
        images_dir = os.path.abspath(images_dir)
        out = []
        for root, dirs, files in os.walk(images_dir):
            dirs[:] = [d for d in dirs if d not in SELF_DIRS]   # skip our scratch dirs
            for f in files:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    out.append(os.path.join(root, f))
        out.sort()
        return out

# rev ("_unscanned_review") must end empty in the happy path.
BUCKETS = {"01": "01_iphone14plus_nomark", "02": "02_other_nomark",
           "03": "03_watermarked", "rev": "_unscanned_review"}
# Exact, stable CSV schema — Audit fills `suspected_miss`; do not reorder/rename.
CSV_FIELDS = ["id", "bucket", "basis", "scan_status", "iphone_models",
              "suspected_miss", "source_path"]


def load_ledger(path):
    """abspath(image) -> status. Keyed by os.path.abspath (NOT realpath) because
    that is exactly how run_bulk's discover_images wrote the ledger paths — keying
    any other way would mismatch on symlinked roots. Robust to blank/garbage lines:
    an interrupted scan can leave a truncated final record, which we skip + count."""
    status, bad = {}, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                status[os.path.abspath(rec["path"])] = rec.get("status")
            except (json.JSONDecodeError, KeyError, TypeError):
                bad += 1
    return status, bad


def classify(name, abspath, status):
    """Route one image. Precedence per spec: 01 rule -> 03 flagged -> 02 clean ->
    quarantine. 'error' and missing both quarantine: an errored scan never produced
    a clean verdict, so per never-assume-clean it is not published as no-mark."""
    if is_skippable_model(name):
        return "01", "rule:iphone>=14"
    st = status.get(abspath)
    if st == "flagged":
        return "03", "scan:flagged"
    if st == "clean":
        return "02", "scan:clean"
    if st == "error":
        return "rev", "scan:error"         # scanned but errored -> quarantine, not clean
    return "rev", "unscanned"              # no ledger record -> never assume clean


def main():
    ap = argparse.ArgumentParser(description="Split the asset library into mark-status folders for QA.")
    ap.add_argument("--assets", required=True, help="root dir of product images (the scanned library)")
    ap.add_argument("--scan", required=True, help="scanner ledger _wm_scan.jsonl")
    ap.add_argument("--out", default="scan_split", help="output root (default: scan_split)")
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copy (zero extra disk)")
    args = ap.parse_args()

    if "inlined mirror" in _RULE_SRC:
        print(f"NOTE: skip rule / enumerator -> {_RULE_SRC}", file=sys.stderr)

    assets_abs = os.path.abspath(args.assets)
    out_abs = os.path.abspath(args.out)
    if not os.path.isdir(assets_abs):
        sys.exit(f"ERROR: --assets is not a directory: {assets_abs}")
    if not os.path.isfile(args.scan):
        sys.exit(f"ERROR: --scan ledger not found: {args.scan}")

    # Destructive-action guard: we rmtree(out) to rebuild. Refuse any layout where
    # that could delete the live library. (out *inside* assets is fine — we exclude it.)
    if out_abs == assets_abs:
        sys.exit("ERROR: --out must differ from --assets (refusing to delete the library)")
    if (assets_abs + os.sep).startswith(out_abs + os.sep):   # out is an ancestor of assets
        sys.exit(f"ERROR: --assets ({assets_abs}) is inside --out ({out_abs}); "
                 "rebuild would delete the library")

    # ---- load ledger ----
    status, bad_lines = load_ledger(args.scan)
    led_clean = sum(1 for v in status.values() if v == "clean")
    led_flag = sum(1 for v in status.values() if v == "flagged")
    led_err = sum(1 for v in status.values() if v == "error")
    led_other = len(status) - led_clean - led_flag - led_err

    # ---- idempotent rebuild ----
    out = Path(out_abs)
    if out.exists():
        prior = out / "scan_routing.csv"
        n_prior = sum(1 for _ in open(prior)) - 1 if prior.exists() else 0
        print(f"note: removing existing {args.out}/ for idempotent rebuild "
              f"(~{max(n_prior,0)} prior rows; any Audit suspected_miss stamps are discarded — "
              "re-run the audit pass after this)")
        shutil.rmtree(out)
    for d in BUCKETS.values():
        (out / d).mkdir(parents=True, exist_ok=True)

    # ---- enumerate assets (identical logic to the scanner's producer) ----
    assets = [p for p in discover_images(assets_abs)
              if not (os.path.abspath(p) + os.sep).startswith(out_abs + os.sep)]  # never re-sort our own output
    assets = sorted(set(os.path.abspath(p) for p in assets))

    # ---- route, copy/symlink, trace ----
    rows, tally = [], {k: 0 for k in BUCKETS}
    used_stems = set()           # globally-unique ids -> unique dest files, unambiguous for Audit
    matched_paths = set()        # ledger keys actually consumed (for orphan reconciliation)
    conflicts = []               # skippable-by-rule yet flagged in ledger (should be impossible)
    quarantine = {"unscanned": 0, "scan:error": 0}

    for p in assets:
        name = os.path.basename(p)
        st = status.get(p)
        bucket, basis = classify(name, p, status)
        if p in status:
            matched_paths.add(p)
        if bucket == "01" and st == "flagged":
            conflicts.append(name)        # surfaced below; spec precedence keeps it in 01
        if bucket == "rev":
            quarantine[basis] = quarantine.get(basis, 0) + 1

        # globally-unique destination name (id == dest stem == row key Audit maps files by)
        stem, suffix = Path(name).stem, Path(name).suffix
        cand, n = stem, 1
        while cand in used_stems:
            cand, n = f"{stem}__{n}", n + 1
        used_stems.add(cand)
        dst = out / BUCKETS[bucket] / f"{cand}{suffix}"

        if args.symlink:
            os.symlink(p, dst)            # p is absolute -> link resolves from any CWD
        else:
            shutil.copy2(p, dst)
        rows.append({"id": cand, "bucket": BUCKETS[bucket], "basis": basis,
                     "scan_status": st or "", "iphone_models": ",".join(_IPHONE_MODEL_RE.findall(name)),
                     "suspected_miss": "", "source_path": p})
        tally[bucket] += 1

    # ---- write trace CSV ----
    with open(out / "scan_routing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ---- reconciliation report ----
    total = len(assets)
    orphans = len(status) - len(matched_paths)     # ledger entries with no file under --assets
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assets_total": total,
        "buckets": {BUCKETS[k]: tally[k] for k in BUCKETS},
        "no_action_set_01_plus_02": tally["01"] + tally["02"],
        "clean_pipeline_03": tally["03"],
        "quarantine_breakdown": quarantine,
        "ledger": {"records": len(status), "clean": led_clean, "flagged": led_flag,
                   "error": led_err, "other": led_other, "bad_lines": bad_lines,
                   "orphans_not_under_assets": orphans},
        "mode": "symlink" if args.symlink else "copy",
        "rule_source": _RULE_SRC,
    }
    with open(out / "scan_split_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\ntotal {total} -> " + "  ".join(f"{BUCKETS[k]}={tally[k]}" for k in BUCKETS))
    print(f"ledger: {len(status)} records | clean={led_clean} flagged={led_flag} "
          f"error={led_err}" + (f" other={led_other}" if led_other else "")
          + (f" | bad_lines={bad_lines}" if bad_lines else ""))
    print(f"reconcile: 02 vs ledger-clean {tally['02']}/{led_clean}, "
          f"03 vs ledger-flagged {tally['03']}/{led_flag}, "
          f"01 (by-rule, not scanned) {tally['01']}")
    if orphans:
        print(f"note: {orphans} ledger path(s) have no file under --assets "
              "(moved/deleted, or --assets is not the scanned root)")

    # Hard invariant: every asset routed exactly once.
    assert sum(tally.values()) == total, "uncategorized images — routing dropped some assets!"
    print(f"OK: 01+02+03+quarantine == total assets ({total}); rows == {len(rows)}")

    if conflicts:
        print(f"NOTE: {len(conflicts)} skippable-by-rule image(s) are flagged in the ledger "
              f"(kept in 01 per precedence; first few: {conflicts[:5]}) — investigate, "
              "a by-rule 'clean' image carrying a mark is a contradiction.")

    if tally["rev"]:
        print(f"WARNING: {tally['rev']} image(s) -> {BUCKETS['rev']}/ "
              f"(unscanned={quarantine['unscanned']}, scan_error={quarantine['scan:error']}). "
              "NOT published as clean — these need a scan or a human glance.")
        if quarantine["unscanned"] and orphans == 0:
            print("         (unscanned with zero ledger orphans suggests --assets and the "
                  "ledger may not share the same root — check the paths.)")
    else:
        print(f"clean: {BUCKETS['rev']}/ is empty (every asset is skippable-by-rule or scanned).")


if __name__ == "__main__":
    main()
