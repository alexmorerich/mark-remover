#!/usr/bin/env python3
"""Reconcile the re-clean results into the authoritative process manifest.

Keeps ONE manifest-compatible source of truth (governance: manifest compatibility):
overlays _reclean.jsonl onto _wm_process.jsonl, preserving the existing schema and
every original field. reclean paths get the canonical bbox + residual_after + a
reclean marker; restore paths get status='restored'. The pre-reclean manifest is
backed up first; the rewrite is atomic (temp + os.replace).
"""
import os, json, csv, shutil

A = "/Users/alexkou/Documents/github/b2bweb/content/products/assets"
PROC = os.path.join(A, "_wm_process.jsonl")
BAK = PROC + ".pre_reclean.bak"
RECLEAN = "/Users/alexkou/Documents/github/mark-remover/_reclean.jsonl"
ROUTE = "/Users/alexkou/Documents/github/mark-remover/reclean_routing.csv"


def main():
    action = {r["filename"]: r["action"] for r in csv.DictReader(open(ROUTE))}
    # latest re-clean record per path
    rc = {}
    for l in open(RECLEAN):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("path"):
            rc[r["path"]] = r

    if not os.path.exists(BAK):
        shutil.copy2(PROC, BAK)
        print("backed up process manifest ->", os.path.basename(BAK))

    out_lines, n_reclean, n_restore, n_untouched = [], 0, 0, 0
    for l in open(PROC):
        l = l.strip()
        if not l:
            continue
        rec = json.loads(l)
        p = rec.get("path")
        fn = os.path.basename(p) if p else None
        act = action.get(fn)
        rr = rc.get(p)
        if act == "reclean" and rr:
            rec["status"] = rr.get("status", rec.get("status"))
            rec["bbox_padded"] = rr.get("bbox_padded", rec.get("bbox_padded"))
            rec["residual_after"] = rr.get("residual_after", 0)
            rec["reclean"] = True
            rec["method"] = rr.get("method", "canonical_band")
            n_reclean += 1
        elif act == "restore":
            rec["status"] = "restored"
            rec["reclean"] = True
            rec["method"] = "restore"
            rec["residual_after"] = 0
            n_restore += 1
        else:
            n_untouched += 1
        out_lines.append(json.dumps(rec))

    tmp = PROC + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    os.replace(tmp, PROC)
    print(f"reconciled _wm_process.jsonl: reclean={n_reclean} restored={n_restore} untouched={n_untouched} total={len(out_lines)}")
    # sanity: schema still loads + status tally
    import collections
    tally = collections.Counter(json.loads(l).get("status") for l in open(PROC) if l.strip())
    print("status tally now:", dict(tally))


if __name__ == "__main__":
    main()
