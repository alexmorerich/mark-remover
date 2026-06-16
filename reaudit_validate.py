#!/usr/bin/env python3
"""Precision/recall gate for the leak-fix: re-audit all 182 Phase-4 PUBLISHED finals (originally
all PASS) with the patched audit. Expected: exactly 1 flips to REJECT (the leak); any other flip
is a NEW false reject to weigh."""
import sys, glob, os, json, time; sys.path.insert(0, ".")
import run_bulk as rb, audit
rb._init_worker("mps"); reader = rb._get_reader()
PUB, ORIG = "job_p4r/publish", "job_p4r/originals"
LEAK = "lcd-battery-iron-sheet-cover-set-with-sticker-screws-for-iphone-12-12-pro-1"
flips, n, dec = [], 0, {}; t0 = time.time()
for f in sorted(os.listdir(PUB)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")): continue
    iid = os.path.splitext(f)[0]
    op = glob.glob(os.path.join(ORIG, iid + ".*"))
    if not op: continue
    r = audit.audit_pair(op[0], os.path.join(PUB, f), reader=reader)
    n += 1; d = r["audit_decision"]; dec[d] = dec.get(d, 0) + 1
    if not r["publish_allowed"]:
        flips.append((iid, d, r.get("retry_hint")))
    if n % 40 == 0: print(f"[{n}] flips={len(flips)} {n/(time.time()-t0):.2f}/s", flush=True)
out = {"checked": n, "now_reject": len(flips), "decisions": dec,
       "leak_caught": any(x[0] == LEAK for x in flips),
       "flips": [{"id": x[0], "decision": x[1], "hint": x[2]} for x in flips]}
json.dump(out, open("job_p4r/reaudit_validation.json", "w"), indent=1)
print("\n=== RE-AUDIT 182 PUBLISHED (patched audit) ===")
print(f"checked={n} now_REJECT={len(flips)} (was 0)  leak_caught={out['leak_caught']}")
print("decisions:", dec)
for x in flips: print(f"  FLIP {x[0][:55]:55s} {x[1]} hint={bool(x[2])}")
