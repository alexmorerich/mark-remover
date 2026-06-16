#!/usr/bin/env python3
"""orchestrator.py — deterministic workflow controller for ClearMark.

NOT an AI agent: a plain Python state machine. Owner produces, Audit validates, the
Orchestrator decides. Agents never talk directly — only through files
(owner_manifest.jsonl, audit_results.jsonl, state.json). Audit has veto authority; the
Orchestrator owns retry + publish + reject and is fully resumable from state.json.

  NEW → OWNER_RUNNING → AUDIT_RUNNING → PASS | RETRY | REJECT      (no other transitions)

Per batch:
  job_xxx/{originals,finals,publish,rejected,retry,masks,logs}/
          owner_manifest.jsonl  audit_results.jsonl  audit_feedback.jsonl  state.json

Usage:
  python3 orchestrator.py --job job_001 --device mps
"""
import argparse, json, os, shutil, subprocess, sys
from collections import Counter
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_RETRY = 2
SUBDIRS = ("originals", "finals", "publish", "rejected", "retry", "masks", "logs")
NEW, OWNER_RUNNING, AUDIT_RUNNING, PASS, RETRY, REJECT = (
    "NEW", "OWNER_RUNNING", "AUDIT_RUNNING", "PASS", "RETRY", "REJECT")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

# audit.py decision (REJECT_*) → Owner feedback verdict (its retry vocab)
AUDIT2FAIL = {
    "REJECT_RESIDUAL_WATERMARK": "FAIL_RESIDUAL_WATERMARK",
    "REJECT_VISIBLE_PATCH": "FAIL_VISIBLE_PATCH",
    "REJECT_PRODUCT_DAMAGE": "FAIL_PRODUCT_DAMAGE",
    "REJECT_PROTECTED_TEXT_DAMAGE": "FAIL_PROTECTED_TEXT_DAMAGE",
    "REJECT_UNNATURAL_IMAGE": "FAIL_VISIBLE_PATCH",
    "REJECT_UNCERTAIN": "FAIL_UNCERTAIN",
}
RETRYABLE_ACTIONS = {"retry_repair", "try_cover"}      # else → reject


def _now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def jp(job, *p):
    return os.path.join(job, *p)


def _load_state(job):
    p = jp(job, "state.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _save_state(job, st):
    st["updated"] = _now()
    tmp = jp(job, "state.json.tmp")
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, jp(job, "state.json"))             # atomic → resumable


def _read_jsonl_latest(path):
    """id → latest record (last line wins)."""
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("id"):
                    out[r["id"]] = r
            except Exception:
                pass
    return out


def init_job(job):
    for d in SUBDIRS:
        os.makedirs(jp(job, d), exist_ok=True)
    st = _load_state(job)
    if st is None:
        imgs = {}
        for f in sorted(os.listdir(jp(job, "originals"))):
            if f.lower().endswith(IMG_EXT):
                imgs[os.path.splitext(f)[0]] = {
                    "file": f, "state": NEW, "retry_count": 0,
                    "owner_status": None, "audit_decision": None}
        st = {"round": 0, "max_retry": MAX_RETRY, "images": imgs, "updated": _now()}
        _save_state(job, st)
    return st


def _run(cmd, logfile):
    with open(logfile, "a") as lf:
        lf.write(f"\n# {_now()} $ {' '.join(cmd)}\n")
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False).returncode


def run_owner(job, device, feedback):
    cmd = [sys.executable, jp(HERE, "owner_agent.py"), "--job", job, "--device", device]
    if feedback:
        cmd.append("--feedback")
    _run(cmd, jp(job, "logs", "owner.log"))


def run_audit(job, items, device):
    items_path = jp(job, "logs", "audit_items.json")
    json.dump(items, open(items_path, "w"))
    _run([sys.executable, jp(HERE, "audit.py"), "--manifest", items_path, "--out-dir", job],
         jp(job, "logs", "audit.log"))


def _publish(job, v):
    src = jp(job, "finals", v["file"])
    if os.path.exists(src):
        shutil.copy2(src, jp(job, "publish", v["file"]))


def _reject_file(job, v):
    src = jp(job, "originals", v["file"])               # keep the original in rejected/
    if os.path.exists(src):
        shutil.copy2(src, jp(job, "rejected", v["file"]))


def orchestrate(job, device):
    st = init_job(job)
    for v in st["images"].values():                     # resume: redo any transient (crashed) state
        if v["state"] in (OWNER_RUNNING, AUDIT_RUNNING):
            v["state"] = NEW
    _save_state(job, st)
    while True:
        imgs = st["images"]
        pending = [i for i, v in imgs.items() if v["state"] in (NEW, RETRY)]
        if not pending:
            break
        st["round"] += 1
        is_retry = any(imgs[i]["state"] == RETRY for i in pending)
        print(f"[round {st['round']}] {len(pending)} pending ({'retry' if is_retry else 'fresh'})", flush=True)

        # ── OWNER produces ──
        for i in pending:
            imgs[i]["state"] = OWNER_RUNNING
        _save_state(job, st)
        owner = {}
        for attempt in range(1, 21):                    # owner is resumable; relaunch if it died mid-batch
            run_owner(job, device, feedback=is_retry)   # (MPS can crash ~every 1100 imgs — don't false-reject)
            owner = _read_jsonl_latest(jp(job, "owner_manifest.jsonl"))
            if all(i in owner for i in pending):
                break
            print(f"  owner incomplete {sum(i in owner for i in pending)}/{len(pending)} — relaunch {attempt}", flush=True)

        to_audit = []
        for i in pending:
            rec = owner.get(i)
            if rec is None:
                imgs[i].update(state=REJECT, owner_status="failed"); _reject_file(job, imgs[i]); continue
            imgs[i]["owner_status"] = rec.get("owner_status")
            if rec.get("final") and rec.get("owner_status") in ("cleaned", "copied_no_watermark"):
                imgs[i]["state"] = AUDIT_RUNNING
                to_audit.append({"id": i, "original": jp(job, rec["original"]), "final": jp(job, rec["final"])})
            else:                                           # owner auto_rejected / failed
                imgs[i]["state"] = REJECT
                _reject_file(job, imgs[i])
        _save_state(job, st)

        # ── AUDIT validates (veto authority) ──
        if to_audit:
            audit = {}
            for attempt in range(1, 21):                # audit resumable per round; relaunch if it died mid-batch
                run_audit(job, to_audit, device)
                audit = _read_jsonl_latest(jp(job, "audit_results.jsonl"))
                if all(it["id"] in audit for it in to_audit):
                    break
                print(f"  audit incomplete {sum(it['id'] in audit for it in to_audit)}/{len(to_audit)} "
                      f"— relaunch {attempt}", flush=True)
            else:                                       # still incomplete → HALT, never mass-reject
                print("ERROR: audit could not complete after retries (see logs/audit.log). Halting — "
                      "items stay AUDIT_RUNNING and re-audit next run. Infra failure must not reject.", flush=True)
                _save_state(job, st)
                return report(job, st)
            feedback = []
            for it in to_audit:
                i = it["id"]
                ar = audit.get(i, {})
                dec = ar.get("audit_decision", "REJECT_UNCERTAIN")
                action = ar.get("recommended_next_action", "auto_reject")
                imgs[i]["audit_decision"] = dec
                if ar.get("publish_allowed"):
                    imgs[i]["state"] = PASS
                    _publish(job, imgs[i])
                else:                                       # FAIL → Orchestrator decides retry vs reject
                    imgs[i]["retry_count"] += 1
                    if action in RETRYABLE_ACTIONS and imgs[i]["retry_count"] <= st["max_retry"]:
                        imgs[i]["state"] = RETRY
                        fl = {"id": i, "verdict": AUDIT2FAIL.get(dec, "FAIL_UNCERTAIN")}
                        hint = ar.get("retry_hint")          # additive; Audit emits it ONLY on retry_repair
                        if hint and hint.get("box"):         # tell the Owner WHERE the mark still is
                            imgs[i]["retry_box"] = [int(v) for v in hint["box"]]  # stash on state (resumable)
                            fl["retry_box"] = imgs[i]["retry_box"]                # rides audit_feedback.jsonl → Owner
                        feedback.append(fl)
                    else:
                        imgs[i]["state"] = REJECT
                        _reject_file(job, imgs[i])
            with open(jp(job, "audit_feedback.jsonl"), "w") as f:   # owner reads this next round
                for fl in feedback:
                    f.write(json.dumps(fl) + "\n")
            _save_state(job, st)

    return report(job, st)


def report(job, st):
    c = Counter(v["state"] for v in st["images"].values())
    n = len(st["images"])
    pub, rej = c.get(PASS, 0), c.get(REJECT, 0)
    rep = {"job": os.path.basename(os.path.abspath(job)), "total": n,
           "published": pub, "rejected": rej,
           "publish_rate": round(pub / n, 4) if n else 0.0,
           "reject_rate": round(rej / n, 4) if n else 0.0,
           "rounds": st["round"], "states": dict(c), "updated": _now()}
    json.dump(rep, open(jp(job, "logs", "orchestrator_report.json"), "w"), indent=2)
    print("ORCHESTRATOR DONE:", json.dumps(rep))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="batch dir (contains originals/)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    orchestrate(args.job, args.device)


if __name__ == "__main__":
    main()
