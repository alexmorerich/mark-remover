#!/usr/bin/env python3
"""state_monitor.py — persist long-run state to files (decoupled from the engine).

Reads the run_bulk manifests (the REAL checkpoint) + _run_full.log and derives:
  state/progress.json   (rewritten every +100 images and on stage change)
  state/heartbeat.json  (every 60s)
  state/summary.md      (every +1000 images)
  logs/run.log          (key events, appended)
  failures.csv          (error/residual samples, appended, deduped)

Non-destructive: only writes the above; never deletes manifests/backups/outputs.
The run resumes from the manifests, so after ANY interruption this monitor just
re-derives state on restart (no separate checkpoint needed). Safe writes:
JSON/markdown via temp+os.replace (atomic), log/csv via append.
"""
import json, os, time, csv, subprocess
from datetime import datetime

MR = "/Users/alexkou/Documents/github/mark-remover"
ASSETS = "/Users/alexkou/Documents/github/b2bweb/content/products/assets"
SCAN_MANIFEST = os.path.join(ASSETS, "_wm_scan.jsonl")
PROC_MANIFEST = os.path.join(ASSETS, "_wm_process.jsonl")
RUN_LOG = os.path.join(MR, "_run_full.log")
SCAN_TOTAL = 17149

STATE_DIR = os.path.join(MR, "state")
LOGS_DIR = os.path.join(MR, "logs")
PROGRESS = os.path.join(STATE_DIR, "progress.json")
HEARTBEAT = os.path.join(STATE_DIR, "heartbeat.json")
SUMMARY = os.path.join(STATE_DIR, "summary.md")
KEYLOG = os.path.join(LOGS_DIR, "run.log")
FAILURES = os.path.join(MR, "failures.csv")


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _append(path, text):
    with open(path, "a") as f:
        f.write(text)


def keylog(msg):
    _append(KEYLOG, f"[{now_iso()}] {msg}\n")


def read_manifest(path):
    """Full re-read (cheap, robust). Returns count, status-tally, last path,
    and a list of (path,status,detail) for error/residual records."""
    count, tally, last_path, fails = 0, {}, None, []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                count += 1
                s = r.get("status", "?")
                tally[s] = tally.get(s, 0) + 1
                if r.get("path"):
                    last_path = r["path"]
                if s in ("error", "residual"):
                    detail = r.get("err") or f"residual_after={r.get('residual_after','')}"
                    fails.append((r.get("path"), s, detail))
    return count, tally, last_path, fails


def detect_stage(proc_count=0):
    tail = ""
    if os.path.exists(RUN_LOG):
        try:
            with open(RUN_LOG, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 8000))
                tail = f.read().decode("utf-8", "ignore")
        except Exception:
            pass
    if "=== ALL DONE ===" in tail:
        return "done"
    # Robust cleaning signal: process manifest grew past the 2 smoke records, OR
    # 'SCAN COMPLETE' still in the log tail. proc_count is the reliable one —
    # 'SCAN COMPLETE' scrolls out of the 8KB tail once cleaning logs accumulate.
    if proc_count > 2 or "SCAN COMPLETE" in tail:
        return "cleaning"
    return "scanning"


def find_pid():
    try:
        out = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
        for ln in out.splitlines():
            if "run_bulk.py" in ln and "grep" not in ln and "Python" in ln:
                return int(ln.split()[0])
    except Exception:
        pass
    return None


def base(p):
    return os.path.basename(p) if p else None


def write_summary(stage, status, scan_count, scan_tally, proc_count, proc_tally, flagged_total, speed, eta_hours):
    pct = 100 * scan_count / SCAN_TOTAL if SCAN_TOTAL else 0
    t = [
        "# Watermark clean — run summary\n",
        f"_updated {now_iso()}_\n\n",
        f"**Status:** {status}  ·  **Stage:** {stage}\n\n",
        "## Scan\n",
        f"- scanned: **{scan_count} / {SCAN_TOTAL}** ({pct:.1f}%)\n",
        f"- flagged (watermarked): {scan_tally.get('flagged', 0)}\n",
        f"- clean: {scan_tally.get('clean', 0)}\n",
        f"- scan errors: {scan_tally.get('error', 0)}\n\n",
        "## Clean / inpaint\n",
        f"- processed: **{proc_count} / {flagged_total}** flagged\n",
        f"- cleaned: {proc_tally.get('cleaned', 0)}\n",
        f"- residual (still readable): {proc_tally.get('residual', 0)}\n",
        f"- no_mark_skip: {proc_tally.get('no_mark_skip', 0)}\n",
        f"- errors: {proc_tally.get('error', 0)}\n\n",
        "## Throughput\n",
        f"- speed: {speed:.2f} img/s\n",
        f"- eta: {eta_hours} h\n\n",
        "## Recovery (resume after any interruption)\n",
        f"- One command: `bash {MR}/resume.sh`\n",
        f"- Checkpoint = manifests at `{SCAN_MANIFEST}` and `{PROC_MANIFEST}` (resumable; already-done images are skipped)\n",
        f"- Originals backup: `{ASSETS}/_wm_backup/`\n",
        f"- Raw engine log: `{RUN_LOG}`\n",
    ]
    _atomic_write(SUMMARY, "".join(t))
    keylog(f"summary.md written (stage={stage} scan={scan_count}/{SCAN_TOTAL})")


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    seen = set()
    if os.path.exists(FAILURES):
        try:
            with open(FAILURES) as f:
                for i, row in enumerate(csv.reader(f)):
                    if i and len(row) >= 3:
                        seen.add(row[2])
        except Exception:
            pass
    else:
        _append(FAILURES, "timestamp,stage,path,status,detail\n")

    keylog("state_monitor started (pid %d)" % os.getpid())
    last_prog, last_summ, last_hb, last_stage, last_prog_time = -1, -1, 0.0, None, 0.0
    prev_count, prev_t = None, None

    while True:
        try:
            t = time.time()
            sc, st, slast, sfail = read_manifest(SCAN_MANIFEST)
            pc, pt, plast, pfail = read_manifest(PROC_MANIFEST)
            stage = detect_stage(pc)
            pid = find_pid()
            alive = pid is not None
            flagged_total = st.get("flagged", 0)

            if stage == "scanning":
                processed, total = sc, SCAN_TOTAL
                failed = st.get("error", 0)
                success = processed - failed
                last_file = slast
            else:
                processed, total = pc, max(flagged_total, pc)
                failed = pt.get("error", 0) + pt.get("residual", 0)
                success = pt.get("cleaned", 0) + pt.get("no_mark_skip", 0)
                last_file = plast or slast

            speed = 0.0
            if prev_count is not None and t > (prev_t or 0):
                speed = max(0.0, (processed - prev_count) / (t - prev_t))
            prev_count, prev_t = processed, t
            remaining = max(0, total - processed)
            eta_hours = round(remaining / speed / 3600, 2) if speed > 0.01 else None
            status = "done" if stage == "done" else ("running" if alive else "stalled")

            if t - last_hb >= 60:
                _atomic_write(HEARTBEAT, json.dumps({
                    "alive": bool(alive), "pid": pid,
                    "updated_at": now_iso(), "last_processed_file": base(last_file),
                }, indent=2))
                last_hb = t

            for (p, s, detail) in (sfail + pfail):
                if p and p not in seen:
                    seen.add(p)
                    d = str(detail).replace('"', "").replace("\n", " ")[:200]
                    _append(FAILURES, f'{now_iso()},{stage},{p},{s},"{d}"\n')
                    keylog(f"FAILURE {s}: {base(p)} ({d})")

            milestone = processed - last_prog >= 100 or stage != last_stage or status == "done"
            if milestone or t - last_prog_time >= 90:  # +100 imgs OR 90s timer (keep speed/eta fresh when slow)
                _atomic_write(PROGRESS, json.dumps({
                    "status": status, "processed": processed, "total": total,
                    "speed_img_per_sec": round(speed, 3), "eta_hours": eta_hours,
                    "success": success, "failed": failed, "current_stage": stage,
                    "last_file": base(last_file), "updated_at": now_iso(),
                    "scan_processed": sc, "scan_total": SCAN_TOTAL,
                    "flagged_total": flagged_total, "clean_processed": pc, "pid": pid,
                }, indent=2))
                last_prog_time = t
                if milestone:
                    keylog(f"progress stage={stage} {processed}/{total} ok={success} fail={failed} {speed:.2f}/s eta={eta_hours}h")
                    last_prog = processed
                    if stage != last_stage:
                        keylog(f"STAGE -> {stage}")
                        last_stage = stage

            if processed - last_summ >= 1000 or status == "done":
                write_summary(stage, status, sc, st, pc, pt, flagged_total, speed, eta_hours)
                last_summ = processed

            if status == "done":
                keylog("ALL DONE — final state written, monitor exiting")
                break
        except Exception as e:
            try:
                keylog(f"monitor cycle error: {type(e).__name__}: {e}")
            except Exception:
                pass
        time.sleep(15)


if __name__ == "__main__":
    main()
