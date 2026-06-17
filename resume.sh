#!/bin/bash
# resume.sh — one-command recovery after ANY interruption (reboot, sleep, crash).
# Resumes from the manifest checkpoint: already-scanned/cleaned images are skipped,
# nothing is re-done, no output is deleted. Starts BOTH the engine and the monitor.
# Usage:  bash resume.sh            (MPS fast path; falls back per watchdog)
#         DEVICE=cpu bash resume.sh (stable CPU path if Metal is wedged)
MR=/Users/alexkou/Documents/github/mark-remover
mkdir -p "$MR/logs"
# stop any stale instances first (never touches manifests/backups)
pkill -9 -f state_monitor.py 2>/dev/null
pkill -9 -f run_overnight.sh 2>/dev/null
pkill -9 -f run_bulk.py 2>/dev/null
sleep 1

nohup python3 -u "$MR/state_monitor.py" >> "$MR/logs/monitor.out" 2>&1 &
echo "monitor started (pid $!)"

DEVICE="${DEVICE:-mps}" WORKERS=1 WM_OCR_MAXPASS="${WM_OCR_MAXPASS:-1}" \
  nohup bash "$MR/run_overnight.sh" >> "$MR/logs/run_wrapper.out" 2>&1 &
echo "run started (device=${DEVICE:-mps}, pid $!)"
echo "watch: tail -f $MR/state/progress.json ; cat $MR/state/heartbeat.json"
