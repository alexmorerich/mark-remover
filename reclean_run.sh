#!/bin/bash
# Chunked, resumable re-clean runner (governance: REPAIR pass with inline AUDIT).
# Fresh process every CHUNK images bounds MPS memory creep (MPS wedged ~1100 imgs
# in the original run; CHUNK=400 stays well under). Resumable via skip-done in the
# manifest, so any crash/restart just continues. Writes state/reclean_progress.json.
set -u
MR="/Users/alexkou/Documents/github/mark-remover"
CSV="$MR/reclean_routing.csv"
DEVICE="${DEVICE:-mps}"
CHUNK="${CHUNK:-400}"
ACTION="${ACTION:-reclean}"
LOG="$MR/_reclean.jsonl"
REVIEW="$MR/_reclean_review.txt"
RUNLOG="$MR/_reclean_run.log"
STATE="$MR/state/reclean_progress.json"
cd "$MR" || exit 1
mkdir -p "$MR/state"

total=$(python3 -c "import csv;print(sum(1 for r in csv.DictReader(open('$CSV')) if r['action']=='$ACTION'))")
echo "=== RECLEAN START action=$ACTION device=$DEVICE chunk=$CHUNK total=$total $(date '+%F %T') ===" | tee -a "$RUNLOG"

while true; do
  done=$(python3 -c "
import json,os
p='$LOG'; a='$ACTION'; s=set()
if os.path.exists(p):
  for l in open(p):
    try:
      r=json.loads(l)
      if r.get('action')==a: s.add(r.get('path'))
    except Exception: pass
print(len(s))")
  ts=$(date '+%F %T')
  python3 -c "import json;json.dump({'action':'$ACTION','done':$done,'total':$total,'device':'$DEVICE','updated_at':'$ts'},open('$STATE','w'),indent=2)"
  echo "[$ts] progress action=$ACTION: $done/$total" | tee -a "$RUNLOG"
  if [ "$done" -ge "$total" ]; then
    echo "=== RECLEAN COMPLETE action=$ACTION $ts ===" | tee -a "$RUNLOG"
    break
  fi
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 reclean.py --run --csv "$CSV" --action "$ACTION" \
    --device "$DEVICE" --limit "$CHUNK" --apply --log "$LOG" --review "$REVIEW" >> "$RUNLOG" 2>&1 \
    || echo "[$(date '+%T')] chunk exited non-zero $?" | tee -a "$RUNLOG"
done
