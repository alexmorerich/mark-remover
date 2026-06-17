#!/bin/bash
# Bulk watermark clean — CHUNKED single-worker, resumable, MEMORY-BOUNDED.
# Each chunk is a FRESH process (--limit CHUNK) that exits before MPS memory creep
# degrades throughput (degradation onset ~2000 imgs on this M4; chunk=500 stays in
# the fast regime), then the loop relaunches and resumes from the manifest. This is
# what stops speed collapsing to ~0.07 img/s over a single long-lived process.
# Completion = a chunk that exits 0 but adds no new manifest lines (nothing left).
# Device-parametrized: DEVICE=mps (fast) | cpu (stable fallback). WORKERS keep at 1.
set -u
MR=/Users/alexkou/Documents/github/mark-remover
ASSETS=/Users/alexkou/Documents/github/b2bweb/content/products/assets
LOG="$MR/_run_full.log"
SCAN_MANIFEST="$ASSETS/_wm_scan.jsonl"
PROC_MANIFEST="$ASSETS/_wm_process.jsonl"

unset WM_OCR_BASE_CAP
export WM_OCR_MAXPASS="${WM_OCR_MAXPASS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
DEVICE="${DEVICE:-mps}"
WORKERS="${WORKERS:-1}"
CHUNK="${CHUNK:-500}"

lines(){ wc -l < "$1" 2>/dev/null | tr -d ' '; }

echo "=== RUN start $(date) | maxpass=$WM_OCR_MAXPASS device=$DEVICE workers=$WORKERS chunk=$CHUNK ===" >> "$LOG"

run_phase(){            # $1=scan|process  $2=manifest  $3=extra args
  local phase="$1" manifest="$2" extra="$3" stall=0 b a rc
  while :; do
    b=$(lines "$manifest"); b=${b:-0}
    python3 -u "$MR/run_bulk.py" "$phase" --images-dir "$ASSETS" --workers "$WORKERS" --device "$DEVICE" --limit "$CHUNK" $extra >> "$LOG" 2>&1
    rc=$?
    a=$(lines "$manifest"); a=${a:-0}
    echo "=== $phase chunk: $b -> $a (rc=$rc) $(date) ===" >> "$LOG"
    if [ "$a" -gt "$b" ]; then
      stall=0                              # made progress -> fresh chunk
    elif [ "$rc" -eq 0 ]; then
      break                                # clean exit, nothing left = DONE
    else
      stall=$((stall+1))                   # crashed before any progress; retry
      echo "=== $phase crash no-progress (stall #$stall) ===" >> "$LOG"
      [ "$stall" -ge 10 ] && { echo "$phase STALLED — aborting" >> "$LOG"; return 1; }
    fi
  done
}

run_phase scan    "$SCAN_MANIFEST" ""      || exit 1
echo "=== SCAN COMPLETE $(date) ===" >> "$LOG"
run_phase process "$PROC_MANIFEST" "--yes" || exit 1
echo "=== ALL DONE $(date) ===" >> "$LOG"
