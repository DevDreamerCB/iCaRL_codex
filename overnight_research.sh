#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/metrics"
LOG_FILE="$LOG_DIR/overnight_runner.log"
PID_FILE="$LOG_DIR/overnight_runner.pid"
HOURS="${1:-10}"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "overnight runner already active: pid=$OLD_PID"
    exit 0
  fi
fi

cd "$ROOT_DIR"
nohup python -u overnight_research.py "$HOURS" >> "$LOG_FILE" 2>&1 < /dev/null &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
sleep 2
if ! kill -0 "$NEW_PID" 2>/dev/null; then
  echo "overnight runner exited early"
  tail -n 40 "$LOG_FILE" 2>/dev/null || true
  exit 1
fi
echo "started overnight runner: pid=$NEW_PID"
echo "log: $LOG_FILE"
