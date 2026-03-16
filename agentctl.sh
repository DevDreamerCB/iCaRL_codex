#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_DIR="$ROOT_DIR/metrics"
AGENT_PID_FILE="$METRICS_DIR/agent_loop.pid"
RUNNER_PID_FILE="$METRICS_DIR/autoresearch.pid"
AGENT_LOG="$METRICS_DIR/agent_loop.log"
RUNNER_LOG="$METRICS_DIR/autoresearch.log"
STATUS_MD="$METRICS_DIR/agent_status.md"

usage() {
  cat <<'EOF'
Usage:
  ./agentctl.sh start [hours]
  ./agentctl.sh start-runner [hours]
  ./agentctl.sh stop
  ./agentctl.sh stop-runner
  ./agentctl.sh status
  ./agentctl.sh logs

Commands:
  start         Start the agent controller loop.
  start-runner  Start the lower-level executor only.
  stop          Stop the agent controller loop.
  stop-runner   Stop the lower-level executor.
  status        Show pid, runtime, and current status markdown.
  logs          Tail both agent and runner logs.
EOF
}

is_alive() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

show_proc() {
  local label="$1"
  local pid_file="$2"
  if is_alive "$pid_file"; then
    ps -p "$(cat "$pid_file")" -o pid,etime,cmd | sed "1s/^/${label}\n/"
  else
    echo "${label}"
    echo "not running"
  fi
}

stop_proc() {
  local pid_file="$1"
  if is_alive "$pid_file"; then
    kill "$(cat "$pid_file")"
    echo "stopped pid $(cat "$pid_file")"
  else
    echo "not running"
  fi
  rm -f "$pid_file"
}

cmd="${1:-status}"

case "$cmd" in
  start)
    hours="${2:-12}"
    cd "$ROOT_DIR"
    ./run_agent_loop.sh "$hours"
    ;;
  start-runner)
    hours="${2:-12}"
    cd "$ROOT_DIR"
    ./run_autoresearch.sh "$hours"
    ;;
  stop)
    stop_proc "$AGENT_PID_FILE"
    ;;
  stop-runner)
    stop_proc "$RUNNER_PID_FILE"
    ;;
  status)
    cd "$ROOT_DIR"
    show_proc "Agent Controller" "$AGENT_PID_FILE"
    echo
    show_proc "Executor Runner" "$RUNNER_PID_FILE"
    echo
    if [[ -f "$STATUS_MD" ]]; then
      sed -n '1,120p' "$STATUS_MD"
    else
      echo "No agent status markdown yet: $STATUS_MD"
    fi
    ;;
  logs)
    cd "$ROOT_DIR"
    tail -n 40 "$AGENT_LOG" || true
    echo
    tail -n 40 "$RUNNER_LOG" || true
    ;;
  *)
    usage
    exit 1
    ;;
esac
