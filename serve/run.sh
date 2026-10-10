#!/usr/bin/env bash
# run.sh - start serve.sh in the background and wait until the engine answers.
#
#   sudo bash run.sh [profile.env] [--stop]
#
# The engine is detached (setsid + nohup) so it survives the shell that started
# it, its log gets a timestamped name plus a stable symlink, and liveness is
# decided by the pid file that serve.sh writes before its exec - not by pgrep.
# A single `pgrep -f "vllm serve"` right after the fork is not evidence: the
# process only appears at the end of serve.sh, and a supervisor that asks too
# early kills a model that is still loading.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-$HERE/awq-pp2.env}"
LOGDIR="${RUN_LOGDIR:-/srv/llm/logs}"
PIDFILE="${FLASHNEXT_PIDFILE:-/run/flashnext-engine.pid}"
PORT="${RUN_PORT:-8008}"
TIMEOUT="${RUN_TIMEOUT_S:-900}"
STEP="${RUN_STEP_S:-5}"
GRACE="${RUN_LIVENESS_GRACE_S:-60}"      # seconds of "no pid file yet" is normal
DEAD_SAMPLES="${RUN_DEAD_SAMPLES:-3}"    # consecutive empty samples after grace = dead
LOG="$LOGDIR/flashnext_$(date +%Y%m%d-%H%M%S).log"

engine_pid() { cat "$PIDFILE" 2>/dev/null; }

eng_alive() {
  local p
  p=$(engine_pid)
  if [ -n "$p" ]; then
    kill -0 "$p" 2>/dev/null
    return $?
  fi
  pgrep -f 'vllm serve' >/dev/null 2>&1
}

if [ "${1:-}" = "--stop" ] || [ "${2:-}" = "--stop" ]; then
  p=$(engine_pid)
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
    echo "=== stopping engine pid $p"
    kill "$p"
    for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -0 "$p" 2>/dev/null && kill -9 "$p"
  else
    echo "=== no engine in $PIDFILE"
  fi
  rm -f "$PIDFILE"
  exit 0
fi

[ -f "$PROFILE" ] || { echo "no profile: $PROFILE"; exit 1; }
mkdir -p "$LOGDIR" 2>/dev/null || true

if eng_alive; then
  echo "=== an engine is already running (pid $(engine_pid)); use --stop first"
  exit 1
fi

rm -f "$PIDFILE"
echo "=== starting: $PROFILE"
setsid nohup bash "$HERE/serve.sh" "$PROFILE" > "$LOG" 2>&1 < /dev/null &
ln -sfn "$LOG" "$LOGDIR/flashnext.log" 2>/dev/null || true
echo "  log: $LOG"

echo "=== waiting for /health (up to ${TIMEOUT} s; liveness: $PIDFILE, grace ${GRACE} s)"
dead=0
code=""
for ((i = 1; i <= TIMEOUT / STEP; i++)); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
          "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "  ready after $((i * STEP)) s: http://127.0.0.1:${PORT}/v1"
    break
  fi
  if (( i % 12 == 0 )); then
    echo "  [$((i * STEP)) s] $(grep -aE 'Error|Traceback|KV cache size|init engine' "$LOG" | tail -1 | cut -c1-160)"
  fi
  if eng_alive; then
    dead=0
    [ "$i" -eq 1 ] && echo "  [$((i * STEP)) s] engine is starting: pid $(engine_pid)"
  elif [ -n "$(engine_pid)" ]; then
    echo "  !!! the engine died: pid $(engine_pid) from $PIDFILE does not exist. Log tail:"
    tail -25 "$LOG"
    exit 1
  elif [ $((i * STEP)) -le "$GRACE" ]; then
    [ "$i" -eq 1 ] && echo "  [$((i * STEP)) s] no pid file yet - normal (grace ${GRACE} s)"
  else
    dead=$((dead + 1))
    if [ "$dead" -ge "$DEAD_SAMPLES" ]; then
      echo "  !!! no vLLM process and no pid file ($dead samples after ${GRACE} s). Log tail:"
      tail -25 "$LOG"
      exit 1
    fi
    echo "  [$((i * STEP)) s] engine is not in $PIDFILE nor in the process list ($dead/$DEAD_SAMPLES)"
  fi
  sleep "$STEP"
done

echo "=== /health: ${code:-no answer}"
if [ "$code" != "200" ]; then
  echo "=== log tail ==="
  tail -25 "$LOG"
  exit 1
fi
grep -aE 'GPU KV cache size|model weights take|Maximum concurrency|PLE non-PLE loader' "$LOG" | tail -4 || true
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader | sed 's/^/  /'