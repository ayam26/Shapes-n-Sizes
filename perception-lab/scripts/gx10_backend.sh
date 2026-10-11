#!/usr/bin/env bash
# Run ON THE GX10. Manages the Perception Lab API (loopback-only) with a pidfile.
# Usage: scripts/gx10_backend.sh start|stop|restart|status   (PL_OFFLINE_GUARD=1 to enable outbound guard)
set -euo pipefail
cd "$(dirname "$0")/.."
PIDF=data/backend.pid; LOG=data/backend.log; PORT=${PL_PORT:-8765}
mkdir -p data
running() { [[ -f $PIDF ]] && kill -0 "$(cat $PIDF)" 2>/dev/null; }
case "${1:-status}" in
  start)
    if running; then echo "already running pid $(cat $PIDF)"; exit 0; fi
    ROOT=$PWD
    cd backend
    PL_ROOT="$ROOT" setsid nohup ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT" </dev/null >>"$ROOT/$LOG" 2>&1 &
    echo $! >"$ROOT/$PIDF"
    cd "$ROOT"
    for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break; sleep 0.25; done
    curl -s "http://127.0.0.1:$PORT/api/health"; echo ;;
  stop) if running; then kill "$(cat $PIDF)"; rm -f $PIDF; echo stopped; else echo "not running"; fi ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status) if running; then echo "running pid $(cat $PIDF)"; curl -s "http://127.0.0.1:$PORT/api/health"; echo; else echo "not running"; fi ;;
esac
