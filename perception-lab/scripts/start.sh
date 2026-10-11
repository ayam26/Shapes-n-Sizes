#!/usr/bin/env bash
# Single start command (run on the laptop):  ./scripts/start.sh
# 1. builds the frontend if needed, 2. syncs code to the GX10, 3. installs backend deps in a venv there,
# 4. starts the loopback-only backend on the GX10 (outbound guard ON), 5. opens an SSH tunnel,
# 6. prints the URL. Rendering runs in this laptop's browser; inference runs on the GX10 (Ollama).
set -euo pipefail
GX10=${PL_GX10:-asus@100.104.225.90}
PORT=${PL_PORT:-8765}
cd "$(dirname "$0")/.."

echo "[1/5] frontend build"
if [[ ! -d frontend/node_modules ]]; then (cd frontend && npm install --silent); fi
(cd frontend && npx tsc -p . && npx vite build --logLevel warn)

echo "[2/5] sync to $GX10:~/perception-lab"
ssh -o BatchMode=yes "$GX10" 'mkdir -p ~/perception-lab/frontend'
rsync -az --exclude __pycache__ backend experiments scripts tests "$GX10":perception-lab/
rsync -az frontend/dist "$GX10":perception-lab/frontend/

echo "[3/5] backend dependencies (venv on GX10)"
ssh -o BatchMode=yes "$GX10" 'cd ~/perception-lab && (test -d .venv || python3 -m venv .venv) && .venv/bin/pip -q install -r backend/requirements.txt' </dev/null

echo "[4/5] start backend + check Ollama"
ssh -o BatchMode=yes "$GX10" "cd ~/perception-lab && curl -sf 127.0.0.1:11434/api/tags >/dev/null || { echo 'Ollama not reachable on GX10 (start it: ollama serve)'; exit 1; }; PL_PORT=$PORT scripts/gx10_backend.sh start >/dev/null" </dev/null

echo "[5/5] tunnel localhost:$PORT -> GX10 loopback:$PORT"
if ! curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "$PORT:127.0.0.1:$PORT" "$GX10"
  sleep 1
fi
curl -s "http://127.0.0.1:$PORT/api/health" | python3 -c "import sys,json;h=json.load(sys.stdin);print('health:',{k:h[k] for k in ('provider_mode','configured_model','digest_matches_lock','offline_guard','prompts_frozen')})"
echo "Open http://127.0.0.1:$PORT  (first request after idle loads the model: ~12 s)"
