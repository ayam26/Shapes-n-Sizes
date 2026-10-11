# Checkpoint — 2026-10-11T06:05Z (elapsed ~0:48, ~552 min remaining; deadline 15:17:24Z)

Status: MVP complete (see artifacts/report.md §10). All milestones committed and pushed to origin/main.

Running processes created by this project:
- GX10: uvicorn backend, pid in ~/perception-lab/data/backend.pid (loopback :8765, PL_OFFLINE_GUARD=1). Stop: `scripts/gx10_backend.sh stop`.
- Mac: SSH tunnel `ssh -f -N -L 8765:127.0.0.1:8765 asus@100.104.225.90` (pgrep -f 8765:127.0.0.1:8765).
- Mac: vite dev server on :5173 (only needed for render_cases.mjs; started with nohup npx vite).
- Mac: caffeinate -t 43200 (keeps Mac awake).

Key artifacts: artifacts/report.md, artifacts/verification.json (60/60), artifacts/scene-checks.json (60/60), data/results/*.zip,
artifacts/screenshots/, artifacts/demo-script.md, artifacts/environment.md, artifacts/source-register.jsonl.

Frozen: experiments/prompts.json (rev 2, frozen 05:46:21Z), experiments/model-lock.json. Do not modify; held-out results are final.
Resume: `./scripts/start.sh`, then `python3 scripts/verify_artifacts.py`.
Optional remaining (not required for MVP): second-model exploratory comparison; higher-fidelity render study.
