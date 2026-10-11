# Decisions log

- 2026-10-11T05:17Z — Project location: `perception-lab/` inside repo `Shapes-n-Sizes` (empty repo apart from design doc). Design doc stays at repo root.
- 2026-10-11T05:17Z — Topology: Claude Code + browser rendering on the user's Mac (arm64, macOS 26.6); FastAPI backend, SQLite, artifact store and Ollama inference on the GX10 (aarch64, GB10). Backend binds 127.0.0.1 on the GX10; the Mac reaches it via an SSH tunnel over Tailscale. No public binding.
- 2026-10-11T05:17Z — Model candidate order: nemotron3:33b (Ollama reports `vision`), then qwen3.6:35b, then gemma4:26b. nemotron-3-super is text-only and is excluded.
- 2026-10-11T05:17Z — Git policy (user instruction): pull --rebase, commit, push to origin/main after each milestone. No force-push.
