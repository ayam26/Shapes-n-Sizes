# Decisions log

- 2026-10-11T05:17Z — Project location: `perception-lab/` inside repo `Shapes-n-Sizes` (empty repo apart from design doc). Design doc stays at repo root.
- 2026-10-11T05:17Z — Topology: Claude Code + browser rendering on the user's Mac (arm64, macOS 26.6); FastAPI backend, SQLite, artifact store and Ollama inference on the GX10 (aarch64, GB10). Backend binds 127.0.0.1 on the GX10; the Mac reaches it via an SSH tunnel over Tailscale. No public binding.
- 2026-10-11T05:17Z — Model candidate order: nemotron3:33b (Ollama reports `vision`), then qwen3.6:35b, then gemma4:26b. nemotron-3-super is text-only and is excluded.
- 2026-10-11T05:17Z — Git policy (user instruction): pull --rebase, commit, push to origin/main after each milestone. No force-push.
- 05:30Z — Output constrained with Ollama structured outputs (`format` = JSON schema with label enum), applied identically to both methods; parser stays strict (exact keys, allowed labels, optional single code fence, no repair).
- 05:30Z — Baseline prompt = design-doc text + one neutral sentence: "The target is the object inside the yellow rectangle." Structure variant appends one fixed sentence (thickness, perspective, support, occlusion).
- 05:35Z — Scene: depiction texture = the truck photographed from the saved ego-lane camera (`TEXTURE_CAMERA`) with an off-axis frustum covering the board, so the ego-lane view of the board has the truck's own perspective. Board = same geometry for depiction and plain control (frame, posts, braces).
- 05:35Z — Splits: development = `cam_a_ego_lane` (azimuth 9°), held-out = `cam_b_oblique` (azimuth 50°). The oblique view exposes board thickness/support much more; held-out is therefore NOT the same difficulty as development — report per split, never pool.
- 05:35Z — Marker = projected bbox of union(truck extents, board extents) + 8 px, yellow 3 px. Identical across target types for a camera, so it does not disclose class. Limitation: in the oblique view it is wide and includes background trees.
- 05:40Z — Dusk hemisphere intensity raised 0.45→0.75 before any model run (targets were nearly black when back-lit). Scene frozen after this change; manifest sha256 recorded in state/run.json.
- 05:40Z — Observability left `unchecked` for all cases: no independent reviewer; the builder is not independent.
