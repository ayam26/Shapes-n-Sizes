export interface Health {
  app: string; fixture_mode: boolean; provider_mode: string; configured_model: string; offline_guard: boolean;
  worker_alive: boolean; queue_depth: number; prompts_frozen: boolean; lock_frozen: boolean;
  provider_status: string; actual_model_digest?: string; digest_matches_lock?: boolean; model_config_hash: string;
  rendering: string; inference: string;
}

export interface Frac { n: number; d: number; pct: number | null }
export interface Summary {
  attempted: number; valid_responses: number; operational_failures: number; invalid_output: number; timeout: number;
  provider_error: number; abstentions: number; correct_over_attempted: Frac; accuracy_answered: Frac; coverage: Frac;
  false_vehicle_on_depictions: Frac; vehicle_recall: Frac; control_correct: Frac;
  outcomes: Record<string, number>; latency_ms: { n: number; p50: number | null; p95: number | null };
}
export interface Paired {
  both_correct: number; baseline_only: number; variant_only: number; neither: number; unresolved: number;
  pairs: number; hash_mismatch: number; cases: Record<string, string>;
}
export interface Metrics {
  by_split_method: Record<string, Record<string, Summary>>;
  paired_by_split: Record<string, Paired>;
}

export interface RunRow {
  run_id: string; run_key: string | null; label: string | null; created_at: string; status: string;
  provider_mode: string; offline_guard: number; use_cache: number; counts: Record<string, number>;
}
export interface RunResult {
  result_id: number; case_id: string; method_id: string; status: string; label: string | null; outcome: string | null;
  elapsed_ms: number | null; origin: string | null; split: string; truth: string; family: string;
  observability: string; image_sha256: string; conditions: string;
}
export interface RunDetail extends RunRow {
  total: number; results: RunResult[]; error_summary: { case_id: string; method_id: string; status: string; error: string }[];
  model_config: { lock: Record<string, unknown>; methods: Record<string, { prompt: string; prompt_hash: string }>;
    prompts_frozen: boolean; lock_frozen: boolean; lock_hash: string };
}
export interface CaseResult {
  result_id: number; run_id: string; run_label: string | null; provider_mode: string; method_id: string; status: string;
  label: string | null; outcome: string | null; parsed: string | null; raw_output: string | null; parse_error: string | null;
  elapsed_ms: number | null; origin: string | null; prompt_hash: string | null; image_sha256: string | null;
  model_id: string | null; model_digest: string | null; quantization: string | null; runtime_version: string | null;
  options: string | null; provider_meta: string | null; queued_at: string; started_at: string | null; finished_at: string | null;
}
export interface CaseDetail {
  case_id: string; split: string; family: string; truth: string; observability: string; image_sha256: string;
  width: number; height: number; renderer_version: string; created_at: string; scene_spec: Record<string, unknown>;
  conditions: Record<string, unknown>; results: CaseResult[];
}

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) throw new Error(`${init?.method ?? "GET"} ${path} → ${r.status}: ${(await r.text()).slice(0, 300)}`);
  return r.json() as Promise<T>;
}

export interface ModelLock { lock_file: string; model_id: string; digest: string; quantization: string;
  parameter_size: string; frozen: boolean; headline: boolean }
export interface ModelsInfo { locks: ModelLock[]; loaded: { model?: string; size_bytes?: number; vram_bytes?: number; error?: string }[] }

export const api = {
  models: () => j<ModelsInfo>("/api/models"),
  health: () => j<Health>("/api/health"),
  runs: () => j<RunRow[]>("/api/runs"),
  run: (id: string) => j<RunDetail>(`/api/runs/${id}`),
  metrics: (id: string) => j<Metrics>(`/api/metrics?run_id=${encodeURIComponent(id)}`),
  case: (id: string) => j<CaseDetail>(`/api/cases/${id}`),
  imageUrl: (id: string) => `/api/cases/${id}/image`,
  exportUrl: (id: string, fmt: string) => `/api/export/${id}?fmt=${fmt}`,
  async createCase(meta: object, png: Blob) {
    const fd = new FormData();
    fd.append("meta", JSON.stringify(meta));
    fd.append("image", png, "input.png");
    return j<{ case_id: string; image_sha256: string; existing: boolean }>("/api/cases", { method: "POST", body: fd });
  },
  createRun: (body: object) =>
    j<{ run_id: string }>("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
};

export const METHODS = ["rgb_baseline", "rgb_structure_prompt"] as const;
export const METHOD_LABEL: Record<string, string> = {
  rgb_baseline: "RGB baseline",
  rgb_structure_prompt: "RGB + structure prompt",
  rgb_scene_context: "RGB + scene-context prompt",
  rgb_two_step: "RGB + two-step decision",
  rgb_context_two_step: "RGB + scene-context + two-step",
  rgb_think: "RGB + reasoning mode",
};
