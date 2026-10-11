import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  api, CaseDetail, CaseResult, Health, METHOD_LABEL, METHODS, Metrics, RunDetail, RunRow, Summary, Frac,
} from "./api";
import { BuiltScene, HELPER_LAYER, buildScene } from "./scene/build";
import { renderSensor } from "./scene/capture";
import {
  APPEARANCE_LABELS, Appearance, CAMERA_PRESETS, FAMILY, Lighting, RENDERER_VERSION, SLOT, TARGET_LABELS, TRUTH,
  TargetType, makeSpec,
} from "./scene/spec";
import "./styles.css";

// ---------- small presentational helpers ----------
const LABEL_TEXT: Record<string, string> = {
  physical_vehicle: "physical_vehicle", flat_depiction: "flat_depiction", other: "other", uncertain: "uncertain",
};

function LabelChip({ label }: { label: string | null }) {
  if (!label) return <span className="chip chip-none">no label</span>;
  return <span className={`chip lab-${label}`}>{LABEL_TEXT[label] ?? label}</span>;
}

function OutcomeChip({ outcome }: { outcome: string | null }) {
  if (!outcome) return <span className="chip chip-none">pending</span>;
  const ok = outcome === "correct";
  const cls = ok ? "out-ok" : outcome === "abstention" ? "out-abst" :
    ["invalid_output", "timeout", "provider_error"].includes(outcome) ? "out-op" : "out-bad";
  return <span className={`chip ${cls}`}>{ok ? "✓ correct" : outcome === "abstention" ? "◌ abstention" : `✗ ${outcome}`}</span>;
}

function StatusChip({ status }: { status: string }) {
  return <span className={`chip st-${status}`}>{status}</span>;
}

const fmtFrac = (f: Frac) => (f.d === 0 ? "–" : `${f.n}/${f.d} (${f.pct}%)`);
const ms = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v)} ms`);
const short = (h: string | null | undefined, n = 12) => (h ? h.slice(0, n) : "–");

// ---------- scene controls ----------
interface Controls {
  target: TargetType; appearance: Appearance; preset: string; azimuth: number; distance: number;
  lighting: Lighting; occluder: boolean;
}
const DEFAULTS: Controls = {
  target: "truck", appearance: "white_livery", preset: "cam_a_ego_lane", azimuth: 9, distance: 22.3,
  lighting: "day", occluder: false,
};

function ObserverView({ built }: { built: BuiltScene }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const st = useRef<{ r: THREE.WebGLRenderer; cam: THREE.PerspectiveCamera; controls: OrbitControls } | null>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const r = new THREE.WebGLRenderer({ canvas, antialias: true });
    r.shadowMap.enabled = true;
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const cam = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    cam.position.set(SLOT.x + 26, 16, SLOT.z + 20);
    cam.layers.enable(HELPER_LAYER);
    const controls = new OrbitControls(cam, canvas);
    controls.target.set(SLOT.x, 1.5, SLOT.z);
    controls.update();
    st.current = { r, cam, controls };
    return () => { controls.dispose(); r.dispose(); };
  }, []);
  useEffect(() => {
    const s = st.current!;
    const canvas = ref.current!;
    const render = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (w && h) {
        s.r.setSize(w, h, false);
        s.cam.aspect = w / h;
        s.cam.updateProjectionMatrix();
      }
      s.r.render(built.scene, s.cam);
    };
    s.controls.addEventListener("change", render);
    const ro = new ResizeObserver(render);
    ro.observe(canvas);
    render();
    return () => { s.controls.removeEventListener("change", render); ro.disconnect(); };
  }, [built]);
  return <canvas ref={ref} className="observer" aria-label="Observer 3D view; drag to orbit, scroll to zoom" tabIndex={0} />;
}

function SensorView({ built }: { built: BuiltScene }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => { renderSensor(built, 512, ref.current!); }, [built]);
  return <canvas ref={ref} className="sensor" width={512} height={512}
    aria-label="Sensor camera preview: exactly the image that will be sent to the model" />;
}

// ---------- analysis state ----------
type Phase = "idle" | "capturing" | "uploading" | "queued" | "running" | "completed" | "error";
interface Analysis { phase: Phase; caseId?: string; runId?: string; error?: string; progress?: Record<string, string>; sha?: string }

// ---------- evidence panel ----------
function EvidencePanel({ caseId, refresh }: { caseId: string | null; refresh: number }) {
  const [c, setC] = useState<CaseDetail | null>(null);
  const [runSel, setRunSel] = useState<string | null>(null);
  const [runs, setRuns] = useState<Record<string, RunDetail>>({});
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!caseId) return;
    let live = true;
    api.case(caseId).then(async (d) => {
      if (!live) return;
      setC(d);
      setErr(null);
      const ids = [...new Set(d.results.map((r) => r.run_id))];
      setRunSel(ids[0] ?? null);
      const loaded: Record<string, RunDetail> = {};
      for (const id of ids) loaded[id] = await api.run(id);
      if (live) setRuns(loaded);
    }).catch((e) => live && setErr(String(e)));
    return () => { live = false; };
  }, [caseId, refresh]);
  if (!caseId) return <p className="muted">Click <b>Analyze</b> or pick a case from the gallery to inspect its evidence.</p>;
  if (err) return <p className="error" role="alert">{err}</p>;
  if (!c) return <p className="muted">Loading case…</p>;
  const runIds = [...new Set(c.results.map((r) => r.run_id))];
  const shown = c.results.filter((r) => r.run_id === runSel);
  const byMethod: Record<string, CaseResult> = {};
  for (const r of shown) byMethod[r.method_id] ??= r;
  const hashes = new Set(shown.map((r) => r.image_sha256).filter(Boolean));
  const run = runSel ? runs[runSel] : undefined;
  return (
    <div className="evidence">
      <div className="truthbox">
        <div><span className="k">Ground truth (simulator, never sent to the model)</span><br />
          <LabelChip label={c.truth} /> <span className="muted">family {c.family}</span></div>
        <div className="muted small">case <code>{c.case_id}</code> · split <b>{c.split}</b> · observability {c.observability}<br />
          input sha256 <code>{short(c.image_sha256, 16)}</code> · {c.width}×{c.height} · {c.renderer_version}</div>
      </div>
      {runIds.length > 1 && (
        <label className="small">Run:&nbsp;
          <select value={runSel ?? ""} onChange={(e) => setRunSel(e.target.value)}>
            {runIds.map((id) => <option key={id} value={id}>{runs[id]?.label ?? id}</option>)}
          </select>
        </label>
      )}
      {run && (
        <p className="small muted">Run <code>{run.run_id}</code> · provider <b>{run.provider_mode}</b> ·
          prompts frozen {String(run.model_config.prompts_frozen)} · model lock frozen {String(run.model_config.lock_frozen)}
          {hashes.size === 1 && <> · ✓ both methods received byte-identical input</>}
          {hashes.size > 1 && <span className="error"> · ⚠ methods received different inputs</span>}</p>
      )}
      <div className="methods">
        {METHODS.map((m) => {
          const r = byMethod[m];
          return (
            <div className="method" key={m}>
              <h4>{METHOD_LABEL[m]}</h4>
              {!r ? <p className="muted small">no result in this run</p> : (
                <>
                  <div className="row"><StatusChip status={r.status} /> <LabelChip label={r.label} /> <OutcomeChip outcome={r.outcome} /></div>
                  {r.parsed && <blockquote>{(JSON.parse(r.parsed) as { visible_evidence: string }).visible_evidence}
                    <footer className="small muted">model's stated evidence — not a verified account of its reasoning</footer></blockquote>}
                  {r.parse_error && <p className="error small">parse/provider error: {r.parse_error}</p>}
                  <p className="small muted">{ms(r.elapsed_ms)} · origin <b>{r.origin ?? "–"}</b> · {r.model_id} {short(r.model_digest)} · {r.quantization}</p>
                  <details><summary>Exact prompt</summary><pre>{run?.model_config.methods[m]?.prompt ?? "(run not loaded)"}</pre>
                    <p className="small muted">prompt sha256 {short(r.prompt_hash, 16)}</p></details>
                  <details><summary>Raw model response</summary><pre>{r.raw_output ?? "(none)"}</pre></details>
                  <details><summary>Model configuration &amp; timing</summary><pre>{JSON.stringify({
                    model_id: r.model_id, model_digest: r.model_digest, quantization: r.quantization, runtime: r.runtime_version,
                    options: r.options && JSON.parse(r.options), provider_meta: r.provider_meta && JSON.parse(r.provider_meta),
                    queued_at: r.queued_at, started_at: r.started_at, finished_at: r.finished_at, image_sha256: r.image_sha256,
                  }, null, 2)}</pre></details>
                </>
              )}
            </div>
          );
        })}
      </div>
      <details><summary className="small">Scene specification</summary><pre>{JSON.stringify(c.scene_spec, null, 2)}</pre></details>
      <p className="small"><a href={api.imageUrl(c.case_id)} download={`${c.case_id}.png`}>Download exact model input PNG</a></p>
    </div>
  );
}

// ---------- metrics ----------
function MetricsTable({ m, run }: { m: Metrics; run: RunDetail }) {
  const dist = (split: string, method: string) => {
    const c: Record<string, number> = {};
    for (const r of run.results) if (r.split === split && r.method_id === method && r.label) c[r.label] = (c[r.label] ?? 0) + 1;
    return Object.entries(c).map(([k, v]) => `${k} ${v}`).join(", ") || "–";
  };
  return (
    <>
      {Object.entries(m.by_split_method).map(([split, by]) => {
        const p = m.paired_by_split[split];
        return (
          <div key={split} className="metricblock">
            <h4>Split: {split}{split === "heldout" && " (headline)"}{split === "discovery" && " (exploratory — never pooled)"}</h4>
            <div className="tablewrap"><table>
              <thead><tr><th>Method</th><th>Attempted</th><th>Valid</th><th>Op. failures</th><th>Abstain</th>
                <th>Correct / attempted</th><th>Acc. answered</th><th>Coverage</th><th>False vehicle on depictions</th>
                <th>Vehicle recall</th><th>Control → other</th><th>Latency p50 / p95 (n)</th><th>Predicted labels</th></tr></thead>
              <tbody>{Object.entries(by).map(([meth, s]: [string, Summary]) => (
                <tr key={meth}><td>{METHOD_LABEL[meth] ?? meth}</td><td>{s.attempted}</td><td>{s.valid_responses}</td>
                  <td>{s.operational_failures}</td><td>{s.abstentions}</td><td><b>{fmtFrac(s.correct_over_attempted)}</b></td>
                  <td>{fmtFrac(s.accuracy_answered)}</td><td>{fmtFrac(s.coverage)}</td>
                  <td>{fmtFrac(s.false_vehicle_on_depictions)}</td><td>{fmtFrac(s.vehicle_recall)}</td>
                  <td>{fmtFrac(s.control_correct)}</td>
                  <td>{s.latency_ms.n ? `${Math.round(s.latency_ms.p50!)} / ${Math.round(s.latency_ms.p95!)} ms (${s.latency_ms.n})` : "–"}</td>
                  <td className="small">{dist(split, meth)}</td></tr>
              ))}</tbody>
            </table></div>
            {p && p.pairs > 0 && (
              <p className="small">Paired on identical inputs ({p.pairs} pairs, hash mismatches {p.hash_mismatch}):
                both correct <b>{p.both_correct}</b> · baseline-only (regressions) <b>{p.baseline_only}</b> ·
                variant-only (repairs) <b>{p.variant_only}</b> · neither <b>{p.neither}</b> · unresolved <b>{p.unresolved}</b></p>
            )}
          </div>
        );
      })}
      <p className="small muted">Small synthetic sample; no statistical significance or real-world safety claim. Operational failures and
        abstentions count as not correct in “correct / attempted”.</p>
    </>
  );
}

// ---------- gallery ----------
interface GalleryCase { case_id: string; split: string; truth: string; cond: Record<string, unknown>;
  res: Record<string, { label: string | null; outcome: string | null; status: string }> }

function Gallery({ run, pairedCases, onPick, selected }: {
  run: RunDetail; pairedCases: Record<string, string>; onPick: (id: string) => void; selected: string | null;
}) {
  const [f, setF] = useState({ outcome: "all", target: "all", lighting: "all", appearance: "all", pair: "all" });
  const cases = useMemo(() => {
    const m: Record<string, GalleryCase> = {};
    for (const r of run.results) {
      m[r.case_id] ??= { case_id: r.case_id, split: r.split, truth: r.truth, cond: JSON.parse(r.conditions || "{}"), res: {} };
      m[r.case_id].res[r.method_id] = { label: r.label, outcome: r.outcome, status: r.status };
    }
    return Object.values(m);
  }, [run]);
  const shown = cases.filter((c) => {
    const outs = Object.values(c.res).map((r) => r.outcome);
    if (f.outcome === "any_error" && outs.every((o) => o === "correct")) return false;
    if (f.outcome !== "all" && f.outcome !== "any_error" && !outs.includes(f.outcome)) return false;
    if (f.target !== "all" && c.cond.target !== f.target) return false;
    if (f.lighting !== "all" && c.cond.lighting !== f.lighting) return false;
    if (f.appearance !== "all" && c.cond.appearance !== f.appearance) return false;
    if (f.pair !== "all" && pairedCases[c.case_id] !== f.pair) return false;
    return true;
  });
  const sel = (key: keyof typeof f, opts: [string, string][]) => (
    <label className="small">{key}&nbsp;
      <select value={f[key]} onChange={(e) => setF({ ...f, [key]: e.target.value })}>
        {opts.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
      </select>
    </label>
  );
  return (
    <>
      <div className="filters">
        {sel("outcome", [["all", "all"], ["any_error", "any non-correct"], ["correct", "correct"], ["abstention", "abstention"],
          ["depiction_as_vehicle", "depiction_as_vehicle"], ["vehicle_as_depiction", "vehicle_as_depiction"],
          ["wrong_other_label", "wrong_other_label"], ["invalid_output", "invalid_output"], ["timeout", "timeout"]])}
        {sel("pair", [["all", "all"], ["both_correct", "both correct"], ["baseline_only", "baseline-only"],
          ["variant_only", "variant-only"], ["neither", "neither"], ["unresolved", "unresolved"]])}
        {sel("target", [["all", "all"], ["truck", "truck"], ["depiction_board", "depiction board"], ["plain_board", "plain board"]])}
        {sel("lighting", [["all", "all"], ["day", "day"], ["dusk", "dusk"]])}
        {sel("appearance", [["all", "all"], ["white_livery", "white"], ["striped_livery", "striped"], ["dark_livery", "dark"]])}
        <span className="small muted">{shown.length} / {cases.length} cases</span>
      </div>
      <div className="gallery">
        {shown.map((c) => (
          <button key={c.case_id} className={`card ${selected === c.case_id ? "sel" : ""}`} onClick={() => onPick(c.case_id)}
            aria-label={`Case ${c.case_id}, truth ${c.truth}`}>
            <img src={api.imageUrl(c.case_id)} alt={`model input for ${c.case_id}`} loading="lazy" />
            <div className="cardmeta">
              <div className="small"><code>{c.case_id}</code> · truth <LabelChip label={c.truth} /></div>
              <div className="small muted">{String(c.cond.target)} · {String(c.cond.appearance)} · {String(c.cond.lighting)}
                {c.cond.variant ? ` · ${String(c.cond.variant)}` : ""}</div>
              {METHODS.map((m) => c.res[m] && (
                <div key={m} className="small">{m === "rgb_baseline" ? "base" : "struct"}: <LabelChip label={c.res[m].label} />{" "}
                  <OutcomeChip outcome={c.res[m].outcome} /></div>
              ))}
            </div>
          </button>
        ))}
      </div>
    </>
  );
}

// ---------- app ----------
export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);
  const [ctl, setCtl] = useState<Controls>(DEFAULTS);
  const [analysis, setAnalysis] = useState<Analysis>({ phase: "idle" });
  const [caseId, setCaseId] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => {
    const tick = () => api.health().then((h) => { setHealth(h); setHealthErr(null); }).catch((e) => setHealthErr(String(e)));
    tick();
    const t = setInterval(tick, 10000);
    return () => clearInterval(t);
  }, []);

  const loadRuns = useCallback(() => api.runs().then((rs) => {
    setRuns(rs);
    setRunId((cur) => cur ?? rs.find((r) => r.run_key === "heldout-v1")?.run_id ?? rs[0]?.run_id ?? null);
  }), []);
  useEffect(() => { loadRuns(); }, [loadRuns]);
  useEffect(() => {
    if (!runId) return;
    api.run(runId).then(setRun);
    api.metrics(runId).then(setMetrics);
  }, [runId, refresh]);

  const spec = useMemo(() => makeSpec({
    target: ctl.target, appearance: ctl.appearance, camera_preset: ctl.preset,
    camera: { azimuth_deg: ctl.azimuth, distance_m: ctl.distance }, lighting: ctl.lighting, occluder: ctl.occluder,
  }), [ctl]);
  const built = useMemo(() => buildScene(spec), [spec]);
  useEffect(() => () => built.dispose(), [built]);

  const setPreset = (p: string) => {
    const c = CAMERA_PRESETS[p];
    setCtl({ ...ctl, preset: p, azimuth: c.azimuth_deg, distance: c.distance_m });
  };

  const busy = ["capturing", "uploading", "queued", "running"].includes(analysis.phase);

  async function analyze() {
    try {
      setAnalysis({ phase: "capturing" });
      const canvas = renderSensor(built, spec.resolution);
      const blob = await new Promise<Blob>((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error("capture failed"))), "image/png"));
      setAnalysis({ phase: "uploading" });
      const c = await api.createCase({
        split: "interactive", family: FAMILY[spec.target], truth: TRUTH[spec.target], scene_seed: spec.seed, scene_spec: spec,
        observability: "unchecked", source_kind: "procedural", license: "self-created", renderer_version: RENDERER_VERSION,
      }, blob);
      const r = await api.createRun({ case_ids: [c.case_id], method_ids: [...METHODS], label: "interactive analyze" });
      setAnalysis({ phase: "queued", caseId: c.case_id, runId: r.run_id, sha: c.image_sha256 });
      for (;;) {
        await new Promise((res) => setTimeout(res, 400));
        const d = await api.run(r.run_id);
        const progress: Record<string, string> = {};
        for (const x of d.results) progress[x.method_id] = x.status;
        const phase: Phase = d.status === "completed" ? "completed" : d.status === "running" ? "running" : "queued";
        setAnalysis({ phase, caseId: c.case_id, runId: r.run_id, progress, sha: c.image_sha256 });
        if (d.status === "completed" || d.status === "cancelled") break;
      }
      setCaseId(c.case_id);
      setRefresh((x) => x + 1);
      loadRuns();
    } catch (e) {
      setAnalysis({ phase: "error", error: String(e) });
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>Perception Lab</h1>
          <p className="tagline">Change appearance without changing physical geometry, and inspect when a local vision model changes its answer.</p>
        </div>
        <div className="health" aria-live="polite">
          {healthErr && <span className="chip out-bad">backend unreachable</span>}
          {health && <>
            <span className={`chip ${health.provider_mode === "live" ? "out-ok" : "out-abst"}`}>
              {health.provider_mode === "live" ? "LIVE local inference" : "FIXTURE MODE — not model output"}</span>
            <span className="small">model <b>{health.configured_model}</b> {health.digest_matches_lock ? "✓ digest" : "⚠ digest mismatch"}</span>
            <span className="small">prompts {health.prompts_frozen ? "frozen" : "NOT frozen"} · lock {health.lock_frozen ? "frozen" : "NOT frozen"}</span>
            <span className="small">offline guard {health.offline_guard ? "on" : "off"} · queue {health.queue_depth}</span>
            <span className="small muted">Rendering: this browser (WebGL) · Inference: GX10 (Ollama)</span>
          </>}
        </div>
      </header>

      <main className="cols">
        <section className="col" aria-labelledby="h-scene">
          <h2 id="h-scene">1 · Scene (observer view)</h2>
          <ObserverView built={built} />
          <p className="small muted">Drag to orbit. Yellow box = target slot; wireframe = sensor camera. Helpers are on a separate layer and never appear in the model image.</p>
          <fieldset>
            <legend>Target preset</legend>
            {(Object.keys(TARGET_LABELS) as TargetType[]).map((t) => (
              <label key={t} className="radio"><input type="radio" name="target" checked={ctl.target === t}
                onChange={() => setCtl({ ...ctl, target: t })} /> {TARGET_LABELS[t]}</label>
            ))}
          </fieldset>
          <div className="grid2">
            <label>Appearance<select value={ctl.appearance} onChange={(e) => setCtl({ ...ctl, appearance: e.target.value as Appearance })}>
              {(Object.keys(APPEARANCE_LABELS) as Appearance[]).map((a) => <option key={a} value={a}>{APPEARANCE_LABELS[a]}</option>)}
            </select></label>
            <label>Lighting<select value={ctl.lighting} onChange={(e) => setCtl({ ...ctl, lighting: e.target.value as Lighting })}>
              <option value="day">Day</option><option value="dusk">Dusk (back-lit)</option></select></label>
            <label>Camera preset<select value={ctl.preset} onChange={(e) => setPreset(e.target.value)}>
              <option value="cam_a_ego_lane">A · ego lane (development)</option>
              <option value="cam_b_oblique">B · oblique (held-out)</option>
              {ctl.preset === "custom" && <option value="custom">custom</option>}</select></label>
            <label className="radio"><input type="checkbox" checked={ctl.occluder} onChange={(e) => setCtl({ ...ctl, occluder: e.target.checked })} /> Hedge occluder</label>
            <label>Viewing angle {ctl.azimuth}°<input type="range" min={0} max={70} step={1} value={ctl.azimuth}
              onChange={(e) => setCtl({ ...ctl, preset: "custom", azimuth: +e.target.value })} /></label>
            <label>Distance {ctl.distance.toFixed(1)} m<input type="range" min={12} max={40} step={0.5} value={ctl.distance}
              onChange={(e) => setCtl({ ...ctl, preset: "custom", distance: +e.target.value })} /></label>
          </div>
          <div className="truthbox small"><span className="k">Ground truth for this scene</span> <LabelChip label={TRUTH[ctl.target]} />
            <span className="muted"> — shown here only; never drawn into the model image.</span></div>
        </section>

        <section className="col" aria-labelledby="h-sensor">
          <h2 id="h-sensor">2 · Exact model input</h2>
          <SensorView built={built} />
          <p className="small muted">Fixed sensor camera, 512×512 PNG, neutral yellow marker around the target slot (same box for every target type).</p>
          <button className="primary" onClick={analyze} disabled={busy || !health}>
            {busy ? "Analyzing…" : "Analyze with both methods"}</button>
          <div className="status" aria-live="polite">
            {analysis.phase !== "idle" && <p className="small">Status: <StatusChip status={analysis.phase} />
              {analysis.sha && <> · stored input sha256 <code>{short(analysis.sha, 16)}</code></>}</p>}
            {analysis.progress && <ul className="small">{METHODS.map((m) => (
              <li key={m}>{METHOD_LABEL[m]}: <StatusChip status={analysis.progress![m] ?? "queued"} /></li>))}</ul>}
            {analysis.error && <p className="error" role="alert">{analysis.error}</p>}
          </div>
          {caseId && <>
            <h3>Stored input for selected case <code>{caseId}</code> (bytes sent to the model)</h3>
            <img className="sensor" src={`${api.imageUrl(caseId)}?r=${refresh}`} alt="Stored image exactly as sent to the model" />
          </>}
        </section>

        <section className="col" aria-labelledby="h-evidence">
          <h2 id="h-evidence">3 · Method comparison &amp; evidence</h2>
          <EvidencePanel caseId={caseId} refresh={refresh} />
        </section>
      </main>

      <section className="results" aria-labelledby="h-results">
        <h2 id="h-results">Results</h2>
        <div className="row">
          <label>Run&nbsp;
            <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value)}>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>
                {r.run_key ?? "adhoc"} · {r.label} · {r.provider_mode} · {r.status}</option>)}
            </select></label>
          {runId && <span className="small">Export: <a href={api.exportUrl(runId, "zip")}>bundle (.zip)</a> ·{" "}
            <a href={api.exportUrl(runId, "csv")}>CSV</a> · <a href={api.exportUrl(runId, "jsonl")}>JSONL</a> ·{" "}
            <a href={api.exportUrl(runId, "md")}>report (.md)</a></span>}
        </div>
        {run && metrics && <MetricsTable m={metrics} run={run} />}
        {run && metrics && <>
          <h3>Case gallery — complete run (not selected examples)</h3>
          <Gallery run={run} onPick={(id) => { setCaseId(id); window.scrollTo({ top: 0, behavior: "smooth" }); }} selected={caseId}
            pairedCases={Object.values(metrics.paired_by_split).reduce((a, p) => ({ ...a, ...p.cases }), {} as Record<string, string>)} />
        </>}
      </section>
      <footer className="small muted">Educational research prototype. Not a driving controller, ADAS, or evidence about production vehicles.
        Synthetic procedural scenes; small samples; no significance or safety claims.</footer>
    </div>
  );
}
