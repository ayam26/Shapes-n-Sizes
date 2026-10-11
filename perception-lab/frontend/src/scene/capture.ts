import * as THREE from "three";
import { BuiltScene, buildScene, depictionTexture } from "./build";
import { RENDERER_VERSION, SceneSpec } from "./spec";

let sensorRenderer: THREE.WebGLRenderer | null = null;

function getSensorRenderer(): THREE.WebGLRenderer {
  if (!sensorRenderer) {
    sensorRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    sensorRenderer.setPixelRatio(1);
    sensorRenderer.shadowMap.enabled = true;
    sensorRenderer.shadowMap.type = THREE.PCFShadowMap;
  }
  return sensorRenderer;
}

export interface MarkerRect { x: number; y: number; w: number; h: number }

/** The target marker is the projected bounding box of the target slot (the union of truck and
 *  board extents), so it is identical for every target type under the same camera. */
export function markerRect(built: BuiltScene, size: number): MarkerRect {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of built.markerPoints) {
    const q = p.clone().project(built.sensorCamera);
    const sx = ((q.x + 1) / 2) * size, sy = ((1 - q.y) / 2) * size;
    x0 = Math.min(x0, sx); y0 = Math.min(y0, sy); x1 = Math.max(x1, sx); y1 = Math.max(y1, sy);
  }
  const pad = 8;
  x0 = Math.max(2, Math.floor(x0 - pad)); y0 = Math.max(2, Math.floor(y0 - pad));
  x1 = Math.min(size - 2, Math.ceil(x1 + pad)); y1 = Math.min(size - 2, Math.ceil(y1 + pad));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Render the fixed sensor camera and draw the neutral marker. This canvas is exactly what
 *  is encoded to PNG and sent to the model; nothing else (no labels, no text) is drawn. */
export function renderSensor(built: BuiltScene, size: number, out?: HTMLCanvasElement): HTMLCanvasElement {
  const r = getSensorRenderer();
  const hq = !!built.scene.userData.hq;
  r.toneMapping = hq ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
  r.shadowMap.type = hq ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
  const ss = hq ? 2 : 1; // high-fidelity study renders at 2x and downsamples
  r.setSize(size * ss, size * ss, false);
  r.render(built.scene, built.sensorCamera);
  const canvas = out ?? document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const g = canvas.getContext("2d")!;
  if (hq) {
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    g.drawImage(r.domElement, 0, 0, size, size);
  } else g.drawImage(r.domElement, 0, 0);
  const m = markerRect(built, size);
  g.strokeStyle = "#ffd400";
  g.lineWidth = 3;
  g.strokeRect(m.x + 0.5, m.y + 0.5, m.w, m.h);
  return canvas;
}

export interface CaptureResult {
  dataUrl: string;
  width: number;
  height: number;
  marker: MarkerRect;
  textureDataUrl: string | null;
  rendererVersion: string;
}

export function captureSpec(spec: SceneSpec): CaptureResult {
  const built = buildScene(spec);
  try {
    const canvas = renderSensor(built, spec.resolution);
    return {
      dataUrl: canvas.toDataURL("image/png"),
      width: canvas.width,
      height: canvas.height,
      marker: markerRect(built, spec.resolution),
      textureDataUrl: spec.target === "depiction_board" ? depictionTexture(spec.appearance, spec.quality === "high").dataUrl : null,
      rendererVersion: RENDERER_VERSION,
    };
  } finally {
    built.dispose();
  }
}
