/** High-fidelity materials for the exploratory render study (spec.quality === "high").
 *  Everything is procedural and seeded, so renders stay deterministic. The standard path never calls this file. */
import * as THREE from "three";
import { Lighting, rng } from "./spec";

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return [c, c.getContext("2d")!];
}

function rgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

const clamp = (x: number) => Math.max(0, Math.min(255, Math.round(x)));

/** Two-octave value noise: a smooth low-frequency layer plus fine grain. */
function noiseCanvas(seed: number, base: string, amp: number, grain: number, size = 256): HTMLCanvasElement {
  const r = rng(seed);
  const [lo, lg] = canvas(size / 8, size / 8);
  const loImg = lg.createImageData(size / 8, size / 8);
  for (let i = 0; i < (size / 8) ** 2; i++) {
    const v = 128 + (r() - 0.5) * 255 * amp;
    loImg.data.set([v, v, v, 255], i * 4);
  }
  lg.putImageData(loImg, 0, 0);
  const [c, g] = canvas(size, size);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(lo, 0, 0, size, size);
  const img = g.getImageData(0, 0, size, size);
  const [R, G, B] = rgb(base);
  for (let i = 0; i < size * size; i++) {
    const low = (img.data[i * 4] - 128) / 255;
    const n = low + (r() - 0.5) * grain;
    img.data.set([clamp(R * (1 + n)), clamp(G * (1 + n)), clamp(B * (1 + n)), 255], i * 4);
  }
  g.putImageData(img, 0, 0);
  return c;
}

export function colorNoise(seed: number, base: string, amp: number, grain: number, repeat: [number, number]): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(noiseCanvas(seed, base, amp, grain));
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(...repeat);
  t.anisotropy = 4;
  return t;
}

export function grayNoise(seed: number, level: string, amp: number, repeat: [number, number]): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(noiseCanvas(seed, level, amp, amp));
  t.colorSpace = THREE.NoColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(...repeat);
  return t;
}

/** Corrugated trailer panel: vertical ribs with subtle grime toward the bottom. */
export function ribTexture(base: string, ribs: number): THREE.CanvasTexture {
  const [c, g] = canvas(512, 256);
  const [R, G, B] = rgb(base);
  for (let x = 0; x < 512; x++) {
    const s = 0.93 + 0.07 * Math.cos((x / 512) * ribs * Math.PI * 2);
    g.fillStyle = `rgb(${clamp(R * s)},${clamp(G * s)},${clamp(B * s)})`;
    g.fillRect(x, 0, 1, 256);
  }
  const grad = g.createLinearGradient(0, 160, 0, 256);
  grad.addColorStop(0, "rgba(60,50,40,0)");
  grad.addColorStop(1, "rgba(60,50,40,0.18)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 512, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function windowTexture(seed: number, wall: string): THREE.CanvasTexture {
  const r = rng(seed);
  const [c, g] = canvas(256, 256);
  g.fillStyle = wall;
  g.fillRect(0, 0, 256, 256);
  for (let y = 12; y < 248; y += 28)
    for (let x = 10; x < 246; x += 24) {
      g.fillStyle = r() < 0.15 ? "#d9c98f" : "#3a4652";
      g.fillRect(x, y, 14, 16);
    }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Equirectangular sky used as an environment (reflection + ambient) map. CPU-sourced, so every
 *  WebGL context (sensor, observer, texture camera) converts it itself. */
const envCache = new Map<string, THREE.Texture>();
export function skyEnvironment(kind: Lighting | "studio"): THREE.Texture {
  const hit = envCache.get(kind);
  if (hit) return hit;
  const [c, g] = canvas(512, 256);
  const pal = {
    day: ["#5f93d1", "#cfe2f2", "#7d8a6a", "#3d4a30"],
    dusk: ["#2a2547", "#d98a5a", "#3b3329", "#17140f"],
    studio: ["#e8ecef", "#ffffff", "#9aa0a6", "#6d7378"],
  }[kind];
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, pal[0]);
  grad.addColorStop(0.49, pal[1]);
  grad.addColorStop(0.51, pal[2]);
  grad.addColorStop(1, pal[3]);
  g.fillStyle = grad;
  g.fillRect(0, 0, 512, 256);
  const sun = kind === "dusk" ? { x: 330, y: 118, r: 26, col: "rgba(255,170,90," } : { x: 150, y: 50, r: 22, col: "rgba(255,250,235," };
  const sg = g.createRadialGradient(sun.x, sun.y, 0, sun.x, sun.y, sun.r * 3);
  sg.addColorStop(0, sun.col + "1)");
  sg.addColorStop(1, sun.col + "0)");
  g.fillStyle = sg;
  g.fillRect(0, 0, 512, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.mapping = THREE.EquirectangularReflectionMapping;
  envCache.set(kind, t);
  return t;
}

export const paint = (color: string, o: Partial<THREE.MeshPhysicalMaterialParameters> = {}) =>
  new THREE.MeshPhysicalMaterial({ color, roughness: 0.38, metalness: 0.15, clearcoat: 0.6, clearcoatRoughness: 0.15, ...o });
export const metal = (color: string, roughness = 0.3) =>
  new THREE.MeshStandardMaterial({ color, metalness: 0.9, roughness });
export const glass = () =>
  new THREE.MeshPhysicalMaterial({ color: "#0d1620", roughness: 0.04, metalness: 0.2, clearcoat: 1, reflectivity: 0.9 });
export const rubber = () =>
  new THREE.MeshStandardMaterial({ color: "#141414", roughness: 0.92, roughnessMap: grayNoise(91, "#e0e0e0", 0.3, [2, 1]) });
