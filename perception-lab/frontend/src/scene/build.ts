import * as THREE from "three";
import {
  Appearance, CameraSpec, LOOK_AT_HEIGHT, Lighting, SLOT, SceneSpec, TEXTURE_CAMERA, rng,
} from "./spec";

const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const std = (color: THREE.ColorRepresentation, o: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.7, metalness: 0.05, ...o });

export function cameraPosition(c: CameraSpec): THREE.Vector3 {
  const a = THREE.MathUtils.degToRad(c.azimuth_deg);
  return V(SLOT.x + c.distance_m * Math.sin(a), c.height_m, SLOT.z + c.distance_m * Math.cos(a));
}

export function makeCamera(c: CameraSpec, aspect = 1): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(c.fov_deg, aspect, 0.1, 700);
  cam.position.copy(cameraPosition(c));
  cam.lookAt(SLOT.x, LOOK_AT_HEIGHT, SLOT.z);
  cam.updateMatrixWorld();
  return cam;
}

// ---------- truck (local frame: faces +z, ground at y=0) ----------
const TRUCK_BOX = { x: 1.45, yTop: 3.9, zBack: -5.0, zFront: 5.0 };

export function truckCorners(): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  for (const x of [-TRUCK_BOX.x, TRUCK_BOX.x])
    for (const y of [0, TRUCK_BOX.yTop])
      for (const z of [TRUCK_BOX.zBack, TRUCK_BOX.zFront]) out.push(V(SLOT.x + x, y, SLOT.z + z));
  return out;
}

function stripeTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d")!;
  g.fillStyle = "#f4f1ea";
  g.fillRect(0, 0, 256, 256);
  const cols = ["#c0392b", "#1f3a5f"];
  for (let i = -8; i < 16; i++) {
    g.fillStyle = cols[((i % 2) + 2) % 2];
    g.beginPath();
    g.moveTo(i * 32, 0);
    g.lineTo(i * 32 + 16, 0);
    g.lineTo(i * 32 + 16 - 256, 256);
    g.lineTo(i * 32 - 256, 256);
    g.closePath();
    g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

const LIVERY: Record<Appearance, { cab: string; trailer: string; stripes: boolean }> = {
  white_livery: { cab: "#efefec", trailer: "#f2f2ef", stripes: false },
  striped_livery: { cab: "#b8332a", trailer: "#f4f1ea", stripes: true },
  dark_livery: { cab: "#2d3035", trailer: "#33363b", stripes: false },
};

export function buildTruck(app: Appearance): THREE.Group {
  const L = LIVERY[app];
  const g = new THREE.Group();
  g.name = "target_truck";
  g.position.set(SLOT.x, 0, SLOT.z);
  const add = (geo: THREE.BufferGeometry, mat: THREE.Material | THREE.Material[], x: number, y: number, z: number) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    g.add(m);
    return m;
  };
  const trailerMat = std(L.trailer);
  let trailerMats: THREE.Material | THREE.Material[] = trailerMat;
  if (L.stripes) {
    const t = stripeTexture();
    t.repeat.set(3, 1);
    const side = std("#ffffff", { map: t });
    trailerMats = [side, side, trailerMat, trailerMat, trailerMat, trailerMat];
  }
  add(new THREE.BoxGeometry(2.5, 2.9, 7.0), trailerMats, 0, 2.45, -1.5);
  add(new THREE.BoxGeometry(1.0, 0.25, 7.0), std("#26282b"), 0, 0.85, -1.5);
  add(new THREE.BoxGeometry(2.4, 2.5, 2.5), std(L.cab, { roughness: 0.45, metalness: 0.2 }), 0, 2.2, 3.55);
  const glass = std("#18232e", { roughness: 0.15, metalness: 0.6 });
  add(new THREE.BoxGeometry(2.1, 1.0, 0.05), glass, 0, 2.85, 4.81);
  add(new THREE.BoxGeometry(0.05, 0.8, 1.0), glass, 1.21, 2.85, 4.0);
  add(new THREE.BoxGeometry(0.05, 0.8, 1.0), glass, -1.21, 2.85, 4.0);
  add(new THREE.BoxGeometry(1.6, 0.6, 0.05), std("#3a3c40"), 0, 1.65, 4.81);
  add(new THREE.BoxGeometry(2.45, 0.35, 0.25), std("#9aa0a6", { metalness: 0.7, roughness: 0.3 }), 0, 0.78, 4.85);
  const lamp = std("#fff3c4", { emissive: new THREE.Color("#fff3c4"), emissiveIntensity: 0.35 });
  add(new THREE.BoxGeometry(0.35, 0.2, 0.05), lamp, 0.85, 1.15, 4.86);
  add(new THREE.BoxGeometry(0.35, 0.2, 0.05), lamp, -0.85, 1.15, 4.86);
  add(new THREE.BoxGeometry(0.08, 0.45, 0.2), std("#222"), 1.36, 2.6, 4.55);
  add(new THREE.BoxGeometry(0.08, 0.45, 0.2), std("#222"), -1.36, 2.6, 4.55);
  const wheelGeo = new THREE.CylinderGeometry(0.5, 0.5, 0.35, 24).rotateZ(Math.PI / 2);
  const tyre = std("#151515", { roughness: 0.9 });
  for (const z of [4.0, 2.6, -3.4, -4.4]) for (const x of [-1.08, 1.08]) add(wheelGeo, tyre, x, 0.5, z);
  return g;
}

// ---------- board (shared by depiction and plain control) ----------
export interface BoardGeom {
  F: THREE.Vector3; n: THREE.Vector3; t: THREE.Vector3;
  uMin: number; uMax: number; vMin: number; vMax: number;
  width: number; height: number; center: THREE.Vector3;
}

let boardCache: BoardGeom | null = null;

/** The board stands at the truck's front, faces the texture camera, and is just large enough
 *  to contain the truck's silhouette as seen from that camera. Identical for both board types. */
export function boardGeometry(): BoardGeom {
  if (boardCache) return boardCache;
  const A = cameraPosition(TEXTURE_CAMERA);
  const F = V(SLOT.x, 0, SLOT.z + TRUCK_BOX.zFront + 0.3);
  const n = V(A.x - F.x, 0, A.z - F.z).normalize();
  const t = V(n.z, 0, -n.x);
  let uMin = Infinity, uMax = -Infinity, vMax = -Infinity;
  for (const P of truckCorners()) {
    const d = P.clone().sub(A);
    const s = F.clone().sub(A).dot(n) / d.dot(n);
    const X = A.clone().addScaledVector(d, s);
    const u = X.clone().sub(F).dot(t);
    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
    vMax = Math.max(vMax, X.y);
  }
  const m = 0.35;
  uMin -= m; uMax += m; vMax += m;
  const vMin = 0.0;
  const center = F.clone().addScaledVector(t, (uMin + uMax) / 2).setY((vMin + vMax) / 2);
  boardCache = { F, n, t, uMin, uMax, vMin, vMax, width: uMax - uMin, height: vMax - vMin, center };
  return boardCache;
}

export function boardCorners(): THREE.Vector3[] {
  const b = boardGeometry();
  const out: THREE.Vector3[] = [];
  for (const u of [b.uMin - 0.12, b.uMax + 0.12])
    for (const v of [0, b.vMax + 0.12]) out.push(b.F.clone().addScaledVector(b.t, u).setY(v));
  return out;
}

const textureCache = new Map<string, { texture: THREE.Texture; dataUrl: string }>();

/** Photograph the truck from the saved texture camera, with an off-axis frustum that exactly
 *  covers the board rectangle. Returns the texture and the PNG asset (for hashing/provenance). */
export function depictionTexture(app: Appearance): { texture: THREE.Texture; dataUrl: string } {
  const key = `depiction:${app}`;
  const hit = textureCache.get(key);
  if (hit) return hit;
  const b = boardGeometry();
  const TW = 1024, TH = Math.round((1024 * b.height) / b.width);
  const canvas = document.createElement("canvas");
  canvas.width = TW;
  canvas.height = TH;
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  r.setPixelRatio(1);
  r.setSize(TW, TH, false);
  r.shadowMap.enabled = true;
  r.shadowMap.type = THREE.PCFShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#cdd6dd");
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(300, 300), std("#80858a"));
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(SLOT.x, 0, SLOT.z);
  ground.receiveShadow = true;
  scene.add(ground);
  const truck = buildTruck(app);
  truck.traverse((o) => { o.castShadow = true; });
  scene.add(truck);
  scene.add(new THREE.HemisphereLight("#ffffff", "#6d6d6d", 1.1));
  const sun = new THREE.DirectionalLight("#ffffff", 2.2);
  sun.position.set(SLOT.x + 15, 35, SLOT.z + 20);
  sun.target.position.set(SLOT.x, 0, SLOT.z);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -15, right: 15, top: 15, bottom: -15, near: 1, far: 100 });
  scene.add(sun, sun.target);
  const A = cameraPosition(TEXTURE_CAMERA);
  const cam = new THREE.PerspectiveCamera(30, TW / TH, 0.1, 500);
  cam.position.copy(A);
  cam.lookAt(A.clone().sub(b.n));
  cam.updateMatrixWorld();
  const D = A.clone().sub(b.F).dot(b.n);
  const off = b.F.clone().sub(A).dot(b.t);
  const near = 0.1, k = near / D;
  cam.projectionMatrix.makePerspective((b.uMin + off) * k, (b.uMax + off) * k, (b.vMax - A.y) * k, (b.vMin - A.y) * k, near, 500);
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
  r.render(scene, cam);
  const dataUrl = canvas.toDataURL("image/png");
  const copy = document.createElement("canvas");
  copy.width = TW;
  copy.height = TH;
  copy.getContext("2d")!.drawImage(canvas, 0, 0);
  const texture = new THREE.CanvasTexture(copy);
  texture.colorSpace = THREE.SRGBColorSpace;
  r.dispose();
  r.forceContextLoss();
  const out = { texture, dataUrl };
  textureCache.set(key, out);
  return out;
}

function plainBoardMaterial(app: Appearance): THREE.Material {
  if (app === "striped_livery") {
    const t = stripeTexture();
    const b = boardGeometry();
    t.repeat.set(b.width / 3, b.height / 3);
    return std("#ffffff", { map: t });
  }
  return std(app === "white_livery" ? "#f2f2ef" : "#33363b");
}

export function buildBoard(front: THREE.Material): THREE.Group {
  const b = boardGeometry();
  const g = new THREE.Group();
  g.name = "target_board";
  g.position.copy(b.center);
  g.rotation.y = Math.atan2(b.n.x, b.n.z);
  const W = b.width, H = b.height;
  const edge = std("#9a9ea3");
  const back = std("#5d6166");
  g.add(new THREE.Mesh(new THREE.BoxGeometry(W, H, 0.06), [edge, edge, edge, edge, front, back]));
  const frame = std("#3b3f44", { metalness: 0.4, roughness: 0.5 });
  const bar = (w: number, h: number, x: number, y: number) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.1), frame);
    m.position.set(x, y, 0.0);
    g.add(m);
  };
  bar(W + 0.24, 0.12, 0, H / 2 + 0.06);
  bar(W + 0.24, 0.12, 0, -H / 2 + 0.06);
  bar(0.12, H + 0.24, -W / 2 - 0.06, 0);
  bar(0.12, H + 0.24, W / 2 + 0.06, 0);
  const metal = std("#4a4e53", { metalness: 0.5, roughness: 0.5 });
  for (const x of [-W / 3, W / 3]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.14, H * 0.92, 0.14), metal);
    post.position.set(x, -H / 2 + (H * 0.92) / 2, -0.12);
    g.add(post);
    const len = Math.hypot(H * 0.75, 1.6);
    const brace = new THREE.Mesh(new THREE.BoxGeometry(0.1, len, 0.1), metal);
    brace.position.set(x, -H / 2 + (H * 0.75) / 2, -0.12 - 0.8);
    brace.rotation.x = -Math.atan2(1.6, H * 0.75);
    g.add(brace);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.08, 1.9), metal);
    foot.position.set(x, -H / 2 + 0.04, -0.95);
    g.add(foot);
  }
  return g;
}

// ---------- environment ----------
function circlesTexture(seed: number): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 256;
  const g = c.getContext("2d")!;
  const r = rng(seed * 7 + 3);
  g.fillStyle = "#f6e7c8";
  g.fillRect(0, 0, 512, 256);
  const pal = ["#e4572e", "#17bebb", "#ffc914", "#2e282a", "#76b041"];
  for (let i = 0; i < 14; i++) {
    g.fillStyle = pal[Math.floor(r() * pal.length)];
    g.beginPath();
    g.arc(r() * 512, r() * 256, 15 + r() * 55, 0, Math.PI * 2);
    g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildEnvironment(scene: THREE.Scene, seed: number) {
  const r = rng(seed);
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), std("#5d7a3a", { roughness: 1 }));
  grass.rotation.x = -Math.PI / 2;
  grass.position.z = -150;
  scene.add(grass);
  const flat = (w: number, l: number, color: string, x: number, y: number, z: number) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, l), std(color, { roughness: 0.95 }));
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, y, z);
    scene.add(m);
  };
  flat(9.0, 420, "#4b4e52", 0, 0.004, -150);
  flat(0.15, 420, "#e6e6e2", 3.6, 0.008, -150);
  flat(0.15, 420, "#e6e6e2", -3.6, 0.008, -150);
  for (let z = 45; z > -360; z -= 9) flat(0.15, 3, "#e6e6e2", 0, 0.008, z);

  const trunk = std("#5a3d22");
  for (let i = 0; i < 44; i++) {
    const side = i % 2 ? 1 : -1;
    const x = side * (8 + r() * 22);
    const z = 25 - r() * 230;
    const s = 0.8 + r() * 0.7;
    const hue = 0.25 + r() * 0.08;
    const light = 0.24 + r() * 0.08;
    if (side > 0 && z > -42) continue; // keep the oblique camera's line of sight clear
    if (side < 0 && z > -34 && z < -12) continue;
    const tr = new THREE.Mesh(new THREE.CylinderGeometry(0.22 * s, 0.3 * s, 2.4 * s, 10), trunk);
    tr.position.set(x, 1.2 * s, z);
    const crown = new THREE.Mesh(new THREE.ConeGeometry(1.9 * s, 4.6 * s, 12),
      std(new THREE.Color().setHSL(hue, 0.45, light)));
    crown.position.set(x, 2.4 * s + 2.2 * s, z);
    scene.add(tr, crown);
  }
  for (let i = 0; i < 6; i++) {
    const w = 14 + r() * 18, h = 8 + r() * 20, d = 12 + r() * 10;
    const bld = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
      std(["#b9b2a5", "#8f969c", "#c9bfae", "#9da59a"][i % 4]));
    bld.position.set(-48 - r() * 35, h / 2, -110 - i * 38);
    scene.add(bld);
  }
  const poleMat = std("#6f7378", { metalness: 0.6, roughness: 0.4 });
  for (const z of [-5, -45, -85, -125, -165]) {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.12, 7.5, 8), poleMat);
    pole.position.set(5.0, 3.75, z);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.1, 0.1), poleMat);
    arm.position.set(4.2, 7.4, z);
    scene.add(pole, arm);
  }
  // Background billboard (scenery, never the target): abstract ad, no vehicle content.
  const bbPole = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.3, 7, 10), poleMat);
  bbPole.position.set(13, 3.5, -78);
  const ad = std("#ffffff", { map: circlesTexture(seed) });
  const bbEdge = std("#3b3f44");
  const bb = new THREE.Mesh(new THREE.BoxGeometry(9, 4.5, 0.25), [bbEdge, bbEdge, bbEdge, bbEdge, ad, bbEdge]);
  bb.position.set(13, 9.2, -78);
  bb.rotation.y = -0.45;
  scene.add(bbPole, bb);
}

function addLights(scene: THREE.Scene, lighting: Lighting) {
  const day = lighting === "day";
  scene.background = new THREE.Color(day ? "#a9cbe8" : "#3e3a55");
  scene.fog = new THREE.Fog(day ? "#bcd4e6" : "#4a4560", day ? 160 : 90, day ? 480 : 320);
  scene.add(new THREE.HemisphereLight(day ? "#dbeeff" : "#7a6e9a", day ? "#4f5f3a" : "#1d1a14", day ? 1.1 : 0.75));
  const sun = new THREE.DirectionalLight(day ? "#fff4e0" : "#ff9550", day ? 2.6 : 1.3);
  const o = day ? V(25, 45, 15) : V(-45, 7, -35);
  sun.position.set(SLOT.x + o.x, o.y, SLOT.z + o.z);
  sun.target.position.set(SLOT.x, 0, SLOT.z);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.02;
  Object.assign(sun.shadow.camera, { left: -35, right: 35, top: 35, bottom: -35, near: 1, far: 220 });
  scene.add(sun, sun.target);
}

function buildOccluder(spec: SceneSpec): THREE.Mesh {
  const cam = cameraPosition(spec.camera);
  const dir = V(cam.x - SLOT.x, 0, cam.z - SLOT.z).normalize();
  const hedge = new THREE.Mesh(new THREE.BoxGeometry(4.2, 1.4, 1.0), std("#3f5a2a", { roughness: 1 }));
  hedge.position.set(SLOT.x + dir.x * 10, 0.7, SLOT.z + dir.z * 10);
  hedge.rotation.y = Math.atan2(dir.x, dir.z);
  return hedge;
}

// ---------- assembly ----------
export interface BuiltScene {
  scene: THREE.Scene;
  sensorCamera: THREE.PerspectiveCamera;
  markerPoints: THREE.Vector3[];
  helpers: THREE.Object3D[];
  dispose: () => void;
}

export const HELPER_LAYER = 1;

export function buildScene(spec: SceneSpec): BuiltScene {
  const scene = new THREE.Scene();
  addLights(scene, spec.lighting);
  buildEnvironment(scene, spec.seed);
  let target: THREE.Object3D;
  if (spec.target === "truck") target = buildTruck(spec.appearance);
  else if (spec.target === "depiction_board") {
    target = buildBoard(std("#ffffff", { map: depictionTexture(spec.appearance).texture, roughness: 0.85 }));
  } else target = buildBoard(plainBoardMaterial(spec.appearance));
  scene.add(target);
  if (spec.occluder) scene.add(buildOccluder(spec));
  scene.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });
  const sensorCamera = makeCamera(spec.camera, 1);
  const markerPoints = [...truckCorners(), ...boardCorners()];
  const camHelper = new THREE.CameraHelper(sensorCamera);
  const slotBox = new THREE.Box3().setFromPoints(markerPoints);
  const boxHelper = new THREE.Box3Helper(slotBox, new THREE.Color("#ffd400"));
  const helpers = [camHelper, boxHelper];
  for (const h of helpers) {
    h.layers.set(HELPER_LAYER);
    scene.add(h);
  }
  const cached = new Set([...textureCache.values()].map((v) => v.texture));
  return {
    scene, sensorCamera, markerPoints, helpers,
    dispose: () => scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
      for (const mat of mats) {
        const map = (mat as THREE.MeshStandardMaterial).map;
        if (map && !cached.has(map)) map.dispose();
        mat.dispose();
      }
    }),
  };
}
