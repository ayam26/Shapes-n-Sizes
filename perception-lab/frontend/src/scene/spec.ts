import { REVISION } from "three";

export type TargetType = "truck" | "depiction_board" | "plain_board";
export type Appearance = "white_livery" | "striped_livery" | "dark_livery";
export type Lighting = "day" | "dusk";
export type Quality = "standard" | "high";

export interface CameraSpec {
  azimuth_deg: number; // orbit angle around the target slot, 0 = looking straight down the road
  distance_m: number;
  height_m: number;
  fov_deg: number;
}

export interface SceneSpec {
  schema_version: 1;
  seed: number;
  target: TargetType;
  appearance: Appearance;
  camera_preset: string;
  camera: CameraSpec;
  lighting: Lighting;
  occluder: boolean;
  resolution: number;
  /** Absent = standard (the frozen core corpus). "high" = exploratory high-fidelity render study. */
  quality?: Quality;
  conditions: Record<string, string | number | boolean>;
}

export const RENDERER_VERSION = `pl-scene-1.0+three-r${REVISION}`;

/** Ground point at the centre of the target slot (left lane, oncoming side). */
export const SLOT = { x: -1.75, z: -22 };
export const LOOK_AT_HEIGHT = 1.8;

/** The "saved camera" used to photograph the truck for the depiction board. It equals the
 *  ego-lane camera, so from that viewpoint the picture has the same perspective as the truck. */
export const TEXTURE_CAMERA: CameraSpec = { azimuth_deg: 9, distance_m: 22.3, height_m: 1.5, fov_deg: 32 };

export const CAMERA_PRESETS: Record<string, CameraSpec> = {
  cam_a_ego_lane: { azimuth_deg: 9, distance_m: 22.3, height_m: 1.5, fov_deg: 32 },
  cam_b_oblique: { azimuth_deg: 50, distance_m: 22.3, height_m: 1.5, fov_deg: 32 },
};

export const TRUTH: Record<TargetType, "physical_vehicle" | "flat_depiction" | "other"> = {
  truck: "physical_vehicle",
  depiction_board: "flat_depiction",
  plain_board: "other",
};

export const FAMILY: Record<TargetType, string> = {
  truck: "physical_vehicle",
  depiction_board: "flat_depiction",
  plain_board: "negative_control",
};

export const TARGET_LABELS: Record<TargetType, string> = {
  truck: "Physical truck",
  depiction_board: "Truck picture on a flat board",
  plain_board: "Plain board (negative control)",
};

export const APPEARANCE_LABELS: Record<Appearance, string> = {
  white_livery: "White livery",
  striped_livery: "Striped livery",
  dark_livery: "Dark livery",
};

export function makeSpec(p: {
  seed?: number;
  target: TargetType;
  appearance: Appearance;
  camera_preset: string;
  camera?: Partial<CameraSpec>;
  lighting: Lighting;
  occluder?: boolean;
  quality?: Quality;
}): SceneSpec {
  const base = CAMERA_PRESETS[p.camera_preset] ?? CAMERA_PRESETS.cam_a_ego_lane;
  const camera = { ...base, ...(p.camera ?? {}) };
  const spec: SceneSpec = {
    schema_version: 1,
    seed: p.seed ?? 17,
    target: p.target,
    appearance: p.appearance,
    camera_preset: p.camera_preset,
    camera,
    lighting: p.lighting,
    occluder: p.occluder ?? false,
    resolution: 512,
    conditions: {
      target: p.target,
      appearance: p.appearance,
      camera: p.camera_preset,
      azimuth_deg: camera.azimuth_deg,
      distance_m: camera.distance_m,
      lighting: p.lighting,
      occluder: p.occluder ?? false,
    },
  };
  if (p.quality === "high") {  // only added when used, so standard specs are unchanged
    spec.quality = "high";
    spec.conditions.quality = "high";
  }
  return spec;
}

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
