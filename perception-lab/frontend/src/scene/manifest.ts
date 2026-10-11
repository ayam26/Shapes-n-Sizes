import { Appearance, FAMILY, Lighting, SceneSpec, TRUTH, TargetType, makeSpec } from "./spec";

export interface ManifestEntry {
  case_id: string;
  split: "development" | "heldout" | "discovery";
  family: string;
  truth: string;
  observability: "clear" | "ambiguous" | "unchecked";
  spec: SceneSpec;
}

const TARGETS: TargetType[] = ["truck", "depiction_board", "plain_board"];
const APPEARANCES: Appearance[] = ["white_livery", "striped_livery", "dark_livery"];
const LIGHTS: Lighting[] = ["day", "dusk"];
export const SEED = 17;

/** Core corpus: 3 targets x 3 appearances x 2 cameras x 2 lights = 36, balanced.
 *  Development = ego-lane camera; held-out = oblique camera (sealed until freeze). */
export function buildCoreManifest(): ManifestEntry[] {
  const out: ManifestEntry[] = [];
  let i = 0;
  for (const [cam, split] of [["cam_a_ego_lane", "development"], ["cam_b_oblique", "heldout"]] as const)
    for (const target of TARGETS)
      for (const appearance of APPEARANCES)
        for (const lighting of LIGHTS) {
          i += 1;
          out.push({
            case_id: `core-${String(i).padStart(3, "0")}`, split, family: FAMILY[target], truth: TRUTH[target],
            observability: "unchecked",
            spec: makeSpec({ seed: SEED, target, appearance, camera_preset: cam, lighting }),
          });
        }
  return out;
}

/** Exploratory discovery corpus (24): distance, intermediate angle and occlusion; never pooled
 *  into held-out headline metrics. */
export function buildDiscoveryManifest(): ManifestEntry[] {
  const variants: { name: string; preset: string; camera?: { distance_m?: number; azimuth_deg?: number }; occluder?: boolean }[] = [
    { name: "near", preset: "cam_a_ego_lane", camera: { distance_m: 14 } },
    { name: "far", preset: "cam_a_ego_lane", camera: { distance_m: 34 } },
    { name: "angle25", preset: "cam_a_ego_lane", camera: { azimuth_deg: 25 } },
    { name: "occluded", preset: "cam_a_ego_lane", occluder: true },
  ];
  const out: ManifestEntry[] = [];
  let i = 0;
  for (const v of variants)
    for (const target of TARGETS)
      for (const lighting of LIGHTS) {
        i += 1;
        const spec = makeSpec({ seed: SEED, target, appearance: "white_livery", camera_preset: v.preset,
          camera: v.camera, lighting, occluder: v.occluder });
        spec.conditions.variant = v.name;
        out.push({ case_id: `disc-${String(i).padStart(3, "0")}`, split: "discovery", family: FAMILY[target],
          truth: TRUTH[target], observability: "unchecked", spec });
      }
  return out;
}
