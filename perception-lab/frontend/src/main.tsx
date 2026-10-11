import { createRoot } from "react-dom/client";
import { buildScene } from "./scene/build";
import { captureSpec } from "./scene/capture";
import { RENDERER_VERSION, SceneSpec } from "./scene/spec";

declare global {
  interface Window {
    __pl?: { version: string; renderCase: (spec: SceneSpec) => ReturnType<typeof captureSpec>; inspect: (spec: SceneSpec) => unknown };
  }
}

const mode = new URLSearchParams(location.search).get("mode");
if (mode === "capture") {
  // Headless corpus renderer: same scene code as the UI, no UI mounted.
  window.__pl = {
    version: RENDERER_VERSION,
    renderCase: captureSpec,
    // Scene/metadata consistency check: which target objects exist and what the sensor camera can see.
    inspect: (spec: SceneSpec) => {
      const b = buildScene(spec);
      const names: string[] = [];
      b.scene.traverse((o) => { if (o.name.startsWith("target_")) names.push(o.name); });
      const out = {
        targets: names,
        sensor_layer_mask: b.sensorCamera.layers.mask,
        helper_layer_masks: b.helpers.map((h) => h.layers.mask),
        helpers_visible_to_sensor: b.helpers.some((h) => b.sensorCamera.layers.test(h.layers)),
      };
      b.dispose();
      return out;
    },
  };
  document.body.dataset.ready = "1";
} else {
  import("./App").then(({ default: App }) => {
    createRoot(document.getElementById("root")!).render(<App />);
  });
}
