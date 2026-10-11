import { createRoot } from "react-dom/client";
import { captureSpec } from "./scene/capture";
import { RENDERER_VERSION, SceneSpec } from "./scene/spec";

declare global {
  interface Window {
    __pl?: { version: string; renderCase: (spec: SceneSpec) => ReturnType<typeof captureSpec> };
  }
}

const mode = new URLSearchParams(location.search).get("mode");
if (mode === "capture") {
  // Headless corpus renderer: same scene code as the UI, no UI mounted.
  window.__pl = { version: RENDERER_VERSION, renderCase: captureSpec };
  document.body.dataset.ready = "1";
} else {
  import("./App").then(({ default: App }) => {
    createRoot(document.getElementById("root")!).render(<App />);
  });
}
