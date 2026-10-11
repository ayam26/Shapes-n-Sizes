import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api is proxied to the GX10 backend through the SSH tunnel on localhost:8765.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, proxy: { "/api": "http://127.0.0.1:8765" } },
});
