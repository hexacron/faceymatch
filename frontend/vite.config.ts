import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend serves the built assets from frontend/dist at `/`, with the API
// mounted under `/api`. In dev, Vite serves the app instead, so `/api` is
// proxied to the locally running backend and the app can keep using relative
// URLs in both modes. Invariant 11: never bind anything but the loopback
// interface; remote access goes through `tailscale serve`.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        ws: true,
      },
    },
  },
  // `bun run preview` serves the production build; keep it loopback too.
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
