import path from "node:path";
import sirv from "sirv";
import { defineConfig, type Connect, type Plugin } from "vite";

const OUT_DIR = path.resolve(import.meta.dirname, "..", "out");

function serveOut(): Plugin {
  // Serves pipeline output at /out for "Open latest". sirv handles the range requests video
  // seeking needs. This is a dev convenience: the built viewer needs only static files.
  const middleware = (): Connect.NextHandleFunction =>
    sirv(OUT_DIR, { dev: true, etag: true }) as Connect.NextHandleFunction;
  return {
    name: "clayview-serve-out",
    configureServer(server) {
      server.middlewares.use("/out", middleware());
    },
    configurePreviewServer(server) {
      server.middlewares.use("/out", middleware());
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [serveOut()],
  server: { port: 5199, strictPort: true },
  preview: { port: 5199, strictPort: true },
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
});
