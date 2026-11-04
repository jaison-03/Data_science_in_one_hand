import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              debugName: "3D renderer dependencies",
              name: (moduleId) => {
                if (/node_modules[\\/]three[\\/]/.test(moduleId)) return "three";
                if (/node_modules[\\/]@react-three[\\/]fiber[\\/]/.test(moduleId)) return "fiber";
                if (/node_modules[\\/]@react-three[\\/]drei[\\/]/.test(moduleId)) return "drei";
                return "camera-controls";
              },
              test: /node_modules[\\/](?:three|@react-three[\\/](?:fiber|drei)|camera-controls)[\\/]/,
              minSize: 30_000,
              maxSize: 420_000,
              includeDependenciesRecursively: false,
            },
          ],
        },
      },
    },
  },
});
