import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss()],
  build: {
    outDir: "app/static",
    emptyOutDir: false,
    sourcemap: true,
    target: "es2022",
    rollupOptions: {
      input: {
        common: "app/assets/ts/common.ts",
        classifier: "app/assets/ts/classifier.ts",
        paywall: "app/assets/ts/paywall.ts",
        storefront: "app/assets/ts/storefront.ts",
        styles: "app/assets/css/input.css",
      },
      output: {
        entryFileNames: "js/[name].js",
        chunkFileNames: "js/[name].js",
        assetFileNames: (assetInfo) =>
          assetInfo.names?.some((name) => name.endsWith(".css"))
            ? "css/styles.css"
            : "assets/[name][extname]",
      },
    },
  },
});
