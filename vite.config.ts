import { defineConfig } from 'vite';

export default defineConfig({
  assetsInclude: ['**/*.wgsl', '**/*.hdr', '**/*.glb', '**/*.gltf'],
  server: {
    port: 3000,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
