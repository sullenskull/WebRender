# WebRender

Small WebGPU 3D viewer focused on testing rendering features quickly from the browser.

## Current features

- WebGPU renderer with real-time mode
- GLB loading
- GLTF loading with local `.bin` and texture dependencies when selected together
- Skybox / HDRI environment support
- TAA pipeline groundwork
- Path tracing playground mode

## Run locally

```bash
npm install
npm run dev
```

Then open the local Vite URL and load:

- a `.glb`, or
- a `.gltf` together with its `.bin` and texture files

## Notes

- This project is currently geared toward rapid WebGPU experimentation.
- Recent debugging and handoff notes live in [`docs/handoffs/2026-05-11-webgpu-viewer.md`](docs/handoffs/2026-05-11-webgpu-viewer.md).
