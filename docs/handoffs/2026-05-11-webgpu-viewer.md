# WebGPU Viewer Handoff - 2026-05-11

## Current status

- Fixed the skybox uniform-size validation error by allocating a padded 32-byte uniform buffer for `SkyParams`.
- Fixed the Windows `requestAdapter()` hint handling by skipping `powerPreference` on Windows.
- Fixed the TAA history path:
  - TAA now resolves into a dedicated `taa-resolve` texture.
  - History is copied into `hdr-prev` only after the resolve.
- Fixed transient-buffer lifetime bugs:
  - Temporary buffers used by the shadow pass and skybox are now destroyed only after `queue.onSubmittedWorkDone()`.

## GLTF support added

- Added minimal support for `.gltf` scenes with external local resources selected together by the user.
- The model file input now accepts multiple files:
  - `.gltf`
  - `.glb`
  - `.bin`
  - common image formats used by textures
- `main.ts` now passes the full selected file list to the loader.
- `gltf-loader.ts` now resolves `buffers[i].uri` and `images[i].uri` against the selected local files.
- Resolution is intentionally simple and pragmatic:
  - exact relative path match if present in the selected file name
  - fallback to basename match

## New helper modules

- `src/core/uniforms.ts`
- `src/core/webgpu-config.ts`
- `src/core/frame-plan.ts`
- `src/scene/gltf-support.ts`
- `src/scene/gltf-file-resolver.ts`

## Tests added

- `tests/skybox-params.test.mjs`

Current targeted assertions cover:

- skybox uniform buffer sizing
- Windows adapter-hint logic
- TAA resolve/history separation
- unsupported external-resource messaging
- local GLTF file resolution

## Manual runtime validation still useful

- Reload the dev viewer and verify the previous WebGPU validation errors are gone in the browser console.
- Test a real `.gltf + .bin + textures` selection through the file picker.
- Test drag-and-drop with a batch of files containing one `.gltf` plus its dependencies.

## Known limitations / next useful steps

- File resolution is local-selection based only. It does not fetch sibling files from disk automatically beyond the files explicitly selected/dropped.
- Relative path handling is intentionally lightweight. If asset packs rely on more complex directory layouts, the next step would be a more robust path normalizer.
- The viewer still picks the first `.gltf` / `.glb` found in the selection.
- No dedicated UI guidance yet if the user selects only the `.gltf` without the `.bin` or textures; the loader throws a clearer error, but UX could be improved.
- A good next step would be a status message listing the missing referenced files by name.

## Verification run

- Targeted Node tests passed.
- `npm run build` passed.
