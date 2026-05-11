import test from 'node:test';
import assert from 'node:assert/strict';

import { SKYBOX_PARAMS_BUFFER_SIZE, createSkyboxParamsData } from '../.tmp-tests/core/uniforms.js';
import { getRealtimeResolvePlan } from '../.tmp-tests/core/frame-plan.js';
import { isWindowsUserAgent, shouldRequestHighPerformanceAdapter } from '../.tmp-tests/core/webgpu-config.js';
import {
  buildLocalFileIndex,
  getEntryFile,
  resolveLocalGLTFResource,
} from '../.tmp-tests/scene/gltf-file-resolver.js';
import { collectExternalGLTFDependencies } from '../.tmp-tests/scene/gltf-manifest.js';
import {
  getMissingLocalResourceMessage,
  getUnsupportedExternalResourceMessage,
  isDataUri,
} from '../.tmp-tests/scene/gltf-support.js';

test('skybox params uniform allocates the padded WebGPU buffer size', () => {
  assert.equal(SKYBOX_PARAMS_BUFFER_SIZE, 32);

  const data = createSkyboxParamsData(1.5);
  assert.equal(data.byteLength, 16);
  assert.deepEqual(Array.from(data), [1.5, 0, 0, 0]);
});

test('windows skips the high-performance adapter hint', () => {
  assert.equal(isWindowsUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64)'), true);
  assert.equal(shouldRequestHighPerformanceAdapter('Mozilla/5.0 (Windows NT 10.0; Win64; x64)'), false);
  assert.equal(shouldRequestHighPerformanceAdapter('Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)'), true);
});

test('taa resolve plan keeps history input distinct from taa output', () => {
  const taaPlan = getRealtimeResolvePlan(true);
  assert.equal(taaPlan.shouldCopyResolvedToHistory, true);
  assert.notEqual(taaPlan.taaOutputTexture, taaPlan.historyTexture);

  const noTaaPlan = getRealtimeResolvePlan(false);
  assert.equal(noTaaPlan.taaOutputTexture, 'hdr');
  assert.equal(noTaaPlan.shouldCopyResolvedToHistory, false);
});

test('gltf helper reports unsupported external resources clearly', () => {
  assert.equal(isDataUri('data:application/octet-stream;base64,AAAA'), true);
  assert.equal(isDataUri('buffers/model.bin'), false);
  assert.match(
    getUnsupportedExternalResourceMessage('scene.gltf', 'buffer', 'buffers/model.bin'),
    /scene\.gltf references an external buffer \(buffers\/model\.bin\)/,
  );
});

test('gltf local file resolver matches referenced bin and texture files', () => {
  const files = [
    { name: 'DamagedHelmet.gltf' },
    { name: 'DamagedHelmet.bin' },
    { name: 'textures/baseColor.png' },
  ];

  const entryFile = getEntryFile(files, ['.gltf', '.glb']);
  assert.equal(entryFile?.name, 'DamagedHelmet.gltf');

  const fileIndex = buildLocalFileIndex(files);
  assert.equal(resolveLocalGLTFResource(fileIndex, entryFile, 'DamagedHelmet.bin')?.name, 'DamagedHelmet.bin');
  assert.equal(resolveLocalGLTFResource(fileIndex, entryFile, './textures/baseColor.png')?.name, 'textures/baseColor.png');
  assert.equal(resolveLocalGLTFResource(fileIndex, entryFile, 'missing.bin'), null);
});

test('gltf manifest lists external file dependencies and formats a useful missing-files message', () => {
  const dependencies = collectExternalGLTFDependencies({
    buffers: [{ uri: 'scene.bin' }],
    images: [{ uri: 'textures/baseColor.png' }, { uri: 'data:image/png;base64,AAAA' }],
  });

  assert.deepEqual(dependencies.buffers, ['scene.bin']);
  assert.deepEqual(dependencies.images, ['textures/baseColor.png']);
  assert.match(
    getMissingLocalResourceMessage('scene.gltf', ['scene.bin', 'textures/baseColor.png']),
    /needs local dependency files selected together/,
  );
});
