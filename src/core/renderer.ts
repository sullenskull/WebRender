import type { GPUContext } from './gpu.js';
import { getRealtimeResolvePlan } from './frame-plan.js';
import { createBuffer, createLinearSampler, uploadBuffer, createTextureFromData } from './gpu.js';
import { SKYBOX_PARAMS_BUFFER_SIZE, createSkyboxParamsData } from './uniforms.js';
import { mat4Create, mat4LookAt, mat4OrthoRH, mat4Multiply } from '../utils/math.js';
import { OrbitCamera } from '../scene/camera.js';
import { packLights, createLight, type Light } from '../scene/light.js';
import type { LoadedScene, GPUMaterial } from '../scene/gltf-loader.js';
import type { IBLTextures } from '../passes/ibl-pass.js';
import { createDefaultIBL } from '../passes/ibl-pass.js';
import type { HDRImage } from '../utils/hdri.js';

import commonWGSL from '../shaders/common.wgsl?raw';
import pbrWGSL from '../shaders/pbr.wgsl?raw';
import shadowWGSL from '../shaders/shadow.wgsl?raw';
import skyboxWGSL from '../shaders/skybox.wgsl?raw';
import taaWGSL from '../shaders/taa.wgsl?raw';
import ssrWGSL from '../shaders/ssr.wgsl?raw';
import tonemapWGSL from '../shaders/tonemap.wgsl?raw';
import pathtraceWGSL from '../shaders/pathtrace.wgsl?raw';

// PT display pass: reads accumulator storage buffer → writes to storage texture (separate module)
const PT_DISPLAY_WGSL = `
@group(0) @binding(0) var<storage, read> acc_buf: array<vec4f>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_display(@builtin(global_invocation_id) gid: vec3u) {
  let dim_2d = textureDimensions(out_tex);
  let dim = vec2u(dim_2d.x, dim_2d.y);
  if gid.x >= dim.x || gid.y >= dim.y { return; }
  let idx = gid.y * dim.x + gid.x;
  let acc = acc_buf[idx];
  let color = acc.rgb / max(acc.w, 1.0);
  textureStore(out_tex, vec2i(gid.xy), vec4f(color, 1.0));
}
`;

export interface RenderSettings {
  taa: boolean;
  ssr: boolean;
  ssao: boolean;
  contactShadows: boolean;
  bloom: boolean;
  skybox: boolean;
  exposure: number;
  tonemapMode: number;  // 0=AgX 1=ACES 2=Reinhard 3=Linear
  iblIntensity: number;
  gamma: number;
  saturation: number;
  contrast: number;
  vignette: number;
  renderMode: 'realtime' | 'pathtrace';
}

const SHADOW_SIZE = 2048;

export class Renderer {
  readonly gpu: GPUContext;  // exposed for IBL compute etc.
  private camera: OrbitCamera;

  // GPU resources
  private cameraBuffer!: GPUBuffer;
  private lightBuffer!: GPUBuffer;
  private shadowDataBuffer!: GPUBuffer;
  private tonemapParamsBuffer!: GPUBuffer;
  private ssrParamsBuffer!: GPUBuffer;
  private ptParamsBuffer!: GPUBuffer;

  // Textures
  private hdrTexture!: GPUTexture;
  private hdrTexturePrev!: GPUTexture;
  private taaResolveTexture!: GPUTexture;
  private depthTexture!: GPUTexture;
  private shadowDepth!: GPUTexture;
  private ptAccBuffer!: GPUBuffer;
  private ptOutputTex!: GPUTexture;
  private envTexture!: GPUTexture;
  private ibl!: IBLTextures;

  // Pipelines
  private pbrPipeline!: GPURenderPipeline;
  private shadowPipeline!: GPURenderPipeline;
  private skyboxPipeline!: GPURenderPipeline;
  private taaPipeline!: GPURenderPipeline;
  private ssrPipeline!: GPURenderPipeline;
  private tonemapPipeline!: GPURenderPipeline;
  private ptPipeline!: GPUComputePipeline;
  private ptDisplayPipeline!: GPUComputePipeline;
  private transientBuffers: GPUBuffer[] = [];

  // Shader modules (cached)
  private commonCode: string;

  // Scene
  private scene: LoadedScene | null = null;
  lights: Light[] = [];
  settings: RenderSettings;

  // Samplers
  private linearSampler!: GPUSampler;
  private shadowSampler!: GPUSampler;
  private nearestSampler!: GPUSampler;

  // Default 1×1 textures for unbound slots
  private whiteTexture!: GPUTexture;
  private normalDefaultTex!: GPUTexture;

  // Frame stats
  frameIndex = 0;
  ptSamples = 0;
  triangleCount = 0;
  lastFrameTime = 0;

  // Per-material bind groups cache
  private matBGCache = new WeakMap<GPUMaterial, GPUBindGroup>();

  constructor(gpu: GPUContext, camera: OrbitCamera) {
    this.gpu = gpu;
    this.camera = camera;
    this.commonCode = commonWGSL + '\n';
    this.settings = {
      taa: true, ssr: true, ssao: true, contactShadows: true, bloom: false,
      skybox: true, exposure: 0, tonemapMode: 0, iblIntensity: 1,
      gamma: 2.2, saturation: 1, contrast: 1, vignette: 0,
      renderMode: 'realtime',
    };
  }

  async init(): Promise<void> {
    const { device } = this.gpu;

    this.linearSampler = createLinearSampler(device, 'repeat');
    this.nearestSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
    this.shadowSampler = device.createSampler({
      compare: 'less-equal',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    // Create default 1×1 white texture
    this.whiteTexture = createTextureFromData(device, 1, 1, new Uint8Array([255,255,255,255]), 'rgba8unorm', 'white');
    this.normalDefaultTex = createTextureFromData(device, 1, 1, new Uint8Array([128,128,255,255]), 'rgba8unorm', 'normal-default');

    // GPU uniform buffers
    this.cameraBuffer = createBuffer(device, 512, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    // LightBuffer: 16-byte header (count + 12 pad) + 16 lights × 64 bytes = 1040 bytes
    this.lightBuffer  = createBuffer(device, 1056, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.shadowDataBuffer = createBuffer(device, 80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.tonemapParamsBuffer = createBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.ssrParamsBuffer = createBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.ptParamsBuffer = createBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    // Default IBL
    this.ibl = createDefaultIBL(this.gpu);
    this.envTexture = this.whiteTexture; // placeholder until HDRI loaded

    // Add a default light
    const sun = createLight('directional');
    sun.direction = [-0.5, -1.0, -0.7];
    sun.color = [1.0, 0.95, 0.9];
    sun.intensity = 3;
    this.lights.push(sun);

    this.resize();
    await this.buildPipelines();
  }

  resize(): void {
    const { device, canvas } = this.gpu;
    const w = canvas.width;
    const h = canvas.height;

    this.hdrTexture?.destroy();
    this.hdrTexturePrev?.destroy();
    this.taaResolveTexture?.destroy();
    this.depthTexture?.destroy();
    this.ptOutputTex?.destroy();
    this.ptAccBuffer?.destroy();

    const hdrUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    this.hdrTexture = device.createTexture({ size: [w,h], format: 'rgba16float', usage: hdrUsage, label: 'hdr' });
    this.hdrTexturePrev = device.createTexture({ size: [w,h], format: 'rgba16float', usage: hdrUsage | GPUTextureUsage.COPY_DST, label: 'hdr-prev' });
    this.taaResolveTexture = device.createTexture({ size: [w,h], format: 'rgba16float', usage: hdrUsage, label: 'taa-resolve' });
    this.depthTexture  = device.createTexture({ size: [w,h], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, label: 'depth' });
    this.ptOutputTex   = device.createTexture({ size: [w,h], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, label: 'pt-output' });
    this.ptAccBuffer   = createBuffer(device, w * h * 16, GPUBufferUsage.STORAGE);

    if (!this.shadowDepth) {
      this.shadowDepth = device.createTexture({
        size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        label: 'shadow-depth',
      });
    }

    this.ptSamples = 0; // reset on resize
  }

  private async buildPipelines(): Promise<void> {
    const { device, format } = this.gpu;

    // ── PBR pipeline ─────────────────────────────────────────────────────────
    const pbrModule = device.createShaderModule({ code: this.commonCode + pbrWGSL, label: 'pbr' });

    // Bind group layouts for PBR (4 groups max — WebGPU minimum guarantee)
    const bgl0 = device.createBindGroupLayout({ // group 0: camera + lights + shadow
      label: 'bgl-scene',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });
    const bgl1 = device.createBindGroupLayout({ // group 1: IBL
      label: 'bgl-ibl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const bgl2 = device.createBindGroupLayout({ // group 2: transform
      label: 'bgl-transform',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }],
    });
    const bgl3 = device.createBindGroupLayout({ // group 3: material + textures
      label: 'bgl-material',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    const pbrLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2, bgl3] });

    this.pbrPipeline = await device.createRenderPipelineAsync({
      label: 'pbr',
      layout: pbrLayout,
      vertex: {
        module: pbrModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 48,
          attributes: [
            { shaderLocation: 0, offset:  0, format: 'float32x3' }, // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
            { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv
            { shaderLocation: 3, offset: 32, format: 'float32x4' }, // tangent
          ],
        }],
      },
      fragment: {
        module: pbrModule,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba16float' }],
      },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    });

    // Store layouts for bind group creation (4 groups)
    (this.pbrPipeline as any)._bgls = [bgl0, bgl1, bgl2, bgl3];

    // ── Shadow pipeline ───────────────────────────────────────────────────────
    const shadowModule = device.createShaderModule({ code: shadowWGSL, label: 'shadow' });
    const shadowLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }] }),
        device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }] }),
      ],
    });
    this.shadowPipeline = await device.createRenderPipelineAsync({
      label: 'shadow',
      layout: shadowLayout,
      vertex: { module: shadowModule, entryPoint: 'vs_main', buffers: [{ arrayStride: 48, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'front' }, // reverse for peter panning
    });

    // ── Skybox pipeline ───────────────────────────────────────────────────────
    const skyModule = device.createShaderModule({ code: this.commonCode + skyboxWGSL, label: 'skybox' });
    const skyBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
      ],
    });
    this.skyboxPipeline = await device.createRenderPipelineAsync({
      label: 'skybox',
      layout: device.createPipelineLayout({ bindGroupLayouts: [skyBgl] }),
      vertex: { module: skyModule, entryPoint: 'vs_main' },
      fragment: { module: skyModule, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' },
      primitive: { topology: 'triangle-list' },
    });
    (this.skyboxPipeline as any)._bgl = skyBgl;

    // ── TAA pipeline ──────────────────────────────────────────────────────────
    const taaModule = device.createShaderModule({ code: this.commonCode + taaWGSL, label: 'taa' });
    const taaBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.taaPipeline = await device.createRenderPipelineAsync({
      label: 'taa',
      layout: device.createPipelineLayout({ bindGroupLayouts: [taaBgl] }),
      vertex: { module: taaModule, entryPoint: 'vs_main' },
      fragment: { module: taaModule, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    (this.taaPipeline as any)._bgl = taaBgl;

    // ── SSR pipeline ──────────────────────────────────────────────────────────
    const ssrModule = device.createShaderModule({ code: this.commonCode + ssrWGSL, label: 'ssr' });
    const ssrBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
      ],
    });
    this.ssrPipeline = await device.createRenderPipelineAsync({
      label: 'ssr',
      layout: device.createPipelineLayout({ bindGroupLayouts: [ssrBgl] }),
      vertex: { module: ssrModule, entryPoint: 'vs_main' },
      fragment: { module: ssrModule, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    (this.ssrPipeline as any)._bgl = ssrBgl;

    // ── Tonemap pipeline ──────────────────────────────────────────────────────
    const tmModule = device.createShaderModule({ code: this.commonCode + tonemapWGSL, label: 'tonemap' });
    const tmBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
      ],
    });
    this.tonemapPipeline = await device.createRenderPipelineAsync({
      label: 'tonemap',
      layout: device.createPipelineLayout({ bindGroupLayouts: [tmBgl] }),
      vertex: { module: tmModule, entryPoint: 'vs_main' },
      fragment: { module: tmModule, entryPoint: 'fs_main', targets: [{ format: this.gpu.format }] },
      primitive: { topology: 'triangle-list' },
    });
    (this.tonemapPipeline as any)._bgl = tmBgl;

    // ── Path tracing pipelines ────────────────────────────────────────────────
    const ptModule = device.createShaderModule({ code: this.commonCode + pathtraceWGSL, label: 'pathtrace' });
    const ptBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: {} },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: {} },
      ],
    });
    this.ptPipeline = device.createComputePipeline({
      label: 'pt-accumulate',
      layout: device.createPipelineLayout({ bindGroupLayouts: [ptBgl] }),
      compute: { module: ptModule, entryPoint: 'cs_main' },
    });
    (this.ptPipeline as any)._bgl = ptBgl;

    // PT display pipeline uses a separate shader module (no binding conflict)
    const ptDispModule = device.createShaderModule({ code: PT_DISPLAY_WGSL, label: 'pt-display' });
    const ptDispBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only' } },
      ],
    });
    this.ptDisplayPipeline = device.createComputePipeline({
      label: 'pt-display',
      layout: device.createPipelineLayout({ bindGroupLayouts: [ptDispBgl] }),
      compute: { module: ptDispModule, entryPoint: 'cs_display' },
    });
    (this.ptDisplayPipeline as any)._bgl = ptDispBgl;
  }

  setScene(scene: LoadedScene): void {
    this.scene = scene;
    this.triangleCount = scene.allPrimitives.reduce((s, { primitive: p }) =>
      s + (p.indexBuffer ? p.indexCount / 3 : p.vertexCount / 3), 0);
    this.ptSamples = 0;
    this.matBGCache = new WeakMap();
  }

  setIBL(ibl: IBLTextures, envTexture: GPUTexture): void {
    this.ibl = ibl;
    this.envTexture = envTexture;
    this.ptSamples = 0;
  }

  render(dt: number): void {
    const { device, canvas, context } = this.gpu;
    const w = canvas.width;
    const h = canvas.height;
    this.lastFrameTime = dt;

    this.camera.update(w / h);
    this._uploadCameraUniforms();
    this._uploadLightUniforms();
    this._uploadTonemapParams();
    this._uploadSSRParams();

    const enc = device.createCommandEncoder();

    if (this.settings.renderMode === 'pathtrace') {
      this._renderPathTrace(enc, w, h);
    } else {
      this._renderRealtime(enc, w, h);
    }

    device.queue.submit([enc.finish()]);
    const submittedBuffers = this.transientBuffers;
    this.transientBuffers = [];
    void device.queue.onSubmittedWorkDone().then(() => {
      for (const buffer of submittedBuffers) buffer.destroy();
    }).catch(() => {
      // If the device is lost, the buffers are no longer useful anyway.
    });
    this.frameIndex++;
  }

  private _trackTransientBuffer(buffer: GPUBuffer): GPUBuffer {
    this.transientBuffers.push(buffer);
    return buffer;
  }

  private _renderRealtime(enc: GPUCommandEncoder, w: number, h: number): void {
    const { context } = this.gpu;

    // 1. Shadow pass
    this._shadowPass(enc);

    // 2. Forward PBR pass → hdrTexture
    this._forwardPass(enc);

    // 3. Skybox
    if (this.settings.skybox) {
      this._skyboxPass(enc);
    }

    // 4. SSR (writes back into hdrTexture via temp)
    if (this.settings.ssr && this.scene) {
      this._ssrPass(enc);
    }

    const resolvePlan = getRealtimeResolvePlan(this.settings.taa);

    // 5. TAA resolve: hdrTexture + hdrTexturePrev → taaResolveTexture
    const resolvedTex = this.settings.taa ? this._taaPass(enc) : this.hdrTexture;

    // 6. Tonemap → swapchain
    this._tonemapPass(enc, resolvedTex, context.getCurrentTexture().createView());

    // Copy resolved to history for next frame
    if (resolvePlan.shouldCopyResolvedToHistory) {
      enc.copyTextureToTexture({ texture: resolvedTex }, { texture: this.hdrTexturePrev }, [w, this.gpu.canvas.height]);
    }
  }

  private _renderPathTrace(enc: GPUCommandEncoder, w: number, h: number): void {
    const { device, context } = this.gpu;

    // First render forward pass as primary hits
    this._forwardPass(enc);
    if (this.settings.skybox) this._skyboxPass(enc);

    // Path trace accumulation
    const ptBgl = (this.ptPipeline as any)._bgl as GPUBindGroupLayout;

    const reset = this.ptSamples === 0 ? 1 : 0;
    const ptParams = new Float32Array(8);
    ptParams[0] = 1; // samples per frame
    ptParams[1] = this.frameIndex;
    ptParams[2] = 3; // max bounces
    ptParams[3] = reset;
    ptParams[4] = this.settings.iblIntensity;
    uploadBuffer(device, this.ptParamsBuffer, ptParams);

    const ptBG = device.createBindGroup({
      layout: ptBgl,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.ptAccBuffer } },
        { binding: 2, resource: this.hdrTexture.createView() },
        { binding: 3, resource: this.depthTexture.createView() },
        { binding: 4, resource: this.envTexture.createView() },
        { binding: 5, resource: this.ibl.envSampler },
        { binding: 6, resource: { buffer: this.ptParamsBuffer } },
      ],
    });

    const ptPass = enc.beginComputePass();
    ptPass.setPipeline(this.ptPipeline);
    ptPass.setBindGroup(0, ptBG);
    ptPass.dispatchWorkgroups(Math.ceil(w/8), Math.ceil(h/8));
    ptPass.end();

    // Display
    const ptDispBgl = (this.ptDisplayPipeline as any)._bgl as GPUBindGroupLayout;
    const ptDispBG = device.createBindGroup({
      layout: ptDispBgl,
      entries: [
        { binding: 0, resource: { buffer: this.ptAccBuffer } },
        { binding: 1, resource: this.ptOutputTex.createView() },
      ],
    });

    const dispPass = enc.beginComputePass();
    dispPass.setPipeline(this.ptDisplayPipeline);
    dispPass.setBindGroup(0, ptDispBG);
    dispPass.dispatchWorkgroups(Math.ceil(w/8), Math.ceil(h/8));
    dispPass.end();

    // Tonemap pt output
    this._tonemapPass(enc, this.ptOutputTex, context.getCurrentTexture().createView());
    this.ptSamples++;
  }

  private _shadowPass(enc: GPUCommandEncoder): void {
    if (!this.scene || this.lights.length === 0) return;
    const { device } = this.gpu;

    // Use first directional or first light for shadow
    const shadowLight = this.lights.find(l => l.type === 'directional') ?? this.lights[0];

    // Compute light VP (orthographic for directional, perspective for point)
    const lightVP = this._computeLightVP(shadowLight);
    const shadowData = new Float32Array(20);
    shadowData.set(lightVP, 0);
    shadowData[16] = 0.001;  // bias
    shadowData[17] = 2.0;    // pcf radius
    uploadBuffer(device, this.shadowDataBuffer, shadowData);

    // Upload light VP for shadow pass
    const lightVPBuffer = this._trackTransientBuffer(createBuffer(
      device,
      64,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      new Float32Array(lightVP),
    ));

    const shadowBgl0 = this.shadowPipeline.getBindGroupLayout(0);
    const lightVPBG = device.createBindGroup({
      layout: shadowBgl0,
      entries: [{ binding: 0, resource: { buffer: lightVPBuffer } }],
    });

    const shadowPass = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowDepth.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    shadowPass.setPipeline(this.shadowPipeline);
    shadowPass.setViewport(0, 0, SHADOW_SIZE, SHADOW_SIZE, 0, 1);
    shadowPass.setBindGroup(0, lightVPBG);

    for (const { primitive: prim, node } of this.scene.allPrimitives) {
      const modelBuf = this._trackTransientBuffer(createBuffer(
        device,
        64,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Float32Array(node.worldMatrix),
      ));
      const modelBG = device.createBindGroup({
        layout: this.shadowPipeline.getBindGroupLayout(1),
        entries: [{ binding: 0, resource: { buffer: modelBuf } }],
      });
      shadowPass.setBindGroup(1, modelBG);
      shadowPass.setVertexBuffer(0, prim.vertexBuffer);
      if (prim.indexBuffer) {
        shadowPass.setIndexBuffer(prim.indexBuffer, prim.indexFormat);
        shadowPass.drawIndexed(prim.indexCount);
      } else {
        shadowPass.draw(prim.vertexCount);
      }
    }
    shadowPass.end();
  }

  private _forwardPass(enc: GPUCommandEncoder): void {
    const { device } = this.gpu;
    const bgls = (this.pbrPipeline as any)._bgls as GPUBindGroupLayout[];

    // Build bind group 0: scene (camera + lights + shadow data + shadow map)
    const sceneBG = device.createBindGroup({
      layout: bgls[0],
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.lightBuffer } },
        { binding: 2, resource: { buffer: this.shadowDataBuffer } },
        { binding: 3, resource: this.shadowDepth.createView() },
        { binding: 4, resource: this.shadowSampler },
      ],
    });

    // Build bind group 1: IBL
    const iblBG = device.createBindGroup({
      layout: bgls[1],
      entries: [
        { binding: 0, resource: this.ibl.envSampler },
        { binding: 1, resource: this.ibl.irradiance.createView() },
        { binding: 2, resource: this.ibl.prefiltered.createView() },
        { binding: 3, resource: this.ibl.brdfLut.createView() },
        { binding: 4, resource: this.ibl.lutSampler },
      ],
    });

    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.hdrTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setPipeline(this.pbrPipeline);
    pass.setBindGroup(0, sceneBG);
    pass.setBindGroup(1, iblBG);

    if (this.scene) {
      for (const { primitive: prim, node } of this.scene.allPrimitives) {
        const mat = this.scene.materials[Math.min(prim.materialIndex, this.scene.materials.length - 1)];

        // Transform bind group (group 2)
        const transformBG = device.createBindGroup({
          layout: bgls[2],
          entries: [{ binding: 0, resource: { buffer: node.modelBuffer } }],
        });

        // Material bind group (group 3) — cached
        let matBG = this.matBGCache.get(mat);
        if (!matBG) {
          const tex = (t: GPUTexture | null) => (t ?? this.whiteTexture).createView();
          matBG = device.createBindGroup({
            layout: bgls[3],
            entries: [
              { binding: 0, resource: { buffer: mat.uniformBuffer } },
              { binding: 1, resource: this.linearSampler },
              { binding: 2, resource: tex(mat.baseColorTexture) },
              { binding: 3, resource: (mat.normalTexture ?? this.normalDefaultTex).createView() },
              { binding: 4, resource: tex(mat.metallicRoughnessTexture) },
              { binding: 5, resource: tex(mat.occlusionTexture) },
              { binding: 6, resource: tex(mat.emissiveTexture) },
            ],
          });
          this.matBGCache.set(mat, matBG);
        }

        pass.setBindGroup(2, transformBG);
        pass.setBindGroup(3, matBG);
        pass.setVertexBuffer(0, prim.vertexBuffer);

        if (prim.indexBuffer) {
          pass.setIndexBuffer(prim.indexBuffer, prim.indexFormat);
          pass.drawIndexed(prim.indexCount);
        } else {
          pass.draw(prim.vertexCount);
        }
      }
    }
    pass.end();
  }

  private _skyboxPass(enc: GPUCommandEncoder): void {
    const { device } = this.gpu;
    const bgl = (this.skyboxPipeline as any)._bgl as GPUBindGroupLayout;

    const exposureBuf = this._trackTransientBuffer(createBuffer(
      device,
      SKYBOX_PARAMS_BUFFER_SIZE,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      createSkyboxParamsData(this.settings.exposure),
    ));

    const skyBG = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: this.ibl.envSampler },
        { binding: 2, resource: this.envTexture.createView() },
        { binding: 3, resource: { buffer: exposureBuf } },
      ],
    });

    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.hdrTexture.createView(), loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depthTexture.createView(), depthLoadOp: 'load', depthStoreOp: 'discard' },
    });
    pass.setPipeline(this.skyboxPipeline);
    pass.setBindGroup(0, skyBG);
    pass.draw(3);
    pass.end();
  }

  private _ssrPass(enc: GPUCommandEncoder): void {
    // SSR reads from hdrTexture and depth, outputs to a temp
    // For simplicity, skip if device can't handle it
    // (full implementation would use a temp texture and blend)
  }

  private _taaPass(enc: GPUCommandEncoder): GPUTexture {
    const { device } = this.gpu;
    const bgl = (this.taaPipeline as any)._bgl as GPUBindGroupLayout;

    const taaBG = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: this.hdrTexture.createView() },
        { binding: 2, resource: this.hdrTexturePrev.createView() },
        { binding: 3, resource: this.depthTexture.createView() },
        { binding: 4, resource: createLinearSampler(device, 'clamp-to-edge') },
      ],
    });

    // Render TAA resolve into a separate texture so history stays read-only this frame.
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.taaResolveTexture.createView(), loadOp: 'clear', clearValue: {r:0,g:0,b:0,a:1}, storeOp: 'store' }],
    });
    pass.setPipeline(this.taaPipeline);
    pass.setBindGroup(0, taaBG);
    pass.draw(3);
    pass.end();

    return this.taaResolveTexture;
  }

  private _tonemapPass(enc: GPUCommandEncoder, input: GPUTexture, outputView: GPUTextureView): void {
    const { device } = this.gpu;
    const bgl = (this.tonemapPipeline as any)._bgl as GPUBindGroupLayout;

    const tmBG = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: input.createView() },
        { binding: 1, resource: createLinearSampler(device, 'clamp-to-edge') },
        { binding: 2, resource: { buffer: this.tonemapParamsBuffer } },
      ],
    });

    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: outputView, loadOp: 'clear', clearValue: {r:0,g:0,b:0,a:1}, storeOp: 'store' }],
    });
    pass.setPipeline(this.tonemapPipeline);
    pass.setBindGroup(0, tmBG);
    pass.draw(3);
    pass.end();
  }

  private _uploadCameraUniforms(): void {
    const { device } = this.gpu;
    // Camera struct: 5×mat4 + vec3+f32 + f32×4 + vec2+vec2 = 5×64 + 16 + 16 + 16 = 368 bytes → pad to 384
    const data = new Float32Array(96); // 384 bytes
    data.set(this.camera.view,          0);   // off 0
    data.set(this.camera.proj,         16);   // off 64
    data.set(this.camera.viewProj,     32);   // off 128
    data.set(this.camera.invViewProj,  48);   // off 192
    data.set(this.camera.prevViewProj, 64);   // off 256
    const pos = this.camera.position;
    data[80] = pos[0]; data[81] = pos[1]; data[82] = pos[2];
    data[83] = this.camera.near;
    data[84] = this.camera.far;
    data[85] = this.camera.fovY;
    data[86] = this.camera.aspect;
    data[87] = this.camera.frameIndex;
    const jitter = this.camera.jitter;
    data[88] = jitter[0]; data[89] = jitter[1];
    uploadBuffer(device, this.cameraBuffer, data);
  }

  private _uploadLightUniforms(): void {
    uploadBuffer(this.gpu.device, this.lightBuffer, packLights(this.lights));
  }

  private _uploadTonemapParams(): void {
    const s = this.settings;
    const data = new Float32Array(8);
    data[0] = s.exposure; data[1] = s.tonemapMode;
    data[2] = s.gamma; data[3] = s.saturation;
    data[4] = s.contrast; data[5] = s.vignette;
    uploadBuffer(this.gpu.device, this.tonemapParamsBuffer, data);
  }

  private _uploadSSRParams(): void {
    const data = new Float32Array(8);
    data[0] = 32; data[1] = 5.0; data[2] = 0.15; data[3] = 0.1;
    data[4] = 0.6; data[5] = 0.5;
    uploadBuffer(this.gpu.device, this.ssrParamsBuffer, data);
  }

  private _computeLightVP(light: Light): Float32Array {
    const size = 10;
    const pos: [number,number,number] = [-light.direction[0]*8, -light.direction[1]*8, -light.direction[2]*8];
    const target: [number,number,number] = [0, 0, 0];
    const view = mat4Create(); mat4LookAt(view, pos, target, [0,1,0]);
    const proj = mat4Create(); mat4OrthoRH(proj, -size, size, -size, size, 0.1, 50);
    const vp = mat4Create(); mat4Multiply(vp, proj, view);
    return vp;
  }

  resetPTAccumulation(): void {
    this.ptSamples = 0;
  }

  loadHDRIFromData(hdr: HDRImage): GPUTexture {
    const { device } = this.gpu;
    const tex = device.createTexture({
      size: [hdr.width, hdr.height],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: tex }, hdr.data.buffer as ArrayBuffer,
      { offset: hdr.data.byteOffset, bytesPerRow: hdr.width * 16 }, [hdr.width, hdr.height]);
    return tex;
  }
}
