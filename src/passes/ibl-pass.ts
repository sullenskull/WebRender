// Precomputes irradiance map, prefiltered specular map, and BRDF LUT from an equirectangular HDRI.
// Each pass uses its own shader module to avoid binding conflicts.
import type { GPUContext } from '../core/gpu.js';
import { createLinearSampler } from '../core/gpu.js';
import commonWGSL from '../shaders/common.wgsl?raw';

const IBL_SIZE = 64;        // irradiance map
const PREFILTER_SIZE = 128; // prefiltered specular
const BRDF_SIZE = 128;
const PREFILTER_MIPS = 5;

export interface IBLTextures {
  irradiance: GPUTexture;
  prefiltered: GPUTexture;
  brdfLut: GPUTexture;
  envSampler: GPUSampler;
  lutSampler: GPUSampler;
}

// ─── Irradiance shader ────────────────────────────────────────────────────────
const IRRADIANCE_WGSL = `
@group(0) @binding(0) var env_tex:  texture_2d<f32>;
@group(0) @binding(1) var env_samp: sampler;
@group(0) @binding(2) var irr_out:  texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_irradiance(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(irr_out);
  if gid.x >= dim.x || gid.y >= dim.y { return; }
  let uv    = (vec2f(gid.xy) + 0.5) / vec2f(dim);
  let phi   = uv.x * 2.0 * PI;
  let theta = (1.0 - uv.y) * PI;
  let N = vec3f(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
  let up = select(vec3f(1,0,0), vec3f(0,1,0), abs(N.y) < 0.999);
  let T  = normalize(cross(up, N));
  let B  = cross(N, T);
  var irr = vec3f(0.0);
  var n_s  = 0u;
  let steps = 24u;
  let delta = PI / f32(steps);
  for (var pi = 0u; pi < steps * 2u; pi++) {
    for (var ti = 0u; ti < steps / 2u; ti++) {
      let sp = f32(pi) * delta;
      let st = f32(ti) * delta;
      let d_ts = vec3f(sin(st)*cos(sp), sin(st)*sin(sp), cos(st));
      let d_ws = T*d_ts.x + B*d_ts.y + N*d_ts.z;
      irr += textureSampleLevel(env_tex, env_samp, dir_to_equirect(d_ws), 0.0).rgb * cos(st) * sin(st);
      n_s++;
    }
  }
  irr = PI * irr / f32(n_s);
  textureStore(irr_out, vec2i(gid.xy), vec4f(irr, 1.0));
}
`;

// ─── Prefilter shader ─────────────────────────────────────────────────────────
const PREFILTER_WGSL = `
@group(0) @binding(0) var env_tex:   texture_2d<f32>;
@group(0) @binding(1) var env_samp:  sampler;
@group(0) @binding(2) var pf_out:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform>  roughness: f32;

fn van_der_corput(n_in: u32) -> f32 {
  var n = n_in; var vdc = 0.0; var denom = 1.0;
  for (var k = 0u; k < 20u; k++) {
    denom *= 2.0;
    if (n & 1u) != 0u { vdc += 1.0/denom; }
    n >>= 1u;
    if n == 0u { break; }
  }
  return vdc;
}

@compute @workgroup_size(8, 8)
fn cs_prefilter(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(pf_out);
  if gid.x >= dim.x || gid.y >= dim.y { return; }
  let uv  = (vec2f(gid.xy) + 0.5) / vec2f(dim);
  let phi   = uv.x * 2.0 * PI;
  let theta = (1.0 - uv.y) * PI;
  let N = vec3f(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
  let up = select(vec3f(1,0,0), vec3f(0,1,0), abs(N.y) < 0.999);
  let T  = normalize(cross(up, N));
  let B  = cross(N, T);
  let n_samples = 64u;
  var pf  = vec3f(0.0);
  var wsum = 0.0;
  for (var i = 0u; i < n_samples; i++) {
    let xi = vec2f(f32(i) / f32(n_samples), van_der_corput(i));
    let H  = importance_sample_GGX(xi, N, roughness);
    let L  = normalize(2.0 * dot(N, H) * H - N);
    let NoL = max(dot(N, L), 0.0);
    if NoL > 0.0 {
      pf   += textureSampleLevel(env_tex, env_samp, dir_to_equirect(L), 0.0).rgb * NoL;
      wsum += NoL;
    }
  }
  textureStore(pf_out, vec2i(gid.xy), vec4f(pf / max(wsum, 0.001), 1.0));
}
`;

// ─── BRDF LUT shader ─────────────────────────────────────────────────────────
const BRDF_LUT_WGSL = `
@group(0) @binding(0) var brdf_out: texture_storage_2d<rg16float, write>;

fn van_der_corput(n_in: u32) -> f32 {
  var n = n_in; var vdc = 0.0; var denom = 1.0;
  for (var k = 0u; k < 20u; k++) {
    denom *= 2.0;
    if (n & 1u) != 0u { vdc += 1.0/denom; }
    n >>= 1u;
    if n == 0u { break; }
  }
  return vdc;
}

@compute @workgroup_size(8, 8)
fn cs_brdf_lut(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(brdf_out);
  if gid.x >= dim.x || gid.y >= dim.y { return; }
  let NoV  = clamp((f32(gid.x) + 0.5) / f32(dim.x), 0.001, 1.0);
  let roughness = clamp((f32(gid.y) + 0.5) / f32(dim.y), 0.045, 1.0);
  let V = vec3f(sqrt(1.0 - NoV*NoV), 0.0, NoV);
  let N = vec3f(0.0, 0.0, 1.0);
  let n_samples = 256u;
  var A = 0.0; var Bv = 0.0;
  for (var i = 0u; i < n_samples; i++) {
    let xi = vec2f(f32(i) / f32(n_samples), van_der_corput(i));
    let H  = importance_sample_GGX(xi, N, roughness);
    let L  = normalize(2.0 * dot(V, H) * H - V);
    let NoL = max(L.z, 0.0);
    let NoH = max(H.z, 0.0);
    let VoH = max(dot(V, H), 0.0);
    if NoL > 0.0 {
      let G  = V_SmithGGXCorrelated(NoV, NoL, roughness) * 4.0 * NoL * NoV;
      let Fc = pow(1.0 - VoH, 5.0);
      A  += G * (1.0 - Fc);
      Bv += G * Fc;
    }
  }
  textureStore(brdf_out, vec2i(gid.xy), vec4f(A/f32(n_samples), Bv/f32(n_samples), 0.0, 1.0));
}
`;

export async function computeIBL(gpu: GPUContext, envTexture: GPUTexture): Promise<IBLTextures> {
  const { device } = gpu;
  const prefix = commonWGSL + '\n';
  const envSampler = createLinearSampler(device, 'repeat');
  const lutSampler = createLinearSampler(device, 'clamp-to-edge');

  // ── Irradiance ────────────────────────────────────────────────────────────
  const irradiance = device.createTexture({
    size: [IBL_SIZE, IBL_SIZE], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const irrMod = device.createShaderModule({ code: prefix + IRRADIANCE_WGSL, label: 'ibl-irr' });
  const irrBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: {} },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only' } },
  ]});
  const irrPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [irrBGL] }),
    compute: { module: irrMod, entryPoint: 'cs_irradiance' },
  });
  const irrBG = device.createBindGroup({ layout: irrBGL, entries: [
    { binding: 0, resource: envTexture.createView() },
    { binding: 1, resource: envSampler },
    { binding: 2, resource: irradiance.createView() },
  ]});
  const enc1 = device.createCommandEncoder();
  const p1 = enc1.beginComputePass();
  p1.setPipeline(irrPipeline); p1.setBindGroup(0, irrBG);
  p1.dispatchWorkgroups(Math.ceil(IBL_SIZE/8), Math.ceil(IBL_SIZE/8));
  p1.end();
  device.queue.submit([enc1.finish()]);

  // ── Prefiltered ───────────────────────────────────────────────────────────
  const prefiltered = device.createTexture({
    size: [PREFILTER_SIZE, PREFILTER_SIZE], format: 'rgba16float',
    mipLevelCount: PREFILTER_MIPS,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const pfMod = device.createShaderModule({ code: prefix + PREFILTER_WGSL, label: 'ibl-pf' });
  const pfBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: {} },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});
  const pfPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [pfBGL] }),
    compute: { module: pfMod, entryPoint: 'cs_prefilter' },
  });

  for (let mip = 0; mip < PREFILTER_MIPS; mip++) {
    const rough = mip / (PREFILTER_MIPS - 1);
    const roughBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Float32Array(roughBuf.getMappedRange()).set([rough, 0, 0, 0]);
    roughBuf.unmap();
    const mipSize = Math.max(1, PREFILTER_SIZE >> mip);
    const pfBG = device.createBindGroup({ layout: pfBGL, entries: [
      { binding: 0, resource: envTexture.createView() },
      { binding: 1, resource: envSampler },
      { binding: 2, resource: prefiltered.createView({ baseMipLevel: mip, mipLevelCount: 1 }) },
      { binding: 3, resource: { buffer: roughBuf } },
    ]});
    const enc2 = device.createCommandEncoder();
    const p2 = enc2.beginComputePass();
    p2.setPipeline(pfPipeline); p2.setBindGroup(0, pfBG);
    p2.dispatchWorkgroups(Math.ceil(mipSize/8), Math.ceil(mipSize/8));
    p2.end();
    device.queue.submit([enc2.finish()]);
    roughBuf.destroy();
  }

  // ── BRDF LUT ──────────────────────────────────────────────────────────────
  const brdfLut = device.createTexture({
    size: [BRDF_SIZE, BRDF_SIZE], format: 'rg16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const brdfMod = device.createShaderModule({ code: prefix + BRDF_LUT_WGSL, label: 'ibl-brdf' });
  const brdfBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rg16float', access: 'write-only' } },
  ]});
  const brdfPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [brdfBGL] }),
    compute: { module: brdfMod, entryPoint: 'cs_brdf_lut' },
  });
  const brdfBG = device.createBindGroup({ layout: brdfBGL, entries: [
    { binding: 0, resource: brdfLut.createView() },
  ]});
  const enc3 = device.createCommandEncoder();
  const p3 = enc3.beginComputePass();
  p3.setPipeline(brdfPipeline); p3.setBindGroup(0, brdfBG);
  p3.dispatchWorkgroups(Math.ceil(BRDF_SIZE/8), Math.ceil(BRDF_SIZE/8));
  p3.end();
  device.queue.submit([enc3.finish()]);
  await device.queue.onSubmittedWorkDone();

  return { irradiance, prefiltered, brdfLut, envSampler, lutSampler };
}

// Create flat default IBL (gray ambient) for when no HDRI is loaded
export function createDefaultIBL(gpu: GPUContext): IBLTextures {
  const { device } = gpu;
  const makeFlat = (r: number, g: number, b: number, format: 'rgba8unorm' | 'rg8unorm' = 'rgba8unorm'): GPUTexture => {
    const tex = device.createTexture({
      size: [1, 1], format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const data = format === 'rgba8unorm'
      ? new Uint8Array([Math.round(r*255), Math.round(g*255), Math.round(b*255), 255])
      : new Uint8Array([Math.round(r*255), Math.round(g*255)]);
    device.queue.writeTexture({ texture: tex }, data.buffer as ArrayBuffer,
      { bytesPerRow: data.byteLength }, [1, 1]);
    return tex;
  };
  return {
    irradiance:  makeFlat(0.3, 0.3, 0.3),
    prefiltered: makeFlat(0.15, 0.15, 0.15),
    brdfLut:     makeFlat(0.5, 0.5, 0, 'rgba8unorm'), // rg only matters, using rgba8 is fine (sampleType:float)
    envSampler:  createLinearSampler(device, 'repeat'),
    lutSampler:  createLinearSampler(device, 'clamp-to-edge'),
  };
}
