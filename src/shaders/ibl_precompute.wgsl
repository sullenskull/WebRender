// ─── IBL Precomputation (compute shaders) ────────────────────────────────────
// Two passes: irradiance map + BRDF LUT

// ── Pass 1: Irradiance map (diffuse IBL) ─────────────────────────────────────
@group(0) @binding(0) var env_src:     texture_2d<f32>;
@group(0) @binding(1) var env_sampler: sampler;
@group(0) @binding(2) var irr_out:     texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_irradiance(@builtin(global_invocation_id) gid: vec3u) {
  let dim   = textureDimensions(irr_out);
  if gid.x >= dim.x || gid.y >= dim.y { return; }

  let uv  = (vec2f(gid.xy) + 0.5) / vec2f(dim);
  // Convert UV to direction
  let phi   = uv.x * 2.0 * PI;
  let theta = (1.0 - uv.y) * PI;
  let N = vec3f(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));

  let up = select(vec3f(1,0,0), vec3f(0,1,0), abs(N.y) < 0.999);
  let T  = normalize(cross(up, N));
  let B  = cross(N, T);

  var irradiance = vec3f(0.0);
  let sample_count = 64u;
  let delta = PI / f32(sample_count);

  var n_samples = 0u;
  for (var phi_i = 0u; phi_i < sample_count * 2u; phi_i++) {
    for (var theta_i = 0u; theta_i < sample_count / 2u; theta_i++) {
      let sp = f32(phi_i) * delta;
      let st = f32(theta_i) * delta;
      let sample_dir_ts = vec3f(sin(st)*cos(sp), sin(st)*sin(sp), cos(st));
      let sample_dir_ws = T*sample_dir_ts.x + B*sample_dir_ts.y + N*sample_dir_ts.z;

      let env_uv = dir_to_equirect(sample_dir_ws);
      irradiance += textureSampleLevel(env_src, env_sampler, env_uv, 0.0).rgb * cos(st) * sin(st);
      n_samples++;
    }
  }
  irradiance = PI * irradiance / f32(n_samples);
  textureStore(irr_out, vec2i(gid.xy), vec4f(irradiance, 1.0));
}

// ── Pass 2: Prefiltered specular map ─────────────────────────────────────────
@group(0) @binding(0) var env_src2:    texture_2d<f32>;
@group(0) @binding(1) var env_sampler2:sampler;
@group(0) @binding(2) var pf_out:      texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> roughness_level: f32;

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

  let roughness  = roughness_level;
  let sample_count = 128u;
  var prefiltered = vec3f(0.0);
  var weight_sum  = 0.0;

  for (var i = 0u; i < sample_count; i++) {
    // Van der Corput sequence
    var ri = i; var vdc = 0.0; var denom = 1.0;
    for (var k = 0u; k < 20u; k++) { denom *= 2.0; if (ri & 1u) != 0u { vdc += 1.0/denom; } ri >>= 1u; if ri == 0u { break; } }
    let xi = vec2f(f32(i) / f32(sample_count), vdc);
    let H  = importance_sample_GGX(xi, N, roughness);
    let L  = normalize(2.0 * dot(N, H) * H - N);
    let NoL = max(dot(N, L), 0.0);
    if NoL > 0.0 {
      let env_uv = dir_to_equirect(L);
      prefiltered += textureSampleLevel(env_src2, env_sampler2, env_uv, 0.0).rgb * NoL;
      weight_sum  += NoL;
    }
  }
  prefiltered = prefiltered / max(weight_sum, 0.001);
  textureStore(pf_out, vec2i(gid.xy), vec4f(prefiltered, 1.0));
}

// ── Pass 3: BRDF Integration LUT ─────────────────────────────────────────────
// Output: RG = scale+bias for Fresnel split-sum
@group(0) @binding(0) var brdf_out: texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8)
fn cs_brdf_lut(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(brdf_out);
  if gid.x >= dim.x || gid.y >= dim.y { return; }

  let NoV       = (f32(gid.x) + 0.5) / f32(dim.x);
  let roughness = (f32(gid.y) + 0.5) / f32(dim.y);

  let V = vec3f(sqrt(1.0 - NoV*NoV), 0.0, NoV);
  let N = vec3f(0.0, 0.0, 1.0);

  let sample_count = 1024u;
  var A = 0.0; var B_val = 0.0;

  for (var i = 0u; i < sample_count; i++) {
    var ri = i; var vdc = 0.0; var denom = 1.0;
    for (var k = 0u; k < 20u; k++) { denom *= 2.0; if (ri & 1u) != 0u { vdc += 1.0/denom; } ri >>= 1u; if ri == 0u { break; } }
    let xi = vec2f(f32(i) / f32(sample_count), vdc);
    let H  = importance_sample_GGX(xi, N, roughness);
    let L  = normalize(2.0 * dot(V, H) * H - V);
    let NoL = max(L.z, 0.0);
    let NoH = max(H.z, 0.0);
    let VoH = max(dot(V, H), 0.0);
    if NoL > 0.0 {
      let G    = V_SmithGGXCorrelated(NoV, NoL, roughness) * 4.0 * NoL * NoV;
      let Fc   = pow(1.0 - VoH, 5.0);
      A += G * (1.0 - Fc);
      B_val += G * Fc;
    }
  }
  A /= f32(sample_count);
  B_val /= f32(sample_count);
  textureStore(brdf_out, vec2i(gid.xy), vec4f(A, B_val, 0.0, 1.0));
}
