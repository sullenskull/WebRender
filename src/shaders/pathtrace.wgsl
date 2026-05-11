// ─── Progressive Path Tracing (compute shader) ───────────────────────────────
// Screen-space: primary hits from G-Buffer, secondary from ray march.
// Each dispatch adds N samples to the accumulation buffer.

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(1) var<storage, read_write> accumulator: array<vec4f>;  // w = sample count
@group(0) @binding(2) var hdr_in:     texture_2d<f32>;    // forward-pass color (primary hit)
@group(0) @binding(3) var depth_tex:  texture_depth_2d;
@group(0) @binding(4) var env_tex:    texture_2d<f32>;
@group(0) @binding(5) var env_sampler:sampler;
@group(0) @binding(6) var<uniform> pt_params: PTParams;

struct PTParams {
  sample_count: u32,   // samples this dispatch
  frame_index:  u32,   // global frame counter
  max_bounces:  u32,
  reset:        u32,   // 1 = reset accumulator
  env_intensity:f32,
  _pad:         vec3f,
}

// ─── PCG random ──────────────────────────────────────────────────────────────
fn pcg_hash(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand_float(seed: ptr<function, u32>) -> f32 {
  *seed = pcg_hash(*seed);
  return f32(*seed) * 2.3283064365e-10; // / 2^32
}

fn rand_vec2(seed: ptr<function, u32>) -> vec2f {
  return vec2f(rand_float(seed), rand_float(seed));
}

// ─── Cosine hemisphere sampling ───────────────────────────────────────────────
fn cosine_hemisphere(xi: vec2f, N: vec3f) -> vec3f {
  let phi       = 2.0 * PI * xi.x;
  let cos_theta = sqrt(xi.y);
  let sin_theta = sqrt(1.0 - xi.y);
  let up = select(vec3f(1,0,0), vec3f(0,1,0), abs(N.y) < 0.999);
  let T  = normalize(cross(up, N));
  let B  = cross(N, T);
  return normalize(T * (cos(phi)*sin_theta) + B * (sin(phi)*sin_theta) + N * cos_theta);
}

// ─── GGX importance sampling ─────────────────────────────────────────────────
fn ggx_sample(xi: vec2f, N: vec3f, roughness: f32) -> vec3f {
  let a     = roughness * roughness;
  let phi   = 2.0 * PI * xi.x;
  let cosT  = sqrt((1.0 - xi.y) / (1.0 + (a*a - 1.0) * xi.y));
  let sinT  = sqrt(max(0.0, 1.0 - cosT*cosT));
  let H_local = vec3f(cos(phi)*sinT, sin(phi)*sinT, cosT);
  let up = select(vec3f(1,0,0), vec3f(0,1,0), abs(N.y) < 0.999);
  let T  = normalize(cross(up, N));
  let B  = cross(N, T);
  return normalize(T*H_local.x + B*H_local.y + N*H_local.z);
}

// Screen-space ray march (for secondary bounce)
fn trace_ss(ray_dir: vec3f, ray_origin: vec3f) -> vec3f {
  let steps = 32u;
  let step_size = 0.08;

  for (var i = 1u; i <= steps; i++) {
    let p    = ray_origin + ray_dir * (f32(i) * step_size);
    let clip = camera.view_proj * vec4f(p, 1.0);
    if clip.w <= 0.0 { break; }
    let ndc  = clip.xyz / clip.w;
    let uv   = ndc.xy * 0.5 + 0.5;
    if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) { break; }

    let dim  = vec2i(textureDimensions(depth_tex));
    let tc   = vec2i(uv * vec2f(dim));
    let s_d  = textureLoad(depth_tex, tc, 0);
    let clip_d = ndc.z;

    let s_lin = linearize_depth(s_d, camera.near, camera.far);
    let r_lin = linearize_depth(clip_d, camera.near, camera.far);
    if r_lin > s_lin && (r_lin - s_lin) < 0.3 {
      // Hit — return color from forward pass
      return textureLoad(hdr_in, tc, 0).rgb;
    }
  }
  // Miss — sample environment
  let env_uv = dir_to_equirect(ray_dir);
  return textureSampleLevel(env_tex, env_sampler, env_uv, 0.0).rgb * pt_params.env_intensity;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let dim_2d = textureDimensions(hdr_in);
  let dim = vec2u(dim_2d.x, dim_2d.y);
  if gid.x >= dim.x || gid.y >= dim.y { return; }

  let pixel_idx = gid.y * dim.x + gid.x;
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dim);

  // Reset accumulator on camera movement
  if pt_params.reset != 0u {
    accumulator[pixel_idx] = vec4f(0.0);
  }

  let depth = textureLoad(depth_tex, vec2i(gid.xy), 0);
  var color_sum = accumulator[pixel_idx].rgb * accumulator[pixel_idx].w;
  let old_count = accumulator[pixel_idx].w;

  if depth >= 1.0 {
    // Skybox pixel — just use env
    let clip = vec4f(uv * 2.0 - 1.0, 1.0, 1.0);
    let wh   = camera.inv_view_proj * clip;
    let dir  = normalize(wh.xyz / wh.w - camera.position);
    let env  = textureSampleLevel(env_tex, env_sampler, dir_to_equirect(dir), 0.0).rgb * pt_params.env_intensity;
    color_sum += env * f32(pt_params.sample_count);
    accumulator[pixel_idx] = vec4f(color_sum, old_count + f32(pt_params.sample_count));
    return;
  }

  let world_p = world_pos_from_depth(uv, depth, camera.inv_view_proj);

  // Get normal from depth gradient (screen-space approximation)
  let dim_f   = vec2f(dim);
  let d_r = textureLoad(depth_tex, vec2i(gid.xy) + vec2i(1,0), 0);
  let d_u = textureLoad(depth_tex, vec2i(gid.xy) + vec2i(0,1), 0);
  let p_r = world_pos_from_depth((vec2f(gid.xy) + vec2f(1.5,0.5)) / dim_f, d_r, camera.inv_view_proj);
  let p_u = world_pos_from_depth((vec2f(gid.xy) + vec2f(0.5,1.5)) / dim_f, d_u, camera.inv_view_proj);
  let N   = safe_normalize(cross(p_r - world_p, p_u - world_p));

  let V   = safe_normalize(camera.position - world_p);
  // Primary color from forward pass (already has direct light + IBL)
  let primary = textureLoad(hdr_in, vec2i(gid.xy), 0).rgb;

  // Trace secondary bounce samples
  for (var s = 0u; s < pt_params.sample_count; s++) {
    var seed = pcg_hash(pixel_idx + (pt_params.frame_index + s) * 19349663u);

    // Cosine-weighted secondary ray
    let xi  = rand_vec2(&seed);
    let L   = cosine_hemisphere(xi, N);
    let dot_nl = max(dot(N, L), 0.0);

    let hit = trace_ss(L, world_p + N * 0.02);
    let contrib = primary + hit * dot_nl * 2.0; // lambertian * pi cancels

    color_sum += contrib;
  }

  let new_count = old_count + f32(pt_params.sample_count);
  accumulator[pixel_idx] = vec4f(color_sum, new_count);
}

// (Display pass is in a separate shader module in renderer.ts — PT_DISPLAY_WGSL)
