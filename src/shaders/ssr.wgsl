// ─── Screen-Space Reflections (SSR) ──────────────────────────────────────────
// Ray-march in view space along reflection direction, sample if hit.

@group(0) @binding(0) var<uniform> camera:    Camera;
@group(0) @binding(1) var hdr_tex:    texture_2d<f32>;  // current frame HDR color
@group(0) @binding(2) var depth_tex:  texture_depth_2d;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;  // world-space normals (in HDR pass)
@group(0) @binding(4) var lin_sampler:sampler;
@group(0) @binding(5) var<uniform> ssr_params: SSRParams;

struct SSRParams {
  max_steps:     u32,
  max_distance:  f32,
  thickness:     f32,
  stride:        f32,
  intensity:     f32,
  roughness_cut: f32,  // skip SSR when roughness > this
  _pad:          vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  return vec4f(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
  let dim  = vec2f(textureDimensions(hdr_tex));
  let uv   = frag_coord.xy / dim;
  let tc   = vec2i(frag_coord.xy);

  // Sample current pixel color
  let color = textureLoad(hdr_tex, tc, 0).rgb;

  // Reconstruct world position
  let depth = textureLoad(depth_tex, tc, 0);
  if depth >= 1.0 { return vec4f(color, 1.0); } // skybox

  let world_p = world_pos_from_depth(uv, depth, camera.inv_view_proj);

  // Get normal (packed in hdr output as second RT, simplified: skip normal RT)
  // For a real impl we'd read from a G-Buffer normal RT.
  // Here we approximate using depth gradient.
  let d_right = textureLoad(depth_tex, tc + vec2i(1, 0), 0);
  let d_up    = textureLoad(depth_tex, tc + vec2i(0, 1), 0);
  let p_right = world_pos_from_depth((frag_coord.xy + vec2f(1,0)) / dim, d_right, camera.inv_view_proj);
  let p_up    = world_pos_from_depth((frag_coord.xy + vec2f(0,1)) / dim, d_up,    camera.inv_view_proj);
  let N = safe_normalize(cross(p_right - world_p, p_up - world_p));

  let V = safe_normalize(camera.position - world_p);
  let R = reflect(-V, N);

  // Ray march in view space
  var hit_color = vec3f(0.0);
  var hit_weight = 0.0;

  let ray_origin = world_p + N * 0.01;
  let ray_step = R * (ssr_params.max_distance / f32(ssr_params.max_steps));

  for (var i = 0u; i < ssr_params.max_steps; i++) {
    let ray_pos = ray_origin + R * (f32(i) * ssr_params.stride);

    // Project to screen
    let clip = camera.view_proj * vec4f(ray_pos, 1.0);
    if clip.w <= 0.0 { break; }
    let ndc   = clip.xyz / clip.w;
    let s_uv  = ndc.xy * 0.5 + 0.5;

    if any(s_uv < vec2f(0.0)) || any(s_uv > vec2f(1.0)) { break; }

    let s_tc = vec2i(s_uv * dim);
    let s_depth = textureLoad(depth_tex, s_tc, 0);

    // Check depth
    let s_p = world_pos_from_depth(s_uv, s_depth, camera.inv_view_proj);
    let ray_depth = ndc.z;
    let scene_depth_linear = linearize_depth(s_depth, camera.near, camera.far);
    let ray_depth_linear   = linearize_depth(clip.z / clip.w, camera.near, camera.far);

    let depth_diff = ray_depth_linear - scene_depth_linear;
    if depth_diff > 0.0 && depth_diff < ssr_params.thickness {
      // Hit! Sample color
      hit_color  = textureSampleLevel(hdr_tex, lin_sampler, s_uv, 0.0).rgb;
      // Fade at edges
      let fade_uv   = abs(s_uv - 0.5) * 2.0;
      let edge_fade = 1.0 - max(fade_uv.x, fade_uv.y);
      // Fade by distance
      let dist_fade = 1.0 - f32(i) / f32(ssr_params.max_steps);
      hit_weight = edge_fade * dist_fade * ssr_params.intensity;
      break;
    }
  }

  // Blend SSR on top of original color
  let out = mix(color, hit_color, hit_weight);
  return vec4f(out, 1.0);
}
