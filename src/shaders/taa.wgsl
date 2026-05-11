// ─── Temporal Anti-Aliasing (TAA) ─────────────────────────────────────────────
// Resolve: blend current jittered frame with reprojected history.
// Uses YCoCg color space + neighborhood clamping to reduce ghosting.

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(1) var current_tex:  texture_2d<f32>;   // current HDR frame (jittered)
@group(0) @binding(2) var history_tex:  texture_2d<f32>;   // previous resolved frame
@group(0) @binding(3) var depth_tex:    texture_depth_2d;  // current depth
@group(0) @binding(4) var lin_sampler:  sampler;

// fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  return vec4f(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
  let dim = vec2f(textureDimensions(current_tex));
  let dim_i = vec2i(textureDimensions(current_tex));
  let uv  = frag_coord.xy / dim;

  // Sample current pixel and 3x3 neighborhood for AABB clamping
  let tc = vec2i(frag_coord.xy);
  var c_min = vec3f( 1e8);
  var c_max = vec3f(-1e8);
  var c_current = vec3f(0.0);

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let sample_tc = clamp(tc + vec2i(dx, dy), vec2i(0), dim_i - vec2i(1));
      let s = textureLoad(current_tex, sample_tc, 0).rgb;
      let ycocg = rgb_to_ycocg(s);
      c_min = min(c_min, ycocg);
      c_max = max(c_max, ycocg);
      if dx == 0 && dy == 0 { c_current = ycocg; }
    }
  }

  // Reproject: find where this pixel was in the previous frame
  let depth   = textureLoad(depth_tex, tc, 0);
  let world_p = world_pos_from_depth(uv, depth, camera.inv_view_proj);
  let prev_clip = camera.prev_view_proj * vec4f(world_p, 1.0);
  let prev_ndc  = prev_clip.xy / prev_clip.w;
  let prev_uv   = prev_ndc * 0.5 + 0.5;

  // Sample history (remove jitter from UV)
  let prev_uv_unjittered = prev_uv - camera.jitter * 0.5 / dim;
  var c_history = vec3f(0.0);
  var valid_history = false;
  if all(prev_uv_unjittered >= vec2f(0.0)) && all(prev_uv_unjittered <= vec2f(1.0)) {
    c_history = rgb_to_ycocg(textureSampleLevel(history_tex, lin_sampler, prev_uv_unjittered, 0.0).rgb);
    valid_history = true;
  }

  // Neighborhood clamp to prevent ghosting
  let c_clamped = clamp(c_history, c_min, c_max);

  // Blend weight: more history weight = smoother but more ghosting
  let blend = select(1.0, 0.1, valid_history);
  let result_ycocg = mix(c_clamped, c_current, blend);

  return vec4f(ycocg_to_rgb(result_ycocg), 1.0);
}
