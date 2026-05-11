// ─── Skybox / HDRI background ─────────────────────────────────────────────────
// Fullscreen triangle, reconstructs world direction from NDC + inv_view_proj.

@group(0) @binding(0) var<uniform> camera:    Camera;
@group(0) @binding(1) var env_sampler: sampler;
@group(0) @binding(2) var env_tex:     texture_2d<f32>;
@group(0) @binding(3) var<uniform> sky_params: SkyParams;

struct SkyParams {
  exposure: f32,
  _pad: vec3f,
}

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0)       uv:  vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertOut {
  // Fullscreen triangle slightly behind everything (depth = 0.9999)
  var corners = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 1.0),
    vec2f(2.0, 1.0),
    vec2f(0.0,-1.0),
  );
  var out: VertOut;
  out.pos = vec4f(corners[idx], 0.9999, 1.0);
  out.uv  = uvs[idx];
  return out;
}

@fragment
fn fs_main(v: VertOut) -> @location(0) vec4f {
  // Reconstruct world direction from UV (NDC)
  let ndc    = v.uv * 2.0 - 1.0;
  let clip_h = vec4f(ndc, 1.0, 1.0);
  let world_h = camera.inv_view_proj * clip_h;
  let dir    = normalize(world_h.xyz / world_h.w - camera.position);

  let env_uv = dir_to_equirect(dir);
  var color  = textureSample(env_tex, env_sampler, env_uv).rgb;
  color     *= pow(2.0, sky_params.exposure);
  return vec4f(color, 1.0);
}
