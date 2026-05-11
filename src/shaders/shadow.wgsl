// ─── Shadow map depth pass ────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> light_vp:   mat4x4f;
@group(1) @binding(0) var<uniform> model:       mat4x4f;

struct VertIn {
  @location(0) position: vec3f,
}

@vertex
fn vs_main(v: VertIn) -> @builtin(position) vec4f {
  return light_vp * model * vec4f(v.position, 1.0);
}

// No fragment shader needed — depth is written automatically
