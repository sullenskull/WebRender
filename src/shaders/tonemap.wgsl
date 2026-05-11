// ─── Tone mapping + Gamma correction pass ────────────────────────────────────

@group(0) @binding(0) var hdr_tex:     texture_2d<f32>;
@group(0) @binding(1) var lin_sampler: sampler;
@group(0) @binding(2) var<uniform> params: TonemapParams;

struct TonemapParams {
  exposure:    f32,
  mode:        u32,   // 0=AgX 1=ACES 2=Reinhard 3=Linear
  gamma:       f32,
  saturation:  f32,
  contrast:    f32,
  vignette:    f32,
  _pad:        vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  return vec4f(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
  let dim = vec2f(textureDimensions(hdr_tex));
  let uv  = frag_coord.xy / dim;

  var color = textureSample(hdr_tex, lin_sampler, uv).rgb;

  // Apply contrast (in linear space)
  let contrast = params.contrast;
  if abs(contrast - 1.0) > 0.001 {
    color = pow(max(color, vec3f(0.0)), vec3f(contrast));
  }

  // Apply saturation
  if abs(params.saturation - 1.0) > 0.001 {
    let lum = luminance(color);
    color = mix(vec3f(lum), color, params.saturation);
  }

  // Tone mapping
  var ldr: vec3f;
  switch params.mode {
    case 0u: { ldr = tonemap_agx(color, params.exposure); }
    case 1u: { ldr = tonemap_aces(color, params.exposure); }
    case 2u: { ldr = tonemap_reinhard(color, params.exposure); }
    default: {
      ldr = color * pow(2.0, params.exposure);
      ldr = clamp(ldr, vec3f(0.0), vec3f(1.0));
    }
  }

  // Gamma correction
  ldr = pow(max(ldr, vec3f(0.0)), vec3f(1.0 / params.gamma));

  // Vignette
  if params.vignette > 0.001 {
    let uv_c = uv - 0.5;
    let vig = 1.0 - params.vignette * dot(uv_c, uv_c) * 4.0;
    ldr *= clamp(vig, 0.0, 1.0);
  }

  return vec4f(ldr, 1.0);
}
