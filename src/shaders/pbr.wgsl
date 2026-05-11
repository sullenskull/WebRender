// ─── PBR Forward Shader (4 bind groups max) ──────────────────────────────────

// ── Group 0: Scene (camera, lights, shadow) ───────────────────────────────────
@group(0) @binding(0) var<uniform> camera:       Camera;
@group(0) @binding(1) var<uniform> lights:       LightBuffer;
@group(0) @binding(2) var<uniform> shadow_data:  ShadowData;
@group(0) @binding(3) var shadow_map:            texture_depth_2d;
@group(0) @binding(4) var shadow_sampler:        sampler_comparison;

// ── Group 1: IBL ──────────────────────────────────────────────────────────────
@group(1) @binding(0) var env_sampler:   sampler;
@group(1) @binding(1) var irradiance:    texture_2d<f32>;
@group(1) @binding(2) var prefiltered:   texture_2d<f32>;
@group(1) @binding(3) var brdf_lut:      texture_2d<f32>;
@group(1) @binding(4) var lut_sampler:   sampler;

// ── Group 2: Per-object transform ─────────────────────────────────────────────
@group(2) @binding(0) var<uniform> transform: Transform;

// ── Group 3: Material ─────────────────────────────────────────────────────────
@group(3) @binding(0) var<uniform> mat_data:      Material;
@group(3) @binding(1) var tex_sampler:            sampler;
@group(3) @binding(2) var base_color_tex:         texture_2d<f32>;
@group(3) @binding(3) var normal_tex:             texture_2d<f32>;
@group(3) @binding(4) var mr_tex:                 texture_2d<f32>;
@group(3) @binding(5) var ao_tex:                 texture_2d<f32>;
@group(3) @binding(6) var emissive_tex:           texture_2d<f32>;

struct ShadowData {
  light_vp:   mat4x4f,
  bias:       f32,
  pcf_radius: f32,
  _pad:       vec2f,
}

// ─── Vertex ───────────────────────────────────────────────────────────────────
struct VertIn {
  @location(0) position: vec3f,
  @location(1) normal:   vec3f,
  @location(2) uv:       vec2f,
  @location(3) tangent:  vec4f,
}

struct VertOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal:    vec3f,
  @location(2) uv:        vec2f,
  @location(3) tangent:   vec4f,
}

@vertex
fn vs_main(v: VertIn) -> VertOut {
  var out: VertOut;
  let world_pos = transform.model * vec4f(v.position, 1.0);
  out.world_pos = world_pos.xyz;
  out.clip_pos  = camera.view_proj * world_pos;
  out.normal    = normalize((transform.normal * vec4f(v.normal, 0.0)).xyz);
  out.uv        = v.uv;
  out.tangent   = vec4f(normalize((transform.model * vec4f(v.tangent.xyz, 0.0)).xyz), v.tangent.w);
  return out;
}

// ─── Fragment ─────────────────────────────────────────────────────────────────
@fragment
fn fs_main(v: VertOut) -> @location(0) vec4f {
  // Base color
  var base = mat_data.base_color;
  if (mat_data.flags & 1u) != 0u {
    base *= textureSample(base_color_tex, tex_sampler, v.uv);
  }
  if base.a < mat_data.alpha_cut && (mat_data.flags & 64u) != 0u { discard; }

  // Normal mapping
  var N = normalize(v.normal);
  if (mat_data.flags & 2u) != 0u {
    let T = normalize(v.tangent.xyz);
    let B = v.tangent.w * cross(N, T);
    let ts = textureSample(normal_tex, tex_sampler, v.uv).xyz * 2.0 - 1.0;
    N = normalize(T * ts.x + B * ts.y + N * ts.z);
  }

  // PBR params
  var metallic  = mat_data.metallic;
  var roughness = mat_data.roughness;
  if (mat_data.flags & 4u) != 0u {
    let mr = textureSample(mr_tex, tex_sampler, v.uv);
    roughness *= mr.g;
    metallic  *= mr.b;
  }
  roughness = clamp(roughness, 0.045, 1.0);
  metallic  = clamp(metallic,  0.0,   1.0);

  var ao = 1.0;
  if (mat_data.flags & 8u) != 0u {
    ao = mix(1.0, textureSample(ao_tex, tex_sampler, v.uv).r, mat_data.ao_strength);
  }

  var emissive = mat_data.emissive;
  if (mat_data.flags & 16u) != 0u {
    emissive *= textureSample(emissive_tex, tex_sampler, v.uv).rgb;
  }

  let albedo         = base.rgb;
  let V              = normalize(camera.position - v.world_pos);
  let NoV            = max(dot(N, V), 0.001);
  let f0             = mix(vec3f(0.04), albedo, metallic);
  let diffuse_color  = albedo * (1.0 - metallic);

  // ── Direct lights ──────────────────────────────────────────────────────────
  var Lo = vec3f(0.0);
  for (var i = 0u; i < lights.count; i++) {
    let l = lights.lights[i];
    var L: vec3f;
    var att: f32 = 1.0;

    if l.kind == 0u {
      L = safe_normalize(-l.direction);
    } else {
      let dv  = v.world_pos - l.position;
      L       = -normalize(dv);
      let d   = length(dv);
      att     = 1.0 / max(d * d, 0.0001);
      let rf  = clamp(1.0 - pow(d / l.range, 4.0), 0.0, 1.0);
      att    *= rf * rf;
      if l.kind == 2u {
        let ct  = dot(normalize(dv), l.direction);
        let eps = l.inner_cone - l.outer_cone;
        att    *= clamp((ct - l.outer_cone) / eps, 0.0, 1.0);
      }
    }

    let H   = normalize(V + L);
    let NoL = max(dot(N, L), 0.0);
    if NoL <= 0.0 { continue; }
    let NoH = max(dot(N, H), 0.0);
    let LoH = max(dot(L, H), 0.0);

    let spec = BRDF_Specular(f0, NoH, NoV, NoL, LoH, roughness);
    let diff = diffuse_color * Fd_Burley(NoV, NoL, LoH, roughness);
    let kD   = (1.0 - F_Schlick(LoH, f0)) * (1.0 - metallic);
    Lo      += (kD * diff + spec) * l.color * l.intensity * att * NoL;
  }

  // ── Shadow (directional / first light) ────────────────────────────────────
  var shadow = 1.0;
  {
    let sc  = shadow_data.light_vp * vec4f(v.world_pos, 1.0);
    let ndc = sc.xyz / sc.w;
    let uv  = ndc.xy * 0.5 + 0.5;
    let d   = ndc.z - shadow_data.bias;
    if all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) && d >= 0.0 {
      let ts  = vec2f(1.0 / 2048.0) * shadow_data.pcf_radius;
      var sum = 0.0;
      for (var sx = -1; sx <= 1; sx++) {
        for (var sy = -1; sy <= 1; sy++) {
          sum += textureSampleCompareLevel(shadow_map, shadow_sampler,
                                          uv + vec2f(f32(sx), f32(sy)) * ts, d);
        }
      }
      shadow = sum / 9.0;
    }
  }
  Lo *= shadow;

  // ── IBL ───────────────────────────────────────────────────────────────────
  let R          = reflect(-V, N);
  let diff_uv    = dir_to_equirect(N);
  let spec_uv    = dir_to_equirect(R);
  let irr        = textureSample(irradiance, env_sampler, diff_uv).rgb;
  let pf         = textureSampleLevel(prefiltered, env_sampler, spec_uv, roughness * 4.0).rgb;
  let brdf       = textureSampleLevel(brdf_lut, lut_sampler, vec2f(NoV, roughness), 0.0).rg;
  let F_ibl      = F_Schlick(NoV, f0);
  let kD_ibl     = (1.0 - F_ibl) * (1.0 - metallic);
  let ibl_diff   = kD_ibl * irr * diffuse_color;
  let ibl_spec   = pf * (F_ibl * brdf.x + brdf.y);
  let ambient    = (ibl_diff + ibl_spec) * ao;

  return vec4f(Lo + ambient + emissive, base.a);
}
