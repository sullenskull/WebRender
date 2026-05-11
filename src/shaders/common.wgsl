// ─── Common PBR math, shared across all shaders ───────────────────────────────

const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067;
const HALF_PI: f32 = 1.5707963267948966;
const EPSILON: f32 = 1e-6;

// ── Camera uniform (binding group 0, binding 0) ─────────────────────────────
struct Camera {
  view:         mat4x4f,
  proj:         mat4x4f,
  view_proj:    mat4x4f,
  inv_view_proj:mat4x4f,
  prev_view_proj:mat4x4f,
  position:     vec3f,
  near:         f32,
  far:          f32,
  fov_y:        f32,
  aspect:       f32,
  frame_index:  u32,
  jitter:       vec2f,
  _pad:         vec2f,
}

// ── Light (matches packLights() in light.ts) ─────────────────────────────────
struct Light {
  position:   vec3f,
  kind:       u32,   // 0=directional 1=point 2=spot
  color:      vec3f,
  intensity:  f32,
  direction:  vec3f,
  range:      f32,
  inner_cone: f32,
  outer_cone: f32,
  _pad:       vec2f,
}

struct LightBuffer {
  count:  u32,
  _pad:   vec3u,
  lights: array<Light, 16>,
}

// ── Material uniform ──────────────────────────────────────────────────────────
struct Material {
  base_color: vec4f,
  emissive:   vec3f,
  metallic:   f32,
  roughness:  f32,
  ao_strength:f32,
  alpha_cut:  f32,
  flags:      u32,   // bit0=has_basecolor, bit1=has_normal, bit2=has_mr, bit3=has_ao, bit4=has_emissive
}

// ── Transform uniform ─────────────────────────────────────────────────────────
struct Transform {
  model:   mat4x4f,
  normal:  mat4x4f,
}

// ─── PBR BRDF Functions ───────────────────────────────────────────────────────

// GGX Normal Distribution Function
fn D_GGX(NoH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = (NoH * a2 - NoH) * NoH + 1.0;
  return a2 / max(PI * d * d, EPSILON);
}

// Smith Height-Correlated GGX Geometric Shadowing
fn V_SmithGGXCorrelated(NoV: f32, NoL: f32, roughness: f32) -> f32 {
  let a2 = roughness * roughness * roughness * roughness;
  let GGXL = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
  let GGXV = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
  return 0.5 / max(GGXV + GGXL, EPSILON);
}

// Schlick Fresnel
fn F_Schlick(u: f32, f0: vec3f) -> vec3f {
  let f = pow(clamp(1.0 - u, 0.0, 1.0), 5.0);
  return f + f0 * (1.0 - f);
}

fn F_Schlick_scalar(u: f32, f0: f32) -> f32 {
  return f0 + (1.0 - f0) * pow(clamp(1.0 - u, 0.0, 1.0), 5.0);
}

// Disney diffuse (Burley) — better than Lambertian at grazing angles
fn Fd_Burley(NoV: f32, NoL: f32, LoH: f32, roughness: f32) -> f32 {
  let f90 = 0.5 + 2.0 * roughness * LoH * LoH;
  let lightScatter = F_Schlick_scalar(NoL, 1.0) * (f90 - 1.0) + 1.0;
  let viewScatter  = F_Schlick_scalar(NoV, 1.0) * (f90 - 1.0) + 1.0;
  return lightScatter * viewScatter * INV_PI;
}

// Full specular BRDF term (D*V*F)
fn BRDF_Specular(f0: vec3f, NoH: f32, NoV: f32, NoL: f32, LoH: f32, roughness: f32) -> vec3f {
  let perceptualRoughness = max(roughness, 0.045);
  let D = D_GGX(NoH, perceptualRoughness);
  let V = V_SmithGGXCorrelated(NoV, NoL, perceptualRoughness);
  let F = F_Schlick(LoH, f0);
  return D * V * F;
}

// ─── IBL helpers ──────────────────────────────────────────────────────────────

// Equirectangular UV from direction
fn dir_to_equirect(d: vec3f) -> vec2f {
  return vec2f(
    0.5 + 0.5 * atan2(d.z, d.x) * INV_PI,
    0.5 - asin(clamp(d.y, -1.0, 1.0)) * INV_PI
  );
}

// Importance sample GGX for IBL prefilter (on CPU, not used at runtime)
fn importance_sample_GGX(xi: vec2f, N: vec3f, roughness: f32) -> vec3f {
  let a = roughness * roughness;
  let phi = 2.0 * PI * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a*a - 1.0) * xi.y));
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  let H = vec3f(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
  // TBN
  let up = select(vec3f(1.0,0.0,0.0), vec3f(0.0,1.0,0.0), abs(N.y) < 0.999);
  let T = normalize(cross(up, N));
  let B = cross(N, T);
  return normalize(T*H.x + B*H.y + N*H.z);
}

// ─── Tone mapping ─────────────────────────────────────────────────────────────

// AgX — by Troy Sobotka, natural tone response
fn agx_default_contrast_approx(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  let x6 = x4 * x2;
  return  - 17.86     * x6 * x
          + 78.01     * x6
          - 126.7     * x4 * x
          + 92.06     * x4
          - 28.72     * x2 * x
          + 4.361     * x2
          - 0.1718    * x
          + 0.002857;
}

fn agx(val_in: vec3f) -> vec3f {
  // AgX Input Transform (linear → log)
  let AgXInset = mat3x3f(
    vec3f(0.842479062253094,  0.0423282422610123, 0.0423756549057051),
    vec3f(0.0784335999999992, 0.878468636469772,  0.0784336),
    vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104)
  );
  let MIN_EV = -12.47393;
  let MAX_EV =  4.026069;

  var col = AgXInset * val_in;
  col = clamp(log2(max(col, vec3f(1e-10))), vec3f(MIN_EV), vec3f(MAX_EV));
  col = (col - vec3f(MIN_EV)) / (MAX_EV - MIN_EV);
  return agx_default_contrast_approx(col);
}

fn agx_eotf(val_in: vec3f) -> vec3f {
  let AgXOutset = mat3x3f(
    vec3f(1.19687900512017,   -0.0528968517574562, -0.0529716355144438),
    vec3f(-0.0980208811401368, 1.15190312990417,   -0.0980434501171241),
    vec3f(-0.0990297440797205,-0.0989611768448433,  1.15107367264116)
  );
  return AgXOutset * val_in;
}

fn tonemap_agx(color: vec3f, exposure: f32) -> vec3f {
  let c = color * pow(2.0, exposure);
  let mapped = agx(c);
  let out = agx_eotf(mapped);
  return clamp(out, vec3f(0.0), vec3f(1.0));
}

// ACES filmic
fn tonemap_aces(color: vec3f, exposure: f32) -> vec3f {
  var c = color * pow(2.0, exposure) * 0.6;
  let a = 2.51; let b = 0.03; let cc = 2.43; let d = 0.59; let e = 0.14;
  return clamp((c*(a*c+b))/(c*(cc*c+d)+e), vec3f(0.0), vec3f(1.0));
}

// Simple Reinhard
fn tonemap_reinhard(color: vec3f, exposure: f32) -> vec3f {
  var c = color * pow(2.0, exposure);
  return c / (c + vec3f(1.0));
}

// ─── Depth reconstruction ─────────────────────────────────────────────────────

fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
  return near * far / (far - depth * (far - near));
}

fn world_pos_from_depth(uv: vec2f, depth: f32, inv_vp: mat4x4f) -> vec3f {
  let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
  let world_h = inv_vp * ndc;
  return world_h.xyz / world_h.w;
}

// ─── Misc ─────────────────────────────────────────────────────────────────────

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

fn rgb_to_ycocg(rgb: vec3f) -> vec3f {
  let Y  =  0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b;
  let Co =  0.5  * rgb.r                - 0.5  * rgb.b;
  let Cg = -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b;
  return vec3f(Y, Co, Cg);
}

fn ycocg_to_rgb(ycocg: vec3f) -> vec3f {
  let Y = ycocg.x; let Co = ycocg.y; let Cg = ycocg.z;
  return vec3f(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

fn safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  return select(vec3f(0.0,1.0,0.0), v/l, l > 1e-6);
}
