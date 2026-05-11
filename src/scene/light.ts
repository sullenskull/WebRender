import type { Vec3 } from '../utils/math.js';

export const MAX_LIGHTS = 16;

export type LightType = 'directional' | 'point' | 'spot';

export interface Light {
  id: string;
  type: LightType;
  position: Vec3;
  direction: Vec3;
  color: [number, number, number]; // linear RGB
  intensity: number;
  range: number;
  innerCone: number;  // radians
  outerCone: number;  // radians
  castShadow: boolean;
}

export function createLight(type: LightType = 'point'): Light {
  return {
    id: Math.random().toString(36).slice(2),
    type,
    position: [2, 3, 2],
    direction: [-0.5, -1, -0.5],
    color: [1, 1, 1],
    intensity: 5,
    range: 20,
    innerCone: 0.3,
    outerCone: 0.5,
    castShadow: true,
  };
}

// Pack lights into a Float32Array for GPU upload
// Layout per light (16 floats = 64 bytes):
//   [0-2]  position   [3]    type (0=dir,1=point,2=spot)
//   [4-6]  color      [7]    intensity
//   [8-10] direction  [11]   range
//   [12]   innerCone  [13]   outerCone  [14-15] padding
export function packLights(lights: Light[]): Float32Array {
  // Header: count (4 bytes) + 12 bytes padding = 16 bytes
  // Then lights array (64 bytes each)
  const stride = 16; // floats per light
  const buf = new Float32Array(4 + MAX_LIGHTS * stride);
  buf[0] = lights.length; // count

  for (let i = 0; i < Math.min(lights.length, MAX_LIGHTS); i++) {
    const l = lights[i];
    const off = 4 + i * stride;
    buf[off+0] = l.position[0]; buf[off+1] = l.position[1]; buf[off+2] = l.position[2];
    buf[off+3] = l.type === 'directional' ? 0 : l.type === 'point' ? 1 : 2;
    buf[off+4] = l.color[0]; buf[off+5] = l.color[1]; buf[off+6] = l.color[2];
    buf[off+7] = l.intensity;
    buf[off+8] = l.direction[0]; buf[off+9] = l.direction[1]; buf[off+10] = l.direction[2];
    buf[off+11] = l.range;
    buf[off+12] = l.innerCone; buf[off+13] = l.outerCone;
    buf[off+14] = 0; buf[off+15] = 0;
  }
  return buf;
}

// Hex color ↔ linear RGB
export function hexToLinear(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1,3), 16) / 255;
  const g = parseInt(hex.slice(3,5), 16) / 255;
  const b = parseInt(hex.slice(5,7), 16) / 255;
  return [r**2.2, g**2.2, b**2.2];
}

export function linearToHex(rgb: [number, number, number]): string {
  const toHex = (v: number) => Math.round(Math.pow(Math.min(v, 1), 1/2.2) * 255).toString(16).padStart(2,'0');
  return '#' + toHex(rgb[0]) + toHex(rgb[1]) + toHex(rgb[2]);
}
