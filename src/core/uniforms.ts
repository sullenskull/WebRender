export const SKYBOX_PARAMS_BUFFER_SIZE = 32;

export function createSkyboxParamsData(exposure: number): Float32Array {
  return new Float32Array([exposure, 0, 0, 0]);
}
