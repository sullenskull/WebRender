import { isDataUri } from './gltf-support.js';

export interface GLTFExternalDependencies {
  buffers: string[];
  images: string[];
}

export function collectExternalGLTFDependencies(json: any): GLTFExternalDependencies {
  const buffers = (json.buffers ?? [])
    .map((buffer: any) => buffer?.uri)
    .filter((uri: unknown): uri is string => typeof uri === 'string' && !isDataUri(uri));

  const images = (json.images ?? [])
    .map((image: any) => image?.uri)
    .filter((uri: unknown): uri is string => typeof uri === 'string' && !isDataUri(uri));

  return { buffers, images };
}
