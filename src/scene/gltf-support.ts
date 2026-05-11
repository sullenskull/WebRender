export function isDataUri(uri: string): boolean {
  return uri.startsWith('data:');
}

export function getUnsupportedExternalResourceMessage(filename: string, kind: 'buffer' | 'image', uri?: string): string {
  const target = uri ? ` (${uri})` : '';
  return `${filename} references an external ${kind}${target}. This viewer currently supports .glb files or .gltf files with embedded data URIs only.`;
}

export function getMissingLocalResourceMessage(filename: string, missing: readonly string[]): string {
  const suffix = missing.length > 0 ? ` Missing: ${missing.join(', ')}.` : '';
  return `${filename} needs local dependency files selected together with the .gltf.${suffix} Select the .gltf, its .bin, and its textures in one action or drop them together.`;
}
