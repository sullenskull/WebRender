export function isDataUri(uri: string): boolean {
  return uri.startsWith('data:');
}

export function getUnsupportedExternalResourceMessage(filename: string, kind: 'buffer' | 'image', uri?: string): string {
  const target = uri ? ` (${uri})` : '';
  return `${filename} references an external ${kind}${target}. This viewer currently supports .glb files or .gltf files with embedded data URIs only.`;
}
