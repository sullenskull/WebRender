export interface FileLike {
  name: string;
}

function normalizePath(path: string): string {
  return path
    .replace(/\\/g, '/')
    .replace(/\/+/g, '/')
    .replace(/^\.\//, '')
    .replace(/\/\.\//g, '/');
}

function basename(path: string): string {
  const normalized = normalizePath(path);
  const parts = normalized.split('/');
  return parts[parts.length - 1] ?? normalized;
}

export function buildLocalFileIndex<T extends FileLike>(files: readonly T[]): Map<string, T> {
  const index = new Map<string, T>();

  for (const file of files) {
    const normalized = normalizePath(file.name);
    index.set(normalized, file);
    index.set(basename(normalized), file);
  }

  return index;
}

export function getEntryFile<T extends FileLike>(files: readonly T[], extensions: readonly string[]): T | null {
  const lowered = extensions.map((extension) => extension.toLowerCase());
  return files.find((file) => lowered.some((extension) => file.name.toLowerCase().endsWith(extension))) ?? null;
}

export function resolveLocalGLTFResource<T extends FileLike>(
  fileIndex: Map<string, T>,
  _entryFile: T | null,
  uri: string,
): T | null {
  const normalized = normalizePath(uri);
  return fileIndex.get(normalized) ?? fileIndex.get(basename(normalized)) ?? null;
}
