export function isWindowsUserAgent(userAgent: string): boolean {
  return /Windows/i.test(userAgent);
}

export function shouldRequestHighPerformanceAdapter(userAgent: string): boolean {
  return !isWindowsUserAgent(userAgent);
}
