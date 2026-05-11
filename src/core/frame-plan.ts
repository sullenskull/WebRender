export interface RealtimeResolvePlan {
  taaOutputTexture: 'hdr' | 'taa-resolve';
  historyTexture: 'hdr-prev';
  shouldCopyResolvedToHistory: boolean;
}

export function getRealtimeResolvePlan(taaEnabled: boolean): RealtimeResolvePlan {
  if (taaEnabled) {
    return {
      taaOutputTexture: 'taa-resolve',
      historyTexture: 'hdr-prev',
      shouldCopyResolvedToHistory: true,
    };
  }

  return {
    taaOutputTexture: 'hdr',
    historyTexture: 'hdr-prev',
    shouldCopyResolvedToHistory: false,
  };
}
