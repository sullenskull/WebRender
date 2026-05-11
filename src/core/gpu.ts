export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GPUContext> {
  if (!navigator.gpu) {
    throw new Error(
      'WebGPU not supported. Enable it in:\n' +
      '• Chrome/Opera GX: chrome://flags/#enable-unsafe-webgpu → Enabled\n' +
      '• Make sure you have a compatible GPU driver.'
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No suitable GPU adapter found.');

  const requiredFeatures: GPUFeatureName[] = [];
  // Request timestamp queries if available (for GPU timing)
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxTextureDimension2D: Math.min(adapter.limits.maxTextureDimension2D, 8192),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 256 * 1024 * 1024),
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 128 * 1024 * 1024),
    },
  });

  device.lost.then((info) => {
    console.error('GPU device lost:', info.message);
    if (info.reason !== 'destroyed') {
      // Could attempt reinitialization here
      alert('GPU device lost. Please refresh the page.');
    }
  });

  device.onuncapturederror = (ev) => {
    console.error('Uncaptured WebGPU error:', ev.error);
  };

  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: 'opaque',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  console.log(`WebGPU initialized: ${adapter.info?.vendor ?? 'unknown'} — ${format}`);
  return { adapter, device, canvas, context, format };
}

// Helper: create a GPU buffer and optionally upload data
export function createBuffer(
  device: GPUDevice,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
): GPUBuffer {
  const buf = device.createBuffer({ size: Math.max(size, 4), usage, mappedAtCreation: !!data });
  if (data) {
    const src = data instanceof ArrayBuffer ? data : data.buffer as ArrayBuffer;
    const offset = data instanceof ArrayBuffer ? 0 : (data as ArrayBufferView).byteOffset;
    const len = data instanceof ArrayBuffer ? data.byteLength : (data as ArrayBufferView).byteLength;
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(src, offset, len));
    buf.unmap();
  }
  return buf;
}

export function uploadBuffer(device: GPUDevice, buf: GPUBuffer, data: ArrayBufferView, offset = 0): void {
  device.queue.writeBuffer(buf, offset, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
}

// Create a texture from raw pixel data
export function createTextureFromData(
  device: GPUDevice,
  width: number,
  height: number,
  data: Uint8Array | Float32Array,
  format: GPUTextureFormat,
  label?: string
): GPUTexture {
  const tex = device.createTexture({
    label,
    size: [width, height],
    format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const bytesPerPixel = format.includes('32float') ? 16 : (format.includes('16float') ? 8 : 4);
  device.queue.writeTexture(
    { texture: tex },
    data.buffer as ArrayBuffer,
    { offset: data.byteOffset, bytesPerRow: width * bytesPerPixel },
    [width, height]
  );
  return tex;
}

// Create a simple linear sampler
export function createLinearSampler(device: GPUDevice, addressMode: GPUAddressMode = 'clamp-to-edge'): GPUSampler {
  return device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: addressMode,
    addressModeV: addressMode,
  });
}

export function createNearestSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
}

export function align(n: number, alignment: number): number {
  return Math.ceil(n / alignment) * alignment;
}
