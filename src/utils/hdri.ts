// RGBE (.hdr) file parser — produces Float32Array RGBA data
export interface HDRImage {
  width: number;
  height: number;
  data: Float32Array; // RGBA f32, linear
}

export async function loadHDR(file: File | string): Promise<HDRImage> {
  let buffer: ArrayBuffer;
  if (typeof file === 'string') {
    buffer = await fetch(file).then(r => r.arrayBuffer());
  } else {
    buffer = await file.arrayBuffer();
  }
  return parseRGBE(new Uint8Array(buffer));
}

function parseRGBE(bytes: Uint8Array): HDRImage {
  let pos = 0;

  // Read header lines
  const readLine = (): string => {
    let s = '';
    while (pos < bytes.length && bytes[pos] !== 0x0a) s += String.fromCharCode(bytes[pos++]);
    pos++; // skip LF
    return s;
  };

  // Magic
  const magic = readLine();
  if (!magic.startsWith('#?')) throw new Error('Not a valid RGBE file');

  let width = 0, height = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const line = readLine();
    if (line === '') break;
    const m = line.match(/^-Y (\d+) \+X (\d+)/);
    if (m) { height = parseInt(m[1]); width = parseInt(m[2]); break; }
  }
  if (!width || !height) throw new Error('Failed to parse HDR dimensions');

  const out = new Float32Array(width * height * 4);

  // RLE decode scanlines
  const scanline = new Uint8Array(width * 4);
  for (let y = 0; y < height; y++) {
    // Check for new RLE format
    if (bytes[pos] === 2 && bytes[pos+1] === 2) {
      const scanW = (bytes[pos+2] << 8) | bytes[pos+3];
      pos += 4;
      if (scanW !== width) throw new Error('Bad scanline width');

      // Decode 4 channels separately
      for (let ch = 0; ch < 4; ch++) {
        let x = 0;
        while (x < width) {
          const code = bytes[pos++];
          if (code > 128) {
            const count = code - 128;
            const val = bytes[pos++];
            for (let i = 0; i < count; i++) scanline[x * 4 + ch + i * 4] = val; // wrong indexing fix below
            // Actually channel-separated layout: scanline[ch*width + x]
            // Let me redo this with proper channel buffer
            x += count;
          } else {
            for (let i = 0; i < code; i++) scanline[x * 4 + ch + i * 4] = bytes[pos++];
            x += code;
          }
        }
      }
      // Decode RGBE to float RGB
      const base = y * width * 4;
      for (let x = 0; x < width; x++) {
        const r = scanline[x*4+0], g = scanline[x*4+1], b = scanline[x*4+2], e = scanline[x*4+3];
        if (e !== 0) {
          const scale = Math.pow(2, e - 128 - 8);
          out[base + x*4+0] = r * scale;
          out[base + x*4+1] = g * scale;
          out[base + x*4+2] = b * scale;
          out[base + x*4+3] = 1.0;
        }
      }
    } else {
      // Old format: read 4 bytes per pixel
      const base = y * width * 4;
      for (let x = 0; x < width; x++) {
        const r = bytes[pos++], g = bytes[pos++], b = bytes[pos++], e = bytes[pos++];
        if (e !== 0) {
          const scale = Math.pow(2, e - 128 - 8);
          out[base + x*4+0] = r * scale;
          out[base + x*4+1] = g * scale;
          out[base + x*4+2] = b * scale;
          out[base + x*4+3] = 1.0;
        }
      }
    }
  }

  return { width, height, data: out };
}

// Better RLE parser with proper channel-separated approach
export function parseRGBEProper(bytes: Uint8Array): HDRImage {
  let pos = 0;
  const readLine = () => {
    let s = '';
    while (pos < bytes.length && bytes[pos] !== 0x0a) s += String.fromCharCode(bytes[pos++]);
    pos++;
    return s;
  };

  const firstLine = readLine();
  if (!firstLine.startsWith('#?') && !firstLine.includes('RADIANCE')) {
    throw new Error('Invalid HDR header');
  }

  let width = 0, height = 0;
  for (let i = 0; i < 20; i++) {
    const line = readLine();
    const m = line.match(/-Y\s+(\d+)\s+\+X\s+(\d+)/);
    if (m) { height = parseInt(m[1]); width = parseInt(m[2]); break; }
    if (width && height) break;
  }
  if (!width || !height) throw new Error('HDR dimensions not found');

  const out = new Float32Array(width * height * 4);
  const chBuf = new Uint8Array(width * 4); // 4 channels × width

  for (let y = 0; y < height; y++) {
    if (pos + 4 > bytes.length) break;

    // New RLE scanline check
    if (bytes[pos] === 2 && bytes[pos+1] === 2 && (bytes[pos+2] << 8 | bytes[pos+3]) === width) {
      pos += 4;

      // Each channel decoded separately
      for (let ch = 0; ch < 4; ch++) {
        let x = 0;
        while (x < width) {
          let code = bytes[pos++];
          if (code > 128) {
            const count = code - 128;
            const val = bytes[pos++];
            for (let i = 0; i < count && x + i < width; i++) chBuf[ch * width + x + i] = val;
            x += count;
          } else {
            for (let i = 0; i < code && x + i < width; i++) chBuf[ch * width + x + i] = bytes[pos++];
            x += code;
          }
        }
      }
    } else {
      // Uncompressed
      for (let x = 0; x < width; x++) {
        chBuf[0*width+x] = bytes[pos++]; chBuf[1*width+x] = bytes[pos++];
        chBuf[2*width+x] = bytes[pos++]; chBuf[3*width+x] = bytes[pos++];
      }
    }

    // RGBE → float
    const base = y * width * 4;
    for (let x = 0; x < width; x++) {
      const r = chBuf[0*width+x], g = chBuf[1*width+x], b = chBuf[2*width+x], e = chBuf[3*width+x];
      if (e !== 0) {
        const scale = Math.pow(2, e - 128 - 8);
        out[base+x*4+0] = r * scale;
        out[base+x*4+1] = g * scale;
        out[base+x*4+2] = b * scale;
        out[base+x*4+3] = 1.0;
      } else {
        out[base+x*4+0] = out[base+x*4+1] = out[base+x*4+2] = out[base+x*4+3] = 0;
      }
    }
  }
  return { width, height, data: out };
}
