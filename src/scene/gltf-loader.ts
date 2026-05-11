import { collectExternalGLTFDependencies } from './gltf-manifest.js';
import type { GPUContext } from '../core/gpu.js';
import { getMissingLocalResourceMessage, getUnsupportedExternalResourceMessage, isDataUri } from './gltf-support.js';
import { buildLocalFileIndex, resolveLocalGLTFResource } from './gltf-file-resolver.js';
import { createBuffer } from '../core/gpu.js';
import { mat4Create, mat4FromTRS, mat4Multiply, mat4Invert, mat4Transpose } from '../utils/math.js';
import type { Mat4 } from '../utils/math.js';

export interface GPUPrimitive {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer | null;
  indexCount: number;
  indexFormat: GPUIndexFormat;
  vertexCount: number;
  materialIndex: number;
  // For bounding box
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
}

export interface GPUMaterial {
  baseColorFactor: [number, number, number, number];
  emissiveFactor: [number, number, number];
  metallicFactor: number;
  roughnessFactor: number;
  aoStrength: number;
  alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff: number;
  doubleSided: boolean;
  // Textures (null = use factor only)
  baseColorTexture: GPUTexture | null;
  normalTexture: GPUTexture | null;
  metallicRoughnessTexture: GPUTexture | null;
  occlusionTexture: GPUTexture | null;
  emissiveTexture: GPUTexture | null;
  // Packed uniform buffer
  uniformBuffer: GPUBuffer;
}

export interface GPUMesh {
  primitives: GPUPrimitive[];
}

export interface SceneNode {
  name: string;
  mesh: GPUMesh | null;
  children: SceneNode[];
  localMatrix: Mat4;
  worldMatrix: Mat4;
  normalMatrix: Mat4;
  modelBuffer: GPUBuffer; // mat4 model + mat4 normal = 128 bytes
}

export interface LoadedScene {
  nodes: SceneNode[];
  materials: GPUMaterial[];
  meshes: GPUMesh[];
  allPrimitives: { primitive: GPUPrimitive; node: SceneNode }[];
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
}

// Vertex layout: interleaved position(12) + normal(12) + uv(8) + tangent(16) = 48 bytes
export const VERTEX_STRIDE = 48;

export async function loadGLTF(gpu: GPUContext, file: File, files: readonly File[] = [file]): Promise<LoadedScene> {
  const buffer = await file.arrayBuffer();
  const ext = file.name.toLowerCase();
  const fileIndex = buildLocalFileIndex(files);

  if (ext.endsWith('.glb')) {
    return parseGLB(gpu, buffer);
  } else {
    const json = JSON.parse(new TextDecoder().decode(buffer));
    const dependencies = collectExternalGLTFDependencies(json);
    const missing = [...dependencies.buffers, ...dependencies.images].filter((uri) => {
      return !resolveLocalGLTFResource(fileIndex, file, uri);
    });
    if (missing.length > 0) {
      throw new Error(getMissingLocalResourceMessage(file.name, missing));
    }
    return parseGLTF(gpu, json, null, file.name, fileIndex, file);
  }
}

async function parseGLB(gpu: GPUContext, buffer: ArrayBuffer): Promise<LoadedScene> {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== 0x46546C67) throw new Error('Not a GLB file');

  const version = view.getUint32(4, true);
  if (version !== 2) throw new Error(`Unsupported GLB version: ${version}`);

  let pos = 12;
  let jsonChunk: ArrayBuffer | null = null;
  let binChunk: ArrayBuffer | null = null;

  while (pos < buffer.byteLength) {
    const chunkLength = view.getUint32(pos, true);
    const chunkType = view.getUint32(pos + 4, true);
    pos += 8;

    if (chunkType === 0x4E4F534A) {
      jsonChunk = buffer.slice(pos, pos + chunkLength);
    } else if (chunkType === 0x004E4942) {
      binChunk = buffer.slice(pos, pos + chunkLength);
    }
    pos += chunkLength;
  }

  if (!jsonChunk) throw new Error('No JSON chunk in GLB');
  const json = JSON.parse(new TextDecoder().decode(jsonChunk));
  return parseGLTF(gpu, json, binChunk, 'model.glb', new Map(), null);
}

async function parseGLTF(
  gpu: GPUContext,
  json: any,
  binChunk: ArrayBuffer | null,
  _filename: string,
  fileIndex: Map<string, File>,
  entryFile: File | null,
): Promise<LoadedScene> {
  const { device } = gpu;

  // Resolve buffers
  const bufferData: (ArrayBuffer | null)[] = await Promise.all(
    (json.buffers ?? []).map(async (buf: any, i: number) => {
      if (i === 0 && binChunk) return binChunk;
      if (buf.uri && isDataUri(buf.uri)) {
        const base64 = buf.uri.split(',')[1];
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let j = 0; j < binary.length; j++) bytes[j] = binary.charCodeAt(j);
        return bytes.buffer;
      }
      if (buf.uri) {
        const localFile = resolveLocalGLTFResource(fileIndex, entryFile, buf.uri);
        if (localFile) return await localFile.arrayBuffer();
        throw new Error(getUnsupportedExternalResourceMessage(_filename, 'buffer', buf.uri));
      }
      return null;
    })
  );

  const getAccessorData = (accessorIdx: number): { data: ArrayBuffer; componentType: number; type: string; count: number; byteStride: number } => {
    const acc = json.accessors[accessorIdx];
    const bv = json.bufferViews[acc.bufferView];
    const buf = bufferData[bv.buffer];
    if (!buf) {
      throw new Error(getUnsupportedExternalResourceMessage(_filename, 'buffer', json.buffers?.[bv.buffer]?.uri));
    }
    const byteOffset = (acc.byteOffset ?? 0) + (bv.byteOffset ?? 0);
    const byteStride = bv.byteStride ?? 0;
    const componentSizes: Record<number, number> = { 5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4 };
    const typeCounts: Record<string, number> = { SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT2:4, MAT3:9, MAT4:16 };
    const compSize = componentSizes[acc.componentType] ?? 4;
    const typeCount = typeCounts[acc.type] ?? 1;
    const elementSize = compSize * typeCount;
    const totalBytes = byteStride > 0
      ? byteStride * (acc.count - 1) + elementSize
      : elementSize * acc.count;
    return {
      data: buf.slice(byteOffset, byteOffset + totalBytes),
      componentType: acc.componentType,
      type: acc.type,
      count: acc.count,
      byteStride: byteStride > 0 ? byteStride : elementSize,
    };
  };

  // Extract float array from accessor (handles stride properly)
  const getFloatArray = (accessorIdx: number): Float32Array => {
    const { data, componentType, type, count, byteStride } = getAccessorData(accessorIdx);
    const typeCount = { SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT4:16 }[type] ?? 1;
    const result = new Float32Array(count * typeCount);
    const view = new DataView(data);
    const elementSize = typeCount * 4;

    for (let i = 0; i < count; i++) {
      const base = i * byteStride;
      for (let j = 0; j < typeCount; j++) {
        if (componentType === 5126) { // FLOAT
          result[i*typeCount+j] = view.getFloat32(base + j*4, true);
        } else if (componentType === 5121) { // UNSIGNED_BYTE normalized
          result[i*typeCount+j] = view.getUint8(base + j) / 255;
        } else if (componentType === 5123) { // UNSIGNED_SHORT normalized
          result[i*typeCount+j] = view.getUint16(base + j*2, true) / 65535;
        }
      }
    }
    return result;
  };

  const getIndexArray = (accessorIdx: number): { data: Uint16Array | Uint32Array; format: GPUIndexFormat } => {
    const { data, componentType, count } = getAccessorData(accessorIdx);
    const view = new DataView(data);
    if (componentType === 5123) { // UNSIGNED_SHORT
      const arr = new Uint16Array(count);
      for (let i = 0; i < count; i++) arr[i] = view.getUint16(i*2, true);
      return { data: arr, format: 'uint16' };
    } else { // UNSIGNED_INT
      const arr = new Uint32Array(count);
      for (let i = 0; i < count; i++) arr[i] = view.getUint32(i*4, true);
      return { data: arr, format: 'uint32' };
    }
  };

  // Load textures
  const textures: (GPUTexture | null)[] = [];
  const images = json.images ?? [];
  for (const img of images) {
    try {
      let blob: Blob;
      if (img.bufferView !== undefined) {
        const bv = json.bufferViews[img.bufferView];
        const buf = bufferData[bv.buffer]!;
        const slice = buf.slice(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
        const mimeType = img.mimeType ?? 'image/png';
        blob = new Blob([slice], { type: mimeType });
      } else if (img.uri && isDataUri(img.uri)) {
        const arr = atob(img.uri.split(',')[1]);
        const bytes = new Uint8Array(arr.length);
        for (let i = 0; i < arr.length; i++) bytes[i] = arr.charCodeAt(i);
        blob = new Blob([bytes]);
      } else if (img.uri) {
        const localFile = resolveLocalGLTFResource(fileIndex, entryFile, img.uri);
        if (localFile) {
          blob = localFile;
        } else {
          throw new Error(getUnsupportedExternalResourceMessage(_filename, 'image', img.uri));
        }
      } else {
        textures.push(null); continue;
      }
      const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
      const tex = device.createTexture({
        size: [bitmap.width, bitmap.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        // We currently do not generate mip chains after upload, so keep imported textures
        // to a single mip level to avoid sampling uninitialized mip data on some drivers.
        mipLevelCount: 1,
      });
      device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex }, [bitmap.width, bitmap.height]);
      // Generate mipmaps via compute (simplified: just mark as done)
      textures.push(tex);
    } catch {
      textures.push(null);
    }
  }

  const getTexture = (texRef: any): GPUTexture | null => {
    if (!texRef || texRef.index === undefined) return null;
    const gltfTex = json.textures?.[texRef.index];
    if (!gltfTex || gltfTex.source === undefined) return null;
    return textures[gltfTex.source] ?? null;
  };

  // Build materials
  const materials: GPUMaterial[] = [];
  for (const mat of (json.materials ?? [{ name: 'default' }])) {
    const pbr = mat.pbrMetallicRoughness ?? {};
    const baseColor = pbr.baseColorFactor ?? [1,1,1,1];
    const emissive = mat.emissiveFactor ?? [0,0,0];
    const metallic = pbr.metallicFactor ?? 1.0;
    const roughness = pbr.roughnessFactor ?? 1.0;

    // Uniform buffer: baseColor(16) + emissive(12) + metallic(4) + roughness(4) + ao(4) + alphaCut(4) + flags(4) = 64 bytes
    const uniformData = new Float32Array(16);
    uniformData[0]=baseColor[0]; uniformData[1]=baseColor[1]; uniformData[2]=baseColor[2]; uniformData[3]=baseColor[3];
    uniformData[4]=emissive[0]; uniformData[5]=emissive[1]; uniformData[6]=emissive[2];
    uniformData[7]=metallic;
    uniformData[8]=roughness;
    uniformData[9]=1.0; // ao strength
    uniformData[10]=mat.alphaCutoff ?? 0.5;
    uniformData[11]=0; // flags

    const uniformBuffer = createBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, uniformData);

    materials.push({
      baseColorFactor: baseColor as [number,number,number,number],
      emissiveFactor: emissive as [number,number,number],
      metallicFactor: metallic,
      roughnessFactor: roughness,
      aoStrength: 1.0,
      alphaMode: mat.alphaMode ?? 'OPAQUE',
      alphaCutoff: mat.alphaCutoff ?? 0.5,
      doubleSided: mat.doubleSided ?? false,
      baseColorTexture: getTexture(pbr.baseColorTexture),
      normalTexture: getTexture(mat.normalTexture),
      metallicRoughnessTexture: getTexture(pbr.metallicRoughnessTexture),
      occlusionTexture: getTexture(mat.occlusionTexture),
      emissiveTexture: getTexture(mat.emissiveTexture),
      uniformBuffer,
    });
  }

  // Build meshes
  const meshes: GPUMesh[] = [];
  for (const mesh of (json.meshes ?? [])) {
    const primitives: GPUPrimitive[] = [];
    for (const prim of mesh.primitives) {
      const attrs = prim.attributes;
      const positions = attrs.POSITION !== undefined ? getFloatArray(attrs.POSITION) : new Float32Array(0);
      const normals = attrs.NORMAL !== undefined ? getFloatArray(attrs.NORMAL) : null;
      const uvs = attrs.TEXCOORD_0 !== undefined ? getFloatArray(attrs.TEXCOORD_0) : null;
      const tangents = attrs.TANGENT !== undefined ? getFloatArray(attrs.TANGENT) : null;

      const vertexCount = positions.length / 3;

      // Compute AABB
      let minX=Infinity,minY=Infinity,minZ=Infinity;
      let maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
      for (let i = 0; i < vertexCount; i++) {
        const x=positions[i*3],y=positions[i*3+1],z=positions[i*3+2];
        if(x<minX)minX=x; if(y<minY)minY=y; if(z<minZ)minZ=z;
        if(x>maxX)maxX=x; if(y>maxY)maxY=y; if(z>maxZ)maxZ=z;
      }

      // Interleave vertices: pos(3) + normal(3) + uv(2) + tangent(4) = 12 floats = 48 bytes
      const interleaved = new Float32Array(vertexCount * 12);
      for (let i = 0; i < vertexCount; i++) {
        const off = i * 12;
        interleaved[off+0] = positions[i*3+0];
        interleaved[off+1] = positions[i*3+1];
        interleaved[off+2] = positions[i*3+2];
        interleaved[off+3] = normals ? normals[i*3+0] : 0;
        interleaved[off+4] = normals ? normals[i*3+1] : 1;
        interleaved[off+5] = normals ? normals[i*3+2] : 0;
        interleaved[off+6] = uvs ? uvs[i*2+0] : 0;
        interleaved[off+7] = uvs ? uvs[i*2+1] : 0;
        interleaved[off+8] = tangents ? tangents[i*4+0] : 1;
        interleaved[off+9] = tangents ? tangents[i*4+1] : 0;
        interleaved[off+10] = tangents ? tangents[i*4+2] : 0;
        interleaved[off+11] = tangents ? tangents[i*4+3] : 1;
      }

      const vertexBuffer = createBuffer(device, interleaved.byteLength,
        GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, interleaved);

      let indexBuffer: GPUBuffer | null = null;
      let indexCount = 0;
      let indexFormat: GPUIndexFormat = 'uint16';

      if (prim.indices !== undefined) {
        const { data, format } = getIndexArray(prim.indices);
        indexCount = data.length;
        indexFormat = format;
        indexBuffer = createBuffer(device, data.byteLength,
          GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, data);
      }

      primitives.push({
        vertexBuffer,
        indexBuffer,
        indexCount,
        indexFormat,
        vertexCount,
        materialIndex: prim.material ?? 0,
        aabbMin: [minX, minY, minZ],
        aabbMax: [maxX, maxY, maxZ],
      });
    }
    meshes.push({ primitives });
  }

  // Build node tree
  const buildNode = (nodeIdx: number, parentWorld: Mat4 | null): SceneNode => {
    const n = json.nodes[nodeIdx];
    const local = mat4Create();

    if (n.matrix) {
      local.set(n.matrix);
    } else {
      const t: [number,number,number] = n.translation ?? [0,0,0];
      const r: [number,number,number,number] = n.rotation ?? [0,0,0,1];
      const s: [number,number,number] = n.scale ?? [1,1,1];
      mat4FromTRS(local, t, r, s);
    }

    const world = mat4Create();
    if (parentWorld) mat4Multiply(world, parentWorld, local);
    else world.set(local);

    const normalMatrix = mat4Create();
    const invWorld = mat4Create();
    mat4Invert(invWorld, world);
    mat4Transpose(normalMatrix, invWorld);

    // Pack model + normal matrix into 128-byte buffer
    const modelData = new Float32Array(32);
    modelData.set(world, 0);
    modelData.set(normalMatrix, 16);
    const modelBuffer = createBuffer(device, 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, modelData);

    const children = (n.children ?? []).map((ci: number) => buildNode(ci, world));
    const mesh = n.mesh !== undefined ? meshes[n.mesh] : null;

    return { name: n.name ?? `node_${nodeIdx}`, mesh, children, localMatrix: local, worldMatrix: world, normalMatrix, modelBuffer };
  };

  const rootScene = json.scenes?.[json.scene ?? 0] ?? { nodes: [] };
  const identity = mat4Create();
  const nodes = (rootScene.nodes ?? []).map((i: number) => buildNode(i, identity));

  // Flatten all primitives
  const allPrimitives: { primitive: GPUPrimitive; node: SceneNode }[] = [];
  const collectPrimitives = (node: SceneNode) => {
    if (node.mesh) {
      for (const prim of node.mesh.primitives) {
        allPrimitives.push({ primitive: prim, node });
      }
    }
    node.children.forEach(collectPrimitives);
  };
  nodes.forEach(collectPrimitives);

  // Scene AABB
  let sMinX=Infinity,sMinY=Infinity,sMinZ=Infinity;
  let sMaxX=-Infinity,sMaxY=-Infinity,sMaxZ=-Infinity;
  for (const { primitive: p, node } of allPrimitives) {
    const w = node.worldMatrix;
    const corners = [p.aabbMin, p.aabbMax].flatMap(([x,y,z]) => [
      [x,y,z],[x,y,p.aabbMax[2]],[x,p.aabbMax[1],z],[x,p.aabbMax[1],p.aabbMax[2]],
    ]);
    for (const [cx,cy,cz] of corners) {
      const wx = w[0]*cx+w[4]*cy+w[8]*cz+w[12];
      const wy = w[1]*cx+w[5]*cy+w[9]*cz+w[13];
      const wz = w[2]*cx+w[6]*cy+w[10]*cz+w[14];
      if(wx<sMinX)sMinX=wx; if(wy<sMinY)sMinY=wy; if(wz<sMinZ)sMinZ=wz;
      if(wx>sMaxX)sMaxX=wx; if(wy>sMaxY)sMaxY=wy; if(wz>sMaxZ)sMaxZ=wz;
    }
  }

  return {
    nodes,
    materials: materials.length > 0 ? materials : [createDefaultMaterial(device)],
    meshes,
    allPrimitives,
    aabbMin: [sMinX,sMinY,sMinZ],
    aabbMax: [sMaxX,sMaxY,sMaxZ],
  };
}

function createDefaultMaterial(device: GPUDevice): GPUMaterial {
  const uniformData = new Float32Array(16);
  uniformData[0]=0.8; uniformData[1]=0.8; uniformData[2]=0.8; uniformData[3]=1.0;
  uniformData[7]=0.0; uniformData[8]=0.5; uniformData[9]=1.0;
  const uniformBuffer = createBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, uniformData);
  return {
    baseColorFactor: [0.8,0.8,0.8,1.0],
    emissiveFactor: [0,0,0],
    metallicFactor: 0.0,
    roughnessFactor: 0.5,
    aoStrength: 1.0,
    alphaMode: 'OPAQUE',
    alphaCutoff: 0.5,
    doubleSided: false,
    baseColorTexture: null,
    normalTexture: null,
    metallicRoughnessTexture: null,
    occlusionTexture: null,
    emissiveTexture: null,
    uniformBuffer,
  };
}
