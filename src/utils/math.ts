export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];
export type Mat4 = Float32Array; // column-major, 16 floats

export const PI = Math.PI;
export const TAU = Math.PI * 2;

export function mat4Create(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4Multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
  const a00=a[0],a01=a[1],a02=a[2],a03=a[3];
  const a10=a[4],a11=a[5],a12=a[6],a13=a[7];
  const a20=a[8],a21=a[9],a22=a[10],a23=a[11];
  const a30=a[12],a31=a[13],a32=a[14],a33=a[15];
  let b0=b[0],b1=b[1],b2=b[2],b3=b[3];
  out[0]=b0*a00+b1*a10+b2*a20+b3*a30; out[1]=b0*a01+b1*a11+b2*a21+b3*a31;
  out[2]=b0*a02+b1*a12+b2*a22+b3*a32; out[3]=b0*a03+b1*a13+b2*a23+b3*a33;
  b0=b[4];b1=b[5];b2=b[6];b3=b[7];
  out[4]=b0*a00+b1*a10+b2*a20+b3*a30; out[5]=b0*a01+b1*a11+b2*a21+b3*a31;
  out[6]=b0*a02+b1*a12+b2*a22+b3*a32; out[7]=b0*a03+b1*a13+b2*a23+b3*a33;
  b0=b[8];b1=b[9];b2=b[10];b3=b[11];
  out[8]=b0*a00+b1*a10+b2*a20+b3*a30; out[9]=b0*a01+b1*a11+b2*a21+b3*a31;
  out[10]=b0*a02+b1*a12+b2*a22+b3*a32; out[11]=b0*a03+b1*a13+b2*a23+b3*a33;
  b0=b[12];b1=b[13];b2=b[14];b3=b[15];
  out[12]=b0*a00+b1*a10+b2*a20+b3*a30; out[13]=b0*a01+b1*a11+b2*a21+b3*a31;
  out[14]=b0*a02+b1*a12+b2*a22+b3*a32; out[15]=b0*a03+b1*a13+b2*a23+b3*a33;
  return out;
}

// WebGPU: right-handed, NDC depth [0, 1]
export function mat4Perspective(out: Mat4, fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovY * 0.5);
  out.fill(0);
  out[0]  = f / aspect;
  out[5]  = f;
  out[10] = far / (near - far);       // maps far→0, near→1 (reversed-Z) ... actually maps near→-1 then
  out[11] = -1;
  out[14] = (near * far) / (near - far);
  // This maps: near → 0, far → 1 in WebGPU NDC (right-handed, depth 0..1)
  return out;
}

export function mat4LookAt(out: Mat4, eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  let fx=center[0]-eye[0], fy=center[1]-eye[1], fz=center[2]-eye[2];
  let len=1/Math.hypot(fx,fy,fz); fx*=len; fy*=len; fz*=len;
  let sx=fy*up[2]-fz*up[1], sy=fz*up[0]-fx*up[2], sz=fx*up[1]-fy*up[0];
  len=Math.hypot(sx,sy,sz); if(len>0){len=1/len;} sx*=len;sy*=len;sz*=len;
  const ux=sy*fz-sz*fy, uy=sz*fx-sx*fz, uz=sx*fy-sy*fx;
  out[0]=sx; out[1]=ux; out[2]=-fx; out[3]=0;
  out[4]=sy; out[5]=uy; out[6]=-fy; out[7]=0;
  out[8]=sz; out[9]=uz; out[10]=-fz; out[11]=0;
  out[12]=-(sx*eye[0]+sy*eye[1]+sz*eye[2]);
  out[13]=-(ux*eye[0]+uy*eye[1]+uz*eye[2]);
  out[14]=fx*eye[0]+fy*eye[1]+fz*eye[2];
  out[15]=1;
  return out;
}

export function mat4Invert(out: Mat4, a: Mat4): Mat4 {
  const a00=a[0],a01=a[1],a02=a[2],a03=a[3];
  const a10=a[4],a11=a[5],a12=a[6],a13=a[7];
  const a20=a[8],a21=a[9],a22=a[10],a23=a[11];
  const a30=a[12],a31=a[13],a32=a[14],a33=a[15];
  const b00=a00*a11-a01*a10, b01=a00*a12-a02*a10;
  const b02=a00*a13-a03*a10, b03=a01*a12-a02*a11;
  const b04=a01*a13-a03*a11, b05=a02*a13-a03*a12;
  const b06=a20*a31-a21*a30, b07=a20*a32-a22*a30;
  const b08=a20*a33-a23*a30, b09=a21*a32-a22*a31;
  const b10=a21*a33-a23*a31, b11=a22*a33-a23*a32;
  let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if(!det) return out; det=1/det;
  out[0]=(a11*b11-a12*b10+a13*b09)*det; out[1]=(a02*b10-a01*b11-a03*b09)*det;
  out[2]=(a31*b05-a32*b04+a33*b03)*det; out[3]=(a22*b04-a21*b05-a23*b03)*det;
  out[4]=(a12*b08-a10*b11-a13*b07)*det; out[5]=(a00*b11-a02*b08+a03*b07)*det;
  out[6]=(a32*b02-a30*b05-a33*b01)*det; out[7]=(a20*b05-a22*b02+a23*b01)*det;
  out[8]=(a10*b10-a11*b08+a13*b06)*det; out[9]=(a01*b08-a00*b10-a03*b06)*det;
  out[10]=(a30*b04-a31*b02+a33*b00)*det; out[11]=(a21*b02-a20*b04-a23*b00)*det;
  out[12]=(a11*b07-a10*b09-a12*b06)*det; out[13]=(a00*b09-a01*b07+a02*b06)*det;
  out[14]=(a31*b01-a30*b03-a32*b00)*det; out[15]=(a20*b03-a21*b01+a22*b00)*det;
  return out;
}

export function mat4Transpose(out: Mat4, a: Mat4): Mat4 {
  if (out === a) {
    let t;
    t=a[1];out[1]=a[4];out[4]=t; t=a[2];out[2]=a[8];out[8]=t;
    t=a[3];out[3]=a[12];out[12]=t; t=a[6];out[6]=a[9];out[9]=t;
    t=a[7];out[7]=a[13];out[13]=t; t=a[11];out[11]=a[14];out[14]=t;
  } else {
    out[0]=a[0];out[1]=a[4];out[2]=a[8];out[3]=a[12];
    out[4]=a[1];out[5]=a[5];out[6]=a[9];out[7]=a[13];
    out[8]=a[2];out[9]=a[6];out[10]=a[10];out[11]=a[14];
    out[12]=a[3];out[13]=a[7];out[14]=a[11];out[15]=a[15];
  }
  return out;
}

export function mat4Identity(out: Mat4): Mat4 {
  out.fill(0); out[0]=out[5]=out[10]=out[15]=1; return out;
}

export function mat4FromTRS(out: Mat4, t: Vec3, r: [number,number,number,number], s: Vec3): Mat4 {
  const [x,y,z,w]=r;
  const x2=x+x,y2=y+y,z2=z+z;
  const xx=x*x2,xy=x*y2,xz=x*z2;
  const yy=y*y2,yz=y*z2,zz=z*z2;
  const wx=w*x2,wy=w*y2,wz=w*z2;
  const sx=s[0],sy=s[1],sz=s[2];
  out[0]=(1-(yy+zz))*sx; out[1]=(xy+wz)*sx; out[2]=(xz-wy)*sx; out[3]=0;
  out[4]=(xy-wz)*sy; out[5]=(1-(xx+zz))*sy; out[6]=(yz+wx)*sy; out[7]=0;
  out[8]=(xz+wy)*sz; out[9]=(yz-wx)*sz; out[10]=(1-(xx+yy))*sz; out[11]=0;
  out[12]=t[0]; out[13]=t[1]; out[14]=t[2]; out[15]=1;
  return out;
}

export function vec3Normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0]/l, v[1]/l, v[2]/l] : [0,0,0];
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
}

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
}

export function vec3Scale(v: Vec3, s: number): Vec3 {
  return [v[0]*s, v[1]*s, v[2]*s];
}

export function vec3Length(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

// Halton sequence for TAA jitter
export function halton(index: number, base: number): number {
  let f = 1, r = 0;
  while (index > 0) { f /= base; r += f * (index % base); index = Math.floor(index / base); }
  return r;
}

export function mat4OrthoRH(out: Mat4, l: number, r: number, b: number, t: number, near: number, far: number): Mat4 {
  out.fill(0);
  out[0]=2/(r-l); out[5]=2/(t-b); out[10]=1/(near-far);
  out[12]=-(r+l)/(r-l); out[13]=-(t+b)/(t-b); out[14]=near/(near-far); out[15]=1;
  return out;
}

export function mat4FromQuat(out: Mat4, q: [number,number,number,number]): Mat4 {
  return mat4FromTRS(out, [0,0,0], q, [1,1,1]);
}

export function quatMul(a: [number,number,number,number], b: [number,number,number,number]): [number,number,number,number] {
  return [
    a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
    a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
    a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
    a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2],
  ];
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
