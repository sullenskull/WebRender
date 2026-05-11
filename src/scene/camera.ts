import { mat4Create, mat4Perspective, mat4LookAt, mat4Invert, mat4Transpose, Vec3, halton } from '../utils/math.js';
import type { Mat4 } from '../utils/math.js';

export interface CameraUniforms {
  view: Mat4;
  proj: Mat4;
  viewProj: Mat4;
  invViewProj: Mat4;
  prevViewProj: Mat4;
  position: Vec3;
  near: number;
  far: number;
  fovY: number;
  aspect: number;
  jitter: [number, number];
  frameIndex: number;
}

export class OrbitCamera {
  // Orbit state
  target: Vec3 = [0, 0, 0];
  radius = 3;
  theta = Math.PI * 0.25;  // horizontal angle
  phi = Math.PI * 0.3;     // vertical angle

  fovY = 60 * Math.PI / 180;
  near = 0.01;
  far = 1000;
  aspect = 1;

  private _frameIndex = 0;
  private _jitterEnabled = true;
  private _prevViewProj: Mat4 = mat4Create();

  // Computed matrices
  view: Mat4 = mat4Create();
  proj: Mat4 = mat4Create();
  viewProj: Mat4 = mat4Create();
  invViewProj: Mat4 = mat4Create();

  // Mouse state
  private _dragging = false;
  private _panning = false;
  private _lastX = 0;
  private _lastY = 0;

  constructor(canvas: HTMLCanvasElement) {
    this._bindEvents(canvas);
  }

  get position(): Vec3 {
    const sinPhi = Math.sin(this.phi);
    const cosPhi = Math.cos(this.phi);
    const sinTheta = Math.sin(this.theta);
    const cosTheta = Math.cos(this.theta);
    return [
      this.target[0] + this.radius * sinPhi * sinTheta,
      this.target[1] + this.radius * cosPhi,
      this.target[2] + this.radius * sinPhi * cosTheta,
    ];
  }

  update(aspect: number): void {
    this.aspect = aspect;

    // Save previous
    this._prevViewProj.set(this.viewProj);

    const pos = this.position;
    mat4LookAt(this.view, pos, this.target, [0,1,0]);
    mat4Perspective(this.proj, this.fovY, aspect, this.near, this.far);

    // Apply TAA sub-pixel jitter (~1 pixel in NDC)
    if (this._jitterEnabled) {
      const scale = 1.5 / 1080; // ~1.5 pixels at 1080p
      const jx = (halton(this._frameIndex, 2) - 0.5) * scale * 2;
      const jy = (halton(this._frameIndex, 3) - 0.5) * scale * 2;
      this.proj[8] += jx;
      this.proj[9] += jy;
    }

    // viewProj = proj * view
    multiplyMat4(this.viewProj, this.proj, this.view);
    mat4Invert(this.invViewProj, this.viewProj);

    this._frameIndex++;
  }

  get jitter(): [number, number] {
    if (!this._jitterEnabled) return [0, 0];
    const idx = this._frameIndex - 1;
    return [
      (halton(idx, 2) - 0.5) * 2,
      (halton(idx, 3) - 0.5) * 2,
    ];
  }

  get prevViewProj(): Mat4 { return this._prevViewProj; }
  get frameIndex(): number { return this._frameIndex; }

  set jitterEnabled(v: boolean) { this._jitterEnabled = v; }

  fitTo(center: Vec3, size: number): void {
    this.target = [...center];
    this.radius = size * 2.0;
    this.theta = Math.PI * 0.35;
    this.phi = Math.PI * 0.35;
  }

  reset(): void {
    this.target = [0, 0, 0];
    this.radius = 3;
    this.theta = Math.PI * 0.25;
    this.phi = Math.PI * 0.3;
    this._frameIndex = 0;
  }

  private _bindEvents(canvas: HTMLCanvasElement): void {
    canvas.addEventListener('mousedown', (e) => {
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      if (e.button === 0) this._dragging = true;
      if (e.button === 1 || e.button === 2) this._panning = true;
      e.preventDefault();
    });
    window.addEventListener('mouseup', () => { this._dragging = false; this._panning = false; });
    window.addEventListener('mousemove', (e) => {
      const dx = e.clientX - this._lastX;
      const dy = e.clientY - this._lastY;
      this._lastX = e.clientX;
      this._lastY = e.clientY;

      if (this._dragging) {
        this.theta -= dx * 0.005;
        this.phi = Math.max(0.05, Math.min(Math.PI - 0.05, this.phi - dy * 0.005));
        this._frameIndex = 0; // reset TAA on movement
      }
      if (this._panning) {
        const speed = this.radius * 0.001;
        // compute right and up vectors
        const pos = this.position;
        const fwd: Vec3 = [this.target[0]-pos[0], this.target[1]-pos[1], this.target[2]-pos[2]];
        const len = Math.hypot(...fwd); fwd[0]/=len; fwd[1]/=len; fwd[2]/=len;
        const right: Vec3 = [fwd[2], 0, -fwd[0]]; // simplified
        const up: Vec3 = [0, 1, 0];
        this.target[0] -= right[0] * dx * speed + up[0] * dy * speed;
        this.target[1] -= right[1] * dx * speed + up[1] * dy * speed;
        this.target[2] -= right[2] * dx * speed + up[2] * dy * speed;
        this._frameIndex = 0;
      }
    });
    canvas.addEventListener('wheel', (e) => {
      this.radius *= 1 + e.deltaY * 0.001;
      this.radius = Math.max(0.01, this.radius);
      this._frameIndex = 0;
      e.preventDefault();
    }, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    // Touch support
    let lastTouchDist = 0;
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this._dragging = true;
        this._lastX = e.touches[0].clientX;
        this._lastY = e.touches[0].clientY;
      }
      if (e.touches.length === 2) {
        this._dragging = false;
        lastTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    });
    canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && this._dragging) {
        const dx = e.touches[0].clientX - this._lastX;
        const dy = e.touches[0].clientY - this._lastY;
        this._lastX = e.touches[0].clientX;
        this._lastY = e.touches[0].clientY;
        this.theta -= dx * 0.006;
        this.phi = Math.max(0.05, Math.min(Math.PI-0.05, this.phi - dy * 0.006));
        this._frameIndex = 0;
      }
      if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        this.radius *= lastTouchDist / dist;
        this.radius = Math.max(0.01, this.radius);
        lastTouchDist = dist;
        this._frameIndex = 0;
      }
      e.preventDefault();
    }, { passive: false });
    canvas.addEventListener('touchend', () => { this._dragging = false; });
  }
}

// Inline mat4 multiply (avoids import cycle)
function multiplyMat4(out: Mat4, a: Mat4, b: Mat4): void {
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
}
