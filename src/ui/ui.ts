import type { Renderer, RenderSettings } from '../core/renderer.js';
import { createLight, hexToLinear, linearToHex, type Light } from '../scene/light.js';
import type { OrbitCamera } from '../scene/camera.js';

export class UI {
  private renderer: Renderer;
  private camera: OrbitCamera;
  private panelOpen = false;

  constructor(renderer: Renderer, camera: OrbitCamera) {
    this.renderer = renderer;
    this.camera = camera;
  }

  init(): void {
    this._bindTogglePanel();
    this._bindSettings();
    this._bindLights();
    this._bindToolbar();
    this._renderLightList();
  }

  private _bindTogglePanel(): void {
    const btn = document.getElementById('toggle-panel-btn')!;
    const panel = document.getElementById('panel')!;
    btn.addEventListener('click', () => {
      this.panelOpen = !this.panelOpen;
      panel.classList.toggle('open', this.panelOpen);
      btn.textContent = this.panelOpen ? 'Settings ◂' : 'Settings ▸';
    });
  }

  private _bindSettings(): void {
    const r = this.renderer;
    const s = r.settings;

    // Render mode
    const modeSelect = document.getElementById('render-mode') as HTMLSelectElement;
    modeSelect.addEventListener('change', () => {
      s.renderMode = modeSelect.value as 'realtime' | 'pathtrace';
      r.resetPTAccumulation();
      const overlay = document.getElementById('photomode-overlay')!;
      overlay.style.display = s.renderMode === 'pathtrace' ? 'block' : 'none';
    });

    // Toggles
    document.querySelectorAll<HTMLElement>('[data-toggle]').forEach(toggle => {
      const key = toggle.dataset.toggle as keyof RenderSettings;
      toggle.classList.toggle('active', s[key] as boolean);
      toggle.addEventListener('click', () => {
        (s as any)[key] = !s[key];
        toggle.classList.toggle('active', s[key] as boolean);
        r.resetPTAccumulation();
      });
    });

    // Exposure
    const exposureSlider = document.getElementById('exposure') as HTMLInputElement;
    const exposureVal = document.getElementById('exposure-val')!;
    exposureSlider.addEventListener('input', () => {
      s.exposure = parseFloat(exposureSlider.value);
      exposureVal.textContent = s.exposure.toFixed(1);
    });

    // Tonemap
    const tmSelect = document.getElementById('tonemap-mode') as HTMLSelectElement;
    tmSelect.addEventListener('change', () => {
      s.tonemapMode = parseInt(tmSelect.value);
    });

    // IBL intensity
    const iblSlider = document.getElementById('ibl-intensity') as HTMLInputElement;
    const iblVal = document.getElementById('ibl-val')!;
    iblSlider.addEventListener('input', () => {
      s.iblIntensity = parseFloat(iblSlider.value);
      iblVal.textContent = s.iblIntensity.toFixed(1);
    });

    // FOV
    const fovSlider = document.getElementById('fov') as HTMLInputElement;
    const fovVal = document.getElementById('fov-val')!;
    fovSlider.addEventListener('input', () => {
      this.camera.fovY = parseFloat(fovSlider.value) * Math.PI / 180;
      fovVal.textContent = fovSlider.value + '°';
      r.resetPTAccumulation();
    });

    // Near / Far
    const nearInput = document.getElementById('near') as HTMLInputElement;
    const farInput = document.getElementById('far') as HTMLInputElement;
    nearInput.addEventListener('change', () => { this.camera.near = parseFloat(nearInput.value); });
    farInput.addEventListener('change', () => { this.camera.far = parseFloat(farInput.value); });

    // Reset / Fit camera
    document.getElementById('reset-camera')?.addEventListener('click', () => {
      this.camera.reset();
      r.resetPTAccumulation();
    });
    document.getElementById('fit-camera')?.addEventListener('click', () => {
      // Dispatches fitToScene event for main.ts to handle
      window.dispatchEvent(new CustomEvent('fit-camera'));
    });
  }

  private _bindLights(): void {
    document.getElementById('add-point')?.addEventListener('click', () => {
      this.renderer.lights.push(createLight('point'));
      this._renderLightList();
      this.renderer.resetPTAccumulation();
    });
    document.getElementById('add-dir')?.addEventListener('click', () => {
      this.renderer.lights.push(createLight('directional'));
      this._renderLightList();
      this.renderer.resetPTAccumulation();
    });
  }

  private _bindToolbar(): void {
    const r = this.renderer;
    const s = r.settings;

    document.getElementById('tb-realtime')?.addEventListener('click', () => {
      s.renderMode = 'realtime';
      document.getElementById('tb-realtime')?.classList.add('active');
      document.getElementById('tb-photo')?.classList.remove('active');
      (document.getElementById('render-mode') as HTMLSelectElement).value = 'realtime';
      document.getElementById('photomode-overlay')!.style.display = 'none';
    });

    document.getElementById('tb-photo')?.addEventListener('click', () => {
      s.renderMode = 'pathtrace';
      r.resetPTAccumulation();
      document.getElementById('tb-photo')?.classList.add('active');
      document.getElementById('tb-realtime')?.classList.remove('active');
      (document.getElementById('render-mode') as HTMLSelectElement).value = 'pathtrace';
      document.getElementById('photomode-overlay')!.style.display = 'block';
    });

    document.getElementById('tb-load')?.addEventListener('click', () => {
      document.getElementById('file-input')?.click();
    });

    document.getElementById('tb-hdri')?.addEventListener('click', () => {
      document.getElementById('hdri-input')?.click();
    });

    document.getElementById('load-hdri-btn')?.addEventListener('click', () => {
      document.getElementById('hdri-input')?.click();
    });

    document.getElementById('tb-screenshot')?.addEventListener('click', () => {
      this._takeScreenshot();
    });
  }

  private _renderLightList(): void {
    const list = document.getElementById('lights-list')!;
    list.innerHTML = '';

    this.renderer.lights.forEach((light, idx) => {
      const item = document.createElement('div');
      item.className = 'light-item';
      item.innerHTML = `
        <div class="light-item-header">
          <span>${light.type === 'directional' ? '☀' : '💡'} Light ${idx + 1} (${light.type})</span>
          <span class="del" data-idx="${idx}">✕</span>
        </div>
        <div class="light-item-body">
          <div class="row">
            <label>Color</label>
            <input type="color" class="light-color" value="${linearToHex(light.color)}" />
          </div>
          <div class="row">
            <label>Intensity</label>
            <input type="range" class="light-intensity" min="0" max="20" step="0.1" value="${light.intensity}" />
            <span class="val">${light.intensity.toFixed(1)}</span>
          </div>
          ${light.type !== 'directional' ? `
          <div class="row">
            <label>Pos X</label>
            <input type="number" class="light-px" step="0.1" value="${light.position[0].toFixed(2)}" style="width:60px" />
          </div>
          <div class="row">
            <label>Pos Y</label>
            <input type="number" class="light-py" step="0.1" value="${light.position[1].toFixed(2)}" style="width:60px" />
          </div>
          <div class="row">
            <label>Pos Z</label>
            <input type="number" class="light-pz" step="0.1" value="${light.position[2].toFixed(2)}" style="width:60px" />
          </div>
          <div class="row">
            <label>Range</label>
            <input type="range" class="light-range" min="0.5" max="100" step="0.5" value="${light.range}" />
            <span class="val">${light.range.toFixed(0)}</span>
          </div>
          ` : `
          <div class="row">
            <label>Dir X</label>
            <input type="number" class="light-dx" step="0.1" value="${light.direction[0].toFixed(2)}" style="width:60px" />
          </div>
          <div class="row">
            <label>Dir Y</label>
            <input type="number" class="light-dy" step="0.1" value="${light.direction[1].toFixed(2)}" style="width:60px" />
          </div>
          <div class="row">
            <label>Dir Z</label>
            <input type="number" class="light-dz" step="0.1" value="${light.direction[2].toFixed(2)}" style="width:60px" />
          </div>
          `}
        </div>
      `;

      // Toggle expand
      item.querySelector('.light-item-header')?.addEventListener('click', (e) => {
        if ((e.target as HTMLElement).classList.contains('del')) return;
        item.classList.toggle('expanded');
      });

      // Delete
      item.querySelector('.del')?.addEventListener('click', () => {
        this.renderer.lights.splice(idx, 1);
        this._renderLightList();
        this.renderer.resetPTAccumulation();
      });

      // Color
      item.querySelector<HTMLInputElement>('.light-color')?.addEventListener('input', (e) => {
        light.color = hexToLinear((e.target as HTMLInputElement).value);
        this.renderer.resetPTAccumulation();
      });

      // Intensity
      const intSlider = item.querySelector<HTMLInputElement>('.light-intensity')!;
      const intVal = intSlider.nextElementSibling as HTMLElement;
      intSlider.addEventListener('input', () => {
        light.intensity = parseFloat(intSlider.value);
        intVal.textContent = light.intensity.toFixed(1);
        this.renderer.resetPTAccumulation();
      });

      // Position inputs
      const setPos = (cl: string, axis: number) => {
        item.querySelector<HTMLInputElement>(cl)?.addEventListener('change', (e) => {
          light.position[axis] = parseFloat((e.target as HTMLInputElement).value);
          this.renderer.resetPTAccumulation();
        });
      };
      setPos('.light-px', 0); setPos('.light-py', 1); setPos('.light-pz', 2);

      // Direction inputs
      const setDir = (cl: string, axis: number) => {
        item.querySelector<HTMLInputElement>(cl)?.addEventListener('change', (e) => {
          light.direction[axis] = parseFloat((e.target as HTMLInputElement).value);
          this.renderer.resetPTAccumulation();
        });
      };
      setDir('.light-dx', 0); setDir('.light-dy', 1); setDir('.light-dz', 2);

      // Range
      const rangeSlider = item.querySelector<HTMLInputElement>('.light-range');
      if (rangeSlider) {
        const rangeVal = rangeSlider.nextElementSibling as HTMLElement;
        rangeSlider.addEventListener('input', () => {
          light.range = parseFloat(rangeSlider.value);
          rangeVal.textContent = light.range.toFixed(0);
        });
      }

      list.appendChild(item);
    });
  }

  updateStats(fps: number, frameMs: number): void {
    const fpsEl = document.getElementById('fps');
    const ftEl = document.getElementById('frame-time');
    const triEl = document.getElementById('tri-count');
    const ptEl = document.getElementById('pt-samples');
    if (fpsEl) fpsEl.textContent = fps.toFixed(0);
    if (ftEl) ftEl.textContent = frameMs.toFixed(1);
    if (triEl) triEl.textContent = this.renderer.triangleCount.toLocaleString();
    if (ptEl) ptEl.textContent = this.renderer.ptSamples.toString();
  }

  private _takeScreenshot(): void {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `webgpu-render-${Date.now()}.png`;
    link.click();
  }

  refreshLights(): void {
    this._renderLightList();
  }
}
