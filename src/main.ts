import { initWebGPU, type GPUContext } from './core/gpu.js';
import { Renderer } from './core/renderer.js';
import { OrbitCamera } from './scene/camera.js';
import { getEntryFile } from './scene/gltf-file-resolver.js';
import { loadGLTF } from './scene/gltf-loader.js';
import { computeIBL } from './passes/ibl-pass.js';
import { parseRGBEProper } from './utils/hdri.js';
import { UI } from './ui/ui.js';

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const canvas      = document.getElementById('canvas') as HTMLCanvasElement;
const welcome     = document.getElementById('welcome') as HTMLDivElement;
const dropOverlay = document.getElementById('drop-overlay') as HTMLDivElement;
const fileInput   = document.getElementById('file-input') as HTMLInputElement;
const hdriInput   = document.getElementById('hdri-input') as HTMLInputElement;
const loadBtn     = document.getElementById('load-btn') as HTMLButtonElement;

// ─── Canvas sizing ────────────────────────────────────────────────────────────
function setSize() {
  const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
  canvas.width  = Math.floor(window.innerWidth  * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width  = window.innerWidth  + 'px';
  canvas.style.height = window.innerHeight + 'px';
}
setSize();

// ─── Status bar ───────────────────────────────────────────────────────────────
let _statusTimeout: ReturnType<typeof setTimeout>;
function showStatus(msg: string, isError = false) {
  clearTimeout(_statusTimeout);
  let el = document.getElementById('_status');
  if (!el) {
    el = document.createElement('div');
    el.id = '_status';
    Object.assign(el.style, {
      position: 'fixed', bottom: '70px', left: '50%', transform: 'translateX(-50%)',
      padding: '8px 18px', borderRadius: '8px', fontSize: '0.8rem', zIndex: '200',
      pointerEvents: 'none', fontFamily: 'Segoe UI, system-ui, sans-serif',
    });
    document.body.appendChild(el);
  }
  Object.assign(el.style, {
    background: isError ? 'rgba(200,60,60,0.95)' : 'rgba(20,20,30,0.9)',
    border: isError ? '1px solid #f66' : '1px solid rgba(255,255,255,0.1)',
    color: '#fff', display: msg ? 'block' : 'none',
  });
  el.textContent = msg;
  if (msg && !isError) _statusTimeout = setTimeout(() => { el!.style.display = 'none'; }, 4000);
}

// ─── Error screen ─────────────────────────────────────────────────────────────
function showError(title: string, detail: string) {
  document.body.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:center;height:100vh;
                flex-direction:column;color:#f88;font-family:monospace;padding:40px;text-align:center;">
      <h2 style="color:#f44;margin-bottom:16px;">⚠ ${title}</h2>
      <pre style="background:rgba(255,0,0,0.1);padding:20px;border-radius:8px;max-width:700px;
                  white-space:pre-wrap;text-align:left;font-size:0.78rem;color:#fcc;">${detail}</pre>
      <p style="margin-top:20px;color:#888;font-size:0.85rem;">
        <b>Activer WebGPU :</b><br>
        Opera GX / Chrome → <code>chrome://flags/#enable-unsafe-webgpu</code> → <b>Enabled</b><br>
        Redémarre le navigateur, puis recharge la page.
      </p>
    </div>`;
}

// ─── State ────────────────────────────────────────────────────────────────────
let rendererReady = false;
let renderer: Renderer | null = null;
let camera: OrbitCamera | null = null;
let ui: UI | null = null;
let gpu: GPUContext | null = null;
let pendingModelFiles: File[] | null = null;
let pendingHdriFile: File | null = null;

// ─── File processing (called once renderer is ready) ─────────────────────────
async function processModel(files: File[]) {
  const entryFile = getEntryFile(files, ['.gltf', '.glb']);
  if (!entryFile) {
    showStatus('Aucun fichier .gltf ou .glb trouve dans la selection.', true);
    return;
  }
  if (!renderer) { pendingModelFiles = files; return; }
  showStatus(`Chargement de ${entryFile.name}…`);
  try {
    const scene = await loadGLTF(renderer.gpu, entryFile, files);
    renderer.setScene(scene);
    const cx = (scene.aabbMin[0] + scene.aabbMax[0]) * 0.5;
    const cy = (scene.aabbMin[1] + scene.aabbMax[1]) * 0.5;
    const cz = (scene.aabbMin[2] + scene.aabbMax[2]) * 0.5;
    const sz = Math.max(
      scene.aabbMax[0] - scene.aabbMin[0],
      scene.aabbMax[1] - scene.aabbMin[1],
      scene.aabbMax[2] - scene.aabbMin[2],
    ) || 1;
    camera!.fitTo([cx, cy, cz], sz);
    hideWelcome();
    showStatus(`✓ ${entryFile.name} — ${renderer.triangleCount.toLocaleString()} triangles`);
  } catch (e: any) {
    console.error('[loadGLTF]', e);
    showStatus(`Erreur : ${e.message}`, true);
  }
}

async function processHDRI(file: File) {
  if (!renderer) { pendingHdriFile = file; return; }
  showStatus('Chargement HDRI…');
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const hdr   = parseRGBEProper(bytes);
    const envTex = renderer.loadHDRIFromData(hdr);
    showStatus('Calcul IBL (irradiance, prefilter, LUT)…');
    const ibl = await computeIBL(renderer.gpu, envTex);
    renderer.setIBL(ibl, envTex);
    renderer.resetPTAccumulation();
    showStatus(`✓ HDRI ${file.name} chargé (${hdr.width}×${hdr.height})`);
  } catch (e: any) {
    console.error('[loadHDRI]', e);
    showStatus(`Erreur HDRI : ${e.message}`, true);
  }
}

function hideWelcome() {
  welcome.style.opacity = '0';
  welcome.style.pointerEvents = 'none';
}

// ─── PHASE 1 — Brancher les inputs IMMÉDIATEMENT (avant init renderer) ───────
// The file inputs and drag & drop work even before WebGPU is ready.
// Files are queued if renderer isn't initialized yet.

fileInput.addEventListener('change', () => {
  const files = Array.from(fileInput.files ?? []);
  if (files.length > 0) processModel(files);
  fileInput.value = '';
});

hdriInput.addEventListener('change', () => {
  const f = hdriInput.files?.[0];
  if (f) processHDRI(f);
  hdriInput.value = '';
});

loadBtn.addEventListener('click', () => fileInput.click());

// Toolbar buttons — bound early too (querySelector might return null if renderer hasn't run ui.init yet)
document.getElementById('tb-load')?.addEventListener('click', () => fileInput.click());
document.getElementById('tb-hdri')?.addEventListener('click', () => hdriInput.click());
document.getElementById('load-hdri-btn')?.addEventListener('click', () => hdriInput.click());

// Drag & drop
let _dragCount = 0;
document.addEventListener('dragenter', (e) => {
  _dragCount++;
  dropOverlay.classList.add('active');
  e.preventDefault();
});
document.addEventListener('dragleave', () => {
  _dragCount = Math.max(0, _dragCount - 1);
  if (_dragCount === 0) dropOverlay.classList.remove('active');
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
  e.preventDefault();
  _dragCount = 0;
  dropOverlay.classList.remove('active');
  const files = Array.from(e.dataTransfer?.files ?? []);
  if (files.length === 0) return;
  const modelFile = getEntryFile(files, ['.gltf', '.glb']);
  const hdriFile = getEntryFile(files, ['.hdr']);
  if (modelFile) processModel(files);
  else if (hdriFile) processHDRI(hdriFile);
  else showStatus(`Format non reconnu : ${files[0].name}`, true);
});

// ─── PHASE 2 — Init WebGPU & Renderer (async) ────────────────────────────────
async function main() {
  // Small delay to let the browser paint the HTML first
  await new Promise(r => setTimeout(r, 50));

  try {
    gpu = await initWebGPU(canvas);
  } catch (err: any) {
    showError('WebGPU non disponible', err.message);
    return;
  }

  camera = new OrbitCamera(canvas);
  renderer = new Renderer(gpu, camera);

  try {
    showStatus('Initialisation WebGPU…');
    await renderer.init();
    showStatus('');
  } catch (err: any) {
    console.error('[renderer.init]', err);
    showError('Erreur d\'initialisation du renderer', err.message ?? String(err));
    return;
  }

  // Renderer ready — wire resize and UI
  window.addEventListener('resize', () => { setSize(); renderer!.resize(); });

  ui = new UI(renderer, camera);
  ui.init();
  rendererReady = true;

  // Process any files dropped before renderer was ready
  if (pendingModelFiles) { const files = pendingModelFiles; pendingModelFiles = null; await processModel(files); }
  if (pendingHdriFile)  { const f = pendingHdriFile;  pendingHdriFile  = null; await processHDRI(f); }

  // Fit camera on event from UI
  window.addEventListener('fit-camera', () => {
    const s = (renderer as any)?.scene;
    if (!s) return;
    const cx = (s.aabbMin[0] + s.aabbMax[0]) * 0.5;
    const cy = (s.aabbMin[1] + s.aabbMax[1]) * 0.5;
    const cz = (s.aabbMin[2] + s.aabbMax[2]) * 0.5;
    const sz = Math.max(s.aabbMax[0]-s.aabbMin[0], s.aabbMax[1]-s.aabbMin[1], s.aabbMax[2]-s.aabbMin[2]) || 1;
    camera!.fitTo([cx, cy, cz], sz);
  });

  // Screenshot
  document.getElementById('tb-screenshot')?.addEventListener('click', () => {
    const link = document.createElement('a');
    link.href  = canvas.toDataURL('image/png');
    link.download = `render-${Date.now()}.png`;
    link.click();
  });

  // ── Render loop ─────────────────────────────────────────────────────────────
  let lastTime = performance.now();
  let fpsFrames = 0, fpsAccum = 0, lastFpsUpdate = 0;

  function frame(time: number) {
    const dt = time - lastTime;
    lastTime = time;

    try {
      renderer!.render(dt);
    } catch (e: any) {
      console.error('[render loop]', e);
      showStatus('Erreur de rendu : ' + e.message, true);
    }

    fpsFrames++;
    fpsAccum += dt;
    if (time - lastFpsUpdate > 600) {
      ui?.updateStats(fpsFrames * 1000 / fpsAccum, fpsAccum / fpsFrames);
      fpsFrames = 0; fpsAccum = 0; lastFpsUpdate = time;
    }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
