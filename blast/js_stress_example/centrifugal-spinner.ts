/**
 * Centrifugal Spinner Demo
 *
 * The clearest possible A/B for the stress solver's centrifugal acceleration. A handful of
 * free-floating segmented beams are spun hard in zero gravity. Nothing else acts on them, so:
 *
 *   - Centrifugal OFF → the beams spin forever, perfectly intact.
 *   - Centrifugal ON  → each beam's segments need an inward (centripetal) pull to stay on their
 *                        circular path; the solver supplies it as compression in the radial bonds,
 *                        which overstresses and the beams shatter from the inside out.
 *
 * Flip the "Centrifugal acceleration" checkbox in the panel to see the difference live. The same
 * scene + spin is exercised headlessly in `centrifugal.showcase.test.ts`.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { buildDestructibleCore } from 'blast-stress-solver/rapier';
import {
  createDestructibleThreeBundle,
  RapierDebugRenderer,
} from 'blast-stress-solver/three';
import { buildSpinningBeamsScenario } from 'blast-stress-solver/scenarios';

// ── Config ────────────────────────────────────────────────────

const CONFIG = {
  centrifugal: false, // start OFF so the user sees it spin intact, then flips it on
  spin: 20, // rad/s about +Y (live)
  // One centered beam by default: it spins about its own centre and snaps in the middle.
  // Raising "Beams" adds more (they ride one free body and break together).
  beams: 1,
  segments: 9,
  // Compression fatal limit (Reset to apply). The centrifugal stress is large in solver units even
  // at modest spin, so this needs a wide range: start strong enough that the beam holds together,
  // then lower it (or raise Spin rate) until it shatters.
  bondStrength: 1.0e6,
};

// ── Three.js setup ────────────────────────────────────────────

const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d13);

const camera = new THREE.PerspectiveCamera(
  55,
  canvas.clientWidth / canvas.clientHeight,
  0.1,
  400,
);
// The beams float at y≈6 (above the core's invisible ground plane), so look there.
camera.position.set(0, 8, 18);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 6, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffeedd, 1.0);
dirLight.position.set(8, 14, 10);
scene.add(dirLight);

// A faint grid at the ground plane (y=0) for orientation; the beams float well above it.
const grid = new THREE.GridHelper(40, 40, 0x223044, 0x152030);
grid.position.y = 0;
scene.add(grid);

// ── Stats panel ──────────────────────────────────────────────

const stats = new Stats();
stats.dom.style.position = 'absolute';
stats.dom.style.top = '0';
stats.dom.style.left = '0';
(document.querySelector('.viewport') as HTMLElement)?.appendChild(stats.dom);

let _physicsMs = 0;
let _renderMs = 0;
const EMA = 0.12;

function updatePerfStats() {
  const el = (id: string) => document.getElementById(id);
  el('stat-physics-ms')!.textContent = _physicsMs.toFixed(1) + ' ms';
  el('stat-render-ms')!.textContent = _renderMs.toFixed(1) + ' ms';
  el('stat-draw-calls')!.textContent = String(renderer.info.render.calls);
  el('stat-triangles')!.textContent = renderer.info.render.triangles.toLocaleString();
}

function updateStatus(core: any) {
  const el = (id: string) => document.getElementById(id);
  el('stat-bodies')!.textContent = String(core.getRigidBodyCount());
  el('stat-bonds')!.textContent = String(core.getActiveBondsCount());
  const active = core.chunks.filter((c: any) => c.active).length;
  const detached = core.chunks.filter((c: any) => c.detached).length;
  el('stat-chunks')!.textContent = `${active} / ${detached} detached`;
  el('stat-fragments')!.textContent = String(core.chunks.length);
}

// ── Main ─────────────────────────────────────────────────────

let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let showDebug = false;

async function initScene() {
  const scenario = buildSpinningBeamsScenario({
    beams: CONFIG.beams,
    segments: CONFIG.segments,
    segmentSize: 0.6,
  });

  const core = await buildDestructibleCore({
    scenario,
    gravity: 0, // zero gravity isolates the centrifugal effect
    // Weak in compression (the spin loads bonds inward), strong otherwise, so only centrifugal
    // load can break the beams.
    solverSettings: {
      compressionElasticLimit: CONFIG.bondStrength * 0.5,
      compressionFatalLimit: CONFIG.bondStrength,
      tensionElasticLimit: 1.0e6,
      tensionFatalLimit: 1.0e7,
      shearElasticLimit: 1.0e6,
      shearFatalLimit: 1.0e7,
    },
    // Run the solver every frame so toggling the checkbox responds immediately, and keep the
    // bodies awake/undamped so they keep spinning.
    fracturePolicy: { idleSkip: false },
    sleepMode: 'off',
    smallBodyDamping: { mode: 'off' },
    debrisCleanup: { mode: 'off' },
    damage: { enabled: false },
  });

  core.setSolverCentrifugalEnabled(CONFIG.centrifugal);

  const group = new THREE.Group();
  scene.add(group);

  const visuals = createDestructibleThreeBundle({
    core,
    scenario,
    root: group,
    useBatchedMesh: true,
    batchedMeshOptions: { enableBVH: false, bvhMargin: 5 },
    includeDebugLines: true,
  });

  rapierDebug?.dispose();
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: showDebug });

  coreRef = core;
  visualsRef = visuals;
}

/** Drive every live dynamic body at the configured spin about +Y, so beams keep tumbling. */
function applySpin(core: any) {
  const world = core.world;
  const seen = new Set<number>();
  for (const chunk of core.chunks) {
    if (!chunk.active || chunk.bodyHandle == null || seen.has(chunk.bodyHandle)) continue;
    seen.add(chunk.bodyHandle);
    const body = world.getRigidBody(chunk.bodyHandle);
    if (!body || body.isFixed()) continue;
    body.setAngularDamping(0);
    body.setAngvel({ x: 0, y: CONFIG.spin, z: 0 }, true);
  }
}

// ── UI wiring ────────────────────────────────────────────────

document.getElementById('btn-reset')?.addEventListener('click', async () => {
  visualsRef?.dispose();
  coreRef?.dispose();
  coreRef = null;
  visualsRef = null;
  await initScene();
});

document.getElementById('btn-debug')?.addEventListener('click', () => {
  showDebug = !showDebug;
  rapierDebug?.setEnabled(showDebug);
  const btn = document.getElementById('btn-debug')!;
  btn.textContent = showDebug ? '◈ Hide Debug' : '◇ Show Debug';
});

function bindSlider(id: string, obj: Record<string, any>, key: string, fmt?: (v: number) => string) {
  const slider = document.getElementById(id) as HTMLInputElement | null;
  const display = document.getElementById(id + '-value');
  if (!slider) return;
  slider.value = String(obj[key]);
  if (display) display.textContent = fmt ? fmt(obj[key]) : String(obj[key]);
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    obj[key] = v;
    if (display) display.textContent = fmt ? fmt(v) : String(v);
  });
}

function bindCheckbox(id: string, obj: Record<string, any>, key: string, onChange?: (v: boolean) => void) {
  const checkbox = document.getElementById(id) as HTMLInputElement | null;
  if (!checkbox) return;
  checkbox.checked = !!obj[key];
  checkbox.addEventListener('change', () => {
    obj[key] = checkbox.checked;
    onChange?.(checkbox.checked);
  });
}

// Live: toggling centrifugal applies immediately to the running core.
bindCheckbox('cfg-centrifugal', CONFIG, 'centrifugal', (v) => coreRef?.setSolverCentrifugalEnabled(v));
// Live: spin rate is read every frame in applySpin().
bindSlider('cfg-spin', CONFIG, 'spin', (v) => v.toFixed(0));
// Deferred (need Reset): scene shape and bond strength.
bindSlider('cfg-beams', CONFIG, 'beams');
bindSlider('cfg-segments', CONFIG, 'segments');
// Bond strength is logarithmic: the slider holds log10(compressionFatalLimit) over a wide range
// (1 … 1e8) so there is actually a setting where the beam survives the spin. Reset to apply.
{
  const slider = document.getElementById('cfg-bond-strength') as HTMLInputElement | null;
  const display = document.getElementById('cfg-bond-strength-value');
  const fmt = (v: number) => (v >= 1000 ? v.toExponential(1) : v.toFixed(1));
  if (slider) {
    slider.value = String(Math.log10(CONFIG.bondStrength));
    if (display) display.textContent = fmt(CONFIG.bondStrength);
    slider.addEventListener('input', () => {
      CONFIG.bondStrength = Math.pow(10, parseFloat(slider.value));
      if (display) display.textContent = fmt(CONFIG.bondStrength);
    });
  }
}

// ── Render loop ──────────────────────────────────────────────

const clock = new THREE.Clock();

function loop() {
  requestAnimationFrame(loop);
  stats.begin();
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();

  if (coreRef && visualsRef) {
    applySpin(coreRef);
    const t0 = performance.now();
    coreRef.step(dt);
    _physicsMs += ((performance.now() - t0) - _physicsMs) * EMA;
    visualsRef.update({ debug: showDebug, updateBVH: false, updateProjectiles: false });
    rapierDebug?.update();
    updateStatus(coreRef);
  }

  const t1 = performance.now();
  renderer.render(scene, camera);
  _renderMs += ((performance.now() - t1) - _renderMs) * EMA;
  updatePerfStats();
  stats.end();
}

// ── Resize ───────────────────────────────────────────────────

function onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

// ── Boot ─────────────────────────────────────────────────────

initScene().then(() => loop()).catch((err) => {
  console.error('Failed to initialize centrifugal spinner demo:', err);
  const hint = document.querySelector('.viewport-hint');
  if (hint) hint.textContent = `Error: ${err.message}`;
});
