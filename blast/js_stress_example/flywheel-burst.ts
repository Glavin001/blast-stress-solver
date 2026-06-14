/**
 * Flywheel Overspeed Burst Demo
 *
 * The centrifugal stress solver on a recognizable machine part. A single steel flywheel — hub,
 * radial spokes, and a closed rim hoop — spins flat about its axis in zero gravity. The only thing
 * that can stress it is its own rotation: the solver's centrifugal term loads every bond inward,
 * hardest out at the rim where ω²r is largest. Below the burst speed it just hums; ramp the RPM up
 * and the rim cracks and sheds arcs outward stage by stage, leaving the hub and spokes spinning in a
 * cloud of debris — the classic turbine/flywheel overspeed failure. (Drop the bond strength and the
 * remaining core lets go too.)
 *
 *   - Centrifugal OFF → spins forever intact, even at redline.
 *   - Centrifugal ON  → crosses the burst RPM and comes apart, shedding fragments outward.
 *
 * "Spin up to burst" ramps the RPM for you. The same scene is exercised headlessly in
 * `flywheel.showcase.test.ts`.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { buildDestructibleCore, createFrameProfilerOverlay, createRecordingOverlay } from 'blast-stress-solver/rapier';
import {
  createDestructibleThreeBundle,
  RapierDebugRenderer,
} from 'blast-stress-solver/three';
import { buildFlywheelScenario } from 'blast-stress-solver/scenarios';
import { mountPhysicsControls, physicsCoreOverrides } from './physics-controls.js';

// ── Config ────────────────────────────────────────────────────

const TAU = Math.PI * 2;
const rpmToOmega = (rpm: number) => (rpm * TAU) / 60;

const CONFIG = {
  centrifugal: true, // the wheel spins intact below burst RPM, so start ON and let the user rev it
  rpm: 100, // below the burst speed: the wheel just hums (live)
  // Compression fatal limit (the rim/spoke bonds load inward → compression). At 1e5 the burst curve
  // measured in this same JS/WASM pipeline is: ≲140 rpm intact, ~160 first rim arcs, ~220 more,
  // ~300 spokes tear, ~400 full disintegration. Lower strength bursts earlier (and throws debris
  // slower, since burst tip speed scales with √strength); slide it to move the whole threshold.
  bondStrength: 1.0e5,
  // Wheel shape (rebuilds the scene on release).
  rimBlocks: 72,
  spokes: 6,
};

const REDLINE_RPM = 400;

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
// The wheel lies flat at y≈6 (above the core's invisible ground plane); pull back and look down so
// the debris fan stays in frame when it bursts.
camera.position.set(14, 21, 18);

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

// A faint grid at the ground plane (y=0) for orientation; the wheel floats well above it.
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

// Total nodes in the wheel, captured at build time so we can report an "intact" fraction.
let totalChunks = 1;

function updateStatus(core: any) {
  const el = (id: string) => document.getElementById(id);
  el('stat-bodies')!.textContent = String(core.getRigidBodyCount());
  el('stat-bonds')!.textContent = String(core.getActiveBondsCount());
  // Fragments shed = dynamic bodies beyond the one intact wheel (minus the invisible ground body).
  const fragments = Math.max(0, core.getRigidBodyCount() - 2);
  el('stat-fragments')!.textContent = String(fragments);
  const intactPct = Math.round((1 - fragments / Math.max(1, totalChunks)) * 100);
  el('stat-intact')!.textContent = `${Math.max(0, intactPct)}%`;
  el('stat-rpm')!.textContent = `${Math.round(CONFIG.rpm)}`;
}

// ── Main ─────────────────────────────────────────────────────

let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let showDebug = false;

// Standard session recorder (● Record / ⬇ Save) + live frame profiler.
const profiler = createFrameProfilerOverlay();
const recorder = createRecordingOverlay({
  mount: document.getElementById('recorder-slot') ?? undefined,
  exportName: 'flywheel-burst-recording',
  getProfilerExport: () => profiler.exportData(),
});

async function initScene() {
  const scenario = buildFlywheelScenario({
    rimBlocks: CONFIG.rimBlocks,
    spokes: CONFIG.spokes,
  });
  totalChunks = scenario.nodes.length;

  const core = await buildDestructibleCore({
    scenario,
    gravity: 0, // zero gravity isolates the centrifugal effect
    // Weak in compression (the spin loads bonds inward), strong otherwise, so only centrifugal
    // load can burst the wheel.
    solverSettings: {
      compressionElasticLimit: CONFIG.bondStrength * 0.5,
      compressionFatalLimit: CONFIG.bondStrength,
      tensionElasticLimit: 1.0e9,
      tensionFatalLimit: 1.0e10,
      shearElasticLimit: 1.0e9,
      shearFatalLimit: 1.0e10,
    },
    // Solve every frame so the RPM slider responds immediately, and keep bodies awake/undamped so
    // they keep spinning.
    fracturePolicy: { idleSkip: false },
    sleepMode: 'off',
    // Shared Physics/Optimization controls (debris cleanup forced 'off' below — spinning debris in
    // zero-g should persist so the burst stays on screen).
    ...physicsCoreOverrides(),
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
  recorder.attach(core, { scenario, meta: { demo: 'flywheel-burst', config: CONFIG } });
  profiler.attach(core);
}

/** Drive every live dynamic body at the configured spin about +Y, so the wheel and its fragments
 *  keep turning. Centrifugal stress is read from each body's actual angular velocity. Shed fragments
 *  leave at the rim's tangential speed (tens of m/s); a little linear damping bleeds that off so the
 *  debris fans out and stays on screen instead of teleporting away — the intact wheel is a single
 *  body whose centre never translates, so the damping only affects fragments. */
function applySpin(core: any) {
  const world = core.world;
  const omega = rpmToOmega(CONFIG.rpm);
  const seen = new Set<number>();
  for (const chunk of core.chunks) {
    if (!chunk.active || chunk.bodyHandle == null || seen.has(chunk.bodyHandle)) continue;
    seen.add(chunk.bodyHandle);
    const body = world.getRigidBody(chunk.bodyHandle);
    if (!body || body.isFixed()) continue;
    body.setAngularDamping(0);
    body.setLinearDamping(2.5);
    body.setAngvel({ x: 0, y: omega, z: 0 }, true);
  }
}

// ── UI wiring ────────────────────────────────────────────────

// Rebuild the scene (Reset, and the deferred shape/strength sliders so they apply without a
// separate Reset click — the wheel is a single cheap body).
async function rebuild() {
  rampActive = false;
  visualsRef?.dispose();
  coreRef?.dispose();
  coreRef = null;
  visualsRef = null;
  await initScene();
}

document.getElementById('btn-reset')?.addEventListener('click', () => { void rebuild(); });

document.getElementById('btn-debug')?.addEventListener('click', () => {
  showDebug = !showDebug;
  rapierDebug?.setEnabled(showDebug);
  const btn = document.getElementById('btn-debug')!;
  btn.textContent = showDebug ? '◈ Hide Debug' : '◇ Show Debug';
});

// "Spin up to burst": ramp the RPM from its current value to redline over a few seconds so the
// progressive overspeed failure plays out hands-free.
let rampActive = false;
const rpmSlider = () => document.getElementById('cfg-rpm') as HTMLInputElement | null;
const rpmDisplay = () => document.getElementById('cfg-rpm-value');
function setRpm(v: number) {
  CONFIG.rpm = v;
  const s = rpmSlider();
  if (s) s.value = String(Math.round(v));
  const d = rpmDisplay();
  if (d) d.textContent = String(Math.round(v));
}
document.getElementById('btn-spinup')?.addEventListener('click', () => {
  rampActive = !rampActive;
  const btn = document.getElementById('btn-spinup')!;
  btn.textContent = rampActive ? '⏸ Stop ramp' : '⏵ Spin up to burst';
  if (rampActive && CONFIG.rpm >= REDLINE_RPM) setRpm(80); // restart from a hum if already maxed
});

function bindSlider(id: string, obj: Record<string, any>, key: string, fmt?: (v: number) => string, onChange?: () => void) {
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
  if (onChange) slider.addEventListener('change', onChange);
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

// Shared Physics / Optimization controls (this demo keeps its own spin + debug toggles).
mountPhysicsControls({
  getCore: () => coreRef,
  onRebuild: () => void rebuild(),
  include: { centrifugal: false, debug: false, damage: false },
  defaults: { debrisCleanupMode: 'off' }, // keep spinning debris in zero-g
});

// Live: RPM is read every frame in applySpin(); dragging it cancels any auto-ramp.
bindSlider('cfg-rpm', CONFIG, 'rpm', (v) => String(Math.round(v)), undefined);
rpmSlider()?.addEventListener('input', () => { rampActive = false; const b = document.getElementById('btn-spinup'); if (b) b.textContent = '⏵ Spin up to burst'; });

// Shape rebuilds the scene on release ('change') so it applies live.
bindSlider('cfg-rim-blocks', CONFIG, 'rimBlocks', undefined, () => void rebuild());
bindSlider('cfg-spokes', CONFIG, 'spokes', undefined, () => void rebuild());

// Bond strength is logarithmic: the slider holds log10(compressionFatalLimit) so both a "holds at
// redline" and a "bursts early" regime are reachable. Rebuilds on release.
{
  const slider = document.getElementById('cfg-bond-strength') as HTMLInputElement | null;
  const display = document.getElementById('cfg-bond-strength-value');
  const fmt = (v: number) => v.toExponential(1);
  if (slider) {
    slider.value = String(Math.log10(CONFIG.bondStrength));
    if (display) display.textContent = fmt(CONFIG.bondStrength);
    slider.addEventListener('input', () => {
      CONFIG.bondStrength = Math.pow(10, parseFloat(slider.value));
      if (display) display.textContent = fmt(CONFIG.bondStrength);
    });
    slider.addEventListener('change', () => void rebuild());
  }
}

// ── Render loop ──────────────────────────────────────────────

const clock = new THREE.Clock();

function loop() {
  requestAnimationFrame(loop);
  profiler.render();
  recorder.render();
  stats.begin();
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();

  if (rampActive) {
    // ~70 RPM/s ramp: slow enough to watch each shedding stage as the threshold sweeps the curve.
    setRpm(Math.min(REDLINE_RPM, CONFIG.rpm + 70 * dt));
    if (CONFIG.rpm >= REDLINE_RPM) {
      rampActive = false;
      const b = document.getElementById('btn-spinup');
      if (b) b.textContent = '⏵ Spin up to burst';
    }
  }

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
  console.error('Failed to initialize flywheel burst demo:', err);
  const hint = document.querySelector('.viewport-hint');
  if (hint) hint.textContent = `Error: ${err.message}`;
});
