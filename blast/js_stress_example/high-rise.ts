/**
 * High-Rise Apartment Demolition Demo
 *
 * A mid-rise reinforced-concrete apartment building (flat-slab skeleton + frangible
 * drywall infill) loaded from the SAME shared scene-pack JSON the Rust/Bevy demo uses.
 *
 * Destruction is driven by the stress solver: a wrecking ball is spawned just in front
 * of the clicked surface (a LOCAL impact) and the solver fractures bonds where stress
 * exceeds the material limits. The top-right panel exposes the projectile, gravity,
 * material strength, physics, and optimization knobs so the behavior can be tuned live.
 *
 * The optional per-chunk "Custom damage system" is OFF by default (it can over-soften
 * the structure); toggle it on to compare.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { buildDestructibleCore, loadScenePackFromUrl , createFrameProfilerOverlay, createRecordingOverlay } from 'blast-stress-solver/rapier';
import { createDestructibleThreeBundle, RapierDebugRenderer } from 'blast-stress-solver/three';
import { pipelineCoreOverrides, mountPipelineControls } from './pipeline-controls.js';

const SCENE_URL = '/vendor/blast-stress-solver/high-rise.json';

// ── Mutable demo config (driven by the control panel) ──────────
const CONFIG = {
  projectile: { radius: 0.6, mass: 2500, speed: 8 },
  solver: {
    gravity: -9.81,
    // Multiplier on the pack's concrete stress limits: >1 = stronger / harder to break,
    // <1 = more fragile. This is the "material strength" knob (clearer than materialScale).
    strength: 1.0,
    // Couples projectile impacts into the *stress* solver. Kept low by default: the
    // high-rise's stress response is near-bimodal, and a high value lets one hit seed a
    // slow global collapse. Raise it for a more impact-reactive (and collapse-prone) frame.
    contactForceScale: 8,
    // Stress-solver internals (Blast knobs). These secretly set the fracture THRESHOLD:
    // - iterations: CG steps/frame. Low = under-converged stress (under-reports → robust
    //   but physically inaccurate); high = converged true stress (accurate → readily
    //   cascades) and much costlier. There is a sharp cliff, not a gradient.
    // - graphReduction: coarsens the support graph; higher = lower peak stress (sparser /
    //   chunkier cracks, eventually none) and cheaper.
    iterations: 24,
    graphReduction: 0,
  },
  physics: { debrisCollisionMode: 'all', friction: 0.25, restitution: 0 },
  optimization: {
    // Damp small debris only after it lands; 'always' also damps it mid-air (caps the fall at
    // ~g/damping ≈ 5 m/s → "floaty" collapse). See rapier.smallBodyDamping.fall.test.ts.
    smallBodyDampingMode: 'afterGroundCollision',
    debrisCleanupMode: 'always',
    debrisTtlMs: 10000,
    maxCollidersForDebris: 3,
  },
  features: { damage: false, debug: false },
};

// Captured from the scene pack at load.
let baseLimits: Record<string, number> = {};
let packDamage: Record<string, unknown> = {};
let projectileTtlMs = 3000;
const STANDOFF = 6; // metres in front of the clicked surface to spawn the ball
const buildingBox = new THREE.Box3();

function scaledLimits(): Record<string, number> {
  const s = CONFIG.solver.strength;
  const out: Record<string, number> = {};
  for (const k of Object.keys(baseLimits)) out[k] = baseLimits[k] * s;
  return out;
}

// ── Three.js setup ────────────────────────────────────────────
const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d13);
scene.fog = new THREE.FogExp2(0x0a0d13, 0.006);

const camera = new THREE.PerspectiveCamera(55, canvas.clientWidth / canvas.clientHeight, 0.1, 600);
camera.position.set(28, 18, 44);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 12, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const dirLight = new THREE.DirectionalLight(0xffeedd, 1.0);
dirLight.position.set(20, 36, 24);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -40;
dirLight.shadow.camera.right = 40;
dirLight.shadow.camera.top = 50;
dirLight.shadow.camera.bottom = -5;
scene.add(dirLight);

const groundMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(300, 300),
  new THREE.MeshStandardMaterial({ color: 0x1a1e2f, roughness: 0.9, metalness: 0.05 }),
);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.position.y = -0.02;
groundMesh.receiveShadow = true;
scene.add(groundMesh);

const stats = new Stats();
stats.dom.style.position = 'absolute';
stats.dom.style.top = '0';
stats.dom.style.left = '0';
(document.querySelector('.viewport') as HTMLElement)?.appendChild(stats.dom);

// ── Stats / perf ──────────────────────────────────────────────
let _physicsMs = 0;
let _renderMs = 0;
const EMA = 0.12;
let initialBonds = 0;
let baselineY = 0;

function setText(id: string, value: string) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
function avgDynamicY(core: any): number {
  const dyn = core.chunks.filter((c: any) => c.active && !c.isSupport);
  if (!dyn.length) return 0;
  return dyn.reduce((s: number, c: any) => s + (c.worldPosition ?? c.baseLocalOffset).y, 0) / dyn.length;
}
function updateStatus(core: any) {
  setText('stat-bodies', String(core.getRigidBodyCount()));
  setText('stat-bonds', `${core.getActiveBondsCount()} / ${initialBonds}`);
  setText('stat-projectiles', String(core.projectiles.length));
  const active = core.chunks.filter((c: any) => c.active).length;
  const detached = core.chunks.filter((c: any) => c.detached).length;
  setText('stat-chunks', `${active} / ${detached} detached`);
  setText('stat-fragments', String(core.chunks.length));
  setText('stat-settle', (baselineY - avgDynamicY(core)).toFixed(2) + ' m');
}
function updatePerf() {
  setText('stat-physics-ms', _physicsMs.toFixed(1) + ' ms');
  setText('stat-render-ms', _renderMs.toFixed(1) + ' ms');
  setText('stat-draw-calls', String(renderer.info.render.calls));
  setText('stat-triangles', renderer.info.render.triangles.toLocaleString());
}

// ── Scene lifecycle ───────────────────────────────────────────
let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
// Reusable, self-mounting live frame-profiler overlay (per-phase cost + A/B).
const profiler = createFrameProfilerOverlay();

// Reusable session recorder — ● Record captures every dynamic body's per-frame
// position/orientation + linear/angular velocity, every input (projectiles,
// forces, gravity) and every fracture/topology change into a single gzipped
// bug-report bundle (⬇ Save). Zero allocation on the hot path while recording.
const recorder = createRecordingOverlay({
  exportName: 'high-rise-recording',
  getProfilerExport: () => profiler.exportData(),
});
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let rebuilding = false;

async function initScene() {
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = 'Loading high-rise...';

  const pack = await loadScenePackFromUrl(SCENE_URL);
  const { scenario, defaults } = pack;
  baseLimits = (defaults.solverSettings as Record<string, number>) ?? {};
  packDamage = (defaults.damage as Record<string, unknown>) ?? {};
  projectileTtlMs = Math.min((defaults.projectile as any)?.ttlMs ?? 3000, 3000);

  buildingBox.makeEmpty();
  const boundsPoint = new THREE.Vector3();
  for (const n of scenario.nodes) {
    buildingBox.expandByPoint(boundsPoint.set(n.centroid.x, n.centroid.y, n.centroid.z));
  }
  buildingBox.expandByScalar(1.0);

  controls.target.set(defaults.camera.target.x, defaults.camera.target.y, defaults.camera.target.z);
  controls.update();

  const core = await buildDestructibleCore({
    scenario,
    gravity: CONFIG.solver.gravity,
    materialScale: defaults.materialScale,
    solverSettings: {
      ...scaledLimits(),
      maxSolverIterationsPerFrame: CONFIG.solver.iterations,
      graphReductionLevel: CONFIG.solver.graphReduction,
    },
    friction: CONFIG.physics.friction,
    restitution: CONFIG.physics.restitution,
    contactForceScale: CONFIG.solver.contactForceScale,
    debrisCollisionMode: CONFIG.physics.debrisCollisionMode as any,
    damage: { ...packDamage, enabled: CONFIG.features.damage } as any,
    debrisCleanup: {
      mode: CONFIG.optimization.debrisCleanupMode as any,
      debrisTtlMs: CONFIG.optimization.debrisTtlMs,
      maxCollidersForDebris: CONFIG.optimization.maxCollidersForDebris,
    },
    smallBodyDamping: {
      mode: CONFIG.optimization.smallBodyDampingMode as any,
      colliderCountThreshold: 3,
      minLinearDamping: 2,
      minAngularDamping: 2,
    },
    ...pipelineCoreOverrides(),
  });

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
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: CONFIG.features.debug });

  coreRef = core;
  recorder.attach(core, { scenario, meta: { demo: 'high-rise', config: CONFIG } });
  profiler.attach(core);
  visualsRef = visuals;
  initialBonds = core.getActiveBondsCount();
  baselineY = avgDynamicY(core);
  if (hint) hint.textContent = 'Click the building to throw the wrecking ball';
}

async function rebuild() {
  if (rebuilding) return;
  rebuilding = true;
  visualsRef?.dispose();
  coreRef?.dispose();
  coreRef = null;
  visualsRef = null;
  try {
    await initScene();
  } finally {
    rebuilding = false;
  }
}

// ── Projectile ────────────────────────────────────────────────
function shootProjectile(ndcX: number, ndcY: number) {
  const core = coreRef;
  if (!core) return;
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  const dir = raycaster.ray.direction.clone().normalize();
  // Spawn the ball just in FRONT of where the click ray meets the building, so it
  // delivers a local hit instead of plowing ~45 m diagonally through the structure.
  const entry = new THREE.Vector3();
  if (!raycaster.ray.intersectBox(buildingBox, entry)) return; // clicked empty space
  const spawn = entry.addScaledVector(dir, -STANDOFF);
  const speed = CONFIG.projectile.speed;
  core.enqueueProjectile({
    position: { x: spawn.x, y: spawn.y, z: spawn.z },
    velocity: { x: dir.x * speed, y: dir.y * speed, z: dir.z * speed },
    radius: CONFIG.projectile.radius,
    mass: CONFIG.projectile.mass,
    ttl: projectileTtlMs,
  });
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  shootProjectile(ndcX, ndcY);
});

// ── Control panel wiring ──────────────────────────────────────
function bindSlider(
  id: string,
  obj: Record<string, any>,
  key: string,
  fmt?: (v: number) => string,
  onInput?: (v: number) => void,
) {
  const slider = document.getElementById(id) as HTMLInputElement | null;
  const display = document.getElementById(id + '-value');
  if (!slider) return;
  slider.value = String(obj[key]);
  if (display) display.textContent = fmt ? fmt(obj[key]) : String(obj[key]);
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    obj[key] = v;
    if (display) display.textContent = fmt ? fmt(v) : String(v);
    onInput?.(v);
  });
}
function bindSelect(id: string, obj: Record<string, any>, key: string, onChange?: (v: string) => void) {
  const select = document.getElementById(id) as HTMLSelectElement | null;
  if (!select) return;
  select.value = String(obj[key]);
  select.addEventListener('change', () => { obj[key] = select.value; onChange?.(select.value); });
}

// Projectile (read live at shoot time)
bindSlider('cfg-proj-radius', CONFIG.projectile, 'radius', (v) => v.toFixed(2) + ' m');
bindSlider('cfg-proj-mass', CONFIG.projectile, 'mass', (v) => v.toLocaleString() + ' kg');
bindSlider('cfg-proj-speed', CONFIG.projectile, 'speed', (v) => v.toFixed(0) + ' m/s');

// Structure / solver
bindSlider('cfg-strength', CONFIG.solver, 'strength', (v) => v.toFixed(2) + '×');
bindSlider('cfg-contact-force', CONFIG.solver, 'contactForceScale', (v) => v.toFixed(0));
bindSlider('cfg-iterations', CONFIG.solver, 'iterations', (v) => v.toFixed(0));
bindSlider('cfg-graph-reduction', CONFIG.solver, 'graphReduction', (v) => v.toFixed(0));
bindSlider('cfg-gravity', CONFIG.solver, 'gravity', (v) => v.toFixed(1) + ' m/s²', (v) => coreRef?.setGravity(v));

// Physics
bindSelect('cfg-debris-collision', CONFIG.physics, 'debrisCollisionMode', (v) => coreRef?.setDebrisCollisionMode(v as any));
bindSlider('cfg-friction', CONFIG.physics, 'friction', (v) => v.toFixed(2));
bindSlider('cfg-restitution', CONFIG.physics, 'restitution', (v) => v.toFixed(2));

// Optimization (live)
bindSelect('cfg-damping-mode', CONFIG.optimization, 'smallBodyDampingMode', (v) => coreRef?.setSmallBodyDamping?.({ mode: v as any }));
bindSelect('cfg-cleanup-mode', CONFIG.optimization, 'debrisCleanupMode', (v) =>
  coreRef?.setDebrisCleanup?.({ mode: v as any, debrisTtlMs: CONFIG.optimization.debrisTtlMs }));
bindSlider('cfg-debris-ttl', CONFIG.optimization, 'debrisTtlMs', (v) => (v / 1000).toFixed(1) + 's', (v) =>
  coreRef?.setDebrisCleanup?.({ mode: CONFIG.optimization.debrisCleanupMode as any, debrisTtlMs: v }));

// Features
const optDamage = document.getElementById('opt-damage') as HTMLInputElement | null;
if (optDamage) optDamage.checked = CONFIG.features.damage;
optDamage?.addEventListener('change', () => { CONFIG.features.damage = !!optDamage.checked; void rebuild(); });

const optDebug = document.getElementById('opt-debug') as HTMLInputElement | null;
if (optDebug) optDebug.checked = CONFIG.features.debug;
optDebug?.addEventListener('change', () => {
  CONFIG.features.debug = !!optDebug.checked;
  rapierDebug?.setEnabled(CONFIG.features.debug);
});

document.getElementById('btn-reset')?.addEventListener('click', () => { void rebuild(); });

// ── Render loop ───────────────────────────────────────────────
const clock = new THREE.Clock();
function loop() {
  requestAnimationFrame(loop);
  profiler.render();
  recorder.render();
  stats.begin();
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();

  if (coreRef && visualsRef) {
    const t0 = performance.now();
    coreRef.step(dt);
    _physicsMs += ((performance.now() - t0) - _physicsMs) * EMA;
    visualsRef.update({ debug: CONFIG.features.debug, updateBVH: false, updateProjectiles: true });
    rapierDebug?.update();
    updateStatus(coreRef);
  }

  const t1 = performance.now();
  renderer.render(scene, camera);
  _renderMs += ((performance.now() - t1) - _renderMs) * EMA;
  updatePerf();
  stats.end();
}

function onResize() {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

mountPipelineControls();
initScene()
  .then(() => loop())
  .catch((err) => {
    console.error('Failed to initialize high-rise demo:', err);
    const hint = document.querySelector('.viewport-hint');
    if (hint) hint.textContent = `Error: ${err.message}`;
  });
