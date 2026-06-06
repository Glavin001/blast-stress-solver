/**
 * Destructible House Demo
 *
 * A realistic, low-poly, *walkable* wood-framed house: a static foundation anchors a floor
 * slab; framed exterior walls carry a front door + windows; an interior partition splits a
 * living room and kitchen; a ring of top-plate beams, interior posts and ceiling ties form
 * the frame; a pitched gabled roof sits on top; and the rooms are furnished (table, chairs,
 * kitchen counter, wall shelves). It is ONE heterogeneous body — per-material bond strengths
 * make it fail non-uniformly: a hit blows out wall panels, caves the roof on its own, and
 * the barely-attached furniture goes flying on the lightest touch.
 *
 * Press V to walk inside in first person (WASD + mouse). Click to throw a ball. The right
 * panel tunes material strength / projectile / physics and flips between the realistic
 * material colors and the standard orange/gray physics-state view.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import * as pinata from '@dgreenheck/three-pinata';
import { buildDestructibleCore, createFrameProfilerOverlay, createRecordingOverlay } from 'blast-stress-solver/rapier';
import { createDestructibleThreeBundle, RapierDebugRenderer, setPinataModule } from 'blast-stress-solver/three';
import {
  buildHouseScenario,
  buildHouseScenarioAsync,
  HOUSE_PALETTE,
  type HouseOptions,
  type HouseFractureMode,
} from 'blast-stress-solver/scenarios';
import { mountShooter } from './shooter-fps.js';
import { mountPhysicsControls, physicsCoreOverrides } from './physics-controls.js';

// Pre-register three-pinata so the synchronous Voronoi fracturer resolves in the browser
// ESM environment (used only when the "Fracture" option is on).
setPinataModule(pinata as any);

// ── Mutable demo config (driven by the control panel) ──────────
const CONFIG = {
  // House geometry — rebuilt live (apply via Reset).
  house: {
    width: 10,
    depth: 8,
    wallHeight: 2.6,
    roofRise: 1.6,
    furniture: true,
    // Voronoi-shatter selected parts (three-pinata) instead of boxes: none | walls |
    // wallsRoof | all. Anything but 'none' uses the async builder + WASM auto-bonder.
    fracture: 'none' as HouseFractureMode,
    // Fracture detail knobs (only matter when fracturing). Defaults are perf-tuned;
    // raise shards / lower cell size for finer, more detailed breakage at a perf cost.
    fragmentsPerPiece: 3,
    fractureCellSize: 1.05,
  },
  projectile: { radius: 0.35, mass: 800, speed: 28 },
  solver: {
    gravity: -9.81,
    // Master strength: effective material limits = materialScale * strength. Wood is
    // weaker than the concrete demos (1e10), so a thrown ball readily blows out panels.
    materialScale: 4e9,
    strength: 1.0,
    // Couples projectile impacts into the stress solver (same load path as a blast).
    contactForceScale: 14,
    iterations: 24,
    graphReduction: 0,
  },
  // Physics / Optimization / Features (debris collision, friction, restitution, small-body
  // damping, debris cleanup, custom damage, debug) come from the shared physics-controls.ts.
  features: { materialColors: true, debug: false },
};

// ── Three.js setup (daylight) ─────────────────────────────────
const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xaecbe8); // soft daytime sky
scene.fog = new THREE.FogExp2(0xaecbe8, 0.012);

const camera = new THREE.PerspectiveCamera(58, canvas.clientWidth / canvas.clientHeight, 0.1, 400);
// In front of (−Z) the front door, centered on X, so pressing V spawns you just outside
// the door facing in.
camera.position.set(0, 3.4, -13);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.4, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495; // don't dip under the ground
controls.update();

scene.add(new THREE.HemisphereLight(0xbfd6f2, 0x55503f, 0.85));
const sun = new THREE.DirectionalLight(0xfff2dd, 1.5);
sun.position.set(14, 22, -10);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -18;
sun.shadow.camera.right = 18;
sun.shadow.camera.top = 22;
sun.shadow.camera.bottom = -8;
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 70;
sun.shadow.bias = -0.0004;
scene.add(sun);

const groundMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(300, 300),
  new THREE.MeshStandardMaterial({ color: 0x6f8f5a, roughness: 0.95, metalness: 0 }),
);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.position.y = -0.01; // flush with the runtime ground (top at y=0); foundation buried below
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
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let shooter: ReturnType<typeof mountShooter> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let rebuilding = false;

// Standard session recorder (● Record / ⬇ Save) + live frame profiler, shared by all
// destruction demos so bug reports / perf captures are consistent across pages.
const profiler = createFrameProfilerOverlay();
const recorder = createRecordingOverlay({
  mount: document.getElementById('recorder-slot') ?? undefined,
  exportName: 'house-recording',
  getProfilerExport: () => profiler.exportData(),
});

/** Map the scenario's per-node fragment types + furniture accents → render colors. */
function buildNodeColors(scenario: any): THREE.Color[] {
  const house = scenario.parameters?.house ?? {};
  const types: string[] = house.fragmentTypes ?? [];
  const accents: (number | undefined)[] = house.nodeColors ?? [];
  return types.map(
    (t, i) => new THREE.Color(accents[i] ?? HOUSE_PALETTE[t] ?? 0xcccccc),
  );
}

async function initScene() {
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = 'Building the house…';

  const h = CONFIG.house;
  const opts: HouseOptions = {
    width: h.width,
    depth: h.depth,
    wallHeight: h.wallHeight,
    roofRise: h.roofRise,
    furniture: h.furniture,
  };
  const scenario =
    h.fracture === 'none'
      ? buildHouseScenario(opts)
      : await buildHouseScenarioAsync({
          ...opts,
          fracture: h.fracture,
          fragmentsPerPiece: h.fragmentsPerPiece,
          fractureCellSize: h.fractureCellSize,
          pinata: pinata as any,
        });

  const fractured = h.fracture !== 'none';
  const core = await buildDestructibleCore({
    scenario,
    gravity: CONFIG.solver.gravity,
    materialScale: CONFIG.solver.materialScale * CONFIG.solver.strength,
    solverSettings: {
      maxSolverIterationsPerFrame: CONFIG.solver.iterations,
      graphReductionLevel: CONFIG.solver.graphReduction,
    },
    contactForceScale: CONFIG.solver.contactForceScale,
    // Shared Physics/Optimization/Features controls (debris collision, friction, restitution,
    // small-body damping, debris cleanup/TTL, custom damage) — see physics-controls.ts.
    ...physicsCoreOverrides(),
    // Fractured mode spawns many small shards: force debris↔debris collisions off and cap the
    // per-frame split churn (this overrides the shared "Debris Collision" control while on).
    ...(fractured
      ? { debrisCollisionMode: 'noDebrisPairs' as const, fracturePolicy: { maxNewBodiesPerFrame: 30, maxColliderMigrationsPerFrame: 60 } }
      : {}),
  });

  const group = new THREE.Group();
  scene.add(group);
  const visuals = createDestructibleThreeBundle({
    core,
    scenario,
    root: group,
    useBatchedMesh: true,
    batchedMeshOptions: { enableBVH: false },
    includeDebugLines: true,
    nodeColors: buildNodeColors(scenario),
    materialColors: CONFIG.features.materialColors,
  });

  rapierDebug?.dispose();
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: CONFIG.features.debug });

  coreRef = core;
  visualsRef = visuals;
  recorder.attach(core, { scenario, meta: { demo: 'house', config: CONFIG } });
  profiler.attach(core);
  initialBonds = core.getActiveBondsCount();
  baselineY = avgDynamicY(core);
  if (hint) hint.textContent = 'Click to throw a ball · Press V to walk inside (WASD + mouse)';
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

// House geometry (Reset to apply).
bindSlider('cfg-width', CONFIG.house, 'width', (v) => v.toFixed(0) + ' m');
bindSlider('cfg-depth', CONFIG.house, 'depth', (v) => v.toFixed(0) + ' m');
bindSlider('cfg-wall-height', CONFIG.house, 'wallHeight', (v) => v.toFixed(1) + ' m');
bindSlider('cfg-roof-rise', CONFIG.house, 'roofRise', (v) => v.toFixed(1) + ' m');
const optFurniture = document.getElementById('cfg-furniture') as HTMLInputElement | null;
if (optFurniture) optFurniture.checked = CONFIG.house.furniture;
optFurniture?.addEventListener('change', () => {
  CONFIG.house.furniture = !!optFurniture.checked;
  void rebuild();
});

// Voronoi fracture selector (opt-in; uses the async builder + WASM auto-bonder).
const selFracture = document.getElementById('cfg-fracture') as HTMLSelectElement | null;
if (selFracture) selFracture.value = CONFIG.house.fracture;
selFracture?.addEventListener('change', () => {
  CONFIG.house.fracture = selFracture.value as HouseFractureMode;
  void rebuild();
});
// Fracture detail (Reset to apply): higher shards + smaller cell = finer breakage.
bindSlider('cfg-shards', CONFIG.house, 'fragmentsPerPiece', (v) => v.toFixed(0) + ' / piece');
bindSlider('cfg-fracture-cell', CONFIG.house, 'fractureCellSize', (v) => v.toFixed(2) + ' m');

// Projectile (live at next throw).
bindSlider('cfg-proj-radius', CONFIG.projectile, 'radius', (v) => v.toFixed(2) + ' m');
bindSlider('cfg-proj-mass', CONFIG.projectile, 'mass', (v) => v.toLocaleString() + ' kg');
bindSlider('cfg-proj-speed', CONFIG.projectile, 'speed', (v) => v.toFixed(0) + ' m/s');

// Structure / solver (Reset to apply, except gravity).
bindSlider('cfg-strength', CONFIG.solver, 'strength', (v) => v.toFixed(2) + '×');
bindSlider('cfg-contact-force', CONFIG.solver, 'contactForceScale', (v) => v.toFixed(0));
bindSlider('cfg-iterations', CONFIG.solver, 'iterations', (v) => v.toFixed(0));
bindSlider('cfg-graph-reduction', CONFIG.solver, 'graphReduction', (v) => v.toFixed(0));
bindSlider('cfg-gravity', CONFIG.solver, 'gravity', (v) => v.toFixed(1) + ' m/s²', (v) => coreRef?.setGravity(v));

// Material colors (house-specific). Friction/restitution, debris collision, small-body
// damping, debris cleanup/TTL, custom damage and the debug wireframe are all provided by the
// shared sections injected by mountPhysicsControls below.
const optColors = document.getElementById('opt-material-colors') as HTMLInputElement | null;
if (optColors) optColors.checked = CONFIG.features.materialColors;
optColors?.addEventListener('change', () => {
  CONFIG.features.materialColors = !!optColors.checked;
  visualsRef?.setMaterialColors(CONFIG.features.materialColors);
});

mountPhysicsControls({
  getCore: () => coreRef,
  onDebug: (on) => { CONFIG.features.debug = on; rapierDebug?.setEnabled(on); },
  onRebuild: () => { void rebuild(); },
  include: { centrifugal: false }, // a house doesn't spin
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

  shooter?.update();

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

shooter = mountShooter({
  canvas,
  camera,
  controls,
  scene,
  getCore: () => coreRef,
  getBallParams: () => CONFIG.projectile,
  floorY: 0, // outside ground level (the player steps up onto the floor slab)
  eyeHeight: 1.6,
  // Real-world avatar: ≈1.6 m tall, ≈0.56 m wide — fits comfortably through the doors.
  playerHalfHeight: 0.5,
  playerRadius: 0.28,
});
initScene()
  .then(() => loop())
  .catch((err) => {
    console.error('Failed to initialize house demo:', err);
    const hint = document.querySelector('.viewport-hint');
    if (hint) hint.textContent = `Error: ${err.message}`;
  });
