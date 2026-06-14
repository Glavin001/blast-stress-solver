/**
 * Brick Castle Siege — large-scale anchored destruction showcase.
 *
 * A procedural masonry castle (running-bond curtain walls, four corner towers,
 * an arched gatehouse, a central keep, crenellated battlements) built as ~2,700
 * Voronoi-fractured bricks. ALL bonds are derived from real surface contact by
 * the WASM auto-bonder, then scaled into a three-tier strength hierarchy:
 *   intra-brick  ≫  mortar (brick↔brick)  >  inter-structure (wall↔tower seam).
 *
 * The castle is anchored to a static foundation, so — unlike a free body — the
 * stress solver genuinely propagates gravity + siege impacts into a progressive,
 * physically-plausible collapse. Lazy intact colliders + the island solver keep
 * the large scene interactive: nothing is "live" in the broadphase until a siege
 * tool actually reaches it.
 *
 * Three siege tools: click to fire a cannonball, swing a wrecking ball, or call
 * down a boulder storm.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import * as pinata from '@dgreenheck/three-pinata';
import { buildDestructibleCore, createFrameProfilerOverlay, createRecordingOverlay } from 'blast-stress-solver/rapier';
import { createDestructibleThreeBundle, RapierDebugRenderer, setPinataModule } from 'blast-stress-solver/three';
import { buildBrickCastleScenario, type BrickCastleOptions, type CastleStructureKind } from 'blast-stress-solver/scenarios';
import { mountShooter } from './shooter-fps.js';

// Pre-register three-pinata so the synchronous Voronoi fracturer resolves under
// browser ESM (bare-specifier dynamic imports don't resolve there).
setPinataModule(pinata as any);

// ── Scale presets (Reset to apply) ────────────────────────────
type SizeKey = 'Small' | 'Medium' | 'Hero' | 'Max';
const SIZE_PRESETS: Record<SizeKey, Partial<BrickCastleOptions>> = {
  Small: { wallLengthBricks: 11, wallCourses: 8, towerCourses: 11, keepSideBricks: 6, keepCourses: 15 },
  Medium: { wallLengthBricks: 14, wallCourses: 9, towerCourses: 14, keepSideBricks: 7, keepCourses: 18 },
  Hero: { wallLengthBricks: 18, wallCourses: 11, towerCourses: 18, keepSideBricks: 8, keepCourses: 24 },
  Max: { wallLengthBricks: 24, wallCourses: 13, towerCourses: 22, keepSideBricks: 9, keepCourses: 28 },
};
const SIZE_ORDER: SizeKey[] = ['Small', 'Medium', 'Hero', 'Max'];

// ── Mutable demo config (driven by the control panel) ──────────
const CONFIG = {
  size: 'Hero' as SizeKey,
  chunksPerBrick: 2,
  battlements: true,
  // Strength hierarchy (deferred — applied on Rebuild).
  materialExp: 9.1, // materialScale = 10^materialExp ("stone strength")
  intra: 9.0,
  mortar: 1.6,
  inter: 0.5,
  // Scalability (live).
  lazy: true,
  island: true,
  sleep: true,
  gravity: -9.81,
  // Cannonball (read live at shoot time).
  projectile: { radius: 0.7, mass: 3500, speed: 92 },
};

// ── Three.js setup ────────────────────────────────────────────
const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9fb6cf); // hazy siege sky
scene.fog = new THREE.FogExp2(0x9fb6cf, 0.004);

const camera = new THREE.PerspectiveCamera(55, canvas.clientWidth / canvas.clientHeight, 0.1, 1200);
// Look at the south (gatehouse) face from a 3/4 angle — this is also the side the
// siege tools attack from, so projectiles arc in toward the wall you're watching.
camera.position.set(40, 26, -52);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 7, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dirLight = new THREE.DirectionalLight(0xfff1dd, 1.05);
dirLight.position.set(36, 54, 30);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -60;
dirLight.shadow.camera.right = 60;
dirLight.shadow.camera.top = 70;
dirLight.shadow.camera.bottom = -10;
scene.add(dirLight);

const groundMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(600, 600),
  new THREE.MeshStandardMaterial({ color: 0x5a6b4a, roughness: 0.95, metalness: 0.02 }),
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

// ── Perf / HUD ────────────────────────────────────────────────
let _physicsMs = 0;
let _renderMs = 0;
const EMA = 0.12;
let initialBonds = 0;

function setText(id: string, value: string) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
function updateStatus(core: any) {
  setText('stat-bodies', core.getRigidBodyCount().toLocaleString());
  setText('stat-bonds', `${core.getActiveBondsCount().toLocaleString()} / ${initialBonds.toLocaleString()}`);
  setText('stat-broken', Math.max(0, initialBonds - core.getActiveBondsCount()).toLocaleString());
  const active = core.chunks.filter((c: any) => c.active).length;
  setText('stat-chunks', `${active.toLocaleString()} / ${core.chunks.length.toLocaleString()}`);
  const isl = core.getIslandSolverStats?.() ?? { islandCount: 0, islandsSkipped: 0 };
  setText('stat-islands', `${isl.islandsSkipped} / ${isl.islandCount}`);
  const lazy = core.getLazyColliderStats?.() ?? { activeLeafFragments: 0 };
  setText('stat-colliders', lazy.activeLeafFragments.toLocaleString());
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
let rapierDebug: RapierDebugRenderer | null = null;
let shooter: ReturnType<typeof mountShooter> | null = null;
let rebuilding = false;
let showDebug = false;
let colorMode: 'stone' | 'structure' = 'stone';
// Castle extents (set at build) for aiming the siege tools.
let castleHalf = 30;
let castleTop = 16;

const profiler = createFrameProfilerOverlay();
const recorder = createRecordingOverlay({
  mount: document.getElementById('recorder-slot') ?? undefined,
  exportName: 'brick-castle-recording',
  getProfilerExport: () => profiler.exportData(),
});

// Per-structure palette for the "structure" colour view (shows the hierarchy).
const STRUCTURE_COLORS: Record<CastleStructureKind, number> = {
  wall: 0x8fb0d8,
  tower: 0xe0a85a,
  gatehouse: 0xd86a6a,
  keep: 0x9b6cd8,
  battlement: 0x6ad8a8,
  foundation: 0x55524a,
};

function buildNodeColors(scenario: any): THREE.Color[] {
  const p = scenario.parameters ?? {};
  const base: number[] = p.baseColorByNode ?? [];
  const kinds: CastleStructureKind[] = p.kindByNode ?? [];
  return scenario.nodes.map((_: unknown, i: number) =>
    colorMode === 'structure'
      ? new THREE.Color(STRUCTURE_COLORS[kinds[i]] ?? 0x999999)
      : new THREE.Color(base[i] ?? 0x9a8f7d),
  );
}

async function initScene() {
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = `Building ${CONFIG.size} castle… (auto-bonding ~2–3k bricks)`;

  const scenario = await buildBrickCastleScenario({
    ...SIZE_PRESETS[CONFIG.size],
    chunksPerBrick: CONFIG.chunksPerBrick,
    battlements: CONFIG.battlements,
    bondMode: 'auto',
    multipliers: { intraBrick: CONFIG.intra, mortar: CONFIG.mortar, interStructure: CONFIG.inter },
    pinata: pinata as any,
  });

  const dims = (scenario.parameters as any)?.dims ?? { x: 60, y: 16, z: 60 };
  castleHalf = dims.x * 0.5;
  castleTop = dims.y;

  console.log(
    `[brick-castle] ${scenario.nodes.length} chunks, ${scenario.bonds.length} bonds, ` +
    `${(scenario.parameters as any)?.brickCount} bricks, tiers(anchor/intra/mortar/inter)=` +
    `${Array.from((scenario.parameters as any)?.tierCounts ?? [])}`,
  );

  const core = await buildDestructibleCore({
    scenario,
    gravity: CONFIG.gravity,
    materialScale: Math.pow(10, CONFIG.materialExp),
    // Avoid the overlapping-chunk explosion when many fragments detach at once
    // (debris bounces off the castle + ground, not off other debris).
    debrisCollisionMode: 'noDebrisPairs',
    friction: 0.7,
    restitution: 0.0,
    contactForceScale: 20,
    // Scalability: only materialize colliders for the struck region.
    lazyIntactColliders: CONFIG.lazy,
    // Let settled bodies sleep so the idle castle costs almost nothing.
    sleepMode: CONFIG.sleep ? 'always' : 'off',
    sleepLinearThreshold: 0.18,
    sleepAngularThreshold: 0.22,
    // Keep a heavy collapse smooth by capping per-frame work.
    fracturePolicy: { maxNewBodiesPerFrame: 130, maxColliderMigrationsPerFrame: 220 },
  });

  // Island-aware solving: skip islands that have come to rest.
  core.setIslandSolver({ enabled: CONFIG.island, skipSettled: true });

  const group = new THREE.Group();
  scene.add(group);
  const visuals = createDestructibleThreeBundle({
    core,
    scenario,
    root: group,
    useBatchedMesh: true,
    batchedMeshOptions: { enableBVH: false, bvhMargin: 5 },
    includeDebugLines: true,
    nodeColors: buildNodeColors(scenario),
  });

  rapierDebug?.dispose();
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: showDebug });

  coreRef = core;
  visualsRef = visuals;
  initialBonds = core.getActiveBondsCount();
  recorder.attach(core, { scenario, meta: { demo: 'brick-castle', config: { ...CONFIG } } });
  profiler.attach(core);

  if (hint) hint.textContent = 'Click the castle to fire a cannonball';
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

// ── Siege tools ───────────────────────────────────────────────

/** A heavy iron ball swung in a low arc into the nearest (south) wall. */
function wreckingBall() {
  const core = coreRef;
  if (!core) return;
  const y = castleTop * 0.42;
  core.enqueueProjectile({
    position: { x: (Math.random() - 0.5) * castleHalf * 0.5, y, z: -(castleHalf + 12) },
    velocity: { x: 0, y: 2, z: 34 },
    radius: 1.5,
    mass: 9000,
    ttl: 20,
  });
}

/** A rain of boulders over the whole footprint — mass destruction + load test. */
function boulderStorm(count = 30) {
  const core = coreRef;
  if (!core) return;
  for (let i = 0; i < count; i++) {
    core.enqueueProjectile({
      position: {
        x: (Math.random() - 0.5) * castleHalf * 2.1,
        y: castleTop + 24 + Math.random() * 16,
        z: (Math.random() - 0.5) * castleHalf * 2.1,
      },
      velocity: { x: (Math.random() - 0.5) * 6, y: -38, z: (Math.random() - 0.5) * 6 },
      radius: 0.55 + Math.random() * 0.35,
      mass: 1600,
      ttl: 18,
    });
  }
}

/** Fire a volley of cannonballs at the south wall (used by the headless soak). */
function fireVolley(count = 5) {
  const core = coreRef;
  if (!core) return;
  for (let i = 0; i < count; i++) {
    // Aim low and at the central south wall / gatehouse (where the load above is
    // greatest), so a breach drops the wall over it rather than pinging a tower.
    core.enqueueProjectile({
      position: { x: (Math.random() - 0.5) * castleHalf * 0.6, y: castleTop * (0.2 + Math.random() * 0.25), z: -(castleHalf + 18) },
      velocity: { x: 0, y: 0, z: CONFIG.projectile.speed },
      radius: CONFIG.projectile.radius,
      mass: CONFIG.projectile.mass,
      ttl: 14,
    });
  }
}

// ── Headless metrics probe (for scripts/soak-castle.mjs) ──────
function maxBodyMetrics(core: any): { maxSpeed: number; fastBodies: number; bodies: number } {
  let maxSpeed = 0;
  let fastBodies = 0;
  let bodies = 0;
  const world = core.world;
  world.forEachRigidBody((b: any) => {
    if (b.isFixed?.() || b.bodyType?.() === 1) return; // skip fixed bodies
    if (b.userData?.projectile) return; // skip the siege projectiles themselves
    const v = b.linvel();
    const s = Math.hypot(v.x, v.y, v.z);
    bodies++;
    if (s > maxSpeed) maxSpeed = s;
    if (s > 60) fastBodies++;
  });
  return { maxSpeed, fastBodies, bodies };
}

(window as any).__castleDemo = {
  metrics() {
    const core = coreRef;
    if (!core) return { ready: false };
    const m = maxBodyMetrics(core);
    return {
      ready: true,
      ...m,
      chunks: core.chunks.length,
      activeChunks: core.chunks.filter((c: any) => c.active).length,
      initialBonds,
      activeBonds: core.getActiveBondsCount(),
      brokenBonds: Math.max(0, initialBonds - core.getActiveBondsCount()),
      activeColliders: core.getLazyColliderStats?.().activeLeafFragments ?? 0,
      islandsSkipped: core.getIslandSolverStats?.().islandsSkipped ?? 0,
    };
  },
  wreckingBall,
  boulderStorm,
  fireVolley,
  isReady: () => !!coreRef && !rebuilding,
};

// ── Control panel wiring ──────────────────────────────────────
function bindSlider(id: string, obj: Record<string, any>, key: string, fmt?: (v: number) => string, onInput?: (v: number) => void) {
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
function bindCheckbox(id: string, get: () => boolean, set: (v: boolean) => void) {
  const el = document.getElementById(id) as HTMLInputElement | null;
  if (!el) return;
  el.checked = get();
  el.addEventListener('change', () => set(!!el.checked));
}

// Siege tools (live).
document.getElementById('btn-wrecking')?.addEventListener('click', () => wreckingBall());
document.getElementById('btn-storm')?.addEventListener('click', () => boulderStorm());
bindSlider('cfg-proj-radius', CONFIG.projectile, 'radius', (v) => v.toFixed(2) + ' m');
bindSlider('cfg-proj-mass', CONFIG.projectile, 'mass', (v) => v.toLocaleString() + ' kg');
bindSlider('cfg-proj-speed', CONFIG.projectile, 'speed', (v) => v.toFixed(0) + ' m/s');

// Strength hierarchy (deferred — applied on Rebuild).
{
  const slider = document.getElementById('cfg-material') as HTMLInputElement | null;
  const display = document.getElementById('cfg-material-value');
  if (slider) {
    slider.value = String(CONFIG.materialExp);
    if (display) display.textContent = `1e${CONFIG.materialExp.toFixed(0)}`;
    slider.addEventListener('input', () => {
      CONFIG.materialExp = parseFloat(slider.value);
      if (display) display.textContent = `1e${CONFIG.materialExp.toFixed(1)}`;
    });
  }
}
bindSlider('cfg-intra', CONFIG, 'intra', (v) => v.toFixed(1));
bindSlider('cfg-mortar', CONFIG, 'mortar', (v) => v.toFixed(1));
bindSlider('cfg-inter', CONFIG, 'inter', (v) => v.toFixed(1));

// Castle (deferred).
{
  const slider = document.getElementById('cfg-scale') as HTMLInputElement | null;
  const display = document.getElementById('cfg-scale-value');
  if (slider) {
    slider.value = String(SIZE_ORDER.indexOf(CONFIG.size));
    if (display) display.textContent = CONFIG.size;
    slider.addEventListener('input', () => {
      CONFIG.size = SIZE_ORDER[parseInt(slider.value, 10)] ?? 'Hero';
      if (display) display.textContent = CONFIG.size;
    });
  }
}
bindSlider('cfg-chunks', CONFIG, 'chunksPerBrick', (v) => String(v));
bindCheckbox('cfg-battlements', () => CONFIG.battlements, (v) => { CONFIG.battlements = v; });

// Scalability (live).
bindCheckbox('cfg-lazy', () => CONFIG.lazy, (v) => { CONFIG.lazy = v; coreRef?.setLazyIntactColliders(v); });
bindCheckbox('cfg-island', () => CONFIG.island, (v) => { CONFIG.island = v; coreRef?.setIslandSolver({ enabled: v, skipSettled: true }); });
bindCheckbox('cfg-sleep', () => CONFIG.sleep, (v) => { CONFIG.sleep = v; coreRef?.setSleepMode(v ? 'always' : 'off'); });
bindSlider('cfg-gravity', CONFIG, 'gravity', (v) => v.toFixed(1) + ' m/s²', (v) => coreRef?.setGravity(v));

// Actions.
document.getElementById('btn-reset')?.addEventListener('click', () => { void rebuild(); });
document.getElementById('btn-debug')?.addEventListener('click', () => {
  showDebug = !showDebug;
  rapierDebug?.setEnabled(showDebug);
  setText('btn-debug', showDebug ? '◈ Hide Debug' : '◇ Show Debug');
  const b = document.getElementById('btn-debug'); if (b) b.textContent = showDebug ? '◈ Hide Debug' : '◇ Show Debug';
});
document.getElementById('btn-color')?.addEventListener('click', () => {
  colorMode = colorMode === 'stone' ? 'structure' : 'stone';
  const b = document.getElementById('btn-color');
  if (b) b.textContent = colorMode === 'stone' ? '🎨 Colour: Stone' : '🎨 Colour: Structure';
  void rebuild();
});

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
    visualsRef.update({ debug: showDebug, updateBVH: false, updateProjectiles: true });
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
});
initScene()
  .then(() => loop())
  .catch((err) => {
    console.error('Failed to initialize brick castle demo:', err);
    const hint = document.querySelector('.viewport-hint');
    if (hint) hint.textContent = `Error: ${err.message}`;
  });
