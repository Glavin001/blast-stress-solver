/**
 * Destructible Vehicle Demo
 *
 * Loads a GLB model (a junkyard buggy), decomposes it into its named parts,
 * classifies each part into a structural role (frame / wheel / panel / cargo /
 * accessory) and wires it up as ONE destructible body with a *hierarchy* of bond
 * strengths:
 *
 *   frame↔frame  (roll cage / chassis)  strongest  ── keeps the shell together
 *   frame↔wheel  (hub / axle)           very strong
 *   frame↔panel                         strong
 *   …↔cargo / …↔accessory               weak       ── strapped-on payload sheds first
 *
 * So a light hit knocks the barrels and crates off while the cage holds; a hard
 * enough hit (or a big drop) shatters the skeleton itself. Click to shoot, or hit
 * "Drop from height". Parts are coloured by role so the hierarchy is visible.
 *
 * The decomposition + bond hierarchy live in ./glb-vehicle.ts; this file is just
 * the scene/render/UI shell (mirrors fractured-wall.ts).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import Stats from 'three/addons/libs/stats.module.js';
import * as pinata from '@dgreenheck/three-pinata';
import { buildDestructibleCore, createFrameProfilerOverlay, createRecordingOverlay } from 'blast-stress-solver/rapier';
import { createDestructibleThreeBundle, RapierDebugRenderer } from 'blast-stress-solver/three';
import { pipelineCoreOverrides, mountPipelineControls } from './pipeline-controls.js';
import { mountPhysicsControls, physicsCoreOverrides, physicsConfig } from './physics-controls.js';
import { mountShooter } from './shooter-fps.js';
import {
  buildVehicleScenarioFromAsset,
  extractVehicleParts,
  ROLE_COLORS,
  ROLE_LABELS,
  type VehiclePart,
  type VehiclePartRole,
  type VehiclePiecesAsset,
} from './glb-vehicle.js';

// ── Config ────────────────────────────────────────────────────

// Pre-decomposed pieces asset (CoACD pipeline output): tight, non-overlapping
// convex collider pieces, so full debris collision doesn't explode.
const ASSET_URL = './assets/buggy.pieces.json';
// Original full-detail model — used ONLY for rendering (render != collision). Each
// collider piece keeps its tight CoACD hull for physics but draws a slice of this
// model, so the car looks like the real vehicle, not faceted hulls.
const MODEL_URL = './assets/buggy.glb';

const CONFIG = {
  vehicle: {
    totalMass: 1800,
    // Fracture large concave structural parts into ~this many metres per chunk so
    // their colliders are tight (a whole roll cage as one convex hull is a blob).
    // 0 = keep parts whole.
    fractureCellSize: 0.6,
    bondMaxSeparation: 0.12, // m — max surface gap auto-bonding treats as contact
    // Per-role attachment strength (LIVE). 1 = the baseline hierarchy (cargo is
    // already the weakest via its base threshold, the frame the strongest); lower
    // a role to make it shed more easily, raise it to weld it on. Drives both
    // impact and motion shedding instantly.
    bondStrength: { frame: 1, wheel: 1, panel: 1, cargo: 1, accessory: 1 } as Record<VehiclePartRole, number>,
  },
  projectile: {
    // A heavy ball (~wrecking-ball): at the calibrated materialScale this sheds a
    // satisfying few parts per hit (cargo first, the cage holding) without
    // shattering the car or exploding. Swept in scripts/probe-vehicle.mjs.
    radius: 0.25,
    mass: 150,
    speed: 45,
    ttlMs: 8000,
  },
  solver: {
    gravity: -9.81,
    // Holds the intact car rock-solid at rest. The CoACD pieces have small contact
    // areas, so the per-area bonds need a high materialScale not to sag under
    // gravity; impact breaking is then driven by contactForceScale (below) so a hit
    // still spikes local bond stress past the limit. Soak-verified: holds at rest,
    // sheds on light/heavy shots, stable drops.
    materialScale: 1e13,
  },
  physics: {
    // High on purpose: with the strong (high-materialScale) bonds, this amplifies
    // projectile/ground contact forces into the stress solver so a hit overstresses
    // and sheds local parts (light shot ~3, heavy ~9 in the soak).
    contactForceScale: 200,
    skipSingleBodies: false,
  },
};

// ── Three.js setup ────────────────────────────────────────────

const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d13);
scene.fog = new THREE.FogExp2(0x0a0d13, 0.02);

const camera = new THREE.PerspectiveCamera(52, canvas.clientWidth / canvas.clientHeight, 0.1, 200);
camera.position.set(6.5, 3.6, 7.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.7, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffeedd, 1.05);
dirLight.position.set(8, 14, 10);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -12;
dirLight.shadow.camera.right = 12;
dirLight.shadow.camera.top = 12;
dirLight.shadow.camera.bottom = -6;
scene.add(dirLight);

const rimLight = new THREE.DirectionalLight(0x88aaff, 0.3);
rimLight.position.set(-8, 6, -10);
scene.add(rimLight);

// Ground plane (visual; the solver has its own collision ground at y=0)
const groundMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(80, 80),
  new THREE.MeshStandardMaterial({ color: 0x1a1e2f, roughness: 0.9, metalness: 0.05 }),
);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.position.y = 0;
groundMesh.receiveShadow = true;
scene.add(groundMesh);

// ── Stats panel ───────────────────────────────────────────────

const stats = new Stats();
stats.dom.style.position = 'absolute';
stats.dom.style.top = '0';
stats.dom.style.left = '0';
(document.querySelector('.viewport') as HTMLElement)?.appendChild(stats.dom);

// ── Perf tracking ─────────────────────────────────────────────

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

// ── Status HUD ────────────────────────────────────────────────

function updateStatus(core: any) {
  const el = (id: string) => document.getElementById(id);
  el('stat-bodies')!.textContent = String(core.getRigidBodyCount());
  el('stat-bonds')!.textContent = String(core.getActiveBondsCount());
  el('stat-projectiles')!.textContent = String(core.projectiles.length);
  // Parts the stress solver has broken off (their own dynamic bodies now).
  const detached = (core.chunks as any[]).filter((c) => c.detached || c.destroyed).length;
  el('stat-detached')!.textContent = `${detached} / ${core.chunks.length}`;
}

// ── Main ──────────────────────────────────────────────────────

let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let shooter: ReturnType<typeof mountShooter> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let showDebug = false;
let colorByRole = true;

// Cached pieces asset so Reset / Drop rebuild without re-fetching.
let piecesAsset: VehiclePiecesAsset | null = null;
// Cached full-detail render parts (from buggy.glb) for the render!=collision path.
let renderParts: VehiclePart[] | null = null;

// Breaking is fully stress-driven: the solver sees gravity + the ground-contact
// reaction + projectile contact forces, and breaks bonds wherever the stress
// exceeds the (area-encoded) per-role limit. No scripted onImpact/inertial-shed
// path — `materialScale` is calibrated so the intact car holds at rest yet a
// hit/drop overstresses the weak joints. (See freeBodyGroundStress.test.ts.)

const profiler = createFrameProfilerOverlay();
const recorder = createRecordingOverlay({
  mount: document.getElementById('recorder-slot') ?? undefined,
  exportName: 'destructible-vehicle-recording',
  getProfilerExport: () => profiler.exportData(),
});

function setHint(text: string) {
  const hint = document.querySelector('.viewport-hint');
  if (hint) hint.textContent = text;
}

async function ensureAssetLoaded(): Promise<VehiclePiecesAsset> {
  if (piecesAsset) return piecesAsset;
  setHint('Loading model…');
  const res = await fetch(ASSET_URL);
  if (!res.ok) throw new Error(`failed to load ${ASSET_URL}: ${res.status} ${res.statusText}`);
  piecesAsset = (await res.json()) as VehiclePiecesAsset;
  return piecesAsset;
}

/** Load the original full-detail model and extract its world-space parts (render only). */
async function ensureRenderPartsLoaded(): Promise<VehiclePart[] | null> {
  if (renderParts) return renderParts;
  try {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(MODEL_URL);
    const { parts } = extractVehicleParts(gltf.scene, { splitComponents: false });
    renderParts = parts;
    console.log(`Loaded render model: ${parts.length} detail parts from ${MODEL_URL}`);
  } catch (err) {
    console.warn('Failed to load detailed render model; rendering collider hulls:', err);
    renderParts = null;
  }
  return renderParts;
}

/** Build (or rebuild) the destructible vehicle. `dropHeight` lifts it for a drop test. */
async function initScene(dropHeight = 0) {
  const asset = await ensureAssetLoaded();
  const detailParts = await ensureRenderPartsLoaded();

  const { scenario, nodeColors, summary } = await buildVehicleScenarioFromAsset(asset, {
    totalMass: CONFIG.vehicle.totalMass,
    bondMaxSeparation: CONFIG.vehicle.bondMaxSeparation,
    roleStrength: CONFIG.vehicle.bondStrength,
    groundGap: 0.03 + Math.max(0, dropHeight),
    renderParts: detailParts ?? undefined,
  });

  console.log(
    `Destructible vehicle: ${summary.parts} parts → ${summary.nodes} nodes, ${summary.bonds} bonds`,
    summary.roleCounts,
  );

  const core = await buildDestructibleCore({
    scenario,
    gravity: CONFIG.solver.gravity,
    materialScale: CONFIG.solver.materialScale,
    contactForceScale: CONFIG.physics.contactForceScale,
    skipSingleBodies: CONFIG.physics.skipSingleBodies,
    ...physicsCoreOverrides(),
    ...pipelineCoreOverrides(),
  });

  const group = new THREE.Group();
  scene.add(group);

  const visuals = createDestructibleThreeBundle({
    core,
    scenario,
    root: group,
    // Individual meshes (not BatchedMesh): only ~dozens of parts, some high-poly,
    // so per-mesh is simpler and avoids batched vertex-capacity limits.
    useBatchedMesh: false,
    includeDebugLines: true,
    nodeColors,
    materialColors: colorByRole,
  });

  rapierDebug?.dispose();
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: showDebug });

  coreRef = core;
  core.setSolverCentrifugalEnabled(physicsConfig.centrifugal);
  recorder.attach(core, { scenario, meta: { demo: 'destructible-vehicle' } });
  profiler.attach(core);
  visualsRef = visuals;

  renderLegend(summary.roleCounts);
  setHint(
    dropHeight > 0
      ? 'Dropped! Click to shoot · drag to orbit'
      : 'Click to shoot · drag to orbit · cargo sheds first, the cage holds',
  );
}

function disposeCurrent() {
  visualsRef?.dispose();
  // Remove the visuals group from the scene.
  if (visualsRef?.object?.parent) visualsRef.object.parent.remove(visualsRef.object);
  coreRef?.dispose();
  coreRef = null;
  visualsRef = null;
}

// ── Role legend ───────────────────────────────────────────────

function renderLegend(roleCounts: Record<VehiclePartRole, number>) {
  const host = document.getElementById('role-legend');
  if (!host) return;
  const order: VehiclePartRole[] = ['frame', 'wheel', 'panel', 'cargo', 'accessory'];
  host.innerHTML = order
    .filter((r) => roleCounts[r] > 0)
    .map((r) => {
      const hex = '#' + ROLE_COLORS[r].toString(16).padStart(6, '0');
      return (
        `<div class="legend-row">` +
        `<span class="legend-swatch" style="background:${hex}"></span>` +
        `<span class="legend-label">${ROLE_LABELS[r]}</span>` +
        `<span class="legend-count">${roleCounts[r]}</span>` +
        `</div>`
      );
    })
    .join('');
}

// ── UI wiring ─────────────────────────────────────────────────

document.getElementById('btn-reset')?.addEventListener('click', async () => {
  disposeCurrent();
  await initScene(0);
});

document.getElementById('btn-drop')?.addEventListener('click', async () => {
  disposeCurrent();
  await initScene(4.0);
});

document.getElementById('btn-debug')?.addEventListener('click', () => {
  showDebug = !showDebug;
  rapierDebug?.setEnabled(showDebug);
  document.getElementById('btn-debug')!.textContent = showDebug ? '◈ Hide Debug' : '◇ Show Debug';
});

document.getElementById('btn-color')?.addEventListener('click', () => {
  colorByRole = !colorByRole;
  visualsRef?.setMaterialColors(colorByRole);
  document.getElementById('btn-color')!.textContent = colorByRole ? '🎨 Color: Role' : '🎨 Color: State';
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

function bindCheckbox(id: string, obj: Record<string, any>, key: string) {
  const checkbox = document.getElementById(id) as HTMLInputElement | null;
  if (!checkbox) return;
  checkbox.checked = !!obj[key];
  checkbox.addEventListener('change', () => {
    obj[key] = checkbox.checked;
  });
}

// Vehicle config (needs Reset)
bindSlider('cfg-mass', CONFIG.vehicle, 'totalMass', (v) => v.toLocaleString() + ' kg');
bindSlider('cfg-fracture', CONFIG.vehicle, 'fractureCellSize', (v) =>
  v <= 0 ? 'off (parts whole)' : `${v.toFixed(2)} m/chunk`,
);
bindSlider('cfg-bond-sep', CONFIG.vehicle, 'bondMaxSeparation', (v) => v.toFixed(2) + ' m');

// Bond strength hierarchy — per-role attachment (Reset to apply).
const strengthFmt = (v: number) => `${v.toFixed(2)}×`;
bindSlider('cfg-bs-frame', CONFIG.vehicle.bondStrength, 'frame', strengthFmt);
bindSlider('cfg-bs-wheel', CONFIG.vehicle.bondStrength, 'wheel', strengthFmt);
bindSlider('cfg-bs-panel', CONFIG.vehicle.bondStrength, 'panel', strengthFmt);
bindSlider('cfg-bs-cargo', CONFIG.vehicle.bondStrength, 'cargo', strengthFmt);
bindSlider('cfg-bs-accessory', CONFIG.vehicle.bondStrength, 'accessory', strengthFmt);

// Projectile (immediate)
bindSlider('cfg-proj-radius', CONFIG.projectile, 'radius', (v) => v.toFixed(2) + ' m');
bindSlider('cfg-proj-mass', CONFIG.projectile, 'mass', (v) => v.toLocaleString() + ' kg');
bindSlider('cfg-proj-speed', CONFIG.projectile, 'speed', (v) => v.toFixed(0) + ' m/s');

// Solver (immediate-ish; gravity needs Reset)
bindSlider('cfg-gravity', CONFIG.solver, 'gravity', (v) => v.toFixed(1));
{
  const slider = document.getElementById('cfg-material') as HTMLInputElement | null;
  const display = document.getElementById('cfg-material-value');
  if (slider) {
    const exp = Math.log10(CONFIG.solver.materialScale);
    slider.value = String(exp);
    if (display) display.textContent = `1e${exp.toFixed(0)}`;
    slider.addEventListener('input', () => {
      const e = parseFloat(slider.value);
      CONFIG.solver.materialScale = Math.pow(10, e);
      if (display) display.textContent = `1e${e.toFixed(1)}`;
    });
  }
}

// Detached debris bounces off the intact car and the ground, but NOT off other
// debris. This is the one genuine collider-fidelity mitigation: the runtime builds
// a single convex hull per node, and fractured/concave pieces have *overlapping*
// Debris collision: the CoACD pipeline (split → weld+simplify → CoACD → clip)
// makes tight, near-non-overlapping collider pieces. A few thin sliver pieces from
// the messy source can't be fully de-interpenetrated, so under full 'all' debris
// collision a mass shatter can still cascade off those residual overlaps. The
// runtime-robust choice is 'noDebrisPairs' (debris still bounces off the car +
// ground, just not off each other), which passes settle/light/heavy/drop in
// scripts/soak-vehicle.mjs. Eliminating the slivers entirely (more aggressive
// simplify) would let this be 'all'.
physicsConfig.debrisCollisionMode = 'all';

// Shared Physics / Optimization / Features controls.
mountPhysicsControls({ getCore: () => coreRef, include: { debug: false } });
bindSlider('cfg-contact-force', CONFIG.physics, 'contactForceScale', (v) => v.toFixed(0));
bindCheckbox('cfg-skip-single', CONFIG.physics, 'skipSingleBodies');

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

  updatePerfStats();
  stats.end();
}

// ── Resize ────────────────────────────────────────────────────

function onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

// Debug hook for headless inspection / controlled camera angles.
(window as any).__vehicleDemo = {
  get camera() { return camera; },
  get controls() { return controls; },
  get core() { return coreRef; },
  get visuals() { return visualsRef; },
  setView(pos: [number, number, number], target: [number, number, number] = [0, 0.7, 0]) {
    camera.position.set(pos[0], pos[1], pos[2]);
    controls.target.set(target[0], target[1], target[2]);
    controls.update();
  },
  setDebug(colliders: boolean, bonds: boolean) {
    showDebug = !!bonds;
    rapierDebug?.setEnabled(!!colliders);
  },
  // Stability probe for headless soak tests: max linear speed and max distance
  // from origin across dynamic bodies (an explosion spikes both), plus counts.
  metrics() {
    const core = coreRef;
    if (!core) return null;
    let maxSpeed = 0, maxDist = 0, dyn = 0, fast = 0;
    const w: any = core.world;
    w.forEachRigidBody((rb: any) => {
      if (rb.isFixed?.()) return;
      const v = rb.linvel();
      const sp = Math.hypot(v.x, v.y, v.z);
      const t = rb.translation();
      const d = Math.hypot(t.x, t.y, t.z);
      if (sp > maxSpeed) maxSpeed = sp;
      if (d > maxDist) maxDist = d;
      if (sp > 60) fast++; // way above projectile speed → exploding debris
      dyn++;
    });
    const shed = (core.chunks as any[]).filter((c) => c.detached || c.destroyed).length;
    return { maxSpeed, maxDist, fastBodies: fast, dynamicBodies: dyn, bodies: core.getRigidBodyCount(), shed };
  },
  // Set debris collision mode at runtime (for the shatter test / QA).
  setDebris(mode: string) { (coreRef as any)?.setDebrisCollisionMode?.(mode); },
  // Acceptance test for non-overlapping colliders: cut every bond so all chunks
  // become free rigid bodies at once. With debris collisions ON and tight,
  // non-overlapping colliders, this must just collapse and settle — NOT explode.
  // Overlapping convex hulls instead resolve their penetration as a blast.
  shatterAll() {
    const core = coreRef as any;
    if (!core) { console.warn('[vehicle] shatterAll: no core'); return 0; }
    // Lift any body cap so every chunk can become its own rigid body.
    core.setFracturePolicy?.({ maxDynamicBodies: 0, maxFracturesPerFrame: 0, maxNewBodiesPerFrame: 0 });
    if (typeof core.shatterAll !== 'function') { console.warn('[vehicle] shatterAll: core has no shatterAll'); return 0; }
    return core.shatterAll();
  },
};

// ── Boot ──────────────────────────────────────────────────────

mountPipelineControls();
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
    console.error('Failed to initialize destructible vehicle demo:', err);
    setHint(`Error: ${err?.message ?? err}`);
  });
