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
  extractVehicleParts,
  buildVehicleScenario,
  ROLE_COLORS,
  ROLE_LABELS,
  type VehiclePartRole,
} from './glb-vehicle.js';

// ── Config ────────────────────────────────────────────────────

const MODEL_URL = './assets/buggy.glb';

const CONFIG = {
  vehicle: {
    totalMass: 1800,
    // Fracture large concave structural parts into ~this many metres per chunk so
    // their colliders are tight (a whole roll cage as one convex hull is a blob).
    // 0 = keep parts whole.
    fractureCellSize: 0.6,
    bondMaxSeparation: 0.10, // m — max surface gap auto-bonding treats as contact
    // Per-role attachment strength (LIVE). 1 = the baseline hierarchy (cargo is
    // already the weakest via its base threshold, the frame the strongest); lower
    // a role to make it shed more easily, raise it to weld it on. Drives both
    // impact and motion shedding instantly.
    bondStrength: { frame: 1, wheel: 1, panel: 1, cargo: 1, accessory: 1 } as Record<VehiclePartRole, number>,
  },
  // Live breaking knobs (no Reset needed).
  breaking: {
    impactSensitivity: 1.0, // global: how easily a direct hit detaches a part
    shedOnMotion: 1.0, // global: how readily parts tear off a violently moving body (0 = off)
  },
  projectile: {
    radius: 0.25,
    mass: 150,
    speed: 38,
    ttlMs: 8000,
  },
  solver: {
    gravity: -9.81,
    // High on purpose: the stress solver should hold the car rock-solid under
    // gravity. All breaking is driven by the controlled onImpact path, not stress,
    // so a free body never sags or sheds parts just sitting/settling.
    materialScale: 1e12,
  },
  physics: {
    contactForceScale: 30,
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
  // Parts shed via impact (cutNodeBonds doesn't set chunk.detached, so track cuts).
  el('stat-detached')!.textContent = `${_impactCuts} / ${core.chunks.length}`;
}

// ── Main ──────────────────────────────────────────────────────

let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let shooter: ReturnType<typeof mountShooter> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let showDebug = false;
let colorByRole = true;

// Cached parsed model so Reset / Drop rebuild without re-fetching 8 MB.
let gltfScene: THREE.Object3D | null = null;

// Per-node role + centroid for the impact handler (free-body breaking).
let nodeRolesRef: VehiclePartRole[] | null = null;
let nodeCentroidsRef: Array<{ x: number; y: number; z: number }> | null = null;

// Impact → detach response. A free-floating car barely stresses its bonds under a
// hit, so instead we detach weak parts locally on impact: when a node is struck
// hard enough for its role, cut its bonds (it falls off as debris, keeping its
// mesh) and splash to nearby same-or-weaker parts for a harder hit. Frame/wheels
// need a much harder hit; cargo/accessories shed easily.
// Contact-force (N) needed to tear a part of each role off. Calibrated against
// observed impacts: a ~4 m drop lands wheels at ~7–13 kN, a fast heavy projectile
// hits much harder. Cargo/accessories shed easily; wheels only pop off on a hard
// hit (and as a whole unit); the frame only lets go under a catastrophic blow.
const IMPACT_DETACH_THRESHOLD: Record<VehiclePartRole, number> = {
  accessory: 500,
  cargo: 1500,
  panel: 5000,
  wheel: 16000,
  frame: 22000,
};
// Weaker (lower) parts shed before stronger ones in a splash.
const ROLE_RANK: Record<VehiclePartRole, number> = { accessory: 0, cargo: 1, panel: 2, wheel: 3, frame: 4 };
let _impactCuts = 0;
// Ignore impacts briefly after (re)build so the vehicle can settle onto the
// ground without shedding parts. A drop from height lands well after this.
let sceneSettleUntil = 0;

// Effective detach threshold = base × the role's live attachment strength,
// divided by the global impact-sensitivity. So the per-role sliders and the
// global slider both take effect immediately on the next hit (no Reset). This is
// what actually governs destruction (the stress solver is kept stiff on purpose).
const thrOf = (role: VehiclePartRole) =>
  (IMPACT_DETACH_THRESHOLD[role] * (CONFIG.vehicle.bondStrength[role] ?? 1)) /
  Math.max(0.02, CONFIG.breaking.impactSensitivity);

function handleImpact(e: { nodeIndex: number; force: number; otherBodyHandle: number }) {
  const roles = nodeRolesRef;
  const cents = nodeCentroidsRef;
  const core = coreRef;
  if (!roles || !cents || !core) return;
  if (performance.now() < sceneSettleUntil) return; // let it settle on spawn
  const role = roles[e.nodeIndex];
  if (!role) return;
  const thr = thrOf(role);
  if (e.force < thr) return;

  // Detach the struck part (cut its bonds → it falls off as debris, keeps mesh).
  if (core.cutNodeBonds(e.nodeIndex)) _impactCuts++;

  // Harder hits shed a small local cluster of same-or-weaker parts (splash),
  // tightly bounded in both radius and count so one hit can't level the vehicle.
  const over = Math.min(4, e.force / thr);
  if (over > 1.5) {
    const radius = 0.25 + 0.06 * over; // ~0.34 .. 0.49 m
    const r2 = radius * radius;
    const maxSplash = Math.round(1 + over); // ~3 .. 5 extra parts
    const c0 = cents[e.nodeIndex];
    const rank = ROLE_RANK[role];
    const cand: Array<{ j: number; d2: number }> = [];
    for (let j = 0; j < roles.length; j++) {
      if (j === e.nodeIndex) continue;
      const rj = roles[j];
      if (!rj || ROLE_RANK[rj] > rank) continue; // only same-or-weaker
      if (e.force < thrOf(rj)) continue;
      const cj = cents[j];
      const dx = c0.x - cj.x, dy = c0.y - cj.y, dz = c0.z - cj.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < r2) cand.push({ j, d2 });
    }
    cand.sort((a, b) => a.d2 - b.d2);
    for (let k = 0; k < Math.min(maxSplash, cand.length); k++) {
      if (core.cutNodeBonds(cand[k].j)) _impactCuts++;
    }
  }
}

// Inertial shedding: a free body that's flung/spun hard doesn't stress its bonds
// in the solver, but physically the loosely-lashed cargo would tear off. So when
// a body carrying cargo/accessories moves or spins violently, progressively cut
// those weak parts' bonds (they fly off as debris). "Shed on motion" scales it.
// Base "violence" (m/s-equivalent: linear speed + weighted spin) at which each
// role starts to tear loose from a flung/spinning body. Cargo/accessories are
// low (shed readily); frame/wheels are high (only a savage tumble shakes them
// off) — but "Shed on motion" can scale all of it for full control.
const INERTIAL_SHED_BASE: Record<VehiclePartRole, number> = {
  accessory: 6, cargo: 9, panel: 28, wheel: 55, frame: 80,
};

function processInertialShedding() {
  const roles = nodeRolesRef;
  const core = coreRef;
  if (!roles || !core) return;
  if (performance.now() < sceneSettleUntil) return;
  const sens = CONFIG.breaking.shedOnMotion;
  if (sens <= 0) return;

  // "Violence" per body = linear speed + a weighting of angular speed (spin).
  const violence = new Map<number, number>();
  for (const ch of core.chunks as any[]) {
    if (!ch.active || ch.bodyHandle == null || violence.has(ch.bodyHandle)) continue;
    const body = core.world.getRigidBody(ch.bodyHandle);
    if (!body || body.isFixed?.()) continue;
    const v = body.linvel();
    const w = body.angvel();
    violence.set(ch.bodyHandle, Math.hypot(v.x, v.y, v.z) + Math.hypot(w.x, w.y, w.z) * 1.2);
  }

  for (let i = 0; i < core.chunks.length; i++) {
    const ch = (core.chunks as any[])[i];
    if (!ch.active || ch.bodyHandle == null) continue;
    const role = roles[i];
    if (!role) continue;
    const vv = violence.get(ch.bodyHandle) ?? 0;
    // Threshold scales with the role's live attachment strength, so the same
    // per-role sliders that govern impact breaking also govern motion shedding.
    const thr = (INERTIAL_SHED_BASE[role] * (CONFIG.vehicle.bondStrength[role] ?? 1)) / sens;
    if (vv > thr && Math.random() < Math.min(0.5, 0.05 * (vv / thr - 1))) {
      if (core.cutNodeBonds(i)) _impactCuts++;
    }
  }
}

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

async function ensureModelLoaded(): Promise<THREE.Object3D> {
  if (gltfScene) return gltfScene;
  setHint('Loading model…');
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(MODEL_URL);
  gltfScene = gltf.scene;
  return gltfScene;
}

/** Build (or rebuild) the destructible vehicle. `dropHeight` lifts it for a drop test. */
async function initScene(dropHeight = 0) {
  const root = await ensureModelLoaded();

  // Fresh parts every build — extract clones + bakes geometry, so repeated
  // rebuilds never accumulate transforms.
  const { parts, bounds } = extractVehicleParts(root);

  const { scenario, nodeColors, nodeRoles, summary } = await buildVehicleScenario(parts, bounds, {
    totalMass: CONFIG.vehicle.totalMass,
    fractureCellSize: CONFIG.vehicle.fractureCellSize,
    bondMaxSeparation: CONFIG.vehicle.bondMaxSeparation,
    roleStrength: CONFIG.vehicle.bondStrength,
    pinata,
    groundGap: 0.03 + Math.max(0, dropHeight),
  });
  nodeRolesRef = nodeRoles;
  nodeCentroidsRef = scenario.nodes.map((n: any) => n.centroid);
  _impactCuts = 0;
  // Grace window: settle a resting spawn (~0.1 s) or let a height-drop fall first.
  sceneSettleUntil = performance.now() + (dropHeight > 0 ? 500 : 1100);

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
    onImpact: handleImpact,
    onImpactMinForce: 1,
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

// Breaking sensitivity — LIVE (no Reset).
bindSlider('cfg-impact-sens', CONFIG.breaking, 'impactSensitivity', (v) => v.toFixed(2) + '×');
bindSlider('cfg-shed-motion', CONFIG.breaking, 'shedOnMotion', (v) =>
  v <= 0 ? 'off' : v.toFixed(2) + '×',
);

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

// Detached debris collides with the GROUND only. Fractured/split chunks have
// overlapping convex hulls; once bonds are pruned sparser, a hard hit detaches
// chunks that still overlap the intact body, and debris-vs-body penetration
// resolves as a violent explosion (noDebrisPairs only stops debris-vs-debris).
// Ground-only is density-independent and rock-solid; the trade-off is shed parts
// fall through the car/each other rather than bouncing off it. Verified stable
// under heavy destruction by scripts/soak-vehicle.mjs.
physicsConfig.debrisCollisionMode = 'noDebrisPairs';

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
    processInertialShedding(); // cargo tears off a flung/spinning body
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
    return { maxSpeed, maxDist, fastBodies: fast, dynamicBodies: dyn, bodies: core.getRigidBodyCount(), shed: _impactCuts };
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
