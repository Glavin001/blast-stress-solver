/**
 * Keystone — a dry-stone arch held up by nothing but compression.
 *
 * A semicircular ring of wedge-shaped voussoirs spans two stone abutments. There is no mortar, no
 * glue, no bolts: the only thing keeping every block aloft is the compression passing from stone to
 * stone, down through the haunches and into the ground. That load path is the arch's "thrust line",
 * and you can watch it live — the coloured lines are the per-bond stress straight from the solver
 * (getSolverDebugLines), defaulting to the Compression view so the thrust line glows through the
 * ring (green = relaxed, red = highly compressed).
 *
 * The centre block is the keystone (tinted gold): the last stone placed, the one that locks the
 * ring. Hit "Knock Out Keystone" and a heavy striker overstresses the crown joints until they
 * fracture — the ring splits into two unsupported half-arches whose weight now overhangs their
 * feet, so they peel off the abutments and the whole span comes down. (Cutting bonds alone only
 * removes them from the stress graph; the pieces physically separate only when a joint is
 * *overstressed* — by the striker, by a crank of the Load, by a dropped weight, or by a shot.)
 *
 * The scenario builder (and that it stands intact yet collapses under overload) is exercised
 * headlessly in `arch.showcase.test.ts`.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { buildDestructibleCore, createFrameProfilerOverlay, createRecordingOverlay } from 'blast-stress-solver/rapier';
import { createDestructibleThreeBundle } from 'blast-stress-solver/three';
import { buildArchScenario } from 'blast-stress-solver/scenarios';
import { mountPhysicsControls, physicsCoreOverrides } from './physics-controls.js';
import { mountShooter } from './shooter-fps.js';

// ── Scene constants ───────────────────────────────────────────
const ARCH = { radius: 7, thickness: 1.4, depth: 3.5, springingHeight: 4 };

// Stone material limits — tuned so the self-weight thrust sits ~40% of the compression limit at
// Load 1× (the thrust line reads green→amber and the arch holds), while dry-stone joints are weak
// in tension/shear so an unbalanced load hinges them. Scaled live by `strength`; if the default
// reads too hot or too cold for your machine, nudge Strength.
const BASE_LIMITS = {
  compressionElasticLimit: 2e5, compressionFatalLimit: 6e5,
  tensionElasticLimit: 8e4, tensionFatalLimit: 2.5e5,
  shearElasticLimit: 1.5e5, shearFatalLimit: 5e5,
};
const G = 9.81;
const STRESS_MODE = { Max: 0, Compression: 1, Tension: 2, Shear: 3 } as const;

const CONFIG = { load: 1.0, strength: 1.0, mode: STRESS_MODE.Compression as number, voussoirs: 15, cracksOnly: false };

// ── Three.js setup ────────────────────────────────────────────
const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d13);

const camera = new THREE.PerspectiveCamera(52, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
camera.position.set(2, 9, 30);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, ARCH.springingHeight + ARCH.radius * 0.4, 0);
controls.enableDamping = true;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dir = new THREE.DirectionalLight(0xffeedd, 1.1);
dir.position.set(10, 22, 14);
dir.castShadow = true;
dir.shadow.camera.near = 1;
dir.shadow.camera.far = 80;
(['left', 'right', 'top', 'bottom'] as const).forEach((s) => ((dir.shadow.camera as any)[s] = s === 'left' || s === 'bottom' ? -30 : 30));
scene.add(dir);

// Ground + a dark "chasm" the arch spans, for orientation.
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(400, 400),
  new THREE.MeshStandardMaterial({ color: 0x141826, roughness: 0.96 }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0;
ground.receiveShadow = true;
scene.add(ground);
const chasm = new THREE.Mesh(
  new THREE.PlaneGeometry(2 * ARCH.radius - ARCH.thickness, ARCH.depth * 2.4),
  new THREE.MeshStandardMaterial({ color: 0x06080f, roughness: 1 }),
);
chasm.rotation.x = -Math.PI / 2;
chasm.position.set(0, 0.02, 0);
scene.add(chasm);

// ── Live per-bond stress overlay (the "thrust line") ──────────
const lineGeom = new THREE.BufferGeometry();
const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthTest: false });
const stressLines = new THREE.LineSegments(lineGeom, lineMat);
stressLines.renderOrder = 999;
scene.add(stressLines);
const decode = (c: number) => ({ r: ((c >> 16) & 255) / 255, g: ((c >> 8) & 255) / 255, b: (c & 255) / 255 });
const stressOf = (c: number) => { const r = (c >> 16) & 255, g = (c >> 8) & 255; return (r - g + 255) / 510; };

function setText(id: string, v: string) { const el = document.getElementById(id); if (el) el.textContent = v; }

// ── Scene lifecycle ───────────────────────────────────────────
let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let shooter: ReturnType<typeof mountShooter> | null = null;
let keystoneIndex = 0;
let archNodeCount = 0;
let initialBonds = 0;
let keystonePulled = false;
let rebuilding = false;

const profiler = createFrameProfilerOverlay();
const recorder = createRecordingOverlay({
  mount: document.getElementById('recorder-slot') ?? undefined,
  exportName: 'keystone-arch-recording',
  getProfilerExport: () => profiler.exportData(),
});

function scaledLimits() {
  const s = CONFIG.strength, o: Record<string, number> = {};
  for (const k of Object.keys(BASE_LIMITS)) o[k] = (BASE_LIMITS as Record<string, number>)[k] * s;
  return o;
}

/** Build wedge ConvexGeometry per voussoir from the hull corners the scenario emitted. */
function geometriesFromScenario(scenario: any): (THREE.BufferGeometry | undefined)[] {
  const hulls = (scenario.parameters?.voussoirHulls ?? []) as number[][];
  return hulls.map((hull) => {
    if (!hull || hull.length < 12) return undefined; // abutments → box fallback
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < hull.length; i += 3) pts.push(new THREE.Vector3(hull[i], hull[i + 1], hull[i + 2]));
    return new ConvexGeometry(pts);
  });
}

function nodeColorsFromScenario(scenario: any, keystone: number): THREE.Color[] {
  const raw = (scenario.parameters?.nodeColors ?? []) as number[];
  const colors = raw.map((hex, i) => {
    const c = new THREE.Color(hex);
    if (i < archNodeCount) c.offsetHSL(0, 0, ((i * 37) % 9 - 4) / 110); // subtle per-stone variation
    return c;
  });
  if (colors[keystone]) colors[keystone] = new THREE.Color(0xd9a441); // gold keystone
  return colors;
}

async function initScene() {
  keystonePulled = false;
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = 'The arch stands on pure compression — watch the thrust line glow. Knock out the keystone (gold) to bring it down.';

  const { scenario, keystoneIndex: ks, archNodeCount: an } = buildArchScenario({
    voussoirs: CONFIG.voussoirs,
    radius: ARCH.radius,
    thickness: ARCH.thickness,
    depth: ARCH.depth,
    springingHeight: ARCH.springingHeight,
  });
  keystoneIndex = ks;
  archNodeCount = an;
  (scenario.parameters as any).fragmentGeometries = geometriesFromScenario(scenario);
  const nodeColors = nodeColorsFromScenario(scenario, ks);

  const core = await buildDestructibleCore({
    scenario,
    gravity: -G * CONFIG.load,
    materialScale: 1,
    solverSettings: { ...scaledLimits(), maxSolverIterationsPerFrame: 100, graphReductionLevel: 0 },
    contactForceScale: 25,
    ...physicsCoreOverrides(),
  });
  // Solve every frame so the live Load slider and the thrust-line colours stay current.
  core.setFracturePolicy({ idleSkip: false });

  const group = new THREE.Group();
  scene.add(group);
  // Per-chunk meshes (not batched): the voussoirs are non-indexed ConvexGeometry wedges and the
  // abutments are indexed boxes, which a single BatchedMesh cannot mix. ~17 meshes is cheap.
  const visuals = createDestructibleThreeBundle({
    core, scenario, root: group, useBatchedMesh: false, includeDebugLines: false,
    nodeColors, materialColors: true,
  });

  coreRef = core;
  visualsRef = visuals;
  recorder.attach(core, { scenario, meta: { demo: 'keystone-arch', config: CONFIG } });
  profiler.attach(core);
  initialBonds = core.getActiveBondsCount();
}

async function rebuild() {
  if (rebuilding) return;
  rebuilding = true;
  visualsRef?.dispose();
  coreRef?.dispose();
  coreRef = null; visualsRef = null;
  try { await initScene(); } finally { rebuilding = false; }
}

function knockOutKeystone() {
  const core = coreRef; if (!core) return;
  // A heavy striker dropped onto the crown overstresses the keystone's joints until they fracture.
  // Once the crown is broken the ring is two unsupported half-arches → they peel off and fall. (A
  // bare cutNodeBonds would only desync the stress graph; the pieces separate on *overstress*.)
  const crownY = ARCH.springingHeight + ARCH.radius + ARCH.thickness / 2;
  core.cutNodeBonds?.(keystoneIndex); // also drop it from the stress graph so the halves go free
  core.enqueueProjectile({
    position: { x: 0, y: crownY + 9, z: 0 },
    velocity: { x: 0, y: -38, z: 0 },
    radius: 0.8, mass: 16000, ttl: 8000,
  });
  keystonePulled = true;
  const hint = document.querySelector('.viewport-hint');
  if (hint) hint.textContent = 'Keystone knocked out — with the crown gone the half-arches have nothing to lean on, so they fall.';
}

function dropLoad() {
  const core = coreRef; if (!core) return;
  const crownY = ARCH.springingHeight + ARCH.radius + ARCH.thickness / 2;
  // Land it over a haunch (off-centre): an asymmetric load is what actually hinges an arch.
  core.enqueueProjectile({
    position: { x: ARCH.radius * 0.5, y: crownY + 7, z: 0 },
    velocity: { x: 0, y: -16, z: 0 },
    radius: 1.0, mass: 12000, ttl: 8000,
  });
}

// ── Controls ──────────────────────────────────────────────────
function bindSlider(id: string, get: () => number, set: (v: number) => void, fmt: (v: number) => string, onInput?: (v: number) => void, onChange?: () => void) {
  const s = document.getElementById(id) as HTMLInputElement | null;
  const d = document.getElementById(id + '-value');
  if (!s) return;
  s.value = String(get());
  if (d) d.textContent = fmt(get());
  s.addEventListener('input', () => { const v = parseFloat(s.value); set(v); if (d) d.textContent = fmt(v); onInput?.(v); });
  if (onChange) s.addEventListener('change', onChange);
}
bindSlider('cfg-load', () => CONFIG.load, (v) => CONFIG.load = v, (v) => v.toFixed(1) + '×',
  (v) => coreRef?.setGravity(-G * v)); // LIVE: effective self-weight
bindSlider('cfg-strength', () => CONFIG.strength, (v) => CONFIG.strength = v, (v) => v.toFixed(2) + '×',
  undefined, () => void rebuild());
bindSlider('cfg-voussoirs', () => CONFIG.voussoirs, (v) => CONFIG.voussoirs = Math.round(v), (v) => String(Math.round(v)),
  undefined, () => void rebuild());

const modeSel = document.getElementById('cfg-mode') as HTMLSelectElement | null;
if (modeSel) { modeSel.value = String(CONFIG.mode); modeSel.addEventListener('change', () => { CONFIG.mode = parseInt(modeSel.value, 10) || 0; }); }

const cracksOnly = document.getElementById('opt-cracks-only') as HTMLInputElement | null;
cracksOnly?.addEventListener('change', () => { CONFIG.cracksOnly = !!cracksOnly.checked; });

document.getElementById('btn-keystone')?.addEventListener('click', () => knockOutKeystone());
document.getElementById('btn-drop')?.addEventListener('click', () => dropLoad());
document.getElementById('btn-reset')?.addEventListener('click', () => { void rebuild(); });

mountPhysicsControls({
  getCore: () => coreRef,
  onRebuild: () => void rebuild(),
  include: { centrifugal: false, debug: false },
});

// ── Render loop ───────────────────────────────────────────────
function updateStress() {
  const core = coreRef; if (!core) return;
  const lines = core.getSolverDebugLines(CONFIG.mode);
  const pos: number[] = [], col: number[] = [];
  let maxS = 0;
  for (const ln of lines) {
    const s = Math.max(stressOf(ln.color0), stressOf(ln.color1));
    if (s > maxS) maxS = s;
    if (CONFIG.cracksOnly && s < 0.35) continue;
    const a = decode(ln.color0), b = decode(ln.color1);
    pos.push(ln.p0.x, ln.p0.y, ln.p0.z, ln.p1.x, ln.p1.y, ln.p1.z);
    col.push(a.r, a.g, a.b, b.r, b.g, b.b);
  }
  lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  lineGeom.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  setText('stat-max-stress', (maxS * 100).toFixed(0) + '%');
  setText('stat-broken', String(initialBonds - core.getActiveBondsCount()));
  setText('stat-bonds', String(core.getActiveBondsCount()));
  setText('stat-bodies', String(core.getRigidBodyCount()));
}

const clock = new THREE.Clock();
function loop() {
  requestAnimationFrame(loop);
  profiler.render();
  recorder.render();
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();
  if (coreRef && visualsRef) {
    coreRef.step(dt);
    visualsRef.update({ debug: false, updateBVH: false, updateProjectiles: true });
    updateStress();
  }
  shooter?.update();
  renderer.render(scene, camera);
}

function onResize() {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

shooter = mountShooter({
  canvas, camera, controls, scene,
  getCore: () => coreRef,
  getBallParams: () => ({ radius: 0.45, mass: 700, speed: 48 }),
});
initScene().then(() => loop()).catch((err) => {
  console.error('Failed to initialize keystone-arch demo:', err);
  const hint = document.querySelector('.viewport-hint');
  if (hint) hint.textContent = `Error: ${err.message}`;
});

void STRESS_MODE;
