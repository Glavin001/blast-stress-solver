/**
 * Stress Cracks — a cantilever beam you overload until it cracks and snaps.
 *
 * The clearest demonstration of the stress solver's *directional* fidelity and of the
 * elastic→fatal band as a VISUAL cracking signal. A beam is built into a wall and sticks
 * out horizontally. Increase the Load (its effective self-weight) and watch stress
 * concentrate at the ROOT: the bonds there bloom green → yellow → red as they cross the
 * elastic limit (cracking), and when they reach the fatal limit the root ruptures and the
 * beam falls. The failure is tension-driven at the TOP fiber — switch the Stress view to
 * "Tension" to see it light up while "Compression" (the bottom fiber) stays dark.
 *
 * The colored lines ARE the live per-bond stress straight from the solver
 * (getSolverDebugLines): green = relaxed, red = at the fatal limit, gone = broken.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildDestructibleCore } from 'blast-stress-solver/rapier';
import { createDestructibleThreeBundle } from 'blast-stress-solver/three';

// ── Cantilever scenario ───────────────────────────────────────
const BEAM = { length: 8, height: 3, mass: 80 };
function buildCantilever() {
  const { length, height, mass } = BEAM;
  const nodes: any[] = [];
  const sizes: any[] = [];
  const idx = new Map<string, number>();
  for (let x = 0; x <= length; x++) {
    for (let y = 1; y <= height; y++) {
      idx.set(`${x},${y}`, nodes.length);
      nodes.push({ centroid: { x, y, z: 0 }, mass: x === 0 ? 0 : mass, volume: 1 }); // x=0 column = fixed wall
      sizes.push({ x: 1, y: 1, z: 1 });
    }
  }
  const bonds: any[] = [];
  const add = (a: number, b: number, cx: number, cy: number, nx: number, ny: number) =>
    bonds.push({ node0: a, node1: b, centroid: { x: cx, y: cy, z: 0 }, normal: { x: nx, y: ny, z: 0 }, area: 1 });
  for (let x = 0; x <= length; x++) {
    for (let y = 1; y <= height; y++) {
      const a = idx.get(`${x},${y}`)!;
      if (x < length) add(a, idx.get(`${x + 1},${y}`)!, x + 0.5, y, 1, 0); // horizontal
      if (y < height) add(a, idx.get(`${x},${y + 1}`)!, x, y + 0.5, 0, 1); // vertical
    }
  }
  const tipNodes: number[] = [];
  for (let y = 1; y <= height; y++) tipNodes.push(idx.get(`${length},${y}`)!);
  return { scenario: { nodes, bonds, parameters: { fragmentSizes: sizes } }, tipNodes };
}

// Material stress limits — concrete-like (weak in tension), scaled by `strength`.
const BASE_LIMITS = {
  compressionElasticLimit: 2e6, compressionFatalLimit: 6e6,
  tensionElasticLimit: 2e5, tensionFatalLimit: 6e5,
  shearElasticLimit: 4e5, shearFatalLimit: 1.2e6,
};
const G = 9.81;
const STRESS_MODE = { Max: 0, Compression: 1, Tension: 2, Shear: 3 } as const;

const CONFIG = { loadFactor: 4.6, strength: 1.0, mode: 0, cracksOnly: false };

// ── Three.js setup ────────────────────────────────────────────
const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d13);

const camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 400);
camera.position.set(4, 3, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(4, 0, 0);
controls.enableDamping = true;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dir = new THREE.DirectionalLight(0xffeedd, 1.0);
dir.position.set(8, 16, 12);
dir.castShadow = true;
scene.add(dir);
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(200, 200),
  new THREE.MeshStandardMaterial({ color: 0x161a28, roughness: 0.95 }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -10;
ground.receiveShadow = true;
scene.add(ground);
// A slab behind the wall column so the beam reads as "built into a wall".
const wall = new THREE.Mesh(
  new THREE.BoxGeometry(1.4, BEAM.height + 3, 4),
  new THREE.MeshStandardMaterial({ color: 0x2a3147, roughness: 0.9 }),
);
wall.position.set(-0.7, BEAM.height / 2 + 0.5, 0);
scene.add(wall);

// ── Live per-bond stress overlay (the "cracks") ───────────────
const crackGeom = new THREE.BufferGeometry();
const crackMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthTest: false });
const crackLines = new THREE.LineSegments(crackGeom, crackMat);
crackLines.renderOrder = 999;
scene.add(crackLines);
const decode = (c: number) => ({ r: ((c >> 16) & 255) / 255, g: ((c >> 8) & 255) / 255, b: (c & 255) / 255 });
const stressOf = (c: number) => { const r = (c >> 16) & 255, g = (c >> 8) & 255; return (r - g + 255) / 510; };

function setText(id: string, v: string) { const el = document.getElementById(id); if (el) el.textContent = v; }

// ── Scene lifecycle ───────────────────────────────────────────
let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let rebuilding = false;
let initialBonds = 0;

function scaledLimits() {
  const s = CONFIG.strength, o: any = {};
  for (const k of Object.keys(BASE_LIMITS)) o[k] = (BASE_LIMITS as any)[k] * s;
  return o;
}

async function initScene() {
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = 'Beam loaded near its limit (deep red at the root) — nudge Load up to snap it, or down to relieve the cracks';
  const { scenario } = buildCantilever();
  const core = await buildDestructibleCore({
    scenario,
    gravity: -G * CONFIG.loadFactor,
    materialScale: 1,
    solverSettings: { ...scaledLimits(), maxSolverIterationsPerFrame: 100, graphReductionLevel: 0 },
    friction: 0.4, restitution: 0, contactForceScale: 12,
    damage: { enabled: false },
    debrisCleanup: { mode: 'always' as any, debrisTtlMs: 12000, maxCollidersForDebris: 3 },
  });
  // Solve every frame (no idle-skip) so the live Load slider and the stress colors stay current.
  core.setFracturePolicy({ idleSkip: false });
  const group = new THREE.Group();
  scene.add(group);
  const visuals = createDestructibleThreeBundle({
    core, scenario, root: group, useBatchedMesh: true, includeDebugLines: false,
  });
  coreRef = core;
  visualsRef = visuals;
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

function shoot(ndcX: number, ndcY: number) {
  const core = coreRef; if (!core) return;
  const rc = new THREE.Raycaster();
  rc.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  const d = rc.ray.direction.clone().normalize();
  // aim a small fast ball at the beam plane (z=0)
  const t = -rc.ray.origin.z / (d.z || -1);
  const hit = rc.ray.origin.clone().addScaledVector(d, Math.max(2, t));
  core.enqueueProjectile({
    position: { x: hit.x - d.x * 5, y: hit.y - d.y * 5, z: hit.z - d.z * 5 },
    velocity: { x: d.x * 22, y: d.y * 22, z: d.z * 22 },
    radius: 0.35, mass: 400, ttl: 2500,
  });
}
canvas.addEventListener('click', (e) => {
  const r = canvas.getBoundingClientRect();
  shoot(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
});

// ── Controls ──────────────────────────────────────────────────
function bindSlider(id: string, get: () => number, set: (v: number) => void, fmt: (v: number) => string, onInput?: (v: number) => void) {
  const s = document.getElementById(id) as HTMLInputElement | null;
  const d = document.getElementById(id + '-value');
  if (!s) return;
  s.value = String(get());
  if (d) d.textContent = fmt(get());
  s.addEventListener('input', () => { const v = parseFloat(s.value); set(v); if (d) d.textContent = fmt(v); onInput?.(v); });
}
bindSlider('cfg-load', () => CONFIG.loadFactor, (v) => CONFIG.loadFactor = v, (v) => v.toFixed(1) + '×',
  (v) => coreRef?.setGravity(-G * v)); // LIVE: load = effective gravity
bindSlider('cfg-strength', () => CONFIG.strength, (v) => CONFIG.strength = v, (v) => v.toFixed(2) + '×'); // reset to apply

const modeSel = document.getElementById('cfg-mode') as HTMLSelectElement | null;
modeSel?.addEventListener('change', () => { CONFIG.mode = parseInt(modeSel.value, 10) || 0; });
const cracksOnly = document.getElementById('opt-cracks-only') as HTMLInputElement | null;
cracksOnly?.addEventListener('change', () => { CONFIG.cracksOnly = !!cracksOnly.checked; });
document.getElementById('btn-reset')?.addEventListener('click', () => { void rebuild(); });

// ── Render loop ───────────────────────────────────────────────
function updateCracks() {
  const core = coreRef; if (!core) return;
  const lines = core.getSolverDebugLines(CONFIG.mode);
  const pos: number[] = [], col: number[] = [];
  let maxS = 0, cracking = 0;
  for (const ln of lines) {
    const s = Math.max(stressOf(ln.color0), stressOf(ln.color1));
    maxS = Math.max(maxS, s);
    if (s > 0.45) cracking++;
    if (CONFIG.cracksOnly && s < 0.35) continue; // hide relaxed bonds → only cracks show
    const a = decode(ln.color0), b = decode(ln.color1);
    pos.push(ln.p0.x, ln.p0.y, ln.p0.z, ln.p1.x, ln.p1.y, ln.p1.z);
    col.push(a.r, a.g, a.b, b.r, b.g, b.b);
  }
  crackGeom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  crackGeom.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  setText('stat-max-stress', (maxS * 100).toFixed(0) + '%');
  setText('stat-cracking', String(cracking));
  setText('stat-broken', String(initialBonds - core.getActiveBondsCount()));
}

const clock = new THREE.Clock();
function loop() {
  requestAnimationFrame(loop);
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();
  if (coreRef && visualsRef) {
    coreRef.step(dt);
    visualsRef.update({ debug: false, updateBVH: false, updateProjectiles: true });
    updateCracks();
  }
  renderer.render(scene, camera);
}

function onResize() {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

initScene().then(() => loop()).catch((err) => {
  console.error('Failed to initialize cracking demo:', err);
  const hint = document.querySelector('.viewport-hint');
  if (hint) hint.textContent = `Error: ${err.message}`;
});

void STRESS_MODE;
