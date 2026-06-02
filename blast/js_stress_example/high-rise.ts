/**
 * High-Rise Apartment Demolition Demo
 *
 * A mid-rise reinforced-concrete apartment building (flat-slab skeleton + frangible
 * drywall infill) destroyed by a thrown "wrecking ball". The building is loaded from
 * the SAME shared scene-pack JSON the Rust/Bevy demo uses (single source of truth),
 * including its realistic, decoupled concrete stress limits.
 *
 * Click the viewport to launch the wrecking ball. The HUD shows bonds broken over
 * time + rigid-body count — the key signals for "local damage, not glass shatter".
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import {
  buildDestructibleCore,
  loadScenePackFromUrl,
  createBondBreakRecorder,
} from 'blast-stress-solver/rapier';
import {
  createDestructibleThreeBundle,
  RapierDebugRenderer,
} from 'blast-stress-solver/three';

// Served via serve.js: /vendor/blast-stress-solver/ -> blast-stress-solver/dist/
const SCENE_URL = '/vendor/blast-stress-solver/high-rise.json';

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

function setHud(id: string, value: string) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

// ── Main ─────────────────────────────────────────────────────
let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let recorder: ReturnType<typeof createBondBreakRecorder> | null = null;
let frame = 0;
let rebuilding = false;

// User-controlled settings (top-right panel). `damage` is OFF by default: the
// default experience is the pure stress-solver response to impacts; the toggle
// flips on the per-chunk contact-damage layer (local holes) and rebuilds.
const settings = { damage: false, debug: false, mass: 2500, speed: 18 };
// Projectile radius + ttl come from the scene pack; mass + speed are slider-driven.
let projectileShape = { radius: 0.6, ttlMs: 8000 };
let packDamage: Record<string, unknown> = {};

async function initScene() {
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = 'Loading high-rise...';

  const pack = await loadScenePackFromUrl(SCENE_URL);
  const { scenario, defaults } = pack;
  projectileShape = {
    radius: (defaults.projectile as any)?.radius ?? 0.6,
    ttlMs: (defaults.projectile as any)?.ttlMs ?? 8000,
  };
  packDamage = (defaults.damage as Record<string, unknown>) ?? {};
  console.log(
    `High-rise: ${scenario.nodes.length} nodes, ${scenario.bonds.length} bonds, ` +
      `limits comp/ten/shear=${defaults.solverSettings?.compressionFatalLimit}/` +
      `${defaults.solverSettings?.tensionFatalLimit}/${defaults.solverSettings?.shearFatalLimit}`,
  );

  controls.target.set(defaults.camera.target.x, defaults.camera.target.y, defaults.camera.target.z);
  controls.update();

  const core = await buildDestructibleCore({
    scenario,
    gravity: defaults.gravity,
    materialScale: defaults.materialScale,
    // Decoupled concrete limits from the pack win over the materialScale defaults.
    solverSettings: defaults.solverSettings,
    friction: defaults.physics.friction,
    restitution: defaults.physics.restitution,
    contactForceScale: defaults.physics.contactForceScale,
    debrisCollisionMode: defaults.physics.debrisCollisionMode as any,
    // Per-chunk contact damage (health + splash) localizes wrecking-ball destruction
    // into a local hole instead of a global stress cascade. Tuned params come from the
    // scene pack; the `enabled` flag is driven by the top-right "Custom damage system"
    // toggle — OFF by default, so the default impact response is the pure stress solver.
    damage: { ...packDamage, enabled: settings.damage } as any,
    debrisCleanup: {
      mode: defaults.optimization.debrisCleanupMode as any,
      debrisTtlMs: defaults.optimization.debrisTtlMs,
      maxCollidersForDebris: defaults.optimization.maxCollidersForDebris,
    },
    smallBodyDamping: {
      mode: defaults.optimization.smallBodyDampingMode as any,
      colliderCountThreshold: 3,
      minLinearDamping: 2,
      minAngularDamping: 2,
    },
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
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: settings.debug });

  coreRef = core;
  visualsRef = visuals;
  recorder = createBondBreakRecorder(core);
  frame = 0;
  if (hint) hint.textContent = 'Click to throw the wrecking ball';
}

function shootProjectile(ndcX: number, ndcY: number) {
  const core = coreRef;
  if (!core) return;
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  const dir = raycaster.ray.direction.clone().normalize();
  core.enqueueProjectile({
    position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
    velocity: { x: dir.x * settings.speed, y: dir.y * settings.speed, z: dir.z * settings.speed },
    radius: projectileShape.radius,
    mass: settings.mass,
    ttl: projectileShape.ttlMs,
  });
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  shootProjectile(ndcX, ndcY);
});

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

// ── Top-right panel: feature toggles + settings ──────────────
document.getElementById('btn-reset')?.addEventListener('click', () => { void rebuild(); });

const optDamage = document.getElementById('opt-damage') as HTMLInputElement | null;
if (optDamage) optDamage.checked = settings.damage;
optDamage?.addEventListener('change', () => {
  settings.damage = !!optDamage.checked;
  // Per-chunk health is allocated at construction, so applying the damage feature
  // rebuilds the scene with the current settings.
  void rebuild();
});

const optDebug = document.getElementById('opt-debug') as HTMLInputElement | null;
if (optDebug) optDebug.checked = settings.debug;
optDebug?.addEventListener('change', () => {
  settings.debug = !!optDebug.checked;
  rapierDebug?.setEnabled(settings.debug);
});

const optMass = document.getElementById('opt-mass') as HTMLInputElement | null;
const optSpeed = document.getElementById('opt-speed') as HTMLInputElement | null;
function syncBallLabels() {
  setHud('val-mass', `${settings.mass} kg`);
  setHud('val-speed', `${settings.speed} m/s`);
}
optMass?.addEventListener('input', () => { settings.mass = Number(optMass.value); syncBallLabels(); });
optSpeed?.addEventListener('input', () => { settings.speed = Number(optSpeed.value); syncBallLabels(); });
// Initialize settings + labels from the control defaults.
if (optMass) settings.mass = Number(optMass.value);
if (optSpeed) settings.speed = Number(optSpeed.value);
syncBallLabels();

// ── Render loop ──────────────────────────────────────────────
const clock = new THREE.Clock();
function loop() {
  requestAnimationFrame(loop);
  stats.begin();
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();

  if (coreRef && visualsRef) {
    coreRef.step(dt);
    visualsRef.update({ debug: settings.debug, updateBVH: false, updateProjectiles: true });
    rapierDebug?.update();
    const s = recorder?.sample(frame++);
    if (s) {
      setHud('stat-bodies', String(s.rigidBodies));
      setHud('stat-bonds', String(s.activeBonds));
      setHud('stat-bonds-broken', String(s.bondsBrokenCumulative));
      setHud('stat-com', s.comHeight.toFixed(1) + ' m');
    }
  }
  renderer.render(scene, camera);
  stats.end();
}

function onResize() {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

initScene()
  .then(() => loop())
  .catch((err) => {
    console.error('Failed to initialize high-rise demo:', err);
    const hint = document.querySelector('.viewport-hint');
    if (hint) hint.textContent = `Error: ${err.message}`;
  });
