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
let showDebug = false;
let projectile = { radius: 0.6, mass: 2500, speed: 18, ttlMs: 8000 };

async function initScene() {
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = 'Loading high-rise...';

  const pack = await loadScenePackFromUrl(SCENE_URL);
  const { scenario, defaults } = pack;
  projectile = defaults.projectile;
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
    // Contact damage (per-chunk health + splash) localizes wrecking-ball destruction:
    // it blows out a local hole instead of cascading like glass. From the scene pack.
    damage: (defaults.damage as any) ?? { enabled: false },
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
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: showDebug });

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
    velocity: { x: dir.x * projectile.speed, y: dir.y * projectile.speed, z: dir.z * projectile.speed },
    radius: projectile.radius,
    mass: projectile.mass,
    ttl: projectile.ttlMs,
  });
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  shootProjectile(ndcX, ndcY);
});

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
});

// ── Render loop ──────────────────────────────────────────────
const clock = new THREE.Clock();
function loop() {
  requestAnimationFrame(loop);
  stats.begin();
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();

  if (coreRef && visualsRef) {
    coreRef.step(dt);
    visualsRef.update({ debug: showDebug, updateBVH: false, updateProjectiles: true });
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
