/**
 * Micro-benchmark: THREE.BatchedMesh per-frame CPU cost (`onBeforeRender`).
 *
 * A large city renders all its fragments through one BatchedMesh. Every frame (and again for the
 * shadow pass) the renderer calls `BatchedMesh.onBeforeRender`, which — with the library defaults
 * `sortObjects = true` and `perObjectFrustumCulled = true` — frustum-tests and front-to-back
 * sorts every *visible* instance on the CPU. This isolates that cost with no GPU/context.
 *
 * Run: node scripts/bench-batched-mesh.mjs [instanceCount]
 *
 * Representative result (27,539 instances, the grid-10 mini-city):
 *   sort + frustum-cull (THREE default) ~4.95 ms/call   (~9.9 ms/frame across 2 passes)
 *   frustum-cull only (no sort)         ~1.22 ms/call    <- ~4x cheaper
 *   no sort, no cull                    ~0.00 ms/call
 *   all instances hidden                ~0.06 ms/call    <- hidden instances are skipped
 *
 * Takeaways, both applied in the renderer:
 *   1. Opaque chunks don't need the front-to-back sort — disabling `sortObjects` is the single
 *      biggest render-thread win at scale (see destructible-adapter.ts).
 *   2. Cost scales with the *visible* instance count, so the intact-building proxy LOD (which hides
 *      far buildings' fragments) compounds with (1).
 */
import * as THREE from 'three';

const N = Number(process.argv[2] || 27539);
const REPS = 300;

const box = new THREE.BoxGeometry(1, 1, 1);
const mesh = new THREE.BatchedMesh(
  N,
  box.attributes.position.count,
  box.index.count,
  new THREE.MeshStandardMaterial(),
);
const gid = mesh.addGeometry(box);
const m = new THREE.Matrix4();
for (let i = 0; i < N; i++) {
  const id = mesh.addInstance(gid);
  m.makeTranslation((Math.random() - 0.5) * 300, Math.random() * 60, (Math.random() - 0.5) * 300);
  mesh.setMatrixAt(id, m);
}

const scene = new THREE.Scene();
scene.add(mesh);
const cam = new THREE.PerspectiveCamera(55, 1.6, 0.1, 1200);
cam.position.set(60, 40, 80);
cam.lookAt(0, 8, 0);
scene.updateMatrixWorld(true);
cam.updateMatrixWorld(true);
cam.updateProjectionMatrix();
const rendererStub = { coordinateSystem: THREE.WebGLCoordinateSystem };

function bench(label) {
  for (let k = 0; k < 20; k++) mesh.onBeforeRender(rendererStub, scene, cam, mesh.geometry, mesh.material, null);
  const t = performance.now();
  for (let k = 0; k < REPS; k++) mesh.onBeforeRender(rendererStub, scene, cam, mesh.geometry, mesh.material, null);
  const ms = (performance.now() - t) / REPS;
  console.log(`${label.padEnd(36)} ${ms.toFixed(3)} ms/call`);
}

console.log(`BatchedMesh.onBeforeRender — ${N} instances\n`);
mesh.sortObjects = true;  mesh.perObjectFrustumCulled = true;  bench('sort + frustum-cull (THREE default)');
mesh.sortObjects = false; mesh.perObjectFrustumCulled = true;  bench('frustum-cull only (no sort)');
mesh.sortObjects = false; mesh.perObjectFrustumCulled = false; bench('no sort, no cull');
for (let i = 0; i < N; i++) mesh.setVisibleAt(i, false);
mesh.sortObjects = true;  mesh.perObjectFrustumCulled = true;  bench('all instances hidden (sort+cull)');
