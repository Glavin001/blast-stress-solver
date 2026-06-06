/**
 * Collision-LOD benchmark (headless): eager vs lazy(binary) vs lazy(hierarchical).
 *
 * Reports, per scene:
 *   - idle rapierStep  — Rapier world.step() broadphase/solve cost while nothing is being destroyed
 *   - post-hit rapierStep — same, averaged over a window after a single localized impact
 *   - peak active fragments — most colliders the lazy modes had enabled at once (locality metric)
 *
 * Two scenes exercise the two wins:
 *   A. Big city (many short towers) → cross-building locality (idle + un-hit buildings stay dormant)
 *   B. Tall high-rise (one structure) → intra-building locality (a localized hit only descends the
 *      struck region of the LOD tree). Run rigid (1e10) so it's idle-stable and damage stays local.
 *
 * Usage: node scripts/lod-bench.mjs
 */
import { buildDestructibleCore, buildSpatialCollisionTree } from '../dist/rapier.js';
import * as scen from '../dist/scenarios.js';

const mean = (a) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);
const clone = (s) => ({
  nodes: s.nodes.map((n) => ({ centroid: { ...n.centroid }, mass: n.mass, volume: n.volume })),
  bonds: s.bonds.map((b) => ({ node0: b.node0, node1: b.node1, centroid: { ...b.centroid }, normal: { ...b.normal }, area: b.area })),
  parameters: {},
});

async function cityScenario(grid) {
  const t = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
  const W = 8, STREET = 9, cell = W + STREET, half = ((grid - 1) * cell) / 2;
  const nodes = [], bonds = [];
  let base = 0;
  for (let r = 0; r < grid; r++) for (let c = 0; c < grid; c++) {
    const ox = -half + c * cell, oz = -half + r * cell;
    for (const n of t.nodes) nodes.push({ centroid: { x: n.centroid.x + ox, y: n.centroid.y, z: n.centroid.z + oz }, mass: n.mass, volume: n.volume });
    for (const b of t.bonds) bonds.push({ node0: b.node0 + base, node1: b.node1 + base, centroid: { x: b.centroid.x + ox, y: b.centroid.y, z: b.centroid.z + oz }, normal: { ...b.normal }, area: b.area });
    base += t.nodes.length;
  }
  return { nodes, bonds, parameters: {}, _hit: { x: -half, y: 4, z: -30, vz: 60, mass: 3000, r: 0.8 } };
}
async function highRiseScenario() {
  const b = await (scen.buildHighRiseScenarioAsync ?? scen.buildHighRiseScenario)({});
  const s = b.nodes ? b : (b.scenario ?? b);
  let ymin = Infinity, ymax = -Infinity;
  for (const n of s.nodes) { ymin = Math.min(ymin, n.centroid.y); ymax = Math.max(ymax, n.centroid.y); }
  s._hit = { x: 0, y: ymin + (ymax - ymin) * 0.15, z: -40, vz: 95, mass: 12000, r: 0.9 };
  return s;
}

async function measure(baseScenario, mode, MAT, postFrames) {
  const s = clone(baseScenario); s._hit = baseScenario._hit;
  if (mode === 'hier') s.collisionTree = buildSpatialCollisionTree(s, { leafMaxFragments: 24 });
  let sample = null;
  const core = await buildDestructibleCore({ scenario: s, gravity: -9.81, materialScale: MAT, friction: 0.25, restitution: 0, contactForceScale: 30, debrisCollisionMode: 'all', damage: { enabled: false }, lazyIntactColliders: mode !== 'eager' });
  core.setProfiler?.({ enabled: true, onSample: (x) => { sample = x; } });
  const dt = 1 / 60;
  for (let i = 0; i < 40; i++) core.step(dt);
  const idle = []; for (let i = 0; i < 40; i++) { core.step(dt); idle.push(sample?.rapierStepMs ?? 0); }
  const h = s._hit; core.enqueueProjectile({ position: { x: h.x, y: h.y, z: h.z }, velocity: { x: 0, y: 0, z: h.vz }, radius: h.r, mass: h.mass, ttl: 4000 });
  const act = []; let peak = 0;
  for (let i = 0; i < postFrames; i++) { core.step(dt); act.push(sample?.rapierStepMs ?? 0); const lz = core.getLazyColliderStats?.(); if (lz) peak = Math.max(peak, lz.activeLeafFragments); }
  const bodies = core.getRigidBodyCount();
  core.dispose?.();
  return { idle: mean(idle), active: mean(act), peak, bodies, frags: s.nodes.length };
}

async function run(name, baseScenario, MAT, postFrames) {
  console.log(`\n### ${name}  (${baseScenario.nodes.length} fragments, materialScale ${MAT.toExponential(0)})`);
  const e = await measure(baseScenario, 'eager', MAT, postFrames);
  const b = await measure(baseScenario, 'binary', MAT, postFrames);
  const h = await measure(baseScenario, 'hier', MAT, postFrames);
  console.log('mode        | idle rapierStep | post-hit rapierStep | peak active frags');
  console.log('eager       | %s ms | %s ms | %d (all)', e.idle.toFixed(3).padStart(7), e.active.toFixed(3).padStart(7), e.frags);
  console.log('lazy binary | %s ms | %s ms | %d', b.idle.toFixed(3).padStart(7), b.active.toFixed(3).padStart(7), b.peak);
  console.log('lazy hier   | %s ms | %s ms | %d', h.idle.toFixed(3).padStart(7), h.active.toFixed(3).padStart(7), h.peak);
  console.log('  idle eager→lazy: %sx | post-hit eager→hier: %sx | binary→hier: %sx',
    (e.idle / Math.max(1e-6, h.idle)).toFixed(1), (e.active / Math.max(1e-6, h.active)).toFixed(2), (b.active / Math.max(1e-6, h.active)).toFixed(2));
}

await run('Big city 8×8 (64 towers), single wrecking-ball', await cityScenario(8), 5e6, 150);
await run('High-rise (1 tall tower), localized low hit, rigid', await highRiseScenario(), 1e10, 60);
