/**
 * Cross-check a ScenePack's authored bonds against NvBlast's own contact-based
 * bond generator (ExtAuthoring `bondsFromPrefractured`, EXACT mode) via WASM.
 *
 *   node scripts/verify-bonds-against-autobonding.mjs <pack.json> [--tolerance 0.02]
 *
 * Generators that compute contact in closed form (export-fractured-city.mjs
 * derives shard~shard area from the shared Voronoi edge and shard~frame area
 * from a polygon-rectangle intersection) are fast and deterministic, but the
 * arithmetic is only trustworthy if something independent agrees with it. This
 * runs the library's own triangle-based generator over the same geometry and
 * reports where the two disagree.
 *
 * Three failure kinds are worth different reactions:
 *   MISSING   the generator found a contact the pack does not bond. Either a
 *             real omitted load path, or a patch below the pack's MIN_BOND_AREA
 *             floor (those are dropped on purpose — check the area).
 *   EXTRA     the pack bonds a pair the generator did not. Only a bug when the
 *             two bodies are actually apart, so every extra pair is re-tested
 *             for a real gap and reported split:
 *               GAPPED   — separated. A bond across empty space; fix the pack.
 *               TOUCHING — flush. EXACT mode searches for *common surface* and
 *                          does not reliably find exactly-coplanar faces
 *                          between independently triangulated meshes (a panel
 *                          bearing on a slab ledge, say). Measured on the
 *                          fractured-city pack, all 526 extras were flush at a
 *                          gap of 0.000000 m. Not a defect. AVERAGE mode is
 *                          worse here, not better: it covered 45% of authored
 *                          bonds against EXACT's 82%.
 *   AREA      both agree a contact exists but disagree on how big it is.
 *
 * So a clean result is zero GAPPED and zero MISSING, with AREA agreement on the
 * pairs both find. That is what caught two real bugs in export-fractured-city:
 * columns bonded to their footings across a 1 mm gap, and columns bonded to
 * only one of the several slab cells they straddle.
 *
 * Requires the WASM runtime (blast/js_stress_example: npm run build).
 */
import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import { generateAutoBondsFromChunks } from '../dist/three.js';

const packPath = process.argv[2];
if (!packPath) {
  console.error('usage: node scripts/verify-bonds-against-autobonding.mjs <pack.json> [--tolerance F]');
  process.exit(2);
}
const tolIndex = process.argv.indexOf('--tolerance');
const TOL = tolIndex > 0 ? Number(process.argv[tolIndex + 1]) : 0.02;

const pack = JSON.parse(await readFile(packPath, 'utf8'));
const sc = pack.scenario;
const nodes = sc.nodes;
const types = sc.nodeTypes ?? nodes.map(() => 'node');
const sizes = sc.nodeSizes;
const meshes = pack.nodeMeshes ?? sc.nodeMeshes ?? nodes.map(() => null);

// Rebuild each node's world-space surface. Shards carry a real mesh (stored
// centroid-relative, matching the ScenePack convention); frame members are
// boxes described by nodeSizes.
const chunks = nodes.map((node, i) => {
  const c = node.centroid;
  let geometry;
  const mesh = meshes[i];
  if (mesh && mesh.positions?.length) {
    geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(Float32Array.from(mesh.positions), 3));
    if (mesh.indices?.length) geometry.setIndex(Array.from(mesh.indices));
    geometry.translate(c.x, c.y, c.z);
  } else {
    const s = sizes[i];
    geometry = new THREE.BoxGeometry(s.x, s.y, s.z);
    geometry.translate(c.x, c.y, c.z);
  }
  return { geometry };
});

process.stderr.write(`running NvBlast EXACT auto-bonding over ${chunks.length} chunks...\n`);
const auto = await generateAutoBondsFromChunks(chunks, { mode: 'exact' });
if (!auto) {
  console.error('auto-bonding failed (is the WASM runtime built?)');
  process.exit(1);
}

const keyOf = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);
const autoMap = new Map();
for (const b of auto) {
  const k = keyOf(b.node0, b.node1);
  autoMap.set(k, (autoMap.get(k) ?? 0) + b.area); // sum multi-patch contacts
}
const packMap = new Map();
for (const b of sc.bonds) {
  const k = keyOf(b.node0, b.node1);
  packMap.set(k, (packMap.get(k) ?? 0) + b.area);
}

const cls = (k) => {
  const [i, j] = k.split(':').map(Number);
  return [types[i], types[j]].sort().join('~');
};
// World-space AABB per node, used to re-test EXTRA pairs for a real gap.
function nodeAabb(i) {
  const c = nodes[i].centroid, m = meshes[i];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const put = (p) => { for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); } };
  if (m && m.positions?.length) {
    for (let k = 0; k < m.positions.length; k += 3) put([m.positions[k] + c.x, m.positions[k + 1] + c.y, m.positions[k + 2] + c.z]);
  } else {
    const s = sizes[i];
    for (const dx of [-1, 1]) for (const dy of [-1, 1]) for (const dz of [-1, 1]) put([c.x + dx * s.x / 2, c.y + dy * s.y / 2, c.z + dz * s.z / 2]);
  }
  return [lo, hi];
}
const GAP_EPS = 1e-6;
function separation(a, b) {
  const [l0, h0] = nodeAabb(a), [l1, h1] = nodeAabb(b);
  let gap = 0;
  for (let d = 0; d < 3; d++) gap = Math.max(gap, Math.max(l0[d] - h1[d], l1[d] - h0[d]));
  return gap;
}

const stats = new Map();
const bump = (k, field, value = 1) => {
  const c = cls(k);
  if (!stats.has(c)) stats.set(c, { matched: 0, missing: 0, gapped: 0, touching: 0, areaOff: 0, relSum: 0, missingArea: 0 });
  stats.get(c)[field] += value;
};

for (const [k, packArea] of packMap) {
  const autoArea = autoMap.get(k);
  if (autoArea === undefined) {
    const [i, j] = k.split(':').map(Number);
    bump(k, separation(i, j) > GAP_EPS ? 'gapped' : 'touching');
    continue;
  }
  const rel = Math.abs(autoArea - packArea) / Math.max(autoArea, packArea);
  bump(k, 'matched');
  bump(k, 'relSum', rel);
  if (rel > TOL) bump(k, 'areaOff');
}
for (const [k, autoArea] of autoMap) {
  if (!packMap.has(k)) { bump(k, 'missing'); bump(k, 'missingArea', autoArea); }
}

console.log(`pack bonds=${packMap.size}  autoBonds=${autoMap.size}  tolerance=${(TOL * 100).toFixed(0)}%\n`);
console.log(`${'class'.padEnd(24)} ${'matched'.padStart(8)} ${'meanAreaErr'.padStart(12)} ${'over tol'.padStart(9)} ${'missing'.padStart(8)} ${'gapped'.padStart(7)} ${'touching'.padStart(9)}`);
let gapped = 0, missing = 0, off = 0, touching = 0;
for (const [c, s] of [...stats].sort((a, b) => b[1].matched - a[1].matched)) {
  const mean = s.matched ? s.relSum / s.matched : 0;
  console.log(`${c.padEnd(24)} ${String(s.matched).padStart(8)} ${(mean * 100).toFixed(2).padStart(11)}% ${String(s.areaOff).padStart(9)} ${String(s.missing).padStart(8)} ${String(s.gapped).padStart(7)} ${String(s.touching).padStart(9)}`);
  gapped += s.gapped; missing += s.missing; off += s.areaOff; touching += s.touching;
}
console.log(`\nGAPPED   (pack bonds across empty space — real bug): ${gapped}`);
console.log(`TOUCHING (flush, EXACT mode just did not report):    ${touching}`);
console.log(`MISSING  (contacts the pack does not bond):          ${missing}`);
console.log(`AREA disagreements over ${(TOL * 100).toFixed(0)}%:                        ${off}`);
process.exit(gapped > 0 ? 1 : 0);
