#!/usr/bin/env node
/**
 * Cross-check a pack's bonds against NvBlast's own contact-based bond
 * generator (ExtAuthoring `bondsFromPrefractured`, EXACT mode) via WASM.
 *
 *   node structures/verify-autobonds.mjs <pack.json> [--tolerance 0.02]
 *
 * Adapted from scripts/verify-bonds-against-autobonding.mjs, which reads a
 * pack's `nodeMeshes`. These packs do not carry meshes — every chunk's shape IS
 * its collider — so geometry is rebuilt from `nodeColliders` instead: a cuboid
 * from its half-extents, a hull from its points.
 *
 * The point is not to replace the pack's own bonding. Contact here is computed
 * in closed form — a shared Voronoi edge, a shared cell boundary, a
 * separating-axis test with a shadow overlap — which is fast and deterministic
 * but only trustworthy if something independent agrees with it. This is that
 * something.
 *
 * How to read the result, per the upstream script's own findings:
 *
 *   MISSING   the generator found a contact the pack does not bond. Either a
 *             real omitted load path, or a patch under the pack's minimum bond
 *             area, which are dropped on purpose.
 *   GAPPED    the pack bonds a pair that is genuinely apart. Always a bug.
 *   TOUCHING  the pack bonds a flush pair the generator did not find. EXACT
 *             mode searches for common SURFACE and does not reliably find
 *             exactly-coplanar faces between independently triangulated
 *             meshes — which is most bearings in a building. Upstream measured
 *             all 526 of these on the city pack at a gap of 0.000000 m.
 *   AREA      both agree there is a contact and disagree on how big.
 *
 * A clean result is zero GAPPED and few MISSING.
 */
import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { generateAutoBondsFromChunks } from '../dist/three.js';

const packPath = process.argv[2];
if (!packPath) {
  console.error('usage: node structures/verify-autobonds.mjs <pack.json> [--tolerance F]');
  process.exit(2);
}
const tolIndex = process.argv.indexOf('--tolerance');
const TOL = tolIndex > 0 ? Number(process.argv[tolIndex + 1]) : 0.02;

const pack = JSON.parse(await readFile(packPath, 'utf8'));
const sc = pack.scenario;
const nodes = sc.nodes;
const types = sc.nodeTypes ?? nodes.map(() => 'node');

/** A chunk's real surface, in world space. */
const chunks = nodes.map((node, i) => {
  const c = node.centroid;
  const col = sc.nodeColliders[i];
  let geometry;
  if (col.kind === 'convex_hull') {
    const pts = [];
    for (let k = 0; k + 2 < col.points.length; k += 3) {
      pts.push(new THREE.Vector3(col.points[k], col.points[k + 1], col.points[k + 2]));
    }
    geometry = new ConvexGeometry(pts);
  } else {
    const h = col.halfExtents;
    geometry = new THREE.BoxGeometry(h.x * 2, h.y * 2, h.z * 2);
  }
  geometry.translate(c.x, c.y, c.z);
  return { geometry };
});

process.stderr.write(`NvBlast EXACT auto-bonding over ${chunks.length} chunks...\n`);
const auto = await generateAutoBondsFromChunks(chunks, { mode: 'exact' });
if (!auto) {
  console.error('auto-bonding failed (is the WASM runtime built?)');
  process.exit(1);
}

const keyOf = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);
const sum = (list, get) => {
  const m = new Map();
  for (const b of list) {
    const k = keyOf(b.node0, b.node1);
    m.set(k, (m.get(k) ?? 0) + get(b));
  }
  return m;
};
const autoMap = sum(auto, (b) => b.area);
const packMap = sum(sc.bonds, (b) => b.area);

/** World AABB, for re-testing a pair the generator did not find. */
const boxes = nodes.map((node, i) => {
  const c = node.centroid, col = sc.nodeColliders[i];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const put = (p) => { for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); } };
  if (col.kind === 'convex_hull') {
    for (let k = 0; k + 2 < col.points.length; k += 3) {
      put([col.points[k] + c.x, col.points[k + 1] + c.y, col.points[k + 2] + c.z]);
    }
  } else {
    const h = col.halfExtents;
    for (const dx of [-1, 1]) for (const dy of [-1, 1]) for (const dz of [-1, 1]) {
      put([c.x + dx * h.x, c.y + dy * h.y, c.z + dz * h.z]);
    }
  }
  return [lo, hi];
});
const separation = (a, b) => {
  const [l0, h0] = boxes[a], [l1, h1] = boxes[b];
  let gap = 0;
  for (let d = 0; d < 3; d++) gap = Math.max(gap, Math.max(l0[d] - h1[d], l1[d] - h0[d]));
  return gap;
};

let matched = 0, gapped = 0, touching = 0, areaOff = 0, missing = 0;
let relSum = 0, missingArea = 0, worstRel = 0, worstPair = null;
const gappedPairs = [];
let subMm = 0;
// Which way the disagreement runs, and for which kind of joint. An overstated
// area is the dangerous direction: stress is force over area, so a bond claimed
// bigger than it is reports less stress than it carries and does not break when
// it should.
let over = 0, under = 0;
const byClass = new Map();

for (const [k, packArea] of packMap) {
  const autoArea = autoMap.get(k);
  if (autoArea === undefined) {
    const [i, j] = k.split(':').map(Number);
    const gap = separation(i, j);
    // A bond is only wrong if it spans a REAL gap. The builder treats surfaces
    // within a millimetre as touching, so anything under that is flushness the
    // generator's AABB test resolves differently, not a bond across empty space.
    if (gap > 1e-3) { gapped += 1; gappedPairs.push([k, gap]); }
    else if (gap > 1e-6) subMm += 1;
    else touching += 1;
    continue;
  }
  matched += 1;
  const rel = Math.abs(packArea - autoArea) / Math.max(autoArea, 1e-9);
  relSum += rel;
  const [ii, jj] = k.split(':').map(Number);
  const cls = [types[ii], types[jj]].sort().join('~');
  const acc = byClass.get(cls) ?? { n: 0, over: 0, ratios: [] };
  acc.n += 1;
  // Ratios are collected and reported as a MEDIAN. A mean is useless here: a
  // handful of pairs where the generator reports an area of ~0 send it to
  // thousands and hide what the bulk of the bonds are doing.
  acc.ratios.push(packArea / Math.max(autoArea, 1e-9));
  if (packArea > autoArea * (1 + TOL)) { acc.over += 1; over += 1; }
  else if (packArea < autoArea * (1 - TOL)) under += 1;
  byClass.set(cls, acc);
  if (rel > TOL) {
    areaOff += 1;
    if (rel > worstRel) { worstRel = rel; worstPair = [k, packArea, autoArea]; }
  }
}
for (const [k, autoArea] of autoMap) {
  if (!packMap.has(k)) { missing += 1; missingArea += autoArea; }
}

const pct = (n, d) => (d ? (100 * n / d).toFixed(1) : '0.0');
console.log(`\n${pack.title ?? packPath}`);
console.log(`  pack bonds ${packMap.size}   auto bonds ${autoMap.size}`);
console.log(`  matched  ${matched} (${pct(matched, packMap.size)}% of pack)   mean area error ` +
  `${matched ? (100 * relSum / matched).toFixed(1) : '0.0'}%   over ${TOL * 100}%: ${areaOff}`);
if (worstPair) {
  console.log(`    worst: pair ${worstPair[0]} pack ${worstPair[1].toFixed(4)} m^2 vs auto ` +
    `${worstPair[2].toFixed(4)} (${(worstRel * 100).toFixed(0)}% off)`);
}
console.log(`  TOUCHING ${touching} — flush pairs EXACT mode did not find (expected; not a defect)`);
console.log(`  MISSING  ${missing} contacts the generator found and the pack does not bond ` +
  `(${missingArea.toFixed(2)} m^2 total)`);
console.log(`  direction: ${over} overstated, ${under} understated (of ${matched} matched)`);
const worstClasses = [...byClass.entries()]
  .filter(([, a]) => a.n >= 5)
  .map(([c, a]) => {
    const r = a.ratios.slice().sort((x, y) => x - y);
    return [c, { ...a, median: r[r.length >> 1], p95: r[Math.floor(r.length * 0.95)] }];
  })
  .sort((a, b) => b[1].median - a[1].median)
  .slice(0, 6);
for (const [cls, a] of worstClasses) {
  console.log(`    ${cls.padEnd(24)} n=${String(a.n).padStart(5)} median pack/auto ` +
    `${a.median.toFixed(2)}x  p95 ${a.p95.toFixed(2)}x  over-tol ${a.over} (${pct(a.over, a.n)}%)`);
}
if (subMm) console.log(`  (${subMm} pair(s) separated by under a millimetre — within the builder's touch tolerance)`);
console.log(`  GAPPED   ${gapped} bonds across a real gap` + (gapped ? ' <-- BUG' : ''));
for (const [k, gap] of gappedPairs.slice(0, 8)) {
  const [i, j] = k.split(':').map(Number);
  console.log(`    ${types[i]}~${types[j]} pair ${k} separated by ${gap.toFixed(4)} m`);
}
process.exit(gapped > 0 ? 1 : 0);
