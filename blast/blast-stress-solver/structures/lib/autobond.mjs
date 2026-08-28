/**
 * Re-measure a pack's bond areas with NvBlast's own contact-based bond
 * generator (ExtAuthoring `bondsFromPrefractured`, EXACT mode) via WASM.
 *
 * The builder computes contact in closed form — a shared Voronoi edge, a shared
 * cell boundary, a separating-axis test with a shadow overlap. Checked against
 * the generator, the typical bond agrees exactly (median ratio 1.00x across
 * every joint class), but there is a tail that does not: the shadow overlap
 * projects a WHOLE shard onto the contact plane, so where two pieces touch over
 * part of their outlines it reports the overlap of their silhouettes instead of
 * the contact. On the tower that put 22% of bonds over a 2% tolerance, with a
 * 95th percentile of 2.3x on beam-to-slab and 3.4x on balcony-to-slab, always
 * in the same direction — too big.
 *
 * Too big is the direction that matters. Stress is force over area, so a bond
 * claimed larger than it is reports less stress than it carries, and does not
 * break when it should.
 *
 * So the generator's area is taken wherever it finds the same contact. What it
 * is NOT used for is deciding which pairs are bonded, because EXACT mode
 * searches for common SURFACE and does not reliably find exactly-coplanar faces
 * between independently triangulated meshes — which is most bearings in a
 * building. It misses 6-9% of the pack's bonds that way, and they are load
 * paths: a column on a slab, a parapet on a deck. Those are kept, and checked
 * for a real gap instead.
 */
import * as THREE from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { generateAutoBondsFromChunks } from '../../dist/three.js';
import { bondMaterialName, MATERIAL_INDEX } from './materials.mjs';
import { colliderOf } from './colliders.mjs';

/** Two surfaces this close are touching, matching the builder's own tolerance. */
const TOUCH_EPS = 1e-3;

const keyOf = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);

function chunkGeometry(pack, i) {
  const c = pack.scenario.nodes[i].centroid;
  const col = colliderOf(pack.scenario, i);
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
  return geometry;
}

function aabb(pack, i) {
  const c = pack.scenario.nodes[i].centroid;
  const col = colliderOf(pack.scenario, i);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const put = (p) => { for (let d = 0; d < 3; d += 1) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); } };
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
}

/**
 * Rewrite `pack.scenario.bonds` using the generator's measurements.
 *
 * @returns a report of what changed, for the build log.
 */
export async function applyAutoBonds(pack, { minArea = 8e-3, batch = 9000 } = {}) {
  const s = pack.scenario;
  const boxes0 = s.nodes.map((_, i) => aabb(pack, i));
  const groups = s.nodeGroups ?? [];

  // Bond each logical unit, then bond the units to each other.
  //
  // Running the generator over a whole scene at once does not scale: the WASM
  // build is 32-bit and its heap tops out at 2 GB, which 37,200 hulls spread
  // over 240 m of city ran straight into. It is also the wrong shape of work.
  // A bond only ever forms between chunks that touch, so almost all of them
  // are INSIDE something -- a wall panel, a house -- and only a thin skin of
  // chunks per unit can reach anything outside it.
  //
  // So: one pass per unit over its own chunks, then a single pass over just
  // the chunks that lie within reach of a different unit. The second set is a
  // small fraction of the scene, which is what makes the whole thing cheap.
  // Two chunks can only bond if they touch, so "near a different unit" is a
  // matter of centimetres. Sizing this off the largest chunk in the scene (the
  // terrain, at 13 m) instead swept nearly every chunk into the skin and gave
  // back the whole-scene pass this exists to avoid.
  const margin = 0.5;

  const autoArea = new Map();
  const autoBond = new Map();
  const report = { remeasured: 0, shrunk: 0, grew: 0, kept: 0, dropped: 0, added: 0,
    areaBefore: 0, areaAfter: 0, units: 0, interfaceChunks: 0 };

  const measure = async (members, label) => {
    if (members.length < 2) return;
    const auto = await generateAutoBondsFromChunks(
      members.map((i) => ({ geometry: chunkGeometry(pack, i) })), { mode: 'exact' },
    );
    if (!auto) {
      throw new Error(`auto-bonding returned nothing for ${label} ` +
        `(${members.length} chunks) — is the WASM runtime built?`);
    }
    for (const bond of auto) {
      const n0 = members[bond.node0], n1 = members[bond.node1];
      const k = keyOf(n0, n1);
      if (autoBond.has(k)) continue;
      autoArea.set(k, bond.area);
      autoBond.set(k, { ...bond, node0: n0, node1: n1 });
    }
  };

  // Pass 1: inside each unit.
  const byGroup = new Map();
  for (let i = 0; i < s.nodes.length; i += 1) {
    const g = groups[i] ?? 'scene';
    const list = byGroup.get(g) ?? [];
    list.push(i);
    byGroup.set(g, list);
  }
  for (const [name, members] of byGroup) {
    report.units += 1;
    await measure(members, `unit ${name}`);
  }

  // Pass 2: the skin. A chunk is on it if anything from another unit comes
  // within `margin` of it -- so every cross-unit contact has both sides here.
  const skin = [];
  for (let i = 0; i < s.nodes.length; i += 1) {
    const [li, hi_] = boxes0[i];
    const gi = groups[i] ?? 'scene';
    let touching = false;
    for (let j = 0; j < s.nodes.length && !touching; j += 1) {
      if ((groups[j] ?? 'scene') === gi) continue;
      const [lj, hj] = boxes0[j];
      touching = hj[0] >= li[0] - margin && lj[0] <= hi_[0] + margin
        && hj[1] >= li[1] - margin && lj[1] <= hi_[1] + margin
        && hj[2] >= li[2] - margin && lj[2] <= hi_[2] + margin;
    }
    if (touching) skin.push(i);
  }
  report.interfaceChunks = skin.length;
  // The skin can still be large on a scene whose units all abut, so it goes
  // through the same size cap the units do.
  const CAP = 9000;
  if (skin.length <= CAP) {
    await measure(skin, 'interfaces');
  } else {
    const cells = Math.ceil(skin.length / CAP);
    const sorted = skin.slice().sort((a, c) => boxes0[a][0][0] - boxes0[c][0][0]);
    const per = Math.ceil(sorted.length / cells);
    for (let c = 0; c < cells; c += 1) {
      const lo = c * per;
      const slab = sorted.slice(lo, lo + per);
      if (!slab.length) continue;
      // Overlap by the margin so a contact on a slab edge is seen whole.
      const x0 = boxes0[slab[0]][0][0] - margin;
      const x1 = boxes0[slab[slab.length - 1]][1][0] + margin;
      await measure(sorted.filter((i) => boxes0[i][1][0] >= x0 && boxes0[i][0][0] <= x1),
        `interfaces ${c + 1}/${cells}`);
    }
  }

  // The generator can report several contact patches for one pair, and summing
  // them can exceed what the pieces could possibly share -- patches that
  // overlap, or a coincident face counted from both sides. Clamp to the same
  // physical bound verify.mjs enforces: the largest planar cross-section of
  // the smaller piece.
  const crossSection = (i) => {
    const z = s.nodeSizes[i];
    return Math.max(z.x * Math.hypot(z.y, z.z), z.y * Math.hypot(z.x, z.z), z.z * Math.hypot(z.x, z.y));
  };
  const bound = (a, b) => Math.min(crossSection(a), crossSection(b));

  const boxes = boxes0;
  const gap = (a, b) => {
    let g = 0;
    for (let d = 0; d < 3; d += 1) {
      g = Math.max(g, Math.max(boxes[a][0][d] - boxes[b][1][d], boxes[b][0][d] - boxes[a][1][d]));
    }
    return g;
  };

  const seen = new Set();
  const out = [];

  for (const bond of s.bonds) {
    const k = keyOf(bond.node0, bond.node1);
    seen.add(k);
    const measured = autoArea.get(k);
    report.areaBefore += bond.area;
    if (measured === undefined) {
      // The generator did not find this contact. Keep it if the two really are
      // flush — that is the coplanar case it cannot see — and drop it if they
      // are apart, which would be a bond across a gap.
      if (gap(bond.node0, bond.node1) > TOUCH_EPS) { report.dropped += 1; continue; }
      report.kept += 1;
      report.areaAfter += bond.area;
      out.push(bond);
      continue;
    }
    const area = Math.min(measured, bound(bond.node0, bond.node1));
    if (area < minArea) { report.dropped += 1; continue; }
    if (area < bond.area) report.shrunk += 1; else if (area > bond.area) report.grew += 1;
    report.remeasured += 1;
    report.areaAfter += area;
    out.push({ ...bond, area: round(area) });
  }

  // Contacts the generator found that the closed-form pass missed.
  for (const [k, raw] of autoArea) {
    if (seen.has(k)) continue;
    const b = autoBond.get(k);
    const area = Math.min(raw, bound(b.node0, b.node1));
    if (area < minArea) continue;
    const m = bondMaterialName(s.nodeMaterials[b.node0], s.nodeMaterials[b.node1]);
    const n0 = s.nodes[b.node0].centroid, n1 = s.nodes[b.node1].centroid;
    const len = Math.hypot(b.normal.x, b.normal.y, b.normal.z) || 1;
    out.push({
      node0: b.node0,
      node1: b.node1,
      centroid: { x: round(b.centroid.x), y: round(b.centroid.y), z: round(b.centroid.z) },
      normal: { x: round(b.normal.x / len), y: round(b.normal.y / len), z: round(b.normal.z / len) },
      area: round(area),
      m: MATERIAL_INDEX[m],
    });
    report.added += 1;
    report.areaAfter += area;
    void n0; void n1;
  }

  s.bonds = out;
  return report;
}

const round = (n) => Math.round(n * 1e5) / 1e5;
