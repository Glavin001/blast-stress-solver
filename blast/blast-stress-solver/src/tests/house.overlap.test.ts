/**
 * Geometry sanity check for the house: no two chunks may *overlap* in 3D space.
 *
 * Touching (shared faces) is fine and expected — it's how bonds form. But interpenetration
 * is a bug: while the structure is one rigid body it's invisible, but the instant a piece
 * fractures off, the physics engine sees two solids occupying the same space and violently
 * shoves them apart ("pieces thrust out in weird ways"). This catches that at build time.
 *
 * Exact for axis-aligned boxes (the structural pieces); the sloped roof tiles tile along a
 * world axis so the world-axis SAT used here doesn't false-positive them.
 */
import { describe, it, expect } from 'vitest';
import { buildHouseScenario } from '../scenarios/houseScenario';
import { projectExtentsOnAxisWorld } from '../three/fracture';

type Ax = { x: number; y: number; z: number };

/** The 3 unique face-normal axes of a (possibly rotated) box geometry — the SAT axes that
 *  make this exact for oriented boxes, so tilted roof tiles aren't false-flagged. */
function boxAxes(geom: any): Ax[] {
  const nrm = geom.getAttribute('normal');
  const axes: Ax[] = [];
  for (let i = 0; i < nrm.count && axes.length < 3; i++) {
    let x = nrm.getX(i), y = nrm.getY(i), z = nrm.getZ(i);
    const len = Math.hypot(x, y, z);
    if (len < 1e-6) continue;
    x /= len; y /= len; z /= len;
    if (!axes.some((a) => Math.abs(a.x * x + a.y * y + a.z * z) > 0.999)) axes.push({ x, y, z });
  }
  return axes;
}

describe('house geometry has no overlapping chunks', () => {
  const scenario = buildHouseScenario();
  const nodes = scenario.nodes;
  const params = scenario.parameters as any;
  const geoms = params.fragmentGeometries as any[];
  const sizes = params.fragmentSizes as { x: number; y: number; z: number }[];
  const types = params.house.fragmentTypes as string[];
  const n = nodes.length;

  // Penetration deeper than this counts as overlap (small intentional contact "bites" for
  // bonding are below it). Touching ⇒ ~0.
  const TOL = 0.03;

  const overlaps: Array<{ i: number; j: number; depth: number; pair: string; at: any }> = [];

  for (let i = 0; i < n; i++) {
    const ci = nodes[i].centroid;
    const si = sizes[i];
    for (let j = i + 1; j < n; j++) {
      const cj = nodes[j].centroid;
      const sj = sizes[j];
      // Broadphase: AABB reject.
      if (
        Math.abs(ci.x - cj.x) >= (si.x + sj.x) * 0.5 - TOL ||
        Math.abs(ci.y - cj.y) >= (si.y + sj.y) * 0.5 - TOL ||
        Math.abs(ci.z - cj.z) >= (si.z + sj.z) * 0.5 - TOL
      ) {
        continue;
      }
      // Exact OBB separating-axis test over both boxes' face normals (a separating axis
      // ⇒ min overlap ≤ 0 ⇒ not interpenetrating).
      let minOv = Infinity;
      for (const ax of [...boxAxes(geoms[i]), ...boxAxes(geoms[j])]) {
        const a = projectExtentsOnAxisWorld(geoms[i], ci, ax);
        const b = projectExtentsOnAxisWorld(geoms[j], cj, ax);
        const ov = Math.min(a.max, b.max) - Math.max(a.min, b.min);
        if (ov < minOv) minOv = ov;
      }
      if (minOv > TOL) {
        const pair = [types[i], types[j]].sort().join('~');
        overlaps.push({ i, j, depth: minOv, pair, at: { x: +ci.x.toFixed(2), y: +ci.y.toFixed(2), z: +ci.z.toFixed(2) } });
      }
    }
  }

  it('reports any interpenetrating chunk pairs', () => {
    overlaps.sort((a, b) => b.depth - a.depth);
    const byPair: Record<string, number> = {};
    for (const o of overlaps) byPair[o.pair] = (byPair[o.pair] ?? 0) + 1;
    // eslint-disable-next-line no-console
    console.log(`[overlaps] ${overlaps.length} pairs > ${TOL} m. By type:`, JSON.stringify(byPair));
    // eslint-disable-next-line no-console
    for (const o of overlaps.slice(0, 25)) console.log(`  ${o.pair}  depth=${o.depth.toFixed(3)}m at (${o.at.x},${o.at.y},${o.at.z})`);
    expect(overlaps.length, 'interpenetrating chunk pairs (see log)').toBe(0);
  });
});
