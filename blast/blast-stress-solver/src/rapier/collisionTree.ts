import type { ScenarioDesc, CollisionGroup } from './types';

export type SpatialCollisionTreeOptions = {
  /** Maximum fragments in a leaf group. Smaller ⇒ finer descent (more locality), more nodes to
   *  test. Default 24. */
  leafMaxFragments?: number;
};

/**
 * Build a hierarchical collision-LOD tree (see `CollisionGroup`) from a flat scenario, with no
 * authoring metadata required:
 *   1. group nodes into buildings = connected components of the bond graph (supports included),
 *   2. recursively median-split each building's fragments along their longest centroid axis until
 *      each leaf holds ≤ `leafMaxFragments` fragments (a balanced BVH-like tree).
 *
 * The result is shape-agnostic and works for any structure. Pass it as `scenario.collisionTree`
 * to let `lazyIntactColliders` descend the collider frontier per-region: a localized hit only
 * materializes the struck leaf's fragments instead of the whole building. It is orthogonal to the
 * bond/stress graph and cannot change fracture output.
 */
export function buildSpatialCollisionTree(
  scenario: Pick<ScenarioDesc, 'nodes' | 'bonds'>,
  options: SpatialCollisionTreeOptions = {},
): CollisionGroup[] {
  const leafMax = Math.max(1, Math.floor(options.leafMaxFragments ?? 24));
  const nodes = scenario.nodes;
  const n = nodes.length;
  if (n === 0) return [];

  // 1. Connected components via union-find over all bonds.
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x: number): number => {
    let r = x;
    while (parent[r] !== r) r = parent[r];
    while (parent[x] !== r) { const nx = parent[x]; parent[x] = r; x = nx; }
    return r;
  };
  for (const b of scenario.bonds) {
    const r0 = find(b.node0), r1 = find(b.node1);
    if (r0 !== r1) parent[r0] = r1;
  }
  const components = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    let arr = components.get(r);
    if (!arr) { arr = []; components.set(r, arr); }
    arr.push(i);
  }

  // 2. Median-split each component into a balanced tree.
  const cx = (i: number) => nodes[i].centroid.x;
  const cy = (i: number) => nodes[i].centroid.y;
  const cz = (i: number) => nodes[i].centroid.z;

  function split(indices: number[]): CollisionGroup {
    if (indices.length <= leafMax) return { fragments: indices };
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const i of indices) {
      const x = cx(i), y = cy(i), z = cz(i);
      if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
      if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
    }
    const ex = maxX - minX, ey = maxY - minY, ez = maxZ - minZ;
    const axis = ex >= ey && ex >= ez ? cx : ey >= ez ? cy : cz;
    const sorted = indices.slice().sort((a, b) => axis(a) - axis(b));
    const mid = sorted.length >> 1;
    const left = sorted.slice(0, mid), right = sorted.slice(mid);
    // Degenerate (all coincident along every axis) → can't reduce; emit a leaf to avoid recursion.
    if (left.length === 0 || right.length === 0) return { fragments: indices };
    return { children: [split(left), split(right)] };
  }

  const roots: CollisionGroup[] = [];
  for (const indices of components.values()) roots.push(split(indices));
  return roots;
}
