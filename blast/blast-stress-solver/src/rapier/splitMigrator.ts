import type { SplitChild } from "./types";

export type ExistingBodyState = {
  handle: number;
  nodeIndices: ReadonlySet<number>;
  isFixed: boolean;
};

export type PlannerChild = SplitChild & { index: number };

export type SplitMigrationPlan = {
  reuse: Array<{ childIndex: number; bodyHandle: number }>;
  create: Array<{ childIndex: number }>;
};

export type SplitMigrationTimings = { onDuration?: (ms: number) => void };

/**
 * Plan body reuse vs. creation for split children.
 *
 * When a parent rigid body fractures, its chunks are partitioned into N child
 * islands. Each island can either *reuse* an existing body (keep its Rapier
 * handle + colliders + pose — the cheapest topology edit) or get a *new* body.
 * The planner maximizes total node overlap between reused (body, child) pairs,
 * which is the proxy for "colliders that stay attached to their body".
 *
 * The shipping path (this function) solves that assignment by decomposing the
 * body↔child overlap graph into connected components and solving each in
 * isolation — argmax for single-body / single-child components (covering the
 * 1×N cascade and the N×1 merge), a small dense Hungarian only for a component
 * with multiple bodies *and* multiple mutually-overlapping children. This is
 * provably identical to one global maximum-overlap matching (an optimal match
 * never pairs a body with a 0-overlap child, so no beneficial edge crosses a
 * component boundary) but avoids the O(max(M,N)^3) blow-up of a single
 * square-padded Hungarian over the whole (usually sparse) set.
 *
 * {@link planSplitMigrationReference} keeps the original single dense Hungarian
 * as an independent oracle; the equivalence test suite proves the two agree on
 * the optimum for every case, so this optimization is free (same result, faster).
 */
export function planSplitMigration(
  bodies: ExistingBodyState[],
  children: PlannerChild[],
  timings?: SplitMigrationTimings,
): SplitMigrationPlan {
  return planSplitMigrationInner(bodies, children, assignByComponents, timings);
}

/**
 * Reference planner — the original single square-padded dense Hungarian over the
 * whole unmatched body × child set (no component decomposition, no argmax fast
 * path). Retained as an independent oracle to prove the shipping planner is
 * equally optimal and to A/B their cost at scale. Not used by the runtime.
 */
export function planSplitMigrationReference(
  bodies: ExistingBodyState[],
  children: PlannerChild[],
  timings?: SplitMigrationTimings,
): SplitMigrationPlan {
  return planSplitMigrationInner(bodies, children, assignByDenseHungarian, timings);
}

type Assigner = (
  bodies: ExistingBodyState[],
  children: PlannerChild[],
) => SplitMigrationPlan;

function planSplitMigrationInner(
  bodies: ExistingBodyState[],
  children: PlannerChild[],
  assign: Assigner,
  timings?: SplitMigrationTimings,
): SplitMigrationPlan {
  const start = now();

  if (bodies.length === 0 || children.length === 0) {
    const plan: SplitMigrationPlan = {
      reuse: [],
      create: children.map((child) => ({ childIndex: child.index })),
    };
    timings?.onDuration?.(now() - start);
    return plan;
  }

  // Phase 1 — exact node-set matches reuse their own body (the cheapest possible
  // edit: nothing moves). A fixed body may only be exact-matched by a support
  // child; a dynamic body matches any child with the same node set.
  const bodyHashes = buildBodyHashIndex(bodies);
  const reuse: SplitMigrationPlan["reuse"] = [];
  const assignedBodies = new Set<number>();
  const unmatchedChildren: PlannerChild[] = [];

  for (const child of children) {
    const hash = hashNodes(child.nodes);
    const candidates = bodyHashes.get(hash);
    if (!candidates) {
      unmatchedChildren.push(child);
      continue;
    }
    const body = candidates.find(
      (candidate) =>
        !assignedBodies.has(candidate.handle) &&
        (candidate.isFixed ? child.isSupport : true),
    );
    if (!body) {
      unmatchedChildren.push(child);
      continue;
    }
    reuse.push({ childIndex: child.index, bodyHandle: body.handle });
    assignedBodies.add(body.handle);
  }

  const unmatchedBodies = bodies.filter((body) => !assignedBodies.has(body.handle));

  // Phase 2 — maximum-overlap assignment over what's left.
  let remaining: SplitMigrationPlan;
  if (unmatchedBodies.length > 0 && unmatchedChildren.length > 0) {
    remaining = assign(unmatchedBodies, unmatchedChildren);
  } else {
    remaining = {
      reuse: [],
      create: unmatchedChildren.map((child) => ({ childIndex: child.index })),
    };
  }

  const plan: SplitMigrationPlan = {
    reuse: [...reuse, ...remaining.reuse],
    create: remaining.create,
  };
  timings?.onDuration?.(now() - start);
  return plan;
}

/**
 * Maximum-overlap assignment via connected-component decomposition of the
 * body↔child overlap graph (the shipping path). Each component is solved
 * independently and the results concatenated. Produces a deterministic plan
 * (reuse pairs sorted by child index).
 *
 * Assumes existing bodies own disjoint node sets (each chunk belongs to exactly
 * one rigid body), which holds for the destruction pipeline — so a node maps to
 * at most one body and overlaps can be built sparsely from a node→body map.
 */
function assignByComponents(
  bodies: ExistingBodyState[],
  children: PlannerChild[],
): SplitMigrationPlan {
  const nb = bodies.length;
  const nc = children.length;

  // node -> local body index (bodies are disjoint node sets).
  const nodeToBody = new Map<number, number>();
  for (let bi = 0; bi < nb; bi += 1) {
    for (const node of bodies[bi].nodeIndices) nodeToBody.set(node, bi);
  }

  // Sparse overlaps (per child: body index -> shared node count) + union the
  // body and child for every *allowed* positive-overlap edge. A (body, child)
  // edge is suppressed when a fixed body would adopt a non-support child — that
  // pairing is forbidden, exactly as the dense path scores it 0.
  const overlaps: Array<Map<number, number>> = Array.from({ length: nc }, () => new Map());
  const uf = new UnionFind(nb + nc); // bodies [0, nb), children [nb, nb + nc)
  for (let ci = 0; ci < nc; ci += 1) {
    const child = children[ci];
    for (const node of child.nodes) {
      const bi = nodeToBody.get(node);
      if (bi === undefined) continue;
      if (bodies[bi].isFixed && !child.isSupport) continue; // forbidden pairing
      overlaps[ci].set(bi, (overlaps[ci].get(bi) ?? 0) + 1);
      uf.union(bi, nb + ci);
    }
  }

  // Group bodies and (overlapping) children by component root. A child with no
  // allowed overlap is left out — it is created, not reused; a body in no edge
  // stays unmatched and is retired.
  const compBodies = new Map<number, number[]>();
  const compChildren = new Map<number, number[]>();
  for (let bi = 0; bi < nb; bi += 1) pushTo(compBodies, uf.find(bi), bi);
  for (let ci = 0; ci < nc; ci += 1) {
    if (overlaps[ci].size > 0) pushTo(compChildren, uf.find(nb + ci), ci);
  }

  // Solve each component (deterministic order: ascending root).
  const roots = [...compChildren.keys()].sort((a, b) => a - b);
  const pairs: Array<[number, number]> = []; // [local body, local child]
  for (const root of roots) {
    solveComponent(compBodies.get(root) ?? [], compChildren.get(root) ?? [], overlaps, pairs);
  }

  pairs.sort((a, b) => a[1] - b[1]); // ascending child index -> deterministic plan
  const reuse: SplitMigrationPlan["reuse"] = [];
  const reusedLocalChildren = new Set<number>();
  for (const [bi, ci] of pairs) {
    reuse.push({ childIndex: children[ci].index, bodyHandle: bodies[bi].handle });
    reusedLocalChildren.add(ci);
  }
  const create: SplitMigrationPlan["create"] = [];
  for (let ci = 0; ci < nc; ci += 1) {
    if (!reusedLocalChildren.has(ci)) create.push({ childIndex: children[ci].index });
  }
  return { reuse, create };
}

/**
 * Optimal maximum-overlap matching within one component, appended to `pairs` as
 * [local body, local child]. Argmax for single-body / single-child components
 * (the 1×N cascade and the N×1 merge); a small dense Hungarian otherwise.
 */
function solveComponent(
  compBodies: number[],
  compChildren: number[],
  overlaps: Array<Map<number, number>>,
  pairs: Array<[number, number]>,
): void {
  const ov = (bi: number, ci: number): number => overlaps[ci].get(bi) ?? 0;

  if (compBodies.length === 1) {
    const bi = compBodies[0];
    let bestChild = -1;
    let bestOverlap = 0;
    for (const ci of compChildren) {
      const o = ov(bi, ci);
      if (o > bestOverlap) {
        bestOverlap = o;
        bestChild = ci;
      }
    }
    if (bestChild >= 0) pairs.push([bi, bestChild]);
    return;
  }
  if (compChildren.length === 1) {
    const ci = compChildren[0];
    let bestBody = -1;
    let bestOverlap = 0;
    for (const bi of compBodies) {
      const o = ov(bi, ci);
      if (o > bestOverlap) {
        bestOverlap = o;
        bestBody = bi;
      }
    }
    if (bestBody >= 0) pairs.push([bestBody, ci]);
    return;
  }

  // Dense component (multiple bodies AND children): Hungarian over just these
  // members — small, because cross-component pairs share no node.
  const matrix = compBodies.map((bi) => compChildren.map((ci) => ov(bi, ci)));
  const assignment = hungarianMax(matrix);
  for (let row = 0; row < assignment.length; row += 1) {
    const col = assignment[row];
    if (col < 0 || col >= compChildren.length) continue;
    if ((matrix[row]?.[col] ?? 0) <= 0) continue;
    pairs.push([compBodies[row], compChildren[col]]);
  }
}

/**
 * Reference assignment: the original single square-padded dense Hungarian over
 * all unmatched bodies × children. Kept as an independent oracle (see
 * {@link planSplitMigrationReference}); not used by the runtime.
 */
function assignByDenseHungarian(
  bodies: ExistingBodyState[],
  children: PlannerChild[],
): SplitMigrationPlan {
  const overlapMatrix = buildOverlapMatrix(bodies, children);
  const assignments = hungarianMax(overlapMatrix);
  const reuse: SplitMigrationPlan["reuse"] = [];
  const reusedChildren = new Set<number>();

  bodies.forEach((body, rowIdx) => {
    const assignedCol = assignments[rowIdx];
    if (assignedCol == null || assignedCol < 0 || assignedCol >= children.length) {
      return;
    }
    const overlap = overlapMatrix[rowIdx][assignedCol] ?? 0;
    if (overlap > 0) {
      reuse.push({
        childIndex: children[assignedCol].index,
        bodyHandle: body.handle,
      });
      reusedChildren.add(children[assignedCol].index);
    }
  });

  const create = children
    .filter((child) => !reusedChildren.has(child.index))
    .map((child) => ({ childIndex: child.index }));
  return { reuse, create };
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function pushTo<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const bucket = map.get(key);
  if (bucket) bucket.push(value);
  else map.set(key, [value]);
}

function buildBodyHashIndex(
  bodies: ExistingBodyState[],
): Map<string, ExistingBodyState[]> {
  const map = new Map<string, ExistingBodyState[]>();
  for (const body of bodies) {
    const hash = hashNodes(body.nodeIndices);
    const bucket = map.get(hash);
    if (bucket) {
      bucket.push(body);
    } else {
      map.set(hash, [body]);
    }
  }
  return map;
}

function hashNodes(nodes: Iterable<number>): string {
  const list = Array.isArray(nodes) ? nodes.slice() : Array.from(nodes);
  list.sort((a, b) => a - b);
  return list.join(",");
}

function buildOverlapMatrix(
  bodies: ExistingBodyState[],
  children: PlannerChild[],
): number[][] {
  return bodies.map((body) => {
    return children.map((child) => {
      if (body.isFixed && !child.isSupport) {
        return 0;
      }
      let overlap = 0;
      for (const nodeIndex of child.nodes) {
        if (body.nodeIndices.has(nodeIndex)) overlap += 1;
      }
      return overlap;
    });
  });
}

/** Minimal union-find with path compression; the smaller root wins so a
 *  component's root is its lowest member (keeps component order stable). */
class UnionFind {
  private parent: number[];
  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
  }
  find(x: number): number {
    let root = x;
    while (this.parent[root] !== root) root = this.parent[root];
    let cur = x;
    while (this.parent[cur] !== cur) {
      const next = this.parent[cur];
      this.parent[cur] = root;
      cur = next;
    }
    return root;
  }
  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent[Math.max(ra, rb)] = Math.min(ra, rb);
  }
}

/** Maximum-weight assignment for one row per body. Returns, for each row, the
 *  assigned column (or -1). Square-pads to max(rows, cols) and delegates to the
 *  min-cost Hungarian — O(max(rows, cols)^3). The component decomposition keeps
 *  the matrices passed here small. */
function hungarianMax(matrix: number[][]): number[] {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  if (!rows || !cols) return new Array(rows).fill(-1);
  const size = Math.max(rows, cols);
  let maxVal = 0;
  for (const row of matrix) {
    for (const value of row) maxVal = Math.max(maxVal, value);
  }
  const cost = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => {
      const value = matrix[i]?.[j] ?? 0;
      return maxVal - value;
    }),
  );
  const assignment = hungarian(cost);
  return assignment.slice(0, rows);
}

function hungarian(cost: number[][]): number[] {
  const n = cost.length;
  const m = cost[0]?.length ?? 0;
  const size = Math.max(n, m);
  const paddedCost = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => cost[i]?.[j] ?? 0),
  );
  const u = new Array(size + 1).fill(0);
  const v = new Array(size + 1).fill(0);
  const p = new Array(size + 1).fill(0);
  const way = new Array(size + 1).fill(0);

  for (let i = 1; i <= size; i++) {
    p[0] = i;
    const minv = new Array(size + 1).fill(Infinity);
    const used = new Array(size + 1).fill(false);
    let j0 = 0;
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= size; j++) {
        if (used[j]) continue;
        const cur = paddedCost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= size; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const result = new Array(size).fill(-1);
  for (let j = 1; j <= size; j++) {
    if (p[j] > 0) {
      result[p[j] - 1] = j - 1;
    }
  }
  return result;
}
