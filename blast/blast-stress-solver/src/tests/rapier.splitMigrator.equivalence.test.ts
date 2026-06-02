/**
 * Anti-regression guard for the split planner optimization.
 *
 * The shipping planner ({@link planSplitMigration}) replaced a single dense
 * Hungarian over the whole unmatched body × child set — O(max(M,N)^3) — with a
 * connected-component decomposition + argmax fast path. This suite proves that
 * trade is *free*: the shipping planner reaches the SAME optimum as the retained
 * reference Hungarian ({@link planSplitMigrationReference}) and as an independent
 * brute-force optimum, on degenerate cascades, genuine multi-body reparenting,
 * ties, exact matches, a 400-case randomized sweep, and at large scale — so the
 * speed can never be bought by degrading the assignment. (Mirrors the Rust
 * split_migrator equivalence suite.)
 */
import { describe, expect, it } from "vitest";
import {
  planSplitMigration,
  planSplitMigrationReference,
  type ExistingBodyState,
  type PlannerChild,
  type SplitMigrationPlan,
} from "../rapier/splitMigrator";

// ── Builders ─────────────────────────────────────────────────────────────────

function mkBodies(sets: number[][]): ExistingBodyState[] {
  return sets.map((nodes, i) => ({
    handle: i + 1,
    nodeIndices: new Set(nodes),
    isFixed: false,
  }));
}

function mkChildren(sets: number[][]): PlannerChild[] {
  return sets.map((nodes, i) => ({
    index: i,
    actorIndex: i,
    nodes: [...nodes],
    isSupport: false,
  }));
}

const plan = (b: ExistingBodyState[], c: PlannerChild[]) => planSplitMigration(b, c);
const planRef = (b: ExistingBodyState[], c: PlannerChild[]) => planSplitMigrationReference(b, c);

// ── Scoring + validation ─────────────────────────────────────────────────────

function overlap(body: ExistingBodyState, child: PlannerChild): number {
  let o = 0;
  for (const n of child.nodes) if (body.nodeIndices.has(n)) o += 1;
  return o;
}

/**
 * Assert the plan is a valid partition of the children (each assigned exactly
 * once, reuse XOR create; each body reused at most once; reused pairs share ≥1
 * node) and return the total reuse-overlap that the planner maximizes.
 */
function validateAndScore(
  p: SplitMigrationPlan,
  bodies: ExistingBodyState[],
  children: PlannerChild[],
): number {
  const assigned = new Array(children.length).fill(0);
  for (const r of p.reuse) assigned[r.childIndex] += 1;
  for (const c of p.create) assigned[c.childIndex] += 1;
  expect(assigned.every((t) => t === 1)).toBe(true);

  const reuseCount = new Map<number, number>();
  let total = 0;
  for (const r of p.reuse) {
    reuseCount.set(r.bodyHandle, (reuseCount.get(r.bodyHandle) ?? 0) + 1);
    const body = bodies.find((b) => b.handle === r.bodyHandle);
    expect(body).toBeDefined();
    const ov = overlap(body!, children[r.childIndex]);
    expect(ov).toBeGreaterThan(0); // a reused pair must share a node
    total += ov;
  }
  for (const c of reuseCount.values()) expect(c).toBeLessThanOrEqual(1); // body reused ≤ once
  return total;
}

/** Independent brute-force optimum: max total overlap over all
 *  body→distinct-child assignments (a body may also stay unmatched). */
function bruteOptimum(bodies: ExistingBodyState[], children: PlannerChild[]): number {
  const used = new Array(children.length).fill(false);
  const rec = (bi: number): number => {
    if (bi === bodies.length) return 0;
    let best = rec(bi + 1); // body bi takes no child
    for (let ci = 0; ci < children.length; ci += 1) {
      if (used[ci]) continue;
      const ov = overlap(bodies[bi], children[ci]);
      if (ov === 0) continue;
      used[ci] = true;
      best = Math.max(best, ov + rec(bi + 1));
      used[ci] = false;
    }
    return best;
  };
  return rec(0);
}

function reusePairs(p: SplitMigrationPlan): Array<[number, number]> {
  return p.reuse
    .map((r) => [r.childIndex, r.bodyHandle] as [number, number])
    .sort((a, b) => a[0] - b[0]);
}
function createSet(p: SplitMigrationPlan): number[] {
  return p.create.map((c) => c.childIndex).sort((a, b) => a - b);
}

// ── 1. Degenerate cascade (the fast path): one parent -> N children. ─────────

describe("split planner: core assignment (dynamic bodies)", () => {
  it("cascade 1->N equals reference and is optimal", () => {
    for (const n of [2, 5, 33, 256]) {
      const bodies = mkBodies([Array.from({ length: n }, (_, i) => i)]);
      const children = mkChildren(Array.from({ length: n }, (_, k) => [k]));

      const fast = plan(bodies, children);
      const reference = planRef(bodies, children);
      const sFast = validateAndScore(fast, bodies, children);
      const sRef = validateAndScore(reference, bodies, children);

      expect(fast.reuse.length).toBe(1); // one body reuses one child
      expect(fast.create.length).toBe(n - 1); // the rest are created
      expect(sFast).toBe(1); // best single-node overlap is 1
      expect(sFast).toBe(sRef);
      if (n <= 8) expect(sFast).toBe(bruteOptimum(bodies, children));
    }
  });

  // ── 2. Complex partial reparenting (the real worst case). ─────────────────
  it("complex 4x4 reparenting is optimal and matches reference", () => {
    // 4 bodies partition nodes 0..16; a shear regroups them into 4 children that
    // each straddle two old bodies — a genuine 4×4 assignment (no exact match).
    const bodies = mkBodies([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15],
    ]);
    const children = mkChildren([
      [0, 1, 4], // body0:2 body1:1 -> body0
      [2, 3, 5, 6, 7], // body0:2 body1:3 -> body1
      [8, 9, 12], // body2:2 body3:1 -> body2
      [10, 11, 13, 14, 15], // body2:2 body3:3 -> body3
    ]);

    const fast = plan(bodies, children);
    const reference = planRef(bodies, children);
    const sFast = validateAndScore(fast, bodies, children);
    const sRef = validateAndScore(reference, bodies, children);

    expect(sFast).toBe(bruteOptimum(bodies, children));
    expect(sFast).toBe(sRef);
    expect(createSet(fast)).toEqual([]); // all 4 chunks reuse a body
  });

  // ── 3. Realistic mixed scene: most bodies survive, one shatters. ──────────
  it("mostly-persist + one shatter uses the fast path correctly", () => {
    const persist = [
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7],
    ];
    const bsets = [...persist, Array.from({ length: 8 }, (_, i) => 100 + i)];
    const bodies = mkBodies(bsets);
    const csets = [...persist, ...Array.from({ length: 8 }, (_, i) => [100 + i])];
    const children = mkChildren(csets);

    const fast = plan(bodies, children);
    const sFast = validateAndScore(fast, bodies, children);
    const sRef = validateAndScore(planRef(bodies, children), bodies, children);

    // Survivors reuse their *own* body (exact match, never recreated).
    for (let ci = 0; ci < persist.length; ci += 1) {
      const r = fast.reuse.find((x) => x.childIndex === ci);
      expect(r).toBeDefined();
      expect(r!.bodyHandle).toBe(bodies[ci].handle);
    }
    expect(fast.create.length).toBe(7); // shattered body reuses 1 shard, creates 7
    expect(sFast).toBe(sRef);
    expect(sFast).toBe(bruteOptimum(bodies, children)); // = 8 (survivors) + 1 (shard)
  });

  // ── 4. Ties resolved equally optimally. ───────────────────────────────────
  it("ties are resolved equally optimally", () => {
    const bodies = mkBodies([[0, 1, 2, 3]]);
    const children = mkChildren([
      [0, 1],
      [2, 3],
    ]); // both overlap the body by 2
    const fast = plan(bodies, children);
    const sFast = validateAndScore(fast, bodies, children);
    const sRef = validateAndScore(planRef(bodies, children), bodies, children);
    expect([fast.reuse.length, fast.create.length]).toEqual([1, 1]);
    expect(sFast).toBe(2);
    expect(sFast).toBe(sRef);
    expect(sFast).toBe(bruteOptimum(bodies, children));
  });

  // ── 5. Exact node-set matches must always be reused. ──────────────────────
  it("exact node-set matches are always reused", () => {
    const bodies = mkBodies([
      [0, 1, 2],
      [3, 4],
      [5, 6, 7, 8],
    ]);
    const children = mkChildren([
      [5, 6, 7, 8],
      [0, 1, 2],
      [3, 4],
    ]); // permuted
    const fast = plan(bodies, children);
    validateAndScore(fast, bodies, children);
    expect(fast.create).toEqual([]);
    expect(fast.reuse.length).toBe(3);
    for (const r of fast.reuse) {
      const body = bodies.find((b) => b.handle === r.bodyHandle)!;
      const cset = new Set(children[r.childIndex].nodes);
      expect(body.nodeIndices).toEqual(cset);
    }
  });

  // ── 6. Randomized property sweep — the core guard. ────────────────────────
  it("400 randomized fractures: planner == reference == brute optimum, deterministic", () => {
    // mulberry32 — dependency-free deterministic PRNG.
    let state = 0xdeadbeef >>> 0;
    const next = () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const below = (n: number) => Math.floor(next() * n);

    let checked = 0;
    for (let cas = 0; cas < 400; cas += 1) {
      const nodeCount = 1 + below(24);
      const nb = 1 + below(6); // up to 6 bodies  -> small enough for brute force
      const nc = 1 + below(7); // up to 7 children -> spans degenerate..complex
      const bn: number[][] = Array.from({ length: nb }, () => []);
      const cn: number[][] = Array.from({ length: nc }, () => []);
      for (let node = 0; node < nodeCount; node += 1) {
        bn[below(nb)].push(node); // one body label
        cn[below(nc)].push(node); // one child label (disjoint partition)
      }
      const bsets = bn.filter((v) => v.length > 0);
      const csets = cn.filter((v) => v.length > 0);
      if (bsets.length === 0 || csets.length === 0) continue;

      const bodies = mkBodies(bsets);
      const children = mkChildren(csets);

      const fast = plan(bodies, children);
      const reference = planRef(bodies, children);
      const sFast = validateAndScore(fast, bodies, children);
      const sRef = validateAndScore(reference, bodies, children);
      const opt = bruteOptimum(bodies, children);

      expect(sFast).toBe(opt); // shipping planner globally optimal
      expect(sRef).toBe(opt); // reference globally optimal
      expect(sFast).toBe(sRef);

      // determinism: identical plan on a repeat run.
      const again = plan(bodies, children);
      expect(reusePairs(fast)).toEqual(reusePairs(again));
      expect(createSet(fast)).toEqual(createSet(again));
      checked += 1;
    }
    expect(checked).toBeGreaterThan(300);
  });

  // ── 7. Scale: fast path stays O(N) at huge N; large complex stays valid. ──
  it("scales to large inputs (1->8192 fast path; 64-body complex)", () => {
    // 7a. one body -> 8192 shards: reuse exactly one, create the rest.
    const bodies = mkBodies([Array.from({ length: 8192 }, (_, i) => i)]);
    const children = mkChildren(Array.from({ length: 8192 }, (_, k) => [k]));
    const p = plan(bodies, children);
    validateAndScore(p, bodies, children);
    expect([p.reuse.length, p.create.length]).toEqual([1, 8191]);

    // 7b. 64 bodies: 32 persist exactly, 32 reshuffle into children straddling
    // two adjacent old bodies — a 32×32 assignment after exact matches drop out.
    const bsets2 = Array.from({ length: 64 }, (_, k) => [k * 2, k * 2 + 1]);
    const bodies2 = mkBodies(bsets2);
    const csets2: number[][] = [
      ...Array.from({ length: 32 }, (_, k) => [k * 2, k * 2 + 1]),
      ...Array.from({ length: 32 }, (_, i) => {
        const k = 32 + i;
        return [k * 2, ((k + 1) % 64) * 2 + 1];
      }),
    ];
    const children2 = mkChildren(csets2);
    const p2 = plan(bodies2, children2);
    const s2 = validateAndScore(p2, bodies2, children2);
    const sRef2 = validateAndScore(planRef(bodies2, children2), bodies2, children2);
    expect(s2).toBe(sRef2);
    for (let k = 0; k < 32; k += 1) {
      const r = p2.reuse.find((x) => x.childIndex === k);
      expect(r).toBeDefined();
      expect(r!.bodyHandle).toBe(bodies2[k].handle); // survivor keeps its own body
    }
  });
});

// ── 8. TS-specific: the fixed/support constraint is preserved exactly. ───────

describe("split planner: fixed/support constraint (production == reference)", () => {
  const mkBody = (handle: number, nodes: number[], isFixed: boolean): ExistingBodyState => ({
    handle,
    nodeIndices: new Set(nodes),
    isFixed,
  });
  const mkChild = (index: number, nodes: number[], isSupport: boolean): PlannerChild => ({
    index,
    actorIndex: index,
    nodes,
    isSupport,
  });

  it("a non-support child never reuses a fixed body (both planners)", () => {
    const bodies = [mkBody(10, [0, 1, 2], true)];
    const children = [mkChild(0, [0], false), mkChild(1, [1, 2], false)];
    for (const planner of [planSplitMigration, planSplitMigrationReference]) {
      const p = planner(bodies, children);
      for (const r of p.reuse) expect(r.bodyHandle).not.toBe(10);
      expect(p.create.length).toBe(2);
    }
  });

  it("a support child may reuse a fixed body; non-support sibling is created", () => {
    const bodies = [mkBody(10, [0, 1, 2], true)];
    const children = [mkChild(0, [0], true), mkChild(1, [1, 2], false)];
    for (const planner of [planSplitMigration, planSplitMigrationReference]) {
      const p = planner(bodies, children);
      const supportReuse = p.reuse.find((r) => r.childIndex === 0);
      expect(supportReuse?.bodyHandle).toBe(10);
      expect(p.create).toContainEqual({ childIndex: 1 });
    }
  });

  it("mixed fixed + dynamic bodies: production matches reference", () => {
    // node 0..1 on a fixed support body; 2..5 on a dynamic body that shatters.
    const bodies = [mkBody(10, [0, 1], true), mkBody(20, [2, 3, 4, 5], false)];
    const children = [
      mkChild(0, [0, 1], true), // exact support match -> reuses fixed body 10
      mkChild(1, [2, 3], false),
      mkChild(2, [4, 5], false),
    ];
    const prod = planSplitMigration(bodies, children);
    const ref = planSplitMigrationReference(bodies, children);
    expect(prod.reuse.find((r) => r.childIndex === 0)?.bodyHandle).toBe(10);
    // Same reuse/create cardinality + same set of reused body handles.
    expect(prod.reuse.length).toBe(ref.reuse.length);
    expect(new Set(prod.reuse.map((r) => r.bodyHandle))).toEqual(
      new Set(ref.reuse.map((r) => r.bodyHandle)),
    );
  });
});
