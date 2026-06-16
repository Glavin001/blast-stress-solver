/**
 * Integration tests for the brick-castle scenario.
 *
 * Verifies the structural & strength-hierarchy invariants that the demo relies on:
 * - a valid ScenarioDesc with bonds derived from contact (auto-bonding),
 * - a static foundation anchoring the structure (mass-0 supports),
 * - all three+anchor bond tiers present, and
 * - the strength hierarchy actually orders the bond areas
 *   (intra-brick ≫ mortar > inter-structure), plus
 * - a collision-LOD tree (castle → structure → brick) for lazy colliders.
 *
 * Requires @dgreenheck/three-pinata (skips gracefully if unavailable).
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { CastleBondTier } from '../scenarios/brickCastleScenario';

let pinataAvailable = false;
try {
  require.resolve('@dgreenheck/three-pinata');
  pinataAvailable = true;
} catch {
  pinataAvailable = false;
}

// A deliberately small castle so the WASM auto-bonder runs quickly.
const SMALL = {
  wallLengthBricks: 6,
  wallCourses: 5,
  towerSideBricks: 4,
  towerCourses: 6,
  keepSideBricks: 4,
  keepCourses: 7,
  chunksPerBrick: 2,
} as const;

describe('brick-castle scenario (requires three-pinata)', () => {
  beforeAll(async () => {
    if (pinataAvailable) {
      const { ensurePinataLoaded } = await import('../three/pinataFracture');
      await ensurePinataLoaded();
    }
  });

  it.skipIf(!pinataAvailable)('builds an anchored, auto-bonded castle with the strength hierarchy', async () => {
    const { buildBrickCastleScenario } = await import('../scenarios/brickCastleScenario');
    const scenario = await buildBrickCastleScenario({ ...SMALL, bondMode: 'auto', bondAcrossStructures: true });
    const p = scenario.parameters as any;

    // Valid graph.
    expect(scenario.nodes.length).toBeGreaterThan(200);
    expect(scenario.bonds.length).toBeGreaterThan(scenario.nodes.length * 0.5);

    // Per-node metadata is complete (one entry per node).
    expect(p.brickIdByNode).toHaveLength(scenario.nodes.length);
    expect(p.structureIdByNode).toHaveLength(scenario.nodes.length);
    expect(p.kindByNode).toHaveLength(scenario.nodes.length);
    expect(p.baseColorByNode).toHaveLength(scenario.nodes.length);

    // Anchored: a static foundation grid plus footing (bottom-course) bricks are
    // mass-0 supports. The foundation tiles are a subset of all supports.
    const supports = scenario.nodes.filter((n) => n.mass === 0).length;
    const foundation = p.kindByNode.filter((k: string) => k === 'foundation').length;
    expect(foundation).toBeGreaterThan(0);
    expect(supports).toBeGreaterThanOrEqual(foundation);

    // Dynamic bricks carry physical (volume*density) mass.
    expect(scenario.nodes.some((n) => n.mass > 0)).toBe(true);

    // The structural tiers (intra-brick, mortar) must be populated. The anchor
    // tier is intentionally empty — foundation bonds are dropped so the slab can't
    // couple structures (anchoring is via each structure's own mass-0 footing). The
    // inter-structure tier may also be empty (structures are spaced overlap-free).
    const tc: number[] = Array.from(p.tierCounts);
    expect(tc[CastleBondTier.Anchor]).toBe(0);
    expect(tc[CastleBondTier.IntraBrick]).toBeGreaterThan(0);
    expect(tc[CastleBondTier.Mortar]).toBeGreaterThan(0);

    // Strength ordering of the populated tiers: intra-brick ≫ mortar (by area).
    const tiers: Uint8Array = p.bondTiers;
    const sum = [0, 0, 0, 0];
    const cnt = [0, 0, 0, 0];
    scenario.bonds.forEach((b, i) => { sum[tiers[i]] += b.area; cnt[tiers[i]]++; });
    const mean = (t: CastleBondTier) => sum[t] / Math.max(1, cnt[t]);
    expect(mean(CastleBondTier.IntraBrick)).toBeGreaterThan(mean(CastleBondTier.Mortar));

    // Collision-LOD tree: one castle root whose children are the structures.
    expect(scenario.collisionTree).toBeDefined();
    expect(scenario.collisionTree!.length).toBe(1);
    expect((scenario.collisionTree![0].children?.length ?? 0)).toBeGreaterThan(1);
  }, 60_000);

  it.skipIf(!pinataAvailable)('honors custom tier multipliers', async () => {
    const { buildBrickCastleScenario } = await import('../scenarios/brickCastleScenario');
    // Make mortar far stronger than intra-brick and confirm the ordering flips.
    const scenario = await buildBrickCastleScenario({
      ...SMALL,
      bondMode: 'auto',
      multipliers: { intraBrick: 1, mortar: 20, interStructure: 0.5, anchor: 3 },
    });
    const p = scenario.parameters as any;
    const tiers: Uint8Array = p.bondTiers;
    const sum = [0, 0, 0, 0];
    const cnt = [0, 0, 0, 0];
    scenario.bonds.forEach((b, i) => { sum[tiers[i]] += b.area; cnt[tiers[i]]++; });
    const mean = (t: number) => sum[t] / Math.max(1, cnt[t]);
    expect(mean(CastleBondTier.Mortar)).toBeGreaterThan(mean(CastleBondTier.IntraBrick));
  }, 60_000);

  it.skipIf(!pinataAvailable)('decouples structures into separate stress islands by default', async () => {
    const { buildBrickCastleScenario } = await import('../scenarios/brickCastleScenario');
    // Default (bondAcrossStructures:false) drops the inter-structure bonds so a hit
    // stays local. Confirm the tier is empty and the largest dynamic island is well
    // below the whole castle (no single ring island).
    const scenario = await buildBrickCastleScenario({ ...SMALL, bondMode: 'auto' });
    const p = scenario.parameters as any;
    expect(Array.from(p.tierCounts)[CastleBondTier.InterStructure]).toBe(0);

    const n = scenario.nodes.length;
    const dyn = (i: number) => scenario.nodes[i].mass > 0;
    const dynCount = scenario.nodes.filter((nd) => nd.mass > 0).length;
    const par = Array.from({ length: n }, (_, i) => i);
    const find = (x: number): number => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    for (const b of scenario.bonds) {
      if (!dyn(b.node0) || !dyn(b.node1)) continue;
      const a = find(b.node0), c = find(b.node1);
      if (a !== c) par[a] = c;
    }
    const size = new Map<number, number>();
    for (let i = 0; i < n; i++) { if (!dyn(i)) continue; const r = find(i); size.set(r, (size.get(r) ?? 0) + 1); }
    const largest = Math.max(...size.values());
    // No single island should span more than ~60% of the dynamic bricks.
    expect(largest).toBeLessThan(dynCount * 0.6);
  }, 60_000);

  it.skipIf(!pinataAvailable)('places no overlapping colliders between distinct bricks', async () => {
    // Regression guard for the catastrophic-collapse bug: authored colliders that
    // interpenetrate get ejected violently the instant they materialize as separate
    // rigid bodies (correct physics under full collision). The scene must therefore
    // contain NO inter-brick collider overlaps — junctions (wall↔tower), the gate,
    // and battlements were the offenders.
    const { buildBrickCastleScenario } = await import('../scenarios/brickCastleScenario');
    const scenario = await buildBrickCastleScenario({ ...SMALL, bondMode: 'auto' });
    const p = scenario.parameters as any;
    const cen = scenario.nodes.map((nd) => nd.centroid);
    const sz = p.fragmentSizes as Array<{ x: number; y: number; z: number }>;
    const brick = p.brickIdByNode as number[];
    const N = scenario.nodes.length;
    const half = (i: number) => ({ x: Math.max(sz[i].x * 0.5, 0.02), y: Math.max(sz[i].y * 0.5, 0.02), z: Math.max(sz[i].z * 0.5, 0.02) });
    // Spatial hash so this stays O(N).
    const cell = 1.0;
    const H = new Map<string, number[]>();
    const key = (a: number, b: number, c: number) => `${a},${b},${c}`;
    for (let i = 0; i < N; i++) {
      const c = cen[i];
      const k = key(Math.floor(c.x / cell), Math.floor(c.y / cell), Math.floor(c.z / cell));
      const arr = H.get(k); if (arr) arr.push(i); else H.set(k, [i]);
    }
    const pen = (i: number, j: number) => {
      const a = cen[i], b = cen[j], ha = half(i), hb = half(j);
      const ox = (ha.x + hb.x) - Math.abs(a.x - b.x); if (ox <= 0) return 0;
      const oy = (ha.y + hb.y) - Math.abs(a.y - b.y); if (oy <= 0) return 0;
      const oz = (ha.z + hb.z) - Math.abs(a.z - b.z); if (oz <= 0) return 0;
      return Math.min(ox, oy, oz);
    };
    let deep = 0;
    // Bricks are placed TOUCHING (gap 0), so any real penetration is a bug. The
    // half-extents bake in each brick's 0°/90° rotation, so this AABB test is exact
    // for the castle. 1 cm only absorbs float error — it is NOT a mortar gap.
    const DEEP = 0.01; // metres
    for (let i = 0; i < N; i++) {
      const c = cen[i];
      const cx = Math.floor(c.x / cell), cy = Math.floor(c.y / cell), cz = Math.floor(c.z / cell);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const arr = H.get(key(cx + dx, cy + dy, cz + dz)); if (!arr) continue;
        for (const j of arr) {
          if (j <= i) continue;
          if (brick[i] === brick[j]) continue; // same-brick Voronoi siblings legitimately share faces
          if (pen(i, j) > DEEP) deep++;
        }
      }
    }
    expect(deep).toBe(0);
  }, 60_000);

  it.skipIf(!pinataAvailable)('keeps every structure an independent island (no foundation coupling)', async () => {
    // Regression guard for "hit a corner tower, the centre keep also breaks": the
    // shared foundation slab used to bond all structures into one connected graph,
    // so a hit propagated through it into every structure. No bonded component may
    // span more than one structure (foundation bonds are dropped entirely).
    const { buildBrickCastleScenario } = await import('../scenarios/brickCastleScenario');
    const scenario = await buildBrickCastleScenario({ ...SMALL, bondMode: 'auto' });
    const p = scenario.parameters as any;
    const kind = p.kindByNode as string[];
    const sid = p.structureIdByNode as number[];
    const N = scenario.nodes.length;
    const par = Array.from({ length: N }, (_, i) => i);
    const find = (x: number): number => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    for (const b of scenario.bonds) { const a = find(b.node0), c = find(b.node1); if (a !== c) par[a] = c; }
    // No foundation bonds should survive at all.
    expect(scenario.bonds.some((b) => kind[b.node0] === 'foundation' || kind[b.node1] === 'foundation')).toBe(false);
    // Every connected component must contain at most one (non-foundation) structure.
    const compStructs = new Map<number, Set<number>>();
    for (let i = 0; i < N; i++) {
      if (kind[i] === 'foundation') continue;
      const r = find(i);
      const set = compStructs.get(r) ?? new Set<number>();
      set.add(sid[i]); compStructs.set(r, set);
    }
    const maxStructsPerComponent = Math.max(...[...compStructs.values()].map((s) => s.size));
    expect(maxStructsPerComponent).toBe(1);
  }, 60_000);
});
