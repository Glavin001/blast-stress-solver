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

  it.skipIf(!pinataAvailable)('builds an anchored, auto-bonded castle with a four-tier hierarchy', async () => {
    const { buildBrickCastleScenario } = await import('../scenarios/brickCastleScenario');
    // Keep the inter-structure bonds so all four tiers are exercised here.
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

    // All four tiers are represented (anchor, intra-brick, mortar, inter-structure).
    const tc: number[] = Array.from(p.tierCounts);
    expect(tc[CastleBondTier.Anchor]).toBeGreaterThan(0);
    expect(tc[CastleBondTier.IntraBrick]).toBeGreaterThan(0);
    expect(tc[CastleBondTier.Mortar]).toBeGreaterThan(0);
    expect(tc[CastleBondTier.InterStructure]).toBeGreaterThan(0);

    // The strength hierarchy must actually order the (area) strengths:
    // intra-brick ≫ mortar > inter-structure. Compare per-tier mean areas.
    const tiers: Uint8Array = p.bondTiers;
    const sum = [0, 0, 0, 0];
    const cnt = [0, 0, 0, 0];
    scenario.bonds.forEach((b, i) => { sum[tiers[i]] += b.area; cnt[tiers[i]]++; });
    const mean = (t: CastleBondTier) => sum[t] / Math.max(1, cnt[t]);
    const intra = mean(CastleBondTier.IntraBrick);
    const mortar = mean(CastleBondTier.Mortar);
    const inter = mean(CastleBondTier.InterStructure);
    expect(intra).toBeGreaterThan(mortar);
    expect(mortar).toBeGreaterThan(inter);

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
});
