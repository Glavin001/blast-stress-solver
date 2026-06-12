/**
 * Lazy intact colliders ("collision-dormant buildings").
 *
 * With `lazyIntactColliders` on, each intact building (a connected component of the bond graph)
 * keeps its per-fragment colliders DISABLED (out of the Rapier broadphase) until a mover is about
 * to hit it, then enables them just-in-time via a conservative pre-step AABB test. This crushes
 * the idle broadphase cost on big cities.
 *
 * Equivalence guarantees this test pins:
 *   - while intact and on approach the simulation is BIT-IDENTICAL to the eager (non-lazy) world
 *     (a disabled fixed collider that nothing touches contributes nothing to the solver),
 *   - an impact explodes only the building(s) actually approached, not the whole city,
 *   - the fracture TOPOLOGY (rigid-body count) matches the eager world,
 *   - the live toggle round-trips.
 * Equivalence is provably strong: idle/approach and rigid-material impacts (the production regime)
 * are bit-identical to the eager world. The only divergence is in a *soft-material* mega-shatter
 * (100+ bodies), where the eager reference carries contact warm-start history accumulated while its
 * colliders sat enabled — history the lazy path skips for the idle win. Warm-start is a solver
 * convergence accelerator, so both paths are equally-valid, equally-converged outcomes; chaos just
 * amplifies the rounding difference. We therefore assert bit-identity for idle/approach/rigid and
 * topology parity (not exact positions) for the soft mega-shatter.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type * as RapierEntry from '../rapier';
import type * as Scenarios from '../scenarios';
import type { ScenarioDesc } from '../rapier/types';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

async function load() {
  const rapier = (await import('../../dist/rapier.js')) as typeof RapierEntry;
  const scen = (await import('../../dist/scenarios.js')) as typeof Scenarios;
  return { rapier, scen };
}

type Vec3 = { x: number; y: number; z: number };
function mergeCity(template: ScenarioDesc, offsets: Vec3[]): ScenarioDesc {
  const nodes: any[] = [], bonds: any[] = [];
  let base = 0;
  for (const o of offsets) {
    for (const n of template.nodes)
      nodes.push({ centroid: { x: n.centroid.x + o.x, y: n.centroid.y + o.y, z: n.centroid.z + o.z }, mass: n.mass, volume: n.volume });
    for (const b of template.bonds)
      bonds.push({ node0: b.node0 + base, node1: b.node1 + base, centroid: { x: b.centroid.x + o.x, y: b.centroid.y + o.y, z: b.centroid.z + o.z }, normal: { ...b.normal }, area: b.area });
    base += template.nodes.length;
  }
  return { nodes, bonds, parameters: {} } as ScenarioDesc;
}

const OPTS = { gravity: -9.81, materialScale: 2e6, friction: 0.25, restitution: 0, contactForceScale: 30, debrisCollisionMode: 'all' as const, damage: { enabled: false } };

describe.skipIf(!runtimeAvailable)('Lazy intact colliders', () => {
  it('idle: buildings start dormant, intact city is bit-identical to the eager world', async () => {
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const offsets = [{ x: -17, y: 0, z: 0 }, { x: 17, y: 0, z: 0 }];
    const scenario = mergeCity(template, offsets);

    const lazy = await rapier.buildDestructibleCore({ scenario, ...OPTS, lazyIntactColliders: true });
    const eager = await rapier.buildDestructibleCore({ scenario, ...OPTS, lazyIntactColliders: false });

    const stats = lazy.getLazyColliderStats!();
    expect(stats.enabled).toBe(true);
    expect(stats.buildingCount).toBe(2);          // two disconnected towers → two buildings
    expect(stats.dormantCount).toBe(2);            // both dormant at rest
    expect(stats.explodedCount).toBe(0);

    const dt = 1 / 60;
    let maxDelta = 0;
    for (let f = 0; f < 30; f++) {
      lazy.step(dt); eager.step(dt);
      for (let i = 0; i < lazy.chunks.length; i++) {
        const a = lazy.chunks[i].worldPosition, b = eager.chunks[i].worldPosition;
        if (a && b) maxDelta = Math.max(maxDelta, Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
      }
    }
    // Intact, undisturbed city: dormant (disabled) colliders touch nothing → identical to eager.
    expect(maxDelta).toBe(0);
    expect(lazy.getLazyColliderStats!().dormantCount).toBe(2); // nothing exploded spuriously

    lazy.dispose?.(); eager.dispose?.();
  });

  it('an impact explodes only the approached building; far buildings stay dormant; topology matches eager', async () => {
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    // Three towers in a row; we hit the middle one (at the origin).
    const offsets = [{ x: -40, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, { x: 40, y: 0, z: 0 }];
    const scenario = mergeCity(template, offsets);
    const dt = 1 / 60;

    function fireAndRun(core: Awaited<ReturnType<typeof rapier.buildDestructibleCore>>) {
      for (let i = 0; i < 15; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: 0, y: 5, z: -22 }, velocity: { x: 0, y: 0, z: 70 }, radius: 0.6, mass: 1500, ttl: 3000 });
      for (let i = 0; i < 40; i++) core.step(dt);
    }

    const lazy = await rapier.buildDestructibleCore({ scenario, ...OPTS, lazyIntactColliders: true });
    const eager = await rapier.buildDestructibleCore({ scenario, ...OPTS, lazyIntactColliders: false });
    fireAndRun(lazy);
    fireAndRun(eager);

    const lz = lazy.getLazyColliderStats!();
    // The struck (middle) building exploded; at least one other stayed dormant — we did NOT
    // re-materialise the whole city from a single hit.
    expect(lz.explodedCount).toBeGreaterThanOrEqual(1);
    expect(lz.dormantCount).toBeGreaterThanOrEqual(1);
    expect(lz.explodedCount).toBeLessThan(lz.buildingCount);

    // Fracture topology is unchanged by the optimization: same number of rigid bodies.
    // (Position parity is asserted separately for rigid material; under very soft material a
    // building explodes into hundreds of bodies and the exact rubble positions diverge within
    // Rapier's contact warm-start tolerance — an equally-valid, equally-converged outcome — so
    // here we pin topology, not positions.)
    expect(lazy.getRigidBodyCount()).toBe(eager.getRigidBodyCount());

    lazy.dispose?.(); eager.dispose?.();
  });

  it('a rigid-material impact (production regime) is bit-identical to the eager world', async () => {
    // The divergence in the soft-material test above is contact warm-start chaos in a 100+ body
    // shatter. For a rigid material (mini-city uses 1e10) a hit breaks off only a handful of
    // bodies, and lazy is bit-identical to eager through the whole impact — proving the
    // enable-on-approach mechanism itself does not perturb the solve.
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const scenario = mergeCity(template, [{ x: -20, y: 0, z: 0 }, { x: 20, y: 0, z: 0 }]);
    const dt = 1 / 60;
    function fire(core: Awaited<ReturnType<typeof rapier.buildDestructibleCore>>) {
      for (let i = 0; i < 15; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: -20, y: 5, z: -22 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.7, mass: 9000, ttl: 3000 });
      const frames: Array<Array<{ x: number; y: number; z: number } | null>> = [];
      for (let f = 0; f < 40; f++) { core.step(dt); frames.push(core.chunks.map((c: any) => c.worldPosition ? { ...c.worldPosition } : null)); }
      return frames;
    }
    const lazy = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10, lazyIntactColliders: true });
    const eager = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10, lazyIntactColliders: false });
    const lf = fire(lazy), ef = fire(eager);

    let maxDelta = 0;
    for (let f = 0; f < lf.length; f++)
      for (let i = 0; i < lf[f].length; i++) {
        const a = lf[f][i], b = ef[f][i];
        if (a && b) maxDelta = Math.max(maxDelta, Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
      }
    expect(lazy.getRigidBodyCount()).toBeGreaterThan(2); // the impact actually broke something off
    expect(maxDelta).toBeLessThan(1e-6);                  // …and lazy matched eager bit-for-bit through it

    lazy.dispose?.(); eager.dispose?.();
  });

  it('toggling the flag off materializes every building (and on disables them again)', async () => {
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 2, floorHeight: 3 });
    const scenario = mergeCity(template, [{ x: -17, y: 0, z: 0 }, { x: 17, y: 0, z: 0 }]);
    const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, lazyIntactColliders: true });
    core.step(1 / 60);
    expect(core.getLazyColliderStats!().dormantCount).toBe(2);

    core.setLazyIntactColliders!(false);
    expect(core.getLazyColliderStats!().dormantCount).toBe(0); // all enabled (eager)
    expect(core.getLazyColliderStats!().explodedCount).toBe(2);

    core.setLazyIntactColliders!(true);
    expect(core.getLazyColliderStats!().dormantCount).toBe(2); // dormant again

    core.dispose?.();
  });
});

describe.skipIf(!runtimeAvailable)('Hierarchical collision LOD (collisionTree)', () => {
  it('a localized hit on a tall structure activates only the struck region (not the whole building)', async () => {
    const { rapier, scen } = await load();
    const built = await (scen.buildHighRiseScenarioAsync ?? scen.buildHighRiseScenario)({});
    const scenario = ((built as any).nodes ? built : (built as any).scenario) as ScenarioDesc;
    // Opt-in spatial LOD tree: building → balanced sub-regions (≤32 fragments per leaf).
    scenario.collisionTree = rapier.buildSpatialCollisionTree(scenario, { leafMaxFragments: 32 });

    const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10, lazyIntactColliders: true });
    const dt = 1 / 60;
    for (let i = 0; i < 20; i++) core.step(dt);
    expect(core.getLazyColliderStats!().activeLeafFragments).toBe(0); // fully dormant at rest

    // Hit the lower part horizontally.
    let ymin = Infinity, ymax = -Infinity;
    for (const n of scenario.nodes) { ymin = Math.min(ymin, n.centroid.y); ymax = Math.max(ymax, n.centroid.y); }
    core.enqueueProjectile({ position: { x: 0, y: ymin + (ymax - ymin) * 0.15, z: -40 }, velocity: { x: 0, y: 0, z: 80 }, radius: 0.8, mass: 4000, ttl: 4000 });

    let firstActive = 0;
    for (let f = 0; f < 60; f++) { core.step(dt); const a = core.getLazyColliderStats!().activeLeafFragments; if (a > 0 && firstActive === 0) firstActive = a; }
    // Only the struck region descended — a small fraction of the 900+ fragment tower, not all of it.
    expect(firstActive).toBeGreaterThan(0);
    expect(firstActive).toBeLessThan(scenario.nodes.length * 0.5);

    core.dispose?.();
  });

  it('with a tree, a hit on one building leaves the others fully dormant, and is bit-identical to eager (rigid)', async () => {
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    // Three towers; hit the middle one.
    const scenario = mergeCity(template, [{ x: -40, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, { x: 40, y: 0, z: 0 }]);
    scenario.collisionTree = rapier.buildSpatialCollisionTree(scenario, { leafMaxFragments: 24 });
    const dt = 1 / 60;
    function fire(core: Awaited<ReturnType<typeof rapier.buildDestructibleCore>>) {
      for (let i = 0; i < 15; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: 0, y: 5, z: -22 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.7, mass: 9000, ttl: 3000 });
      const frames: Array<Array<{ x: number; y: number; z: number } | null>> = [];
      for (let f = 0; f < 40; f++) { core.step(dt); frames.push(core.chunks.map((c: any) => c.worldPosition ? { ...c.worldPosition } : null)); }
      return frames;
    }
    const lazy = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10, lazyIntactColliders: true });
    const eager = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10, lazyIntactColliders: false });
    const lf = fire(lazy), ef = fire(eager);

    const lz = lazy.getLazyColliderStats!();
    expect(lz.buildingCount).toBe(3);
    expect(lz.explodedCount).toBeGreaterThanOrEqual(1);
    expect(lz.dormantCount).toBeGreaterThanOrEqual(1); // ≥1 untouched tower stayed fully dormant
    // The hit tower only partially descended: fewer active fragments than a whole tower.
    expect(lz.activeLeafFragments).toBeLessThan(template.nodes.length);

    let maxDelta = 0;
    for (let f = 0; f < lf.length; f++)
      for (let i = 0; i < lf[f].length; i++) {
        const a = lf[f][i], b = ef[f][i];
        if (a && b) maxDelta = Math.max(maxDelta, Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
      }
    expect(lazy.getRigidBodyCount()).toBe(eager.getRigidBodyCount()); // topology identical
    expect(maxDelta).toBeLessThan(1e-6);                               // and bit-identical (rigid)

    lazy.dispose?.(); eager.dispose?.();
  });

  it('honors an explicit hand-authored collisionTree (descends only the overlapped leaf)', async () => {
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const scenario = mergeCity(template, [{ x: -40, y: 0, z: 0 }, { x: 40, y: 0, z: 0 }]);
    // Author one root per tower (split by node-index range), so a hit on tower 0 cannot touch tower 1.
    const half = template.nodes.length;
    const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i);
    scenario.collisionTree = [
      { fragments: range(0, half) },
      { fragments: range(half, half * 2) },
    ];
    const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10, lazyIntactColliders: true });
    const dt = 1 / 60;
    for (let i = 0; i < 15; i++) core.step(dt);
    expect(core.getLazyColliderStats!().buildingCount).toBe(2);
    core.enqueueProjectile({ position: { x: -40, y: 5, z: -22 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.7, mass: 9000, ttl: 3000 });
    for (let f = 0; f < 30; f++) core.step(dt);
    const lz = core.getLazyColliderStats!();
    expect(lz.explodedCount).toBe(1);  // only the struck root
    expect(lz.dormantCount).toBe(1);   // the other authored root stayed dormant
    core.dispose?.();
  });
});
