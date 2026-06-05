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
 * (Once a building is being violently shattered the exact rubble positions diverge within Rapier's
 *  order-sensitive contact-solver tolerance — topology stays identical — so we do not assert
 *  bit-identity through a multi-body fracture.)
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
    expect(lazy.getRigidBodyCount()).toBe(eager.getRigidBodyCount());

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
