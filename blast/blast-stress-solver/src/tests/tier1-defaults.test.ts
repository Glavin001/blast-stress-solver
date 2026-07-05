/**
 * Tier-1 performance defaults (docs/perf-city-scale-roadmap.md §3).
 *
 * Pins the new `buildDestructibleCore` defaults and their safety rails:
 *   - the island-aware stress solve (+ settled-island skip) is ON by default, with the
 *     `islandSolver` option (boolean or granular object) as the opt-out — and an idle world
 *     under the default is bit-identical to the legacy whole-graph solve,
 *   - `lazyIntactColliders` stays opt-in at the library level but is forced OFF (with the
 *     option set) for free-floating scenarios, whose dynamic root would desync the LOD AABBs,
 *   - EXTERNAL bodies (vehicles / character controllers / anything the host app adds to the
 *     core's world) wake dormant buildings via the predictive pass — a driven body cannot pass
 *     through an intact dormant structure, and the collision resolves identically to eager.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import RAPIER from '@dimforge/rapier3d-compat';
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

/** Two bonded free-floating cubes — no supports (mass 0 nodes), so the root body is dynamic. */
function freeFloatingScenario(): ScenarioDesc {
  return {
    nodes: [
      { centroid: { x: -0.5, y: 5, z: 0 }, mass: 10, volume: 1 },
      { centroid: { x: 0.5, y: 5, z: 0 }, mass: 10, volume: 1 },
    ],
    bonds: [
      { node0: 0, node1: 1, centroid: { x: 0, y: 5, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 1 },
    ],
    parameters: {},
  } as ScenarioDesc;
}

const OPTS = { gravity: -9.81, materialScale: 1e10, friction: 0.25, restitution: 0, contactForceScale: 30, debrisCollisionMode: 'all' as const, damage: { enabled: false } };

describe.skipIf(!runtimeAvailable)('Island-aware solve default', () => {
  it('is ON (with settled-skip) by default, and the option opts out', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 4, depth: 4, floorCount: 2, floorHeight: 3 });

    const def = await rapier.buildDestructibleCore({ scenario, ...OPTS });
    expect(def.getIslandSolverStats!().enabled).toBe(true);
    expect(def.getIslandSolverStats!().skipSettled).toBe(true);
    def.dispose?.();

    const off = await rapier.buildDestructibleCore({ scenario, ...OPTS, islandSolver: false });
    expect(off.getIslandSolverStats!().enabled).toBe(false);
    off.dispose?.();

    const granular = await rapier.buildDestructibleCore({ scenario, ...OPTS, islandSolver: { enabled: true, skipSettled: false } });
    expect(granular.getIslandSolverStats!().enabled).toBe(true);
    expect(granular.getIslandSolverStats!().skipSettled).toBe(false);
    granular.dispose?.();
  });

  it('an idle two-tower city under the default is bit-identical to the whole-graph solve', async () => {
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 6, depth: 6, floorCount: 3, floorHeight: 3 });
    const scenario = mergeCity(template, [{ x: -15, y: 0, z: 0 }, { x: 15, y: 0, z: 0 }]);

    const island = await rapier.buildDestructibleCore({ scenario, ...OPTS });                       // default: island ON
    const whole = await rapier.buildDestructibleCore({ scenario, ...OPTS, islandSolver: false });   // legacy whole-graph

    const dt = 1 / 60;
    let maxDelta = 0;
    for (let f = 0; f < 60; f++) {
      island.step(dt); whole.step(dt);
      for (let i = 0; i < island.chunks.length; i++) {
        const a = island.chunks[i].worldPosition, b = whole.chunks[i].worldPosition;
        if (a && b) maxDelta = Math.max(maxDelta, Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
      }
    }
    // An intact, converged structure fractures in neither mode and the rigid bodies never move:
    // the solve-mode change cannot leak into the simulation.
    expect(maxDelta).toBe(0);
    expect(island.getActiveBondsCount()).toBe(whole.getActiveBondsCount());
    island.dispose?.(); whole.dispose?.();
  });

  it('fracturing still works under the default (impact breaks bodies off)', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 2e6 });
    const dt = 1 / 60;
    for (let i = 0; i < 15; i++) core.step(dt);
    core.enqueueProjectile({ position: { x: 0, y: 5, z: -22 }, velocity: { x: 0, y: 0, z: 70 }, radius: 0.6, mass: 1500, ttl: 3000 });
    for (let i = 0; i < 40; i++) core.step(dt);
    expect(core.getRigidBodyCount()).toBeGreaterThan(2); // ground + root + debris
    core.dispose?.();
  });
});

describe.skipIf(!runtimeAvailable)('Lazy intact colliders safety rails', () => {
  it('is forced OFF for a free-floating scenario (dynamic root), even when requested', async () => {
    const { rapier } = await load();
    const core = await rapier.buildDestructibleCore({ scenario: freeFloatingScenario(), ...OPTS, lazyIntactColliders: true });
    expect(core.getLazyColliderStats!().enabled).toBe(false);
    core.setLazyIntactColliders!(true); // the live toggle must refuse too
    expect(core.getLazyColliderStats!().enabled).toBe(false);
    core.dispose?.();
  });

  it('an EXTERNAL body (e.g. a vehicle) wakes a dormant building and collides identically to eager', async () => {
    const { rapier, scen } = await load();
    const template = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    // Two towers; ram the one at x = -20 with an app-created ball the core knows nothing about.
    const scenario = mergeCity(template, [{ x: -20, y: 0, z: 0 }, { x: 20, y: 0, z: 0 }]);
    const dt = 1 / 60;

    function ramWithExternalBall(core: Awaited<ReturnType<typeof rapier.buildDestructibleCore>>) {
      for (let i = 0; i < 10; i++) core.step(dt);
      const world = core.world as RAPIER.World;
      const body = world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic().setTranslation(-20, 2, -15).setLinvel(0, 0, 25).setCcdEnabled(true),
      );
      world.createCollider(RAPIER.ColliderDesc.ball(0.8).setDensity(400).setFriction(0.3), body);
      const track: Vec3[] = [];
      for (let f = 0; f < 90; f++) {
        core.step(dt);
        const t = body.translation();
        track.push({ x: t.x, y: t.y, z: t.z });
      }
      return track;
    }

    const lazy = await rapier.buildDestructibleCore({ scenario, ...OPTS, lazyIntactColliders: true });
    const eager = await rapier.buildDestructibleCore({ scenario, ...OPTS, lazyIntactColliders: false });
    expect(lazy.getLazyColliderStats!().dormantCount).toBe(2);

    const lt = ramWithExternalBall(lazy);
    const et = ramWithExternalBall(eager);

    // The predictive pass saw the external mover: the rammed building materialized…
    const stats = lazy.getLazyColliderStats!();
    expect(stats.explodedCount).toBeGreaterThanOrEqual(1);
    expect(stats.dormantCount).toBeGreaterThanOrEqual(1); // …but the far tower stayed dormant.

    // The ball did NOT sail through the dormant building: it stopped on the near side of the
    // tower's footprint instead of continuing to +z. (Tower walls span z ∈ [-4, 4] at x=-20.)
    const lastLazy = lt[lt.length - 1];
    expect(lastLazy.z).toBeLessThan(0);

    // And the whole encounter matched the eager world bit-for-bit (rigid material: nothing
    // fractures, so the trajectories must be identical — the strongest equivalence we can pin).
    let maxDelta = 0;
    for (let f = 0; f < lt.length; f++)
      maxDelta = Math.max(maxDelta, Math.abs(lt[f].x - et[f].x), Math.abs(lt[f].y - et[f].y), Math.abs(lt[f].z - et[f].z));
    expect(maxDelta).toBeLessThan(1e-6);

    lazy.dispose?.(); eager.dispose?.();
  });
});
