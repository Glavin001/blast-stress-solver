/**
 * Solver-irrelevant contact filtering (drainContactForces).
 *
 * The WASM stress solver DROPS forces applied to single-node actors (its actor force
 * path requires >1 graph node — a lone fragment carries no bonds to stress), and a
 * single-chunk body has no same-body splash neighbours. Yet during a debris grind the
 * overwhelming majority of contact events are debris↔debris on exactly such bodies, and
 * each one used to be buffered, rotation-resolved, stashed and splash-walked before the
 * solver discarded it. The filter skips that dead work at the source.
 *
 * These tests pin that the filter is OUTPUT-IDENTICAL — same chunk trajectories
 * (bit-for-bit), same fracture topology, same surviving bonds — while actually skipping
 * events once a debris field exists, and that damage processing still sees every event.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type * as RapierEntry from '../rapier';
import type * as Scenarios from '../scenarios';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

async function load() {
  const rapier = (await import('../../dist/rapier.js')) as typeof RapierEntry;
  const scen = (await import('../../dist/scenarios.js')) as typeof Scenarios;
  return { rapier, scen };
}

type CoreInstance = Awaited<ReturnType<typeof RapierEntry.buildDestructibleCore>>;

const skippedCount = (core: CoreInstance) =>
  (core as unknown as { __solverIrrelevantContactsSkipped: () => number }).__solverIrrelevantContactsSkipped();

function trajectory(core: CoreInstance, frames: number, dt: number) {
  const out: Array<Array<{ x: number; y: number; z: number } | null>> = [];
  for (let f = 0; f < frames; f++) {
    core.step(dt);
    out.push(core.chunks.map((c) => (c.worldPosition ? { ...c.worldPosition } : null)));
  }
  return out;
}

function maxTrajectoryDelta(
  a: Array<Array<{ x: number; y: number; z: number } | null>>,
  b: Array<Array<{ x: number; y: number; z: number } | null>>,
) {
  let d = 0;
  for (let f = 0; f < Math.min(a.length, b.length); f++) {
    for (let i = 0; i < a[f].length; i++) {
      const pa = a[f][i], pb = b[f][i];
      if (!pa || !pb) continue;
      d = Math.max(d, Math.abs(pa.x - pb.x), Math.abs(pa.y - pb.y), Math.abs(pa.z - pb.z));
    }
  }
  return d;
}

const OPTS = { gravity: -9.81, friction: 0.25, restitution: 0, contactForceScale: 30, debrisCollisionMode: 'all' as const };

describe.skipIf(!runtimeAvailable)('Solver-irrelevant contact filtering', () => {
  it('a collapse + debris grind is bit-identical with the filter on vs off', async () => {
    const { rapier, scen } = await load();
    // Soft material so the impact genuinely shatters: the run covers the pre-fracture
    // frames (multi-node hits — never filtered), the cascade, and 60+ frames of
    // debris-pile grinding (where almost every event is filterable).
    const scenario = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const dt = 1 / 60;

    async function run(filterOn: boolean) {
      const core = await rapier.buildDestructibleCore({
        scenario, ...OPTS, materialScale: 2e6, skipSolverIrrelevantContacts: filterOn,
      });
      for (let i = 0; i < 15; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: 0, y: 2, z: -8 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.7, mass: 9000, ttl: 3000 });
      const frames = trajectory(core, 110, dt);
      const summary = {
        bodies: core.getRigidBodyCount(),
        bonds: core.getActiveBondsCount(),
        lastSkipped: skippedCount(core),
      };
      core.dispose?.();
      return { frames, summary };
    }

    const on = await run(true);
    const off = await run(false);

    expect(on.summary.bodies).toBeGreaterThan(10);             // the shatter actually happened
    expect(maxTrajectoryDelta(on.frames, off.frames)).toBe(0); // bit-identical trajectories
    expect(on.summary.bodies).toBe(off.summary.bodies);        // same fracture topology
    expect(on.summary.bonds).toBe(off.summary.bonds);          // same surviving bonds
    expect(off.summary.lastSkipped).toBe(0);                   // off really buffers everything
  });

  it('actually skips events once single-chunk debris exists (and is on by default)', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const dt = 1 / 60;
    const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 2e6 }); // default: filter ON

    let maxSkipped = 0;
    for (let i = 0; i < 15; i++) core.step(dt);
    expect(skippedCount(core)).toBe(0); // intact structure: every node sits on a multi-chunk body

    core.enqueueProjectile({ position: { x: 0, y: 2, z: -8 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.7, mass: 9000, ttl: 3000 });
    for (let i = 0; i < 110; i++) {
      core.step(dt);
      maxSkipped = Math.max(maxSkipped, skippedCount(core));
    }
    expect(core.getRigidBodyCount()).toBeGreaterThan(10); // the shatter actually happened
    expect(maxSkipped).toBeGreaterThan(0);                // …and the pile produced filterable events
    core.dispose?.();
  });

  it('damage processing still sees filtered events (filter only gates the stress buffer)', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 6, depth: 6, floorCount: 3, floorHeight: 3 });
    const dt = 1 / 60;
    const damage = { enabled: true, kImpact: 0.02, strengthPerVolume: 600, autoDetachOnDestroy: true };

    async function run(filterOn: boolean) {
      const core = await rapier.buildDestructibleCore({
        scenario, ...OPTS, materialScale: 1e9, damage, skipSolverIrrelevantContacts: filterOn,
      });
      for (let i = 0; i < 10; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: 0, y: 1.5, z: -6 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.8, mass: 4000, ttl: 3000 });
      const frames = trajectory(core, 80, dt);
      const destroyed = core.chunks.filter((c) => c.destroyed).length;
      core.dispose?.();
      return { frames, destroyed };
    }

    const on = await run(true);
    const off = await run(false);
    expect(off.destroyed).toBeGreaterThan(0);                  // damage really fired (non-vacuous)
    expect(on.destroyed).toBe(off.destroyed);                  // identical damage outcomes
    expect(maxTrajectoryDelta(on.frames, off.frames)).toBe(0); // and identical motion
  });
});
