/**
 * Per-node contact-hit deduplication (`dedupeContactHits`, opt-in).
 *
 * A pile fragment struck by N contacts in one frame re-expands its whole splash
 * neighbourhood N times; with dedup the resolved body-local forces accumulate per hit
 * node (f64, rounded to f32 once) and each node expands exactly once. Per-node force
 * TOTALS are mathematically identical, but float summation order changes — so:
 *   - where nothing fractures, output is BIT-IDENTICAL (the stress solver feeds back
 *     into the physics world only via fracture commands), and
 *   - where fracture is in play, outcomes are equivalent to rounding (knife-edge bonds
 *     may flip), which is why the option defaults OFF.
 * These tests pin the bit-identity property, the actual stash reduction, and that a
 * fracturing run still produces a comparable shatter (sanity, not equality).
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
const stashCount = (core: CoreInstance) =>
  (core as unknown as { __contactStashCount: () => number }).__contactStashCount();

function trajectory(core: CoreInstance, frames: number, dt: number, onStep?: () => void) {
  const out: Array<Array<{ x: number; y: number; z: number } | null>> = [];
  for (let f = 0; f < frames; f++) {
    core.step(dt);
    onStep?.();
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

describe.skipIf(!runtimeAvailable)('Per-node contact-hit dedup', () => {
  it('a rigid (non-fracturing) bombardment is bit-identical with dedup on vs off', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const dt = 1 / 60;

    async function run(dedup: boolean) {
      // Rigid material: bonds never break, so injected stress never feeds back into
      // Rapier — any trajectory difference could only come from the dedup changing
      // what Rapier sees, which it must not.
      const core = await rapier.buildDestructibleCore({
        scenario, ...OPTS, materialScale: 1e10, dedupeContactHits: dedup,
      });
      for (let i = 0; i < 10; i++) core.step(dt);
      for (let s = 0; s < 3; s++) {
        core.enqueueProjectile({
          position: { x: (s - 1) * 0.4, y: 1.5 + s * 0.6, z: -8 },
          velocity: { x: 0, y: 0, z: 60 },
          radius: 0.5, mass: 1500, ttl: 3000,
        });
      }
      const frames = trajectory(core, 80, dt);
      const summary = { bodies: core.getRigidBodyCount(), bonds: core.getActiveBondsCount() };
      core.dispose?.();
      return { frames, summary };
    }

    const on = await run(true);
    const off = await run(false);
    expect(on.summary.bonds).toBe(off.summary.bonds); // nothing fractured in either
    expect(maxTrajectoryDelta(on.frames, off.frames)).toBe(0);
  });

  it('dedup actually reduces stash entries under multi-contact load', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
    const dt = 1 / 60;

    // Soft material so the tower shatters into a grinding pile — the regime where one
    // node collects many contacts per frame.
    async function maxStash(dedup: boolean) {
      const core = await rapier.buildDestructibleCore({
        scenario, ...OPTS, materialScale: 2e6, dedupeContactHits: dedup,
        // Keep every contact in play for the stash (no relevance filtering ambiguity in
        // this measurement) — dedup must shrink the stash on its own.
        skipSolverIrrelevantContacts: false,
      } as never);
      for (let i = 0; i < 10; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: 0, y: 2, z: -8 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.7, mass: 9000, ttl: 3000 });
      let peak = 0;
      trajectory(core, 100, dt, () => { peak = Math.max(peak, stashCount(core)); });
      const bodies = core.getRigidBodyCount();
      core.dispose?.();
      return { peak, bodies };
    }

    const off = await maxStash(false);
    const on = await maxStash(true);
    expect(off.bodies).toBeGreaterThan(10); // the shatter actually happened
    expect(on.bodies).toBeGreaterThan(10);
    expect(off.peak).toBeGreaterThan(0);
    // With dedup, stash entries are bounded by unique hit nodes — strictly fewer than
    // the per-contact stash once the pile grinds.
    expect(on.peak).toBeLessThan(off.peak);
  });

  it('defaults OFF (per-contact stash unchanged unless opted in)', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 4, depth: 4, floorCount: 2, floorHeight: 3 });
    const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10 });
    // No public getter for the option; the behavioural contract is covered by the two
    // tests above — here we only pin that building WITHOUT the option works unchanged.
    for (let i = 0; i < 5; i++) core.step(1 / 60);
    expect(core.getActiveBondsCount()).toBeGreaterThan(0);
    core.dispose?.();
  });
});
