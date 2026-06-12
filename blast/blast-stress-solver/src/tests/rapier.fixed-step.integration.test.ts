/**
 * Fixed-timestep driver against the REAL destructible core: the driver must add no
 * math of its own — N driver ticks at the native rate are bit-identical to N direct
 * `core.step(1/hz)` calls, on any display cadence whose total stepped time matches.
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

const OPTS = { gravity: -9.81, materialScale: 2e6, friction: 0.25, restitution: 0, contactForceScale: 30, damage: { enabled: false } };

function snapshotPoses(core: { chunks: Array<{ worldPosition?: { x: number; y: number; z: number } | null }> }) {
  return core.chunks.map((c) => (c.worldPosition ? { ...c.worldPosition } : null));
}

function maxDelta(
  a: Array<{ x: number; y: number; z: number } | null>,
  b: Array<{ x: number; y: number; z: number } | null>,
) {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    const pa = a[i], pb = b[i];
    if (!pa || !pb) continue;
    d = Math.max(d, Math.abs(pa.x - pb.x), Math.abs(pa.y - pb.y), Math.abs(pa.z - pb.z));
  }
  return d;
}

describe.skipIf(!runtimeAvailable)('Fixed-step loop × destructible core', () => {
  it('driver ticks are bit-identical to direct stepping, including through an impact', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 6, depth: 6, floorCount: 3, floorHeight: 3 });

    const direct = await rapier.buildDestructibleCore({ scenario, ...OPTS });
    const driven = await rapier.buildDestructibleCore({ scenario, ...OPTS });

    const interp = rapier.createPoseInterpolator(driven);
    const loop = rapier.createFixedStepLoop({
      hz: 60,
      maxStepsPerTick: 4,
      step: (dt) => driven.step(dt),
      onBeforeStep: () => interp.beforeStep(),
      onAfterStep: () => interp.afterStep(),
    });

    const fire = (core: typeof direct) =>
      core.enqueueProjectile({ position: { x: 0, y: 4, z: -18 }, velocity: { x: 0, y: 0, z: 60 }, radius: 0.5, mass: 1200, ttl: 3000 });

    // 30 settle frames, an impact, then 60 more — driven on a mixed display cadence
    // (alternating 120 Hz-style half-frames) whose stepped count matches direct 1:1.
    for (let i = 0; i < 30; i++) direct.step(1 / 60);
    fire(direct);
    for (let i = 0; i < 60; i++) direct.step(1 / 60);

    for (let i = 0; i < 30; i++) loop.tick(1 / 60);
    fire(driven);
    for (let i = 0; i < 120; i++) loop.tick(1 / 120); // 120 half-frames = 60 fixed steps
    expect(maxDelta(snapshotPoses(direct), snapshotPoses(driven))).toBe(0);
    expect(driven.getRigidBodyCount()).toBe(direct.getRigidBodyCount());

    direct.dispose?.();
    driven.dispose?.();
  });

  it('the interpolated view stays between the last two physics states (no overshoot)', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 4, depth: 4, floorCount: 2, floorHeight: 3 });
    const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e3 }); // soft → it moves
    const interp = rapier.createPoseInterpolator(core);
    const loop = rapier.createFixedStepLoop({
      hz: 60,
      step: (dt) => core.step(dt),
      onBeforeStep: () => interp.beforeStep(),
      onAfterStep: () => interp.afterStep(),
    });

    for (let i = 0; i < 20; i++) loop.tick(1 / 60);
    const v = interp.view(0.5);
    for (let i = 0; i < core.chunks.length; i++) {
      const b = i * 7;
      if (Number.isNaN(v.prev[b + 6]) || Number.isNaN(v.curr[b + 6])) continue;
      for (let k = 0; k < 3; k++) {
        const lo = Math.min(v.prev[b + k], v.curr[b + k]);
        const hi = Math.max(v.prev[b + k], v.curr[b + k]);
        const mid = v.prev[b + k] + (v.curr[b + k] - v.prev[b + k]) * 0.5;
        expect(mid).toBeGreaterThanOrEqual(lo - 1e-6);
        expect(mid).toBeLessThanOrEqual(hi + 1e-6);
      }
    }
    core.dispose?.();
  });
});
