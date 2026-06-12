/**
 * Settled-debris freeze lifecycle (`debrisSettle: { mode: 'freeze' }`, opt-in).
 *
 * Small debris bodies that stay Rapier-asleep for `settleDelayMs` convert to FIXED —
 * leaving the dynamic solve, the rollback snapshot, and the island/contact graphs —
 * and thaw back to dynamic when a strong contact lands on them or a neighbouring body
 * is removed. This is an explicit behaviour change, so:
 *   - default 'off' is pinned as exactly the legacy behaviour (nothing ever freezes),
 *   - freezing is pinned to actually happen on a settled pile (with the dynamic-body
 *     count dropping accordingly), and
 *   - an impact on the frozen pile is pinned to thaw bodies (frozenCount drops and a
 *     previously-frozen body moves again).
 *
 * NOTE: `settleDelayMs` is wall-clock (like projectile TTL and debris cleanup), so the
 * tests use a tiny delay — headless steps run much faster than real time.
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
const settleStats = (core: CoreInstance) =>
  (core as unknown as { getDebrisSettleStats: () => { mode: string; frozenCount: number } }).getDebrisSettleStats();

const OPTS = {
  gravity: -9.81, friction: 0.25, restitution: 0, contactForceScale: 30,
  debrisCollisionMode: 'all' as const,
  // Make debris fall asleep promptly so the wall-clock settle delay can trigger inside
  // a fast headless run.
  sleepMode: 'always' as const, sleepLinearThreshold: 0.5, sleepAngularThreshold: 0.5,
  // Keep the pile around: the default cleanup would start despawning debris.
  debrisCleanup: { mode: 'off' as const },
};

/** Shatter the 8×8×3 tower into a pile and run until it settles. */
async function shatterAndSettle(rapier: typeof RapierEntry, scen: typeof Scenarios, extra: Record<string, unknown>) {
  const scenario = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 3, floorHeight: 3 });
  const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 2e6, ...extra } as never);
  const dt = 1 / 60;
  for (let i = 0; i < 10; i++) core.step(dt);
  core.enqueueProjectile({ position: { x: 0, y: 2, z: -8 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.7, mass: 9000, ttl: 0.5 });
  for (let i = 0; i < 240; i++) core.step(dt); // collapse + settle
  return core;
}

function countDynamic(core: CoreInstance): number {
  let n = 0;
  core.world.forEachRigidBody((b: { isDynamic?: () => boolean }) => {
    if (b.isDynamic?.()) n++;
  });
  return n;
}

describe.skipIf(!runtimeAvailable)('Settled-debris freeze lifecycle', () => {
  it("default 'off': nothing ever freezes (legacy behaviour preserved)", async () => {
    const { rapier, scen } = await load();
    const core = await shatterAndSettle(rapier, scen, {});
    expect(core.getRigidBodyCount()).toBeGreaterThan(10); // the shatter actually happened
    const stats = settleStats(core);
    expect(stats.mode).toBe('off');
    expect(stats.frozenCount).toBe(0);
    core.dispose?.();
  });

  it('freeze mode converts settled debris to fixed bodies (dynamic count drops)', async () => {
    const { rapier, scen } = await load();
    const core = await shatterAndSettle(rapier, scen, {
      debrisSettle: { mode: 'freeze', settleDelayMs: 50, maxCollidersForSettle: 4 },
    });
    const dynBefore = countDynamic(core);
    // Give the wall-clock delay room, then sweep a few more frames.
    await new Promise((r) => setTimeout(r, 120));
    for (let i = 0; i < 30; i++) core.step(1 / 60);

    const stats = settleStats(core);
    expect(stats.frozenCount).toBeGreaterThan(0);          // a settled pile froze
    expect(countDynamic(core)).toBeLessThan(dynBefore);    // and left the dynamic solve
    core.dispose?.();
  });

  it('a strong impact thaws frozen debris and it moves again', async () => {
    const { rapier, scen } = await load();
    const core = await shatterAndSettle(rapier, scen, {
      debrisSettle: { mode: 'freeze', settleDelayMs: 50, maxCollidersForSettle: 4, unfreezeImpulse: 50 },
    });
    await new Promise((r) => setTimeout(r, 120));
    for (let i = 0; i < 30; i++) core.step(1 / 60);
    const frozenBefore = settleStats(core).frozenCount;
    // Stretch the delay so nothing RE-freezes inside the assertion window below —
    // frozenCount is intentionally not monotonic in normal operation (thawed debris
    // re-settles and freezes again; the first run of this test proved exactly that).
    core.setDebrisSettle!({ settleDelayMs: 60_000 });
    expect(frozenBefore).toBeGreaterThan(0);

    // Snapshot poses of currently-frozen chunks, then slam the pile.
    const frozenPoses = core.chunks
      .filter((c) => c.bodyHandle != null && c.worldPosition && core.world.getRigidBody(c.bodyHandle)?.isFixed())
      .map((c) => ({ nodeIndex: c.nodeIndex, y: c.worldPosition!.y, z: c.worldPosition!.z }));
    expect(frozenPoses.length).toBeGreaterThan(0);

    core.enqueueProjectile({ position: { x: 0, y: 1.2, z: -8 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.9, mass: 9000, ttl: 0.5 });
    for (let i = 0; i < 90; i++) core.step(1 / 60);

    expect(settleStats(core).frozenCount).toBeLessThan(frozenBefore); // some thawed
    // At least one previously-frozen chunk actually moved after thawing.
    let moved = 0;
    for (const p of frozenPoses) {
      const c = core.chunks[p.nodeIndex];
      if (!c.worldPosition) continue;
      if (Math.abs(c.worldPosition.y - p.y) > 0.05 || Math.abs(c.worldPosition.z - p.z) > 0.05) moved++;
    }
    expect(moved).toBeGreaterThan(0);
    core.dispose?.();
  });

  it("live-disabling thaws everything (setDebrisSettle({ mode: 'off' }))", async () => {
    const { rapier, scen } = await load();
    const core = await shatterAndSettle(rapier, scen, {
      debrisSettle: { mode: 'freeze', settleDelayMs: 50, maxCollidersForSettle: 4 },
    });
    await new Promise((r) => setTimeout(r, 120));
    for (let i = 0; i < 30; i++) core.step(1 / 60);
    expect(settleStats(core).frozenCount).toBeGreaterThan(0);

    core.setDebrisSettle!({ mode: 'off' });
    core.step(1 / 60); // thaws apply on the next pre-step sweep
    expect(settleStats(core).frozenCount).toBe(0);
    // No debris body may remain fixed: every fixed chunk body left must be a support.
    let frozenLeft = 0;
    for (const c of core.chunks) {
      if (c.bodyHandle == null || c.isSupport) continue;
      const body = core.world.getRigidBody(c.bodyHandle);
      if (body?.isFixed() && c.detached) frozenLeft++;
    }
    expect(frozenLeft).toBe(0);
    core.dispose?.();
  });
});
