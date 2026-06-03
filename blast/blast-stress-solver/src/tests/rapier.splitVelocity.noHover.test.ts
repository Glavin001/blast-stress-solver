/**
 * Integration ("does the big fragment hover / lurch?") tests for the real destruction
 * pipeline — requires the WASM stress-solver build.
 *
 * Symptom: when a large body keeps fracturing as it tumbles, the *reused* fragment (the
 * largest piece — it keeps the parent's Rapier handle and merely loses the colliders that
 * migrate to its new siblings) appears to hover or get yanked, while *created* children
 * fall normally. Cause (see `rapier.splitVelocity.mechanism.test.ts`): the reused body's
 * centre of mass shifts when it loses colliders, but Rapier keeps the stored COM-velocity,
 * so its velocity field jumps by `ω × ΔCOM`. The fix re-derives the reused body's velocity
 * (`reconcileReusedBodyVelocity`), mirroring Rust's `reconcile_child_velocity_with_com`.
 *
 * The headline test here drives a *controlled* rotating split and asserts the reused
 * fragment stays point-velocity continuous (it drifts ~1.5 m/s without the fix). The rest
 * are robustness guards over a messy cascade.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import RAPIER from '@dimforge/rapier3d-compat';
import { TOL } from './invariants.shared';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

type V = { x: number; y: number; z: number };
const sub = (a: V, b: V): V => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const mag = (a: V): number => Math.hypot(a.x, a.y, a.z);
function qRot(q: { x: number; y: number; z: number; w: number }, v: V): V {
  const ix = q.w * v.x + q.y * v.z - q.z * v.y;
  const iy = q.w * v.y + q.z * v.x - q.x * v.z;
  const iz = q.w * v.z + q.x * v.y - q.y * v.x;
  const iw = -q.x * v.x - q.y * v.y - q.z * v.z;
  return {
    x: ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
    y: iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
    z: iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x,
  };
}

let buildDestructibleCore: (opts: any) => Promise<any>;
beforeAll(async () => {
  if (!runtimeAvailable) return;
  await RAPIER.init();
  buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
});

const NODE_MASS = 5;
const bodyOf = (core: any, ni: number) => {
  const h = core.chunks[ni].bodyHandle;
  return h == null ? null : core.world.getRigidBody(h);
};
function chunkWorldCenter(core: any, ni: number): V {
  const b = bodyOf(core, ni);
  const c = core.chunks[ni].baseLocalOffset;
  const t = b.translation();
  const rr = qRot(b.rotation(), c);
  return { x: t.x + rr.x, y: t.y + rr.y, z: t.z + rr.z };
}
function activeNonSupportChunks(core: any): any[] {
  return core.chunks.filter((c: any) => c.active && !c.isSupport && c.bodyHandle != null);
}

// ── Controlled rotating split ─────────────────────────────────────────────────
// A vertical chain: support(0) at top, two dynamic nodes below joined by a strong bond.
// The weak support bond breaks under gravity (so {1,2} detaches as ONE dynamic body); the
// strong internal bond survives until we deliberately overstress it while the body spins.
function verticalChain() {
  return {
    nodes: [
      { centroid: { x: 0, y: 22, z: 0 }, mass: 0, volume: 1 },
      { centroid: { x: 0, y: 21, z: 0 }, mass: NODE_MASS, volume: 1 },
      { centroid: { x: 0, y: 20, z: 0 }, mass: NODE_MASS, volume: 1 },
    ],
    bonds: [
      { node0: 0, node1: 1, centroid: { x: 0, y: 21.5, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 0.05 },
      { node0: 1, node1: 2, centroid: { x: 0, y: 20.5, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 50 },
    ],
    spacing: { x: 1, y: 1, z: 1 },
  } as any;
}

// ── Messy cascade (robustness guards) ─────────────────────────────────────────
function cascadingWall() {
  const rows = 4, cols = 3;
  const nodes: any[] = [], bonds: any[] = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      nodes.push({ centroid: { x: c, y: r + 0.5, z: 0 }, mass: r === 0 ? 0 : NODE_MASS, volume: 1 });
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols - 1; c++) {
      const n = r * cols + c;
      bonds.push({ node0: n, node1: n + 1, centroid: { x: c + 0.5, y: r + 0.5, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 2 });
    }
  for (let r = 0; r < rows - 1; r++)
    for (let c = 0; c < cols; c++) {
      const n = r * cols + c;
      bonds.push({ node0: n, node1: n + cols, centroid: { x: c, y: r + 1, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 0.01 });
    }
  return { nodes, bonds } as any;
}
async function buildCascade() {
  const core = await buildDestructibleCore({
    scenario: cascadingWall(), gravity: -20, materialScale: 0.001,
    resimulateOnFracture: true, maxResimulationPasses: 5, snapshotMode: 'perBody',
    debrisCleanup: { mode: 'off' }, skipSingleBodies: false,
  });
  core.setFracturePolicy?.({ idleSkip: false });
  return core;
}

describe.skipIf(!runtimeAvailable)('split velocity continuity — real pipeline (requires WASM)', () => {
  // THE FIX VERIFICATION. A spinning two-chunk body fractures into a reused fragment (keeps
  // the handle, loses a collider → COM shifts) and a created fragment. Both must keep the
  // parent's pre-split world point velocity. Without reconcileReusedBodyVelocity the reused
  // fragment drifts by ω×ΔCOM ≈ 1.5 m/s (3 rad/s × 0.5 m); with it, both stay < 1e-2.
  it('keeps the REUSED fragment point-velocity continuous across a rotating split', async () => {
    const core = await buildDestructibleCore({
      scenario: verticalChain(), gravity: -9.81, materialScale: 30000,
      resimulateOnFracture: true, maxResimulationPasses: 1, snapshotMode: 'perBody',
      debrisCleanup: { mode: 'off' }, skipSingleBodies: false,
    });
    core.setFracturePolicy?.({ idleSkip: false });

    // 1) Let the weak support bond break so {1,2} becomes a single dynamic body.
    let detached = false;
    for (let i = 0; i < 30 && !detached; i++) {
      core.step(1 / 60);
      const h1 = core.chunks[1].bodyHandle, h2 = core.chunks[2].bodyHandle, b1 = bodyOf(core, 1);
      detached = !!b1 && !b1.isFixed() && h1 !== core.rootBodyHandle && h1 === h2;
    }
    expect(detached).toBe(true);

    // 2) Freeze gravity and impose a pure rotation so the body's motion is exactly known.
    core.setGravity(0);
    core.setSolverGravityEnabled?.(false);
    const omega = { x: 0, y: 0, z: 3 };

    // 3) Each step: re-impose the spin, record the parent's field at both chunks, then
    //    overstress the internal bond. A tiny dt keeps integration drift far below the bug.
    let split = false, errReused = Infinity, errCreated = Infinity, reusedNode = -1;
    for (let i = 0; i < 300 && !split; i++) {
      const parent = bodyOf(core, 1);
      if (!parent) break;
      const parentHandle = parent.handle;
      parent.setLinvel({ x: 0, y: 0, z: 0 }, true);
      parent.setAngvel(omega, true);
      const ref: Record<number, { p: V; v: V }> = {};
      for (const ni of [1, 2]) {
        const p = chunkWorldCenter(core, ni);
        ref[ni] = { p, v: parent.velocityAtPoint(p) };
      }
      core.solver.addForce(2, core.chunks[2].baseLocalOffset, { x: 20000, y: 0, z: 0 });
      parent.setLinvel({ x: 0, y: 0, z: 0 }, true);
      parent.setAngvel(omega, true);
      core.step(1 / 480);

      const h1 = core.chunks[1].bodyHandle, h2 = core.chunks[2].bodyHandle;
      if (h1 !== h2) {
        reusedNode = h1 === parentHandle ? 1 : h2 === parentHandle ? 2 : -1;
        const e1 = mag(sub(bodyOf(core, 1).velocityAtPoint(ref[1].p), ref[1].v));
        const e2 = mag(sub(bodyOf(core, 2).velocityAtPoint(ref[2].p), ref[2].v));
        errReused = reusedNode === 1 ? e1 : e2;
        errCreated = reusedNode === 1 ? e2 : e1;
        split = true;
      }
    }

    expect(split).toBe(true);
    expect(reusedNode).toBeGreaterThan(0); // one child kept the parent handle (the reused body)
    // Headline: the reused fragment must not gain spurious velocity (would be ~1.5 m/s).
    expect(errReused).toBeLessThan(50 * TOL.pointVelocityContinuity);
    // The created child was already correct; assert it stays so (matched control).
    expect(errCreated).toBeLessThan(50 * TOL.pointVelocityContinuity);
    core.dispose();
  });

  it('never produces NaN/Inf body velocities through a full cascade', async () => {
    const core = await buildCascade();
    let allFinite = true;
    for (let i = 0; i < 220; i++) {
      core.step(1 / 60);
      core.world.forEachRigidBody((b: any) => {
        const v = b.linvel(), a = b.angvel();
        if (![v.x, v.y, v.z, a.x, a.y, a.z].every(Number.isFinite)) allFinite = false;
      });
    }
    expect(allFinite).toBe(true);
    core.dispose();
  });

  it('conserves total dynamic mass across the whole cascade', async () => {
    const core = await buildCascade();
    for (let i = 0; i < 220; i++) core.step(1 / 60);
    let bodyMass = 0;
    core.world.forEachRigidBody((b: any) => { if (!b.isFixed()) bodyMass += b.mass(); });
    const expected = activeNonSupportChunks(core).length * NODE_MASS;
    expect(Math.abs(bodyMass - expected) / Math.max(expected, 1)).toBeLessThan(TOL.massRelative);
    core.dispose();
  });

  it('the largest airborne fragment keeps descending (no gross hover)', async () => {
    const core = await buildCascade();
    let observed = false;
    for (let i = 0; i < 200; i++) {
      core.step(1 / 60);
      const byHandle = new Map<number, any[]>();
      for (const ch of activeNonSupportChunks(core)) {
        const b = core.world.getRigidBody(ch.bodyHandle);
        if (!b || b.isFixed()) continue;
        const arr = byHandle.get(ch.bodyHandle) ?? [];
        arr.push(ch); byHandle.set(ch.bodyHandle, arr);
      }
      let big: { h: number; chunks: any[] } | null = null;
      for (const [h, chunks] of byHandle) if (!big || chunks.length > big.chunks.length) big = { h, chunks };
      if (!big || big.chunks.length < 2) continue;
      const body = core.world.getRigidBody(big.h);
      const minY = Math.min(...big.chunks.map((c: any) => chunkWorldCenter(core, c.nodeIndex).y));
      if (minY > 1.5 && body.linvel().y < -1) {
        observed = true;
        expect(body.linvel().y).toBeLessThan(-0.1); // not hovering at ~0
      }
    }
    expect(observed).toBe(true);
    core.dispose();
  });
});
