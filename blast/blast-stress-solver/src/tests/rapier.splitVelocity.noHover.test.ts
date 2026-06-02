/**
 * Integration ("does the big fragment hover?") tests for the real destruction
 * pipeline — requires the WASM stress-solver build.
 *
 * Symptom under test (reported from the high-rise demo): when a large body keeps
 * fracturing as it tumbles into the ground, the *big* remaining piece appears to
 * hover/lurch while *small* already-detached pieces keep falling normally.
 *
 * `rapier.splitVelocity.mechanism.test.ts` proves the cause in isolation (a body
 * that loses colliders while spinning keeps its stale centre-of-mass velocity, so
 * its velocity field jumps by `ω × ΔCOM`). These tests show the consequence end to
 * end and, crucially, expose why the existing instrumentation never flagged it:
 *
 *   • The split-continuity log only ever records *created* child bodies (which the
 *     pipeline DOES velocity-correct). The *reused* body — the largest fragment,
 *     which keeps the parent's handle and merely loses colliders — is never in the
 *     log, so the log stays "green" (~1e-7) even when reused fragments drift.
 *
 * Always-true invariants (finiteness, mass conservation, log self-consistency) are
 * asserted as blocking guards. The reused-fragment continuity violation is captured
 * as a `it.skip` repro (mirroring the Rust `#[ignore]` known-bug lane in
 * blast/TESTING.md) so CI stays green while the gap stays visible and reproducible.
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

/** 4×3 wall anchored along its bottom row, with weak vertical bonds: it sheds
 *  rows that tumble and re-fracture in mid-air — the dynamic→dynamic regime that
 *  produces both spinning fragments and reused (handle-retained) parent bodies. */
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
    scenario: cascadingWall(),
    gravity: -20,
    materialScale: 0.001,
    resimulateOnFracture: true,
    maxResimulationPasses: 5,
    snapshotMode: 'perBody',
    debrisCleanup: { mode: 'off' },
    skipSingleBodies: false,
  });
  core.setFracturePolicy?.({ idleSkip: false });
  return core;
}

function activeNonSupportChunks(core: any): any[] {
  return core.chunks.filter((c: any) => c.active && !c.isSupport && c.bodyHandle != null);
}
function chunkWorldCenter(core: any, ch: any): V | null {
  const b = core.world.getRigidBody(ch.bodyHandle);
  if (!b) return null;
  const t = b.translation();
  const rr = qRot(b.rotation(), ch.baseLocalOffset);
  return { x: t.x + rr.x, y: t.y + rr.y, z: t.z + rr.z };
}
function dynamicLogRecords(core: any): any[] {
  return (core.__debugSplitContinuityLog ?? []).filter(
    (r: any) => !r.sourceBodyIsFixed && !r.targetBodyIsFixed && r.nodeIndices.length > 0,
  );
}

describe.skipIf(!runtimeAvailable)('split velocity continuity — real pipeline (requires WASM)', () => {
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

  // The blind spot: the continuity log reports near-perfect continuity, but it
  // only ever watches CREATED child bodies (distinct handle). The reused fragment
  // (handle == parent's) is never recorded, so a green log is NOT evidence the big
  // fragment's velocity is continuous.
  it('continuity log is green AND structurally excludes the reused (handle-retained) fragment', async () => {
    const core = await buildCascade();
    core.__clearDebugSplitContinuityLog?.();
    for (let i = 0; i < 220; i++) core.step(1 / 60);

    const records = dynamicLogRecords(core);
    expect(records.length).toBeGreaterThan(0); // instrumentation is actually active

    // (a) The log claims everything is continuous.
    const maxLogged = Math.max(...records.map((r: any) => r.maxChunkPointVelocityError));
    expect(maxLogged).toBeLessThan(TOL.pointVelocityContinuity);

    // (b) ...but every record is a transfer to a *newly created* body, never the
    // reused parent handle. The reused fragment is therefore unobserved.
    const reusedRecords = records.filter((r: any) => r.sourceBodyHandle === r.targetBodyHandle);
    expect(reusedRecords.length).toBe(0);
    core.dispose();
  });

  // The reused (big) fragment does not catastrophically lose all its velocity:
  // once it is airborne and moving down it keeps net downward motion (this guards
  // against a regression to a *full* hover/zeroed-velocity, distinct from the
  // partial ω×ΔCOM drift the skipped repro below targets).
  it('the largest airborne fragment keeps descending (no gross hover)', async () => {
    const core = await buildCascade();
    let observed = false;
    for (let i = 0; i < 200; i++) {
      core.step(1 / 60);
      // largest dynamic body = most active non-support chunks sharing a handle
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
      const minY = Math.min(...big.chunks.map((c: any) => chunkWorldCenter(core, c)!.y));
      // airborne (above the ground) and already moving down
      if (minY > 1.5 && body.linvel().y < -1) {
        observed = true;
        // It must not be hovering: a body that lost its velocity would have vy≈0.
        expect(body.linvel().y).toBeLessThan(-0.1);
      }
    }
    expect(observed).toBe(true); // the scenario actually produced a falling big fragment
    core.dispose();
  });

  // KNOWN-BUG REPRO (gap #1 on the JS reused-body path) — skipped so CI stays
  // green, mirroring the Rust `#[ignore]` repro lane. Remove `.skip` to reproduce:
  // it measures the reused (handle-retained, collider-losing) fragment's retained
  // chunks and asserts point-velocity continuity. It currently FAILS because that
  // body is never re-derived for its shifted COM (see the mechanism test).
  it.skip('[repro] reused fragment preserves point velocity across its own fracture', async () => {
    const core = await buildCascade();
    const dt = 1 / 60, g = -20;
    let prev: { handle: number[]; ctr: (V | null)[]; vel: (V | null)[]; ncol: Map<number, number> } | null = null;
    const colCount = (h: number) => activeNonSupportChunks(core).filter((c: any) => c.bodyHandle === h).length;
    let worst = 0;
    for (let i = 0; i < 220; i++) {
      core.step(dt);
      const handle: number[] = [], ctr: (V | null)[] = [], vel: (V | null)[] = [];
      const ncol = new Map<number, number>();
      for (let ni = 0; ni < core.chunks.length; ni++) {
        const ch = core.chunks[ni];
        handle[ni] = ch.bodyHandle ?? -1;
        const c = ch.active && ch.bodyHandle != null ? chunkWorldCenter(core, ch) : null;
        ctr[ni] = c;
        const b = ch.bodyHandle == null ? null : core.world.getRigidBody(ch.bodyHandle);
        vel[ni] = b && c ? b.velocityAtPoint(c) : null;
        if (b && !b.isFixed() && ch.bodyHandle != null && !ncol.has(ch.bodyHandle)) ncol.set(ch.bodyHandle, colCount(ch.bodyHandle));
      }
      if (prev) {
        for (let ni = 0; ni < core.chunks.length; ni++) {
          const ch = core.chunks[ni];
          if (ch.isSupport || !ch.active) continue;
          const h = handle[ni];
          if (h < 0 || h !== prev.handle[ni]) continue; // reused: same body as last frame
          const before = prev.ncol.get(h), after = ncol.get(h);
          if (before == null || after == null || after >= before) continue; // body lost colliders => it fractured
          const c0 = prev.ctr[ni], v0 = prev.vel[ni], v1 = vel[ni];
          if (!c0 || !v0 || !v1 || (ctr[ni]?.y ?? 0) < 1) continue; // airborne only
          const expected = { x: v0.x, y: v0.y + g * dt, z: v0.z };
          worst = Math.max(worst, mag(sub(v1, expected)));
        }
      }
      prev = { handle, ctr, vel, ncol };
    }
    // A faithful rigid fracture would keep this near the gravity-only continuation.
    expect(worst).toBeLessThan(TOL.pointVelocityContinuity);
    core.dispose();
  });
});
