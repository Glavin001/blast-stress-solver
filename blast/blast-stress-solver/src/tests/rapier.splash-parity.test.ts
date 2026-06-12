/**
 * Solver-side (WASM) splash expansion vs the JS expansion loop it replaces.
 *
 * The WASM path ships the static splash adjacency once and submits only the resolved
 * contact hits each frame; the solver expands hit → same-actor neighbours internally in
 * the same (contact, neighbour) order and with the same rounding sequence
 * (float(double(f32 force) × double weight))) as the JS loop. These tests pin that the
 * two paths are BYTE-IDENTICAL through real impacts — same chunk trajectories, same
 * fracture topology, same surviving bonds — and that the destroyed-node filter stays in
 * lockstep when the damage system is driving destruction.
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

function splashPipeline(core: CoreInstance): { wasmActive: boolean } {
  return (core as unknown as { __splashPipeline: () => { wasmActive: boolean } }).__splashPipeline();
}

function trajectory(core: CoreInstance, frames: number, dt: number, onFrame?: (f: number) => void) {
  const out: Array<Array<{ x: number; y: number; z: number } | null>> = [];
  for (let f = 0; f < frames; f++) {
    onFrame?.(f);
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

describe.skipIf(!runtimeAvailable)('WASM splash expansion parity', () => {
  it('activates on this runtime and respects the splashInWasm opt-out', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 4, depth: 4, floorCount: 2, floorHeight: 3 });

    const on = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10 });
    expect(splashPipeline(on).wasmActive).toBe(true);
    on.dispose?.();

    const off = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e10, splashInWasm: false });
    expect(splashPipeline(off).wasmActive).toBe(false);
    off.dispose?.();
  });

  it('a fracturing impact is byte-identical between the WASM and JS splash paths', async () => {
    const { rapier, scen } = await load();
    // Soft enough to fracture: the splash forces materially decide which bonds break,
    // so any divergence in expansion order/rounding would change the outcome.
    const scenario = await scen.buildTowerScenario({ width: 8, depth: 8, floorCount: 4, floorHeight: 3 });
    const dt = 1 / 60;

    async function run(splashInWasm: boolean) {
      const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 2e6, splashInWasm });
      expect(splashPipeline(core).wasmActive).toBe(splashInWasm);
      for (let i = 0; i < 15; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: 0, y: 6, z: -22 }, velocity: { x: 0, y: 0, z: 70 }, radius: 0.6, mass: 1500, ttl: 3000 });
      const frames = trajectory(core, 60, dt);
      const summary = {
        bodies: core.getRigidBodyCount(),
        bonds: core.getActiveBondsCount(),
      };
      core.dispose?.();
      return { frames, summary };
    }

    const wasm = await run(true);
    const js = await run(false);

    expect(wasm.summary.bodies).toBe(js.summary.bodies); // same fracture topology
    expect(wasm.summary.bonds).toBe(js.summary.bonds);   // same surviving bonds
    expect(maxTrajectoryDelta(wasm.frames, js.frames)).toBe(0); // same trajectories, bit-for-bit
  });

  it('stays byte-identical when damage destroys nodes mid-impact (inactive-filter lockstep)', async () => {
    const { rapier, scen } = await load();
    const scenario = await scen.buildTowerScenario({ width: 6, depth: 6, floorCount: 3, floorHeight: 3 });
    const dt = 1 / 60;
    const damage = { enabled: true, kImpact: 0.02, strengthPerVolume: 800, autoDetachOnDestroy: true };

    async function run(splashInWasm: boolean) {
      const core = await rapier.buildDestructibleCore({ scenario, ...OPTS, materialScale: 1e9, damage, splashInWasm });
      for (let i = 0; i < 10; i++) core.step(dt);
      core.enqueueProjectile({ position: { x: 0, y: 4, z: -20 }, velocity: { x: 0, y: 0, z: 90 }, radius: 0.8, mass: 4000, ttl: 3000 });
      const frames = trajectory(core, 70, dt);
      const destroyed = core.chunks.filter((c) => c.destroyed).length;
      const summary = { bodies: core.getRigidBodyCount(), bonds: core.getActiveBondsCount(), destroyed };
      core.dispose?.();
      return { frames, summary };
    }

    const wasm = await run(true);
    const js = await run(false);

    expect(wasm.summary.destroyed).toBe(js.summary.destroyed);
    expect(wasm.summary.bodies).toBe(js.summary.bodies);
    expect(wasm.summary.bonds).toBe(js.summary.bonds);
    expect(maxTrajectoryDelta(wasm.frames, js.frames)).toBe(0);
  });
});
