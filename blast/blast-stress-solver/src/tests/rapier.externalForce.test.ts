/**
 * The virtual external-force primitive (applyExternalForce) must behave like a
 * real contact impact, without a colliding body:
 *   1. feed the STRESS SOLVER so a bonded node overstresses and fractures, and
 *   2. feed the PHYSICS so the freed piece flies — both in the SAME tick (the
 *      impact is re-applied to the freed body during the resimulation pass).
 * A control core with no force applied must stay intact and at rest, proving the
 * fracture + motion come from the registered force (not gravity or anything else).
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

type Vec3 = { x: number; y: number; z: number };

let buildDestructibleCore: (opts: any) => Promise<any>;
async function loadModules() {
  if (buildDestructibleCore) return;
  buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
}

// node0 = fixed support (top), node1 = mass hanging from it by one vertical bond.
// Shear (sideways) limits are tiny, compression is huge, gravity is OFF — so the
// ONLY thing that can break the bond or move node1 is a registered external force.
function twoNode() {
  return {
    nodes: [
      { centroid: { x: 0, y: 2, z: 0 }, mass: 0, volume: 1 },
      { centroid: { x: 0, y: 1, z: 0 }, mass: 5, volume: 1 },
    ],
    bonds: [
      { node0: 0, node1: 1, centroid: { x: 0, y: 1.5, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 1 },
    ],
  };
}

const opts = (extra: Record<string, unknown>) => ({
  scenario: twoNode(),
  gravity: 0, // isolate the external force as the only load
  materialScale: 1,
  resimulateOnFracture: true,
  maxResimulationPasses: 2,
  snapshotMode: 'perBody',
  skipSingleBodies: false,
  solverSettings: {
    maxSolverIterationsPerFrame: 25,
    compressionElasticLimit: 1e9, compressionFatalLimit: 1e10,
    tensionElasticLimit: 1, tensionFatalLimit: 2,
    shearElasticLimit: 1, shearFatalLimit: 2,
  },
  ...extra,
});

function node1Speed(core: any): number {
  const h = core.chunks[1].bodyHandle;
  const b = h == null ? null : core.world.getRigidBody(h);
  if (!b) return 0;
  const v = b.linvel() as Vec3;
  return Math.hypot(v.x, v.y, v.z);
}

describe.skipIf(!runtimeAvailable)('applyExternalForce primitive (requires WASM build)', () => {
  it('control: with no external force the bond holds and the node stays put', async () => {
    await loadModules();
    const core = await buildDestructibleCore(opts({}));
    for (let i = 0; i < 5; i++) core.step(1 / 60);
    expect(core.getActiveBondsCount()).toBe(1);
    expect(node1Speed(core)).toBeLessThan(0.5);
    core.dispose();
  });

  it('fractures the bond AND flings the freed piece in one tick', async () => {
    await loadModules();
    const core = await buildDestructibleCore(opts({}));
    expect(core.getActiveBondsCount()).toBe(1);

    // One sideways virtual impact on node1, then a single step.
    core.applyExternalForce(1, { x: 0, y: 1, z: 0 }, { x: 1e4, y: 0, z: 0 });
    core.step(1 / 60);

    // (1) solver fractured the bond …
    expect(core.getActiveBondsCount()).toBe(0);
    // … (2) and the freed piece is moving outward in the force direction — same tick.
    const h = core.chunks[1].bodyHandle;
    const body = core.world.getRigidBody(h);
    expect(body.isDynamic()).toBe(true);
    const v = body.linvel() as Vec3;
    expect(v.x).toBeGreaterThan(2); // flew along +x (the applied force)
    core.dispose();
  });

  it('does not keep pushing on later ticks (one-shot, like a contact)', async () => {
    await loadModules();
    const core = await buildDestructibleCore(opts({}));
    core.applyExternalForce(1, { x: 0, y: 1, z: 0 }, { x: 1e4, y: 0, z: 0 });
    core.step(1 / 60);
    const vx1 = (core.world.getRigidBody(core.chunks[1].bodyHandle).linvel() as Vec3).x;
    // No new force registered; velocity must not keep growing (force isn't persisting).
    for (let i = 0; i < 5; i++) core.step(1 / 60);
    const vx2 = (core.world.getRigidBody(core.chunks[1].bodyHandle).linvel() as Vec3).x;
    expect(Math.abs(vx2 - vx1)).toBeLessThan(0.5);
    core.dispose();
  });
});
