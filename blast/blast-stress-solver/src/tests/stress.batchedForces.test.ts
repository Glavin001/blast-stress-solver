import { describe, it, expect } from 'vitest';
import type * as Runtime from '..';
import type { ExtStressBondDesc, ExtStressNodeDesc, ExtStressSolverSettings } from '../types';

// End-to-end equivalence check through the real WASM runtime: submitting a set
// of external forces in one batched FFI crossing (addAllForces) must leave the
// solver in exactly the same state as submitting them one at a time (addForce).
// Mirrors the native Rust test (batched_forces_test.rs) but exercises the
// JavaScript wrapper + freshly built WASM the demos actually load.

async function importRuntime(): Promise<typeof Runtime> {
  return (await import('../../dist/index.js')) as typeof Runtime;
}

const nodes: ExtStressNodeDesc[] = [
  { centroid: { x: -1, y: 0, z: 0 }, mass: 0, volume: 1 }, // support
  { centroid: { x: 1, y: 0, z: 0 }, mass: 0, volume: 1 }, // support
  { centroid: { x: 0, y: 1.5, z: 0 }, mass: 15, volume: 1 } // mass
];
const bonds: ExtStressBondDesc[] = [
  { centroid: { x: -0.5, y: 0.75, z: 0 }, normal: { x: 0.55, y: 0.83, z: 0 }, area: 0.6, node0: 0, node1: 2 },
  { centroid: { x: 0.5, y: 0.75, z: 0 }, normal: { x: -0.55, y: 0.83, z: 0 }, area: 0.6, node0: 1, node1: 2 },
  { centroid: { x: 0, y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 0.9, node0: 0, node1: 1 }
];
const settings: ExtStressSolverSettings = {
  compressionElasticLimit: 0.001,
  compressionFatalLimit: 0.002,
  tensionElasticLimit: 0.001,
  tensionFatalLimit: 0.002,
  shearElasticLimit: 0.001,
  shearFatalLimit: 0.002
};

// node_index, application point, force vector. Node 2 appears twice so the test
// also covers force accumulation on a repeated node (the hit + splash pattern).
const forces: Array<{ node: number; pos: [number, number, number]; vec: [number, number, number] }> = [
  { node: 2, pos: [0, 1.5, 0], vec: [1000, 0, 0] },
  { node: 0, pos: [-1, 0, 0], vec: [0, -250, 30] },
  { node: 1, pos: [1, 0, 0], vec: [-120, 80, -10] },
  { node: 2, pos: [0, 1.5, 0], vec: [200, -400, 15] }
];

describe('ExtStressSolver addAllForces (batched contact injection)', () => {
  it('exposes the batched-force entry point in the freshly built runtime', async () => {
    const { loadStressSolver } = await importRuntime();
    const rt = await loadStressSolver();
    const solver = rt.createExtSolver({ nodes, bonds, settings });
    expect(solver.supportsBatchedForces()).toBe(true);
    solver.destroy();
  });

  it('leaves the solver in an identical state to per-call addForce', async () => {
    const { loadStressSolver } = await importRuntime();
    const rt = await loadStressSolver();

    // Reference: one FFI crossing per force.
    const perCall = rt.createExtSolver({ nodes, bonds, settings });
    for (const f of forces) {
      perCall.addForce(f.node, { x: f.pos[0], y: f.pos[1], z: f.pos[2] }, { x: f.vec[0], y: f.vec[1], z: f.vec[2] });
    }
    perCall.update();

    // Optimised: the identical set in a single batched crossing.
    const batched = rt.createExtSolver({ nodes, bonds, settings });
    const idx = new Uint32Array(forces.map((f) => f.node));
    const pos = new Float32Array(forces.flatMap((f) => f.pos));
    const vec = new Float32Array(forces.flatMap((f) => f.vec));
    const applied = batched.addAllForces(idx, pos, vec, forces.length);
    expect(applied).toBe(forces.length);
    batched.update();

    // The forces are large enough to do real, non-trivial work — assert that so
    // the equivalence below can't pass by both solvers simply doing nothing.
    expect(perCall.overstressedBondCount()).toBeGreaterThan(0);
    const ea = perCall.stressError();
    expect(ea.lin).toBeGreaterThan(0);

    // Equivalence: identical solver state after one batched vs many per-call
    // crossings. stressError (continuous) is the strongest signal — it comes
    // out bit-identical because both paths apply the same forces in the same
    // order — and overstressed/converged corroborate it.
    expect(batched.overstressedBondCount()).toBe(perCall.overstressedBondCount());
    expect(batched.converged()).toBe(perCall.converged());
    const eb = batched.stressError();
    expect(eb.lin).toBeCloseTo(ea.lin, 6);
    expect(eb.ang).toBeCloseTo(ea.ang, 6);

    perCall.destroy();
    batched.destroy();
  });

  it('treats an empty batch as a no-op', async () => {
    const { loadStressSolver } = await importRuntime();
    const rt = await loadStressSolver();

    const untouched = rt.createExtSolver({ nodes, bonds, settings });
    untouched.update();

    const empty = rt.createExtSolver({ nodes, bonds, settings });
    const applied = empty.addAllForces(new Uint32Array(0), new Float32Array(0), new Float32Array(0), 0);
    expect(applied).toBe(0);
    empty.update();

    expect(empty.overstressedBondCount()).toBe(untouched.overstressedBondCount());
    expect(empty.stressError().lin).toBeCloseTo(untouched.stressError().lin, 5);

    untouched.destroy();
    empty.destroy();
  });
});
