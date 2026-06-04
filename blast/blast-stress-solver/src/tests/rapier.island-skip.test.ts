/**
 * Settled-island skipping (Stage 3): an island whose velocity inputs are bit-identical to its last
 * solve and that already converged is skipped — its solve would be a guaranteed no-op. This must be
 * OBSERVABLY IDENTICAL to not skipping (correctness is never traded), it must actually engage on a
 * settled scene (the optimization works), and a settled island must re-solve the SAME FRAME when its
 * load changes (paused, never evicted/frozen).
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type * as Runtime from '..';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

type Vec3 = { x: number; y: number; z: number };
type Node = { centroid: Vec3; mass: number; volume: number };
type Bond = { centroid: Vec3; normal: Vec3; area: number; node0: number; node1: number };
type Scenario = { nodes: Node[]; bonds: Bond[] };

async function loadRuntime() {
  return (await import('../../dist/index.js')) as typeof Runtime;
}

const rigid = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 1e12, compressionFatalLimit: 2e12,
  tensionElasticLimit: 1e12, tensionFatalLimit: 2e12,
  shearElasticLimit: 1e12, shearFatalLimit: 2e12,
};
// Settles under light gravity, overstresses decisively under heavy gravity (used for the wake test).
const breakable = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 5e4, compressionFatalLimit: 1e5,
  tensionElasticLimit: 50, tensionFatalLimit: 100,
  shearElasticLimit: 50, shearFatalLimit: 100,
};
const fracture = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 50, compressionFatalLimit: 100,
  tensionElasticLimit: 1, tensionFatalLimit: 2,
  shearElasticLimit: 1, shearFatalLimit: 2,
};

// Two arms sharing one static ground node → one actor, two islands.
function twoArmsSharingGround(len = 4): Scenario {
  const nodes: Node[] = [{ centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 }];
  const bonds: Bond[] = [];
  const chain = (sign: number) => {
    let prev = 0;
    for (let i = 1; i <= len; i++) {
      const idx = nodes.length;
      nodes.push({ centroid: { x: sign * i, y: 0, z: 0 }, mass: 10, volume: 1 });
      bonds.push({ centroid: { x: sign * (i - 0.5), y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 1, node0: prev, node1: idx });
      prev = idx;
    }
  };
  chain(1); chain(-1);
  return { nodes, bonds };
}

// N arms radiating from a shared static ground node → one actor, N islands.
function manyArms(arms: number, len = 3): Scenario {
  const nodes: Node[] = [{ centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 }];
  const bonds: Bond[] = [];
  const dirs = [{ x: 1, z: 0 }, { x: -1, z: 0 }, { x: 0, z: 1 }, { x: 0, z: -1 }];
  for (let a = 0; a < arms; a++) {
    const d = dirs[a % dirs.length];
    let prev = 0;
    for (let i = 1; i <= len; i++) {
      const idx = nodes.length;
      nodes.push({ centroid: { x: d.x * i, y: 0, z: d.z * i }, mass: 10, volume: 1 });
      bonds.push({ centroid: { x: d.x * (i - 0.5), y: 0, z: d.z * (i - 0.5) }, normal: { x: d.x || 1, y: 0, z: d.z }, area: 1, node0: prev, node1: idx });
      prev = idx;
    }
  }
  return { nodes, bonds };
}

const sortedPartition = (actors: Array<{ actorIndex: number; nodes: number[] }>): number[][] =>
  actors.map((a) => a.nodes.slice().sort((x, y) => x - y)).sort((a, b) => (a[0] ?? -1) - (b[0] ?? -1));

describe.skipIf(!runtimeAvailable)('Settled-island skipping (requires WASM build)', () => {
  it('engages on a stable multi-island scene (all islands skipped once settled)', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    const solver = rt.createExtSolver({ ...twoArmsSharingGround(), settings: rigid });
    solver.setIslandAware(true);
    solver.setSkipSettled(true);
    let skipped = 0;
    for (let f = 0; f < 6; f++) { solver.addGravity({ x: 0, y: -5, z: 0 }); solver.update(); skipped = solver.islandsSkipped(); }
    expect(solver.islandCount()).toBe(2);
    expect(solver.overstressedBondCount()).toBe(0);
    expect(skipped).toBe(2);                 // both islands recognized as settled and skipped
    solver.destroy();
  });

  it('is observationally identical to not skipping, across settle → wake → fracture', async () => {
    // Light load for a while (islands settle and skip), then a heavy load that wakes and fractures
    // them. With skipping on vs off the entire evolution must be identical.
    const gravities = [-0.2, -0.2, -0.2, -0.2, -0.2, -400, -400, -400, -400, -400];
    const run = async (skip: boolean) => {
      const rt = await (await loadRuntime()).loadStressSolver();
      const solver = rt.createExtSolver({ ...twoArmsSharingGround(), settings: breakable });
      solver.setIslandAware(true);
      solver.setSkipSettled(skip);
      const over: number[] = [];
      let maxSkipped = 0;
      for (const g of gravities) {
        solver.addGravity({ x: 0, y: g, z: 0 });
        solver.update();
        over.push(solver.overstressedBondCount());
        maxSkipped = Math.max(maxSkipped, solver.islandsSkipped());
        if (solver.overstressedBondCount() > 0) solver.applyFractureCommands(solver.generateFractureCommandsPerActor());
      }
      const partition = sortedPartition(solver.actors());
      solver.destroy();
      return { over, partition, maxSkipped };
    };
    const noskip = await run(false);
    const withskip = await run(true);
    expect(withskip.over).toEqual(noskip.over);            // identical fracture decisions, every frame
    expect(withskip.partition).toEqual(noskip.partition);  // identical resulting actor partition
    expect(noskip.maxSkipped).toBe(0);                     // skip truly off in the baseline run
    expect(withskip.maxSkipped).toBeGreaterThan(0);        // ...and it really did skip during the settle phase
  });

  it('re-solves a settled island the SAME frame its load changes (paused, never frozen)', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    const solver = rt.createExtSolver({ ...twoArmsSharingGround(), settings: breakable });
    solver.setIslandAware(true);
    solver.setSkipSettled(true);
    // Settle under a light, constant load — no overstress, both islands become skippable.
    for (let f = 0; f < 5; f++) { solver.addGravity({ x: 0, y: -0.2, z: 0 }); solver.update(); }
    expect(solver.islandsSkipped()).toBe(2);
    expect(solver.overstressedBondCount()).toBe(0);
    // A heavy load arrives. The velocity input changes, so the islands must re-solve THIS frame and
    // detect the overstress immediately — not skipped, not delayed.
    solver.addGravity({ x: 0, y: -400, z: 0 });
    solver.update();
    expect(solver.islandsSkipped()).toBeLessThan(2);              // the loaded islands woke
    expect(solver.overstressedBondCount()).toBeGreaterThan(0);    // and the fracture is seen the same frame
    solver.destroy();
  });
});
