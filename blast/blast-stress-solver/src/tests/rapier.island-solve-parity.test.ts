/**
 * Island-aware solve parity (Stage 2b): the island-aware solver must produce the
 * SAME simulation as the legacy whole-graph solver — just computed per disconnected
 * component. Correctness is never traded for performance.
 *
 * Guarantees under test:
 *   1. Single island (intact structure) → island-aware falls back to the whole-graph
 *      path, so it is BIT-IDENTICAL to legacy (stressError and overstressed count match
 *      exactly, every frame).
 *   2. Multiple islands → each component is solved independently with the same scaling
 *      and matrix entries; the fracture sequence and final actor partition are identical
 *      to legacy (decisions preserved), and both modes converge.
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

// High limits → never fractures (used to test bit-identical fallback transparency).
const rigidSettings = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 1e12, compressionFatalLimit: 2e12,
  tensionElasticLimit: 1e12, tensionFatalLimit: 2e12,
  shearElasticLimit: 1e12, shearFatalLimit: 2e12,
};
// Low limits → decisively fractures under the test gravity (clear of thresholds, so the
// tiny per-island vs whole-graph numeric difference can't flip a fracture decision).
const fractureSettings = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 5, compressionFatalLimit: 10,
  tensionElasticLimit: 1, tensionFatalLimit: 2,
  shearElasticLimit: 1, shearFatalLimit: 2,
};

const bond = (a: number, b: number, x: number): Bond =>
  ({ centroid: { x, y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 1, node0: a, node1: b });

// One static anchor (node 0) + a horizontal chain of dynamic masses → a cantilever.
function cantilever(len = 4): Scenario {
  const nodes: Node[] = [{ centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 }];
  const bonds: Bond[] = [];
  for (let i = 1; i <= len; i++) {
    nodes.push({ centroid: { x: i, y: 0, z: 0 }, mass: 10, volume: 1 });
    bonds.push(bond(i - 1, i, i - 0.5));
  }
  return { nodes, bonds };
}

// Two cantilever arms sharing only the static ground node → ONE actor, TWO islands.
function twoArmsSharingGround(len = 4): Scenario {
  const nodes: Node[] = [{ centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 }];
  const bonds: Bond[] = [];
  for (let i = 1; i <= len; i++) { nodes.push({ centroid: { x: i, y: 0, z: 0 }, mass: 10, volume: 1 }); bonds.push(bond(i === 1 ? 0 : nodes.length - 2, nodes.length - 1, i - 0.5)); }
  const base = nodes.length;
  for (let i = 1; i <= len; i++) { nodes.push({ centroid: { x: -i, y: 0, z: 0 }, mass: 10, volume: 1 }); bonds.push(bond(i === 1 ? 0 : nodes.length - 2, nodes.length - 1, -(i - 0.5))); }
  void base;
  return { nodes, bonds };
}

// N cantilever arms radiating from a single shared static ground node → ONE actor, N islands.
// Scales the gather path to many components (more counting-sort buckets, larger partition).
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

function sortedPartition(actors: Array<{ actorIndex: number; nodes: number[] }>): number[][] {
  return actors.map((a) => a.nodes.slice().sort((x, y) => x - y)).sort((a, b) => (a[0] ?? -1) - (b[0] ?? -1));
}

// Run a gravity-loaded, self-fracturing simulation and record its evolution.
async function simulate(scenario: Scenario, gravity: Vec3, islandAware: boolean, frames: number, settings = fractureSettings) {
  const rt = await (await loadRuntime()).loadStressSolver();
  const solver = rt.createExtSolver({ nodes: scenario.nodes, bonds: scenario.bonds, settings });
  solver.setIslandAware(islandAware);
  expect(solver.islandAware()).toBe(islandAware);
  const over: number[] = [];
  const islands: number[] = [];
  const errLin: number[] = [];
  for (let f = 0; f < frames; f++) {
    solver.addGravity(gravity);
    solver.update();
    over.push(solver.overstressedBondCount());
    islands.push(solver.islandCount());
    errLin.push(solver.stressError().lin);
    if (solver.overstressedBondCount() > 0) {
      solver.applyFractureCommands(solver.generateFractureCommandsPerActor());
    }
  }
  const partition = sortedPartition(solver.actors());
  const converged = solver.converged();
  solver.destroy();
  return { over, islands, errLin, partition, converged };
}

describe.skipIf(!runtimeAvailable)('Island-aware solve parity (requires WASM build)', () => {
  it('single island: island-aware is bit-identical to legacy (transparent fallback)', async () => {
    // Rigid (never fractures) so it stays one island the whole time → must match exactly.
    const legacy = await simulate(cantilever(), { x: 0, y: -50, z: 0 }, false, 6, rigidSettings);
    const island = await simulate(cantilever(), { x: 0, y: -50, z: 0 }, true, 6, rigidSettings);
    expect(island.islands).toEqual([1, 1, 1, 1, 1, 1]);     // single island throughout
    expect(island.over).toEqual(legacy.over);                // no fractures, identical
    expect(island.errLin).toEqual(legacy.errLin);            // EXACT float equality (same code path)
  });

  it('multi-island fracture sequence matches legacy (two arms sharing ground)', async () => {
    const g = { x: 0, y: -100, z: 0 };
    const legacy = await simulate(twoArmsSharingGround(), g, false, 8);
    const island = await simulate(twoArmsSharingGround(), g, true, 8);
    expect(island.islands[0]).toBeGreaterThanOrEqual(2);     // gather path is exercised from frame 0
    expect(island.over).toEqual(legacy.over);                // identical fracture decisions, every frame
    expect(island.partition).toEqual(legacy.partition);      // identical resulting actor partition
    expect(island.converged).toBe(legacy.converged);
  });

  it('scales to many islands with identical fracture evolution (4 arms = 4 islands)', async () => {
    const g = { x: 0, y: -100, z: 0 };
    const legacy = await simulate(manyArms(4), g, false, 8);
    const island = await simulate(manyArms(4), g, true, 8);
    expect(island.islands[0]).toBe(4);                       // four independent islands solved per-component
    expect(island.over).toEqual(legacy.over);                // identical fracture decisions, every frame
    expect(island.partition).toEqual(legacy.partition);      // identical resulting actor partition
    expect(island.converged).toBe(legacy.converged);
  });
});
