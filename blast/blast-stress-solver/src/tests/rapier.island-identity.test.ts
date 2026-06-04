/**
 * Island identity (Stage 2a): the solver partitions its graph into connected
 * components ("islands") that are independent of Blast actors. This is the
 * foundation for solving each island separately and skipping settled ones.
 *
 * The load-bearing proof: two arms joined ONLY through a shared static (mass=0)
 * ground node are a SINGLE Blast actor (they're connected in the support graph)
 * but TWO stress islands (a static node carries no coupling, so it's a cut
 * point). islandCount() must report 2 while actorCount() reports 1.
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

async function loadRuntime() {
  return (await import('../../dist/index.js')) as typeof Runtime;
}

const settings = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 1e5, compressionFatalLimit: 1e6,
  tensionElasticLimit: 0.05, tensionFatalLimit: 0.1,
  shearElasticLimit: 0.05, shearFatalLimit: 0.1,
};

const bond = (a: number, b: number, x: number): Bond =>
  ({ centroid: { x, y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 0.5, node0: a, node1: b });

// Static ground node (0) with two cantilever arms (+X and -X) hanging off it.
// The arms touch only at the static node → one actor, two islands.
function twoArmsSharingGround(): { nodes: Node[]; bonds: Bond[] } {
  const nodes: Node[] = [{ centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 }];
  for (let i = 1; i <= 3; i++) nodes.push({ centroid: { x: i, y: 0, z: 0 }, mass: 1, volume: 1 });
  for (let i = 1; i <= 3; i++) nodes.push({ centroid: { x: -i, y: 0, z: 0 }, mass: 1, volume: 1 });
  const bonds = [
    bond(0, 1, 0.5), bond(1, 2, 1.5), bond(2, 3, 2.5),     // arm A: ground-1-2-3
    bond(0, 4, -0.5), bond(4, 5, -1.5), bond(5, 6, -2.5),  // arm B: ground-4-5-6
  ];
  return { nodes, bonds };
}

function oneArm(): { nodes: Node[]; bonds: Bond[] } {
  const nodes: Node[] = [{ centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 }];
  for (let i = 1; i <= 3; i++) nodes.push({ centroid: { x: i, y: 0, z: 0 }, mass: 1, volume: 1 });
  return { nodes, bonds: [bond(0, 1, 0.5), bond(1, 2, 1.5), bond(2, 3, 2.5)] };
}

describe.skipIf(!runtimeAvailable)('Island identity (requires WASM build)', () => {
  it('disconnected components are separate islands, distinct from actors', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    const { nodes, bonds } = twoArmsSharingGround();
    const solver = rt.createExtSolver({ nodes, bonds, settings });
    solver.addGravity({ x: 0, y: -10, z: 0 });
    solver.update();
    // One Blast actor (arms connect through the shared ground chunk)...
    expect(solver.actorCount()).toBe(1);
    // ...but two independent stress islands (static ground node is a cut point).
    expect(solver.islandCount()).toBe(2);
    solver.destroy();
  });

  it('a single connected structure is one island', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    const { nodes, bonds } = oneArm();
    const solver = rt.createExtSolver({ nodes, bonds, settings });
    solver.addGravity({ x: 0, y: -10, z: 0 });
    solver.update();
    expect(solver.islandCount()).toBe(1);
    solver.destroy();
  });
});
