/**
 * Empirical tests of two physics mechanisms at the JS SOLVER level — the mirror of the Rust
 * `tests/solver_mechanisms_test.rs`. We assert the actual numeric behavior rather than
 * trusting that the feature is wired up, so we can be confident BOTH libraries' solvers
 * implement orientation-dependent gravity and excess-force-on-fracture.
 *
 * Whether each library's high-level destruction pipeline FEEDS these mechanisms correctly is
 * a separate, integration-level question (see gaps #7/#8 in blast/TESTING.md). Notably, the
 * JS pipeline DOES rotate gravity per actor (destructible-core.ts), while the Rust pipeline
 * does not; conversely the Rust pipeline applies excess forces while the JS pipeline does not.
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

/** 6-node horizontal cantilever along +X, node 0 fixed (mass 0). */
function cantilever(): { nodes: Node[]; bonds: Bond[] } {
  const n = 6;
  const nodes: Node[] = [];
  const bonds: Bond[] = [];
  for (let i = 0; i < n; i += 1) {
    nodes.push({ centroid: { x: i, y: 0, z: 0 }, mass: i === 0 ? 0 : 1, volume: 1 });
  }
  for (let i = 0; i < n - 1; i += 1) {
    bonds.push({
      centroid: { x: i + 0.5, y: 0, z: 0 },
      normal: { x: 1, y: 0, z: 0 },
      area: 0.5,
      node0: i,
      node1: i + 1,
    });
  }
  return { nodes, bonds };
}

// Weak tension/shear, very strong compression: a bending (perpendicular) load snaps the
// beam; an axial load only compresses it.
const bendSensitive = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 1e5,
  compressionFatalLimit: 1e6,
  tensionElasticLimit: 0.05,
  tensionFatalLimit: 0.1,
  shearElasticLimit: 0.05,
  shearFatalLimit: 0.1,
};

describe.skipIf(!runtimeAvailable)('Solver mechanisms (requires WASM build)', () => {
  async function overstressForGravity(gravity: Vec3, perActor: boolean): Promise<number> {
    const rt = await (await loadRuntime()).loadStressSolver();
    const { nodes, bonds } = cantilever();
    const solver = rt.createExtSolver({ nodes, bonds, settings: bendSensitive });
    if (perActor) {
      const [actor] = solver.actors();
      expect(actor).toBeDefined();
      expect(solver.addActorGravity(actor!.actorIndex, gravity)).toBe(true);
    } else {
      solver.addGravity(gravity);
    }
    solver.update();
    const count = solver.overstressedBondCount();
    solver.destroy();
    return count;
  }

  it('addGravity is direction sensitive (bending vs axial)', async () => {
    const bending = await overstressForGravity({ x: 0, y: -50, z: 0 }, false);
    const axial = await overstressForGravity({ x: -50, y: 0, z: 0 }, false);
    expect(bending).toBeGreaterThan(axial);
  });

  it('addActorGravity is direction sensitive (the API the JS pipeline uses for rotation)', async () => {
    const bending = await overstressForGravity({ x: 0, y: -50, z: 0 }, true);
    const axial = await overstressForGravity({ x: -50, y: 0, z: 0 }, true);
    expect(bending).toBeGreaterThan(axial);
  });

  it('getExcessForces reports the released load after a bond fractures', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    // A 10 kg mass hung from a fixed support by one bond under g = 100 carries ~1000 N.
    const nodes: Node[] = [
      { centroid: { x: 0, y: 1, z: 0 }, mass: 0, volume: 1 },
      { centroid: { x: 0, y: 0, z: 0 }, mass: 10, volume: 1 },
    ];
    const bonds: Bond[] = [
      { centroid: { x: 0, y: 0.5, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 1, node0: 0, node1: 1 },
    ];
    const solver = rt.createExtSolver({
      nodes,
      bonds,
      settings: {
        maxSolverIterationsPerFrame: 25,
        compressionElasticLimit: 1, compressionFatalLimit: 2,
        tensionElasticLimit: 1, tensionFatalLimit: 2,
        shearElasticLimit: 1, shearFatalLimit: 2,
      },
    });
    solver.addGravity({ x: 0, y: -100, z: 0 });
    solver.update();
    expect(solver.overstressedBondCount()).toBe(1);

    const perActor = solver.generateFractureCommandsPerActor();
    const events = solver.applyFractureCommands(perActor);
    expect(events.length).toBeGreaterThan(0);

    const freed = solver.actors().find((a) => a.nodes.includes(1));
    expect(freed).toBeDefined();
    const excess = solver.getExcessForces(freed!.actorIndex, { x: 0, y: 0, z: 0 });
    expect(excess).not.toBeNull();
    const mag = Math.hypot(excess!.force.x, excess!.force.y, excess!.force.z);
    // Near-zero would mean the momentum-transfer mechanism is dormant.
    expect(mag).toBeGreaterThan(100);
    solver.destroy();
  });
});
