/**
 * Tests for solver eviction — retiring settled/detached debris from the stress
 * solve so the solver stops spending iterations on inert fragments.
 *
 * Two layers:
 *  1. Solver level — proves `deactivateActor` actually removes an actor's nodes
 *     and bonds from the WASM solve (not just JS-side bookkeeping): after
 *     eviction the previously-overstressed bonds are no longer evaluated.
 *  2. Integration level — proves the destructible pipeline evicts settled
 *     clusters while preserving the rigid bodies (debris stays visible/physical),
 *     by A/B-ing the same destruction with eviction off vs on.
 *
 * Requires the full WASM + TS build (dist/). Skips gracefully otherwise.
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

/** 6-node horizontal cantilever along +X, node 0 fixed (mass 0). One actor. */
function cantilever(): { nodes: Node[]; bonds: Bond[] } {
  const n = 6;
  const nodes: Node[] = [];
  const bonds: Bond[] = [];
  for (let i = 0; i < n; i += 1) {
    nodes.push({ centroid: { x: i, y: 0, z: 0 }, mass: i === 0 ? 0 : 1, volume: 1 });
  }
  for (let i = 0; i < n - 1; i += 1) {
    bonds.push({ centroid: { x: i + 0.5, y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 0.5, node0: i, node1: i + 1 });
  }
  return { nodes, bonds };
}

const bendSensitive = {
  maxSolverIterationsPerFrame: 25,
  compressionElasticLimit: 1e5,
  compressionFatalLimit: 1e6,
  tensionElasticLimit: 0.05,
  tensionFatalLimit: 0.1,
  shearElasticLimit: 0.05,
  shearFatalLimit: 0.1,
};

describe.skipIf(!runtimeAvailable)('Solver eviction — solver level (requires WASM build)', () => {
  it('deactivateActor drops the actor and its bonds from the solve', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    const { nodes, bonds } = cantilever();
    const solver = rt.createExtSolver({ nodes, bonds, settings: bendSensitive });

    // One actor spanning all 6 nodes / 5 bonds.
    expect(solver.actorCount()).toBe(1);
    const [actor] = solver.actors();
    expect(actor).toBeDefined();

    // Bending gravity overstresses the cantilever — bonds are actively solved.
    expect(solver.addActorGravity(actor!.actorIndex, { x: 0, y: -50, z: 0 })).toBe(true);
    solver.update();
    expect(solver.overstressedBondCount()).toBeGreaterThan(0);

    // Evict the actor without splitting it.
    expect(solver.deactivateActor(actor!.actorIndex)).toBe(true);
    expect(solver.actorCount()).toBe(0);

    // The actor is gone: per-actor gravity is a no-op, and after the next update
    // there are no bonds left in the solve to overstress. This is the proof that
    // the WASM graph shrank, not merely the JS-side tracking.
    expect(solver.addActorGravity(actor!.actorIndex, { x: 0, y: -50, z: 0 })).toBe(false);
    solver.update();
    expect(solver.overstressedBondCount()).toBe(0);

    // Idempotent / safe to call on an unknown actor.
    expect(solver.deactivateActor(actor!.actorIndex)).toBe(false);
    solver.destroy();
  });
});

describe.skipIf(!runtimeAvailable)('Solver eviction — integration (requires WASM build)', () => {
  let buildDestructibleCore: (opts: any) => Promise<any>;
  let buildTowerScenario: (opts?: any) => any;

  async function loadModules() {
    if (buildDestructibleCore) return;
    buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
    buildTowerScenario = (await import('../../dist/scenarios.js')).buildTowerScenario;
  }

  // Topple a tower with a strong low impact, then settle. Strong bonds keep large
  // multi-collider clusters intact so there is something to evict once they rest.
  async function runDestruction(evictionMode: 'off' | 'always') {
    const scenario = buildTowerScenario({ side: 5, stories: 10, totalMass: 4000 });
    const core = await buildDestructibleCore({
      scenario, gravity: -9.81, materialScale: 1e10,
      resimulateOnFracture: true, maxResimulationPasses: 1, snapshotMode: 'perBody',
      sleepMode: 'always', sleepLinearThreshold: 0.3, sleepAngularThreshold: 0.3,
      debrisCleanup: { mode: 'off' }, // isolate eviction: nothing else removes bodies
      solverEviction: { mode: evictionMode, minColliders: 2 },
    });
    const dt = 1 / 60;
    const safeStep = () => { try { core.step(dt); } catch { /* detached-buffer guard */ } };
    for (let i = 0; i < 60; i++) safeStep();
    core.enqueueProjectile({ position: { x: 0, y: 0.4, z: 5 }, velocity: { x: 0, y: 0, z: -55 }, radius: 0.5, mass: 30000, ttl: 5000 });
    for (let i = 0; i < 120; i++) safeStep();
    const bodiesAfterImpact = core.getRigidBodyCount();
    for (let i = 0; i < 700; i++) safeStep(); // settle — eviction fires as clusters sleep
    return {
      stats: core.getSolverEvictionStats?.() ?? { evictedBodies: 0, evictedBonds: 0 },
      activeBonds: core.getActiveBondsCount(),
      finalBodies: core.getRigidBodyCount(),
      bodiesAfterImpact,
    };
  }

  it('evicts settled clusters while preserving rigid bodies', async () => {
    await loadModules();
    const off = await runDestruction('off');
    const on = await runDestruction('always');

    // Eviction never fires when disabled.
    expect(off.stats.evictedBodies).toBe(0);

    // Eviction fires when enabled: settled multi-collider clusters are retired.
    expect(on.stats.evictedBodies).toBeGreaterThan(0);
    expect(on.stats.evictedBonds).toBeGreaterThan(0);

    // Bodies are PRESERVED — eviction retires fragments from the solve only, it
    // never deletes the rigid body. The body population must match the off run.
    expect(on.finalBodies).toBe(off.finalBodies);

    // And the solver graph shrank: fewer active bonds remain in the solve.
    expect(on.activeBonds).toBeLessThan(off.activeBonds);
  }, 60_000);
});
