/**
 * Proof that the "Keystone" arch showcase demonstrates what it claims.
 *
 * Scene: a semicircular voussoir ring resting (by contact, not bonds) on two fixed abutments under
 * gravity. The ring is held up purely by compression in its radial joints — the thrust line.
 *
 *   - With strong stone, the thrust line stays inside the section: joints survive and the ring
 *     stays aloft (it stands on pure compression).
 *   - With weak stone, the joints cannot carry the self-weight: they overstress and fracture, the
 *     ring splits into free bodies, and the stones fall.
 *
 * Collapse here is driven by *overstress* (the only thing that re-partitions the rigid bodies),
 * which is exactly what the demo's "Knock Out Keystone" striker, Load slider, dropped weight and
 * shots all do. This is the structural complement to the centrifugal showcase: the same solver,
 * exercised in compression. Like the other WASM-backed tests here it is skipped unless the runtime
 * is built into `dist/`.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// Direct (no-WASM) import: the scenario builder is pure data and can be checked unconditionally.
import { buildArchScenario } from '../scenarios/index';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, '../../dist/stress_solver.wasm');
const runtimeAvailable = existsSync(wasmPath);

let buildDestructibleCore: (opts: any) => Promise<any>;

async function loadModules() {
  if (buildDestructibleCore) return;
  const rapier = await import('../../dist/rapier.js');
  buildDestructibleCore = rapier.buildDestructibleCore;
}

const G = 9.81;
// Strong stone: comfortably above the self-weight thrust → the arch holds.
const STRONG_STONE = {
  compressionElasticLimit: 2e7, compressionFatalLimit: 6e7,
  tensionElasticLimit: 1e7, tensionFatalLimit: 3e7,
  shearElasticLimit: 1e7, shearFatalLimit: 3e7,
};
// Weak stone: the joints cannot carry the ring's own weight → guaranteed overstress + collapse.
const WEAK_STONE = {
  compressionElasticLimit: 50, compressionFatalLimit: 150,
  tensionElasticLimit: 20, tensionFatalLimit: 60,
  shearElasticLimit: 30, shearFatalLimit: 90,
};

async function buildArchCore(stone: Record<string, number>) {
  const { scenario, keystoneIndex, archNodeCount } = buildArchScenario({ voussoirs: 15 });
  const core = await buildDestructibleCore({
    scenario,
    gravity: -G,
    solverSettings: { ...stone, maxSolverIterationsPerFrame: 100, graphReductionLevel: 0 },
    fracturePolicy: { idleSkip: false },
    sleepMode: 'off',
    damage: { enabled: false },
  });
  return { core, keystoneIndex, archNodeCount };
}

/** Mean Y of the voussoir bodies (nodeIndex < archNodeCount), i.e. how high the ring is sitting. */
function meanVoussoirY(core: any, archNodeCount: number): number {
  const world = core.world;
  let sum = 0, n = 0;
  for (const chunk of core.chunks) {
    if (chunk.nodeIndex >= archNodeCount || chunk.bodyHandle == null) continue;
    const body = world.getRigidBody(chunk.bodyHandle);
    if (!body) continue;
    sum += body.translation().y;
    n += 1;
  }
  return n ? sum / n : 0;
}

function step(core: any, frames: number) {
  for (let i = 0; i < frames; i++) core.step(1 / 60);
}

describe('keystone-arch showcase: scenario builder', () => {
  it('builds a voussoir ring on two fixed abutments with a central keystone', () => {
    const { scenario, keystoneIndex, archNodeCount } = buildArchScenario({ voussoirs: 15 });
    // 15 voussoirs (dynamic) + 2 abutments (mass-0 supports).
    expect(archNodeCount).toBe(15);
    expect(scenario.nodes).toHaveLength(17);
    expect(scenario.nodes.filter((n) => n.mass === 0)).toHaveLength(2); // the abutments anchor the world
    expect(scenario.nodes.slice(0, archNodeCount).every((n) => n.mass > 0)).toBe(true);
    // A chain of N voussoirs has N-1 radial joints; the abutments are not bonded to the ring.
    expect(scenario.bonds).toHaveLength(14);
    // Keystone is the centre stone, and every bond joins consecutive voussoirs.
    expect(keystoneIndex).toBe(7);
    expect(scenario.bonds.every((b) => Math.abs(b.node0 - b.node1) === 1)).toBe(true);
  });

  it('forces an odd voussoir count so there is a single keystone', () => {
    const even = buildArchScenario({ voussoirs: 14 });
    expect(even.archNodeCount).toBe(15);
    expect(even.keystoneIndex).toBe(7);
  });

  it('points each joint bond along the circumferential (thrust) direction', () => {
    const { scenario } = buildArchScenario({ voussoirs: 15 });
    // Every bond normal is a unit vector in the X/Y plane (the thrust runs around the ring).
    for (const b of scenario.bonds) {
      const len = Math.hypot(b.normal.x, b.normal.y, b.normal.z);
      expect(len).toBeCloseTo(1, 5);
      expect(Math.abs(b.normal.z)).toBeLessThan(1e-9);
    }
  });
});

describe.skipIf(!runtimeAvailable)('keystone-arch showcase: behavior (requires WASM build)', () => {
  beforeAll(async () => {
    await loadModules();
  });

  it('stands intact on pure compression when the stone is strong', async () => {
    const { core, archNodeCount } = await buildArchCore(STRONG_STONE);
    const bonds0 = core.getActiveBondsCount();
    const y0 = meanVoussoirY(core, archNodeCount);
    step(core, 180);
    // The thrust line stays inside the section: the ring barely settles and joints survive (allow a
    // single stray break so the gate pins behaviour, not bit-exactness).
    expect(bonds0 - core.getActiveBondsCount()).toBeLessThanOrEqual(1);
    expect(meanVoussoirY(core, archNodeCount)).toBeGreaterThan(y0 - 0.5);
  });

  it('overstresses and collapses when the stone cannot carry its own weight', async () => {
    const { core, archNodeCount } = await buildArchCore(WEAK_STONE);
    const bonds0 = core.getActiveBondsCount();
    const bodies0 = core.getRigidBodyCount();
    const y0 = meanVoussoirY(core, archNodeCount);
    step(core, 240);
    // Joints give way, the ring splits into more bodies, and the stones drop off the abutments.
    expect(core.getActiveBondsCount()).toBeLessThan(bonds0);
    expect(core.getRigidBodyCount()).toBeGreaterThan(bodies0);
    expect(meanVoussoirY(core, archNodeCount)).toBeLessThan(y0 - 1.0);
  });
});
