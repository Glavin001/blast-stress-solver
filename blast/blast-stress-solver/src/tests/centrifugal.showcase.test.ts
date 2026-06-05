/**
 * Proof that the centrifugal-spinner showcase demonstrates what it claims.
 *
 * Scene: free-floating segmented beams (no anchor), zero gravity, each beam spun hard about an axis
 * perpendicular to its length. The only thing that can stress these beams is centrifugal load, so
 * this is a clean A/B of `setSolverCentrifugalEnabled`:
 *
 *   - OFF → no force reaches the solver → bonds stay intact, body count unchanged.
 *   - ON  → inward (centripetal) acceleration loads the radial bonds in compression → bonds break
 *           and beams shatter into multiple bodies.
 *
 * This mirrors the Rust `centrifugal_wiring_test` (a spinning bar fractures in compression only with
 * the flag on) at the full JS/WASM core level. Like the other WASM-backed tests here it is skipped
 * unless the WASM runtime has been built into `dist/`.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// Direct (no-WASM) import: the scenario builder is pure data and can be checked unconditionally.
import { buildSpinningBeamsScenario } from '../scenarios/index';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, '../../dist/stress_solver.wasm');
const runtimeAvailable = existsSync(wasmPath);

let buildDestructibleCore: (opts: any) => Promise<any>;

async function loadModules() {
  if (buildDestructibleCore) return;
  const rapier = await import('../../dist/rapier.js');
  buildDestructibleCore = rapier.buildDestructibleCore;
}

// Weak in compression (a spinning beam loads its bonds inward → compression), strong otherwise, so
// only the centrifugal failure mode can break it.
const SHOWCASE_SOLVER_SETTINGS = {
  compressionElasticLimit: 0.2,
  compressionFatalLimit: 0.4,
  tensionElasticLimit: 1.0e6,
  tensionFatalLimit: 1.0e7,
  shearElasticLimit: 1.0e6,
  shearFatalLimit: 1.0e7,
};

const SPIN_RAD_PER_S = 60; // hard spin about +Y (perpendicular to each beam's +X axis)

/** Set a fixed spin on every live dynamic body so the beams keep tumbling deterministically. */
function spinAllDynamicBodies(core: any, omega: number) {
  const world = core.world;
  const seen = new Set<number>();
  for (const chunk of core.chunks) {
    if (!chunk.active || chunk.bodyHandle == null || seen.has(chunk.bodyHandle)) continue;
    seen.add(chunk.bodyHandle);
    const body = world.getRigidBody(chunk.bodyHandle);
    if (!body || body.isFixed()) continue;
    body.setAngularDamping(0);
    body.setAngvel({ x: 0, y: omega, z: 0 }, true);
  }
}

async function buildShowcaseCore(centrifugal: boolean) {
  const scenario = buildSpinningBeamsScenario({ beams: 3, segments: 7, segmentSize: 0.6 });
  const core = await buildDestructibleCore({
    scenario,
    gravity: 0, // isolate centrifugal: a uniform field (gravity) would induce no internal stress
    solverSettings: SHOWCASE_SOLVER_SETTINGS,
    // Run the solver every frame so the A/B is deterministic (no idle-skip), and keep the bodies
    // from being damped to sleep before the spin can do its work.
    fracturePolicy: { idleSkip: false },
    sleepMode: 'off',
    smallBodyDamping: { mode: 'off' },
    damage: { enabled: false },
  });
  core.setSolverCentrifugalEnabled(centrifugal);
  return { core, scenario };
}

/** Spin + step the scene, returning bonds/bodies before and after. */
function runShowcase(core: any, frames = 150) {
  const initialBonds = core.getActiveBondsCount();
  const initialBodies = core.getRigidBodyCount();
  for (let i = 0; i < frames; i++) {
    spinAllDynamicBodies(core, SPIN_RAD_PER_S);
    core.step(1 / 60);
  }
  return {
    initialBonds,
    initialBodies,
    finalBonds: core.getActiveBondsCount(),
    finalBodies: core.getRigidBodyCount(),
  };
}

describe('centrifugal-spinner showcase: scenario builder', () => {
  it('produces independent, anchor-free beams', () => {
    const beams = 4;
    const segments = 6;
    const scenario = buildSpinningBeamsScenario({ beams, segments });
    // No support nodes — every node is dynamic so each beam is a free body.
    expect(scenario.nodes.every((n) => n.mass > 0)).toBe(true);
    expect(scenario.nodes).toHaveLength(beams * segments);
    // Each beam contributes (segments - 1) internal bonds and nothing links beams together.
    expect(scenario.bonds).toHaveLength(beams * (segments - 1));
  });
});

describe.skipIf(!runtimeAvailable)('centrifugal-spinner showcase: behavior (requires WASM build)', () => {
  beforeAll(async () => {
    await loadModules();
  });

  it('does NOT fracture the spinning beams when centrifugal is OFF', async () => {
    const { core } = await buildShowcaseCore(false);
    const r = runShowcase(core);
    // No gravity and no centrifugal ⇒ nothing stresses the beams.
    expect(r.finalBonds).toBe(r.initialBonds);
    expect(r.finalBodies).toBe(r.initialBodies);
  });

  it('DOES shatter the spinning beams when centrifugal is ON', async () => {
    const { core } = await buildShowcaseCore(true);
    const r = runShowcase(core);
    // Centripetal compression breaks bonds and splits each beam into multiple bodies.
    expect(r.finalBonds).toBeLessThan(r.initialBonds);
    expect(r.finalBodies).toBeGreaterThan(r.initialBodies);
  });

  it('breaks strictly more bonds with centrifugal ON than OFF (same scene, same spin)', async () => {
    const off = runShowcase((await buildShowcaseCore(false)).core);
    const on = runShowcase((await buildShowcaseCore(true)).core);
    const brokenOff = off.initialBonds - off.finalBonds;
    const brokenOn = on.initialBonds - on.finalBonds;
    expect(brokenOff).toBe(0);
    expect(brokenOn).toBeGreaterThan(brokenOff);
  });
});
