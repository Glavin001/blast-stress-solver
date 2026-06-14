/**
 * Proof that the flywheel-burst showcase does what it claims.
 *
 * Scene: a single free-floating wheel (hub + spokes + rim hoop, no anchor), zero gravity, spun hard
 * about +Y. The only thing that can stress it is centrifugal load, so this is a clean A/B of
 * `setSolverCentrifugalEnabled` on a structured part rather than a bare beam:
 *
 *   - OFF → no force reaches the solver → the wheel spins forever intact, one body.
 *   - ON  → the rim and spoke bonds overstress (worst at the rim, where ω²r is largest) → the wheel
 *           bursts into multiple bodies.
 *
 * Mirrors `centrifugal.showcase.test.ts`. Like the other WASM-backed tests here it is skipped unless
 * the WASM runtime has been built into `dist/`.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildFlywheelScenario } from '../scenarios/index';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, '../../dist/stress_solver.wasm');
const runtimeAvailable = existsSync(wasmPath);

let buildDestructibleCore: (opts: any) => Promise<any>;

async function loadModules() {
  if (buildDestructibleCore) return;
  const rapier = await import('../../dist/rapier.js');
  buildDestructibleCore = rapier.buildDestructibleCore;
}

// Weak in compression (the spin loads bonds inward → compression), strong otherwise, so only the
// centrifugal failure mode can burst the wheel. Same shape as the spinner showcase.
const SHOWCASE_SOLVER_SETTINGS = {
  compressionElasticLimit: 0.5,
  compressionFatalLimit: 1.0,
  tensionElasticLimit: 1.0e6,
  tensionFatalLimit: 1.0e7,
  shearElasticLimit: 1.0e6,
  shearFatalLimit: 1.0e7,
};

const SPIN_RAD_PER_S = 50; // hard spin about +Y

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
  const scenario = buildFlywheelScenario();
  const core = await buildDestructibleCore({
    scenario,
    gravity: 0,
    solverSettings: SHOWCASE_SOLVER_SETTINGS,
    fracturePolicy: { idleSkip: false },
    sleepMode: 'off',
    smallBodyDamping: { mode: 'off' },
    damage: { enabled: false },
  });
  core.setSolverCentrifugalEnabled(centrifugal);
  return { core, scenario };
}

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

describe('flywheel showcase: scenario builder', () => {
  it('produces a single anchor-free wheel (hub + spokes + rim hoop)', () => {
    const s = buildFlywheelScenario({ rimBlocks: 24, spokes: 6, spokeSegments: 3 });
    // No support nodes — the whole wheel is one free body.
    expect(s.nodes.every((n) => n.mass > 0)).toBe(true);
    // 1 hub + spokes*segments + rimBlocks nodes.
    expect(s.nodes).toHaveLength(1 + 6 * 3 + 24);
    // spokes*segments spoke bonds + rimBlocks hoop bonds (closed loop) + spokes welds.
    expect(s.bonds).toHaveLength(6 * 3 + 24 + 6);
    expect(s.roles.filter((r) => r === 'rim')).toHaveLength(24);
  });

  it('keeps spokes a divisor of the rim count', () => {
    // 7 does not divide 24, so the builder steps spokes down to a divisor (6).
    const s = buildFlywheelScenario({ rimBlocks: 24, spokes: 7, spokeSegments: 2 });
    const spokeNodes = s.roles.filter((r) => r === 'spoke').length;
    expect(spokeNodes % 2).toBe(0);
    expect(24 % (spokeNodes / 2)).toBe(0);
  });
});

describe.skipIf(!runtimeAvailable)('flywheel showcase: behavior (requires WASM build)', () => {
  beforeAll(async () => {
    await loadModules();
  });

  // Body count is the robust burst signal: when the wheel splits, each fragment becomes its own
  // Rapier body, so finalBodies climbs far above the single starting body.
  it('does NOT burst the wheel when centrifugal is OFF', async () => {
    const { core } = await buildShowcaseCore(false);
    const r = runShowcase(core);
    // No load reaches the solver: the wheel keeps spinning as one body (allow one stray split).
    expect(r.finalBodies - r.initialBodies).toBeLessThanOrEqual(1);
  });

  it('DOES burst the wheel when centrifugal is ON', async () => {
    const { core } = await buildShowcaseCore(true);
    const r = runShowcase(core);
    // The rim/spoke bonds overstress and the wheel shatters into many separate bodies.
    expect(r.finalBodies - r.initialBodies).toBeGreaterThanOrEqual(10);
  });

  it('bursts with centrifugal ON but stays intact with it OFF (same scene, same spin)', async () => {
    const off = runShowcase((await buildShowcaseCore(false)).core);
    const on = runShowcase((await buildShowcaseCore(true)).core);
    // The whole point: the same spinning wheel comes apart only when centrifugal is enabled.
    expect(off.finalBodies - off.initialBodies).toBeLessThanOrEqual(1);
    expect(on.finalBodies - on.initialBodies).toBeGreaterThan(off.finalBodies - off.initialBodies);
  });
});
