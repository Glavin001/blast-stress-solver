/**
 * Tier-1 physics invariants for the JS destruction pipeline (requires the WASM build).
 *
 * This is the JS half of the shared invariant spec (`invariants.shared.ts`). It proves
 * the JS library satisfies the same point-velocity continuity + finiteness invariants
 * that the Rust `kinematic_invariants_test.rs` asserts — and that the Rust offset-COM
 * repro currently violates. Together they localize the "sudden movement after
 * destruction" bug to the Rust split path rather than the algorithm in general.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  stepN,
  getContinuityRecords,
  getDynamicRecords,
  assertContinuity,
  assertFiniteRecords,
  type ContinuityObservableCore,
} from './invariants.shared';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, '../../dist/stress_solver.wasm');
const runtimeAvailable = existsSync(wasmPath);

type Vec3 = { x: number; y: number; z: number };

let buildDestructibleCore: (opts: any) => Promise<ContinuityObservableCore & { dispose: () => void }>;
let buildWallScenario: (opts?: any) => any;

async function loadModules() {
  if (buildDestructibleCore) return;
  const rapier = await import('../../dist/rapier.js');
  const scenarios = await import('../../dist/scenarios.js');
  buildDestructibleCore = rapier.buildDestructibleCore;
  buildWallScenario = scenarios.buildWallScenario;
}

/** A 4x3 grid anchored along its top row with very weak vertical bonds: it cascades
 * into many dynamic→dynamic splits, the case that exercises split continuity. */
function createCascadingGridScenario() {
  const rows = 4;
  const cols = 3;
  const nodes: Array<{ centroid: Vec3; mass: number; volume: number }> = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      nodes.push({ centroid: { x: col, y: row + 0.5, z: 0 }, mass: row === 0 ? 0 : 5, volume: 1 });
    }
  }
  const bonds: Array<{ node0: number; node1: number; centroid: Vec3; normal: Vec3; area: number }> = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      const node0 = row * cols + col;
      bonds.push({ node0, node1: node0 + 1, centroid: { x: col + 0.5, y: row + 0.5, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 2.0 });
    }
  }
  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const node0 = row * cols + col;
      bonds.push({ node0, node1: node0 + cols, centroid: { x: col, y: row + 1.0, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 0.01 });
    }
  }
  return { nodes, bonds };
}

describe.skipIf(!runtimeAvailable)('Kinematic invariants after fracture (requires WASM build)', () => {
  it('preserves point-velocity continuity across a cascading split', async () => {
    await loadModules();
    const core = await buildDestructibleCore({
      scenario: createCascadingGridScenario(),
      gravity: -20,
      materialScale: 0.001,
      resimulateOnFracture: true,
      maxResimulationPasses: 5,
      snapshotMode: 'perBody',
      skipSingleBodies: false,
    });
    core.__clearDebugSplitContinuityLog?.();
    stepN(core, 220);

    const dynamicRecords = getDynamicRecords(core);
    // The JS reference path must keep every migrated chunk's point velocity continuous.
    assertContinuity(expect, dynamicRecords);
    assertFiniteRecords(expect, getContinuityRecords(core));
    core.dispose();
  });

  it('never produces NaN/Inf continuity samples under a wall collapse', async () => {
    await loadModules();
    const core = await buildDestructibleCore({
      scenario: buildWallScenario(),
      gravity: -9.81,
      materialScale: 0.0005, // weak material -> the wall fractures
      resimulateOnFracture: true,
      maxResimulationPasses: 4,
      snapshotMode: 'perBody',
      skipSingleBodies: false,
    });
    core.__clearDebugSplitContinuityLog?.();
    stepN(core, 200);

    const records = getContinuityRecords(core);
    // A collapsing wall must still split cleanly: every recorded sample is finite and
    // every genuine dynamic transfer is point-velocity continuous.
    assertFiniteRecords(expect, records);
    const dynamicRecords = getDynamicRecords(core);
    if (dynamicRecords.length > 0) {
      assertContinuity(expect, dynamicRecords);
    }
    core.dispose();
  });
});
