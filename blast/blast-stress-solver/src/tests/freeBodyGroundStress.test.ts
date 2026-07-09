/**
 * Does a free-floating body *resting on the ground* develop real gravity+contact
 * bond stress and break via the existing overstress path — with NO support nodes
 * and NO scripted onImpact callback?
 *
 * This pins down the physics behind making the destructible-vehicle demo break on
 * stress instead of the onImpact hack. The demo kept materialScale=1e12 (bonds
 * effectively unbreakable) and faked breaking via onImpact, on the premise that
 * "a free body barely stresses its bonds". That premise is only true in FREE-FALL.
 * A body resting on the ground is supported by the ground reaction, which the core
 * already injects into the stress solver as a contact force — so gravity + that
 * reaction form a real load path the overstress fracture path handles natively.
 *
 * We build a heavy vertical column of dynamic cubes (no mass=0 supports) and show:
 *   - RESTING on the ground: at a high materialScale it holds (like the demo); at
 *     a low materialScale the weight-bearing bonds overstress and it breaks — no
 *     onImpact, no supports.
 *   - The SAME column in FREE-FALL stays intact even at a materialScale that
 *     shatters the resting one (free-fall is stress-free — the equivalence
 *     principle), proving it is the GROUND REACTION, not gravity alone, that
 *     stresses the bonds.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import RAPIER from '@dimforge/rapier3d-compat';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

let buildDestructibleCore: (opts: any) => Promise<any>;
beforeAll(async () => {
  if (!runtimeAvailable) return;
  await RAPIER.init();
  buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
});

const CUBE = 1.0;   // edge length (m)
const N = 5;        // cubes in the column
const MASS = 500;   // kg per cube (heavy, so self-weight is significant)
const AREA = 0.05;  // bond contact area (m^2)
const START_BONDS = N - 1;

// High materialScale → effectively unbreakable (the demo's regime); low → the
// real self-weight stress crosses the bond limit. The transition sits between
// these for this weight/area (observed: holds at 1e9, breaks by 1e7).
const HOLD_SCALE = 1e12;
const BREAK_SCALE = 1e6;

/** Vertical column of N dynamic cubes, lowest cube's bottom face at y=baseY. */
function columnScenario(baseY: number) {
  const nodes = [];
  const bonds = [];
  for (let i = 0; i < N; i++) {
    const cy = baseY + CUBE / 2 + i * CUBE;
    nodes.push({ centroid: { x: 0, y: cy, z: 0 }, mass: MASS, volume: CUBE ** 3 });
    if (i > 0) {
      bonds.push({
        node0: i - 1,
        node1: i,
        centroid: { x: 0, y: baseY + i * CUBE, z: 0 },
        normal: { x: 0, y: 1, z: 0 },
        area: AREA,
      });
    }
  }
  return { nodes, bonds, spacing: { x: CUBE, y: CUBE, z: CUBE } } as any;
}

async function makeCore(baseY: number, materialScale: number) {
  const core = await buildDestructibleCore({
    scenario: columnScenario(baseY),
    gravity: -9.81,
    materialScale,
    contactForceScale: 1,          // raw ground reaction, no amplification
    debrisCleanup: { mode: 'off' },
  });
  core.setFracturePolicy?.({ idleSkip: false, maxFracturesPerFrame: 0 });
  return core;
}

describe.skipIf(!runtimeAvailable)('free body resting on ground develops real stress', () => {
  it('holds at a high materialScale (the demo regime)', async () => {
    const core = await makeCore(0.0, HOLD_SCALE);
    for (let i = 0; i < 120; i++) core.step(1 / 60);
    expect(core.getActiveBondsCount()).toBe(START_BONDS); // intact under self-weight
    core.dispose();
  }, 30000);

  it('breaks under its own weight at a low materialScale — no supports, no onImpact', async () => {
    const core = await makeCore(0.0, BREAK_SCALE);
    for (let i = 0; i < 120; i++) core.step(1 / 60);
    expect(core.getActiveBondsCount()).toBeLessThan(START_BONDS); // ground reaction + weight overstressed bonds
    core.dispose();
  }, 30000);

  it('the SAME column in FREE-FALL stays intact even at the breaking materialScale', async () => {
    // Released high; stepped only while airborne (never reaches the ground here).
    const core = await makeCore(40.0, BREAK_SCALE);
    for (let i = 0; i < 30; i++) core.step(1 / 60); // ~0.5 s of free-fall
    expect(core.getActiveBondsCount()).toBe(START_BONDS); // free-fall is stress-free
    core.dispose();
  }, 30000);
});
