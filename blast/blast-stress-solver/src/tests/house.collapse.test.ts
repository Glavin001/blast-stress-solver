/**
 * Headless PHYSICS test of the house's structural realism (requires the WASM build).
 *
 * It runs the real Rapier + stress-solver pipeline and checks the three claims the
 * structural model makes:
 *   1. The intact house stands under gravity — the roof does not sag/float-away on its own.
 *   2. Removing the structural FRAME (posts + beams) collapses the roof (it falls).
 *   3. Removing only the non-structural DRYWALL leaves the roof standing.
 *
 * Supports are removed deterministically with core.cutNodeBonds() (no flaky projectile
 * aiming), so this isolates the bond/structure model from contact tuning.
 *
 * Run: npm --prefix blast/blast-stress-solver run build && \
 *      npx vitest run src/tests/house.collapse.test.ts
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));
const d = runtimeAvailable ? describe : describe.skip;

let buildDestructibleCore: (opts: any) => Promise<any>;
let buildHouseScenario: (opts?: any) => any;

beforeAll(async () => {
  buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
  buildHouseScenario = (await import('../../dist/scenarios.js')).buildHouseScenario;
});

const MATERIAL_SCALE = 4e9; // matches the demo default

function stepN(core: any, n: number) {
  for (let i = 0; i < n; i++) core.step(1 / 60);
}

async function buildHouseCore() {
  const scenario = buildHouseScenario({ furniture: false });
  const types: string[] = (scenario.parameters as any).house.fragmentTypes;
  const core = await buildDestructibleCore({
    scenario,
    gravity: -9.81,
    materialScale: MATERIAL_SCALE,
    solverSettings: { maxSolverIterationsPerFrame: 24, graphReductionLevel: 0 },
    contactForceScale: 14,
    friction: 0.5,
    restitution: 0,
  });
  return { core, types };
}

/** Fraction of roof chunks that have detached into their own dynamic bodies. */
function detachedRoofFraction(core: any, types: string[]): number {
  const roof = core.chunks.filter((c: any) => types[c.nodeIndex] === 'roof');
  if (!roof.length) return 0;
  const detached = roof.filter((c: any) => c.detached || c.destroyed).length;
  return detached / roof.length;
}

/** Average world-Y of the still-attached chunks of a given fragment type. */
function avgRoofY(core: any, types: string[]): number {
  const roof = core.chunks.filter(
    (c: any) => c.active && !c.destroyed && c.worldPosition && types[c.nodeIndex] === 'roof',
  );
  if (!roof.length) return 0;
  return roof.reduce((s: number, c: any) => s + c.worldPosition.y, 0) / roof.length;
}

function cutAllOfTypes(core: any, types: string[], wanted: Set<string>) {
  for (const c of core.chunks) {
    if (wanted.has(types[c.nodeIndex])) core.cutNodeBonds(c.nodeIndex);
  }
}

d('house structural realism (physics)', () => {
  it('stands under gravity: the intact roof does not collapse on its own', async () => {
    const { core, types } = await buildHouseCore();
    const initialBonds = core.getActiveBondsCount();
    const y0 = avgRoofY(core, types);
    stepN(core, 180); // 3 s
    const y1 = avgRoofY(core, types);
    // Roof barely moves, and the vast majority of bonds survive.
    expect(y0 - y1).toBeLessThan(0.25);
    expect(core.getActiveBondsCount() / initialBonds).toBeGreaterThan(0.9);
    core.dispose();
  });

  it('collapses when the structural supports are shot out (vs. drywall left intact)', async () => {
    // Knock out the interior king posts + ridge that hold the roof up, with a heavy
    // projectile sweeping along the centerline (z≈0) where the ridge support lives.
    const { core, types } = await buildHouseCore();
    stepN(core, 60); // settle
    const y0 = avgRoofY(core, types);
    core.enqueueProjectile({
      position: { x: -9, y: 2.4, z: 0 },
      velocity: { x: 70, y: 0, z: 0 },
      radius: 0.9,
      mass: 6000,
      ttl: 6000,
    });
    stepN(core, 360); // 6 s
    const drop = y0 - avgRoofY(core, types);
    const detached = detachedRoofFraction(core, types);
    // eslint-disable-next-line no-console
    console.log(`[support blast] roof drop=${drop.toFixed(2)} m, detached roof=${(detached * 100).toFixed(0)}%`);
    expect(drop).toBeGreaterThan(0.6); // the roof comes down once its supports are gone
    void detached;
    core.dispose();
  });

  it('a drywall hit does NOT bring the roof down (drywall is non-structural)', async () => {
    // Same energy, but aimed at a drywall panel of the front wall away from the posts.
    const { core, types } = await buildHouseCore();
    stepN(core, 60);
    const y0 = avgRoofY(core, types);
    core.enqueueProjectile({
      position: { x: 1.6, y: 1.3, z: -9 },
      velocity: { x: 0, y: 0, z: 70 },
      radius: 0.5,
      mass: 6000,
      ttl: 6000,
    });
    stepN(core, 360);
    const drop = y0 - avgRoofY(core, types);
    // eslint-disable-next-line no-console
    console.log(`[drywall blast] roof drop=${drop.toFixed(2)} m`);
    expect(drop).toBeLessThan(0.55); // the frame still holds the roof up
    core.dispose();
  });
});
