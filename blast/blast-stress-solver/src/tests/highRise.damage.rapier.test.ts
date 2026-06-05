/**
 * Full-pipeline (Rapier + WASM) test for the high-rise CONTACT-DAMAGE layer.
 *
 * Guards the local-destruction behavior that the global stress solver alone cannot
 * produce (it is spatially all-or-nothing): with the damage layer enabled, a wrecking
 * ball deposits per-chunk health damage and punches a LOCAL hole, while the stress
 * solver keeps the structure standing by redistributing gravity around the missing
 * nodes. Impacts are decoupled from the stress solver (damageContactStressScale
 * defaults to 0 when damage is enabled), so a large contact force no longer drives the
 * global cascade.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 * Run: npm run build && npx vitest run src/tests/highRise.damage.rapier.test.ts
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildHighRiseScenario } from '../scenarios/highRiseScenario';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

// Decoupled concrete limits (Pa), matching the shipped scene pack.
const CONCRETE_LIMITS = {
  compressionElasticLimit: 12e6,
  compressionFatalLimit: 30e6,
  tensionElasticLimit: 1.2e6,
  tensionFatalLimit: 3e6,
  shearElasticLimit: 1.6e6,
  shearFatalLimit: 4e6,
};

// Contact-damage params matching the shipped high-rise scene pack (export-high-rise.mjs).
const DAMAGE = {
  enabled: true,
  strengthPerVolume: 200,
  kImpact: 0.15,
  contactDamageScale: 1,
  minImpulseThreshold: 5,
  internalMinImpulseThreshold: 8,
  splashRadius: 3.0,
  splashFalloffExp: 1.5,
};

let buildDestructibleCore: (opts: any) => Promise<any>;
async function loadCore() {
  if (!buildDestructibleCore) {
    buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
  }
}

function stepN(core: any, n: number, dt = 1 / 60) {
  for (let i = 0; i < n; i++) core.step(dt);
}
function avgDynamicY(core: any): number {
  const dyn = core.chunks.filter((c: any) => c.active && !c.isSupport);
  if (!dyn.length) return 0;
  return dyn.reduce(
    (s: number, c: any) => s + (c.worldPosition ?? c.baseLocalOffset ?? c.localOffset).y,
    0,
  ) / dyn.length;
}
function destroyedByType(core: any, nodeTypes: string[]) {
  const counts: Record<string, number> = {};
  for (const c of core.chunks) {
    if (!c.destroyed) continue;
    const t = nodeTypes[c.nodeIndex] ?? 'other';
    counts[t] = (counts[t] ?? 0) + 1;
  }
  return counts;
}

describe.skipIf(!runtimeAvailable)('high-rise contact damage (Rapier)', () => {
  it('gravity is stable with the damage layer enabled (no spurious destruction)', async () => {
    await loadCore();
    const scenario = buildHighRiseScenario({ floorCount: 4, columnsX: 3, columnsZ: 2 });
    const core = await buildDestructibleCore({
      scenario,
      gravity: -9.81,
      materialScale: 1e10,
      solverSettings: CONCRETE_LIMITS,
      damage: DAMAGE,
    });
    expect(core.damageEnabled).toBe(true);
    const initialBonds = core.getActiveBondsCount();
    stepN(core, 150);
    const destroyed = core.chunks.filter((c: any) => c.destroyed).length;
    expect(destroyed).toBe(0); // resting contacts must not deplete health
    expect(core.getActiveBondsCount() / initialBonds).toBeGreaterThan(0.97);
    core.dispose();
  });

  it('a wrecking ball punches a LOCAL hole and the building stays standing', async () => {
    await loadCore();
    const scenario = buildHighRiseScenario({ floorCount: 4, columnsX: 3, columnsZ: 2 });
    const nodeTypes: string[] = (scenario.parameters as any).highRise.fragmentTypes;

    // Target a front-wall (min-Z) infill panel at mid height.
    let ylo = Infinity, yhi = -Infinity, zlo = Infinity;
    for (const n of scenario.nodes) {
      ylo = Math.min(ylo, n.centroid.y);
      yhi = Math.max(yhi, n.centroid.y);
      zlo = Math.min(zlo, n.centroid.z);
    }
    const target = { x: 0, y: (ylo + yhi) / 2, z: zlo };
    let hit = 0, best = Infinity;
    for (let i = 0; i < scenario.nodes.length; i++) {
      if (nodeTypes[i] !== 'infill') continue;
      const c = scenario.nodes[i].centroid;
      const d = (c.x - target.x) ** 2 + (c.y - target.y) ** 2 + (c.z - target.z) ** 2;
      if (d < best) { best = d; hit = i; }
    }
    const t = scenario.nodes[hit].centroid;

    const core = await buildDestructibleCore({
      scenario,
      gravity: -9.81,
      materialScale: 1e10,
      solverSettings: CONCRETE_LIMITS,
      friction: 0.25,
      restitution: 0,
      contactForceScale: 30,
      damage: DAMAGE,
    });
    const initialBonds = core.getActiveBondsCount();
    const y0 = avgDynamicY(core);
    core.step(1 / 60);
    core.enqueueProjectile({
      position: { x: t.x, y: t.y, z: t.z - 9 },
      velocity: { x: 0, y: 0, z: 18 },
      radius: 0.6,
      mass: 2500,
      ttl: 2500,
    });
    stepN(core, 180);

    const destroyed = destroyedByType(core, nodeTypes);
    const infillDestroyed = destroyed.infill ?? 0;
    const structuralDestroyed = (destroyed.column ?? 0) + (destroyed.slab ?? 0) + (destroyed.foundation ?? 0);
    const survival = core.getActiveBondsCount() / initialBonds;
    const drop = y0 - avgDynamicY(core);

    // A local hole formed in the wall...
    expect(infillDestroyed).toBeGreaterThan(0);
    // ...but it is LOCAL, not a uniform shatter (540 infill chunks exist in the full
    // building; a single realistic hit blows out a handful, not the whole skin).
    expect(infillDestroyed).toBeLessThan(60);
    // The frame is barely touched by a single wall hit.
    expect(structuralDestroyed).toBeLessThan(6);
    // No global cascade: the vast majority of bonds survive and the building stands.
    expect(survival).toBeGreaterThan(0.85);
    expect(drop).toBeLessThan(1.0);
    core.dispose();
  });
});
