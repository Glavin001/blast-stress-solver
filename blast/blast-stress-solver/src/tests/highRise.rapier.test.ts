/**
 * Full-pipeline (Rapier + WASM) soak test for the high-rise.
 *
 * Non-blocking (soak suite): verifies the building is STABLE under gravity in the
 * real physics runtime — it neither fractures nor slides off its anchored base.
 * Behavior under impact is covered exhaustively + deterministically by the Rust
 * headless tests (`high_rise_scenarios_test.rs`); here we just confirm the shared
 * structure + decoupled concrete limits hold up in the Rapier integration.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 * Run: npm run build && npx vitest run src/tests/highRise.rapier.test.ts
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
  return dyn.reduce((s: number, c: any) => s + (c.worldPosition ?? c.baseLocalOffset ?? c.localOffset).y, 0) / dyn.length;
}

describe.skipIf(!runtimeAvailable)('high-rise (Rapier)', () => {
  it('stays stable under gravity (no fracture, no slide)', async () => {
    await loadCore();
    // Smaller building for a fast test; same structural model.
    const scenario = buildHighRiseScenario({ floorCount: 4, columnsX: 3, columnsZ: 2 });
    const core = await buildDestructibleCore({
      scenario,
      gravity: -9.81,
      materialScale: 1e10,
      solverSettings: CONCRETE_LIMITS,
      damage: { enabled: false },
      resimulateOnFracture: true,
      maxResimulationPasses: 1,
    });
    const initialBonds = core.getActiveBondsCount();
    const y0 = avgDynamicY(core);
    stepN(core, 150);
    const survival = core.getActiveBondsCount() / initialBonds;
    const drop = y0 - avgDynamicY(core);
    expect(survival).toBeGreaterThan(0.97); // essentially no bonds break under gravity
    expect(drop).toBeLessThan(0.5); // building does not collapse or slide off its base
    core.dispose();
  });
});
