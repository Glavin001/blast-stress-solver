/**
 * Headless perf check for the FRACTURED house (requires the WASM build).
 *
 * Fine Voronoi fracturing used to explode the chunk/contact count (a recording showed
 * ~2718 nodes / 7824 bonds and ~18.7 ms/frame spent in JS contact injection alone). The
 * builder now coarsens the base grid before shattering, and the demo caps fracture churn,
 * so the piece count — and therefore the per-frame contact-injection cost — stays sane.
 *
 * This logs the piece count + per-frame step time (intact and after a hit) and guards the
 * piece count against regressing back to the explosion. Timing is logged, not asserted
 * (machine-dependent).
 *
 * Run: npm --prefix blast/blast-stress-solver run build && \
 *      npx vitest run src/tests/house.fracture.perf.test.ts
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
let buildHouseScenarioAsync: (opts?: any) => Promise<any>;

beforeAll(async () => {
  const rapier = await import('../../dist/rapier.js');
  const scenarios = await import('../../dist/scenarios.js');
  buildDestructibleCore = rapier.buildDestructibleCore;
  buildHouseScenario = scenarios.buildHouseScenario;
  buildHouseScenarioAsync = scenarios.buildHouseScenarioAsync;
});

function timeSteps(core: any, n: number): { avg: number; max: number } {
  let total = 0;
  let max = 0;
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    core.step(1 / 60);
    const dt = performance.now() - t0;
    total += dt;
    if (dt > max) max = dt;
  }
  return { avg: total / n, max };
}

d('fractured house performance', () => {
  it('keeps the piece count sane and the step time reasonable', async () => {
    const plain = buildHouseScenario({ furniture: false });
    const fractured = await buildHouseScenarioAsync({ furniture: false, fracture: 'wallsRoof' });

    const plainNodes = plain.nodes.length;
    const fracturedNodes = fractured.nodes.length;
    const fracturedBonds = fractured.bonds.length;
    // eslint-disable-next-line no-console
    console.log(`[pieces] plain=${plainNodes} nodes/${plain.bonds.length} bonds · fractured=${fracturedNodes} nodes/${fracturedBonds} bonds (${(fracturedNodes / plainNodes).toFixed(1)}x)`);

    const core = await buildDestructibleCore({
      scenario: fractured,
      gravity: -9.81,
      materialScale: 4e9,
      solverSettings: { maxSolverIterationsPerFrame: 24, graphReductionLevel: 0 },
      contactForceScale: 14,
      friction: 0.5,
      restitution: 0,
      debrisCollisionMode: 'noDebrisPairs',
      fracturePolicy: { maxNewBodiesPerFrame: 30, maxColliderMigrationsPerFrame: 60 },
    });

    const intact = timeSteps(core, 120);
    core.enqueueProjectile({ position: { x: -9, y: 2.4, z: 0 }, velocity: { x: 67, y: 0, z: 0 }, radius: 0.9, mass: 4000, ttl: 6000 });
    const afterHit = timeSteps(core, 240);
    // eslint-disable-next-line no-console
    console.log(`[step ms] intact avg=${intact.avg.toFixed(2)} max=${intact.max.toFixed(2)} · after-hit avg=${afterHit.avg.toFixed(2)} max=${afterHit.max.toFixed(2)}`);

    // Regression guard: the coarsen-then-shatter path must stay far below the old explosion
    // (~2718 nodes); ~3x the plain house was the symptom.
    expect(fracturedNodes).toBeLessThan(plainNodes * 2);
    core.dispose();
  }, 60000);
});
