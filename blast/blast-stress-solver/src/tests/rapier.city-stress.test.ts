/**
 * City-scale destruction stress tests.
 *
 * Two layers:
 *  1. Pure-data tests (always run): validate the scenario composer + city builders
 *     — index remapping, heterogeneous sizing via fragmentSizes, island separation,
 *     and determinism. No WASM needed.
 *  2. Full-pipeline stress (skipIf no WASM): run the city scenarios through
 *     buildDestructibleCore, collect profiler samples, and print the costliest-step
 *     report. Mirrors rapier.perf.test.ts conventions.
 *
 * Run: npm run build && npx vitest run src/tests/rapier.city-stress.test.ts
 */
import { describe, it, expect, afterAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { composeScenarios } from './stress/composeScenarios';
import { buildCityScenario, CITY_SCENARIOS, CITY_SCENARIO_NAMES, mulberry32 } from './stress/cityScenario';
import { runStressScenario, printReport, type BuildCore, type ScenarioResult } from './stress/harness';
import { buildTowerScenario } from '../scenarios/towerScenario';
import { buildWallScenario } from '../scenarios/wallScenario';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

// ── Layer 1: pure-data composer / builder validation ──────────────────────────
describe('scenario composer', () => {
  it('concatenates nodes/bonds and remaps indices without cross-building bonds', () => {
    const a = buildTowerScenario({ side: 3, stories: 4 });
    const b = buildWallScenario({ spanSegments: 4, heightSegments: 3 });
    const composed = composeScenarios([
      { scenario: a, offset: { x: 0, y: 0, z: 0 }, tag: 'A' },
      { scenario: b, offset: { x: 20, y: 0, z: 0 }, tag: 'B' },
    ]);

    expect(composed.nodes.length).toBe(a.nodes.length + b.nodes.length);
    expect(composed.bonds.length).toBe(a.bonds.length + b.bonds.length);
    // One size per node — the only mechanism that preserves heterogeneous sizing.
    expect(composed.parameters.fragmentSizes.length).toBe(composed.nodes.length);

    // No bond may straddle two buildings → each part is its own island.
    for (const range of composed.parameters.buildings) {
      const lo = range.nodeStart;
      const hi = range.nodeStart + range.nodeCount;
      for (let i = range.bondStart; i < range.bondStart + range.bondCount; i++) {
        const bond = composed.bonds[i];
        expect(bond.node0).toBeGreaterThanOrEqual(lo);
        expect(bond.node0).toBeLessThan(hi);
        expect(bond.node1).toBeGreaterThanOrEqual(lo);
        expect(bond.node1).toBeLessThan(hi);
      }
    }
  });

  it('applies translation to centroids', () => {
    const a = buildTowerScenario({ side: 3, stories: 4 });
    const composed = composeScenarios([{ scenario: a, offset: { x: 100, y: 5, z: -50 } }]);
    for (let i = 0; i < a.nodes.length; i++) {
      expect(composed.nodes[i].centroid.x).toBeCloseTo(a.nodes[i].centroid.x + 100, 6);
      expect(composed.nodes[i].centroid.y).toBeCloseTo(a.nodes[i].centroid.y + 5, 6);
      expect(composed.nodes[i].centroid.z).toBeCloseTo(a.nodes[i].centroid.z - 50, 6);
    }
  });

  it('rotateY=90 swaps X/Z collider extents', () => {
    const a = buildTowerScenario({ side: 2, stories: 3, spacing: { x: 1, y: 0.5, z: 2 } });
    const straight = composeScenarios([{ scenario: a }]);
    const turned = composeScenarios([{ scenario: a, rotateY: 90 }]);
    expect(turned.parameters.fragmentSizes[0].x).toBeCloseTo(straight.parameters.fragmentSizes[0].z, 6);
    expect(turned.parameters.fragmentSizes[0].z).toBeCloseTo(straight.parameters.fragmentSizes[0].x, 6);
  });
});

describe('city builders', () => {
  it('build deterministically for a given seed', () => {
    const c1 = buildCityScenario({ gridX: 3, gridZ: 3, seed: 42 });
    const c2 = buildCityScenario({ gridX: 3, gridZ: 3, seed: 42 });
    expect(c1.nodes.length).toBe(c2.nodes.length);
    expect(c1.bonds.length).toBe(c2.bonds.length);
    expect(c1.nodes[0].centroid).toEqual(c2.nodes[0].centroid);
    expect(c1.parameters.buildings.map((b) => b.tag)).toEqual(c2.parameters.buildings.map((b) => b.tag));
  });

  it('mulberry32 is reproducible', () => {
    const r1 = mulberry32(7);
    const r2 = mulberry32(7);
    expect([r1(), r1(), r1()]).toEqual([r2(), r2(), r2()]);
  });

  it.each(CITY_SCENARIO_NAMES)('preset %s builds a valid, island-separated world', (name) => {
    const useCase = CITY_SCENARIOS[name]('small');
    const s = useCase.scenario;
    expect(s.nodes.length).toBeGreaterThan(0);
    expect(s.bonds.length).toBeGreaterThan(0);
    expect(s.parameters.fragmentSizes.length).toBe(s.nodes.length);
    expect(s.parameters.buildings.length).toBeGreaterThan(0);
    // Every building has at least one static base node (mass 0) anchoring it.
    for (const range of s.parameters.buildings) {
      let hasStatic = false;
      for (let i = range.nodeStart; i < range.nodeStart + range.nodeCount; i++) {
        if (s.nodes[i].mass === 0) { hasStatic = true; break; }
      }
      expect(hasStatic).toBe(true);
    }
    expect(useCase.impactPlan.length).toBeGreaterThan(0);
  });
});

// ── Layer 2: full pipeline (requires WASM build) ──────────────────────────────
const stressResults: ScenarioResult[] = [];

describe.skipIf(!runtimeAvailable)('City destruction stress (requires WASM build)', () => {
  let buildDestructibleCore: BuildCore;

  async function load() {
    if (buildDestructibleCore) return;
    const rapier = await import('../../dist/rapier.js');
    buildDestructibleCore = rapier.buildDestructibleCore as unknown as BuildCore;
  }

  it.each(CITY_SCENARIO_NAMES)('runs %s through the pipeline and profiles it', async (name) => {
    await load();
    const useCase = CITY_SCENARIOS[name]('small');
    const result = await runStressScenario({
      name: useCase.name,
      scenario: useCase.scenario,
      buildCore: buildDestructibleCore,
      coreOpts: useCase.coreOpts,
      impactPlan: useCase.impactPlan,
      postImpactFrames: useCase.postImpactFrames,
    });
    stressResults.push(result);
    expect(result.samples.length).toBeGreaterThan(0);
    // Every preset's impact plan is destructive — some bonds must break (unless a
    // crash cut the run short, which we tolerate at extreme scale).
    if (!result.crashed && result.bondsFinal >= 0) {
      expect(result.bondsFinal).toBeLessThan(result.bondsInitial);
    }
  }, 180_000);

  afterAll(() => {
    if (stressResults.length) printReport(stressResults);
  });
});
