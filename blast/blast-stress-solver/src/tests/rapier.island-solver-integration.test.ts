/**
 * Stage 4: island-aware solving wired through the real destructible-core / Rapier pipeline.
 *
 * Confirms that DestructibleCore.setIslandSolver() actually drives the solver, that settled islands
 * are skipped in the full pipeline (the per-actor gravity rotated by a sleeping body's constant
 * orientation is bit-stable, so the skip engages), and that enabling it does not change the outcome.
 * ON by default since the Tier-1 perf defaults (docs/perf-city-scale-roadmap.md §3); the
 * `islandSolver: false` build option is the whole-graph opt-out.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

let buildDestructibleCore: (opts: any) => Promise<any>;
async function loadCore() {
  if (!buildDestructibleCore) buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
}

// Two separate anchored pillars with no bonds between them → two independent stress islands.
function twoPillars(h = 4, gap = 3) {
  const nodes: any[] = [];
  const bonds: any[] = [];
  const pillar = (x: number) => {
    nodes.push({ centroid: { x, y: 0, z: 0 }, mass: 0, volume: 0 }); // anchored support (mass 0)
    let prev = nodes.length - 1;
    for (let i = 1; i <= h; i++) {
      const idx = nodes.length;
      nodes.push({ centroid: { x, y: i, z: 0 }, mass: 1, volume: 1 });
      bonds.push({ node0: prev, node1: idx, centroid: { x, y: i - 0.5, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 1 });
      prev = idx;
    }
  };
  pillar(0);
  pillar(gap);
  return { nodes, bonds, spacing: { x: 1, y: 1, z: 1 } };
}

async function run(enabled: boolean) {
  const core = await buildDestructibleCore({ scenario: twoPillars(), gravity: -9.81, materialScale: 1e10, islandSolver: enabled });
  let maxIslandCount = 0;
  let maxSkipped = 0;
  for (let i = 0; i < 120; i++) {
    core.step(1 / 60);
    const s = core.getIslandSolverStats();
    maxIslandCount = Math.max(maxIslandCount, s.islandCount);
    maxSkipped = Math.max(maxSkipped, s.islandsSkipped);
  }
  const result = { enabled: core.getIslandSolverStats().enabled, maxIslandCount, maxSkipped, bonds: core.getActiveBondsCount() };
  core.dispose();
  return result;
}

describe.skipIf(!runtimeAvailable)('Island solver integration (requires WASM build)', () => {
  it('is ON by default (with settled-skip); islandSolver:false opts out completely', async () => {
    await loadCore();
    const def = await buildDestructibleCore({ scenario: twoPillars(), gravity: -9.81, materialScale: 1e10 });
    expect(def.getIslandSolverStats().enabled).toBe(true);
    expect(def.getIslandSolverStats().skipSettled).toBe(true);
    def.dispose();

    const r = await run(false);
    expect(r.enabled).toBe(false);
    expect(r.maxSkipped).toBe(0);
    expect(r.maxIslandCount).toBe(0);   // island partition isn't computed when island solving is off — zero-overhead
  });

  it('skips settled islands in the full pipeline, with the same outcome as disabled', async () => {
    await loadCore();
    const off = await run(false);
    const on = await run(true);
    expect(on.enabled).toBe(true);
    expect(on.maxIslandCount).toBe(2);
    expect(on.maxSkipped).toBeGreaterThan(0);   // settled islands really were skipped in the live sim
    expect(on.bonds).toBe(off.bonds);           // identical structural outcome — no correctness change
  });

  it('setIslandSolver drives the underlying solver flags (skip is gated by enabled)', async () => {
    await loadCore();
    const core = await buildDestructibleCore({ scenario: twoPillars(), gravity: -9.81, materialScale: 1e10, islandSolver: false });
    expect(core.solver.islandAware()).toBe(false);
    expect(core.solver.skipSettled()).toBe(false);
    core.setIslandSolver({ enabled: true });
    expect(core.solver.islandAware()).toBe(true);
    expect(core.solver.skipSettled()).toBe(true);
    core.setIslandSolver({ skipSettled: false });
    expect(core.solver.islandAware()).toBe(true);
    expect(core.solver.skipSettled()).toBe(false);
    core.setIslandSolver({ enabled: false, skipSettled: true });
    expect(core.solver.islandAware()).toBe(false);
    expect(core.solver.skipSettled()).toBe(false);   // enabled=false forces skip off too
    core.dispose();
  });
});
