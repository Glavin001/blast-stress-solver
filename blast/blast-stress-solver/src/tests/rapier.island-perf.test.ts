/**
 * Island-aware solving — performance comparison vs the legacy whole-graph solve.
 *
 * The headline win is a large, complexly-bonded world with LOCALIZED activity: most of the world has
 * settled, only a small part is being disturbed. The legacy solver re-iterates the entire graph every
 * frame to resolve the disturbed part; island+skip solves only the active island(s) and skips the
 * settled ones. The benefit scales with how much of the world is settled.
 *
 * Convention (matching rapier.perf.test.ts): measure + log timings, assert behavioral facts that are
 * noise-free (skip engaged, same outcome) plus a generous timing inequality that the large margin
 * makes robust across machines.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type * as Runtime from '..';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

async function loadRuntime() {
  return (await import('../../dist/index.js')) as typeof Runtime;
}

const rigid = {
  maxSolverIterationsPerFrame: 50,
  compressionElasticLimit: 1e12, compressionFatalLimit: 2e12,
  tensionElasticLimit: 1e12, tensionFatalLimit: 2e12,
  shearElasticLimit: 1e12, shearFatalLimit: 2e12,
};

// `count` independent anchored pillars (each `h` tall) → `count` disconnected stress islands.
function pillars(count: number, h: number) {
  const nodes: any[] = [];
  const bonds: any[] = [];
  const tops: number[] = [];
  const topPos: Array<{ x: number; y: number; z: number }> = [];
  for (let p = 0; p < count; p++) {
    const x = p * 2;
    nodes.push({ centroid: { x, y: 0, z: 0 }, mass: 0, volume: 0 }); // anchored support
    let prev = nodes.length - 1;
    for (let i = 1; i <= h; i++) {
      const idx = nodes.length;
      nodes.push({ centroid: { x, y: i, z: 0 }, mass: 1, volume: 1 });
      bonds.push({ centroid: { x, y: i - 0.5, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 1, node0: prev, node1: idx });
      prev = idx;
    }
    tops.push(nodes.length - 1);
    topPos.push({ x, y: h, z: 0 });
  }
  return { nodes, bonds, tops, topPos };
}

const G = { x: 0, y: -5, z: 0 };

describe.skipIf(!runtimeAvailable)('Island-aware solve performance', () => {
  it('large world + localized activity: island+skip beats the legacy whole-graph solve', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    const scene = pillars(120, 10); // 120 islands, ~1320 nodes, 1200 bonds
    const FRAMES = 60;

    // One island is disturbed every frame (a time-varying force at its top — like an ongoing impact);
    // the other 119 sit settled. Returns timing + how many islands were skipped on average + outcome.
    function measure(islandAware: boolean, skipSettled: boolean) {
      const solver = rt.createExtSolver({ nodes: scene.nodes, bonds: scene.bonds, settings: rigid });
      if (islandAware) solver.setIslandAware(true);
      if (skipSettled) solver.setSkipSettled(true);
      for (let f = 0; f < 15; f++) { solver.addGravity(G); solver.update(); } // settle
      let totalSkipped = 0;
      const t0 = performance.now();
      for (let f = 0; f < FRAMES; f++) {
        solver.addGravity(G);
        solver.addForce(scene.tops[0], scene.topPos[0], { x: Math.sin(f * 0.5) * 80, y: 0, z: Math.cos(f * 0.5) * 80 });
        solver.update();
        totalSkipped += solver.islandsSkipped();
      }
      const ms = performance.now() - t0;
      const overstressed = solver.overstressedBondCount();
      solver.destroy();
      return { ms, avgSkipped: totalSkipped / FRAMES, overstressed };
    }

    const legacy = measure(false, false);
    const islandOnly = measure(true, false);
    const islandSkip = measure(true, true);

    /* eslint-disable no-console */
    console.log(`\n  [island perf] 120 islands x 10, 1 active/frame, ${FRAMES} frames`);
    console.log(`    legacy (whole-graph)   ${legacy.ms.toFixed(1)} ms`);
    console.log(`    island-aware (no skip) ${islandOnly.ms.toFixed(1)} ms`);
    console.log(`    island + skip-settled  ${islandSkip.ms.toFixed(1)} ms   avgSkipped=${islandSkip.avgSkipped.toFixed(0)}/120`);
    console.log(`    speedup (skip vs legacy): ${(legacy.ms / islandSkip.ms).toFixed(1)}x\n`);
    /* eslint-enable no-console */

    // Correctness: none of the modes break a bond here (rigid material) — same outcome.
    expect(legacy.overstressed).toBe(0);
    expect(islandSkip.overstressed).toBe(0);
    // The skip engaged on the settled majority while keeping the active island live (so not all 120).
    expect(islandSkip.avgSkipped).toBeGreaterThan(100);
    expect(islandSkip.avgSkipped).toBeLessThan(120);
    // And it is faster than re-solving the whole graph (large real margin makes this robust).
    expect(islandSkip.ms).toBeLessThan(legacy.ms);
  });

  it('benefit scales with the settled fraction', async () => {
    const rt = await (await loadRuntime()).loadStressSolver();
    const scene = pillars(80, 10);
    const FRAMES = 40;

    // Disturb the first `active` islands each frame; measure island+skip vs legacy.
    function measure(islandSkip: boolean, active: number) {
      const solver = rt.createExtSolver({ nodes: scene.nodes, bonds: scene.bonds, settings: rigid });
      if (islandSkip) { solver.setIslandAware(true); solver.setSkipSettled(true); }
      for (let f = 0; f < 15; f++) { solver.addGravity(G); solver.update(); }
      const t0 = performance.now();
      for (let f = 0; f < FRAMES; f++) {
        solver.addGravity(G);
        for (let a = 0; a < active; a++) solver.addForce(scene.tops[a], scene.topPos[a], { x: Math.sin(f * 0.5 + a) * 80, y: 0, z: 0 });
        solver.update();
      }
      const ms = performance.now() - t0;
      solver.destroy();
      return ms;
    }

    const speedups: Array<{ active: number; speedup: number }> = [];
    for (const active of [1, 20, 80]) {
      const legacy = measure(false, active);
      const skip = measure(true, active);
      speedups.push({ active, speedup: legacy / skip });
    }
    /* eslint-disable no-console */
    console.log(`\n  [island perf] 80 islands, speedup by active fraction:`);
    for (const s of speedups) console.log(`    ${s.active}/80 active: ${s.speedup.toFixed(1)}x`);
    console.log('');
    /* eslint-enable no-console */

    // More settled (fewer active) → larger speedup. With nearly everything active there is no win to
    // expect, but with mostly-settled worlds island+skip should be clearly ahead.
    const mostlySettled = speedups.find((s) => s.active === 1)!;
    const mostlyActive = speedups.find((s) => s.active === 80)!;
    expect(mostlySettled.speedup).toBeGreaterThan(mostlyActive.speedup);
    expect(mostlySettled.speedup).toBeGreaterThan(1);
  });
});
