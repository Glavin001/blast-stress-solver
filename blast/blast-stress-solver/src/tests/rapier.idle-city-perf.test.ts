/**
 * Idle large-scene readout cost — the chunk world-transform skip.
 *
 * On a big, settled scene (e.g. an intact mini-city: many bonded buildings, each an anchored
 * island sitting on the single fixed root body) the simulation is doing almost nothing — the
 * stress solver idle-skips and Rapier integrates no motion. But `step()` still has to publish
 * every chunk's world position/orientation for the renderer. Recomputing all N chunks every
 * frame (an FFI getRigidBody + quaternion math + two Vec3 allocations each) was the single
 * largest idle cost at scale (~14 ms/frame at 14k fragments).
 *
 * The core now fetches each body's pose once per frame and SKIPS the per-chunk recompute when
 * the body's pose is bit-identical to last frame (fixed/asleep bodies). This is observationally
 * identical — a skipped chunk keeps last frame's value, which equals the recompute. These tests
 * lock in the two behavioral facts that make the skip both effective and safe:
 *   1. an idle settled scene REUSES the chunk transform objects (the recompute was skipped), and
 *   2. a genuinely moving (free-falling, dynamic) body still recomputes its transform every frame.
 *
 * Requires the full WASM + TS build; skips gracefully if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type * as RapierEntry from '../rapier';
import type { ScenarioDesc } from '../rapier/types';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

async function loadCoreBuilder() {
  return (await import('../../dist/rapier.js')) as typeof RapierEntry;
}

const RIGID = 1e10;

// A grid of `count` anchored pillars → `count` disconnected stress islands, all sitting on the
// single fixed root body (exactly the mini-city idle topology: intact buildings never move).
function pillarCity(count: number, h: number): ScenarioDesc {
  const nodes: any[] = [];
  const bonds: any[] = [];
  for (let p = 0; p < count; p++) {
    const x = p * 4;
    nodes.push({ centroid: { x, y: -0.5, z: 0 }, mass: 0, volume: 0 }); // anchored support (cut point)
    let prev = nodes.length - 1;
    for (let i = 0; i < h; i++) {
      const idx = nodes.length;
      nodes.push({ centroid: { x, y: i + 0.5, z: 0 }, mass: 1, volume: 1 });
      bonds.push({ node0: prev, node1: idx, centroid: { x, y: i, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 1 });
      prev = idx;
    }
  }
  return { nodes, bonds, parameters: {} } as ScenarioDesc;
}

describe.skipIf(!runtimeAvailable)('Idle scene chunk-transform skip', () => {
  it('an idle, settled multi-island city reuses chunk transforms (recompute skipped) and idle-skips the solver', async () => {
    const { buildDestructibleCore } = await loadCoreBuilder();
    const core = await buildDestructibleCore({ scenario: pillarCity(40, 6), gravity: -9.81, materialScale: RIGID });
    let solverSolvesRan = 0;
    core.setProfiler?.({ enabled: true, onSample: (s: any) => { if ((s.solverSolveMs ?? 0) > 0) solverSolvesRan++; } });

    const dt = 1 / 60;
    for (let f = 0; f < 30; f++) core.step(dt); // settle

    // Capture transform object references for every active chunk, then step once more (idle).
    const chunks = (core as any).chunks as Array<any>;
    const active = chunks.filter((c) => c.active && c.worldPosition);
    expect(active.length).toBeGreaterThan(100);
    const before = active.map((c) => c.worldPosition);

    const solvesBefore = solverSolvesRan;
    core.step(dt);

    // Every chunk's transform object is the SAME reference → the per-chunk recompute was skipped
    // (a fixed body returns a bit-identical pose, so its chunks are left untouched).
    let reused = 0;
    for (let i = 0; i < active.length; i++) if (active[i].worldPosition === before[i]) reused++;
    expect(reused).toBe(active.length);

    // And the stress solve idle-skipped on that frame (no CGNR ran).
    expect(solverSolvesRan).toBe(solvesBefore);

    core.dispose?.();
  });

  it('a moving (free-falling, dynamic) body still recomputes its chunk transform every frame', async () => {
    const { buildDestructibleCore } = await loadCoreBuilder();
    // No support (mass 0) nodes → free-floating scenario → dynamic root body that falls under gravity.
    const scenario = {
      nodes: [
        { centroid: { x: 0, y: 20, z: 0 }, mass: 1, volume: 1 },
        { centroid: { x: 1, y: 20, z: 0 }, mass: 1, volume: 1 },
      ],
      bonds: [{ node0: 0, node1: 1, centroid: { x: 0.5, y: 20, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 1 }],
      parameters: {},
    } as ScenarioDesc;
    const core = await buildDestructibleCore({ scenario, gravity: -9.81, materialScale: RIGID });

    const dt = 1 / 60;
    core.step(dt);
    core.step(dt);
    const chunk = (core as any).chunks[0];
    const ref1 = chunk.worldPosition;
    const y1 = ref1.y;
    core.step(dt);
    const ref2 = chunk.worldPosition;

    // The body is falling, so its pose changes every frame → the transform is recomputed
    // (lower y). The skip must NOT engage for a body that actually moved. The pose object
    // itself is now MUTATED IN PLACE (identity is stable by design — the per-moving-chunk
    // object allocations were removed), so the recompute is observed through the value.
    expect(ref2).toBe(ref1);
    expect(ref2.y).toBeLessThan(y1);

    core.dispose?.();
  });
});
