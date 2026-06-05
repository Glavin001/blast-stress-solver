/**
 * Regression tests for the precomputed static splash adjacency.
 *
 * contactInjectSplash applies an attenuated force to each hit node's same-body
 * neighbours within SPLASH_RADIUS. Because baseLocalOffset (the chunk's original
 * asset-space centroid) never changes, those neighbours and their quadratic
 * falloff weights are precomputed once at init. These tests lock in that the
 * precompute is correct: right radius cutoff, right falloff, self excluded, and
 * byte-for-byte agreement with an independent spatial computation.
 *
 * Requires the full build (WASM + TS); skips if dist is unavailable.
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, '../../dist/stress_solver.wasm');
const runtimeAvailable = existsSync(wasmPath);

// Nodes on the x-axis at f32-exact coordinates, so distances (0.5, 1.0, 1.5) and
// the resulting weights (0.25, 0.0625, 0.5625) are exact in both f32 and f64 —
// no rounding slack in the assertions. SPLASH_RADIUS is 2.0; a node at exactly
// dist 2.0 has weight 0 and is excluded.
const NODES = [
  { centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 }, // support
  { centroid: { x: 1, y: 0, z: 0 }, mass: 1, volume: 1 },
  { centroid: { x: 1.5, y: 0, z: 0 }, mass: 1, volume: 1 },
  { centroid: { x: 3, y: 0, z: 0 }, mass: 1, volume: 1 },
];
const BONDS = [
  { node0: 0, node1: 1, centroid: { x: 0.5, y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 1 },
  { node0: 1, node1: 2, centroid: { x: 1.25, y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 1 },
  { node0: 2, node1: 3, centroid: { x: 2.25, y: 0, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 1 },
];

type Adj = { radius: number; perNode: Array<Array<{ node: number; weight: number }>> };
const sortByNode = (l: Array<{ node: number; weight: number }>) => [...l].sort((a, b) => a.node - b.node);

describe.skipIf(!runtimeAvailable)('splash adjacency precompute (requires WASM build)', () => {
  async function buildCore(): Promise<any> {
    const mod = await import('../../dist/rapier.js');
    return mod.buildDestructibleCore({
      scenario: { nodes: NODES, bonds: BONDS },
      gravity: -9.81,
      materialScale: 1.0,
    });
  }

  it('matches an independent brute-force spatial computation', async () => {
    const core = await buildCore();
    try {
      const adj: Adj = core.__debugSplashAdjacency();
      expect(adj.radius).toBeGreaterThan(0);
      expect(adj.perNode).toHaveLength(NODES.length);

      const R = adj.radius;
      for (let i = 0; i < NODES.length; i++) {
        // Replicate the precompute exactly: the runtime hit position arrives via a
        // Float32Array, so the origin is fround'd; the neighbour stays full-f64.
        const px = Math.fround(NODES[i].centroid.x);
        const py = Math.fround(NODES[i].centroid.y);
        const pz = Math.fround(NODES[i].centroid.z);
        const expected: Array<{ node: number; weight: number }> = [];
        for (let j = 0; j < NODES.length; j++) {
          if (j === i) continue;
          const dx = NODES[j].centroid.x - px;
          const dy = NODES[j].centroid.y - py;
          const dz = NODES[j].centroid.z - pz;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          const falloff = 1 - dist / R;
          const weight = falloff * falloff;
          if (dist < R && weight > 0) expected.push({ node: j, weight });
        }
        // Exact equality (not approximate): the precompute and this reference run
        // the identical float computation, so the stored weights must match bit-for-bit.
        expect(sortByNode(adj.perNode[i])).toEqual(sortByNode(expected));
      }
    } finally {
      core.dispose();
    }
  });

  it('applies quadratic falloff and excludes self + nodes at/beyond the radius', async () => {
    const core = await buildCore();
    try {
      const { perNode }: Adj = core.__debugSplashAdjacency();

      const n0 = new Map(perNode[0].map((e) => [e.node, e.weight]));
      expect(n0.has(0)).toBe(false); // self excluded
      expect(n0.get(1)).toBeCloseTo(0.25, 12); // dist 1.0 -> (1 - 0.5)^2
      expect(n0.get(2)).toBeCloseTo(0.0625, 12); // dist 1.5 -> (1 - 0.75)^2
      expect(n0.has(3)).toBe(false); // dist 3.0 (> R) excluded

      const n1 = new Map(perNode[1].map((e) => [e.node, e.weight]));
      expect(n1.get(2)).toBeCloseTo(0.5625, 12); // dist 0.5 -> (1 - 0.25)^2
      expect(n1.has(3)).toBe(false); // dist exactly 2.0 (== R, weight 0) excluded
    } finally {
      core.dispose();
    }
  });

  it('produces symmetric weights between mutual neighbours', async () => {
    const core = await buildCore();
    try {
      const { perNode }: Adj = core.__debugSplashAdjacency();
      const weight = (i: number, j: number) => perNode[i].find((e) => e.node === j)?.weight;
      expect(weight(0, 1)).toBe(weight(1, 0)); // dist 1.0
      expect(weight(1, 2)).toBe(weight(2, 1)); // dist 0.5
      expect(weight(2, 3)).toBe(weight(3, 2)); // dist 1.5
    } finally {
      core.dispose();
    }
  });
});
