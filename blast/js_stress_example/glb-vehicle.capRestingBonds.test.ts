/**
 * Unit tests for capRestingBonds — the pure bond-graph logic that collapses a
 * resting cargo/accessory part's many footprint bonds down to a single weak
 * contact. No GLB / WASM needed.
 */
import { describe, it, expect, vi } from 'vitest';

// glb-vehicle.ts imports browser-only modules (three, blast-stress-solver/three)
// that are provided via importmap in the demo, not installed for node. capRestingBonds
// is pure graph logic and touches none of them, so stub them out to load the module.
vi.mock('three', () => ({ Box3: class {}, Vector3: class {} }));
vi.mock('blast-stress-solver/three', () => ({
  buildScenarioFromFragments: () => undefined,
  buildScenarioFromFragmentsAsync: async () => undefined,
  fractureGeometryAsync: async () => undefined,
  recenterGeometry: () => undefined,
}));

import { capRestingBonds, type VehiclePartRole } from './glb-vehicle';

type Meta = { role: VehiclePartRole };
const bond = (node0: number, node1: number, area: number) => ({ node0, node1, area });

describe('capRestingBonds', () => {
  it('reduces a cargo part glued along a footprint to its single strongest bond', () => {
    // node0 = frame, node1 = cargo crate bonded to the frame by 4 footprint bonds.
    const meta: Meta[] = [{ role: 'frame' }, { role: 'cargo' }];
    const scenario = { bonds: [bond(0, 1, 0.05), bond(0, 1, 0.2), bond(0, 1, 0.1), bond(0, 1, 0.02)] };
    const removed = capRestingBonds(scenario, meta);
    expect(removed).toBe(3);
    expect(scenario.bonds).toHaveLength(1);
    expect(scenario.bonds[0].area).toBe(0.2); // strongest contact kept
  });

  it('leaves structural (non-capped) bonds untouched', () => {
    const meta: Meta[] = [{ role: 'frame' }, { role: 'frame' }, { role: 'wheel' }];
    const scenario = { bonds: [bond(0, 1, 0.3), bond(0, 1, 0.4), bond(1, 2, 0.2)] };
    const removed = capRestingBonds(scenario, meta);
    expect(removed).toBe(0);
    expect(scenario.bonds).toHaveLength(3);
  });

  it('keeps a bond that is the strongest link of EITHER endpoint', () => {
    // cargo(0)––cargo(1) stacked, plus each glued to the frame(2,3).
    // The cargo–cargo bond is node1's strongest, so it survives even if it is not
    // node0's strongest — neither cargo gets orphaned.
    const meta: Meta[] = [{ role: 'cargo' }, { role: 'cargo' }, { role: 'frame' }, { role: 'frame' }];
    const scenario = {
      bonds: [
        bond(0, 2, 0.30), // cargo0 strongest → frame
        bond(0, 1, 0.10), // cargo0–cargo1: not cargo0's best, but IS cargo1's best
        bond(1, 3, 0.05), // cargo1 weak → frame
      ],
    };
    const removed = capRestingBonds(scenario, meta);
    // cargo0 keeps bond#0; cargo1 keeps bond#1 (its strongest). bond#2 dropped.
    expect(scenario.bonds.map((b) => b.area).sort()).toEqual([0.1, 0.3]);
    expect(removed).toBe(1);
    // Every cargo node still has at least one bond (no orphans).
    for (const node of [0, 1]) {
      expect(scenario.bonds.some((b) => b.node0 === node || b.node1 === node)).toBe(true);
    }
  });

  it('respects maxBondsPerNode and the roles filter', () => {
    const meta: Meta[] = [{ role: 'frame' }, { role: 'accessory' }];
    const scenario = { bonds: [bond(0, 1, 0.1), bond(0, 1, 0.2), bond(0, 1, 0.3)] };
    // Keep the top 2 accessory bonds.
    const removed = capRestingBonds(scenario, meta, { maxBondsPerNode: 2 });
    expect(removed).toBe(1);
    expect(scenario.bonds.map((b) => b.area).sort()).toEqual([0.2, 0.3]);

    // With accessory excluded from the roles filter, nothing is capped.
    const scenario2 = { bonds: [bond(0, 1, 0.1), bond(0, 1, 0.2)] };
    expect(capRestingBonds(scenario2, meta, { roles: ['cargo'] })).toBe(0);
  });
});
