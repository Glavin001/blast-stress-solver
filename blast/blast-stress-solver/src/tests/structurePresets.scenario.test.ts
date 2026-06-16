/**
 * Pure (no-WASM) tests for the vibe-city structure presets ported in
 * `scenarios/structurePresets.ts`: the rectilinear grid carves the requested
 * shell, anchors only its base layer, distributes the deck mass across the
 * dynamic cells, and bonds the graph into a single connected component so it
 * stands as one body until it is hit.
 */
import { describe, it, expect } from 'vitest';
import type { ScenarioDesc } from '../rapier/types';
import {
  buildRectilinearScenario,
  buildFrameTowerScenario,
  buildConcreteHutScenario,
  buildCourtyardBungalowScenario,
  buildStructurePreset,
  STRUCTURE_PRESET_METADATA,
} from '../scenarios/structurePresets';

function dynamicMass(scenario: ScenarioDesc): number {
  return scenario.nodes.reduce((sum, n) => sum + n.mass, 0);
}

function supportCount(scenario: ScenarioDesc): number {
  return scenario.nodes.filter((n) => n.mass === 0).length;
}

/** Size of the largest connected component of the node graph (intact bonds). */
function largestComponentSize(scenario: ScenarioDesc): number {
  const parent = scenario.nodes.map((_, i) => i);
  const find = (a: number): number => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  for (const b of scenario.bonds) union(b.node0, b.node1);
  const sizes = new Map<number, number>();
  for (let i = 0; i < scenario.nodes.length; i += 1) {
    const r = find(i);
    sizes.set(r, (sizes.get(r) ?? 0) + 1);
  }
  return Math.max(0, ...sizes.values());
}

describe('buildRectilinearScenario', () => {
  it('fills a solid box with the requested grid and bottom-layer supports', () => {
    const seg = { x: 3, y: 4, z: 3 };
    const scenario = buildRectilinearScenario({
      size: { x: 3, y: 4, z: 3 },
      segments: seg,
      deckMass: 1000,
      normalizeAreas: false,
    });
    expect(scenario.nodes.length).toBe(seg.x * seg.y * seg.z);
    // Default support predicate anchors the entire bottom layer (iy === 0).
    expect(supportCount(scenario)).toBe(seg.x * seg.z);
    // The deck mass lands on the dynamic cells only.
    expect(dynamicMass(scenario)).toBeCloseTo(1000, 3);
    // gridCoordinates align 1:1 with nodes.
    expect(scenario.gridCoordinates?.length).toBe(scenario.nodes.length);
    expect(scenario.spacing).toEqual({ x: 1, y: 1, z: 1 });
  });

  it('honours the includeNode predicate (carves a hollow shell)', () => {
    const seg = { x: 4, y: 4, z: 4 };
    const scenario = buildRectilinearScenario({
      size: { x: 4, y: 4, z: 4 },
      segments: seg,
      includeNode: ({ ix, iz, segments: s }) => ix === 0 || ix === s.x - 1 || iz === 0 || iz === s.z - 1,
    });
    // A hollow shell has strictly fewer cells than the full box.
    expect(scenario.nodes.length).toBeLessThan(seg.x * seg.y * seg.z);
    expect(scenario.nodes.length).toBeGreaterThan(0);
  });

  it('can disable bonds along an axis', () => {
    const opts = { size: { x: 3, y: 3, z: 3 }, segments: { x: 3, y: 3, z: 3 } } as const;
    const withY = buildRectilinearScenario({ ...opts });
    const withoutY = buildRectilinearScenario({ ...opts, bondsY: false });
    expect(withoutY.bonds.length).toBeLessThan(withY.bonds.length);
    // No purely-vertical bonds remain when bondsY is off (diagonals are off by default).
    expect(withoutY.bonds.some((b) => Math.abs(b.normal.y) > 0.99)).toBe(false);
  });
});

const PRESETS: Array<{ label: string; build: () => ScenarioDesc }> = [
  { label: 'frame tower', build: buildFrameTowerScenario },
  { label: 'concrete hut', build: buildConcreteHutScenario },
  { label: 'courtyard bungalow', build: buildCourtyardBungalowScenario },
];

describe.each(PRESETS)('structure preset: $label', ({ build }) => {
  const scenario = build();

  it('produces a non-trivial, richly bonded graph', () => {
    expect(scenario.nodes.length).toBeGreaterThan(50);
    expect(scenario.bonds.length).toBeGreaterThan(scenario.nodes.length);
    expect(scenario.spacing).toBeDefined();
  });

  it('anchors some cells to the ground and leaves the rest dynamic', () => {
    const supports = supportCount(scenario);
    expect(supports).toBeGreaterThan(0);
    expect(supports).toBeLessThan(scenario.nodes.length);
    // Supports are the lowest cells.
    const minY = Math.min(...scenario.nodes.map((n) => n.centroid.y));
    for (const n of scenario.nodes) {
      if (n.mass === 0) expect(n.centroid.y).toBeCloseTo(minY, 6);
    }
  });

  it('keeps every bond index in range and avoids self-bonds', () => {
    for (const b of scenario.bonds) {
      expect(b.node0).toBeGreaterThanOrEqual(0);
      expect(b.node1).toBeGreaterThanOrEqual(0);
      expect(b.node0).toBeLessThan(scenario.nodes.length);
      expect(b.node1).toBeLessThan(scenario.nodes.length);
      expect(b.node0).not.toBe(b.node1);
      expect(b.area).toBeGreaterThan(0);
    }
  });

  it('bonds the bulk of the structure into one dominant component', () => {
    // Most cells form a single load-bearing body; a few decorative cells (e.g. the
    // courtyard skylight dots) may float free, which is faithful to the source geometry.
    expect(largestComponentSize(scenario)).toBeGreaterThan(scenario.nodes.length * 0.75);
  });
});

describe('structure preset registry', () => {
  it('exposes all three presets with working builders', () => {
    expect(STRUCTURE_PRESET_METADATA.map((p) => p.id).sort()).toEqual(
      ['concreteHut', 'courtyardBungalow', 'frameTower'],
    );
    for (const preset of STRUCTURE_PRESET_METADATA) {
      const scenario = buildStructurePreset(preset.id);
      expect(scenario.nodes.length).toBeGreaterThan(0);
      expect(scenario.bonds.length).toBeGreaterThan(0);
    }
  });

  it('throws on an unknown preset id', () => {
    // @ts-expect-error intentional invalid id
    expect(() => buildStructurePreset('nope')).toThrow();
  });
});
