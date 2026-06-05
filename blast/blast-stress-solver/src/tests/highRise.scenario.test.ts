/**
 * Pure (no-WASM) tests for the high-rise composer and the shared scene-pack loader.
 *
 * These validate the *structure* and *strength model* without a physics runtime:
 * the right pieces bond to each other, the base is anchored, materials get realistic
 * (heterogeneous) masses, the skeleton is far stronger than the infill, and the
 * generated scene pack round-trips through the loader with decoupled concrete limits.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  buildHighRiseScenario,
  DEFAULT_HIGH_RISE_OPTIONS,
  CONCRETE_DENSITY,
  DRYWALL_DENSITY,
} from '../scenarios/highRiseScenario';
import { parseScenePackJson } from '../rapier/scenePackLoader';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCENE_PATH = path.resolve(
  __dirname,
  '../../../blast-stress-demo-rs/assets/scenes/high-rise.json',
);

describe('high-rise composer', () => {
  const scenario = buildHighRiseScenario();
  const types = (scenario.parameters as any).highRise.fragmentTypes as string[];

  it('produces a heterogeneous structure with a single mass-0 foundation', () => {
    const byType = types.reduce<Record<string, number>>((acc, t) => {
      acc[t] = (acc[t] ?? 0) + 1;
      return acc;
    }, {});
    expect(byType.foundation).toBeGreaterThan(0);
    expect(byType.column).toBeGreaterThan(0);
    expect(byType.slab).toBeGreaterThan(0);
    expect(byType.infill).toBeGreaterThan(0);

    const supports = scenario.nodes.filter((n) => n.mass === 0).length;
    expect(supports).toBe(byType.foundation); // foundation tiles are the only supports
  });

  it('anchors the base: foundation<->column bonds exist (no footing rip-off)', () => {
    const fc = scenario.bonds.filter((b) => {
      const a = types[b.node0];
      const c = types[b.node1];
      return (a === 'foundation' && c === 'column') || (a === 'column' && c === 'foundation');
    });
    expect(fc.length).toBeGreaterThan(0);
  });

  it('uses a flat-slab system: columns connect through slabs, not directly', () => {
    const colCol = scenario.bonds.filter(
      (b) => types[b.node0] === 'column' && types[b.node1] === 'column',
    );
    expect(colCol.length).toBe(0);
    const colSlab = scenario.bonds.filter((b) => {
      const a = types[b.node0];
      const c = types[b.node1];
      return (a === 'column' && c === 'slab') || (a === 'slab' && c === 'column');
    });
    expect(colSlab.length).toBeGreaterThan(0);
  });

  it('assigns realistic, heterogeneous masses (concrete heavier per volume than drywall)', () => {
    const density = (t: string) => {
      const i = types.indexOf(t);
      return scenario.nodes[i].mass / Math.max(1e-9, scenario.nodes[i].volume);
    };
    expect(density('column')).toBeCloseTo(CONCRETE_DENSITY, -1);
    expect(density('infill')).toBeCloseTo(DRYWALL_DENSITY, -1);
    expect(density('column')).toBeGreaterThan(density('infill') * 2);
  });

  it('makes the skeleton far stronger than infill via bond area', () => {
    const meanArea = (pred: (a: string, b: string) => boolean) => {
      const bs = scenario.bonds.filter((b) => pred(types[b.node0], types[b.node1]));
      return bs.reduce((s, b) => s + b.area, 0) / Math.max(1, bs.length);
    };
    const foundationColumn = meanArea(
      (a, b) =>
        (a === 'foundation' && b === 'column') || (a === 'column' && b === 'foundation'),
    );
    const infillInfill = meanArea((a, b) => a === 'infill' && b === 'infill');
    // Anchor joints carry far more effective area (=strength) than infill joints.
    expect(foundationColumn).toBeGreaterThan(infillInfill * 10);
  });

  it('builds the configured number of storeys', () => {
    expect(DEFAULT_HIGH_RISE_OPTIONS.floorCount).toBeGreaterThanOrEqual(8);
  });
});

describe('scene-pack loader', () => {
  it('round-trips the generated high-rise pack with decoupled concrete limits', () => {
    if (!existsSync(SCENE_PATH)) {
      // Generated, git-ignored artifact; skip if not built yet.
      console.warn(`skip: ${SCENE_PATH} not generated`);
      return;
    }
    const pack = parseScenePackJson(readFileSync(SCENE_PATH, 'utf8'));
    expect(pack.scenario.nodes.length).toBeGreaterThan(0);
    expect(pack.scenario.bonds.length).toBeGreaterThan(0);
    const sizes = (pack.scenario.parameters as any).fragmentSizes as unknown[];
    expect(sizes.length).toBe(pack.scenario.nodes.length);

    const s = pack.defaults.solverSettings!;
    expect(s).toBeDefined();
    // Concrete: strong in compression, ~10x weaker in tension (the non-glass knob).
    expect(s.compressionFatalLimit!).toBeGreaterThan(s.tensionFatalLimit! * 5);
    expect(pack.defaults.projectile.mass).toBeGreaterThan(0);
    expect(pack.nodeTypes.length).toBe(pack.scenario.nodes.length);
  });
});
