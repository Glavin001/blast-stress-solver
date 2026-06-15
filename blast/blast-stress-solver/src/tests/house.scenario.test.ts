/**
 * Pure (no-WASM) tests for the house composer: the right pieces exist and bond, the base
 * is the only static anchor, materials get heterogeneous masses, the doorway is a real
 * hole, and the per-material strength model matches intent (furniture barely on the floor,
 * shelves clinging to walls, roof bonded but weaker than the frame).
 */
import { describe, it, expect } from 'vitest';
import {
  buildHouseScenario,
  makeHouseBondMultiplier,
  DEFAULT_HOUSE_OPTIONS,
  HOUSE_ROOF_DENSITY,
  HOUSE_FURNITURE_DENSITY,
} from '../scenarios/houseScenario';

describe('house composer', () => {
  const scenario = buildHouseScenario();
  const types = (scenario.parameters as any).house.fragmentTypes as string[];

  const byType = types.reduce<Record<string, number>>((acc, t) => {
    acc[t] = (acc[t] ?? 0) + 1;
    return acc;
  }, {});

  it('produces every house subsystem', () => {
    for (const t of ['foundation', 'floor', 'wall', 'beam', 'column', 'roof', 'furniture', 'shelf']) {
      expect(byType[t], `expected some "${t}" chunks`).toBeGreaterThan(0);
    }
  });

  it('anchors only the foundation (it is the sole static support)', () => {
    const supports = scenario.nodes.filter((n) => n.mass === 0).length;
    expect(supports).toBe(byType.foundation);
  });

  it('connects the body and anchors the floor to the foundation', () => {
    expect(scenario.bonds.length).toBeGreaterThan(scenario.nodes.length); // richly connected
    const foundationFloor = scenario.bonds.filter((b) => {
      const a = types[b.node0];
      const c = types[b.node1];
      return (a === 'foundation' && c === 'floor') || (a === 'floor' && c === 'foundation');
    });
    expect(foundationFloor.length).toBeGreaterThan(0);
  });

  it('leaves a walkable front doorway (no wall chunks fill the door hole)', () => {
    const { width, depth, wallThickness } = DEFAULT_HOUSE_OPTIONS;
    const frontZ = -depth * 0.5 + wallThickness * 0.5;
    const inDoor = scenario.nodes.filter((n, i) => {
      if (types[i] !== 'wall') return false;
      const c = n.centroid;
      return Math.abs(c.z - frontZ) < 0.25 && Math.abs(c.x) < 0.45 && c.y < 1.8;
    });
    expect(inDoor.length).toBe(0);
    void width;
  });

  it('roof bonds to itself and attaches to the frame', () => {
    const roofRoof = scenario.bonds.filter(
      (b) => types[b.node0] === 'roof' && types[b.node1] === 'roof',
    );
    expect(roofRoof.length).toBeGreaterThan(0);
    const roofToFrame = scenario.bonds.filter((b) => {
      const a = types[b.node0];
      const c = types[b.node1];
      const pair = (x: string) => (a === 'roof' && c === x) || (a === x && c === 'roof');
      return pair('beam') || pair('wall');
    });
    expect(roofToFrame.length, 'roof should bond to the ridge/plate or gable wall').toBeGreaterThan(0);
  });

  it('authors a single gable-prism collision node grouping every roof slope', () => {
    const roots = scenario.collisionTree!;
    expect(roots.length).toBe(1); // one building
    const children = roots[0].children!;
    const prismNodes = children.filter((c) => c.shape === 'prism');
    expect(prismNodes.length).toBe(1); // the gable roof → one ridge-prism render proxy
    // No bare roof leaves left at the top level: every roof fragment lives under the prism node.
    const roofIdx = new Set(types.flatMap((t, i) => (t === 'roof' ? [i] : [])));
    expect(roofIdx.size).toBeGreaterThan(0);
    const underPrism = new Set<number>();
    for (const slope of prismNodes[0].children ?? []) for (const f of slope.fragments ?? []) underPrism.add(f);
    expect(prismNodes[0].children!.length).toBe(2); // two gable slopes kept as separate collision leaves
    for (const i of roofIdx) expect(underPrism.has(i), `roof fragment ${i} under the prism node`).toBe(true);
    // And the prism node holds ONLY roof fragments.
    for (const i of underPrism) expect(roofIdx.has(i), `prism fragment ${i} is roof`).toBe(true);
  });

  it('has interior king posts that rise to support the ridge', () => {
    const ridgeY = (scenario.parameters as any).house.ridgeY as number;
    // King posts (columns) reach up near the ridge — the dispersed roof support.
    const tallColumns = scenario.nodes.filter(
      (n, i) => types[i] === 'column' && n.centroid.y > ridgeY - 1.0,
    );
    expect(tallColumns.length).toBeGreaterThan(0);
    // The roof load path runs through the frame beams (ridge / top plate).
    const roofBeam = scenario.bonds.filter((b) => {
      const a = types[b.node0];
      const c = types[b.node1];
      return (a === 'roof' && c === 'beam') || (a === 'beam' && c === 'roof');
    });
    expect(roofBeam.length).toBeGreaterThan(0);
  });

  it('assigns realistic heterogeneous masses (heavy floor slab, light roof/furniture)', () => {
    const density = (t: string) => {
      const i = types.indexOf(t);
      return scenario.nodes[i].mass / Math.max(1e-9, scenario.nodes[i].volume);
    };
    expect(density('roof')).toBeCloseTo(HOUSE_ROOF_DENSITY, -1);
    expect(density('furniture')).toBeCloseTo(HOUSE_FURNITURE_DENSITY, -1);
    expect(density('floor')).toBeGreaterThan(density('roof') * 2);
  });

  it('rests the structure on the ground (dynamic floor on y=0, foundation buried below)', () => {
    const sizes = (scenario.parameters as any).fragmentSizes as { y: number }[];
    let minDynamicBottom = Infinity;
    let hasBuriedSupport = false;
    scenario.nodes.forEach((n, i) => {
      const bottom = n.centroid.y - (sizes[i]?.y ?? 0) * 0.5;
      if (n.mass > 0) minDynamicBottom = Math.min(minDynamicBottom, bottom);
      else if (n.centroid.y < 0) hasBuriedSupport = true;
    });
    // Nothing dynamic dips below the runtime's static ground (top at y=0); foundation is buried.
    expect(minDynamicBottom).toBeGreaterThan(-0.05);
    expect(hasBuriedSupport).toBe(true);
  });

  it('carries furniture accent colors for a colorful interior', () => {
    const nodeColors = (scenario.parameters as any).house.nodeColors as (number | undefined)[];
    const furnitureColors = nodeColors.filter((c, i) => types[i] === 'furniture' && c != null);
    expect(furnitureColors.length).toBeGreaterThan(0);
  });
});

describe('house bond-strength model (structural intent)', () => {
  const m = makeHouseBondMultiplier();

  it('furniture is barely attached to the floor, far weaker than the foundation anchor', () => {
    expect(m('furniture', 'floor')).toBeLessThan(0.1);
    expect(m('foundation', 'floor')).toBeGreaterThan(m('furniture', 'floor') * 100);
  });

  it('shelves cling to walls more firmly than plain wall-to-wall', () => {
    expect(m('shelf', 'wall')).toBeGreaterThan(m('wall', 'wall'));
  });

  it('drywall is non-structural: its joints are far weaker than the wood frame', () => {
    // The load path post↔beam and post↔foundation must dwarf any drywall joint.
    expect(m('column', 'beam')).toBeGreaterThan(m('wall', 'column') * 10);
    expect(m('foundation', 'column')).toBeGreaterThan(m('wall', 'beam') * 10);
    expect(m('wall', 'column')).toBeLessThan(1);
    expect(m('wall', 'beam')).toBeLessThan(1);
  });

  it('the roof is carried by the frame but is not a rigid self-supporting plate', () => {
    // Roof attaches to the frame (beam/post) far more strongly than to itself, so an
    // unsupported span sags/fails instead of floating.
    expect(m('roof', 'beam')).toBeGreaterThan(m('roof', 'roof') * 2);
    expect(m('roof', 'column')).toBeGreaterThan(m('roof', 'roof') * 2);
    // ...yet the roof↔frame joint is still weaker than the primary load path.
    expect(m('roof', 'beam')).toBeLessThan(m('column', 'beam'));
    expect(m('roof', 'beam')).toBeLessThan(m('foundation', 'column'));
  });

  it('is symmetric in argument order', () => {
    expect(m('roof', 'beam')).toBe(m('beam', 'roof'));
    expect(m('furniture', 'floor')).toBe(m('floor', 'furniture'));
    expect(m('wall', 'column')).toBe(m('column', 'wall'));
  });
});
