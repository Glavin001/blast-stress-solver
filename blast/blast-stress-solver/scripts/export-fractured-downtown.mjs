// Export a DENSE downtown: many buildings of mixed height packed at street
// distance, so a toppling tower lands on its neighbours.
//
// This is the deliberate opposite of export-fractured-district.mjs. That script
// spaces buildings by what they can REACH when they fall (`pitch = width +
// height * clearance`), which keeps every debris field separate. The reason is
// real and worth restating: PhysX sleeps rigid bodies per CONTACT ISLAND, so
// once two rubble piles touch they become one island, and no body in it can
// sleep until the last one settles. Reach-based spacing is how the district
// keeps 16k bodies from all waiting on each other.
//
// The cost of that safety is a map you cannot knock over into itself: buildings
// stand too far apart to interact, which is most of the drama of a destructible
// city. This pack takes the other side of that trade on purpose. Streets are
// ~12 m, so a 36 m tower reaches two plots and an 84 m spire reaches most of a
// block. Expect merged rubble fields and a longer settle; that is the point,
// and it is why both packs exist rather than one replacing the other.
//
// Layout is row-packed rather than gridded: each row lays buildings left to
// right with a fixed street gap between FACES, so plots of different widths sit
// flush against the same street and no two buildings can overlap by
// construction. Heights are deliberately interleaved -- a 2-storey house next
// to an 84 m spire -- because a tower that only ever falls on other towers is
// not what makes a skyline interesting to demolish.
//
// Usage: SEED=7 node scripts/export-fractured-downtown.mjs

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FLOOR_HEIGHT, MATERIALS, buildBuilding, mulberry32, round, v } from './export-fractured-city.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SEED = Number(process.env.SEED ?? 7);
const OUTPUT = path.resolve(__dirname, process.env.OUTPUT ?? '../assets/mini-city/fractured-downtown.json');

/** PhysX cooks GPU convex meshes from at most 64 vertices. */
const MAX_HULL_POINTS = 64;

/**
 * Gap between building FACES, i.e. the street. Not a function of height: that
 * is exactly the reach-based rule this pack is rejecting.
 */
const STREET_M = Number(process.env.STREET_M ?? 12);

/** Extra breathing room between rows, on top of STREET_M. */
const AVENUE_M = Number(process.env.AVENUE_M ?? 4);

const ARCHETYPES = {
  house: { width: 8, floors: 2, label: 'house' },
  shop: { width: 12, floors: 3, label: 'shop row' },
  midrise: { width: 12, floors: 6, label: 'mid-rise' },
  block: { width: 20, floors: 5, label: 'wide block' },
  tower: { width: 12, floors: 12, label: 'tower' },
  skyscraper: { width: 16, floors: 20, label: 'skyscraper' },
  spire: { width: 18, floors: 28, label: 'spire' },
};

/**
 * Rows, north to south. Heights are interleaved on purpose so that every tall
 * building has something short beside it to fall on.
 */
const ROWS = [
  ['skyscraper', 'house', 'tower', 'shop', 'midrise'],
  ['shop', 'midrise', 'house', 'spire', 'house', 'tower'],
  ['block', 'tower', 'house', 'midrise', 'shop'],
  ['house', 'house', 'midrise', 'tower', 'block', 'house'],
  ['shop', 'tower', 'midrise', 'house', 'shop'],
];

function hullVolumeCentroid(flat) {
  // Same convex-hull volume/centroid reduction the district export uses: the
  // builder emits hull points about a nominal centre, and the solver needs the
  // true centroid so a chunk's rest offset matches its mass distribution.
  let volume = 0;
  const centroid = [0, 0, 0];
  const points = [];
  for (let i = 0; i < flat.length; i += 3) {
    points.push([flat[i], flat[i + 1], flat[i + 2]]);
  }
  let minimum = [Infinity, Infinity, Infinity];
  let maximum = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis], point[axis]);
      maximum[axis] = Math.max(maximum[axis], point[axis]);
    }
  }
  volume = (maximum[0] - minimum[0]) * (maximum[1] - minimum[1]) * (maximum[2] - minimum[2]);
  for (let axis = 0; axis < 3; axis++) {
    centroid[axis] = (minimum[axis] + maximum[axis]) / 2;
  }
  return { volume, centroid, count: points.length };
}

function correctCentroids(building) {
  let maxPoints = 0;
  for (let i = 0; i < building.nodeColliders.length; i++) {
    const collider = building.nodeColliders[i];
    if (!collider || collider.kind !== 'hull') continue;
    const { volume, centroid, count } = hullVolumeCentroid(collider.points);
    maxPoints = Math.max(maxPoints, count);
    const node = building.nodes[i];
    node.centroid = v(
      node.centroid.x + centroid[0],
      node.centroid.y + centroid[1],
      node.centroid.z + centroid[2],
    );
    node.volume = round(volume);
  }
  return maxPoints;
}

/**
 * Row-pack the archetypes into plot centres.
 *
 * Advancing the cursor by half of each neighbour's width plus the street keeps
 * the gap between FACES constant, which is what makes a mixed-width row read as
 * a street rather than as a grid with holes in it.
 */
function layout(rng) {
  const plots = [];
  const rowDepths = ROWS.map((row) => Math.max(...row.map((kind) => ARCHETYPES[kind].width)));
  const totalDepth =
    rowDepths.reduce((a, b) => a + b, 0) + (ROWS.length - 1) * (STREET_M + AVENUE_M);
  let cz = -totalDepth / 2;

  ROWS.forEach((row, rowIndex) => {
    const depth = rowDepths[rowIndex];
    cz += depth / 2;
    const widths = row.map((kind) => ARCHETYPES[kind].width);
    const rowWidth = widths.reduce((a, b) => a + b, 0) + (row.length - 1) * STREET_M;
    let cx = -rowWidth / 2;

    row.forEach((kind, index) => {
      const archetype = ARCHETYPES[kind];
      cx += archetype.width / 2;
      // Jitter is a fraction of the street, never enough to close it: two
      // buildings touching at rest would be one island before a shot is fired.
      const jitter = () => (rng() - 0.5) * STREET_M * 0.25;
      plots.push({
        kind,
        archetype,
        cx: cx + jitter(),
        cz: cz + jitter(),
        height: archetype.floors * FLOOR_HEIGHT,
      });
      cx += archetype.width / 2 + STREET_M;
    });

    cz += depth / 2 + STREET_M + AVENUE_M;
  });

  return plots;
}

/** Fail loudly if the layout ever lets two footprints intersect. */
function assertNoOverlap(plots) {
  for (let i = 0; i < plots.length; i++) {
    for (let j = i + 1; j < plots.length; j++) {
      const a = plots[i];
      const b = plots[j];
      const gapX = Math.abs(a.cx - b.cx) - (a.archetype.width + b.archetype.width) / 2;
      const gapZ = Math.abs(a.cz - b.cz) - (a.archetype.width + b.archetype.width) / 2;
      if (gapX < 0 && gapZ < 0) {
        throw new Error(
          `${a.kind} at (${round(a.cx)}, ${round(a.cz)}) overlaps ` +
            `${b.kind} at (${round(b.cx)}, ${round(b.cz)})`,
        );
      }
    }
  }
}

function main() {
  const rng = mulberry32(SEED * 2654435761);
  const nodes = [];
  const nodeTypes = [];
  const nodeSizes = [];
  const nodeColliders = [];
  const bonds = [];
  const placed = [];
  let maxPoints = 0;

  const plots = layout(rng);
  assertNoOverlap(plots);

  for (const plot of plots) {
    const building = buildBuilding({
      width: plot.archetype.width,
      floors: plot.archetype.floors,
      rng,
    });
    maxPoints = Math.max(maxPoints, correctCentroids(building));

    const base = nodes.length;
    for (const node of building.nodes) {
      nodes.push({
        centroid: v(node.centroid.x + plot.cx, node.centroid.y, node.centroid.z + plot.cz),
        mass: node.mass,
        volume: node.volume,
      });
    }
    nodeTypes.push(...building.nodeTypes);
    nodeSizes.push(...building.nodeSizes);
    nodeColliders.push(...building.nodeColliders);
    for (const bond of building.bonds) {
      bonds.push({
        node0: bond.node0 + base,
        node1: bond.node1 + base,
        centroid: v(bond.centroid.x + plot.cx, bond.centroid.y, bond.centroid.z + plot.cz),
        normal: bond.normal,
        area: bond.area,
        ...(bond.m ? { m: bond.m } : {}),
      });
    }
    placed.push({
      kind: plot.archetype.label,
      cx: plot.cx,
      cz: plot.cz,
      height: plot.height,
      nodes: building.nodes.length,
    });
  }

  if (maxPoints > MAX_HULL_POINTS) {
    throw new Error(`hull with ${maxPoints} points exceeds the ${MAX_HULL_POINTS}-point GPU cap`);
  }

  return {
    pack: {
      version: 2,
      key: 'fractured-downtown',
      title: 'Fractured downtown (dense mixed-height blocks, street spacing)',
      defaults: { solver: { gravity: -9.81, materialScale: 1, materials: MATERIALS } },
      scenario: { nodeTypes, nodes, bonds, nodeSizes, nodeColliders },
    },
    placed,
    maxPoints,
  };
}

const { pack, placed, maxPoints } = main();
await mkdir(path.dirname(OUTPUT), { recursive: true });
await writeFile(OUTPUT, JSON.stringify(pack));

const extentX = placed.reduce((acc, p) => Math.max(acc, Math.abs(p.cx)), 0);
const extentZ = placed.reduce((acc, p) => Math.max(acc, Math.abs(p.cz)), 0);
process.stderr.write(`wrote ${OUTPUT}\n`);
process.stderr.write(
  `buildings=${placed.length} nodes=${pack.scenario.nodes.length} ` +
    `bonds=${pack.scenario.bonds.length} maxHullPoints=${maxPoints} ` +
    `extent=${Math.round(extentX * 2)}x${Math.round(extentZ * 2)} m ` +
    `tallest=${Math.max(...placed.map((p) => p.height))} m street=${STREET_M} m\n`,
);
for (const [kind, count] of Object.entries(
  placed.reduce((acc, p) => ({ ...acc, [p.kind]: (acc[p.kind] ?? 0) + 1 }), {}),
)) {
  process.stderr.write(`  ${count} x ${kind}\n`);
}
