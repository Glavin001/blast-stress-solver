// Export a city DISTRICT: several buildings of mixed character, laid out so
// their debris fields stay separate.
//
// Why this exists rather than another uniform grid: PhysX sleeps rigid bodies
// per CONTACT ISLAND. A grid of identical towers at identical spacing
// collapses into one continuous rubble field, which is a single island --
// every body in the city then waits on every other body before any of them
// can sleep, and one late-settling piece anywhere keeps 20,000 bodies awake.
//
// Spacing is therefore derived from what each building can REACH when it
// falls. A toppling tower's debris lands within roughly its own height, so a
// gap wider than that means the pile it makes cannot merge with its
// neighbour's. Districts then settle independently, and the map is more
// interesting to look at than a grid besides.
//
// The layout is deliberately not uniform:
//   - houses cluster tightly; they cannot reach each other anyway
//   - mid-rises sit at conversational distance, so a big collapse can clip a
//     neighbour without guaranteeing it
//   - towers and the skyscraper get clearance proportional to their height,
//     so they fall dramatically without welding the whole map together
//
// Usage: SEED=7 node scripts/export-fractured-district.mjs

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FLOOR_HEIGHT, MATERIALS, buildBuilding, mulberry32, round, v } from './export-fractured-city.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SEED = Number(process.env.SEED ?? 7);
const OUTPUT = path.resolve(__dirname, process.env.OUTPUT ?? '../assets/mini-city/fractured-district.json');

/** PhysX cooks GPU convex meshes from at most 64 vertices. */
const MAX_HULL_POINTS = 64;

/**
 * Building archetypes. `width` is the footprint; `floors` sets the height and
 * therefore the reach. `clearance` is the gap left around the building beyond
 * its own footprint, expressed as a multiple of its height -- 1.0 means "a
 * fallen tower's debris stops short of the next plot".
 */
const ARCHETYPES = {
  house: { width: 8, floors: 2, clearance: 0.55, label: 'house' },
  shop: { width: 12, floors: 3, clearance: 0.6, label: 'shop row' },
  midrise: { width: 12, floors: 6, clearance: 0.75, label: 'mid-rise' },
  block: { width: 20, floors: 5, clearance: 0.7, label: 'wide block' },
  tower: { width: 12, floors: 12, clearance: 0.9, label: 'tower' },
  skyscraper: { width: 16, floors: 20, clearance: 1.0, label: 'skyscraper' },
};

/**
 * Districts, each a cluster of one character, placed far enough apart that
 * their rubble cannot merge. Positions are hand-placed rather than gridded so
 * the skyline reads as a city with a centre and outskirts.
 */
const DISTRICTS = [
  // Downtown: the tall stuff, widely spaced. A skyscraper coming down here is
  // the set piece, and it has room to fall without reaching the suburbs.
  { at: [0, 0], kind: 'skyscraper', count: 1 },
  { at: [64, 18], kind: 'tower', count: 1 },
  { at: [-58, 26], kind: 'tower', count: 1 },
  { at: [10, -70], kind: 'tower', count: 1 },

  // Midtown: mid-rises and wide blocks, close enough that a big collapse can
  // clip a neighbour. This is where cascades are supposed to happen.
  { at: [86, -46], kind: 'midrise', count: 2 },
  { at: [-78, -40], kind: 'block', count: 2 },
  { at: [96, 74], kind: 'midrise', count: 2 },

  // Outskirts: dense low-rise. These cannot reach each other, so a levelled
  // suburb is many small independent piles rather than one continuous field.
  { at: [-104, 96], kind: 'house', count: 6 },
  { at: [118, 130], kind: 'house', count: 6 },
  { at: [-140, -104], kind: 'shop', count: 4 },
  { at: [40, 150], kind: 'house', count: 5 },
];

function hullVolumeCentroid(flat) {
  const points = [];
  for (let i = 0; i < flat.length; i += 3) points.push([flat[i], flat[i + 1], flat[i + 2]]);
  const faces = convexFaces(points);
  const origin = points
    .reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0])
    .map((c) => c / points.length);
  let volume = 0;
  const centroid = [0, 0, 0];
  for (const [ia, ib, ic] of faces) {
    const a = points[ia];
    const b = points[ib];
    const c = points[ic];
    const tet = Math.abs(dot(sub(a, origin), cross(sub(b, origin), sub(c, origin)))) / 6;
    if (tet <= 0) continue;
    for (let k = 0; k < 3; k++) centroid[k] += tet * ((a[k] + b[k] + c[k] + origin[k]) / 4);
    volume += tet;
  }
  if (volume <= 0) return { centroid: origin, volume: 0 };
  return { centroid: centroid.map((c) => c / volume), volume };
}

function convexFaces(points) {
  const faces = [];
  const n = points.length;
  const EPS = 1e-7;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        const normal = cross(sub(points[j], points[i]), sub(points[k], points[i]));
        const length = Math.hypot(normal[0], normal[1], normal[2]);
        if (length < EPS) continue;
        const unit = normal.map((c) => c / length);
        let positive = false;
        let negative = false;
        for (let m = 0; m < n; m++) {
          if (m === i || m === j || m === k) continue;
          const d = dot(unit, sub(points[m], points[i]));
          if (d > EPS) positive = true;
          else if (d < -EPS) negative = true;
          if (positive && negative) break;
        }
        if (!(positive && negative)) faces.push([i, j, k]);
      }
    }
  }
  return faces;
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** Exact volume centroids, authored in rather than repaired later. */
function correctCentroids(building) {
  let maxPoints = 0;
  for (let i = 0; i < building.nodeColliders.length; i++) {
    const collider = building.nodeColliders[i];
    if (collider.kind !== 'convex_hull') continue;
    const points = collider.points;
    maxPoints = Math.max(maxPoints, points.length / 3);
    const { centroid, volume } = hullVolumeCentroid(points);
    for (let p = 0; p < points.length; p += 3) {
      points[p] = round(points[p] - centroid[0]);
      points[p + 1] = round(points[p + 1] - centroid[1]);
      points[p + 2] = round(points[p + 2] - centroid[2]);
    }
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

function main() {
  const rng = mulberry32(SEED * 2654435761);
  const nodes = [];
  const nodeTypes = [];
  const nodeSizes = [];
  const nodeColliders = [];
  const bonds = [];
  const placed = [];
  let maxPoints = 0;

  for (const district of DISTRICTS) {
    const archetype = ARCHETYPES[district.kind];
    const height = archetype.floors * FLOOR_HEIGHT;
    // Plot pitch: the building's own footprint plus what it can reach when it
    // topples. This is the number that keeps debris fields apart.
    const pitch = archetype.width + height * archetype.clearance;
    const perRow = Math.ceil(Math.sqrt(district.count));

    for (let index = 0; index < district.count; index++) {
      const row = Math.floor(index / perRow);
      const col = index % perRow;
      // Jitter so a district reads as organic rather than as a sub-grid, but
      // never enough to close the clearance gap.
      const jitter = () => (rng() - 0.5) * pitch * 0.12;
      const cx = district.at[0] + (col - (perRow - 1) / 2) * pitch + jitter();
      const cz = district.at[1] + (row - (perRow - 1) / 2) * pitch + jitter();

      const building = buildBuilding({
        width: archetype.width,
        floors: archetype.floors,
        rng,
      });
      maxPoints = Math.max(maxPoints, correctCentroids(building));

      const base = nodes.length;
      for (const node of building.nodes) {
        nodes.push({
          centroid: v(node.centroid.x + cx, node.centroid.y, node.centroid.z + cz),
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
          centroid: v(bond.centroid.x + cx, bond.centroid.y, bond.centroid.z + cz),
          normal: bond.normal,
          area: bond.area,
          ...(bond.m ? { m: bond.m } : {}),
        });
      }
      placed.push({ kind: archetype.label, cx, cz, height, pitch, nodes: building.nodes.length });
    }
  }

  if (maxPoints > MAX_HULL_POINTS) {
    throw new Error(`hull with ${maxPoints} points exceeds the ${MAX_HULL_POINTS}-point GPU cap`);
  }

  return {
    pack: {
      version: 2,
      key: 'fractured-district',
      title: 'Fractured city district (mixed archetypes, reach-based spacing)',
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

const extent = placed.reduce(
  (acc, p) => Math.max(acc, Math.abs(p.cx), Math.abs(p.cz)),
  0,
);
process.stderr.write(`wrote ${OUTPUT}\n`);
process.stderr.write(
  `buildings=${placed.length} nodes=${pack.scenario.nodes.length} ` +
    `bonds=${pack.scenario.bonds.length} maxHullPoints=${maxPoints} ` +
    `extent=±${Math.round(extent)} m\n`,
);
for (const [kind, count] of Object.entries(
  placed.reduce((acc, p) => ({ ...acc, [p.kind]: (acc[p.kind] ?? 0) + 1 }), {}),
)) {
  process.stderr.write(`  ${count} x ${kind}\n`);
}
