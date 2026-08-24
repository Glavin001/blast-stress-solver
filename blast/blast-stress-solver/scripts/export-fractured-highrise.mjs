// Export ONE fractured high-rise as a ScenePack v2 building asset.
//
// This is the single-building sibling of export-fractured-city.mjs. That
// script bakes a whole grid of buildings into one flat scenario, which suits a
// standalone demo but not a consumer that tiles a pack itself -- vibe-land
// stamps one pack across its city grid, so it needs a pack that IS one
// building, centred on the origin.
//
// Everything structural is reused from the city exporter rather than
// reimplemented: the same Voronoi panel fracture, the same closed-form contact
// areas, the same material table. Only the packaging differs.
//
// INVARIANT INHERITED FROM buildBuilding, AND LOAD-BEARING HERE: no shard
// spans a storey boundary. Wall panels are fractured within each storey's
// clear height, columns are segmented per storey, slabs occupy the band at the
// top. vibe-land builds shorter variants of a pack by slicing it at a Y
// cutoff, so a shard crossing a floor line would be cut in half -- geometry
// kept, but with its mass and bonds describing a piece that no longer exists.
//
// Usage:
//   FLOORS=10 SHARDS_PER_PANEL=10 node scripts/export-fractured-highrise.mjs
//
// Determinism: seeded PRNG only, no wall-clock or hash iteration, so repeated
// runs are byte-identical. Verify with two runs and cmp before committing an
// asset -- client and server both derive geometry from the manifest hash, so a
// nondeterministic exporter would be a desync waiting to happen.

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { MATERIALS, buildBuilding, mulberry32, round, v } from './export-fractured-city.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const FLOORS = Number(process.env.FLOORS ?? 10);
const WIDTH = Number(process.env.WIDTH ?? 12);
const SEED = Number(process.env.SEED ?? 7);
const OUTPUT = path.resolve(
  __dirname,
  process.env.OUTPUT ?? `../assets/mini-city/fractured-highrise-${FLOORS}f.json`,
);

/**
 * PhysX cooks GPU-compatible convex meshes from at most 64 vertices. Assets
 * are expected to author within that; exceeding it means the cooker has to
 * approximate, which silently moves colliders away from the rendered shape.
 */
const MAX_HULL_POINTS = 64;

function main() {
  const rng = mulberry32(SEED * 2654435761);
  const building = buildBuilding({ width: WIDTH, floors: FLOORS, rng });

  // Author exact volume centroids. A hull's centroid is what every consumer
  // reconstructs its world pose against and what the body's centre of mass is
  // declared on, so an approximate one makes chunks tumble about a point that
  // is not their mass centre -- and shifts them visibly the moment they become
  // separate bodies.
  let corrected = 0;
  let maxPoints = 0;
  for (let i = 0; i < building.nodeColliders.length; i++) {
    const collider = building.nodeColliders[i];
    if (collider.kind !== 'convex_hull') continue;
    const points = collider.points;
    maxPoints = Math.max(maxPoints, points.length / 3);
    const { centroid, volume } = hullVolumeCentroid(points);
    if (Math.hypot(centroid[0], centroid[1], centroid[2]) > 1e-6) corrected++;
    // Shift the points by the same delta so every world position is
    // unchanged: this restates the shape, it does not move it.
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

  if (maxPoints > MAX_HULL_POINTS) {
    throw new Error(
      `hull with ${maxPoints} points exceeds the ${MAX_HULL_POINTS}-point GPU cap; ` +
        'lower SHARDS_PER_PANEL or simplify the panel geometry',
    );
  }

  const pack = {
    version: 2,
    key: `fractured-highrise-${FLOORS}f`,
    title: `Fractured high-rise (${FLOORS} floors, exact-contact bonds)`,
    defaults: {
      solver: { gravity: -9.81, materialScale: 1, materials: MATERIALS },
    },
    scenario: {
      nodeTypes: building.nodeTypes,
      nodes: building.nodes,
      bonds: building.bonds,
      nodeSizes: building.nodeSizes,
      nodeColliders: building.nodeColliders,
    },
  };

  return { pack, corrected, maxPoints };
}

/**
 * Exact centroid and volume of a convex point cloud, by tetrahedron
 * decomposition of its hull about an interior point. Closed form -- there is
 * no reason for a building asset to carry an approximate one.
 */
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
    const ab = sub(a, origin);
    const bb = sub(b, origin);
    const cb = sub(c, origin);
    const tet = Math.abs(dot(ab, cross(bb, cb))) / 6;
    if (tet <= 0) continue;
    for (let k = 0; k < 3; k++) {
      centroid[k] += tet * ((a[k] + b[k] + c[k] + origin[k]) / 4);
    }
    volume += tet;
  }
  if (volume <= 0) return { centroid: origin, volume: 0 };
  return { centroid: centroid.map((c) => c / volume), volume };
}

/** Hull faces as triangles, by brute-force supporting-plane test. */
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
        // All other points on one side: this triangle lies on the hull.
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

const { pack, corrected, maxPoints } = main();
await mkdir(path.dirname(OUTPUT), { recursive: true });
await writeFile(OUTPUT, JSON.stringify(pack));
const hulls = pack.scenario.nodeColliders.filter((c) => c.kind === 'convex_hull').length;
process.stderr.write(`wrote ${OUTPUT}\n`);
process.stderr.write(
  `floors=${FLOORS} width=${WIDTH} nodes=${pack.scenario.nodes.length} ` +
    `bonds=${pack.scenario.bonds.length} hulls=${hulls} maxHullPoints=${maxPoints} ` +
    `centroidsCorrected=${corrected}\n`,
);
