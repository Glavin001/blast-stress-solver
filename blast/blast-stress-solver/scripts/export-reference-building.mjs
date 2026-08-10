/**
 * The worked reference building — a copyable starting point for authoring.
 *
 *   node scripts/export-reference-building.mjs
 *   -> assets/reference/reference-building.json  (ScenePack v2)
 *
 * Small on purpose (64 nodes, 132 bonds, 3 floors of a 2x2 bay frame): big
 * enough to have a real load path — footings, columns, slab diaphragms and
 * non-structural cladding — and small enough to read end to end and to
 * re-measure in a fraction of a second while you iterate.
 *
 * Everything here follows the rule the format exists to enforce
 * (SCENE_PACK_FORMAT.md): AREA IS GEOMETRY, MATERIAL IS STRENGTH. Every bond
 * area below is computed from the section it represents — a column's area is
 * its cross-section, a panel clip's is the panel edge against the column — and
 * strength differences come only from the material table.
 *
 * The material values are NOT a blessed library. They are one calibrated
 * example, and they were arrived at by measuring, not guessing:
 *
 *   1. author geometry + a first guess at materials
 *   2. run the demo with a negligible projectile to get gravity-only loads:
 *        blast_stress_demo --physics cpu --grid 1 --settle 2 \
 *          --scene assets/reference/reference-building.json \
 *          --projectile-mass-scale 0.0001 --projectile-speed-scale 0.0001
 *   3. read the "gravity load path" table it prints
 *   4. move the material whose safety factor is outside its target band, in
 *      the MODE that is actually binding (peak c/t/s tells you which)
 *
 * The first pass here put the facade clip at safety factor 1.06 — the building
 * technically stood, but its cladding was one gust from letting go. The
 * binding mode was shear (5.67e4 Pa against a 0.06e6 limit), so the clip's
 * shear limits moved and nothing else did. That is the whole loop.
 *
 * Target bands used (see .cursor/skills/blast-structure-authoring/SKILL.md):
 *   base anchor  30-70    never the failure point
 *   frame        5-20     real structural design margin
 *   facade        2-4     the deliberate weak link
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT = path.resolve(__dirname, '../assets/reference/reference-building.json');

// ── Geometry ────────────────────────────────────────────────────────────────
const BAY = 4.0; // m between column lines
const FLOOR = 3.0; // m floor to floor
const FLOORS = 3;
const COL = 0.4; // column section, square
const SLAB_T = 0.25;
const PANEL_T = 0.12;
const CONCRETE = 2400; // kg/m^3
const DRYWALL = 700;

// ── Materials (Pa) ──────────────────────────────────────────────────────────
// Index 0 is the structure default: a bond with no `m` gets it.
// Ductility is the fatal-elastic band: the frame yields over many frames, the
// clip lets go quickly, which is what makes cladding shed cleanly.
// Ductility of the structural frame: fatal = elastic * FRAME_BAND. Elastic
// (strength) is untouched, so gravity safety factors are IDENTICAL for any
// value here — this dial changes only HOW the frame fails, never whether it
// holds itself up. Override to compare: FRAME_BAND=1.05 node scripts/...
//
// Measured on this building (standing fraction after a single impact):
//   band 1.05 (brittle): 1.00 -> 0.06 with NO partial state at all
//   band 2.5:            one partial step, ~1000-1400 kg wide
//   band 10  (ductile):  partial state holds 1000-5000 kg, still fully
//                        destructible at 20 t
// See demos/blast-stress-demo/tests/material_behavior_test.cpp
// (testBandWidthControlsBrittleVsDuctile) for the isolated mechanism.
const FRAME_BAND = Number(process.env.FRAME_BAND ?? 10);

const MATERIALS = [
  {
    name: 'reinforced-concrete',
    compressionElastic: 24e6, compressionFatal: 24e6 * FRAME_BAND,
    tensionElastic: 3.0e6, tensionFatal: 3.0e6 * FRAME_BAND,
    shearElastic: 4.0e6, shearFatal: 4.0e6 * FRAME_BAND,
  },
  {
    name: 'concrete-slab',
    compressionElastic: 12e6, compressionFatal: 30e6,
    tensionElastic: 1.2e6, tensionFatal: 3.0e6,
    shearElastic: 1.6e6, shearFatal: 4.0e6,
  },
  {
    // Shear raised 3x when the panels were split in two: halving a panel
    // halves its bonded edge, which doubles the stress the same weight puts
    // through it. Geometry changed, so the material had to follow — that is
    // the calibration loop, not tuning.
    name: 'drywall-panel',
    compressionElastic: 0.8e6, compressionFatal: 2.0e6,
    tensionElastic: 0.12e6, tensionFatal: 0.40e6,
    shearElastic: 0.45e6, shearFatal: 1.2e6,
  },
  {
    // Same story, and the second time this material's SHEAR was the binding
    // mode (the first pass had it at safety factor 1.06). Read peak(c/t/s) in
    // the load-path report and move only the mode that is actually loaded.
    name: 'facade-clip',
    compressionElastic: 0.5e6, compressionFatal: 0.9e6,
    tensionElastic: 0.09e6, tensionFatal: 0.22e6,
    shearElastic: 0.60e6, shearFatal: 1.3e6,
  },
];
const [M_FRAME, M_SLAB, M_PANEL, M_CLIP] = [0, 1, 2, 3];

const round = (n) => Math.round(n * 1e5) / 1e5;
const v = (x, y, z) => ({ x: round(x), y: round(y), z: round(z) });

const nodes = [];
const nodeTypes = [];
const nodeSizes = [];
const nodeColliders = [];

function addNode(role, centre, half, density, fixed = false) {
  const volume = 8 * half[0] * half[1] * half[2];
  nodes.push({
    centroid: v(...centre),
    // mass 0 marks a support: pinned to the world. ONLY footings get it —
    // pinning columns too was how an earlier revision made a tower that
    // projectiles bounced off (it had no load path, just scenery).
    mass: fixed ? 0 : round(volume * density),
    volume: round(volume),
  });
  nodeTypes.push(role);
  nodeSizes.push(v(half[0] * 2, half[1] * 2, half[2] * 2));
  nodeColliders.push({ kind: 'cuboid', halfExtents: v(...half) });
  return nodes.length - 1;
}

const bonds = [];
function addBond(a, b, area, material) {
  const ca = nodes[a].centroid;
  const cb = nodes[b].centroid;
  const d = [cb.x - ca.x, cb.y - ca.y, cb.z - ca.z];
  const len = Math.hypot(...d) || 1;
  const bond = {
    node0: a,
    node1: b,
    centroid: v((ca.x + cb.x) / 2, (ca.y + cb.y) / 2, (ca.z + cb.z) / 2),
    normal: v(d[0] / len, d[1] / len, d[2] / len),
    area: round(area),
  };
  if (material) bond.m = material; // omitted means 0
  bonds.push(bond);
}

const xs = [-BAY / 2, BAY / 2];
const zs = [-BAY / 2, BAY / 2];

// ── Fracture granularity ────────────────────────────────────────────────────
// Each structural ELEMENT is built from several chunks, not one. This is what
// makes breaks look like pieces instead of dust: an inter-element joint
// failing releases a multi-chunk piece (half a column, a slab quadrant) that
// holds together because its INTRA-element bonds are full-section monolithic
// concrete.
//
// This is a measured lesson, not a style choice. The first version of this
// file used one node per element; a moderate hit atomized the building into
// 30 bodies out of 31 chunks — largest surviving piece: 2 chunks. Granularity
// is what buys recognisable debris. See
// demos/blast-stress-demo/tests/destruction_quality_test.cpp.
const COL_SEGMENTS = 2; // per floor
const SLAB_SPLIT = 2; // per axis -> 2x2 quadrants
const PANEL_SPLIT = 2; // per wall face

const foundation = new Map();
const columns = new Map(); // key(x,z,floor,segment)
const slabs = new Map(); // key(floor,i,j)
const panels = []; // [node, floor, axis, coord, side]
const key = (...parts) => parts.join(':');

for (const x of xs)
  for (const z of zs)
    foundation.set(key(x, z), addNode('foundation', [x, 0.3, z], [0.6, 0.3, 0.6], CONCRETE, true));

const SEG_H = FLOOR / COL_SEGMENTS;
const SLAB_HALF = (BAY / 2 + COL / 2) / SLAB_SPLIT;

for (let f = 0; f < FLOORS; ++f) {
  const y0 = 0.6 + f * FLOOR;
  for (const x of xs)
    for (const z of zs)
      for (let s = 0; s < COL_SEGMENTS; ++s)
        columns.set(key(x, z, f, s), addNode('column',
          [x, y0 + SEG_H * (s + 0.5), z], [COL / 2, SEG_H / 2, COL / 2], CONCRETE));

  const slabY = y0 + FLOOR + SLAB_T / 2;
  for (let i = 0; i < SLAB_SPLIT; ++i)
    for (let j = 0; j < SLAB_SPLIT; ++j)
      slabs.set(key(f, i, j), addNode('slab',
        [(i * 2 - 1) * SLAB_HALF, slabY, (j * 2 - 1) * SLAB_HALF],
        [SLAB_HALF, SLAB_T / 2, SLAB_HALF], CONCRETE));
}

for (let f = 0; f < FLOORS; ++f) {
  const y = 0.6 + f * FLOOR + FLOOR / 2;
  const halfW = (BAY / 2 - COL / 2) / PANEL_SPLIT;
  for (const z of zs)
    for (let s = 0; s < PANEL_SPLIT; ++s)
      panels.push([addNode('infill', [(s * 2 - 1) * halfW, y, z],
        [halfW, FLOOR / 2 - 0.1, PANEL_T / 2], DRYWALL), f, 'z', z, s * 2 - 1]);
  for (const x of xs)
    for (let s = 0; s < PANEL_SPLIT; ++s)
      panels.push([addNode('infill', [x, y, (s * 2 - 1) * halfW],
        [PANEL_T / 2, FLOOR / 2 - 0.1, halfW], DRYWALL), f, 'x', x, s * 2 - 1]);
}

// ── Bond areas: each one is a real contact patch, not a strength knob ───────
const COL_AREA = COL * COL; // column cross-section
const SLAB_CUT = 2 * SLAB_HALF * SLAB_T; // quadrant-to-quadrant cut face
const PANEL_EDGE = 2 * ((BAY / 2 - COL / 2) / PANEL_SPLIT) * PANEL_T; // panel head into slab
const PANEL_SEAM = (FLOOR - 0.2) * PANEL_T; // vertical seam between panel halves
const CLIP_AREA = (FLOOR - 0.2) * PANEL_T; // panel edge against a column

for (const x of xs)
  for (const z of zs)
    addBond(foundation.get(key(x, z)), columns.get(key(x, z, 0, 0)), COL_AREA, M_FRAME);

for (let f = 0; f < FLOORS; ++f) {
  for (const x of xs)
    for (const z of zs) {
      // INTRA-element: monolithic column, full section. Keeps a broken-off
      // column piece in one lump instead of a pile of cubes.
      for (let s = 0; s + 1 < COL_SEGMENTS; ++s)
        addBond(columns.get(key(x, z, f, s)), columns.get(key(x, z, f, s + 1)), COL_AREA, M_FRAME);
      const top = columns.get(key(x, z, f, COL_SEGMENTS - 1));
      const qi = x < 0 ? 0 : 1;
      const qj = z < 0 ? 0 : 1;
      addBond(top, slabs.get(key(f, qi, qj)), COL_AREA, M_FRAME);
      if (f + 1 < FLOORS)
        addBond(slabs.get(key(f, qi, qj)), columns.get(key(x, z, f + 1, 0)), COL_AREA, M_FRAME);
    }
  // INTRA-element: slab diaphragm continuity between quadrants. This is the
  // structure's only load-path redundancy: a quadrant whose column is gone is
  // still held by its neighbours, which is what produces a partial collapse
  // instead of an all-or-nothing one.
  for (let i = 0; i < SLAB_SPLIT; ++i)
    for (let j = 0; j < SLAB_SPLIT; ++j) {
      if (i + 1 < SLAB_SPLIT)
        addBond(slabs.get(key(f, i, j)), slabs.get(key(f, i + 1, j)), SLAB_CUT, M_SLAB);
      if (j + 1 < SLAB_SPLIT)
        addBond(slabs.get(key(f, i, j)), slabs.get(key(f, i, j + 1)), SLAB_CUT, M_SLAB);
    }
}

for (const [panel, f, axis, coord, side] of panels) {
  const qi = axis === 'z' ? (side < 0 ? 0 : 1) : (coord < 0 ? 0 : 1);
  const qj = axis === 'z' ? (coord < 0 ? 0 : 1) : (side < 0 ? 0 : 1);
  addBond(slabs.get(key(f, qi, qj)), panel, PANEL_EDGE, M_PANEL);
  const cx = axis === 'z' ? (side < 0 ? xs[0] : xs[1]) : coord;
  const cz = axis === 'z' ? coord : (side < 0 ? zs[0] : zs[1]);
  addBond(columns.get(key(cx, cz, f, 0)), panel, CLIP_AREA / 2, M_CLIP);
  addBond(columns.get(key(cx, cz, f, COL_SEGMENTS - 1)), panel, CLIP_AREA / 2, M_CLIP);
}
for (let f = 0; f < FLOORS; ++f) {
  for (const axis of ['z', 'x'])
    for (const coord of axis === 'z' ? zs : xs) {
      const pair = panels.filter(([, pf, pa, pc]) => pf === f && pa === axis && pc === coord);
      if (pair.length === 2) addBond(pair[0][0], pair[1][0], PANEL_SEAM, M_PANEL);
    }
}

const pack = {
  version: 2,
  key: 'reference-building',
  title: 'Reference building (2x2 bay, 3 floors)',
  defaults: {
    camera: { target: v(0, 5, 0), distance: 26 },
    projectile: { radius: 0.5, mass: 2000, speed: 18, ttlMs: 8000 },
    solver: { gravity: -9.81, materialScale: 1, materials: MATERIALS },
    physics: {
      debrisCollisionMode: 'all', friction: 0.25, restitution: 0,
      // 1 is the physically correct transfer; see SCENE_PACK_FORMAT.md.
      contactForceScale: 1, skipSingleBodies: false,
    },
    optimization: {
      smallBodyDampingMode: 'always', debrisCleanupMode: 'always',
      debrisTtlMs: 10000, maxCollidersForDebris: 3,
    },
  },
  scenario: { nodeTypes, nodes, bonds, nodeSizes, nodeColliders },
  nodeMeshes: [],
};

await mkdir(path.dirname(OUTPUT), { recursive: true });
await writeFile(OUTPUT, `${JSON.stringify(pack, null, 1)}\n`, 'utf8');

const mass = nodes.reduce((sum, n) => sum + n.mass, 0);
console.log(`wrote ${OUTPUT}`);
console.log(`nodes=${nodes.length} bonds=${bonds.length} materials=${MATERIALS.length} mass=${Math.round(mass)}kg`);
