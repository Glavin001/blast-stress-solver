/**
 * The worked reference building — a copyable starting point for authoring.
 *
 *   node scripts/export-reference-building.mjs
 *   -> assets/reference/reference-building.json  (ScenePack v2)
 *
 * Small on purpose (31 nodes, 62 bonds, 3 floors of a 2x2 bay frame): big
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
const MATERIALS = [
  {
    name: 'reinforced-concrete',
    compressionElastic: 24e6, compressionFatal: 60e6,
    tensionElastic: 3.0e6, tensionFatal: 8.0e6,
    shearElastic: 4.0e6, shearFatal: 10.0e6,
  },
  {
    name: 'concrete-slab',
    compressionElastic: 12e6, compressionFatal: 30e6,
    tensionElastic: 1.2e6, tensionFatal: 3.0e6,
    shearElastic: 1.6e6, shearFatal: 4.0e6,
  },
  {
    name: 'drywall-panel',
    compressionElastic: 0.8e6, compressionFatal: 2.0e6,
    tensionElastic: 0.12e6, tensionFatal: 0.40e6,
    shearElastic: 0.15e6, shearFatal: 0.5e6,
  },
  {
    // Raised from the first pass: shear was binding at safety factor 1.06.
    name: 'facade-clip',
    compressionElastic: 0.5e6, compressionFatal: 0.9e6,
    tensionElastic: 0.09e6, tensionFatal: 0.22e6,
    shearElastic: 0.20e6, shearFatal: 0.45e6,
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
const foundation = new Map();
const columns = new Map();
const slabs = new Map();
const panels = [];
const key = (...parts) => parts.join(':');

for (const x of xs)
  for (const z of zs)
    foundation.set(key(x, z), addNode('foundation', [x, 0.3, z], [0.6, 0.3, 0.6], CONCRETE, true));

for (let f = 0; f < FLOORS; ++f) {
  const y0 = 0.6 + f * FLOOR;
  for (const x of xs)
    for (const z of zs)
      columns.set(key(x, z, f),
        addNode('column', [x, y0 + FLOOR / 2, z], [COL / 2, FLOOR / 2, COL / 2], CONCRETE));
  slabs.set(f, addNode('slab', [0, y0 + FLOOR + SLAB_T / 2, 0],
    [BAY / 2 + COL / 2, SLAB_T / 2, BAY / 2 + COL / 2], CONCRETE));
}

for (let f = 0; f < FLOORS; ++f) {
  const y = 0.6 + f * FLOOR + FLOOR / 2;
  for (const z of zs)
    panels.push([addNode('infill', [0, y, z],
      [BAY / 2 - COL / 2, FLOOR / 2 - 0.1, PANEL_T / 2], DRYWALL), f, 'z', z]);
  for (const x of xs)
    panels.push([addNode('infill', [x, y, 0],
      [PANEL_T / 2, FLOOR / 2 - 0.1, BAY / 2 - COL / 2], DRYWALL), f, 'x', x]);
}

// ── Bond areas: each one is a real contact patch, not a strength knob ───────
const COL_AREA = COL * COL; // column cross-section
const SLAB_EDGE = (BAY + COL) * SLAB_T; // slab edge bearing on a column line
const PANEL_EDGE = (BAY - COL) * PANEL_T; // panel head/base into the slab band
const CLIP_AREA = (FLOOR - 0.2) * PANEL_T; // panel vertical edge against a column

for (const x of xs)
  for (const z of zs)
    addBond(foundation.get(key(x, z)), columns.get(key(x, z, 0)), COL_AREA, M_FRAME);

for (let f = 0; f < FLOORS; ++f) {
  for (const x of xs)
    for (const z of zs) {
      addBond(columns.get(key(x, z, f)), slabs.get(f), COL_AREA, M_FRAME);
      if (f + 1 < FLOORS)
        addBond(slabs.get(f), columns.get(key(x, z, f + 1)), COL_AREA, M_FRAME);
    }
}
for (let f = 0; f + 1 < FLOORS; ++f)
  addBond(slabs.get(f), slabs.get(f + 1), SLAB_EDGE * 0.25, M_SLAB);

for (const [panel, f, axis, coord] of panels) {
  addBond(slabs.get(f), panel, PANEL_EDGE, M_PANEL);
  for (const x of xs)
    for (const z of zs)
      if ((axis === 'z' && z === coord) || (axis === 'x' && x === coord))
        addBond(columns.get(key(x, z, f)), panel, CLIP_AREA, M_CLIP);
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
