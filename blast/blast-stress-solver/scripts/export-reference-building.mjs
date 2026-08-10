/**
 * The worked reference building — a copyable starting point for authoring.
 *
 *   node scripts/export-reference-building.mjs
 *   -> assets/reference/reference-building.json  (ScenePack v2)
 *
 * Small on purpose (76 nodes, 180 bonds, 3 floors of a 2x2 bay frame): big
 * enough to have a real load path — footings, columns, a ring beam at every
 * floor, slab diaphragms and non-structural cladding — and small enough to
 * read end to end and to re-measure in a fraction of a second while you
 * iterate.
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
 *
 * ── The beam ring, and what it actually bought ──────────────────────────────
 *
 * This building used to be a TREE: each slab quadrant hung off exactly one
 * column, and its only redundancy was the quadrant-to-quadrant diaphragm. The
 * ring beams were added on the hypothesis that tying the column heads together
 * would give a GENTLER damage curve — lose a column, the ring spans, the bay
 * does not simply drop.
 *
 * The ring does exactly what it was supposed to do structurally, and the
 * hypothesis was still wrong. Measured with destruction_quality_test --sweep,
 * mass ramped at a fixed 16 m/s (`largest` is the biggest connected piece):
 *
 *        mass    beamless: standing/largest      ring: standing/largest
 *         400          1.00 / 64  intact            1.00 / 76  intact
 *         600          0.50 / 32  frame hit         0.66 / 50  FACADE ONLY
 *        1000          0.50 / 32  frame hit         0.66 / 50  facade only
 *        1400          0.39 / 25  frame hit         0.66 / 50  facade only
 *        1800          0.41 / 25  frame hit         0.64 / 48  frame starts
 *        2400          0.33 / 18  frame hit         0.64 / 47  frame
 *        2800          0.06 /  6  down              0.64 / 47  frame
 *        3200          0.06 / 12  down              0.05 / 45  down, one lump
 *        4000          0.06 / 10  down              0.05 /  3  down
 *
 * By the letter of the metric the ring wins: "standing between 0.25 and 0.85"
 * holds over 600-3100 kg (5.2x) instead of 600-2400 kg (4.0x), and the
 * surviving piece is 47-50 chunks instead of 12-32.
 *
 * By the SPIRIT of it the ring loses. 0.66 standing is almost exactly "all 52
 * frame chunks up, all 24 facade panels shed" — and `moved(column)` is 0
 * everywhere below 1.8 t. The wide band is not partial destruction, it is an
 * intact frame in its underwear. The frame itself went from a graded response
 * (0.50 -> 0.39 -> 0.33 over 600-2400 kg) to a step function: nothing, then
 * everything, over a 300 kg interval near 3.2 t. Worse, the 3200 kg case
 * detaches 45 of 76 chunks as ONE piece — the whole superstructure shearing
 * off its footings intact, which is not a look anyone wants.
 *
 * Two attempts to recover the gradient, both measured, both failed:
 *   - Weakening the beam~column joint to the bottom of the frame band (safety
 *     factor 5.01 instead of 6.26) moved the cliff from 3200 to 2800 kg and
 *     changed nothing about its shape. The plateau is structural, not a
 *     calibration artifact.
 *   - Splitting each beam in two so it can fail at mid-span (BEAM_SPLIT = 2,
 *     88 nodes) destroyed monotonicity instead of restoring the gradient:
 *     standing went 0.67, 0.05, 0.36, 0.05, 0.67, 0.05 over 400-1800 kg. A
 *     mid-span hinge releases half a ring, which is too big an event to grade.
 *
 * Cost of the ring: +12 nodes (64->76), +48 bonds (132->180), +13.4 t
 * (58.8->72.2, +23%), ~12% GPU frame time at grid 2, and a calibration that is
 * meaningfully more coupled — ten joint classes instead of seven, and two of
 * them (column~slab, beam~slab) have to sit on the SLAB material rather than
 * the frame material or they read 24-35, over the frame band.
 *
 * Kept here because the geometry and the calibration are correct and the
 * measurement is the point. If you want a graded frame response, the beamless
 * tree gave one and this does not; if you want a frame that survives a hit and
 * sheds cladding convincingly, this is better. Set BEAM_SPLIT = 2 to reproduce
 * the chaotic variant.
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
const BEAM_H = 0.45; // ring-beam depth  (~span/9 at a 4 m bay)
const BEAM_W = 0.3; // ring-beam width
const CONCRETE = 2400; // kg/m^3
const DRYWALL = 700;

// Storey stack, so the beam ring occupies the top BEAM_H of every storey and
// the overall building height is UNCHANGED from the beamless version. That
// matters: the point of this asset is a controlled comparison, so the columns
// and panels give up the height the beams take rather than the building
// growing taller and loading its own base differently.
const CLEAR = FLOOR - BEAM_H; // column height, and clear height under the beam

// ── Materials (Pa) ──────────────────────────────────────────────────────────
// Index 0 is the structure default: a bond with no `m` gets it.
// Ductility is the fatal-elastic band: the frame yields over many frames, the
// clip lets go quickly, which is what makes cladding shed cleanly.
// Recalibrated for the beam ring. The ring adds 13.4 t of beams that every
// column head now carries, and it moved the slab off the columns entirely, so
// every class had to be re-measured. Each move below is ONE mode of ONE
// material — the mode `peak(c/t/s)` named as binding — and the ductility
// ratios (fatal/elastic) are unchanged, so only thresholds moved, not the
// brittle/ductile character of anything.
const MATERIALS = [
  {
    // compression 24 -> 30 MPa: the anchor (column~foundation) is the ONLY
    // class where this material's compression binds, and the ring's extra
    // weight had pushed it to safety factor 26, under its 30-70 band.
    // shear 4 -> 5 MPa: shear binds every frame class, and the beam~column
    // joint had landed at 5.01 — inside the 5-20 band but only just, and only
    // 1.7x above the facade, which is how cladding starts taking the frame
    // with it. This restores the gap to ~2x.
    name: 'reinforced-concrete',
    compressionElastic: 30e6, compressionFatal: 75e6,
    tensionElastic: 3.0e6, tensionFatal: 8.0e6,
    shearElastic: 5.0e6, shearFatal: 12.5e6,
  },
  {
    // Unchanged. It now also carries the two slab-bearing classes (see below).
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
    //
    // The ring shortened the panels (they now stop under the beam), which
    // moved this material twice more, in two DIFFERENT modes:
    //   tension 0.12 -> 0.26 MPa — the panel-to-panel seam is pure tension
    //     (peak c/t/s is 0/7.9e4/3) and had fallen to safety factor 1.52.
    //   shear 0.45 -> 0.55 MPa — the panel head into the beam is pure shear,
    //     and a shorter panel on the same head area sat at 2.62.
    name: 'drywall-panel',
    compressionElastic: 0.8e6, compressionFatal: 2.0e6,
    tensionElastic: 0.26e6, tensionFatal: 0.85e6,
    shearElastic: 0.55e6, shearFatal: 1.47e6,
  },
  {
    // Same story, and the second time this material's SHEAR was the binding
    // mode (the first pass had it at safety factor 1.06). Read peak(c/t/s) in
    // the load-path report and move only the mode that is actually loaded.
    // Shear 0.60 -> 0.80 MPa here puts the clip back at 3.03, the exact figure
    // the beamless building was calibrated to, so the facade weak link is held
    // FIXED across the experiment and any change in the damage curve is the
    // ring's doing rather than a differently-tuned facade.
    name: 'facade-clip',
    compressionElastic: 0.5e6, compressionFatal: 0.9e6,
    tensionElastic: 0.09e6, tensionFatal: 0.22e6,
    shearElastic: 0.80e6, shearFatal: 1.73e6,
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
const sgn = (value) => (value < 0 ? -1 : 1);

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
// Each ring beam is TWO chunks meeting at mid-span. A single-node beam cannot
// break in the middle, and measuring that version showed why that matters: the
// ring became so rigid that the frame was either untouched or entirely gone,
// with a 5x-wide band of impact energies in between where nothing happened to
// it but the cladding fell off. Mid-span is where a beam bridging a lost
// column is actually overstressed, so it is where the break has to be able to
// happen.
const BEAM_SPLIT = 1;

const foundation = new Map();
const columns = new Map(); // key(x,z,floor,segment)
const beams = new Map(); // key(floor,axis,coord) - axis is the axis it SPANS
const slabs = new Map(); // key(floor,i,j)
const panels = []; // [node, floor, axis, coord, side]
const key = (...parts) => parts.join(':');

for (const x of xs)
  for (const z of zs)
    foundation.set(key(x, z), addNode('foundation', [x, 0.3, z], [0.6, 0.3, 0.6], CONCRETE, true));

const SEG_H = CLEAR / COL_SEGMENTS;
const SLAB_HALF = (BAY / 2 + COL / 2) / SLAB_SPLIT;
const BEAM_HALF_SPAN = BAY / 2 + COL / 2; // reaches the outer face of both columns
const BEAM_SEG = BEAM_HALF_SPAN / BEAM_SPLIT; // half-length of one beam chunk
const PANEL_HALF_H = CLEAR / 2 - 0.1;
// Which beam chunk covers the half of the span with this sign. With
// BEAM_SPLIT = 1 there is only one chunk and everything bonds to it.
const bseg = (side) => (BEAM_SPLIT === 1 ? 0 : side < 0 ? 0 : 1);

for (let f = 0; f < FLOORS; ++f) {
  const y0 = 0.6 + f * FLOOR;
  for (const x of xs)
    for (const z of zs)
      for (let s = 0; s < COL_SEGMENTS; ++s)
        columns.set(key(x, z, f, s), addNode('column',
          [x, y0 + SEG_H * (s + 0.5), z], [COL / 2, SEG_H / 2, COL / 2], CONCRETE));

  // ── The beam ring ────────────────────────────────────────────────────────
  // Four beams per floor tying the four column heads into a closed ring, one
  // node each. ONE node per beam is deliberate: a beam that cannot break in
  // the middle is exactly the member that keeps spanning after the column
  // under it is gone, which is the whole reason the ring is here.
  //
  // It sits in the top BEAM_H of the storey, between the column heads and the
  // slab, and NOTHING overlaps it: column heads stop at its soffit, the slab
  // starts at its top, and the panels below were shortened to clear it. An
  // earlier attempt left the beams intersecting the facade panels, which is
  // how the panels ended up carrying frame load.
  const beamY = y0 + CLEAR + BEAM_H / 2;
  for (let k = 0; k < BEAM_SPLIT; ++k) {
    const at = -BEAM_HALF_SPAN + BEAM_SEG * (2 * k + 1); // chunk centre along the span
    for (const z of zs)
      beams.set(key(f, 'x', z, k), addNode('beam', [at, beamY, z],
        [BEAM_SEG, BEAM_H / 2, BEAM_W / 2], CONCRETE));
    for (const x of xs)
      beams.set(key(f, 'z', x, k), addNode('beam', [x, beamY, at],
        [BEAM_W / 2, BEAM_H / 2, BEAM_SEG], CONCRETE));
  }

  const slabY = y0 + FLOOR + SLAB_T / 2;
  for (let i = 0; i < SLAB_SPLIT; ++i)
    for (let j = 0; j < SLAB_SPLIT; ++j)
      slabs.set(key(f, i, j), addNode('slab',
        [(i * 2 - 1) * SLAB_HALF, slabY, (j * 2 - 1) * SLAB_HALF],
        [SLAB_HALF, SLAB_T / 2, SLAB_HALF], CONCRETE));
}

for (let f = 0; f < FLOORS; ++f) {
  const y = 0.6 + f * FLOOR + CLEAR / 2;
  const halfW = (BAY / 2 - COL / 2) / PANEL_SPLIT;
  for (const z of zs)
    for (let s = 0; s < PANEL_SPLIT; ++s)
      panels.push([addNode('infill', [(s * 2 - 1) * halfW, y, z],
        [halfW, PANEL_HALF_H, PANEL_T / 2], DRYWALL), f, 'z', z, s * 2 - 1]);
  for (const x of xs)
    for (let s = 0; s < PANEL_SPLIT; ++s)
      panels.push([addNode('infill', [x, y, (s * 2 - 1) * halfW],
        [PANEL_T / 2, PANEL_HALF_H, halfW], DRYWALL), f, 'x', x, s * 2 - 1]);
}

// ── Bond areas: each one is a real contact patch, not a strength knob ───────
const COL_AREA = COL * COL; // column cross-section
const SLAB_CUT = 2 * SLAB_HALF * SLAB_T; // quadrant-to-quadrant cut face
const PANEL_EDGE = 2 * ((BAY / 2 - COL / 2) / PANEL_SPLIT) * PANEL_T; // panel head into the beam
const PANEL_SEAM = 2 * PANEL_HALF_H * PANEL_T; // vertical seam between panel halves
const CLIP_AREA = 2 * PANEL_HALF_H * PANEL_T; // panel edge against a column
const BEAM_SECTION = BEAM_H * BEAM_W; // beam cross-section, for the ring corners
// The column head is ONE 0.4x0.4 patch that TWO beams cross. Giving each bond
// the full COL*BEAM_W strip would bill the head at 1.5x its real section, so
// the head is split between the two beams and the two bonds sum to exactly
// COL_AREA. Area is geometry; if the joint needs to be stronger, the material
// moves, not this.
const COL_HEAD_SHARE = COL_AREA / 2;
// Slab soffit bearing on a beam: a strip one beam wide along the quadrant edge.
const SLAB_BEARING = 2 * SLAB_HALF * BEAM_W;

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
      // The column head carries the two beams crossing it, and NOTHING else.
      // The slab does not touch a column any more: its whole weight goes
      // slab -> beam -> column head. That indirection is the point of the
      // ring — with a direct column~slab bond the beams would sit beside the
      // load path instead of in it, and losing a column would still drop the
      // quadrant straight down.
      addBond(top, beams.get(key(f, 'x', z, bseg(sgn(x)))), COL_HEAD_SHARE, M_FRAME);
      addBond(top, beams.get(key(f, 'z', x, bseg(sgn(z)))), COL_HEAD_SHARE, M_FRAME);
      // The next storey's column starts off the slab, as before. This is a
      // slab-governed joint (the slab would punch before the column would), and
      // on the frame material it measured 24 — over the frame band; on the slab
      // material it is 7.7, mid-band.
      if (f + 1 < FLOORS)
        addBond(slabs.get(key(f, qi, qj)), columns.get(key(x, z, f + 1, 0)), COL_AREA, M_SLAB);
    }
  // INTRA-element: the two halves of each beam, full section at mid-span.
  // This is the joint that lets a beam bridging a lost column fail where a
  // real one would, instead of the ring being all-or-nothing.
  for (let k = 0; k + 1 < BEAM_SPLIT; ++k) {
    for (const z of zs)
      addBond(beams.get(key(f, 'x', z, k)), beams.get(key(f, 'x', z, k + 1)),
        BEAM_SECTION, M_FRAME);
    for (const x of xs)
      addBond(beams.get(key(f, 'z', x, k)), beams.get(key(f, 'z', x, k + 1)),
        BEAM_SECTION, M_FRAME);
  }
  // The RING: the four beams are continuous around the corners. Without these
  // four bonds there is no ring, just four beams that happen to share columns,
  // and a corner column's loss still releases its neighbours.
  for (const x of xs)
    for (const z of zs)
      addBond(beams.get(key(f, 'x', z, bseg(sgn(x)))), beams.get(key(f, 'z', x, bseg(sgn(z)))),
        BEAM_SECTION, M_FRAME);
  // Each slab quadrant bears on the two beams under its two outer edges. A
  // bearing strip is a big contact patch (0.66 m^2), so the stress here is
  // genuinely low — on the frame material that read safety factor 35, above
  // the frame band. The fix is the material, not the area: this is the slab
  // bearing and the slab governs it.
  for (let i = 0; i < SLAB_SPLIT; ++i)
    for (let j = 0; j < SLAB_SPLIT; ++j) {
      addBond(slabs.get(key(f, i, j)), beams.get(key(f, 'x', zs[j], bseg(sgn(xs[i])))),
        SLAB_BEARING, M_SLAB);
      addBond(slabs.get(key(f, i, j)), beams.get(key(f, 'z', xs[i], bseg(sgn(zs[j])))),
        SLAB_BEARING, M_SLAB);
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
  // The panel head now dies into the BEAM above it, not the slab — the beam is
  // what is physically there. This also stops the panel being a parallel strut
  // from slab to column, which is what made the facade structural (and put it
  // at safety factor 0.4) the first time this ring was tried.
  addBond(beams.get(key(f, axis === 'z' ? 'x' : 'z', coord, bseg(side))), panel, PANEL_EDGE, M_PANEL);
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
