/**
 * SUPERSEDED by scripts/export-fractured-city.mjs — prefer that for new work.
 *
 * Packs from this script contain ~44 m^3 of real solid-solid interpenetration
 * per city: the frame is deliberately embedded in the wall's thickness band so
 * the proximity bonder finds contacts (see "Hybrid bonding" below). Bonded
 * bodies that overlap are compressed springs — the contact solver ejects them
 * when a bond breaks — which reads on screen as the structure detonating rather
 * than collapsing. Measure any pack from here with
 * demos/blast-stress-demo/tools/analyze_overlap.py before trusting it, and see
 * .cursor/skills/blast-destruction-diagnostics/SKILL.md for the full diagnosis.
 * export-fractured-city.mjs replaces the proximity bonder with exact contact
 * areas and authors the wall skin flush outside the frame instead.
 *
 * Mini-city — ScenePack v2, materials-based (NOT Rapier's multiplier-based area hack).
 *
 *   node scripts/export-mini-city-v2.mjs
 *   -> assets/mini-city/mini-city-v2.json
 *
 * A GRID x GRID skyline of independent buildings, each with:
 *   - a box-authored structural frame (foundation footings, columns, slab
 *     diaphragms) — cheap to render, cheap to collide, and bonded with EXACT
 *     geometric areas the same way scripts/export-reference-building.mjs does.
 *   - real Voronoi-fractured exterior wall panels (convex-hull shards, one
 *     fracture per building face spanning its full height) — irregular
 *     fragments, not boxes, so the facade actually looks broken when it breaks.
 *
 * THE ONE RULE (see SCENE_PACK_FORMAT.md, "The one rule that matters"):
 *   bonds[].area is REAL GEOMETRIC contact area (m^2), always. Strength comes
 *   ONLY from bonds[].m (a material index). This script never scales an area
 *   to fake a strength difference — not even by a hair. Frame bonds reuse the
 *   reference building's exact section-area formulas; wall bonds come straight
 *   out of computeBondsFromFragments() (proximity/SAT overlap on the actual
 *   fractured geometry), completely unnormalized (no perAxis/uniform rescale
 *   — that rescale is itself a departure from raw geometric area, which is why
 *   this script calls computeBondsFromFragments() directly instead of going
 *   through buildScenarioFromFragments()/buildFracturedTowerScenario(), both
 *   of which also bake in applyBondStrengthMultipliers()).
 *
 * Hybrid bonding, per building:
 *   1. Author the frame (foundation/column/slab) as plain boxes with bonds
 *      computed from the exact cross-sections they represent — proven pattern,
 *      copied from export-reference-building.mjs.
 *   2. Wrap each of the 4 exterior faces (full building height, one wall per
 *      face) around a shard pattern drawn from a small pool of pre-fractured
 *      UNIT wall templates (buildWallTemplates()/instantiateWall(), wraps
 *      @dgreenheck/three-pinata) — the Voronoi fracture itself runs only
 *      WALL_TEMPLATE_COUNT times for the whole city, not once per wall; every
 *      actual wall reuses a template's shard pattern rescaled to its real
 *      span/height. Offline/build-time only, no runtime fracturing.
 *   3. Wrap every frame node as a lightweight box FragmentInfo proxy, run
 *      computeBondsFromFragments() over [frame proxies + wall shards]
 *      TOGETHER, then keep only the bonds that touch a wall shard (wall~wall
 *      seams, wall~column / wall~slab clips) — the frame~frame bonds that
 *      same pass would also produce are discarded because step 1 already
 *      authored those with hand-derived exact areas, and keeping both would
 *      double-bond the frame.
 *
 * Materials table mirrors export-reference-building.mjs's shape and is
 * retuned independently for this (differently proportioned) city — see the
 * calibration loop in .cursor/skills/blast-structure-authoring/SKILL.md.
 *
 * Env vars (all optional):
 *   GRID              city is GRID x GRID buildings (default 3)
 *   FLOORS_MIN/MAX    per-building floor count range (default 3 / 7)
 *   WIDTH_MIN/MAX     per-building footprint width in meters (default 8 / 16)
 *   STREET            gap between building footprints, meters (default 6)
 *   SEED              RNG seed for reproducible layout (default 7)
 *   FRAME_BAND        frame ductility: fatal = elastic * FRAME_BAND (default 10)
 *   FRAGMENTS_PER_WALL Voronoi shard count per building FACE (default 8) — total
 *                     wall-shard nodes = GRID^2 * 4 faces * FRAGMENTS_PER_WALL,
 *                     independent of floor count by construction (see below).
 *   FRAME_SCALE / SLAB_SCALE / FACADE_SCALE / ANCHOR_SCALE  material grade scales
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import * as pinata from '@dgreenheck/three-pinata';
import { fractureGeometry, computeBondsFromFragments } from '../dist/three.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT = path.resolve(__dirname, '../assets/mini-city/mini-city-v2.json');

// ── City knobs ──────────────────────────────────────────────────────────────
const GRID = Number(process.env.GRID ?? 3);
const FLOORS_MIN = Number(process.env.FLOORS_MIN ?? 3);
const FLOORS_MAX = Number(process.env.FLOORS_MAX ?? 7);
const WIDTH_MIN = Number(process.env.WIDTH_MIN ?? 8);
const WIDTH_MAX = Number(process.env.WIDTH_MAX ?? 16);
const STREET = Number(process.env.STREET ?? 6);
const SEED = Number(process.env.SEED ?? 7);
// Voronoi shards per building FACE (not per floor — see file header: fracturing
// the whole face once, at full height, keeps the total shard count independent
// of FLOORS so the render-cost budget below holds regardless of how tall
// buildings get. Individual shards still land on, and bond to, whichever
// floor's column/slab they geometrically overlap.
const FRAGMENTS_PER_WALL = Number(process.env.FRAGMENTS_PER_WALL ?? 8);

// ── Geometry ────────────────────────────────────────────────────────────────
const BAY = 4.0; // m between column lines
const FLOOR_HEIGHT = 3.0; // m floor to floor
const COL = Number(process.env.COL ?? 0.4); // column section, square
const SLAB_T = 0.25;
const WALL_T = 0.15;
const FOUND_HALF = 0.3; // foundation footing half-extent (matches export-reference-building.mjs)
const CONCRETE = 2400; // kg/m^3
const DRYWALL = 700; // kg/m^3, facade shard density
const COL_SEGMENTS = 2; // per floor — see export-reference-building.mjs's granularity note
const SLAB_SPLIT = 2; // per axis -> 2x2 quadrants per floor

// ── Materials (Pa) ──────────────────────────────────────────────────────────
// Ductility is the fatal-elastic band (see export-reference-building.mjs /
// SKILL.md): the frame yields over many frames, the facade lets go quickly.
// Elastic (strength) is untouched by FRAME_BAND, so gravity safety factors are
// IDENTICAL for any value here — it changes only HOW the frame fails.
const FRAME_BAND = Number(process.env.FRAME_BAND ?? 10);
// The frame is deliberately a grade above the reference building's. Measured
// at scale 1 these column joints sat at safety factor ~3.3 — below SKILL.md's
// 5-20 frame band, and below the facade clips holding the cladding onto them.
// That inversion is what made projectiles shear columns instead of peeling
// facade: the sacrificial layer was stronger than the structure behind it.
// At 2 the frame lands at ~6.5, inside its band and above the facade, so the
// cladding is what gives way first.
const FRAME_SCALE = Number(process.env.FRAME_SCALE ?? 2);
const SLAB_SCALE = Number(process.env.SLAB_SCALE ?? 1);
const FACADE_SCALE = Number(process.env.FACADE_SCALE ?? 1);
const ANCHOR_SCALE = Number(process.env.ANCHOR_SCALE ?? 1);

const MATERIALS = [
  {
    name: 'reinforced-concrete',
    compressionElastic: 24e6 * FRAME_SCALE, compressionFatal: 24e6 * FRAME_SCALE * FRAME_BAND,
    tensionElastic: 3.0e6 * FRAME_SCALE, tensionFatal: 3.0e6 * FRAME_SCALE * FRAME_BAND,
    // Shear raised from the reference building's 4.0e6: this city's taller
    // buildings (up to FLOORS_MAX floors on a fixed COL section) put more
    // shear through column~slab/column~column than the 3-floor reference
    // ever measured — see the calibration-loop note at the bottom of this file.
    shearElastic: 5.8e6 * FRAME_SCALE, shearFatal: 5.8e6 * FRAME_SCALE * FRAME_BAND,
  },
  {
    name: 'concrete-slab',
    compressionElastic: 12e6 * SLAB_SCALE, compressionFatal: 30e6 * SLAB_SCALE,
    tensionElastic: 1.2e6 * SLAB_SCALE, tensionFatal: 3.0e6 * SLAB_SCALE,
    shearElastic: 1.6e6 * SLAB_SCALE, shearFatal: 4.0e6 * SLAB_SCALE,
  },
  {
    // Shard-to-shard seams. Meant to shatter visibly, but must still hold
    // under gravity (target safety factor 2-4, same facade band as the
    // reference building). Recalibrated after the walls were banded per floor:
    // shards now stack within a band and hand weight down through their seams,
    // so shear binds here (measured 4.44e5 Pa peak) where tension used to.
    // The old 0.45 MPa shear limit sat 1% above that peak — a facade holding
    // itself up with no margin, which is what let one impact unzip a whole
    // building. 1.5 MPa is an ordinary grouted precast-panel seam and puts
    // this class back in the 2-4 band. Strength moves on the material axis
    // only; areas stay pure geometry (see the note at the bottom of this file).
    name: 'facade-panel',
    compressionElastic: 2.4e6 * FACADE_SCALE, compressionFatal: 6.0e6 * FACADE_SCALE,
    tensionElastic: 1.5e6 * FACADE_SCALE, tensionFatal: 5.0e6 * FACADE_SCALE,
    shearElastic: 1.5e6 * FACADE_SCALE, shearFatal: 4.0e6 * FACADE_SCALE,
  },
  {
    // Shard-to-frame clips (wall~column, wall~slab, wall~foundation). More
    // brittle than the panel-panel seam is intentional — clips are the
    // deliberate weak link. Sized to the hardest-loaded pair sharing this
    // material (column~wall, compression-binding at 5.15e5 Pa peak); the
    // lighter-loaded slab~wall / foundation~wall pairs land above the 2-4
    // facade band as a result (safety ~5-7) but nowhere near "vacuous"
    // (thousands) — see the calibration-loop note at the bottom of this file.
    name: 'facade-clip',
    compressionElastic: 1.6e6 * FACADE_SCALE, compressionFatal: 2.9e6 * FACADE_SCALE,
    tensionElastic: 2.7e5 * FACADE_SCALE, tensionFatal: 6.6e5 * FACADE_SCALE,
    shearElastic: 9.0e5 * FACADE_SCALE, shearFatal: 2.0e6 * FACADE_SCALE,
  },
  {
    // Footing anchorage — a distinct design lever from the frame itself
    // (base overturn dominates over vertical cascade on taller buildings).
    // Target band is 30-70 (SKILL.md: "never the failure point"). Sized well
    // above plain reinforced-concrete because a real footing anchor is not
    // just a concrete-to-concrete bearing joint — it is doweled/bolted rebar
    // continuity into the pedestal, which is exactly the kind of capacity
    // difference this material axis exists to express (never by inflating
    // bond area — see the calibration-loop note at the bottom of this file).
    name: 'footing-anchor',
    compressionElastic: 1.0e8 * ANCHOR_SCALE, compressionFatal: 1.0e8 * ANCHOR_SCALE * FRAME_BAND,
    tensionElastic: 1.3e7 * ANCHOR_SCALE, tensionFatal: 1.3e7 * ANCHOR_SCALE * FRAME_BAND,
    shearElastic: 5.0e7 * ANCHOR_SCALE, shearFatal: 5.0e7 * ANCHOR_SCALE * FRAME_BAND,
  },
];
const [M_FRAME, M_SLAB, M_PANEL, M_CLIP, M_ANCHOR] = [0, 1, 2, 3, 4];

// ── RNG (deterministic city layout) ─────────────────────────────────────────
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const round = (n) => Math.round(n * 1e5) / 1e5;
const v = (x, y, z) => ({ x: round(x), y: round(y), z: round(z) });

// ── Mesh export (positions/normals/indices) — copied pattern from
// scripts/export-rust-scenes.mjs's exportGeometry(). ─────────────────────────
function exportGeometry(geometry) {
  if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  if (!position || !normal) throw new Error('geometry missing position/normal attributes');
  const index = geometry.getIndex();
  return {
    positions: Array.from(position.array, (x) => round(x)),
    normals: Array.from(normal.array, (x) => round(x)),
    indices: index
      ? Array.from(index.array)
      : Array.from({ length: position.count }, (_, i) => i),
  };
}

// Volume enclosed by a closed triangle mesh (divergence theorem). The shards
// are convex hulls straight out of the fracture, so this is their true volume.
function hullVolume(mesh) {
  const p = mesh.positions, idx = mesh.indices;
  let total = 0;
  for (let i = 0; i < idx.length; i += 3) {
    const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
    total +=
      p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) -
      p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) +
      p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return Math.abs(total) / 6;
}

// ── Wall fracture templates ──────────────────────────────────────────────────
// three-pinata's Voronoi fracture is (a) the slow step and (b) not fully
// seed-controlled by SEED above (its internal cell-placement RNG drifts
// between identical-input runs). Rather than re-fracturing a fresh box per
// wall — GRID^2 * 4 calls — fracture a small pool of UNIT-width/height panels
// ONCE and reuse each shard pattern for every actual wall by rescaling X
// (span) and Y (height). Thickness (Z) is authored real (WALL_T) in the
// template and never rescaled, so shard depth always matches the true wall
// section. This is also offline/build-time only — the demo loads the frozen
// JSON, there is no runtime fracturing.
const WALL_TEMPLATE_COUNT = 3;

function buildWallTemplates(fragmentCount, pinataModule) {
  const templates = [];
  for (let i = 0; i < WALL_TEMPLATE_COUNT; i++) {
    const unit = new THREE.BoxGeometry(1, 1, WALL_T, 2, 3, 1);
    templates.push(fractureGeometry(unit, { fragmentCount, pinata: pinataModule }));
    unit.dispose();
  }
  return templates;
}

// Reposition/rescale a cached unit template into a real wall instance. Each
// fragment's geometry is already recentered on its own centroid (that's what
// fractureGeometry() returns), so scaling both the shard geometry and its
// worldPosition by the same per-axis factor stretches shape and layout
// consistently — the same trick buildWallFragments() already used for
// rotation (applyMatrix4 handles the normal matrix for us).
function instantiateWall(template, { span, height, centerX = 0, centerZ = 0, rotationY = 0, baseY = 0, minHalfExtent = 0.05 }) {
  const cy = height * 0.5 + baseY;
  const scaleMatrix = new THREE.Matrix4().makeScale(span, height, 1);
  const rotMatrix = Math.abs(rotationY) > 0.001 ? new THREE.Matrix4().makeRotationY(rotationY) : null;

  return template.map((f) => {
    const geometry = f.geometry.clone();
    geometry.applyMatrix4(scaleMatrix);
    let position = new THREE.Vector3(f.worldPosition.x * span, f.worldPosition.y * height, f.worldPosition.z);
    if (rotMatrix) {
      geometry.applyMatrix4(rotMatrix);
      position = position.applyMatrix4(rotMatrix);
    }
    geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    return {
      worldPosition: { x: position.x + centerX, y: position.y + cy, z: position.z + centerZ },
      halfExtents: {
        x: Math.max(minHalfExtent, size.x * 0.5),
        y: Math.max(minHalfExtent, size.y * 0.5),
        z: Math.max(minHalfExtent, size.z * 0.5),
      },
      geometry,
      isSupport: false,
      fragmentType: 'wall',
    };
  });
}

// ── Per-building authoring ──────────────────────────────────────────────────
// Builds one building's local nodes/bonds (node indices are LOCAL to the
// building — 0-based — and offset by the caller when merging into the city).
function buildBuilding({ width, floors, wallTemplates, bondOptions }) {
  const nodes = [];
  const nodeTypes = [];
  const nodeSizes = [];
  const nodeColliders = [];
  const nodeMeshes = [];
  const bonds = [];

  function addNode(type, centre, half, density, fixed = false) {
    const volume = 8 * half[0] * half[1] * half[2];
    nodes.push({ centroid: v(...centre), mass: fixed ? 0 : round(volume * density), volume: round(volume) });
    nodeTypes.push(type);
    nodeSizes.push(v(half[0] * 2, half[1] * 2, half[2] * 2));
    nodeColliders.push({ kind: 'cuboid', halfExtents: v(...half) });
    // No nodeMesh for structural (box) nodes: the renderer instances all
    // boxes in one shared draw call, and only fracture-shard geometry (real
    // irregular hulls) needs its own per-actor mesh + draw call. Populating
    // every node here regressed to ~3300 unique-mesh draw calls per camera
    // pane instead of ~300 (one per wall shard) plus one instanced box call —
    // see mini_city_main.cpp / recorder/src/renderer.rs.
    nodeMeshes.push(null);
    return nodes.length - 1;
  }

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
    if (material) bond.m = material;
    bonds.push(bond);
  }

  // Column grid, foundation, frame — same shape as export-reference-building.mjs.
  const BAYS = Math.max(1, Math.round(width / BAY));
  const FOOTPRINT = BAYS * BAY + COL;
  const lines = Array.from({ length: BAYS + 1 }, (_, i) => (i - BAYS / 2) * BAY);
  const xs = lines, zs = lines;
  const SLAB_CELLS = BAYS * SLAB_SPLIT;
  const SLAB_HALF = FOOTPRINT / (2 * SLAB_CELLS);
  const cellOf = (p) => Math.min(SLAB_CELLS - 1, Math.max(0, Math.floor((p + FOOTPRINT / 2) / (FOOTPRINT / SLAB_CELLS))));

  const foundation = new Map();
  const columns = new Map();
  const slabs = new Map();
  const key = (...parts) => parts.join(':');

  for (const x of xs)
    for (const z of zs)
      foundation.set(key(x, z), addNode('foundation', [x, FOUND_HALF, z], [0.6, FOUND_HALF, 0.6], CONCRETE, true));

  const BASE_Y = 2 * FOUND_HALF; // foundation top
  const SEG_H = FLOOR_HEIGHT / COL_SEGMENTS;
  for (let f = 0; f < floors; ++f) {
    const y0 = BASE_Y + f * FLOOR_HEIGHT;
    for (const x of xs)
      for (const z of zs)
        for (let s = 0; s < COL_SEGMENTS; ++s)
          columns.set(key(x, z, f, s), addNode('column', [x, y0 + SEG_H * (s + 0.5), z], [COL / 2, SEG_H / 2, COL / 2], CONCRETE));

    const slabY = y0 + FLOOR_HEIGHT + SLAB_T / 2;
    for (let i = 0; i < SLAB_CELLS; ++i)
      for (let j = 0; j < SLAB_CELLS; ++j)
        slabs.set(key(f, i, j), addNode('slab',
          [-FOOTPRINT / 2 + SLAB_HALF * (2 * i + 1), slabY, -FOOTPRINT / 2 + SLAB_HALF * (2 * j + 1)],
          [SLAB_HALF, SLAB_T / 2, SLAB_HALF], CONCRETE));
  }

  const COL_AREA = COL * COL;
  const SLAB_CUT = 2 * SLAB_HALF * SLAB_T;

  for (const x of xs)
    for (const z of zs)
      addBond(foundation.get(key(x, z)), columns.get(key(x, z, 0, 0)), COL_AREA, M_ANCHOR);

  for (let f = 0; f < floors; ++f) {
    for (const x of xs)
      for (const z of zs) {
        for (let s = 0; s + 1 < COL_SEGMENTS; ++s)
          addBond(columns.get(key(x, z, f, s)), columns.get(key(x, z, f, s + 1)), COL_AREA, M_FRAME);
        const top = columns.get(key(x, z, f, COL_SEGMENTS - 1));
        const qi = cellOf(x), qj = cellOf(z);
        addBond(top, slabs.get(key(f, qi, qj)), COL_AREA, M_FRAME);
        if (f + 1 < floors)
          addBond(slabs.get(key(f, qi, qj)), columns.get(key(x, z, f + 1, 0)), COL_AREA, M_FRAME);
      }
    for (let i = 0; i < SLAB_CELLS; ++i)
      for (let j = 0; j < SLAB_CELLS; ++j) {
        if (i + 1 < SLAB_CELLS) addBond(slabs.get(key(f, i, j)), slabs.get(key(f, i + 1, j)), SLAB_CUT, M_SLAB);
        if (j + 1 < SLAB_CELLS) addBond(slabs.get(key(f, i, j)), slabs.get(key(f, i, j + 1)), SLAB_CUT, M_SLAB);
      }
  }

  const FRAME_COUNT = nodes.length; // everything below this index is frame

  // ── Exterior walls: one Voronoi fracture per face PER FLOOR ─────────────
  // Fracturing each face once over the full height produced 10-19 m shards
  // (multi-tonne slabs on a 27 m building). Those are wrong three ways: a
  // projectile cannot punch a local hole in one, losing a single shard strips
  // a third of a facade at once (the "no damage, then the building explodes"
  // cliff), and a full-height shard straddles any floor cutoff — the demo's
  // height-truncated grid variants left half their shards cantilevered up to
  // 7.6 m above the frame, which drove the column~wall clips past their
  // elastic limit under gravity alone (safety factor 0.99 at rest).
  // Banding the fracture per floor fixes all three: shards are floor-height
  // panel pieces, and every band boundary lands exactly on a floor cutoff.
  const halfFootprint = FOOTPRINT / 2;
  const sideSpan = FOOTPRINT - WALL_T * 2; // side walls fit between front/back, no corner double-fill
  const faces = [
    { span: FOOTPRINT, centerX: 0, centerZ: -halfFootprint + WALL_T * 0.5, rotationY: 0 },
    { span: FOOTPRINT, centerX: 0, centerZ: halfFootprint - WALL_T * 0.5, rotationY: 0 },
    { span: sideSpan, centerX: -halfFootprint + WALL_T * 0.5, centerZ: 0, rotationY: Math.PI * 0.5 },
    { span: sideSpan, centerX: halfFootprint - WALL_T * 0.5, centerZ: 0, rotationY: Math.PI * 0.5 },
  ];
  const wallFragments = [];
  for (let f = 0; f < floors; ++f) {
    for (let face = 0; face < faces.length; ++face) {
      // Rotate through the template pool by floor AND face so the facade does
      // not visibly repeat one shard pattern up a column or around a corner.
      const template = wallTemplates[(f + face) % wallTemplates.length];
      wallFragments.push(...instantiateWall(template, {
        ...faces[face],
        height: FLOOR_HEIGHT,
        baseY: BASE_Y + f * FLOOR_HEIGHT,
      }));
    }
  }

  // Append wall shards as real nodes (convex-hull collider + real mesh), in
  // the SAME order as wallFragments so their node index (FRAME_COUNT + j)
  // matches their position in the `combined` array used for bonding below.
  for (const f of wallFragments) {
    const hx = f.halfExtents.x, hy = f.halfExtents.y, hz = f.halfExtents.z;
    const mesh = exportGeometry(f.geometry);
    // Mass comes from the shard's REAL hull volume, not its bounding box. A
    // Voronoi shard is a slanted convex cell, so its AABB overstates it by
    // ~2x on average (worst case ~4x here) — that was silently making the
    // whole facade twice as heavy as the geometry it is drawn from, loading
    // the clips that hold it on. halfExtents stay the AABB: that is what
    // broadphase and the visual box path legitimately want.
    const volume = hullVolume(mesh);
    nodes.push({ centroid: v(f.worldPosition.x, f.worldPosition.y, f.worldPosition.z), mass: round(volume * DRYWALL), volume: round(volume) });
    nodeTypes.push('wall');
    nodeSizes.push(v(hx * 2, hy * 2, hz * 2));
    nodeMeshes.push(mesh);
    nodeColliders.push({ kind: 'convex_hull', points: mesh.positions });
  }

  // ── Hybrid bonding: frame proxies (boxes) + wall shards (real geometry),
  // proximity-bonded together; keep only bonds that touch a wall shard. ──
  const frameProxies = [];
  for (let i = 0; i < FRAME_COUNT; i++) {
    const s = nodeSizes[i];
    frameProxies.push({
      worldPosition: nodes[i].centroid,
      halfExtents: { x: s.x / 2, y: s.y / 2, z: s.z / 2 },
      geometry: new THREE.BoxGeometry(s.x, s.y, s.z),
      isSupport: nodes[i].mass === 0,
      fragmentType: nodeTypes[i],
    });
  }
  const combined = [...frameProxies, ...wallFragments.map((f) => ({ ...f, fragmentType: 'wall' }))];
  const detected = computeBondsFromFragments(combined, bondOptions);

  // A full-height wall shard's bounding box geometrically overlaps every
  // column segment / slab quadrant it happens to pass by along its height
  // and length (by construction — columns are embedded in the wall's
  // thickness band so contacts are real overlaps, not marginal touches), so
  // an unpruned proximity pass produces many more "clip" bonds per shard
  // than a real facade clip fixing would ever have. Keep only each wall
  // shard's nearest few contacts per frame type (by centroid distance) —
  // physically: a panel clips to the column(s)/slab(s) it actually sits
  // against, not to every member whose footprint its bounding box crosses.
  const CLIPS_PER_TYPE = 2;
  const wallFrameCandidates = new Map(); // wallNodeIndex -> Map<frameType, [{bond,dist}]>
  const wallWallBonds = [];
  for (const b of detected) {
    if (b.node0 < FRAME_COUNT && b.node1 < FRAME_COUNT) continue; // frame~frame: already authored exactly above
    const t0 = combined[b.node0].fragmentType;
    const t1 = combined[b.node1].fragmentType;
    if (t0 === 'wall' && t1 === 'wall') {
      wallWallBonds.push(b);
      continue;
    }
    const wallIdx = t0 === 'wall' ? b.node0 : b.node1;
    const frameIdx = t0 === 'wall' ? b.node1 : b.node0;
    const frameType = combined[frameIdx].fragmentType;
    const wp = combined[wallIdx].worldPosition, fp = combined[frameIdx].worldPosition;
    const dist = Math.hypot(wp.x - fp.x, wp.y - fp.y, wp.z - fp.z);
    let byType = wallFrameCandidates.get(wallIdx);
    if (!byType) { byType = new Map(); wallFrameCandidates.set(wallIdx, byType); }
    let list = byType.get(frameType);
    if (!list) { list = []; byType.set(frameType, list); }
    list.push({ b, dist });
  }

  const pushDetected = (b, material) =>
    bonds.push({ node0: b.node0, node1: b.node1, centroid: v(b.centroid.x, b.centroid.y, b.centroid.z), normal: v(b.normal.x, b.normal.y, b.normal.z), area: round(b.area), m: material });

  for (const b of wallWallBonds) pushDetected(b, M_PANEL);
  for (const byType of wallFrameCandidates.values()) {
    for (const list of byType.values()) {
      list.sort((a, c) => a.dist - c.dist);
      for (const { b } of list.slice(0, CLIPS_PER_TYPE)) pushDetected(b, M_CLIP);
    }
  }

  for (const g of frameProxies) g.geometry.dispose();

  const mass = nodes.reduce((s, n) => s + n.mass, 0);
  return { nodes, nodeTypes, nodeSizes, nodeColliders, nodeMeshes, bonds, mass, frameCount: FRAME_COUNT, wallCount: wallFragments.length };
}

// ── City assembly ────────────────────────────────────────────────────────────
async function main() {
  const rng = mulberry32(SEED * 2654435761);
  const cellPitch = WIDTH_MAX + STREET;
  const span = (GRID - 1) * cellPitch;
  const half = span / 2;

  // Fracture the reusable wall-shard pool ONCE for the whole city (see
  // buildWallTemplates() above) instead of per building/per wall.
  const wallTemplates = buildWallTemplates(FRAGMENTS_PER_WALL, pinata);

  const cityNodes = [];
  const cityNodeTypes = [];
  const cityNodeSizes = [];
  const cityNodeColliders = [];
  const cityNodeMeshes = [];
  const cityBonds = [];

  let totalMass = 0;
  let totalWallNodes = 0;
  let totalFrameNodes = 0;
  const perBuildingSummary = [];

  // Proximity bond detection tuned for embedded box-in-wall contacts: columns
  // and slabs are geometrically embedded in the wall's thickness band (by
  // construction — see buildBuilding), so overlaps are real, not razor-thin,
  // but Voronoi shard boundaries are irregular, so a slightly wider tolerance
  // than the library default catches genuine adjacent-shard seams reliably.
  const bondOptions = { toleranceFactor: 0.18, minOverlapRatio: 0.12, minGapTolerance: 0.02, pairGapScale: 0.25 };

  let idx = 0;
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      const width = WIDTH_MIN + rng() * (WIDTH_MAX - WIDTH_MIN);
      const floors = FLOORS_MIN + Math.floor(rng() * (FLOORS_MAX - FLOORS_MIN + 1));
      const cx = -half + c * cellPitch;
      const cz = -half + r * cellPitch;

      const b = buildBuilding({ width, floors, wallTemplates, bondOptions });

      const base = cityNodes.length;
      for (const n of b.nodes) {
        cityNodes.push({ centroid: v(n.centroid.x + cx, n.centroid.y, n.centroid.z + cz), mass: n.mass, volume: n.volume });
      }
      cityNodeTypes.push(...b.nodeTypes);
      cityNodeSizes.push(...b.nodeSizes);
      cityNodeColliders.push(...b.nodeColliders);
      cityNodeMeshes.push(...b.nodeMeshes);
      for (const bond of b.bonds) {
        cityBonds.push({
          node0: bond.node0 + base,
          node1: bond.node1 + base,
          centroid: v(bond.centroid.x + cx, bond.centroid.y, bond.centroid.z + cz),
          normal: bond.normal,
          area: bond.area,
          ...(bond.m ? { m: bond.m } : {}),
        });
      }

      totalMass += b.mass;
      totalWallNodes += b.wallCount;
      totalFrameNodes += b.frameCount;
      idx += 1;
      perBuildingSummary.push({ idx, width: round(width), floors, nodes: b.nodes.length, wallShards: b.wallCount });
      process.stderr.write(`building ${idx}/${GRID * GRID}: width=${width.toFixed(1)}m floors=${floors} nodes=${b.nodes.length} wallShards=${b.wallCount}\n`);
    }
  }

  const cityRadius = Math.max(30, half + WIDTH_MAX);
  const cityMaxHeight = FLOORS_MAX * FLOOR_HEIGHT;

  const pack = {
    version: 2,
    key: 'mini-city-v2',
    title: `Mini-city v2 (${GRID}x${GRID} buildings, ScenePack v2 materials)`,
    defaults: {
      camera: { target: v(0, cityMaxHeight * 0.35, 0), distance: cityRadius * 2.2 },
      projectile: { radius: 0.6, mass: 2000, speed: 18, ttlMs: 8000 },
      solver: { gravity: -9.81, materialScale: 1, materials: MATERIALS },
      physics: {
        debrisCollisionMode: 'all', friction: 0.25, restitution: 0,
        contactForceScale: 1, skipSingleBodies: false,
      },
      optimization: {
        smallBodyDampingMode: 'always', debrisCleanupMode: 'always',
        debrisTtlMs: 10000, maxCollidersForDebris: 3,
      },
    },
    scenario: {
      nodeTypes: cityNodeTypes,
      nodes: cityNodes,
      bonds: cityBonds,
      nodeSizes: cityNodeSizes,
      nodeColliders: cityNodeColliders,
    },
    nodeMeshes: cityNodeMeshes,
  };

  await mkdir(path.dirname(OUTPUT), { recursive: true });
  await writeFile(OUTPUT, `${JSON.stringify(pack, null, 1)}\n`, 'utf8');

  console.log(`wrote ${OUTPUT}`);
  console.log(
    `buildings=${GRID * GRID} nodes=${cityNodes.length} bonds=${cityBonds.length} materials=${MATERIALS.length} ` +
    `mass=${Math.round(totalMass)}kg frameNodes=${totalFrameNodes} wallShardNodes=${totalWallNodes}`,
  );
}

await main();
