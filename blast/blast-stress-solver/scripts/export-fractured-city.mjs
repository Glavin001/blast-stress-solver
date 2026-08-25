/**
 * Fractured city — ScenePack v2, exact-contact bonding, zero interpenetration.
 *
 *   node scripts/export-fractured-city.mjs > /dev/null
 *   -> assets/mini-city/fractured-city.json
 *
 * WHY THIS EXISTS (replacing export-mini-city-v2.mjs's wall system)
 *
 * The previous generator fractured each wall with three-pinata and then found
 * bonds by proximity. Measurement of the resulting pack showed three defects
 * that together produced the "balls bounce off, then the building detonates"
 * failure:
 *
 *   1. ~44 m^3 of REAL solid-solid interpenetration (93/120 sampled
 *      column~wall pairs, worst pair 37.5% of the column buried inside a
 *      shard). The frame was deliberately embedded in the wall's thickness
 *      band so proximity bonding would find contacts. Every embedded pair is
 *      a loaded spring: the bond hides it while intact, and the contact
 *      solver ejects both bodies the instant it breaks.
 *   2. Bond areas came from a proximity heuristic with tolerance fudge
 *      factors rather than from actual shared surface.
 *   3. three-pinata's cell RNG is not seed-controlled, so the same inputs
 *      produced different packs run to run.
 *
 * All three are geometry problems, so this generator fixes them in the
 * geometry rather than compensating in materials:
 *
 *   - Each wall face/floor is a flat slab fractured as a 2D Voronoi diagram
 *     computed by half-plane clipping (clipConvex/voronoiCells below). The
 *     cells TILE the panel rectangle exactly: their areas sum to the
 *     rectangle's area and no two overlap, by construction rather than by
 *     tolerance. Extruding each cell by the wall thickness gives a convex
 *     prism shard — still an irregular fragment, just provably well formed.
 *   - Bonds come from exact shared geometry: shard~shard area is the shared
 *     polygon EDGE length times wall thickness; shard~frame area is the exact
 *     polygon-rectangle intersection on the shared contact plane. This is the
 *     same quantity a contact-based auto-bonder recovers, computed in closed
 *     form because we author the geometry.
 *   - The wall skin sits OUTSIDE the frame envelope, its inner face exactly
 *     coplanar with the frame's outer face. Flush, never overlapping.
 *   - One seeded RNG (mulberry32) drives every random choice.
 *
 * THE ONE RULE (SCENE_PACK_FORMAT.md): bonds[].area is real geometric contact
 * area in m^2, always. Strength comes only from bonds[].m (material index).
 * No area is ever scaled to express a strength difference.
 *
 * Env vars: GRID, FLOORS_MIN/MAX, WIDTH_MIN/MAX, STREET, SEED, SHARDS_PER_PANEL,
 * FRAME_BAND, FACADE_BAND, FRAME_SCALE/SLAB_SCALE/FACADE_SCALE/ANCHOR_SCALE, COL.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT = path.resolve(__dirname, '../assets/mini-city/fractured-city.json');

const GRID = Number(process.env.GRID ?? 3);
const FLOORS_MIN = Number(process.env.FLOORS_MIN ?? 3);
const FLOORS_MAX = Number(process.env.FLOORS_MAX ?? 7);
const WIDTH_MIN = Number(process.env.WIDTH_MIN ?? 8);
const WIDTH_MAX = Number(process.env.WIDTH_MAX ?? 16);
const STREET = Number(process.env.STREET ?? 6);
const SEED = Number(process.env.SEED ?? 7);
/** Voronoi shards per wall panel, where a panel is one face of one floor. */
const SHARDS_PER_PANEL = Number(process.env.SHARDS_PER_PANEL ?? 10);

/**
 * How many DISTINCT fracture patterns exist per panel class. 0 = unlimited,
 * which is the original behaviour: every panel gets its own random seeds and
 * therefore its own one-of-a-kind shards.
 *
 * Unlimited is what makes the city expensive to draw. A shard's shape is unique
 * to its panel, so no two shards share geometry, so nothing can be instanced --
 * downtown ends up with 7,160 distinct hulls and the renderer pays one draw
 * range per shard. Bounding the pattern count is what gives the renderer
 * something to instance, and it is a look-versus-cost dial rather than a
 * correctness one: N patterns per class, stamped at random onto panels.
 *
 * The unit that must match for two panels to share shard geometry is the panel
 * CLASS -- same span, same span height, same face axis. Shards are authored
 * centroid-relative, so floor height and building position do not enter into
 * it, and a pattern stamped on floor 1 of one tower yields byte-identical hulls
 * on floor 9 of another.
 *
 * Downtown has 8 classes (4 footprints x 2 face axes), so distinct wall hulls
 * come to roughly `8 * N * SHARDS_PER_PANEL`. Against 7,160 today: N=2 gives
 * ~160, N=4 ~320, N=8 ~640.
 */
const SHARD_PATTERNS = Number(process.env.SHARD_PATTERNS ?? 0);

/**
 * Seed for the pattern library itself, kept apart from the city's own seed.
 *
 * The library must not depend on how many buildings were laid out before a
 * panel asked for it, or the same pattern index would mean different shapes in
 * two runs -- and worse, changing the row layout would silently re-fracture
 * every wall in the city.
 */
const PATTERN_SEED = Number(process.env.PATTERN_SEED ?? 0x5eed);

/**
 * Normalised seed sets per panel class, built on first use.
 *
 * Seeds live in [0,1]^2 and are mapped onto the panel rect by the caller, so a
 * class is identified by its rect dimensions rather than its position.
 */
const patternLibrary = new Map();

export function panelPatternClass(axis, spanU, spanW) {
  return `${axis}:${round(spanU)}:${round(spanW)}`;
}

/**
 * The `index`-th fracture pattern for a panel class, as normalised seeds.
 *
 * Each class gets its own RNG stream seeded from the class key, so adding a new
 * building width cannot shift the patterns of the widths already present.
 */
export function panelPattern(axis, spanU, spanW, index) {
  const key = panelPatternClass(axis, spanU, spanW);
  let patterns = patternLibrary.get(key);
  if (!patterns) {
    let hash = PATTERN_SEED;
    for (let i = 0; i < key.length; i++) hash = (Math.imul(hash, 31) + key.charCodeAt(i)) | 0;
    const rng = mulberry32(hash);
    patterns = Array.from({ length: Math.max(1, SHARD_PATTERNS) }, () =>
      Array.from({ length: SHARDS_PER_PANEL }, () => [rng(), rng()]));
    patternLibrary.set(key, patterns);
  }
  return patterns[index % patterns.length];
}

/** Distinct panel classes seen, for the exporter's summary. */
export function panelPatternStats() {
  return { classes: patternLibrary.size, patternsPerClass: Math.max(1, SHARD_PATTERNS) };
}

/**
 * The shard shape library: every distinct fractured piece, stored once.
 *
 * This is the point of bounding the pattern count, and it has to be built HERE,
 * as each cut is made -- not recovered afterwards by comparing meshes. The
 * fracturer already knows it is stamping cell `c` of pattern `k` onto a panel
 * of class `cls`; that triple IS the shape's identity, known before a single
 * vertex is written. A consumer then reads an id instead of hashing point
 * arrays to guess which shards are alike.
 *
 * The library entry is also AUTHORITATIVE: the first cut of a given id defines
 * the geometry and every later instance references it rather than emitting its
 * own copy. That removes a real hazard. A panel's polygon is computed in
 * absolute panel coordinates and then made centroid-relative, so the same cell
 * on floor 1 and floor 9 goes through a different subtraction and can land a
 * few ULPs apart. Recovered-by-comparison identity is at the mercy of that;
 * referenced identity is not, and cannot drift by construction.
 */
const shapeLibrary = [];
const shapeIdByKey = new Map();
/** Largest deviation seen between a reused shape and the cut it replaced. */
let shapeReuseDriftM = 0;

/**
 * Identity of a shard: panel class, pattern, and which cell of it.
 *
 * `sign` is deliberately absent. A panel's two facing directions differ only in
 * the sign of the thickness axis, and the prism spans +/- half a thickness
 * either way, so the two carry the SAME point set walked in a different order.
 * Treating them as one shape is what turns 320 emitted arrays into 160 solids.
 * The drift assertion below is what proves that claim rather than assuming it.
 */
export function shardShapeId(cls, patternIndex, cellIndex, positions) {
  const key = `${cls}#${patternIndex}#${cellIndex}`;
  const existing = shapeIdByKey.get(key);
  if (existing !== undefined) {
    const stored = shapeLibrary[existing].points;
    if (stored.length === positions.length) {
      // Same id must mean same solid. Compare as SETS: the mirrored face emits
      // the identical points in a different order.
      const a = [...stored].sort((x, y) => x - y);
      const b = [...positions].sort((x, y) => x - y);
      for (let i = 0; i < a.length; i++) {
        shapeReuseDriftM = Math.max(shapeReuseDriftM, Math.abs(a[i] - b[i]));
      }
    } else {
      throw new Error(`shape ${key} reused with ${positions.length} coords, library has ${stored.length}`);
    }
    return existing;
  }
  const id = shapeLibrary.length;
  shapeLibrary.push({ kind: 'convex_hull', points: positions });
  shapeIdByKey.set(key, id);
  return id;
}

export function shardShapeLibrary() {
  return { shapes: shapeLibrary, driftM: shapeReuseDriftM };
}

const BAY = 4.0;
export const FLOOR_HEIGHT = 3.0;
const COL = Number(process.env.COL ?? 0.4);
const SLAB_T = 0.25;
const WALL_T = 0.15;
const FOUND_HALF = 0.3;
const CONCRETE = 2400;
const FACADE_DENSITY = 900; // lightweight precast cladding
const COL_SEGMENTS = 2;
const SLAB_SPLIT = 2;
const BASE_Y = 0.001 + FOUND_HALF * 2;
/** Smallest contact patch treated as a structural bond (m^2). */
const MIN_BOND_AREA = 1e-3;

// ── Materials (Pa) ──────────────────────────────────────────────────────────
// Ductility = fatal/elastic. The frame is ductile (FRAME_BAND) so it yields
// over many frames. The facade is deliberately BRITTLE (FACADE_BAND ~1.2):
// measurement of the previous pack showed 13-56 bonds sitting overstressed for
// six seconds of simulated time, silently draining their damage pool with
// nothing visibly moving, until the pool emptied and the released load
// cascaded through the structure in one burst. A narrow fatal-elastic band
// means an overstressed cladding joint lets go at once, at the impact site,
// which is both what a struck panel does and what makes the hit readable.
const FRAME_BAND = Number(process.env.FRAME_BAND ?? 10);
const FACADE_BAND = Number(process.env.FACADE_BAND ?? 1.2);
const FRAME_SCALE = Number(process.env.FRAME_SCALE ?? 2);
const SLAB_SCALE = Number(process.env.SLAB_SCALE ?? 1);
const FACADE_SCALE = Number(process.env.FACADE_SCALE ?? 1);
const ANCHOR_SCALE = Number(process.env.ANCHOR_SCALE ?? 1);

export const MATERIALS = [
  {
    name: 'reinforced-concrete',
    compressionElastic: 24e6 * FRAME_SCALE, compressionFatal: 24e6 * FRAME_SCALE * FRAME_BAND,
    tensionElastic: 3.0e6 * FRAME_SCALE, tensionFatal: 3.0e6 * FRAME_SCALE * FRAME_BAND,
    shearElastic: 5.8e6 * FRAME_SCALE, shearFatal: 5.8e6 * FRAME_SCALE * FRAME_BAND,
  },
  {
    name: 'concrete-slab',
    compressionElastic: 12e6 * SLAB_SCALE, compressionFatal: 12e6 * SLAB_SCALE * FRAME_BAND,
    tensionElastic: 1.2e6 * SLAB_SCALE, tensionFatal: 1.2e6 * SLAB_SCALE * FRAME_BAND,
    shearElastic: 1.6e6 * SLAB_SCALE, shearFatal: 1.6e6 * SLAB_SCALE * FRAME_BAND,
  },
  {
    // Shard-to-shard seams within a panel. Brittle: a struck panel cracks.
    // Compression is concrete-grade because a shard pressing on its neighbour
    // bears through concrete — the joint's weakness is in tension and shear,
    // which is what lets an impact separate panels. Authoring a low
    // compression limit here instead put these seams at safety factor ~1
    // under nothing but their own weight.
    name: 'facade-panel',
    compressionElastic: 12e6 * FACADE_SCALE, compressionFatal: 12e6 * FACADE_SCALE * FACADE_BAND,
    tensionElastic: 1.2e6 * FACADE_SCALE, tensionFatal: 1.2e6 * FACADE_SCALE * FACADE_BAND,
    shearElastic: 2.0e6 * FACADE_SCALE, shearFatal: 2.0e6 * FACADE_SCALE * FACADE_BAND,
  },
  {
    // Shard-to-frame: bearing on the slab ledge and tie against the column.
    // Same reasoning — compression goes through concrete; the tie is what is
    // weak, so tension is low and shear modest. Brittle band so a struck shard
    // detaches at once rather than hanging on a half-damaged joint.
    name: 'facade-clip',
    compressionElastic: 12e6 * FACADE_SCALE, compressionFatal: 12e6 * FACADE_SCALE * FACADE_BAND,
    // 0.4 MPa tension modelled an unreinforced mortar joint, which is not what
    // holds a precast panel on: real cladding is anchored to the frame with
    // steel connections, so the tie carries meaningful tension and shear. That
    // under-authoring, not the geometry, was the last thing keeping the facade
    // near safety factor 1 standing still.
    tensionElastic: 1.5e6 * FACADE_SCALE, tensionFatal: 1.5e6 * FACADE_SCALE * FACADE_BAND,
    shearElastic: 2.5e6 * FACADE_SCALE, shearFatal: 2.5e6 * FACADE_SCALE * FACADE_BAND,
  },
  {
    name: 'footing-anchor',
    compressionElastic: 1.0e8 * ANCHOR_SCALE, compressionFatal: 1.0e8 * ANCHOR_SCALE * FRAME_BAND,
    tensionElastic: 1.3e7 * ANCHOR_SCALE, tensionFatal: 1.3e7 * ANCHOR_SCALE * FRAME_BAND,
    shearElastic: 5.0e7 * ANCHOR_SCALE, shearFatal: 5.0e7 * ANCHOR_SCALE * FRAME_BAND,
  },
];
const [M_FRAME, M_SLAB, M_PANEL, M_CLIP, M_ANCHOR] = [0, 1, 2, 3, 4];

// ── Utilities ───────────────────────────────────────────────────────────────
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export const round = (n) => Math.round(n * 1e5) / 1e5;
export const v = (x, y, z) => ({ x: round(x), y: round(y), z: round(z) });

// ── 2D convex geometry ──────────────────────────────────────────────────────
const EPS = 1e-9;

/** Clip a convex polygon by the half-plane {p : dot(n,p) <= d} (Sutherland-Hodgman). */
function clipConvex(poly, nx, ny, d) {
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const da = nx * a[0] + ny * a[1] - d;
    const db = nx * b[0] + ny * b[1] - d;
    if (da <= EPS) out.push(a);
    if ((da < -EPS && db > EPS) || (da > EPS && db < -EPS)) {
      const t = da / (da - db);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}

function polygonArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return Math.abs(a) / 2;
}

function polygonCentroid(poly) {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const cross = p[0] * q[1] - q[0] * p[1];
    a += cross; cx += (p[0] + q[0]) * cross; cy += (p[1] + q[1]) * cross;
  }
  if (Math.abs(a) < EPS) {
    const n = poly.length;
    return [poly.reduce((s, p) => s + p[0], 0) / n, poly.reduce((s, p) => s + p[1], 0) / n];
  }
  return [cx / (3 * a), cy / (3 * a)];
}

/**
 * Voronoi cells of `seeds` clipped to a rectangle, by half-plane clipping.
 * Each cell is the set of points closer to its seed than to any other, so the
 * cells tile the rectangle exactly: no gaps, no overlaps, areas summing to the
 * rectangle area. Verified by the caller.
 */
function voronoiCells(seeds, x0, y0, x1, y1) {
  const rect = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return seeds.map((s, i) => {
    let poly = rect;
    for (let j = 0; j < seeds.length && poly.length; j++) {
      if (j === i) continue;
      const o = seeds[j];
      // Bisector: points p with dot(p - mid, o - s) <= 0 are on s's side.
      const nx = o[0] - s[0], ny = o[1] - s[1];
      const mx = (s[0] + o[0]) / 2, my = (s[1] + o[1]) / 2;
      poly = clipConvex(poly, nx, ny, nx * mx + ny * my);
    }
    return poly;
  });
}

/**
 * Length of the boundary shared by two convex polygons that meet along a line.
 * Cells i and j of a Voronoi diagram share their bisector, so we measure the
 * part of i's boundary lying on that bisector — the exact contact edge.
 */
function sharedEdgeLength(poly, nx, ny, d) {
  const len = Math.hypot(nx, ny);
  if (len < EPS) return 0;
  nx /= len; ny /= len; d /= len;
  let total = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const da = nx * a[0] + ny * a[1] - d;
    const db = nx * b[0] + ny * b[1] - d;
    if (Math.abs(da) < 1e-7 && Math.abs(db) < 1e-7) {
      total += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
  }
  return total;
}

/** Exact area of (convex polygon) ∩ (axis-aligned rect). */
function polygonRectArea(poly, x0, y0, x1, y1) {
  let p = poly;
  p = clipConvex(p, -1, 0, -x0); if (!p.length) return 0;
  p = clipConvex(p, 1, 0, x1); if (!p.length) return 0;
  p = clipConvex(p, 0, -1, -y0); if (!p.length) return 0;
  p = clipConvex(p, 0, 1, y1); if (!p.length) return 0;
  return polygonArea(p);
}

/** Convex-hull triangle mesh of a prism: polygon extruded +/- t/2 along its normal. */
function prismMesh(poly, thickness, toWorld, centroid) {
  const h = thickness / 2;
  const positions = [], normals = [], indices = [];
  const push = (p, n) => {
    const w = toWorld(p[0], p[1], p[2]);
    positions.push(round(w[0] - centroid[0]), round(w[1] - centroid[1]), round(w[2] - centroid[2]));
    const nn = toWorld(n[0], n[1], n[2]);
    const on = toWorld(0, 0, 0);
    const d = [nn[0] - on[0], nn[1] - on[1], nn[2] - on[2]];
    const L = Math.hypot(...d) || 1;
    normals.push(round(d[0] / L), round(d[1] / L), round(d[2] / L));
  };
  const n = poly.length;
  // Front face (+normal), back face (-normal), each as its own vertex ring so
  // normals stay flat, then a quad per side edge.
  for (const p of poly) push([p[0], p[1], h], [0, 0, 1]);
  for (const p of poly) push([p[0], p[1], -h], [0, 0, -1]);
  for (let i = 1; i + 1 < n; i++) indices.push(0, i, i + 1);
  for (let i = 1; i + 1 < n; i++) indices.push(n, n + i + 1, n + i);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ex = poly[j][0] - poly[i][0], ey = poly[j][1] - poly[i][1];
    const L = Math.hypot(ex, ey) || 1;
    const sn = [ey / L, -ex / L, 0];
    const base = positions.length / 3;
    push([poly[i][0], poly[i][1], h], sn);
    push([poly[j][0], poly[j][1], h], sn);
    push([poly[j][0], poly[j][1], -h], sn);
    push([poly[i][0], poly[i][1], -h], sn);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { positions, normals, indices };
}

// ── Per-building authoring ──────────────────────────────────────────────────
export function buildBuilding({ width, floors, rng }) {
  const nodes = [], nodeTypes = [], nodeSizes = [], nodeColliders = [], nodeMeshes = [], bonds = [];
  // Shape-library id per node, or -1 for a node that carries its own geometry
  // (every cuboid, and every shard when the pattern pool is off).
  const nodeShapeIds = [];

  function addBox(type, centre, half, density, fixed = false) {
    const volume = 8 * half[0] * half[1] * half[2];
    nodes.push({ centroid: v(...centre), mass: fixed ? 0 : round(volume * density), volume: round(volume) });
    nodeTypes.push(type);
    nodeSizes.push(v(half[0] * 2, half[1] * 2, half[2] * 2));
    nodeColliders.push({ kind: 'cuboid', halfExtents: v(...half) });
    // Cuboids are already one shared shape carried in the instance matrix, so
    // they need no library entry. Pushed anyway to keep the array parallel to
    // `nodes` -- a hole here would silently misalign every node after it.
    nodeShapeIds.push(-1);
    nodeMeshes.push(null); // boxes render through the shared instanced draw call
    return nodes.length - 1;
  }
  /**
   * `normal` MUST be the true contact-surface normal, not the direction
   * between centroids. The solver splits a bond's load into normal
   * (compression/tension) and tangential (shear) components about this
   * vector, so a wrong normal books compression as shear. A shard bearing on
   * a slab ledge sits above and outboard of the slab's centre, so the
   * centroid-to-centroid direction is diagonal, and its own weight was
   * being charged against the weak shear limit instead of the concrete
   * compression limit — enough on its own to put the facade below a safety
   * factor of 1 standing still.
   */
  function addBond(a, b, area, material, normal) {
    // Contact patches below ~10 cm^2 are geometrically real (a shard corner
    // grazing a slab cell boundary) but structurally meaningless, and because
    // stress is force/area they show up as singularities: 9 such bonds out of
    // 2470 produced a peak stress 144x the class mean and set the reported
    // safety factor for the whole facade. Dropping them changes no load path
    // — the shards involved keep their real bonds.
    if (!(area > MIN_BOND_AREA)) return;
    const ca = nodes[a].centroid, cb = nodes[b].centroid;
    let d = normal;
    if (!d) {
      d = [cb.x - ca.x, cb.y - ca.y, cb.z - ca.z];
    }
    const len = Math.hypot(...d) || 1;
    const bond = {
      node0: a, node1: b,
      centroid: v((ca.x + cb.x) / 2, (ca.y + cb.y) / 2, (ca.z + cb.z) / 2),
      normal: v(d[0] / len, d[1] / len, d[2] / len),
      area: round(area),
    };
    if (material) bond.m = material;
    bonds.push(bond);
  }
  const UP = [0, 1, 0];

  const BAYS = Math.max(1, Math.round(width / BAY));
  const FOOTPRINT = BAYS * BAY + COL;
  const half = FOOTPRINT / 2;
  const lines = Array.from({ length: BAYS + 1 }, (_, i) => (i - BAYS / 2) * BAY);
  const SLAB_CELLS = BAYS * SLAB_SPLIT;
  const SLAB_HALF = FOOTPRINT / (2 * SLAB_CELLS);

  const foundation = new Map(), columns = new Map(), slabs = new Map();
  const key = (...p) => p.join(':');
  const CELL_W = FOOTPRINT / SLAB_CELLS;
  /** [low, high] extent of slab cell `i` along one axis, with perimeter overhang. */
  const slabRange = (i) => [
    -half + i * CELL_W - (i === 0 ? WALL_T : 0),
    -half + (i + 1) * CELL_W + (i === SLAB_CELLS - 1 ? WALL_T : 0),
  ];

  for (const x of lines) for (const z of lines) {
    // Sit the footing so its TOP is exactly BASE_Y. Centring it at FOUND_HALF
    // left its top at 2*FOUND_HALF while columns started at BASE_Y =
    // 0.001 + 2*FOUND_HALF, i.e. every column was bonded to its footing across
    // a 1 mm gap with no contact surface at all. NvBlast's own contact-based
    // bond generator finds zero column~foundation contacts in that layout.
    foundation.set(key(x, z), addBox('foundation', [x, BASE_Y - FOUND_HALF, z], [FOUND_HALF, FOUND_HALF, FOUND_HALF], CONCRETE, true));
  }
  // Columns stop at the slab soffit, not at the floor line: the slab occupies
  // the top SLAB_T of each storey. Running them the full storey height buried
  // every column in the slab above it (measured 5.7 m^3 of interpenetration
  // across one building), which the contact solver would resolve explosively
  // the moment a column~slab bond broke.
  const COL_CLEAR = FLOOR_HEIGHT - SLAB_T;
  const segH = COL_CLEAR / COL_SEGMENTS;
  for (let f = 0; f < floors; f++) {
    for (const x of lines) for (const z of lines) for (let s = 0; s < COL_SEGMENTS; s++) {
      const y = BASE_Y + f * FLOOR_HEIGHT + (s + 0.5) * segH;
      columns.set(key(f, x, z, s), addBox('column', [x, y, z], [COL / 2, segH / 2, COL / 2], CONCRETE));
    }
    const slabY = BASE_Y + (f + 1) * FLOOR_HEIGHT - SLAB_T / 2;
    for (let i = 0; i < SLAB_CELLS; i++) for (let j = 0; j < SLAB_CELLS; j++) {
      // Perimeter cells overhang outward by the wall thickness so the slab
      // edge forms a real bearing ledge under the cladding above it. Without
      // the ledge the facade has no horizontal support at all and its entire
      // weight has to go through the vertical clip patches in shear, which
      // measured a gravity safety factor below 1 on three facade classes.
      const [x0, x1] = slabRange(i), [z0, z1] = slabRange(j);
      slabs.set(key(f, i, j), addBox(
        'slab',
        [(x0 + x1) / 2, slabY, (z0 + z1) / 2],
        [(x1 - x0) / 2, SLAB_T / 2, (z1 - z0) / 2],
        CONCRETE));
    }
  }
  // Frame bonds at exact section areas.
  const COL_AREA = COL * COL;
  const SLAB_CUT = (FOOTPRINT / SLAB_CELLS) * SLAB_T;
  for (const x of lines) for (const z of lines) {
    addBond(foundation.get(key(x, z)), columns.get(key(0, x, z, 0)), COL_AREA, M_ANCHOR, UP);
  }
  for (let f = 0; f < floors; f++) {
    for (const x of lines) for (const z of lines) {
      for (let s = 0; s + 1 < COL_SEGMENTS; s++) {
        addBond(columns.get(key(f, x, z, s)), columns.get(key(f, x, z, s + 1)), COL_AREA, M_FRAME, UP);
      }
      const top = columns.get(key(f, x, z, COL_SEGMENTS - 1));
      // A column on a bay line straddles two or four slab cells, so bonding
      // only the cell containing its centre both omitted load paths and gave
      // that one cell the column's whole section area. Split the contact by
      // the real overlap of the column footprint with each cell.
      for (let i = 0; i < SLAB_CELLS; i++) for (let j = 0; j < SLAB_CELLS; j++) {
        const [cx0, cx1] = slabRange(i), [cz0, cz1] = slabRange(j);
        const ox = Math.min(x + COL / 2, cx1) - Math.max(x - COL / 2, cx0);
        const oz = Math.min(z + COL / 2, cz1) - Math.max(z - COL / 2, cz0);
        if (ox <= 0 || oz <= 0) continue;
        addBond(top, slabs.get(key(f, i, j)), ox * oz, M_FRAME, UP);
        if (f + 1 < floors) addBond(slabs.get(key(f, i, j)), columns.get(key(f + 1, x, z, 0)), ox * oz, M_FRAME, UP);
      }
    }
    for (let i = 0; i < SLAB_CELLS; i++) for (let j = 0; j < SLAB_CELLS; j++) {
      // Cut area is the real shared face: the neighbour's extent along the
      // other axis times slab thickness (perimeter cells are wider).
      const [z0, z1] = slabRange(j), [x0, x1] = slabRange(i);
      if (i + 1 < SLAB_CELLS) addBond(slabs.get(key(f, i, j)), slabs.get(key(f, i + 1, j)), (z1 - z0) * SLAB_T, M_SLAB, [1, 0, 0]);
      if (j + 1 < SLAB_CELLS) addBond(slabs.get(key(f, i, j)), slabs.get(key(f, i, j + 1)), (x1 - x0) * SLAB_T, M_SLAB, [0, 0, 1]);
    }
  }

  // ── Wall skin: OUTSIDE the frame envelope, inner face flush at +/- half ──
  // Panel local coords: u along the face, w vertical, thickness along the
  // outward normal. Front/back panels span the footprint; side panels are
  // widened by WALL_T at each end so the four faces close the corners by
  // butting flush against each other rather than overlapping.
  const FACES = [
    { axis: 'z', sign: -1, spanU: FOOTPRINT, extend: 0 },
    { axis: 'z', sign: +1, spanU: FOOTPRINT, extend: 0 },
    { axis: 'x', sign: -1, spanU: FOOTPRINT + 2 * WALL_T, extend: WALL_T },
    { axis: 'x', sign: +1, spanU: FOOTPRINT + 2 * WALL_T, extend: WALL_T },
  ];

  const panels = []; // { face, floor, cells:[{poly,node,seed}] }
  for (let f = 0; f < floors; f++) {
    for (const face of FACES) {
      // The panel fills the CLEAR storey height only; the slab's outward
      // ledge occupies the band above it. So each panel sits on the ledge
      // below (bearing, in compression) and meets the ledge above, instead of
      // hanging off clips for its whole weight.
      const u0 = -face.spanU / 2, u1 = face.spanU / 2;
      const w0 = BASE_Y + f * FLOOR_HEIGHT, w1 = w0 + COL_CLEAR;
      // Everything downstream -- polygons, bond areas, masses, volumes, hull
      // points -- is derived from these seeds, so bounding the pattern count
      // changes WHICH fracture a panel gets and nothing about how it is built.
      // That is the whole reason to do the pooling here rather than by
      // substituting shapes at render time, where the shards stop tiling the
      // panel and an intact wall reads as rubble.
      const patternIndex = SHARD_PATTERNS > 0 ? Math.floor(rng() * SHARD_PATTERNS) : -1;
      const panelClass = panelPatternClass(face.axis, u1 - u0, w1 - w0);
      const seeds = SHARD_PATTERNS > 0
        ? panelPattern(face.axis, u1 - u0, w1 - w0, patternIndex)
          .map(([su, sw]) => [u0 + su * (u1 - u0), w0 + sw * (w1 - w0)])
        : Array.from({ length: SHARDS_PER_PANEL }, () => [
          u0 + rng() * (u1 - u0), w0 + rng() * (w1 - w0),
        ]);
      const polys = voronoiCells(seeds, u0, w0, u1, w1);

      // Panel plane: inner face flush with the frame's outer face at +/- half.
      const inner = face.sign * half;
      const mid = inner + face.sign * WALL_T / 2;
      const toWorld = face.axis === 'z'
        ? (u, w, t) => [u, w, mid + face.sign * t]
        : (u, w, t) => [mid + face.sign * t, w, u];

      const cells = [];
      for (let c = 0; c < polys.length; c++) {
        const poly = polys[c];
        if (poly.length < 3) { cells.push(null); continue; }
        const area = polygonArea(poly);
        if (area < 1e-4) { cells.push(null); continue; }
        const [cu, cw] = polygonCentroid(poly);
        const centre = toWorld(cu, cw, 0);
        const volume = area * WALL_T;
        // Local polygon coords are relative to the cell centroid so the mesh
        // (and the collider hull built from it) is centroid-relative, matching
        // the ScenePack convention for nodeMeshes/nodeColliders.
        const local = poly.map(([u, w]) => [u - cu, w - cw]);
        const mesh = prismMesh(local, WALL_T, (a, b, t) => toWorld(cu + a, cw + b, t), centre);
        let minU = Infinity, maxU = -Infinity, minW = Infinity, maxW = -Infinity;
        for (const [u, w] of poly) {
          minU = Math.min(minU, u); maxU = Math.max(maxU, u);
          minW = Math.min(minW, w); maxW = Math.max(maxW, w);
        }
        const sizeU = maxU - minU, sizeW = maxW - minW;
        nodes.push({ centroid: v(...centre), mass: round(volume * FACADE_DENSITY), volume: round(volume) });
        nodeTypes.push('wall');
        nodeSizes.push(face.axis === 'z' ? v(sizeU, sizeW, WALL_T) : v(WALL_T, sizeW, sizeU));
        if (SHARD_PATTERNS > 0) {
          // Identity is known here, before the geometry is stored: this is
          // cell `c` of pattern `patternIndex` on a panel of `panelClass`.
          const shapeId = shardShapeId(panelClass, patternIndex, c, mesh.positions);
          nodeShapeIds.push(shapeId);
          nodeColliders.push({ kind: 'shape', shape: shapeId });
        } else {
          nodeShapeIds.push(-1);
          nodeColliders.push({ kind: 'convex_hull', points: mesh.positions });
        }
        nodeMeshes.push(mesh);
        cells.push({ poly, node: nodes.length - 1, seed: seeds[c], area });
      }
      panels.push({ face, floor: f, cells, seeds, u0, u1, w0, w1 });
    }
  }

  // Shard~shard seams inside a panel: exact shared Voronoi edge x thickness.
  for (const panel of panels) {
    const { cells, seeds } = panel;
    for (let i = 0; i < cells.length; i++) {
      if (!cells[i]) continue;
      for (let j = i + 1; j < cells.length; j++) {
        if (!cells[j]) continue;
        const a = seeds[i], b = seeds[j];
        const nx = b[0] - a[0], ny = b[1] - a[1];
        const d = nx * (a[0] + b[0]) / 2 + ny * (a[1] + b[1]) / 2;
        const edge = sharedEdgeLength(cells[i].poly, nx, ny, d);
        // The seam plane is the Voronoi bisector, so its normal is the
        // in-panel bisector direction lifted into world axes (panel u maps to
        // world x on a z-facing wall, to world z on an x-facing one).
        const seamNormal = panel.face.axis === 'z' ? [nx, ny, 0] : [0, ny, nx];
        addBond(cells[i].node, cells[j].node, edge * WALL_T, M_PANEL, seamNormal);
      }
    }
  }

  // Shard~frame clips: exact polygon-rectangle intersection on the contact
  // plane (the frame's outer face, which the shard's inner face lies on).
  for (const panel of panels) {
    const { face, floor, cells } = panel;
    const perimeter = face.sign < 0 ? lines[0] : lines[lines.length - 1];
    for (const cell of panel.cells) {
      if (!cell) continue;
      // Columns whose outer face is coplanar with this panel.
      for (const other of lines) {
        const cu = face.axis === 'z' ? other : other;
        for (let s = 0; s < COL_SEGMENTS; s++) {
          const cy = BASE_Y + floor * FLOOR_HEIGHT + (s + 0.5) * segH;
          const area = polygonRectArea(cell.poly, cu - COL / 2, cy - segH / 2, cu + COL / 2, cy + segH / 2);
          if (area > 1e-6) {
            const node = columns.get(face.axis === 'z' ? key(floor, other, perimeter, s) : key(floor, perimeter, other, s));
            if (node !== undefined) addBond(cell.node, node, area, M_CLIP, face.axis === 'z' ? [0, 0, 1] : [1, 0, 0]);
          }
        }
      }
      // BEARING: the shard's bottom edge rests on the ledge of the slab below
      // (or, on the ground floor, on the foundation). Contact is horizontal,
      // so the area is the shard's footprint there: the length of its polygon
      // edge lying on w0, times the wall thickness. This is the load path that
      // actually carries the facade's weight, in compression.
      // SUSPENSION: its top edge meets the ledge above the same way.
      const edgeIndex = face.sign < 0 ? 0 : SLAB_CELLS - 1;
      const bottom = sharedEdgeLength(cell.poly, 0, -1, -panel.w0) * WALL_T;
      const top = sharedEdgeLength(cell.poly, 0, 1, panel.w1) * WALL_T;
      const spanOf = (poly, w) => {
        const us = poly.filter(([, ww]) => Math.abs(ww - w) < 1e-7).map(([u]) => u);
        return us.length ? [Math.min(...us), Math.max(...us)] : null;
      };
      for (const [area, w, slabFloor] of [[bottom, panel.w0, floor - 1], [top, panel.w1, floor]]) {
        if (!(area > 1e-6)) continue;
        const range = spanOf(cell.poly, w);
        if (!range) continue;
        if (slabFloor < 0) {
          // Ground floor bears on the footings it actually sits over.
          for (const other of lines) {
            const share = Math.max(0, Math.min(range[1], other + FOUND_HALF) - Math.max(range[0], other - FOUND_HALF));
            if (share > 1e-6) {
              const node = foundation.get(face.axis === 'z' ? key(other, perimeter) : key(perimeter, other));
              if (node !== undefined) addBond(cell.node, node, share * WALL_T, M_CLIP, UP);
            }
          }
          continue;
        }
        for (let k = 0; k < SLAB_CELLS; k++) {
          const [c0, c1] = slabRange(k);
          const share = Math.max(0, Math.min(range[1], c1) - Math.max(range[0], c0));
          if (share > 1e-6) {
            const node = slabs.get(face.axis === 'z' ? key(slabFloor, k, edgeIndex) : key(slabFloor, edgeIndex, k));
            if (node !== undefined) addBond(cell.node, node, share * WALL_T, M_CLIP, UP);
          }
        }
      }
    }
  }

  // Vertical seams between stacked panels on the same face, and corner seams
  // between adjacent faces, both at exact shared area.
  for (const panel of panels) {
    for (const other of panels) {
      if (panel === other) continue;
      const sameFace = panel.face === other.face;
      if (sameFace && other.floor !== panel.floor + 1) continue;
      if (!sameFace) continue; // corners are handled by the flush butt joint below
      for (const a of panel.cells) {
        if (!a) continue;
        for (const b of other.cells) {
          if (!b) continue;
          // Horizontal seam at the floor line: overlap of the two cells' u-ranges.
          const wLine = panel.w1;
          const aOn = sharedEdgeLength(a.poly, 0, 1, wLine);
          const bOn = sharedEdgeLength(b.poly, 0, -1, -wLine);
          if (aOn < 1e-7 || bOn < 1e-7) continue;
          const aU = a.poly.filter(([, w]) => Math.abs(w - wLine) < 1e-7).map(([u]) => u);
          const bU = b.poly.filter(([, w]) => Math.abs(w - wLine) < 1e-7).map(([u]) => u);
          if (!aU.length || !bU.length) continue;
          const lo = Math.max(Math.min(...aU), Math.min(...bU));
          const hi = Math.min(Math.max(...aU), Math.max(...bU));
          if (hi - lo > 1e-6) addBond(a.node, b.node, (hi - lo) * WALL_T, M_PANEL, UP);
        }
      }
    }
  }

  const mass = nodes.reduce((s, n) => s + n.mass, 0);
  const wallCount = nodeTypes.filter((t) => t === 'wall').length;
  if (nodeShapeIds.length !== nodes.length) {
    throw new Error(`nodeShapeIds ${nodeShapeIds.length} != nodes ${nodes.length}`);
  }
  return { nodes, nodeTypes, nodeSizes, nodeColliders, nodeMeshes, nodeShapeIds, bonds, mass, wallCount, footprint: FOOTPRINT };
}

// ── City assembly ───────────────────────────────────────────────────────────
async function main() {
  const rng = mulberry32(SEED * 2654435761);
  const cellPitch = WIDTH_MAX + STREET;
  const span = (GRID - 1) * cellPitch;
  const halfSpan = span / 2;

  const cityNodes = [], cityTypes = [], citySizes = [], cityColliders = [], cityMeshes = [], cityBonds = [];
  let totalMass = 0, totalWall = 0;

  for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) {
    const width = WIDTH_MIN + rng() * (WIDTH_MAX - WIDTH_MIN);
    const floors = FLOORS_MIN + Math.floor(rng() * (FLOORS_MAX - FLOORS_MIN + 1));
    const cx = -halfSpan + c * cellPitch, cz = -halfSpan + r * cellPitch;
    const b = buildBuilding({ width, floors, rng });
    const base = cityNodes.length;
    for (const n of b.nodes) cityNodes.push({ centroid: v(n.centroid.x + cx, n.centroid.y, n.centroid.z + cz), mass: n.mass, volume: n.volume });
    cityTypes.push(...b.nodeTypes);
    citySizes.push(...b.nodeSizes);
    cityColliders.push(...b.nodeColliders);
    cityMeshes.push(...b.nodeMeshes);
    for (const bond of b.bonds) {
      cityBonds.push({
        node0: bond.node0 + base, node1: bond.node1 + base,
        centroid: v(bond.centroid.x + cx, bond.centroid.y, bond.centroid.z + cz),
        normal: bond.normal, area: bond.area, ...(bond.m ? { m: bond.m } : {}),
      });
    }
    totalMass += b.mass; totalWall += b.wallCount;
    process.stderr.write(`building ${r * GRID + c + 1}/${GRID * GRID}: width=${width.toFixed(1)}m floors=${floors} nodes=${b.nodes.length} shards=${b.wallCount}\n`);
  }

  const cityRadius = Math.max(30, halfSpan + WIDTH_MAX);
  const pack = {
    version: 2,
    key: 'fractured-city',
    title: `Fractured city (${GRID}x${GRID}, exact-contact bonds)`,
    defaults: {
      camera: { target: v(0, FLOORS_MAX * FLOOR_HEIGHT * 0.35, 0), distance: cityRadius * 2.2 },
      projectile: { radius: 0.6, mass: 2000, speed: 18, ttlMs: 8000 },
      solver: { gravity: -9.81, materialScale: 1, materials: MATERIALS },
      physics: { debrisCollisionMode: 'all', friction: 0.25, restitution: 0, contactForceScale: 1, skipSingleBodies: false },
      optimization: { smallBodyDampingMode: 'always', debrisCleanupMode: 'always', debrisTtlMs: 10000, maxCollidersForDebris: 3 },
    },
    scenario: {
      nodeTypes: cityTypes, nodes: cityNodes, bonds: cityBonds,
      nodeSizes: citySizes, nodeColliders: cityColliders,
    },
    nodeMeshes: cityMeshes,
  };

  await mkdir(path.dirname(OUTPUT), { recursive: true });
  await writeFile(OUTPUT, JSON.stringify(pack));
  process.stderr.write(`wrote ${OUTPUT}\n`);
  process.stderr.write(`buildings=${GRID * GRID} nodes=${cityNodes.length} bonds=${cityBonds.length} shards=${totalWall} mass=${Math.round(totalMass)}kg\n`);
}

// Only run the city export when invoked directly; the building generator and
// its helpers are imported by sibling exporters.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
