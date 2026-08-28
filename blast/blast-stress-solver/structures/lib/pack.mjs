/**
 * ScenePackBuilder — author a structure as convex prisms, get a v2 ScenePack.
 *
 * A PIECE is a convex polygon extruded along one axis. That single shape covers
 * everything a building is made of: a box is a rectangle extruded (column,
 * slab, wall, pane), and a curved balcony segment is a wedge extruded. Having
 * one primitive rather than two means the fracture path, the contact path and
 * the collider path each exist once.
 *
 * The format has NO ROTATION and NO SCALE (see SCENE_PACK_FORMAT.md), so any
 * curvature has to be baked into hull points. Extruding a polygon does exactly
 * that.
 *
 * Bond areas are exact wherever it matters:
 *   - shard <-> shard inside one piece: the shared Voronoi edge x thickness,
 *     via sharedEdgeLength.
 *   - piece <-> piece stacked along their common extrude axis (column onto
 *     slab, balcony onto slab -- the load path): convexIntersectArea of the two
 *     cross-sections, exact even for curved pieces.
 *   - everything else: the overlap rectangle of the two axis-aligned bounding
 *     boxes, which is exact for boxes and an over-estimate only for a curved
 *     piece meeting something side-on. verify.mjs bounds that case by asserting
 *     area <= min(face areas).
 * None of it is a centroid-distance proximity heuristic, which is what the
 * upstream exporter was rewritten to get away from.
 */
import {
  clipConvex, polygonArea, polygonCentroid, voronoiCells, sharedEdgeLength,
  prismMesh, mulberry32, round, v, convexIntersectArea,
} from '../../scripts/export-fractured-city.mjs';
import { MATERIALS, MATERIAL_INDEX, materialTable, bondMaterialName, material } from './materials.mjs';
import { shardsFor, maxChunkVolume, cellVolumeFor } from './fracture.mjs';
import { resolveBackend, unavailable } from './fracture-backends.mjs';
import { prismContact } from './contact.mjs';

/**
 * PhysX's GPU convex cook limit. prismMesh emits 6 vertices per polygon edge
 * (two face rings plus a quad per side), so a polygon may have at most 10
 * vertices. Throw rather than clamp: a silently simplified collider no longer
 * matches the mesh that is drawn.
 */
export const MAX_HULL_POINTS = 64;
/**
 * Lloyd relaxation passes applied to the fracture seeds before cutting.
 *
 * 0 restores the un-relaxed jittered grid. Two is where the seam widths stop
 * improving much on the structures in this repo; more mostly makes the cells
 * rounder without removing more hairlines.
 */
const LLOYD_PASSES = Number(process.env.BLAST_LLOYD_PASSES ?? 2);

const MAX_POLY_VERTS = Math.floor(MAX_HULL_POINTS / 6);

/**
 * Contact patches below this are geometrically real but structurally
 * meaningless, and because stress is force/area they show up as singularities
 * that set the reported safety factor for a whole structure. Upstream measured
 * 9 such bonds out of 2470 producing a peak stress 144x the class mean.
 *
 * 120 cm^2, not the 10 cm^2 upstream uses, because the pieces here are larger
 * and the slivers scale with them: a 50 cm^2 phantom contact between two stair
 * wedges — which meet along a line, not a face — took 26 t and reported a
 * flight at 112% of yield. Raising it further starts dropping real bonds; at
 * 250 cm^2 Villa Savoye's ribbon walls come loose.
 */
const MIN_BOND_AREA = 8e-3;

/** Two surfaces this close count as touching (1 mm). */
const TOUCH_EPS = 1e-3;

/** Plane axes (u, w) and the world mapping, per extrude axis. */
const FRAME = {
  x: { u: 1, w: 2, t: 0, toWorld: (u, w, t) => [t, u, w] },
  y: { u: 0, w: 2, t: 1, toWorld: (u, w, t) => [u, t, w] },
  z: { u: 0, w: 1, t: 2, toWorld: (u, w, t) => [u, w, t] },
};

const rect = (u0, w0, u1, w1) => [[u0, w0], [u1, w0], [u1, w1], [u0, w1]];

function polyBounds(poly) {
  let u0 = Infinity, w0 = Infinity, u1 = -Infinity, w1 = -Infinity;
  for (const [u, w] of poly) {
    u0 = Math.min(u0, u); u1 = Math.max(u1, u);
    w0 = Math.min(w0, w); w1 = Math.max(w1, w);
  }
  return [u0, w0, u1, w1];
}

/** Is this polygon an axis-aligned rectangle? Those get a cheap cuboid collider. */
function isAxisRect(poly) {
  if (poly.length !== 4) return false;
  const [u0, w0, u1, w1] = polyBounds(poly);
  const corners = rect(u0, w0, u1, w1);
  return poly.every(([u, w]) => corners.some(([cu, cw]) => Math.abs(u - cu) < 1e-7 && Math.abs(w - cw) < 1e-7));
}

export class ScenePackBuilder {
  /**
   * Distinct fracture patterns per panel class.
   *
   * The renderer instances shards city-wide on geometric identity, and a
   * building is the same few panels stamped over and over — 432 Park's 2,443
   * panels are 34 classes. Drawing every panel's jitter from one stream made
   * all 56,583 shards one-of-a-kind and defeated that entirely: measured 1.0x
   * reuse, 56,289 distinct shapes. Seeding on the panel's own class instead
   * means identical panels shatter identically and share one upload.
   *
   * More than one variant so a facade does not read as tiled. Three is where
   * the eye stops seeing a repeat; the cost is linear in the shape library.
   */
  static SHAPE_VARIANTS = 3;

  constructor({ key, title, seed = 0x5eed }) {
    this.key = key;
    this.title = title;
    this.rng = mulberry32(seed);
    /** How many panels of each class have been cut, to cycle the variants. */
    this.patternCounts = new Map();
    /**
     * The logical unit pieces are currently being added to -- a house, a wall
     * ring, one tier's rock. Auto-bonding uses it to work unit by unit rather
     * than over the whole scene at once, which is both far cheaper and a more
     * honest description of the structure: a house's chunks bond to each other,
     * and separately the house bonds to the street.
     */
    this.currentGroup = 'scene';
    this.nodeGroups = [];
    this.pieces = [];
    /** Populated by build(). */
    this.nodes = []; this.nodeTypes = []; this.nodeMaterials = [];
    this.nodeSizes = []; this.nodeColliders = []; this.bonds = []; this.nodePieces = [];
    this.shardStats = [];
    /** Contacts formed through slight overlap rather than a clean face. */
    this.penetratingContacts = 0;
    this.built = false;
  }

  /**
   * Everything authored inside `fn` belongs to the named logical unit.
   *
   * Units are what auto-bonding works over: a house's chunks are bonded among
   * themselves, and separately the house is bonded to the street it stands on.
   * Nesting is allowed and the previous unit is restored on the way out.
   */
  group(name, fn) {
    const previous = this.currentGroup;
    this.currentGroup = name;
    try { return fn(); } finally { this.currentGroup = previous; }
  }

  /**
   * Register a piece.
   *
   * @param axis      extrude axis: 'x' | 'y' | 'z'
   * @param poly      convex cross-section in that axis' plane
   *                  (x:(y,z)  y:(x,z)  z:(x,y)), or omit and pass `box`
   * @param lo,hi     extent along the extrude axis
   * @param material  material name; drives strength, look, density and fracture
   * @param type      structural role, for scenario.nodeTypes
   * @param fixed     mass 0 -- pinned to the world. Foundations only.
   * @param fracture  false leaves the piece whole (one hull)
   * @param cellVolume  overrides the material's subdivision cell size, in m^3.
   *
   * `cellVolume` exists for terrain. The cap a material carries is chosen so
   * that a BUILDING made of it breaks up convincingly, and the subdivision
   * grid applies whether or not the piece fractures -- so a pinned mountain
   * shell authored in stone dices into 2 m^3 cells and costs thousands of
   * nodes that can never move. Raising it for a fixed piece is the same
   * argument `footing-anchor` already makes at 12 m^3: a support never breaks,
   * so its size is invisible and splitting it buys nothing.
   */
  piece({ type, material: matName, axis, poly, lo, hi,
          fixed = false, fracture = true, cellVolume = null,
          fractureBackend = undefined }) {
    if (!FRAME[axis]) throw new Error(`bad axis "${axis}"`);
    if (!(hi > lo)) throw new Error(`${type}/${matName}: empty extent ${lo}..${hi} on ${axis}`);
    if (poly.length > MAX_POLY_VERTS) {
      throw new Error(`${type}/${matName}: ${poly.length}-vertex cross-section exceeds ` +
        `${MAX_POLY_VERTS} (prismMesh emits 6 verts/edge, PhysX GPU cooks at most ${MAX_HULL_POINTS})`);
    }
    if (cellVolume !== null && !(cellVolume > 0)) {
      throw new Error(`${type}/${matName}: cellVolume must be positive, got ${cellVolume}`);
    }
    material(matName); // validates the name
    this.pieces.push({
      type, material: matName, axis, poly, lo, hi, fixed, fracture, cellVolume,
      fractureBackend, group: this.currentGroup, shards: [],
    });
    return this.pieces.length - 1;
  }

  /** Convenience: an axis-aligned box, given world-space min/max corners. */
  box({ type, material, min, max, axis = 'y', fixed = false, fracture = true, cellVolume = null,
        fractureBackend = undefined }) {
    const f = FRAME[axis];
    return this.piece({
      type, material, axis, fixed, fracture, cellVolume, fractureBackend,
      poly: rect(min[f.u], min[f.w], max[f.u], max[f.w]),
      lo: min[f.t], hi: max[f.t],
    });
  }

  /** World-space AABB of a cross-section polygon + extent. */
  static aabb({ axis, poly, lo, hi }) {
    const f = FRAME[axis];
    const [u0, w0, u1, w1] = polyBounds(poly);
    const lo3 = f.toWorld(u0, w0, lo), hi3 = f.toWorld(u1, w1, hi);
    return [
      [Math.min(lo3[0], hi3[0]), Math.min(lo3[1], hi3[1]), Math.min(lo3[2], hi3[2])],
      [Math.max(lo3[0], hi3[0]), Math.max(lo3[1], hi3[1]), Math.max(lo3[2], hi3[2])],
    ];
  }

  // ── emission ──────────────────────────────────────────────────────────────

  #addNode({ type, matName, axis, poly, lo, hi, fixed, piece }) {
    const f = FRAME[axis];
    const thickness = hi - lo;
    const area = polygonArea(poly);
    const [cu, cw] = polygonCentroid(poly);
    const ct = (lo + hi) / 2;
    const centre = f.toWorld(cu, cw, ct);
    const volume = area * thickness;
    const dens = material(matName).density;
    const [u0, w0, u1, w1] = polyBounds(poly);
    const size = f.toWorld(u1 - u0, w1 - w0, thickness);

    this.nodes.push({
      centroid: v(...centre),
      mass: fixed ? 0 : round(volume * dens),
      volume: round(volume),
      m: MATERIAL_INDEX[matName],
    });
    this.nodeTypes.push(type);
    this.nodeMaterials.push(matName);
    this.nodePieces.push(piece);
    this.nodeGroups.push(this.pieces[piece]?.group ?? 'scene');
    this.nodeSizes.push(v(Math.abs(size[0]), Math.abs(size[1]), Math.abs(size[2])));

    if (isAxisRect(poly)) {
      this.nodeColliders.push({
        kind: 'cuboid',
        halfExtents: v(Math.abs(size[0]) / 2, Math.abs(size[1]) / 2, Math.abs(size[2]) / 2),
      });
    } else {
      const local = poly.map(([pu, pw]) => [pu - cu, pw - cw]);
      const mesh = prismMesh(local, thickness, (a, b, t) => f.toWorld(cu + a, cw + b, ct + t), centre);
      if (mesh.positions.length / 3 > MAX_HULL_POINTS) {
        throw new Error(`${type}/${matName}: hull has ${mesh.positions.length / 3} points > ${MAX_HULL_POINTS}`);
      }
      // Rounded before storing. Two identical panels sit at different world
      // positions, so the same pattern comes out differing in the last ULP;
      // at 1e-5 m they compare equal and share a shape-library entry.
      this.nodeColliders.push({ kind: 'convex_hull', points: mesh.positions.map(round) });
    }
    return this.nodes.length - 1;
  }

  /**
   * `normal` MUST be the true contact-surface normal, not the direction between
   * centroids. The solver splits a bond's load into normal (compression /
   * tension) and tangential (shear) components about this vector, so a wrong
   * normal books compression as shear -- upstream traced a facade sitting below
   * a safety factor of 1 while standing still to exactly that.
   */
  #addBond(a, b, area, normal, matName) {
    if (!(area > MIN_BOND_AREA)) return false;
    const ca = this.nodes[a].centroid, cb = this.nodes[b].centroid;
    const len = Math.hypot(...normal) || 1;
    this.bonds.push({
      node0: a, node1: b,
      centroid: v((ca.x + cb.x) / 2, (ca.y + cb.y) / 2, (ca.z + cb.z) / 2),
      normal: v(normal[0] / len, normal[1] / len, normal[2] / len),
      area: round(area),
      m: MATERIAL_INDEX[matName],
    });
    return true;
  }

  // ── build ─────────────────────────────────────────────────────────────────

  /** Fracture every piece, then bond seams within pieces and contacts between them. */
  build() {
    if (this.built) throw new Error('build() called twice');
    for (const p of this.pieces) this.#fracturePiece(p);
    for (const p of this.pieces) this.#bondSeams(p);
    this.#bondContacts();
    this.built = true;
    return this;
  }

  /**
   * Split a piece into cells small enough that fracturing each one produces
   * chunks under the material's volume cap, then fracture every cell.
   *
   * The cap is on chunk VOLUME, not shard count, because count alone is the
   * wrong control: clamping a 640 m^2 floor plate to eight shards gives eight
   * pieces of 80 m^3, and a building toppling into a handful of intact slabs
   * reads as one solid object falling rather than as a building breaking up.
   *
   * Cells are cut in the cross-section plane and, for a chunky piece, along the
   * extrusion too. Keeping the grid and cutting each CELL — rather than
   * fracturing the whole plate at once — is also what makes shard shapes
   * repeat: a cell's shards are keyed on the cell's own size, so the same few
   * shapes recur instead of every shard in the building being unique.
   */
  #fracturePiece(p) {
    const pieceId = this.pieces.indexOf(p);
    const { axis, poly, lo, hi, material: matName, type, fixed, fracture } = p;
    const thickness = hi - lo;
    const [u0, w0, u1, w1] = polyBounds(poly);
    const area = polygonArea(poly);

    // How many cells this piece needs, from its ACTUAL volume — not from its
    // bounding box. A curved facade band is a thin strip whose box is several
    // times its own volume, and sizing the grid off the box cut it into dozens
    // of slivers.
    const cellVolume = p.cellVolume ?? cellVolumeFor(matName, fracture);
    const cellsWanted = Math.max(1, Math.ceil((area * thickness) / cellVolume));

    // Split greedily, always halving whichever dimension currently has the
    // coarsest cell. That keeps cells roughly cubic without assuming anything
    // about the piece's shape, and leaves a plate's thickness alone.
    let nu = 1, nw = 1, nt = 1;
    const eu = u1 - u0, ew = w1 - w0;
    while (nu * nw * nt < cellsWanted) {
      const cu = eu / nu, cw = ew / nw, ct = thickness / nt;
      if (cu >= cw && cu >= ct) nu += 1;
      else if (cw >= ct) nw += 1;
      else nt += 1;
    }

    let cell = 0;
    for (let i = 0; i < nu; i += 1) {
      for (let j = 0; j < nw; j += 1) {
        const cu0 = u0 + ((u1 - u0) * i) / nu, cu1 = u0 + ((u1 - u0) * (i + 1)) / nu;
        const cw0 = w0 + ((w1 - w0) * j) / nw, cw1 = w0 + ((w1 - w0) * (j + 1)) / nw;
        // Clip the piece's own outline to the cell, so a non-rectangular piece
        // (a balcony wedge, a stair waist) is gridded within its own shape.
        const cellPoly = nu === 1 && nw === 1
          ? poly
          : clipToRect(poly, cu0, cw0, cu1, cw1);
        if (cellPoly.length < 3 || polygonArea(cellPoly) < 2e-3) continue;
        for (let k = 0; k < nt; k += 1) {
          const tLo = lo + ((hi - lo) * k) / nt, tHi = lo + ((hi - lo) * (k + 1)) / nt;
          this.#fractureCell(p, pieceId, cellPoly, tLo, tHi, { i, j, k, u0: cu0, u1: cu1, w0: cw0, w1: cw1 });
          cell += 1;
        }
      }
    }
  }

  /**
   * A generator for one panel class, cycling through `SHAPE_VARIANTS` patterns.
   *
   * The key is the panel's outline translated to its own origin, plus its
   * thickness and shard count — everything that decides the resulting geometry
   * and nothing about where the panel sits. Two windows on opposite ends of a
   * facade hash alike; a window and a door do not.
   */
  #patternRng(poly, u0, w0, thickness, n) {
    let key = `${round(thickness)}|${n}`;
    for (const [pu, pw] of poly) {
      key += `|${Math.round((pu - u0) * 1e4)},${Math.round((pw - w0) * 1e4)}`;
    }
    const seen = this.patternCounts.get(key) ?? 0;
    this.patternCounts.set(key, seen + 1);
    // FNV-1a over the key and the variant: a stable 32-bit seed that does not
    // depend on insertion order, so a pack is reproducible chunk for chunk.
    let h = 0x811c9dc5;
    const tag = `${key}#${seen % ScenePackBuilder.SHAPE_VARIANTS}`;
    for (let i = 0; i < tag.length; i += 1) {
      h ^= tag.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return mulberry32(h);
  }

  /** Fracture one cell of a piece, with whichever backend it selected. */
  #fractureCell(p, pieceId, poly, lo, hi, cell) {
    // Resolved per piece, so one structure can be cut two ways while the
    // backends are being compared rather than the choice being global-only.
    const backend = resolveBackend(p.fractureBackend);
    if (backend !== 'voronoi-2d') unavailable(backend);
    const { axis, material: matName, type, fixed } = p;
    const [u0, w0, u1, w1] = polyBounds(poly);
    const faceArea = polygonArea(poly);
    // `fracture: false` means no Voronoi, not "one enormous chunk": the piece is
    // still cut on the size grid. A foundation left whole came out at 363 m^3.
    const n = p.fracture ? shardsFor(matName, faceArea, faceArea * (hi - lo)) : 1;
    this.shardStats.push({ material: matName, shards: n });

    if (n <= 1) {
      p.shards.push({
        poly, seed: polygonCentroid(poly), cell, lo, hi,
        node: this.#addNode({ type, matName, axis, poly, lo, hi, fixed, piece: pieceId }),
      });
      return;
    }
    // Seeds on a jittered grid, not uniformly random.
    //
    // Random seeds cluster, and clustered seeds make wildly uneven Voronoi
    // cells: asking for eight shards can yield one the size of the other seven
    // and a couple too small to keep, which defeats a cap on chunk volume. A
    // grid with jitter keeps the cells within a factor of about two of each
    // other while still looking irregular.
    // Jitter from the panel's CLASS, not from the builder's stream, so two
    // identical panels produce byte-identical shards and collapse to one shape.
    const rng = this.#patternRng(poly, u0, w0, hi - lo, n);
    const gu = Math.max(1, Math.round(Math.sqrt(n * ((u1 - u0) / Math.max(w1 - w0, 1e-6)))));
    const gw = Math.max(1, Math.ceil(n / gu));
    const seeds = [];
    for (let a = 0; a < gu && seeds.length < n; a += 1) {
      for (let b = 0; b < gw && seeds.length < n; b += 1) {
        const ju = (a + 0.25 + rng() * 0.5) / gu;
        const jw = (b + 0.25 + rng() * 0.5) / gw;
        seeds.push([u0 + ju * (u1 - u0), w0 + jw * (w1 - w0)]);
      }
    }
    // Lloyd relaxation before cutting.
    //
    // A jittered grid keeps cell SIZES even, which is what it was chosen for.
    // It does not keep cell ADJACENCIES sane: two seeds that land near each
    // other still produce neighbours meeting along a hairline, and the bond
    // between them is that hairline times the wall thickness. Measured on a
    // 47,631-chunk masonry structure, 23% of all seams came out under 5 cm
    // wide, the narrowest at 1 mm, between shards a metre across. Stress is
    // force over area, so those seams read as failing under ordinary load,
    // crack, and hand their share to their neighbours -- which is what a
    // building slowly falling apart on its own looks like from outside.
    //
    // Moving each seed to the centre of its own cell and re-cutting pushes
    // seeds apart where they crowded. A few rounds is the standard cure and
    // costs nothing here: the cut is half-plane clipping over a handful of
    // seeds, run once per panel class at build time.
    //
    // Determinism is preserved -- relaxation is a pure function of the seeds,
    // which are already drawn from the panel class's own stream, so identical
    // panels still produce identical shards and still share one upload.
    for (let pass = 0; pass < LLOYD_PASSES; pass += 1) {
      const relaxing = voronoiCells(seeds, u0, w0, u1, w1);
      for (let i = 0; i < seeds.length; i += 1) {
        const cellPoly = relaxing[i];
        if (!cellPoly || cellPoly.length < 3) continue;
        const c = polygonCentroid(cellPoly);
        if (Number.isFinite(c[0]) && Number.isFinite(c[1])) seeds[i] = c;
      }
    }
    const cells = voronoiCells(seeds, u0, w0, u1, w1);
    for (let i = 0; i < cells.length; i += 1) {
      let shard = cells[i];
      if (!isAxisRect(poly)) shard = clipToConvex(shard, poly);
      // Drop slivers. A Voronoi cell that comes out at a fraction of a percent
      // of its parent is an artifact of two seeds landing close together, and
      // it has no contact worth the name: all five orphan chunks in the
      // Petronas towers were slivers whose every bond fell under the minimum
      // area, leaving them bonded to nothing at all.
      if (shard.length < 3 || polygonArea(shard) < Math.max(1e-4, faceArea * 0.015)) continue;
      if (shard.length > MAX_POLY_VERTS) shard = simplifyToBudget(shard);
      p.shards.push({
        poly: shard, seed: seeds[i], cell, lo, hi,
        node: this.#addNode({ type, matName, axis, poly: shard, lo, hi, fixed, piece: pieceId }),
      });
    }
  }

  /**
   * Seams inside one piece.
   *
   * Within a cell the shards are a Voronoi diagram, so their shared edge is
   * exact and cheap: the bisector between two seeds. Across cells they are two
   * independent diagrams with no such relationship, so those pairs go through
   * the same contact test used between pieces.
   */
  #bondSeams(p) {
    const f = FRAME[p.axis];
    const mat = p.material;
    for (let i = 0; i < p.shards.length; i++) {
      for (let j = i + 1; j < p.shards.length; j++) {
        const si = p.shards[i], sj = p.shards[j];
        const ci = si.cell, cj = sj.cell;
        const sameCell = ci.i === cj.i && ci.j === cj.j && ci.k === cj.k;

        if (sameCell) {
          // Two cells of one Voronoi diagram: their shared edge is the
          // bisector between the seeds, exactly.
          const a = si.seed, b = sj.seed;
          const nu = b[0] - a[0], nw = b[1] - a[1];
          const d = nu * (a[0] + b[0]) / 2 + nw * (a[1] + b[1]) / 2;
          const edge = sharedEdgeLength(si.poly, nu, nw, d);
          if (edge > 1e-6) {
            const t = Math.min(si.hi, sj.hi) - Math.max(si.lo, sj.lo);
            if (t > 1e-6) this.#addBond(si.node, sj.node, edge * t, f.toWorld(nu, nw, 0), mat);
          }
          continue;
        }

        const du = Math.abs(ci.i - cj.i), dw = Math.abs(ci.j - cj.j), dt = Math.abs(ci.k - cj.k);
        // Only face-adjacent cells bond. Diagonal neighbours meet along an edge
        // or at a corner, which is no bearing at all.
        if (du + dw + dt !== 1) continue;

        if (dt === 1) {
          // Stacked along the extrusion: the contact is the overlap of the two
          // cross-sections, which convexIntersectArea gives exactly.
          const area = convexIntersectArea(si.poly, sj.poly);
          if (area > 0) this.#addBond(si.node, sj.node, area, f.toWorld(0, 0, 1), mat);
          continue;
        }

        // Side by side in the cross-section plane. Both polygons were clipped
        // to their cell, so both carry an edge lying exactly on the shared cell
        // boundary; the seam is where those two edges overlap.
        //
        // Measured rather than taken from prismContact's shadow overlap, which
        // projects a whole shard and so overstates a seam between two irregular
        // cells — by up to 4x here, past what the geometry can support.
        const k = du === 1 ? 0 : 1;
        const line = du === 1
          ? (ci.i < cj.i ? ci.u1 : cj.u1)
          : (ci.j < cj.j ? ci.w1 : cj.w1);
        const a = spanOnLine(si.poly, k, line);
        const b = spanOnLine(sj.poly, k, line);
        if (!a || !b) continue;
        const edge = Math.min(a[1], b[1]) - Math.max(a[0], b[0]);
        const t = Math.min(si.hi, sj.hi) - Math.max(si.lo, sj.lo);
        if (edge > 1e-6 && t > 1e-6) {
          const n = k === 0 ? [1, 0] : [0, 1];
          this.#addBond(si.node, sj.node, edge * t, f.toWorld(n[0], n[1], 0), mat);
        }
      }
    }
  }

  /**
   * Contacts between pieces.
   *
   * AABBs are the broad phase only. The bearing itself comes from prismContact,
   * which does a separating-axis test on the real face normals and measures the
   * patch as a shadow overlap. Boxes cannot decide this: a roof sheet's AABB
   * runs from its eave to its ridge, so it reads as deeply overlapping the
   * ridge beam it is merely resting on.
   */
  #bondContacts() {
    const pieceBoxes = this.pieces.map((p) => ScenePackBuilder.aabb(p));
    for (let i = 0; i < this.pieces.length; i++) {
      for (let j = i + 1; j < this.pieces.length; j++) {
        if (!aabbTouch(pieceBoxes[i], pieceBoxes[j])) continue;
        const A = this.pieces[i], B = this.pieces[j];
        const matName = bondMaterialName(A.material, B.material);
        for (const sa of A.shards) {
          const ba = ScenePackBuilder.aabb({ axis: A.axis, poly: sa.poly, lo: sa.lo, hi: sa.hi });
          for (const sb of B.shards) {
            const bb = ScenePackBuilder.aabb({ axis: B.axis, poly: sb.poly, lo: sb.lo, hi: sb.hi });
            if (!aabbTouch(ba, bb)) continue;
            const c = prismContact(
              { axis: A.axis, poly: sa.poly, lo: sa.lo, hi: sa.hi },
              { axis: B.axis, poly: sb.poly, lo: sb.lo, hi: sb.hi },
            );
            if (!c) continue;
            if (c.penetration > TOUCH_EPS) this.penetratingContacts++;
            this.#addBond(sa.node, sb.node, c.area, c.normal, matName);
          }
        }
      }
    }
  }

  // ── output ────────────────────────────────────────────────────────────────

  emit({ cameraTarget, cameraDistance } = {}) {
    if (!this.built) throw new Error('emit() before build()');
    if (this.nodes.length !== this.nodeSizes.length || this.nodes.length !== this.nodeColliders.length) {
      throw new Error('parallel node arrays out of step'); // the parser's CountMismatch
    }
    return {
      version: 2,
      key: this.key,
      title: this.title,
      defaults: {
        camera: { target: v(...(cameraTarget ?? [0, 5, 0])), distance: cameraDistance ?? 60 },
        projectile: { radius: 0.5, mass: 1800, speed: 20, ttlMs: 8000 },
        solver: { gravity: -9.81, materialScale: 1, materials: materialTable() },
        physics: { debrisCollisionMode: 'all', friction: 0.25, restitution: 0, contactForceScale: 1, skipSingleBodies: false },
        optimization: { smallBodyDampingMode: 'always', debrisCleanupMode: 'always', debrisTtlMs: 10000, maxCollidersForDebris: 3 },
      },
      scenario: {
        nodeTypes: this.nodeTypes,
        // Additions to the format. Ignored by scene_pack.rs (no
        // deny_unknown_fields); read by the standalone viewer to pick a
        // material's colour, opacity and texture layer. nodes[].m carries the
        // same information as an index, which is what Rust will parse.
        nodeMaterials: this.nodeMaterials,
        // Which authored piece each node was fractured from. Lets a consumer tell
        // "two shards of one wall" from "two different walls", which is what
        // makes an interpenetration check meaningful: shards of one piece tile
        // exactly but their bounding boxes overlap, so they would otherwise read
        // as false positives.
        nodePieces: this.nodePieces,
        // Which logical unit each node belongs to. Authoring information, kept
        // so a consumer can tell "two shards of one house" from "two houses".
        nodeGroups: this.nodeGroups,
        nodes: this.nodes,
        bonds: this.bonds,
        nodeSizes: this.nodeSizes,
        nodeColliders: this.nodeColliders,
        // Distinct shard hulls, stored once. Written by buildShapeLibrary()
        // after emit; absent when every shard is one-of-a-kind.
      },
    };
  }
}

/**
 * The extent of a polygon's edge lying on the line `axis == value`, or null.
 * Both neighbours were clipped to their cell, so both have such an edge on the
 * boundary they share.
 */
function spanOnLine(poly, k, value) {
  const on = poly.filter((pt) => Math.abs(pt[k] - value) < 1e-6).map((pt) => pt[1 - k]);
  return on.length > 1 ? [Math.min(...on), Math.max(...on)] : null;
}

/** Clip a convex polygon to an axis-aligned cell of the subdivision grid. */
function clipToRect(poly, u0, w0, u1, w1) {
  let p = clipConvex(poly, -1, 0, -u0);
  if (p.length) p = clipConvex(p, 1, 0, u1);
  if (p.length) p = clipConvex(p, 0, -1, -w0);
  if (p.length) p = clipConvex(p, 0, 1, w1);
  return p;
}

/** Clip a Voronoi cell to a convex outline (used for non-rectangular pieces). */
function clipToConvex(cell, outline) {
  const sign = polygonArea(outline, true) >= 0 ? 1 : -1;
  let poly = cell;
  for (let i = 0; i < outline.length && poly.length; i++) {
    const p = outline[i], q = outline[(i + 1) % outline.length];
    const nx = (q[1] - p[1]) * sign, ny = -(q[0] - p[0]) * sign;
    poly = clipConvex(poly, nx, ny, nx * p[0] + ny * p[1]);
  }
  return poly;
}

/**
 * Drop the least significant vertices until the polygon fits the hull budget.
 * Removes whichever vertex changes the area least, which for a Voronoi cell
 * clipped against a curved outline is always a near-collinear pair.
 */
function simplifyToBudget(poly) {
  let p = poly.slice();
  while (p.length > MAX_POLY_VERTS) {
    let best = -1, bestLoss = Infinity;
    for (let i = 0; i < p.length; i++) {
      const trial = p.filter((_, k) => k !== i);
      const loss = Math.abs(polygonArea(poly) - polygonArea(trial));
      if (loss < bestLoss) { bestLoss = loss; best = i; }
    }
    p = p.filter((_, k) => k !== best);
  }
  return p;
}

/** Broad phase: do two AABBs come within TOUCH_EPS on every axis? */
function aabbTouch(A, B) {
  for (let k = 0; k < 3; k++) {
    if (A[0][k] - B[1][k] > TOUCH_EPS || B[0][k] - A[1][k] > TOUCH_EPS) return false;
  }
  return true;
}


