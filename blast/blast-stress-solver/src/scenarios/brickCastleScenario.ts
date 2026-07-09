/**
 * Brick-castle scenario builder.
 *
 * Builds a large, realistic masonry castle (curtain walls in running bond, four
 * corner towers, an arched gatehouse, a central keep, and crenellated
 * battlements) as a single anchored `ScenarioDesc`, then derives ALL of its
 * bonds from real surface contact via the WASM auto-bonder.
 *
 * The whole point of this scenario is to exercise the stress solver on a *large
 * anchored structure* — the case it is actually good at. Unlike a free body
 * (where impacts become momentum, not internal stress), the castle is anchored
 * to a static foundation, so gravity + siege impacts create real internal
 * stress that the solver propagates into progressive, plausible collapse.
 *
 * Three-tier bond-strength hierarchy (expressed by scaling auto-bond areas):
 *   1. INTRA-BRICK   — chunks of one fractured brick. Strongest: a brick holds
 *      together until real violence cracks it.
 *   2. MORTAR        — brick-to-brick within one structure (e.g. a wall course).
 *      Medium: mortar joints fail before the bricks themselves.
 *   3. INTER-STRUCT  — between distinct structures (wall↔tower, tower↔gatehouse).
 *      Weakest: the seams where a wall meets a tower let go first.
 *   (ANCHOR — any bond touching the static foundation — is kept strong so the
 *   base stays put and the structure above is what collapses.)
 *
 * Bricks are fractured with cached Voronoi: each distinct brick *shape* is
 * fractured ONCE, then its chunk set is cloned/rotated/translated to every
 * placement — so three-pinata runs a handful of times, not thousands.
 */
import * as THREE from 'three';
import type { CollisionGroup, ScenarioBond, ScenarioDesc, Vec3 } from '../rapier/types';
import type { FragmentInfo, FragmentType } from '../three/fracture';
import { buildGridFoundationFragments } from '../three/fractureBuilders';
import { fractureGeometry, type PinataModule } from '../three/pinataFracture';
import { buildScenarioFromFragments, buildScenarioFromFragmentsAsync } from '../three/scenarioFromFragments';

// Stone-ish density (kg/m^3). Sandstone/limestone ≈ 2200–2600.
export const CASTLE_STONE_DENSITY = 2400;

/** Coarse structural role of a node — drives both colouring and the bond tier. */
export type CastleStructureKind =
  | 'wall'
  | 'tower'
  | 'gatehouse'
  | 'keep'
  | 'battlement'
  | 'foundation';

/** Bond tier (index into the multiplier table). */
export enum CastleBondTier {
  Anchor = 0,
  IntraBrick = 1,
  Mortar = 2,
  InterStructure = 3,
}

export type CastleBondMultipliers = {
  /** Bonds touching the static foundation (keeps the base planted). */
  anchor: number;
  /** Chunks of the same fractured brick. */
  intraBrick: number;
  /** Brick-to-brick within one structure (mortar joints). */
  mortar: number;
  /** Between two different structures (wall↔tower seams). */
  interStructure: number;
};

export const DEFAULT_CASTLE_BOND_MULTIPLIERS: CastleBondMultipliers = {
  anchor: 3.0,
  intraBrick: 9.0,
  mortar: 1.6,
  interStructure: 0.5,
};

export type BrickCastleOptions = {
  // ── Brick geometry (metres) ──
  /** Stretcher brick length (long axis). Default 0.9. */
  brickLength?: number;
  /** Brick / course height. Default 0.42. */
  brickHeight?: number;
  /**
   * Brick depth (one leaf's thickness). Default 0.45 = brickLength/2 — a classic
   * 2:1 brick. This ratio is what lets headers, closers and cross-lapped corners
   * tile exactly, so multi-wythe walls and solid tower corners have no gaps.
   */
  brickDepth?: number;
  /** Mortar gap between adjacent bricks. Default 0.025. */
  mortarGap?: number;

  // ── Castle layout (in bricks / courses) ──
  /** Number of stretchers along each curtain-wall span (between towers). Default 14. */
  wallLengthBricks?: number;
  /** Curtain-wall height in courses. Default 9. */
  wallCourses?: number;
  /**
   * Curtain-wall thickness in wythes (parallel brick leaves, each `brickDepth`
   * deep). 1 = a thin single-leaf wall (tips over from a tap); 3 = a proper
   * castle wall — outer leaf carries the merlons, the inner leaves form a
   * walkway. Default 3.
   */
  wallThicknessBricks?: number;
  /** Corner-tower wall thickness in wythes. Default 3. */
  towerThicknessBricks?: number;
  /** Corner-tower side length in bricks. Default 5. */
  towerSideBricks?: number;
  /** Corner-tower height in courses. Default 14. */
  towerCourses?: number;
  /** Central keep side length in bricks. Default 7. */
  keepSideBricks?: number;
  /** Central keep height in courses. Default 18. */
  keepCourses?: number;
  /** Gate opening width in bricks (front wall). Default 4. */
  gateWidthBricks?: number;
  /** Gate opening height in courses before the arch springs. Default 5. */
  gateHeightCourses?: number;
  /** Add crenellated battlements on top of walls/towers/keep. Default true. */
  battlements?: boolean;

  // ── Fracturing ──
  /**
   * Voronoi chunks per brick. 1 = unfractured (a brick is a single rigid box,
   * no intra-brick tier). 2–4 gives crackable bricks at a sane node budget.
   * Default 3.
   */
  chunksPerBrick?: number;

  // ── Bonding ──
  /** 'auto' = WASM triangle-contact bonder (the showcase path). Default 'auto'. */
  bondMode?: 'auto' | 'proximity';
  /** Auto-bond contact tolerance (the mortar gap). Default = mortarGap * 1.5. */
  maxSeparation?: number;
  /**
   * Keep the weak inter-structure bonds that stitch adjacent structures together.
   * Default false: structures are left as independent stress islands so a hit
   * stays local instead of cascading around the whole castle. true restores the
   * full four-tier hierarchy (and the single-island behaviour).
   */
  bondAcrossStructures?: boolean;

  // ── Strength hierarchy ──
  multipliers?: Partial<CastleBondMultipliers>;

  /** Stone density (kg/m^3). Default CASTLE_STONE_DENSITY. */
  density?: number;

  // ── Dependencies ──
  /** Pre-imported three-pinata module (required in browser ESM). */
  pinata?: PinataModule;
  /** Rapier module for convex-hull colliders. If omitted, colliderDescForNode is null. */
  rapier?: { ColliderDesc: { cuboid(hx: number, hy: number, hz: number): unknown; convexHull(points: Float32Array): unknown | null } };
};

const DEFAULTS: Required<Omit<BrickCastleOptions, 'multipliers' | 'pinata' | 'rapier' | 'maxSeparation'>> = {
  brickLength: 0.9,
  brickHeight: 0.42,
  brickDepth: 0.45, // = brickLength / 2 (2:1 brick → exact tiling)
  // Bricks TOUCH exactly (gap 0): adjacent faces are coincident so the EXACT
  // triangle bonder welds every contact — side-by-side, course-to-course AND the
  // merlons seated on top. (The 'average' bonder silently drops every *vertical*
  // (Y-normal) contact — see autoBondingOptions below — so any gap there would
  // leave the courses unbonded and the castle collapses as loose stacked rings.)
  mortarGap: 0,
  wallLengthBricks: 18,
  wallCourses: 11,
  wallThicknessBricks: 3,
  towerThicknessBricks: 3,
  towerSideBricks: 5,
  towerCourses: 18,
  keepSideBricks: 8,
  keepCourses: 24,
  gateWidthBricks: 4,
  gateHeightCourses: 5,
  battlements: true,
  // Solid bricks by default (1 chunk each). Fracturing each brick into pieces is a
  // separate concern; keep the structure simple and obviously-correct first.
  chunksPerBrick: 1,
  bondMode: 'auto',
  bondAcrossStructures: false,
  density: CASTLE_STONE_DENSITY,
};

// ── Cached brick archetypes ────────────────────────────────────────────────

/** One pre-fractured chunk of a brick, in brick-local space (centred at origin). */
type CachedChunk = { offset: Vec3; halfExtents: Vec3; geometry: THREE.BufferGeometry };
type Archetype = { size: Vec3; chunks: CachedChunk[] };

function makeArchetype(size: Vec3, chunksPerBrick: number, pinata?: PinataModule): Archetype {
  if (chunksPerBrick <= 1) {
    return {
      size,
      chunks: [{
        offset: { x: 0, y: 0, z: 0 },
        halfExtents: { x: size.x * 0.5, y: size.y * 0.5, z: size.z * 0.5 },
        geometry: new THREE.BoxGeometry(size.x, size.y, size.z),
      }],
    };
  }
  // A couple of segments helps three-pinata cut cleaner Voronoi cells.
  const box = new THREE.BoxGeometry(size.x, size.y, size.z, 2, 1, 1);
  const frags = fractureGeometry(box, {
    fragmentCount: chunksPerBrick,
    worldOffset: { x: 0, y: 0, z: 0 },
    minHalfExtent: 0.02,
    pinata,
  });
  box.dispose();
  return {
    size,
    chunks: frags.map((f) => ({ offset: f.worldPosition, halfExtents: f.halfExtents, geometry: f.geometry })),
  };
}

// ── Builder ─────────────────────────────────────────────────────────────────

export async function buildBrickCastleScenario(options: BrickCastleOptions = {}): Promise<ScenarioDesc> {
  const opt = { ...DEFAULTS, ...options };
  const mult: CastleBondMultipliers = { ...DEFAULT_CASTLE_BOND_MULTIPLIERS, ...options.multipliers };
  const L = opt.brickLength;
  const H = opt.brickHeight;
  const D = opt.brickDepth;
  const gap = opt.mortarGap;
  const pitchX = L + gap; // horizontal pitch along a course
  const pitchY = H + gap; // vertical pitch between courses
  const density = opt.density;
  // maxSeparation only matters for the 'average' bonder (unused here — see below).
  const maxSeparation = options.maxSeparation ?? Math.max(gap * 7, 0.05);

  // Archetype library (cached Voronoi). "stretcher" = standard brick; "halfBrick"
  // = the closer that fills running-bond end notches; "merlon" = battlement tooth.
  // (The gate is a carved opening in the wall — no separate pier/voussoir bricks.)
  const stretcher = makeArchetype({ x: L, y: H, z: D }, opt.chunksPerBrick, options.pinata);
  // Half-brick "closer": fills the half-pitch notch at the ends of every offset
  // (running-bond) course so wall runs are flush — no toothing holes at corners.
  const halfBrick = makeArchetype({ x: L * 0.5, y: H, z: D }, opt.chunksPerBrick, options.pinata);
  // Merlon height == one course so it rests cleanly on the top course.
  const merlon = makeArchetype({ x: L, y: H, z: D }, Math.max(1, opt.chunksPerBrick - 1), options.pinata);

  // ── Accumulators ──
  const allFragments: FragmentInfo[] = [];
  const brickIdByNode: number[] = [];
  const structureIdByNode: number[] = [];
  const kindByNode: CastleStructureKind[] = [];
  const baseColorByNode: number[] = [];

  // Brick groups (for the collision-LOD tree): per structure → per brick → node indices.
  type BrickGroup = { structureId: number; indices: number[] };
  const brickGroups: BrickGroup[] = [];

  let nextBrickId = 0;
  const reuseM = new THREE.Matrix4();
  const reuseV = new THREE.Vector3();
  const reuseSize = new THREE.Vector3();

  /** Place one brick: clone its cached chunks, bake rotationY, translate to centre. */
  function placeBrick(
    arch: Archetype,
    center: Vec3,
    rotationY: number,
    structureId: number,
    kind: CastleStructureKind,
    baseColor: number,
    isSupport = false,
  ): void {
    const brickId = nextBrickId++;
    const indices: number[] = [];
    const rot = Math.abs(rotationY) > 1e-6 ? reuseM.makeRotationY(rotationY).clone() : null;
    for (const ch of arch.chunks) {
      const geo = ch.geometry.clone();
      let off = ch.offset;
      let he = ch.halfExtents;
      if (rot) {
        geo.applyMatrix4(rot);
        reuseV.set(ch.offset.x, ch.offset.y, ch.offset.z).applyMatrix4(rot);
        off = { x: reuseV.x, y: reuseV.y, z: reuseV.z };
        geo.computeBoundingBox();
        (geo.boundingBox as THREE.Box3).getSize(reuseSize);
        he = { x: reuseSize.x * 0.5, y: reuseSize.y * 0.5, z: reuseSize.z * 0.5 };
      }
      const nodeIndex = allFragments.length;
      allFragments.push({
        worldPosition: { x: center.x + off.x, y: center.y + off.y, z: center.z + off.z },
        halfExtents: he,
        geometry: geo,
        isSupport,
        fragmentType: kindToFragmentType(kind),
        density: isSupport ? undefined : density,
      });
      brickIdByNode[nodeIndex] = brickId;
      structureIdByNode[nodeIndex] = structureId;
      kindByNode[nodeIndex] = kind;
      baseColorByNode[nodeIndex] = jitterColor(baseColor, brickId);
      indices.push(nodeIndex);
    }
    brickGroups.push({ structureId, indices });
  }

  // ── Layout geometry ──
  // Footprint: a square ring of curtain walls between four corner towers,
  // a gatehouse on the south (−Z) wall, and a keep in the centre. Every wall is
  // multi-wythe (T parallel leaves) so it has real thickness — a walkway on top,
  // the outer leaf carrying the merlons — and the tower corners are solid,
  // cross-lapped blocks (no toothing holes).
  const T = Math.max(1, Math.floor(opt.wallThicknessBricks));     // curtain-wall leaves
  const Tt = Math.max(1, Math.floor(opt.towerThicknessBricks));   // tower-wall leaves
  const pitchP = D + gap;                                         // leaf-to-leaf pitch (across thickness)
  const span = opt.wallLengthBricks * pitchX;                     // curtain-wall length, edge to edge
  const towerSpan = opt.towerSideBricks * pitchX;                 // tower outer footprint side
  const halfFoot = span * 0.5 + towerSpan * 0.5;                  // origin → tower centre
  const sideCenter = halfFoot;                                    // each wall mid-line == tower-centre coord
  let structureId = 0;

  // Colours (warm stone, varied per structure kind).
  const colWall = 0x9a8f7d;
  const colTower = 0x877c69;
  const colGate = 0xb0a48d;
  const colKeep = 0x726a5a;
  const colMerlon = 0xa49a87;

  // Foundation top sits at y=0; bricks start at y=0.
  const baseY = 0;

  /** Is (course, along) inside the carved gate opening? Rectangular below the
   *  spring line, then corbelled inward one brick per course to a point — so the
   *  arch is formed by *omitting* wall bricks, never by overlaying extra ones. */
  function gateOpening(c: number, along: number, gateHalf: number): boolean {
    if (Math.abs(along) >= gateHalf) return false;
    if (c < opt.gateHeightCourses) return true;
    const narrow = (c - opt.gateHeightCourses + 1) * pitchX;
    return Math.abs(along) < gateHalf - narrow;
  }

  /**
   * Lay ONE running-bond leaf of stretchers filling the run [edge0, edge1] (whose
   * length is a whole number of brick lengths) at perpendicular coordinate `perp`
   * and height `y`. When `offset`, the course shifts half a brick and the two end
   * notches are filled with half-brick closers — so the run is flush at both ends
   * (no toothing holes) and breaks joint with the course below/beside it.
   * `gate(centerAlong)` (optional) omits bricks inside the carved opening.
   */
  function placeLeaf(
    axis: 'x' | 'z', edge0: number, edge1: number, perp: number, y: number,
    offset: boolean, sid: number, kind: CastleStructureKind, color: number,
    isSupport: boolean, gate?: (alongCenter: number) => boolean,
  ): void {
    const rotY = axis === 'x' ? 0 : Math.PI * 0.5;
    const pt = (a: number): Vec3 => (axis === 'x' ? { x: a, y, z: perp } : { x: perp, y, z: a });
    const place = (centerAlong: number, arch: Archetype): void => {
      if (gate && gate(centerAlong)) return; // carved opening
      const jamb = !!gate && (gate(centerAlong - pitchX) || gate(centerAlong + pitchX));
      placeBrick(arch, pt(centerAlong), rotY, sid, jamb ? 'gatehouse' : kind, jamb ? colGate : color, isSupport);
    };
    const n = Math.round((edge1 - edge0) / pitchX);
    if (!offset) {
      for (let i = 0; i < n; i++) place(edge0 + (i + 0.5) * pitchX, stretcher);
    } else {
      place(edge0 + pitchX * 0.25, halfBrick);                       // start closer
      for (let i = 0; i < n - 1; i++) place(edge0 + pitchX * (i + 1), stretcher);
      place(edge1 - pitchX * 0.25, halfBrick);                       // end closer
    }
  }

  // ── Four curtain walls ── (multi-wythe; the run [−span/2, span/2] butts the
  // tower inner faces exactly because halfFoot = span/2 + towerSpan/2.)
  buildCurtainWall('x', -sideCenter, true);   // south (carries the gate)
  buildCurtainWall('x', +sideCenter, false);  // north
  buildCurtainWall('z', -sideCenter, false);  // west
  buildCurtainWall('z', +sideCenter, false);  // east

  function buildCurtainWall(axis: 'x' | 'z', lineConst: number, withGate: boolean): void {
    const sid = structureId++;
    const edge0 = -span * 0.5, edge1 = span * 0.5;
    const gateHalf = (opt.gateWidthBricks * pitchX) * 0.5;
    const outwardSign = lineConst >= 0 ? 1 : -1;
    const leafPerp = (w: number) => lineConst + (w - (T - 1) / 2) * pitchP;
    for (let c = 0; c < opt.wallCourses; c++) {
      const y = baseY + H * 0.5 + c * pitchY;
      const isSupport = c === 0; // first course is the anchored footing
      const gate = withGate ? (a: number) => gateOpening(c, a, gateHalf) : undefined;
      for (let w = 0; w < T; w++) {
        const offset = ((c + w) & 1) === 1; // break joint across courses AND leaves
        placeLeaf(axis, edge0, edge1, leafPerp(w), y, offset, sid, 'wall', colWall, isSupport, gate);
      }
    }
    // Merlons: only on the OUTER leaf; the inner leaves' top course is the walkway.
    if (opt.battlements) {
      const yTop = baseY + H * 0.5 + opt.wallCourses * pitchY;
      const perp = leafPerp(outwardSign > 0 ? T - 1 : 0);
      const rotY = axis === 'x' ? 0 : Math.PI * 0.5;
      for (let i = 0; i < opt.wallLengthBricks; i++) {
        if (i % 2 === 1) continue; // crenel gap
        const a = edge0 + (i + 0.5) * pitchX;
        // Skip merlons with no top-course brick under them (over the gate arch).
        if (withGate && gateOpening(opt.wallCourses - 1, a, gateHalf)) continue;
        const center: Vec3 = axis === 'x' ? { x: a, y: yTop, z: perp } : { x: perp, y: yTop, z: a };
        placeBrick(merlon, center, rotY, sid, 'battlement', colMerlon, false);
      }
    }
  }

  // ── Four corner towers + central keep ──
  for (const s of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    buildSquareTower(s[0] * sideCenter, s[1] * sideCenter, opt.towerSideBricks, opt.towerCourses, Tt, 'tower', colTower);
  }
  buildSquareTower(0, 0, opt.keepSideBricks, opt.keepCourses, Tt, 'keep', colKeep);

  /**
   * Thick-walled hollow square tower with SOLID, cross-lapped corners.
   *
   * Each course is a square ring `thickness` leaves deep. The two walls on the
   * "full" axis run corner-to-corner (their leaves fill the corner squares); the
   * two walls on the other axis run only the middle, butting the corner fill. The
   * full axis alternates every course, so each corner is owned (and filled solid)
   * by alternating directions — a cross-lapped quoin with no holes and no overlap.
   * Each leaf also breaks joint by course (running bond), via half-brick closers.
   */
  function buildSquareTower(
    cx: number, cz: number, sideBricks: number, courses: number, thickness: number,
    kind: CastleStructureKind, color: number,
  ): void {
    const sid = structureId++;
    const th = Math.min(Math.max(1, thickness), sideBricks - 1);
    const S = sideBricks * pitchX;          // outer side length
    const band = th * pitchP;               // wall-band (and corner-square) thickness
    const x0 = cx - S * 0.5, x1 = cx + S * 0.5;
    const z0 = cz - S * 0.5, z1 = cz + S * 0.5;
    for (let c = 0; c < courses; c++) {
      const y = baseY + H * 0.5 + c * pitchY;
      const isSupport = c === 0;
      const fbFull = (c & 1) === 0; // even: front/back full (own corners); odd: left/right full
      for (let w = 0; w < th; w++) {
        const offset = ((c + w) & 1) === 1;
        const dep = (w + 0.5) * pitchP; // this leaf's centre, measured inward from its outer edge
        if (fbFull) {
          placeLeaf('x', x0, x1, z0 + dep, y, offset, sid, kind, color, isSupport);            // front (full)
          placeLeaf('x', x0, x1, z1 - dep, y, offset, sid, kind, color, isSupport);            // back  (full)
          placeLeaf('z', z0 + band, z1 - band, x0 + dep, y, offset, sid, kind, color, isSupport); // left  (middle)
          placeLeaf('z', z0 + band, z1 - band, x1 - dep, y, offset, sid, kind, color, isSupport); // right (middle)
        } else {
          placeLeaf('z', z0, z1, x0 + dep, y, offset, sid, kind, color, isSupport);            // left  (full)
          placeLeaf('z', z0, z1, x1 - dep, y, offset, sid, kind, color, isSupport);            // right (full)
          placeLeaf('x', x0 + band, x1 - band, z0 + dep, y, offset, sid, kind, color, isSupport); // front (middle)
          placeLeaf('x', x0 + band, x1 - band, z1 - dep, y, offset, sid, kind, color, isSupport); // back  (middle)
        }
      }
    }
    // Merlons around the OUTER leaf ring; front/back own the corners, left/right
    // skip the corner cells so corner merlons are never double-placed.
    if (opt.battlements) {
      const y = baseY + H * 0.5 + courses * pitchY;
      const zf = z0 + pitchP * 0.5, zb = z1 - pitchP * 0.5; // outer-leaf z
      const xl = x0 + pitchP * 0.5, xr = x1 - pitchP * 0.5; // outer-leaf x
      for (let i = 0; i < sideBricks; i++) {
        if (i % 2 === 1) continue;
        const x = x0 + (i + 0.5) * pitchX;
        placeBrick(merlon, { x, y, z: zf }, 0, sid, 'battlement', colMerlon, false);
        placeBrick(merlon, { x, y, z: zb }, 0, sid, 'battlement', colMerlon, false);
      }
      for (let j = 1; j < sideBricks - 1; j++) {
        if (j % 2 === 1) continue;
        const z = z0 + (j + 0.5) * pitchX;
        placeBrick(merlon, { x: xl, y, z }, Math.PI * 0.5, sid, 'battlement', colMerlon, false);
        placeBrick(merlon, { x: xr, y, z }, Math.PI * 0.5, sid, 'battlement', colMerlon, false);
      }
    }
  }

  // ── Foundation (static anchor + courtyard floor) ──
  const footprint = (halfFoot + towerSpan * 0.5) * 2;
  const { fragments: foundationFragments } = buildGridFoundationFragments({
    width: footprint,
    depth: footprint,
    height: 0.6,
    groundClearance: 0.001,
  });
  const foundationStructureId = structureId++;
  for (const f of foundationFragments) {
    const nodeIndex = allFragments.length;
    // Lift so the foundation top sits at y = 0 (bricks rest on it).
    allFragments.push({
      ...f,
      worldPosition: { x: f.worldPosition.x, y: f.worldPosition.y - (0.6 + 0.001), z: f.worldPosition.z },
      fragmentType: 'foundation',
    });
    brickIdByNode[nodeIndex] = nextBrickId++;
    structureIdByNode[nodeIndex] = foundationStructureId;
    kindByNode[nodeIndex] = 'foundation';
    baseColorByNode[nodeIndex] = 0x4a4842;
    brickGroups.push({ structureId: foundationStructureId, indices: [nodeIndex] });
  }

  // ── Bonds from contacts (auto-bonding) ──
  const dims = { x: footprint, y: (opt.keepCourses + 2) * pitchY, z: footprint };
  const scenarioOptions = {
    bondMode: opt.bondMode,
    areaNormalization: 'none' as const, // physical contact areas; tiers scale them
    dimensions: dims,
    // EXACT (not 'average'): bricks touch with coincident faces, so the exact
    // coplanar-overlap bonder welds every contact and — crucially — it is the only
    // mode that bonds VERTICAL (Y-normal) course-to-course contacts. 'average'
    // silently drops all vertical bonds, which collapses the structure.
    autoBondingOptions: { mode: 'exact' as const, maxSeparation, label: 'BrickCastle' },
    rapier: options.rapier,
  };

  const scenario = opt.bondMode === 'auto'
    ? await buildScenarioFromFragmentsAsync(allFragments, scenarioOptions)
    : buildScenarioFromFragments(allFragments, scenarioOptions);

  // ── Apply the strength hierarchy, and KEEP STRUCTURES INDEPENDENT ──
  // Two back-doors otherwise stitch the whole castle into one stress island, so a
  // hit on one structure wakes/shatters every other (e.g. hitting a corner tower
  // collapses the centre keep):
  //   1. Direct wall↔tower seam bonds (inter-structure tier).
  //   2. The shared foundation slab: every structure's footing bonds to a single
  //      connected grid of foundation tiles, so a hit routes THROUGH the foundation
  //      into all structures. (This one is the real culprit — verified from a
  //      recorded session where a corner-tower hit woke 262 keep chunks the same
  //      frame, the keep and tower sharing one foundation-linked component.)
  // Each structure's bottom course is already a static (mass-0) footing that anchors
  // it on its own, so we DROP every foundation-touching bond and leave the
  // foundation as an unbonded static floor. We also drop inter-structure bonds by
  // default. Result: each wall / tower / keep is its own island and a hit stays
  // local. Set bondAcrossStructures:true to restore the weak wall↔tower seam tier
  // (the foundation stays decoupled regardless).
  const isFoundation = (i: number) => kindByNode[i] === 'foundation';
  const tierOf = (b: ScenarioBond): CastleBondTier => {
    const a = b.node0, c = b.node1;
    if (isFoundation(a) || isFoundation(c)) return CastleBondTier.Anchor;
    if (brickIdByNode[a] === brickIdByNode[c]) return CastleBondTier.IntraBrick;
    if (structureIdByNode[a] === structureIdByNode[c]) return CastleBondTier.Mortar;
    return CastleBondTier.InterStructure;
  };
  const tierMult = [mult.anchor, mult.intraBrick, mult.mortar, mult.interStructure];
  const keptBonds: ScenarioBond[] = [];
  const keptTiers: number[] = [];
  for (const b of scenario.bonds) {
    const tier = tierOf(b);
    if (tier === CastleBondTier.Anchor) continue; // foundation = unbonded static floor
    if (tier === CastleBondTier.InterStructure && !opt.bondAcrossStructures) continue;
    keptBonds.push({ ...b, area: Math.max(b.area * tierMult[tier], 1e-8) });
    keptTiers.push(tier);
  }
  scenario.bonds = keptBonds;
  const bondTiers = Uint8Array.from(keptTiers);

  // ── Collider skin: cuboids (not hulls) for solid box bricks ──
  // A Rapier convex hull carries a ~6 mm collision margin, so two bricks placed
  // face-to-face (gap 0 — required by the EXACT vertical bonder) overlap by that
  // margin and shove apart with a "pop" the instant a bond between them breaks. A
  // box brick is exactly a cuboid (rotation is baked into the AABB half-extents),
  // so a slightly-inset cuboid collider matches the brick, leaves a hair of
  // clearance between neighbours (no pop), and is cheaper than a hull.
  if (options.rapier && opt.chunksPerBrick <= 1) {
    const rapier = options.rapier;
    const sizes = (scenario.parameters?.fragmentSizes ?? []) as Vec3[];
    const SKIN = 0.004; // 4 mm inset per side
    scenario.colliderDescForNode = scenario.nodes.map((_n, i) => {
      const s = sizes[i] ?? { x: 0.1, y: 0.1, z: 0.1 };
      const hx = Math.max(s.x * 0.5 - SKIN, s.x * 0.25);
      const hy = Math.max(s.y * 0.5 - SKIN, s.y * 0.25);
      const hz = Math.max(s.z * 0.5 - SKIN, s.z * 0.25);
      return () => rapier.ColliderDesc.cuboid(hx, hy, hz);
    }) as NonNullable<typeof scenario.colliderDescForNode>;
  }

  // ── Collision-LOD tree: castle → structure → brick(leaf) ──
  const byStructure = new Map<number, number[][]>();
  for (const g of brickGroups) {
    const arr = byStructure.get(g.structureId);
    if (arr) arr.push(g.indices); else byStructure.set(g.structureId, [g.indices]);
  }
  const structureNodes: CollisionGroup[] = [];
  for (const indicesList of byStructure.values()) {
    structureNodes.push({ children: indicesList.map((fragments) => ({ fragments })) });
  }
  scenario.collisionTree = [{ children: structureNodes }];

  // ── Metadata for the demo (colours, HUD, tier breakdown) ──
  const tierCounts = [0, 0, 0, 0];
  for (let i = 0; i < bondTiers.length; i++) tierCounts[bondTiers[i]]++;
  scenario.parameters = {
    ...scenario.parameters,
    castle: true,
    brickIdByNode,
    structureIdByNode,
    kindByNode,
    baseColorByNode,
    bondTiers,
    tierCounts,
    structureCount: structureId,
    brickCount: nextBrickId,
    multipliers: mult,
    dims,
  };

  return scenario;
}

// ── helpers ─────────────────────────────────────────────────────────────────

function kindToFragmentType(kind: CastleStructureKind): FragmentType {
  return kind === 'foundation' ? 'foundation' : 'wall';
}

/** Deterministic small brightness jitter per brick so individual bricks read. */
function jitterColor(base: number, seed: number): number {
  const c = new THREE.Color(base);
  // Cheap hash → [-0.08, 0.08] multiplier on lightness.
  const h = Math.sin(seed * 12.9898) * 43758.5453;
  const f = (h - Math.floor(h)) - 0.5;
  const k = 1 + f * 0.16;
  c.r = Math.min(1, Math.max(0, c.r * k));
  c.g = Math.min(1, Math.max(0, c.g * k));
  c.b = Math.min(1, Math.max(0, c.b * k));
  return c.getHex();
}
