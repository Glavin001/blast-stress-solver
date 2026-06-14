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
  /** Brick depth (wall thickness for a single-leaf wall). Default 0.5. */
  brickDepth?: number;
  /** Mortar gap between adjacent bricks. Default 0.025. */
  mortarGap?: number;

  // ── Castle layout (in bricks / courses) ──
  /** Number of stretchers along each curtain-wall span (between towers). Default 14. */
  wallLengthBricks?: number;
  /** Curtain-wall height in courses. Default 9. */
  wallCourses?: number;
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
  brickDepth: 0.5,
  mortarGap: 0.025,
  wallLengthBricks: 18,
  wallCourses: 11,
  towerSideBricks: 5,
  towerCourses: 18,
  keepSideBricks: 8,
  keepCourses: 24,
  gateWidthBricks: 4,
  gateHeightCourses: 5,
  battlements: true,
  chunksPerBrick: 2,
  bondMode: 'auto',
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
  // Bond contact tolerance. Irregular Voronoi faces across the mortar gap only
  // bond reliably when the 'average' bonder inflates each chunk enough to
  // overlap its neighbour — so this is several × the mortar gap. Too small and
  // the walls fragment into many loosely-stacked clusters that jitter on settle;
  // this value keeps each structure essentially monolithic.
  const maxSeparation = options.maxSeparation ?? gap * 7;

  // Archetype library (cached Voronoi). "stretcher" = standard brick;
  // "header"/half = end filler for running bond; "ashlar" = big keep block;
  // "merlon" = short battlement tooth; "voussoir" = gate-arch wedge.
  const stretcher = makeArchetype({ x: L, y: H, z: D }, opt.chunksPerBrick, options.pinata);
  const half = makeArchetype({ x: (L - gap) * 0.5, y: H, z: D }, Math.max(1, opt.chunksPerBrick - 1), options.pinata);
  const ashlar = makeArchetype({ x: L * 1.1, y: H, z: D * 1.2 }, opt.chunksPerBrick, options.pinata);
  const merlon = makeArchetype({ x: L, y: H * 1.1, z: D }, Math.max(1, opt.chunksPerBrick - 1), options.pinata);
  const voussoir = makeArchetype({ x: L * 0.7, y: H, z: D }, Math.max(1, opt.chunksPerBrick - 1), options.pinata);

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
  // a gatehouse on the south (−Z) wall, and a keep in the centre.
  const span = opt.wallLengthBricks * pitchX;            // clear wall span between towers
  const towerSpan = opt.towerSideBricks * pitchX;        // tower footprint side
  const halfFoot = span * 0.5 + towerSpan * 0.5;         // distance origin→tower centre
  let structureId = 0;

  // Colours (warm stone, varied per structure kind).
  const colWall = 0x9a8f7d;
  const colTower = 0x877c69;
  const colGate = 0xb0a48d;
  const colKeep = 0x726a5a;
  const colMerlon = 0xa49a87;

  // Foundation top sits at y=0; bricks start at y=0.
  const baseY = 0;

  // ── Four curtain walls ──
  // South wall carries the gatehouse gap; the others are plain running-bond
  // walls. Each wall's mid-line is collinear with its two corner-tower centres
  // (±halfFoot), and the run (length `span`, centred on the origin) butts the
  // towers' inner faces — so the wall ends do NOT overlap the towers (an overlap
  // there puts two separate bodies inside each other → a settle "pop").
  const sideCenter = halfFoot; // |coordinate| of each wall's mid-line == tower centre

  placeWallRun('south', -sideCenter, true);
  placeWallRun('north', +sideCenter, false);
  placeWallRun('west', -sideCenter, false);
  placeWallRun('east', +sideCenter, false);

  function placeWallRun(side: 'north' | 'south' | 'east' | 'west', lineConst: number, withGate: boolean): void {
    const sid = structureId++;
    const alongX = side === 'north' || side === 'south';
    const rotY = alongX ? 0 : Math.PI * 0.5;
    const n = opt.wallLengthBricks;
    const startAlong = -((n - 1) * pitchX) * 0.5; // centre the run on origin
    const gateHalf = (opt.gateWidthBricks * pitchX) * 0.5;
    for (let c = 0; c < opt.wallCourses; c++) {
      const y = baseY + H * 0.5 + c * pitchY;
      const isSupport = c === 0; // first course is anchored footing
      const odd = c % 2 === 1;
      // Running bond: odd courses shift half a pitch and carry one fewer brick
      // (the half-pitch end gaps are tucked behind the corner towers).
      const count = odd ? n - 1 : n;
      const shift = odd ? pitchX * 0.5 : 0;
      for (let i = 0; i < count; i++) {
        const along = startAlong + i * pitchX + shift;
        // Gate opening: skip bricks in the central gap for the lower courses.
        if (withGate && c < opt.gateHeightCourses && Math.abs(along) < gateHalf) continue;
        const center = wallPoint(side, lineConst, along, y);
        placeBrick(stretcher, center, rotY, sid, 'wall', colWall, isSupport);
      }
    }
    if (withGate) buildGateArch(sid, side, lineConst, rotY, gateHalf);
    if (opt.battlements) buildBattlement(sid, side, lineConst, rotY, n, startAlong);
  }

  /** Map a position along a wall side to a world point. */
  function wallPoint(side: 'north' | 'south' | 'east' | 'west', lineConst: number, along: number, y: number): Vec3 {
    switch (side) {
      case 'south': return { x: along, y, z: lineConst };
      case 'north': return { x: along, y, z: lineConst };
      case 'west': return { x: lineConst, y, z: along };
      case 'east': return { x: lineConst, y, z: along };
    }
  }

  /** Semicircular-ish arch of voussoir wedges over the gate opening. */
  function buildGateArch(sid: number, side: 'north' | 'south' | 'east' | 'west', lineConst: number, rotY: number, gateHalf: number): void {
    const springY = baseY + opt.gateHeightCourses * pitchY; // arch springs here
    const radius = gateHalf + L * 0.2;
    const count = Math.max(5, Math.round(opt.gateWidthBricks * 1.6) | 1); // odd → keystone in middle
    for (let k = 0; k < count; k++) {
      const t = count > 1 ? k / (count - 1) : 0.5;
      const ang = Math.PI * (1 - t); // PI → 0 (left springer to right springer)
      const along = Math.cos(ang) * radius;
      const y = springY + Math.sin(ang) * radius * 0.6 + H * 0.5;
      // Wedge faces radially; rotate the brick so its long axis follows the arch.
      const tilt = rotY + (ang - Math.PI * 0.5);
      const center = wallPoint(side, lineConst, along, y);
      placeBrick(voussoir, center, tilt, sid, 'gatehouse', colGate, false);
    }
    // Two gate piers (thicken the jambs) for a proper gatehouse read.
    for (const s of [-1, 1]) {
      for (let c = 0; c < opt.gateHeightCourses; c++) {
        const y = baseY + H * 0.5 + c * pitchY;
        const along = s * (gateHalf + (L - gap) * 0.25);
        const center = wallPoint(side, lineConst, along, y);
        placeBrick(half, center, rotY, sid, 'gatehouse', colGate, c === 0);
      }
    }
  }

  /** Alternating merlons / crenels along a wall top. */
  function buildBattlement(sid: number, side: 'north' | 'south' | 'east' | 'west', lineConst: number, rotY: number, n: number, startAlong: number): void {
    const y = baseY + H * 0.5 + opt.wallCourses * pitchY;
    for (let i = 0; i < n; i++) {
      if (i % 2 === 1) continue; // crenel (gap)
      const along = startAlong + i * pitchX;
      const center = wallPoint(side, lineConst, along, y);
      placeBrick(merlon, center, rotY, sid, 'battlement', colMerlon, false);
    }
  }

  // ── Four corner towers ──
  const towerCenters: Array<{ x: number; z: number }> = [
    { x: -sideCenter, z: -sideCenter },
    { x: +sideCenter, z: -sideCenter },
    { x: -sideCenter, z: +sideCenter },
    { x: +sideCenter, z: +sideCenter },
  ];
  for (const tc of towerCenters) {
    buildSquareTower(tc.x, tc.z, opt.towerSideBricks, opt.towerCourses, 'tower', colTower, stretcher);
  }

  // ── Central keep ──
  buildSquareTower(0, 0, opt.keepSideBricks, opt.keepCourses, 'keep', colKeep, ashlar);

  /**
   * Hollow square tower as a stack of running-bond rings. Corners interlock by
   * alternating which pair of walls runs "full" each course (a quoin pattern),
   * which also avoids double-placing corner bricks.
   */
  function buildSquareTower(cx: number, cz: number, sideBricks: number, courses: number, kind: CastleStructureKind, color: number, arch: Archetype): void {
    const sid = structureId++;
    const aSize = arch.size;
    const sidePitch = aSize.x + gap;
    const halfSide = ((sideBricks - 1) * sidePitch) * 0.5;
    const extent = halfSide + sidePitch * 0.5; // outer face offset for the perpendicular runs
    for (let c = 0; c < courses; c++) {
      const y = baseY + aSize.y * 0.5 + c * (aSize.y + gap);
      const isSupport = c === 0;
      const fbFull = c % 2 === 0; // even course: front/back full; odd: left/right full
      // Front (−Z) and back (+Z) runs (vary along X).
      for (const zc of [-extent, +extent]) {
        const lo = fbFull ? 0 : 1;
        const hi = fbFull ? sideBricks : sideBricks - 1;
        for (let i = lo; i < hi; i++) {
          const x = cx - halfSide + i * sidePitch;
          placeBrick(arch, { x, y, z: cz + zc }, 0, sid, kind, color, isSupport);
        }
      }
      // Left (−X) and right (+X) runs (vary along Z).
      for (const xc of [-extent, +extent]) {
        const lo = fbFull ? 1 : 0;
        const hi = fbFull ? sideBricks - 1 : sideBricks;
        for (let j = lo; j < hi; j++) {
          const z = cz - halfSide + j * sidePitch;
          placeBrick(arch, { x: cx + xc, y, z }, Math.PI * 0.5, sid, kind, color, isSupport);
        }
      }
    }
    // Battlements on the tower top.
    if (opt.battlements) {
      const y = baseY + aSize.y * 0.5 + courses * (aSize.y + gap);
      for (let i = 0; i < sideBricks; i++) {
        if (i % 2 === 1) continue;
        const x = cx - halfSide + i * sidePitch;
        placeBrick(merlon, { x, y, z: cz - extent }, 0, sid, 'battlement', colMerlon, false);
        placeBrick(merlon, { x, y, z: cz + extent }, 0, sid, 'battlement', colMerlon, false);
      }
      for (let j = 0; j < sideBricks; j++) {
        if (j % 2 === 1) continue;
        const z = cz - halfSide + j * sidePitch;
        placeBrick(merlon, { x: cx - extent, y, z }, Math.PI * 0.5, sid, 'battlement', colMerlon, false);
        placeBrick(merlon, { x: cx + extent, y, z }, Math.PI * 0.5, sid, 'battlement', colMerlon, false);
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
    autoBondingOptions: { mode: 'average' as const, maxSeparation, label: 'BrickCastle' },
    rapier: options.rapier,
  };

  const scenario = opt.bondMode === 'auto'
    ? await buildScenarioFromFragmentsAsync(allFragments, scenarioOptions)
    : buildScenarioFromFragments(allFragments, scenarioOptions);

  // ── Apply the three-tier strength hierarchy ──
  const isFoundation = (i: number) => kindByNode[i] === 'foundation';
  const tierOf = (b: ScenarioBond): CastleBondTier => {
    const a = b.node0, c = b.node1;
    if (isFoundation(a) || isFoundation(c)) return CastleBondTier.Anchor;
    if (brickIdByNode[a] === brickIdByNode[c]) return CastleBondTier.IntraBrick;
    if (structureIdByNode[a] === structureIdByNode[c]) return CastleBondTier.Mortar;
    return CastleBondTier.InterStructure;
  };
  const tierMult = [mult.anchor, mult.intraBrick, mult.mortar, mult.interStructure];
  const bondTiers = new Uint8Array(scenario.bonds.length);
  scenario.bonds = scenario.bonds.map((b, i) => {
    const tier = tierOf(b);
    bondTiers[i] = tier;
    return { ...b, area: Math.max(b.area * tierMult[tier], 1e-8) };
  });

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
