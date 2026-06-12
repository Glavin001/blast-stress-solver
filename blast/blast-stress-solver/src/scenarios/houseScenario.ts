/**
 * Single-storey wood-framed HOUSE scenario — a realistic, low-poly, walkable home that
 * battle-tests the stress solver on a genuinely heterogeneous, *structurally realistic*
 * body (rather than a uniform box that floats when you knock its walls out).
 *
 * Structural model (this is the important part — it gives a real load path):
 *
 *   - A static FOUNDATION (mass 0) anchors a FLOOR slab.
 *   - A wood POST-AND-BEAM FRAME is the load path: vertical posts (corner + intermediate
 *     wall posts to the eave, plus interior "king posts" that rise to the ridge), tied by
 *     a top-plate ring BEAM at the eave and a RIDGE beam at the apex. Posts → beams → roof
 *     and posts → floor → foundation carry all the weight.
 *   - The pitched gabled ROOF rests on the ridge beam (held by the king posts + gable
 *     ends) and on the top-plate (held by the wall posts). It is NOT a self-supporting
 *     rigid plate: roof↔roof bonds are only moderate, so an unsupported span sags and
 *     fails instead of floating — knock out the posts under it and that bay collapses.
 *   - DRYWALL wall panels are non-structural infill hung weakly on the frame between the
 *     posts (with door/window openings). Blowing them out does NOT bring the house down;
 *     they carry almost no load. This is the cosmetic-vs-structural distinction.
 *   - FURNITURE (table, chairs, counter, shelves) is part of the body but barely attached
 *     to the floor, so any contact frees a piece; shelves cling a bit to their wall.
 *
 * Every chunk is a single convex box → the runtime gives it a cheap cuboid / convex-hull
 * collider (never a concave trimesh); multi-part pieces are a convex decomposition. With
 * the async builder + `fracture`, selected parts are Voronoi-shattered (three-pinata) into
 * irregular shards instead of boxes, for nicer-looking breakage.
 *
 * Material strength is expressed through bond *area* (stress = impulse / area), since the
 * solver's stress limits are global — see {@link makeHouseBondMultiplier}.
 */
import * as THREE from 'three';
import type { ScenarioDesc, ScenarioBond } from '../rapier/types';
import type { FragmentInfo, FragmentType } from '../three/fracture';
import {
  applyBondStrengthMultipliers,
  buildRotatedBoxGridFragments,
  subdivideBoxFragments,
  type BondStrengthMultiplierFn,
} from '../three/fractureBuilders';
import {
  buildScenarioFromFragments,
  buildScenarioFromFragmentsAsync,
} from '../three/scenarioFromFragments';
import { ensurePinataLoaded, fractureGeometry, type PinataModule } from '../three/pinataFracture';

// ── Material densities (kg/m^3) ─────────────────────────────────────────────
export const HOUSE_WOOD_DENSITY = 600; // framing lumber (posts / beams)
export const HOUSE_DRYWALL_DENSITY = 700; // drywall infill panels
export const HOUSE_ROOF_DENSITY = 560; // sheathing + shingles (heavy enough to want to fall)
export const HOUSE_FOUNDATION_DENSITY = 2200; // concrete slab/footing
export const HOUSE_FURNITURE_DENSITY = 480; // light wood furniture

// ── Realistic-but-distinct material palette (hex). Structural defaults live here so the
//    demo and any consumer share one source of truth; furniture carries per-piece accents.
export const HOUSE_PALETTE: Record<string, number> = {
  foundation: 0x8d8a86, // concrete
  floor: 0xc2a878, // wood floor
  wall: 0xe8e2d4, // off-white drywall (non-structural)
  beam: 0x8b5a2b, // wood frame (top plate / ridge)
  column: 0x7a4a22, // wood posts (structural)
  roof: 0x77503c, // warm shingle brown
  furniture: 0x9c6b3f,
  shelf: 0x8b5a2b,
};

/** Which parts get Voronoi-shattered (async builder only). */
export type HouseFractureMode = 'none' | 'walls' | 'wallsRoof' | 'all';

export type HouseOptions = {
  /** Footprint width along X (m). Default 10. */
  width?: number;
  /** Footprint depth along Z (m). Default 8. */
  depth?: number;
  /** Eave (wall) height from floor to top plate (m). Default 2.6. */
  wallHeight?: number;
  /** Exterior wall (drywall) thickness (m). Default 0.15. */
  wallThickness?: number;
  /** Interior partition thickness (m). Default 0.1. */
  interiorWallThickness?: number;
  /** Floor slab thickness (m). Default 0.18. */
  floorThickness?: number;
  /** Static foundation thickness (m). Default 0.4. */
  foundationThickness?: number;
  /** Top-plate / ring-beam thickness (m). Default 0.14. */
  plateThickness?: number;
  /** Structural post cross-section (m). Default 0.18. */
  postSize?: number;
  /** Ridge rise above the eave (m). Default 1.6. */
  roofRise?: number;
  /** Roof eave overhang past the wall (m). Default 0.45. */
  eaveOverhang?: number;
  /** Roof panel (sheathing) thickness (m). Default 0.12. */
  roofThickness?: number;
  /** Approx. wall/roof chunk edge length (m) — smaller = finer. Default 0.62. */
  panelCell?: number;
  /** Vertical wall courses. Default 5. */
  wallCourses?: number;
  /** Number of interior "king posts" under the ridge (incl. the gable ends). Default 5. */
  ridgePosts?: number;
  /** Include interior furniture. Default true. */
  furniture?: boolean;
  /** Bond detection: 'proximity' (pure-JS, default) or 'auto' (WASM, async builder). */
  bondMode?: 'proximity' | 'auto';
  /** Voronoi-shatter selected parts instead of boxes (async builder only). Default 'none'. */
  fracture?: HouseFractureMode;
  /** Voronoi shards per box when fracturing — higher = finer/more detailed. Default 3. */
  fragmentsPerPiece?: number;
  /**
   * Base wall/roof cell size (m) used when fracturing, before shattering each cell into
   * `fragmentsPerPiece` shards. Smaller = finer (more, smaller shards) but more chunks /
   * contacts to simulate. Default 1.05 (coarsen-then-shatter keeps the count sane). Drop it
   * toward `panelCell` (~0.6) for much finer fracturing at a performance cost.
   */
  fractureCellSize?: number;
  /** Pre-imported three-pinata module (required for `fracture` in browser ESM). */
  pinata?: PinataModule;
  /** Override individual bond-strength multipliers. */
  multipliers?: Partial<HouseMultipliers>;
};

export const DEFAULT_HOUSE_OPTIONS: Required<
  Omit<HouseOptions, 'multipliers' | 'bondMode' | 'fracture' | 'fragmentsPerPiece' | 'fractureCellSize' | 'pinata'>
> & { bondMode: 'proximity' | 'auto' } = {
  width: 10,
  depth: 8,
  wallHeight: 2.6,
  wallThickness: 0.15,
  interiorWallThickness: 0.1,
  floorThickness: 0.18,
  foundationThickness: 0.4,
  plateThickness: 0.14,
  postSize: 0.18,
  roofRise: 1.6,
  eaveOverhang: 0.45,
  roofThickness: 0.12,
  panelCell: 0.62,
  wallCourses: 5,
  ridgePosts: 5,
  furniture: true,
  bondMode: 'proximity',
};

// ── Bond-strength table (multipliers on bond AREA → effective strength) ─────
//
// The structural load path (foundation → post → beam → roof, and post → floor) is STRONG;
// drywall is non-structural and hangs weakly on the frame; the roof is carried by the
// beams/posts and only moderately bonded to itself; furniture barely rests on the floor.
export type HouseMultipliers = {
  // Anchor + frame (structural — the load path that holds the house up).
  foundationColumn: number;
  foundationFloor: number;
  foundationOther: number;
  floorColumn: number;
  floorFloor: number;
  floorBeam: number;
  frameFrame: number; // post↔beam, beam↔beam, post↔post
  // Roof (carried by the frame; only moderately self-bonded).
  roofRoof: number;
  roofBeam: number;
  roofColumn: number;
  roofWall: number;
  // Drywall (non-structural cosmetic infill — weak everywhere).
  wallWall: number;
  wallColumn: number;
  wallBeam: number;
  wallFloor: number;
  wallFoundation: number;
  // Contents.
  shelfWall: number;
  shelfOther: number;
  furnitureFloor: number;
  furnitureFurniture: number;
  furnitureOther: number;
};

export const DEFAULT_HOUSE_MULTIPLIERS: HouseMultipliers = {
  foundationColumn: 28, // posts anchored to the ground — strongest joint
  foundationFloor: 30,
  foundationOther: 12,
  floorColumn: 18, // post base on the slab
  floorFloor: 10,
  floorBeam: 6,
  frameFrame: 16, // post↔beam / beam↔beam — the stiff wood frame
  roofRoof: 1.0, // sheathing continuity: low, so the roof can't span/cantilever as a rigid plate
  roofBeam: 6, // ridge / top plate carry the roof (the structural load path)
  roofColumn: 6, // king post directly under the ridge
  roofWall: 0.35, // roof rests lightly on the gable skin — which is too weak to carry it
  wallWall: 0.6, // drywall panel-to-panel continuity (weak, non-structural)
  wallColumn: 0.35, // drywall hung on a post
  wallBeam: 0.3, // drywall hung on the plate
  wallFloor: 1.0, // drywall bottom plate — holds the panel's own weight, not a roof
  wallFoundation: 1.0,
  shelfWall: 3.0, // shelves cling to their wall
  shelfOther: 2.0,
  furnitureFloor: 0.02, // barely resting — any contact frees & shoves the piece
  furnitureFurniture: 3.0, // holds a chair/table together until hit
  furnitureOther: 0.3,
};

const isFrame = (t?: FragmentType) => t === 'beam' || t === 'column';

/**
 * Build the house bond-strength multiplier. Resolve contents → drywall → roof → frame, so
 * a "drywall ↔ anything" joint always stays weak (non-structural) and a "roof ↔ frame"
 * joint is the structural one that carries the roof.
 */
export function makeHouseBondMultiplier(
  overrides?: Partial<HouseMultipliers>,
): BondStrengthMultiplierFn {
  const M = { ...DEFAULT_HOUSE_MULTIPLIERS, ...overrides };
  return (t0, t1) => {
    const pair = (a: FragmentType, b: FragmentType) =>
      (t0 === a && t1 === b) || (t0 === b && t1 === a);
    const has = (a: FragmentType) => t0 === a || t1 === a;

    // Contents.
    if (has('furniture') || has('shelf')) {
      if (pair('furniture', 'floor')) return M.furnitureFloor;
      if (t0 === 'furniture' && t1 === 'furniture') return M.furnitureFurniture;
      if (has('shelf')) return has('wall') ? M.shelfWall : M.shelfOther;
      return M.furnitureOther;
    }

    // Drywall infill is non-structural — every drywall joint is weak.
    if (has('wall')) {
      if (t0 === 'wall' && t1 === 'wall') return M.wallWall;
      if (has('roof')) return M.roofWall;
      if (has('column')) return M.wallColumn;
      if (has('beam')) return M.wallBeam;
      if (has('floor')) return M.wallFloor;
      if (has('foundation')) return M.wallFoundation;
      return M.wallWall;
    }

    // Roof skin (carried by the frame).
    if (has('roof')) {
      if (t0 === 'roof' && t1 === 'roof') return M.roofRoof;
      if (has('beam')) return M.roofBeam;
      if (has('column')) return M.roofColumn;
      return M.roofRoof;
    }

    // Structural load path.
    if (pair('foundation', 'column')) return M.foundationColumn;
    if (pair('foundation', 'floor')) return M.foundationFloor;
    if (has('foundation')) return M.foundationOther;
    if (pair('floor', 'column')) return M.floorColumn;
    if (t0 === 'floor' && t1 === 'floor') return M.floorFloor;
    if (has('floor')) return M.floorBeam;
    if (isFrame(t0) && isFrame(t1)) return M.frameFrame;

    return 1.0;
  };
}

export const HOUSE_BOND_MULTIPLIERS: BondStrengthMultiplierFn = makeHouseBondMultiplier();

// ── Geometry assembly ───────────────────────────────────────────────────────

type Vec3 = { x: number; y: number; z: number };
type WallRunOpening = { center: number; width: number; sill: number; height: number };

type MergedOptions = typeof DEFAULT_HOUSE_OPTIONS;
type CollectedHouse = {
  fragments: FragmentInfo[];
  nodeColors: (number | undefined)[];
  dims: Record<string, number>;
};

function overlaps1D(aMin: number, aMax: number, bMin: number, bMax: number): boolean {
  return aMin < bMax && bMin < aMax;
}

function linspace(min: number, max: number, n: number): number[] {
  if (n <= 1) return [(min + max) * 0.5];
  const step = (max - min) / (n - 1);
  return Array.from({ length: n }, (_v, i) => min + step * i);
}

/**
 * Build a vertical wall run (drywall infill) as a grid of box chunks, skipping any cell
 * overlapping a door/window opening. The grid naturally leaves a header course above each
 * opening and an apron below a window.
 */
function buildWallRun(opts: {
  axis: 'x' | 'z';
  fixed: number;
  runCenter: number;
  runLength: number;
  baseY: number;
  height: number;
  thickness: number;
  density: number;
  panelCell: number;
  courses: number;
  openings?: WallRunOpening[];
}): FragmentInfo[] {
  const { axis, fixed, runCenter, runLength, baseY, height, thickness, density } = opts;
  const cols = Math.max(1, Math.round(runLength / opts.panelCell));
  const rows = Math.max(1, opts.courses);
  const cellU = runLength / cols;
  const cellV = height / rows;
  const uMin = runCenter - runLength * 0.5;
  const openings = opts.openings ?? [];

  const out: FragmentInfo[] = [];
  for (let iu = 0; iu < cols; iu++) {
    const cu = uMin + cellU * (iu + 0.5);
    for (let iv = 0; iv < rows; iv++) {
      const cv = baseY + cellV * (iv + 0.5);
      const skip = openings.some((o) =>
        overlaps1D(cu - cellU * 0.5, cu + cellU * 0.5, o.center - o.width * 0.5, o.center + o.width * 0.5) &&
        overlaps1D(cv - cellV * 0.5, cv + cellV * 0.5, baseY + o.sill, baseY + o.sill + o.height),
      );
      if (skip) continue;

      const worldPosition: Vec3 =
        axis === 'x' ? { x: cu, y: cv, z: fixed } : { x: fixed, y: cv, z: cu };
      const size: Vec3 =
        axis === 'x' ? { x: cellU, y: cellV, z: thickness } : { x: thickness, y: cellV, z: cellU };
      out.push({
        worldPosition,
        halfExtents: { x: size.x * 0.5, y: size.y * 0.5, z: size.z * 0.5 },
        geometry: new THREE.BoxGeometry(size.x, size.y, size.z),
        isSupport: false,
        fragmentType: 'wall',
        density,
      });
    }
  }
  return out;
}

/** A single structural post (column) from baseY up to topY, subdivided vertically. */
function buildPost(cx: number, cz: number, baseY: number, topY: number, size: number, panelCell: number): FragmentInfo[] {
  const h = topY - baseY;
  return subdivideBoxFragments({
    center: { x: cx, y: (baseY + topY) * 0.5, z: cz },
    size: { x: size, y: h, z: size },
    divisions: { x: 1, y: Math.max(1, Math.round(h / Math.max(panelCell, 0.5))), z: 1 },
    fragmentType: 'column',
    density: HOUSE_WOOD_DENSITY,
  });
}

function collectHouseFragments(o: MergedOptions): CollectedHouse {
  const {
    width, depth, wallHeight, wallThickness, interiorWallThickness, floorThickness,
    foundationThickness, plateThickness, postSize, roofRise, eaveOverhang, roofThickness,
    panelCell, wallCourses, ridgePosts, furniture,
  } = o;

  const fragments: FragmentInfo[] = [];
  const nodeColors: (number | undefined)[] = [];
  const hw = width * 0.5;
  const hd = depth * 0.5;

  const push = (fs: FragmentInfo[], color?: number) => {
    for (const f of fs) { fragments.push(f); nodeColors.push(color); }
  };

  // Datum: floor top = 0 (lifted onto the runtime ground at the end).
  const floorTopY = 0;
  const floorBottomY = floorTopY - floorThickness;
  const foundationTopY = floorBottomY;
  const foundationBottomY = foundationTopY - foundationThickness;
  const eaveY = wallHeight;

  // Roof geometry — derive so the underside meets the top plate at the eave (small bite for
  // a real contact bond) and rises to the ridge; everything tucks UNDER the roof.
  const plateTopY = wallHeight + plateThickness;
  const eaveBite = 0.02;
  const roofCenterAtWall = plateTopY - eaveBite + roofThickness * 0.5;
  const ridgeCenterY = roofCenterAtWall + roofRise;
  const ridgeUndersideY = ridgeCenterY - roofThickness * 0.5;
  const slope = Math.atan2(roofRise, hd);
  const slopeRatio = roofRise / hd; // dy per dz of the roof centerline

  // ── Foundation (static) + floor slab ──────────────────────────────────────
  push(
    subdivideBoxFragments({
      center: { x: 0, y: (foundationTopY + foundationBottomY) * 0.5, z: 0 },
      size: { x: width, y: foundationThickness, z: depth },
      divisions: { x: Math.max(3, Math.round(width / 2.5)), y: 1, z: Math.max(2, Math.round(depth / 2.5)) },
      fragmentType: 'foundation', isSupport: true,
    }),
  );
  push(
    subdivideBoxFragments({
      center: { x: 0, y: (floorTopY + floorBottomY) * 0.5, z: 0 },
      size: { x: width, y: floorThickness, z: depth },
      divisions: { x: Math.max(3, Math.round(width / 1.8)), y: 1, z: Math.max(3, Math.round(depth / 1.8)) },
      fragmentType: 'floor', density: HOUSE_FOUNDATION_DENSITY,
    }),
  );

  // ── Structural frame: posts sit JUST INSIDE the walls (touching the inner face), so the
  // frame never interpenetrates the drywall. ───────────────────────────────
  const postFrontZ = -hd + wallThickness + postSize * 0.5;
  const postBackZ = hd - wallThickness - postSize * 0.5;
  const postLeftX = -hw + wallThickness + postSize * 0.5;
  const postRightX = hw - wallThickness - postSize * 0.5;
  const plateStrip = wallThickness + postSize; // top-plate width: covers the wall + inset post

  // Ridge-beam geometry (the king posts stop at its underside; its top sits at the roof apex).
  const roofLenX = width + 0.6;
  const ridgeBeamH = 0.16;
  const ridgeBeamZ = 0.1;
  const ridgeBeamCenterY = ridgeUndersideY - ridgeBeamH * 0.5; // top ≈ roof underside at the ridge
  const kingTopY = ridgeUndersideY - ridgeBeamH; // king posts stop at the ridge-beam underside

  // Eave posts: 4 corners + intermediates along each exterior wall (floor → eave).
  for (const x of [postLeftX, postRightX]) for (const z of [postFrontZ, postBackZ]) push(buildPost(x, z, floorTopY, eaveY, postSize, panelCell), HOUSE_PALETTE.column);
  const interXs = [-width * 0.25, width * 0.25];
  const interZs = [-depth * 0.25, depth * 0.25];
  for (const x of interXs) { push(buildPost(x, postFrontZ, floorTopY, eaveY, postSize, panelCell), HOUSE_PALETTE.column); push(buildPost(x, postBackZ, floorTopY, eaveY, postSize, panelCell), HOUSE_PALETTE.column); }
  for (const z of interZs) { push(buildPost(postLeftX, z, floorTopY, eaveY, postSize, panelCell), HOUSE_PALETTE.column); push(buildPost(postRightX, z, floorTopY, eaveY, postSize, panelCell), HOUSE_PALETTE.column); }

  // Ridge "king posts" on the centerline (z=0), floor → ridge-beam underside. The partition
  // drywall is offset just behind them (below) so they touch instead of interpenetrate.
  const partitionDoor = { center: -1.2, width: 1.0, height: 2.1 };
  const kingHalfRange = hw - plateStrip - postSize * 0.5; // keep king posts clear of the side plates
  const kingXs = linspace(-kingHalfRange, kingHalfRange, Math.max(2, Math.round(ridgePosts))).filter(
    (x) => Math.abs(x - partitionDoor.center) > partitionDoor.width * 0.5 + postSize,
  );
  for (const x of kingXs) push(buildPost(x, 0, floorTopY, kingTopY, postSize, panelCell), HOUSE_PALETTE.column);

  // ── Beams: top-plate ring + ridge beam. Plates cover the wall+post strip and butt at the
  // corners (front/back plates run full width; side plates fit strictly between them). ──
  const plateY = plateTopY - plateThickness * 0.5;
  for (const sign of [-1, 1] as const) {
    push(subdivideBoxFragments({
      center: { x: 0, y: plateY, z: sign * (hd - plateStrip * 0.5) },
      size: { x: width, y: plateThickness, z: plateStrip },
      divisions: { x: Math.max(2, Math.round(width / panelCell)), y: 1, z: 1 },
      fragmentType: 'beam', density: HOUSE_WOOD_DENSITY,
    }), HOUSE_PALETTE.beam);
  }
  const sidePlateLen = depth - 2 * plateStrip;
  for (const sign of [-1, 1] as const) {
    push(subdivideBoxFragments({
      center: { x: sign * (hw - plateStrip * 0.5), y: plateY, z: 0 },
      size: { x: plateStrip, y: plateThickness, z: sidePlateLen },
      divisions: { x: 1, y: 1, z: Math.max(2, Math.round(sidePlateLen / panelCell)) },
      fragmentType: 'beam', density: HOUSE_WOOD_DENSITY,
    }), HOUSE_PALETTE.beam);
  }
  push(subdivideBoxFragments({
    center: { x: 0, y: ridgeBeamCenterY, z: 0 },
    size: { x: roofLenX, y: ridgeBeamH, z: ridgeBeamZ },
    divisions: { x: Math.max(2, Math.round(roofLenX / panelCell)), y: 1, z: 1 },
    fragmentType: 'beam', density: HOUSE_WOOD_DENSITY,
  }), HOUSE_PALETTE.beam);

  // ── Drywall infill walls (non-structural) with door + window openings ─────
  const drywall = HOUSE_DRYWALL_DENSITY;
  const dr = (fs: FragmentInfo[]) => push(fs, HOUSE_PALETTE.wall);
  // Front (−Z): centered front door + two flanking windows.
  dr(buildWallRun({ axis: 'x', fixed: -hd + wallThickness * 0.5, runCenter: 0, runLength: width, baseY: floorTopY, height: wallHeight, thickness: wallThickness, density: drywall, panelCell, courses: wallCourses, openings: [
    { center: 0, width: 1.2, sill: 0, height: 2.1 },
    { center: -hw * 0.6, width: 1.2, sill: 0.9, height: 1.2 },
    { center: hw * 0.6, width: 1.2, sill: 0.9, height: 1.2 },
  ] }));
  // Back (+Z): centered back door + a window.
  dr(buildWallRun({ axis: 'x', fixed: hd - wallThickness * 0.5, runCenter: 0, runLength: width, baseY: floorTopY, height: wallHeight, thickness: wallThickness, density: drywall, panelCell, courses: wallCourses, openings: [
    { center: 0, width: 1.0, sill: 0, height: 2.05 },
    { center: hw * 0.6, width: 1.3, sill: 0.9, height: 1.2 },
  ] }));
  // Sides (±X): a window each. Run strictly BETWEEN the front/back walls so the corners
  // (filled by the full-width front/back walls) don't double up.
  const sideWallLen = depth - 2 * wallThickness;
  for (const sideX of [-hw + wallThickness * 0.5, hw - wallThickness * 0.5]) {
    dr(buildWallRun({ axis: 'z', fixed: sideX, runCenter: 0, runLength: sideWallLen, baseY: floorTopY, height: wallHeight, thickness: wallThickness, density: drywall, panelCell, courses: wallCourses, openings: [
      { center: -hd * 0.4, width: 1.3, sill: 0.9, height: 1.2 },
    ] }));
  }
  // Interior partition: living room (front) | kitchen (back). Offset just behind the king
  // posts so they touch instead of interpenetrate; doorway in a clear bay between posts.
  const partitionZ = postSize * 0.5 + interiorWallThickness * 0.5;
  dr(buildWallRun({ axis: 'x', fixed: partitionZ, runCenter: 0, runLength: width - 2 * wallThickness, baseY: floorTopY, height: wallHeight, thickness: interiorWallThickness, density: drywall, panelCell, courses: wallCourses, openings: [
    { center: partitionDoor.center, width: partitionDoor.width, sill: 0, height: partitionDoor.height },
  ] }));

  // ── Gabled roof (two sloped planes; everything below fits UNDER it) ────────
  const slopeLen = (hd + eaveOverhang) / Math.cos(slope);
  const roofCols = Math.max(2, Math.round(roofLenX / panelCell));
  const roofRows = Math.max(2, Math.round(slopeLen / panelCell));
  const ridgePivot: Vec3 = { x: 0, y: ridgeUndersideY, z: 0 };
  for (const sign of [-1, 1] as const) {
    push(buildRotatedBoxGridFragments({
      center: { x: 0, y: ridgeUndersideY + roofThickness * 0.5, z: sign * slopeLen * 0.5 },
      size: { x: roofLenX, y: roofThickness, z: slopeLen },
      divisions: { x: roofCols, y: 1, z: roofRows },
      fragmentType: 'roof', density: HOUSE_ROOF_DENSITY,
      rotation: { axis: 'x', angle: sign * slope, pivot: ridgePivot },
    }), HOUSE_PALETTE.roof);
  }

  // ── Gable-end drywall triangles: ABOVE the top plate, each course sized so its top stays
  // UNDER the sloped roof (touching, never poking through). ──
  const gableCourses = 5;
  const gableBaseY = plateTopY;
  const courseH = (ridgeUndersideY - gableBaseY) / gableCourses;
  for (const sideX of [-hw + wallThickness * 0.5, hw - wallThickness * 0.5]) {
    for (let k = 0; k < gableCourses; k++) {
      const yBot = gableBaseY + k * courseH;
      const yTop = yBot + courseH;
      const halfZ = (ridgeUndersideY - yTop) / slopeRatio;
      if (halfZ < 0.12) continue;
      dr(subdivideBoxFragments({
        center: { x: sideX, y: (yBot + yTop) * 0.5, z: 0 },
        size: { x: wallThickness, y: courseH, z: 2 * halfZ },
        divisions: { x: 1, y: 1, z: Math.max(1, Math.round((2 * halfZ) / panelCell)) },
        fragmentType: 'wall', density: drywall,
      }));
    }
  }

  // ── Furniture ─────────────────────────────────────────────────────────────
  if (furniture) addFurniture(push, { hw, hd, floorTopY, wallThickness });

  // Lift the whole house so the dynamic floor rests ON the runtime ground (top at y=0).
  const liftY = floorThickness;
  for (const f of fragments) f.worldPosition = { ...f.worldPosition, y: f.worldPosition.y + liftY };

  return {
    fragments,
    nodeColors,
    dims: {
      width, depth, wallHeight,
      floorTopY: floorTopY + liftY,
      eaveY: eaveY + liftY,
      ridgeY: ridgeCenterY + liftY,
      foundationBottomY: foundationBottomY + liftY,
      plateTopY: plateTopY + liftY,
    },
  };
}

const FURNITURE_ACCENTS = [0xc0473b, 0x3b6fc0, 0x3ba35a, 0xd6a73a, 0x9b59b6, 0xe07b39];

function addFurniture(
  push: (fs: FragmentInfo[], color?: number) => void,
  g: { hw: number; hd: number; floorTopY: number; wallThickness: number },
) {
  const { hw, hd, floorTopY, wallThickness } = g;
  const box = (cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, type: FragmentType): FragmentInfo => ({
    worldPosition: { x: cx, y: cy, z: cz },
    halfExtents: { x: sx * 0.5, y: sy * 0.5, z: sz * 0.5 },
    geometry: new THREE.BoxGeometry(sx, sy, sz),
    isSupport: false, fragmentType: type, density: HOUSE_FURNITURE_DENSITY,
  });

  // Dining table + 4 chairs in the living room (front half, −Z).
  const tx = hw * 0.42;
  const tz = -hd * 0.45;
  const wood = 0x9c6b3f;
  push([box(tx, floorTopY + 0.74, tz, 1.3, 0.06, 0.85, 'furniture')], wood);
  for (const dx of [-0.55, 0.55]) for (const dz of [-0.36, 0.36]) push([box(tx + dx, floorTopY + 0.355, tz + dz, 0.07, 0.71, 0.07, 'furniture')], wood);
  const chairOffsets: Array<[number, number]> = [[-0.95, 0], [0.95, 0], [0, -0.7], [0, 0.7]];
  chairOffsets.forEach(([ox, oz], i) => {
    const accent = FURNITURE_ACCENTS[i % FURNITURE_ACCENTS.length];
    const cx = tx + ox, cz = tz + oz;
    push([box(cx, floorTopY + 0.46, cz, 0.42, 0.05, 0.42, 'furniture')], accent);
    push([box(cx, floorTopY + 0.72, cz - 0.18, 0.42, 0.5, 0.05, 'furniture')], accent);
    for (const lx of [-0.17, 0.17]) for (const lz of [-0.17, 0.17]) push([box(cx + lx, floorTopY + 0.225, cz + lz, 0.05, 0.45, 0.05, 'furniture')], accent);
  });

  // Kitchen counter (back half, +Z), placed in a clear bay so it doesn't sit on a wall post.
  const counterX = -hw * 0.72;
  const counterZ = hd - wallThickness - 0.45;
  push([box(counterX, floorTopY + 0.45, counterZ, 1.8, 0.9, 0.55, 'furniture')], 0xd9dde0);
  push([box(counterX, floorTopY + 0.92, counterZ, 1.9, 0.06, 0.6, 'furniture')], wood);
  // Upper wall cabinet (wall-mounted → 'shelf'), back flush against the back wall.
  push([box(counterX, floorTopY + 1.7, hd - wallThickness - 0.17, 1.6, 0.7, 0.34, 'shelf')], 0xb24c4c);

  // Wall shelves on the left living-room wall (back flush to the wall, clear of the posts).
  const shelfX = -hw + wallThickness + 0.13;
  for (const sy of [1.35, 1.8]) push([box(shelfX, floorTopY + sy, -1.0, 0.26, 0.04, 0.95, 'shelf')], 0x8b5a2b);
}

// ── Voronoi shatter (async path) ────────────────────────────────────────────

function fractureTypesFor(mode: HouseFractureMode): Set<FragmentType> {
  switch (mode) {
    case 'walls': return new Set<FragmentType>(['wall']);
    case 'wallsRoof': return new Set<FragmentType>(['wall', 'roof']);
    case 'all': return new Set<FragmentType>(['wall', 'roof', 'floor', 'beam', 'column', 'furniture', 'shelf']);
    default: return new Set<FragmentType>();
  }
}

/** Replace each box of a selected type with Voronoi shards (keeping type/density/color). */
function fractureSelected(
  fragments: FragmentInfo[],
  nodeColors: (number | undefined)[],
  types: Set<FragmentType>,
  perPiece: number,
  pinata?: PinataModule,
): { fragments: FragmentInfo[]; nodeColors: (number | undefined)[] } {
  const outF: FragmentInfo[] = [];
  const outC: (number | undefined)[] = [];
  for (let i = 0; i < fragments.length; i++) {
    const f = fragments[i];
    const t = f.fragmentType;
    if (!f.isSupport && t && types.has(t)) {
      try {
        const shards = fractureGeometry(f.geometry, {
          fragmentCount: Math.max(2, perPiece), voronoiMode: '3D',
          worldOffset: f.worldPosition, minHalfExtent: 0.07, pinata,
        });
        if (shards.length) {
          for (const s of shards) { outF.push({ ...s, isSupport: false, fragmentType: t, density: f.density }); outC.push(nodeColors[i]); }
          continue;
        }
      } catch { /* fall through to the original box */ }
    }
    outF.push(f); outC.push(nodeColors[i]);
  }
  return { fragments: outF, nodeColors: outC };
}

// ── Public builders ─────────────────────────────────────────────────────────

function finalize(
  scenario: ScenarioDesc, options: HouseOptions,
  fragments: FragmentInfo[], nodeColors: (number | undefined)[], dims: Record<string, number>,
): ScenarioDesc {
  const fragmentTypes = fragments.map((f) => f.fragmentType);
  scenario.bonds = applyBondStrengthMultipliers(
    scenario.bonds as ScenarioBond[], fragmentTypes, makeHouseBondMultiplier(options.multipliers),
  );
  scenario.parameters = { ...scenario.parameters, house: { ...dims, fragmentTypes, nodeColors } };
  return scenario;
}

/**
 * Build the house synchronously with the pure-JS proximity bonder (all-boxes). For Voronoi
 * fracturing or WASM auto-bonding use {@link buildHouseScenarioAsync}.
 */
export function buildHouseScenario(options: HouseOptions = {}): ScenarioDesc {
  const o = { ...DEFAULT_HOUSE_OPTIONS, ...options };
  const { fragments, nodeColors, dims } = collectHouseFragments(o);
  const scenario = buildScenarioFromFragments(fragments, { areaNormalization: 'none' });
  return finalize(scenario, options, fragments, nodeColors, dims);
}

/**
 * Async house builder. Adds `fracture` (Voronoi-shatter selected parts via three-pinata)
 * and `bondMode: 'auto'` (WASM triangle-overlap bonding, more robust for the irregular
 * shards and the sloped roof). Falls back to proximity bonds on any failure.
 */
export async function buildHouseScenarioAsync(options: HouseOptions = {}): Promise<ScenarioDesc> {
  const mode = options.fracture ?? 'none';
  const types = fractureTypesFor(mode);

  // When fracturing, build a COARSER base grid (bigger boxes, fewer courses) before
  // shattering each box into a few shards. Otherwise we multiply an already-fine grid and
  // the chunk/contact count explodes — and the per-frame *contact injection* (not the WASM
  // stress solve, which stays cheap) becomes the bottleneck. Coarse base × few shards keeps
  // the total piece count close to the unfractured house while still looking shattered.
  const coarse = types.size > 0
    ? {
        panelCell: options.fractureCellSize ?? 1.05,
        wallCourses: options.wallCourses ?? 3,
      }
    : {};
  const o = { ...DEFAULT_HOUSE_OPTIONS, ...options, ...coarse };

  const collected = collectHouseFragments(o);
  let { fragments, nodeColors } = collected;
  if (types.size > 0) {
    if (!options.pinata) await ensurePinataLoaded();
    ({ fragments, nodeColors } = fractureSelected(fragments, nodeColors, types, options.fragmentsPerPiece ?? 3, options.pinata));
  }

  // Fractured shards bond best with the WASM auto-bonder; plain boxes are fine with proximity.
  const bondMode = options.bondMode ?? (types.size > 0 ? 'auto' : 'proximity');
  const scenario = await buildScenarioFromFragmentsAsync(fragments, { bondMode, areaNormalization: 'none' });
  return finalize(scenario, options, fragments, nodeColors, collected.dims);
}
