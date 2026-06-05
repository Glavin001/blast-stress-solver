/**
 * Single-storey wood-framed HOUSE scenario — a realistic, low-poly, walkable home that
 * battle-tests the stress solver on a genuinely heterogeneous, civil-engineering-style
 * structure (rather than a uniform box).
 *
 * It is built as ONE body (one ScenarioDesc) from many composable box "chunks", but the
 * bonds between them are tuned per material so it fails *non-uniformly*:
 *
 *   - A static FOUNDATION (mass 0) anchors a FLOOR slab to the ground.
 *   - Wood-framed EXTERIOR WALLS with real door + window openings, an INTERIOR partition
 *     splitting a living room and kitchen, a ring of top-plate BEAMS, a few interior
 *     POSTS (columns) and ceiling tie-BEAMS — the load path floor → wall → plate → roof.
 *   - A pitched, gabled ROOF (two sloped planes + ridge beam + gable-end triangles) that
 *     is bonded *moderately* to the frame, so a hit caves the roof differently than it
 *     blows out a wall.
 *   - FURNITURE (table, chairs, kitchen counter, wall shelves) that is part of the same
 *     body but barely attached to the floor: the furniture↔floor bond is near-zero, so
 *     any contact frees a piece and shoves it; wall-mounted SHELVES are bonded more
 *     firmly to their wall.
 *
 * Every chunk is a single convex box, so the runtime gives it a cheap cuboid / convex-hull
 * collider (never a concave trimesh); multi-part pieces (a table = top + 4 legs) are a
 * manual convex decomposition bonded together — Rapier's recommended approach for complex
 * dynamic objects.
 *
 * Material strength is expressed through bond *area* (stress = impulse / area), since the
 * solver's stress limits are global — see {@link makeHouseBondMultiplier}. Mirrors the
 * high-rise composer (highRiseScenario.ts) end to end.
 */
import * as THREE from 'three';
import type { ScenarioDesc } from '../rapier/types';
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

// ── Material densities (kg/m^3) ─────────────────────────────────────────────
export const HOUSE_WOOD_DENSITY = 600; // framing lumber / posts / beams
export const HOUSE_DRYWALL_DENSITY = 700; // drywall / sheathed wall panels
export const HOUSE_ROOF_DENSITY = 420; // sheathing + shingles
export const HOUSE_FOUNDATION_DENSITY = 2200; // concrete slab/footing
export const HOUSE_FURNITURE_DENSITY = 480; // light wood furniture

// ── Realistic-but-distinct material palette (hex). Structural defaults live here so the
//    demo and any other consumer share one source of truth; furniture pieces carry their
//    own per-piece accent colors (see collectHouseFragments). ───────────────────────────
export const HOUSE_PALETTE: Record<string, number> = {
  foundation: 0x8d8a86, // concrete
  floor: 0xc2a878, // wood floor
  wall: 0xe8e2d4, // off-white drywall
  beam: 0x8b5a2b, // wood frame (top plate / ridge / ties)
  column: 0x7a4a22, // darker wood posts
  roof: 0x77503c, // warm shingle brown
  furniture: 0x9c6b3f,
  shelf: 0x8b5a2b,
};

export type HouseOptions = {
  /** Footprint width along X (m). Default 10. */
  width?: number;
  /** Footprint depth along Z (m). Default 8. */
  depth?: number;
  /** Eave (wall) height from floor to top plate (m). Default 2.7. */
  wallHeight?: number;
  /** Exterior wall thickness (m). Default 0.15. */
  wallThickness?: number;
  /** Interior partition thickness (m). Default 0.1. */
  interiorWallThickness?: number;
  /** Floor slab thickness (m). Default 0.18. */
  floorThickness?: number;
  /** Static foundation thickness (m). Default 0.4. */
  foundationThickness?: number;
  /** Top-plate / ring-beam thickness (m). Default 0.12. */
  plateThickness?: number;
  /** Ridge rise above the eave (m). Default 1.8. */
  roofRise?: number;
  /** Roof eave overhang past the wall (m). Default 0.45. */
  eaveOverhang?: number;
  /** Roof panel (sheathing) thickness (m). Default 0.12. */
  roofThickness?: number;
  /** Approx. wall/roof chunk edge length (m) — smaller = finer. Default 0.62. */
  panelCell?: number;
  /** Vertical wall courses. Default 5. */
  wallCourses?: number;
  /** Include interior furniture (table, chairs, counter, shelves). Default true. */
  furniture?: boolean;
  /** Bond detection: 'proximity' (pure-JS, default) or 'auto' (WASM, needs async builder). */
  bondMode?: 'proximity' | 'auto';
  /** Override individual bond-strength multipliers. */
  multipliers?: Partial<HouseMultipliers>;
};

export const DEFAULT_HOUSE_OPTIONS: Required<
  Omit<HouseOptions, 'multipliers' | 'bondMode'>
> & { bondMode: 'proximity' | 'auto' } = {
  width: 10,
  depth: 8,
  wallHeight: 2.7,
  wallThickness: 0.15,
  interiorWallThickness: 0.1,
  floorThickness: 0.18,
  foundationThickness: 0.4,
  plateThickness: 0.12,
  roofRise: 1.8,
  eaveOverhang: 0.45,
  roofThickness: 0.12,
  panelCell: 0.62,
  wallCourses: 5,
  furniture: true,
  bondMode: 'proximity',
};

// ── Bond-strength table (multipliers on bond AREA → effective strength) ─────
export type HouseMultipliers = {
  furnitureFloor: number;
  furnitureWall: number;
  furnitureFurniture: number;
  shelfWall: number;
  shelfOther: number;
  roofRoof: number;
  roofBeam: number;
  roofWall: number;
  foundationFloor: number;
  foundationFrame: number;
  floorFloor: number;
  floorWall: number;
  floorFrame: number;
  frameFrame: number;
  beamWall: number;
  wallWall: number;
};

export const DEFAULT_HOUSE_MULTIPLIERS: HouseMultipliers = {
  furnitureFloor: 0.02, // barely resting — any contact frees & shoves the piece
  furnitureWall: 0.5, // counter lightly against a wall
  furnitureFurniture: 3.0, // holds a chair/table together until hit
  shelfWall: 6.0, // wall-mounted shelves cling to their wall
  shelfOther: 3.0,
  roofRoof: 2.5, // sheathing plane continuity
  roofBeam: 1.5, // roof ↔ ridge/top plate — caves differently than a wall
  roofWall: 1.2, // roof ↔ gable wall
  foundationFloor: 30.0, // strongest: slab anchored to ground (no liftoff)
  foundationFrame: 20.0, // sill anchor of walls/posts to foundation
  floorFloor: 10.0, // slab diaphragm
  floorWall: 8.0, // bottom plate: wall sits on floor
  floorFrame: 14.0, // floor ↔ post/beam base
  frameFrame: 12.0, // wood frame joints (post↔beam, beam↔beam)
  beamWall: 8.0, // top plate ↔ studs
  wallWall: 4.0, // stud/sheathing baseline
};

const isFrame = (t?: FragmentType) => t === 'beam' || t === 'column';

/** Build the house bond-strength multiplier function (resolve weak/special pairs first). */
export function makeHouseBondMultiplier(
  overrides?: Partial<HouseMultipliers>,
): BondStrengthMultiplierFn {
  const M = { ...DEFAULT_HOUSE_MULTIPLIERS, ...overrides };
  return (t0, t1) => {
    const pair = (a: FragmentType, b: FragmentType) =>
      (t0 === a && t1 === b) || (t0 === b && t1 === a);
    const has = (a: FragmentType) => t0 === a || t1 === a;

    // Furniture & shelves first, so any "<something> ↔ furniture" stays weak/special.
    if (has('furniture') || has('shelf')) {
      if (pair('furniture', 'floor')) return M.furnitureFloor;
      if (t0 === 'furniture' && t1 === 'furniture') return M.furnitureFurniture;
      if (has('shelf')) return has('wall') ? M.shelfWall : M.shelfOther;
      return M.furnitureWall; // furniture ↔ wall / frame / roof: light attachment
    }

    // Roof bonds (moderate; weaker than the frame so the roof caves on its own).
    if (has('roof')) {
      if (t0 === 'roof' && t1 === 'roof') return M.roofRoof;
      if (isFrame(t0) || isFrame(t1)) return M.roofBeam;
      return M.roofWall;
    }

    // Base anchoring.
    if (pair('foundation', 'floor')) return M.foundationFloor;
    if (has('foundation')) return M.foundationFrame;

    // Wood frame load path.
    if (t0 === 'floor' && t1 === 'floor') return M.floorFloor;
    if (pair('floor', 'wall')) return M.floorWall;
    if (pair('floor', 'beam') || pair('floor', 'column')) return M.floorFrame;
    if (isFrame(t0) && isFrame(t1)) return M.frameFrame;
    if (pair('beam', 'wall') || pair('column', 'wall')) return M.beamWall;
    if (t0 === 'wall' && t1 === 'wall') return M.wallWall;

    return 1.0;
  };
}

export const HOUSE_BOND_MULTIPLIERS: BondStrengthMultiplierFn = makeHouseBondMultiplier();

// ── Geometry assembly ───────────────────────────────────────────────────────

type Vec3 = { x: number; y: number; z: number };
type WallRunOpening = {
  /** Center along the run (X for a front/back wall, Z for a side wall), in meters. */
  center: number;
  width: number;
  /** Sill height above the wall base (m); 0 for a door. */
  sill: number;
  height: number;
};

type CollectedHouse = {
  fragments: FragmentInfo[];
  /** Per-fragment accent color (hex) or undefined to let the palette pick by type. */
  nodeColors: (number | undefined)[];
  dims: Record<string, number>;
};

function overlaps1D(aMin: number, aMax: number, bMin: number, bMax: number): boolean {
  return aMin < bMax && bMin < aMax;
}

/**
 * Build a vertical wall run (along X or Z) as a grid of box chunks, skipping any cell that
 * overlaps a door/window opening. The grid naturally leaves a header course above each
 * opening and an apron below a window, so no separate lintel is needed. Remaining chunks
 * stay flush with their neighbors, so the proximity/auto bonder reconnects them around the
 * hole.
 */
function buildWallRun(opts: {
  axis: 'x' | 'z';
  /** Perpendicular world position: z for an x-run, x for a z-run. */
  fixed: number;
  /** Center along the run. */
  runCenter: number;
  runLength: number;
  baseY: number;
  height: number;
  thickness: number;
  fragmentType: FragmentType;
  density: number;
  panelCell: number;
  courses: number;
  openings?: WallRunOpening[];
}): FragmentInfo[] {
  const { axis, fixed, runCenter, runLength, baseY, height, thickness, fragmentType, density } =
    opts;
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
      // Skip cells that overlap any opening.
      const skip = openings.some((o) =>
        overlaps1D(cu - cellU * 0.5, cu + cellU * 0.5, o.center - o.width * 0.5, o.center + o.width * 0.5) &&
        overlaps1D(cv - cellV * 0.5, cv + cellV * 0.5, baseY + o.sill, baseY + o.sill + o.height),
      );
      if (skip) continue;

      const worldPosition: Vec3 =
        axis === 'x' ? { x: cu, y: cv, z: fixed } : { x: fixed, y: cv, z: cu };
      const size: Vec3 =
        axis === 'x'
          ? { x: cellU, y: cellV, z: thickness }
          : { x: thickness, y: cellV, z: cellU };
      out.push({
        worldPosition,
        halfExtents: { x: size.x * 0.5, y: size.y * 0.5, z: size.z * 0.5 },
        geometry: new THREE.BoxGeometry(size.x, size.y, size.z),
        isSupport: false,
        fragmentType,
        density,
      });
    }
  }
  return out;
}

function collectHouseFragments(o: Required<Omit<HouseOptions, 'multipliers' | 'bondMode'>>): CollectedHouse {
  const {
    width,
    depth,
    wallHeight,
    wallThickness,
    interiorWallThickness,
    floorThickness,
    foundationThickness,
    plateThickness,
    roofRise,
    eaveOverhang,
    roofThickness,
    panelCell,
    wallCourses,
    furniture,
  } = o;

  const fragments: FragmentInfo[] = [];
  const nodeColors: (number | undefined)[] = [];
  const hw = width * 0.5;
  const hd = depth * 0.5;

  const push = (fs: FragmentInfo[], color?: number) => {
    for (const f of fs) {
      fragments.push(f);
      nodeColors.push(color);
    }
  };

  // ── Foundation: static (mass 0) tiles directly under the floor ────────────
  const floorTopY = 0;
  const floorBottomY = floorTopY - floorThickness;
  const foundationTopY = floorBottomY;
  const foundationBottomY = foundationTopY - foundationThickness;
  push(
    subdivideBoxFragments({
      center: { x: 0, y: (foundationTopY + foundationBottomY) * 0.5, z: 0 },
      size: { x: width, y: foundationThickness, z: depth },
      divisions: { x: Math.max(3, Math.round(width / 2.5)), y: 1, z: Math.max(2, Math.round(depth / 2.5)) },
      fragmentType: 'foundation',
      isSupport: true,
    }),
  );

  // ── Floor slab the player walks on ────────────────────────────────────────
  push(
    subdivideBoxFragments({
      center: { x: 0, y: (floorTopY + floorBottomY) * 0.5, z: 0 },
      size: { x: width, y: floorThickness, z: depth },
      divisions: { x: Math.max(3, Math.round(width / 1.8)), y: 1, z: Math.max(3, Math.round(depth / 1.8)) },
      fragmentType: 'floor',
      density: HOUSE_FOUNDATION_DENSITY,
    }),
  );

  // ── Exterior walls (with openings) ────────────────────────────────────────
  const wallBaseY = floorTopY;
  const frontZ = -hd + wallThickness * 0.5;
  const backZ = hd - wallThickness * 0.5;
  const leftX = -hw + wallThickness * 0.5;
  const rightX = hw - wallThickness * 0.5;

  // Front wall (−Z): front door centered + two flanking windows.
  push(
    buildWallRun({
      axis: 'x', fixed: frontZ, runCenter: 0, runLength: width, baseY: wallBaseY,
      height: wallHeight, thickness: wallThickness, fragmentType: 'wall',
      density: HOUSE_DRYWALL_DENSITY, panelCell, courses: wallCourses,
      openings: [
        { center: 0, width: 1.1, sill: 0, height: 2.12 },
        { center: -hw * 0.62, width: 1.2, sill: 0.9, height: 1.2 },
        { center: hw * 0.62, width: 1.2, sill: 0.9, height: 1.2 },
      ],
    }),
  );
  // Back wall (+Z): back door to the kitchen + a window.
  push(
    buildWallRun({
      axis: 'x', fixed: backZ, runCenter: 0, runLength: width, baseY: wallBaseY,
      height: wallHeight, thickness: wallThickness, fragmentType: 'wall',
      density: HOUSE_DRYWALL_DENSITY, panelCell, courses: wallCourses,
      openings: [
        { center: hw * 0.55, width: 1.0, sill: 0, height: 2.05 },
        { center: -hw * 0.5, width: 1.3, sill: 0.9, height: 1.2 },
      ],
    }),
  );
  // Side walls (±X): one window each.
  for (const sideX of [leftX, rightX]) {
    push(
      buildWallRun({
        axis: 'z', fixed: sideX, runCenter: 0, runLength: depth, baseY: wallBaseY,
        height: wallHeight, thickness: wallThickness, fragmentType: 'wall',
        density: HOUSE_DRYWALL_DENSITY, panelCell, courses: wallCourses,
        openings: [{ center: -hd * 0.35, width: 1.3, sill: 0.9, height: 1.2 }],
      }),
    );
  }

  // ── Interior partition: living room (front) | kitchen (back), with a doorway ─
  const partitionZ = hd * 0.12;
  push(
    buildWallRun({
      axis: 'x', fixed: partitionZ, runCenter: 0, runLength: width - 2 * wallThickness,
      baseY: wallBaseY, height: wallHeight, thickness: interiorWallThickness,
      fragmentType: 'wall', density: HOUSE_DRYWALL_DENSITY, panelCell, courses: wallCourses,
      openings: [{ center: -hw * 0.5, width: 1.0, sill: 0, height: 2.1 }],
    }),
  );

  // ── Frame: top-plate ring beams, interior posts, ceiling tie-beams ────────
  const plateCenterY = wallHeight + plateThickness * 0.5;
  // Front/back plates (span X).
  for (const z of [frontZ, backZ]) {
    push(
      subdivideBoxFragments({
        center: { x: 0, y: plateCenterY, z },
        size: { x: width, y: plateThickness, z: wallThickness * 1.2 },
        divisions: { x: Math.max(2, Math.round(width / panelCell)), y: 1, z: 1 },
        fragmentType: 'beam', density: HOUSE_WOOD_DENSITY,
      }),
    );
  }
  // Side plates (span Z).
  for (const x of [leftX, rightX]) {
    push(
      subdivideBoxFragments({
        center: { x, y: plateCenterY, z: 0 },
        size: { x: wallThickness * 1.2, y: plateThickness, z: depth },
        divisions: { x: 1, y: 1, z: Math.max(2, Math.round(depth / panelCell)) },
        fragmentType: 'beam', density: HOUSE_WOOD_DENSITY,
      }),
    );
  }
  // Ceiling tie-beams (span Z) at a few X stations — visible "joists" tying the long walls.
  for (const x of [-width * 0.25, 0, width * 0.25]) {
    push(
      subdivideBoxFragments({
        center: { x, y: wallHeight - 0.06, z: 0 },
        size: { x: 0.12, y: 0.12, z: depth - 2 * wallThickness },
        divisions: { x: 1, y: 1, z: Math.max(2, Math.round(depth / panelCell)) },
        fragmentType: 'beam', density: HOUSE_WOOD_DENSITY,
      }),
    );
  }
  // Interior support posts along the partition line.
  for (const x of [-width * 0.25, width * 0.25]) {
    push(
      subdivideBoxFragments({
        center: { x, y: wallBaseY + wallHeight * 0.5, z: partitionZ },
        size: { x: 0.16, y: wallHeight, z: 0.16 },
        divisions: { x: 1, y: Math.max(2, Math.round(wallHeight / panelCell)), z: 1 },
        fragmentType: 'column', density: HOUSE_WOOD_DENSITY,
      }),
    );
  }

  // ── Gabled roof (two sloped planes + ridge beam + gable triangles) ────────
  const plateTopY = wallHeight + plateThickness;
  const eaveBite = 0.02; // roof underside dips into the plate so a real contact bond forms
  const roofCenterAtWall = plateTopY - eaveBite + roofThickness * 0.5; // centerline y at the wall line
  const ridgeCenterY = roofCenterAtWall + roofRise; // centerline y at the ridge
  const slope = Math.atan2(roofRise, hd); // pitch from horizontal
  const cosS = Math.cos(slope);
  const roofLenX = width + 0.6; // small rake overhang past the gable ends
  const horizHalfRun = hd + eaveOverhang;
  const slopeLen = horizHalfRun / cosS; // slab length from ridge to eave tip
  const roofCols = Math.max(2, Math.round(roofLenX / panelCell));
  const roofRows = Math.max(2, Math.round(slopeLen / panelCell));
  const ridgePivot: Vec3 = { x: 0, y: ridgeCenterY, z: 0 };

  for (const sign of [-1, 1] as const) {
    // Flat slab from ridge (z=0) outward, then tilt about the ridge so the far edge drops.
    push(
      buildRotatedBoxGridFragments({
        center: { x: 0, y: ridgeCenterY, z: sign * slopeLen * 0.5 },
        size: { x: roofLenX, y: roofThickness, z: slopeLen },
        divisions: { x: roofCols, y: 1, z: roofRows },
        fragmentType: 'roof',
        density: HOUSE_ROOF_DENSITY,
        rotation: { axis: 'x', angle: sign * slope, pivot: ridgePivot },
      }),
    );
  }
  // Ridge beam straddling the apex.
  push(
    subdivideBoxFragments({
      center: { x: 0, y: ridgeCenterY, z: 0 },
      size: { x: roofLenX, y: 0.16, z: 0.16 },
      divisions: { x: Math.max(2, Math.round(roofLenX / panelCell)), y: 1, z: 1 },
      fragmentType: 'beam', density: HOUSE_WOOD_DENSITY,
    }),
  );
  // Gable-end triangles above the ±X short walls (stepped courses shrinking to the ridge).
  const gableCourses = 4;
  const gableH = ridgeCenterY - wallHeight;
  for (const sideX of [-hw + wallThickness * 0.5, hw - wallThickness * 0.5]) {
    for (let k = 0; k < gableCourses; k++) {
      const yMid = wallHeight + (k + 0.5) * (gableH / gableCourses);
      const halfZ = hd * (ridgeCenterY - yMid) / gableH;
      if (halfZ < 0.15) continue;
      push(
        subdivideBoxFragments({
          center: { x: sideX, y: yMid, z: 0 },
          size: { x: wallThickness, y: gableH / gableCourses, z: 2 * halfZ },
          divisions: { x: 1, y: 1, z: Math.max(1, Math.round((2 * halfZ) / panelCell)) },
          fragmentType: 'wall', density: HOUSE_DRYWALL_DENSITY,
        }),
      );
    }
  }

  // ── Furniture (a convex decomposition of small boxes per piece) ───────────
  if (furniture) {
    addFurniture(push, { width, depth, hw, hd, partitionZ, floorTopY, wallThickness });
  }

  // Lift the whole house so the *dynamic* floor slab rests ON the runtime's static ground
  // collider (its top is at y=0) instead of overlapping it: after the shift the foundation
  // (static) is buried below ground, the floor bottom sits at y=0, and the only "step" into
  // the house is the floor thickness (well within the FPS controller's auto-step). The
  // shift is uniform, so all relative geometry/bonds are unchanged.
  const liftY = floorThickness;
  for (const f of fragments) {
    f.worldPosition = { ...f.worldPosition, y: f.worldPosition.y + liftY };
  }

  return {
    fragments,
    nodeColors,
    dims: {
      width, depth, wallHeight,
      floorTopY: floorTopY + liftY,
      ridgeY: ridgeCenterY + liftY,
      foundationBottomY: foundationBottomY + liftY,
      plateTopY: plateTopY + liftY,
    },
  };
}

// Furniture accent colors, rotated per piece for a colorful interior.
const FURNITURE_ACCENTS = [0xc0473b, 0x3b6fc0, 0x3ba35a, 0xd6a73a, 0x9b59b6, 0xe07b39];

function addFurniture(
  push: (fs: FragmentInfo[], color?: number) => void,
  g: { width: number; depth: number; hw: number; hd: number; partitionZ: number; floorTopY: number; wallThickness: number },
) {
  const { hw, hd, partitionZ, floorTopY, wallThickness } = g;
  const box = (cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, type: FragmentType): FragmentInfo => ({
    worldPosition: { x: cx, y: cy, z: cz },
    halfExtents: { x: sx * 0.5, y: sy * 0.5, z: sz * 0.5 },
    geometry: new THREE.BoxGeometry(sx, sy, sz),
    isSupport: false,
    fragmentType: type,
    density: HOUSE_FURNITURE_DENSITY,
  });

  // Dining table + 4 chairs in the living room (front half, −Z of the partition).
  const tx = hw * 0.45;
  const tz = -hd * 0.45;
  const wood = 0x9c6b3f;
  const tableTopY = 0.76;
  push([box(tx, floorTopY + tableTopY, tz, 1.3, 0.06, 0.85, 'furniture')], wood);
  for (const dx of [-0.55, 0.55]) {
    for (const dz of [-0.36, 0.36]) {
      push([box(tx + dx, floorTopY + 0.37, tz + dz, 0.07, 0.74, 0.07, 'furniture')], wood);
    }
  }
  const chairOffsets: Array<[number, number, number]> = [
    [-0.95, 0, 0], [0.95, 0, 0], [0, 0, -0.7], [0, 0, 0.7],
  ];
  chairOffsets.forEach(([ox, , oz], i) => {
    const accent = FURNITURE_ACCENTS[i % FURNITURE_ACCENTS.length];
    const cx = tx + ox;
    const cz = tz + oz;
    push([box(cx, floorTopY + 0.46, cz, 0.42, 0.05, 0.42, 'furniture')], accent); // seat
    push([box(cx, floorTopY + 0.72, cz - 0.18, 0.42, 0.5, 0.05, 'furniture')], accent); // back
    for (const lx of [-0.17, 0.17]) {
      for (const lz of [-0.17, 0.17]) {
        push([box(cx + lx, floorTopY + 0.225, cz + lz, 0.05, 0.45, 0.05, 'furniture')], accent); // legs
      }
    }
  });

  // Kitchen counter (back half, +Z) against the back wall.
  const counterZ = hd - wallThickness - 0.3;
  const counterX = hw * 0.4;
  push([box(counterX, floorTopY + 0.45, counterZ, 2.2, 0.9, 0.6, 'furniture')], 0xd9dde0); // carcass
  push([box(counterX, floorTopY + 0.92, counterZ, 2.35, 0.06, 0.66, 'furniture')], wood); // countertop
  // Upper wall cabinet (wall-mounted → 'shelf', clings to the wall).
  push([box(counterX, floorTopY + 1.75, hd - wallThickness - 0.18, 1.7, 0.7, 0.34, 'shelf')], 0xb24c4c);

  // Wall shelves on the left living-room wall.
  const shelfX = -hw + wallThickness + 0.13;
  for (const sy of [1.4, 1.85]) {
    push([box(shelfX, floorTopY + sy, -hd * 0.4, 0.26, 0.04, 0.95, 'shelf')], 0x8b5a2b);
  }

  void partitionZ;
}

// ── Public builders ─────────────────────────────────────────────────────────

function finalize(scenario: ScenarioDesc, o: HouseOptions, collected: CollectedHouse): ScenarioDesc {
  const fragmentTypes = collected.fragments.map((f) => f.fragmentType);
  scenario.bonds = applyBondStrengthMultipliers(
    scenario.bonds,
    fragmentTypes,
    makeHouseBondMultiplier(o.multipliers),
  );
  scenario.parameters = {
    ...scenario.parameters,
    house: {
      ...collected.dims,
      fragmentTypes,
      nodeColors: collected.nodeColors,
    },
  };
  return scenario;
}

/**
 * Build the house scenario synchronously with the pure-JS proximity bonder. Every chunk is
 * a box, so no WASM/Voronoi is needed. For more accurate bonds on the sloped roof contacts
 * use {@link buildHouseScenarioAsync} with `bondMode: 'auto'`.
 */
export function buildHouseScenario(options: HouseOptions = {}): ScenarioDesc {
  const o = { ...DEFAULT_HOUSE_OPTIONS, ...options };
  const collected = collectHouseFragments(o);
  const scenario = buildScenarioFromFragments(collected.fragments, {
    areaNormalization: 'none',
  });
  return finalize(scenario, options, collected);
}

/**
 * Async house builder. Adds `bondMode: 'auto'` — the WASM triangle-overlap bonder, which is
 * more robust for the pitched roof's sloped contacts than the JS proximity heuristic. Falls
 * back to proximity bonds on any failure.
 */
export async function buildHouseScenarioAsync(options: HouseOptions = {}): Promise<ScenarioDesc> {
  const o = { ...DEFAULT_HOUSE_OPTIONS, ...options };
  const collected = collectHouseFragments(o);
  const scenario = await buildScenarioFromFragmentsAsync(collected.fragments, {
    bondMode: o.bondMode ?? 'auto',
    areaNormalization: 'none',
  });
  return finalize(scenario, options, collected);
}
