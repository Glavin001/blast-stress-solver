/**
 * Rectilinear "structure preset" scenario builders.
 *
 * Ported from vibe-city's `/destructible-stress` page (src/lib/stress/scenarios/
 * structurePresets.ts). A single grid builder — {@link buildRectilinearScenario} —
 * carves a hollow voxel shell out of a box (walls, floors, openings) via an
 * `includeNode` predicate, bonds neighbours on a regular lattice (with optional
 * face diagonals for shear stiffness), and normalizes bond area per axis so the
 * structure stands under gravity until it is hit.
 *
 * Three presets are recreated here:
 *   - {@link buildFrameTowerScenario}        — "Multi-storey frame tower"
 *   - {@link buildConcreteHutScenario}       — "Mini concrete hut"
 *   - {@link buildCourtyardBungalowScenario} — "Courtyard bungalow"
 *
 * Each node is a uniform cell (the scenario's `spacing`), so the runtime sizes
 * both the box collider and the render mesh from `spacing` — no per-node collider
 * descriptors are needed. Support cells (mass 0) are the static foundation.
 */
import type { ScenarioBond, ScenarioDesc, ScenarioNode, Vec3 } from '../rapier/types';

const EPSILON = 1e-8;

type Vec3i = { x: number; y: number; z: number };

type IncludeArgs = {
  ix: number;
  iy: number;
  iz: number;
  segments: Vec3i;
  position: Vec3;
};

export type RectilinearOptions = {
  /** Overall footprint of the structure in metres (X × Y × Z). */
  size: Vec3;
  /** Voxel resolution of the grid along each axis. */
  segments: Vec3i;
  /** World-space centre of the footprint (Y defaults to half the height). */
  center?: Vec3;
  /** Keep a voxel cell only when this returns true (carves walls / openings). */
  includeNode?: (args: IncludeArgs) => boolean;
  /** Cells matching this are static supports (mass 0). Default: bottom layer. */
  supportPredicate?: (args: IncludeArgs) => boolean;
  /** Total dynamic mass distributed across the included cells (kg). */
  deckMass?: number;
  /** Multiplier on raw cell face areas before normalization. */
  areaScale?: number;
  /** Add face-diagonal bonds for extra shear/bending stiffness. */
  addDiagonals?: boolean;
  /** Area multiplier for the diagonal bonds. */
  diagScale?: number;
  /** Per-axis bond-area normalization (keeps the shell standing). */
  normalizeAreas?: boolean;
  /** Toggle bonds along each axis (useful for stress experiments). */
  bondsX?: boolean;
  bondsY?: boolean;
  bondsZ?: boolean;
};

function clampSegments(value: number): number {
  return Math.max(1, Math.floor(value));
}

function makeVec(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

function subVec(a: Vec3, b: Vec3): Vec3 {
  return makeVec(a.x - b.x, a.y - b.y, a.z - b.z);
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v.x, v.y, v.z);
  if (len <= EPSILON) return makeVec(0, 0, 0);
  return makeVec(v.x / len, v.y / len, v.z / len);
}

/**
 * Build a voxel-grid scenario from a box footprint and an `includeNode` predicate.
 * Returns a flat node/bond stress graph plus `gridCoordinates` and `spacing` so the
 * runtime can size colliders and render meshes from the uniform cell.
 */
export function buildRectilinearScenario({
  size,
  segments,
  center,
  includeNode,
  supportPredicate,
  deckMass = 14_000,
  areaScale = 0.05,
  addDiagonals = false,
  diagScale = 0.75,
  normalizeAreas = true,
  bondsX = true,
  bondsY = true,
  bondsZ = true,
}: RectilinearOptions): ScenarioDesc {
  const segX = clampSegments(segments.x);
  const segY = clampSegments(segments.y);
  const segZ = clampSegments(segments.z);

  const cellX = size.x / segX;
  const cellY = size.y / segY;
  const cellZ = size.z / segZ;

  const origin = makeVec(
    (center?.x ?? 0) - size.x * 0.5 + cellX * 0.5,
    (center?.y ?? size.y * 0.5) - size.y * 0.5 + cellY * 0.5,
    (center?.z ?? 0) - size.z * 0.5 + cellZ * 0.5,
  );

  const grid: number[][][] = Array.from({ length: segX }, () =>
    Array.from({ length: segY }, () => Array.from({ length: segZ }, () => -1)),
  );

  const nodes: ScenarioNode[] = [];
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];

  const include = includeNode ?? (() => true);
  const support = supportPredicate ?? (({ iy }) => iy === 0);

  const cellVolume = cellX * cellY * cellZ;

  let totalVolume = 0;

  for (let ix = 0; ix < segX; ix += 1) {
    for (let iy = 0; iy < segY; iy += 1) {
      for (let iz = 0; iz < segZ; iz += 1) {
        const position = makeVec(origin.x + ix * cellX, origin.y + iy * cellY, origin.z + iz * cellZ);
        if (!include({ ix, iy, iz, segments: { x: segX, y: segY, z: segZ }, position })) continue;
        const isSupport = support({ ix, iy, iz, segments: { x: segX, y: segY, z: segZ }, position });
        const nodeIndex = nodes.length;
        const volume = isSupport ? 0 : cellVolume;
        if (!isSupport) totalVolume += volume;
        nodes.push({ centroid: position, mass: volume, volume });
        grid[ix][iy][iz] = nodeIndex;
        gridCoordinates[nodeIndex] = { ix, iy, iz };
      }
    }
  }

  const massScale = totalVolume > 0 ? deckMass / totalVolume : 0;
  nodes.forEach((node) => {
    node.mass = massScale > 0 && node.volume > 0 ? node.volume * massScale : 0;
  });

  const bonds: ScenarioBond[] = [];

  const areaX = cellY * cellZ * areaScale;
  const areaY = cellX * cellZ * areaScale;
  const areaZ = cellX * cellY * areaScale;

  const addBond = (a: number, b: number, area: number) => {
    if (a < 0 || b < 0) return;
    const na = nodes[a];
    const nb = nodes[b];
    const centroid = makeVec(
      (na.centroid.x + nb.centroid.x) * 0.5,
      (na.centroid.y + nb.centroid.y) * 0.5,
      (na.centroid.z + nb.centroid.z) * 0.5,
    );
    const normal = normalize(subVec(nb.centroid, na.centroid));
    bonds.push({ node0: a, node1: b, centroid, normal, area: Math.max(area, EPSILON) });
  };

  for (let ix = 0; ix < segX; ix += 1) {
    for (let iy = 0; iy < segY; iy += 1) {
      for (let iz = 0; iz < segZ; iz += 1) {
        const current = grid[ix][iy][iz];
        if (current < 0) continue;
        if (bondsX && ix + 1 < segX) addBond(current, grid[ix + 1][iy][iz], areaX);
        if (bondsY && iy + 1 < segY) addBond(current, grid[ix][iy + 1][iz], areaY);
        if (bondsZ && iz + 1 < segZ) addBond(current, grid[ix][iy][iz + 1], areaZ);
        if (addDiagonals) {
          if (bondsX && bondsY && ix + 1 < segX && iy + 1 < segY) addBond(current, grid[ix + 1][iy + 1][iz], 0.5 * (areaX + areaY) * diagScale);
          if (bondsX && bondsZ && ix + 1 < segX && iz + 1 < segZ) addBond(current, grid[ix + 1][iy][iz + 1], 0.5 * (areaX + areaZ) * diagScale);
          if (bondsY && bondsZ && iy + 1 < segY && iz + 1 < segZ) addBond(current, grid[ix][iy + 1][iz + 1], 0.5 * (areaY + areaZ) * diagScale);
        }
      }
    }
  }

  if (normalizeAreas && bonds.length) {
    const target = { x: size.y * size.z, y: size.x * size.z, z: size.x * size.y };
    const sum = { x: 0, y: 0, z: 0 };
    const pick = (n: Vec3): 'x' | 'y' | 'z' => {
      const ax = Math.abs(n.x);
      const ay = Math.abs(n.y);
      const az = Math.abs(n.z);
      if (ax >= ay && ax >= az) return 'x';
      if (ay >= az) return 'y';
      return 'z';
    };
    bonds.forEach((bond) => {
      sum[pick(bond.normal)] += bond.area;
    });
    const scale = {
      x: sum.x > 0 ? target.x / sum.x : 1,
      y: sum.y > 0 ? target.y / sum.y : 1,
      z: sum.z > 0 ? target.z / sum.z : 1,
    } as const;
    bonds.forEach((bond) => {
      bond.area *= scale[pick(bond.normal)];
    });
  }

  return {
    nodes,
    bonds,
    gridCoordinates,
    spacing: makeVec(cellX, cellY, cellZ),
    parameters: { size, segments, deckMass, areaScale, addDiagonals },
  } satisfies ScenarioDesc;
}

export type StructurePresetOptions = {
  bondsX?: boolean;
  bondsY?: boolean;
  bondsZ?: boolean;
};

/**
 * "Mini concrete hut" — a hollow four-wall shelter with an open roof, a front
 * doorway carve-out and a side window. Showcases shell fragmentation.
 */
export function buildConcreteHutScenario({ bondsX = true, bondsY = true, bondsZ = true }: StructurePresetOptions = {}): ScenarioDesc {
  const segments = { x: 18, y: 9, z: 14 };
  const doorStart = Math.floor(segments.x * 0.38);
  const doorEnd = Math.ceil(segments.x * 0.62);
  const doorHeight = Math.floor(segments.y * 0.55);
  const windowRow = Math.floor(segments.y * 0.65);
  const windowStart = Math.floor(segments.z * 0.35);
  const windowEnd = Math.ceil(segments.z * 0.65);

  return buildRectilinearScenario({
    size: makeVec(6.5, 3.4, 5.2),
    segments,
    deckMass: 19_000,
    areaScale: 0.052,
    addDiagonals: true,
    diagScale: 0.6,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onX = ix === 0 || ix === seg.x - 1;
      const onZ = iz === 0 || iz === seg.z - 1;
      const onTop = iy === seg.y - 1;
      if (!onX && !onZ) return false;
      if (onTop) return false; // open roof
      // doorway carve-out on front wall (iz === 0)
      if (iz === 0 && ix >= doorStart && ix <= doorEnd && iy <= doorHeight) {
        return ix === doorStart || ix === doorEnd || iy === doorHeight;
      }
      // side window on the right wall
      if (ix === seg.x - 1 && iy === windowRow && iz >= windowStart && iz <= windowEnd) {
        return iy === windowRow && (iz === windowStart || iz === windowEnd);
      }
      return true;
    },
  });
}

const FRAME_TOWER_CONFIG = {
  /** Number of vertical layers; tweak this to make the tower taller or shorter. */
  segmentCount: 58,
  /**
   * Maintain the original per-layer scale (9.2 m over 22 layers) so a new height
   * simply stretches the existing detailing proportionally.
   */
  metersPerSegment: 9.2 / 22,
} as const;

/**
 * "Multi-storey frame tower" — a tall framed shell with interior columns, floor
 * plates and slit windows for progressive-collapse testing.
 */
export function buildFrameTowerScenario({ bondsX = true, bondsY = true, bondsZ = true }: StructurePresetOptions = {}): ScenarioDesc {
  const towerSegments = { x: 16, y: FRAME_TOWER_CONFIG.segmentCount, z: 16 };
  const towerHeightMeters = towerSegments.y * FRAME_TOWER_CONFIG.metersPerSegment;
  const floorHeights = [
    0,
    Math.floor(towerSegments.y * 0.33),
    Math.floor(towerSegments.y * 0.66),
    towerSegments.y - 2,
  ];
  const columnPositions = [Math.floor(towerSegments.x * 0.25), Math.floor(towerSegments.x * 0.75)];

  return buildRectilinearScenario({
    size: makeVec(6.8, towerHeightMeters, 6.8),
    segments: towerSegments,
    deckMass: 280_000,
    areaScale: 0.055,
    addDiagonals: true,
    diagScale: 0.65,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onShell = ix === 0 || ix === seg.x - 1 || iz === 0 || iz === seg.z - 1;
      const isRoof = iy === seg.y - 1;
      const isFloor = floorHeights.includes(iy);
      const inColumn = columnPositions.includes(ix) && columnPositions.includes(iz);
      const hasWindowBand = iy === Math.floor(seg.y * 0.5) && (ix + iz) % 2 === 0;
      if (isRoof) return true;
      if (onShell) {
        if (iy > Math.floor(seg.y * 0.4) && (ix === Math.floor(seg.x * 0.5) || iz === Math.floor(seg.z * 0.5))) {
          return (iy - Math.floor(seg.y * 0.4)) % 2 === 0; // vertical slit windows
        }
        return true;
      }
      if (isFloor) return true;
      if (inColumn) return true;
      if (hasWindowBand) return true;
      return false;
    },
  });
}

/**
 * "Courtyard bungalow" — a low-rise home wrapping a central garden, with a
 * breezeway doorway, lintels, a waist-high courtyard ring and clerestory openings.
 */
export function buildCourtyardBungalowScenario({ bondsX = true, bondsY = true, bondsZ = true }: StructurePresetOptions = {}): ScenarioDesc {
  const segments = { x: 28, y: 14, z: 28 };
  const courtyardMinX = Math.floor(segments.x * 0.28);
  const courtyardMaxX = segments.x - courtyardMinX - 1;
  const courtyardMinZ = Math.floor(segments.z * 0.28);
  const courtyardMaxZ = segments.z - courtyardMinZ - 1;
  const skylightRow = Math.floor(segments.y * 0.85);
  const lintelRow = Math.floor(segments.y * 0.45);

  return buildRectilinearScenario({
    size: makeVec(12.5, 4.4, 12.5),
    segments,
    deckMass: 880_000,
    areaScale: 0.054,
    addDiagonals: true,
    diagScale: 0.58,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onOuterShell = ix === 0 || ix === seg.x - 1 || iz === 0 || iz === seg.z - 1;
      const onCourtyardRing =
        ix === courtyardMinX ||
        ix === courtyardMaxX ||
        iz === courtyardMinZ ||
        iz === courtyardMaxZ;
      const inCourtyardVoid = ix > courtyardMinX && ix < courtyardMaxX && iz > courtyardMinZ && iz < courtyardMaxZ;
      const galleryBand = iy === Math.floor(seg.y * 0.32);

      if (iy === 0) return true; // slab
      if (galleryBand && onCourtyardRing) return true;
      if (onOuterShell) {
        const doorSpanStart = Math.floor(seg.z * 0.4);
        const doorSpanEnd = Math.floor(seg.z * 0.6);
        const doorHeight = Math.floor(seg.y * 0.35);
        if (ix === 0 && iz >= doorSpanStart && iz <= doorSpanEnd && iy <= doorHeight) {
          return iy === doorHeight || iz === doorSpanStart || iz === doorSpanEnd;
        }
        if (iy === lintelRow && (ix === 0 || ix === seg.x - 1)) {
          return true;
        }
        return true;
      }

      if (onCourtyardRing) {
        if (iy >= Math.floor(seg.y * 0.6)) {
          return false; // open clerestory around courtyard
        }
        if (iy === Math.floor(seg.y * 0.25)) {
          return true; // waist-high garden wall
        }
        return iy <= Math.floor(seg.y * 0.5);
      }

      if (inCourtyardVoid) {
        return iy === skylightRow && (ix - courtyardMinX) % 3 === 0 && (iz - courtyardMinZ) % 3 === 0;
      }

      return false;
    },
  });
}

export type StructurePresetId = 'frameTower' | 'concreteHut' | 'courtyardBungalow';

export const STRUCTURE_PRESET_METADATA: Array<{
  id: StructurePresetId;
  label: string;
  description: string;
  build: (opts?: StructurePresetOptions) => ScenarioDesc;
}> = [
  {
    id: 'frameTower',
    label: 'Multi-storey frame tower',
    description: 'Tall frame with interior columns and floor plates for progressive collapse testing.',
    build: buildFrameTowerScenario,
  },
  {
    id: 'concreteHut',
    label: 'Mini concrete hut',
    description: 'Hollow four-wall shelter with a doorway and side window to showcase shell fragmentation.',
    build: buildConcreteHutScenario,
  },
  {
    id: 'courtyardBungalow',
    label: 'Courtyard bungalow',
    description: 'Low-rise home wrapping a central garden with breezeways, lintels, and clerestory openings.',
    build: buildCourtyardBungalowScenario,
  },
];

/** Build a structure-preset scenario by id. */
export function buildStructurePreset(id: StructurePresetId, opts?: StructurePresetOptions): ScenarioDesc {
  const preset = STRUCTURE_PRESET_METADATA.find((p) => p.id === id);
  if (!preset) throw new Error(`Unknown structure preset: ${id}`);
  return preset.build(opts);
}
