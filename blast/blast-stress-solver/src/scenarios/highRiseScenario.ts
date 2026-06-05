/**
 * High-rise apartment building scenario (mid-rise, ~8-10 floors).
 *
 * Goal: a structure that a wrecking ball destroys *realistically* — punching local
 * holes and causing incremental / partial collapse — instead of shattering uniformly
 * like glass. The key is a heterogeneous structure rather than a uniform-strength one:
 *
 *   - A stiff reinforced-concrete SKELETON: columns + flat floor slabs. Coarse
 *     subdivision (few large chunks), large bond contact areas, high bond-strength
 *     multipliers, and a strong anchor to static (mass 0) foundation nodes.
 *   - Frangible INFILL walls (drywall/masonry) filling the perimeter bays between
 *     columns: fine subdivision (many small chunks), thin (small contact area), and
 *     bonded *weakly* to the frame so whole panels pop out locally without
 *     propagating a stress wave into the skeleton.
 *
 * Structural system: "flat slab" (columns + slabs, no beams) — common in real
 * apartment buildings. The vertical load path is foundation -> column -> slab ->
 * column -> ... -> roof. Columns are interrupted by the slabs at each floor (their
 * tops/bottoms touch the slab, forming column<->slab joints), so there is no
 * geometric overlap and the geometry is fully deterministic.
 *
 * Material strength is expressed through bond *area* (stress = impulse / area), since
 * the Blast solver's stress limits are global. See HIGH_RISE_BOND_MULTIPLIERS.
 */
import type { ScenarioDesc } from '../rapier/types';
import type { FragmentInfo, FragmentType } from '../three/fracture';
import {
  applyBondStrengthMultipliers,
  subdivideBoxFragments,
  type BondStrengthMultiplierFn,
} from '../three/fractureBuilders';
import { buildScenarioFromFragments } from '../three/scenarioFromFragments';

// ── Material densities (kg/m^3) ─────────────────────────────────────────────
export const CONCRETE_DENSITY = 2400; // reinforced concrete
export const DRYWALL_DENSITY = 800; // light drywall/partition infill

export type HighRiseOptions = {
  /** Number of occupied storeys (default 9). */
  floorCount?: number;
  /** Floor-to-floor height in meters (default 3.2). */
  floorHeight?: number;
  /** Footprint width along X in meters (default 18). */
  width?: number;
  /** Footprint depth along Z in meters (default 14). */
  depth?: number;
  /** Column count along X (default 4). */
  columnsX?: number;
  /** Column count along Z (default 3). */
  columnsZ?: number;
  /** Square column cross-section size in meters (default 0.5). */
  columnSize?: number;
  /** Concrete floor-slab thickness in meters (default 0.22). */
  slabThickness?: number;
  /** Slab grid divisions along X / Z (default ~one tile per 3 m). */
  slabDivX?: number;
  slabDivZ?: number;
  /** Infill wall panel thickness in meters (default 0.15). */
  infillThickness?: number;
  /** Approx. infill chunk edge length in meters — smaller = finer (default 1.8). */
  infillCell?: number;
  /** Foundation slab thickness in meters (default 0.6). */
  foundationThickness?: number;
  /** Gap between foundation bottom and ground plane (default 0.02). */
  groundClearance?: number;
  /** Density of infill walls (default DRYWALL_DENSITY). */
  infillDensity?: number;
  /** Density of the concrete skeleton (default CONCRETE_DENSITY). */
  concreteDensity?: number;
  /** Bond mode for the assembler. 'proximity' is pure-JS (default); 'auto' needs WASM. */
  bondMode?: 'proximity' | 'auto';
  /** Override individual bond-strength multipliers (for parameter sweeps / tuning). */
  multipliers?: Partial<HighRiseMultipliers>;
  /**
   * Vertical separation-seam planes, given as X positions (meters). Bonds whose two
   * endpoints straddle a seam are weakened by `seamMultiplier`, compartmentalizing the
   * footprint so a collapse on one side detaches at the seam instead of dragging the
   * rest down (a structural "fuse" / expansion joint). Place seams in the gaps
   * *between* column lines so each compartment keeps its own vertical load path.
   */
  seamsX?: number[];
  /** Strength multiplier applied to seam-straddling bonds (default 0.12; <1 = weak fuse). */
  seamMultiplier?: number;
  /**
   * Storey indices whose slab acts as a stiff "transfer/firewall" belt: bonds in that
   * slab's Y-band are strengthened by `transferMultiplier` so a cascade can't punch
   * vertically through it (cf. real outrigger/transfer floors).
   */
  transferFloors?: number[];
  /** Strength multiplier applied to transfer-floor bonds (default 2.5; >1 = firewall). */
  transferMultiplier?: number;
};

export const DEFAULT_HIGH_RISE_OPTIONS: Required<
  Omit<
    HighRiseOptions,
    'bondMode' | 'multipliers' | 'seamsX' | 'seamMultiplier' | 'transferFloors' | 'transferMultiplier'
  >
> & { bondMode: 'proximity' | 'auto' } = {
  floorCount: 9,
  floorHeight: 3.2,
  width: 18,
  depth: 14,
  columnsX: 4,
  columnsZ: 3,
  columnSize: 0.5,
  slabThickness: 0.22,
  slabDivX: 6,
  slabDivZ: 5,
  infillThickness: 0.15,
  infillCell: 1.8,
  foundationThickness: 0.6,
  groundClearance: 0.02,
  infillDensity: DRYWALL_DENSITY,
  concreteDensity: CONCRETE_DENSITY,
  bondMode: 'proximity',
};

/**
 * Bond-area multipliers, expressed as a tunable table.
 *
 * The GLOBAL solver stress limits represent concrete (see the scene-pack `solver`
 * block). Multipliers > 1 strengthen a joint relative to plain concrete (rebar
 * continuity / monolithic pours); multipliers < 1 weaken it (drywall infill, which
 * per unit area is much weaker than concrete, and its light attachment to the frame
 * weaker still). These are the primary knobs the parameter sweep explores; the
 * defaults below were tuned against the FULL Rapier pipeline (contacts + collapse),
 * not just the headless stress solver, so a realistic ball blows out local infill
 * while the frame survives a single hit.
 */
export type HighRiseMultipliers = {
  foundationColumn: number;
  foundationSkeleton: number;
  columnColumn: number;
  columnSlab: number;
  beamColumn: number;
  beamBeam: number;
  beamSlab: number;
  slabSlab: number;
  infillInfill: number;
  slabInfill: number;
  frameInfill: number;
};

export const DEFAULT_HIGH_RISE_MULTIPLIERS: HighRiseMultipliers = {
  foundationColumn: 24.0, // strongest joint: base anchor (no "footing rips off")
  foundationSkeleton: 12.0, // foundation <-> slab/beam
  columnColumn: 16.0, // stacked columns (if any touch directly)
  columnSlab: 14.0, // primary vertical load path — must survive a single heavy hit
  beamColumn: 12.0,
  beamBeam: 10.0,
  beamSlab: 9.0,
  slabSlab: 7.0, // floor-plate diaphragm continuity
  infillInfill: 0.03, // drywall internal — weak so a realistic ball blows panels out
  slabInfill: 0.02, // panel sits in the slab band
  frameInfill: 0.015, // column/beam/foundation <-> drywall (light attachment)
};

/** Build a bond-strength multiplier function from the (optionally overridden) table. */
export function makeHighRiseBondMultiplier(
  overrides?: Partial<HighRiseMultipliers>,
): BondStrengthMultiplierFn {
  const M = { ...DEFAULT_HIGH_RISE_MULTIPLIERS, ...overrides };
  const isInfill = (t?: FragmentType) => t === 'infill' || t === 'wall';
  return (t0, t1) => {
    const pair = (a: FragmentType, b: FragmentType) =>
      (t0 === a && t1 === b) || (t0 === b && t1 === a);
    const has = (a: FragmentType) => t0 === a || t1 === a;

    // Weak infill joints first, so any skeleton<->infill pair resolves to "weak".
    if (isInfill(t0) || isInfill(t1)) {
      if (isInfill(t0) && isInfill(t1)) return M.infillInfill;
      if (has('slab') || has('floor')) return M.slabInfill;
      return M.frameInfill; // column / beam / foundation <-> infill
    }

    // Anchoring: the base must be the strongest joint in the structure.
    if (pair('foundation', 'column')) return M.foundationColumn;
    if (has('foundation')) return M.foundationSkeleton;

    // Stiff reinforced-concrete skeleton.
    if (pair('column', 'column')) return M.columnColumn;
    if (pair('column', 'beam')) return M.beamColumn;
    if (pair('beam', 'beam')) return M.beamBeam;
    if (pair('column', 'slab') || pair('column', 'floor')) return M.columnSlab;
    if (pair('beam', 'slab') || pair('beam', 'floor')) return M.beamSlab;
    if (has('slab') || has('floor')) return M.slabSlab;

    return 1.0;
  };
}

/** Default high-rise bond-strength multiplier (uses DEFAULT_HIGH_RISE_MULTIPLIERS). */
export const HIGH_RISE_BOND_MULTIPLIERS: BondStrengthMultiplierFn = makeHighRiseBondMultiplier();

function linspace(min: number, max: number, n: number): number[] {
  if (n <= 1) return [(min + max) * 0.5];
  const step = (max - min) / (n - 1);
  return Array.from({ length: n }, (_v, i) => min + step * i);
}

/**
 * Build the full high-rise scenario (nodes + bonds + parameters for serialization).
 */
export function buildHighRiseScenario(options: HighRiseOptions = {}): ScenarioDesc {
  const o = { ...DEFAULT_HIGH_RISE_OPTIONS, ...options };
  const {
    floorCount,
    floorHeight,
    width,
    depth,
    columnsX,
    columnsZ,
    columnSize,
    slabThickness,
    slabDivX,
    slabDivZ,
    infillThickness,
    infillCell,
    foundationThickness,
    groundClearance,
    infillDensity,
    concreteDensity,
    bondMode,
  } = o;

  const foundationTopY = groundClearance + foundationThickness;
  const colXs = linspace(-width * 0.5 + columnSize * 0.5, width * 0.5 - columnSize * 0.5, columnsX);
  const colZs = linspace(-depth * 0.5 + columnSize * 0.5, depth * 0.5 - columnSize * 0.5, columnsZ);

  // Y boundary helpers (flat-slab geometry; everything aligns exactly).
  const slabTopY = (k: number) => foundationTopY + k * floorHeight; // top of slab at floor k (k=1..floorCount)
  const storeyBottomY = (k: number) => (k === 0 ? foundationTopY : slabTopY(k)); // clear-space bottom of storey k
  const storeyTopY = (k: number) => slabTopY(k + 1) - slabThickness; // clear-space top of storey k
  const storeyClearH = floorHeight - slabThickness;

  const fragments: FragmentInfo[] = [];

  // ── Foundation: static (mass 0) tiles covering the footprint ──────────────
  fragments.push(
    ...subdivideBoxFragments({
      center: { x: 0, y: groundClearance + foundationThickness * 0.5, z: 0 },
      size: { x: width, y: foundationThickness, z: depth },
      divisions: { x: Math.max(2, columnsX), y: 1, z: Math.max(2, columnsZ) },
      fragmentType: 'foundation',
      isSupport: true,
    }),
  );

  // ── Columns: one chunk per storey at each grid position ───────────────────
  for (let k = 0; k < floorCount; k++) {
    const yBottom = storeyBottomY(k);
    const yTop = storeyTopY(k);
    const cy = (yBottom + yTop) * 0.5;
    const h = yTop - yBottom;
    for (const cx of colXs) {
      for (const cz of colZs) {
        fragments.push(
          ...subdivideBoxFragments({
            center: { x: cx, y: cy, z: cz },
            size: { x: columnSize, y: h, z: columnSize },
            divisions: { x: 1, y: 1, z: 1 },
            fragmentType: 'column',
            density: concreteDensity,
          }),
        );
      }
    }
  }

  // ── Floor slabs (incl. roof): full-footprint concrete plates ──────────────
  for (let k = 1; k <= floorCount; k++) {
    const topY = slabTopY(k);
    fragments.push(
      ...subdivideBoxFragments({
        center: { x: 0, y: topY - slabThickness * 0.5, z: 0 },
        size: { x: width, y: slabThickness, z: depth },
        divisions: { x: slabDivX, y: 1, z: slabDivZ },
        fragmentType: 'slab',
        density: concreteDensity,
      }),
    );
  }

  // ── Infill walls: frangible panels in the perimeter bays between columns ──
  const addInfillPanel = (
    centerX: number,
    centerZ: number,
    spanX: number,
    spanZ: number,
    yBottom: number,
    yTop: number,
  ) => {
    const h = yTop - yBottom;
    const spanH = Math.max(spanX, spanZ); // the wide (in-plane horizontal) span
    fragments.push(
      ...subdivideBoxFragments({
        center: { x: centerX, y: (yBottom + yTop) * 0.5, z: centerZ },
        size: { x: spanX, y: h, z: spanZ },
        divisions: {
          x: spanX > spanZ ? Math.max(1, Math.round(spanH / infillCell)) : 1,
          y: Math.max(1, Math.round(h / infillCell)),
          z: spanZ > spanX ? Math.max(1, Math.round(spanH / infillCell)) : 1,
        },
        fragmentType: 'infill',
        density: infillDensity,
      }),
    );
  };

  for (let k = 0; k < floorCount; k++) {
    const yBottom = storeyBottomY(k);
    const yTop = storeyTopY(k);

    // Front (min Z) and back (max Z) walls: panels span X between adjacent columns.
    for (const cz of [colZs[0], colZs[colZs.length - 1]]) {
      for (let i = 0; i < colXs.length - 1; i++) {
        const x0 = colXs[i] + columnSize * 0.5;
        const x1 = colXs[i + 1] - columnSize * 0.5;
        const spanX = x1 - x0;
        if (spanX <= 0.05) continue;
        addInfillPanel((x0 + x1) * 0.5, cz, spanX, infillThickness, yBottom, yTop);
      }
    }

    // Left (min X) and right (max X) walls: panels span Z between adjacent columns.
    for (const cx of [colXs[0], colXs[colXs.length - 1]]) {
      for (let j = 0; j < colZs.length - 1; j++) {
        const z0 = colZs[j] + columnSize * 0.5;
        const z1 = colZs[j + 1] - columnSize * 0.5;
        const spanZ = z1 - z0;
        if (spanZ <= 0.05) continue;
        addInfillPanel(cx, (z0 + z1) * 0.5, infillThickness, spanZ, yBottom, yTop);
      }
    }
  }

  if (bondMode === 'auto') {
    // Auto-bonding needs WASM (buildScenarioFromFragmentsAsync). For the synchronous
    // path used here we fall back to proximity; callers that need WASM bonding should
    // assemble fragments themselves. We keep the option for API symmetry.
    // eslint-disable-next-line no-console
    console.warn('[high-rise] bondMode "auto" requires the async assembler; using proximity.');
  }

  const scenario = buildScenarioFromFragments(fragments, {
    areaNormalization: 'none',
    // density on fragments drives mass; totalMass is unused for density-tagged nodes.
  });

  const fragmentTypes = fragments.map((f) => f.fragmentType);
  scenario.bonds = applyBondStrengthMultipliers(
    scenario.bonds,
    fragmentTypes,
    makeHighRiseBondMultiplier(o.multipliers),
  );

  // ── Spatial heterogeneity: separation seams + transfer (firewall) floors ──
  // These run *after* the type-based multipliers and scale bond area further, so a
  // seam-straddling slab bond becomes a weak fuse and a transfer-floor bond a strong
  // belt. Strong "blocks" joined by weak "seams" fail at the seams (compartmentalized
  // collapse) instead of propagating a brittle stress wave across the whole structure.
  const seamsX = o.seamsX ?? [];
  const seamMultiplier = o.seamMultiplier ?? 0.12;
  const transferFloors = o.transferFloors ?? [];
  const transferMultiplier = o.transferMultiplier ?? 2.5;
  if (seamsX.length > 0 || transferFloors.length > 0) {
    const nodeX = (i: number) => scenario.nodes[i]?.centroid.x ?? 0;
    const transferYs = transferFloors.map((k) => slabTopY(k) - slabThickness * 0.5);
    const yBand = slabThickness * 0.75;
    scenario.bonds = scenario.bonds.map((bond) => {
      let mul = 1;
      // Weaken bonds whose endpoints sit on opposite sides of a seam plane.
      for (const sx of seamsX) {
        if ((nodeX(bond.node0) - sx) * (nodeX(bond.node1) - sx) < 0) {
          mul *= seamMultiplier;
          break;
        }
      }
      // Strengthen bonds sitting within a transfer floor's slab band.
      for (const ty of transferYs) {
        if (Math.abs(bond.centroid.y - ty) <= yBand) {
          mul *= transferMultiplier;
          break;
        }
      }
      return mul === 1 ? bond : { ...bond, area: bond.area * mul };
    });
  }

  scenario.parameters = {
    ...scenario.parameters,
    highRise: {
      ...o,
      foundationTopY,
      totalHeight: slabTopY(floorCount),
      fragmentTypes,
    },
  };
  return scenario;
}
