/**
 * "Mini city" scenario builders for stress/perf profiling.
 *
 * INTERNAL / unstable — NOT exported from `src/scenarios/index.ts`. These compose
 * the shipped single-structure builders into large multi-building worlds via
 * {@link composeScenarios}, to exercise the destruction pipeline at city scale:
 * many independent islands, dense vs. sparse destruction, and cascading collapse.
 *
 * Everything here is pure data (no WASM); the high-rise builder pulls in three.js
 * for geometry math, which runs fine headlessly in Node.
 *
 * Determinism: building-kind selection and any placement jitter use a seeded
 * `mulberry32` PRNG — never `Math.random` — so a given (preset, tier, seed) always
 * produces the identical world. This is a hard requirement: the goal is to find
 * bottlenecks WITHOUT changing behavior, so runs must be reproducible.
 */
import type { ProjectileSpawn, ScenarioDesc, Vec3 } from '../../rapier/types';
import { buildTowerScenario } from '../../scenarios/towerScenario';
import { buildWallScenario } from '../../scenarios/wallScenario';
import { buildHighRiseScenario } from '../../scenarios/highRiseScenario';
import { buildBeamBridgeScenario } from '../../scenarios/bridgeScenario';
import { composeScenarios, type ComposedScenario, type ScenarioPart } from './composeScenarios';
import type { ImpactEvent } from './harness';

// ── Deterministic PRNG ──────────────────────────────────────────────────────
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SizeTier = 'small' | 'medium' | 'large' | 'xl';

/** Per-tier multiplier applied to grid dimensions. Building sizes stay fixed so
 *  output is identical per building; only the COUNT of buildings scales. */
const TIER_GRID: Record<SizeTier, number> = { small: 1, medium: 1.5, large: 2, xl: 3 };

// ── Building-kind catalogue ─────────────────────────────────────────────────
// Each returns a ScenarioDesc centered at origin, sitting on the ground (its
// static base at y≈0). Sizes are deliberately modest so a grid of them stays
// tractable; the world scales by adding MORE buildings, not bigger ones.
export type BuildingKind = 'smallTower' | 'midTower' | 'tallTower' | 'lowWall' | 'miniHighRise';

const KIND_BUILDERS: Record<BuildingKind, () => ScenarioDesc> = {
  smallTower: () => buildTowerScenario({ side: 3, stories: 5, totalMass: 1500 }),
  midTower: () => buildTowerScenario({ side: 4, stories: 8, totalMass: 4000 }),
  tallTower: () => buildTowerScenario({ side: 4, stories: 14, totalMass: 7000 }),
  lowWall: () => buildWallScenario({ spanSegments: 6, heightSegments: 4, layers: 1, deckMass: 6000 }),
  miniHighRise: () =>
    buildHighRiseScenario({
      floorCount: 4,
      floorHeight: 3.0,
      width: 10,
      depth: 8,
      columnsX: 3,
      columnsZ: 2,
      infillCell: 2.6,
      slabDivX: 4,
      slabDivZ: 3,
    }),
};

/** Approximate XZ footprint per kind (m), so we can space plots without overlap. */
const KIND_FOOTPRINT: Record<BuildingKind, number> = {
  smallTower: 2.0,
  midTower: 2.5,
  tallTower: 2.5,
  lowWall: 6.5,
  miniHighRise: 11,
};

// ── Generic city builder ────────────────────────────────────────────────────
export interface CityOptions {
  gridX: number;
  gridZ: number;
  /** Plot pitch (center-to-center spacing) in meters. */
  cellSize?: number;
  /** Building kinds to sample from, one per cell (seeded). */
  kinds?: BuildingKind[];
  seed?: number;
  /** Fraction of a cell a building may be randomly nudged within (0 = grid-perfect). */
  jitter?: number;
}

export function buildCityScenario(opts: CityOptions): ComposedScenario {
  const { gridX, gridZ, cellSize = 8, kinds = ['smallTower', 'midTower', 'lowWall'], seed = 1, jitter = 0 } = opts;
  const rng = mulberry32(seed);
  const parts: ScenarioPart[] = [];

  const halfX = ((gridX - 1) * cellSize) / 2;
  const halfZ = ((gridZ - 1) * cellSize) / 2;

  for (let iz = 0; iz < gridZ; iz++) {
    for (let ix = 0; ix < gridX; ix++) {
      const kind = kinds[Math.floor(rng() * kinds.length) % kinds.length];
      const jx = jitter ? (rng() - 0.5) * jitter * cellSize : 0;
      const jz = jitter ? (rng() - 0.5) * jitter * cellSize : 0;
      parts.push({
        scenario: KIND_BUILDERS[kind](),
        offset: { x: ix * cellSize - halfX + jx, y: 0, z: iz * cellSize - halfZ + jz },
        rotateY: ([0, 90, 180, 270] as const)[Math.floor(rng() * 4) % 4],
        tag: `${kind}-${ix}x${iz}`,
      });
    }
  }

  return composeScenarios(parts);
}

// ── Impact helpers (aim at placed buildings via their AABBs) ─────────────────
function horizontalHit(target: { center: Vec3; halfExtents: Vec3 }, opts?: { speed?: number; mass?: number; radius?: number; heightFrac?: number }): ProjectileSpawn {
  const { speed = 45, mass = 16000, radius = 0.4, heightFrac = 0.5 } = opts ?? {};
  const standoff = target.halfExtents.z + 4;
  // Buildings sit on the ground (base at y≈0), so a height fraction maps to world Y.
  return {
    position: { x: target.center.x, y: target.halfExtents.y * 2 * heightFrac, z: target.center.z + standoff },
    velocity: { x: 0, y: 0, z: -speed },
    radius,
    mass,
    ttl: 6000,
  };
}

function topDrop(target: { center: Vec3; halfExtents: Vec3 }, opts?: { speed?: number; mass?: number; radius?: number }): ProjectileSpawn {
  const { speed = 35, mass = 22000, radius = 0.5 } = opts ?? {};
  return {
    position: { x: target.center.x, y: target.halfExtents.y * 2 + 6, z: target.center.z },
    velocity: { x: 0, y: -speed, z: 0 },
    radius,
    mass,
    ttl: 6000,
  };
}

// ── Stress cases (preset world + impact plan + tuning) ───────────────────────
export interface StressCase {
  name: string;
  scenario: ComposedScenario;
  impactPlan: ImpactEvent[];
  coreOpts?: Record<string, unknown>;
  warmupFrames?: number;
  postImpactFrames?: number;
}

const scaled = (n: number, tier: SizeTier) => Math.max(1, Math.round(n * TIER_GRID[tier]));

/** Sparse grid of small towers — lots of independent, mostly-settled islands.
 *  Edge case for island skip-settled: hitting ONE building should leave the rest idle. */
function manyIslandsCity(tier: SizeTier): StressCase {
  const gx = scaled(6, tier);
  const gz = scaled(6, tier);
  const scenario = buildCityScenario({ gridX: gx, gridZ: gz, cellSize: 7, kinds: ['smallTower'], seed: 11 });
  const target = scenario.parameters.buildings[Math.floor(scenario.parameters.buildings.length / 2)];
  return {
    name: `manyIslands ${gx}x${gz}`,
    scenario,
    impactPlan: [{ frame: 0, projectiles: [horizontalHit(target, { mass: 20000, speed: 55 })] }],
    postImpactFrames: 180,
  };
}

/** Tightly packed mix of towers + walls — dense, overlapping debris fields.
 *  (High-rises are heavier and get their own preset, `highRiseDistrict`.) */
function denseMixedCity(tier: SizeTier): StressCase {
  const gx = scaled(4, tier);
  const gz = scaled(4, tier);
  const scenario = buildCityScenario({
    gridX: gx,
    gridZ: gz,
    cellSize: 6,
    kinds: ['smallTower', 'midTower', 'tallTower', 'lowWall'],
    seed: 23,
  });
  const b = scenario.parameters.buildings;
  // Carpet the city: a wave of projectiles sweeping across the front row.
  const projectiles: ProjectileSpawn[] = b
    .filter((x) => x.center.z >= 0)
    .map((x) => horizontalHit(x, { mass: 14000, speed: 50, radius: 0.35 }));
  return {
    name: `denseMixed ${gx}x${gz}`,
    scenario,
    impactPlan: [{ frame: 0, projectiles }],
    postImpactFrames: 240,
  };
}

/** Grid of mid-rise apartment buildings (heterogeneous skeleton/infill). */
function highRiseDistrict(tier: SizeTier): StressCase {
  const gx = scaled(2, tier);
  const gz = scaled(2, tier);
  const scenario = buildCityScenario({ gridX: gx, gridZ: gz, cellSize: 16, kinds: ['miniHighRise'], seed: 37 });
  const target = scenario.parameters.buildings[0];
  return {
    name: `highRiseDistrict ${gx}x${gz}`,
    scenario,
    impactPlan: [{ frame: 0, projectiles: [horizontalHit(target, { mass: 34000, speed: 70, radius: 0.6, heightFrac: 0.4 })] }],
    postImpactFrames: 240,
  };
}

/** One tall tower at the back of a line of short towers, hit at the base from the
 *  side so it topples ONTO its neighbors — cascading destruction from a falling,
 *  partially-fractured building. */
function cascadeCity(tier: SizeTier): StressCase {
  const count = scaled(5, tier);
  const parts: ScenarioPart[] = [];
  const pitch = 3.0;
  // The faller, offset behind the row in +Z, leaning impact pushes it toward -Z.
  parts.push({ scenario: buildTowerScenario({ side: 4, stories: 18, totalMass: 9000 }), offset: { x: 0, y: 0, z: 3.0 }, tag: 'faller' });
  for (let i = 0; i < count; i++) {
    parts.push({
      scenario: buildTowerScenario({ side: 3, stories: 6, totalMass: 1800 }),
      offset: { x: 0, y: 0, z: -(i + 1) * pitch },
      tag: `neighbor-${i}`,
    });
  }
  const scenario = composeScenarios(parts);
  const faller = scenario.parameters.buildings[0];
  // Hit the faller low and hard from +Z so it tips toward the row (-Z).
  return {
    name: `cascade row of ${count + 1}`,
    scenario,
    impactPlan: [
      {
        frame: 0,
        projectiles: [
          {
            position: { x: 0, y: faller.halfExtents.y * 0.35, z: faller.center.z + faller.halfExtents.z + 5 },
            velocity: { x: 0, y: 0, z: -85 },
            radius: 0.6,
            mass: 60000,
            ttl: 8000,
          },
        ],
      },
    ],
    postImpactFrames: 360,
  };
}

/** A finely-segmented beam bridge (fractures into many deck chunks under load)
 *  flanked by tower clusters; drop a heavy load on the deck mid-span. We use the
 *  synchronous beam-bridge builder rather than the async/auto-bonded fractured
 *  bridge so the whole suite stays pure-data (no WASM at scenario-build time). */
function bridgeAndTowers(tier: SizeTier): StressCase {
  const clusters = scaled(2, tier);
  const parts: ScenarioPart[] = [];
  parts.push({
    scenario: buildBeamBridgeScenario({ spanSegments: 28, widthSegments: 6, thicknessLayers: 2, deckMass: 50000 }),
    offset: { x: 0, y: 0, z: 0 },
    tag: 'bridge',
  });
  for (let i = 0; i < clusters; i++) {
    const x = (i - (clusters - 1) / 2) * 8;
    parts.push({ scenario: buildTowerScenario({ side: 3, stories: 7, totalMass: 2200 }), offset: { x, y: 0, z: 14 }, tag: `tower-${i}` });
  }
  const scenario = composeScenarios(parts);
  const bridge = scenario.parameters.buildings[0];
  return {
    name: `bridge + ${clusters} towers`,
    scenario,
    impactPlan: [{ frame: 0, projectiles: [topDrop(bridge, { mass: 40000, speed: 45, radius: 0.6 })] }],
    postImpactFrames: 300,
  };
}

export const CITY_SCENARIOS: Record<string, (tier: SizeTier) => StressCase> = {
  manyIslands: manyIslandsCity,
  denseMixed: denseMixedCity,
  highRiseDistrict,
  cascade: cascadeCity,
  bridgeAndTowers,
};

export const CITY_SCENARIO_NAMES = Object.keys(CITY_SCENARIOS);
