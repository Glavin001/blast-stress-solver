/**
 * Scenario composer — places several independent {@link ScenarioDesc} structures
 * into a single world so we can stress-test "mini city" scale.
 *
 * INTERNAL / unstable. Lives under `src/tests/` and is intentionally NOT exported
 * from `src/scenarios/index.ts` — it is a profiling/test utility, not public API.
 *
 * Why this is needed: every shipped scenario builder centers its structure at the
 * origin and either tags nodes with a uniform `spacing` (grid builders) or a
 * per-node `parameters.fragmentSizes` array (heterogeneous builders like the
 * high-rise). Collider sizing resolves via `resolveScenarioNodeSize`
 * (fragmentSizes[i] -> spacing -> default). To place buildings of DIFFERENT sizes
 * in one world we therefore have to flatten every sub-scenario down to an explicit
 * per-node `fragmentSizes` array — that is the only mechanism that preserves
 * heterogeneous sizing once a single top-level `spacing` no longer applies.
 *
 * Buildings stay independent simply by never creating a bond between them, so each
 * is its own solver island (anchored to its own static, mass-0 base nodes).
 */
import type { ScenarioBond, ScenarioDesc, ScenarioNode, Vec3 } from '../../rapier/types';
import { resolveScenarioNodeSize } from '../../rapier/scenario';

/** A single placed structure. Rotation is axis-aligned only so box colliders stay
 *  undistorted (we only ever swap X/Z extents for 90/270). */
export interface ScenarioPart {
  scenario: ScenarioDesc;
  /** World-space translation applied to every centroid (default origin). */
  offset?: Vec3;
  /** Yaw in degrees, axis-aligned only. Default 0. */
  rotateY?: 0 | 90 | 180 | 270;
  /** Label used for per-building attribution in the report. */
  tag?: string;
}

/** Where each building landed in the merged node/bond arrays, plus its placed AABB
 *  (handy for aiming projectiles at a specific building from the report/harness). */
export interface BuildingRange {
  tag: string;
  nodeStart: number;
  nodeCount: number;
  bondStart: number;
  bondCount: number;
  /** Count of dynamic (mass > 0) nodes — i.e. destructible chunks. */
  dynamicNodes: number;
  center: Vec3;
  halfExtents: Vec3;
}

export interface ComposedScenario extends ScenarioDesc {
  parameters: {
    fragmentSizes: Vec3[];
    buildings: BuildingRange[];
    [key: string]: unknown;
  };
}

/** Rotate a vector about +Y by an axis-aligned yaw. About +Y: x' = x cosθ + z sinθ,
 *  z' = -x sinθ + z cosθ. */
function rotateYVec(p: Vec3, deg: 0 | 90 | 180 | 270): Vec3 {
  switch (deg) {
    case 90:
      return { x: p.z, y: p.y, z: -p.x };
    case 180:
      return { x: -p.x, y: p.y, z: -p.z };
    case 270:
      return { x: -p.z, y: p.y, z: p.x };
    default:
      return { x: p.x, y: p.y, z: p.z };
  }
}

/** Collider box extents swap X/Z under a quarter turn. */
function rotateYSize(s: Vec3, deg: 0 | 90 | 180 | 270): Vec3 {
  return deg === 90 || deg === 270 ? { x: s.z, y: s.y, z: s.x } : { x: s.x, y: s.y, z: s.z };
}

/**
 * Concatenate + reindex several scenarios into one world.
 *
 * - Node/bond indices are remapped by a running offset.
 * - Centroids/bond-centroids/normals are rotated (axis-aligned) then translated.
 * - Each sub-scenario's per-node size is resolved via {@link resolveScenarioNodeSize}
 *   and flattened into one combined `parameters.fragmentSizes`.
 * - No cross-building bonds are created, so every part is its own island.
 */
export function composeScenarios(parts: ScenarioPart[]): ComposedScenario {
  const nodes: ScenarioNode[] = [];
  const bonds: ScenarioBond[] = [];
  const fragmentSizes: Vec3[] = [];
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];
  const buildings: BuildingRange[] = [];

  // Keep grid coordinates unique across buildings (sizing ignores them, but some
  // adapters key off them). A large stride per building avoids collisions.
  let gridStride = 0;

  parts.forEach((part, partIndex) => {
    const { scenario, offset = { x: 0, y: 0, z: 0 }, rotateY = 0, tag = `b${partIndex}` } = part;
    const nodeStart = nodes.length;
    const bondStart = bonds.length;
    const srcGrid = scenario.gridCoordinates;
    let dynamicNodes = 0;

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (let i = 0; i < scenario.nodes.length; i++) {
      const src = scenario.nodes[i];
      const rc = rotateYVec(src.centroid, rotateY);
      const centroid: Vec3 = { x: rc.x + offset.x, y: rc.y + offset.y, z: rc.z + offset.z };
      nodes.push({ centroid, mass: src.mass, volume: src.volume });
      if (src.mass > 0) dynamicNodes++;

      const size = rotateYSize(resolveScenarioNodeSize(i, scenario), rotateY);
      fragmentSizes.push(size);

      const g = srcGrid?.[i];
      gridCoordinates.push(
        g
          ? { ix: g.ix + gridStride, iy: g.iy, iz: g.iz + gridStride }
          : { ix: gridStride, iy: 0, iz: gridStride },
      );

      const hx = size.x / 2;
      const hy = size.y / 2;
      const hz = size.z / 2;
      minX = Math.min(minX, centroid.x - hx);
      maxX = Math.max(maxX, centroid.x + hx);
      minY = Math.min(minY, centroid.y - hy);
      maxY = Math.max(maxY, centroid.y + hy);
      minZ = Math.min(minZ, centroid.z - hz);
      maxZ = Math.max(maxZ, centroid.z + hz);
    }

    for (const b of scenario.bonds) {
      const rc = rotateYVec(b.centroid, rotateY);
      bonds.push({
        node0: b.node0 + nodeStart,
        node1: b.node1 + nodeStart,
        centroid: { x: rc.x + offset.x, y: rc.y + offset.y, z: rc.z + offset.z },
        normal: rotateYVec(b.normal, rotateY),
        area: b.area,
      });
    }

    buildings.push({
      tag,
      nodeStart,
      nodeCount: scenario.nodes.length,
      bondStart,
      bondCount: scenario.bonds.length,
      dynamicNodes,
      center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2, z: (minZ + maxZ) / 2 },
      halfExtents: { x: (maxX - minX) / 2, y: (maxY - minY) / 2, z: (maxZ - minZ) / 2 },
    });

    gridStride += 1000;
  });

  // No top-level `spacing`: the world is heterogeneous and every node's size is
  // carried explicitly in fragmentSizes.
  return { nodes, bonds, gridCoordinates, parameters: { fragmentSizes, buildings } };
}
