/**
 * glb-vehicle.ts — turn an arbitrary GLB model into a destructible scenario with
 * a *hierarchy* of bond strengths.
 *
 * Pipeline:
 *   1. extractVehicleParts(root)   — walk the loaded glTF scene, bake each mesh
 *      to world space, drop the source ground plane, and record name/size/colour.
 *   2. classifyVehiclePart(...)     — assign every part a structural role
 *      (frame / wheel / panel / cargo / accessory). Mirrors scripts/analyze-glb.mjs.
 *   3. buildVehicleScenario(parts)  — convert parts to FragmentInfo, optionally
 *      Voronoi-fracture the structural parts into chunks, detect bonds by
 *      proximity, then scale every bond's area by a role-pair multiplier so the
 *      skeleton holds while the payload tears off. Returns a ScenarioDesc plus
 *      per-node colours for the "show the hierarchy" view.
 *
 * Strength is encoded as bond *area* (stress = force / area, so a bigger area is
 * a stronger joint). The skeleton-to-skeleton joints get the largest area; the
 * strapped-on cargo gets the smallest, so it sheds first. A global materialScale
 * (set on the solver) plus these per-bond multipliers give the full hierarchy the
 * brief asked for: wheels strong, axle→body stronger, chassis strongest, exterior
 * payload weak — and a hard enough hit still shatters the whole thing.
 *
 * Only depends on already-published blast-stress-solver/three exports, so it
 * builds with esbuild/tsc without rebuilding the WASM library.
 */
import * as THREE from 'three';
import {
  buildScenarioFromFragments,
  fractureGeometryAsync,
  recenterGeometry,
  type FragmentInfo,
} from 'blast-stress-solver/three';

// ── Roles & visual identity ──────────────────────────────────────────────────

export type VehiclePartRole = 'frame' | 'wheel' | 'panel' | 'cargo' | 'accessory';

/** Per-role colour for the "hierarchy" view (warmer/redder = stronger skeleton). */
export const ROLE_COLORS: Record<VehiclePartRole, number> = {
  frame: 0xd6453c, // crimson — the strong roll-cage / chassis skeleton
  wheel: 0x3a4150, // slate — wheels, firmly bolted to the frame
  panel: 0xe08a2e, // orange — body panels / floor pans / seats
  cargo: 0xd9c27a, // tan — strapped-on payload (sheds first)
  accessory: 0x9aa6b2, // light grey — loose bits (chain, bucket — first to go)
};

export const ROLE_LABELS: Record<VehiclePartRole, string> = {
  frame: 'Frame / roll cage',
  wheel: 'Wheels',
  panel: 'Body panels',
  cargo: 'Cargo',
  accessory: 'Accessories',
};

// ── Bond-strength hierarchy (area multipliers) ───────────────────────────────
//
// Symmetric role-pair multipliers applied to each bond's area. >1 strengthens a
// joint, <1 weakens it. This table *is* the destructible hierarchy.
const INTER_ROLE_MULTIPLIER: Record<string, number> = {
  'frame|frame': 8.0, // skeleton welds — strongest; keeps the shell together
  'frame|wheel': 5.0, // hub/axle to chassis — very strong
  'frame|panel': 3.0, // panels bolted to the frame
  'frame|cargo': 0.25, // payload lashed to the frame — weak, sheds first
  'frame|accessory': 0.12,
  'wheel|wheel': 1.5,
  'wheel|panel': 2.0,
  'wheel|cargo': 0.2,
  'wheel|accessory': 0.1,
  'panel|panel': 2.0,
  'panel|cargo': 0.3,
  'panel|accessory': 0.15,
  'cargo|cargo': 0.35, // payload items tied to each other
  'cargo|accessory': 0.18,
  'accessory|accessory': 0.1,
};

/** Internal joints between chunks of the *same* fractured part. */
const INTERNAL_ROLE_MULTIPLIER: Record<VehiclePartRole, number> = {
  frame: 6.0,
  wheel: 4.0,
  panel: 2.5,
  cargo: 1.5,
  accessory: 1.2,
};

function interRoleMultiplier(a: VehiclePartRole, b: VehiclePartRole): number {
  return INTER_ROLE_MULTIPLIER[`${a}|${b}`] ?? INTER_ROLE_MULTIPLIER[`${b}|${a}`] ?? 1.0;
}

// ── Classification (kept in sync with scripts/analyze-glb.mjs) ────────────────

export type VehicleBounds = {
  /** min corner of the whole-vehicle bbox (ground excluded). */
  lo: THREE.Vector3;
  /** size of the whole-vehicle bbox. */
  size: THREE.Vector3;
};

/**
 * Assign a structural role to a part from its name, size and position.
 * Name keywords win when present; otherwise geometry/position cues are used so
 * generically-named parts (Circle/Plane/Cube…) still classify sensibly.
 */
export function classifyVehiclePart(
  name: string,
  size: THREE.Vector3,
  center: THREE.Vector3,
  bounds: VehicleBounds,
): VehiclePartRole {
  const n = name.toLowerCase();
  const maxDim = Math.max(size.x, size.y, size.z);
  const minDim = Math.min(size.x, size.y, size.z);
  const carLen = Math.max(bounds.size.x, bounds.size.z) || 1;
  const carHeight = bounds.size.y || 1;
  const relY = (center.y - bounds.lo.y) / carHeight; // 0 = floor, 1 = roof
  const spans = maxDim / carLen;

  // 1) Name keywords (strongest signal).
  if (/wheel|tire|tyre|\brim\b/.test(n)) return 'wheel';
  if (/cage|chassis|\bframe\b|\bbody|rollbar|roll_?cage|rollcage/.test(n)) return 'frame';
  if (/door|hood|bonnet|fender|bumper|panel|windshield|windscreen|glass|window/.test(n)) return 'panel';
  if (/seat|interior|dash|steer/.test(n)) return 'panel';
  if (/engine|motor|axle|suspension|exhaust|drivetrain|gearbox/.test(n)) return 'frame';
  if (/^aset_|barrel|drum|crate|\bbox\b|\blog\b|rock|tarp|jerry|canister|\bcan\b|bag|sack|cargo|container|plastic|wood|fabric|cloth/.test(n)) {
    return 'cargo';
  }
  if (/chain|rope|cable|hook|bucket|pipe|wire|strap|antenna/.test(n)) return 'accessory';

  // 2) Geometry / position fallback for generic names.
  if (spans > 0.6) return 'frame'; // long central members = skeleton
  if (relY < 0.38 && minDim > maxDim * 0.4 && spans < 0.32) return 'wheel';
  if (minDim < maxDim * 0.18) return 'panel'; // thin flat slab
  if (relY > 0.5 && spans < 0.3) return 'cargo'; // small + perched high
  return 'accessory';
}

// ── Part extraction from a loaded glTF scene ─────────────────────────────────

export type VehiclePart = {
  name: string;
  /** Geometry baked into world space (NOT recentred yet). */
  geometry: THREE.BufferGeometry;
  center: THREE.Vector3;
  size: THREE.Vector3;
  /** Base colour pulled from the source material (for an optional texture-ish view). */
  materialColor: THREE.Color;
  role: VehiclePartRole;
};

const _box = new THREE.Box3();
const _v = new THREE.Vector3();

/**
 * Walk a loaded glTF scene and return one VehiclePart per mesh, with geometry
 * baked to world space. Oversized planes (the source scene's ground/backdrop)
 * are dropped. Roles are assigned after the whole-vehicle bounds are known.
 */
export function extractVehicleParts(root: THREE.Object3D): {
  parts: VehiclePart[];
  bounds: VehicleBounds;
} {
  root.updateMatrixWorld(true);

  type Raw = Omit<VehiclePart, 'role'>;
  const raw: Raw[] = [];
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;

    const geometry = mesh.geometry.clone();
    geometry.applyMatrix4(mesh.matrixWorld); // bake world transform
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox as THREE.Box3;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    bbox.getSize(size);
    bbox.getCenter(center);

    // material base colour (best-effort)
    let color = new THREE.Color(0xb0b0b0);
    const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    if (mat && (mat as THREE.MeshStandardMaterial).color) {
      color = (mat as THREE.MeshStandardMaterial).color.clone();
    }

    raw.push({ name: mesh.name || `part_${raw.length}`, geometry, center, size, materialColor: color });
  });

  // whole-vehicle bounds, excluding any oversized ground/backdrop plane
  _box.makeEmpty();
  let carMaxDim = 0;
  for (const r of raw) carMaxDim = Math.max(carMaxDim, r.size.x, r.size.y, r.size.z);
  // first pass to estimate car length without the ground plane
  let estLen = 0;
  for (const r of raw) {
    const m = Math.max(r.size.x, r.size.y, r.size.z);
    if (m < carMaxDim) estLen = Math.max(estLen, m);
  }
  const groundThreshold = Math.max(estLen * 3, 20);

  const kept: Raw[] = [];
  for (const r of raw) {
    if (Math.max(r.size.x, r.size.y, r.size.z) > groundThreshold) {
      r.geometry.dispose();
      continue; // ground plane / backdrop
    }
    kept.push(r);
    _box.expandByPoint(r.center.clone().sub(_v.copy(r.size).multiplyScalar(0.5)));
    _box.expandByPoint(r.center.clone().add(_v.copy(r.size).multiplyScalar(0.5)));
  }

  const bounds: VehicleBounds = {
    lo: _box.min.clone(),
    size: _box.getSize(new THREE.Vector3()),
  };

  const parts: VehiclePart[] = kept.map((r) => ({
    ...r,
    role: classifyVehiclePart(r.name, r.size, r.center, bounds),
  }));

  return { parts, bounds };
}

// ── Scenario assembly ────────────────────────────────────────────────────────

export type BuildVehicleScenarioOptions = {
  /** Total mass (kg) spread across all parts by volume. Default 1500. */
  totalMass?: number;
  /**
   * Voronoi-fracture structural parts (frame/panel/wheel) into this many chunks
   * each so a hard hit shatters them. 0 (default) keeps every part intact —
   * robust, and still sheds cargo / breaks the skeleton joints. Chunks of the
   * same part get strong internal bonds so light hits keep the shell together.
   */
  structuralFractureChunks?: number;
  /** Pre-loaded three-pinata module (required only when fracturing). */
  pinata?: Parameters<typeof fractureGeometryAsync>[1]['pinata'];
  /** Lift so the lowest point sits at this Y (rest on the ground). Default 0.03. */
  groundGap?: number;
  /** Re-centre the vehicle on X/Z to the world origin. Default true. */
  centerOnOrigin?: boolean;
};

export type VehicleScenarioResult = {
  scenario: any; // ScenarioDesc
  /** Per-node colour (indexed by node) for createDestructibleThreeBundle. */
  nodeColors: THREE.Color[];
  /** Per-node role (indexed by node), for HUD / inspection. */
  nodeRoles: VehiclePartRole[];
  summary: {
    parts: number;
    nodes: number;
    bonds: number;
    roleCounts: Record<VehiclePartRole, number>;
  };
};

type FragMeta = { partId: number; role: VehiclePartRole };

const STRUCTURAL_ROLES: ReadonlySet<VehiclePartRole> = new Set<VehiclePartRole>(['frame', 'panel', 'wheel']);

/**
 * Build a destructible ScenarioDesc from classified vehicle parts.
 * Async because optional Voronoi fracturing loads three-pinata.
 */
export async function buildVehicleScenario(
  parts: VehiclePart[],
  bounds: VehicleBounds,
  options: BuildVehicleScenarioOptions = {},
): Promise<VehicleScenarioResult> {
  const totalMass = options.totalMass ?? 1500;
  const fractureChunks = Math.max(0, Math.round(options.structuralFractureChunks ?? 0));
  const groundGap = options.groundGap ?? 0.03;
  const centerOnOrigin = options.centerOnOrigin ?? true;

  // World-space offset so the vehicle rests on the ground at the origin.
  const offset = new THREE.Vector3(
    centerOnOrigin ? -(bounds.lo.x + bounds.size.x * 0.5) : 0,
    -bounds.lo.y + groundGap,
    centerOnOrigin ? -(bounds.lo.z + bounds.size.z * 0.5) : 0,
  );

  const fragments: FragmentInfo[] = [];
  const meta: FragMeta[] = [];

  for (let partId = 0; partId < parts.length; partId++) {
    const part = parts[partId];

    // Optionally Voronoi-fracture structural parts into chunks. The part geometry
    // is already baked into world space, and fractureGeometry preserves that frame
    // (fragment.worldPosition = worldOffset + pieceWorldPos), so we pass only the
    // global vehicle offset and use the returned worldPosition directly.
    let fractured = false;
    if (fractureChunks > 1 && STRUCTURAL_ROLES.has(part.role) && options.pinata) {
      try {
        const chunks = await fractureGeometryAsync(part.geometry, {
          pinata: options.pinata,
          fragmentCount: fractureChunks,
          voronoiMode: '3D',
          worldOffset: { x: offset.x, y: offset.y, z: offset.z },
        });
        if (chunks.length > 1) {
          part.geometry.dispose(); // replaced by fresh per-chunk geometries
          for (const c of chunks) {
            fragments.push({
              worldPosition: c.worldPosition,
              halfExtents: c.halfExtents,
              geometry: c.geometry,
              isSupport: false,
            });
            meta.push({ partId, role: part.role });
          }
          fractured = true;
        }
      } catch (err) {
        // Non-manifold / awkward mesh: fall back to keeping the part intact.
        console.warn(`[glb-vehicle] fracture failed for "${part.name}", keeping intact:`, err);
      }
    }
    if (fractured) continue;

    // Intact part: recentre the world-space geometry to its own origin and use
    // its centroid (plus the global offset) as the fragment world position.
    const geom = part.geometry;
    const { offset: localCenter } = recenterGeometry(geom);
    geom.computeBoundingBox();
    const size = (geom.boundingBox as THREE.Box3).getSize(new THREE.Vector3());
    fragments.push({
      worldPosition: {
        x: localCenter.x + offset.x,
        y: localCenter.y + offset.y,
        z: localCenter.z + offset.z,
      },
      halfExtents: { x: size.x * 0.5, y: size.y * 0.5, z: size.z * 0.5 },
      geometry: geom,
      isSupport: false,
    });
    meta.push({ partId, role: part.role });
  }

  // Detect bonds by proximity (the parts overlap/touch, so this connects them),
  // and normalise areas to a consistent base so the role multipliers — not the
  // incidental overlap geometry — drive relative joint strength.
  const scenario = buildScenarioFromFragments(fragments, {
    totalMass,
    areaNormalization: 'uniform',
    dimensions: { x: bounds.size.x, y: bounds.size.y, z: bounds.size.z },
    bondDetectionOptions: {
      // Be generous: strapped cargo can rest with a small visible gap.
      toleranceFactor: 0.25,
      minGapTolerance: 0.02,
      minOverlapRatio: 0.12,
    },
  });

  // A representative base area (median of the normalised areas) used for the
  // stitch bonds we may add for connectivity.
  const baseArea = medianArea(scenario.bonds) || 0.1;

  // Scale each bond by its role-pair (or same-part internal) multiplier.
  for (const bond of scenario.bonds) {
    const a = meta[bond.node0];
    const b = meta[bond.node1];
    if (!a || !b) continue;
    const mult =
      a.partId === b.partId
        ? INTERNAL_ROLE_MULTIPLIER[a.role]
        : interRoleMultiplier(a.role, b.role);
    bond.area = Math.max(bond.area * mult, 1e-6);
  }

  // Guarantee the bond graph is a single connected component, otherwise an
  // isolated part would split off and drop on the first solver step.
  stitchConnectivity(scenario, fragments, meta, baseArea);

  // Per-node colours + roles for the hierarchy view / HUD.
  const nodeColors: THREE.Color[] = meta.map((m) => new THREE.Color(ROLE_COLORS[m.role]));
  const nodeRoles: VehiclePartRole[] = meta.map((m) => m.role);

  const roleCounts: Record<VehiclePartRole, number> = {
    frame: 0, wheel: 0, panel: 0, cargo: 0, accessory: 0,
  };
  for (const p of parts) roleCounts[p.role]++;

  return {
    scenario,
    nodeColors,
    nodeRoles,
    summary: {
      parts: parts.length,
      nodes: scenario.nodes.length,
      bonds: scenario.bonds.length,
      roleCounts,
    },
  };
}

// ── Connectivity helpers ─────────────────────────────────────────────────────

function medianArea(bonds: Array<{ area: number }>): number {
  if (!bonds.length) return 0;
  const a = bonds.map((b) => b.area).sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

/**
 * Union-find over the bond graph; for every component beyond the first, add a
 * bond between the closest pair of fragments bridging it to an already-connected
 * component. Stitch bonds use the role-pair multiplier so the joins respect the
 * same hierarchy.
 */
function stitchConnectivity(
  scenario: any,
  fragments: FragmentInfo[],
  meta: FragMeta[],
  baseArea: number,
): void {
  const n = fragments.length;
  if (n <= 1) return;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  const union = (a: number, b: number) => {
    parent[find(a)] = find(b);
  };
  for (const bond of scenario.bonds) union(bond.node0, bond.node1);

  // Group node indices by component root.
  const comps = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    (comps.get(r) ?? comps.set(r, []).get(r)!).push(i);
  }
  if (comps.size <= 1) return;

  // Largest component is the "main" body; attach the rest to it (or to the
  // growing connected set) by nearest fragment pair.
  const groups = [...comps.values()].sort((a, b) => b.length - a.length);
  const connected = [...groups[0]];
  for (let g = 1; g < groups.length; g++) {
    const group = groups[g];
    let best = { i: -1, j: -1, d2: Infinity };
    for (const i of group) {
      const pi = fragments[i].worldPosition;
      for (const j of connected) {
        const pj = fragments[j].worldPosition;
        const d2 = (pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2;
        if (d2 < best.d2) best = { i, j, d2 };
      }
    }
    if (best.i >= 0) {
      const a = meta[best.i];
      const b = meta[best.j];
      const mult = a.partId === b.partId
        ? INTERNAL_ROLE_MULTIPLIER[a.role]
        : interRoleMultiplier(a.role, b.role);
      const pi = fragments[best.i].worldPosition;
      const pj = fragments[best.j].worldPosition;
      const dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
      const len = Math.hypot(dx, dy, dz) || 1;
      scenario.bonds.push({
        node0: best.i,
        node1: best.j,
        centroid: { x: (pi.x + pj.x) / 2, y: (pi.y + pj.y) / 2, z: (pi.z + pj.z) / 2 },
        normal: { x: dx / len, y: dy / len, z: dz / len },
        area: Math.max(baseArea * mult, 1e-6),
      });
      // merge the group into the connected set
      for (const i of group) connected.push(i);
    }
  }
}
