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
  buildScenarioFromFragmentsAsync,
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
 * Split a triangle mesh into its connected components (topological islands).
 *
 * Artists routinely model several physically-separate shapes (e.g. the tubes of
 * a roll cage) as a single mesh. Treated as one node, the runtime builds ONE
 * convex-hull collider that spans the gaps between those shapes — a nonsensical
 * blob. Splitting first means each real shape becomes its own piece with a tight
 * hull, and auto-bonding can reconnect the ones that actually touch.
 *
 * Vertices are welded by quantized position so coincident-but-duplicated verts
 * (common in GLB exports) count as connected, while shapes separated by a gap
 * become distinct geometries. Tiny stray islands are merged into the nearest
 * kept island to avoid a flood of micro-nodes.
 */
export function splitConnectedComponents(
  geometry: THREE.BufferGeometry,
  opts: { weldTol?: number; minTriangles?: number; maxComponents?: number } = {},
): THREE.BufferGeometry[] {
  const weldTol = opts.weldTol ?? 1e-4;
  const minTriangles = opts.minTriangles ?? 6;
  const maxComponents = opts.maxComponents ?? 24;

  const pos = geometry.getAttribute('position') as THREE.BufferAttribute | undefined;
  if (!pos) return [geometry];
  const index = geometry.getIndex();
  const triCount = Math.floor((index ? index.count : pos.count) / 3);
  if (triCount < 2) return [geometry];
  const vert = (i: number) => (index ? index.getX(i) : i);

  // Weld vertices by quantized position → canonical id.
  const inv = 1 / weldTol;
  const canonOf = new Int32Array(pos.count);
  const keyToCanon = new Map<string, number>();
  let canonCount = 0;
  for (let i = 0; i < pos.count; i++) {
    const key =
      Math.round(pos.getX(i) * inv) + '_' +
      Math.round(pos.getY(i) * inv) + '_' +
      Math.round(pos.getZ(i) * inv);
    let c = keyToCanon.get(key);
    if (c === undefined) { c = canonCount++; keyToCanon.set(key, c); }
    canonOf[i] = c;
  }

  // Union-find over canonical verts, unioning the 3 verts of each triangle.
  const parent = new Int32Array(canonCount);
  for (let i = 0; i < canonCount; i++) parent[i] = i;
  const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (a: number, b: number) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
  for (let t = 0; t < triCount; t++) {
    const a = canonOf[vert(t * 3)], b = canonOf[vert(t * 3 + 1)], c = canonOf[vert(t * 3 + 2)];
    union(a, b); union(b, c);
  }

  // Group triangle indices by component root.
  const groups = new Map<number, number[]>();
  for (let t = 0; t < triCount; t++) {
    const root = find(canonOf[vert(t * 3)]);
    let g = groups.get(root);
    if (!g) { g = []; groups.set(root, g); }
    g.push(t);
  }
  if (groups.size <= 1) return [geometry];

  const nrm = geometry.getAttribute('normal') as THREE.BufferAttribute | undefined;
  const buildGeom = (tris: number[]): THREE.BufferGeometry => {
    const positions = new Float32Array(tris.length * 9);
    const normals = nrm ? new Float32Array(tris.length * 9) : null;
    let o = 0;
    for (const t of tris) {
      for (let k = 0; k < 3; k++) {
        const v = vert(t * 3 + k);
        positions[o] = pos.getX(v); positions[o + 1] = pos.getY(v); positions[o + 2] = pos.getZ(v);
        if (normals) { normals[o] = nrm!.getX(v); normals[o + 1] = nrm!.getY(v); normals[o + 2] = nrm!.getZ(v); }
        o += 3;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (normals) g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    else g.computeVertexNormals();
    return g;
  };
  const centroidOf = (tris: number[]): THREE.Vector3 => {
    const c = new THREE.Vector3();
    for (const t of tris) for (let k = 0; k < 3; k++) {
      const v = vert(t * 3 + k);
      c.x += pos.getX(v); c.y += pos.getY(v); c.z += pos.getZ(v);
    }
    return c.multiplyScalar(1 / (tris.length * 3));
  };

  // Keep the largest components; merge tiny/overflow ones into the nearest kept.
  const sorted = [...groups.values()].sort((a, b) => b.length - a.length);
  const kept: number[][] = [];
  const keptCentroids: THREE.Vector3[] = [];
  for (const tris of sorted) {
    if (kept.length < maxComponents && (tris.length >= minTriangles || kept.length === 0)) {
      kept.push(tris.slice());
      keptCentroids.push(centroidOf(tris));
    } else {
      const c = centroidOf(tris);
      let best = 0, bestD = Infinity;
      for (let i = 0; i < keptCentroids.length; i++) {
        const d = c.distanceToSquared(keptCentroids[i]);
        if (d < bestD) { bestD = d; best = i; }
      }
      for (const t of tris) kept[best].push(t);
    }
  }
  if (kept.length <= 1) return [geometry];
  return kept.map(buildGeom);
}

/**
 * Walk a loaded glTF scene and return VehicleParts with geometry baked to world
 * space. Each mesh is split into its connected-component islands (so separate
 * shapes sharing a mesh become separate pieces). Oversized planes (the source
 * scene's ground/backdrop) are dropped. Each island inherits its parent mesh's
 * role so classification stays semantic.
 */
export function extractVehicleParts(
  root: THREE.Object3D,
  opts: { splitComponents?: boolean } = {},
): { parts: VehiclePart[]; bounds: VehicleBounds } {
  root.updateMatrixWorld(true);
  const splitComponents = opts.splitComponents ?? true;

  // Collect baked meshes first (need vehicle-wide info to drop the ground plane).
  type MeshRec = { name: string; baked: THREE.BufferGeometry; center: THREE.Vector3; size: THREE.Vector3; color: THREE.Color };
  const meshes: MeshRec[] = [];
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const baked = mesh.geometry.clone();
    baked.applyMatrix4(mesh.matrixWorld);
    baked.computeBoundingBox();
    const bbox = baked.boundingBox as THREE.Box3;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    bbox.getSize(size);
    bbox.getCenter(center);
    let color = new THREE.Color(0xb0b0b0);
    const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    if (mat && (mat as THREE.MeshStandardMaterial).color) color = (mat as THREE.MeshStandardMaterial).color.clone();
    meshes.push({ name: mesh.name || `part_${meshes.length}`, baked, center, size, color });
  });

  // Drop any oversized ground/backdrop plane (much bigger than the vehicle).
  let maxDim = 0;
  for (const m of meshes) maxDim = Math.max(maxDim, m.size.x, m.size.y, m.size.z);
  let estLen = 0;
  for (const m of meshes) {
    const d = Math.max(m.size.x, m.size.y, m.size.z);
    if (d < maxDim) estLen = Math.max(estLen, d);
  }
  const groundThreshold = Math.max(estLen * 3, 20);
  const kept = meshes.filter((m) => {
    if (Math.max(m.size.x, m.size.y, m.size.z) > groundThreshold) { m.baked.dispose(); return false; }
    return true;
  });

  // Whole-vehicle bounds (from kept mesh bboxes) — used for classification and placement.
  _box.makeEmpty();
  for (const m of kept) {
    _box.expandByPoint(m.center.clone().sub(_v.copy(m.size).multiplyScalar(0.5)));
    _box.expandByPoint(m.center.clone().add(_v.copy(m.size).multiplyScalar(0.5)));
  }
  const bounds: VehicleBounds = { lo: _box.min.clone(), size: _box.getSize(new THREE.Vector3()) };

  // Classify each mesh, then split it into physical islands — EXCEPT roles that
  // should stay cohesive units (wheels are rubbery; they fall off whole rather
  // than separating into tire/rim/nuts). The frame is the key case for splitting
  // (welded but physically-separate cage tubes). Each island keeps its mesh role.
  const parts: VehiclePart[] = [];
  const makePart = (name: string, geometry: THREE.BufferGeometry, center: THREE.Vector3, size: THREE.Vector3, color: THREE.Color, role: VehiclePartRole) =>
    parts.push({ name, geometry, center, size, materialColor: color, role });

  for (const m of kept) {
    const role = classifyVehiclePart(m.name, m.size, m.center, bounds);
    const doSplit = splitComponents && !NO_SPLIT_ROLES.has(role);
    const islands = doSplit ? splitConnectedComponents(m.baked) : [m.baked];
    if (islands.length === 1 && islands[0] === m.baked) {
      makePart(m.name, m.baked, m.center.clone(), m.size.clone(), m.color, role);
    } else {
      m.baked.dispose(); // replaced by per-island geometries
      islands.forEach((g, i) => {
        g.computeBoundingBox();
        const s = new THREE.Vector3();
        const c = new THREE.Vector3();
        (g.boundingBox as THREE.Box3).getSize(s);
        (g.boundingBox as THREE.Box3).getCenter(c);
        makePart(`${m.name}#${i}`, g, c, s, m.color, role);
      });
    }
  }

  return { parts, bounds };
}

// ── Scenario assembly ────────────────────────────────────────────────────────

export type BuildVehicleScenarioOptions = {
  /** Total mass (kg) spread across all parts by volume. Default 1500. */
  totalMass?: number;
  /**
   * Voronoi-fracture *large* connected structural parts (frame/panel/wheel whose
   * longest side exceeds ~the cell size) into chunks roughly this many metres
   * across. This is essential for concave welded shapes like a roll cage: kept
   * whole, their ONE convex-hull collider spans the whole vehicle (a nonsensical
   * blob) and auto-bonds to everything from a central point. Fracturing gives
   * each chunk a tight local hull + local bonds, and chunks of the same part get
   * strong internal bonds so the shell holds together until hit hard. Set to 0 to
   * disable (parts kept whole). Default 0.6 m.
   */
  fractureCellSize?: number;
  /** Pre-loaded three-pinata module (required only when fracturing). */
  pinata?: Parameters<typeof fractureGeometryAsync>[1]['pinata'];
  /** Lift so the lowest point sits at this Y (rest on the ground). Default 0.03. */
  groundGap?: number;
  /** Re-centre the vehicle on X/Z to the world origin. Default true. */
  centerOnOrigin?: boolean;
  /**
   * Bond detection strategy:
   * - 'auto' (default): WASM triangle-surface bonding (createBondsFromTriangles) —
   *   connects parts where their meshes actually touch, with real contact
   *   area/normal/location. This is what makes the hierarchy come apart correctly.
   * - 'proximity': fast JS centroid/AABB-overlap bonding (fallback, no WASM).
   */
  bondMode?: 'auto' | 'proximity';
  /**
   * Max gap (m) between two parts' surfaces still treated as a contact bond, used
   * by 'auto' (average) bonding. Artist meshes touch/interpenetrate within a few
   * cm; default 0.06.
   */
  bondMaxSeparation?: number;
  /**
   * Per-role attachment-strength scale, multiplied into every bond touching a
   * part of that role (so e.g. roleStrength.cargo = 0.5 halves how strongly all
   * cargo is held). Default 1 for each. This is the main "make cargo weaker" knob.
   */
  roleStrength?: Partial<Record<VehiclePartRole, number>>;
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

// Roles eligible for Voronoi fracture when a part is large + concave. The frame
// (welded roll cage / chassis) is the main case; large flat panels too. Wheels,
// cargo and accessories are cohesive roundish/boxy props whose convex hull is
// already a fine collider, so they stay whole (and don't shatter unrealistically).
const FRACTURE_ROLES: ReadonlySet<VehiclePartRole> = new Set<VehiclePartRole>(['frame', 'panel']);

// Roles kept as ONE cohesive piece (never split into connected-component islands).
// A wheel is rubbery — it should fall off as a unit, not separate into tire / rim /
// nuts — so its whole mesh becomes a single (cylinder-ish hull) piece.
const NO_SPLIT_ROLES: ReadonlySet<VehiclePartRole> = new Set<VehiclePartRole>(['wheel']);

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
  const fractureCell = Math.max(0, options.fractureCellSize ?? 0.6);
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

    // Voronoi-fracture LARGE connected structural parts into chunks. The part
    // geometry is already baked into world space, and fractureGeometry preserves
    // that frame (fragment.worldPosition = worldOffset + pieceWorldPos), so we
    // pass only the global vehicle offset and use the returned worldPosition.
    const partMaxDim = Math.max(part.size.x, part.size.y, part.size.z);
    const shouldFracture =
      fractureCell > 0 &&
      FRACTURE_ROLES.has(part.role) &&
      !!options.pinata &&
      partMaxDim > Math.max(0.9, fractureCell * 1.6);
    let fractured = false;
    if (shouldFracture) {
      // Chunk count scales with the part's VOLUME so big parts break down to ~cell
      // size while small ones get only a few — with a floor by longest dimension so
      // long thin members (cage tubes) still get cut into segments along their length.
      const cell = fractureCell;
      const byVolume = (part.size.x * part.size.y * part.size.z) / (cell * cell * cell);
      const byLongest = partMaxDim / cell;
      const chunkCount = Math.max(3, Math.min(32, Math.round(Math.max(byVolume, byLongest))));
      try {
        const chunks = await fractureGeometryAsync(part.geometry, {
          pinata: options.pinata,
          fragmentCount: chunkCount,
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

  // Bond the parts. Default: WASM triangle-surface auto-bonding — it connects
  // parts where their meshes actually touch (wheel↔axle↔frame, cargo↔the surface
  // it rests on) with real contact area/normal/location, so the assembly comes
  // apart along real seams instead of a centroid star. areaNormalization 'none'
  // keeps the real contact areas; role multipliers below set material strength.
  const bondMode = options.bondMode ?? 'auto';
  const bondMaxSeparation = options.bondMaxSeparation ?? 0.06;
  const dimensions = { x: bounds.size.x, y: bounds.size.y, z: bounds.size.z };
  const proximityOptions = {
    // Fallback proximity detection (also used if auto-bonding yields nothing).
    toleranceFactor: 0.25,
    minGapTolerance: 0.02,
    minOverlapRatio: 0.12,
  };
  const scenario =
    bondMode === 'auto'
      ? await buildScenarioFromFragmentsAsync(fragments, {
          totalMass,
          bondMode: 'auto',
          areaNormalization: 'none', // keep real triangle-contact areas
          dimensions,
          bondDetectionOptions: proximityOptions,
          autoBondingOptions: { mode: 'average', maxSeparation: bondMaxSeparation, label: 'vehicle' },
        })
      : buildScenarioFromFragments(fragments, {
          totalMass,
          areaNormalization: 'uniform',
          dimensions,
          bondDetectionOptions: proximityOptions,
        });

  // A representative base area (median of the normalised areas) used for the
  // stitch bonds we may add for connectivity.
  const baseArea = medianArea(scenario.bonds) || 0.1;

  // Scale each bond by its role-pair (or same-part internal) multiplier, then by
  // the per-role attachment-strength knobs (both endpoints).
  const rs = options.roleStrength ?? {};
  const rsOf = (r: VehiclePartRole) => (rs[r] ?? 1);
  for (const bond of scenario.bonds) {
    const a = meta[bond.node0];
    const b = meta[bond.node1];
    if (!a || !b) continue;
    const mult =
      a.partId === b.partId
        ? INTERNAL_ROLE_MULTIPLIER[a.role]
        : interRoleMultiplier(a.role, b.role);
    bond.area = Math.max(bond.area * mult * rsOf(a.role) * rsOf(b.role), 1e-6);
  }

  // Guarantee the bond graph is a single connected component, otherwise an
  // isolated part would split off and drop on the first solver step.
  const stitched = stitchConnectivity(scenario, fragments, meta, baseArea);

  // ── Diagnostics ───────────────────────────────────────────────────────────
  // Bond distribution by role pair (verifies sensible connectivity).
  const bondsByRolePair: Record<string, number> = {};
  for (const bond of scenario.bonds) {
    const a = meta[bond.node0]?.role;
    const b = meta[bond.node1]?.role;
    if (!a || !b) continue;
    const key = [a, b].sort().join('↔');
    bondsByRolePair[key] = (bondsByRolePair[key] ?? 0) + 1;
  }
  let undefMeta = 0;
  let maxNode = -1;
  for (const bond of scenario.bonds) {
    if (!meta[bond.node0] || !meta[bond.node1]) undefMeta++;
    maxNode = Math.max(maxNode, bond.node0, bond.node1);
  }
  console.log(
    '[glb-vehicle] bonds by role pair:', JSON.stringify(bondsByRolePair),
    `| total=${scenario.bonds.length} nodes=${scenario.nodes.length} meta=${meta.length} maxNodeIdx=${maxNode} undefMeta=${undefMeta}`,
  );
  // Orphan stitches (centroid bonds) — these are the long "star" debug lines.
  console.log(`[glb-vehicle] orphan stitches added: ${stitched}`);
  // Parts whose collider spans a large fraction of the vehicle (likely a single
  // mesh holding multiple disconnected shapes — e.g. wheel-nuts across wheels).
  const carLen = Math.max(bounds.size.x, bounds.size.z) || 1;
  const spanning = fragments
    .map((f, i) => ({ i, role: meta[i].role, span: 2 * Math.max(f.halfExtents.x, f.halfExtents.y, f.halfExtents.z) }))
    .filter((p) => p.span > 0.45 * carLen)
    .sort((a, b) => b.span - a.span);
  if (spanning.length) {
    console.log(
      `[glb-vehicle] ${spanning.length} spanning parts (>${(0.45 * carLen).toFixed(2)}m):`,
      spanning.slice(0, 10).map((p) => `node${p.i}/${p.role}=${p.span.toFixed(2)}m`).join(', '),
    );
  }

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
): number {
  const n = fragments.length;
  if (n <= 1) return 0;
  let added = 0;
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
  if (comps.size <= 1) return 0;

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
      added++;
      // merge the group into the connected set
      for (const i of group) connected.push(i);
    }
  }
  return added;
}
