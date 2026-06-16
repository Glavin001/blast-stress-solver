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

/**
 * Internal joints between collider pieces of the *same* part. Cohesive props —
 * wheels, the bucket/chain (accessory) — must behave as ONE rigid body: their
 * pieces give a tight CoACD collider but never fracture apart, so their internal
 * bonds are effectively unbreakable. Frame and panels DO break into their pieces
 * (the destructible skeleton), so theirs are moderate (and further scaled by piece
 * thickness — thin bars weaker than thick rails). Cargo mostly holds (a crate
 * sheds as a unit) but can break under a hard hit.
 */
const UNBREAKABLE = 1000;
const INTERNAL_ROLE_MULTIPLIER: Record<VehiclePartRole, number> = {
  frame: 6.0,
  panel: 3.0,
  cargo: 8.0,
  wheel: UNBREAKABLE,
  accessory: UNBREAKABLE,
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
  /**
   * Original full-detail model parts (world-space geometry + role), used ONLY for
   * RENDERING. When supplied (asset path), each collider piece (node) keeps its
   * tight CoACD convex hull for COLLISION but draws a slice of the original
   * detailed mesh, so the car looks like the real model instead of faceted hulls.
   * Produced by `extractVehicleParts(gltfScene)`. Omit to render the hulls.
   */
  renderParts?: VehiclePart[];
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

  const partRoleCounts: Record<VehiclePartRole, number> = { frame: 0, wheel: 0, panel: 0, cargo: 0, accessory: 0 };
  for (const p of parts) partRoleCounts[p.role]++;
  return assembleVehicleScenario(fragments, meta, bounds, options, parts.length, partRoleCounts);
}

/**
 * Shared tail: bond the fragments, scale bond strength by the role hierarchy and by
 * piece geometry (thin members are weaker than thick ones), cap resting cargo,
 * stitch connectivity, and package the result. Used by both the GLB path
 * (buildVehicleScenario) and the pre-decomposed asset path.
 */
async function assembleVehicleScenario(
  fragments: FragmentInfo[],
  meta: FragMeta[],
  bounds: VehicleBounds,
  options: BuildVehicleScenarioOptions,
  partCount: number,
  roleCounts: Record<VehiclePartRole, number>,
): Promise<VehicleScenarioResult> {
  const totalMass = options.totalMass ?? 1500;
  const bondMode = options.bondMode ?? 'auto';
  const bondMaxSeparation = options.bondMaxSeparation ?? 0.06;
  const dimensions = { x: bounds.size.x, y: bounds.size.y, z: bounds.size.z };
  const proximityOptions = { toleranceFactor: 0.25, minGapTolerance: 0.02, minOverlapRatio: 0.12 };
  const scenario =
    bondMode === 'auto'
      ? await buildScenarioFromFragmentsAsync(fragments, {
          totalMass,
          bondMode: 'auto',
          areaNormalization: 'none',
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

  const baseArea = medianArea(scenario.bonds) || 0.1;

  // Geometry-aware strength: a bond is only as strong as the thinnest member it
  // joins (a thin metal bar should fail before a thick frame rail). thickness =
  // the piece's smallest cross-section (2*min half-extent); the bond is scaled by
  // the thinner of the two pieces, normalised so a ~10cm member is the baseline.
  const REF_THICK = 0.1; // m — reference cross-section (×1)
  const thicknessOf = (i: number) => {
    const h = fragments[i]?.halfExtents;
    return h ? 2 * Math.min(h.x, h.y, h.z) : REF_THICK;
  };

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
    // Thinner member governs; clamp so thin bars are weaker than thick rails but
    // not so weak they drop under their own weight (0.5×..2×).
    const thinFactor = Math.min(2, Math.max(0.5, Math.min(thicknessOf(bond.node0), thicknessOf(bond.node1)) / REF_THICK));
    bond.area = Math.max(bond.area * mult * rsOf(a.role) * rsOf(b.role) * thinFactor, 1e-6);
  }

  const cappedBonds = capRestingBonds(scenario, meta, { roles: ['cargo', 'accessory'], maxBondsPerNode: 1 });
  const stitched = stitchConnectivity(scenario, fragments, meta, baseArea);

  const bondsByRolePair: Record<string, number> = {};
  for (const bond of scenario.bonds) {
    const a = meta[bond.node0]?.role;
    const b = meta[bond.node1]?.role;
    if (!a || !b) continue;
    const key = [a, b].sort().join('↔');
    bondsByRolePair[key] = (bondsByRolePair[key] ?? 0) + 1;
  }
  console.log(
    '[glb-vehicle] bonds by role pair:', JSON.stringify(bondsByRolePair),
    `| total=${scenario.bonds.length} nodes=${scenario.nodes.length} parts=${partCount}`,
  );
  console.log(`[glb-vehicle] resting bonds capped: ${cappedBonds} | orphan stitches added: ${stitched}`);

  const nodeColors: THREE.Color[] = meta.map((m) => new THREE.Color(ROLE_COLORS[m.role]));
  const nodeRoles: VehiclePartRole[] = meta.map((m) => m.role);

  return {
    scenario,
    nodeColors,
    nodeRoles,
    summary: { parts: partCount, nodes: scenario.nodes.length, bonds: scenario.bonds.length, roleCounts },
  };
}

// ── Pre-decomposed asset (CoACD pipeline output) ─────────────────────────────

export type VehiclePiecesAsset = {
  source?: string;
  parts: Array<{
    name: string;
    centroid: [number, number, number];
    extents: [number, number, number];
    pieces: Array<{ vertices: number[][]; faces: number[][] }>;
  }>;
};

/**
 * Build the destructible scenario from a pre-decomposed pieces asset (produced by
 * scripts/glb-pipeline/build_destructible_asset.py: split → CoACD → clip so every
 * collider piece is a tight convex hull and no two pieces of different parts
 * overlap). Each piece becomes a node (render + collider); pieces of the same part
 * share a partId so they get strong INTERNAL bonds and behave as one unit until a
 * hard enough hit. This is the high-quality path — non-overlapping colliders mean
 * full debris collision doesn't explode.
 */
export async function buildVehicleScenarioFromAsset(
  asset: VehiclePiecesAsset,
  options: BuildVehicleScenarioOptions = {},
): Promise<VehicleScenarioResult> {
  const totalMass = options.totalMass ?? 1500;
  const groundGap = options.groundGap ?? 0.03;

  // Whole-vehicle bounds from part centroids ± extents.
  _box.makeEmpty();
  for (const p of asset.parts) {
    const c = new THREE.Vector3(p.centroid[0], p.centroid[1], p.centroid[2]);
    const h = new THREE.Vector3(p.extents[0], p.extents[1], p.extents[2]).multiplyScalar(0.5);
    _box.expandByPoint(c.clone().sub(h));
    _box.expandByPoint(c.clone().add(h));
  }
  const bounds: VehicleBounds = { lo: _box.min.clone(), size: _box.getSize(new THREE.Vector3()) };
  const offset = new THREE.Vector3(
    -(bounds.lo.x + bounds.size.x * 0.5),
    -bounds.lo.y + groundGap,
    -(bounds.lo.z + bounds.size.z * 0.5),
  );

  const fragments: FragmentInfo[] = [];
  const meta: FragMeta[] = [];
  const roleCounts: Record<VehiclePartRole, number> = { frame: 0, wheel: 0, panel: 0, cargo: 0, accessory: 0 };

  for (let partId = 0; partId < asset.parts.length; partId++) {
    const part = asset.parts[partId];
    const size = new THREE.Vector3(...part.extents);
    const center = new THREE.Vector3(...part.centroid);
    const role = classifyVehiclePart(part.name, size, center, bounds);
    roleCounts[role]++;

    // Wheels (3.b): a tyre must never shatter like glass and the 4 wheels must stay
    // separate. We keep the pipeline's tight, NON-OVERLAPPING CoACD hulls for the
    // wheel (so debris collision stays explosion-free) and make the wheel behave as
    // ONE rigid unit via UNBREAKABLE internal bonds (see INTERNAL_ROLE_MULTIPLIER).
    // The wheel still RENDERS as the smooth detailed tyre (render != collision), so
    // it never looks faceted. (No collider merge — that would reintroduce overlap
    // with the hub/brake parts nested inside the tyre.)
    for (const piece of part.pieces) {
      if (!piece.vertices?.length || !piece.faces?.length) continue;
      // Piece vertices are in GLB world space; recentre to the piece origin and
      // place it via worldPosition = pieceCentroid + global offset.
      let cx = 0, cy = 0, cz = 0;
      for (const v of piece.vertices) { cx += v[0]; cy += v[1]; cz += v[2]; }
      const n = piece.vertices.length;
      cx /= n; cy /= n; cz /= n;
      const pos = new Float32Array(n * 3);
      let minx = Infinity, miny = Infinity, minz = Infinity, maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
      for (let i = 0; i < n; i++) {
        const lx = piece.vertices[i][0] - cx, ly = piece.vertices[i][1] - cy, lz = piece.vertices[i][2] - cz;
        pos[i * 3] = lx; pos[i * 3 + 1] = ly; pos[i * 3 + 2] = lz;
        if (lx < minx) minx = lx; if (ly < miny) miny = ly; if (lz < minz) minz = lz;
        if (lx > maxx) maxx = lx; if (ly > maxy) maxy = ly; if (lz > maxz) maxz = lz;
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const idx: number[] = [];
      for (const f of piece.faces) { idx.push(f[0], f[1], f[2]); }
      geom.setIndex(idx);
      geom.computeVertexNormals();
      fragments.push({
        worldPosition: { x: cx + offset.x, y: cy + offset.y, z: cz + offset.z },
        halfExtents: { x: (maxx - minx) * 0.5, y: (maxy - miny) * 0.5, z: (maxz - minz) * 0.5 },
        geometry: geom,
        isSupport: false,
      });
      meta.push({ partId, role });
    }
  }

  const result = await assembleVehicleScenario(fragments, meta, bounds, options, asset.parts.length, roleCounts);

  // High-fidelity render: keep the tight CoACD hulls for COLLISION, but draw each
  // node with a slice of the original detailed model (render != collision).
  if (options.renderParts?.length) {
    try {
      attachDetailedRenderGeometry(result.scenario, meta, offset, options.renderParts);
    } catch (err) {
      console.warn('[glb-vehicle] detailed render attach failed; rendering hulls:', err);
    }
  }
  return result;
}

// ── High-fidelity render geometry (render != collision) ──────────────────────

/**
 * Replace each node's RENDER geometry with a slice of the original detailed model,
 * while preserving the CoACD convex hull as the node's COLLISION geometry.
 *
 * The runtime builds one convex-hull collider per node from its geometry, so using
 * the CoACD pieces as render geometry makes the car look like faceted hulls. Here we
 * move those hulls to `scenario.parameters.colliderGeometries` (the core prefers
 * that channel for collision) and rebuild `fragmentGeometries` (what the chunk mesh
 * draws) from the original model: every original triangle is assigned to the node
 * whose collider best owns it (nearest piece, role-preferred), so the union of the
 * slices reproduces the original mesh and each slice rides with its node when it
 * detaches. Nodes that collect no triangles keep their hull geometry.
 */
function attachDetailedRenderGeometry(
  scenario: any,
  meta: FragMeta[],
  offset: THREE.Vector3,
  renderParts: VehiclePart[],
): void {
  const nodes = scenario.nodes as Array<{ centroid: { x: number; y: number; z: number } }>;
  const nodeCount = nodes.length;
  if (!nodeCount) return;
  const hulls = (scenario.parameters?.fragmentGeometries ?? []) as THREE.BufferGeometry[];

  // Per-node world-space (pre-offset) centroid + role.
  const cx = new Float64Array(nodeCount);
  const cy = new Float64Array(nodeCount);
  const cz = new Float64Array(nodeCount);
  const roleOf: VehiclePartRole[] = new Array(nodeCount);
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < nodeCount; i++) {
    const c = nodes[i].centroid;
    const x = c.x - offset.x, y = c.y - offset.y, z = c.z - offset.z;
    cx[i] = x; cy[i] = y; cz[i] = z;
    roleOf[i] = meta[i]?.role ?? 'frame';
    if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
    if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
  }

  // Uniform grid over node centroids for fast nearest-node queries.
  const spanX = Math.max(1e-3, maxX - minX);
  const spanY = Math.max(1e-3, maxY - minY);
  const spanZ = Math.max(1e-3, maxZ - minZ);
  const maxSpan = Math.max(spanX, spanY, spanZ);
  const cell = Math.max(0.12, maxSpan / 24);
  const inv = 1 / cell;
  const gx = Math.max(1, Math.ceil(spanX * inv));
  const gy = Math.max(1, Math.ceil(spanY * inv));
  const gz = Math.max(1, Math.ceil(spanZ * inv));
  const cellOf = (x: number, y: number, z: number) => {
    let ix = Math.floor((x - minX) * inv); if (ix < 0) ix = 0; else if (ix >= gx) ix = gx - 1;
    let iy = Math.floor((y - minY) * inv); if (iy < 0) iy = 0; else if (iy >= gy) iy = gy - 1;
    let iz = Math.floor((z - minZ) * inv); if (iz < 0) iz = 0; else if (iz >= gz) iz = gz - 1;
    return (ix * gy + iy) * gz + iz;
  };
  const grid = new Map<number, number[]>();
  for (let i = 0; i < nodeCount; i++) {
    const k = cellOf(cx[i], cy[i], cz[i]);
    const list = grid.get(k);
    if (list) list.push(i); else grid.set(k, [i]);
  }

  /** Nearest node to (x,y,z): prefer the same role, fall back to any role. */
  function nearestNode(x: number, y: number, z: number, role: VehiclePartRole): number {
    let ix = Math.floor((x - minX) * inv); if (ix < 0) ix = 0; else if (ix >= gx) ix = gx - 1;
    let iy = Math.floor((y - minY) * inv); if (iy < 0) iy = 0; else if (iy >= gy) iy = gy - 1;
    let iz = Math.floor((z - minZ) * inv); if (iz < 0) iz = 0; else if (iz >= gz) iz = gz - 1;
    let bestRole = -1, bestRoleD = Infinity;
    let bestAny = -1, bestAnyD = Infinity;
    const maxR = Math.max(gx, gy, gz);
    let foundRing = -1;
    for (let r = 0; r <= maxR; r++) {
      // Stop expanding once we are a full ring past the first hit (guarantees the
      // true nearest within the searched roles is found).
      if (foundRing >= 0 && r > foundRing + 1) break;
      const x0 = Math.max(0, ix - r), x1 = Math.min(gx - 1, ix + r);
      const y0 = Math.max(0, iy - r), y1 = Math.min(gy - 1, iy + r);
      const z0 = Math.max(0, iz - r), z1 = Math.min(gz - 1, iz + r);
      for (let a = x0; a <= x1; a++) {
        for (let b = y0; b <= y1; b++) {
          for (let c = z0; c <= z1; c++) {
            // Only the shell of the (2r+1)^3 box is new for r>0.
            if (r > 0 && a > x0 && a < x1 && b > y0 && b < y1 && c > z0 && c < z1) continue;
            const list = grid.get((a * gy + b) * gz + c);
            if (!list) continue;
            for (const n of list) {
              const dx = cx[n] - x, dy = cy[n] - y, dz = cz[n] - z;
              const d = dx * dx + dy * dy + dz * dz;
              if (d < bestAnyD) { bestAnyD = d; bestAny = n; }
              if (roleOf[n] === role && d < bestRoleD) { bestRoleD = d; bestRole = n; }
            }
          }
        }
      }
      if (foundRing < 0 && (bestAny >= 0 || bestRole >= 0)) foundRing = r;
    }
    // Prefer same-role unless it is much farther than the overall nearest (handles
    // role classification drift between the asset and the original model).
    if (bestRole >= 0 && (bestAny < 0 || bestRoleD <= bestAnyD * 2.25)) return bestRole;
    return bestAny;
  }

  // Accumulate assigned triangles (world-space verts + normals) per node.
  const accPos: number[][] = Array.from({ length: nodeCount }, () => []);
  const accNrm: number[][] = Array.from({ length: nodeCount }, () => []);

  for (const part of renderParts) {
    const geom = part.geometry;
    const posAttr = geom.getAttribute('position') as THREE.BufferAttribute | undefined;
    if (!posAttr) continue;
    const nrmAttr = geom.getAttribute('normal') as THREE.BufferAttribute | undefined;
    const index = geom.getIndex();
    const triCount = Math.floor((index ? index.count : posAttr.count) / 3);
    const vi = (t: number, k: number) => (index ? index.getX(t * 3 + k) : t * 3 + k);
    for (let t = 0; t < triCount; t++) {
      const a = vi(t, 0), b = vi(t, 1), c = vi(t, 2);
      const ax = posAttr.getX(a), ay = posAttr.getY(a), az = posAttr.getZ(a);
      const bx = posAttr.getX(b), by = posAttr.getY(b), bz = posAttr.getZ(b);
      const ccx = posAttr.getX(c), ccy = posAttr.getY(c), ccz = posAttr.getZ(c);
      const tcx = (ax + bx + ccx) / 3, tcy = (ay + by + ccy) / 3, tcz = (az + bz + ccz) / 3;
      const node = nearestNode(tcx, tcy, tcz, part.role);
      if (node < 0) continue;
      const ap = accPos[node];
      ap.push(ax, ay, az, bx, by, bz, ccx, ccy, ccz);
      const an = accNrm[node];
      if (nrmAttr) {
        an.push(
          nrmAttr.getX(a), nrmAttr.getY(a), nrmAttr.getZ(a),
          nrmAttr.getX(b), nrmAttr.getY(b), nrmAttr.getZ(b),
          nrmAttr.getX(c), nrmAttr.getY(c), nrmAttr.getZ(c),
        );
      }
    }
  }

  // Build per-node render geometry; keep hulls for collision.
  const renderGeoms: THREE.BufferGeometry[] = new Array(nodeCount);
  let withDetail = 0, withoutDetail = 0;
  for (let i = 0; i < nodeCount; i++) {
    const ap = accPos[i];
    if (ap.length >= 9) {
      const local = new Float32Array(ap.length);
      // Local frame = world - pieceCentroid (matches the hull geometry's frame).
      for (let j = 0; j < ap.length; j += 3) {
        local[j] = ap[j] - cx[i];
        local[j + 1] = ap[j + 1] - cy[i];
        local[j + 2] = ap[j + 2] - cz[i];
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(local, 3));
      const an = accNrm[i];
      if (an.length === ap.length) g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(an), 3));
      else g.computeVertexNormals();
      renderGeoms[i] = g;
      withDetail++;
    } else {
      renderGeoms[i] = hulls[i]; // fall back to the hull when no triangles landed here
      withoutDetail++;
    }
  }

  scenario.parameters = scenario.parameters ?? {};
  scenario.parameters.colliderGeometries = hulls; // tight convex hulls = collision
  scenario.parameters.fragmentGeometries = renderGeoms; // detailed slices = render
  console.log(`[glb-vehicle] detailed render: ${withDetail}/${nodeCount} nodes got model geometry, ${withoutDetail} fell back to hull`);
}

// ── Connectivity helpers ─────────────────────────────────────────────────────

function medianArea(bonds: Array<{ area: number }>): number {
  if (!bonds.length) return 0;
  const a = bonds.map((b) => b.area).sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

/**
 * Cargo / accessories are NOT structurally attached — they just rest in the
 * vehicle. Their bonds exist only so they ride along as one reduced rigid body
 * until disturbed (a perf convenience), not because they are welded on. The
 * triangle-surface auto-bonder, however, glues a resting box along its whole
 * contact footprint (many bonds), which over-constrains it into "welded".
 *
 * Reduce each such part to a single strongest bond so it behaves like a resting
 * contact: one weak link that lets go cleanly under the slightest stress. A bond
 * is kept if it is a top-`maxBondsPerNode` bond (by area) of EITHER endpoint, so
 * every capped node keeps at least its best link; full disconnection (if any) is
 * repaired afterwards by stitchConnectivity. Bonds between two un-capped
 * (structural) parts are never touched.
 *
 * Exported for unit testing the pure graph logic without a GLB.
 */
export function capRestingBonds(
  scenario: { bonds: Array<{ node0: number; node1: number; area: number }> },
  meta: Array<{ role: VehiclePartRole } | undefined>,
  options: { roles?: VehiclePartRole[]; maxBondsPerNode?: number } = {},
): number {
  const roles = new Set<VehiclePartRole>(options.roles ?? ['cargo', 'accessory']);
  const maxPer = Math.max(1, Math.floor(options.maxBondsPerNode ?? 1));
  const bonds = scenario.bonds;
  const isCapped = (n: number) => roles.has(meta[n]?.role as VehiclePartRole);

  // Collect bond indices touching each capped node.
  const perNode = new Map<number, number[]>();
  for (let bi = 0; bi < bonds.length; bi++) {
    const { node0, node1 } = bonds[bi];
    if (isCapped(node0)) (perNode.get(node0) ?? perNode.set(node0, []).get(node0)!).push(bi);
    if (isCapped(node1)) (perNode.get(node1) ?? perNode.set(node1, []).get(node1)!).push(bi);
  }
  if (perNode.size === 0) return 0;

  // Keep each capped node's strongest `maxPer` bonds (by area).
  const keep = new Set<number>();
  for (const biList of perNode.values()) {
    biList.sort((a, b) => bonds[b].area - bonds[a].area);
    for (let k = 0; k < Math.min(maxPer, biList.length); k++) keep.add(biList[k]);
  }

  // Drop bonds that touch a capped node but aren't kept by either endpoint.
  const next: typeof bonds = [];
  let removed = 0;
  for (let bi = 0; bi < bonds.length; bi++) {
    const { node0, node1 } = bonds[bi];
    if ((isCapped(node0) || isCapped(node1)) && !keep.has(bi)) { removed++; continue; }
    next.push(bonds[bi]);
  }
  scenario.bonds = next;
  return removed;
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
