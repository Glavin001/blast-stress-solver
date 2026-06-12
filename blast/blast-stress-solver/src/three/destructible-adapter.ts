import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import type { ChunkData, DestructibleCore } from '../rapier/types';
import type { InterpolatedPoseView } from '../rapier/fixedStepLoop';

const shadowsEnabled = true;

const HEALTHY_COLOR = new THREE.Color(0x2fbf71);
const CRITICAL_COLOR = new THREE.Color(0xd72638);

const KINEMATIC_COLOR = 0x2a6ddb;
const FIXED_COLOR = 0xbababa;
const DYNAMIC_COLOR = 0xff9147;

type ChunkLike = Pick<ChunkData, 'nodeIndex'>;
type RigidBodyLike = Pick<RAPIER.RigidBody, 'isKinematic' | 'isFixed' | 'isDynamic'>;

export type ChunkMeshBuildResult = {
  objects: THREE.Mesh[];
  dispose: () => void;
};

export type ChunkMeshBuildOptions = {
  /** Clone provided geometries before attaching them to meshes. Default: false. */
  cloneGeometries?: boolean;
  /** Remove geometry groups on prepared geometries. Default: true. */
  clearGroups?: boolean;
  /** Dispose external source geometries on `dispose()`. Default: false. */
  disposeSourceGeometries?: boolean;
  /**
   * Optional per-node base ("material") colors, indexed by `chunk.nodeIndex`. When
   * provided (and the per-chunk damage overlay is off), chunks render in their material
   * color instead of the fixed/kinematic/dynamic physics-state colors — so a demo can
   * show realistic materials. Pass `undefined` (or omit) to fall back to the standard
   * state coloring; toggling the two is how demos flip between "material" and
   * "collision/state" views at runtime without rebuilding.
   */
  nodeColors?: THREE.Color[];
};

export type BatchedChunkMeshOptions = ChunkMeshBuildOptions & {
  /**
   * Retained for backward compatibility with Vibe City. Standard THREE.BatchedMesh
   * does not support per-instance shader uniforms here, so this flag is ignored.
   */
  enablePerInstanceUniforms?: boolean;
  /** Enable optional prototype-extended BVH helpers if present. Default: true. */
  enableBVH?: boolean;
  /** Margin forwarded to optional `computeBVH` helper. Default: 0.5. */
  bvhMargin?: number;
  /**
   * Per-instance front-to-back sort each frame (THREE.BatchedMesh default `true`). For opaque
   * chunks this only trims overdraw, but it costs an O(visible·log visible) sort inside
   * `onBeforeRender` every frame/pass — ~5 ms for 27k instances, the dominant render-thread cost
   * on a large city. Default: `false`.
   */
  sortObjects?: boolean;
  /** Per-instance frustum culling (THREE.BatchedMesh default `true`). Keep on to skip off-screen
   *  instances when zoomed in. Default: leave the BatchedMesh default. */
  perObjectFrustumCulled?: boolean;
};

type OptionalBvhApi = {
  computeBVH?: (coordinateSystem?: number, options?: { margin?: number }) => void;
  bvh?: {
    move: (instanceId: number) => void;
  };
};

type ColorWritableBatchedMesh = THREE.BatchedMesh & OptionalBvhApi & {
  setColorAt?: (instanceId: number, color: THREE.Color) => THREE.BatchedMesh;
};

export type BatchedChunkMeshResult = {
  batchedMesh: ColorWritableBatchedMesh;
  chunkToInstanceId: Map<number, number>;
  geometryIds: number[];
  dispose: () => void;
};

const PROJECTILE_MAX_LIFETIME = 12;
const PROJECTILE_MIN_Y = -50;
const nowSeconds = () =>
  (typeof performance !== 'undefined' ? performance.now() : Date.now()) / 1000;

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _quaternion = new THREE.Quaternion();
const _scale = new THREE.Vector3(1, 1, 1);
const _localOffset = new THREE.Vector3();
const _colorTmp = new THREE.Color();
const _batchedDefaultColor = new THREE.Color(0x00ff00);

/**
 * Per-instance "last applied" state for {@link updateBatchedChunkMesh}, so that a frame can
 * skip the expensive `setMatrixAt`/`setColorAt`/`setVisibleAt` writes (and the matrix-texture
 * re-upload they trigger) for instances that did not change. This mirrors the physics core's
 * unchanged-body transform skip: an intact/dormant city writes nothing per frame.
 */
type BatchedSyncCache = {
  count: number;
  px: Float32Array;
  py: Float32Array;
  pz: Float32Array;
  qx: Float32Array;
  qy: Float32Array;
  qz: Float32Array;
  qw: Float32Array;
  health: Float32Array;
  visible: Uint8Array;
  lastNodeColors: THREE.Color[] | undefined;
  primed: boolean;
};

const _batchedSyncCaches = new WeakMap<THREE.BatchedMesh, BatchedSyncCache>();

function getBatchedSyncCache(batchedMesh: THREE.BatchedMesh, count: number): BatchedSyncCache {
  let cache = _batchedSyncCaches.get(batchedMesh);
  if (!cache || cache.count < count) {
    // NaN pose / 255 visibility are sentinels that force a write on the first frame.
    cache = {
      count,
      px: new Float32Array(count).fill(NaN),
      py: new Float32Array(count).fill(NaN),
      pz: new Float32Array(count).fill(NaN),
      qx: new Float32Array(count).fill(NaN),
      qy: new Float32Array(count).fill(NaN),
      qz: new Float32Array(count).fill(NaN),
      qw: new Float32Array(count).fill(NaN),
      health: new Float32Array(count).fill(NaN),
      visible: new Uint8Array(count).fill(255),
      lastNodeColors: undefined,
      primed: false,
    };
    _batchedSyncCaches.set(batchedMesh, cache);
  }
  return cache;
}

function toMeshStandardMaterial(material: THREE.Material | THREE.Material[] | undefined): THREE.MeshStandardMaterial | null {
  return material instanceof THREE.MeshStandardMaterial ? material : null;
}

function prepareGeometry(
  geometry: THREE.BufferGeometry,
  options: ChunkMeshBuildOptions | undefined,
  ownedGeometries: Set<THREE.BufferGeometry>,
): THREE.BufferGeometry {
  let prepared = geometry;
  if (options?.cloneGeometries) {
    prepared = geometry.clone();
    ownedGeometries.add(prepared);
  } else if (options?.disposeSourceGeometries) {
    ownedGeometries.add(prepared);
  }

  if (options?.clearGroups ?? true) {
    try {
      prepared.clearGroups();
    } catch {}
  }

  return prepared;
}

function disposeOwnedGeometries(ownedGeometries: Set<THREE.BufferGeometry>) {
  for (const geometry of ownedGeometries) {
    try {
      geometry.dispose();
    } catch {}
  }
  ownedGeometries.clear();
}

function disposeMeshMaterials(meshes: THREE.Mesh[]) {
  for (const mesh of meshes) {
    const material = mesh.material;
    if (Array.isArray(material)) {
      for (const entry of material) {
        try {
          entry.dispose();
        } catch {}
      }
    } else {
      try {
        material?.dispose?.();
      } catch {}
    }
  }
}

function getBatchedMeshCount(geometries: THREE.BufferGeometry[]) {
  let vertexCount = 0;
  let indexCount = 0;

  for (const geometry of geometries) {
    const position = geometry.getAttribute('position');
    if (!position) continue;
    vertexCount += position.count;
    indexCount += geometry.index?.count ?? position.count;
  }

  return { vertexCount, indexCount };
}

function makeDefaultMaterials(materials?: { deck?: THREE.Material; support?: THREE.Material }) {
  return {
    deck:
      materials?.deck ??
      new THREE.MeshStandardMaterial({
        color: 0xbababa,
        roughness: 0.62,
        metalness: 0.05,
      }),
    support:
      materials?.support ??
      new THREE.MeshStandardMaterial({
        color: 0x7a889a,
        roughness: 0.7,
        metalness: 0.15,
      }),
  };
}

/**
 * Resolve a chunk's color. Precedence:
 *   1. Damage overlay (if enabled): green→red health gradient.
 *   2. Material `baseColor` (if provided): the chunk's material color, kept even after
 *      it breaks free (debris is darkened a touch so motion still reads).
 *   3. Physics state: kinematic (blue) / fixed (gray) / dynamic (orange).
 * Mutates and returns the provided color instance.
 */
export function applyChunkColor(opts: {
  core: DestructibleCore;
  chunk: ChunkLike;
  body: RigidBodyLike;
  color: THREE.Color;
  /** Material color for "material" view. Omit/null → standard state coloring. */
  baseColor?: THREE.Color | null;
}): THREE.Color {
  const { core, chunk, body, color, baseColor } = opts;

  const damageEnabled = core.damageEnabled === true;
  const healthGetter = core.getNodeHealth;
  if (damageEnabled && typeof healthGetter === 'function') {
    const info = healthGetter(chunk.nodeIndex);
    if (info && info.maxHealth > 0) {
      const ratio = Math.max(0, Math.min(1, info.health / info.maxHealth));
      color.copy(HEALTHY_COLOR).lerp(CRITICAL_COLOR, 1 - ratio);
      return color;
    }
  }

  if (baseColor) {
    color.copy(baseColor);
    if (body.isDynamic()) color.multiplyScalar(0.82);
    return color;
  }

  if (body.isKinematic()) color.setHex(KINEMATIC_COLOR);
  else if (body.isFixed()) color.setHex(FIXED_COLOR);
  else if (body.isDynamic()) color.setHex(DYNAMIC_COLOR);

  return color;
}

/** Per-node base color lookup (by chunk.nodeIndex), or null when colors aren't supplied. */
function baseColorForNode(
  nodeColors: THREE.Color[] | undefined,
  nodeIndex: number,
): THREE.Color | null {
  return nodeColors ? nodeColors[nodeIndex] ?? null : null;
}

export function buildChunkMeshes(
  core: DestructibleCore,
  materials?: { deck?: THREE.Material; support?: THREE.Material },
): ChunkMeshBuildResult {
  const mats = makeDefaultMaterials(materials);
  const meshes: THREE.Mesh[] = [];
  const ownedGeometries = new Set<THREE.BufferGeometry>();

  for (const chunk of core.chunks) {
    const material = (chunk.isSupport ? mats.support : mats.deck).clone();
    const geometry = new THREE.BoxGeometry(chunk.size.x, chunk.size.y, chunk.size.z);
    ownedGeometries.add(geometry);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.nodeIndex = chunk.nodeIndex;
    mesh.castShadow = shadowsEnabled;
    mesh.receiveShadow = shadowsEnabled;
    meshes.push(mesh);
  }

  return {
    objects: meshes,
    dispose: () => {
      disposeMeshMaterials(meshes);
      disposeOwnedGeometries(ownedGeometries);
    },
  };
}

export function buildChunkMeshesFromGeometries(
  core: DestructibleCore,
  geometries: THREE.BufferGeometry[],
  materials?: { deck?: THREE.Material; support?: THREE.Material },
  options?: ChunkMeshBuildOptions,
): ChunkMeshBuildResult {
  const mats = makeDefaultMaterials(materials);
  const meshes: THREE.Mesh[] = [];
  const ownedGeometries = new Set<THREE.BufferGeometry>();

  for (let i = 0; i < core.chunks.length; i += 1) {
    const chunk = core.chunks[i];
    const geometry = chunk.isSupport
      ? new THREE.BoxGeometry(chunk.size.x, chunk.size.y, chunk.size.z)
      : prepareGeometry(
          geometries[i] ?? new THREE.BoxGeometry(chunk.size.x, chunk.size.y, chunk.size.z),
          options,
          ownedGeometries,
        );

    if (chunk.isSupport || !geometries[i]) {
      ownedGeometries.add(geometry);
      if (options?.clearGroups ?? true) {
        try {
          geometry.clearGroups();
        } catch {}
      }
    }

    const material = (chunk.isSupport ? mats.support : mats.deck).clone();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.nodeIndex = chunk.nodeIndex;
    mesh.castShadow = shadowsEnabled;
    mesh.receiveShadow = shadowsEnabled;
    meshes.push(mesh);
  }

  return {
    objects: meshes,
    dispose: () => {
      disposeMeshMaterials(meshes);
      disposeOwnedGeometries(ownedGeometries);
    },
  };
}

export function buildBatchedChunkMesh(
  core: DestructibleCore,
  options?: BatchedChunkMeshOptions,
): BatchedChunkMeshResult {
  const geometries: THREE.BufferGeometry[] = [];
  const ownedGeometries = new Set<THREE.BufferGeometry>();

  for (const chunk of core.chunks) {
    const geometry = new THREE.BoxGeometry(chunk.size.x, chunk.size.y, chunk.size.z);
    ownedGeometries.add(geometry);
    if (options?.clearGroups ?? true) {
      try {
        geometry.clearGroups();
      } catch {}
    }
    geometries.push(geometry);
  }

  return buildBatchedChunkMeshInternal(core, geometries, ownedGeometries, options);
}

export function buildBatchedChunkMeshFromGeometries(
  core: DestructibleCore,
  geometries: THREE.BufferGeometry[],
  options?: BatchedChunkMeshOptions,
): BatchedChunkMeshResult {
  const finalGeometries: THREE.BufferGeometry[] = [];
  const ownedGeometries = new Set<THREE.BufferGeometry>();

  for (let i = 0; i < core.chunks.length; i += 1) {
    const chunk = core.chunks[i];
    if (chunk.isSupport) {
      const supportGeometry = new THREE.BoxGeometry(chunk.size.x, chunk.size.y, chunk.size.z);
      ownedGeometries.add(supportGeometry);
      if (options?.clearGroups ?? true) {
        try {
          supportGeometry.clearGroups();
        } catch {}
      }
      finalGeometries.push(supportGeometry);
      continue;
    }

    const sourceGeometry = geometries[i] ?? new THREE.BoxGeometry(chunk.size.x, chunk.size.y, chunk.size.z);
    if (!geometries[i]) {
      ownedGeometries.add(sourceGeometry);
    }
    finalGeometries.push(prepareGeometry(sourceGeometry, options, ownedGeometries));
  }

  return buildBatchedChunkMeshInternal(core, finalGeometries, ownedGeometries, options);
}

function buildBatchedChunkMeshInternal(
  core: DestructibleCore,
  geometries: THREE.BufferGeometry[],
  ownedGeometries: Set<THREE.BufferGeometry>,
  options?: BatchedChunkMeshOptions,
): BatchedChunkMeshResult {
  const chunkCount = core.chunks.length;
  const { vertexCount, indexCount } = getBatchedMeshCount(geometries);

  const material = new THREE.MeshStandardMaterial({
    color: 0xbababa,
    roughness: 0.5,
    metalness: 0.1,
  });

  const batchedMesh = new THREE.BatchedMesh(
    Math.max(1, chunkCount),
    Math.max(3, vertexCount),
    Math.max(3, indexCount),
    material,
  ) as ColorWritableBatchedMesh;

  batchedMesh.castShadow = shadowsEnabled;
  batchedMesh.receiveShadow = shadowsEnabled;
  batchedMesh.frustumCulled = false;
  // Opaque chunks: skip the per-frame front-to-back sort (THREE default on). It dominates the
  // render thread at scale (~5 ms for 27k instances) and only trims overdraw for opaque material.
  batchedMesh.sortObjects = options?.sortObjects ?? false;
  if (options?.perObjectFrustumCulled !== undefined) {
    batchedMesh.perObjectFrustumCulled = options.perObjectFrustumCulled;
  }

  const geometryIds: number[] = [];
  const chunkToInstanceId = new Map<number, number>();

  for (let i = 0; i < chunkCount; i += 1) {
    const chunk = core.chunks[i];
    const geometry = geometries[i];
    const geometryId = batchedMesh.addGeometry(geometry);
    geometryIds.push(geometryId);

    const instanceId = batchedMesh.addInstance(geometryId);
    chunkToInstanceId.set(chunk.nodeIndex, instanceId);

    const body = core.world.getRigidBody(chunk.bodyHandle ?? -1);
    if (body) {
      const translation = body.translation();
      const rotation = body.rotation();
      _position.set(translation.x, translation.y, translation.z);
      _quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
      _localOffset
        .set(chunk.baseLocalOffset.x, chunk.baseLocalOffset.y, chunk.baseLocalOffset.z)
        .applyQuaternion(_quaternion);
      _position.add(_localOffset);
      _matrix.compose(_position, _quaternion, _scale);
    } else {
      _position.set(chunk.baseLocalOffset.x, chunk.baseLocalOffset.y, chunk.baseLocalOffset.z);
      _quaternion.identity();
      _matrix.compose(_position, _quaternion, _scale);
    }

    batchedMesh.setMatrixAt(instanceId, _matrix);
    batchedMesh.setVisibleAt(instanceId, !chunk.destroyed);
    if (typeof batchedMesh.setColorAt === 'function') {
      const baseColor = baseColorForNode(options?.nodeColors, chunk.nodeIndex);
      if (body) {
        applyChunkColor({ core, chunk, body, color: _colorTmp, baseColor });
        batchedMesh.setColorAt(instanceId, _colorTmp);
      } else {
        batchedMesh.setColorAt(instanceId, baseColor ?? _batchedDefaultColor);
      }
    }
  }

  try {
    batchedMesh.computeBoundingBox();
  } catch {}
  try {
    batchedMesh.computeBoundingSphere();
  } catch {}

  if ((options?.enableBVH ?? true) && typeof batchedMesh.computeBVH === 'function') {
    try {
      batchedMesh.computeBVH(2000, { margin: options?.bvhMargin ?? 0.5 });
    } catch {}
  }

  return {
    batchedMesh,
    chunkToInstanceId,
    geometryIds,
    dispose: () => {
      try {
        batchedMesh.dispose();
      } catch {}
      try {
        material.dispose();
      } catch {}
      disposeOwnedGeometries(ownedGeometries);
    },
  };
}

export function updateChunkMeshes(
  core: DestructibleCore,
  meshes: THREE.Mesh[],
  options?: { nodeColors?: THREE.Color[] },
) {
  const nodeColors = options?.nodeColors;
  const tmp = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const isDev = typeof process !== 'undefined' ? process.env.NODE_ENV !== 'production' : true;

  if (isDev && meshes.length !== core.chunks.length) {
    console.error('[Adapter] Chunk mesh count mismatch', {
      meshes: meshes.length,
      chunks: core.chunks.length,
    });
    throw new Error('Chunk mesh count mismatch');
  }

  for (let i = 0; i < core.chunks.length; i += 1) {
    const chunk = core.chunks[i];
    const mesh = meshes[i];
    if (!mesh) continue;

    if (chunk.destroyed) {
      mesh.visible = false;
      continue;
    }
    mesh.visible = true;

    const handle = chunk.bodyHandle;
    if (handle == null) continue;

    const body = core.world.getRigidBody(handle);
    if (!body) continue;

    const translation = body.translation();
    const rotation = body.rotation();
    mesh.position.set(translation.x, translation.y, translation.z);
    quat.set(rotation.x, rotation.y, rotation.z, rotation.w);
    mesh.quaternion.copy(quat);
    tmp
      .set(chunk.baseLocalOffset.x, chunk.baseLocalOffset.y, chunk.baseLocalOffset.z)
      .applyQuaternion(mesh.quaternion);
    mesh.position.add(tmp);

    const material = toMeshStandardMaterial(mesh.material);
    if (material) {
      const baseColor = baseColorForNode(nodeColors, chunk.nodeIndex);
      applyChunkColor({ core, chunk, body, color: material.color, baseColor });
    }
  }
}

export function updateBatchedChunkMesh(
  core: DestructibleCore,
  batchedMesh: ColorWritableBatchedMesh,
  chunkToInstanceId: Map<number, number>,
  options?: {
    updateBVH?: boolean;
    /** Per-node material colors (by chunk.nodeIndex). Omit for state coloring. */
    nodeColors?: THREE.Color[];
    /** Rewrite every instance this frame, bypassing the unchanged-pose skip. */
    forceFullUpdate?: boolean;
    /** Mask (by instanceId) of fragments hidden by an external LOD layer (e.g. an intact-building
     *  proxy box stands in for them). Non-zero → the fragment instance is forced invisible. */
    instanceHidden?: Uint8Array;
    /** Fixed-timestep render interpolation: blend each chunk's pose between the last two
     *  physics states (`createPoseInterpolator(core).view(alpha)`). Chunks without a captured
     *  pose (NaN sentinel) fall back to the live chunk pose. Settled chunks interpolate to an
     *  identical pose, so the unchanged-pose skip still elides their writes. */
    interpolation?: InterpolatedPoseView;
  },
) {
  const updateBVH = options?.updateBVH ?? false;
  const nodeColors = options?.nodeColors;
  const instanceHidden = options?.instanceHidden;
  const forceFull = options?.forceFullUpdate === true;
  const interp = options?.interpolation;
  const interpPrev = interp?.prev;
  const interpCurr = interp?.curr;
  const interpAlpha = interp ? Math.min(1, Math.max(0, interp.alpha)) : 0;
  const canColor = typeof batchedMesh.setColorAt === 'function';
  // Health-based coloring only applies when damage is enabled; capture the getter once.
  const healthGetter =
    canColor && core.damageEnabled === true && typeof core.getNodeHealth === 'function'
      ? core.getNodeHealth
      : undefined;

  const chunks = core.chunks;
  const cache = getBatchedSyncCache(batchedMesh, chunks.length);

  // The active color source changed (material-color toggle flips nodeColors between an array
  // and undefined, damage toggled, or first frame): recolor every visible instance even if it
  // didn't move.
  const colorSourceChanged = forceFull || !cache.primed || cache.lastNodeColors !== nodeColors;
  cache.lastNodeColors = nodeColors;
  cache.primed = true;

  for (let i = 0; i < chunks.length; i += 1) {
    const chunk = chunks[i];
    const instanceId = chunkToInstanceId.get(chunk.nodeIndex);
    if (instanceId == null || instanceId >= cache.count) continue;

    // Visibility is derived purely from chunk state — debris cleanup marks removed chunks
    // `destroyed`/inactive — so the dormant fast path never has to touch the physics body.
    const handle = chunk.bodyHandle;
    const wantVisible =
      !chunk.destroyed &&
      chunk.active !== false &&
      handle != null &&
      !(instanceHidden !== undefined && instanceHidden[instanceId] !== 0);
    if (!wantVisible) {
      if (cache.visible[instanceId] !== 0) {
        batchedMesh.setVisibleAt(instanceId, false);
        cache.visible[instanceId] = 0;
      }
      continue;
    }

    // Prefer the core's cached world pose (composed end-of-step, and held stable across frames
    // where the body didn't move) over re-reading the rigid body every frame.
    let px: number, py: number, pz: number, qx: number, qy: number, qz: number, qw: number;
    const wp = chunk.worldPosition;
    const wq = chunk.worldQuaternion;
    const ib = i * 7;
    if (
      interpPrev !== undefined &&
      interpCurr !== undefined &&
      // NaN sentinel in either buffer → no captured pose for this chunk yet; fall through.
      interpPrev[ib + 6] === interpPrev[ib + 6] &&
      interpCurr[ib + 6] === interpCurr[ib + 6]
    ) {
      const t = interpAlpha;
      px = interpPrev[ib] + (interpCurr[ib] - interpPrev[ib]) * t;
      py = interpPrev[ib + 1] + (interpCurr[ib + 1] - interpPrev[ib + 1]) * t;
      pz = interpPrev[ib + 2] + (interpCurr[ib + 2] - interpPrev[ib + 2]) * t;
      // nlerp with hemisphere correction: adjacent physics states are near-identical
      // rotations, where nlerp ≈ slerp and stays monotonic.
      const pqx = interpPrev[ib + 3], pqy = interpPrev[ib + 4], pqz = interpPrev[ib + 5], pqw = interpPrev[ib + 6];
      let cqx = interpCurr[ib + 3], cqy = interpCurr[ib + 4], cqz = interpCurr[ib + 5], cqw = interpCurr[ib + 6];
      if (pqx * cqx + pqy * cqy + pqz * cqz + pqw * cqw < 0) { cqx = -cqx; cqy = -cqy; cqz = -cqz; cqw = -cqw; }
      qx = pqx + (cqx - pqx) * t;
      qy = pqy + (cqy - pqy) * t;
      qz = pqz + (cqz - pqz) * t;
      qw = pqw + (cqw - pqw) * t;
      const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
      if (ql > 1e-12) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; }
      else { qx = 0; qy = 0; qz = 0; qw = 1; }
    } else if (wp && wq) {
      px = wp.x; py = wp.y; pz = wp.z;
      qx = wq.x; qy = wq.y; qz = wq.z; qw = wq.w;
    } else {
      // Pre-first-step fallback: compose from the body directly.
      const body = core.world.getRigidBody(handle as number);
      if (!body) {
        if (cache.visible[instanceId] !== 0) {
          batchedMesh.setVisibleAt(instanceId, false);
          cache.visible[instanceId] = 0;
        }
        continue;
      }
      const t = body.translation();
      const r = body.rotation();
      _quaternion.set(r.x, r.y, r.z, r.w);
      _localOffset
        .set(chunk.baseLocalOffset.x, chunk.baseLocalOffset.y, chunk.baseLocalOffset.z)
        .applyQuaternion(_quaternion);
      px = t.x + _localOffset.x; py = t.y + _localOffset.y; pz = t.z + _localOffset.z;
      qx = r.x; qy = r.y; qz = r.z; qw = r.w;
    }

    const becameVisible = cache.visible[instanceId] !== 1;
    if (becameVisible) {
      batchedMesh.setVisibleAt(instanceId, true);
      cache.visible[instanceId] = 1;
    }

    const poseChanged =
      forceFull ||
      becameVisible ||
      cache.px[instanceId] !== px ||
      cache.py[instanceId] !== py ||
      cache.pz[instanceId] !== pz ||
      cache.qx[instanceId] !== qx ||
      cache.qy[instanceId] !== qy ||
      cache.qz[instanceId] !== qz ||
      cache.qw[instanceId] !== qw;

    if (poseChanged) {
      _position.set(px, py, pz);
      _quaternion.set(qx, qy, qz, qw);
      _matrix.compose(_position, _quaternion, _scale);
      batchedMesh.setMatrixAt(instanceId, _matrix);
      cache.px[instanceId] = px; cache.py[instanceId] = py; cache.pz[instanceId] = pz;
      cache.qx[instanceId] = qx; cache.qy[instanceId] = qy;
      cache.qz[instanceId] = qz; cache.qw[instanceId] = qw;
      if (updateBVH && batchedMesh.bvh) {
        try {
          batchedMesh.bvh.move(instanceId);
        } catch {}
      }
    }

    if (!canColor) continue;

    // Recolor on movement / (un)hide / color-source change, plus health changes that occur
    // without movement (e.g. cracking before collapse).
    let colorNeeded = poseChanged || colorSourceChanged;
    let healthRatio = NaN;
    if (healthGetter) {
      const info = healthGetter(chunk.nodeIndex);
      if (info && info.maxHealth > 0) {
        healthRatio = Math.max(0, Math.min(1, info.health / info.maxHealth));
        if (healthRatio !== cache.health[instanceId]) colorNeeded = true;
      }
    }
    if (!colorNeeded) continue;

    if (!Number.isNaN(healthRatio)) {
      // Health-driven tint (matches applyChunkColor's damage branch) — no body read needed.
      _colorTmp.copy(HEALTHY_COLOR).lerp(CRITICAL_COLOR, 1 - healthRatio);
      cache.health[instanceId] = healthRatio;
      batchedMesh.setColorAt(instanceId, _colorTmp);
    } else {
      // State / material color needs the body; only reached on change, so FFI stays bounded.
      const body = core.world.getRigidBody(handle as number);
      if (body) {
        const baseColor = baseColorForNode(nodeColors, chunk.nodeIndex);
        applyChunkColor({ core, chunk, body, color: _colorTmp, baseColor });
        batchedMesh.setColorAt(instanceId, _colorTmp);
      }
    }
  }
}

export { SolverDebugLinesHelper } from './solver-debug-lines';

const _projPose = { px: 0, py: 0, pz: 0, qx: 0, qy: 0, qz: 0, qw: 1 };

export function updateProjectileMeshes(
  core: DestructibleCore,
  root: THREE.Group,
  options?: {
    /** Fixed-timestep render interpolation for projectile bodies (same view handed to
     *  `updateBatchedChunkMesh`). Projectiles are the fastest movers on screen, so without
     *  this they would visibly snap at the physics rate while chunks glide. */
    interpolation?: InterpolatedPoseView;
  },
) {
  const interp = options?.interpolation;
  const profilerRecorder = (
    core as unknown as {
      recordProjectileCleanupDuration?: (durationMs: number) => void;
    }
  ).recordProjectileCleanupDuration;
  const hasPerf =
    typeof performance !== 'undefined' && typeof performance.now === 'function';
  const timeNow = hasPerf ? () => performance.now() : () => Date.now();
  const timerStart = profilerRecorder ? timeNow() : null;
  const projectiles = core.projectiles as Array<{
    bodyHandle: number;
    radius: number;
    type: 'ball' | 'box';
    mesh?: THREE.Mesh;
    spawnTime?: number;
  }>;
  const now = nowSeconds();

  for (let i = projectiles.length - 1; i >= 0; i -= 1) {
    const projectile = projectiles[i];
    const body = core.world.getRigidBody(projectile.bodyHandle);
    const lifetime =
      typeof projectile.spawnTime === 'number' ? now - projectile.spawnTime : 0;
    const shouldCullLifetime = lifetime > PROJECTILE_MAX_LIFETIME;
    const shouldCullBody = !body;
    const bodyTranslation = body?.translation();
    const shouldCullFall = bodyTranslation ? bodyTranslation.y < PROJECTILE_MIN_Y : false;

    if (shouldCullLifetime || shouldCullBody || shouldCullFall) {
      if (projectile.mesh) {
        root.remove(projectile.mesh);
        try {
          projectile.mesh.geometry?.dispose?.();
        } catch {}
        try {
          (projectile.mesh.material as THREE.Material | undefined)?.dispose?.();
        } catch {}
      }
      if (body) {
        try {
          core.world.removeRigidBody(body);
        } catch {}
      }
      projectiles.splice(i, 1);
      continue;
    }

    if (!projectile.mesh) {
      const geometry =
        projectile.type === 'ball'
          ? new THREE.SphereGeometry(projectile.radius, 24, 24)
          : new THREE.BoxGeometry(
              projectile.radius * 2,
              projectile.radius * 2,
              projectile.radius * 2,
            );
      const material = new THREE.MeshStandardMaterial({
        color: 0xff9147,
        emissive: 0x331100,
        roughness: 0.4,
        metalness: 0.2,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = shadowsEnabled;
      mesh.receiveShadow = shadowsEnabled;
      projectile.mesh = mesh;
      root.add(mesh);
    }

    if (!body || !bodyTranslation) continue;

    const mesh = projectile.mesh as THREE.Mesh;
    mesh.visible = true;
    if (interp && interp.getBodyPose(projectile.bodyHandle, _projPose)) {
      mesh.position.set(_projPose.px, _projPose.py, _projPose.pz);
      mesh.quaternion.set(_projPose.qx, _projPose.qy, _projPose.qz, _projPose.qw);
    } else {
      const rotation = body.rotation();
      mesh.position.set(bodyTranslation.x, bodyTranslation.y, bodyTranslation.z);
      mesh.quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
    }
  }

  if (profilerRecorder && timerStart != null) {
    profilerRecorder(Math.max(0, timeNow() - timerStart));
  }
}
