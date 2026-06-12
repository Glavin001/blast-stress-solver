/**
 * Shared-memory pose channel for off-main-thread physics (roadmap Tier 3.1,
 * docs/perf-city-scale-roadmap.md §3).
 *
 * The physics worker owns a `DestructibleCore` and PUBLISHES the render-facing state —
 * every chunk's world pose + liveness flags + body kind, and the live projectiles —
 * into one SharedArrayBuffer each fixed step. The main thread MIRRORS that state into
 * adapter-compatible chunk objects (mutating the same objects every frame, so the
 * Three.js batched-sync cache and its unchanged-instance skip keep working untouched).
 *
 * Consistency model (deliberately simple v1): the writer bumps the frame counter with a
 * release store after writing the body of the frame; the reader acquire-loads it before
 * mirroring. A reader that overlaps a writer can observe a mix of two adjacent fixed
 * steps for different chunks (and, in principle, a torn 7-float pose) for ONE render
 * frame — visually sub-pixel at 60 Hz physics and self-correcting next frame. Commands
 * and stats ride ordinary postMessage; only the hot per-frame state lives in the SAB.
 *
 * Layout (one SAB):
 *   I32 header: [0] frameIndex  [1] chunkCount  [2] projectileCount  [3] reserved
 *   F32 chunk poses:   7 per chunk (px py pz qx qy qz qw)
 *   U32 chunk bodies:  1 per chunk (body handle; NO_BODY = none)
 *   U8  chunk flags:   1 per chunk (bit0 active, bit1 destroyed, bit2 has-pose)
 *   U8  chunk body kind: 1 per chunk (0 fixed / 1 kinematic / 2 dynamic)
 *   F32 projectiles:   9 per slot (id px py pz qx qy qz qw radius), `type` in a U8 lane
 */

export const POSE_STRIDE = 7;
export const PROJ_STRIDE = 9;
export const NO_BODY = 0xffffffff;

export const CHUNK_FLAG_ACTIVE = 1;
export const CHUNK_FLAG_DESTROYED = 2;
export const CHUNK_FLAG_HAS_POSE = 4;
export const CHUNK_FLAG_DETACHED = 8;

export type PoseChannelViews = {
  header: Int32Array;
  poses: Float32Array;
  bodies: Uint32Array;
  flags: Uint8Array;
  bodyKind: Uint8Array;
  projectiles: Float32Array;
  projectileType: Uint8Array;
  chunkCapacity: number;
  projectileCapacity: number;
};

const HEADER_INTS = 4;

export function poseChannelByteLength(chunkCapacity: number, projectileCapacity: number): number {
  let bytes = HEADER_INTS * 4;
  bytes += chunkCapacity * POSE_STRIDE * 4; // poses f32
  bytes += chunkCapacity * 4; // bodies u32
  bytes += chunkCapacity; // flags u8
  bytes += chunkCapacity; // body kind u8
  bytes = (bytes + 3) & ~3; // realign for f32
  bytes += projectileCapacity * PROJ_STRIDE * 4;
  bytes += projectileCapacity; // projectile type u8
  return (bytes + 3) & ~3;
}

/** Build typed views over a pose-channel buffer (SharedArrayBuffer or ArrayBuffer). */
export function viewPoseChannel(
  buffer: ArrayBufferLike,
  chunkCapacity: number,
  projectileCapacity: number,
): PoseChannelViews {
  let off = 0;
  const header = new Int32Array(buffer, off, HEADER_INTS);
  off += HEADER_INTS * 4;
  const poses = new Float32Array(buffer, off, chunkCapacity * POSE_STRIDE);
  off += chunkCapacity * POSE_STRIDE * 4;
  const bodies = new Uint32Array(buffer, off, chunkCapacity);
  off += chunkCapacity * 4;
  const flags = new Uint8Array(buffer, off, chunkCapacity);
  off += chunkCapacity;
  const bodyKind = new Uint8Array(buffer, off, chunkCapacity);
  off += chunkCapacity;
  off = (off + 3) & ~3;
  const projectiles = new Float32Array(buffer, off, projectileCapacity * PROJ_STRIDE);
  off += projectileCapacity * PROJ_STRIDE * 4;
  const projectileType = new Uint8Array(buffer, off, projectileCapacity);
  return { header, poses, bodies, flags, bodyKind, projectiles, projectileType, chunkCapacity, projectileCapacity };
}

type PublishableChunk = {
  active?: boolean;
  destroyed?: boolean;
  detached?: boolean;
  bodyHandle?: number | null;
  worldPosition?: { x: number; y: number; z: number } | null;
  worldQuaternion?: { x: number; y: number; z: number; w: number } | null;
};

type PublishableCore = {
  chunks: PublishableChunk[];
  projectiles: Array<{ bodyHandle: number; radius?: number; type?: string }>;
  world: {
    getRigidBody(handle: number):
      | {
          isFixed(): boolean;
          isKinematicVelocityBased?(): boolean;
          isKinematic?(): boolean;
          translation(): { x: number; y: number; z: number };
          rotation(): { x: number; y: number; z: number; w: number };
        }
      | null
      | undefined;
  };
};

/** Worker side: publish the core's render-facing state into the channel. Returns the
 *  new frame index. `frameIndex` is bumped LAST (release order on SABs via Atomics). */
export function publishPoseFrame(views: PoseChannelViews, core: PublishableCore): number {
  const chunks = core.chunks;
  const count = Math.min(chunks.length, views.chunkCapacity);
  const bodyKindCache = new Map<number, number>();

  for (let i = 0; i < count; i++) {
    const c = chunks[i];
    let flags = 0;
    if (c.active !== false) flags |= CHUNK_FLAG_ACTIVE;
    if (c.destroyed) flags |= CHUNK_FLAG_DESTROYED;
    if (c.detached) flags |= CHUNK_FLAG_DETACHED;
    const bh = c.bodyHandle;
    views.bodies[i] = bh == null ? NO_BODY : bh >>> 0;

    const wp = c.worldPosition;
    const wq = c.worldQuaternion;
    if (wp && wq) {
      flags |= CHUNK_FLAG_HAS_POSE;
      const b = i * POSE_STRIDE;
      views.poses[b] = wp.x;
      views.poses[b + 1] = wp.y;
      views.poses[b + 2] = wp.z;
      views.poses[b + 3] = wq.x;
      views.poses[b + 4] = wq.y;
      views.poses[b + 5] = wq.z;
      views.poses[b + 6] = wq.w;
    }
    views.flags[i] = flags;

    if (bh != null) {
      let kind = bodyKindCache.get(bh);
      if (kind === undefined) {
        const body = core.world.getRigidBody(bh);
        kind = 2;
        if (body) {
          if (body.isFixed()) kind = 0;
          else if (body.isKinematic?.() || body.isKinematicVelocityBased?.()) kind = 1;
        }
        bodyKindCache.set(bh, kind);
      }
      views.bodyKind[i] = kind;
    } else {
      views.bodyKind[i] = 0;
    }
  }

  const projectiles = core.projectiles ?? [];
  const pCount = Math.min(projectiles.length, views.projectileCapacity);
  for (let p = 0; p < pCount; p++) {
    const proj = projectiles[p];
    const body = core.world.getRigidBody(proj.bodyHandle);
    const b = p * PROJ_STRIDE;
    views.projectiles[b] = proj.bodyHandle;
    if (body) {
      const t = body.translation();
      const r = body.rotation();
      views.projectiles[b + 1] = t.x;
      views.projectiles[b + 2] = t.y;
      views.projectiles[b + 3] = t.z;
      views.projectiles[b + 4] = r.x;
      views.projectiles[b + 5] = r.y;
      views.projectiles[b + 6] = r.z;
      views.projectiles[b + 7] = r.w;
    }
    views.projectiles[b + 8] = proj.radius ?? 0.5;
    views.projectileType[p] = proj.type === 'box' ? 1 : 0;
  }

  views.header[1] = count;
  views.header[2] = pCount;
  // Release-publish the frame: readers acquire-load [0] before consuming the body.
  return Atomics.add(views.header, 0, 1) + 1;
}

export type MirrorChunk = {
  nodeIndex: number;
  active: boolean;
  destroyed: boolean;
  detached: boolean;
  bodyHandle: number | null;
  worldPosition: { x: number; y: number; z: number } | null;
  worldQuaternion: { x: number; y: number; z: number; w: number } | null;
  baseLocalOffset: { x: number; y: number; z: number };
  localOffset: { x: number; y: number; z: number };
};

export type MirrorProjectile = {
  bodyHandle: number;
  radius: number;
  type: 'ball' | 'box';
  px: number; py: number; pz: number;
  qx: number; qy: number; qz: number; qw: number;
  mesh?: unknown;
  spawnTime?: number;
};

/** Main-thread side: mirror the channel into stable chunk objects (mutated in place so
 *  renderer caches keyed on identity keep working). Returns the consumed frame index,
 *  or -1 when no new frame was available (mirror untouched). */
export function mirrorPoseFrame(
  views: PoseChannelViews,
  lastFrame: number,
  chunks: MirrorChunk[],
  projectiles: MirrorProjectile[],
  bodyKindByHandle: Map<number, number>,
  nowSeconds: number,
): number {
  const frame = Atomics.load(views.header, 0);
  if (frame === lastFrame) return -1;

  const count = Math.min(views.header[1], chunks.length);
  bodyKindByHandle.clear();
  for (let i = 0; i < count; i++) {
    const c = chunks[i];
    const flags = views.flags[i];
    c.active = (flags & CHUNK_FLAG_ACTIVE) !== 0;
    c.destroyed = (flags & CHUNK_FLAG_DESTROYED) !== 0;
    c.detached = (flags & CHUNK_FLAG_DETACHED) !== 0;
    const bodyRaw = views.bodies[i];
    c.bodyHandle = bodyRaw === NO_BODY ? null : bodyRaw;
    if ((flags & CHUNK_FLAG_HAS_POSE) !== 0) {
      const b = i * POSE_STRIDE;
      let wp = c.worldPosition;
      if (!wp) { wp = { x: 0, y: 0, z: 0 }; c.worldPosition = wp; }
      wp.x = views.poses[b];
      wp.y = views.poses[b + 1];
      wp.z = views.poses[b + 2];
      let wq = c.worldQuaternion;
      if (!wq) { wq = { x: 0, y: 0, z: 0, w: 1 }; c.worldQuaternion = wq; }
      wq.x = views.poses[b + 3];
      wq.y = views.poses[b + 4];
      wq.z = views.poses[b + 5];
      wq.w = views.poses[b + 6];
    }
    if (c.bodyHandle != null) bodyKindByHandle.set(c.bodyHandle, views.bodyKind[i]);
  }

  // Projectiles: keep mesh references for slots whose id survived (the adapter stores
  // the THREE.Mesh on the entry), drop the rest.
  const pCount = views.header[2];
  const existingByHandle = new Map<number, MirrorProjectile>();
  for (const p of projectiles) existingByHandle.set(p.bodyHandle, p);
  projectiles.length = 0;
  for (let p = 0; p < pCount; p++) {
    const b = p * PROJ_STRIDE;
    const id = views.projectiles[b];
    let entry = existingByHandle.get(id);
    if (!entry) {
      entry = {
        bodyHandle: id,
        radius: views.projectiles[b + 8],
        type: views.projectileType[p] === 1 ? 'box' : 'ball',
        px: 0, py: 0, pz: 0, qx: 0, qy: 0, qz: 0, qw: 1,
        spawnTime: nowSeconds,
      };
    }
    entry.radius = views.projectiles[b + 8];
    entry.px = views.projectiles[b + 1];
    entry.py = views.projectiles[b + 2];
    entry.pz = views.projectiles[b + 3];
    entry.qx = views.projectiles[b + 4];
    entry.qy = views.projectiles[b + 5];
    entry.qz = views.projectiles[b + 6];
    entry.qw = views.projectiles[b + 7];
    projectiles.push(entry);
  }

  return frame;
}
