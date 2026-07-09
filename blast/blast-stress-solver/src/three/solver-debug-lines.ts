import * as THREE from 'three';
import type { DestructibleCore } from '../rapier/types';

/** Debug line from solver (rest/local space) */
export type DebugLine = {
  p0: { x: number; y: number; z: number };
  p1: { x: number; y: number; z: number };
  color0: number;
  color1: number;
};

const GRID_CELL_SIZE = 2.0;

type Pose = { t: THREE.Vector3; q: THREE.Quaternion };

/**
 * Helper class for rendering stress solver debug lines.
 *
 * The solver emits each line as a pair of node centroids in the *rest* frame
 * (the original, undamaged body's local space) — exactly the values stored in
 * `chunk.baseLocalOffset`. To draw a line in the world we must transform each
 * endpoint by the pose of the rigid body that currently owns that endpoint's
 * chunk.
 *
 * Correctness note: every endpoint is mapped to its *own* chunk and transformed
 * by that chunk's *current* `bodyHandle` (which the core keeps up to date as
 * fragments break off). We deliberately do NOT infer a single owning "actor"
 * per line from `solver.actors()`/`actorMap`: those mappings were cached behind
 * weak version keys (`actorMap.size`, which only ever grows) and went stale when
 * actor indices were recycled, so lines were transformed by the wrong (often
 * flown-away) body and appeared floating in the sky / scattered nonsensically.
 *
 * Usage:
 *   const helper = new SolverDebugLinesHelper();
 *   scene.add(helper.object);
 *   // In render loop:
 *   helper.update(core, core.getSolverDebugLines(), visible);
 *   // On cleanup:
 *   helper.dispose();
 */
export class SolverDebugLinesHelper {
  // THREE.js objects
  private geometry: THREE.BufferGeometry;
  private material: THREE.LineBasicMaterial;
  public readonly object: THREE.LineSegments;

  // Pre-allocated GPU buffers
  private positions: Float32Array;
  private colors: Float32Array;
  private maxLines: number;

  // Spatial grid mapping rest-space cells -> chunk indices (chunk rest centroids
  // never move, so this is built once and only rebuilt if the chunk count changes).
  private chunkGrid: Map<string, number[]> | null = null;
  private chunkGridChunkCount = -1;

  // Per-line endpoint -> chunk caches. A line's endpoints are fixed node centroids
  // in rest space, so the endpoint->chunk mapping is static for a given bond. The
  // rendered-line set only shrinks (a bond drops out when its last sub-bond breaks),
  // so the line count changing is a reliable signal to rebuild these.
  private lineChunk0: Int32Array = new Int32Array(0);
  private lineChunk1: Int32Array = new Int32Array(0);
  private lineChunkCount = -1;

  // Per-frame pose cache keyed by the chunk's CURRENT bodyHandle. Rebuilt every
  // frame (no staleness). Reuses pooled Vector3/Quaternion objects to avoid churn.
  private readonly bodyPosePool: Pose[] = [];
  private readonly bodyPoseMap = new Map<number, Pose>();
  private bodyPoolIndex = 0;

  // Root-body pose, used as the fallback for any chunk without a live body.
  private readonly rootPose: Pose = { t: new THREE.Vector3(), q: new THREE.Quaternion() };

  // Reusable transform scratch
  private readonly v = new THREE.Vector3();

  constructor(initialMaxLines = 50000) {
    this.maxLines = initialMaxLines;
    this.positions = new Float32Array(initialMaxLines * 6);
    this.colors = new Float32Array(initialMaxLines * 6);

    this.geometry = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(this.positions, 3);
    const colAttr = new THREE.BufferAttribute(this.colors, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    colAttr.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('position', posAttr);
    this.geometry.setAttribute('color', colAttr);

    this.material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    });

    this.object = new THREE.LineSegments(this.geometry, this.material);
    this.object.visible = false;
    this.object.frustumCulled = false;
  }

  /**
   * Update debug lines each frame.
   * @param core The destructible core instance
   * @param lines Debug lines from solver (rest/local space)
   * @param visible Whether to show the lines
   */
  update(core: DestructibleCore, lines: DebugLine[], visible: boolean): void {
    if (!visible) {
      this.object.visible = false;
      return;
    }

    const lineCount = lines.length;
    if (lineCount === 0) {
      this.object.visible = false;
      return;
    }

    // Ensure buffers are large enough
    this.ensureBufferCapacity(lineCount);

    // Ensure caches are up to date
    this.ensureChunkGrid(core);
    this.ensureLineChunkCache(core, lines);

    // Snapshot the current body poses (fresh every frame)
    this.updateBodyPoses(core);

    // Transform and write lines to GPU buffers. Returns how many were actually
    // written — bonds that can't be placed on a single live rigid body are skipped.
    const drawn = this.transformLines(core, lines, lineCount);
    this.geometry.setDrawRange(0, drawn * 2);

    if (drawn === 0) {
      this.object.visible = false;
      return;
    }

    // Mark buffers for upload
    const posAttr = this.geometry.getAttribute('position') as THREE.BufferAttribute;
    const colAttr = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;

    this.object.visible = true;
  }

  /**
   * Invalidate all caches. Call when the core is disposed/reset or replaced.
   */
  invalidate(): void {
    this.chunkGrid = null;
    this.chunkGridChunkCount = -1;
    this.lineChunk0 = new Int32Array(0);
    this.lineChunk1 = new Int32Array(0);
    this.lineChunkCount = -1;
    this.bodyPoseMap.clear();
  }

  /**
   * Dispose THREE.js resources.
   */
  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.invalidate();
  }

  // =========================================================================
  // Private: Buffer management
  // =========================================================================

  private ensureBufferCapacity(lineCount: number): void {
    if (lineCount <= this.maxLines) return;

    // Double capacity to avoid frequent reallocations
    this.maxLines = Math.max(lineCount, this.maxLines * 2);
    this.positions = new Float32Array(this.maxLines * 6);
    this.colors = new Float32Array(this.maxLines * 6);

    const posAttr = new THREE.BufferAttribute(this.positions, 3);
    const colAttr = new THREE.BufferAttribute(this.colors, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    colAttr.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('position', posAttr);
    this.geometry.setAttribute('color', colAttr);
  }

  // =========================================================================
  // Private: Chunk grid (built once per core; chunk rest centroids are static)
  // =========================================================================

  private ensureChunkGrid(core: DestructibleCore): void {
    const chunkCount = core.chunks.length;
    if (this.chunkGrid && this.chunkGridChunkCount === chunkCount) return;

    this.chunkGrid = new Map();
    for (let i = 0; i < core.chunks.length; i++) {
      const c = core.chunks[i];
      const key = this.getGridKey(c.baseLocalOffset.x, c.baseLocalOffset.y, c.baseLocalOffset.z);
      let arr = this.chunkGrid.get(key);
      if (!arr) {
        arr = [];
        this.chunkGrid.set(key, arr);
      }
      arr.push(i);
    }
    this.chunkGridChunkCount = chunkCount;
    // Endpoint->chunk indices reference this grid; force a rebuild.
    this.lineChunkCount = -1;
  }

  private getGridKey(x: number, y: number, z: number): string {
    const cx = Math.floor(x / GRID_CELL_SIZE);
    const cy = Math.floor(y / GRID_CELL_SIZE);
    const cz = Math.floor(z / GRID_CELL_SIZE);
    return `${cx},${cy},${cz}`;
  }

  private findNearestChunkIndex(core: DestructibleCore, mx: number, my: number, mz: number): number {
    if (!this.chunkGrid) return -1;

    let bestNode = -1;
    let bestD2 = Infinity;

    const cx = Math.floor(mx / GRID_CELL_SIZE);
    const cy = Math.floor(my / GRID_CELL_SIZE);
    const cz = Math.floor(mz / GRID_CELL_SIZE);

    // Search 3x3x3 neighborhood
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const key = `${cx + dx},${cy + dy},${cz + dz}`;
          const indices = this.chunkGrid.get(key);
          if (!indices) continue;

          for (const i of indices) {
            const c = core.chunks[i];
            const ddx = c.baseLocalOffset.x - mx;
            const ddy = c.baseLocalOffset.y - my;
            const ddz = c.baseLocalOffset.z - mz;
            const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (d2 < bestD2) {
              bestD2 = d2;
              bestNode = i;
            }
          }
        }
      }
    }

    return bestNode;
  }

  // =========================================================================
  // Private: Per-line endpoint -> chunk cache
  // =========================================================================

  private ensureLineChunkCache(core: DestructibleCore, lines: DebugLine[]): void {
    if (this.lineChunkCount === lines.length) return;

    // Rebuild — this only happens when the rendered-line count changes (a bond
    // dropping out as it fully breaks). Each endpoint is matched to the chunk
    // whose rest centroid it coincides with, so it can follow that chunk's body.
    this.lineChunk0 = new Int32Array(lines.length);
    this.lineChunk1 = new Int32Array(lines.length);
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      this.lineChunk0[i] = this.findNearestChunkIndex(core, line.p0.x, line.p0.y, line.p0.z);
      this.lineChunk1[i] = this.findNearestChunkIndex(core, line.p1.x, line.p1.y, line.p1.z);
    }
    this.lineChunkCount = lines.length;
  }

  // =========================================================================
  // Private: Per-frame body poses (no staleness)
  // =========================================================================

  private updateBodyPoses(core: DestructibleCore): void {
    this.bodyPoseMap.clear();
    this.bodyPoolIndex = 0;

    // Root pose: fallback for any chunk whose body can't be resolved.
    const rootBody = core.world.getRigidBody(core.rootBodyHandle);
    if (rootBody) {
      const rt = rootBody.translation();
      const rr = rootBody.rotation();
      this.rootPose.t.set(rt.x, rt.y, rt.z);
      this.rootPose.q.set(rr.x, rr.y, rr.z, rr.w);
    } else {
      this.rootPose.t.set(0, 0, 0);
      this.rootPose.q.identity();
    }
    // Chunks still attached to the root share the root pose directly.
    this.bodyPoseMap.set(core.rootBodyHandle, this.rootPose);
  }

  /** Current world pose of a rigid body, cached per frame. Null if it can't be resolved. */
  private poseForBody(core: DestructibleCore, bodyHandle: number): Pose | null {
    const cached = this.bodyPoseMap.get(bodyHandle);
    if (cached) return cached;

    const body = core.world.getRigidBody(bodyHandle);
    if (!body) return null;

    const pose = this.getPooledPose(this.bodyPoolIndex++);
    const tr = body.translation();
    const rot = body.rotation();
    pose.t.set(tr.x, tr.y, tr.z);
    pose.q.set(rot.x, rot.y, rot.z, rot.w);
    this.bodyPoseMap.set(bodyHandle, pose);
    return pose;
  }

  private getPooledPose(index: number): Pose {
    if (index < this.bodyPosePool.length) {
      return this.bodyPosePool[index];
    }
    const pose: Pose = { t: new THREE.Vector3(), q: new THREE.Quaternion() };
    this.bodyPosePool.push(pose);
    return pose;
  }

  // =========================================================================
  // Private: Transform lines to world space and write to GPU buffers
  // =========================================================================

  private transformLines(core: DestructibleCore, lines: DebugLine[], lineCount: number): number {
    let written = 0;
    for (let i = 0; i < lineCount; i++) {
      const c0 = this.lineChunk0[i];
      const c1 = this.lineChunk1[i];
      const chunk0 = c0 >= 0 ? core.chunks[c0] : undefined;
      const chunk1 = c1 >= 0 ? core.chunks[c1] : undefined;
      if (!chunk0 || !chunk1) continue;

      // A bond's stress is only meaningful within ONE rigid body, and the overlay
      // should only ever appear on geometry you can see. Skip a line if either
      // endpoint sits on a destroyed chunk (its mesh is hidden) or if the two
      // endpoints resolve to different bodies (a bond that has broken / is mid-split).
      // Drawing those was what produced stray green lines floating in mid-air with
      // nothing attached: e.g. an un-cut bond on a destroyed chunk whose body had
      // flown off, or a bond stretched between a settled fragment and a flying one.
      if (chunk0.destroyed || chunk1.destroyed) continue;
      const bh = chunk0.bodyHandle;
      if (bh == null || bh !== chunk1.bodyHandle) continue;

      const pose = this.poseForBody(core, bh);
      if (!pose) continue;

      const line = lines[i];
      const base = written * 6;

      // Both endpoints share the owning body's pose.
      this.v.set(line.p0.x, line.p0.y, line.p0.z).applyQuaternion(pose.q).add(pose.t);
      this.positions[base] = this.v.x;
      this.positions[base + 1] = this.v.y;
      this.positions[base + 2] = this.v.z;

      this.v.set(line.p1.x, line.p1.y, line.p1.z).applyQuaternion(pose.q).add(pose.t);
      this.positions[base + 3] = this.v.x;
      this.positions[base + 4] = this.v.y;
      this.positions[base + 5] = this.v.z;

      // Write colors (inline extraction)
      const col0 = line.color0;
      const col1 = line.color1 ?? col0;
      this.colors[base] = ((col0 >> 16) & 0xff) / 255;
      this.colors[base + 1] = ((col0 >> 8) & 0xff) / 255;
      this.colors[base + 2] = (col0 & 0xff) / 255;
      this.colors[base + 3] = ((col1 >> 16) & 0xff) / 255;
      this.colors[base + 4] = ((col1 >> 8) & 0xff) / 255;
      this.colors[base + 5] = (col1 & 0xff) / 255;

      written++;
    }
    return written;
  }
}
