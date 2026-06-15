import * as THREE from 'three';

/**
 * Intact-building render LOD.
 *
 * A fractured building is drawn as N fragment instances in a {@link THREE.BatchedMesh}. While the
 * building is still intact (un-hit) it looks like one solid shell, so — especially at distance —
 * a single box is visually indistinguishable from those N fragments but costs one instance and a
 * dozen triangles instead of hundreds. This module renders those proxy boxes in one
 * {@link THREE.InstancedMesh} (identical box geometry → the textbook instancing case) and reports,
 * per frame, which fragment instances a proxy currently stands in for so the caller can hide them.
 *
 * A building reverts to its real fragments when (a) it is no longer intact (it got hit), or (b)
 * the camera comes close enough to notice the difference. Distance gating uses hysteresis so a
 * building straddling the threshold doesn't flicker.
 */

export type BuildingRenderState = {
  buildingId: number;
  intact: boolean;
  aabbMin: { x: number; y: number; z: number };
  aabbMax: { x: number; y: number; z: number };
  fragments: number[];
  /**
   * Optional per-element AABBs (the building's LOD leaves: wall / column / slab / roof). When
   * present the proxy draws one cheap box per part — a faithful blocky silhouette (pillars, slabs,
   * roof read correctly) — instead of one building-sized box. Omit (or empty) to fall back to a
   * single box around the whole building.
   */
  parts?: Array<{
    aabbMin: { x: number; y: number; z: number };
    aabbMax: { x: number; y: number; z: number };
  }>;
};

export type IntactBuildingProxyOptions = {
  /** Live per-building render state, e.g. `() => core.getBuildingRenderStates!()`. */
  getStates: () => BuildingRenderState[];
  /** Fragment `nodeIndex` → batched `instanceId`, so a proxy can mask the fragments it replaces. */
  chunkToInstanceId: Map<number, number>;
  /** Length of the fragment instance set (size of the returned hidden mask). */
  instanceCount: number;
  /** Proxy a building once farther than this from the camera. `<= 0` → always proxy while intact. */
  farDistance?: number;
  /** Reveal real fragments again once nearer than this (hysteresis; should be `< farDistance`). */
  nearDistance?: number;
  /** Optional per-building proxy color (aligned with `getStates()` order); omit for the material. */
  buildingColors?: Array<THREE.Color | undefined>;
  material?: THREE.Material;
  castShadow?: boolean;
  receiveShadow?: boolean;
};

export type IntactBuildingProxies = {
  /** The proxy boxes; add to the scene/root. */
  object: THREE.InstancedMesh;
  /** Mask by `instanceId`: non-zero → fragment is currently replaced by a proxy. Feed to
   *  `updateBatchedChunkMesh({ instanceHidden })`. The same array is mutated in place each frame. */
  instanceHidden: Uint8Array;
  /** Recompute proxy-vs-fragments for this camera. MUST run before `updateBatchedChunkMesh`. */
  update: (camera: THREE.Camera) => void;
  /** Buildings currently drawn as a proxy (for stats overlays). */
  proxiedCount: () => number;
  /** Turn the whole layer on/off; when off, every building shows real fragments. */
  setEnabled: (on: boolean) => void;
  dispose: () => void;
};

const _camPos = new THREE.Vector3();
const _center = new THREE.Vector3();
const _size = new THREE.Vector3();
const _IDENT_Q = new THREE.Quaternion();
const _ZERO_MATRIX = new THREE.Matrix4().makeScale(0, 0, 0);

/**
 * Should a building be drawn as a proxy box this frame? Pure so the policy is unit-testable.
 * Hysteresis: a non-proxied building must get farther than `farD` to start; a proxied one stays
 * a proxy until it gets nearer than `nearD`.
 */
export function decideProxied(
  prev: boolean,
  intact: boolean,
  dist: number,
  nearD: number,
  farD: number,
): boolean {
  if (!intact) return false;
  if (farD <= 0) return true; // distance gating disabled → always proxy while intact
  if (prev) return dist > nearD;
  return dist > farD;
}

export function createIntactBuildingProxies(
  options: IntactBuildingProxyOptions,
): IntactBuildingProxies {
  const { getStates, chunkToInstanceId, instanceCount } = options;
  const farD = options.farDistance ?? 70;
  const nearD = options.nearDistance ?? farD * 0.65;

  const states0 = getStates();
  const count = states0.length;

  const material =
    options.material ??
    new THREE.MeshStandardMaterial({ color: 0xbababa, roughness: 0.5, metalness: 0.1 });
  const geometry = new THREE.BoxGeometry(1, 1, 1);

  // Each building is drawn as one box per LOD leaf ("part": wall / column / slab / roof) so the
  // intact silhouette reads faithfully — pillars, slabs and roofs survive — instead of a single
  // building-sized block. A building with no parts falls back to one box around its whole AABB, so
  // the slot layout stays 1:1 with buildings in that case (instance i ⇔ building i).
  const boxMatrices: THREE.Matrix4[][] = new Array(count); // per building → one matrix per slot
  const slotBase: number[] = new Array(count); // first global instance index for this building
  const center: THREE.Vector3[] = new Array(count);
  const fragInstances: number[][] = new Array(count);
  let totalSlots = 0;
  for (let i = 0; i < count; i++) {
    const s = states0[i];
    _center.set(
      (s.aabbMin.x + s.aabbMax.x) / 2,
      (s.aabbMin.y + s.aabbMax.y) / 2,
      (s.aabbMin.z + s.aabbMax.z) / 2,
    );
    center[i] = _center.clone(); // distance test uses the whole-building center
    const boxes = s.parts && s.parts.length > 0 ? s.parts : [{ aabbMin: s.aabbMin, aabbMax: s.aabbMax }];
    const mats: THREE.Matrix4[] = new Array(boxes.length);
    for (let b = 0; b < boxes.length; b++) {
      const box = boxes[b];
      _center.set(
        (box.aabbMin.x + box.aabbMax.x) / 2,
        (box.aabbMin.y + box.aabbMax.y) / 2,
        (box.aabbMin.z + box.aabbMax.z) / 2,
      );
      _size.set(
        Math.max(1e-3, box.aabbMax.x - box.aabbMin.x),
        Math.max(1e-3, box.aabbMax.y - box.aabbMin.y),
        Math.max(1e-3, box.aabbMax.z - box.aabbMin.z),
      );
      mats[b] = new THREE.Matrix4().compose(_center, _IDENT_Q, _size);
    }
    boxMatrices[i] = mats;
    slotBase[i] = totalSlots;
    totalSlots += mats.length;
    const insts: number[] = [];
    for (const ni of s.fragments) {
      const id = chunkToInstanceId.get(ni);
      if (id != null) insts.push(id);
    }
    fragInstances[i] = insts;
  }

  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, totalSlots));
  mesh.castShadow = options.castShadow ?? true;
  mesh.receiveShadow = options.receiveShadow ?? true;
  // Instances are scattered across the whole city; cull per-building via proxy state, not the
  // mesh's (huge, would-be-stale) bounding sphere.
  mesh.frustumCulled = false;
  mesh.count = totalSlots;
  for (let i = 0; i < count; i++) {
    const mats = boxMatrices[i];
    const c = options.buildingColors?.[i];
    for (let b = 0; b < mats.length; b++) {
      // Start hidden (boxes off, real fragments shown) until the first update() decides.
      mesh.setMatrixAt(slotBase[i] + b, _ZERO_MATRIX);
      if (c) mesh.setColorAt(slotBase[i] + b, c);
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

  const instanceHidden = new Uint8Array(instanceCount);
  const proxied = new Uint8Array(count);
  let enabled = true;
  let nProxied = 0;

  function setBuildingProxied(i: number, on: boolean) {
    if (on === (proxied[i] !== 0)) return;
    proxied[i] = on ? 1 : 0;
    const mats = boxMatrices[i];
    for (let b = 0; b < mats.length; b++) mesh.setMatrixAt(slotBase[i] + b, on ? mats[b] : _ZERO_MATRIX);
    const insts = fragInstances[i];
    const v = on ? 1 : 0;
    for (let k = 0; k < insts.length; k++) instanceHidden[insts[k]] = v;
    nProxied += on ? 1 : -1;
  }

  function update(camera: THREE.Camera) {
    const states = getStates();
    camera.getWorldPosition(_camPos);
    const n = Math.min(states.length, count);
    let matrixDirty = false;
    for (let i = 0; i < n; i++) {
      const want = enabled
        ? decideProxied(proxied[i] !== 0, states[i].intact, _camPos.distanceTo(center[i]), nearD, farD)
        : false;
      if (want !== (proxied[i] !== 0)) {
        setBuildingProxied(i, want);
        matrixDirty = true;
      }
    }
    if (matrixDirty) mesh.instanceMatrix.needsUpdate = true;
  }

  function setEnabled(on: boolean) {
    if (on === enabled) return;
    enabled = on;
    if (!on) {
      let dirty = false;
      for (let i = 0; i < count; i++) {
        if (proxied[i]) {
          setBuildingProxied(i, false);
          dirty = true;
        }
      }
      if (dirty) mesh.instanceMatrix.needsUpdate = true;
    }
    // When re-enabled, the next update() re-proxies as the camera/intactness dictate.
  }

  function dispose() {
    try {
      geometry.dispose();
    } catch {}
    if (!options.material) {
      try {
        material.dispose();
      } catch {}
    }
    try {
      mesh.dispose();
    } catch {}
  }

  return {
    object: mesh,
    instanceHidden,
    update,
    proxiedCount: () => nProxied,
    setEnabled,
    dispose,
  };
}
