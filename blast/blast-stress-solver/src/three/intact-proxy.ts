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
   *
   * A part may request a non-box `shape`: `'prism'` draws a triangular gable prism (ridge along the
   * longer horizontal axis of its AABB) so a pitched roof reads as a ridge line, not a flat block.
   */
  parts?: Array<{
    aabbMin: { x: number; y: number; z: number };
    aabbMax: { x: number; y: number; z: number };
    shape?: 'box' | 'prism';
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
  /** The proxy meshes (a group holding the instanced box layer + an instanced gable-prism layer for
   *  roof parts); add to the scene/root. */
  object: THREE.Group;
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
// Quarter-turn about Y: maps the unit prism's ridge (local +X) onto world +Z, for gables whose
// ridge runs along Z (the AABB is deeper than it is wide).
const _RIDGE_Z_Q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
const _ZERO_MATRIX = new THREE.Matrix4().makeScale(0, 0, 0);

/**
 * Unit triangular gable prism, ridge along +X: a 1×1×1 box footprint whose top collapses to a ridge
 * line at y=+0.5, z=0 (eaves at the two lower z edges). Flat-shaded (non-indexed → per-face normals)
 * so the ridge reads crisply. Scaled per part to the roof AABB; rotate 90° about Y for Z-ridge roofs.
 */
function makeUnitGablePrism(): THREE.BufferGeometry {
  // 6 corners: A* on the x=-0.5 end, B* on the x=+0.5 end; *ap = apex (ridge), *0/*1 = z-base edges.
  const A0 = [-0.5, -0.5, -0.5], A1 = [-0.5, -0.5, 0.5], Aap = [-0.5, 0.5, 0];
  const B0 = [0.5, -0.5, -0.5], B1 = [0.5, -0.5, 0.5], Bap = [0.5, 0.5, 0];
  // 8 triangles (CCW outward): 2 gable ends, 2 per sloped roof face, 2 for the underside.
  const tris = [
    A0, A1, Aap,        // -X end
    B0, Bap, B1,        // +X end
    A1, B1, Bap, A1, Bap, Aap, // +Z slope
    A0, Aap, Bap, A0, Bap, B0, // -Z slope
    A0, B1, A1, A0, B0, B1,    // underside (-Y)
  ];
  const positions = new Float32Array(tris.length * 3);
  for (let i = 0; i < tris.length; i++) {
    positions[i * 3] = tris[i][0];
    positions[i * 3 + 1] = tris[i][1];
    positions[i * 3 + 2] = tris[i][2];
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.computeVertexNormals();
  return geom;
}

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
  // Two instanced layers sharing one material: plain boxes (walls / columns / slabs / fallback) and
  // gable prisms (roof parts). A part picks a layer by its `shape`; both are drawn instanced.
  const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
  const prismGeometry = makeUnitGablePrism();

  // Each building is drawn as one shaped proxy per LOD leaf ("part": wall / column / slab / roof) so
  // the intact silhouette reads faithfully — pillars, slabs and the roof ridge survive — instead of
  // a single building-sized block. A building with no parts falls back to one box around its whole
  // AABB (instance i ⇔ building i in the box layer for that simple case).
  type Slot = { layer: 0 | 1; index: number; matrix: THREE.Matrix4 }; // layer 0 = boxes, 1 = prisms
  const buildingSlots: Slot[][] = new Array(count);
  const center: THREE.Vector3[] = new Array(count);
  const fragInstances: number[][] = new Array(count);
  let nBox = 0;
  let nPrism = 0;

  // World matrix for a gable prism filling `box`: the unit prism's ridge (local +X) runs along the
  // AABB's longer horizontal axis (X as-is, or Z via a quarter-turn about Y), apex at the AABB top.
  const prismMatrix = (sx: number, sy: number, sz: number): THREE.Matrix4 => {
    if (sx >= sz) {
      _size.set(sx, sy, sz);
      return new THREE.Matrix4().compose(_center, _IDENT_Q, _size);
    }
    _size.set(sz, sy, sx); // local X (length) → world Z; local Z (base width) → world X
    return new THREE.Matrix4().compose(_center, _RIDGE_Z_Q, _size);
  };

  for (let i = 0; i < count; i++) {
    const s = states0[i];
    _center.set(
      (s.aabbMin.x + s.aabbMax.x) / 2,
      (s.aabbMin.y + s.aabbMax.y) / 2,
      (s.aabbMin.z + s.aabbMax.z) / 2,
    );
    center[i] = _center.clone(); // distance test uses the whole-building center
    const parts = s.parts && s.parts.length > 0 ? s.parts : [{ aabbMin: s.aabbMin, aabbMax: s.aabbMax }];
    const slots: Slot[] = new Array(parts.length);
    for (let b = 0; b < parts.length; b++) {
      const part = parts[b];
      _center.set(
        (part.aabbMin.x + part.aabbMax.x) / 2,
        (part.aabbMin.y + part.aabbMax.y) / 2,
        (part.aabbMin.z + part.aabbMax.z) / 2,
      );
      const sxv = Math.max(1e-3, part.aabbMax.x - part.aabbMin.x);
      const syv = Math.max(1e-3, part.aabbMax.y - part.aabbMin.y);
      const szv = Math.max(1e-3, part.aabbMax.z - part.aabbMin.z);
      if (part.shape === 'prism') {
        slots[b] = { layer: 1, index: nPrism++, matrix: prismMatrix(sxv, syv, szv) };
      } else {
        _size.set(sxv, syv, szv);
        slots[b] = { layer: 0, index: nBox++, matrix: new THREE.Matrix4().compose(_center, _IDENT_Q, _size) };
      }
    }
    buildingSlots[i] = slots;
    const insts: number[] = [];
    for (const ni of s.fragments) {
      const id = chunkToInstanceId.get(ni);
      if (id != null) insts.push(id);
    }
    fragInstances[i] = insts;
  }

  const boxMesh = new THREE.InstancedMesh(boxGeometry, material, Math.max(1, nBox));
  const prismMesh = new THREE.InstancedMesh(prismGeometry, material, Math.max(1, nPrism));
  const layers: THREE.InstancedMesh[] = [boxMesh, prismMesh];
  const group = new THREE.Group();
  for (const mesh of layers) {
    mesh.castShadow = options.castShadow ?? true;
    mesh.receiveShadow = options.receiveShadow ?? true;
    // Instances are scattered across the whole city; cull per-building via proxy state, not the
    // mesh's (huge, would-be-stale) bounding sphere.
    mesh.frustumCulled = false;
    group.add(mesh);
  }
  boxMesh.count = nBox;
  prismMesh.count = nPrism;

  for (let i = 0; i < count; i++) {
    const c = options.buildingColors?.[i];
    for (const slot of buildingSlots[i]) {
      const mesh = layers[slot.layer];
      // Start hidden (proxies off, real fragments shown) until the first update() decides.
      mesh.setMatrixAt(slot.index, _ZERO_MATRIX);
      if (c) mesh.setColorAt(slot.index, c);
    }
  }
  for (const mesh of layers) {
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  const instanceHidden = new Uint8Array(instanceCount);
  const proxied = new Uint8Array(count);
  const layerDirty = [false, false];
  let enabled = true;
  let nProxied = 0;

  function setBuildingProxied(i: number, on: boolean) {
    if (on === (proxied[i] !== 0)) return;
    proxied[i] = on ? 1 : 0;
    for (const slot of buildingSlots[i]) {
      layers[slot.layer].setMatrixAt(slot.index, on ? slot.matrix : _ZERO_MATRIX);
      layerDirty[slot.layer] = true;
    }
    const insts = fragInstances[i];
    const v = on ? 1 : 0;
    for (let k = 0; k < insts.length; k++) instanceHidden[insts[k]] = v;
    nProxied += on ? 1 : -1;
  }

  function flushDirty() {
    for (let l = 0; l < layers.length; l++) {
      if (layerDirty[l]) {
        layers[l].instanceMatrix.needsUpdate = true;
        layerDirty[l] = false;
      }
    }
  }

  function update(camera: THREE.Camera) {
    const states = getStates();
    camera.getWorldPosition(_camPos);
    const n = Math.min(states.length, count);
    for (let i = 0; i < n; i++) {
      const want = enabled
        ? decideProxied(proxied[i] !== 0, states[i].intact, _camPos.distanceTo(center[i]), nearD, farD)
        : false;
      if (want !== (proxied[i] !== 0)) setBuildingProxied(i, want);
    }
    flushDirty();
  }

  function setEnabled(on: boolean) {
    if (on === enabled) return;
    enabled = on;
    if (!on) {
      for (let i = 0; i < count; i++) if (proxied[i]) setBuildingProxied(i, false);
      flushDirty();
    }
    // When re-enabled, the next update() re-proxies as the camera/intactness dictate.
  }

  function dispose() {
    try {
      boxGeometry.dispose();
    } catch {}
    try {
      prismGeometry.dispose();
    } catch {}
    if (!options.material) {
      try {
        material.dispose();
      } catch {}
    }
    for (const mesh of layers) {
      try {
        mesh.dispose();
      } catch {}
    }
  }

  return {
    object: group,
    instanceHidden,
    update,
    proxiedCount: () => nProxied,
    setEnabled,
    dispose,
  };
}
