import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  decideProxied,
  createIntactBuildingProxies,
  type BuildingRenderState,
} from '../three/intact-proxy';

describe('decideProxied', () => {
  it('never proxies a non-intact building', () => {
    expect(decideProxied(false, false, 1000, 30, 50)).toBe(false);
    expect(decideProxied(true, false, 1000, 30, 50)).toBe(false);
  });

  it('always proxies an intact building when distance gating is off (farD <= 0)', () => {
    expect(decideProxied(false, true, 0, 0, 0)).toBe(true);
    expect(decideProxied(false, true, 5, 0, -1)).toBe(true);
  });

  it('applies hysteresis around the near/far band', () => {
    // not yet a proxy: must exceed farD to start
    expect(decideProxied(false, true, 40, 30, 50)).toBe(false); // inside band, stays detailed
    expect(decideProxied(false, true, 60, 30, 50)).toBe(true); // beyond farD, becomes proxy
    // already a proxy: stays until inside nearD
    expect(decideProxied(true, true, 40, 30, 50)).toBe(true); // inside band, stays a proxy
    expect(decideProxied(true, true, 20, 30, 50)).toBe(false); // nearer than nearD, reverts
  });
});

describe('createIntactBuildingProxies', () => {
  function box(cx: number): { aabbMin: THREE.Vector3Like; aabbMax: THREE.Vector3Like } {
    return { aabbMin: { x: cx - 5, y: 0, z: -5 }, aabbMax: { x: cx + 5, y: 20, z: 5 } };
  }

  function setup() {
    // building 0 centered at x=0 (fragments 0,1), building 1 centered at x=100 (fragments 2,3)
    const states: BuildingRenderState[] = [
      { buildingId: 0, intact: true, fragments: [0, 1], ...box(0) },
      { buildingId: 1, intact: true, fragments: [2, 3], ...box(100) },
    ];
    const chunkToInstanceId = new Map<number, number>([
      [0, 0],
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    const proxies = createIntactBuildingProxies({
      getStates: () => states,
      chunkToInstanceId,
      instanceCount: 4,
      farDistance: 50,
      nearDistance: 30,
    });
    return { states, proxies };
  }

  function cameraAt(x: number, y: number, z: number): THREE.Camera {
    const cam = new THREE.PerspectiveCamera();
    cam.position.set(x, y, z);
    cam.updateMatrixWorld(true);
    return cam;
  }

  // The proxy `object` is a group of instanced layers; layer 0 is the box mesh, layer 1 the prisms.
  const boxLayer = (proxies: { object: THREE.Group }) => proxies.object.children[0] as THREE.InstancedMesh;
  const prismLayer = (proxies: { object: THREE.Group }) => proxies.object.children[1] as THREE.InstancedMesh;

  it('proxies all intact buildings when the camera is far, masking their fragments', () => {
    const { proxies } = setup();
    proxies.update(cameraAt(0, 0, 300)); // far from both
    expect(proxies.proxiedCount()).toBe(2);
    expect(Array.from(proxies.instanceHidden)).toEqual([1, 1, 1, 1]);
    expect(boxLayer(proxies).count).toBe(2); // one fallback box per (part-less) building
  });

  it('reveals fragments of the building the camera approaches (hysteresis)', () => {
    const { proxies } = setup();
    proxies.update(cameraAt(0, 0, 300)); // both proxied
    proxies.update(cameraAt(0, 0, 10)); // close to building 0 only
    expect(Array.from(proxies.instanceHidden)).toEqual([0, 0, 1, 1]);
    expect(proxies.proxiedCount()).toBe(1);
  });

  it('reverts a building as soon as it stops being intact, even when far', () => {
    const { states, proxies } = setup();
    proxies.update(cameraAt(0, 0, 300)); // both proxied
    states[1].intact = false; // building 1 gets hit
    proxies.update(cameraAt(0, 0, 300));
    expect(Array.from(proxies.instanceHidden)).toEqual([1, 1, 0, 0]);
    expect(proxies.proxiedCount()).toBe(1);
  });

  it('setEnabled(false) reveals every fragment', () => {
    const { proxies } = setup();
    proxies.update(cameraAt(0, 0, 300));
    proxies.setEnabled(false);
    expect(Array.from(proxies.instanceHidden)).toEqual([0, 0, 0, 0]);
    expect(proxies.proxiedCount()).toBe(0);
  });

  it('writes a non-degenerate box matrix only for proxied buildings', () => {
    const { proxies } = setup();
    const mesh = boxLayer(proxies);
    proxies.update(cameraAt(0, 0, 10)); // building 0 detailed (near), building 1 proxied (far)
    const m0 = new THREE.Matrix4();
    const m1 = new THREE.Matrix4();
    mesh.getMatrixAt(0, m0);
    mesh.getMatrixAt(1, m1);
    expect(m0.elements[0]).toBe(0); // building 0 not proxied → zero-scale
    expect(m1.elements[0]).toBeGreaterThan(0); // building 1 proxied → real box
  });

  it('draws roof parts as prisms and other parts as boxes, masking all fragments', () => {
    // One building with a column box (fragment 0) and a gable roof prism (fragments 1,2).
    const states: BuildingRenderState[] = [
      {
        buildingId: 0,
        intact: true,
        fragments: [0, 1, 2],
        aabbMin: { x: -4, y: 0, z: -3 },
        aabbMax: { x: 4, y: 6, z: 3 },
        parts: [
          { aabbMin: { x: -0.3, y: 0, z: -0.3 }, aabbMax: { x: 0.3, y: 3, z: 0.3 } }, // column → box
          { aabbMin: { x: -4, y: 3, z: -3 }, aabbMax: { x: 4, y: 6, z: 3 }, shape: 'prism' }, // roof → prism
        ],
      },
    ];
    const proxies = createIntactBuildingProxies({
      getStates: () => states,
      chunkToInstanceId: new Map([[0, 0], [1, 1], [2, 2]]),
      instanceCount: 3,
      farDistance: 50,
      nearDistance: 30,
    });
    expect(boxLayer(proxies).count).toBe(1);
    expect(prismLayer(proxies).count).toBe(1);

    proxies.update(cameraAt(0, 0, 300)); // far → proxied
    expect(proxies.proxiedCount()).toBe(1);
    expect(Array.from(proxies.instanceHidden)).toEqual([1, 1, 1]); // every fragment masked

    // The prism slot carries a real (non-degenerate) transform when proxied, with the ridge along
    // the AABB's longer horizontal axis (X here: width 8 > depth 6).
    const pm = new THREE.Matrix4();
    prismLayer(proxies).getMatrixAt(0, pm);
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    pm.decompose(pos, quat, scl);
    expect(scl.x).toBeCloseTo(8); // ridge axis (X) not rotated → local X = world width
    expect(scl.y).toBeCloseTo(3);
    expect(scl.z).toBeCloseTo(6);
    expect(quat.angleTo(new THREE.Quaternion())).toBeCloseTo(0); // no Y rotation for an X ridge
  });
});
