import * as THREE from 'three';
import type { DestructibleCore, ScenarioDesc } from '../rapier/types';
import type { InterpolatedPoseView } from '../rapier/fixedStepLoop';
import {
  buildBatchedChunkMeshFromScenario,
  buildChunkMeshesFromScenario,
} from './scenario';
import {
  SolverDebugLinesHelper,
  updateBatchedChunkMesh,
  updateChunkMeshes,
  updateProjectileMeshes,
  type BatchedChunkMeshOptions,
  type BatchedChunkMeshResult,
  type ChunkMeshBuildOptions,
  type ChunkMeshBuildResult,
} from './destructible-adapter';
import {
  createIntactBuildingProxies,
  type IntactBuildingProxies,
} from './intact-proxy';

export type CreateDestructibleThreeBundleOptions = {
  core: DestructibleCore;
  scenario?: ScenarioDesc;
  root?: THREE.Group;
  useBatchedMesh?: boolean;
  materials?: { deck?: THREE.Material; support?: THREE.Material };
  chunkMeshOptions?: ChunkMeshBuildOptions;
  batchedMeshOptions?: BatchedChunkMeshOptions;
  includeDebugLines?: boolean;
  initialDebugVisible?: boolean;
  /**
   * Optional per-node material colors (indexed by `chunk.nodeIndex`). When provided,
   * chunks render in their material color (the "material" view); call
   * `setMaterialColors(false)` to flip to the standard fixed/kinematic/dynamic state
   * coloring at runtime. Omit for the default state coloring.
   */
  nodeColors?: THREE.Color[];
  /** Start in material-color view when `nodeColors` is set. Default: true. */
  materialColors?: boolean;
  /**
   * Intact-building render LOD: collapse a still-intact building to a single proxy box (batched
   * mode only). Requires `core.getBuildingRenderStates`. Off unless `enabled`. Pass `camera` to
   * `update()` for the distance gating to apply.
   */
  intactProxies?: {
    enabled?: boolean;
    /** Proxy a building once farther than this (world units). `<= 0` → always proxy while intact. */
    farDistance?: number;
    /** Reveal real fragments again once nearer than this (hysteresis; `< farDistance`). */
    nearDistance?: number;
  };
};

export type DestructibleThreeBundle = {
  object: THREE.Group;
  core: DestructibleCore;
  chunkMeshes: THREE.Mesh[] | null;
  batched: BatchedChunkMeshResult | null;
  debugLines: SolverDebugLinesHelper | null;
  /** Intact-building proxy layer, when enabled (batched mode). */
  intactProxies: IntactBuildingProxies | null;
  update: (options?: {
    debug?: boolean;
    updateBVH?: boolean;
    updateProjectiles?: boolean;
    /** Per-frame override of the material-color view. Defaults to the stored toggle. */
    materialColors?: boolean;
    /** Camera for intact-proxy distance gating. Omit → proxies use their last decision. */
    camera?: THREE.Camera;
    /** Fixed-timestep render interpolation (`createPoseInterpolator(core).view(alpha)`):
     *  chunk instances and projectile meshes blend between the last two physics states.
     *  Batched-mesh mode only. */
    interpolation?: InterpolatedPoseView;
  }) => { batchedWrites: number } | undefined;
  /** Flip between material colors (true) and physics-state colors (false) at runtime. */
  setMaterialColors: (on: boolean) => void;
  dispose: () => void;
};

function disposeChunkBuild(
  root: THREE.Group,
  chunkBuild: ChunkMeshBuildResult | null,
) {
  if (!chunkBuild) return;
  for (const mesh of chunkBuild.objects) {
    try {
      root.remove(mesh);
    } catch {}
  }
  try {
    chunkBuild.dispose();
  } catch {}
}

export function createDestructibleThreeBundle(
  options: CreateDestructibleThreeBundleOptions,
): DestructibleThreeBundle {
  const {
    core,
    scenario,
    root = new THREE.Group(),
    useBatchedMesh = false,
    materials,
    chunkMeshOptions,
    batchedMeshOptions,
    includeDebugLines = true,
    initialDebugVisible = false,
    nodeColors,
  } = options;

  // Mutable "material vs. state" color toggle (only meaningful when nodeColors is set).
  let materialColorsEnabled = options.materialColors ?? true;

  let chunkBuild: ChunkMeshBuildResult | null = null;
  let batchedBuild: BatchedChunkMeshResult | null = null;
  let intactProxies: IntactBuildingProxies | null = null;

  if (scenario) {
    if (useBatchedMesh) {
      batchedBuild = buildBatchedChunkMeshFromScenario(core, scenario, {
        ...batchedMeshOptions,
        nodeColors,
      });
      root.add(batchedBuild.batchedMesh);

      // Build the proxy layer whenever it is configured (and the core supports it) so it can be
      // toggled live; `enabled: false` just starts it dormant (boxes off, all fragments shown).
      if (options.intactProxies && typeof core.getBuildingRenderStates === 'function') {
        const getStates = core.getBuildingRenderStates.bind(core);
        // One representative color per building (its first fragment) so distant proxies match the
        // city's material-color view; harmless under state coloring.
        const buildingColors = nodeColors
          ? getStates().map((s) => nodeColors[s.fragments[0]])
          : undefined;
        intactProxies = createIntactBuildingProxies({
          getStates,
          chunkToInstanceId: batchedBuild.chunkToInstanceId,
          instanceCount: core.chunks.length,
          farDistance: options.intactProxies.farDistance,
          nearDistance: options.intactProxies.nearDistance,
          buildingColors,
        });
        if (options.intactProxies.enabled === false) intactProxies.setEnabled(false);
        root.add(intactProxies.object);
      }
    } else {
      chunkBuild = buildChunkMeshesFromScenario(core, scenario, materials, {
        ...chunkMeshOptions,
        nodeColors,
      });
      for (const mesh of chunkBuild.objects) {
        root.add(mesh);
      }
    }
  }

  let debugHelper: SolverDebugLinesHelper | null = null;
  if (includeDebugLines) {
    debugHelper = new SolverDebugLinesHelper();
    debugHelper.object.visible = initialDebugVisible;
    root.add(debugHelper.object);
  }

  return {
    object: root,
    core,
    chunkMeshes: chunkBuild?.objects ?? null,
    batched: batchedBuild,
    debugLines: debugHelper,
    intactProxies,
    update: (updateOptions) => {
      const nc =
        (updateOptions?.materialColors ?? materialColorsEnabled) ? nodeColors : undefined;
      let batchedWrites: number | undefined;
      if (batchedBuild) {
        // Decide proxy-vs-fragments first so the hidden mask reflects this frame's camera.
        if (intactProxies && updateOptions?.camera) {
          intactProxies.update(updateOptions.camera);
        }
        batchedWrites = updateBatchedChunkMesh(core, batchedBuild.batchedMesh, batchedBuild.chunkToInstanceId, {
          updateBVH: updateOptions?.updateBVH,
          nodeColors: nc,
          instanceHidden: intactProxies?.instanceHidden,
          interpolation: updateOptions?.interpolation,
        }).writes;
      } else if (chunkBuild) {
        updateChunkMeshes(core, chunkBuild.objects, { nodeColors: nc });
      }

      if (updateOptions?.updateProjectiles ?? true) {
        updateProjectileMeshes(core, root, { interpolation: updateOptions?.interpolation });
      }

      if (debugHelper) {
        const showDebug = updateOptions?.debug ?? initialDebugVisible;
        if (showDebug) {
          debugHelper.update(core, core.getSolverDebugLines(), true);
        } else {
          debugHelper.update(core, [], false);
        }
      }
      return batchedWrites === undefined ? undefined : { batchedWrites };
    },
    setMaterialColors: (on: boolean) => {
      materialColorsEnabled = on;
    },
    dispose: () => {
      if (intactProxies) {
        try {
          root.remove(intactProxies.object);
        } catch {}
        try {
          intactProxies.dispose();
        } catch {}
      }

      if (batchedBuild) {
        try {
          root.remove(batchedBuild.batchedMesh);
        } catch {}
        try {
          batchedBuild.dispose();
        } catch {}
      }

      disposeChunkBuild(root, chunkBuild);

      if (debugHelper) {
        try {
          root.remove(debugHelper.object);
        } catch {}
        try {
          debugHelper.dispose();
        } catch {}
      }
    },
  };
}
