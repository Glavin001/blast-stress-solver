import * as THREE from 'three';
import type { DestructibleCore, ScenarioDesc } from '../rapier/types';
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
};

export type DestructibleThreeBundle = {
  object: THREE.Group;
  core: DestructibleCore;
  chunkMeshes: THREE.Mesh[] | null;
  batched: BatchedChunkMeshResult | null;
  debugLines: SolverDebugLinesHelper | null;
  update: (options?: {
    debug?: boolean;
    updateBVH?: boolean;
    updateProjectiles?: boolean;
    /** Per-frame override of the material-color view. Defaults to the stored toggle. */
    materialColors?: boolean;
  }) => void;
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

  if (scenario) {
    if (useBatchedMesh) {
      batchedBuild = buildBatchedChunkMeshFromScenario(core, scenario, {
        ...batchedMeshOptions,
        nodeColors,
      });
      root.add(batchedBuild.batchedMesh);
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
    update: (updateOptions) => {
      const nc =
        (updateOptions?.materialColors ?? materialColorsEnabled) ? nodeColors : undefined;
      if (batchedBuild) {
        updateBatchedChunkMesh(core, batchedBuild.batchedMesh, batchedBuild.chunkToInstanceId, {
          updateBVH: updateOptions?.updateBVH,
          nodeColors: nc,
        });
      } else if (chunkBuild) {
        updateChunkMeshes(core, chunkBuild.objects, { nodeColors: nc });
      }

      if (updateOptions?.updateProjectiles ?? true) {
        updateProjectileMeshes(core, root);
      }

      if (debugHelper) {
        const showDebug = updateOptions?.debug ?? initialDebugVisible;
        if (showDebug) {
          debugHelper.update(core, core.getSolverDebugLines(), true);
        } else {
          debugHelper.update(core, [], false);
        }
      }
    },
    setMaterialColors: (on: boolean) => {
      materialColorsEnabled = on;
    },
    dispose: () => {
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
