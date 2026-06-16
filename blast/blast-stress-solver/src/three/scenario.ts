import * as THREE from 'three';
import type { DestructibleCore, ScenarioDesc } from '../rapier/types';
import {
  buildBatchedChunkMesh,
  buildBatchedChunkMeshFromGeometries,
  buildChunkMeshes,
  buildChunkMeshesFromGeometries,
  type BatchedChunkMeshOptions,
  type BatchedChunkMeshResult,
  type ChunkMeshBuildOptions,
  type ChunkMeshBuildResult,
} from './destructible-adapter';

export type ScenarioThreeParameters = {
  /** Per-node RENDER geometry (what the chunk mesh draws). */
  fragmentGeometries?: THREE.BufferGeometry[];
  /**
   * Optional per-node COLLISION geometry, used only to build the convex-hull
   * collider for that node. When present the core prefers it over
   * `fragmentGeometries` for collision, so the render mesh (detailed, possibly
   * concave) and the collider (a tight convex hull) can differ. Omit to keep the
   * legacy behavior of deriving the collider from the render geometry.
   */
  colliderGeometries?: THREE.BufferGeometry[];
};

export function getScenarioFragmentGeometries(
  scenario: ScenarioDesc,
): THREE.BufferGeometry[] | undefined {
  const parameters = scenario.parameters as ScenarioThreeParameters | undefined;
  const geometries = parameters?.fragmentGeometries;
  return Array.isArray(geometries) ? geometries : undefined;
}

export function buildChunkMeshesFromScenario(
  core: DestructibleCore,
  scenario: ScenarioDesc,
  materials?: { deck?: THREE.Material; support?: THREE.Material },
  options?: ChunkMeshBuildOptions,
): ChunkMeshBuildResult {
  const geometries = getScenarioFragmentGeometries(scenario);
  return geometries?.length
    ? buildChunkMeshesFromGeometries(core, geometries, materials, options)
    : buildChunkMeshes(core, materials);
}

export function buildBatchedChunkMeshFromScenario(
  core: DestructibleCore,
  scenario: ScenarioDesc,
  options?: BatchedChunkMeshOptions,
): BatchedChunkMeshResult {
  const geometries = getScenarioFragmentGeometries(scenario);
  return geometries?.length
    ? buildBatchedChunkMeshFromGeometries(core, geometries, options)
    : buildBatchedChunkMesh(core, options);
}
