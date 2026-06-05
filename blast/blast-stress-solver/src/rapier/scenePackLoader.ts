/**
 * Loader for the shared scene-pack JSON (the same files the Bevy demo loads).
 *
 * This lets the web demo render the EXACT building that ships in the Rust demo from
 * one source of truth, instead of rebuilding it procedurally. It mirrors the Rust
 * loader (scene_pack.rs / scene_json.rs): validates the schema, builds a ScenarioDesc
 * (with per-node sizes so the core/bundle render & collide chunks as boxes), and
 * surfaces the scene defaults — including the optional decoupled stress `limits`,
 * returned as `solverSettings` so they win over the materialScale-scaled defaults in
 * buildDestructibleCore.
 *
 * For all-box scenes `nodeMeshes` is omitted; box geometry is derived from
 * `nodeSizes` by the renderer, so no mesh data is needed here.
 */
import type { ScenarioDesc, Vec3 } from './types';
import type { ExtStressSolverSettings } from '../types';

export type ScenePackProjectileDefaults = {
  radius: number;
  mass: number;
  speed: number;
  ttlMs: number;
};

export type ScenePackDefaults = {
  camera: { target: Vec3; distance: number };
  projectile: ScenePackProjectileDefaults;
  gravity: number;
  materialScale: number;
  /** Explicit decoupled stress limits mapped to ExtStressSolverSettings, if present. */
  solverSettings?: ExtStressSolverSettings;
  physics: {
    debrisCollisionMode?: string;
    friction: number;
    restitution: number;
    contactForceScale: number;
    skipSingleBodies?: boolean;
  };
  optimization: {
    smallBodyDampingMode: string;
    debrisCleanupMode: string;
    debrisTtlMs: number;
    maxCollidersForDebris: number;
  };
  /** Contact-damage options (per-chunk health + splash) — the local-destruction knob. */
  damage?: Record<string, unknown>;
};

export type LoadedScenePack = {
  title: string;
  scenario: ScenarioDesc;
  defaults: ScenePackDefaults;
  /** Per-node structural role, if the pack carried `nodeTypes`. */
  nodeTypes: string[];
};

type RawVec3 = { x: number; y: number; z: number };

type RawLimits = {
  compressionElastic: number;
  compressionFatal: number;
  tensionElastic: number;
  tensionFatal: number;
  shearElastic: number;
  shearFatal: number;
};

type RawScenePack = {
  version: number;
  title?: string;
  defaults: {
    camera: { target: RawVec3; distance: number };
    projectile: ScenePackProjectileDefaults;
    solver: { gravity: number; materialScale: number; limits?: RawLimits };
    physics: ScenePackDefaults['physics'];
    optimization: ScenePackDefaults['optimization'];
    damage?: Record<string, unknown>;
  };
  scenario: {
    nodes: Array<{ centroid: RawVec3; mass: number; volume: number }>;
    bonds: Array<{ node0: number; node1: number; centroid: RawVec3; normal: RawVec3; area: number }>;
    nodeSizes: RawVec3[];
    nodeColliders?: unknown[];
    nodeTypes?: string[];
  };
  nodeMeshes?: unknown[];
};

function limitsToSolverSettings(l: RawLimits): ExtStressSolverSettings {
  return {
    compressionElasticLimit: l.compressionElastic,
    compressionFatalLimit: l.compressionFatal,
    tensionElasticLimit: l.tensionElastic,
    tensionFatalLimit: l.tensionFatal,
    shearElasticLimit: l.shearElastic,
    shearFatalLimit: l.shearFatal,
  };
}

/** Parse a scene-pack object (already JSON.parse'd) into a LoadedScenePack. */
export function parseScenePack(raw: unknown): LoadedScenePack {
  const pack = raw as RawScenePack;
  if (!pack || typeof pack !== 'object') throw new Error('scene pack: not an object');
  if (pack.version !== 1) throw new Error(`scene pack: unsupported version ${pack.version}`);

  const s = pack.scenario;
  if (!s || !Array.isArray(s.nodes) || !Array.isArray(s.bonds)) {
    throw new Error('scene pack: missing scenario.nodes / scenario.bonds');
  }
  if (s.nodeSizes.length !== s.nodes.length) {
    throw new Error(
      `scene pack: nodeSizes (${s.nodeSizes.length}) != nodes (${s.nodes.length})`,
    );
  }
  if (pack.nodeMeshes && pack.nodeMeshes.length > 0 && pack.nodeMeshes.length !== s.nodes.length) {
    throw new Error(
      `scene pack: nodeMeshes (${pack.nodeMeshes.length}) != nodes (${s.nodes.length})`,
    );
  }

  const nodes = s.nodes.map((n) => ({
    centroid: { x: n.centroid.x, y: n.centroid.y, z: n.centroid.z },
    mass: n.mass,
    volume: n.volume,
  }));
  const bonds = s.bonds.map((b) => ({
    node0: b.node0,
    node1: b.node1,
    centroid: { x: b.centroid.x, y: b.centroid.y, z: b.centroid.z },
    normal: { x: b.normal.x, y: b.normal.y, z: b.normal.z },
    area: b.area,
  }));
  const fragmentSizes: Vec3[] = s.nodeSizes.map((v) => ({ x: v.x, y: v.y, z: v.z }));

  const scenario: ScenarioDesc = {
    nodes,
    bonds,
    // Per-node sizes drive box colliders (core) and box meshes (bundle); no explicit
    // colliderDescForNode is needed (the core falls back to cuboids from size).
    parameters: { fragmentSizes },
  };

  const d = pack.defaults;
  const defaults: ScenePackDefaults = {
    camera: {
      target: { x: d.camera.target.x, y: d.camera.target.y, z: d.camera.target.z },
      distance: d.camera.distance,
    },
    projectile: d.projectile,
    gravity: d.solver.gravity,
    materialScale: d.solver.materialScale,
    solverSettings: d.solver.limits ? limitsToSolverSettings(d.solver.limits) : undefined,
    physics: d.physics,
    optimization: d.optimization,
    damage: d.damage,
  };

  return {
    title: pack.title ?? 'Scene',
    scenario,
    defaults,
    nodeTypes: Array.isArray(s.nodeTypes) ? s.nodeTypes : [],
  };
}

/** Parse a scene-pack JSON string. */
export function parseScenePackJson(json: string): LoadedScenePack {
  return parseScenePack(JSON.parse(json));
}

/** Fetch and parse a scene-pack JSON file (browser). */
export async function loadScenePackFromUrl(url: string): Promise<LoadedScenePack> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`scene pack: failed to fetch ${url} (${res.status})`);
  return parseScenePack(await res.json());
}
