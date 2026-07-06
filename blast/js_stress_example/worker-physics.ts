/**
 * Main-thread driver for the physics worker (roadmap Tier 3.1).
 *
 * `createWorkerCore(scenario, options)` spawns the bundled physics worker, hands it the
 * geometry-stripped scenario plus a SharedArrayBuffer pose channel, and returns a
 * `WorkerCoreView` — an object implementing the slice of the `DestructibleCore` surface
 * the render bundle and the mini-city HUD consume (chunks with live mirrored poses,
 * projectiles, stats getters, command methods). The Three.js adapter renders it exactly
 * like a local core: mirrored chunk/pose objects are mutated in place so the batched
 * sync cache and intact-proxy layer work unchanged.
 *
 * Requirements: `crossOriginIsolated` (COOP/COEP headers — both demo servers and the
 * Vercel config send them) for SharedArrayBuffer. Callers must check
 * `workerPhysicsSupported()` and fall back to a local core otherwise.
 *
 * Not supported in worker mode (the demo disables them): Rapier debug wireframes,
 * solver debug lines, the frame profiler / session recorder (they attach to a local
 * core), and the FPS shooter / drive mode (external bodies must live in the same world).
 */
import {
  isBuildingRenderIntact,
  mirrorPoseFrame,
  poseChannelByteLength,
  viewPoseChannel,
  NO_BODY,
  type MirrorChunk,
  type MirrorProjectile,
  type PoseChannelViews,
} from 'blast-stress-solver/rapier';

type Vec3 = { x: number; y: number; z: number };

export type WorkerCoreStats = {
  bodies: number;
  bonds: number;
  projectiles: number;
  detached: number;
  islands: { enabled: boolean; skipSettled: boolean; islandCount: number; islandsSkipped: number } | null;
  lazy: { enabled: boolean; buildingCount: number; dormantCount: number; explodedCount: number; activeLeafFragments: number } | null;
};

type BuildingStatic = { buildingId: number; aabbMin: Vec3; aabbMax: Vec3; fragments: number[] };

export function workerPhysicsSupported(): boolean {
  return typeof SharedArrayBuffer !== 'undefined' &&
    (typeof crossOriginIsolated === 'undefined' || crossOriginIsolated === true) &&
    typeof Worker !== 'undefined';
}

/** Strip the render-only (non-cloneable) parameters before posting to the worker. */
function stripScenarioForWorker(scenario: Record<string, unknown>): Record<string, unknown> {
  const params = { ...((scenario.parameters as Record<string, unknown>) ?? {}) };
  delete params.fragmentGeometries; // THREE.BufferGeometry — render-side only
  return { ...scenario, parameters: params };
}

// Pooled fake bodies: the adapter's color path only asks the body for its kind.
class KindBody {
  kind = 2;
  isFixed() { return this.kind === 0; }
  isKinematic() { return this.kind === 1; }
  isDynamic() { return this.kind === 2; }
}

export async function createWorkerCore(
  scenario: { nodes: Array<{ centroid: Vec3 }>; parameters?: unknown },
  options: Record<string, unknown>,
  workerUrl = './dist/mini-city-physics-worker.js',
) {
  const chunkCount = scenario.nodes.length;
  const projectileCapacity = 256;
  const sab = new SharedArrayBuffer(poseChannelByteLength(chunkCount, projectileCapacity));
  const views: PoseChannelViews = viewPoseChannel(sab, chunkCount, projectileCapacity);

  const worker = new Worker(workerUrl, { type: 'module' });

  const chunks: MirrorChunk[] = scenario.nodes.map((n, i) => ({
    nodeIndex: i,
    active: true,
    destroyed: false,
    detached: false,
    bodyHandle: null,
    worldPosition: null,
    worldQuaternion: null,
    // The authored centroid doubles as the local offset, matching the local core's
    // chunk fields the adapter falls back to before the first mirrored frame.
    baseLocalOffset: { ...n.centroid },
    localOffset: { ...n.centroid },
  }));
  const projectiles: MirrorProjectile[] = [];
  const bodyKindByHandle = new Map<number, number>();
  const bodyPool = new Map<number, KindBody>();
  const projByHandle = new Map<number, MirrorProjectile>();

  let lastFrame = 0;
  let stats: WorkerCoreStats = { bodies: 0, bonds: 0, projectiles: 0, detached: 0, islands: null, lazy: null };
  let buildingsStatic: BuildingStatic[] = [];
  let buildingStates: Array<BuildingStatic & { intact: boolean }> = [];
  let disposed = false;

  const ready = new Promise<void>((resolve, reject) => {
    worker.onmessage = (e: MessageEvent) => {
      const m = e.data as Record<string, unknown> & { t: string };
      if (m.t === 'ready') {
        buildingsStatic = (m.buildings as BuildingStatic[]) ?? [];
        buildingStates = buildingsStatic.map((b) => ({ ...b, intact: true }));
        resolve();
      } else if (m.t === 'stats') {
        stats = m as unknown as WorkerCoreStats;
      } else if (m.t === 'error') {
        reject(new Error(String(m.message)));
      }
    };
    worker.onerror = (err) => reject(new Error(`physics worker failed: ${err.message}`));
  });

  worker.postMessage({
    t: 'init',
    scenario: stripScenarioForWorker(scenario as Record<string, unknown>),
    options,
    sab,
    chunkCapacity: chunkCount,
    projectileCapacity,
  });
  await ready;

  const nowSeconds = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()) / 1000;

  const view = {
    /** True marker for demos that need to branch on the mode. */
    isWorkerCore: true as const,
    chunks,
    projectiles,
    // The adapter's color/cull paths only ask bodies for their kind; serve pooled fakes
    // keyed by mirrored handle. Projectile mesh sync also reads translation/rotation —
    // those entries carry their own pose, served by the projectile fake below.
    world: {
      getRigidBody(handle: number) {
        if (handle === NO_BODY) return null;
        const proj = projByHandle.get(handle);
        if (proj) {
          return {
            isFixed: () => false,
            isKinematic: () => false,
            isDynamic: () => true,
            translation: () => ({ x: proj.px, y: proj.py, z: proj.pz }),
            rotation: () => ({ x: proj.qx, y: proj.qy, z: proj.qz, w: proj.qw }),
          };
        }
        const kind = bodyKindByHandle.get(handle);
        if (kind === undefined) return null;
        let body = bodyPool.get(handle);
        if (!body) { body = new KindBody(); bodyPool.set(handle, body); }
        body.kind = kind;
        return body;
      },
    },

    /** Pump wall-clock time to the worker and mirror the freshest published frame. */
    step(dt: number) {
      if (disposed) return;
      worker.postMessage({ t: 'pump', dt });
      const consumed = mirrorPoseFrame(views, lastFrame, chunks, projectiles, bodyKindByHandle, nowSeconds());
      if (consumed >= 0) {
        lastFrame = consumed;
        projByHandle.clear();
        for (const p of projectiles) projByHandle.set(p.bodyHandle, p);
      }
    },

    enqueueProjectile(spawn: unknown) { worker.postMessage({ t: 'projectile', spawn }); },
    applyExternalForce(nodeIndex: number, point: Vec3, force: Vec3) {
      worker.postMessage({ t: 'externalForce', nodeIndex, point, force });
    },
    setIslandSolver(opts: unknown) { worker.postMessage({ t: 'setIslandSolver', opts }); },
    setLazyIntactColliders(enabled: boolean) { worker.postMessage({ t: 'setLazyIntactColliders', enabled }); },
    setGravity(gravity: number) { worker.postMessage({ t: 'setGravity', gravity }); },

    getRigidBodyCount: () => stats.bodies,
    getActiveBondsCount: () => stats.bonds,
    getIslandSolverStats: () => stats.islands ?? { enabled: false, skipSettled: false, islandCount: 0, islandsSkipped: 0 },
    getLazyColliderStats: () => stats.lazy ?? { enabled: false, buildingCount: 0, dormantCount: 0, explodedCount: 0, activeLeafFragments: 0 },
    /** Static fragments/AABBs from the worker; `intact` recomputed from mirrored chunks. */
    getBuildingRenderStates() {
      for (const s of buildingStates) s.intact = isBuildingRenderIntact(s.fragments, chunks as never);
      return buildingStates;
    },
    getSolverDebugLines: () => [] as never[],

    dispose() {
      if (disposed) return;
      disposed = true;
      try { worker.postMessage({ t: 'dispose' }); } catch { /* ignore */ }
      try { worker.terminate(); } catch { /* ignore */ }
    },
  };

  return view;
}
