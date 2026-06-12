/**
 * Physics worker for the mini-city demo (roadmap Tier 3.1).
 *
 * Owns the entire physics stack — Rapier WASM + Blast stress WASM inside a
 * `DestructibleCore` — off the main thread. The main thread pumps wall-clock time over
 * postMessage; this worker advances the simulation on a FIXED 60 Hz accumulator and
 * publishes the render-facing state (chunk poses/flags, projectiles) into the
 * SharedArrayBuffer pose channel after every step batch. Commands (shoot, radial force,
 * optimization toggles) arrive as messages and apply before the next step, and a small
 * stats snapshot is posted back at a low rate for the HUD.
 *
 * Bundled self-contained by `build:demo:mini-city-worker` (import maps don't apply to
 * workers): blast-stress-solver and Rapier are inlined; the stress WASM is fetched
 * relative to this script's URL, so the bundle must be served from `dist/` next to
 * `stress_solver.wasm`.
 */
import {
  buildDestructibleCore,
  viewPoseChannel,
  publishPoseFrame,
  type PoseChannelViews,
} from 'blast-stress-solver/rapier';

type CoreInstance = Awaited<ReturnType<typeof buildDestructibleCore>>;

let core: CoreInstance | null = null;
let views: PoseChannelViews | null = null;

// Fixed-step accumulator (kept inline so this worker has no dependency on the
// fixed-step-loop PR; unify on createFixedStepLoop once both land).
const FIXED_DT = 1 / 60;
const MAX_STEPS_PER_PUMP = 3;
let accumulator = 0;

let statsCounter = 0;

function publish() {
  if (core && views) publishPoseFrame(views, core);
}

function postStats() {
  if (!core) return;
  let detached = 0;
  for (const c of core.chunks) if ((c as { detached?: boolean }).detached) detached++;
  (self as unknown as Worker).postMessage({
    t: 'stats',
    bodies: core.getRigidBodyCount(),
    bonds: core.getActiveBondsCount(),
    projectiles: core.projectiles.length,
    detached,
    islands: core.getIslandSolverStats?.() ?? null,
    lazy: core.getLazyColliderStats?.() ?? null,
  });
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data as Record<string, unknown> & { t: string };
  switch (m.t) {
    case 'init': {
      try {
        core = await buildDestructibleCore({
          scenario: m.scenario as never,
          ...(m.options as Record<string, unknown>),
        } as never);
        views = viewPoseChannel(
          m.sab as SharedArrayBuffer,
          m.chunkCapacity as number,
          m.projectileCapacity as number,
        );
        // One settle step so every chunk has a world pose before the first render.
        core.step(FIXED_DT);
        publish();
        // Static building data for the main thread's render LOD (fragments + AABBs are
        // constant; the live `intact` flag is recomputed main-side from mirrored chunks).
        const buildings = (core.getBuildingRenderStates?.() ?? []).map((s) => ({
          buildingId: s.buildingId,
          aabbMin: { ...s.aabbMin },
          aabbMax: { ...s.aabbMax },
          fragments: [...s.fragments],
        }));
        postStats();
        (self as unknown as Worker).postMessage({ t: 'ready', buildings });
      } catch (err) {
        (self as unknown as Worker).postMessage({ t: 'error', message: String(err) });
      }
      break;
    }
    case 'pump': {
      if (!core) break;
      const dt = typeof m.dt === 'number' && Number.isFinite(m.dt) ? Math.max(0, m.dt) : 0;
      accumulator = Math.min(accumulator + dt, (MAX_STEPS_PER_PUMP + 1) * FIXED_DT);
      let steps = 0;
      while (accumulator >= FIXED_DT && steps < MAX_STEPS_PER_PUMP) {
        core.step(FIXED_DT);
        accumulator -= FIXED_DT;
        steps++;
      }
      if (steps > 0) {
        publish();
        if (++statsCounter % 10 === 0) postStats();
      }
      break;
    }
    case 'projectile':
      core?.enqueueProjectile(m.spawn as never);
      break;
    case 'externalForce': {
      const f = m as unknown as { nodeIndex: number; point: never; force: never };
      (core as unknown as { applyExternalForce?: (n: number, p: never, fo: never) => void })
        ?.applyExternalForce?.(f.nodeIndex, f.point, f.force);
      break;
    }
    case 'setIslandSolver':
      core?.setIslandSolver?.(m.opts as never);
      break;
    case 'setLazyIntactColliders':
      core?.setLazyIntactColliders?.(m.enabled as boolean);
      break;
    case 'setGravity':
      (core as unknown as { setGravity?: (g: number) => void })?.setGravity?.(m.gravity as number);
      break;
    case 'dispose':
      try { core?.dispose?.(); } catch { /* ignore */ }
      core = null;
      self.close();
      break;
  }
};
