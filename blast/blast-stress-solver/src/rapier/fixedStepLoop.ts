/**
 * Fixed-timestep physics driver + pose interpolation (Tier-1 roadmap item:
 * docs/perf-city-scale-roadmap.md §3 Tier 1.3).
 *
 * Every demo used to call `core.step(rafDt)` once per requestAnimationFrame with the
 * *display's* variable delta. That couples simulation cost and behaviour to the monitor:
 * a 120 Hz display pays 120 physics steps/s (double the work for zero visual benefit) and
 * the simulation itself becomes frame-rate dependent (fracture cadence, contact impulses
 * and resim frequency all shift with dt).
 *
 * `createFixedStepLoop` decouples them with the classic accumulator pattern: physics
 * always steps at a fixed `hz` (default 60), regardless of display rate. The remainder
 * (`alpha` ∈ [0,1)) lets the renderer interpolate between the last two physics states so
 * 120 Hz displays still get 120 unique smooth frames — `createPoseInterpolator` maintains
 * those two states for every chunk (and projectile) as flat typed arrays the Three.js
 * adapter can lerp from without allocations.
 *
 * Quality: this *improves* physics fidelity — identical simulation on every machine —
 * and the integration test pins that N driver-ticks produce bit-identical state to N
 * direct `core.step(1/hz)` calls (the driver adds no math of its own).
 *
 * Overload policy: when a frame is so slow that more than `maxStepsPerTick` steps are
 * owed, the excess accumulator time is DROPPED (sim runs slow-motion under overload
 * rather than death-spiralling). `1/30`-style dt clamping lives in `core.step` already;
 * the cap here bounds the *number* of catch-up steps per render frame.
 */
import type { ChunkData } from './types';

export type FixedStepLoopOptions = {
  /** Physics rate in steps per second. Default 60. */
  hz?: number;
  /** Max catch-up steps per tick before excess time is dropped. Default 3. */
  maxStepsPerTick?: number;
  /** Advance the simulation by exactly `fixedDt` seconds. */
  step: (fixedDt: number) => void;
  /** Called immediately BEFORE each `step` (e.g. `interpolator.beforeStep()`). */
  onBeforeStep?: () => void;
  /** Called immediately AFTER each `step` (e.g. `interpolator.afterStep()`). */
  onAfterStep?: () => void;
};

export type FixedStepTickResult = {
  /** Physics steps executed during this tick (0 on fast display frames). */
  steps: number;
  /** Fraction of a physics step left in the accumulator — interpolation factor ∈ [0,1). */
  alpha: number;
  /** The fixed dt used for each step this tick (1/hz). */
  fixedDt: number;
  /** Accumulator time dropped by the overload cap this tick (seconds; normally 0). */
  droppedTime: number;
};

export type FixedStepLoop = {
  /** Feed one render frame's wall-clock delta (seconds); runs 0..maxStepsPerTick steps. */
  tick(frameDtSeconds: number): FixedStepTickResult;
  /** Clear pending accumulator time (use after pauses/rebuilds to avoid a catch-up burst). */
  reset(): void;
  /** Change the physics rate; pending accumulator time is preserved. */
  setHz(hz: number): void;
  readonly fixedDt: number;
  readonly hz: number;
};

export function createFixedStepLoop(options: FixedStepLoopOptions): FixedStepLoop {
  let hz = Math.max(1, options.hz ?? 60);
  let fixedDt = 1 / hz;
  const maxSteps = Math.max(1, options.maxStepsPerTick ?? 3);
  const { step, onBeforeStep, onAfterStep } = options;
  let accumulator = 0;

  return {
    get fixedDt() {
      return fixedDt;
    },
    get hz() {
      return hz;
    },
    tick(frameDtSeconds: number): FixedStepTickResult {
      // Negative / NaN guards (tab restores, clock jumps).
      const frameDt = Number.isFinite(frameDtSeconds) ? Math.max(0, frameDtSeconds) : 0;
      accumulator += frameDt;

      let steps = 0;
      while (accumulator >= fixedDt && steps < maxSteps) {
        onBeforeStep?.();
        step(fixedDt);
        onAfterStep?.();
        accumulator -= fixedDt;
        steps += 1;
      }

      // Overload: drop time we will never catch up on, keeping alpha meaningful.
      let droppedTime = 0;
      if (accumulator >= fixedDt) {
        droppedTime = accumulator - (fixedDt - 1e-9);
        accumulator = fixedDt - 1e-9;
      }

      return { steps, alpha: accumulator / fixedDt, fixedDt, droppedTime };
    },
    reset() {
      accumulator = 0;
    },
    setHz(nextHz: number) {
      if (!Number.isFinite(nextHz) || nextHz <= 0) return;
      hz = nextHz;
      fixedDt = 1 / hz;
    },
  };
}

// ── Pose interpolation ────────────────────────────────────────────────────────────────

/** Flat prev/curr pose buffers (7 floats per chunk: px,py,pz,qx,qy,qz,qw) + alpha, in the
 *  exact layout `updateBatchedChunkMesh` consumes. `qw === NaN` marks "no pose yet" —
 *  consumers fall back to the chunk's live pose. */
export type InterpolatedPoseView = {
  prev: Float32Array;
  curr: Float32Array;
  alpha: number;
  /** Interpolated world pose for an external (non-chunk) body, e.g. a projectile.
   *  Returns false when the body was never captured (caller uses the live pose). */
  getBodyPose(
    bodyHandle: number,
    out: { px: number; py: number; pz: number; qx: number; qy: number; qz: number; qw: number },
  ): boolean;
};

type PoseSourceCore = {
  chunks: ChunkData[];
  projectiles?: Array<{ bodyHandle: number }>;
  world: {
    getRigidBody(handle: number):
      | {
          translation(): { x: number; y: number; z: number };
          rotation(): { x: number; y: number; z: number; w: number };
        }
      | null
      | undefined;
  };
};

export type PoseInterpolator = {
  /** Snapshot the CURRENT state as the interpolation start. Call before every physics step. */
  beforeStep(): void;
  /** Capture the post-step state as the interpolation end. Call after every physics step. */
  afterStep(): void;
  /** Re-prime both buffers from the current state (after rebuilds/teleports — kills lerp ghosts). */
  reset(): void;
  /** A view for the renderer at a given alpha. The buffers are live (no copies). */
  view(alpha: number): InterpolatedPoseView;
};

const STRIDE = 7;

/**
 * Maintains previous/current world poses for every chunk (flat Float32Arrays, no per-frame
 * allocations) and for the few projectile bodies (small reused map). Wire its hooks into
 * `createFixedStepLoop({ onBeforeStep, onAfterStep })` and hand `view(alpha)` to the
 * Three.js adapter each render frame.
 */
export function createPoseInterpolator(core: PoseSourceCore): PoseInterpolator {
  const count = core.chunks.length;
  const prev = new Float32Array(count * STRIDE).fill(Number.NaN);
  const curr = new Float32Array(count * STRIDE).fill(Number.NaN);

  // Projectiles: tiny population, keyed by body handle. Entries are reused; stale handles
  // are pruned on capture so handle reuse can't serve a ghost pose.
  type BodyPose = { ppx: number; ppy: number; ppz: number; pqx: number; pqy: number; pqz: number; pqw: number;
                    cpx: number; cpy: number; cpz: number; cqx: number; cqy: number; cqz: number; cqw: number };
  const bodyPoses = new Map<number, BodyPose>();
  const seenHandles = new Set<number>();

  function captureChunksInto(target: Float32Array) {
    const chunks = core.chunks;
    for (let i = 0; i < chunks.length; i += 1) {
      const base = i * STRIDE;
      const wp = chunks[i].worldPosition;
      const wq = chunks[i].worldQuaternion;
      if (!wp || !wq) {
        target[base + 6] = Number.NaN; // sentinel: no pose yet
        continue;
      }
      target[base] = wp.x;
      target[base + 1] = wp.y;
      target[base + 2] = wp.z;
      target[base + 3] = wq.x;
      target[base + 4] = wq.y;
      target[base + 5] = wq.z;
      target[base + 6] = wq.w;
    }
  }

  function captureBodies(asPrev: boolean) {
    const projectiles = core.projectiles ?? [];
    seenHandles.clear();
    for (const p of projectiles) {
      const body = core.world.getRigidBody(p.bodyHandle);
      if (!body) continue;
      seenHandles.add(p.bodyHandle);
      const t = body.translation();
      const r = body.rotation();
      let e = bodyPoses.get(p.bodyHandle);
      if (!e) {
        // First sighting: prime both ends so the first interpolated frame doesn't lerp
        // from the origin.
        e = { ppx: t.x, ppy: t.y, ppz: t.z, pqx: r.x, pqy: r.y, pqz: r.z, pqw: r.w,
              cpx: t.x, cpy: t.y, cpz: t.z, cqx: r.x, cqy: r.y, cqz: r.z, cqw: r.w };
        bodyPoses.set(p.bodyHandle, e);
        continue;
      }
      if (asPrev) {
        e.ppx = e.cpx; e.ppy = e.cpy; e.ppz = e.cpz;
        e.pqx = e.cqx; e.pqy = e.cqy; e.pqz = e.cqz; e.pqw = e.cqw;
      } else {
        e.cpx = t.x; e.cpy = t.y; e.cpz = t.z;
        e.cqx = r.x; e.cqy = r.y; e.cqz = r.z; e.cqw = r.w;
      }
    }
    // Prune bodies that no longer exist (despawned projectiles, reused handles).
    for (const handle of bodyPoses.keys()) {
      if (!seenHandles.has(handle)) bodyPoses.delete(handle);
    }
  }

  let primed = false;
  function primeIfNeeded() {
    if (primed) return;
    captureChunksInto(curr);
    prev.set(curr);
    captureBodies(false);
    primed = true;
  }

  return {
    beforeStep() {
      primeIfNeeded();
      // The state we are about to overwrite becomes the interpolation start.
      prev.set(curr);
      captureBodies(true);
    },
    afterStep() {
      captureChunksInto(curr);
      captureBodies(false);
      primed = true;
    },
    reset() {
      primed = false;
      bodyPoses.clear();
      primeIfNeeded();
    },
    view(alpha: number): InterpolatedPoseView {
      const a = Math.min(1, Math.max(0, alpha));
      primeIfNeeded();
      return {
        prev,
        curr,
        alpha: a,
        getBodyPose(bodyHandle, out) {
          const e = bodyPoses.get(bodyHandle);
          if (!e) return false;
          const t = a;
          out.px = e.ppx + (e.cpx - e.ppx) * t;
          out.py = e.ppy + (e.cpy - e.ppy) * t;
          out.pz = e.ppz + (e.cpz - e.ppz) * t;
          // nlerp with hemisphere correction (adjacent physics states are near-identical
          // rotations, where nlerp ≈ slerp and is monotonic).
          let qx = e.cqx, qy = e.cqy, qz = e.cqz, qw = e.cqw;
          const dot = e.pqx * qx + e.pqy * qy + e.pqz * qz + e.pqw * qw;
          if (dot < 0) { qx = -qx; qy = -qy; qz = -qz; qw = -qw; }
          let ox = e.pqx + (qx - e.pqx) * t;
          let oy = e.pqy + (qy - e.pqy) * t;
          let oz = e.pqz + (qz - e.pqz) * t;
          let ow = e.pqw + (qw - e.pqw) * t;
          const len = Math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow);
          if (len > 1e-12) { ox /= len; oy /= len; oz /= len; ow /= len; }
          else { ox = 0; oy = 0; oz = 0; ow = 1; }
          out.qx = ox; out.qy = oy; out.qz = oz; out.qw = ow;
          return true;
        },
      };
    },
  };
}
