/**
 * Fixed-timestep driver + pose interpolator (pure logic — no WASM, no Rapier).
 *
 * The driver must (a) convert arbitrary display-frame deltas into an exact fixed-rate
 * step cadence, (b) bound catch-up work under overload by DROPPING excess time, and
 * (c) report an interpolation alpha ∈ [0,1). The interpolator must maintain prev/curr
 * pose buffers that lerp/nlerp faithfully and leave settled chunks bit-stable (so the
 * renderer's unchanged-pose skip still elides their writes).
 */
import { describe, it, expect } from 'vitest';
import { createFixedStepLoop, createPoseInterpolator } from '../rapier/fixedStepLoop';

describe('createFixedStepLoop', () => {
  function makeLoop(hz = 60, maxStepsPerTick = 3) {
    const stepped: number[] = [];
    const loop = createFixedStepLoop({ hz, maxStepsPerTick, step: (dt) => stepped.push(dt) });
    return { loop, stepped };
  }

  it('steps exactly once per frame at the native rate', () => {
    const { loop, stepped } = makeLoop(60);
    for (let i = 0; i < 10; i++) {
      const r = loop.tick(1 / 60);
      expect(r.steps).toBe(1);
      expect(r.droppedTime).toBe(0);
    }
    expect(stepped).toHaveLength(10);
    expect(stepped.every((dt) => dt === 1 / 60)).toBe(true);
  });

  it('steps every other frame on a 120 Hz display (half the physics work)', () => {
    const { loop, stepped } = makeLoop(60);
    let totalSteps = 0;
    for (let i = 0; i < 120; i++) totalSteps += loop.tick(1 / 120).steps;
    expect(totalSteps).toBe(60); // 1 second of 120 Hz frames -> exactly 60 physics steps
    expect(stepped).toHaveLength(60);
  });

  it('catches up with two steps per frame at 30 fps', () => {
    const { loop } = makeLoop(60);
    loop.tick(1 / 30);
    const r = loop.tick(1 / 30);
    expect(r.steps).toBe(2);
  });

  it('alpha is the fractional unstepped remainder in [0,1)', () => {
    const { loop } = makeLoop(60);
    const r1 = loop.tick(0.5 / 60); // half a step accumulated -> no step, alpha 0.5
    expect(r1.steps).toBe(0);
    expect(r1.alpha).toBeCloseTo(0.5, 9);
    const r2 = loop.tick(0.75 / 60); // 1.25 steps total -> one step, alpha 0.25
    expect(r2.steps).toBe(1);
    expect(r2.alpha).toBeCloseTo(0.25, 9);
    expect(r2.alpha).toBeGreaterThanOrEqual(0);
    expect(r2.alpha).toBeLessThan(1);
  });

  it('drops excess time under overload instead of death-spiralling', () => {
    const { loop, stepped } = makeLoop(60, 3);
    const r = loop.tick(1); // owes 60 steps; cap is 3
    expect(r.steps).toBe(3);
    expect(stepped).toHaveLength(3);
    expect(r.droppedTime).toBeGreaterThan(0.9); // ~57 steps' worth dropped
    expect(r.alpha).toBeGreaterThanOrEqual(0);
    expect(r.alpha).toBeLessThan(1);
    // Recovery: the next normal frame owes at most cap+1 steps' worth, not 60.
    const r2 = loop.tick(1 / 60);
    expect(r2.steps).toBeLessThanOrEqual(2);
  });

  it('ignores negative and NaN frame deltas (tab restore, clock jumps)', () => {
    const { loop, stepped } = makeLoop(60);
    expect(loop.tick(-5).steps).toBe(0);
    expect(loop.tick(Number.NaN).steps).toBe(0);
    expect(stepped).toHaveLength(0);
  });

  it('reset clears pending time; setHz changes the cadence (accumulator preserved)', () => {
    const { loop } = makeLoop(60);
    loop.tick(0.9 / 60);
    loop.reset();
    expect(loop.tick(0.5 / 60).steps).toBe(0); // 0.5 steps owed, not 1.4 — reset really cleared
    loop.setHz(120);
    expect(loop.fixedDt).toBeCloseTo(1 / 120, 12);
    // Carried 0.5/60 (= 1/120) + new 1/60 (= 2/120) → exactly 3 steps at the new rate.
    expect(loop.tick(1 / 60).steps).toBe(3);
    loop.reset();
    expect(loop.tick(1 / 60).steps).toBe(2); // clean accumulator: 1/60 = exactly 2 × 1/120
  });

  it('hooks fire around every step in order', () => {
    const order: string[] = [];
    const loop = createFixedStepLoop({
      hz: 60,
      step: () => order.push('step'),
      onBeforeStep: () => order.push('before'),
      onAfterStep: () => order.push('after'),
    });
    loop.tick(2 / 60);
    expect(order).toEqual(['before', 'step', 'after', 'before', 'step', 'after']);
  });
});

describe('createPoseInterpolator', () => {
  type FakeChunk = {
    worldPosition?: { x: number; y: number; z: number };
    worldQuaternion?: { x: number; y: number; z: number; w: number };
  };
  function fakeCore(chunks: FakeChunk[], projectiles: Array<{ bodyHandle: number }> = [], bodies = new Map<number, { t: { x: number; y: number; z: number }; r: { x: number; y: number; z: number; w: number } }>()) {
    return {
      chunks: chunks as any,
      projectiles,
      world: {
        getRigidBody(handle: number) {
          const b = bodies.get(handle);
          if (!b) return null;
          return { translation: () => b.t, rotation: () => b.r };
        },
      },
      bodies,
    };
  }

  it('lerps positions and nlerps rotations between the captured states', () => {
    const chunk: FakeChunk = {
      worldPosition: { x: 0, y: 0, z: 0 },
      worldQuaternion: { x: 0, y: 0, z: 0, w: 1 },
    };
    const core = fakeCore([chunk]);
    const interp = createPoseInterpolator(core);

    interp.beforeStep(); // prev := (0,0,0)
    chunk.worldPosition = { x: 2, y: 4, z: -6 };
    // 90° about Y: q = (0, sin45, 0, cos45)
    chunk.worldQuaternion = { x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 };
    interp.afterStep(); // curr := stepped pose

    const v = interp.view(0.5);
    expect(v.alpha).toBe(0.5);
    expect(v.prev[0]).toBe(0);
    expect(v.curr[0]).toBe(2);
    // Midpoint position via the same formula the adapter applies:
    const px = v.prev[0] + (v.curr[0] - v.prev[0]) * v.alpha;
    const py = v.prev[1] + (v.curr[1] - v.prev[1]) * v.alpha;
    const pz = v.prev[2] + (v.curr[2] - v.prev[2]) * v.alpha;
    expect([px, py, pz]).toEqual([1, 2, -3]);
    // Normalized-lerp midpoint of identity→90°Y is 45°Y.
    let qy = v.prev[4] + (v.curr[4] - v.prev[4]) * v.alpha;
    let qw = v.prev[6] + (v.curr[6] - v.prev[6]) * v.alpha;
    const len = Math.hypot(qy, qw);
    qy /= len; qw /= len;
    expect(qy).toBeCloseTo(Math.sin(Math.PI / 8), 6);
    expect(qw).toBeCloseTo(Math.cos(Math.PI / 8), 6);
  });

  it('a settled chunk has bit-identical prev/curr (renderer skip stays effective)', () => {
    const chunk: FakeChunk = {
      worldPosition: { x: 1.25, y: 2.5, z: -0.75 },
      worldQuaternion: { x: 0.1, y: 0.2, z: 0.3, w: 0.9273 },
    };
    const interp = createPoseInterpolator(fakeCore([chunk]));
    interp.beforeStep();
    interp.afterStep(); // pose unchanged across the step
    const v = interp.view(0.37);
    for (let k = 0; k < 7; k++) expect(v.prev[k]).toBe(v.curr[k]);
  });

  it('marks chunks without a pose via the NaN sentinel', () => {
    const interp = createPoseInterpolator(fakeCore([{}]));
    interp.beforeStep();
    interp.afterStep();
    const v = interp.view(0.5);
    expect(Number.isNaN(v.prev[6])).toBe(true);
    expect(Number.isNaN(v.curr[6])).toBe(true);
  });

  it('interpolates projectile bodies and prunes despawned handles', () => {
    const bodies = new Map([
      [7, { t: { x: 0, y: 10, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 } }],
    ]);
    const projectiles = [{ bodyHandle: 7 }];
    const core = fakeCore([], projectiles, bodies);
    const interp = createPoseInterpolator(core);

    interp.beforeStep();
    bodies.get(7)!.t = { x: 0, y: 8, z: 0 }; // fell 2m during the step
    interp.afterStep();

    const out = { px: 0, py: 0, pz: 0, qx: 0, qy: 0, qz: 0, qw: 1 };
    expect(interp.view(0.5).getBodyPose(7, out)).toBe(true);
    expect(out.py).toBeCloseTo(9, 9);

    // Despawn: handle pruned on the next capture; lookups report "no pose".
    projectiles.length = 0;
    interp.beforeStep();
    interp.afterStep();
    expect(interp.view(0).getBodyPose(7, out)).toBe(false);
  });

  it('takes the short way around for opposite-sign quaternions (hemisphere correction)', () => {
    const bodies = new Map([
      [3, { t: { x: 0, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 } }],
    ]);
    const core = fakeCore([], [{ bodyHandle: 3 }], bodies);
    const interp = createPoseInterpolator(core);
    interp.beforeStep();
    // Same rotation, negated representation: must interpolate to ±identity, not to zero.
    bodies.get(3)!.r = { x: 0, y: 0, z: 0, w: -1 };
    interp.afterStep();
    const out = { px: 0, py: 0, pz: 0, qx: 0, qy: 0, qz: 0, qw: 0 };
    interp.view(0.5).getBodyPose(3, out);
    expect(Math.abs(out.qw)).toBeCloseTo(1, 9);
  });
});
