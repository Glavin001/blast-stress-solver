/**
 * Why a fracturing tower/high-rise "falls slow / floats" — the small-body damping mode.
 *
 * `smallBodyDamping` adds linear+angular damping to small debris bodies (≤ colliderCountThreshold
 * colliders) to settle them. The demos used `mode: 'always'`, which damps debris *the instant it
 * is created — while it is still falling*. With `minLinearDamping: 2` that caps a falling fragment
 * at terminal velocity `g / 2 ≈ 4.9 m/s` instead of accelerating, so a collapse looks like it is
 * floating in slow motion (the reported symptom). The `'afterGroundCollision'` mode exists for
 * exactly this: it leaves airborne debris alone (full-speed fall) and only damps a body once it
 * has touched the ground (settle on landing).
 *
 * These tests pin the behavior down so the demo configs can't silently regress to the floaty mode.
 * (Requires the WASM stress-solver build.)
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import RAPIER from '@dimforge/rapier3d-compat';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

let buildDestructibleCore: (opts: any) => Promise<any>;
beforeAll(async () => {
  if (!runtimeAvailable) return;
  await RAPIER.init();
  buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
});

// support(0) up high + one dynamic node joined by a weak bond: the bond breaks under gravity so
// the dynamic node detaches as a single-collider (= "small") body and free-falls to the ground.
function fallingDebrisScenario(startY: number) {
  return {
    nodes: [
      { centroid: { x: 0, y: startY + 1, z: 0 }, mass: 0, volume: 1 },
      { centroid: { x: 0, y: startY, z: 0 }, mass: 5, volume: 1 },
    ],
    bonds: [
      { node0: 0, node1: 1, centroid: { x: 0, y: startY + 0.5, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 0.02 },
    ],
    spacing: { x: 1, y: 1, z: 1 },
  } as any;
}

async function makeCore(mode: 'always' | 'off' | 'afterGroundCollision', startY: number) {
  const core = await buildDestructibleCore({
    scenario: fallingDebrisScenario(startY), gravity: -9.81, materialScale: 1,
    resimulateOnFracture: true, snapshotMode: 'perBody', debrisCleanup: { mode: 'off' },
    smallBodyDamping: { mode, colliderCountThreshold: 3, minLinearDamping: 2, minAngularDamping: 2 },
  });
  core.setFracturePolicy?.({ idleSkip: false });
  return core;
}
const dampingOf = (core: any) => {
  const b = core.world.getRigidBody(core.chunks[1].bodyHandle);
  return b ? b.linearDamping() : NaN;
};
const speedOf = (core: any) => {
  const b = core.world.getRigidBody(core.chunks[1].bodyHandle);
  return b ? b.linvel().y : NaN;
};

describe.skipIf(!runtimeAvailable)('small-body damping vs falling debris', () => {
  it("'always' damps airborne debris — it floats down at ~g/damping instead of accelerating", async () => {
    const core = await makeCore('always', 20);
    for (let i = 0; i < 90; i++) core.step(1 / 60); // ~1.5 s of fall, still well above ground
    // The body is airborne (y ~ 20 - small drop) yet already damped → capped near terminal speed.
    expect(dampingOf(core)).toBeCloseTo(2, 5);
    expect(Math.abs(speedOf(core))).toBeLessThan(6); // would be ~14.7 m/s in true free-fall
    core.dispose();
  });

  it("'afterGroundCollision' leaves airborne debris UNdamped — it free-falls at full speed", async () => {
    const core = await makeCore('afterGroundCollision', 20);
    for (let i = 0; i < 90; i++) core.step(1 / 60);
    expect(dampingOf(core)).toBe(0); // not yet touched the ground
    expect(Math.abs(speedOf(core))).toBeGreaterThan(12); // true free-fall, not floaty
    core.dispose();
  });

  it("'afterGroundCollision' DOES damp the body once it lands (settles on the ground)", async () => {
    const core = await makeCore('afterGroundCollision', 3); // low start so it reaches the ground
    let landedDamping = 0;
    for (let i = 0; i < 240; i++) {
      core.step(1 / 60);
      const h = core.chunks[1].bodyHandle;
      if (h != null && core.hasBodyCollidedWithGround?.(h)) { landedDamping = dampingOf(core); break; }
    }
    expect(landedDamping).toBeCloseTo(2, 5); // damped after the ground contact, to settle
    core.dispose();
  });
});
