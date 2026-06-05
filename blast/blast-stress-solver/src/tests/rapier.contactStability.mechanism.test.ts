/**
 * Why a big *fracturing* chunk can look choppier than a small piece *flying* through the air —
 * and which parts of the destruction pipeline are (and are NOT) to blame. Pure Rapier, no WASM.
 *
 * A small piece that has finished fracturing is airborne: zero contacts → perfectly smooth
 * ballistic motion. A big chunk in a collapse is resting on / grinding against the debris pile,
 * so its smoothness is at the mercy of contact resolution. These tests isolate what perturbs a
 * resting contact:
 *
 *   • CONTROL  — a settled body just integrating: rock-stable (the baseline).
 *   • RESIM    — re-applying a body's pose+velocity every frame (exactly what
 *                `restoreWorldSnapshot` does to every dynamic body on a fracture frame):
 *                still rock-stable, so resimulation is NOT the source of the choppiness.
 *   • CHURN    — destroying and re-creating the body's collider every frame loses Rapier's
 *                contact warm-starting and the body sinks/jitters violently. Fracture
 *                *collider migration* recreates a fragment's collider once (when it moves to its
 *                new body), so a dense, still-fracturing pile keeps spawning fresh, warm-start-less
 *                contacts → the pile (and the big chunk resting on it) jitters, while airborne
 *                debris is unaffected. This is the real, per-body reason — contact churn in the
 *                fracture zone, not the resim/velocity machinery.
 *
 * The CONTROL and RESIM assertions are regression guards: if the resim path ever starts
 * destabilizing resting contacts (e.g. by recreating colliders on restore), RESIM will fail.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';

beforeAll(async () => {
  await RAPIER.init();
});

function settledBox(): { world: RAPIER.World; body: RAPIER.RigidBody } {
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(50, 0.5, 50).setTranslation(0, -0.5, 0),
    world.createRigidBody(RAPIER.RigidBodyDesc.fixed()),
  );
  const body = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(0, 0.5, 0));
  world.createCollider(RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5).setDensity(1), body);
  for (let i = 0; i < 120; i++) world.step(); // settle on the ground
  return { world, body };
}

/** RMS deviation of y[] from its mean (how much the supposedly-resting body wobbles), in metres. */
function rms(ys: number[]): number {
  const m = ys.reduce((a, b) => a + b, 0) / ys.length;
  return Math.sqrt(ys.reduce((a, b) => a + (b - m) ** 2, 0) / ys.length);
}

describe('contact stability — what perturbs a resting body', () => {
  it('CONTROL: a settled body is rock-stable', () => {
    const { world, body } = settledBox();
    const ys: number[] = [];
    for (let i = 0; i < 120; i++) { world.step(); ys.push(body.translation().y); }
    expect(rms(ys)).toBeLessThan(1e-3);
    world.free();
  });

  it("RESIM: re-applying pose+velocity every frame (the resim restore pattern) does NOT disturb the contact", () => {
    const { world, body } = settledBox();
    const ys: number[] = [];
    for (let i = 0; i < 120; i++) {
      // What restoreWorldSnapshot does to every dynamic body on a fracture frame.
      body.setTranslation(body.translation(), true);
      body.setRotation(body.rotation(), true);
      body.setLinvel(body.linvel(), true);
      body.setAngvel(body.angvel(), true);
      world.step();
      ys.push(body.translation().y);
    }
    expect(rms(ys)).toBeLessThan(1e-3); // resimulation is contact-safe — not the choppiness source
    world.free();
  });

  it('CHURN: re-creating the collider every frame destroys contact warm-starting (documents the real hazard)', () => {
    const { world, body } = settledBox();
    const ys: number[] = [];
    for (let i = 0; i < 120; i++) {
      world.removeCollider(world.getCollider(body.collider(0))!, false);
      world.createCollider(RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5).setDensity(1), body);
      world.step();
      ys.push(body.translation().y);
    }
    // Far from its resting height — collider churn breaks the contact. This is why the pipeline
    // must recreate a fragment's collider at most once (on migration), never per frame, and why a
    // dense, actively-fracturing pile (lots of fresh colliders) is the jittery part of a collapse.
    expect(Math.abs(body.translation().y - 0.5)).toBeGreaterThan(0.5);
    world.free();
  });
});
