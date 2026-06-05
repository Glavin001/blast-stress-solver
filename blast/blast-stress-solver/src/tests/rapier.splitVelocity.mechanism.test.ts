/**
 * Mechanism-level characterization of the fracture velocity-transfer math the
 * destruction split path depends on — pure Rapier, NO stress-solver/WASM, so it
 * is fully deterministic and runs everywhere.
 *
 * Why this exists
 * ---------------
 * Users report that when a large body keeps fracturing as it hits the ground, the
 * big remaining piece appears to "hover" while small already-broken pieces fall
 * normally. The root cause is a Rapier mass-property subtlety the split pipeline
 * has to work around:
 *
 *   • A rigid body stores its linear velocity as the velocity *of its centre of
 *     mass*. The velocity at any other point P is `linvel + ω × (P − COM)`.
 *   • When colliders move between bodies during a split, each body's COM shifts.
 *     Rapier keeps the stored COM-velocity, so the velocity *field* at every other
 *     point jumps by `ω × ΔCOM` unless the velocity is re-derived for the new COM.
 *   • `RigidBody.removeCollider` does NOT recompute mass properties immediately —
 *     the COM is stale until the next `world.step()` (or an explicit
 *     `recomputeMassPropertiesFromColliders()`).
 *
 * `destructible-core.ts` compensates for *created* child bodies
 * (`syncBodyVelocityFromSource`: `v_child = v_parent + ω × (comChild − comParent)`)
 * but the *reused* body (the largest fragment, which keeps the parent's handle and
 * just loses some colliders) gets no such correction — so a spinning reused body's
 * velocity field is corrupted by `ω × ΔCOM` on every fracture. These tests pin the
 * mechanism down with exact magnitudes so the regression can't hide.
 *
 * See blast/TESTING.md (gap #1 "split COM/velocity bug", gaps #6/#8 "JS
 * mass/COM + fragment momentum untested").
 */
import { describe, it, expect, beforeAll } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';

type V = { x: number; y: number; z: number };
const sub = (a: V, b: V): V => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const mag = (a: V): number => Math.hypot(a.x, a.y, a.z);
const cross = (a: V, b: V): V => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});

beforeAll(async () => {
  await RAPIER.init();
});

/** A dynamic body spinning about +z with `colliders` unit cubes at the given
 *  x offsets (equal density => COM is the mean of the offsets). Mass properties
 *  are made fresh without integrating any motion. */
function spinningRow(xs: number[], omega = 5): { world: RAPIER.World; body: RAPIER.RigidBody; handles: number[] } {
  const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic().setTranslation(0, 0, 0).setAngvel({ x: 0, y: 0, z: omega }),
  );
  const handles: number[] = [];
  for (const x of xs) {
    const c = world.createCollider(
      RAPIER.ColliderDesc.cuboid(0.25, 0.25, 0.25).setTranslation(x, 0, 0).setDensity(1),
      body,
    );
    handles.push(c.handle);
  }
  body.recomputeMassPropertiesFromColliders();
  return { world, body, handles };
}

describe('split velocity-transfer mechanism (pure Rapier)', () => {
  it('removeCollider defers the COM/mass recompute until step or explicit recompute', () => {
    const { world, body, handles } = spinningRow([-1, 0, 1]);
    expect(body.worldCom().x).toBeCloseTo(0, 6); // COM of {-1,0,1}

    world.removeCollider(world.getCollider(handles[2])!, false); // drop the +1 cube
    // Stale: Rapier has not yet recomputed, so COM still reads as if +1 were present.
    expect(body.worldCom().x).toBeCloseTo(0, 6);

    body.recomputeMassPropertiesFromColliders();
    // Now it reflects {-1,0} => COM at -0.5.
    expect(body.worldCom().x).toBeCloseTo(-0.5, 5);
    world.free();
  });

  it('a reused body that loses a collider while spinning gains ω×ΔCOM of spurious point velocity', () => {
    const omega = 5;
    const { world, body, handles } = spinningRow([-1, 0, 1], omega);

    // True velocity of the retained chunk at x=-1 while it is part of the parent.
    const pA: V = { x: -1, y: 0, z: 0 };
    const vTrue = body.velocityAtPoint(pA);
    const comBefore = body.worldCom();

    // "Reuse" path: keep this body, just remove the +1 collider (it migrated to a
    // child). The pipeline does NOT re-derive the reused body's velocity.
    world.removeCollider(world.getCollider(handles[2])!, false);
    body.recomputeMassPropertiesFromColliders();
    const comAfter = body.worldCom();

    const vAfter = body.velocityAtPoint(pA);
    const drift = mag(sub(vAfter, vTrue));

    // The drift is exactly ω × ΔCOM — a hard, uncheatable prediction.
    const predicted = mag(cross({ x: 0, y: 0, z: omega }, sub(comAfter, comBefore)));
    expect(predicted).toBeCloseTo(omega * 0.5, 5); // 5 rad/s × 0.5 m COM shift = 2.5 m/s
    expect(drift).toBeCloseTo(predicted, 4);
    expect(drift).toBeGreaterThan(1); // unmistakable "lurch", not numerical noise
    world.free();
  });

  it('symmetric collider loss (no COM shift) preserves the velocity field — the positive control', () => {
    const omega = 5;
    const { world, body, handles } = spinningRow([-1, 0, 1], omega);
    const probe: V = { x: 0.5, y: 0, z: 0 };
    const vTrue = body.velocityAtPoint(probe);

    // Remove BOTH ends symmetrically: COM stays at 0, so the field must not move.
    world.removeCollider(world.getCollider(handles[0])!, false);
    world.removeCollider(world.getCollider(handles[2])!, false);
    body.recomputeMassPropertiesFromColliders();

    expect(body.worldCom().x).toBeCloseTo(0, 5);
    expect(mag(sub(body.velocityAtPoint(probe), vTrue))).toBeLessThan(1e-6);
    world.free();
  });

  it('re-deriving COM velocity (linvel += ω×ΔCOM) restores field continuity — the fix spec', () => {
    const omega = 5;
    const { world, body, handles } = spinningRow([-1, 0, 1], omega);
    const pA: V = { x: -1, y: 0, z: 0 };
    const vTrue = body.velocityAtPoint(pA);
    const comBefore = body.worldCom();

    world.removeCollider(world.getCollider(handles[2])!, false);
    body.recomputeMassPropertiesFromColliders();
    const comAfter = body.worldCom();

    // The correction the reused body is missing: shift COM velocity so the rigid
    // field is preserved at the points that stayed on the body.
    const correction = cross({ x: 0, y: 0, z: omega }, sub(comAfter, comBefore));
    const lv = body.linvel();
    body.setLinvel({ x: lv.x + correction.x, y: lv.y + correction.y, z: lv.z + correction.z }, true);

    expect(mag(sub(body.velocityAtPoint(pA), vTrue))).toBeLessThan(1e-5);
    world.free();
  });

  it('the created-child transfer formula preserves point velocity for an offset-COM child', () => {
    const omega = 5;
    // Parent spins; a child is carved off as its OWN body whose collider COM is
    // offset from the parent (the convex-hull / Voronoi case from TESTING.md).
    const { world, body: parent } = spinningRow([-1, 0, 1], omega);

    const child = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(0, 0, 0).setAngvel({ x: 0, y: 0, z: omega }),
    );
    // Offset-COM collider: a cube whose centre sits at +1.4, not on any node grid point.
    world.createCollider(RAPIER.ColliderDesc.cuboid(0.4, 0.4, 0.4).setTranslation(1.4, 0, 0).setDensity(1), child);
    child.recomputeMassPropertiesFromColliders();

    // syncBodyVelocityFromSource: v_child_com = v_parent_com + ω × (comChild − comParent).
    const comP = parent.worldCom();
    const comC = child.worldCom();
    const lvP = parent.linvel();
    const corr = cross(parent.angvel(), sub(comC, comP));
    child.setLinvel({ x: lvP.x + corr.x, y: lvP.y + corr.y, z: lvP.z + corr.z }, true);

    // The child must now move exactly as the parent did at the child's location.
    const p: V = { x: 1.4, y: 0, z: 0 };
    expect(mag(sub(child.velocityAtPoint(p), parent.velocityAtPoint(p)))).toBeLessThan(1e-5);
    world.free();
  });
});
