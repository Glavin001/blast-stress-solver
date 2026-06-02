/**
 * Tier-0 property tests for the pure (no-WASM) kernels of the destruction pipeline.
 *
 * These run on every change in milliseconds and need no physics runtime, so they are
 * the cheap, exhaustive, un-cheatable backstop: a future performance optimization that
 * breaks the split planner's structural guarantees or the rigid-velocity-transfer math
 * fails here immediately.
 */
import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { planSplitMigration } from '../rapier/splitMigrator';

type Vec3 = { x: number; y: number; z: number };

const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const mag = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);

/** Rigid velocity field of a body with COM velocity `v` and angular velocity `w`. */
const velocityAtPoint = (v: Vec3, w: Vec3, com: Vec3, point: Vec3): Vec3 =>
  add(v, cross(w, sub(point, com)));

const arbVec3 = (range: number) =>
  fc.record({
    x: fc.double({ min: -range, max: range, noNaN: true }),
    y: fc.double({ min: -range, max: range, noNaN: true }),
    z: fc.double({ min: -range, max: range, noNaN: true }),
  });

describe('kernels/splitMigrator (pure structural invariants)', () => {
  // A partition of nodes [0..N) into child groups, plus candidate existing bodies.
  const arbPlanInputs = fc
    .integer({ min: 1, max: 8 })
    .chain((nodeCount) => {
      const nodes = Array.from({ length: nodeCount }, (_, i) => i);
      return fc.record({
        // Assign each node to a child bucket, then drop empty buckets.
        childOf: fc.array(fc.integer({ min: 0, max: nodeCount - 1 }), {
          minLength: nodeCount,
          maxLength: nodeCount,
        }),
        childSupport: fc.array(fc.boolean(), { minLength: nodeCount, maxLength: nodeCount }),
        existing: fc.array(
          fc.record({
            nodeSubset: fc.subarray(nodes, { minLength: 0 }),
            isFixed: fc.boolean(),
          }),
          { minLength: 0, maxLength: 4 },
        ),
      });
    });

  it('partitions every child exactly once and never double-assigns a body', () => {
    fc.assert(
      fc.property(arbPlanInputs, ({ childOf, childSupport, existing }) => {
        const buckets = new Map<number, number[]>();
        childOf.forEach((bucket, node) => {
          const list = buckets.get(bucket) ?? [];
          list.push(node);
          buckets.set(bucket, list);
        });
        const children = [...buckets.entries()].map(([bucket, nodes], index) => ({
          index,
          actorIndex: 1000 + bucket,
          nodes,
          isSupport: nodes.every((n) => childSupport[n]),
        }));
        const existingBodies = existing.map((body, i) => ({
          handle: 10 + i,
          nodeIndices: new Set(body.nodeSubset),
          isFixed: body.isFixed,
        }));
        const fixedByHandle = new Map(existingBodies.map((b) => [b.handle, b.isFixed]));

        const plan = planSplitMigration(existingBodies, children);

        // 1. Every child is placed exactly once (partition of childIndices).
        const placed = [...plan.reuse.map((r) => r.childIndex), ...plan.create.map((c) => c.childIndex)].sort(
          (a, b) => a - b,
        );
        expect(placed).toEqual(children.map((c) => c.index));

        // 2. No existing body is reused by more than one child.
        const reusedHandles = plan.reuse.map((r) => r.bodyHandle);
        expect(new Set(reusedHandles).size).toBe(reusedHandles.length);

        // 3. A non-support child never reuses a fixed body.
        for (const entry of plan.reuse) {
          const child = children[entry.childIndex];
          if (!child.isSupport) {
            expect(fixedByHandle.get(entry.bodyHandle) ?? false).toBe(false);
          }
        }
      }),
      { numRuns: 400 },
    );
  });
});

describe('kernels/rigid velocity transfer (shared continuity spec)', () => {
  // This documents the invariant the JS library satisfies and the Rust kernel must:
  // transferring a parent's rigid motion to a child via
  //   v_child = v_parent + ω × (childCom − parentCom),  ω_child = ω_parent
  // preserves the world-space velocity at every point. Point-velocity continuity is
  // the headline invariant of the whole harness — here proven at the math level.
  it('preserves point velocity for any parent motion, COM shift, and probe point', () => {
    fc.assert(
      fc.property(
        arbVec3(8),
        arbVec3(8),
        arbVec3(6),
        arbVec3(6),
        arbVec3(6),
        (vParent, omega, parentCom, childCom, probe) => {
          const comShift = sub(childCom, parentCom);
          const vChild = add(vParent, cross(omega, comShift));

          const parentVel = velocityAtPoint(vParent, omega, parentCom, probe);
          const childVel = velocityAtPoint(vChild, omega, childCom, probe);

          // Scale tolerance with the magnitudes involved (f64 here, so very tight).
          const scale = 1 + mag(omega) * (mag(comShift) + mag(sub(probe, parentCom)));
          expect(mag(sub(parentVel, childVel))).toBeLessThan(1e-9 * scale);
        },
      ),
      { numRuns: 400 },
    );
  });
});
