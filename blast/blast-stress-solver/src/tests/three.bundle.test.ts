/**
 * Tests for the Three.js bundle helper and SolverDebugLinesHelper.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { SolverDebugLinesHelper, type DebugLine } from '../three/solver-debug-lines';

type FakeBody = {
  translation: () => { x: number; y: number; z: number };
  rotation: () => { x: number; y: number; z: number; w: number };
};

/**
 * Minimal stand-in for DestructibleCore with just the surface the helper reads:
 * chunks (rest centroid + current bodyHandle), a world that resolves bodies, and
 * a root body handle.
 */
function makeCore(opts: {
  chunks: Array<{ baseLocalOffset: { x: number; y: number; z: number }; bodyHandle: number | null }>;
  bodies: Map<number, FakeBody>;
  rootBodyHandle: number;
}) {
  const { chunks, bodies, rootBodyHandle } = opts;
  return {
    chunks,
    rootBodyHandle,
    world: { getRigidBody: (h: number) => bodies.get(h) ?? null },
  } as any;
}

function identityBody(x = 0, y = 0, z = 0): FakeBody {
  return {
    translation: () => ({ x, y, z }),
    rotation: () => ({ x: 0, y: 0, z: 0, w: 1 }),
  };
}

function readPositions(helper: SolverDebugLinesHelper, lineCount: number): number[] {
  const attr = (helper.object.geometry as THREE.BufferGeometry).getAttribute('position') as THREE.BufferAttribute;
  const arr = attr.array as Float32Array;
  return Array.from(arr.slice(0, lineCount * 6));
}

describe('SolverDebugLinesHelper', () => {
  it('creates with a valid Three.js object that starts hidden', () => {
    const helper = new SolverDebugLinesHelper();
    expect(helper.object).toBeInstanceOf(THREE.Object3D);
    expect(helper.object.visible).toBe(false);
  });

  it('update with empty lines hides the object and does not crash', () => {
    const helper = new SolverDebugLinesHelper();
    const core = makeCore({ chunks: [], bodies: new Map(), rootBodyHandle: 0 });
    helper.update(core, [], true);
    expect(helper.object.visible).toBe(false);
  });

  it('update with visible=false hides the object', () => {
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([[0, identityBody()]]);
    const core = makeCore({
      chunks: [{ baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 0 }],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 0, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, false);
    expect(helper.object.visible).toBe(false);
  });

  it('transforms endpoints by the root body pose for an intact structure', () => {
    const helper = new SolverDebugLinesHelper();
    // Root body sitting at the origin (identity).
    const bodies = new Map<number, FakeBody>([[0, identityBody()]]);
    const core = makeCore({
      chunks: [
        { baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 0 },
        { baseLocalOffset: { x: 1, y: 0, z: 0 }, bodyHandle: 0 },
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    expect(helper.object.visible).toBe(true);
    // Identity transform -> endpoints unchanged.
    expect(readPositions(helper, 1)).toEqual([0, 0, 0, 1, 0, 0]);
  });

  it('REGRESSION: each endpoint follows its OWN chunk body (no floating lines)', () => {
    // This is the bug from the screenshots: a debug line whose endpoint belongs to
    // a fragment that flew off must follow that fragment, not be drawn at a stale /
    // wrong actor pose. Here chunk 1 has broken off into body 7 lifted +100 in Y.
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([
      [0, identityBody()], // root, at origin
      [7, identityBody(0, 100, 0)], // flown-away fragment, lifted up
    ]);
    const core = makeCore({
      chunks: [
        { baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 0 }, // still on root
        { baseLocalOffset: { x: 1, y: 0, z: 0 }, bodyHandle: 7 }, // moved to body 7
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    // p0 stays on the root; p1 follows body 7 up by +100.
    expect(readPositions(helper, 1)).toEqual([0, 0, 0, 1, 100, 0]);
  });

  it('reflects body pose changes between frames (re-reads poses every frame)', () => {
    const helper = new SolverDebugLinesHelper();
    let y = 0;
    const movingBody: FakeBody = {
      translation: () => ({ x: 0, y, z: 0 }),
      rotation: () => ({ x: 0, y: 0, z: 0, w: 1 }),
    };
    const bodies = new Map<number, FakeBody>([[0, identityBody()], [3, movingBody]]);
    const core = makeCore({
      chunks: [{ baseLocalOffset: { x: 2, y: 0, z: 0 }, bodyHandle: 3 }],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 2, y: 0, z: 0 }, p1: { x: 2, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];

    helper.update(core, lines, true);
    expect(readPositions(helper, 1).slice(0, 3)).toEqual([2, 0, 0]);

    // Body moves; the same cached line->chunk mapping must pick up the new pose.
    y = 50;
    helper.update(core, lines, true);
    expect(readPositions(helper, 1).slice(0, 3)).toEqual([2, 50, 0]);
  });

  it('falls back to the root pose when a chunk has no live body', () => {
    const helper = new SolverDebugLinesHelper();
    // Root lifted to +5; chunk references a body handle that does not resolve.
    const bodies = new Map<number, FakeBody>([[0, identityBody(0, 5, 0)]]);
    const core = makeCore({
      chunks: [{ baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 999 }],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 0, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    expect(readPositions(helper, 1).slice(0, 3)).toEqual([0, 5, 0]);
  });

  it('writes per-endpoint vertex colors', () => {
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([[0, identityBody()]]);
    const core = makeCore({
      chunks: [{ baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 0 }],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 0, y: 0, z: 0 }, color0: 0xff0000, color1: 0x0000ff },
    ];
    helper.update(core, lines, true);
    const attr = (helper.object.geometry as THREE.BufferGeometry).getAttribute('color') as THREE.BufferAttribute;
    const c = Array.from((attr.array as Float32Array).slice(0, 6));
    expect(c[0]).toBeCloseTo(1, 5); // red endpoint 0
    expect(c[1]).toBeCloseTo(0, 5);
    expect(c[2]).toBeCloseTo(0, 5);
    expect(c[3]).toBeCloseTo(0, 5); // blue endpoint 1
    expect(c[4]).toBeCloseTo(0, 5);
    expect(c[5]).toBeCloseTo(1, 5);
  });

  it('dispose cleans up resources and is idempotent', () => {
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([[0, identityBody()]]);
    const core = makeCore({
      chunks: [{ baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 0 }],
      bodies,
      rootBodyHandle: 0,
    });
    helper.update(
      core,
      [{ p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0xff0000, color1: 0x00ff00 }],
      true,
    );
    helper.dispose();
    // Should not throw on second dispose
    helper.dispose();
  });
});
