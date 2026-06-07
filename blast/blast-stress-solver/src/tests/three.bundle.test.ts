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
 * chunks (rest centroid + current bodyHandle + destroyed flag), a world that
 * resolves bodies, and a root body handle.
 */
function makeCore(opts: {
  chunks: Array<{
    baseLocalOffset: { x: number; y: number; z: number };
    bodyHandle: number | null;
    destroyed?: boolean;
  }>;
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

function drawnVertexCount(helper: SolverDebugLinesHelper): number {
  const geo = helper.object.geometry as THREE.BufferGeometry;
  return geo.drawRange.count;
}

function readPositions(helper: SolverDebugLinesHelper, lineCount: number): number[] {
  const attr = (helper.object.geometry as THREE.BufferGeometry).getAttribute('position') as THREE.BufferAttribute;
  return Array.from((attr.array as Float32Array).slice(0, lineCount * 6));
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

  it('draws an intact-structure bond at the root body pose', () => {
    const helper = new SolverDebugLinesHelper();
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
    expect(drawnVertexCount(helper)).toBe(2);
    expect(readPositions(helper, 1)).toEqual([0, 0, 0, 1, 0, 0]);
  });

  it('a bond within a broken-off fragment follows that fragment body', () => {
    // Both chunks moved together onto fragment body 7, lifted +100 in Y.
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([
      [0, identityBody()],
      [7, identityBody(0, 100, 0)],
    ]);
    const core = makeCore({
      chunks: [
        { baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 7 },
        { baseLocalOffset: { x: 1, y: 0, z: 0 }, bodyHandle: 7 },
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    expect(helper.object.visible).toBe(true);
    // Both endpoints rigidly transformed by body 7 (lifted +100).
    expect(readPositions(helper, 1)).toEqual([0, 100, 0, 1, 100, 0]);
  });

  it('REGRESSION: skips a bond whose endpoints are on different bodies (no floating/stretched lines)', () => {
    // chunk 0 stayed on the root; chunk 1 flew off onto body 7. The bond between
    // them must NOT be drawn — drawing it stretched a line from the ground into the
    // sky. This is the core of the reported "lines that don't make sense" bug.
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([
      [0, identityBody()],
      [7, identityBody(0, 100, 0)],
    ]);
    const core = makeCore({
      chunks: [
        { baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 0 },
        { baseLocalOffset: { x: 1, y: 0, z: 0 }, bodyHandle: 7 },
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    expect(helper.object.visible).toBe(false);
    expect(drawnVertexCount(helper)).toBe(0);
  });

  it('REGRESSION: skips a bond on a destroyed chunk (its mesh is hidden)', () => {
    // A bond left referencing a destroyed chunk would float where the chunk's
    // (hidden) body is — skip it so the overlay only marks visible geometry.
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([[0, identityBody()], [7, identityBody(0, 100, 0)]]);
    const core = makeCore({
      chunks: [
        { baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 7, destroyed: true },
        { baseLocalOffset: { x: 1, y: 0, z: 0 }, bodyHandle: 7 },
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    expect(helper.object.visible).toBe(false);
  });

  it('skips a bond whose body cannot be resolved', () => {
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([[0, identityBody()]]); // body 999 missing
    const core = makeCore({
      chunks: [
        { baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 999 },
        { baseLocalOffset: { x: 1, y: 0, z: 0 }, bodyHandle: 999 },
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    expect(helper.object.visible).toBe(false);
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
      chunks: [
        { baseLocalOffset: { x: 2, y: 0, z: 0 }, bodyHandle: 3 },
        { baseLocalOffset: { x: 3, y: 0, z: 0 }, bodyHandle: 3 },
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      { p0: { x: 2, y: 0, z: 0 }, p1: { x: 3, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];

    helper.update(core, lines, true);
    expect(readPositions(helper, 1)).toEqual([2, 0, 0, 3, 0, 0]);

    // Body moves; the same cached line->chunk mapping must pick up the new pose.
    y = 50;
    helper.update(core, lines, true);
    expect(readPositions(helper, 1)).toEqual([2, 50, 0, 3, 50, 0]);
  });

  it('writes per-endpoint vertex colors for a drawn bond', () => {
    const helper = new SolverDebugLinesHelper();
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
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0xff0000, color1: 0x0000ff },
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

  it('keeps drawing the valid bonds while skipping the bad ones', () => {
    // One good bond on the root, one spanning bond that must be dropped.
    const helper = new SolverDebugLinesHelper();
    const bodies = new Map<number, FakeBody>([[0, identityBody()], [7, identityBody(0, 100, 0)]]);
    const core = makeCore({
      chunks: [
        { baseLocalOffset: { x: 0, y: 0, z: 0 }, bodyHandle: 0 },
        { baseLocalOffset: { x: 1, y: 0, z: 0 }, bodyHandle: 0 },
        { baseLocalOffset: { x: 10, y: 0, z: 0 }, bodyHandle: 7 },
      ],
      bodies,
      rootBodyHandle: 0,
    });
    const lines: DebugLine[] = [
      // good: both on root body 0
      { p0: { x: 0, y: 0, z: 0 }, p1: { x: 1, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
      // bad: spans body 0 and body 7
      { p0: { x: 1, y: 0, z: 0 }, p1: { x: 10, y: 0, z: 0 }, color0: 0x00ff00, color1: 0x00ff00 },
    ];
    helper.update(core, lines, true);
    expect(helper.object.visible).toBe(true);
    // Only the good bond is drawn (2 vertices), and it is the root-space one.
    expect(drawnVertexCount(helper)).toBe(2);
    expect(readPositions(helper, 1)).toEqual([0, 0, 0, 1, 0, 0]);
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
    helper.dispose(); // should not throw on second dispose
  });
});
