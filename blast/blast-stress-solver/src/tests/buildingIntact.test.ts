import { describe, it, expect } from 'vitest';
import { isBuildingRenderIntact, type IntactChunkState } from '../rapier/collisionTree';

const chunk = (bodyHandle: number | null, over: Partial<IntactChunkState> = {}): IntactChunkState => ({
  active: true,
  destroyed: false,
  bodyHandle,
  ...over,
});

describe('isBuildingRenderIntact', () => {
  it('is true when every fragment is alive and shares one body (per-building-body city)', () => {
    // Regression: each building is its own rigid body (≠ a single global root). The old
    // subtreeFullyIntact check required bodyHandle === rootBody.handle and so reported these
    // buildings as not-intact, leaving the proxy LOD stuck at "1 as box".
    const chunks = [chunk(7), chunk(7), chunk(7), chunk(7)];
    expect(isBuildingRenderIntact([0, 1, 2, 3], chunks)).toBe(true);
  });

  it('is true regardless of WHICH shared body it is (root or own body)', () => {
    expect(isBuildingRenderIntact([0, 1], [chunk(0), chunk(0)])).toBe(true);
    expect(isBuildingRenderIntact([0, 1], [chunk(42), chunk(42)])).toBe(true);
  });

  it('is false once fragments split across bodies (fractured)', () => {
    const chunks = [chunk(7), chunk(7), chunk(9) /* split off */, chunk(7)];
    expect(isBuildingRenderIntact([0, 1, 2, 3], chunks)).toBe(false);
  });

  it('is false when any fragment is destroyed, inactive, or bodyless', () => {
    expect(isBuildingRenderIntact([0, 1], [chunk(7), chunk(7, { destroyed: true })])).toBe(false);
    expect(isBuildingRenderIntact([0, 1], [chunk(7), chunk(7, { active: false })])).toBe(false);
    expect(isBuildingRenderIntact([0, 1], [chunk(7), chunk(null)])).toBe(false);
  });

  it('is false for an empty / unknown building', () => {
    expect(isBuildingRenderIntact([], [])).toBe(false);
    expect(isBuildingRenderIntact([0, 5], [chunk(7)])).toBe(false); // index 5 missing
  });
});
