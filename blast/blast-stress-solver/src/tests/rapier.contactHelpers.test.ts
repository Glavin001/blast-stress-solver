import { describe, it, expect } from 'vitest';
import {
  computeSpeedFactor,
  applyQuatToVec3,
  bodyPointVelocity,
  worldPointToBodyLocal,
  chunkWorldCenter,
  applyProjectileMomentumBoost,
  relativeSpeedBetweenBodies,
  type SpeedScalingOptions,
} from '../rapier/contactHelpers';

/** Minimal RigidBody-like stub for velocity helpers (no WASM dependency). */
function mockBody(
  linvel: { x: number; y: number; z: number },
  translation: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 },
  angvel: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 },
) {
  return {
    linvel: () => linvel,
    angvel: () => angvel,
    translation: () => translation,
  } as any;
}

const defaultOpts: SpeedScalingOptions = {
  speedMinExternal: 0.5,
  speedMinInternal: 0.25,
  speedMax: 6.0,
  speedExponent: 1.0,
  slowSpeedFactor: 0.9,
  fastSpeedFactor: 3.0,
};

describe('computeSpeedFactor', () => {
  it('returns slowSpeedFactor when speed is 0 (resting contact)', () => {
    const factor = computeSpeedFactor(0, false, defaultOpts);
    expect(factor).toBeCloseTo(0.9);
  });

  it('returns slowSpeedFactor when speed is below external min', () => {
    const factor = computeSpeedFactor(0.3, false, defaultOpts);
    expect(factor).toBeCloseTo(0.9);
  });

  it('returns slowSpeedFactor when speed equals external min', () => {
    const factor = computeSpeedFactor(0.5, false, defaultOpts);
    expect(factor).toBeCloseTo(0.9);
  });

  it('returns fastSpeedFactor when speed is at or above max', () => {
    const factor = computeSpeedFactor(6.0, false, defaultOpts);
    expect(factor).toBeCloseTo(3.0);
  });

  it('returns fastSpeedFactor for very high speed', () => {
    const factor = computeSpeedFactor(100.0, false, defaultOpts);
    expect(factor).toBeCloseTo(3.0);
  });

  it('interpolates linearly between slow and fast for mid-range speed (exponent=1)', () => {
    // midpoint: speed = (0.5 + 6.0) / 2 = 3.25
    // t = (3.25 - 0.5) / (6.0 - 0.5) = 2.75 / 5.5 = 0.5
    // factor = 0.9 + (3.0 - 0.9) * 0.5 = 0.9 + 1.05 = 1.95
    const factor = computeSpeedFactor(3.25, false, defaultOpts);
    expect(factor).toBeCloseTo(1.95);
  });

  it('uses internal min threshold when isInternal=true', () => {
    // Speed 0.3 is below external min (0.5) but above internal min (0.25)
    const factorExternal = computeSpeedFactor(0.3, false, defaultOpts);
    const factorInternal = computeSpeedFactor(0.3, true, defaultOpts);
    expect(factorExternal).toBeCloseTo(0.9); // below min, returns slow
    expect(factorInternal).toBeGreaterThan(0.9); // above internal min, interpolates
  });

  it('respects speedExponent > 1 (concave curve)', () => {
    const opts: SpeedScalingOptions = { ...defaultOpts, speedExponent: 2.0 };
    // t = ((3.25 - 0.5) / 5.5)^2 = 0.5^2 = 0.25
    // factor = 0.9 + 2.1 * 0.25 = 0.9 + 0.525 = 1.425
    const factor = computeSpeedFactor(3.25, false, opts);
    expect(factor).toBeCloseTo(1.425);
  });

  it('returns 1.0 on internal error (defensive fallback)', () => {
    // Pass null/invalid opts to trigger the catch
    const factor = computeSpeedFactor(1.0, false, null as any);
    expect(factor).toBe(1.0);
  });
});

describe('applyQuatToVec3', () => {
  it('identity quaternion returns same vector', () => {
    const v = { x: 1, y: 2, z: 3 };
    const q = { x: 0, y: 0, z: 0, w: 1 };
    const result = applyQuatToVec3(v, q);
    expect(result.x).toBeCloseTo(1);
    expect(result.y).toBeCloseTo(2);
    expect(result.z).toBeCloseTo(3);
  });

  it('90° rotation around Y axis', () => {
    // Rotate (1, 0, 0) by 90° around Y → (0, 0, -1)
    const angle = Math.PI / 2;
    const q = { x: 0, y: Math.sin(angle / 2), z: 0, w: Math.cos(angle / 2) };
    const v = { x: 1, y: 0, z: 0 };
    const result = applyQuatToVec3(v, q);
    expect(result.x).toBeCloseTo(0);
    expect(result.y).toBeCloseTo(0);
    expect(result.z).toBeCloseTo(-1);
  });

  it('180° rotation around Z axis', () => {
    // Rotate (1, 0, 0) by 180° around Z → (-1, 0, 0)
    const q = { x: 0, y: 0, z: 1, w: 0 }; // 180° around Z
    const v = { x: 1, y: 0, z: 0 };
    const result = applyQuatToVec3(v, q);
    expect(result.x).toBeCloseTo(-1);
    expect(result.y).toBeCloseTo(0);
    expect(result.z).toBeCloseTo(0);
  });
});

describe('relativeSpeedBetweenBodies', () => {
  const atPoint = { x: 0, y: 0, z: 0 };

  it('returns 0 when both bodies are null (resting/unresolved contact)', () => {
    expect(relativeSpeedBetweenBodies(null, null, atPoint)).toBe(0);
  });

  it('returns the moving body speed when the other is null', () => {
    const b = mockBody({ x: 3, y: 4, z: 0 });
    expect(relativeSpeedBetweenBodies(b, null, atPoint)).toBeCloseTo(5);
  });

  it('sums opposing linear velocities into the relative speed', () => {
    const b1 = mockBody({ x: 2, y: 0, z: 0 });
    const b2 = mockBody({ x: -3, y: 0, z: 0 });
    // |v1 - v2| = |2 - (-3)| = 5
    expect(relativeSpeedBetweenBodies(b1, b2, atPoint)).toBeCloseTo(5);
  });

  it('returns 0 for two bodies moving identically (no relative motion)', () => {
    const v = { x: 1, y: 2, z: 3 };
    expect(relativeSpeedBetweenBodies(mockBody(v), mockBody(v), atPoint)).toBeCloseTo(0);
  });

  it('includes the angular (lever-arm) contribution at an offset point', () => {
    // Body spinning about +Z at 1 rad/s, evaluated 1 unit out along +X →
    // tangential velocity ≈ (ω × r) magnitude = 1.
    const spinning = mockBody({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 });
    expect(relativeSpeedBetweenBodies(spinning, null, { x: 1, y: 0, z: 0 })).toBeCloseTo(1);
  });
});

describe('applyProjectileMomentumBoost', () => {
  it('returns currentEffMag when neither body is a projectile', () => {
    const b1 = { userData: {} } as any;
    const b2 = { userData: {} } as any;
    const result = applyProjectileMomentumBoost(b1, b2, 10, 1 / 60, 500);
    expect(result).toBe(500);
  });

  it('returns currentEffMag when bodies are null', () => {
    const result = applyProjectileMomentumBoost(null, null, 10, 1 / 60, 500);
    expect(result).toBe(500);
  });

  it('boosts magnitude when projectile has high momentum', () => {
    const projBody = {
      userData: { projectile: true },
      mass: () => 5,
    } as any;
    const other = { userData: {} } as any;
    // momentum = 5 * 20 = 100, force = 100 / (1/60) = 6000
    // currentEffMag = 100, so should return 6000
    const result = applyProjectileMomentumBoost(projBody, other, 20, 1 / 60, 100);
    expect(result).toBeCloseTo(6000);
  });

  it('keeps currentEffMag when momentum force is lower', () => {
    const projBody = {
      userData: { projectile: true },
      mass: () => 0.1,
    } as any;
    const other = { userData: {} } as any;
    // momentum = 0.1 * 1 = 0.1, force = 0.1 / (1/60) = 6
    // currentEffMag = 1000, so keeps 1000
    const result = applyProjectileMomentumBoost(projBody, other, 1, 1 / 60, 1000);
    expect(result).toBe(1000);
  });
});
