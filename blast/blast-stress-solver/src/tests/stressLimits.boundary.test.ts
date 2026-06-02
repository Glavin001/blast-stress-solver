/**
 * Boundary coverage for `StressLimits.failureMode` — the fatal-threshold classifier uses a
 * strict `>`, so a bond exactly AT its fatal limit must NOT be classified as failed, and just
 * above it must be. Pure (no WASM).
 */
import { describe, it, expect } from 'vitest';
import { StressLimits } from '../../dist/index.js';

const limits = new StressLimits({
  compressionElasticLimit: 1,
  compressionFatalLimit: 100,
  tensionElasticLimit: 1,
  tensionFatalLimit: 50,
  shearElasticLimit: 1,
  shearFatalLimit: 30,
});
const s = (compression = 0, tension = 0, shear = 0) => ({ compression, tension, shear });

describe('StressLimits.failureMode fatal-threshold boundary (strict >)', () => {
  it('does not fail exactly at the limit', () => {
    expect(limits.failureMode(s(100, 0, 0))).toBeNull();
    expect(limits.failureMode(s(0, 50, 0))).toBeNull();
    expect(limits.failureMode(s(0, 0, 30))).toBeNull();
  });

  it('fails just above the limit, per channel', () => {
    expect(limits.failureMode(s(100.001, 0, 0))).toBe('compression');
    expect(limits.failureMode(s(0, 50.001, 0))).toBe('tension');
    expect(limits.failureMode(s(0, 0, 30.001))).toBe('shear');
  });

  it('does not fail just below the limit', () => {
    expect(limits.failureMode(s(99.999, 49.999, 29.999))).toBeNull();
  });

  it('prioritizes compression, then tension, then shear', () => {
    // All three over limit -> compression reported first.
    expect(limits.failureMode(s(200, 200, 200))).toBe('compression');
    // Compression safe, tension+shear over -> tension first.
    expect(limits.failureMode(s(0, 200, 200))).toBe('tension');
  });
});
