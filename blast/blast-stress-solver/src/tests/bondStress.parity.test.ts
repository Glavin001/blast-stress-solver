/**
 * JS `computeBondStress` coverage + cross-language parity lock. The expected values here are
 * identical to the Rust `bond_stress_test.rs` hand-computed cases — both implementations are
 * byte-identical, and asserting the same numbers in both languages keeps them that way.
 * Covers the angular twist/bend paths (the existing Rust tests only used `ang = 0`).
 *
 * `computeBondStress` is pure (no WASM), so this runs everywhere.
 */
import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { computeBondStress } from '../../dist/index.js';

type V = { x: number; y: number; z: number };
const v = (x: number, y: number, z: number): V => ({ x, y, z });
// Two nodes 2 m apart along +X => bond normal = +X, distance = 2.
const nodes = [{ com: v(0, 0, 0) }, { com: v(2, 0, 0) }];
const bond = { node0: 0, node1: 1 };
const stress = (lin: V, ang: V, area: number) => computeBondStress(bond, { lin, ang }, nodes, area);
const near = (a: number, b: number) => Math.abs(a - b) < 1e-4;

describe('computeBondStress — known values (Rust parity lock)', () => {
  it('pure tension along normal', () => {
    const s = stress(v(6, 0, 0), v(0, 0, 0), 2);
    expect(near(s.tension, 3) && near(s.compression, 0) && near(s.shear, 0)).toBe(true);
  });
  it('pure compression against normal', () => {
    const s = stress(v(-6, 0, 0), v(0, 0, 0), 2);
    expect(near(s.compression, 3) && near(s.tension, 0) && near(s.shear, 0)).toBe(true);
  });
  it('pure linear shear', () => {
    const s = stress(v(0, 5, 0), v(0, 0, 0), 1);
    expect(near(s.shear, 5) && near(s.tension, 0) && near(s.compression, 0)).toBe(true);
  });
  it('pure twist adds to shear', () => {
    const s = stress(v(0, 0, 0), v(3, 0, 0), 1);
    expect(near(s.shear, 3) && near(s.tension, 0) && near(s.compression, 0)).toBe(true);
  });
  it('pure bend adds to normal (tension)', () => {
    const s = stress(v(0, 0, 0), v(0, 4, 0), 1);
    expect(near(s.tension, 4) && near(s.compression, 0) && near(s.shear, 0)).toBe(true);
  });
  it('mixed linear shear + twist', () => {
    const s = stress(v(0, 5, 0), v(2, 0, 0), 1);
    expect(near(s.shear, 7) && near(s.tension, 0) && near(s.compression, 0)).toBe(true);
  });
});

describe('computeBondStress — degenerate guards', () => {
  it('non-positive area is zero', () => {
    const s = stress(v(6, 0, 0), v(0, 0, 0), 0);
    expect(s).toEqual({ compression: 0, tension: 0, shear: 0 });
  });
  it('out-of-range nodes are zero', () => {
    const s = computeBondStress({ node0: 0, node1: 9 }, { lin: v(6, 0, 0), ang: v(0, 0, 0) }, nodes, 1);
    expect(s).toEqual({ compression: 0, tension: 0, shear: 0 });
  });
});

describe('computeBondStress — properties', () => {
  it('components are non-negative and compression/tension are mutually exclusive', () => {
    const arb = fc.record({ x: fc.double({ min: -50, max: 50, noNaN: true }), y: fc.double({ min: -50, max: 50, noNaN: true }), z: fc.double({ min: -50, max: 50, noNaN: true }) });
    fc.assert(
      fc.property(arb, arb, fc.double({ min: 0.1, max: 10, noNaN: true }), (lin, ang, area) => {
        const s = stress(lin, ang, area);
        expect(s.compression >= 0 && s.tension >= 0 && s.shear >= 0).toBe(true);
        expect(s.compression === 0 || s.tension === 0).toBe(true);
      }),
      { numRuns: 300 },
    );
  });
});
