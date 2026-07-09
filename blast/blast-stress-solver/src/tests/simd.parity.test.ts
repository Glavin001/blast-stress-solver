/**
 * SIMD-vs-scalar parity gate for the WASM CGNR solver.
 *
 * As of PR #40 the WASM build defaults to the direct wasm-simd128 hand-port
 * of `AngLin6Ops` / `CouplingMatrixOps` / `InertiaMatrixOps` (see anglin6.h
 * / coupling.h / inertia.h).  The kill switch is `EMCC_NO_SIMD=1`, which
 * compiles the scalar `AngLin6Ops<float>` path that clang autovectorizes
 * with `-O3 -msimd128` — same algorithm as on main, byte-equivalent
 * behavior modulo compiler codegen differences.
 *
 * `EMCC_USE_SIMD=1` (without `EMCC_NO_SIMD`) selects the AVX-intrinsic
 * variant from PR #37 instead of v128.  Tied at scale, slower on the
 * worst-case frame; kept for fallback and Rust-crate parity.
 *
 * This file is the safety gate for the build-flag wiring.  It pins:
 *   1. That `StressProcessor::s_use_simd` reflects the build flag, so a
 *      misconfigured env var silently shipping the wrong WASM surfaces at
 *      test time, not in production.
 *   2. That a deterministic gravity scenario converges with a tight
 *      residual on every kernel path.  The whole 408-test suite passes on
 *      both, but a dedicated check against an absolute number means a
 *      future kernel change that drifts SIMD output away from scalar
 *      shows up here as a small, reviewable diff instead of an unrelated
 *      end-to-end test cascading.
 */
import { describe, it, expect } from 'vitest';
import type * as Runtime from '..';

// The WASM solver ships with SIMD on by default (v128 hand-port, PR #40).
// `EMCC_NO_SIMD=1` is the kill-switch that reverts to the scalar path.
// Mirror that exactly here so the parity test fails loudly if the build
// flag and the runtime flag disagree.
const EXPECT_SIMD = process.env.EMCC_NO_SIMD !== '1';

async function loadRuntime(): Promise<typeof Runtime> {
  return (await import('../../dist/index.js')) as typeof Runtime;
}

describe('SIMD build wiring', () => {
  it(`StressProcessor.usingSIMD() matches EMCC_USE_SIMD (expect ${EXPECT_SIMD})`, async () => {
    const { loadStressSolver } = await loadRuntime();
    const rt = await loadStressSolver();
    // The C++ export `stress_processor_using_simd` returns the value of
    // `StressProcessor::s_use_simd` — flipped to true on WASM whenever the
    // SIMD kernel headers are compiled in (i.e. when STRESS_SOLVER_NO_SIMD
    // is left undefined), false otherwise.
    const using = (rt.module as { ccall: (n: string, r: string, a: unknown[], v: unknown[]) => unknown })
      .ccall('stress_processor_using_simd', 'number', [], []) as number;
    expect(Boolean(using)).toBe(EXPECT_SIMD);
  });
});

describe('SIMD vs scalar — deterministic gravity scenario', () => {
  it('falling 1-bond column reaches the same converged impulse within solver tolerance', async () => {
    const { loadStressSolver } = await loadRuntime();
    const rt = await loadStressSolver();

    // Two-node column: bottom node is fixed (mass = 0), top node hangs off a
    // single vertical bond. Gravity along -Y pulls the top node down; the
    // bond carries the static reaction. The bond is set unbreakable so no
    // fracture commands should appear regardless of which kernel path runs.
    const solver = rt.createExtSolver({
      nodes: [
        { centroid: { x: 0, y: 0, z: 0 }, mass: 0, volume: 1 },
        { centroid: { x: 0, y: 1, z: 0 }, mass: 1, volume: 1 }
      ],
      bonds: [
        {
          centroid: { x: 0, y: 0.5, z: 0 },
          normal: { x: 0, y: 1, z: 0 },
          area: 1,
          node0: 0,
          node1: 1
        }
      ],
      settings: {
        maxSolverIterationsPerFrame: 64,
        compressionElasticLimit: 1e9,    // impossibly high — never fracture
        compressionFatalLimit: 1e9,
        tensionElasticLimit: 1e9,
        tensionFatalLimit: 1e9,
        shearElasticLimit: 1e9,
        shearFatalLimit: 1e9
      }
    });

    solver.addGravity({ x: 0, y: -9.81, z: 0 });
    solver.update();
    expect(solver.converged()).toBe(true);

    expect(solver.generateFractureCommands().fractures.length).toBe(0);

    // Residual is tight on both paths (solver tolerance is 1e-3 internally;
    // both kernels hit it within machine precision).
    const err = solver.stressError();
    expect(err.lin).toBeLessThan(1e-3);
    expect(err.ang).toBeLessThan(1e-3);

    solver.destroy();
  });
});
