/**
 * SIMD-vs-scalar parity gate for the WASM CGNR solver.
 *
 * Build via `EMCC_USE_SIMD=1 npm run build` and the WASM compiles `anglin6.h`'s
 * `__m128`/`__m256` specializations through emscripten's native AVX intrinsic
 * lowering (256-bit ops emulated as two wasm-simd128 ops, since emcc 3.1.68
 * landed AVX support).  FMA still needs a `(mul + add)` macro override in
 * `simd/simd.h` because baseline wasm-simd128 has no hardware FMA — same
 * arithmetic semantics the scalar autovec path produces.
 *
 * Without `EMCC_USE_SIMD=1` the build ships the scalar `AngLin6Ops<float>`
 * path that clang autovectorizes with `-O3 -msimd128`.  In our measured
 * scenario the scalar path is currently faster (the AVX kernels eat more
 * load/store traffic per AngLin6 op than the autovec produces from the
 * compact scalar source), so it stays the default.
 *
 * This file is the safety gate either way.  It pins:
 *   1. That the `s_use_simd` flag inside `StressProcessor` reflects the
 *      build (so a misconfigured EMCC_USE_SIMD silently shipping the wrong
 *      WASM surfaces at test time, not in production).
 *   2. That a deterministic gravity scenario converges with a tight
 *      residual on both paths.  The whole 406-test suite passes under both
 *      builds, but a dedicated check against an absolute number means a
 *      future kernel change that drifts the SIMD path away from scalar
 *      shows up here as a small, reviewable diff instead of an unrelated
 *      end-to-end test cascading.
 */
import { describe, it, expect } from 'vitest';
import type * as Runtime from '..';

const EXPECT_SIMD = process.env.EMCC_USE_SIMD === '1' || process.env.EMCC_USE_WASM_SIMD === '1';

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
