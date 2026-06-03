/**
 * Performance-regression guard for the split planner — proves the O(max(M,N)^3)
 * cliff stays gone. These bounds are deliberately loose (10–100× headroom over
 * the real numbers): they are sized to catch a *return to cubic* (which turns
 * milliseconds into minutes), not to police a tight per-frame budget, so they
 * do not flake on slow/contended CI. Correctness at speed (same optimum as the
 * dense reference) is asserted in rapier.splitMigrator.equivalence.test.ts.
 *
 * The pure planner has no WASM/physics dependency, so this always runs.
 */
import { describe, expect, it } from "vitest";
import {
  planSplitMigration,
  planSplitMigrationReference,
} from "../rapier/splitMigrator";
import {
  buildShatterCase,
  buildMergeCase,
  planTotalOverlap,
  timePlanner,
} from "../rapier/splitPlannerBench";

describe("split planner: performance regression guard", () => {
  it("shatter 1×N scales ~linearly, not cubically (the cascade cliff is gone)", () => {
    // O(N^3) would make N=8192 take minutes (8192^3 ≈ 5.5e11 ops). Linear is ~ms.
    const big = timePlanner("production", buildShatterCase(8192), { budgetMs: 80 });
    expect(big.msPerPlan).toBeLessThan(250); // ~3ms in practice; cubic = minutes

    // Doubling N must roughly double the cost (linear), never ×8 (cubic).
    const a = timePlanner("production", buildShatterCase(4096), { budgetMs: 80 });
    const b = timePlanner("production", buildShatterCase(8192), { budgetMs: 80 });
    expect(b.msPerPlan / a.msPerPlan).toBeLessThan(5); // linear ≈ 2×, cubic ≈ 8×
  }, 20_000);

  it("merge M→N (multi-body components) scales sub-cubically", () => {
    // 8192 bodies → 4096 two-node children. The reference would pad to an
    // 8192×8192 Hungarian; production sees 4096 tiny independent components.
    const big = timePlanner("production", buildMergeCase(4096), { budgetMs: 80 });
    expect(big.msPerPlan).toBeLessThan(500); // ~7ms in practice
  }, 20_000);

  it("is dramatically faster than the dense reference, with an identical optimum", () => {
    // At N=256 the reference is well-resolved (~70ms); production is ~0.1ms.
    const testCase = buildShatterCase(256);
    const prod = timePlanner("production", testCase, { budgetMs: 60 });
    const ref = timePlanner("reference", testCase, { budgetMs: 120, maxIters: 20 });

    // Speedup is enormous (≈700× real); require only a very safe 10× to avoid flake.
    expect(ref.msPerPlan / prod.msPerPlan).toBeGreaterThan(10);

    // And the speed is free: same maximum overlap as the dense Hungarian.
    const prodPlan = planSplitMigration(testCase.bodies, testCase.children);
    const refPlan = planSplitMigrationReference(testCase.bodies, testCase.children);
    expect(planTotalOverlap(prodPlan, testCase.bodies, testCase.children)).toBe(
      planTotalOverlap(refPlan, testCase.bodies, testCase.children),
    );
    expect(prodPlan.reuse.length).toBe(refPlan.reuse.length);
  }, 20_000);
});
